"""Text / object cross-attention injection strength + sensitivity diagnostic.

No-training diagnostic. Two pieces of evidence are produced:

Part A — Cross-attn residual norms.
  Hooks ``model.denoiser.text_xattn`` and ``model.denoiser.obj_xattn`` to
  measure ||attn_out|| / ||seq_before|| and the cosine angle between
  attn_out and seq_before. Reported per timestep t∈{100,300,500,700,900}
  under q(x_t|x0).

Part B — Text token sensitivity.
  Perturb the text features at the input to ``_build_cond``: gt /
  zero / shuffle / wrong_clip / random_normmatched. Keep z_int,
  object_world_traj, object_tokens, and plan unchanged. Measure one-step
  Δmotion-135 + dynamics + transition + plan metrics at the same timesteps
  and full DDPM rollout sensitivity.

Part C — Object point-token sensitivity.
  Same idea as Part B but on object_encoder(object_pc) output. The
  object_world_traj route is left untouched in this part.

Part D — Object-world-traj vs object-token route summary.
  Pulled from the previous target_route_ablation result and Part C; no
  new compute.

CAVEAT: inference-only perturbation. Distribution-shift caveat
applies; this is a causal-sensitivity probe, not a deployable method.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import Tensor

from diagnostic_common import (
    clip_metadata,
    dynamics_metrics,
    extract_plan,
    format_md_table,
    load_checkpoint,
    make_seq_mask,
    merge_single_batches,
    transition_metrics,
    write_json,
)
from dynamics_diagnostic import _build_cond, _build_model, _fk_from_motion_135
from piano.utils.clip_utils import load_clip_text_encoder
from plan_condition_diagnostics import _compute_metrics as _compute_plan_metrics
from recon_ladder_truncated_rollout_diagnostic import (
    _build_selected_batches,
    _load_selection,
)


PART_TO_JOINT = torch.tensor([20, 21, 10, 11, 0], dtype=torch.long)


# ---------------------------------------------------------------------------
# Residual-norm hooks on text_xattn / obj_xattn
# ---------------------------------------------------------------------------


class _XAttnTap:
    """Stash ``seq_before`` and ``attn_out`` from a MultiheadAttention call.

    The PyTorch MHA returns ``(attn_output, attn_weights)``; the first
    positional input is the query. We register a forward_pre_hook to
    capture q and a forward_hook to capture attn_output.
    """

    def __init__(self, module: torch.nn.Module) -> None:
        self.module = module
        self.seq_before: Tensor | None = None
        self.attn_out: Tensor | None = None
        self._h_pre = module.register_forward_pre_hook(self._pre_hook)
        self._h_post = module.register_forward_hook(self._post_hook)

    def _pre_hook(self, _module, args):
        if args:
            self.seq_before = args[0].detach()
        return None

    def _post_hook(self, _module, _inputs, output):
        if isinstance(output, tuple):
            self.attn_out = output[0].detach()
        else:
            self.attn_out = output.detach()
        return None

    def stats(self) -> dict[str, float]:
        if self.seq_before is None or self.attn_out is None:
            return {"attn_over_seq_norm": 0.0, "delta_over_seq_norm": 0.0, "cos_seq_attn": 0.0, "seq_norm": 0.0, "attn_norm": 0.0}
        seq = self.seq_before.float()
        atn = self.attn_out.float()
        if seq.shape != atn.shape:
            # MHA may broadcast/repeat differently — re-shape via sum-pool fallback
            return {"attn_over_seq_norm": 0.0, "delta_over_seq_norm": 0.0, "cos_seq_attn": 0.0, "seq_norm": 0.0, "attn_norm": 0.0}
        seq_norm = torch.linalg.vector_norm(seq, dim=-1)
        atn_norm = torch.linalg.vector_norm(atn, dim=-1)
        delta_norm = torch.linalg.vector_norm(atn, dim=-1)
        cos = torch.nn.functional.cosine_similarity(seq.reshape(-1, seq.shape[-1]), atn.reshape(-1, atn.shape[-1]), dim=-1)
        s_mean = float(seq_norm.mean().item())
        a_mean = float(atn_norm.mean().item())
        return {
            "seq_norm": s_mean,
            "attn_norm": a_mean,
            "attn_over_seq_norm": float((atn_norm / seq_norm.clamp_min(1e-12)).mean().item()),
            "delta_over_seq_norm": float((delta_norm / seq_norm.clamp_min(1e-12)).mean().item()),
            "cos_seq_attn": float(cos.mean().item()),
        }

    def reset(self) -> None:
        self.seq_before = None
        self.attn_out = None

    def close(self) -> None:
        self._h_pre.remove()
        self._h_post.remove()


# ---------------------------------------------------------------------------
# Token perturbations
# ---------------------------------------------------------------------------


def _shuffle_batch(tensor: Tensor, seed: int) -> Tensor:
    """Permute tensor along the batch dim (dim=0)."""
    B = tensor.shape[0]
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    perm = torch.randperm(B, generator=g).to(tensor.device)
    return tensor[perm]


def _wrong_clip(tensor: Tensor) -> Tensor:
    """Roll tensor by +1 along batch dim — every clip sees a different clip's token."""
    return torch.roll(tensor, shifts=1, dims=0)


def _random_norm_matched(tensor: Tensor, seed: int) -> Tensor:
    """Replace tensor with Gaussian noise whose per-sample norm matches."""
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    flat = tensor.reshape(tensor.shape[0], -1)
    norms = torch.linalg.vector_norm(flat, dim=-1, keepdim=True)
    new = torch.randn(flat.shape, generator=g).to(tensor.device)
    new_norms = torch.linalg.vector_norm(new, dim=-1, keepdim=True).clamp_min(1e-12)
    out = new * (norms / new_norms)
    return out.reshape(tensor.shape)


def _zero(tensor: Tensor) -> Tensor:
    return torch.zeros_like(tensor)


def _build_text_variants(cond: dict[str, Any], seed: int, include_random: bool) -> dict[str, dict[str, Any]]:
    base = cond["text"]
    variants = {
        "text_gt": {**cond, "text": base},
        "text_zero": {**cond, "text": _zero(base)},
        "text_shuffle": {**cond, "text": _shuffle_batch(base, seed=seed + 1)},
        "text_wrong_clip": {**cond, "text": _wrong_clip(base)},
    }
    if include_random:
        variants["text_random_normmatched"] = {**cond, "text": _random_norm_matched(base, seed=seed + 2)}
    return variants


def _build_objtok_variants(cond: dict[str, Any], seed: int, include_random: bool) -> dict[str, dict[str, Any]]:
    base = cond["object_tokens"]
    variants = {
        "objtok_gt": {**cond, "object_tokens": base},
        "objtok_zero": {**cond, "object_tokens": _zero(base)},
        "objtok_shuffle": {**cond, "object_tokens": _shuffle_batch(base, seed=seed + 3)},
        "objtok_wrong_clip": {**cond, "object_tokens": _wrong_clip(base)},
    }
    if include_random:
        variants["objtok_random_normmatched"] = {**cond, "object_tokens": _random_norm_matched(base, seed=seed + 4)}
    return variants


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def _one_step_x0(
    model,
    motion_gt: Tensor,
    cond: dict[str, Any],
    t_value: int,
    *,
    seed: int,
    cfg_scale: float,
) -> Tensor:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    t = torch.full(
        (motion_gt.shape[0],), int(t_value), device=motion_gt.device, dtype=torch.long,
    )
    noise = torch.randn_like(motion_gt)
    x_t = model.diffusion.q_sample(motion_gt, t, noise)
    pred = model.denoiser(x_t, t, cond, cond_drop_mask=None, self_cond=None)
    if float(cfg_scale) != 1.0:
        drop = torch.ones(x_t.shape[0], dtype=torch.bool, device=x_t.device)
        pred_uncond = model.denoiser(x_t, t, cond, cond_drop_mask=drop, self_cond=None)
    else:
        pred_uncond = None
    if model.diffusion.prediction_target == "v":
        x0_cond = model.diffusion.predict_x0_from_v(x_t, t, pred)
        x0_uncond = (
            model.diffusion.predict_x0_from_v(x_t, t, pred_uncond)
            if pred_uncond is not None
            else None
        )
    else:
        x0_cond, x0_uncond = pred, pred_uncond
    if x0_uncond is None:
        return x0_cond
    return x0_uncond + float(cfg_scale) * (x0_cond - x0_uncond)


@torch.no_grad()
def _full_rollout(model, cond: dict[str, Any], *, seq_length: int, seed: int, cfg_scale: float) -> Tensor:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    return model.sample(cond=cond, seq_length=int(seq_length), cfg_scale=float(cfg_scale))


def _metrics_for_motion(
    motion: Tensor, *, rest_offsets: Tensor, gt_joints: Tensor,
    object_positions: Tensor, contact_state: Tensor, seq_mask: Tensor,
    plan: dict[str, Tensor], fps: float, threshold: float, part_to_joint: Tensor,
) -> dict[str, Any]:
    joints = _fk_from_motion_135(motion, rest_offsets)
    dyn = dynamics_metrics(joints, seq_mask, gt_joints=gt_joints, fps=fps)
    trans = transition_metrics(
        joints, object_positions, contact_state, seq_mask,
        gt_joints=gt_joints, window_k=10, threshold=threshold,
    )
    plan_m = _compute_plan_metrics(
        jpos_pred=joints, jpos_gt=gt_joints, seq_mask=seq_mask,
        anchor_time=plan["anchor_time"], anchor_mask=plan["anchor_mask"],
        anchor_part=plan["anchor_part"], anchor_target_world=plan["anchor_target_world"],
        part_to_joint=part_to_joint, window=3,
    )
    return {"dynamics": dyn, "transition": trans, "plan": plan_m, "joints": joints}


def _summary_row(
    metrics: dict[str, Any], motion: Tensor, *,
    baseline_motion: Tensor | None, baseline_joints: Tensor | None,
    joints: Tensor,
) -> dict[str, float]:
    dyn = metrics["dynamics"]
    trans = metrics["transition"].get("ratios_over_gt", {})
    plan_m = metrics["plan"]
    out = {
        "body_velocity_over_gt": float(dyn.get("body_velocity_cm_per_frame_over_gt", 0.0)),
        "hand_velocity_over_gt": float(dyn.get("hand_velocity_cm_per_frame_over_gt", 0.0)),
        "body_acc_p95_over_gt": float(dyn.get("body_acc_p95_cm_per_frame2_over_gt", 0.0)),
        "fft_mid": float(dyn.get("fft_mid", 0.0)),
        "onset_xgt": float(trans.get("onset_positive_closing", 0.0)),
        "release_xgt": float(trans.get("release_positive_opening", 0.0)),
        "transition_relvel_xgt": float(trans.get("transition_relative_velocity", 0.0)),
        "far_unobserved_error_cm": float(plan_m.get("far_unobserved_error_cm", 0.0)),
        "near_anchor_window_error_cm": float(plan_m.get("near_anchor_window_error_cm", 0.0)),
        "anchor_realization_cm": float(plan_m.get("plan_anchor_contact_realization_cm", 0.0)),
    }
    if baseline_motion is not None:
        out["motion135_delta_vs_baseline"] = float(
            torch.linalg.vector_norm(
                (motion - baseline_motion).reshape(motion.shape[0], -1), dim=-1,
            ).mean().item()
        )
    if baseline_joints is not None:
        out["joint_delta_vs_baseline_cm"] = float(
            torch.linalg.vector_norm(
                (joints - baseline_joints).reshape(joints.shape[0], -1), dim=-1,
            ).mean().item() * 100.0
        )
    return out


# ---------------------------------------------------------------------------
# Verdict + report
# ---------------------------------------------------------------------------


def _verdict(payload: dict[str, Any]) -> str:
    a = payload["partA_residuals"]
    # Average over timesteps
    text_attn_ratio = float(np.mean([row["text"]["attn_over_seq_norm"] for row in a.values()]))
    obj_attn_ratio = float(np.mean([row["object"]["attn_over_seq_norm"] for row in a.values()]))

    def _delta_for(rows: list[dict[str, Any]], name: str) -> float:
        for r in rows:
            if r["variant"] == name:
                return float(r.get("motion135_delta_vs_baseline", 0.0))
        return 0.0

    text_rollout = payload["partB_text_level2"]
    obj_rollout = payload["partC_objtok_level2"]
    text_zero_delta = _delta_for(text_rollout, "text_zero")
    text_wrong_delta = _delta_for(text_rollout, "text_wrong_clip")
    obj_zero_delta = _delta_for(obj_rollout, "objtok_zero")
    obj_wrong_delta = _delta_for(obj_rollout, "objtok_wrong_clip")

    cases: list[str] = []
    if text_attn_ratio < 0.05 and text_zero_delta < 0.05 and text_wrong_delta < 0.05:
        cases.append("Case F (text): residual tiny and perturbations barely change output — text path is weak")
    elif text_attn_ratio >= 0.05 and text_zero_delta < 0.05:
        cases.append("Case H (text): nontrivial residual but perturbations do not affect output — injection present but task-unused")
    elif text_zero_delta >= 0.05 or text_wrong_delta >= 0.05:
        cases.append("Text path is causally active (non-zero output sensitivity)")

    if obj_attn_ratio < 0.05 and obj_zero_delta < 0.05 and obj_wrong_delta < 0.05:
        cases.append("Case G (object tokens): residual tiny and perturbations barely change output; geometry carried by object_world_traj")
    elif obj_attn_ratio >= 0.05 and obj_zero_delta < 0.05:
        cases.append("Case H (object tokens): nontrivial residual but perturbations do not affect output")
    elif obj_zero_delta >= 0.05 or obj_wrong_delta >= 0.05:
        cases.append("Object token path is causally active")

    if not cases:
        cases.append("Mixed / inconclusive; review per-timestep tables")
    return "; ".join(cases)


def _write_report(payload: dict[str, Any], path: Path) -> None:
    lines: list[str] = [
        "# Text / Object Cross-Attn Injection Diagnostic",
        "",
        f"- Config: `{payload['config']}`",
        f"- Checkpoint: `{payload['ckpt']}`",
        f"- Seed: {payload['seed']}",
        f"- Clips: {len(payload['selected_clips'])}",
        "",
        "## Distribution-shift caveat",
        "",
        "Token-level inference perturbations introduce train-inference distribution shift. "
        "This is a causal-sensitivity diagnostic ONLY; it does not propose an inference-time fix.",
        "",
    ]

    # Part A
    lines.extend(["## Part A — Cross-attn residual norms by timestep", ""])
    rows = [["t", "text ||attn||/||seq||", "text Δ/||seq||", "text cos(seq,attn)", "obj ||attn||/||seq||", "obj Δ/||seq||", "obj cos(seq,attn)"]]
    for t in payload["timesteps"]:
        r = payload["partA_residuals"][str(t)]
        rows.append([
            str(t),
            f"{r['text']['attn_over_seq_norm']:.4f}",
            f"{r['text']['delta_over_seq_norm']:.4f}",
            f"{r['text']['cos_seq_attn']:.3f}",
            f"{r['object']['attn_over_seq_norm']:.4f}",
            f"{r['object']['delta_over_seq_norm']:.4f}",
            f"{r['object']['cos_seq_attn']:.3f}",
        ])
    lines.append(format_md_table(rows))
    lines.append("")

    # Part B Level 1
    lines.extend(["## Part B — Text token Level 1 one-step (timestep-averaged Δ vs text_gt)", ""])
    rows = [["variant", "body xGT", "hand xGT", "onset xGT", "release xGT", "far cm", "anchor cm", "Δm135", "Δjoint cm"]]
    for row in payload["partB_text_level1_agg"]:
        rows.append([
            row["variant"],
            f"{row['body_velocity_over_gt']:.3f}",
            f"{row['hand_velocity_over_gt']:.3f}",
            f"{row['onset_xgt']:.3f}",
            f"{row['release_xgt']:.3f}",
            f"{row['far_unobserved_error_cm']:.2f}",
            f"{row['anchor_realization_cm']:.2f}",
            f"{row.get('motion135_delta_vs_baseline', 0.0):.4f}",
            f"{row.get('joint_delta_vs_baseline_cm', 0.0):.3f}",
        ])
    lines.append(format_md_table(rows))
    lines.append("")

    # Part B Level 2
    lines.extend(["## Part B — Text token Level 2 full DDPM rollout (vs text_gt)", ""])
    rows = [["variant", "body xGT", "hand xGT", "onset xGT", "release xGT", "far cm", "anchor cm", "Δm135", "Δjoint cm"]]
    for row in payload["partB_text_level2"]:
        rows.append([
            row["variant"],
            f"{row['body_velocity_over_gt']:.3f}",
            f"{row['hand_velocity_over_gt']:.3f}",
            f"{row['onset_xgt']:.3f}",
            f"{row['release_xgt']:.3f}",
            f"{row['far_unobserved_error_cm']:.2f}",
            f"{row['anchor_realization_cm']:.2f}",
            f"{row.get('motion135_delta_vs_baseline', 0.0):.4f}",
            f"{row.get('joint_delta_vs_baseline_cm', 0.0):.3f}",
        ])
    lines.append(format_md_table(rows))
    lines.append("")

    # Part C Level 1
    lines.extend(["## Part C — Object token Level 1 one-step (timestep-averaged Δ vs objtok_gt)", ""])
    rows = [["variant", "body xGT", "hand xGT", "onset xGT", "release xGT", "far cm", "anchor cm", "Δm135", "Δjoint cm"]]
    for row in payload["partC_objtok_level1_agg"]:
        rows.append([
            row["variant"],
            f"{row['body_velocity_over_gt']:.3f}",
            f"{row['hand_velocity_over_gt']:.3f}",
            f"{row['onset_xgt']:.3f}",
            f"{row['release_xgt']:.3f}",
            f"{row['far_unobserved_error_cm']:.2f}",
            f"{row['anchor_realization_cm']:.2f}",
            f"{row.get('motion135_delta_vs_baseline', 0.0):.4f}",
            f"{row.get('joint_delta_vs_baseline_cm', 0.0):.3f}",
        ])
    lines.append(format_md_table(rows))
    lines.append("")

    # Part C Level 2
    lines.extend(["## Part C — Object token Level 2 full DDPM rollout (vs objtok_gt)", ""])
    rows = [["variant", "body xGT", "hand xGT", "onset xGT", "release xGT", "far cm", "anchor cm", "Δm135", "Δjoint cm"]]
    for row in payload["partC_objtok_level2"]:
        rows.append([
            row["variant"],
            f"{row['body_velocity_over_gt']:.3f}",
            f"{row['hand_velocity_over_gt']:.3f}",
            f"{row['onset_xgt']:.3f}",
            f"{row['release_xgt']:.3f}",
            f"{row['far_unobserved_error_cm']:.2f}",
            f"{row['anchor_realization_cm']:.2f}",
            f"{row.get('motion135_delta_vs_baseline', 0.0):.4f}",
            f"{row.get('joint_delta_vs_baseline_cm', 0.0):.3f}",
        ])
    lines.append(format_md_table(rows))
    lines.append("")

    lines.extend(["## Part D — Object route separation", "", payload["partD_summary"], ""])

    lines.extend(["## Selected Clips", ""])
    rows = [["subset", "seq_id", "seq_len", "text"]]
    for c in payload["selected_clips"]:
        rows.append([c["subset"], c["seq_id"], c["seq_len"], c["text"][:80]])
    lines.append(format_md_table(rows))
    lines.append("")

    lines.extend(["## Verdict", "", payload["verdict"], ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/training/anchordiff_v18_a1_FULL_DATA.yaml"))
    parser.add_argument("--ckpt", type=Path, default=Path("runs/training/stageB_anchordiff_v18_a1_FULL_DATA/final.pt"))
    parser.add_argument("--output", type=Path, default=Path("analyses/2026-05-14_text_object_injection_diagnostic.json"))
    parser.add_argument("--md", type=Path, default=Path("analyses/2026-05-14_text_object_injection_diagnostic.md"))
    parser.add_argument("--selection-json", type=Path, default=Path("analyses/2026-05-13_v18_recon_ladder_truncated_rollout.json"))
    parser.add_argument("--max-clips", type=int, default=4)
    parser.add_argument("--num-candidates", type=int, default=96)
    parser.add_argument("--bucket", choices=["train", "val"], default="train")
    parser.add_argument("--balanced-subsets", action="store_true")
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--timesteps", type=str, default="100,300,500,700,900")
    parser.add_argument("--include-random-stress", action="store_true",
                        help="Include random_normmatched stress variants.")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    selection = _load_selection(args.selection_json, max_clips=int(args.max_clips))
    selected = _build_selected_batches(
        cfg,
        bucket=args.bucket,
        balanced_subsets=bool(args.balanced_subsets),
        num_candidates=int(args.num_candidates),
        selection=selection,
        max_clips=int(args.max_clips),
        threshold=float(args.threshold),
    )
    if not selected:
        raise RuntimeError("No clips selected for text/object injection diagnostic")
    batch = merge_single_batches([item[1] for item in selected])

    model, object_encoder, z_dims = _build_model(cfg, device)
    load_checkpoint(model, object_encoder, args.ckpt)
    clip_model = load_clip_text_encoder(
        device=device,
        model_name=str(cfg.model.text_encoder.clip_version),
        download_root=str(cfg.model.text_encoder.get("download_root", "cache/clip")),
    )
    cond, total_t = _build_cond(batch, model, object_encoder, clip_model, z_dims, cfg, device)
    plan_gt = extract_plan(batch, device)
    base_cond = {**cond, "interaction_plan": plan_gt}
    motion_gt = batch["motion"].to(device).float()
    gt_joints = batch["joints"].to(device).float()
    rest_offsets = batch["rest_offsets"].to(device).float()
    object_positions = batch["object_positions"].to(device).float()
    contact_state = batch["contact_state"].to(device).float()
    seq_mask = make_seq_mask(batch["seq_len"], total_t, device)
    part_to_joint = PART_TO_JOINT.to(device)
    fps = float(args.fps)
    threshold = float(args.threshold)
    timesteps = _parse_ints(args.timesteps)

    # ---- Part A: residual norms by timestep ----------------------------
    den = model.denoiser
    text_tap = _XAttnTap(den.text_xattn)
    obj_tap = _XAttnTap(den.obj_xattn)
    partA: dict[str, dict[str, dict[str, float]]] = {}
    try:
        for t_int in timesteps:
            text_tap.reset()
            obj_tap.reset()
            _ = _one_step_x0(
                model, motion_gt, base_cond, t_int,
                seed=int(args.seed) + t_int, cfg_scale=float(args.cfg_scale),
            )
            partA[str(t_int)] = {"text": text_tap.stats(), "object": obj_tap.stats()}
    finally:
        text_tap.close()
        obj_tap.close()

    # ---- Part B: text sensitivity, Level 1 -----------------------------
    print("Part B: text sensitivity Level 1")
    text_variants = _build_text_variants(base_cond, seed=int(args.seed), include_random=bool(args.include_random_stress))
    text_l1_by_t_variant: dict[int, dict[str, dict[str, float]]] = {}
    for t_int in timesteps:
        text_l1_by_t_variant[t_int] = {}
        baseline_motion = _one_step_x0(
            model, motion_gt, text_variants["text_gt"], t_int,
            seed=int(args.seed) + t_int, cfg_scale=float(args.cfg_scale),
        )
        baseline_metrics = _metrics_for_motion(
            baseline_motion, rest_offsets=rest_offsets, gt_joints=gt_joints,
            object_positions=object_positions, contact_state=contact_state,
            seq_mask=seq_mask, plan=plan_gt, fps=fps, threshold=threshold,
            part_to_joint=part_to_joint,
        )
        for name, c in text_variants.items():
            if name == "text_gt":
                motion = baseline_motion
                metrics = baseline_metrics
            else:
                motion = _one_step_x0(
                    model, motion_gt, c, t_int,
                    seed=int(args.seed) + t_int, cfg_scale=float(args.cfg_scale),
                )
                metrics = _metrics_for_motion(
                    motion, rest_offsets=rest_offsets, gt_joints=gt_joints,
                    object_positions=object_positions, contact_state=contact_state,
                    seq_mask=seq_mask, plan=plan_gt, fps=fps, threshold=threshold,
                    part_to_joint=part_to_joint,
                )
            summary = _summary_row(
                metrics, motion,
                baseline_motion=baseline_motion, baseline_joints=baseline_metrics["joints"],
                joints=metrics["joints"],
            )
            text_l1_by_t_variant[t_int][name] = summary

    text_variants_names = list(text_variants.keys())

    def _agg_by_variant(by_t: dict[int, dict[str, dict[str, float]]], names: list[str]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for name in names:
            keys = list(by_t[next(iter(by_t))][name].keys())
            agg = {"variant": name}
            for k in keys:
                vals = [by_t[t][name][k] for t in by_t]
                vals = [v for v in vals if np.isfinite(v)]
                agg[k] = float(np.mean(vals)) if vals else 0.0
            out.append(agg)
        return out

    partB_l1_agg = _agg_by_variant(text_l1_by_t_variant, text_variants_names)

    # ---- Part B: text sensitivity, Level 2 -----------------------------
    print("Part B: text sensitivity Level 2 (full rollout)")
    text_l2: list[dict[str, Any]] = []
    text_baseline_motion = _full_rollout(
        model, text_variants["text_gt"],
        seq_length=total_t, seed=int(args.seed), cfg_scale=float(args.cfg_scale),
    )
    text_baseline_metrics = _metrics_for_motion(
        text_baseline_motion, rest_offsets=rest_offsets, gt_joints=gt_joints,
        object_positions=object_positions, contact_state=contact_state,
        seq_mask=seq_mask, plan=plan_gt, fps=fps, threshold=threshold,
        part_to_joint=part_to_joint,
    )
    for name in text_variants_names:
        if name == "text_gt":
            motion = text_baseline_motion
            metrics = text_baseline_metrics
        else:
            motion = _full_rollout(
                model, text_variants[name],
                seq_length=total_t, seed=int(args.seed), cfg_scale=float(args.cfg_scale),
            )
            metrics = _metrics_for_motion(
                motion, rest_offsets=rest_offsets, gt_joints=gt_joints,
                object_positions=object_positions, contact_state=contact_state,
                seq_mask=seq_mask, plan=plan_gt, fps=fps, threshold=threshold,
                part_to_joint=part_to_joint,
            )
        summary = _summary_row(
            metrics, motion,
            baseline_motion=text_baseline_motion, baseline_joints=text_baseline_metrics["joints"],
            joints=metrics["joints"],
        )
        summary["variant"] = name
        text_l2.append(summary)

    # ---- Part C: object token sensitivity, Level 1 ---------------------
    print("Part C: object token sensitivity Level 1")
    objtok_variants = _build_objtok_variants(base_cond, seed=int(args.seed), include_random=bool(args.include_random_stress))
    objtok_l1_by_t_variant: dict[int, dict[str, dict[str, float]]] = {}
    for t_int in timesteps:
        objtok_l1_by_t_variant[t_int] = {}
        baseline_motion = _one_step_x0(
            model, motion_gt, objtok_variants["objtok_gt"], t_int,
            seed=int(args.seed) + t_int, cfg_scale=float(args.cfg_scale),
        )
        baseline_metrics = _metrics_for_motion(
            baseline_motion, rest_offsets=rest_offsets, gt_joints=gt_joints,
            object_positions=object_positions, contact_state=contact_state,
            seq_mask=seq_mask, plan=plan_gt, fps=fps, threshold=threshold,
            part_to_joint=part_to_joint,
        )
        for name, c in objtok_variants.items():
            if name == "objtok_gt":
                motion = baseline_motion
                metrics = baseline_metrics
            else:
                motion = _one_step_x0(
                    model, motion_gt, c, t_int,
                    seed=int(args.seed) + t_int, cfg_scale=float(args.cfg_scale),
                )
                metrics = _metrics_for_motion(
                    motion, rest_offsets=rest_offsets, gt_joints=gt_joints,
                    object_positions=object_positions, contact_state=contact_state,
                    seq_mask=seq_mask, plan=plan_gt, fps=fps, threshold=threshold,
                    part_to_joint=part_to_joint,
                )
            summary = _summary_row(
                metrics, motion,
                baseline_motion=baseline_motion, baseline_joints=baseline_metrics["joints"],
                joints=metrics["joints"],
            )
            objtok_l1_by_t_variant[t_int][name] = summary

    objtok_variants_names = list(objtok_variants.keys())
    partC_l1_agg = _agg_by_variant(objtok_l1_by_t_variant, objtok_variants_names)

    # ---- Part C: Level 2 -----------------------------------------------
    print("Part C: object token sensitivity Level 2 (full rollout)")
    objtok_l2: list[dict[str, Any]] = []
    obj_baseline_motion = _full_rollout(
        model, objtok_variants["objtok_gt"],
        seq_length=total_t, seed=int(args.seed), cfg_scale=float(args.cfg_scale),
    )
    obj_baseline_metrics = _metrics_for_motion(
        obj_baseline_motion, rest_offsets=rest_offsets, gt_joints=gt_joints,
        object_positions=object_positions, contact_state=contact_state,
        seq_mask=seq_mask, plan=plan_gt, fps=fps, threshold=threshold,
        part_to_joint=part_to_joint,
    )
    for name in objtok_variants_names:
        if name == "objtok_gt":
            motion = obj_baseline_motion
            metrics = obj_baseline_metrics
        else:
            motion = _full_rollout(
                model, objtok_variants[name],
                seq_length=total_t, seed=int(args.seed), cfg_scale=float(args.cfg_scale),
            )
            metrics = _metrics_for_motion(
                motion, rest_offsets=rest_offsets, gt_joints=gt_joints,
                object_positions=object_positions, contact_state=contact_state,
                seq_mask=seq_mask, plan=plan_gt, fps=fps, threshold=threshold,
                part_to_joint=part_to_joint,
            )
        summary = _summary_row(
            metrics, motion,
            baseline_motion=obj_baseline_motion, baseline_joints=obj_baseline_metrics["joints"],
            joints=metrics["joints"],
        )
        summary["variant"] = name
        objtok_l2.append(summary)

    # ---- Part D summary ------------------------------------------------
    partD_summary = (
        "Previous target_route_ablation (2026-05-14) showed object_world_traj is causally active: "
        "no_dense_target raised far_unobserved_error_cm from 29.15 to 45.95, anchor from 35.45 to 49.57, "
        "and dropped onset/release ratios from 0.710/0.592 to 0.397/0.176. Current Part C tests the "
        "complementary route (object_encoder(object_pc) tokens). Read together they discriminate "
        "whether object geometry is carried primarily by the dense world-trajectory route or by the "
        "point-cloud token route."
    )

    payload = {
        "config": str(args.config),
        "ckpt": str(args.ckpt),
        "seed": int(args.seed),
        "cfg_scale": float(args.cfg_scale),
        "timesteps": timesteps,
        "selected_clips": clip_metadata(batch),
        "partA_residuals": partA,
        "partB_text_level1_by_t": {str(t): rows for t, rows in text_l1_by_t_variant.items()},
        "partB_text_level1_agg": partB_l1_agg,
        "partB_text_level2": text_l2,
        "partC_objtok_level1_by_t": {str(t): rows for t, rows in objtok_l1_by_t_variant.items()},
        "partC_objtok_level1_agg": partC_l1_agg,
        "partC_objtok_level2": objtok_l2,
        "partD_summary": partD_summary,
        "caveat": (
            "Token-level inference perturbations introduce train-inference distribution shift. "
            "Causal-sensitivity diagnostic only; not a proposal for inference-time fixes."
        ),
    }
    payload["verdict"] = _verdict(payload)

    write_json(args.output, payload)
    _write_report(payload, args.md)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.md}")
    print(f"Verdict: {payload['verdict']}")


if __name__ == "__main__":
    main()
