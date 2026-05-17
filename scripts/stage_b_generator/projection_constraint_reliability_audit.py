"""ProjFlow-inspired oracle constraint reliability audit (round 5, Diag 1).

Asks: do reliable spatial constraints fix v18's failures, OR are the
constraints themselves unreliable?

Three sub-diagnostics (oracle / inference-only / diagnostic-only):

  1A. Root world-position projection (modify x0[..., 132:135] each step):
      - root_full              : every frame within seq_len
      - root_event_window      : ±window_k frames of any hand contact event

  1B. Post-hoc GT hand joint replacement (Option 1 per spec):
      No rollout change. After baseline rollout, FK x0 → joints,
      then REPLACE generated L/R hand joint positions with GT hand
      positions for metric computation only. Provides an upper-bound:
      "what would the metric look like if the model could perfectly
      follow GT hand spatial constraints?"

  1C. Post-hoc pseudo-target hand joint replacement:
      Same as 1B but uses pseudo-target hand position (z_int
      contact_target_xyz lifted to world) instead of GT hand. Tells us
      whether forcing the hand to follow the *pseudo* target would
      improve metrics. If 1B improves but 1C does not, pseudo targets
      are unreliable.

NOT a sampler proposal. NOT a deployable correction. Projection is
applied AFTER x0 prediction, BEFORE posterior mean, exactly mirroring
oracle_event_guidance_diagnostic.py's hook point.

Outputs:
  analyses/2026-05-15_projection_constraint_reliability_audit.{json,md}
"""
from __future__ import annotations

import argparse
import contextlib
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
    event_records_from_contact,
    extract_plan,
    format_md_table,
    load_checkpoint,
    make_seq_mask,
    merge_single_batches,
    transition_metrics,
    write_json,
)
from dynamics_diagnostic import _build_cond, _build_model, _fk_from_motion_135
from piano.models.motion_anchordiff import _extract
from piano.utils.clip_utils import load_clip_text_encoder
from plan_condition_diagnostics import _compute_metrics as _compute_plan_metrics
from recon_ladder_truncated_rollout_diagnostic import (
    _build_selected_batches,
    _load_selection,
)


PART_TO_JOINT = torch.tensor([20, 21, 10, 11, 0], dtype=torch.long)


def _axis_angle_to_rot_torch(aa: Tensor) -> Tensor:
    """Batched axis-angle to rotation matrix."""
    theta = aa.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    k = aa / theta
    K = torch.zeros(*aa.shape[:-1], 3, 3, device=aa.device, dtype=aa.dtype)
    K[..., 0, 1] = -k[..., 2]; K[..., 0, 2] = k[..., 1]
    K[..., 1, 0] = k[..., 2];  K[..., 1, 2] = -k[..., 0]
    K[..., 2, 0] = -k[..., 1]; K[..., 2, 1] = k[..., 0]
    eye = torch.eye(3, device=aa.device, dtype=aa.dtype).expand_as(K)
    s = theta.sin().unsqueeze(-1)
    c = theta.cos().unsqueeze(-1)
    return eye + s * K + (1 - c) * (K @ K)


# ---------------------------------------------------------------------------
# Sampler — DDPM with optional per-step x0 root projection
# ---------------------------------------------------------------------------


@torch.no_grad()
def _sample_with_root_projection(
    model,
    cond: dict[str, Any],
    *,
    seq_length: int,
    seed: int,
    cfg_scale: float,
    gt_motion: Tensor | None,
    project_mask_b_t: Tensor | None,
) -> Tensor:
    """v18 default DDPM rollout. If `gt_motion` and `project_mask_b_t` are
    given, replace x0[..., 132:135] (root world position) with
    gt_motion[..., 132:135] at every step on frames where
    project_mask_b_t == True.

    project_mask_b_t: (B, T) bool. None → no projection (baseline).
    """
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    device = cond["z_int"].device
    shape = (cond["z_int"].shape[0], int(seq_length), model.cfg.denoiser.motion_dim)
    x = torch.randn(shape, device=device)
    for t_int in reversed(range(model.diffusion.num_steps)):
        t = torch.full((shape[0],), int(t_int), device=device, dtype=torch.long)
        x0 = model.denoiser(x, t, cond, cond_drop_mask=None, self_cond=None)
        if float(cfg_scale) != 1.0:
            drop = torch.ones(shape[0], dtype=torch.bool, device=device)
            x0_unc = model.denoiser(x, t, cond, cond_drop_mask=drop, self_cond=None)
            x0 = x0_unc + float(cfg_scale) * (x0 - x0_unc)
        if gt_motion is not None and project_mask_b_t is not None:
            # Replace root world-position channels on selected frames
            mask = project_mask_b_t.unsqueeze(-1).to(x0.dtype)  # (B, T, 1)
            x0_proj = x0.clone()
            x0_proj[..., 132:135] = (
                gt_motion[..., 132:135] * mask + x0[..., 132:135] * (1.0 - mask)
            )
            x0 = x0_proj
        mean = model.diffusion.posterior_mean_from_x0(x0, x, t)
        if t_int == 0:
            x = mean
        else:
            noise = torch.randn_like(x)
            log_var = _extract(model.diffusion.posterior_log_variance_clipped, t, x.shape)
            x = mean + (0.5 * log_var).exp() * noise
    return x


def _build_event_mask(
    contact_state: Tensor, seq_lens: Tensor, *, threshold: float, window_k: int,
) -> Tensor:
    """(B, T) bool — True within ±window_k frames of any hand onset/release."""
    events = event_records_from_contact(contact_state, seq_lens, threshold=threshold, hands_only=True)
    B = contact_state.shape[0]
    T = contact_state.shape[1]
    mask = torch.zeros(B, T, dtype=torch.bool, device=contact_state.device)
    for ev in events:
        b = int(ev["batch"])
        f = int(ev["frame"])
        lo = max(0, f - int(window_k))
        hi = min(T - 1, f + int(window_k))
        mask[b, lo : hi + 1] = True
    # Mask out frames past seq_len
    arange = torch.arange(T, device=mask.device).view(1, T)
    valid = arange < seq_lens.to(mask.device).long().view(-1, 1)
    return mask & valid


def _build_full_mask(seq_lens: Tensor, T: int) -> Tensor:
    arange = torch.arange(T, device=seq_lens.device).view(1, T)
    return arange < seq_lens.long().view(-1, 1)


# ---------------------------------------------------------------------------
# Post-hoc joint replacement (Option 1)
# ---------------------------------------------------------------------------


def _replace_hand_joints(
    joints_pred: Tensor, replacement_world: Tensor,
    *, hand_joint_indices: list[int], part_indices: list[int],
    seq_mask: Tensor, event_mask: Tensor,
) -> Tensor:
    """Replace `joints_pred[b, t, hand_joint]` with replacement_world[b, t, part_idx]
    on frames where seq_mask AND event_mask are True. All other frames untouched.

    joints_pred:       (B, T, 22, 3)
    replacement_world: (B, T, P, 3) — usually pseudo target_world or GT hand
    event_mask:        (B, T) bool — only replace within event windows
    """
    out = joints_pred.clone()
    apply = (seq_mask & event_mask).unsqueeze(-1)  # (B, T, 1)
    for j_idx, p_idx in zip(hand_joint_indices, part_indices):
        ja = apply  # (B, T, 1)
        out[..., j_idx, :] = (
            replacement_world[..., p_idx, :] * ja + joints_pred[..., j_idx, :] * (1.0 - ja.to(out.dtype))
        )
    return out


# ---------------------------------------------------------------------------
# Metrics wrapper
# ---------------------------------------------------------------------------


def _metrics_for_joints(
    joints: Tensor,
    *,
    motion_for_delta: Tensor | None,
    baseline_motion: Tensor | None,
    gt_joints: Tensor,
    object_positions: Tensor,
    contact_state: Tensor,
    seq_mask: Tensor,
    plan: dict[str, Tensor],
    fps: float,
    threshold: float,
    part_to_joint: Tensor,
) -> dict[str, float]:
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
    tr = trans.get("ratios_over_gt", {})
    summary = {
        "body_velocity_over_gt": float(dyn.get("body_velocity_cm_per_frame_over_gt", 0.0)),
        "hand_velocity_over_gt": float(dyn.get("hand_velocity_cm_per_frame_over_gt", 0.0)),
        "body_acc_p95_over_gt": float(dyn.get("body_acc_p95_cm_per_frame2_over_gt", 0.0)),
        "body_jerk_p95_over_gt": float(dyn.get("body_jerk_p95_cm_per_frame3_over_gt", 0.0)),
        "onset_xgt": float(tr.get("onset_positive_closing", 0.0)),
        "release_xgt": float(tr.get("release_positive_opening", 0.0)),
        "transition_relvel_xgt": float(tr.get("transition_relative_velocity", 0.0)),
        "far_unobserved_error_cm": float(plan_m.get("far_unobserved_error_cm", 0.0)),
        "near_anchor_window_error_cm": float(plan_m.get("near_anchor_window_error_cm", 0.0)),
        "anchor_realization_cm": float(plan_m.get("plan_anchor_contact_realization_cm", 0.0)),
    }
    if motion_for_delta is not None and baseline_motion is not None:
        summary["motion135_delta_vs_baseline"] = float(
            torch.linalg.vector_norm(
                (motion_for_delta - baseline_motion).reshape(motion_for_delta.shape[0], -1), dim=-1,
            ).mean().item()
        )
    return summary


def _parse_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/training/anchordiff_v18_a1_FULL_DATA.yaml"))
    parser.add_argument("--ckpt", type=Path, default=Path("runs/training/stageB_anchordiff_v18_a1_FULL_DATA/final.pt"))
    parser.add_argument("--output", type=Path, default=Path("analyses/2026-05-15_projection_constraint_reliability_audit.json"))
    parser.add_argument("--md", type=Path, default=Path("analyses/2026-05-15_projection_constraint_reliability_audit.md"))
    parser.add_argument("--selection-json", type=Path, default=Path("analyses/2026-05-13_v18_recon_ladder_truncated_rollout.json"))
    parser.add_argument("--max-clips", type=int, default=16)
    parser.add_argument("--num-candidates", type=int, default=256)
    parser.add_argument("--bucket", choices=["train", "val"], default="train")
    parser.add_argument("--balanced-subsets", action="store_true", default=True)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--event-window", type=int, default=10,
                        help="±frames around hand events for root_event_window projection.")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    selection = _load_selection(args.selection_json, max_clips=int(args.max_clips))
    selected = _build_selected_batches(
        cfg, bucket=args.bucket, balanced_subsets=bool(args.balanced_subsets),
        num_candidates=int(args.num_candidates), selection=selection,
        max_clips=int(args.max_clips), threshold=float(args.threshold),
    )
    if not selected:
        raise SystemExit("No clips selected")
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
    object_rotations = batch["object_rotations"].to(device).float()
    contact_state = batch["contact_state"].to(device).float()
    seq_lens = batch["seq_len"].to(device)
    seq_mask = make_seq_mask(seq_lens, total_t, device)
    part_to_joint = PART_TO_JOINT.to(device)

    # Pseudo target_world via lift of contact_target_xyz_local + object pose.
    contact_target_local = batch["contact_target_xyz"].to(device).float()  # (B, T, 5, 3)
    R_obj = _axis_angle_to_rot_torch(object_rotations)                    # (B, T, 3, 3)
    pseudo_target_world = (
        torch.einsum("btij,btkj->btki", R_obj, contact_target_local)
        + object_positions.unsqueeze(-2)
    )  # (B, T, 5, 3)

    # Masks
    full_mask = _build_full_mask(seq_lens, total_t)
    event_mask = _build_event_mask(contact_state, seq_lens, threshold=float(args.threshold), window_k=int(args.event_window))

    seeds = _parse_ints(args.seeds)
    fps = float(args.fps)
    threshold = float(args.threshold)

    all_results: dict[str, dict[int, dict[str, float]]] = {}

    for seed in seeds:
        print(f"=== seed {seed} ===")
        # baseline (no projection)
        baseline_motion = _sample_with_root_projection(
            model, base_cond, seq_length=total_t,
            seed=int(seed), cfg_scale=float(args.cfg_scale),
            gt_motion=None, project_mask_b_t=None,
        )
        baseline_joints = _fk_from_motion_135(baseline_motion, rest_offsets)
        baseline_metrics = _metrics_for_joints(
            baseline_joints,
            motion_for_delta=baseline_motion, baseline_motion=baseline_motion,
            gt_joints=gt_joints, object_positions=object_positions,
            contact_state=contact_state, seq_mask=seq_mask, plan=plan_gt,
            fps=fps, threshold=threshold, part_to_joint=part_to_joint,
        )

        # root_full
        print("  variant root_full ...")
        rootfull_motion = _sample_with_root_projection(
            model, base_cond, seq_length=total_t,
            seed=int(seed), cfg_scale=float(args.cfg_scale),
            gt_motion=motion_gt, project_mask_b_t=full_mask,
        )
        rootfull_joints = _fk_from_motion_135(rootfull_motion, rest_offsets)
        rootfull_metrics = _metrics_for_joints(
            rootfull_joints,
            motion_for_delta=rootfull_motion, baseline_motion=baseline_motion,
            gt_joints=gt_joints, object_positions=object_positions,
            contact_state=contact_state, seq_mask=seq_mask, plan=plan_gt,
            fps=fps, threshold=threshold, part_to_joint=part_to_joint,
        )

        # root_event_window
        print("  variant root_event_window ...")
        rootevt_motion = _sample_with_root_projection(
            model, base_cond, seq_length=total_t,
            seed=int(seed), cfg_scale=float(args.cfg_scale),
            gt_motion=motion_gt, project_mask_b_t=event_mask,
        )
        rootevt_joints = _fk_from_motion_135(rootevt_motion, rest_offsets)
        rootevt_metrics = _metrics_for_joints(
            rootevt_joints,
            motion_for_delta=rootevt_motion, baseline_motion=baseline_motion,
            gt_joints=gt_joints, object_positions=object_positions,
            contact_state=contact_state, seq_mask=seq_mask, plan=plan_gt,
            fps=fps, threshold=threshold, part_to_joint=part_to_joint,
        )

        # 1B: post-hoc GT hand replacement on baseline rollout (no extra rollout)
        print("  variant 1B post_hoc_gt_hand ...")
        # Build a (B, T, P, 3) replacement that has GT hand at part indices 0,1 (L/R hand) and zeros elsewhere
        gt_replacement_world = torch.zeros(*gt_joints.shape[:2], 5, 3, device=device, dtype=gt_joints.dtype)
        gt_replacement_world[..., 0, :] = gt_joints[..., 20, :]   # L_hand
        gt_replacement_world[..., 1, :] = gt_joints[..., 21, :]   # R_hand
        bg_event = _replace_hand_joints(
            baseline_joints, gt_replacement_world,
            hand_joint_indices=[20, 21], part_indices=[0, 1],
            seq_mask=seq_mask, event_mask=event_mask,
        )
        bg_event_metrics = _metrics_for_joints(
            bg_event,
            motion_for_delta=None, baseline_motion=None,
            gt_joints=gt_joints, object_positions=object_positions,
            contact_state=contact_state, seq_mask=seq_mask, plan=plan_gt,
            fps=fps, threshold=threshold, part_to_joint=part_to_joint,
        )

        # 1C: post-hoc pseudo target_world replacement on baseline rollout
        print("  variant 1C post_hoc_pseudo_target ...")
        bg_pseudo = _replace_hand_joints(
            baseline_joints, pseudo_target_world,
            hand_joint_indices=[20, 21], part_indices=[0, 1],
            seq_mask=seq_mask, event_mask=event_mask,
        )
        bg_pseudo_metrics = _metrics_for_joints(
            bg_pseudo,
            motion_for_delta=None, baseline_motion=None,
            gt_joints=gt_joints, object_positions=object_positions,
            contact_state=contact_state, seq_mask=seq_mask, plan=plan_gt,
            fps=fps, threshold=threshold, part_to_joint=part_to_joint,
        )

        all_results.setdefault("baseline", {})[seed] = baseline_metrics
        all_results.setdefault("root_full", {})[seed] = rootfull_metrics
        all_results.setdefault("root_event_window", {})[seed] = rootevt_metrics
        all_results.setdefault("post_hoc_gt_hand_event_window", {})[seed] = bg_event_metrics
        all_results.setdefault("post_hoc_pseudo_target_event_window", {})[seed] = bg_pseudo_metrics

    # Aggregate across seeds
    variants = [
        "baseline", "root_full", "root_event_window",
        "post_hoc_gt_hand_event_window", "post_hoc_pseudo_target_event_window",
    ]
    metric_keys = sorted({k for v in all_results.values() for d in v.values() for k in d.keys()})
    agg: dict[str, dict[str, float]] = {}
    for var in variants:
        per_seed = all_results.get(var, {})
        if not per_seed:
            continue
        agg[var] = {}
        for k in metric_keys:
            vals = [d.get(k, 0.0) for d in per_seed.values()]
            agg[var][f"{k}_mean"] = float(np.mean(vals))
            agg[var][f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    payload = {
        "config": str(args.config),
        "ckpt": str(args.ckpt),
        "seeds": seeds,
        "event_window": int(args.event_window),
        "selected_clips": clip_metadata(batch),
        "per_seed": all_results,
        "aggregate": agg,
        "caveat": (
            "1A root projection modifies x0 root world position during DDPM rollout — diagnostic only, "
            "introduces train-inference distribution shift. 1B/1C post-hoc joint replacement does NOT "
            "produce a valid motion sample; it gives an upper-bound on what perfect hand-spatial "
            "constraints could yield on the contact metric only."
        ),
    }
    write_json(args.output, payload)

    # Markdown
    lines: list[str] = [
        "# ProjFlow-Inspired Constraint Reliability Audit",
        "",
        f"- Config: `{args.config}`",
        f"- Checkpoint: `{args.ckpt}`",
        f"- Seeds: {seeds}",
        f"- Event window: ±{int(args.event_window)} frames around hand onset/release",
        f"- Clips: {len(payload['selected_clips'])} (subset composition documented in payload)",
        "",
        "## Caveat",
        "",
        payload["caveat"],
        "",
        "## Aggregate metrics across seeds (mean ± std)",
        "",
    ]
    rows = [
        ["variant", "onset xGT", "release xGT", "transvel xGT", "body xGT", "hand xGT",
         "acc p95 xGT", "far cm", "anchor cm", "Δm135 vs baseline"]
    ]
    for var in variants:
        if var not in agg:
            continue
        a = agg[var]
        def _fmt(k: str) -> str:
            m = a.get(f"{k}_mean", 0.0)
            s = a.get(f"{k}_std", 0.0)
            return f"{m:.3f}±{s:.3f}"
        rows.append([
            var,
            _fmt("onset_xgt"), _fmt("release_xgt"), _fmt("transition_relvel_xgt"),
            _fmt("body_velocity_over_gt"), _fmt("hand_velocity_over_gt"),
            _fmt("body_acc_p95_over_gt"),
            f"{a.get('far_unobserved_error_cm_mean', 0.0):.2f}±{a.get('far_unobserved_error_cm_std', 0.0):.2f}",
            f"{a.get('anchor_realization_cm_mean', 0.0):.2f}±{a.get('anchor_realization_cm_std', 0.0):.2f}",
            _fmt("motion135_delta_vs_baseline"),
        ])
    lines.append(format_md_table(rows))
    lines.append("")

    # Interpretation
    base = agg.get("baseline", {})
    rf = agg.get("root_full", {})
    rev = agg.get("root_event_window", {})
    pg = agg.get("post_hoc_gt_hand_event_window", {})
    pp = agg.get("post_hoc_pseudo_target_event_window", {})

    def m(d: dict[str, float], k: str) -> float: return float(d.get(f"{k}_mean", 0.0))

    interp: list[str] = ["## Interpretation", ""]
    interp.append("### 1A root projection")
    interp.append(
        f"- baseline far={m(base,'far_unobserved_error_cm'):.2f} cm, anchor={m(base,'anchor_realization_cm'):.2f} cm"
    )
    interp.append(
        f"- root_full far={m(rf,'far_unobserved_error_cm'):.2f} (Δ={m(rf,'far_unobserved_error_cm')-m(base,'far_unobserved_error_cm'):+.2f}), "
        f"anchor={m(rf,'anchor_realization_cm'):.2f} (Δ={m(rf,'anchor_realization_cm')-m(base,'anchor_realization_cm'):+.2f})"
    )
    interp.append(
        f"- root_event_window far={m(rev,'far_unobserved_error_cm'):.2f} (Δ={m(rev,'far_unobserved_error_cm')-m(base,'far_unobserved_error_cm'):+.2f}), "
        f"anchor={m(rev,'anchor_realization_cm'):.2f} (Δ={m(rev,'anchor_realization_cm')-m(base,'anchor_realization_cm'):+.2f})"
    )
    interp.append("")
    interp.append("### 1B post-hoc GT hand joint replacement (upper bound)")
    interp.append(
        f"- baseline onset={m(base,'onset_xgt'):.3f}, release={m(base,'release_xgt'):.3f}, transvel={m(base,'transition_relvel_xgt'):.3f}"
    )
    interp.append(
        f"- post_hoc_gt_hand onset={m(pg,'onset_xgt'):.3f} (Δ={m(pg,'onset_xgt')-m(base,'onset_xgt'):+.3f}), "
        f"release={m(pg,'release_xgt'):.3f} (Δ={m(pg,'release_xgt')-m(base,'release_xgt'):+.3f}), "
        f"transvel={m(pg,'transition_relvel_xgt'):.3f} (Δ={m(pg,'transition_relvel_xgt')-m(base,'transition_relvel_xgt'):+.3f})"
    )
    interp.append("")
    interp.append("### 1C post-hoc pseudo target_world replacement")
    interp.append(
        f"- post_hoc_pseudo onset={m(pp,'onset_xgt'):.3f} (Δ={m(pp,'onset_xgt')-m(base,'onset_xgt'):+.3f}), "
        f"release={m(pp,'release_xgt'):.3f} (Δ={m(pp,'release_xgt')-m(base,'release_xgt'):+.3f}), "
        f"transvel={m(pp,'transition_relvel_xgt'):.3f} (Δ={m(pp,'transition_relvel_xgt')-m(base,'transition_relvel_xgt'):+.3f})"
    )
    interp.append("")
    interp.append("### Cross-comparison verdict")
    gt_helps = (m(pg,'onset_xgt') > m(base,'onset_xgt') + 0.05) or (m(pg,'release_xgt') > m(base,'release_xgt') + 0.05)
    pseudo_helps = (m(pp,'onset_xgt') > m(base,'onset_xgt') + 0.05) or (m(pp,'release_xgt') > m(base,'release_xgt') + 0.05)
    root_helps_geom = (m(rf,'far_unobserved_error_cm') < m(base,'far_unobserved_error_cm') - 1.0) or \
                      (m(rf,'anchor_realization_cm') < m(base,'anchor_realization_cm') - 1.0)
    if gt_helps and not pseudo_helps:
        verdict = "**Case A**: GT hand oracle improves metrics but pseudo target does not. Pseudo-label / plan target geometry is unreliable; fix targets before any training-side change."
    elif gt_helps and pseudo_helps:
        verdict = "**Case A partial**: Both GT and pseudo target hand replacements improve metrics. Reliable spatial constraints can help, but model is not currently using them — investigate conditioning/sampling."
    elif not gt_helps:
        verdict = "**Case B**: GT hand oracle does NOT improve metrics meaningfully. Either projection method is invalid, metric/event windows are flawed, or failure is not spatial-hand-target-based. Fix metric/visual diagnosis first."
    else:
        verdict = "**Mixed**: see per-metric Δ table."
    if root_helps_geom:
        verdict += " Root projection meaningfully reduces far/anchor error → global placement / root drift is a confound."
    else:
        verdict += " Root projection barely changes far/anchor → root drift is NOT the dominant geometry failure."
    interp.append(verdict)
    interp.append("")
    lines.extend(interp)

    # Selected clips
    lines.append("## Selected clips")
    lines.append("")
    rows = [["subset", "seq_id", "seq_len", "text"]]
    for c in payload["selected_clips"]:
        rows.append([c["subset"], c["seq_id"], c["seq_len"], c["text"][:80]])
    lines.append(format_md_table(rows))
    lines.append("")

    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.md}")


if __name__ == "__main__":
    main()
