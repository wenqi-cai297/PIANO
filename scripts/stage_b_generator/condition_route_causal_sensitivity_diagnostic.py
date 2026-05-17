"""Condition-route causal sensitivity diagnostic for Stage B v18.

No-training diagnostic. For each selected transition-heavy clip we measure
how much x0 prediction (Level 1: one-step under q(x_t|x0)) and full DDPM
rollout (Level 2) respond to small scalings of:

  Perturbation A: alpha_hint on hint_proj(plan_hint) only (post-projection)
  Perturbation B: alpha_z_target on z_int[:, :, 5:20] contact_target_xyz
  Perturbation C (optional, cheap): alpha_dense_target on
                                    object_world_traj[..., 9:]
                                    and alpha_plan_target on
                                    anchor_target_local/world.

The alpha_hint hook is implemented as a forward_pre_hook on
``model.denoiser.v12_input_proj`` so that the same scaling applies at
every denoiser call inside the DDPM rollout. All hooks are removed in a
``finally`` block; no model parameters are modified.

CAVEAT (printed into the report): inference-only scaling introduces
train-inference distribution shift. The point is to discriminate
"route exists" vs "route ignored" vs "route exists but rollout
suppresses it"; not to ship inference-time scaling as a method.
"""
from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
from typing import Any, Callable

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
    safe_div,
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
# Perturbation hooks
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def scaled_hint(model: torch.nn.Module, alpha: float):
    """Forward pre-hook on v12_input_proj that scales plan_hint only.

    V12InputProjection.forward signature:
      forward(x_t, z_int, obj_traj, plan_hint, self_cond=None)
    """
    if abs(float(alpha) - 1.0) < 1e-9:
        yield
        return

    def _hook(_module, args, kwargs):
        a = float(alpha)
        if "plan_hint" in kwargs:
            kwargs["plan_hint"] = kwargs["plan_hint"] * a
        elif len(args) >= 4:
            new_args = list(args)
            new_args[3] = new_args[3] * a
            args = tuple(new_args)
        return args, kwargs

    handle = model.denoiser.v12_input_proj.register_forward_pre_hook(
        _hook, with_kwargs=True,
    )
    try:
        yield
    finally:
        handle.remove()


def _apply_zint_target_scale(cond: dict[str, Any], alpha: float) -> dict[str, Any]:
    """Return a shallow-cloned cond with z_int[:, :, 5:20] scaled by alpha.

    Other z_int channels (contact_state, phase, support), object_world_traj
    and interaction_plan are untouched.
    """
    out: dict[str, Any] = {k: v for k, v in cond.items()}
    z_int = out["z_int"].clone()
    z_int[..., 5:20] = z_int[..., 5:20] * float(alpha)
    out["z_int"] = z_int
    return out


def _apply_dense_target_scale(cond: dict[str, Any], alpha: float) -> dict[str, Any]:
    out: dict[str, Any] = {k: v for k, v in cond.items()}
    obj = out["object_world_traj"].clone()
    obj[..., 9:] = obj[..., 9:] * float(alpha)
    out["object_world_traj"] = obj
    return out


def _apply_plan_target_scale(cond: dict[str, Any], alpha: float) -> dict[str, Any]:
    out: dict[str, Any] = {k: v for k, v in cond.items()}
    plan = {k: v.clone() if isinstance(v, Tensor) else v for k, v in out["interaction_plan"].items()}
    plan["anchor_target_local"] = plan["anchor_target_local"] * float(alpha)
    plan["anchor_target_world"] = plan["anchor_target_world"] * float(alpha)
    out["interaction_plan"] = plan
    return out


# ---------------------------------------------------------------------------
# Level 1: one-step reconstruction sensitivity
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
    alpha_hint: float,
) -> Tensor:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    t = torch.full(
        (motion_gt.shape[0],), int(t_value), device=motion_gt.device, dtype=torch.long,
    )
    noise = torch.randn_like(motion_gt)
    x_t = model.diffusion.q_sample(motion_gt, t, noise)
    with scaled_hint(model, alpha_hint):
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


# ---------------------------------------------------------------------------
# Level 2: full DDPM rollout
# ---------------------------------------------------------------------------


@torch.no_grad()
def _full_rollout(
    model,
    cond: dict[str, Any],
    *,
    seq_length: int,
    seed: int,
    cfg_scale: float,
    alpha_hint: float,
    sampler: str = "ddpm",
) -> Tensor:
    """Full reverse rollout. ``sampler`` is forwarded to ``model.sample`` —
    use 'ddpm' for the v18 default (ancestral) and 'ddim_eta0' for the
    deterministic DDIM η=0 variant.

    DDIM η=0 is a no-training diagnostic-only sampler: it removes the
    posterior-variance noise from the v18 default so route-causality
    can be measured without sampler stochasticity. A positive DDIM
    result does NOT by itself justify training or inference changes;
    it just isolates whether DDPM stochasticity is washing out a real
    route signal.
    """
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    with scaled_hint(model, alpha_hint):
        motion = model.sample(
            cond=cond, seq_length=int(seq_length), cfg_scale=float(cfg_scale),
            sampler=str(sampler),
        )
    return motion


# ---------------------------------------------------------------------------
# Metric harness
# ---------------------------------------------------------------------------


def _metrics_for_motion(
    motion: Tensor,
    *,
    rest_offsets: Tensor,
    gt_joints: Tensor,
    object_positions: Tensor,
    contact_state: Tensor,
    seq_mask: Tensor,
    plan: dict[str, Tensor],
    fps: float,
    threshold: float,
    part_to_joint: Tensor,
) -> dict[str, Any]:
    joints = _fk_from_motion_135(motion, rest_offsets)
    dyn = dynamics_metrics(joints, seq_mask, gt_joints=gt_joints, fps=fps)
    trans = transition_metrics(
        joints,
        object_positions,
        contact_state,
        seq_mask,
        gt_joints=gt_joints,
        window_k=10,
        threshold=threshold,
    )
    plan_m = _compute_plan_metrics(
        jpos_pred=joints,
        jpos_gt=gt_joints,
        seq_mask=seq_mask,
        anchor_time=plan["anchor_time"],
        anchor_mask=plan["anchor_mask"],
        anchor_part=plan["anchor_part"],
        anchor_target_world=plan["anchor_target_world"],
        part_to_joint=part_to_joint,
        window=3,
    )
    return {"dynamics": dyn, "transition": trans, "plan": plan_m, "joints": joints}


def _per_clip_metrics(
    motion: Tensor,
    *,
    rest_offsets: Tensor,
    gt_joints: Tensor,
    object_positions: Tensor,
    contact_state: Tensor,
    seq_mask: Tensor,
    plan: dict[str, Tensor],
    fps: float,
    threshold: float,
    part_to_joint: Tensor,
    baseline_motion: Tensor | None,
    seq_ids: list[str],
    subsets: list[str],
) -> list[dict[str, Any]]:
    """Compute the same metrics as ``_metrics_for_motion`` but per-clip.

    Slices the batch dim and re-uses dynamics_metrics / transition_metrics /
    _compute_plan_metrics with batch=1 for each clip. The result is a list of
    dicts (one per clip) suitable for bootstrap / paired-Δ analysis in
    ``diagnostic_variance_analyzer.py``.
    """
    B = motion.shape[0]
    rows: list[dict[str, Any]] = []
    for b in range(B):
        m_b = motion[b : b + 1]
        gt_b = gt_joints[b : b + 1]
        ro_b = rest_offsets[b : b + 1]
        op_b = object_positions[b : b + 1]
        cs_b = contact_state[b : b + 1]
        sm_b = seq_mask[b : b + 1]
        plan_b = {
            k: (v[b : b + 1] if isinstance(v, Tensor) else v)
            for k, v in plan.items()
        }
        metrics_b = _metrics_for_motion(
            m_b, rest_offsets=ro_b, gt_joints=gt_b, object_positions=op_b,
            contact_state=cs_b, seq_mask=sm_b, plan=plan_b, fps=fps,
            threshold=threshold, part_to_joint=part_to_joint,
        )
        row = _summary_row(
            metrics_b, m_b,
            baseline_motion=(baseline_motion[b : b + 1] if baseline_motion is not None else None),
            baseline_joints=None,
            joints=metrics_b["joints"],
        )
        row["clip_idx"] = b
        row["seq_id"] = str(seq_ids[b]) if b < len(seq_ids) else ""
        row["subset"] = str(subsets[b]) if b < len(subsets) else ""
        rows.append(row)
    return rows


def _summary_row(
    row: dict[str, Any],
    motion: Tensor,
    *,
    baseline_motion: Tensor | None,
    baseline_joints: Tensor | None,
    joints: Tensor,
) -> dict[str, float]:
    dyn = row["dynamics"]
    trans = row["transition"].get("ratios_over_gt", {})
    plan_m = row["plan"]
    summary = {
        "body_velocity_over_gt": float(dyn.get("body_velocity_cm_per_frame_over_gt", 0.0)),
        "hand_velocity_over_gt": float(dyn.get("hand_velocity_cm_per_frame_over_gt", 0.0)),
        "body_acc_p95_over_gt": float(dyn.get("body_acc_p95_cm_per_frame2_over_gt", 0.0)),
        "body_jerk_p95_over_gt": float(dyn.get("body_jerk_p95_cm_per_frame3_over_gt", 0.0)),
        "fft_low": float(dyn.get("fft_low", 0.0)),
        "fft_mid": float(dyn.get("fft_mid", 0.0)),
        "fft_high": float(dyn.get("fft_high", 0.0)),
        "onset_xgt": float(trans.get("onset_positive_closing", 0.0)),
        "release_xgt": float(trans.get("release_positive_opening", 0.0)),
        "transition_relvel_xgt": float(trans.get("transition_relative_velocity", 0.0)),
        "far_unobserved_error_cm": float(plan_m.get("far_unobserved_error_cm", 0.0)),
        "near_anchor_window_error_cm": float(plan_m.get("near_anchor_window_error_cm", 0.0)),
        "anchor_realization_cm": float(plan_m.get("plan_anchor_contact_realization_cm", 0.0)),
    }
    if baseline_motion is not None:
        summary["motion135_delta_vs_alpha1"] = float(
            torch.linalg.vector_norm(
                (motion - baseline_motion).reshape(motion.shape[0], -1), dim=-1,
            ).mean().item()
        )
    if baseline_joints is not None:
        summary["joint_delta_vs_alpha1_cm"] = float(
            torch.linalg.vector_norm(
                (joints - baseline_joints).reshape(joints.shape[0], -1), dim=-1,
            ).mean().item() * 100.0
        )
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_floats(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _write_report(payload: dict[str, Any], path: Path) -> None:
    lines: list[str] = [
        "# Condition-Route Causal Sensitivity Diagnostic",
        "",
        f"- Config: `{payload['config']}`",
        f"- Checkpoint: `{payload['ckpt']}`",
        f"- Seed: {payload['seed']}",
        f"- Sampler: `{payload.get('sampler', 'ddpm')}` "
        f"({'v18 default ancestral DDPM' if payload.get('sampler', 'ddpm') == 'ddpm' else 'DIAGNOSTIC-ONLY deterministic DDIM η=0; not the v18 default'})",
        f"- Clips: {len(payload['selected_clips'])}",
        "",
        "## Distribution-shift caveat",
        "",
        "Inference-only scaling of projected condition branches introduces "
        "train-inference distribution shift. These results are causal-sensitivity "
        "evidence ONLY; they do not propose an inference-time method.",
        "",
    ]

    # alpha_hint Level 1
    lines.extend(["## Perturbation A: alpha_hint, Level 1 one-step (timestep-aggregated)", ""])
    rows = [["alpha", "body xGT", "hand xGT", "acc p95 xGT", "jerk p95 xGT", "onset xGT", "release xGT", "transvel xGT", "far cm", "anchor cm", "Δm135 vs α=1", "Δjoint cm vs α=1"]]
    for row in payload["alpha_hint_level1_agg"]:
        rows.append([
            f"{row['alpha']:.2f}",
            f"{row['body_velocity_over_gt']:.3f}",
            f"{row['hand_velocity_over_gt']:.3f}",
            f"{row['body_acc_p95_over_gt']:.3f}",
            f"{row['body_jerk_p95_over_gt']:.3f}",
            f"{row['onset_xgt']:.3f}",
            f"{row['release_xgt']:.3f}",
            f"{row['transition_relvel_xgt']:.3f}",
            f"{row['far_unobserved_error_cm']:.2f}",
            f"{row['anchor_realization_cm']:.2f}",
            f"{row.get('motion135_delta_vs_alpha1', 0.0):.4f}",
            f"{row.get('joint_delta_vs_alpha1_cm', 0.0):.3f}",
        ])
    lines.append(format_md_table(rows))
    lines.append("")

    # Per-timestep alpha_hint
    lines.extend(["## Perturbation A: alpha_hint, Level 1 per-timestep Δmotion-135", ""])
    rows = [["t"] + [f"α={a:.2f}" for a in payload["alpha_hint"]]]
    for t, row in payload["alpha_hint_level1_by_t"].items():
        rows.append([
            str(t),
            *[f"{row[a]['motion135_delta_vs_alpha1']:.4f}" for a in [str(x) for x in payload["alpha_hint"]]],
        ])
    lines.append(format_md_table(rows))
    lines.append("")

    # alpha_hint Level 2
    lines.extend(["## Perturbation A: alpha_hint, Level 2 full DDPM rollout", ""])
    rows = [["alpha", "body xGT", "hand xGT", "acc p95 xGT", "jerk p95 xGT", "onset xGT", "release xGT", "transvel xGT", "far cm", "anchor cm", "Δm135 vs α=1", "Δjoint cm"]]
    for row in payload["alpha_hint_level2"]:
        rows.append([
            f"{row['alpha']:.2f}",
            f"{row['body_velocity_over_gt']:.3f}",
            f"{row['hand_velocity_over_gt']:.3f}",
            f"{row['body_acc_p95_over_gt']:.3f}",
            f"{row['body_jerk_p95_over_gt']:.3f}",
            f"{row['onset_xgt']:.3f}",
            f"{row['release_xgt']:.3f}",
            f"{row['transition_relvel_xgt']:.3f}",
            f"{row['far_unobserved_error_cm']:.2f}",
            f"{row['anchor_realization_cm']:.2f}",
            f"{row.get('motion135_delta_vs_alpha1', 0.0):.4f}",
            f"{row.get('joint_delta_vs_alpha1_cm', 0.0):.3f}",
        ])
    lines.append(format_md_table(rows))
    lines.append("")

    # alpha_z_target Level 1
    lines.extend(["## Perturbation B: alpha_z_target, Level 1 one-step (timestep-aggregated)", ""])
    rows = [["alpha", "body xGT", "hand xGT", "acc p95 xGT", "jerk p95 xGT", "onset xGT", "release xGT", "transvel xGT", "far cm", "anchor cm", "Δm135", "Δjoint cm"]]
    for row in payload["alpha_z_target_level1_agg"]:
        rows.append([
            f"{row['alpha']:.2f}",
            f"{row['body_velocity_over_gt']:.3f}",
            f"{row['hand_velocity_over_gt']:.3f}",
            f"{row['body_acc_p95_over_gt']:.3f}",
            f"{row['body_jerk_p95_over_gt']:.3f}",
            f"{row['onset_xgt']:.3f}",
            f"{row['release_xgt']:.3f}",
            f"{row['transition_relvel_xgt']:.3f}",
            f"{row['far_unobserved_error_cm']:.2f}",
            f"{row['anchor_realization_cm']:.2f}",
            f"{row.get('motion135_delta_vs_alpha1', 0.0):.4f}",
            f"{row.get('joint_delta_vs_alpha1_cm', 0.0):.3f}",
        ])
    lines.append(format_md_table(rows))
    lines.append("")

    # alpha_z_target Level 2
    lines.extend(["## Perturbation B: alpha_z_target, Level 2 full DDPM rollout", ""])
    rows = [["alpha", "body xGT", "hand xGT", "acc p95 xGT", "jerk p95 xGT", "onset xGT", "release xGT", "transvel xGT", "far cm", "anchor cm", "Δm135", "Δjoint cm"]]
    for row in payload["alpha_z_target_level2"]:
        rows.append([
            f"{row['alpha']:.2f}",
            f"{row['body_velocity_over_gt']:.3f}",
            f"{row['hand_velocity_over_gt']:.3f}",
            f"{row['body_acc_p95_over_gt']:.3f}",
            f"{row['body_jerk_p95_over_gt']:.3f}",
            f"{row['onset_xgt']:.3f}",
            f"{row['release_xgt']:.3f}",
            f"{row['transition_relvel_xgt']:.3f}",
            f"{row['far_unobserved_error_cm']:.2f}",
            f"{row['anchor_realization_cm']:.2f}",
            f"{row.get('motion135_delta_vs_alpha1', 0.0):.4f}",
            f"{row.get('joint_delta_vs_alpha1_cm', 0.0):.3f}",
        ])
    lines.append(format_md_table(rows))
    lines.append("")

    # Optional C — dense / plan target
    if payload.get("alpha_dense_target_level2"):
        lines.extend(["## Perturbation C1: alpha_dense_target (object_world_traj[..., 9:]) Level 2", ""])
        rows = [["alpha", "body xGT", "hand xGT", "onset xGT", "release xGT", "far cm", "anchor cm", "Δm135"]]
        for row in payload["alpha_dense_target_level2"]:
            rows.append([
                f"{row['alpha']:.2f}",
                f"{row['body_velocity_over_gt']:.3f}",
                f"{row['hand_velocity_over_gt']:.3f}",
                f"{row['onset_xgt']:.3f}",
                f"{row['release_xgt']:.3f}",
                f"{row['far_unobserved_error_cm']:.2f}",
                f"{row['anchor_realization_cm']:.2f}",
                f"{row.get('motion135_delta_vs_alpha1', 0.0):.4f}",
            ])
        lines.append(format_md_table(rows))
        lines.append("")

    if payload.get("alpha_plan_target_level2"):
        lines.extend(["## Perturbation C2: alpha_plan_target (anchor_target_local/world) Level 2", ""])
        rows = [["alpha", "body xGT", "hand xGT", "onset xGT", "release xGT", "far cm", "anchor cm", "Δm135"]]
        for row in payload["alpha_plan_target_level2"]:
            rows.append([
                f"{row['alpha']:.2f}",
                f"{row['body_velocity_over_gt']:.3f}",
                f"{row['hand_velocity_over_gt']:.3f}",
                f"{row['onset_xgt']:.3f}",
                f"{row['release_xgt']:.3f}",
                f"{row['far_unobserved_error_cm']:.2f}",
                f"{row['anchor_realization_cm']:.2f}",
                f"{row.get('motion135_delta_vs_alpha1', 0.0):.4f}",
            ])
        lines.append(format_md_table(rows))
        lines.append("")

    lines.extend(["## Selected Clips", ""])
    rows = [["subset", "seq_id", "seq_len", "text"]]
    for c in payload["selected_clips"]:
        rows.append([c["subset"], c["seq_id"], c["seq_len"], c["text"][:80]])
    lines.append(format_md_table(rows))
    lines.append("")

    lines.extend(["## Verdict", "", payload["verdict"], ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _verdict(payload: dict[str, Any]) -> str:
    def get(rows: list[dict[str, Any]], alpha: float, key: str) -> float:
        for r in rows:
            if abs(float(r["alpha"]) - float(alpha)) < 1e-6:
                return float(r.get(key, 0.0))
        return 0.0

    hint1 = payload["alpha_hint_level1_agg"]
    hint2 = payload["alpha_hint_level2"]
    z1 = payload["alpha_z_target_level1_agg"]
    z2 = payload["alpha_z_target_level2"]

    # alpha_hint reaction strength: Δm135 at alpha=1.5 (or 1.25) vs alpha=1
    delta_hint_1step_15 = get(hint1, 1.5, "motion135_delta_vs_alpha1")
    delta_hint_1step_125 = get(hint1, 1.25, "motion135_delta_vs_alpha1")
    delta_hint_roll_15 = get(hint2, 1.5, "motion135_delta_vs_alpha1")
    delta_hint_roll_125 = get(hint2, 1.25, "motion135_delta_vs_alpha1")
    delta_hint_1step_4 = get(hint1, 4.0, "motion135_delta_vs_alpha1") if any(
        abs(float(r["alpha"]) - 4.0) < 1e-6 for r in hint1
    ) else 0.0

    onset_125 = get(hint2, 1.25, "onset_xgt")
    release_125 = get(hint2, 1.25, "release_xgt")
    onset_15 = get(hint2, 1.5, "onset_xgt")
    release_15 = get(hint2, 1.5, "release_xgt")
    onset_base = get(hint2, 1.0, "onset_xgt")
    release_base = get(hint2, 1.0, "release_xgt")

    z0_onset = get(z2, 0.0, "onset_xgt")
    z0_release = get(z2, 0.0, "release_xgt")
    z0_far = get(z2, 0.0, "far_unobserved_error_cm")
    z_base_far = get(z2, 1.0, "far_unobserved_error_cm")

    # Heuristic case selection per spec
    cases: list[str] = []
    if delta_hint_1step_15 < 0.005 and delta_hint_roll_15 < 0.005:
        cases.append("Case C (alpha_hint): no meaningful effect either way")
    elif (delta_hint_1step_15 > 0.01 or delta_hint_1step_125 > 0.005) and delta_hint_roll_15 < 0.005:
        cases.append("Case B (alpha_hint): one-step responds but rollout does not")
    elif (
        (onset_125 > onset_base + 0.02 or onset_15 > onset_base + 0.02)
        or (release_125 > release_base + 0.02 or release_15 > release_base + 0.02)
    ):
        cases.append("Case A candidate (alpha_hint): rollout shows direction-positive small-alpha response")

    if delta_hint_1step_4 > delta_hint_1step_15 * 2.5 and delta_hint_1step_15 < 0.005:
        cases.append("Case I (alpha_hint=4 stress): only OOD scale moves output")

    if z0_onset > get(z2, 1.0, "onset_xgt") + 0.05 or z0_release > get(z2, 1.0, "release_xgt") + 0.05:
        if z0_far > z_base_far + 3.0:
            cases.append("Case E (alpha_z_target=0): transition improves but far/anchor worsens")
        else:
            cases.append("Case D (alpha_z_target < 1): transition improves while far/anchor acceptable")

    if not cases:
        cases.append("Mixed / inconclusive — single-clip-set evidence; expand or reseed before training")

    return "; ".join(cases)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/training/anchordiff_v18_a1_FULL_DATA.yaml"))
    parser.add_argument("--ckpt", type=Path, default=Path("runs/training/stageB_anchordiff_v18_a1_FULL_DATA/final.pt"))
    parser.add_argument("--output", type=Path, default=Path("analyses/2026-05-14_condition_route_causal_sensitivity_diagnostic.json"))
    parser.add_argument("--md", type=Path, default=Path("analyses/2026-05-14_condition_route_causal_sensitivity_diagnostic.md"))
    parser.add_argument("--selection-json", type=Path, default=Path("analyses/2026-05-13_v18_recon_ladder_truncated_rollout.json"))
    parser.add_argument("--max-clips", type=int, default=4)
    parser.add_argument("--num-candidates", type=int, default=96)
    parser.add_argument("--bucket", choices=["train", "val"], default="train")
    parser.add_argument("--balanced-subsets", action="store_true")
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--alpha-hint", type=str, default="0.5,0.75,1.0,1.25,1.5,2.0")
    parser.add_argument("--alpha-hint-stress", type=str, default="4.0",
                        help="Optional stress points appended to alpha-hint sweep; OOD evidence only. "
                             "Pass 'none' to disable.")
    parser.add_argument("--alpha-z-target", type=str, default="0.0,0.25,0.5,0.75,1.0,1.25")
    parser.add_argument("--alpha-z-target-stress", type=str, default="1.5",
                        help="Pass 'none' to disable.")
    parser.add_argument("--alpha-dense-target", type=str, default="0.5,1.0,1.5")
    parser.add_argument("--alpha-plan-target", type=str, default="0.5,1.0,1.5")
    parser.add_argument("--timesteps", type=str, default="100,300,500,700,900")
    parser.add_argument("--skip-optional", action="store_true",
                        help="Skip Perturbation C (dense + plan target scaling).")
    parser.add_argument("--sampler", choices=["ddpm", "ddim_eta0"], default="ddpm",
                        help="Reverse-rollout sampler. 'ddpm' is the v18 default "
                             "(ancestral); 'ddim_eta0' is a diagnostic-only "
                             "deterministic variant used to isolate route causality "
                             "from sampler stochasticity. NOT a deployable change.")
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
        raise RuntimeError("No clips selected for condition-route causal sensitivity")
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

    def _stress_arg(raw: str) -> list[float]:
        if not raw or str(raw).strip().lower() in {"none", "off", "disable"}:
            return []
        return _parse_floats(raw)

    alpha_hint = _parse_floats(args.alpha_hint)
    alpha_hint_stress = _stress_arg(args.alpha_hint_stress)
    alpha_z_target = _parse_floats(args.alpha_z_target)
    alpha_z_target_stress = _stress_arg(args.alpha_z_target_stress)
    alpha_dense_target = _parse_floats(args.alpha_dense_target)
    alpha_plan_target = _parse_floats(args.alpha_plan_target)

    full_hint_alphas = sorted(set(alpha_hint + alpha_hint_stress))
    full_z_alphas = sorted(set(alpha_z_target + alpha_z_target_stress))

    sampler = str(args.sampler)
    print(f"Using sampler: {sampler}")

    # Extract seq_ids / subsets for per-clip metric bookkeeping
    seq_ids = [str(s) for s in batch.get("seq_id", [])]
    subsets = [str(s) for s in batch.get("subset", [])]

    # ---- Level 1 alpha_hint per timestep -------------------------------
    print(f"Level 1 alpha_hint at {len(timesteps)} timesteps × {len(full_hint_alphas)} alphas")
    level1_hint_by_t: dict[int, dict[str, dict[str, float]]] = {}
    level1_hint_motion_by_t_alpha: dict[int, dict[float, Tensor]] = {}
    for t_int in timesteps:
        level1_hint_by_t[t_int] = {}
        level1_hint_motion_by_t_alpha[t_int] = {}
        # alpha=1.0 baseline for this timestep
        baseline_motion = _one_step_x0(
            model, motion_gt, base_cond, t_int,
            seed=int(args.seed) + t_int, cfg_scale=float(args.cfg_scale), alpha_hint=1.0,
        )
        baseline_metrics = _metrics_for_motion(
            baseline_motion, rest_offsets=rest_offsets, gt_joints=gt_joints,
            object_positions=object_positions, contact_state=contact_state,
            seq_mask=seq_mask, plan=plan_gt, fps=fps, threshold=threshold,
            part_to_joint=part_to_joint,
        )
        baseline_joints = baseline_metrics["joints"]
        level1_hint_motion_by_t_alpha[t_int][1.0] = baseline_motion
        for a in full_hint_alphas:
            if abs(a - 1.0) < 1e-9:
                row_motion = baseline_motion
                row_metrics = baseline_metrics
            else:
                row_motion = _one_step_x0(
                    model, motion_gt, base_cond, t_int,
                    seed=int(args.seed) + t_int, cfg_scale=float(args.cfg_scale), alpha_hint=a,
                )
                row_metrics = _metrics_for_motion(
                    row_motion, rest_offsets=rest_offsets, gt_joints=gt_joints,
                    object_positions=object_positions, contact_state=contact_state,
                    seq_mask=seq_mask, plan=plan_gt, fps=fps, threshold=threshold,
                    part_to_joint=part_to_joint,
                )
                level1_hint_motion_by_t_alpha[t_int][a] = row_motion
            summary = _summary_row(
                row_metrics, row_motion,
                baseline_motion=baseline_motion, baseline_joints=baseline_joints,
                joints=row_metrics["joints"],
            )
            level1_hint_by_t[t_int][str(a)] = summary

    # Aggregate (mean across timesteps) for alpha_hint Level 1
    def _aggregate_alpha(by_t: dict[int, dict[str, dict[str, float]]], alphas: list[float]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for a in alphas:
            keys = list(by_t[next(iter(by_t))][str(a)].keys())
            agg = {"alpha": float(a)}
            for k in keys:
                vals = [by_t[t][str(a)][k] for t in by_t]
                vals = [v for v in vals if np.isfinite(v)]
                agg[k] = float(np.mean(vals)) if vals else 0.0
            out.append(agg)
        return out

    alpha_hint_level1_agg = _aggregate_alpha(level1_hint_by_t, full_hint_alphas)

    # ---- Level 2 alpha_hint full rollout -------------------------------
    print(f"Level 2 alpha_hint × {len(full_hint_alphas)} alphas (sampler={sampler})")
    level2_hint_rows: list[dict[str, Any]] = []
    level2_hint_per_clip: dict[str, list[dict[str, Any]]] = {}
    baseline_roll_motion = _full_rollout(
        model, base_cond, seq_length=total_t,
        seed=int(args.seed), cfg_scale=float(args.cfg_scale), alpha_hint=1.0,
        sampler=sampler,
    )
    baseline_roll = _metrics_for_motion(
        baseline_roll_motion, rest_offsets=rest_offsets, gt_joints=gt_joints,
        object_positions=object_positions, contact_state=contact_state,
        seq_mask=seq_mask, plan=plan_gt, fps=fps, threshold=threshold,
        part_to_joint=part_to_joint,
    )
    baseline_roll_joints = baseline_roll["joints"]
    for a in full_hint_alphas:
        if abs(a - 1.0) < 1e-9:
            motion = baseline_roll_motion
            metrics = baseline_roll
        else:
            motion = _full_rollout(
                model, base_cond, seq_length=total_t,
                seed=int(args.seed), cfg_scale=float(args.cfg_scale), alpha_hint=a,
                sampler=sampler,
            )
            metrics = _metrics_for_motion(
                motion, rest_offsets=rest_offsets, gt_joints=gt_joints,
                object_positions=object_positions, contact_state=contact_state,
                seq_mask=seq_mask, plan=plan_gt, fps=fps, threshold=threshold,
                part_to_joint=part_to_joint,
            )
        summary = _summary_row(
            metrics, motion,
            baseline_motion=baseline_roll_motion, baseline_joints=baseline_roll_joints,
            joints=metrics["joints"],
        )
        summary["alpha"] = float(a)
        level2_hint_rows.append(summary)
        # per-clip metrics for bootstrap / paired-Δ analysis
        level2_hint_per_clip[str(a)] = _per_clip_metrics(
            motion, rest_offsets=rest_offsets, gt_joints=gt_joints,
            object_positions=object_positions, contact_state=contact_state,
            seq_mask=seq_mask, plan=plan_gt, fps=fps, threshold=threshold,
            part_to_joint=part_to_joint, baseline_motion=baseline_roll_motion,
            seq_ids=seq_ids, subsets=subsets,
        )

    # ---- Level 1 alpha_z_target per timestep ---------------------------
    print(f"Level 1 alpha_z_target at {len(timesteps)} timesteps × {len(full_z_alphas)} alphas")
    level1_z_by_t: dict[int, dict[str, dict[str, float]]] = {}
    for t_int in timesteps:
        level1_z_by_t[t_int] = {}
        baseline_motion = _one_step_x0(
            model, motion_gt, base_cond, t_int,
            seed=int(args.seed) + t_int, cfg_scale=float(args.cfg_scale), alpha_hint=1.0,
        )
        baseline_metrics = _metrics_for_motion(
            baseline_motion, rest_offsets=rest_offsets, gt_joints=gt_joints,
            object_positions=object_positions, contact_state=contact_state,
            seq_mask=seq_mask, plan=plan_gt, fps=fps, threshold=threshold,
            part_to_joint=part_to_joint,
        )
        baseline_joints = baseline_metrics["joints"]
        for a in full_z_alphas:
            if abs(a - 1.0) < 1e-9:
                row_motion = baseline_motion
                row_metrics = baseline_metrics
            else:
                cond_a = _apply_zint_target_scale(base_cond, a)
                row_motion = _one_step_x0(
                    model, motion_gt, cond_a, t_int,
                    seed=int(args.seed) + t_int, cfg_scale=float(args.cfg_scale), alpha_hint=1.0,
                )
                row_metrics = _metrics_for_motion(
                    row_motion, rest_offsets=rest_offsets, gt_joints=gt_joints,
                    object_positions=object_positions, contact_state=contact_state,
                    seq_mask=seq_mask, plan=plan_gt, fps=fps, threshold=threshold,
                    part_to_joint=part_to_joint,
                )
            summary = _summary_row(
                row_metrics, row_motion,
                baseline_motion=baseline_motion, baseline_joints=baseline_joints,
                joints=row_metrics["joints"],
            )
            level1_z_by_t[t_int][str(a)] = summary

    alpha_z_target_level1_agg = _aggregate_alpha(level1_z_by_t, full_z_alphas)

    # ---- Level 2 alpha_z_target full rollout ---------------------------
    print(f"Level 2 alpha_z_target × {len(full_z_alphas)} alphas (sampler={sampler})")
    level2_z_rows: list[dict[str, Any]] = []
    level2_z_per_clip: dict[str, list[dict[str, Any]]] = {}
    for a in full_z_alphas:
        if abs(a - 1.0) < 1e-9:
            motion = baseline_roll_motion
            metrics = baseline_roll
        else:
            cond_a = _apply_zint_target_scale(base_cond, a)
            motion = _full_rollout(
                model, cond_a, seq_length=total_t,
                seed=int(args.seed), cfg_scale=float(args.cfg_scale), alpha_hint=1.0,
                sampler=sampler,
            )
            metrics = _metrics_for_motion(
                motion, rest_offsets=rest_offsets, gt_joints=gt_joints,
                object_positions=object_positions, contact_state=contact_state,
                seq_mask=seq_mask, plan=plan_gt, fps=fps, threshold=threshold,
                part_to_joint=part_to_joint,
            )
        summary = _summary_row(
            metrics, motion,
            baseline_motion=baseline_roll_motion, baseline_joints=baseline_roll_joints,
            joints=metrics["joints"],
        )
        summary["alpha"] = float(a)
        level2_z_rows.append(summary)
        level2_z_per_clip[str(a)] = _per_clip_metrics(
            motion, rest_offsets=rest_offsets, gt_joints=gt_joints,
            object_positions=object_positions, contact_state=contact_state,
            seq_mask=seq_mask, plan=plan_gt, fps=fps, threshold=threshold,
            part_to_joint=part_to_joint, baseline_motion=baseline_roll_motion,
            seq_ids=seq_ids, subsets=subsets,
        )

    # ---- Optional Perturbation C ---------------------------------------
    level2_dense_rows: list[dict[str, Any]] = []
    level2_plan_rows: list[dict[str, Any]] = []
    if not args.skip_optional:
        print(f"Level 2 alpha_dense_target × {len(alpha_dense_target)} (sampler={sampler})")
        for a in alpha_dense_target:
            if abs(a - 1.0) < 1e-9:
                motion = baseline_roll_motion
                metrics = baseline_roll
            else:
                cond_a = _apply_dense_target_scale(base_cond, a)
                motion = _full_rollout(
                    model, cond_a, seq_length=total_t,
                    seed=int(args.seed), cfg_scale=float(args.cfg_scale), alpha_hint=1.0,
                    sampler=sampler,
                )
                metrics = _metrics_for_motion(
                    motion, rest_offsets=rest_offsets, gt_joints=gt_joints,
                    object_positions=object_positions, contact_state=contact_state,
                    seq_mask=seq_mask, plan=plan_gt, fps=fps, threshold=threshold,
                    part_to_joint=part_to_joint,
                )
            summary = _summary_row(
                metrics, motion,
                baseline_motion=baseline_roll_motion, baseline_joints=baseline_roll_joints,
                joints=metrics["joints"],
            )
            summary["alpha"] = float(a)
            level2_dense_rows.append(summary)

        print(f"Level 2 alpha_plan_target × {len(alpha_plan_target)} (sampler={sampler})")
        for a in alpha_plan_target:
            if abs(a - 1.0) < 1e-9:
                motion = baseline_roll_motion
                metrics = baseline_roll
            else:
                cond_a = _apply_plan_target_scale(base_cond, a)
                motion = _full_rollout(
                    model, cond_a, seq_length=total_t,
                    seed=int(args.seed), cfg_scale=float(args.cfg_scale), alpha_hint=1.0,
                    sampler=sampler,
                )
                metrics = _metrics_for_motion(
                    motion, rest_offsets=rest_offsets, gt_joints=gt_joints,
                    object_positions=object_positions, contact_state=contact_state,
                    seq_mask=seq_mask, plan=plan_gt, fps=fps, threshold=threshold,
                    part_to_joint=part_to_joint,
                )
            summary = _summary_row(
                metrics, motion,
                baseline_motion=baseline_roll_motion, baseline_joints=baseline_roll_joints,
                joints=metrics["joints"],
            )
            summary["alpha"] = float(a)
            level2_plan_rows.append(summary)

    payload = {
        "config": str(args.config),
        "ckpt": str(args.ckpt),
        "seed": int(args.seed),
        "cfg_scale": float(args.cfg_scale),
        "sampler": sampler,
        "timesteps": timesteps,
        "alpha_hint": full_hint_alphas,
        "alpha_z_target": full_z_alphas,
        "alpha_dense_target": alpha_dense_target if not args.skip_optional else [],
        "alpha_plan_target": alpha_plan_target if not args.skip_optional else [],
        "selected_clips": clip_metadata(batch),
        "alpha_hint_level1_by_t": {str(t): rows for t, rows in level1_hint_by_t.items()},
        "alpha_hint_level1_agg": alpha_hint_level1_agg,
        "alpha_hint_level2": level2_hint_rows,
        "alpha_hint_level2_per_clip": level2_hint_per_clip,
        "alpha_z_target_level1_by_t": {str(t): rows for t, rows in level1_z_by_t.items()},
        "alpha_z_target_level1_agg": alpha_z_target_level1_agg,
        "alpha_z_target_level2": level2_z_rows,
        "alpha_z_target_level2_per_clip": level2_z_per_clip,
        "alpha_dense_target_level2": level2_dense_rows,
        "alpha_plan_target_level2": level2_plan_rows,
        "caveat": (
            "Inference-only scaling of projected condition branches introduces "
            "train-inference distribution shift. These results are causal-sensitivity "
            "evidence ONLY; they are not a proposal of inference-time scaling as a "
            "deployable method. The 'sampler' field records the reverse-rollout "
            "sampler used; only 'ddpm' matches the v18 default."
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
