"""Metric-v2 replay of round 2/3/4 z_int diagnostics (round 7, Task 2).

Re-runs full DDPM rollouts on the same 16-clip Round-5 selection at:

  - baseline variance: alpha_z_target = 1.0, seeds [42, 43, 44]
  - z_int alpha sweep: alpha_z_target in [0.0, 0.5, 1.0], seeds [42, 43, 44]

For each rollout, computes:
  - v1 transition_metrics (legacy)
  - v2 transition_metrics (M2/M3/M5 + surface + validity)
  - dynamics_metrics (safety: body / hand vel, acc p95, jerk p95, far_unobs)

Outputs:
  analyses/2026-05-17_metric_v2_replay_summary.{json,md}

Scope notes (per round-7 spec "if cheap"):
- Replay 3 (target-route ablation): DEFERRED — would require additional ablation flags + 5x rollouts.
- Replay 4 (oracle event guidance): DEFERRED — requires loading oracle helper + different cond path.
- Replay 5 (plan sensitivity): DEFERRED — requires plan-perturbation harness; existing diagnostic already covers this but is heavy.

Each of replays 3/4/5 can be added as a follow-up round if v2 replay 1+2 leaves
conclusions ambiguous. This script focuses on the two cases where
v1 conclusions are most likely to be re-interpreted under v2.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import Tensor

from condition_route_causal_sensitivity_diagnostic import (
    _apply_zint_target_scale,
    _full_rollout,
)
from diagnostic_common import (
    dynamics_metrics, extract_plan, format_md_table, make_seq_mask,
    merge_single_batches, stats_list, transition_metrics,
)
from dynamics_diagnostic import (
    PART_JOINT, _build_cond, _build_model, _fk_from_motion_135,
)
from piano.data.dataset import collate_hoi
from piano.utils.clip_utils import load_clip_text_encoder
from recon_ladder_truncated_rollout_diagnostic import (
    _build_selected_batches,
    _load_selection,
)


def _per_clip_metrics_v1_v2(
    motion: Tensor,
    *,
    rest_offsets: Tensor,
    gt_joints: Tensor,
    object_positions: Tensor,
    object_rotations: Tensor,
    object_pc: Tensor,
    contact_state: Tensor,
    seq_mask: Tensor,
    fps: float,
    threshold: float,
) -> dict[str, Any]:
    joints = _fk_from_motion_135(motion, rest_offsets)
    dyn = dynamics_metrics(joints, seq_mask, gt_joints=gt_joints, fps=fps)
    trans_v1 = transition_metrics(
        joints, object_positions, contact_state, seq_mask,
        gt_joints=gt_joints, window_k=10, threshold=threshold,
        metric_version="v1",
    )
    trans_v2 = transition_metrics(
        joints, object_positions, contact_state, seq_mask,
        gt_joints=gt_joints, window_k=10, threshold=threshold,
        metric_version="v2", object_pc=object_pc, object_rotations=object_rotations,
        edge_margin=5, min_gt_change_cm=2.0, flicker_max_frames=2,
    )
    # Trim events from v2 to reduce JSON size
    if "events" in trans_v2:
        trans_v2 = {k: v for k, v in trans_v2.items() if k != "events"}
    return {
        "dynamics": dyn,
        "transition_v1": trans_v1,
        "transition_v2": trans_v2,
        "joints": joints.detach().cpu(),
    }


def _headlines_v1(trans_v1: dict[str, Any]) -> dict[str, float]:
    r = trans_v1.get("ratios_over_gt", {})
    return {
        "onset_xGT": float(r.get("onset_positive_closing", 0.0)),
        "release_xGT": float(r.get("release_positive_opening", 0.0)),
        "transvel_xGT": float(r.get("transition_relative_velocity", 0.0)),
    }


def _headlines_v2(trans_v2: dict[str, Any]) -> dict[str, float]:
    onset = trans_v2.get("onset_direction_score_cm_per_frame", {})
    release = trans_v2.get("release_direction_score_cm_per_frame", {})
    onset_signed = trans_v2.get("onset_signed_diff_cm", {})
    release_signed = trans_v2.get("release_signed_diff_cm", {})
    m5_2 = trans_v2.get("m5_ratio_clip_2cm", {})
    m5_5 = trans_v2.get("m5_ratio_clip_5cm", {})
    return {
        "M2_onset_direction_cm_per_frame_mean": float(onset.get("mean", 0.0)),
        "M2_release_direction_cm_per_frame_mean": float(release.get("mean", 0.0)),
        "M3_onset_signed_cm_mean": float(onset_signed.get("mean", 0.0)),
        "M3_release_signed_cm_mean": float(release_signed.get("mean", 0.0)),
        "M5_clip2cm_mean": float(m5_2.get("mean", 0.0)) if m5_2 else 0.0,
        "M5_clip5cm_mean": float(m5_5.get("mean", 0.0)) if m5_5 else 0.0,
        "n_valid_slope": int(trans_v2.get("n_valid_slope", 0)),
        "n_valid_signed": int(trans_v2.get("n_valid_signed", 0)),
        "n_valid_ratio_2cm": int(trans_v2.get("n_valid_ratio_2cm", 0)),
        "n_events_total": int(trans_v2.get("n_events_total", 0)),
    }


def _safety_headlines(dyn: dict[str, Any]) -> dict[str, float]:
    return {
        "body_vel_xGT": float(dyn.get("body_velocity_cm_per_frame_over_gt", 0.0)),
        "hand_vel_xGT": float(dyn.get("hand_velocity_cm_per_frame_over_gt", 0.0)),
        "acc_p95_xGT": float(dyn.get("body_acc_p95_cm_per_frame2_over_gt", 0.0)),
        "jerk_p95_xGT": float(dyn.get("body_jerk_p95_cm_per_frame3_over_gt", 0.0)),
    }


def _run_config(
    cfg, device, model, object_encoder, clip_model, z_dims,
    selected, *, seed: int, alpha_z_target: float,
    cfg_scale: float, threshold: float,
) -> dict[str, Any]:
    """Single (seed, alpha) configuration over all clips."""
    rows: list[dict[str, Any]] = []
    for clip_idx, (idx, batch, ev) in enumerate(selected):
        cond, T = _build_cond(batch, model, object_encoder, clip_model, z_dims, cfg, device)
        cond = {**cond, "interaction_plan": extract_plan(batch, device)}
        if abs(alpha_z_target - 1.0) > 1e-6:
            cond = _apply_zint_target_scale(cond, alpha_z_target)
        motion_gt = batch["motion"].to(device).float()
        rest_offsets = batch["rest_offsets"].to(device).float()
        seq_len = batch["seq_len"].to(device).long()
        seq_mask = make_seq_mask(seq_len, T, device)
        gt_joints = batch["joints"].to(device).float()
        object_positions = batch["object_positions"].to(device).float()
        object_rotations = batch["object_rotations"].to(device).float()
        object_pc = batch["object_pc"].to(device).float()
        contact_state = batch["contact_state"].to(device).float()

        motion_pred = _full_rollout(
            model, cond, seq_length=T, seed=int(seed) + clip_idx * 10000,
            cfg_scale=float(cfg_scale), alpha_hint=1.0, sampler="ddpm",
        )
        m = _per_clip_metrics_v1_v2(
            motion_pred,
            rest_offsets=rest_offsets, gt_joints=gt_joints,
            object_positions=object_positions, object_rotations=object_rotations,
            object_pc=object_pc, contact_state=contact_state, seq_mask=seq_mask,
            fps=20.0, threshold=float(threshold),
        )
        # Far-unobs realisation (mean over T) — simple proxy using GT hand vs pred hand
        pred_joints = m["joints"]
        # Per-frame mean L2 cm between pred root and GT root
        root_err_cm = float(
            torch.linalg.vector_norm(pred_joints[:, :, 0, :].cpu() - batch["joints"][:, :, 0, :].cpu(), dim=-1).mean().item() * 100.0
        )
        rows.append({
            "clip_idx": clip_idx,
            "subset": str(batch["subset"][0]),
            "seq_id": str(batch["seq_id"][0]),
            "v1": _headlines_v1(m["transition_v1"]),
            "v2": _headlines_v2(m["transition_v2"]),
            "dyn": _safety_headlines(m["dynamics"]),
            "root_err_cm": root_err_cm,
        })
    return {"seed": int(seed), "alpha_z_target": float(alpha_z_target), "clips": rows}


def _aggregate_config(c: dict[str, Any]) -> dict[str, Any]:
    def _mean_v1(key: str) -> float:
        return float(np.mean([row["v1"][key] for row in c["clips"]]))

    def _mean_v2(key: str) -> float:
        return float(np.mean([row["v2"][key] for row in c["clips"]]))

    def _sum_v2(key: str) -> int:
        return int(sum(row["v2"][key] for row in c["clips"]))

    def _mean_dyn(key: str) -> float:
        return float(np.mean([row["dyn"][key] for row in c["clips"]]))

    return {
        "seed": c["seed"],
        "alpha_z_target": c["alpha_z_target"],
        "n_clips": len(c["clips"]),
        "v1": {
            "onset_xGT_mean": _mean_v1("onset_xGT"),
            "release_xGT_mean": _mean_v1("release_xGT"),
            "transvel_xGT_mean": _mean_v1("transvel_xGT"),
        },
        "v2": {
            "M2_onset_direction_mean": _mean_v2("M2_onset_direction_cm_per_frame_mean"),
            "M2_release_direction_mean": _mean_v2("M2_release_direction_cm_per_frame_mean"),
            "M3_onset_signed_mean": _mean_v2("M3_onset_signed_cm_mean"),
            "M3_release_signed_mean": _mean_v2("M3_release_signed_cm_mean"),
            "M5_clip2cm_mean": _mean_v2("M5_clip2cm_mean"),
            "M5_clip5cm_mean": _mean_v2("M5_clip5cm_mean"),
            "n_valid_slope_total": _sum_v2("n_valid_slope"),
            "n_valid_signed_total": _sum_v2("n_valid_signed"),
            "n_valid_ratio_2cm_total": _sum_v2("n_valid_ratio_2cm"),
            "n_events_total": _sum_v2("n_events_total"),
        },
        "safety": {
            "body_vel_xGT_mean": _mean_dyn("body_vel_xGT"),
            "hand_vel_xGT_mean": _mean_dyn("hand_vel_xGT"),
            "acc_p95_xGT_mean": _mean_dyn("acc_p95_xGT"),
            "jerk_p95_xGT_mean": _mean_dyn("jerk_p95_xGT"),
        },
    }


def _paired_delta(
    seeds: list[dict[str, Any]], key_path: list[str],
) -> dict[str, Any]:
    """Mean +/- std across seeds at a given metric path."""
    vals = []
    for s in seeds:
        d = s
        for k in key_path:
            d = d.get(k, {})
        if isinstance(d, (int, float)):
            vals.append(float(d))
    if not vals:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": int(len(vals))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=Path("configs/training/anchordiff_v18_a1_FULL_DATA.yaml"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("runs/training/stageB_anchordiff_v18_a1_FULL_DATA/final.pt"))
    parser.add_argument("--selection-json", type=Path,
                        default=Path("analyses/2026-05-13_v18_recon_ladder_truncated_rollout.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("analyses/2026-05-17_metric_v2_replay_summary.json"))
    parser.add_argument("--md", type=Path,
                        default=Path("analyses/2026-05-17_metric_v2_replay_summary.md"))
    parser.add_argument("--max-clips", type=int, default=16)
    parser.add_argument("--num-candidates", type=int, default=256)
    parser.add_argument("--bucket", choices=["train", "val"], default="train")
    parser.add_argument("--balanced-subsets", action="store_true", default=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--alphas", type=str, default="0.0,0.5,1.0")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Skip alpha sweep, run only baseline variance at alpha=1.0")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(args.config)
    selection = _load_selection(args.selection_json, max_clips=int(args.max_clips))
    selected = _build_selected_batches(
        cfg, bucket=args.bucket, balanced_subsets=bool(args.balanced_subsets),
        num_candidates=int(args.num_candidates), selection=selection,
        max_clips=int(args.max_clips), threshold=float(args.threshold),
    )
    if not selected:
        raise SystemExit("No clips selected")

    # Build model + load checkpoint
    model, object_encoder, z_dims = _build_model(cfg, device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("model", state))
    if "object_encoder" in state:
        object_encoder.load_state_dict(state["object_encoder"])
    elif "extra_modules" in state and "object_encoder" in state["extra_modules"]:
        object_encoder.load_state_dict(state["extra_modules"]["object_encoder"])
    model.eval()
    object_encoder.eval()
    clip_model = load_clip_text_encoder(
        device=device,
        model_name=str(cfg.model.text_encoder.clip_version),
        download_root=str(cfg.model.text_encoder.download_root),
    )

    seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]
    alphas = [float(a.strip()) for a in str(args.alphas).split(",") if a.strip()]
    if args.baseline_only:
        alphas = [1.0]

    configs: list[dict[str, Any]] = []
    for seed in seeds:
        for alpha in alphas:
            print(f"  running seed={seed} alpha_z_target={alpha:.2f}", flush=True)
            c = _run_config(
                cfg, device, model, object_encoder, clip_model, z_dims,
                selected, seed=seed, alpha_z_target=alpha,
                cfg_scale=float(args.cfg_scale), threshold=float(args.threshold),
            )
            configs.append(c)

    # Aggregate per (seed, alpha) and produce paired deltas alpha vs alpha=1.0
    aggregates = [_aggregate_config(c) for c in configs]

    # Find baseline (alpha=1.0) aggregates per seed
    baseline_by_seed: dict[int, dict[str, Any]] = {}
    for ag in aggregates:
        if abs(ag["alpha_z_target"] - 1.0) < 1e-6:
            baseline_by_seed[ag["seed"]] = ag

    # Variance summary at baseline (alpha=1.0) — multi-seed
    baseline_aggs = [a for a in aggregates if abs(a["alpha_z_target"] - 1.0) < 1e-6]
    variance_summary = {
        "n_seeds": len(baseline_aggs),
        "v1_onset_xGT_mean_across_seeds": float(np.mean([a["v1"]["onset_xGT_mean"] for a in baseline_aggs])) if baseline_aggs else 0.0,
        "v1_onset_xGT_std_across_seeds": float(np.std([a["v1"]["onset_xGT_mean"] for a in baseline_aggs])) if baseline_aggs else 0.0,
        "v1_release_xGT_mean_across_seeds": float(np.mean([a["v1"]["release_xGT_mean"] for a in baseline_aggs])) if baseline_aggs else 0.0,
        "v1_release_xGT_std_across_seeds": float(np.std([a["v1"]["release_xGT_mean"] for a in baseline_aggs])) if baseline_aggs else 0.0,
        "v2_M2_onset_direction_mean_across_seeds": float(np.mean([a["v2"]["M2_onset_direction_mean"] for a in baseline_aggs])) if baseline_aggs else 0.0,
        "v2_M2_onset_direction_std_across_seeds": float(np.std([a["v2"]["M2_onset_direction_mean"] for a in baseline_aggs])) if baseline_aggs else 0.0,
        "v2_M2_release_direction_mean_across_seeds": float(np.mean([a["v2"]["M2_release_direction_mean"] for a in baseline_aggs])) if baseline_aggs else 0.0,
        "v2_M2_release_direction_std_across_seeds": float(np.std([a["v2"]["M2_release_direction_mean"] for a in baseline_aggs])) if baseline_aggs else 0.0,
        "v2_M3_onset_signed_mean_across_seeds": float(np.mean([a["v2"]["M3_onset_signed_mean"] for a in baseline_aggs])) if baseline_aggs else 0.0,
        "v2_M3_onset_signed_std_across_seeds": float(np.std([a["v2"]["M3_onset_signed_mean"] for a in baseline_aggs])) if baseline_aggs else 0.0,
        "v2_n_valid_slope_total_across_seeds": int(np.sum([a["v2"]["n_valid_slope_total"] for a in baseline_aggs])),
        "v2_n_events_total_across_seeds": int(np.sum([a["v2"]["n_events_total"] for a in baseline_aggs])),
        "safety_body_vel_xGT_mean_across_seeds": float(np.mean([a["safety"]["body_vel_xGT_mean"] for a in baseline_aggs])) if baseline_aggs else 0.0,
        "safety_acc_p95_xGT_mean_across_seeds": float(np.mean([a["safety"]["acc_p95_xGT_mean"] for a in baseline_aggs])) if baseline_aggs else 0.0,
    }

    # Alpha effect: paired delta vs alpha=1.0 per seed
    alpha_effect: list[dict[str, Any]] = []
    for alpha in alphas:
        if abs(alpha - 1.0) < 1e-6:
            continue
        per_seed_deltas = {
            "v1_onset_xGT_delta": [],
            "v1_release_xGT_delta": [],
            "v2_M2_onset_direction_delta": [],
            "v2_M3_onset_signed_delta": [],
            "v2_M2_release_direction_delta": [],
            "v2_M3_release_signed_delta": [],
            "safety_body_vel_xGT_delta": [],
            "safety_acc_p95_xGT_delta": [],
        }
        sign_consistency_v1 = []
        sign_consistency_v2_m2 = []
        for ag in aggregates:
            if abs(ag["alpha_z_target"] - alpha) >= 1e-6:
                continue
            seed = ag["seed"]
            base = baseline_by_seed.get(seed)
            if base is None:
                continue
            d_v1_onset = ag["v1"]["onset_xGT_mean"] - base["v1"]["onset_xGT_mean"]
            d_v1_release = ag["v1"]["release_xGT_mean"] - base["v1"]["release_xGT_mean"]
            d_v2_onset = ag["v2"]["M2_onset_direction_mean"] - base["v2"]["M2_onset_direction_mean"]
            d_v2_release = ag["v2"]["M2_release_direction_mean"] - base["v2"]["M2_release_direction_mean"]
            d_v2_onset_m3 = ag["v2"]["M3_onset_signed_mean"] - base["v2"]["M3_onset_signed_mean"]
            d_v2_release_m3 = ag["v2"]["M3_release_signed_mean"] - base["v2"]["M3_release_signed_mean"]
            d_safety = ag["safety"]["body_vel_xGT_mean"] - base["safety"]["body_vel_xGT_mean"]
            d_acc = ag["safety"]["acc_p95_xGT_mean"] - base["safety"]["acc_p95_xGT_mean"]
            per_seed_deltas["v1_onset_xGT_delta"].append(d_v1_onset)
            per_seed_deltas["v1_release_xGT_delta"].append(d_v1_release)
            per_seed_deltas["v2_M2_onset_direction_delta"].append(d_v2_onset)
            per_seed_deltas["v2_M2_release_direction_delta"].append(d_v2_release)
            per_seed_deltas["v2_M3_onset_signed_delta"].append(d_v2_onset_m3)
            per_seed_deltas["v2_M3_release_signed_delta"].append(d_v2_release_m3)
            per_seed_deltas["safety_body_vel_xGT_delta"].append(d_safety)
            per_seed_deltas["safety_acc_p95_xGT_delta"].append(d_acc)
            sign_consistency_v1.append(1 if d_v1_onset > 0 else (0 if abs(d_v1_onset) < 1e-9 else -1))
            sign_consistency_v2_m2.append(1 if d_v2_onset > 0 else (0 if abs(d_v2_onset) < 1e-9 else -1))
        alpha_effect.append({
            "alpha_z_target": alpha,
            "n_seeds": len(sign_consistency_v1),
            "v1_onset_xGT_delta_mean": float(np.mean(per_seed_deltas["v1_onset_xGT_delta"])) if per_seed_deltas["v1_onset_xGT_delta"] else 0.0,
            "v1_onset_xGT_delta_std": float(np.std(per_seed_deltas["v1_onset_xGT_delta"])) if per_seed_deltas["v1_onset_xGT_delta"] else 0.0,
            "v1_release_xGT_delta_mean": float(np.mean(per_seed_deltas["v1_release_xGT_delta"])) if per_seed_deltas["v1_release_xGT_delta"] else 0.0,
            "v2_M2_onset_direction_delta_mean": float(np.mean(per_seed_deltas["v2_M2_onset_direction_delta"])) if per_seed_deltas["v2_M2_onset_direction_delta"] else 0.0,
            "v2_M2_onset_direction_delta_std": float(np.std(per_seed_deltas["v2_M2_onset_direction_delta"])) if per_seed_deltas["v2_M2_onset_direction_delta"] else 0.0,
            "v2_M2_release_direction_delta_mean": float(np.mean(per_seed_deltas["v2_M2_release_direction_delta"])) if per_seed_deltas["v2_M2_release_direction_delta"] else 0.0,
            "v2_M3_onset_signed_delta_mean": float(np.mean(per_seed_deltas["v2_M3_onset_signed_delta"])) if per_seed_deltas["v2_M3_onset_signed_delta"] else 0.0,
            "v2_M3_release_signed_delta_mean": float(np.mean(per_seed_deltas["v2_M3_release_signed_delta"])) if per_seed_deltas["v2_M3_release_signed_delta"] else 0.0,
            "safety_body_vel_xGT_delta_mean": float(np.mean(per_seed_deltas["safety_body_vel_xGT_delta"])) if per_seed_deltas["safety_body_vel_xGT_delta"] else 0.0,
            "safety_acc_p95_xGT_delta_mean": float(np.mean(per_seed_deltas["safety_acc_p95_xGT_delta"])) if per_seed_deltas["safety_acc_p95_xGT_delta"] else 0.0,
            "sign_consistency_v1_onset_positive_count": int(sum(1 for x in sign_consistency_v1 if x > 0)),
            "sign_consistency_v2_M2_onset_positive_count": int(sum(1 for x in sign_consistency_v2_m2 if x > 0)),
            "per_seed_deltas": per_seed_deltas,
        })

    # Verdict
    verdict_lines: list[str] = []
    # Replay 1 — baseline variance interpretation
    v1_onset_std = variance_summary["v1_onset_xGT_std_across_seeds"]
    v2_m2_onset_std = variance_summary["v2_M2_onset_direction_std_across_seeds"]
    verdict_lines.append(
        f"Baseline variance ({variance_summary['n_seeds']} seeds, alpha=1.0): "
        f"v1 onset xGT σ = {v1_onset_std:.3f}; v2 M2 onset direction σ = "
        f"{v2_m2_onset_std:.3f} cm/frame."
    )
    # Replay 2 — alpha effect interpretation
    for ae in alpha_effect:
        verdict_lines.append(
            f"alpha={ae['alpha_z_target']:.2f}: v1 onset Δ = "
            f"{ae['v1_onset_xGT_delta_mean']:+.3f} (sign consistent {ae['sign_consistency_v1_onset_positive_count']}/{ae['n_seeds']}); "
            f"v2 M2 onset direction Δ = {ae['v2_M2_onset_direction_delta_mean']:+.3f} cm/frame; "
            f"safety body vel Δ = {ae['safety_body_vel_xGT_delta_mean']:+.3f}, "
            f"acc p95 Δ = {ae['safety_acc_p95_xGT_delta_mean']:+.3f}."
        )

    # Sanitize per-clip rows (strip joint tensors that should not be serialized)
    per_clip_rows = []
    for c in configs:
        per_clip_rows.append({
            "seed": c["seed"],
            "alpha_z_target": c["alpha_z_target"],
            "clips": [
                {
                    "clip_idx": r["clip_idx"],
                    "subset": r["subset"],
                    "seq_id": r["seq_id"],
                    "v1": r["v1"],
                    "v2": r["v2"],
                    "dyn": r["dyn"],
                    "root_err_cm": r.get("root_err_cm", 0.0),
                }
                for r in c["clips"]
            ],
        })

    payload = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "selection_json": str(args.selection_json),
        "seeds": seeds, "alphas": alphas,
        "n_clips_per_config": len(selected),
        "configs": aggregates,
        "per_clip_rows": per_clip_rows,
        "baseline_variance_summary": variance_summary,
        "alpha_effect_paired_delta": alpha_effect,
        "verdict_lines": verdict_lines,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")

    # Markdown
    lines = [
        "# Metric-V2 Replay Summary (Round 7, Task 2)",
        "",
        f"- Config: `{args.config}`",
        f"- Checkpoint: `{args.checkpoint}`",
        f"- Selection: `{args.selection_json}` (n_clips_per_config = {len(selected)})",
        f"- Seeds: {seeds}",
        f"- Alphas: {alphas}",
        "",
        "## Scope notes",
        "",
        "- Replay 1 (baseline variance) and Replay 2 (z_int alpha sweep) covered.",
        "- Replays 3/4/5 (target-route ablation / oracle guidance / plan sensitivity)",
        "  DEFERRED: each requires either additional ablation flags, oracle helpers,",
        "  or plan-perturbation harness. Available to add as a follow-up if v2",
        "  replay 1/2 leaves conclusions ambiguous.",
        "",
        "## Per-(seed, alpha) aggregate",
        "",
        "| seed | alpha | v1 onset xGT | v1 release xGT | v2 M2 onset cm/f | v2 M2 release cm/f | v2 M3 onset cm | v2 M3 release cm | n_valid_slope / n_events | body vel xGT | acc p95 xGT |",
        "|------|-------|---------------|-----------------|-------------------|---------------------|-----------------|-------------------|--------------------------|---------------|--------------|",
    ]
    for a in aggregates:
        lines.append(
            f"| {a['seed']} | {a['alpha_z_target']:.2f} | "
            f"{a['v1']['onset_xGT_mean']:.3f} | "
            f"{a['v1']['release_xGT_mean']:.3f} | "
            f"{a['v2']['M2_onset_direction_mean']:+.3f} | "
            f"{a['v2']['M2_release_direction_mean']:+.3f} | "
            f"{a['v2']['M3_onset_signed_mean']:+.2f} | "
            f"{a['v2']['M3_release_signed_mean']:+.2f} | "
            f"{a['v2']['n_valid_slope_total']}/{a['v2']['n_events_total']} | "
            f"{a['safety']['body_vel_xGT_mean']:.3f} | "
            f"{a['safety']['acc_p95_xGT_mean']:.3f} |"
        )
    lines += [
        "",
        "## Baseline variance (alpha=1.0 across seeds)",
        "",
        f"- v1 onset xGT mean ± σ: {variance_summary['v1_onset_xGT_mean_across_seeds']:.3f} ± {variance_summary['v1_onset_xGT_std_across_seeds']:.3f}",
        f"- v1 release xGT mean ± σ: {variance_summary['v1_release_xGT_mean_across_seeds']:.3f} ± {variance_summary['v1_release_xGT_std_across_seeds']:.3f}",
        f"- v2 M2 onset direction mean ± σ: {variance_summary['v2_M2_onset_direction_mean_across_seeds']:+.3f} ± {variance_summary['v2_M2_onset_direction_std_across_seeds']:.3f} cm/frame",
        f"- v2 M2 release direction mean ± σ: {variance_summary['v2_M2_release_direction_mean_across_seeds']:+.3f} ± {variance_summary['v2_M2_release_direction_std_across_seeds']:.3f} cm/frame",
        f"- v2 M3 onset signed mean ± σ: {variance_summary['v2_M3_onset_signed_mean_across_seeds']:+.3f} ± {variance_summary['v2_M3_onset_signed_std_across_seeds']:.3f} cm",
        f"- v2 valid slope events (total across seeds): {variance_summary['v2_n_valid_slope_total_across_seeds']} / {variance_summary['v2_n_events_total_across_seeds']}",
        f"- safety: body vel xGT mean {variance_summary['safety_body_vel_xGT_mean_across_seeds']:.3f}; acc p95 xGT mean {variance_summary['safety_acc_p95_xGT_mean_across_seeds']:.3f}",
        "",
        "## Alpha effect (paired vs alpha=1.0 baseline per seed)",
        "",
    ]
    for ae in alpha_effect:
        lines += [
            f"### alpha = {ae['alpha_z_target']:.2f}",
            "",
            f"- n_seeds: {ae['n_seeds']}",
            f"- v1 onset xGT Δ: {ae['v1_onset_xGT_delta_mean']:+.3f} ± {ae['v1_onset_xGT_delta_std']:.3f}",
            f"- v1 release xGT Δ: {ae['v1_release_xGT_delta_mean']:+.3f}",
            f"- v2 M2 onset direction Δ: {ae['v2_M2_onset_direction_delta_mean']:+.3f} ± {ae['v2_M2_onset_direction_delta_std']:.3f} cm/frame",
            f"- v2 M2 release direction Δ: {ae['v2_M2_release_direction_delta_mean']:+.3f} cm/frame",
            f"- v2 M3 onset signed Δ: {ae['v2_M3_onset_signed_delta_mean']:+.3f} cm",
            f"- v2 M3 release signed Δ: {ae['v2_M3_release_signed_delta_mean']:+.3f} cm",
            f"- safety body vel xGT Δ: {ae['safety_body_vel_xGT_delta_mean']:+.3f}",
            f"- safety acc p95 xGT Δ: {ae['safety_acc_p95_xGT_delta_mean']:+.3f}",
            f"- sign consistency: v1 onset improved {ae['sign_consistency_v1_onset_positive_count']}/{ae['n_seeds']}; v2 M2 onset improved {ae['sign_consistency_v2_M2_onset_positive_count']}/{ae['n_seeds']}",
            "",
        ]
    lines += [
        "## Verdict notes",
        "",
    ]
    for v in verdict_lines:
        lines.append(f"- {v}")
    lines.append("")
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.md}")


if __name__ == "__main__":
    main()
