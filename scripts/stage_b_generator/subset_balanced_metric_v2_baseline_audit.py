"""Subset-balanced metric-v2 baseline audit (Round 9, Task 2).

Generates v18 baseline samples (default DDPM) on the 24-clip
subset-balanced selection from Task 1, computes v2 transition metrics,
dynamics metrics, plan/geometry metrics, and per-type plan anchor
errors per subset.

No alpha perturbation. No sampler variant. No route ablation. v1 metric
is computed only for legacy reference and is NOT load-bearing.

Outputs:
  analyses/2026-05-19_subset_balanced_metric_v2_baseline_audit.{json,md}
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from condition_route_causal_sensitivity_diagnostic import _full_rollout
from diagnostic_common import (
    dynamics_metrics, extract_plan, format_md_table, make_seq_mask,
    merge_single_batches, stats_list, transition_metrics,
)
from dynamics_diagnostic import (
    _build_cond, _build_dataset, _build_model, _fk_from_motion_135,
)
from piano.data.dataset import collate_hoi
from piano.utils.clip_utils import load_clip_text_encoder
from plan_condition_diagnostics import _compute_metrics as _compute_plan_metrics


HAND_SPECS = (("L_hand", 20, 0), ("R_hand", 21, 1))
PART_NAMES = ("left_hand", "right_hand", "left_foot", "right_foot", "pelvis")
PART_JOINT = (20, 21, 10, 11, 0)
ANCHOR_TYPE_NAMES = {0: "onset", 1: "stable", 2: "release", 3: "phase_change", 4: "support_change"}


def _per_type_plan_errors(batch: dict[str, Any], b: int, gt_joints_np: np.ndarray, seq_len: int) -> dict[str, dict[str, float]]:
    """Per-anchor-type target -> GT hand error on the GT plan (data-side audit)."""
    plan_keys = ["anchor_mask", "anchor_time", "anchor_part", "anchor_target_world", "anchor_type"]
    plan = {k: batch[f"plan_{k}"][b].detach().cpu().numpy() for k in plan_keys}
    out: dict[str, list[float]] = {n: [] for n in ANCHOR_TYPE_NAMES.values()}
    for k in range(int(plan["anchor_mask"].size)):
        if not bool(plan["anchor_mask"][k]):
            continue
        t_a = int(min(max(0, int(plan["anchor_time"][k])), seq_len - 1))
        t_name = ANCHOR_TYPE_NAMES.get(int(plan["anchor_type"][k]), "stable")
        for p in np.where(plan["anchor_part"][k] > 0)[0].tolist():
            joint = PART_JOINT[int(p)]
            target_w = plan["anchor_target_world"][k, int(p)]
            err = float(np.linalg.norm(target_w - gt_joints_np[t_a, joint]) * 100.0)
            out.setdefault(t_name, []).append(err)
    return {
        n: stats_list(vals) if vals else {"mean": 0.0, "n": 0, "median": 0.0, "p25": 0.0, "p75": 0.0, "p95": 0.0, "std": 0.0}
        for n, vals in out.items()
    }


def _audit_clip(
    cfg, device, model, object_encoder, clip_model, z_dims,
    batch: dict[str, Any], *, seed: int, threshold: float, cfg_scale: float,
) -> dict[str, Any]:
    cond, T = _build_cond(batch, model, object_encoder, clip_model, z_dims, cfg, device)
    cond = {**cond, "interaction_plan": extract_plan(batch, device)}
    motion_pred = _full_rollout(
        model, cond, seq_length=T, seed=int(seed), cfg_scale=float(cfg_scale),
        alpha_hint=1.0, sampler="ddpm",
    )
    rest_offsets = batch["rest_offsets"].to(device).float()
    gt_joints = batch["joints"].to(device).float()
    seq_len_t = batch["seq_len"].to(device).long()
    seq_mask = make_seq_mask(seq_len_t, T, device)
    object_positions = batch["object_positions"].to(device).float()
    object_rotations = batch["object_rotations"].to(device).float()
    object_pc = batch["object_pc"].to(device).float()
    contact_state = batch["contact_state"].to(device).float()

    pred_joints = _fk_from_motion_135(motion_pred, rest_offsets)
    dyn = dynamics_metrics(pred_joints, seq_mask, gt_joints=gt_joints, fps=20.0)
    trans_v1 = transition_metrics(
        pred_joints, object_positions, contact_state, seq_mask,
        gt_joints=gt_joints, window_k=10, threshold=float(threshold),
        metric_version="v1",
    )
    trans_v2 = transition_metrics(
        pred_joints, object_positions, contact_state, seq_mask,
        gt_joints=gt_joints, window_k=10, threshold=float(threshold),
        metric_version="v2", object_pc=object_pc, object_rotations=object_rotations,
    )
    # Plan geometry metrics: far_unobs / anchor realization / near anchor
    part_to_joint = torch.tensor(PART_JOINT, dtype=torch.long, device=device)
    plan_keys = ["anchor_time", "anchor_mask", "anchor_part", "anchor_target_world"]
    plan_tensors = {k: batch[f"plan_{k}"].to(device) for k in plan_keys}
    plan_m = _compute_plan_metrics(
        jpos_pred=pred_joints, jpos_gt=gt_joints, seq_mask=seq_mask,
        anchor_time=plan_tensors["anchor_time"],
        anchor_mask=plan_tensors["anchor_mask"],
        anchor_part=plan_tensors["anchor_part"],
        anchor_target_world=plan_tensors["anchor_target_world"],
        part_to_joint=part_to_joint, window=3,
    )

    # Root drift
    pred_root = pred_joints[:, :, 0, :].detach().cpu().numpy()
    gt_root = gt_joints[:, :, 0, :].detach().cpu().numpy()
    seq_len = int(seq_len_t.item())
    root_drift_cm = float(np.linalg.norm(pred_root[0, :seq_len] - gt_root[0, :seq_len], axis=-1).mean() * 100.0)

    # Per-type plan anchor error (data-side; same for all seeds, but we compute once)
    gt_joints_np = batch["joints"][0].detach().cpu().numpy()
    per_type_errs = _per_type_plan_errors(batch, 0, gt_joints_np, seq_len)

    v2 = {
        "M2_onset_direction_cm_per_frame_mean": float(trans_v2.get("onset_direction_score_cm_per_frame", {}).get("mean", 0.0)),
        "M2_release_direction_cm_per_frame_mean": float(trans_v2.get("release_direction_score_cm_per_frame", {}).get("mean", 0.0)),
        "M3_onset_signed_cm_mean": float(trans_v2.get("onset_signed_diff_cm", {}).get("mean", 0.0)),
        "M3_release_signed_cm_mean": float(trans_v2.get("release_signed_diff_cm", {}).get("mean", 0.0)),
        "M5_clip2cm_mean": float(trans_v2.get("m5_ratio_clip_2cm", {}).get("mean", 0.0)) if "m5_ratio_clip_2cm" in trans_v2 else 0.0,
        "M5_clip5cm_mean": float(trans_v2.get("m5_ratio_clip_5cm", {}).get("mean", 0.0)) if "m5_ratio_clip_5cm" in trans_v2 else 0.0,
        "n_valid_slope": int(trans_v2.get("n_valid_slope", 0)),
        "n_valid_signed": int(trans_v2.get("n_valid_signed", 0)),
        "n_valid_ratio_2cm": int(trans_v2.get("n_valid_ratio_2cm", 0)),
        "n_valid_ratio_5cm": int(trans_v2.get("n_valid_ratio_5cm", 0)),
        "n_events_total": int(trans_v2.get("n_events_total", 0)),
        "n_boundary": int(trans_v2.get("n_boundary", 0)),
        "n_flicker": int(trans_v2.get("n_flicker", 0)),
        "n_denom_unstable_2cm": int(trans_v2.get("n_denom_unstable_2cm", 0)),
    }
    v1_ratios = trans_v1.get("ratios_over_gt", {})
    return {
        "subset": str(batch["subset"][0]),
        "seq_id": str(batch["seq_id"][0]),
        "text": str(batch["text"][0])[:140],
        "seq_len": int(seq_len),
        "seed": int(seed),
        "v2": v2,
        "v1_legacy_ratios": {
            "onset_xGT": float(v1_ratios.get("onset_positive_closing", 0.0)),
            "release_xGT": float(v1_ratios.get("release_positive_opening", 0.0)),
            "transvel_xGT": float(v1_ratios.get("transition_relative_velocity", 0.0)),
        },
        "dyn": {
            "body_vel_xGT": float(dyn.get("body_velocity_cm_per_frame_over_gt", 0.0)),
            "hand_vel_xGT": float(dyn.get("hand_velocity_cm_per_frame_over_gt", 0.0)),
            "acc_p95_xGT": float(dyn.get("body_acc_p95_cm_per_frame2_over_gt", 0.0)),
            "jerk_p95_xGT": float(dyn.get("body_jerk_p95_cm_per_frame3_over_gt", 0.0)),
            "fft_low": float(dyn.get("fft_low", 0.0)),
            "fft_mid": float(dyn.get("fft_mid", 0.0)),
            "fft_high": float(dyn.get("fft_high", 0.0)),
            "body_vel_cm_per_frame": float(dyn.get("body_velocity_cm_per_frame", 0.0)),
            "hand_vel_cm_per_frame": float(dyn.get("hand_velocity_cm_per_frame", 0.0)),
        },
        "geom": {
            "far_unobserved_error_cm": float(plan_m.get("far_unobserved_error_cm", 0.0)),
            "far_unobserved_root_aligned_error_cm": float(plan_m.get("far_unobserved_root_aligned_error_cm", 0.0)),
            "near_anchor_window_error_cm": float(plan_m.get("near_anchor_window_error_cm", 0.0)),
            "plan_anchor_contact_realization_cm": float(plan_m.get("plan_anchor_contact_realization_cm", 0.0)),
            "anchor_obs_err_cm": float(plan_m.get("observed_anchor_frame_error_cm", 0.0)),
            "global_err_cm": float(plan_m.get("global_joint_error_cm", 0.0)),
            "global_err_root_aligned_cm": float(plan_m.get("root_aligned_joint_error_cm", 0.0)),
            "root_drift_cm": root_drift_cm,
        },
        "per_type_anchor_err": per_type_errs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=Path("configs/training/anchordiff_v18_a1_FULL_DATA.yaml"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("runs/training/stageB_anchordiff_v18_a1_FULL_DATA/final.pt"))
    parser.add_argument("--selection-json", type=Path,
                        default=Path("analyses/2026-05-19_subset_balanced_failure_selection.json"))
    parser.add_argument("--output-json", type=Path,
                        default=Path("analyses/2026-05-19_subset_balanced_metric_v2_baseline_audit.json"))
    parser.add_argument("--output-md", type=Path,
                        default=Path("analyses/2026-05-19_subset_balanced_metric_v2_baseline_audit.md"))
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--bucket", choices=["train", "val"], default="train")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(args.config)

    # Load selection JSON
    sel_payload = json.loads(args.selection_json.read_text(encoding="utf-8"))
    selected_entries = sel_payload.get("selected", [])
    target_seq_ids = {e["seq_id"]: e for e in selected_entries}
    if not target_seq_ids:
        raise SystemExit(f"Empty selection in {args.selection_json}")

    # Build full dataset, then filter to selected seq_ids
    full_ds = _build_dataset(cfg, args.bucket)
    selected_global_indices: list[int] = []
    seen: set[str] = set()
    # Iterate one pass over the dataset, find indices matching seq_id+subset
    # Fast path: if the selection includes dataset_global_index, use it directly
    if all("dataset_global_index" in e for e in selected_entries):
        selected_global_indices = [int(e["dataset_global_index"]) for e in selected_entries]
    else:
        loader = DataLoader(full_ds, batch_size=1, shuffle=False, collate_fn=collate_hoi, num_workers=0)
        for i, batch in enumerate(loader):
            sid = str(batch["seq_id"][0])
            if sid in target_seq_ids and sid not in seen:
                selected_global_indices.append(i)
                seen.add(sid)
                if len(seen) >= len(target_seq_ids):
                    break

    subset_ds = Subset(full_ds, selected_global_indices)
    loader = DataLoader(subset_ds, batch_size=1, shuffle=False, collate_fn=collate_hoi, num_workers=0)
    clip_batches: list[dict[str, Any]] = [b for b in loader]
    if not clip_batches:
        raise SystemExit("No clips matched the selection JSON.")

    print(f"Loaded {len(clip_batches)} clips from selection.", flush=True)

    # Build model + load checkpoint
    model, object_encoder, z_dims = _build_model(cfg, device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("model", state))
    if "object_encoder" in state:
        object_encoder.load_state_dict(state["object_encoder"])
    elif "extra_modules" in state and "object_encoder" in state["extra_modules"]:
        object_encoder.load_state_dict(state["extra_modules"]["object_encoder"])
    model.eval(); object_encoder.eval()
    clip_model = load_clip_text_encoder(
        device=device, model_name=str(cfg.model.text_encoder.clip_version),
        download_root=str(cfg.model.text_encoder.download_root),
    )

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for clip_idx, batch in enumerate(clip_batches):
            print(f"  seed={seed} clip={clip_idx} {batch['subset'][0]}/{batch['seq_id'][0]}", flush=True)
            r = _audit_clip(
                cfg, device, model, object_encoder, clip_model, z_dims,
                batch, seed=seed + clip_idx * 10000,
                threshold=float(args.threshold), cfg_scale=float(args.cfg_scale),
            )
            r["seed_base"] = int(seed)
            r["clip_idx"] = clip_idx
            rows.append(r)

    # Aggregate per (subset, metric)
    per_subset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        per_subset[r["subset"]].append(r)

    def _agg(rows_sub: list[dict[str, Any]], path: str) -> dict[str, float | int]:
        head, _, tail = path.partition(".")
        vals = []
        for r in rows_sub:
            d = r.get(head, {})
            if isinstance(d, dict):
                v = d.get(tail, None)
                if isinstance(v, (int, float)):
                    vals.append(float(v))
        return stats_list(vals)

    subset_summary: dict[str, Any] = {}
    for sname, sub_rows in per_subset.items():
        subset_summary[sname] = {
            "n_rows": len(sub_rows),
            "n_clips": len(set(r["seq_id"] for r in sub_rows)),
            "n_seeds": len(set(r["seed_base"] for r in sub_rows)),
            # v2 transition
            "v2_M2_onset_direction_cm_per_frame": _agg(sub_rows, "v2.M2_onset_direction_cm_per_frame_mean"),
            "v2_M2_release_direction_cm_per_frame": _agg(sub_rows, "v2.M2_release_direction_cm_per_frame_mean"),
            "v2_M3_onset_signed_cm": _agg(sub_rows, "v2.M3_onset_signed_cm_mean"),
            "v2_M3_release_signed_cm": _agg(sub_rows, "v2.M3_release_signed_cm_mean"),
            "v2_M5_clip2cm": _agg(sub_rows, "v2.M5_clip2cm_mean"),
            # event counts
            "n_valid_slope_total": int(sum(r["v2"]["n_valid_slope"] for r in sub_rows)),
            "n_valid_signed_total": int(sum(r["v2"]["n_valid_signed"] for r in sub_rows)),
            "n_events_total": int(sum(r["v2"]["n_events_total"] for r in sub_rows)),
            "n_boundary_total": int(sum(r["v2"]["n_boundary"] for r in sub_rows)),
            "n_flicker_total": int(sum(r["v2"]["n_flicker"] for r in sub_rows)),
            "n_denom_unstable_2cm_total": int(sum(r["v2"]["n_denom_unstable_2cm"] for r in sub_rows)),
            # dynamics
            "body_vel_xGT": _agg(sub_rows, "dyn.body_vel_xGT"),
            "hand_vel_xGT": _agg(sub_rows, "dyn.hand_vel_xGT"),
            "acc_p95_xGT": _agg(sub_rows, "dyn.acc_p95_xGT"),
            "jerk_p95_xGT": _agg(sub_rows, "dyn.jerk_p95_xGT"),
            "fft_mid": _agg(sub_rows, "dyn.fft_mid"),
            "fft_high": _agg(sub_rows, "dyn.fft_high"),
            "body_vel_cm_per_frame": _agg(sub_rows, "dyn.body_vel_cm_per_frame"),
            "hand_vel_cm_per_frame": _agg(sub_rows, "dyn.hand_vel_cm_per_frame"),
            # geometry
            "far_unobserved_error_cm": _agg(sub_rows, "geom.far_unobserved_error_cm"),
            "far_unobserved_root_aligned_error_cm": _agg(sub_rows, "geom.far_unobserved_root_aligned_error_cm"),
            "near_anchor_window_error_cm": _agg(sub_rows, "geom.near_anchor_window_error_cm"),
            "plan_anchor_contact_realization_cm": _agg(sub_rows, "geom.plan_anchor_contact_realization_cm"),
            "root_drift_cm": _agg(sub_rows, "geom.root_drift_cm"),
            "global_err_root_aligned_cm": _agg(sub_rows, "geom.global_err_root_aligned_cm"),
        }
        # Per-anchor-type plan errors (data-side; identical across seeds)
        per_type: dict[str, list[float]] = defaultdict(list)
        for r in sub_rows:
            for t_name, vals in r["per_type_anchor_err"].items():
                if vals.get("n", 0) > 0:
                    per_type[t_name].append(float(vals["mean"]))
        subset_summary[sname]["per_type_anchor_target_to_GT_hand_cm"] = {
            t_name: {"mean_across_clips": float(np.mean(vs)) if vs else 0.0, "n_clips": len(vs)}
            for t_name, vs in per_type.items()
        }

    # Worst / best 3 clips per subset by M2 onset direction
    worst_best: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for sname, sub_rows in per_subset.items():
        # average M2 onset direction across seeds per clip
        agg_per_clip: dict[str, list[float]] = defaultdict(list)
        for r in sub_rows:
            agg_per_clip[r["seq_id"]].append(r["v2"]["M2_onset_direction_cm_per_frame_mean"])
        per_clip_mean = [
            {"seq_id": sid, "m2_onset_mean_across_seeds": float(np.mean(vs))}
            for sid, vs in agg_per_clip.items()
        ]
        per_clip_mean.sort(key=lambda x: x["m2_onset_mean_across_seeds"])
        worst_best[sname] = {
            "worst": per_clip_mean[:3],
            "best": per_clip_mean[-2:][::-1],
        }

    payload = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "selection_json": str(args.selection_json),
        "seeds": seeds,
        "n_clips": len(clip_batches),
        "subset_composition": {sname: subset_summary[sname]["n_clips"] for sname in subset_summary},
        "subset_summary": subset_summary,
        "worst_best_per_subset": worst_best,
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")

    # Markdown
    lines = [
        "# Subset-Balanced Metric-V2 Baseline Audit (Round 9, Task 2)",
        "",
        f"- Config: `{args.config}`",
        f"- Checkpoint: `{args.checkpoint}`",
        f"- Selection: `{args.selection_json}`",
        f"- Seeds: {seeds}",
        f"- Clips: {len(clip_batches)}",
        f"- Subset composition: {payload['subset_composition']}",
        "",
        "## Per-subset aggregate (mean ± std across {clip × seed} rows)",
        "",
        "### Transition v2 (canonical)",
        "",
        "| subset | M2 onset cm/f | M2 release cm/f | M3 onset cm | M3 release cm | n_valid/n_events |",
        "|--------|---------------|------------------|--------------|---------------|------------------|",
    ]
    for sname, s in subset_summary.items():
        m2o = s["v2_M2_onset_direction_cm_per_frame"]
        m2r = s["v2_M2_release_direction_cm_per_frame"]
        m3o = s["v2_M3_onset_signed_cm"]
        m3r = s["v2_M3_release_signed_cm"]
        lines.append(
            f"| {sname} | "
            f"{m2o['mean']:+.3f} ± {m2o['std']:.3f} | "
            f"{m2r['mean']:+.3f} ± {m2r['std']:.3f} | "
            f"{m3o['mean']:+.2f} ± {m3o['std']:.2f} | "
            f"{m3r['mean']:+.2f} ± {m3r['std']:.2f} | "
            f"{s['n_valid_slope_total']}/{s['n_events_total']} |"
        )
    lines += [
        "",
        "### Dynamics (xGT means)",
        "",
        "| subset | body vel xGT | hand vel xGT | acc p95 xGT | jerk p95 xGT | fft_mid | fft_high |",
        "|--------|--------------|--------------|--------------|--------------|---------|----------|",
    ]
    for sname, s in subset_summary.items():
        lines.append(
            f"| {sname} | "
            f"{s['body_vel_xGT']['mean']:.3f} | "
            f"{s['hand_vel_xGT']['mean']:.3f} | "
            f"{s['acc_p95_xGT']['mean']:.3f} | "
            f"{s['jerk_p95_xGT']['mean']:.3f} | "
            f"{s['fft_mid']['mean']:.3f} | "
            f"{s['fft_high']['mean']:.3f} |"
        )
    lines += [
        "",
        "### Geometry",
        "",
        "| subset | far_unobs cm | far_unobs_ra cm | near_anchor cm | anchor_realiz cm | root_drift cm | global_ra cm |",
        "|--------|--------------|------------------|----------------|------------------|----------------|---------------|",
    ]
    for sname, s in subset_summary.items():
        lines.append(
            f"| {sname} | "
            f"{s['far_unobserved_error_cm']['mean']:.2f} | "
            f"{s['far_unobserved_root_aligned_error_cm']['mean']:.2f} | "
            f"{s['near_anchor_window_error_cm']['mean']:.2f} | "
            f"{s['plan_anchor_contact_realization_cm']['mean']:.2f} | "
            f"{s['root_drift_cm']['mean']:.2f} | "
            f"{s['global_err_root_aligned_cm']['mean']:.2f} |"
        )
    lines += [
        "",
        "### Per-anchor-type plan errors (data-side, target → GT hand cm)",
        "",
        "| subset | onset | stable | release |",
        "|--------|-------|--------|---------|",
    ]
    for sname, s in subset_summary.items():
        types = s["per_type_anchor_target_to_GT_hand_cm"]
        onset = types.get("onset", {})
        stable = types.get("stable", {})
        release = types.get("release", {})
        lines.append(
            f"| {sname} | "
            f"{onset.get('mean_across_clips', 0.0):.2f} (n={onset.get('n_clips', 0)}) | "
            f"{stable.get('mean_across_clips', 0.0):.2f} (n={stable.get('n_clips', 0)}) | "
            f"{release.get('mean_across_clips', 0.0):.2f} (n={release.get('n_clips', 0)}) |"
        )
    lines += [
        "",
        "## Worst 3 / best 2 clips per subset by M2 onset direction",
        "",
    ]
    for sname, wb in worst_best.items():
        lines.append(f"### {sname}")
        lines.append("")
        lines.append("Worst:")
        for r in wb["worst"]:
            lines.append(f"- `{r['seq_id']}` — M2 onset cm/f = {r['m2_onset_mean_across_seeds']:+.3f}")
        lines.append("")
        lines.append("Best:")
        for r in wb["best"]:
            lines.append(f"- `{r['seq_id']}` — M2 onset cm/f = {r['m2_onset_mean_across_seeds']:+.3f}")
        lines.append("")

    lines += [
        "",
        "## Per-row detail (clip × seed)",
        "",
        "| subset | seq_id | seed | M2 onset | M2 release | body vel xGT | acc p95 xGT | far_unobs cm | anchor_realiz cm | root_drift cm |",
        "|--------|--------|------|----------|------------|---------------|--------------|---------------|------------------|----------------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['subset']} | {r['seq_id']} | {r['seed_base']} | "
            f"{r['v2']['M2_onset_direction_cm_per_frame_mean']:+.3f} | "
            f"{r['v2']['M2_release_direction_cm_per_frame_mean']:+.3f} | "
            f"{r['dyn']['body_vel_xGT']:.3f} | "
            f"{r['dyn']['acc_p95_xGT']:.3f} | "
            f"{r['geom']['far_unobserved_error_cm']:.2f} | "
            f"{r['geom']['plan_anchor_contact_realization_cm']:.2f} | "
            f"{r['geom']['root_drift_cm']:.2f} |"
        )
    lines.append("")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
