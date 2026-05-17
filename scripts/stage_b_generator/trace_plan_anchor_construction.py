"""Plan anchor construction trace (round 6, Diag B).

For each selected clip, walk the InteractionPlanCompiler pipeline step
by step on the GT dense pseudo-labels:

    raw contact_state / contact_target_xyz
      -> smooth_contact
      -> smooth_target_local
      -> hysteresis_segments
      -> onset / stable / release candidates
      -> merge_nearby_anchors
      -> temporal_nms_budget
      -> lift_target_local_to_world_np

For each surviving anchor, record:

- anchor index, time, part, type (named), target_local, target_world, conf
- source contact segment (start, end, duration, mean conf)
- raw contact_state / contact_smooth / target_local / target_smooth around anchor
- nearest raw GT 0->1 / 1->0 transition time for same part
- distance from anchor_target_world to GT hand at anchor_time
- distance from anchor_target_world to GT hand at the nearest raw event
- distance from anchor_target_world to nearest object surface point
- whether the anchor was merged from multiple candidates; cluster details
- world-vs-local re-lift consistency

Also produces a "raw-event plan" baseline:
- onset_time = actual 0->1 frame
- release_time = actual 1->0 frame
- stable_time = median frame of longest valid contact segment
- target = same-frame target_local at the chosen time

Compares the existing compiler against the raw-event baseline:
- mean anchor-time error per type
- mean target_world->GT-hand error per type
- correlation between anchor_conf and target error

Outputs:
    analyses/2026-05-16_plan_anchor_construction_trace.json
    analyses/2026-05-16_plan_anchor_construction_trace.md
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from diagnostic_common import format_md_table, merge_single_batches, stats_list
from piano.data.interaction_plan_compiler import (
    ANCHOR_TYPE_ONSET, ANCHOR_TYPE_STABLE, ANCHOR_TYPE_RELEASE,
    ANCHOR_TYPE_PHASE_CHANGE, ANCHOR_TYPE_SUPPORT_CHANGE,
    InteractionPlanCompilerConfig,
    _build_change_anchors, _build_contact_anchors,
    _CandidateAnchor,
    compile_interaction_plan,
    lift_target_local_to_world_np,
    merge_nearby_anchors, smooth_categorical_softmax, smooth_contact,
    smooth_target_local, temporal_nms_budget,
)
from recon_ladder_truncated_rollout_diagnostic import (
    _build_selected_batches,
    _load_selection,
)


ANCHOR_TYPE_NAMES = {
    ANCHOR_TYPE_ONSET: "onset",
    ANCHOR_TYPE_STABLE: "stable",
    ANCHOR_TYPE_RELEASE: "release",
    ANCHOR_TYPE_PHASE_CHANGE: "phase_change",
    ANCHOR_TYPE_SUPPORT_CHANGE: "support_change",
}

# Hand part indices in 5-part layout
HAND_PARTS = (("L_hand", 0, 20), ("R_hand", 1, 21))
PART_JOINT = (20, 21, 10, 11, 0)


def _axis_angle_to_rot(aa: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(aa, axis=-1, keepdims=True).clip(min=1e-12)
    k = aa / theta
    K = np.zeros(aa.shape[:-1] + (3, 3), dtype=aa.dtype)
    K[..., 0, 1] = -k[..., 2]; K[..., 0, 2] = k[..., 1]
    K[..., 1, 0] = k[..., 2];  K[..., 1, 2] = -k[..., 0]
    K[..., 2, 0] = -k[..., 1]; K[..., 2, 1] = k[..., 0]
    eye = np.broadcast_to(np.eye(3), K.shape)
    s = np.sin(theta)[..., None]
    c = np.cos(theta)[..., None]
    return eye + s * K + (1 - c) * (K @ K)


def _raw_event_baseline(
    contact_state: np.ndarray, target_local: np.ndarray, seq_len: int,
    *, threshold: float,
) -> list[dict[str, Any]]:
    """Build the 'raw-event' baseline plan from un-smoothed contact transitions.

    For each part with at least one valid contact segment, emits up to three
    anchors per segment:
        - onset at actual 0->1 frame
        - release at actual 1->0 frame
        - stable at median frame inside the segment
    Target = same-frame target_local at the anchor time.
    """
    out: list[dict[str, Any]] = []
    for p in range(contact_state.shape[1]):
        c_bool = (contact_state[:seq_len, p] > float(threshold))
        if not c_bool.any():
            continue
        # Find contiguous segments
        diff = np.diff(c_bool.astype(np.int8), prepend=0, append=0)
        starts = np.where(diff == 1)[0].tolist()
        ends = (np.where(diff == -1)[0] - 1).tolist()
        for s, e in zip(starts, ends):
            duration = int(e - s + 1)
            if duration < 1:
                continue
            stable = int(s + (duration - 1) // 2)
            for t_idx, type_name in ((s, "onset"), (stable, "stable"), (e, "release")):
                out.append({
                    "time": int(t_idx),
                    "part": int(p),
                    "type": type_name,
                    "target_local": target_local[int(t_idx), p].astype(np.float32).tolist(),
                    "duration": int(duration),
                    "segment_start": int(s),
                    "segment_end": int(e),
                })
    return out


def _candidate_cluster_membership(
    final_anchors: list[_CandidateAnchor],
    pre_merge: list[_CandidateAnchor],
    merge_window: int,
) -> list[list[int]]:
    """Approximate which pre-merge candidates were absorbed into each final anchor."""
    membership = [list() for _ in final_anchors]  # type: list[list[int]]
    used = [False] * len(pre_merge)
    for ai, c in enumerate(final_anchors):
        for ci, p in enumerate(pre_merge):
            if used[ci]:
                continue
            if abs(p.time - c.time) <= merge_window:
                membership[ai].append(ci)
                used[ci] = True
    return membership


def _audit_clip(
    batch: dict[str, Any], b: int,
    *, threshold: float, edge_margin: int, contact_smooth_window: int,
) -> dict[str, Any]:
    seq_len = int(batch["seq_len"][b].item())
    subset = str(batch["subset"][b])
    seq_id = str(batch["seq_id"][b])
    text = str(batch["text"][b])

    gt_joints = batch["joints"][b].detach().cpu().numpy().astype(np.float32)
    contact_state = batch["contact_state"][b].detach().cpu().numpy().astype(np.float32)
    target_local = batch["contact_target_xyz"][b].detach().cpu().numpy().astype(np.float32)
    obj_positions = batch["object_positions"][b].detach().cpu().numpy().astype(np.float32)
    obj_rotations = batch["object_rotations"][b].detach().cpu().numpy().astype(np.float32)
    obj_pc = batch["object_pc"][b].detach().cpu().numpy().astype(np.float32)
    phase = batch["phase"][b].detach().cpu().numpy()
    support = batch["support"][b].detach().cpu().numpy()

    # ---- Run compiler manually to inspect intermediates ----
    cfg = InteractionPlanCompilerConfig(num_parts=int(contact_state.shape[1]))
    contact_prob = contact_state[:seq_len].astype(np.float32)
    target_loc = target_local[:seq_len].astype(np.float32)
    obj_pos_seq = obj_positions[:seq_len]
    obj_rot_seq = obj_rotations[:seq_len]

    contact_smooth = smooth_contact(contact_prob, cfg.contact_smooth_window)
    num_phase = max(int(phase.max()) + 1, cfg.num_phase_classes)
    num_support = cfg.num_support_classes
    phase_softmax = np.zeros((seq_len, num_phase), dtype=np.float32)
    support_softmax = np.zeros((seq_len, num_support), dtype=np.float32)
    phase_safe = np.clip(phase[:seq_len].astype(np.int64), 0, num_phase - 1)
    support_safe = np.clip(support[:seq_len].astype(np.int64), 0, num_support - 1)
    phase_softmax[np.arange(seq_len), phase_safe] = 1.0
    support_softmax[np.arange(seq_len), support_safe] = 1.0
    phase_smooth = smooth_categorical_softmax(phase_softmax, cfg.phase_smooth_window)
    support_smooth = smooth_categorical_softmax(support_softmax, cfg.support_smooth_window)
    target_smooth_local = smooth_target_local(target_loc, contact_smooth, cfg.target_smooth_window)

    from piano.data.interaction_plan_compiler import _entropy
    phase_entropy = _entropy(phase_smooth)
    support_entropy = _entropy(support_smooth)

    contact_cands, segments = _build_contact_anchors(
        contact_smooth, target_smooth_local, phase_smooth, support_smooth,
        phase_entropy, support_entropy, cfg,
    )
    change_cands = _build_change_anchors(phase_smooth, support_smooth, cfg)
    pre_merge = contact_cands + change_cands
    merged = merge_nearby_anchors(pre_merge, cfg)
    kept = temporal_nms_budget(merged, cfg)
    target_world_full = lift_target_local_to_world_np(
        target_smooth_local, obj_pos_seq, obj_rot_seq,
    )

    # Membership: which contact_cands ended up in each merged anchor (approx)
    membership = _candidate_cluster_membership(merged, contact_cands, cfg.merge_window)

    # Per-hand raw event times for nearest-event analysis
    raw_event_times: dict[int, dict[str, list[int]]] = {}
    for p in range(contact_prob.shape[1]):
        c_bool = (contact_prob[:, p] > float(threshold))
        if c_bool.size < 2:
            raw_event_times[p] = {"onsets": [], "releases": []}
            continue
        onset = (c_bool[1:] & ~c_bool[:-1])
        release = (~c_bool[1:] & c_bool[:-1])
        raw_event_times[p] = {
            "onsets": (np.where(onset)[0] + 1).tolist(),
            "releases": (np.where(release)[0] + 1).tolist(),
        }

    # World pose / object surface
    R = _axis_angle_to_rot(obj_rot_seq)
    obj_pc_world = np.einsum("tij,nj->tni", R, obj_pc) + obj_pos_seq[:, None, :]

    # Per-anchor records (only KEPT, those that survive NMS)
    anchor_records: list[dict[str, Any]] = []
    for ai, c in enumerate(kept):
        type_name = ANCHOR_TYPE_NAMES.get(c.type_id, str(c.type_id))
        active_parts = np.where(c.parts > 0)[0].tolist()
        # Re-lift anchor_target_local at anchor_time
        anchor_target_local = c.target_local  # (P, 3) — may carry zeros for inactive parts
        anchor_target_world = target_world_full[c.time]  # (P, 3) — for compiler's reference frame
        per_part: list[dict[str, Any]] = []
        for p in active_parts:
            target_w = (c.parts[p] * target_world_full[c.time, p])  # mask by part
            target_w = target_world_full[c.time, p]
            joint = PART_JOINT[p]
            t_a = int(min(max(0, c.time), seq_len - 1))
            hand_world = gt_joints[t_a, joint]
            err_hand = float(np.linalg.norm(target_w - hand_world) * 100.0)
            d_surface = float(np.linalg.norm(
                obj_pc_world[t_a] - target_w[None, :], axis=-1
            ).min() * 100.0)
            # Nearest raw event for this part
            raw_part = raw_event_times.get(int(p), {"onsets": [], "releases": []})
            raw_times = raw_part["onsets"] + raw_part["releases"]
            if raw_times:
                nearest_t = int(min(raw_times, key=lambda x: abs(x - t_a)))
                time_err = abs(nearest_t - t_a)
                # Distance from anchor_target_world to GT hand at nearest_t
                nearest_clamped = int(min(max(0, nearest_t), seq_len - 1))
                hand_nearest = gt_joints[nearest_clamped, joint]
                err_hand_nearest = float(np.linalg.norm(target_w - hand_nearest) * 100.0)
            else:
                nearest_t = -1
                time_err = -1
                err_hand_nearest = -1.0
            # Local-vs-world re-lift consistency
            local_t = anchor_target_local[p]
            lifted = R[t_a] @ local_t + obj_pos_seq[t_a]
            consistency_err = float(np.linalg.norm(lifted - target_w))
            per_part.append({
                "part": int(p),
                "joint": int(joint),
                "anchor_target_local": [float(v) for v in local_t.tolist()],
                "anchor_target_world": [float(v) for v in target_w.tolist()],
                "anchor_to_GT_hand_at_anchor_time_cm": err_hand,
                "anchor_to_GT_hand_at_nearest_raw_event_cm": err_hand_nearest,
                "anchor_to_object_surface_cm": d_surface,
                "nearest_raw_event_time": int(nearest_t),
                "anchor_time_minus_nearest_event_frames": int(time_err)
                if time_err >= 0 else -1,
                "anchor_target_local_world_consistency_meters": consistency_err,
            })
        # Cluster info
        cluster = membership[merged.index(c)] if c in merged else []
        cluster_info = []
        for ci in cluster:
            sc = contact_cands[ci]
            cluster_info.append({
                "candidate_idx": int(ci),
                "time": int(sc.time),
                "type": ANCHOR_TYPE_NAMES.get(sc.type_id, str(sc.type_id)),
                "part": int(np.argmax(sc.parts).item()) if sc.parts.any() else -1,
                "confidence": float(sc.confidence),
            })
        anchor_records.append({
            "anchor_idx": int(ai),
            "anchor_time": int(c.time),
            "anchor_type": type_name,
            "anchor_parts": active_parts,
            "anchor_conf": float(c.confidence),
            "anchor_duration": int(c.duration),
            "is_change": bool(c.is_change),
            "cluster_size_pre_merge": int(len(cluster)),
            "cluster_members": cluster_info,
            "per_part": per_part,
        })

    # ---- Raw-event baseline ----
    raw_baseline = _raw_event_baseline(
        contact_prob, target_loc, seq_len, threshold=threshold,
    )
    # Compute per-type metrics for raw baseline (only for hand parts, in cm)
    raw_perf: dict[str, list[float]] = {"onset": [], "stable": [], "release": []}
    raw_time_err: dict[str, list[int]] = {"onset": [], "stable": [], "release": []}
    for r in raw_baseline:
        p = int(r["part"])
        if p > 1:
            continue
        joint = PART_JOINT[p]
        t_a = int(min(max(0, r["time"]), seq_len - 1))
        local_v = np.asarray(r["target_local"], dtype=np.float32)
        world_v = R[t_a] @ local_v + obj_pos_seq[t_a]
        hand_world = gt_joints[t_a, joint]
        err = float(np.linalg.norm(world_v - hand_world) * 100.0)
        raw_perf[r["type"]].append(err)
        raw_time_err[r["type"]].append(0)  # raw baseline IS the event

    # Per-anchor-type stats on compiler output
    type_stats: dict[str, dict[str, Any]] = {}
    for t_name in ("onset", "stable", "release", "phase_change", "support_change"):
        errs = []
        time_errs = []
        for rec in anchor_records:
            if rec["anchor_type"] != t_name:
                continue
            for pp in rec["per_part"]:
                if pp["part"] > 1:
                    continue
                errs.append(pp["anchor_to_GT_hand_at_anchor_time_cm"])
                if pp["anchor_time_minus_nearest_event_frames"] >= 0:
                    time_errs.append(pp["anchor_time_minus_nearest_event_frames"])
        type_stats[t_name] = {
            "n": len(errs),
            "target_to_hand_cm": stats_list(errs) if errs else {"mean": 0.0, "n": 0},
            "time_err_frames": stats_list(time_errs) if time_errs else {"mean": 0.0, "n": 0},
        }

    # Part-level stats
    part_stats: dict[str, dict[str, Any]] = {}
    for p_name, p_idx, _joint in HAND_PARTS:
        errs = []
        for rec in anchor_records:
            for pp in rec["per_part"]:
                if pp["part"] != p_idx:
                    continue
                errs.append(pp["anchor_to_GT_hand_at_anchor_time_cm"])
        part_stats[p_name] = {
            "n": len(errs),
            "target_to_hand_cm": stats_list(errs) if errs else {"mean": 0.0, "n": 0},
        }

    # anchor_conf vs target_err correlation (hand parts only)
    pairs = []
    for rec in anchor_records:
        for pp in rec["per_part"]:
            if pp["part"] > 1:
                continue
            pairs.append((rec["anchor_conf"], pp["anchor_to_GT_hand_at_anchor_time_cm"]))
    if len(pairs) >= 3:
        a = np.asarray([x[0] for x in pairs], dtype=np.float64)
        b_arr = np.asarray([x[1] for x in pairs], dtype=np.float64)
        if a.std() > 1e-8 and b_arr.std() > 1e-8:
            corr = float(np.corrcoef(a, b_arr)[0, 1])
        else:
            corr = 0.0
    else:
        corr = 0.0

    # Raw vs compiler comparison
    compiler_onset_errs = type_stats.get("onset", {}).get("target_to_hand_cm", {}).get("mean", 0.0)
    compiler_stable_errs = type_stats.get("stable", {}).get("target_to_hand_cm", {}).get("mean", 0.0)
    compiler_release_errs = type_stats.get("release", {}).get("target_to_hand_cm", {}).get("mean", 0.0)
    raw_onset_mean = float(np.mean(raw_perf["onset"])) if raw_perf["onset"] else 0.0
    raw_stable_mean = float(np.mean(raw_perf["stable"])) if raw_perf["stable"] else 0.0
    raw_release_mean = float(np.mean(raw_perf["release"])) if raw_perf["release"] else 0.0
    raw_vs_compiler = {
        "onset": {
            "compiler_mean_cm": compiler_onset_errs,
            "raw_mean_cm": raw_onset_mean,
            "delta_cm": float(compiler_onset_errs - raw_onset_mean),
            "n_compiler": type_stats.get("onset", {}).get("n", 0),
            "n_raw": len(raw_perf["onset"]),
        },
        "stable": {
            "compiler_mean_cm": compiler_stable_errs,
            "raw_mean_cm": raw_stable_mean,
            "delta_cm": float(compiler_stable_errs - raw_stable_mean),
            "n_compiler": type_stats.get("stable", {}).get("n", 0),
            "n_raw": len(raw_perf["stable"]),
        },
        "release": {
            "compiler_mean_cm": compiler_release_errs,
            "raw_mean_cm": raw_release_mean,
            "delta_cm": float(compiler_release_errs - raw_release_mean),
            "n_compiler": type_stats.get("release", {}).get("n", 0),
            "n_raw": len(raw_perf["release"]),
        },
    }

    return {
        "subset": subset, "seq_id": seq_id, "text": text[:120], "seq_len": seq_len,
        "n_segments": len(segments),
        "n_contact_candidates_pre_merge": len(contact_cands),
        "n_change_candidates_pre_merge": len(change_cands),
        "n_anchors_after_merge": len(merged),
        "n_anchors_kept_after_nms": len(kept),
        "anchor_conf_vs_target_err_corr": corr,
        "type_stats": type_stats,
        "part_stats": part_stats,
        "raw_vs_compiler": raw_vs_compiler,
        "anchors": anchor_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=Path("configs/training/anchordiff_v18_a1_FULL_DATA.yaml"))
    parser.add_argument("--selection-json", type=Path,
                        default=Path("analyses/2026-05-13_v18_recon_ladder_truncated_rollout.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("analyses/2026-05-16_plan_anchor_construction_trace.json"))
    parser.add_argument("--md", type=Path,
                        default=Path("analyses/2026-05-16_plan_anchor_construction_trace.md"))
    parser.add_argument("--max-clips", type=int, default=16)
    parser.add_argument("--num-candidates", type=int, default=256)
    parser.add_argument("--bucket", choices=["train", "val"], default="train")
    parser.add_argument("--balanced-subsets", action="store_true", default=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--edge-margin", type=int, default=5)
    parser.add_argument("--contact-smooth-window", type=int, default=5)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    selection = _load_selection(args.selection_json, max_clips=int(args.max_clips))
    selected = _build_selected_batches(
        cfg, bucket=args.bucket, balanced_subsets=bool(args.balanced_subsets),
        num_candidates=int(args.num_candidates), selection=selection,
        max_clips=int(args.max_clips), threshold=float(args.threshold),
    )
    if not selected:
        raise SystemExit("No clips selected")
    batch = merge_single_batches([item[1] for item in selected])
    B = int(batch["motion"].shape[0])

    audits = [
        _audit_clip(
            batch, b,
            threshold=float(args.threshold),
            edge_margin=int(args.edge_margin),
            contact_smooth_window=int(args.contact_smooth_window),
        )
        for b in range(B)
    ]

    subset_counts: dict[str, int] = {}
    for a in audits:
        subset_counts[a["subset"]] = subset_counts.get(a["subset"], 0) + 1

    # Aggregate across clips per type
    def _agg_type(type_name: str, field_path: list[str]) -> list[float]:
        out = []
        for a in audits:
            d = a["type_stats"].get(type_name, {})
            for k in field_path:
                d = d.get(k, {}) if isinstance(d, dict) else {}
            if isinstance(d, (int, float)):
                out.append(float(d))
        return out

    type_summary: dict[str, dict[str, Any]] = {}
    for t_name in ("onset", "stable", "release", "phase_change", "support_change"):
        target_means = _agg_type(t_name, ["target_to_hand_cm", "mean"])
        time_means = _agg_type(t_name, ["time_err_frames", "mean"])
        type_summary[t_name] = {
            "mean_target_to_hand_cm_across_clips": (
                float(np.mean(target_means)) if target_means else 0.0
            ),
            "mean_time_err_frames_across_clips": (
                float(np.mean(time_means)) if time_means else 0.0
            ),
            "n_clips_with_type": int(len(target_means)),
        }

    # Per-hand-part aggregate
    part_summary: dict[str, dict[str, Any]] = {}
    for p_name in ("L_hand", "R_hand"):
        target_means = [
            float(a["part_stats"].get(p_name, {}).get("target_to_hand_cm", {}).get("mean", 0.0))
            for a in audits
        ]
        part_summary[p_name] = {
            "mean_target_to_hand_cm_across_clips": float(np.mean(target_means)),
        }

    # anchor_conf vs target_err mean correlation
    corrs = [a["anchor_conf_vs_target_err_corr"] for a in audits]
    mean_corr = float(np.mean(corrs)) if corrs else 0.0

    # Raw-vs-compiler aggregate
    raw_vs_compiler_agg = {}
    for t_name in ("onset", "stable", "release"):
        compiler_vals = [a["raw_vs_compiler"][t_name]["compiler_mean_cm"] for a in audits]
        raw_vals = [a["raw_vs_compiler"][t_name]["raw_mean_cm"] for a in audits]
        raw_vs_compiler_agg[t_name] = {
            "compiler_mean_cm_across_clips": float(np.mean(compiler_vals)),
            "raw_mean_cm_across_clips": float(np.mean(raw_vals)),
            "delta_cm_across_clips": float(np.mean(compiler_vals) - np.mean(raw_vals)),
        }

    aggregate = {
        "n_clips": B,
        "subset_counts": subset_counts,
        "type_summary": type_summary,
        "part_summary": part_summary,
        "mean_anchor_conf_target_err_corr_across_clips": mean_corr,
        "raw_vs_compiler_summary": raw_vs_compiler_agg,
    }

    payload = {
        "config": str(args.config),
        "selection_json": str(args.selection_json),
        "n_clips": B,
        "aggregate": aggregate,
        "clips": audits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")

    # Markdown report
    lines = [
        "# Plan Anchor Construction Trace (Round 6, Diag B)",
        "",
        f"- Config: `{args.config}`",
        f"- Selection: `{args.selection_json}`",
        f"- Clips: {B}",
        f"- Subset composition: {subset_counts}",
        "",
        "## Anchor-type summary (target_world -> GT hand error, hand parts only)",
        "",
        "| anchor_type | mean cm | time err (frames) | clips |",
        "|-------------|---------|--------------------|------|",
    ]
    for t_name in ("onset", "stable", "release", "phase_change", "support_change"):
        ts = type_summary[t_name]
        lines.append(
            f"| {t_name} | "
            f"{ts['mean_target_to_hand_cm_across_clips']:.2f} | "
            f"{ts['mean_time_err_frames_across_clips']:.2f} | "
            f"{ts['n_clips_with_type']} |"
        )
    lines += [
        "",
        "## Per-hand-part summary",
        "",
        "| part | mean target -> hand (cm) |",
        "|------|--------------------------|",
    ]
    for p in ("L_hand", "R_hand"):
        lines.append(f"| {p} | {part_summary[p]['mean_target_to_hand_cm_across_clips']:.2f} |")
    lines += [
        "",
        "## anchor_conf vs target-error correlation",
        "",
        f"- Per-clip Pearson r (mean across clips): **{mean_corr:.3f}**",
        "  (negative -> conf usefully predicts low error; near 0 -> unusable as reliability weight)",
        "",
        "## Raw-event baseline vs current compiler (cm, hand parts)",
        "",
        "| anchor_type | compiler | raw baseline | delta |",
        "|-------------|----------|--------------|-------|",
    ]
    for t_name in ("onset", "stable", "release"):
        rv = raw_vs_compiler_agg[t_name]
        lines.append(
            f"| {t_name} | "
            f"{rv['compiler_mean_cm_across_clips']:.2f} | "
            f"{rv['raw_mean_cm_across_clips']:.2f} | "
            f"{rv['delta_cm_across_clips']:+.2f} |"
        )
    lines += [
        "",
        "## Per-clip anchor counts",
        "",
    ]
    rows = [["subset", "seq_id", "T", "segs", "cand_pre", "merge", "kept",
             "conf-err corr", "raw-comp Δonset cm", "raw-comp Δrel cm"]]
    for a in audits:
        rv_on = a["raw_vs_compiler"]["onset"]["delta_cm"]
        rv_rel = a["raw_vs_compiler"]["release"]["delta_cm"]
        rows.append([
            a["subset"], a["seq_id"], a["seq_len"],
            a["n_segments"], a["n_contact_candidates_pre_merge"],
            a["n_anchors_after_merge"], a["n_anchors_kept_after_nms"],
            f"{a['anchor_conf_vs_target_err_corr']:.2f}",
            f"{rv_on:+.2f}", f"{rv_rel:+.2f}",
        ])
    lines.append(format_md_table(rows))
    lines.append("")
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.md}")


if __name__ == "__main__":
    main()
