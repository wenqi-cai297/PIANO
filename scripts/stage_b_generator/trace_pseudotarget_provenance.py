"""Pseudo-target provenance trace (round 6, Diag A).

For each selected clip and each onset/release event on each hand, trace
``contact_target_xyz`` from raw pseudo-label npz through dataset post-
processing through world-frame lift, and compare against GT joint
positions and nearest-object-surface points.

This script does NOT modify pseudo labels or any production code path.
It reads raw npz files directly so the provenance trace is independent
of the dataset's load-time normalisation.

Outputs:
    analyses/2026-05-16_pseudotarget_provenance_trace.json
    analyses/2026-05-16_pseudotarget_provenance_trace.md

Designed to consume the same 16-clip selection used in Round 5 (selection
JSON at analyses/2026-05-13_v18_recon_ladder_truncated_rollout.json) so
findings line up against Round 5 Diag 2 numbers (anchor target → GT hand
35.75 cm, hand → pseudo target 17.24 cm in contact, etc.).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from diagnostic_common import format_md_table, merge_single_batches, stats_list
from recon_ladder_truncated_rollout_diagnostic import (
    _build_selected_batches,
    _load_selection,
)


# Mirrors src/piano/data/interaction_plan_compiler.NUM_PARTS_DEFAULT layout.
HAND_SPECS = (("L_hand", 20, 0), ("R_hand", 21, 1))
PART_NAMES = ("left_hand", "right_hand", "left_foot", "right_foot", "pelvis")
PART_JOINT = (20, 21, 10, 11, 0)


def _axis_angle_to_rot(aa: np.ndarray) -> np.ndarray:
    """Rodrigues axis-angle (..., 3) -> rotation matrix (..., 3, 3)."""
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


def _smooth_target_local_contact_weighted(
    target_local: np.ndarray,           # (T, P, 3)
    contact_smooth: np.ndarray,         # (T, P)
    window: int,
) -> np.ndarray:
    """Replicates interaction_plan_compiler.smooth_target_local."""
    if window <= 1:
        return target_local.astype(np.float32, copy=True)
    from scipy.ndimage import uniform_filter1d
    w = contact_smooth[..., None].astype(np.float32)
    weighted = target_local.astype(np.float32) * w
    num = uniform_filter1d(weighted, size=window, axis=0, mode="nearest")
    den = uniform_filter1d(w, size=window, axis=0, mode="nearest")
    return num / np.clip(den, 1e-6, None)


def _segment_pairs(c_bool: np.ndarray) -> list[tuple[int, int]]:
    """Pair onsets (0->1) and releases (1->0) into (start, end_exclusive) segments."""
    if c_bool.size < 2:
        return []
    onset = (c_bool[1:] & ~c_bool[:-1])
    release = (~c_bool[1:] & c_bool[:-1])
    onset_idx = (np.where(onset)[0] + 1).tolist()
    release_idx = (np.where(release)[0] + 1).tolist()
    segments: list[tuple[int, int]] = []
    ri = 0
    for o in onset_idx:
        while ri < len(release_idx) and release_idx[ri] <= o:
            ri += 1
        if ri < len(release_idx):
            segments.append((o, release_idx[ri]))
            ri += 1
        else:
            segments.append((o, int(c_bool.size)))
    return segments


def _load_raw_npz(npz_path: Path, seq_len: int) -> dict[str, Any]:
    """Read raw pseudo-label npz and report what's there."""
    if not npz_path.exists():
        return {"exists": False, "path": str(npz_path)}
    data = np.load(npz_path, allow_pickle=False)
    fields = list(data.files)
    info: dict[str, Any] = {"exists": True, "path": str(npz_path), "fields": fields}
    if "contact_target_xyz_gt" in fields:
        ct_xyz = data["contact_target_xyz_gt"][:seq_len].astype(np.float32)
        info["source"] = "contact_target_xyz_gt"
        info["raw_target_local"] = ct_xyz
    elif "contact_target" in fields and "patch_centers" in fields:
        soft = data["contact_target"][:seq_len].astype(np.float32)
        pc = data["patch_centers"].astype(np.float32)
        info["source"] = "patch_centroid_fallback"
        info["raw_target_local"] = np.einsum("tbk,kd->tbd", soft, pc).astype(np.float32)
    else:
        info["source"] = "unknown"
        info["raw_target_local"] = None
    # capture other fields summary
    info["has_contact_target"] = "contact_target" in fields
    info["has_patch_centers"] = "patch_centers" in fields
    return info


def _find_npz_path(cfg, subset: str, seq_id: str) -> Path | None:
    """Locate the raw pseudo-label npz for one clip."""
    subdir = cfg.data.get("pseudo_label_subdir", None)
    if subdir is None:
        return None
    for entry in cfg.data.datasets:
        if str(entry.name) == subset:
            return Path(entry.root) / str(subdir) / f"{seq_id}.npz"
    return None


def _audit_clip(
    cfg,
    batch: dict[str, Any], b: int,
    *,
    threshold: float, edge_margin: int, window_k: int,
    target_smooth_window: int, surface_samples: int,
) -> dict[str, Any]:
    seq_len = int(batch["seq_len"][b].item())
    subset = str(batch["subset"][b])
    seq_id = str(batch["seq_id"][b])
    text = str(batch["text"][b])

    # Dataset side: contact_state already passed through suppress_sitting_hand_contact
    gt_joints = batch["joints"][b].detach().cpu().numpy().astype(np.float32)
    contact_state = batch["contact_state"][b].detach().cpu().numpy().astype(np.float32)
    obj_positions = batch["object_positions"][b].detach().cpu().numpy().astype(np.float32)
    obj_rotations = batch["object_rotations"][b].detach().cpu().numpy().astype(np.float32)
    obj_pc = batch["object_pc"][b].detach().cpu().numpy().astype(np.float32)
    ds_target_local = batch["contact_target_xyz"][b].detach().cpu().numpy().astype(np.float32)
    phase = batch["phase"][b].detach().cpu().numpy()
    support = batch["support"][b].detach().cpu().numpy()

    # Raw npz side
    npz_path = _find_npz_path(cfg, subset, seq_id)
    raw_info = _load_raw_npz(npz_path, seq_len) if npz_path else {"exists": False}
    raw_target_local = raw_info.get("raw_target_local")

    # World-pose lift
    R = _axis_angle_to_rot(obj_rotations[:seq_len])
    t_world = obj_positions[:seq_len]

    def _lift(local: np.ndarray) -> np.ndarray:
        return np.einsum("tij,tpj->tpi", R, local[:seq_len]) + t_world[:, None, :]

    ds_target_world = _lift(ds_target_local)
    raw_target_world = _lift(raw_target_local) if raw_target_local is not None else None

    # Subsample object_pc for nearest-surface query (full pc would be slow per-frame)
    pc_sub = obj_pc
    if pc_sub.shape[0] > surface_samples:
        idx = np.random.RandomState(0).choice(pc_sub.shape[0], surface_samples, replace=False)
        pc_sub = pc_sub[idx]
    obj_pc_world = np.einsum("tij,nj->tni", R, pc_sub) + obj_positions[:seq_len, None, :]

    # Smoothed target (mirrors compiler) for hand parts only
    from scipy.ndimage import uniform_filter1d
    contact_smooth = uniform_filter1d(
        contact_state[:seq_len].astype(np.float32), size=5, axis=0, mode="nearest"
    )
    target_smooth = _smooth_target_local_contact_weighted(
        ds_target_local[:seq_len], contact_smooth, target_smooth_window
    )
    target_smooth_world = _lift(target_smooth)

    # Identify events on each hand
    events_records: list[dict[str, Any]] = []
    per_hand_aggregate: dict[str, dict[str, Any]] = {}
    target_non_contact_used = 0
    target_non_contact_total = 0
    local_jump_events = 0
    world_jump_events = 0
    target_world_vs_lift_consistency_max = 0.0

    for part_name, joint, p_idx in HAND_SPECS:
        c_bool = (contact_state[:seq_len, p_idx] > float(threshold))
        segments = _segment_pairs(c_bool)
        onset_idx = [s for s, _ in segments]
        release_idx = [e for _, e in segments if e <= seq_len]
        # Per-hand counts of "target populated during non-contact frames"
        non_contact = ~c_bool
        if non_contact.any():
            tgt_nc = ds_target_local[:seq_len, p_idx][non_contact]
            populated = (np.linalg.norm(tgt_nc, axis=-1) > 1e-4)
            target_non_contact_used += int(populated.sum())
            target_non_contact_total += int(non_contact.sum())

        # Per-event details
        hand_world = gt_joints[:seq_len, joint]
        # event records
        for (s, e) in segments:
            for kind, t_ev in (("onset", s), ("release", e)):
                t_ev = int(min(max(0, t_ev), seq_len - 1))
                tgt_world_ev = ds_target_world[t_ev, p_idx]
                d_hand = float(np.linalg.norm(tgt_world_ev - hand_world[t_ev]) * 100.0)
                d_surface = float(np.linalg.norm(
                    obj_pc_world[t_ev] - tgt_world_ev[None, :], axis=-1
                ).min() * 100.0)
                d_hand_to_surface = float(np.linalg.norm(
                    obj_pc_world[t_ev] - hand_world[t_ev][None, :], axis=-1
                ).min() * 100.0)
                # Local jump magnitude across event (cm) — same-part, frame-to-frame
                window_lo = max(0, t_ev - 2)
                window_hi = min(seq_len - 1, t_ev + 2)
                if window_hi > window_lo:
                    local_seg = ds_target_local[window_lo:window_hi + 1, p_idx]
                    local_jumps_cm = (np.linalg.norm(np.diff(local_seg, axis=0), axis=-1) * 100.0).tolist()
                    max_local_jump = float(max(local_jumps_cm)) if local_jumps_cm else 0.0
                    world_seg = ds_target_world[window_lo:window_hi + 1, p_idx]
                    world_jumps_cm = (np.linalg.norm(np.diff(world_seg, axis=0), axis=-1) * 100.0).tolist()
                    max_world_jump = float(max(world_jumps_cm)) if world_jumps_cm else 0.0
                else:
                    max_local_jump = 0.0
                    max_world_jump = 0.0
                if max_local_jump > 10.0:
                    local_jump_events += 1
                if max_world_jump > 10.0:
                    world_jump_events += 1
                # Smoothed-vs-raw same-frame target gap (local cm)
                gap_smoothed_cm = float(
                    np.linalg.norm(target_smooth[t_ev, p_idx] - ds_target_local[t_ev, p_idx]) * 100.0
                )
                # Boundary flags
                is_boundary = t_ev < int(edge_margin) or t_ev > seq_len - 1 - int(edge_margin)
                # Flicker: segment duration
                duration = max(1, e - s)
                events_records.append({
                    "part": part_name, "kind": kind, "frame": int(t_ev),
                    "segment_start": int(s), "segment_end": int(e),
                    "segment_duration": int(duration),
                    "is_boundary": bool(is_boundary),
                    "is_flicker": bool(duration <= 2),
                    "target_world_to_GT_hand_cm": d_hand,
                    "target_world_to_object_surface_cm": d_surface,
                    "GT_hand_to_object_surface_cm": d_hand_to_surface,
                    "max_local_target_jump_2f_cm": max_local_jump,
                    "max_world_target_jump_2f_cm": max_world_jump,
                    "smoothed_vs_raw_target_local_gap_cm": gap_smoothed_cm,
                })

        # Per-hand aggregate over contact frames
        in_contact = c_bool
        d_target_to_hand_contact = (
            float(np.linalg.norm(
                ds_target_world[in_contact, p_idx] - hand_world[in_contact],
                axis=-1
            ).mean() * 100.0) if in_contact.any() else 0.0
        )
        if in_contact.any():
            d_target_to_surface_contact = []
            for ti in np.where(in_contact)[0].tolist():
                d_target_to_surface_contact.append(float(
                    np.linalg.norm(
                        obj_pc_world[ti] - ds_target_world[ti, p_idx][None, :],
                        axis=-1
                    ).min() * 100.0
                ))
            d_target_to_surface_contact_mean = float(np.mean(d_target_to_surface_contact))
        else:
            d_target_to_surface_contact_mean = 0.0
        per_hand_aggregate[part_name] = {
            "n_contact_frames": int(in_contact.sum()),
            "n_onsets": int(len(onset_idx)),
            "n_releases": int(len(release_idx)),
            "mean_target_to_GT_hand_in_contact_cm": d_target_to_hand_contact,
            "mean_target_to_object_surface_in_contact_cm": d_target_to_surface_contact_mean,
        }

        # World-vs-lift consistency check: dataset stores local; lift here.
        # Compute |R local + t - lift_world| max in contact frames (sanity check
        # that frame indexing matches between dataset and our lift).
        if in_contact.any():
            ti = np.where(in_contact)[0][0]
            R_ti = R[ti]
            local_ti = ds_target_local[ti, p_idx]
            expected = R_ti @ local_ti + t_world[ti]
            got = ds_target_world[ti, p_idx]
            gap = float(np.linalg.norm(expected - got))
            target_world_vs_lift_consistency_max = max(
                target_world_vs_lift_consistency_max, gap
            )

    # Aggregate per-clip
    n_events = len(events_records)
    n_boundary = sum(1 for e in events_records if e["is_boundary"])
    n_flicker = sum(1 for e in events_records if e["is_flicker"])
    n_warn_10 = sum(1 for e in events_records if e["target_world_to_GT_hand_cm"] > 10.0)
    n_warn_15 = sum(1 for e in events_records if e["target_world_to_GT_hand_cm"] > 15.0)
    n_severe_25 = sum(1 for e in events_records if e["target_world_to_GT_hand_cm"] > 25.0)

    return {
        "subset": subset, "seq_id": seq_id, "text": text[:120], "seq_len": seq_len,
        "raw_npz": {k: v for k, v in raw_info.items() if k != "raw_target_local"},
        "per_hand": per_hand_aggregate,
        "events": events_records,
        "summary": {
            "n_events": int(n_events),
            "n_boundary_events": int(n_boundary),
            "n_flicker_events": int(n_flicker),
            "n_warn_target_to_hand_over_10cm": int(n_warn_10),
            "n_warn_target_to_hand_over_15cm": int(n_warn_15),
            "n_severe_target_to_hand_over_25cm": int(n_severe_25),
            "n_target_populated_during_non_contact": int(target_non_contact_used),
            "n_non_contact_frames_total": int(target_non_contact_total),
            "pct_target_populated_during_non_contact": (
                100.0 * target_non_contact_used / target_non_contact_total
                if target_non_contact_total > 0 else 0.0
            ),
            "n_event_local_jump_over_10cm": int(local_jump_events),
            "n_event_world_jump_over_10cm": int(world_jump_events),
            "target_world_vs_lift_consistency_max_meters": float(
                target_world_vs_lift_consistency_max
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/training/anchordiff_v18_a1_FULL_DATA.yaml"),
    )
    parser.add_argument(
        "--selection-json", type=Path,
        default=Path("analyses/2026-05-13_v18_recon_ladder_truncated_rollout.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("analyses/2026-05-16_pseudotarget_provenance_trace.json"),
    )
    parser.add_argument(
        "--md", type=Path,
        default=Path("analyses/2026-05-16_pseudotarget_provenance_trace.md"),
    )
    parser.add_argument("--max-clips", type=int, default=16)
    parser.add_argument("--num-candidates", type=int, default=256)
    parser.add_argument("--bucket", choices=["train", "val"], default="train")
    parser.add_argument("--balanced-subsets", action="store_true", default=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--edge-margin", type=int, default=5)
    parser.add_argument("--window-k", type=int, default=10)
    parser.add_argument("--target-smooth-window", type=int, default=5)
    parser.add_argument("--surface-samples", type=int, default=512)
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

    audits: list[dict[str, Any]] = []
    for b in range(B):
        a = _audit_clip(
            cfg, batch, b,
            threshold=float(args.threshold),
            edge_margin=int(args.edge_margin),
            window_k=int(args.window_k),
            target_smooth_window=int(args.target_smooth_window),
            surface_samples=int(args.surface_samples),
        )
        audits.append(a)

    subset_counts: dict[str, int] = {}
    for a in audits:
        subset_counts[a["subset"]] = subset_counts.get(a["subset"], 0) + 1

    # Cross-clip stats
    all_events = [e for a in audits for e in a["events"]]

    def _agg(field: str) -> dict[str, float | int]:
        return stats_list([e[field] for e in all_events])

    sources = {}
    for a in audits:
        src = a["raw_npz"].get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1

    aggregate = {
        "n_clips": B,
        "subset_counts": subset_counts,
        "raw_source_counts": sources,
        "n_total_events": len(all_events),
        "n_boundary_events": int(sum(1 for e in all_events if e["is_boundary"])),
        "n_flicker_events": int(sum(1 for e in all_events if e["is_flicker"])),
        "target_world_to_GT_hand_cm": _agg("target_world_to_GT_hand_cm"),
        "target_world_to_object_surface_cm": _agg("target_world_to_object_surface_cm"),
        "GT_hand_to_object_surface_cm": _agg("GT_hand_to_object_surface_cm"),
        "smoothed_vs_raw_target_local_gap_cm": _agg("smoothed_vs_raw_target_local_gap_cm"),
        "max_local_target_jump_2f_cm": _agg("max_local_target_jump_2f_cm"),
        "max_world_target_jump_2f_cm": _agg("max_world_target_jump_2f_cm"),
        "pct_events_target_to_hand_over_10cm": (
            100.0 * sum(1 for e in all_events if e["target_world_to_GT_hand_cm"] > 10.0)
            / max(1, len(all_events))
        ),
        "pct_events_target_to_hand_over_15cm": (
            100.0 * sum(1 for e in all_events if e["target_world_to_GT_hand_cm"] > 15.0)
            / max(1, len(all_events))
        ),
        "pct_events_target_to_hand_over_25cm": (
            100.0 * sum(1 for e in all_events if e["target_world_to_GT_hand_cm"] > 25.0)
            / max(1, len(all_events))
        ),
        "mean_pct_target_populated_during_non_contact": float(np.mean([
            a["summary"]["pct_target_populated_during_non_contact"] for a in audits
        ])),
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
    lines: list[str] = [
        "# Pseudo-target Provenance Trace (Round 6, Diag A)",
        "",
        f"- Config: `{args.config}`",
        f"- Selection: `{args.selection_json}`",
        f"- Clips: {B}",
        f"- Subset composition: {subset_counts}",
        f"- Raw target source counts across clips: {sources}",
        "",
        "## Aggregate findings",
        "",
        f"- Total events across all clips: {aggregate['n_total_events']}",
        f"- Boundary events (within {args.edge_margin} frames of edge): "
        f"{aggregate['n_boundary_events']}",
        f"- Flicker events (segment duration <= 2): {aggregate['n_flicker_events']}",
        "",
        "### Target_world to GT hand (cm, event-only)",
        "",
        f"- mean: **{aggregate['target_world_to_GT_hand_cm']['mean']:.2f}** "
        f"median: {aggregate['target_world_to_GT_hand_cm']['median']:.2f} "
        f"p95: {aggregate['target_world_to_GT_hand_cm']['p95']:.2f}",
        f"- % events > 10 cm: **{aggregate['pct_events_target_to_hand_over_10cm']:.1f}%**",
        f"- % events > 15 cm: **{aggregate['pct_events_target_to_hand_over_15cm']:.1f}%**",
        f"- % events > 25 cm: **{aggregate['pct_events_target_to_hand_over_25cm']:.1f}%**",
        "",
        "### Target_world to object surface (cm, event-only)",
        "",
        f"- mean: {aggregate['target_world_to_object_surface_cm']['mean']:.2f} "
        f"median: {aggregate['target_world_to_object_surface_cm']['median']:.2f} "
        f"p95: {aggregate['target_world_to_object_surface_cm']['p95']:.2f}",
        "",
        "### GT hand to object surface (cm, event-only)",
        "",
        f"- mean: {aggregate['GT_hand_to_object_surface_cm']['mean']:.2f} "
        f"median: {aggregate['GT_hand_to_object_surface_cm']['median']:.2f} "
        f"p95: {aggregate['GT_hand_to_object_surface_cm']['p95']:.2f}",
        "",
        "### Target jumps near events",
        "",
        f"- max local target jump (cm, 2-frame window around event) "
        f"mean: {aggregate['max_local_target_jump_2f_cm']['mean']:.2f} "
        f"p95: {aggregate['max_local_target_jump_2f_cm']['p95']:.2f}",
        f"- max world target jump (cm, 2-frame window around event) "
        f"mean: {aggregate['max_world_target_jump_2f_cm']['mean']:.2f} "
        f"p95: {aggregate['max_world_target_jump_2f_cm']['p95']:.2f}",
        "",
        "### Smoothed-vs-raw target local gap",
        "",
        f"- mean cm: {aggregate['smoothed_vs_raw_target_local_gap_cm']['mean']:.3f} "
        f"p95: {aggregate['smoothed_vs_raw_target_local_gap_cm']['p95']:.3f}",
        "",
        "### Target presence during non-contact",
        "",
        f"- Mean % of non-contact frames with populated `contact_target_xyz`: "
        f"**{aggregate['mean_pct_target_populated_during_non_contact']:.1f}%** "
        "(should be 0% if `target_query_contact_only=true`).",
        "",
        "## Per-clip summary",
        "",
    ]
    rows = [[
        "subset", "seq_id", "T", "src", "n_ev", "n_boundary", "n_flicker",
        "warn>15cm", "severe>25cm", "non-contact %", "world-gap m",
    ]]
    for a in audits:
        rows.append([
            a["subset"], a["seq_id"], a["seq_len"],
            a["raw_npz"].get("source", "?"),
            a["summary"]["n_events"],
            a["summary"]["n_boundary_events"],
            a["summary"]["n_flicker_events"],
            a["summary"]["n_warn_target_to_hand_over_15cm"],
            a["summary"]["n_severe_target_to_hand_over_25cm"],
            f"{a['summary']['pct_target_populated_during_non_contact']:.1f}",
            f"{a['summary']['target_world_vs_lift_consistency_max_meters']:.4f}",
        ])
    lines.append(format_md_table(rows))
    lines.append("")
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.md}")


if __name__ == "__main__":
    main()
