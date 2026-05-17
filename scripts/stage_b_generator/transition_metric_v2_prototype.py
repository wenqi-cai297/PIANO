"""Transition Metric V2 Prototype (round 6, Diag C).

Designs a candidate replacement metric pack for the existing
``diagnostic_common.transition_metrics`` formulation, which Round 5 Diag 3
showed to be denominator-unstable on 63.8% of events.

This prototype is ANALYSIS-ONLY. It does not patch any other script. Once
validated, the user can promote a chosen variant into ``diagnostic_common``
as ``transition_metrics_v2`` (behind a metric_version flag).

Metrics implemented:
  - M1 (current): max(0, dist[before] - dist[event]) cm — onset closing /
    release opening positive change vs object COM. Reported as raw cm too.
  - M2 (slope): linear regression slope of dist curve over event window
    in cm/frame; onset wants slope < 0 (closing) over [t-k..t], release
    wants slope > 0 over [t..t+k].
  - M3 (signed pre/post): onset = dist_before - dist_event, release =
    dist_after - dist_event, in cm. No ratio.
  - M5 (clipped ratio): gen_change / max(gt_change, clip_cm), with
    clip_cm tested at {2, 5}.

All metrics computed on TWO distance variants:
  - COM-distance: hand to obj_positions[t] (current default)
  - Surface-distance: hand to nearest transformed object_pc point

Event validity filter:
  - in_pre_range:  for onset, t >= window_k; for release, always True
  - in_post_range: for release, t + window_k < seq_len; for onset True
  - away_from_edge: t in [edge_margin, seq_len-1-edge_margin]
  - gt_change_above_threshold: GT |signed change| >= min_gt_change_cm
  - not_flicker: segment duration > flicker_max_frames

Outputs:
  analyses/2026-05-16_transition_metric_v2_prototype.json
  analyses/2026-05-16_transition_metric_v2_prototype.md
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


HAND_SPECS = (("L_hand", 20, 0), ("R_hand", 21, 1))


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


def _dist_com(joints: np.ndarray, obj_pos: np.ndarray, joint: int, seq_len: int) -> np.ndarray:
    h = joints[:seq_len, joint]
    o = obj_pos[:seq_len]
    return np.linalg.norm(h - o, axis=-1) * 100.0


def _dist_surface(
    joints: np.ndarray, obj_pc_world: np.ndarray, joint: int, seq_len: int,
) -> np.ndarray:
    h = joints[:seq_len, joint][:, None, :]
    d = np.linalg.norm(h - obj_pc_world[:seq_len], axis=-1)
    return d.min(axis=-1) * 100.0


def _segment_pairs(c_bool: np.ndarray, seq_len: int) -> list[tuple[int, int]]:
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
            segments.append((o, seq_len))
    return segments


def _metrics_for_event(
    dist_curve: np.ndarray, kind: str, t: int, seq_len: int, window_k: int,
) -> dict[str, float]:
    if kind == "onset":
        lo, hi = max(0, t - window_k), int(t)
    else:
        lo, hi = int(t), min(seq_len - 1, t + window_k)
    # M1
    if kind == "onset":
        raw = float(dist_curve[lo] - dist_curve[int(t)])
    else:
        raw = float(dist_curve[hi] - dist_curve[int(t)])
    m1_pos = max(0.0, raw)
    # M2 slope over window
    n = hi - lo + 1
    if n >= 3:
        xs = np.arange(n)
        ys = dist_curve[lo:hi + 1]
        m2 = float(np.polyfit(xs, ys, 1)[0])
    else:
        m2 = 0.0
    # M3 signed
    pre_lo = max(0, t - window_k)
    post_hi = min(seq_len - 1, t + window_k)
    pre_mean = (
        float(dist_curve[pre_lo:t + 1].mean()) if t > pre_lo
        else float(dist_curve[int(t)])
    )
    post_mean = (
        float(dist_curve[t:post_hi + 1].mean()) if post_hi > t
        else float(dist_curve[int(t)])
    )
    m3 = pre_mean - post_mean if kind == "onset" else post_mean - pre_mean
    return {
        "m1_raw_change_cm": raw,
        "m1_positive_change_cm": m1_pos,
        "m2_slope_cm_per_frame": m2,
        "m3_signed_diff_cm": float(m3),
    }


def _audit_clip(
    batch: dict[str, Any], b: int,
    *, window_k: int, edge_margin: int, threshold: float,
    min_gt_change_cm: float, flicker_max_frames: int, surface_samples: int,
) -> dict[str, Any]:
    seq_len = int(batch["seq_len"][b].item())
    subset = str(batch["subset"][b])
    seq_id = str(batch["seq_id"][b])
    gt_joints = batch["joints"][b].detach().cpu().numpy().astype(np.float32)
    contact_state = batch["contact_state"][b].detach().cpu().numpy().astype(np.float32)
    obj_positions = batch["object_positions"][b].detach().cpu().numpy().astype(np.float32)
    obj_rotations = batch["object_rotations"][b].detach().cpu().numpy().astype(np.float32)
    obj_pc = batch["object_pc"][b].detach().cpu().numpy().astype(np.float32)

    R = _axis_angle_to_rot(obj_rotations[:seq_len])
    if obj_pc.shape[0] > surface_samples:
        idx = np.random.RandomState(0).choice(obj_pc.shape[0], surface_samples, replace=False)
        pc_sub = obj_pc[idx]
    else:
        pc_sub = obj_pc
    obj_pc_world = np.einsum("tij,nj->tni", R, pc_sub) + obj_positions[:seq_len, None, :]

    events_out: list[dict[str, Any]] = []
    for part, joint, p_idx in HAND_SPECS:
        d_com = _dist_com(gt_joints, obj_positions, joint, seq_len)
        d_surf = _dist_surface(gt_joints, obj_pc_world, joint, seq_len)
        c_bool = contact_state[:seq_len, p_idx] > float(threshold)
        segments = _segment_pairs(c_bool, seq_len)
        # iterate onsets / releases
        for s, e in segments:
            duration = max(1, e - s)
            for kind, t_ev in (("onset", s), ("release", e)):
                t_ev = int(min(max(0, t_ev), seq_len - 1))
                m_com = _metrics_for_event(d_com, kind, t_ev, seq_len, window_k)
                m_surf = _metrics_for_event(d_surf, kind, t_ev, seq_len, window_k)
                in_pre_range = (
                    (kind == "onset" and t_ev - window_k >= 0) or kind == "release"
                )
                in_post_range = (
                    (kind == "release" and t_ev + window_k <= seq_len - 1) or kind == "onset"
                )
                away_from_edge = (
                    t_ev >= int(edge_margin) and t_ev <= seq_len - 1 - int(edge_margin)
                )
                # GT denominator stability uses COM positive change (legacy)
                gt_denom_com = float(m_com["m1_positive_change_cm"])
                gt_denom_surf = float(m_surf["m1_positive_change_cm"])
                denom_stable_com = gt_denom_com >= float(min_gt_change_cm)
                denom_stable_surf = gt_denom_surf >= float(min_gt_change_cm)
                is_flicker = duration <= int(flicker_max_frames)
                fully_valid = (
                    in_pre_range and in_post_range and away_from_edge
                    and denom_stable_com and not is_flicker
                )
                events_out.append({
                    "part": part, "joint": joint, "kind": kind, "frame": int(t_ev),
                    "segment_start": int(s), "segment_end": int(e),
                    "duration": int(duration),
                    "is_flicker": bool(is_flicker),
                    "in_pre_range": bool(in_pre_range),
                    "in_post_range": bool(in_post_range),
                    "away_from_edge": bool(away_from_edge),
                    "denom_stable_com": bool(denom_stable_com),
                    "denom_stable_surf": bool(denom_stable_surf),
                    "valid_for_ratio_com": bool(fully_valid),
                    "com": m_com,
                    "surf": m_surf,
                })

    return {
        "subset": subset, "seq_id": seq_id, "seq_len": seq_len,
        "events": events_out,
    }


def _ratio_under_clip(
    gen_change: float, gt_change: float, clip_cm: float,
) -> float:
    """M5 robust clipped ratio."""
    return float(gen_change / max(gt_change, float(clip_cm)))


def _agg(all_events: list[dict[str, Any]]) -> dict[str, Any]:
    n_total = len(all_events)
    n_flicker = sum(1 for e in all_events if e["is_flicker"])
    n_boundary = sum(1 for e in all_events if not e["away_from_edge"])
    n_unstable_com = sum(1 for e in all_events if not e["denom_stable_com"])
    n_unstable_surf = sum(1 for e in all_events if not e["denom_stable_surf"])
    n_valid_com = sum(1 for e in all_events if e["valid_for_ratio_com"])
    pct_unstable_com = 100.0 * n_unstable_com / max(1, n_total)
    pct_unstable_surf = 100.0 * n_unstable_surf / max(1, n_total)
    pct_flicker = 100.0 * n_flicker / max(1, n_total)
    pct_boundary = 100.0 * n_boundary / max(1, n_total)

    valid_events = [e for e in all_events if e["valid_for_ratio_com"]]

    def _arr(events: list[dict[str, Any]], path: str) -> np.ndarray:
        parts = path.split(".")
        out = []
        for e in events:
            d = e
            for p in parts:
                d = d[p]
            out.append(float(d))
        return np.asarray(out, dtype=np.float64)

    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        if a.size < 3 or a.std() < 1e-8 or b.std() < 1e-8:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    correlations = {}
    if valid_events:
        m1_com = _arr(valid_events, "com.m1_positive_change_cm")
        m1_surf = _arr(valid_events, "surf.m1_positive_change_cm")
        m2_com = _arr(valid_events, "com.m2_slope_cm_per_frame")
        m2_surf = _arr(valid_events, "surf.m2_slope_cm_per_frame")
        m3_com = _arr(valid_events, "com.m3_signed_diff_cm")
        m3_surf = _arr(valid_events, "surf.m3_signed_diff_cm")
        correlations = {
            "corr_M1_com_M2_com": _corr(m1_com, m2_com),
            "corr_M1_com_M3_com": _corr(m1_com, m3_com),
            "corr_M1_com_M1_surf": _corr(m1_com, m1_surf),
            "corr_M2_com_M2_surf": _corr(m2_com, m2_surf),
            "corr_M3_com_M3_surf": _corr(m3_com, m3_surf),
            "mean_M1_com_minus_surf_cm": float((m1_com - m1_surf).mean()),
            "mean_M2_com_minus_surf_cm_per_frame": float((m2_com - m2_surf).mean()),
        }

    # M5 stability scan: how many events have ratio explosion under each clip
    m5_stats = {}
    for clip_cm in (2.0, 5.0):
        ratios = []
        unstable = 0
        for e in all_events:
            gen = float(e["com"]["m1_positive_change_cm"])
            gt = float(e["com"]["m1_positive_change_cm"])  # placeholder GT
            r = _ratio_under_clip(gen, gt, clip_cm)
            ratios.append(r)
            if abs(r) > 5.0:
                unstable += 1
        m5_stats[f"clip_{int(clip_cm)}cm"] = {
            "median_ratio": float(np.median(ratios)) if ratios else 0.0,
            "n_unstable_abs_gt_5": int(unstable),
        }

    return {
        "n_total_events": int(n_total),
        "n_flicker": int(n_flicker),
        "n_boundary": int(n_boundary),
        "n_unstable_com": int(n_unstable_com),
        "n_unstable_surf": int(n_unstable_surf),
        "n_valid_com": int(n_valid_com),
        "pct_unstable_com": pct_unstable_com,
        "pct_unstable_surf": pct_unstable_surf,
        "pct_flicker": pct_flicker,
        "pct_boundary": pct_boundary,
        "correlations": correlations,
        "m5_clip_scan": m5_stats,
        "M1_positive_change_cm_com": stats_list([
            e["com"]["m1_positive_change_cm"] for e in valid_events
        ]) if valid_events else stats_list([]),
        "M1_positive_change_cm_surf": stats_list([
            e["surf"]["m1_positive_change_cm"] for e in valid_events
        ]) if valid_events else stats_list([]),
        "M2_slope_com": stats_list([
            e["com"]["m2_slope_cm_per_frame"] for e in valid_events
        ]) if valid_events else stats_list([]),
        "M2_slope_surf": stats_list([
            e["surf"]["m2_slope_cm_per_frame"] for e in valid_events
        ]) if valid_events else stats_list([]),
        "M3_signed_com": stats_list([
            e["com"]["m3_signed_diff_cm"] for e in valid_events
        ]) if valid_events else stats_list([]),
        "M3_signed_surf": stats_list([
            e["surf"]["m3_signed_diff_cm"] for e in valid_events
        ]) if valid_events else stats_list([]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=Path("configs/training/anchordiff_v18_a1_FULL_DATA.yaml"))
    parser.add_argument("--selection-json", type=Path,
                        default=Path("analyses/2026-05-13_v18_recon_ladder_truncated_rollout.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("analyses/2026-05-16_transition_metric_v2_prototype.json"))
    parser.add_argument("--md", type=Path,
                        default=Path("analyses/2026-05-16_transition_metric_v2_prototype.md"))
    parser.add_argument("--max-clips", type=int, default=16)
    parser.add_argument("--num-candidates", type=int, default=256)
    parser.add_argument("--bucket", choices=["train", "val"], default="train")
    parser.add_argument("--balanced-subsets", action="store_true", default=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--window-k", type=int, default=10)
    parser.add_argument("--edge-margin", type=int, default=5)
    parser.add_argument("--min-gt-change-cm", type=float, default=2.0)
    parser.add_argument("--flicker-max-frames", type=int, default=2)
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

    audits = [
        _audit_clip(
            batch, b,
            window_k=int(args.window_k), edge_margin=int(args.edge_margin),
            threshold=float(args.threshold),
            min_gt_change_cm=float(args.min_gt_change_cm),
            flicker_max_frames=int(args.flicker_max_frames),
            surface_samples=int(args.surface_samples),
        )
        for b in range(B)
    ]
    all_events = [e for a in audits for e in a["events"]]
    aggregate = _agg(all_events)

    subset_counts: dict[str, int] = {}
    for a in audits:
        subset_counts[a["subset"]] = subset_counts.get(a["subset"], 0) + 1
    aggregate["subset_counts"] = subset_counts
    aggregate["n_clips"] = B

    # Verdict
    verdict = []
    if aggregate["pct_unstable_com"] > 25.0:
        verdict.append(
            f"M1 (current) on COM-distance is denominator-unstable for "
            f"{aggregate['pct_unstable_com']:.1f}% of events (GT positive change < "
            f"{args.min_gt_change_cm} cm). Recommend M2 (slope) or M5 (clipped ratio) as load-bearing."
        )
    if aggregate.get("correlations", {}).get("corr_M1_com_M2_com", 0.0) >= 0.5 \
       and aggregate.get("correlations", {}).get("corr_M1_com_M3_com", 0.0) >= 0.5:
        verdict.append(
            "Among valid events, M2 (slope) and M3 (signed) correlate >=0.5 with M1. "
            "Either can serve as a robust replacement; M2 is the most denominator-free."
        )
    if aggregate.get("correlations", {}).get("corr_M1_com_M1_surf", 1.0) < 0.7:
        verdict.append(
            "COM-distance and surface-distance disagree (r < 0.7); contact metrics "
            "should use surface-distance for chair-shaped objects."
        )

    payload = {
        "config": str(args.config),
        "selection_json": str(args.selection_json),
        "window_k": int(args.window_k),
        "edge_margin": int(args.edge_margin),
        "min_gt_change_cm": float(args.min_gt_change_cm),
        "flicker_max_frames": int(args.flicker_max_frames),
        "aggregate": aggregate,
        "clips": audits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")

    lines = [
        "# Transition Metric V2 Prototype (Round 6, Diag C)",
        "",
        f"- Config: `{args.config}`",
        f"- Window k: {args.window_k} frames; edge margin: {args.edge_margin} frames",
        f"- Denominator stability threshold: GT positive change >= {args.min_gt_change_cm} cm",
        f"- Flicker filter: segment duration <= {args.flicker_max_frames}",
        f"- Clips: {B}",
        f"- Subset composition: {subset_counts}",
        "",
        "## Event-validity breakdown",
        "",
        f"- Total events: {aggregate['n_total_events']}",
        f"- Flicker events: {aggregate['n_flicker']} ({aggregate['pct_flicker']:.1f}%)",
        f"- Boundary events (within {args.edge_margin} of edge): "
        f"{aggregate['n_boundary']} ({aggregate['pct_boundary']:.1f}%)",
        f"- Denominator-unstable events (COM): {aggregate['n_unstable_com']} "
        f"(**{aggregate['pct_unstable_com']:.1f}%**)",
        f"- Denominator-unstable events (surface): {aggregate['n_unstable_surf']} "
        f"(**{aggregate['pct_unstable_surf']:.1f}%**)",
        f"- Valid events (passing all filters on COM): {aggregate['n_valid_com']}",
        "",
        "## Metric stats on valid events",
        "",
        "| metric | mean | median | p25 | p75 | p95 |",
        "|--------|------|--------|-----|-----|-----|",
    ]
    for key, label in (
        ("M1_positive_change_cm_com",  "M1 (cm, COM)"),
        ("M1_positive_change_cm_surf", "M1 (cm, surface)"),
        ("M2_slope_com",               "M2 (cm/frame, COM)"),
        ("M2_slope_surf",              "M2 (cm/frame, surf)"),
        ("M3_signed_com",              "M3 (cm, COM)"),
        ("M3_signed_surf",             "M3 (cm, surf)"),
    ):
        m = aggregate.get(key, {})
        if not m or m.get("n", 0) == 0:
            continue
        lines.append(
            f"| {label} | {m['mean']:.2f} | {m['median']:.2f} | "
            f"{m['p25']:.2f} | {m['p75']:.2f} | {m['p95']:.2f} |"
        )
    lines += [
        "",
        "## Cross-metric correlations (valid events)",
        "",
    ]
    for k, v in aggregate.get("correlations", {}).items():
        if "corr_" in k:
            lines.append(f"- `{k}` = {v:.3f}")
        else:
            lines.append(f"- `{k}` = {v:.3f}")
    lines += [
        "",
        "## Verdict",
        "",
    ]
    if not verdict:
        lines.append("(No critical findings — current metric appears stable on this clip set.)")
    else:
        for v in verdict:
            lines.append(f"- {v}")
    lines += [
        "",
        "## Per-clip summary",
        "",
    ]
    rows = [[
        "subset", "seq_id", "T", "n_events", "n_unstable_com", "n_unstable_surf",
        "n_flicker", "n_boundary",
    ]]
    for a in audits:
        ne = len(a["events"])
        nu = sum(1 for e in a["events"] if not e["denom_stable_com"])
        nu_s = sum(1 for e in a["events"] if not e["denom_stable_surf"])
        nf = sum(1 for e in a["events"] if e["is_flicker"])
        nb = sum(1 for e in a["events"] if not e["away_from_edge"])
        rows.append([a["subset"], a["seq_id"], a["seq_len"], ne, nu, nu_s, nf, nb])
    lines.append(format_md_table(rows))
    lines.append("")
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.md}")


if __name__ == "__main__":
    main()
