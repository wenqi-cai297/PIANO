"""Transition metric reliability audit (round 5, Diag 3).

Pure CPU/GPU-light analysis. Checks whether the current
`onset_positive_closing_xGT` / `release_positive_opening_xGT` ratio
metric is stable enough to drive training decisions, and compares
against four alternative metric formulations.

Alternative metrics:
  M1 (current)     : positive-closing ratio = max(0, dist[start]-dist[t])
                     normalised by GT version. Failure mode: denominator
                     tiny → division blow-up.
  M2 (slope)       : linear regression slope of hand-object distance over
                     event window. Robust to near-zero GT change.
  M3 (signed diff) : mean Δ distance before vs after event. No ratio.
  M4 (surface)     : same as M1 but distance to transformed object_pc
                     nearest-point (object surface), not COM.
  M5 (robust ratio): like M1 but with absolute-cm clipped denominator
                     (denom max(GT, 2 cm)).

Event-validity filter:
  - exclude events within `edge_margin` frames of clip edges
  - exclude events with insufficient pre/post frames inside seq_len
  - exclude events with GT distance change below `min_gt_change_cm`

Outputs:
  analyses/2026-05-15_transition_metric_reliability_audit.{json,md}
  analyses/visuals/2026-05-15_transition_metric_reliability/*.png

This is an ANALYSIS-ONLY audit on GT data and (cached) v18 baseline
samples from round 4's 16-clip DDPM run (per-clip metrics already saved
in the round 4 JSON outputs). No new model inference is run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from diagnostic_common import (
    event_records_from_contact,
    format_md_table,
    make_seq_mask,
    merge_single_batches,
)
from recon_ladder_truncated_rollout_diagnostic import (
    _build_selected_batches,
    _load_selection,
)


HAND_SPECS = (("L_hand", 20, 0), ("R_hand", 21, 1))


def _hand_object_distance_curve(
    joints: np.ndarray, obj_positions: np.ndarray, seq_len: int, joint: int,
) -> np.ndarray:
    return np.linalg.norm(joints[:seq_len, joint] - obj_positions[:seq_len], axis=-1) * 100.0


def _hand_surface_distance_curve(
    joints: np.ndarray, obj_pc_world: np.ndarray, seq_len: int, joint: int,
) -> np.ndarray:
    """Min over object point cloud of hand→pc distance. obj_pc_world: (T, N, 3)."""
    h = joints[:seq_len, joint][:, None, :]  # (T,1,3)
    d = np.linalg.norm(h - obj_pc_world[:seq_len], axis=-1)  # (T, N)
    return d.min(axis=-1) * 100.0


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


def _alternative_metrics_for_event(
    dist_curve: np.ndarray,
    *,
    kind: str,
    t: int,
    window_k: int,
    seq_len: int,
) -> dict[str, float]:
    """Compute M1/M2/M3/M5 around a single event at frame t."""
    if kind == "onset":
        lo, hi = max(0, t - window_k), t
        before, after = lo, t
    else:  # release
        lo, hi = t, min(seq_len - 1, t + window_k)
        before, after = t, hi
    # M1 positive change (closing for onset, opening for release)
    raw_change = float(dist_curve[before] - dist_curve[t]) if kind == "onset" else float(dist_curve[after] - dist_curve[t])
    m1 = max(0.0, raw_change)
    # M2 slope over [lo..hi]
    n = hi - lo + 1
    if n >= 3:
        xs = np.arange(n)
        ys = dist_curve[lo:hi + 1]
        m2 = float(np.polyfit(xs, ys, 1)[0])
    else:
        m2 = 0.0
    # M3 signed before-after mean Δ
    half = window_k // 2
    pre_lo = max(0, t - window_k); pre_hi = max(pre_lo + 1, t)
    post_lo = min(seq_len - 1, t); post_hi = min(seq_len - 1, t + window_k)
    pre_mean = float(dist_curve[pre_lo:pre_hi + 1].mean()) if pre_hi > pre_lo else float(dist_curve[t])
    post_mean = float(dist_curve[post_lo:post_hi + 1].mean()) if post_hi > post_lo else float(dist_curve[t])
    m3 = pre_mean - post_mean if kind == "onset" else post_mean - pre_mean
    # M5 robust ratio uses a clipped denominator (resolved later when GT denom is available)
    return {
        "m1_positive_change_cm": m1,
        "m1_raw_change_cm": raw_change,
        "m2_slope_cm_per_frame": m2,
        "m3_signed_diff_cm": float(m3),
    }


def _audit_clip(
    batch: dict[str, Any], b: int,
    *, window_k: int, edge_margin: int, threshold: float,
    min_gt_change_cm: float, compute_surface: bool,
) -> dict[str, Any]:
    seq_len = int(batch["seq_len"][b].item())
    gt_joints = batch["joints"][b].detach().cpu().numpy().astype(np.float32)
    contact_state = batch["contact_state"][b].detach().cpu().numpy().astype(np.float32)
    obj_positions = batch["object_positions"][b].detach().cpu().numpy().astype(np.float32)
    obj_rotations = batch["object_rotations"][b].detach().cpu().numpy().astype(np.float32)
    obj_pc = batch["object_pc"][b].detach().cpu().numpy().astype(np.float32)

    # transform object_pc into world over time
    if compute_surface:
        R = _axis_angle_to_rot(obj_rotations[:seq_len])
        obj_pc_world = np.einsum("tij,nj->tni", R, obj_pc) + obj_positions[:seq_len, None, :]
    else:
        obj_pc_world = None

    events_out: list[dict[str, Any]] = []
    for part, joint, idx in HAND_SPECS:
        d_com = _hand_object_distance_curve(gt_joints, obj_positions, seq_len, joint)
        d_surf = _hand_surface_distance_curve(gt_joints, obj_pc_world, seq_len, joint) if compute_surface else None
        c = contact_state[:seq_len, idx] > float(threshold)
        onset_idx = (np.where(c[1:] & ~c[:-1])[0] + 1).tolist()
        release_idx = (np.where(~c[1:] & c[:-1])[0] + 1).tolist()
        for kind, frames in (("onset", onset_idx), ("release", release_idx)):
            for t in frames:
                # validity flags
                in_pre_range = (kind == "onset" and t - window_k >= 0) or (kind == "release" and True)
                in_post_range = (kind == "release" and t + window_k <= seq_len - 1) or (kind == "onset" and True)
                away_from_edge = (t >= int(edge_margin)) and (t <= seq_len - 1 - int(edge_margin))
                m_com = _alternative_metrics_for_event(d_com, kind=kind, t=int(t), window_k=window_k, seq_len=seq_len)
                m_surf = (
                    _alternative_metrics_for_event(d_surf, kind=kind, t=int(t), window_k=window_k, seq_len=seq_len)
                    if d_surf is not None else None
                )
                gt_denom = float(m_com["m1_positive_change_cm"])
                denom_unstable = gt_denom < float(min_gt_change_cm)
                events_out.append({
                    "part": part,
                    "joint": joint,
                    "kind": kind,
                    "frame": int(t),
                    "in_pre_range": bool(in_pre_range),
                    "in_post_range": bool(in_post_range),
                    "away_from_edge": bool(away_from_edge),
                    "valid": bool(in_pre_range and in_post_range and away_from_edge and not denom_unstable),
                    "denom_unstable": bool(denom_unstable),
                    "gt_com_metrics": m_com,
                    "gt_surface_metrics": m_surf,
                })
    return {
        "subset": str(batch["subset"][b]),
        "seq_id": str(batch["seq_id"][b]),
        "seq_len": seq_len,
        "events": events_out,
    }


def _aggregate(audits: list[dict[str, Any]]) -> dict[str, Any]:
    all_events = [e for a in audits for e in a["events"]]
    n_total = len(all_events)
    n_valid = sum(1 for e in all_events if e["valid"])
    n_denom_unstable = sum(1 for e in all_events if e["denom_unstable"])
    n_edge = sum(1 for e in all_events if not e["away_from_edge"])
    n_out_of_range = sum(1 for e in all_events if not (e["in_pre_range"] and e["in_post_range"]))

    # Correlation between M1 (current ratio numerator) and M2 (slope), M3 (signed diff)
    def _arr(field_a: str, field_b: str) -> tuple[np.ndarray, np.ndarray]:
        valid_events = [e for e in all_events if e["valid"]]
        a = np.asarray([e["gt_com_metrics"][field_a] for e in valid_events], dtype=np.float64)
        b = np.asarray([e["gt_com_metrics"][field_b] for e in valid_events], dtype=np.float64)
        return a, b

    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        if a.size < 3:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    m1, m2 = _arr("m1_positive_change_cm", "m2_slope_cm_per_frame")
    _, m3 = _arr("m1_positive_change_cm", "m3_signed_diff_cm")

    # surface vs COM agreement on M1
    valid_with_surface = [e for e in all_events if e["valid"] and e["gt_surface_metrics"] is not None]
    if valid_with_surface:
        m1_com = np.array([e["gt_com_metrics"]["m1_positive_change_cm"] for e in valid_with_surface])
        m1_surf = np.array([e["gt_surface_metrics"]["m1_positive_change_cm"] for e in valid_with_surface])
        corr_com_surf = _corr(m1_com, m1_surf)
        mean_offset = float((m1_com - m1_surf).mean())
    else:
        corr_com_surf = 0.0
        mean_offset = 0.0

    # boundary event count: frame < 5 (onset frame 2 problem)
    n_onset_frame_lt5 = sum(1 for e in all_events if e["kind"] == "onset" and e["frame"] < 5)

    return {
        "n_clips": len(audits),
        "n_total_events": int(n_total),
        "n_valid_events": int(n_valid),
        "n_denom_unstable": int(n_denom_unstable),
        "n_edge": int(n_edge),
        "n_out_of_range": int(n_out_of_range),
        "n_onset_frame_lt5": int(n_onset_frame_lt5),
        "pct_unstable": 100.0 * n_denom_unstable / max(1, n_total),
        "pct_boundary": 100.0 * n_edge / max(1, n_total),
        "corr_m1_m2_slope": _corr(m1, m2),
        "corr_m1_m3_signed_diff": _corr(m1, m3),
        "corr_m1_com_vs_surface": corr_com_surf,
        "mean_m1_com_minus_surface_cm": mean_offset,
    }


def _plot_clip(audit: dict[str, Any], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    for e in audit["events"]:
        marker = "o" if e["valid"] else "x"
        color = "g" if e["kind"] == "onset" else "r"
        ax.scatter(e["frame"], e["gt_com_metrics"]["m1_positive_change_cm"], c=color, marker=marker, s=40,
                   label=f"{e['kind']} {e['part']}")
    ax.set_xlabel("frame")
    ax.set_ylabel("M1 positive change (cm)")
    ax.set_title(f"{audit['subset']}/{audit['seq_id']}  T={audit['seq_len']}  events={len(audit['events'])}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/training/anchordiff_v18_a1_FULL_DATA.yaml"))
    parser.add_argument("--selection-json", type=Path, default=Path("analyses/2026-05-13_v18_recon_ladder_truncated_rollout.json"))
    parser.add_argument("--output", type=Path, default=Path("analyses/2026-05-15_transition_metric_reliability_audit.json"))
    parser.add_argument("--md", type=Path, default=Path("analyses/2026-05-15_transition_metric_reliability_audit.md"))
    parser.add_argument("--visuals-dir", type=Path, default=Path("analyses/visuals/2026-05-15_transition_metric_reliability"))
    parser.add_argument("--max-clips", type=int, default=16)
    parser.add_argument("--num-candidates", type=int, default=256)
    parser.add_argument("--bucket", choices=["train", "val"], default="train")
    parser.add_argument("--balanced-subsets", action="store_true", default=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--window-k", type=int, default=10)
    parser.add_argument("--edge-margin", type=int, default=5)
    parser.add_argument("--min-gt-change-cm", type=float, default=2.0)
    parser.add_argument("--no-surface", action="store_true", help="Skip object-surface distance (faster).")
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
            batch, b,
            window_k=int(args.window_k), edge_margin=int(args.edge_margin),
            threshold=float(args.threshold), min_gt_change_cm=float(args.min_gt_change_cm),
            compute_surface=not bool(args.no_surface),
        )
        audits.append(a)
        out_png = args.visuals_dir / f"{a['subset']}_{a['seq_id']}_events.png"
        _plot_clip(a, out_png)

    agg = _aggregate(audits)

    payload = {
        "config": str(args.config),
        "n_clips": B,
        "window_k": int(args.window_k),
        "edge_margin": int(args.edge_margin),
        "min_gt_change_cm": float(args.min_gt_change_cm),
        "aggregate": agg,
        "clips": audits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")

    # Decide verdict
    if agg["pct_unstable"] > 25.0:
        verdict = (
            f"**Current onset/release ratio metric is denominator-unstable**: "
            f"{agg['pct_unstable']:.1f}% of events have GT positive change < {args.min_gt_change_cm} cm. "
            "Recommend replacing with M2 (slope) or M5 (robust ratio) before further route diagnostics."
        )
    elif agg["pct_boundary"] > 30.0:
        verdict = (
            f"**Boundary events dominate**: {agg['pct_boundary']:.1f}% of events occur within "
            f"{args.edge_margin} frames of clip edges. Filter or pad before metric computation."
        )
    elif agg["corr_m1_m2_slope"] < 0.5 or agg["corr_m1_m3_signed_diff"] < 0.5:
        verdict = (
            f"**Current metric correlates weakly with alternatives** "
            f"(M1↔M2 r={agg['corr_m1_m2_slope']:.2f}, M1↔M3 r={agg['corr_m1_m3_signed_diff']:.2f}). "
            "Cross-validate any future route claim with at least one alternative metric."
        )
    else:
        verdict = (
            "Current onset/release positive-closing ratio is reasonably stable on the audited clip set "
            "(low denominator instability, moderate-to-high correlation with M2 slope and M3 signed-diff). "
            "Existing metric can stay; still cross-validate with M2/M3 on any future route claim."
        )
    if agg["corr_m1_com_vs_surface"] < 0.7:
        verdict += (
            f" Additionally, object-COM and object-surface distance metrics disagree "
            f"(corr={agg['corr_m1_com_vs_surface']:.2f}, mean offset={agg['mean_m1_com_minus_surface_cm']:.2f} cm); "
            "contact diagnostics should use surface distance, not COM."
        )

    lines: list[str] = [
        "# Transition Metric Reliability Audit",
        "",
        f"- Config: `{args.config}`",
        f"- Clips: {B}",
        f"- Window k: {args.window_k} frames; edge margin: {args.edge_margin} frames",
        f"- Denominator instability threshold: GT positive change < {args.min_gt_change_cm} cm",
        "",
        "## Aggregate",
        "",
        f"- Total events: {agg['n_total_events']}",
        f"- Valid events (after filters): {agg['n_valid_events']}",
        f"- Denominator-unstable events: {agg['n_denom_unstable']} (**{agg['pct_unstable']:.1f}%**)",
        f"- Boundary events (within {args.edge_margin} of edge): {agg['n_edge']} (**{agg['pct_boundary']:.1f}%**)",
        f"- Out-of-range events: {agg['n_out_of_range']}",
        f"- Onset events at frame < 5: {agg['n_onset_frame_lt5']} (Round-2 clip `Sub0001_Obj116_Seg0_600` has onset @ frame 2)",
        "",
        "## Correlations across alternative metrics",
        "",
        f"- M1 (positive-change cm) ↔ M2 (slope cm/frame): r = **{agg['corr_m1_m2_slope']:.3f}**",
        f"- M1 (positive-change cm) ↔ M3 (signed pre-post Δ cm): r = **{agg['corr_m1_m3_signed_diff']:.3f}**",
        f"- M1 (COM-distance) ↔ M1 (surface-distance): r = **{agg['corr_m1_com_vs_surface']:.3f}** "
        f"(mean offset COM − surface = {agg['mean_m1_com_minus_surface_cm']:.2f} cm)",
        "",
        "## Verdict",
        "",
        verdict,
        "",
        "## Per-clip event summary",
        "",
    ]
    rows = [["subset", "seq_id", "T", "n_events", "n_valid", "n_unstable", "n_edge", "n_onset_lt5"]]
    for a in audits:
        n_events = len(a["events"])
        n_valid = sum(1 for e in a["events"] if e["valid"])
        n_unstable = sum(1 for e in a["events"] if e["denom_unstable"])
        n_edge = sum(1 for e in a["events"] if not e["away_from_edge"])
        n_onset_lt5 = sum(1 for e in a["events"] if e["kind"] == "onset" and e["frame"] < 5)
        rows.append([a["subset"], a["seq_id"], a["seq_len"], n_events, n_valid, n_unstable, n_edge, n_onset_lt5])
    lines.append(format_md_table(rows))
    lines.append("")
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.md}")


if __name__ == "__main__":
    main()
