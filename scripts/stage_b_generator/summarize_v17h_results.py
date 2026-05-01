"""Summarise the v17-H (B2) part_margin / segment_consistency sweep + B1
final.pt re-eval + B3 residual drift diagnostic into one comparison table.

Reads from runs/eval/<EVAL_PREFIX>_<ckpt_tag>_{contact_dist, temporal_coupling,
guided_temporal_coupling, alignment_to_gt_roundtrip, guided_alignment_to_gt_roundtrip,
qual}/ for each requested experiment.

Usage::

    python scripts/stage_b_generator/summarize_v17h_results.py \\
        --runs-dir runs/eval

Outputs the table to stdout.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


# Experiments to summarise. Each entry: (label, eval_prefix, ckpt_tag, notes).
EXPERIMENTS: list[tuple[str, str, str, str]] = [
    ("v17-E.20 (best_contact, prior)",   "stageB_v0_17_v16bc_per_step_iters20",                 "bc",    "prior baseline"),
    ("v17-E.50 (best_contact, prior)",   "stageB_v0_17_v16bc_per_step_iters50",                 "bc",    "prior baseline"),
    ("B1: v17-E.20 on final.pt",         "stageB_v0_17_v16final_per_step_iters20",              "final", "B1 — final.pt"),
    ("B1: v17-E.50 on final.pt",         "stageB_v0_17_v16final_per_step_iters50",              "final", "B1 — final.pt"),
    ("B2 sanity (pm=0, sc=0)",           "stageB_v0_17h_v16bc_pm0_sc0",                         "bc",    "should match v17-E.20 prior"),
    ("B2 part_margin=0.5",               "stageB_v0_17h_v16bc_pm0_5",                           "bc",    "B2 — part_margin sweep"),
    ("B2 part_margin=1.0",               "stageB_v0_17h_v16bc_pm1_0",                           "bc",    "B2 — part_margin sweep"),
    ("B2 part_margin=2.0",               "stageB_v0_17h_v16bc_pm2_0",                           "bc",    "B2 — part_margin sweep"),
    ("B2 pm=1.0 + segment_cons=0.1",     "stageB_v0_17h_v16bc_pm10_sc0_1",                      "bc",    "B2 — pm + sc"),
    ("B2 pm=1.0 + segment_cons=0.5",     "stageB_v0_17h_v16bc_pm10_sc0_5",                      "bc",    "B2 — pm + sc"),
    ("B2 pm=1.0 + segment_cons=1.0",     "stageB_v0_17h_v16bc_pm10_sc1_0",                      "bc",    "B2 — pm + sc"),
]


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _condition_mean_min_dist(contact_summary: dict[str, Any], suffix: str) -> float | None:
    if contact_summary is None:
        return None
    for cond_name, cond in contact_summary.get("conditions", {}).items():
        if cond_name.endswith(suffix):
            return float(cond.get("mean_min_dist_m", float("nan")))
    return None


def _summarise_experiment(runs_dir: Path, eval_prefix: str, ckpt_tag: str) -> dict[str, Any]:
    """Read all eval summaries for one experiment and extract key fields."""
    base = runs_dir
    out: dict[str, Any] = {"prefix": eval_prefix, "ckpt": ckpt_tag}

    contact = _load_json(base / f"{eval_prefix}_{ckpt_tag}_contact_dist" / "summary.json")
    out["raw_full_dist"] = _condition_mean_min_dist(contact, "/full")
    out["guided_dist"]   = _condition_mean_min_dist(contact, "/full_guided")
    out["text_only_dist"] = _condition_mean_min_dist(contact, "/text_only")
    out["swap_dist"]      = _condition_mean_min_dist(contact, "/swap")
    out["gt_roundtrip_dist"] = _condition_mean_min_dist(contact, "/gt_roundtrip")

    raw_tc = _load_json(base / f"{eval_prefix}_{ckpt_tag}_temporal_coupling" / "summary.json")
    guided_tc = _load_json(base / f"{eval_prefix}_{ckpt_tag}_guided_temporal_coupling" / "summary.json")

    def tc_field(d: dict[str, Any] | None, key: str) -> float | None:
        if d is None:
            return None
        agg = d.get("aggregate") or d
        return float(agg.get(key, float("nan"))) if key in agg else None

    out["raw_coupled"]    = tc_field(raw_tc, "moving_coupled_frame_frac")
    out["guided_coupled"] = tc_field(guided_tc, "moving_coupled_frame_frac")

    raw_align = _load_json(base / f"{eval_prefix}_{ckpt_tag}_alignment_to_gt_roundtrip" / "summary.json")
    guided_align = _load_json(base / f"{eval_prefix}_{ckpt_tag}_guided_alignment_to_gt_roundtrip" / "summary.json")

    def align_field(d: dict[str, Any] | None, key: str) -> float | None:
        if d is None:
            return None
        agg = d.get("aggregate", {})
        return float(agg.get(key, float("nan"))) if key in agg else None

    out["raw_iou"]      = align_field(raw_align, "moving_contact_temporal_iou")
    out["guided_iou"]   = align_field(guided_align, "moving_contact_temporal_iou")
    out["raw_correct"]  = align_field(raw_align, "moving_right_part_contact_recall_on_gt")
    out["guided_correct"] = align_field(guided_align, "moving_right_part_contact_recall_on_gt")
    out["raw_local"]    = align_field(raw_align, "moving_same_gt_part_local_position_error_m_on_gt_contact")
    out["guided_local"] = align_field(guided_align, "moving_same_gt_part_local_position_error_m_on_gt_contact")
    out["raw_target_local"]    = align_field(raw_align, "moving_target_part_local_error_m_on_gt_contact")
    out["guided_target_local"] = align_field(guided_align, "moving_target_part_local_error_m_on_gt_contact")

    # B3 — residual drift from guidance_trace.json
    trace = _load_json(base / f"{eval_prefix}_{ckpt_tag}_qual" / "full_guided" / "guidance_trace.json")
    drifts: list[float] = []
    losses_opt: list[float] = []
    losses_final: list[float] = []
    if trace is not None:
        for clip in trace.get("per_clip", []):
            ps = (clip.get("info", {}) or {}).get("per_step")
            if not ps:
                continue
            d = ps.get("residual_drift")
            lo = ps.get("loss_opt_last_inner")
            lf = ps.get("loss_final_post_residual")
            if d is not None and d == d:  # not NaN
                drifts.append(float(d))
            if lo is not None and lo == lo:
                losses_opt.append(float(lo))
            if lf is not None and lf == lf:
                losses_final.append(float(lf))
    if drifts:
        out["drift_mean_m"] = statistics.mean(drifts)
        out["drift_abs_mean_m"] = statistics.mean(abs(d) for d in drifts)
        out["drift_max_abs_m"] = max(abs(d) for d in drifts)
        out["drift_n_clips"] = len(drifts)
    else:
        out["drift_mean_m"] = None
        out["drift_abs_mean_m"] = None
        out["drift_max_abs_m"] = None
        out["drift_n_clips"] = 0
    if losses_opt:
        out["loss_opt_mean"] = statistics.mean(losses_opt)
    if losses_final:
        out["loss_final_mean"] = statistics.mean(losses_final)
    return out


def _fmt(x: float | None, *, scale: float = 1.0, digits: int = 2, pct: bool = False) -> str:
    if x is None or x != x:
        return "—"
    v = x * scale
    if pct:
        return f"{v * 100:.{digits}f}"
    return f"{v:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/eval"))
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for label, eval_prefix, ckpt_tag, notes in EXPERIMENTS:
        r = _summarise_experiment(args.runs_dir, eval_prefix, ckpt_tag)
        r["label"] = label
        r["notes"] = notes
        rows.append(r)

    # Header
    cols = [
        ("variant", 38),
        ("raw cont", 8), ("guided cont", 11),
        ("g IoU", 7), ("g correct", 9), ("g local", 8),
        ("drift |d|", 9), ("drift max", 9), ("N", 4),
    ]
    print("=" * sum(w + 1 for _, w in cols))
    print(" ".join(f"{n:<{w}}" for n, w in cols))
    print("-" * sum(w + 1 for _, w in cols))

    def row_str(r: dict[str, Any]) -> str:
        return " ".join([
            f"{r['label'][:38]:<38}",
            f"{_fmt(r.get('raw_full_dist'), scale=100, digits=2):>8}",
            f"{_fmt(r.get('guided_dist'),   scale=100, digits=2):>11}",
            f"{_fmt(r.get('guided_iou'),    digits=3):>7}",
            f"{_fmt(r.get('guided_correct'), digits=3):>9}",
            f"{_fmt(r.get('guided_local'),  scale=100, digits=2):>8}",
            f"{_fmt(r.get('drift_abs_mean_m'), scale=100, digits=2):>9}",
            f"{_fmt(r.get('drift_max_abs_m'),  scale=100, digits=2):>9}",
            f"{r.get('drift_n_clips') or 0:>4d}",
        ])

    for r in rows:
        if r.get("guided_dist") is None and r.get("raw_full_dist") is None:
            continue  # missing experiment
        print(row_str(r))
    print("=" * sum(w + 1 for _, w in cols))
    print()
    print("Units: contact / local error in cm; drift |delta| = mean |loss_final_post_residual - loss_opt_last_inner| in cm.")
    print("Reference baselines: GT_orig 13.09 cm, GT VQ roundtrip 18.47 cm,")
    print("                     v17-C (10 iters, prior) 21.77 / 0.439 / 0.202 / 46.13.")
    print()
    # Also dump full per-experiment JSON for downstream analysis.
    out_path = args.runs_dir.parent / "v17h_summary.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Full summary JSON: {out_path}")


if __name__ == "__main__":
    main()
