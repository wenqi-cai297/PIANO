"""Unified metric summary for the 2026-05-03 metric overhaul.

Reads from one v17-style eval tree:
  - <prefix>_<ckpt>_contact_dist/summary.json     -> mean_min_dist
  - <prefix>_<ckpt>_guided_alignment_to_gt_roundtrip/summary.json
        -> contact IoU, correct-part, same-part local, weighted local, soft IoU
  - <prefix>_<ckpt>_guided_temporal_coupling/summary.json
        -> moving_coupled_frame_frac, moving_mean_best_kin_score (N9)
  - _unified_metrics/penetration/<label>_summary.json   -> N1/N2 penetration
  - _unified_metrics/jerk/<label>_summary.json          -> N7 jerk

Plus computes KS-distance to GT_orig jerk distribution per condition
(reads `_unified_metrics/jerk/<label>_samples.npz`).

For each condition emits:
  Ship gates (with codec-floor-normalized "% absorbed"):
    - mean_min_dist (cm)                + flag if < codec floor (gaming)
    - mean_part_penetration_depth (cm)  + delta vs GT_orig
    - max_part_penetration_depth (cm)
    - moving_correct_part_recall        + % of codec ceiling absorbed
    - moving_weighted_local_error (cm)  + delta vs codec floor
    - moving_weighted_target_error (cm) + delta vs codec floor
  Auxiliary:
    - moving_coupled_frame_frac
    - moving_mean_best_kin_score (N9)
    - moving_contact_temporal_iou (rigid)
    - moving_soft_contact_temporal_iou_pm2 (N6)
    - mean_jerk + jerk_KS_distance_to_GT (N7)

Outputs:
  - text table to stdout
  - <output>/unified_metrics.json with full per-condition data
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


# ============================================================================
# Conditions catalog (matches run_unified_metrics_eval.sh)
# ============================================================================

CONDITIONS: list[tuple[str, str, str]] = [
    # (label, eval_prefix, ckpt_tag) — used to find <prefix>_<ckpt>_* dirs.
    # GT references use a synthetic mapping: contact_dist gives them via the
    # /gt_original /gt_roundtrip subdirectories of <prefix>_gt_roundtrip_80;
    # alignment/coupling/penetration/jerk come from the unified_metrics dirs.
    ("gt_orig",                  "stageB_v0_17_v16final_per_step_iters20", "gt_orig"),
    ("gt_roundtrip",             "stageB_v0_17_v16final_per_step_iters20", "gt_roundtrip"),
    ("v17-C v16bc",              "stageB_v0_17_per_step_v16bc",            "bc"),
    ("v17-C v16bc no-gumbel",    "stageB_v0_17_v16bc_c_no_gumbel",         "bc"),
    ("v17-D stacked",            "stageB_v0_17_v16bc_stacked",             "bc"),
    ("v17-E.20 v16bc",           "stageB_v0_17_v16bc_per_step_iters20",    "bc"),
    ("v17-E.20 v16bc no-gumbel", "stageB_v0_17_v16bc_e20_no_gumbel",       "bc"),
    ("v17-E.50 v16bc",           "stageB_v0_17_v16bc_per_step_iters50",    "bc"),
    ("v17-F.10 Gumbel",          "stageB_v0_17_v16bc_f10_gumbel",          "bc"),
    ("v17-F.20 Gumbel",          "stageB_v0_17_v16bc_f20_gumbel",          "bc"),
    ("v17-G boost=1",            "stageB_v0_17_v16bc_g_b1",                "bc"),
    ("v17-G boost=2",            "stageB_v0_17_v16bc_g_b2",                "bc"),
    ("v17-G boost=5",            "stageB_v0_17_v16bc_g_b5",                "bc"),
    ("v17-G boost=10",           "stageB_v0_17_v16bc_g_b10",               "bc"),
    ("v17-G boost=20",           "stageB_v0_17_v16bc_g_b20",               "bc"),
    ("B1: v17-E.20 final.pt",    "stageB_v0_17_v16final_per_step_iters20", "final"),
    ("B1: v17-E.50 final.pt",    "stageB_v0_17_v16final_per_step_iters50", "final"),
    ("B2 sanity (pm=0)",         "stageB_v0_17h_v16bc_pm0_sc0",            "bc"),
    ("B2 part_margin=0.5",       "stageB_v0_17h_v16bc_pm0_5",              "bc"),
    ("B2 part_margin=1.0",       "stageB_v0_17h_v16bc_pm1_0",              "bc"),
    ("B2 part_margin=2.0",       "stageB_v0_17h_v16bc_pm2_0",              "bc"),
    ("B2 pm=1.0 + sc=0.1",       "stageB_v0_17h_v16bc_pm10_sc0_1",         "bc"),
    ("B2 pm=1.0 + sc=0.5",       "stageB_v0_17h_v16bc_pm10_sc0_5",         "bc"),
    ("B2 pm=1.0 + sc=1.0",       "stageB_v0_17h_v16bc_pm10_sc1_0",         "bc"),
]

# Penetration / jerk label mapping (match run_unified_metrics_eval.sh)
LABEL_MAP: dict[str, str] = {
    "gt_orig": "gt_orig",
    "gt_roundtrip": "gt_roundtrip",
    "v17-C v16bc": "v17C_v16bc",
    "v17-C v16bc no-gumbel": "v17C_v16bc_no_gumbel",
    "v17-D stacked": "v17D_stacked",
    "v17-E.20 v16bc": "v17E20_v16bc",
    "v17-E.20 v16bc no-gumbel": "v17E20_v16bc_no_gumbel",
    "v17-E.50 v16bc": "v17E50_v16bc",
    "v17-F.10 Gumbel": "v17F10_gumbel",
    "v17-F.20 Gumbel": "v17F20_gumbel",
    "v17-G boost=1": "v17G_b1",
    "v17-G boost=2": "v17G_b2",
    "v17-G boost=5": "v17G_b5",
    "v17-G boost=10": "v17G_b10",
    "v17-G boost=20": "v17G_b20",
    "B1: v17-E.20 final.pt": "B1_v17E20_final",
    "B1: v17-E.50 final.pt": "B1_v17E50_final",
    "B2 sanity (pm=0)": "B2_pm0_sc0",
    "B2 part_margin=0.5": "B2_pm0_5",
    "B2 part_margin=1.0": "B2_pm1_0",
    "B2 part_margin=2.0": "B2_pm2_0",
    "B2 pm=1.0 + sc=0.1": "B2_pm10_sc0_1",
    "B2 pm=1.0 + sc=0.5": "B2_pm10_sc0_5",
    "B2 pm=1.0 + sc=1.0": "B2_pm10_sc1_0",
}


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _condition_mean_min_dist(contact_summary: dict, suffix: str) -> float | None:
    if contact_summary is None:
        return None
    for cond_name, cond in contact_summary.get("conditions", {}).items():
        if cond_name.endswith(suffix):
            return float(cond.get("mean_min_dist_m", float("nan")))
    return None


def _summarise_condition(
    runs_dir: Path,
    label: str,
    eval_prefix: str,
    ckpt_tag: str,
    *,
    pen_dir: Path,
    jerk_dir: Path,
) -> dict[str, Any]:
    out: dict[str, Any] = {"label": label, "prefix": eval_prefix, "ckpt": ckpt_tag}

    # mean_min_dist (raw / guided)
    contact = _load_json(runs_dir / f"{eval_prefix}_{ckpt_tag}_contact_dist" / "summary.json")
    if ckpt_tag in {"gt_orig", "gt_roundtrip"}:
        # GT: comes from the gt_roundtrip_80 dir under the contact_dist
        gt_contact = _load_json(runs_dir / f"{eval_prefix}_{ckpt_tag}_contact_dist" / "summary.json")
        if gt_contact is None:
            # Fall back to using v17-E.20 final.pt's gt_roundtrip refs
            gt_contact = _load_json(runs_dir / "stageB_v0_17_v16final_per_step_iters20_final_contact_dist" / "summary.json")
        suffix = "/gt_original" if ckpt_tag == "gt_orig" else "/gt_roundtrip"
        out["raw_dist_m"] = _condition_mean_min_dist(gt_contact, suffix)
        out["guided_dist_m"] = out["raw_dist_m"]   # GT has no "guided"; use same
    else:
        out["raw_dist_m"] = _condition_mean_min_dist(contact, "/full")
        out["guided_dist_m"] = _condition_mean_min_dist(contact, "/full_guided")

    # alignment (guided when it exists, else fall back to raw)
    align_dir = runs_dir / f"{eval_prefix}_{ckpt_tag}_guided_alignment_to_gt_roundtrip"
    if not align_dir.exists():
        align_dir = runs_dir / f"{eval_prefix}_{ckpt_tag}_alignment_to_gt_roundtrip"
    align = _load_json(align_dir / "summary.json")
    agg = (align or {}).get("aggregate", {})

    out["moving_iou"] = agg.get("moving_contact_temporal_iou")
    out["soft_iou_pm2"] = agg.get("moving_soft_contact_temporal_iou_pm2")
    out["correct_part"] = agg.get("moving_right_part_contact_recall_on_gt")
    out["same_part_local_m"] = agg.get("moving_same_gt_part_local_position_error_m_on_gt_contact")
    out["target_local_m"] = agg.get("moving_target_part_local_error_m_on_gt_contact")
    out["weighted_local_m"] = agg.get("moving_weighted_local_error_m")
    out["weighted_target_m"] = agg.get("moving_weighted_target_error_m")

    # temporal coupling (guided)
    tc_dir = runs_dir / f"{eval_prefix}_{ckpt_tag}_guided_temporal_coupling"
    if not tc_dir.exists():
        tc_dir = runs_dir / f"{eval_prefix}_{ckpt_tag}_temporal_coupling"
    tc = _load_json(tc_dir / "summary.json")
    tc_agg = (tc or {}).get("aggregate", tc or {})
    out["coupled_frac"] = tc_agg.get("moving_coupled_frame_frac")
    out["coupled_strength"] = tc_agg.get("moving_mean_best_kin_score")

    # Penetration
    pen = _load_json(pen_dir / f"{LABEL_MAP[label]}_summary.json")
    pen_agg = (pen or {}).get("conditions", {})
    if pen_agg:
        first = next(iter(pen_agg.values()))
        out["mean_pen_m"] = first.get("mean_pen_m")
        out["max_pen_m_avg"] = first.get("max_pen_m_avg")
        out["max_pen_m_overall"] = first.get("max_pen_m_overall")
        out["frac_pen_gt_2cm"] = first.get("frac_frames_pen_gt_2cm")
        out["frac_pen_gt_5cm"] = first.get("frac_frames_pen_gt_5cm")

    # Jerk
    jerk = _load_json(jerk_dir / f"{LABEL_MAP[label]}_summary.json")
    jerk_agg = (jerk or {}).get("conditions", {})
    if jerk_agg:
        first = next(iter(jerk_agg.values()))
        out["mean_jerk"] = first.get("mean_jerk")
        out["max_jerk_avg"] = first.get("max_jerk_avg")
        out["mean_jerk_hands"] = first.get("mean_jerk_hands")

    return out


# ============================================================================
# Codec-floor normalisation
# ============================================================================

def _absorbed_higher_better(model: float | None, codec_floor: float | None) -> float | None:
    """For 'higher better' metrics like recall: returns model / codec_floor.

    1.0 = at codec ceiling (best possible); >1.0 = exceeds (suspicious);
    <1.0 = below codec ceiling, room to improve.
    """
    if model is None or codec_floor is None or codec_floor <= 0:
        return None
    return float(model) / float(codec_floor)


def _absorbed_lower_better(model: float | None, codec_floor: float | None) -> float | None:
    """For 'lower better' metrics: returns codec_floor / model.

    1.0 = at codec floor; >1.0 = exceeds floor (better than VQ codec, suspicious);
    <1.0 = worse than codec floor.
    """
    if model is None or codec_floor is None or model is None or model <= 0:
        return None
    return float(codec_floor) / float(model)


def _ks_distance(samples_a: Any, samples_b: Any) -> float | None:
    """Kolmogorov-Smirnov distance between two 1-D samples."""
    try:
        from scipy.stats import ks_2samp
    except ImportError:
        return None
    if samples_a is None or samples_b is None or len(samples_a) == 0 or len(samples_b) == 0:
        return None
    return float(ks_2samp(samples_a, samples_b).statistic)


def _load_jerk_samples(jerk_dir: Path, label: str):
    import numpy as np
    p = jerk_dir / f"{LABEL_MAP[label]}_samples.npz"
    if not p.exists():
        return None
    d = np.load(p)
    return d["jerk"]


# ============================================================================
# Main
# ============================================================================

def _fmt(x, *, scale=1.0, digits=2, pct=False):
    if x is None or (isinstance(x, float) and (x != x)):
        return "—"
    v = float(x) * scale
    if pct:
        return f"{v * 100:.{digits}f}"
    return f"{v:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/eval"))
    parser.add_argument(
        "--unified-dir", type=Path,
        default=Path("runs/eval/_unified_metrics"),
    )
    parser.add_argument("--output", type=Path, default=Path("runs/v17h_unified_summary.json"))
    args = parser.parse_args()

    pen_dir = args.unified_dir / "penetration"
    jerk_dir = args.unified_dir / "jerk"

    rows = []
    for label, prefix, ckpt in CONDITIONS:
        rows.append(_summarise_condition(
            args.runs_dir, label, prefix, ckpt,
            pen_dir=pen_dir, jerk_dir=jerk_dir,
        ))

    # Find codec floor row (gt_roundtrip vs gt_orig). The alignment summary
    # we already measured: stageB_codec_floor_alignment.
    codec_floor_align = _load_json(
        args.runs_dir / "stageB_codec_floor_alignment" / "summary.json"
    )
    codec_agg = (codec_floor_align or {}).get("aggregate", {})
    codec_floor = {
        "moving_iou":         codec_agg.get("moving_contact_temporal_iou"),
        "soft_iou_pm2":       codec_agg.get("moving_soft_contact_temporal_iou_pm2"),
        "correct_part":       codec_agg.get("moving_right_part_contact_recall_on_gt"),
        "same_part_local_m":  codec_agg.get("moving_same_gt_part_local_position_error_m_on_gt_contact"),
        "target_local_m":     codec_agg.get("moving_target_part_local_error_m_on_gt_contact"),
        "weighted_local_m":   codec_agg.get("moving_weighted_local_error_m"),
        "weighted_target_m":  codec_agg.get("moving_weighted_target_error_m"),
    }
    # GT_orig reference from rows
    gt_orig = next(r for r in rows if r["label"] == "gt_orig")

    # KS distance of jerk to GT_orig
    gt_jerk = _load_jerk_samples(jerk_dir, "gt_orig")
    for r in rows:
        r_jerk = _load_jerk_samples(jerk_dir, r["label"])
        r["jerk_ks_to_gt"] = _ks_distance(r_jerk, gt_jerk)
        r["mean_pen_delta_vs_gt"] = (
            None if r.get("mean_pen_m") is None or gt_orig.get("mean_pen_m") is None
            else float(r["mean_pen_m"]) - float(gt_orig["mean_pen_m"])
        )
        r["correct_part_absorbed"] = _absorbed_higher_better(
            r.get("correct_part"), codec_floor.get("correct_part"),
        )
        r["weighted_local_absorbed"] = _absorbed_lower_better(
            r.get("weighted_local_m"), codec_floor.get("weighted_local_m"),
        )

    # ---- Print table ----
    header = (
        f"{'condition':40} "
        f"{'cont':>6} "
        f"{'pen':>6} "
        f"{'pen-2cm':>7} "
        f"{'IoU':>6} "
        f"{'softIoU':>7} "
        f"{'corPt':>6} "
        f"{'%abs':>6} "
        f"{'wLoc':>6} "
        f"{'wTgt':>6} "
        f"{'cpld':>6} "
        f"{'cpldS':>6} "
        f"{'jerk':>6} "
        f"{'KS':>6}"
    )
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    def row_str(r):
        return (
            f"{r['label'][:40]:40} "
            f"{_fmt(r.get('guided_dist_m'), scale=100, digits=2):>6} "
            f"{_fmt(r.get('mean_pen_m'), scale=100, digits=2):>6} "
            f"{_fmt(r.get('frac_pen_gt_2cm'), digits=1, pct=True):>6}% "
            f"{_fmt(r.get('moving_iou'), digits=3):>6} "
            f"{_fmt(r.get('soft_iou_pm2'), digits=3):>7} "
            f"{_fmt(r.get('correct_part'), digits=3):>6} "
            f"{_fmt(r.get('correct_part_absorbed'), digits=0, pct=True):>5}% "
            f"{_fmt(r.get('weighted_local_m'), scale=100, digits=2):>6} "
            f"{_fmt(r.get('weighted_target_m'), scale=100, digits=2):>6} "
            f"{_fmt(r.get('coupled_frac'), digits=3):>6} "
            f"{_fmt(r.get('coupled_strength'), digits=3):>6} "
            f"{_fmt(r.get('mean_jerk'), digits=0):>6} "
            f"{_fmt(r.get('jerk_ks_to_gt'), digits=3):>6}"
        )

    for r in rows:
        if r.get("guided_dist_m") is None and r.get("raw_dist_m") is None:
            continue
        print(row_str(r))
    print("=" * len(header))
    print()
    print(f"codec floor (GT_rt vs GT_orig):  IoU={_fmt(codec_floor['moving_iou'], digits=3)}  "
          f"corPt={_fmt(codec_floor['correct_part'], digits=3)}  "
          f"wLoc={_fmt(codec_floor['weighted_local_m'], scale=100, digits=2)} cm  "
          f"wTgt={_fmt(codec_floor['weighted_target_m'], scale=100, digits=2)} cm")
    print(f"GT_orig reference:               mean_pen={_fmt(gt_orig.get('mean_pen_m'), scale=100, digits=2)} cm  "
          f"mean_jerk={_fmt(gt_orig.get('mean_jerk'), digits=0)} m/s^3")
    print()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "rows": rows,
            "codec_floor_alignment": codec_floor,
            "gt_orig_reference": {
                "mean_pen_m": gt_orig.get("mean_pen_m"),
                "max_pen_m_avg": gt_orig.get("max_pen_m_avg"),
                "frac_pen_gt_2cm": gt_orig.get("frac_pen_gt_2cm"),
                "mean_jerk": gt_orig.get("mean_jerk"),
                "max_jerk_avg": gt_orig.get("max_jerk_avg"),
                "raw_dist_m": gt_orig.get("raw_dist_m"),
            },
        }, f, indent=2)
    print(f"Saved JSON: {args.output}")


if __name__ == "__main__":
    main()
