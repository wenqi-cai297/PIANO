"""Summarize Round-29 loss-strategy FULL-DATA results into a Markdown report.

Per analyses/2026-05-27_round29_loss_strategy_v2_codex_review.md §Final recommendation.

Reads 4 full-data variants' diag stats from
``analyses/round29_<variant_id>_diag_<kind>_<sublabel>/<kind>_stats.json``
(launcher emits 4 × 3 × 2 = 24 files: 4 variants × 3 diag kinds × 2
selection buckets (train-subset / heldout-val-subset)), then writes a
comparison Markdown report to
``analyses/2026-05-27_round29_loss_strategy_full_data_report.md``.

Per-variant metrics surfaced (same as v2 review):

* sustained contact: drift_max_mean_cm, %drift>5cm, %drift>10cm,
                     track_frac_mean, %track<0.5
* gait:              both_swing, both_stance, trans/s, L_R_corr,
                     step_period_rate
* body action:       mean delta_err, mean dir_cos, mean amp_ratio,
                     LW + RW wrist deltas
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = ROOT / "analyses"
DEFAULT_REPORT_PATH = (
    ROOT / "analyses" / "2026-05-27_round29_loss_strategy_full_data_report.md"
)

# 4 full-data variants per Codex review. Order drives report row order.
VARIANTS: tuple[str, ...] = (
    "r29_lsf_a2_baseline_from_scratch",
    "r29_lsf_a2_anchor2_mixed",
    "r29_lsf_a3_baseline_from_scratch",
    "r29_lsf_a3_anchor2_mixed",
)
SUBLABELS: tuple[str, ...] = ("train", "val")
KINDS: tuple[str, ...] = ("sustained_contact", "gait", "body_action")


def _load_stats(
    results_root: Path, variant_id: str, kind: str, sublabel: str,
) -> dict[str, Any] | None:
    """Load <results_root>/round29_<variant>_diag_<kind>_<sub>/<kind>_stats.json."""
    p = (
        results_root
        / f"round29_{variant_id}_diag_{kind}_{sublabel}"
        / f"{kind}_stats.json"
    )
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _fmt(x: Any, prec: int = 3) -> str:
    if x is None:
        return "-"
    try:
        f = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(f) or math.isinf(f):
        return "-"
    return f"{f:.{prec}f}"


def _pct(x: Any) -> str:
    if x is None:
        return "-"
    try:
        f = float(x)
    except (TypeError, ValueError):
        return "-"
    if math.isnan(f) or math.isinf(f):
        return "-"
    return f"{100.0 * f:.1f}%"


def _sustained_row(stats: dict[str, Any] | None) -> dict[str, Any]:
    if not stats:
        return {k: None for k in (
            "drift_max_mean_cm", "pct_drift_max_above_5cm",
            "pct_drift_max_above_10cm", "track_frac_mean",
            "pct_track_frac_below_0.5", "n_segments",
        )}
    overall = stats.get("overall", {}) or {}
    n_seg = overall.get("n_segments")
    n_above_5 = overall.get("n_drift_max_above_5cm")
    n_above_10 = overall.get("n_drift_max_above_10cm")
    tr = overall.get("tracking_fraction", {}) or {}
    return {
        "drift_max_mean_cm": (overall.get("drift_max_cm", {}) or {}).get("mean"),
        "pct_drift_max_above_5cm": (
            (n_above_5 / n_seg) if (n_seg and n_above_5 is not None) else None
        ),
        "pct_drift_max_above_10cm": (
            (n_above_10 / n_seg) if (n_seg and n_above_10 is not None) else None
        ),
        "track_frac_mean": tr.get("mean"),
        "pct_track_frac_below_0.5": tr.get("rate_below_0.5"),
        "n_segments": n_seg,
    }


_GAIT_FIELDS = (
    "frac_both_swing", "frac_both_stance", "transitions_per_sec",
    "L_R_height_corr", "step_period_rate", "n_walking_segments",
)


def _gait_row(stats: dict[str, Any] | None) -> dict[str, Any]:
    """Pull both pred_aggregate AND gt_aggregate gait stats into a flat row.

    The hardcoded GT reference paragraph in v1 of this report was wrong
    for the heldout-val subset (Codex P1 review 2026-05-27): the val
    subset is a DIFFERENT 48 clips, so its GT walking distribution
    differs from the train subset's. ``round26_gait_diag.py:432`` emits
    ``gt_aggregate`` alongside ``pred_aggregate`` with identical schema;
    we now surface it as ``gt_<field>`` so the renderer can show a
    per-subset GT reference row.
    """
    if not stats:
        out = {k: None for k in _GAIT_FIELDS}
        out.update({f"gt_{k}": None for k in _GAIT_FIELDS})
        return out

    pa = stats.get("pred_aggregate", {}) or {}
    ga = stats.get("gt_aggregate", {}) or {}

    def _extract(agg: dict[str, Any]) -> dict[str, Any]:
        return {
            "frac_both_swing": (agg.get("frac_both_swing", {}) or {}).get("mean"),
            "frac_both_stance": (agg.get("frac_both_stance", {}) or {}).get("mean"),
            "transitions_per_sec": (agg.get("transitions_per_second", {}) or {}).get("mean"),
            "L_R_height_corr": (agg.get("L_R_height_corr", {}) or {}).get("mean"),
            "step_period_rate": (agg.get("step_period_frames", {}) or {}).get("rate_with_period"),
        }

    pred = _extract(pa)
    gt = _extract(ga)
    out = dict(pred)
    out["n_walking_segments"] = stats.get("n_walking_segments")
    out.update({f"gt_{k}": gt[k] for k in pred})
    return out


def _subset_gt_reference(
    rows: dict[str, dict[str, dict[str, dict[str, Any]]]],
    sublabel: str,
) -> dict[str, Any]:
    """Mean GT gait values across all variants for the given sublabel.

    All variants share the same 48-clip selection and run the gait diag
    on the same GT joints → ``gt_aggregate`` is identical across them
    (modulo any floating-point variability). We take the mean of the
    available rows as a defensive aggregate.
    """
    fields = ("frac_both_swing", "frac_both_stance", "transitions_per_sec",
              "L_R_height_corr", "step_period_rate")
    bucket: dict[str, list[float]] = {k: [] for k in fields}
    for variant_rows in rows.values():
        gait = variant_rows.get(sublabel, {}).get("gait", {})
        for f in fields:
            val = gait.get(f"gt_{f}")
            if val is not None:
                try:
                    bucket[f].append(float(val))
                except (TypeError, ValueError):
                    pass
    return {f: (sum(vs) / len(vs)) if vs else None for f, vs in bucket.items()}


def _body_action_row(stats: dict[str, Any] | None) -> dict[str, Any]:
    if not stats:
        return {k: None for k in (
            "delta_err_cm_mean_overall", "direction_cosine_mean_overall",
            "amp_ratio_mean_overall",
            "left_wrist_delta_err_cm_mean", "right_wrist_delta_err_cm_mean",
        )}
    agg = stats.get("aggregate", {}) or {}
    joints = list(agg.keys())
    if not joints:
        return _body_action_row(None)
    de = [agg[j].get("delta_error_cm_mean") for j in joints if agg[j].get("delta_error_cm_mean") is not None]
    dc = [agg[j].get("direction_cosine_mean") for j in joints if agg[j].get("direction_cosine_mean") is not None]
    ar = [agg[j].get("amp_ratio_mean") for j in joints if agg[j].get("amp_ratio_mean") is not None]
    lw = agg.get("left_wrist", {}) or {}
    rw = agg.get("right_wrist", {}) or {}
    return {
        "delta_err_cm_mean_overall": (sum(de) / len(de)) if de else None,
        "direction_cosine_mean_overall": (sum(dc) / len(dc)) if dc else None,
        "amp_ratio_mean_overall": (sum(ar) / len(ar)) if ar else None,
        "left_wrist_delta_err_cm_mean": lw.get("delta_error_cm_mean"),
        "right_wrist_delta_err_cm_mean": rw.get("delta_error_cm_mean"),
    }


def _gather(results_root: Path) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    """Pull (variant × sublabel × kind) rows.

    Returns: ``{variant: {sublabel: {kind: row}}}``.
    """
    out: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for v in VARIANTS:
        out[v] = {}
        for sub in SUBLABELS:
            out[v][sub] = {
                "sustained_contact": _sustained_row(
                    _load_stats(results_root, v, "sustained_contact", sub),
                ),
                "gait": _gait_row(_load_stats(results_root, v, "gait", sub)),
                "body_action": _body_action_row(
                    _load_stats(results_root, v, "body_action", sub),
                ),
            }
    return out


def _render_section_for_sublabel(
    a: Any, rows: dict, sublabel: str, label_order: tuple[str, ...],
) -> None:
    a(f"## Subset: `{sublabel}`")
    a("")
    if sublabel == "train":
        a("In-distribution sanity (same 48-clip balanced subset as the v2 "
          "mechanism screen + the A-group diag).")
    else:
        a("Heldout-val 48-clip balanced subset (built by "
          "`round29_build_val_diag_subset.py`); measures generalization.")
    a("")

    # Sustained contact
    a("### Sustained contact")
    a("")
    a("| variant | n_seg | drift_max mean (cm) | %drift>5cm | %drift>10cm | track_frac mean | %track<0.5 |")
    a("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for lbl in label_order:
        r = rows.get(lbl, {}).get(sublabel, {}).get("sustained_contact", {})
        a(
            f"| `{lbl}` | {_fmt(r.get('n_segments'), 0)} | "
            f"{_fmt(r.get('drift_max_mean_cm'), 2)} | "
            f"{_pct(r.get('pct_drift_max_above_5cm'))} | "
            f"{_pct(r.get('pct_drift_max_above_10cm'))} | "
            f"{_fmt(r.get('track_frac_mean'), 3)} | "
            f"{_pct(r.get('pct_track_frac_below_0.5'))} |"
        )
    a("")

    # Gait — prepend a per-subset GT reference row (read from
    # gt_aggregate in each variant's gait_stats.json; mean across
    # variants). v1 used hardcoded train-subset values uniformly across
    # both subsets, which mis-anchored val-subset interpretation.
    a("### Gait")
    a("")
    gt = _subset_gt_reference(rows, sublabel)
    a("| variant | n_walk_seg | frac_both_swing | frac_both_stance | trans/sec | L_R_corr | step_period_rate |")
    a("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    a(
        f"| **GT ({sublabel} subset)** | - | "
        f"**{_fmt(gt.get('frac_both_swing'), 3)}** | "
        f"**{_fmt(gt.get('frac_both_stance'), 3)}** | "
        f"**{_fmt(gt.get('transitions_per_sec'), 3)}** | "
        f"**{_fmt(gt.get('L_R_height_corr'), 3)}** | "
        f"**{_pct(gt.get('step_period_rate'))}** |"
    )
    for lbl in label_order:
        r = rows.get(lbl, {}).get(sublabel, {}).get("gait", {})
        a(
            f"| `{lbl}` | {_fmt(r.get('n_walking_segments'), 0)} | "
            f"{_fmt(r.get('frac_both_swing'), 3)} | "
            f"{_fmt(r.get('frac_both_stance'), 3)} | "
            f"{_fmt(r.get('transitions_per_sec'), 3)} | "
            f"{_fmt(r.get('L_R_height_corr'), 3)} | "
            f"{_pct(r.get('step_period_rate'))} |"
        )
    a("")

    # Body action
    a("### Body action (mean over 6 joints)")
    a("")
    a("| variant | mean delta_err (cm) | mean dir_cos | mean amp_ratio | LW err (cm) | RW err (cm) |")
    a("| --- | ---: | ---: | ---: | ---: | ---: |")
    for lbl in label_order:
        r = rows.get(lbl, {}).get(sublabel, {}).get("body_action", {})
        a(
            f"| `{lbl}` | "
            f"{_fmt(r.get('delta_err_cm_mean_overall'), 2)} | "
            f"{_fmt(r.get('direction_cosine_mean_overall'), 3)} | "
            f"{_fmt(r.get('amp_ratio_mean_overall'), 3)} | "
            f"{_fmt(r.get('left_wrist_delta_err_cm_mean'), 2)} | "
            f"{_fmt(r.get('right_wrist_delta_err_cm_mean'), 2)} |"
        )
    a("")


def _render_report(rows: dict) -> str:
    today = date.today().isoformat()
    lines: list[str] = []
    a = lines.append

    a(f"# Round-29 loss-strategy full-data report")
    a("")
    a(f"**Date:** {today}")
    a("**Protocol:** FULL InterAct train set, 80 ep, heldout val,")
    a("from-scratch (no init_checkpoint) for all 4 variants.")
    a("")
    a("**Axis A (injection):** A2 = `adapter_only`, A3 = `input_add_adapter`.")
    a("")
    a("**Axis L (loss strategy family):**")
    a("  - `baseline_from_scratch`: original a-group losses (pos_loss=5,")
    a("    anchor_pos=10, anchor_vel=2, world_vel=1, no relative/R29 terms).")
    a("    Fair Rule-1 reference.")
    a("  - `anchor2_mixed`: v2 48-clip winner (anchor_pos=2, anchor_vel=0.5,")
    a("    world_vel=0.5, R29 weights 0.10 each, swing_clearance ON).")
    a("")
    a("Two diag subsets per variant:")
    a("  - **`train`**: 48-clip balanced subset (in-distribution sanity, also")
    a("    used by the v2 48-clip mechanism screen — direct comparison).")
    a("  - **`val`**: 48-clip heldout-val balanced subset (generalization).")
    a("")
    a("GT walking reference for gait is shown per-subset inside each gait")
    a("table (read from `gt_aggregate` in each variant's `gait_stats.json`).")
    a("The train-subset and val-subset GT values differ because they are")
    a("different 48 clips with different walking compositions; using one as")
    a("a reference for the other (as v1 of this summarizer did) would")
    a("mislead `both_stance` / `step_period_rate` interpretations.")
    a("")

    label_order = (
        "r29_lsf_a2_baseline_from_scratch",
        "r29_lsf_a2_anchor2_mixed",
        "r29_lsf_a3_baseline_from_scratch",
        "r29_lsf_a3_anchor2_mixed",
    )

    for sublabel in SUBLABELS:
        _render_section_for_sublabel(a, rows, sublabel, label_order)

    # Decision rule
    a("## Decision rule")
    a("")
    a("Fair comparisons in this report (single-axis):")
    a("")
    a("- **Loss strategy axis** (injection fixed):")
    a("    `a2_baseline_from_scratch` vs `a2_anchor2_mixed`")
    a("    `a3_baseline_from_scratch` vs `a3_anchor2_mixed`")
    a("- **Injection axis** (loss strategy fixed):")
    a("    `a2_baseline_from_scratch` vs `a3_baseline_from_scratch`")
    a("    `a2_anchor2_mixed` vs `a3_anchor2_mixed`")
    a("")
    a("Per Codex v2 review §Final recommendation:")
    a("")
    a("1. If `anchor2_mixed` matches or beats `baseline_from_scratch` on")
    a("   contact AND wins on gait/body, take it as the loss-strategy")
    a("   mainline. Then pick injection (A2 or A3) by full-data deltas.")
    a("2. If `anchor2_mixed` loses contact at full-data (drift_max meaningfully")
    a("   above baseline), revisit weights — the 48-clip winner may not")
    a("   generalize.")
    a("3. If both still show `L_R_corr > 0` and near-zero `step_period_rate`,")
    a("   the next intervention is a P1 phase/footstep supervision loss that")
    a("   reads S4 channels [..., 5:13] — NOT another loss-strategy retune.")
    a("4. Compare train-subset vs val-subset rows per variant: large gap")
    a("   indicates overfitting (expected to be smaller at full-data than")
    a("   at 48-clip).")
    a("")
    a(
        "Do NOT pick the mainline from a single scalar. Cross-check at least "
        "drift_max + track_frac + both_stance + body_delta_err before "
        "committing to the next experiment."
    )

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize the 4 R29 loss-strategy FULL-DATA variants' diag "
            "results (4 variants × 3 diag × 2 subsets = 24 stats files) "
            "into a single comparison Markdown report."
        ),
    )
    parser.add_argument(
        "--results-root", type=Path, default=DEFAULT_RESULTS_ROOT,
        help="Default: analyses/.",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_REPORT_PATH,
    )
    args = parser.parse_args()

    rows = _gather(args.results_root)
    md = _render_report(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf-8")
    n_present = 0
    for v in VARIANTS:
        for sub in SUBLABELS:
            for k in KINDS:
                r = rows[v][sub].get(k, {})
                if r and (
                    r.get("drift_max_mean_cm") is not None
                    or r.get("frac_both_swing") is not None
                    or r.get("delta_err_cm_mean_overall") is not None
                ):
                    n_present += 1
    total = len(VARIANTS) * len(SUBLABELS) * len(KINDS)
    print(f"wrote {args.out}")
    print(f"  diag-section presence: {n_present}/{total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
