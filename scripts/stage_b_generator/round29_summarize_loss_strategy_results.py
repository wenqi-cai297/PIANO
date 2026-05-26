"""Summarize Round-29 loss-strategy ablation results into a Markdown report.

Per analyses/2026-05-27_round29_loss_strategy_ablation_prompt_for_claude_code.md §9.3.

Reads the four loss-strategy variants' diag stats JSONs from
``analyses/round29_<variant_id>_diag_<kind>/<kind>_stats.json`` and the
A2/A3 a-group baselines from ``analyses/round29_agroup_extracted/`` (if
available), then writes a comparison Markdown report to
``analyses/2026-05-27_round29_loss_strategy_ablation_report.md``.

Per-variant fields surfaced:

* sustained contact:
    drift_max_mean_cm
    pct_drift_max_above_5cm
    pct_drift_max_above_10cm
    track_frac_mean
    pct_track_frac_below_0.5
* gait:
    frac_both_swing
    frac_both_stance
    transitions_per_sec
    L_R_height_corr
    step_period_rate
* body action (per joint + summary):
    delta_err_cm_mean
    direction_cosine_mean
    amp_ratio_mean

Usage:
    python scripts/stage_b_generator/round29_summarize_loss_strategy_results.py
    python scripts/stage_b_generator/round29_summarize_loss_strategy_results.py \\
        --results-root analyses --out analyses/.../report.md
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
    ROOT / "analyses" / "2026-05-27_round29_loss_strategy_ablation_report.md"
)

VARIANTS: tuple[str, ...] = (
    "r29_ls_a2_no_dense_pos",
    "r29_ls_a3_no_dense_pos",
    "r29_ls_a2_relative_behavior",
    "r29_ls_a3_relative_behavior",
)

# A2/A3 a-group baselines (v27 warm-start, 48-clip 300ep). Diag results
# may be on disk under round29_agroup_extracted/ if the user has them.
BASELINE_CANDIDATES: dict[str, list[str]] = {
    "a2_baseline": [
        "round29_agroup_extracted/analyses/round29_r29_a2_adapter_only_diag_{kind}",
        "round29_r29_a2_adapter_only_diag_{kind}",
    ],
    "a3_baseline": [
        "round29_agroup_extracted/analyses/round29_r29_a3_input_add_adapter_diag_{kind}",
        "round29_r29_a3_input_add_adapter_diag_{kind}",
    ],
}

KINDS: tuple[str, ...] = ("sustained_contact", "gait", "body_action")


def _load_stats(results_root: Path, variant_id: str, kind: str) -> dict[str, Any] | None:
    """Load <results_root>/round29_<variant>_diag_<kind>/<kind>_stats.json."""
    p = results_root / f"round29_{variant_id}_diag_{kind}" / f"{kind}_stats.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _load_baseline_stats(
    results_root: Path, baseline_key: str, kind: str,
) -> dict[str, Any] | None:
    """Try each candidate path for a baseline (a2_baseline / a3_baseline)."""
    for tmpl in BASELINE_CANDIDATES.get(baseline_key, ()):
        p = results_root / tmpl.format(kind=kind)
        stats = p / f"{kind}_stats.json"
        if stats.exists():
            try:
                return json.loads(stats.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
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
    """Extract the §9.3 sustained-contact fields from a stats JSON."""
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


def _gait_row(stats: dict[str, Any] | None) -> dict[str, Any]:
    if not stats:
        return {k: None for k in (
            "frac_both_swing", "frac_both_stance", "transitions_per_sec",
            "L_R_height_corr", "step_period_rate", "n_walking_segments",
        )}
    pa = stats.get("pred_aggregate", {}) or {}
    return {
        "frac_both_swing": (pa.get("frac_both_swing", {}) or {}).get("mean"),
        "frac_both_stance": (pa.get("frac_both_stance", {}) or {}).get("mean"),
        "transitions_per_sec": (pa.get("transitions_per_second", {}) or {}).get("mean"),
        "L_R_height_corr": (pa.get("L_R_height_corr", {}) or {}).get("mean"),
        "step_period_rate": (pa.get("step_period_frames", {}) or {}).get("rate_with_period"),
        "n_walking_segments": stats.get("n_walking_segments"),
    }


def _body_action_row(stats: dict[str, Any] | None) -> dict[str, Any]:
    if not stats:
        return {k: None for k in (
            "delta_err_cm_mean_overall", "direction_cosine_mean_overall",
            "amp_ratio_mean_overall",
            "left_wrist_delta_err_cm_mean", "right_wrist_delta_err_cm_mean",
            "left_wrist_direction_cosine_mean", "right_wrist_direction_cosine_mean",
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
        "left_wrist_direction_cosine_mean": lw.get("direction_cosine_mean"),
        "right_wrist_direction_cosine_mean": rw.get("direction_cosine_mean"),
    }


def _gather(results_root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Pull (variant or baseline) × (sustained / gait / body_action) rows.

    Returns a dict ``{display_label: {kind: row}}``.
    """
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for vid in VARIANTS:
        out[vid] = {
            "sustained_contact": _sustained_row(
                _load_stats(results_root, vid, "sustained_contact"),
            ),
            "gait": _gait_row(_load_stats(results_root, vid, "gait")),
            "body_action": _body_action_row(_load_stats(results_root, vid, "body_action")),
        }
    for bkey in BASELINE_CANDIDATES:
        out[bkey] = {
            "sustained_contact": _sustained_row(
                _load_baseline_stats(results_root, bkey, "sustained_contact"),
            ),
            "gait": _gait_row(_load_baseline_stats(results_root, bkey, "gait")),
            "body_action": _body_action_row(
                _load_baseline_stats(results_root, bkey, "body_action"),
            ),
        }
    return out


def _render_report(rows: dict[str, dict[str, dict[str, Any]]]) -> str:
    """Render the §9.3 comparison Markdown."""
    today = date.today().isoformat()
    lines: list[str] = []
    a = lines.append

    a(f"# Round-29 loss-strategy ablation report")
    a("")
    a(f"**Date:** {today}")
    a("**Protocol:** 48-clip balanced subset, 300 ep, FULL-DENSE C/I/S/B content.")
    a("**Axis A (injection):** A2 = `adapter_only`, A3 = `input_add_adapter`.")
    a("**Axis L (loss strategy):**")
    a("  - `no_dense_pos`: drop dense FK MSE only. Keeps anchor_joint_pos/vel.")
    a("  - `relative_behavior`: drop pos_loss / anchor_pos / anchor_vel,")
    a("    weak global vel (0.2), enable R29 condition-consistency losses")
    a("    (interaction + both-airborne + stance velocity) + existing")
    a("    relative contact losses (rel_offset / drift / tracking).")
    a("")
    a("Baselines (when available) are pulled from")
    a("`analyses/round29_agroup_extracted/` — A2/A3 a-group ckpts (v27")
    a("warm-start, same 48-clip subset, same 300 ep). Note: baselines and")
    a("ablations have different training-init (warm-start vs from-scratch)")
    a("if the ablation ckpts also train from scratch; see ANALYSIS section.")
    a("")
    a("Variant key:")
    a("- `a2_baseline` — R29 A2 (adapter_only) a-group ckpt, v27 warm-start.")
    a("- `a3_baseline` — R29 A3 (input_add_adapter) a-group ckpt, v27 warm-start.")
    a("- `r29_ls_a*_no_dense_pos` — pos_loss=0 only, anchor kept.")
    a("- `r29_ls_a*_relative_behavior` — pos_loss=0, anchor=0, R29 consistency on.")
    a("")

    label_order = (
        "a2_baseline",
        "r29_ls_a2_no_dense_pos",
        "r29_ls_a2_relative_behavior",
        "a3_baseline",
        "r29_ls_a3_no_dense_pos",
        "r29_ls_a3_relative_behavior",
    )

    # ---------------- Sustained contact ----------------
    a("## Sustained contact")
    a("")
    a("Lower is better for drift_max_mean / %above_5cm / %above_10cm.")
    a("Higher is better for track_frac_mean. Lower is better for %track<0.5.")
    a("")
    a("| variant | n_seg | drift_max mean (cm) | %drift>5cm | %drift>10cm | track_frac mean | %track<0.5 |")
    a("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for lbl in label_order:
        r = rows.get(lbl, {}).get("sustained_contact", {})
        a(
            f"| `{lbl}` | {_fmt(r.get('n_segments'), 0)} | "
            f"{_fmt(r.get('drift_max_mean_cm'), 2)} | "
            f"{_pct(r.get('pct_drift_max_above_5cm'))} | "
            f"{_pct(r.get('pct_drift_max_above_10cm'))} | "
            f"{_fmt(r.get('track_frac_mean'), 3)} | "
            f"{_pct(r.get('pct_track_frac_below_0.5'))} |"
        )
    a("")

    # ---------------- Gait ----------------
    a("## Gait")
    a("")
    a("Reference GT values (from a-group baselines): `frac_both_swing≈0.29`,")
    a("`frac_both_stance≈0.15`, `transitions/sec≈0.79`, `L_R_height_corr≈-0.31`,")
    a("`step_period_rate≈0.27`. Closer to GT on these is better.")
    a("")
    a("| variant | n_walk_seg | frac_both_swing | frac_both_stance | trans/sec | L_R_corr | step_period_rate |")
    a("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for lbl in label_order:
        r = rows.get(lbl, {}).get("gait", {})
        a(
            f"| `{lbl}` | {_fmt(r.get('n_walking_segments'), 0)} | "
            f"{_fmt(r.get('frac_both_swing'), 3)} | "
            f"{_fmt(r.get('frac_both_stance'), 3)} | "
            f"{_fmt(r.get('transitions_per_sec'), 3)} | "
            f"{_fmt(r.get('L_R_height_corr'), 3)} | "
            f"{_pct(r.get('step_period_rate'))} |"
        )
    a("")

    # ---------------- Body action ----------------
    a("## Body action (mean over 48 clips)")
    a("")
    a("Lower is better for `delta_err`. Higher is better for `direction_cosine`")
    a("and for `amp_ratio` close to 1.0 (1.0 = pred matches GT amplitude;")
    a("<1 under-articulates, >1 over-articulates).")
    a("")
    a("| variant | mean delta_err (cm) | mean dir_cos | mean amp_ratio | LW delta_err | RW delta_err | LW dir_cos | RW dir_cos |")
    a("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for lbl in label_order:
        r = rows.get(lbl, {}).get("body_action", {})
        a(
            f"| `{lbl}` | "
            f"{_fmt(r.get('delta_err_cm_mean_overall'), 2)} | "
            f"{_fmt(r.get('direction_cosine_mean_overall'), 3)} | "
            f"{_fmt(r.get('amp_ratio_mean_overall'), 3)} | "
            f"{_fmt(r.get('left_wrist_delta_err_cm_mean'), 2)} | "
            f"{_fmt(r.get('right_wrist_delta_err_cm_mean'), 2)} | "
            f"{_fmt(r.get('left_wrist_direction_cosine_mean'), 3)} | "
            f"{_fmt(r.get('right_wrist_direction_cosine_mean'), 3)} |"
        )
    a("")

    # ---------------- Decision rule ----------------
    a("## Decision rule (per prompt §9.3)")
    a("")
    a(
        "1. **`no_dense_pos` survives** if its sustained-contact, gait, and "
        "body-action metrics do not regress materially against the matching "
        "A2/A3 baseline. If yes, the dense FK position MSE was redundant or "
        "harmful.")
    a(
        "2. **`relative_behavior` is promising** only if (a) sustained "
        "contact does not collapse (drift_max < ~2× baseline), (b) gait "
        "moves closer to GT reference on both-swing / transitions, (c) "
        "body-action error does not blow up, AND (d) qualitative samples "
        "remain plausible. All four checks must hold.")
    a(
        "3. **If `relative_behavior` improves gait/contact but worsens body "
        "detail**, keep it as a direction but reduce its weights or add a "
        "P1 body-refine consistency term before full-data training.")
    a(
        "4. **If both `relative_behavior` variants collapse**, do NOT "
        "conclude the idea is wrong. First sanity-check: (a) are the new "
        "consistency-loss weights too high? (b) is there a scaling bug in "
        "the new losses (e.g. clamp/normalization mismatch with the "
        "condition builder)?")
    a("")
    a(
        "**Important caveat (per analyses/2026-05-27_round29_tier2_verdict.md):** "
        "the A2/A3 a-group baselines were trained as v27 warm-start "
        "fine-tunes. If the loss-strategy ablations train from scratch, "
        "the comparison is contaminated by the training-init axis. Note "
        "in the conclusion which init regime each row used.")
    a("")
    a(
        "Do NOT pick a full-data mainline from a single scalar in this "
        "report. The point of this ablation is the strategic direction "
        "(absolute-GT vs condition-consistency), not the specific full-data "
        "mainline.")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize the 4 R29 loss-strategy ablation variants' diag "
            "results into a single comparison Markdown report."
        ),
    )
    parser.add_argument(
        "--results-root", type=Path, default=DEFAULT_RESULTS_ROOT,
        help=(
            "Root directory containing the round29_<vid>_diag_<kind>/ "
            "folders. Default: analyses/."
        ),
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_REPORT_PATH,
        help="Output Markdown path.",
    )
    args = parser.parse_args()

    rows = _gather(args.results_root)
    md = _render_report(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf-8")
    n_present = sum(
        1 for v in VARIANTS
        if any(rows[v][k] and rows[v][k].get("drift_max_mean_cm") is not None
               or rows[v][k].get("frac_both_swing") is not None
               or rows[v][k].get("delta_err_cm_mean_overall") is not None
               for k in KINDS)
    )
    n_baselines = sum(
        1 for b in BASELINE_CANDIDATES
        if any(rows[b][k] and rows[b][k].get("drift_max_mean_cm") is not None
               or rows[b][k].get("frac_both_swing") is not None
               or rows[b][k].get("delta_err_cm_mean_overall") is not None
               for k in KINDS)
    )
    print(f"wrote {args.out}")
    print(
        f"  variants found: {n_present}/{len(VARIANTS)}    "
        f"baselines found: {n_baselines}/{len(BASELINE_CANDIDATES)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
