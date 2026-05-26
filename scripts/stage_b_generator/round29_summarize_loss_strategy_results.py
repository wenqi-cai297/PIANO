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

# Six v2 variants per Codex review. Three families × A2/A3 injection.
# Order within the tuple drives report row order.
VARIANTS: tuple[str, ...] = (
    "r29_ls_a2_baseline_from_scratch",
    "r29_ls_a2_relbeh_v2_anchor0_low",
    "r29_ls_a2_relbeh_v2_anchor2_mixed",
    "r29_ls_a3_baseline_from_scratch",
    "r29_ls_a3_relbeh_v2_anchor0_low",
    "r29_ls_a3_relbeh_v2_anchor2_mixed",
)

# v2 protocol deliberately drops v27-warm-start a-group baselines from the
# comparison set: training-init confounds the loss-strategy axis (per
# analyses/2026-05-27_round29_tier2_verdict.md). The fair reference is
# now `baseline_from_scratch` inside this manifest itself.
BASELINE_CANDIDATES: dict[str, list[str]] = {}

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

    a(f"# Round-29 loss-strategy ablation report (v2 — Codex review)")
    a("")
    a(f"**Date:** {today}")
    a("**Protocol:** 48-clip balanced subset, 300 ep, FULL-DENSE C/I/S/B content,")
    a("from-scratch (no init_checkpoint) for ALL 6 variants.")
    a("")
    a("**Axis A (injection):** A2 = `adapter_only`, A3 = `input_add_adapter`.")
    a("")
    a("**Axis L (loss strategy family):**")
    a("  - `baseline_from_scratch`: original a-group losses (pos_loss=5,")
    a("    anchor_pos=10, anchor_vel=2, world_vel=1). Fair Rule-1 reference.")
    a("  - `relbeh_v2_anchor0_low`: pure low-weight condition supervision.")
    a("    anchor=0; R29 weights all 0.10; swing_clearance ON.")
    a("  - `relbeh_v2_anchor2_mixed`: weak absolute stabilizer (anchor_pos=2,")
    a("    anchor_vel=0.5) + low R29 weights; swing_clearance ON.")
    a("")
    a("All 6 variants share the same from-scratch protocol → cross-variant")
    a("deltas isolate the loss-strategy axis. v27-warm-start a-group baselines")
    a("are NOT used here (per analyses/2026-05-27_round29_tier2_verdict.md:")
    a("training-init confounds the loss-strategy comparison).")
    a("")

    label_order = (
        "r29_ls_a2_baseline_from_scratch",
        "r29_ls_a2_relbeh_v2_anchor0_low",
        "r29_ls_a2_relbeh_v2_anchor2_mixed",
        "r29_ls_a3_baseline_from_scratch",
        "r29_ls_a3_relbeh_v2_anchor0_low",
        "r29_ls_a3_relbeh_v2_anchor2_mixed",
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
    a("GT physical reference: `frac_both_swing=0.291`, `frac_both_stance=0.153`,")
    a("`trans/s=0.790`, `L_R_height_corr=-0.309`, `step_period_rate=26.6%`.")
    a("Closer to GT on these is better. **Note:** under the current support")
    a("losses (no swing/phase/footstep terms beyond `swing_clearance`),")
    a("`L_R_corr` is a diagnostic only — do not treat its sign as a pass")
    a("criterion (per Codex review).")
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
    a("## Decision rule (per Codex review §revised success criteria)")
    a("")
    a(
        "Reference for each v2 variant is the matching `baseline_from_scratch` "
        "(same injection, same init regime). Pass criteria:")
    a("")
    a(
        "1. **Contact**: `drift_max <= 1.5x baseline_from_scratch` AND "
        "`%track<0.5` not worse by more than 10 pp.")
    a(
        "2. **Gait airborne**: `both_swing` and `trans/sec` should be "
        "materially better than `baseline_from_scratch` (closer to GT).")
    a(
        "3. **Gait planted**: `both_stance < 0.25`. (v1 produced 0.44+; "
        "swing_clearance is the new term added to fight this.)")
    a(
        "4. **L_R_corr**: diagnostic only. Current support losses do NOT "
        "supervise S4 phase/footstep channels, so anti-phase coherence "
        "is not expected from these losses alone. Do not require negative "
        "L_R_corr from v2.")
    a(
        "5. **Body action**: mean delta error within ±2 cm of "
        "`baseline_from_scratch`.")
    a("")
    a(
        "Interpretation paths after the run:")
    a(
        "- If `anchor0_low` passes all five, take it to full-data — cleaner "
        "method (pure condition-consistency).")
    a(
        "- If only `anchor2_mixed` passes, take the mixed strategy to "
        "full-data and describe honestly as weak absolute stabilization + "
        "relative condition losses.")
    a(
        "- If both fail contact, fix the I3 target/mask (Codex recommended "
        "unmasking the target inside the loss or threshold hand_contact "
        "≥ 0.95) before spending full-data time.")
    a(
        "- If both pass contact but gait both_stance is still > 0.25, the "
        "swing_clearance weight may need to be raised, or a phase/footstep "
        "supervision term added before full-data.")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize the 6 R29 loss-strategy ablation variants' diag "
            "results into a single comparison Markdown report (v2 — Codex review)."
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
