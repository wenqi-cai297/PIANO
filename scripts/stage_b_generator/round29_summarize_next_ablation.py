"""Summarize Round-29 next-baseline ablation results into a Markdown report.

Per analyses/2026-05-27_round29_next_ablation_execution_prompt_for_claude_code.md
§"Files To Create" / §"Expected Decision Logic After Training":

  - Reads R0 reference stats from existing R29-FT diag dirs:
        analyses/round29_r29_ft_r0_clean_a3_baseline_diag_<kind>_<sub>/
  - Reads B0/B1/G1/G2/H1 stats from new diag dirs:
        analyses/round29_r29_nb_<variant>_diag_<kind>_<sub>/

Tables emitted (per sublabel = train | val):
  * sustained contact: overall + per-part (LH/RH/LF/RF/pelvis)
  * gait:              with GT row from gt_aggregate (mean across variants)
  * body action:       overall + per-joint (LW/RW/LK/RK/neck/pelvis)

Decision table compares with correct per-variant references:
  - B0 vs R0
  - B1 vs R0 (and vs B0)
  - G1 vs R0
  - G2 vs R0 (and vs G1)
  - H1 vs R0

Degeneration flags (per prompt): a variant is flagged if any of
  - frac_both_swing > 0.70
  - frac_both_stance < 0.02
  - transitions_per_sec < 0.40
on the val walking segments.

If R0 reference stats are missing (e.g. running before R29-FT has been
regenerated on this server), the summarizer FAILS LOUDLY with a clear
message — partial reports without R0 would mislead the verdict.
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
    ROOT / "analyses" / "2026-05-27_round29_next_ablation_report.md"
)

# 5 new train variants in display order. R0 reference (existing R29-FT) is
# loaded separately and rendered first in each table.
NEW_VARIANTS: tuple[str, ...] = (
    "r29_nb_b0_no_r29_cond",
    "r29_nb_b1_c41_only",
    "r29_nb_g1_phasefree_gait_fixed",
    "r29_nb_g2_strong_s4_oracle",
    "r29_nb_h1_r0_plus_oracle_full_hint",
)
R0_REF_VARIANT: str = "r29_ft_r0_clean_a3_baseline"
ALL_VARIANTS: tuple[str, ...] = (R0_REF_VARIANT, *NEW_VARIANTS)

SUBLABELS: tuple[str, ...] = ("train", "val")
KINDS: tuple[str, ...] = ("sustained_contact", "gait", "body_action")

PER_PART_KEYS: tuple[str, ...] = (
    "left_hand", "right_hand", "left_foot", "right_foot", "pelvis",
)

BODY_JOINT_KEYS: tuple[str, ...] = (
    "left_wrist", "right_wrist", "left_knee", "right_knee", "neck", "pelvis",
)

DECISIONS: dict[str, str] = {
    "r29_nb_b0_no_r29_cond": (
        "**B0**: does R0's 8.19 cm result depend on R29 condition families at all? "
        "Compare to R0 — if B0 ≈ R0, R29 C/I/S/B content is not load-bearing. "
        "If B0 << R0, at least one R29 condition family is load-bearing."
    ),
    "r29_nb_b1_c41_only": (
        "**B1**: is C41 alone enough to recover most of R0? "
        "Compare to R0 (and to B0) — if B1 ≈ R0 and B0 << R0, Stage-1.5 "
        "should primarily predict C41-like key-joint deltas."
    ),
    "r29_nb_g1_phasefree_gait_fixed": (
        "**G1**: can phase-free gait losses (soft-stance velocity, transition "
        "rate, duty cycle, both-state match) fix R2's degeneracy without GT "
        "phase locking? Compare to R0 — gait should improve without degeneration."
    ),
    "r29_nb_g2_strong_s4_oracle": (
        "**G2**: does strong explicit S4 schedule beat phase-free G1? "
        "Compare to R0 (and to G1) — if G2 >> G1 without contact/body "
        "regression, Stage-1.5 needs an explicit gait/footstep schedule."
    ),
    "r29_nb_h1_r0_plus_oracle_full_hint": (
        "**H1**: is current I3/I5 too weak, or is the contact bottleneck "
        "deeper than condition content? Compare to R0 — strong H1 win on "
        "hand drift/p95 ⇒ next architecture needs stronger Stage-1.5 hint."
    ),
}

# Per-variant comparison references for the decision table. R0 is the
# default reference; B1 also rendered against B0; G2 also against G1.
REFERENCE_BY_VARIANT: dict[str, str] = {
    "r29_nb_b0_no_r29_cond": R0_REF_VARIANT,
    "r29_nb_b1_c41_only": R0_REF_VARIANT,
    "r29_nb_g1_phasefree_gait_fixed": R0_REF_VARIANT,
    "r29_nb_g2_strong_s4_oracle": R0_REF_VARIANT,
    "r29_nb_h1_r0_plus_oracle_full_hint": R0_REF_VARIANT,
}
SECONDARY_REFERENCE: dict[str, str] = {
    "r29_nb_b1_c41_only": "r29_nb_b0_no_r29_cond",
    "r29_nb_g2_strong_s4_oracle": "r29_nb_g1_phasefree_gait_fixed",
}

# Degeneration thresholds (per prompt §Required Variant Matrix → G1 interpretation).
DEGEN_FRAC_BOTH_SWING_MAX: float = 0.70
DEGEN_FRAC_BOTH_STANCE_MIN: float = 0.02
DEGEN_TRANS_PER_SEC_MIN: float = 0.40


def _load_stats(
    results_root: Path, variant_id: str, kind: str, sublabel: str,
) -> dict[str, Any] | None:
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
        return {
            "drift_max_mean_cm": None, "drift_max_p95_cm": None,
            "pct_drift_max_above_5cm": None, "pct_drift_max_above_10cm": None,
            "track_frac_mean": None, "pct_track_frac_below_0.5": None,
            "n_segments": None,
            **{f"part_{k}_drift_max_mean": None for k in PER_PART_KEYS},
        }
    overall = stats.get("overall", {}) or {}
    n_seg = overall.get("n_segments")
    n_above_5 = overall.get("n_drift_max_above_5cm")
    n_above_10 = overall.get("n_drift_max_above_10cm")
    drift = overall.get("drift_max_cm", {}) or {}
    tr = overall.get("tracking_fraction", {}) or {}
    out: dict[str, Any] = {
        "drift_max_mean_cm": drift.get("mean"),
        "drift_max_p95_cm": drift.get("p95"),
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
    per_part = stats.get("per_part", {}) or {}
    for k in PER_PART_KEYS:
        part = per_part.get(k, {}) or {}
        pd = part.get("drift_max_cm", {}) or {}
        out[f"part_{k}_drift_max_mean"] = pd.get("mean")
    return out


_GAIT_FIELDS = (
    "frac_both_swing", "frac_both_stance", "transitions_per_sec",
    "L_R_height_corr", "step_period_rate", "n_walking_segments",
)


def _gait_row(stats: dict[str, Any] | None) -> dict[str, Any]:
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
    rows: dict, sublabel: str,
) -> dict[str, Any]:
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
        out: dict[str, Any] = {
            "delta_err_cm_mean_overall": None,
            "direction_cosine_mean_overall": None,
            "amp_ratio_mean_overall": None,
        }
        for j in BODY_JOINT_KEYS:
            out[f"joint_{j}_delta_err"] = None
        return out
    agg = stats.get("aggregate", {}) or {}
    joints = list(agg.keys())
    if not joints:
        return _body_action_row(None)
    de = [agg[j].get("delta_error_cm_mean") for j in joints if agg[j].get("delta_error_cm_mean") is not None]
    dc = [agg[j].get("direction_cosine_mean") for j in joints if agg[j].get("direction_cosine_mean") is not None]
    ar = [agg[j].get("amp_ratio_mean") for j in joints if agg[j].get("amp_ratio_mean") is not None]
    out: dict[str, Any] = {
        "delta_err_cm_mean_overall": (sum(de) / len(de)) if de else None,
        "direction_cosine_mean_overall": (sum(dc) / len(dc)) if dc else None,
        "amp_ratio_mean_overall": (sum(ar) / len(ar)) if ar else None,
    }
    for j in BODY_JOINT_KEYS:
        slot = agg.get(j, {}) or {}
        out[f"joint_{j}_delta_err"] = slot.get("delta_error_cm_mean")
    return out


def _gather(
    results_root: Path,
) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    out: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for v in ALL_VARIANTS:
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


def _require_r0_reference(rows: dict, results_root: Path) -> None:
    """Fail loudly if R0 reference stats are missing — partial reports
    without R0 would mislead the verdict."""
    missing: list[str] = []
    for kind in KINDS:
        for sub in SUBLABELS:
            p = (
                results_root
                / f"round29_{R0_REF_VARIANT}_diag_{kind}_{sub}"
                / f"{kind}_stats.json"
            )
            if not p.exists():
                missing.append(str(p))
    if missing:
        msg = (
            f"FATAL: R0 reference diag stats missing — cannot summarize without R0.\n"
            f"Missing files:\n"
            + "\n".join(f"  {p}" for p in missing)
            + "\n\nGenerate R0 diag stats first, e.g.:\n"
            "  bash scripts/stage_b_generator/run_round29_failure_targeted_ablations.sh \\\n"
            f"      --only {R0_REF_VARIANT} --skip-train\n"
        )
        raise FileNotFoundError(msg)


def _is_degenerate_gait(g: dict[str, Any]) -> tuple[bool, list[str]]:
    flags: list[str] = []
    fbs = g.get("frac_both_swing")
    fbst = g.get("frac_both_stance")
    tps = g.get("transitions_per_sec")
    try:
        if fbs is not None and float(fbs) > DEGEN_FRAC_BOTH_SWING_MAX:
            flags.append(f"frac_both_swing={float(fbs):.3f} > {DEGEN_FRAC_BOTH_SWING_MAX}")
    except (TypeError, ValueError):
        pass
    try:
        if fbst is not None and float(fbst) < DEGEN_FRAC_BOTH_STANCE_MIN:
            flags.append(f"frac_both_stance={float(fbst):.3f} < {DEGEN_FRAC_BOTH_STANCE_MIN}")
    except (TypeError, ValueError):
        pass
    try:
        if tps is not None and float(tps) < DEGEN_TRANS_PER_SEC_MIN:
            flags.append(f"transitions_per_sec={float(tps):.3f} < {DEGEN_TRANS_PER_SEC_MIN}")
    except (TypeError, ValueError):
        pass
    return (len(flags) > 0, flags)


def _render_section_for_sublabel(a: Any, rows: dict, sublabel: str) -> None:
    a(f"## Subset: `{sublabel}`")
    a("")
    if sublabel == "train":
        a("In-distribution sanity (same 48-clip balanced subset as the R29 FT diag).")
    else:
        a("Heldout-val 48-clip balanced subset; measures generalization.")
    a("")

    a("### Sustained contact (overall)")
    a("")
    a("| variant | n_seg | drift_max mean (cm) | drift_max p95 (cm) | %drift>5cm | %drift>10cm | track_frac mean | %track<0.5 |")
    a("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for v in ALL_VARIANTS:
        r = rows.get(v, {}).get(sublabel, {}).get("sustained_contact", {})
        a(
            f"| `{v}` | {_fmt(r.get('n_segments'), 0)} | "
            f"{_fmt(r.get('drift_max_mean_cm'), 2)} | "
            f"{_fmt(r.get('drift_max_p95_cm'), 2)} | "
            f"{_pct(r.get('pct_drift_max_above_5cm'))} | "
            f"{_pct(r.get('pct_drift_max_above_10cm'))} | "
            f"{_fmt(r.get('track_frac_mean'), 3)} | "
            f"{_pct(r.get('pct_track_frac_below_0.5'))} |"
        )
    a("")

    a("### Sustained contact (per part — drift_max mean, cm)")
    a("")
    a("| variant | left_hand | right_hand | left_foot | right_foot | pelvis |")
    a("| --- | ---: | ---: | ---: | ---: | ---: |")
    for v in ALL_VARIANTS:
        r = rows.get(v, {}).get(sublabel, {}).get("sustained_contact", {})
        a(
            f"| `{v}` | "
            f"{_fmt(r.get('part_left_hand_drift_max_mean'), 2)} | "
            f"{_fmt(r.get('part_right_hand_drift_max_mean'), 2)} | "
            f"{_fmt(r.get('part_left_foot_drift_max_mean'), 2)} | "
            f"{_fmt(r.get('part_right_foot_drift_max_mean'), 2)} | "
            f"{_fmt(r.get('part_pelvis_drift_max_mean'), 2)} |"
        )
    a("")

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
    for v in ALL_VARIANTS:
        r = rows.get(v, {}).get(sublabel, {}).get("gait", {})
        a(
            f"| `{v}` | {_fmt(r.get('n_walking_segments'), 0)} | "
            f"{_fmt(r.get('frac_both_swing'), 3)} | "
            f"{_fmt(r.get('frac_both_stance'), 3)} | "
            f"{_fmt(r.get('transitions_per_sec'), 3)} | "
            f"{_fmt(r.get('L_R_height_corr'), 3)} | "
            f"{_pct(r.get('step_period_rate'))} |"
        )
    a("")

    a("### Body action (overall mean over reported joints)")
    a("")
    a("| variant | mean delta_err (cm) | mean dir_cos | mean amp_ratio |")
    a("| --- | ---: | ---: | ---: |")
    for v in ALL_VARIANTS:
        r = rows.get(v, {}).get(sublabel, {}).get("body_action", {})
        a(
            f"| `{v}` | "
            f"{_fmt(r.get('delta_err_cm_mean_overall'), 2)} | "
            f"{_fmt(r.get('direction_cosine_mean_overall'), 3)} | "
            f"{_fmt(r.get('amp_ratio_mean_overall'), 3)} |"
        )
    a("")
    a("### Body action (per-joint delta_err, cm)")
    a("")
    header = "| variant |"
    sep = "| --- |"
    for j in BODY_JOINT_KEYS:
        header += f" {j} |"
        sep += " ---: |"
    a(header)
    a(sep)
    for v in ALL_VARIANTS:
        r = rows.get(v, {}).get(sublabel, {}).get("body_action", {})
        row = f"| `{v}` |"
        for j in BODY_JOINT_KEYS:
            row += f" {_fmt(r.get(f'joint_{j}_delta_err'), 2)} |"
        a(row)
    a("")


def _decision_table(a: Any, rows: dict) -> None:
    a("## Automatic decision table")
    a("")
    a("Compares each new variant (B0/B1/G1/G2/H1) against R0 (and against a")
    a("secondary reference where relevant: B1 vs B0, G2 vs G1). Use this as a")
    a("quick read; combine with the per-part + p95 + body tables above before")
    a("committing to a next mainline.")
    a("")
    a("| variant | reference | val drift_max delta (cm) | val %track<0.5 delta | val L_R_corr | val step_period_rate | val body delta_err delta |")
    a("| --- | --- | ---: | ---: | ---: | ---: | ---: |")

    def _val(v: str, kind: str, key: str) -> float | None:
        row = rows.get(v, {}).get("val", {}).get(kind, {})
        x = row.get(key)
        try:
            return float(x) if x is not None else None
        except (TypeError, ValueError):
            return None

    def _delta(x: float | None, base: float | None) -> str:
        if x is None or base is None:
            return "-"
        return f"{x - base:+.2f}"

    def _delta_pct(x: float | None, base: float | None) -> str:
        if x is None or base is None:
            return "-"
        return f"{100.0 * (x - base):+.1f} pp"

    def _emit_row(v: str, ref: str) -> None:
        drift = _val(v, "sustained_contact", "drift_max_mean_cm")
        track = _val(v, "sustained_contact", "pct_track_frac_below_0.5")
        lr = _val(v, "gait", "L_R_height_corr")
        step_rate = _val(v, "gait", "step_period_rate")
        body = _val(v, "body_action", "delta_err_cm_mean_overall")
        ref_drift = _val(ref, "sustained_contact", "drift_max_mean_cm")
        ref_track = _val(ref, "sustained_contact", "pct_track_frac_below_0.5")
        ref_body = _val(ref, "body_action", "delta_err_cm_mean_overall")
        a(
            f"| `{v}` | `{ref}` | {_delta(drift, ref_drift)} | "
            f"{_delta_pct(track, ref_track)} | "
            f"{_fmt(lr, 3)} | {_pct(step_rate)} | "
            f"{_delta(body, ref_body)} |"
        )

    for v in NEW_VARIANTS:
        primary = REFERENCE_BY_VARIANT[v]
        _emit_row(v, primary)
        if v in SECONDARY_REFERENCE:
            _emit_row(v, SECONDARY_REFERENCE[v])
    a("")

    # Degeneration flags on val.
    a("### Gait degeneration flags (val walking segments)")
    a("")
    a("Flagged when any of: `frac_both_swing > 0.70`, `frac_both_stance < 0.02`,")
    a("`transitions_per_sec < 0.40`. (R2 degenerated on these thresholds; G1/G2")
    a("are designed to avoid it.)")
    a("")
    any_flag = False
    for v in NEW_VARIANTS:
        g = rows.get(v, {}).get("val", {}).get("gait", {})
        degen, flags = _is_degenerate_gait(g)
        if degen:
            any_flag = True
            a(f"- ⚠️  `{v}`: {'; '.join(flags)}")
    if not any_flag:
        a("- (none — all new variants pass the degeneration thresholds.)")
    a("")

    a("### Per-variant decision questions")
    a("")
    for v in NEW_VARIANTS:
        if v in DECISIONS:
            a(f"- {DECISIONS[v]}")
    a("")


def _render_report(rows: dict) -> str:
    today = date.today().isoformat()
    lines: list[str] = []
    a = lines.append

    a("# Round-29 next-baseline ablation report")
    a("")
    a(f"**Date:** {today}")
    a("**Protocol:** FULL InterAct train set, 80 ep, heldout val,")
    a("from-scratch (no init_checkpoint), bs=32 / accum=1 (2× 5080).")
    a("R0 reference is the existing R29-FT clean baseline; B0/B1/G1/G2/H1")
    a("are the 5 new train variants in this matrix.")
    a("")
    a("Six rows shown:")
    a("")
    a(f"- **R0 (ref)** `{R0_REF_VARIANT}` — existing R29-FT clean baseline.")
    a("- **B0** `r29_nb_b0_no_r29_cond` — no R29 C/I/S/B injection.")
    a("- **B1** `r29_nb_b1_c41_only` — only C41 extra (dim=18).")
    a("- **G1** `r29_nb_g1_phasefree_gait_fixed` — phase-free gait losses.")
    a("- **G2** `r29_nb_g2_strong_s4_oracle` — strong S4 execution losses.")
    a("- **H1** `r29_nb_h1_r0_plus_oracle_full_hint` — R0 + oracle hint (full, dim=13).")
    a("")
    a("Two diag subsets per variant:")
    a("- **`train`**: 48-clip balanced (in-distribution, same as R29 FT).")
    a("- **`val`**: 48-clip heldout-val balanced (generalization).")
    a("")

    for sublabel in SUBLABELS:
        _render_section_for_sublabel(a, rows, sublabel)

    _decision_table(a, rows)

    a("## Mainline-selection guidance (per prompt §\"Expected Decision Logic\")")
    a("")
    a("**Condition content (B0 / B1 vs R0):**")
    a("- B0 ≈ R0 → R29 C/I/S/B content is not the main reason R0 works.")
    a("- B0 << R0 and B1 ≈ R0 → C41 is the dominant load-bearing condition;")
    a("  Stage-1.5 should primarily predict C41-like key-joint deltas.")
    a("- B1 << R0 → I/S/B also carry useful information; Stage-1.5 output")
    a("  design must include more than C41.")
    a("")
    a("**Gait execution (G1 / G2 vs R0):**")
    a("- G1 succeeds iff it improves gait metrics WITHOUT degeneracy AND")
    a("  WITHOUT contact/body regression.")
    a("- G2 beats G1 ⇒ explicit S4 stance/footstep schedule is required.")
    a("- If both G1 and G2 fail, the bottleneck is not just loss weights —")
    a("  consider injection mode, model capacity, or motion representation.")
    a("")
    a("**Contact execution (H1 vs R0):**")
    a("- H1 strong hand-contact win → current I3/I5 is too weak; next")
    a("  architecture needs a stronger Stage-1.5 contact planner / hint.")
    a("- H1 no win → do not keep multiplying I-condition variants; the")
    a("  bottleneck is likely decoder/motion-representation/absolute objective.")
    a("")
    a("Do not pick a winner from one scalar.")
    a("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize Round-29 next-baseline ablation (B0/B1/G1/G2/H1) "
            "against R0 reference; emits Markdown report."
        ),
    )
    parser.add_argument(
        "--results-root", default=str(DEFAULT_RESULTS_ROOT),
        help="Directory containing round29_<variant>_diag_<kind>_<sub>/ "
             "subdirectories with per-kind *_stats.json files.",
    )
    parser.add_argument(
        "--out", default=str(DEFAULT_REPORT_PATH),
        help="Output Markdown path.",
    )
    parser.add_argument(
        "--allow-missing-r0", action="store_true",
        help="(Testing only) skip the R0-reference existence check.",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root)
    rows = _gather(results_root)
    if not args.allow_missing_r0:
        _require_r0_reference(rows, results_root)
    report = _render_report(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
