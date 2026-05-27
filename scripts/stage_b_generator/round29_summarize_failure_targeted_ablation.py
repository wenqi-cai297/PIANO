"""Summarize Round-29 failure-targeted ablation results into a Markdown report.

Per analyses/2026-05-27_round29_failure_targeted_ablation_prompt_for_claude_code.md §4.3
and §7. Reads 6 full-data variants' diag stats from
``analyses/round29_<variant_id>_diag_<kind>_<sublabel>/<kind>_stats.json``
(launcher emits 6 × 3 × 2 = 36 files: 6 variants × 3 diag kinds × 2 selection
buckets), then writes a comparison Markdown report.

Per-variant metrics surfaced:

* sustained contact: drift_max mean + p95, %drift>5cm, %drift>10cm,
                     tracking_fraction mean + %track<0.5, per-part drift
                     (left_hand / right_hand / left_foot / right_foot /
                     pelvis if available)
* gait:              both_swing, both_stance, trans/s, L_R_corr,
                     step_period_rate (with per-subset GT reference row)
* body action:       mean delta_err, dir_cos, amp_ratio, LW + RW wrist err,
                     plus knee/neck/pelvis if present

Decision table compares R1-R5 against R0 on the question each variant
was designed to answer.
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
    ROOT / "analyses" / "2026-05-27_round29_failure_targeted_ablation_report.md"
)

# 6 failure-targeted variants in display order.
VARIANTS: tuple[str, ...] = (
    "r29_ft_r0_clean_a3_baseline",
    "r29_ft_r1_no_coarse_extra",
    "r29_ft_r2_behavior_gait_loss",
    "r29_ft_r3_oracle_s4_gait_loss",
    "r29_ft_r4_i3_contact_lock",
    "r29_ft_r5_allpart_interaction_lock",
)
SUBLABELS: tuple[str, ...] = ("train", "val")
KINDS: tuple[str, ...] = ("sustained_contact", "gait", "body_action")

# Per-part contact joints (matches diag stats per_part keys).
PER_PART_KEYS: tuple[str, ...] = (
    "left_hand", "right_hand", "left_foot", "right_foot", "pelvis",
)

# Body-action joints to surface (in display order).
BODY_JOINT_KEYS: tuple[str, ...] = (
    "left_wrist", "right_wrist", "left_knee", "right_knee", "neck", "pelvis",
)

# Decision-question text per variant (paired with R0 contrast).
DECISIONS: dict[str, str] = {
    "r29_ft_r1_no_coarse_extra": (
        "**R1**: is C41 extra (18-D coarse-extra) load-bearing? "
        "Compare to R0 — if within ~5% on contact/gait/body, C41 extra is "
        "not load-bearing → defer Stage-1 Coarse-v2."
    ),
    "r29_ft_r2_behavior_gait_loss": (
        "**R2**: does behavior-level gait loss fix walking without GT phase "
        "locking? Compare to R0 — improvement on L_R_corr / step_period_rate "
        "without contact/body regression ⇒ R2 is the next gait mainline."
    ),
    "r29_ft_r3_oracle_s4_gait_loss": (
        "**R3**: does exact S4 schedule execution beat behavior-level gait? "
        "Compare to R2 — if R3 >> R2, Stage-1.5 needs an explicit gait/"
        "phase/footstep schedule; if R2 ~= R3, prefer R2 (multimodal-safe)."
    ),
    "r29_ft_r4_i3_contact_lock": (
        "**R4**: does contact-lock loss fix hand drift with current I3? "
        "Compare to R0 — if R4 improves hand drift_max without hurting "
        "gait/body, contact-lock is a mainline component."
    ),
    "r29_ft_r5_allpart_interaction_lock": (
        "**R5**: does all-part interaction (I5) beat hands-only I3? "
        "Compare to R4 — if R5 >> R4 on feet/pelvis or hand-foot-mixed, "
        "upgrade interaction condition. If R4 ~= R5, I3 was enough."
    ),
}


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
    """Overall + per-part sustained-contact metrics."""
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


def _gather(results_root: Path) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
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


def _render_section_for_sublabel(a: Any, rows: dict, sublabel: str) -> None:
    a(f"## Subset: `{sublabel}`")
    a("")
    if sublabel == "train":
        a("In-distribution sanity (same 48-clip balanced subset as the R29 LSF diag).")
    else:
        a("Heldout-val 48-clip balanced subset; measures generalization.")
    a("")

    # Sustained contact — overall.
    a("### Sustained contact (overall)")
    a("")
    a("| variant | n_seg | drift_max mean (cm) | drift_max p95 (cm) | %drift>5cm | %drift>10cm | track_frac mean | %track<0.5 |")
    a("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for v in VARIANTS:
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

    # Sustained contact — per part.
    a("### Sustained contact (per part — drift_max mean, cm)")
    a("")
    a("| variant | left_hand | right_hand | left_foot | right_foot | pelvis |")
    a("| --- | ---: | ---: | ---: | ---: | ---: |")
    for v in VARIANTS:
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

    # Gait.
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
    for v in VARIANTS:
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

    # Body action — overall + per joint.
    a("### Body action (overall mean over reported joints)")
    a("")
    a("| variant | mean delta_err (cm) | mean dir_cos | mean amp_ratio |")
    a("| --- | ---: | ---: | ---: |")
    for v in VARIANTS:
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
    for v in VARIANTS:
        r = rows.get(v, {}).get(sublabel, {}).get("body_action", {})
        row = f"| `{v}` |"
        for j in BODY_JOINT_KEYS:
            row += f" {_fmt(r.get(f'joint_{j}_delta_err'), 2)} |"
        a(row)
    a("")


def _decision_table(a: Any, rows: dict) -> None:
    a("## Automatic decision table")
    a("")
    a("Compares each non-R0 variant against R0 on val-subset headline metrics.")
    a("Use this as a quick read; combine with the per-part + p95 + body tables")
    a("above before committing to a next mainline.")
    a("")
    a("| variant | val drift_max mean (cm) Δ vs R0 | val %track<0.5 Δ vs R0 | val L_R_corr | val step_period_rate | val body delta_err Δ vs R0 |")
    a("| --- | ---: | ---: | ---: | ---: | ---: |")

    def _val(v: str, kind: str, key: str) -> float | None:
        row = rows.get(v, {}).get("val", {}).get(kind, {})
        x = row.get(key)
        try:
            return float(x) if x is not None else None
        except (TypeError, ValueError):
            return None

    r0_drift = _val("r29_ft_r0_clean_a3_baseline", "sustained_contact", "drift_max_mean_cm")
    r0_track = _val("r29_ft_r0_clean_a3_baseline", "sustained_contact", "pct_track_frac_below_0.5")
    r0_body = _val("r29_ft_r0_clean_a3_baseline", "body_action", "delta_err_cm_mean_overall")

    def _delta(x: float | None, base: float | None) -> str:
        if x is None or base is None:
            return "-"
        return f"{x - base:+.2f}"

    def _delta_pct(x: float | None, base: float | None) -> str:
        if x is None or base is None:
            return "-"
        return f"{100.0 * (x - base):+.1f} pp"

    for v in VARIANTS:
        if v == "r29_ft_r0_clean_a3_baseline":
            continue
        drift = _val(v, "sustained_contact", "drift_max_mean_cm")
        track = _val(v, "sustained_contact", "pct_track_frac_below_0.5")
        lr = _val(v, "gait", "L_R_height_corr")
        step_rate = _val(v, "gait", "step_period_rate")
        body = _val(v, "body_action", "delta_err_cm_mean_overall")
        a(
            f"| `{v}` | {_delta(drift, r0_drift)} | "
            f"{_delta_pct(track, r0_track)} | "
            f"{_fmt(lr, 3)} | {_pct(step_rate)} | "
            f"{_delta(body, r0_body)} |"
        )
    a("")
    a("### Per-variant decision questions")
    a("")
    for v in VARIANTS:
        if v in DECISIONS:
            a(f"- {DECISIONS[v]}")
    a("")


def _render_report(rows: dict) -> str:
    today = date.today().isoformat()
    lines: list[str] = []
    a = lines.append

    a("# Round-29 failure-targeted ablation report")
    a("")
    a(f"**Date:** {today}")
    a("**Protocol:** FULL InterAct train set, 80 ep, heldout val,")
    a("from-scratch (no init_checkpoint), bs=32 / accum=1 (3× 5080).")
    a("Base condition: C41 + I3 + S4 + B4 + input_add_adapter (matches the")
    a("closed R29 winner). Per-variant deltas listed in §Decision table.")
    a("")
    a("Six-variant matrix:")
    a("")
    a("- **R0** `r29_ft_r0_clean_a3_baseline` — clean patched rerun; reference.")
    a("- **R1** `r29_ft_r1_no_coarse_extra` — C23 (no C41 extra) → ablate.")
    a("- **R2** `r29_ft_r2_behavior_gait_loss` — behavior gait (no GT phase).")
    a("- **R3** `r29_ft_r3_oracle_s4_gait_loss` — exact S4 stance BCE + footstep.")
    a("- **R4** `r29_ft_r4_i3_contact_lock` — contact-lock on I3 hands-only.")
    a("- **R5** `r29_ft_r5_allpart_interaction_lock` — I5 all-part + contact-lock.")
    a("")
    a("Two diag subsets per variant:")
    a("- **`train`**: 48-clip balanced (in-distribution, same as R29 LSF).")
    a("- **`val`**: 48-clip heldout-val balanced (generalization).")
    a("")

    for sublabel in SUBLABELS:
        _render_section_for_sublabel(a, rows, sublabel)

    _decision_table(a, rows)

    a("## Mainline-selection guidance (per prompt §7.5)")
    a("")
    a("Do not pick the next mainline from one scalar. Combine:")
    a("")
    a("- drift_max mean + p95")
    a("- %drift > 10 cm")
    a("- tracking_fraction mean + %track<0.5")
    a("- L_R_corr (closer to GT, typically negative)")
    a("- step_period_rate")
    a("- frac_both_swing / frac_both_stance")
    a("- body_action mean delta_err")
    a("- wrist + ankle per-joint errors where available")
    a("- visual review on the same curated clips")
    a("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize Round-29 failure-targeted ablation results into a "
            "Markdown report comparing R0-R5 across contact / gait / body."
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
    args = parser.parse_args()

    rows = _gather(Path(args.results_root))
    report = _render_report(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
