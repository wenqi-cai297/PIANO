"""Summarize Round-29 next-step ablation results into a Markdown report.

Per analyses/2026-05-28_round29_next_step_ablation_execution_prompt_for_claude_code.md §7.

Reads:
  - R0/B1/G1 reference stats from existing R29-FT / R29-NB diag dirs:
        analyses/round29_<variant>_diag_<kind>_<bucket>/<kind>_stats.json
  - 4 new train variants (A0/A1/H1/A2) from:
        analyses/round29_r29_ns_<variant>_diag_<kind>_<bucket>/<kind>_stats.json
  - G1 soft-stance diag from:
        analyses/round29_r29_ns_<variant>_diag_g1_soft_stance_<bucket>/g1_soft_stance_stats.json
  - Motion-repr-floor diag from:
        analyses/round29_repr_floor_<bucket>/repr_floor_stats.json
  - INVALID old H1 (if present on disk): explicitly marked invalid in the
    report — NOT used as a valid decision reference.

Tables emitted (per bucket = train|val):
  1. Sustained contact overall / per-part
  2. Gait with GT row from gt_aggregate
  3. Body action overall / per-joint
  4. Paired bootstrap (val only) — A0 vs B1, A0 vs G1, A1 vs A0, A1 vs G1,
     H1 vs R0 — matched by (subset, seq_id, part_name, t0, t1) for
     sustained-contact and (subset, seq_id, t0, t1) for gait.
  5. G1 soft-stance degeneracy table (A0/A1/G1/A2)
  6. Representation floor table + verdict

Decision text:
  - A0 mainline verdict
  - A1 support-condition verdict
  - H1 contact-content verdict (only if H1 is valid + data supports it)

Fail-closed: missing R0/B1/G1 stats fails by default. ``--allow-partial``
renders an incomplete report.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = ROOT / "analyses"
DEFAULT_REPORT_PATH = (
    ROOT / "analyses" / "2026-05-28_round29_next_step_ablation_report.md"
)

# Variants in display order.
NEW_VARIANTS: tuple[str, ...] = (
    "r29_ns_a0_c41_g1_loss_s4",
    "r29_ns_a1_c41_s4_g1",
    "r29_ns_h1_i5_upper_bound",
    "r29_ns_a2_c41_i5_g1",
)
REFERENCE_VARIANTS: tuple[str, ...] = (
    "r29_ft_r0_clean_a3_baseline",
    "r29_nb_b1_c41_only",
    "r29_nb_g1_phasefree_gait_fixed",
)
INVALID_OLD_VARIANT: str = "r29_nb_h1_r0_plus_oracle_full_hint"

ALL_VARIANTS: tuple[str, ...] = (*REFERENCE_VARIANTS, *NEW_VARIANTS)
SUBLABELS: tuple[str, ...] = ("train", "val")
KINDS: tuple[str, ...] = ("sustained_contact", "gait", "body_action")

PER_PART_KEYS: tuple[str, ...] = (
    "left_hand", "right_hand", "left_foot", "right_foot", "pelvis",
)
BODY_JOINT_KEYS: tuple[str, ...] = (
    "left_wrist", "right_wrist", "left_knee", "right_knee", "neck", "pelvis",
)

# Pair comparisons for the decision table (paired bootstrap, val only).
PAIRED_COMPARISONS: tuple[tuple[str, str, str], ...] = (
    # (variant, reference, descriptor)
    ("r29_ns_a0_c41_g1_loss_s4", "r29_nb_b1_c41_only", "A0 vs B1 (cond)"),
    ("r29_ns_a0_c41_g1_loss_s4", "r29_nb_g1_phasefree_gait_fixed", "A0 vs G1 (gait)"),
    ("r29_ns_a1_c41_s4_g1", "r29_ns_a0_c41_g1_loss_s4", "A1 vs A0 (S4 cond?)"),
    ("r29_ns_a1_c41_s4_g1", "r29_nb_g1_phasefree_gait_fixed", "A1 vs G1 (gait)"),
    ("r29_ns_h1_i5_upper_bound", "r29_ft_r0_clean_a3_baseline", "H1 vs R0 (I5 cond)"),
    ("r29_ns_a2_c41_i5_g1", "r29_ns_a0_c41_g1_loss_s4", "A2 vs A0 (I5 add)"),
    ("r29_ns_a2_c41_i5_g1", "r29_ns_h1_i5_upper_bound", "A2 vs H1 (G1 add)"),
)

# Degeneracy thresholds.
DEGEN_FRAC_BOTH_SWING_MAX: float = 0.70
DEGEN_FRAC_BOTH_STANCE_MIN: float = 0.02
DEGEN_TRANS_PER_SEC_MIN: float = 0.40


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

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


def _load_g1_soft_stats(
    results_root: Path, variant_id: str, sublabel: str,
) -> dict[str, Any] | None:
    p = (
        results_root
        / f"round29_{variant_id}_diag_g1_soft_stance_{sublabel}"
        / "g1_soft_stance_stats.json"
    )
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _load_repr_floor(
    results_root: Path, sublabel: str,
) -> dict[str, Any] | None:
    p = results_root / f"round29_repr_floor_{sublabel}" / "repr_floor_stats.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
# Headline-stats rows (re-uses the schema from round29_summarize_failure_targeted_ablation.py).
# --------------------------------------------------------------------------- #

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


def _gait_row(stats: dict[str, Any] | None) -> dict[str, Any]:
    fields = (
        "frac_both_swing", "frac_both_stance", "transitions_per_sec",
        "L_R_height_corr", "step_period_rate", "n_walking_segments",
    )
    if not stats:
        out = {k: None for k in fields}
        out.update({f"gt_{k}": None for k in fields})
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


def _subset_gt_reference(rows: dict, sublabel: str) -> dict[str, Any]:
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


# --------------------------------------------------------------------------- #
# Paired bootstrap CI
# --------------------------------------------------------------------------- #

def _per_segment_rows_sustained(
    stats: dict[str, Any] | None,
) -> dict[tuple[str, str, str, int, int], float]:
    """Index per-segment drift_max_cm by (subset, seq_id, part_name, t0, t1)."""
    out: dict[tuple[str, str, str, int, int], float] = {}
    if not stats:
        return out
    for r in stats.get("rows", []) or []:
        key = (
            str(r.get("subset", "")),
            str(r.get("seq_id", "")),
            str(r.get("part_name", "")),
            int(r.get("t0", -1)),
            int(r.get("t1", -1)),
        )
        v = r.get("drift_max_cm")
        if v is None:
            continue
        try:
            out[key] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def _per_segment_rows_gait(
    stats: dict[str, Any] | None,
) -> dict[tuple[str, str, int, int], dict[str, float]]:
    """Index gait per-segment pred metrics by (subset, seq_id, t0, t1)."""
    out: dict[tuple[str, str, int, int], dict[str, float]] = {}
    if not stats:
        return out
    for r in stats.get("per_segment", []) or []:
        key = (
            str(r.get("subset", "")),
            str(r.get("seq_id", "")),
            int(r.get("t0", -1)),
            int(r.get("t1", -1)),
        )
        pred = r.get("pred", {}) or {}
        slot: dict[str, float] = {}
        for fld in ("frac_both_swing", "frac_both_stance",
                    "transitions_per_second", "L_R_height_corr"):
            v = pred.get(fld)
            if v is None:
                continue
            try:
                slot[fld] = float(v)
            except (TypeError, ValueError):
                continue
        if slot:
            out[key] = slot
    return out


def _bootstrap_ci(
    deltas: list[float], *,
    n_bootstrap: int = 2000,
    seed: int = 42,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Paired-difference bootstrap of the mean. Returns (mean, lo, hi)."""
    if not deltas:
        return (float("nan"), float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(deltas)
    mean_d = sum(deltas) / n
    means: list[float] = []
    for _ in range(n_bootstrap):
        s = 0.0
        for _i in range(n):
            s += deltas[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo_idx = int((1.0 - ci) / 2.0 * n_bootstrap)
    hi_idx = int((1.0 + ci) / 2.0 * n_bootstrap) - 1
    lo_idx = max(0, min(lo_idx, n_bootstrap - 1))
    hi_idx = max(0, min(hi_idx, n_bootstrap - 1))
    return (mean_d, means[lo_idx], means[hi_idx])


def _paired_drift_delta(
    variant_stats: dict[str, Any] | None,
    ref_stats: dict[str, Any] | None,
) -> list[float]:
    a = _per_segment_rows_sustained(variant_stats)
    b = _per_segment_rows_sustained(ref_stats)
    keys = set(a) & set(b)
    return [a[k] - b[k] for k in keys]


def _paired_gait_delta(
    variant_stats: dict[str, Any] | None,
    ref_stats: dict[str, Any] | None,
    field: str,
) -> list[float]:
    a = _per_segment_rows_gait(variant_stats)
    b = _per_segment_rows_gait(ref_stats)
    keys = set(a) & set(b)
    deltas = []
    for k in keys:
        if field not in a[k] or field not in b[k]:
            continue
        deltas.append(a[k][field] - b[k][field])
    return deltas


# --------------------------------------------------------------------------- #
# Gathering
# --------------------------------------------------------------------------- #

def _gather(
    results_root: Path,
) -> tuple[
    dict[str, dict[str, dict[str, dict[str, Any]]]],  # rows: variant -> sub -> kind -> stats
    dict[str, dict[str, dict[str, Any] | None]],       # raw sustained_contact + gait stats by variant + sub
    dict[str, dict[str, dict[str, Any] | None]],       # g1 soft stats
    dict[str, dict[str, Any] | None],                  # repr floor
]:
    rows: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    raw: dict[str, dict[str, dict[str, Any] | None]] = {}
    g1_soft: dict[str, dict[str, dict[str, Any] | None]] = {}
    for v in ALL_VARIANTS:
        rows[v] = {}
        raw[v] = {}
        for sub in SUBLABELS:
            sc_stats = _load_stats(results_root, v, "sustained_contact", sub)
            gait_stats = _load_stats(results_root, v, "gait", sub)
            ba_stats = _load_stats(results_root, v, "body_action", sub)
            rows[v][sub] = {
                "sustained_contact": _sustained_row(sc_stats),
                "gait": _gait_row(gait_stats),
                "body_action": _body_action_row(ba_stats),
            }
            raw[v][f"sustained_contact_{sub}"] = sc_stats
            raw[v][f"gait_{sub}"] = gait_stats
    for v in NEW_VARIANTS:
        g1_soft[v] = {}
        for sub in SUBLABELS:
            g1_soft[v][sub] = _load_g1_soft_stats(results_root, v, sub)
    # Also include G1 reference soft-stance if present (helps to anchor A0/A1/A2 reading).
    g1_soft[REFERENCE_VARIANTS[2]] = {  # G1 ref
        sub: _load_g1_soft_stats(results_root, REFERENCE_VARIANTS[2], sub)
        for sub in SUBLABELS
    }
    repr_floor = {sub: _load_repr_floor(results_root, sub) for sub in SUBLABELS}
    return rows, raw, g1_soft, repr_floor


def _missing_required_stats(
    rows: dict, raw: dict, g1_soft: dict, repr_floor: dict,
) -> list[str]:
    """Check that ALL decision-critical stats are present.

    Per Codex 2026-05-28 fix prompt §2, the summarizer must fail by default
    if any of the following are missing or invalid:

      (1) Standard stats (sustained_contact / gait / body_action) for R0,
          B1, G1, A0, A1, H1, A2 on train + val buckets.
      (2) Representation floor stats for train + val, each with an
          `interpretation.verdict`.
      (3) G1 soft-stance for the G1 reference and for A0/A1/A2 on train +
          val, each with `soft_aggregate.n_segments > 0` and all required
          fields.
      (4) Paired bootstrap: every required comparison in
          `PAIRED_COMPARISONS` has `n_paired > 0` for both sustained-
          contact drift and gait (L_R_height_corr + frac_both_swing).

    Returns a list of human-readable missing-slot identifiers. Empty list
    means everything is present.
    """
    missing: list[str] = []

    # --- (1) Standard stats for references + new variants.
    for v in REFERENCE_VARIANTS:
        for sub in SUBLABELS:
            for kind in KINDS:
                key = ("n_segments" if kind == "sustained_contact"
                       else "n_walking_segments" if kind == "gait"
                       else "delta_err_cm_mean_overall")
                if rows.get(v, {}).get(sub, {}).get(kind, {}).get(key) is None:
                    missing.append(f"standard:{v}/{kind}_{sub}")
    for v in NEW_VARIANTS:
        for sub in SUBLABELS:
            for kind in KINDS:
                slot = rows.get(v, {}).get(sub, {}).get(kind, {})
                key = ("n_segments" if kind == "sustained_contact"
                       else "n_walking_segments" if kind == "gait"
                       else "delta_err_cm_mean_overall")
                if slot.get(key) is None:
                    missing.append(f"standard:{v}/{kind}_{sub}")

    # --- (2) Representation floor stats, each with interpretation.verdict.
    for sub in SUBLABELS:
        floor = repr_floor.get(sub)
        if floor is None:
            missing.append(f"repr_floor:{sub} (missing stats JSON)")
            continue
        verdict = (floor.get("interpretation") or {}).get("verdict")
        if not verdict:
            missing.append(f"repr_floor:{sub} (missing interpretation.verdict)")

    # --- (3) G1 soft-stance for G1 ref + A0/A1/A2 (variants with G1 losses).
    g1_required_variants = (
        "r29_nb_g1_phasefree_gait_fixed",   # G1 ref
        "r29_ns_a0_c41_g1_loss_s4",
        "r29_ns_a1_c41_s4_g1",
        "r29_ns_a2_c41_i5_g1",
    )
    g1_required_fields = (
        "constant_mid_rate", "soft_alt_std",
        "soft_transition_density", "gt_transition_density",
        "low_alt_amplitude_rate", "low_transition_rate",
    )
    for v in g1_required_variants:
        for sub in SUBLABELS:
            stats = (g1_soft.get(v, {}) or {}).get(sub)
            if stats is None:
                missing.append(f"g1_soft:{v}/{sub} (missing stats JSON)")
                continue
            agg = stats.get("soft_aggregate") or {}
            n_seg = agg.get("n_segments", 0)
            if not n_seg or int(n_seg) <= 0:
                missing.append(f"g1_soft:{v}/{sub} (soft_aggregate.n_segments=0)")
                continue
            for fld in g1_required_fields:
                if fld not in agg or agg[fld] is None:
                    missing.append(f"g1_soft:{v}/{sub} (missing field {fld})")
                    break

    # --- (4) Paired bootstrap matching for every required comparison.
    for variant, ref, label in PAIRED_COMPARISONS:
        # Sustained-contact drift.
        sc_deltas = _paired_drift_delta(
            raw.get(variant, {}).get("sustained_contact_val"),
            raw.get(ref, {}).get("sustained_contact_val"),
        )
        if not sc_deltas:
            missing.append(
                f"paired_bootstrap:{label}/sustained_contact (n_paired=0)"
            )
        # Gait L_R_height_corr.
        gait_lr_deltas = _paired_gait_delta(
            raw.get(variant, {}).get("gait_val"),
            raw.get(ref, {}).get("gait_val"),
            "L_R_height_corr",
        )
        if not gait_lr_deltas:
            missing.append(
                f"paired_bootstrap:{label}/gait_L_R_corr (n_paired=0)"
            )
        # Gait frac_both_swing.
        gait_bs_deltas = _paired_gait_delta(
            raw.get(variant, {}).get("gait_val"),
            raw.get(ref, {}).get("gait_val"),
            "frac_both_swing",
        )
        if not gait_bs_deltas:
            missing.append(
                f"paired_bootstrap:{label}/gait_both_swing (n_paired=0)"
            )

    return missing


# --------------------------------------------------------------------------- #
# Markdown rendering
# --------------------------------------------------------------------------- #

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
    header = "| variant |"; sep = "| --- |"
    for j in BODY_JOINT_KEYS:
        header += f" {j} |"; sep += " ---: |"
    a(header); a(sep)
    for v in ALL_VARIANTS:
        r = rows.get(v, {}).get(sublabel, {}).get("body_action", {})
        row = f"| `{v}` |"
        for j in BODY_JOINT_KEYS:
            row += f" {_fmt(r.get(f'joint_{j}_delta_err'), 2)} |"
        a(row)
    a("")


def _render_paired_bootstrap(a: Any, raw: dict, *, seed: int = 42) -> None:
    a("## Paired bootstrap CI (val)")
    a("")
    a("Per-segment paired differences matched by `(subset, seq_id, part_name, "
      "t0, t1)` for sustained contact and `(subset, seq_id, t0, t1)` for gait. "
      "Mean and 95% bootstrap CI (n_boot=2000). A CI that excludes zero is a "
      "significant difference; a CI containing zero is consistent with noise.")
    a("")
    a("### Sustained contact — drift_max_cm delta")
    a("")
    a("| comparison | n_paired | mean Δ (cm) | 95% CI |")
    a("| --- | ---: | ---: | --- |")
    for variant, ref, label in PAIRED_COMPARISONS:
        v_st = raw.get(variant, {}).get("sustained_contact_val")
        r_st = raw.get(ref, {}).get("sustained_contact_val")
        deltas = _paired_drift_delta(v_st, r_st)
        if not deltas:
            a(f"| {label} | 0 | - | - |")
            continue
        mean_d, lo, hi = _bootstrap_ci(deltas, seed=seed)
        ci_str = f"[{lo:+.2f}, {hi:+.2f}]"
        sig = " 🟢" if (lo > 0 or hi < 0) else ""
        a(f"| {label} | {len(deltas)} | {mean_d:+.2f} | {ci_str}{sig} |")
    a("")
    a("### Gait — L_R_height_corr delta")
    a("")
    a("| comparison | n_paired | mean Δ | 95% CI |")
    a("| --- | ---: | ---: | --- |")
    for variant, ref, label in PAIRED_COMPARISONS:
        v_st = raw.get(variant, {}).get("gait_val")
        r_st = raw.get(ref, {}).get("gait_val")
        deltas = _paired_gait_delta(v_st, r_st, "L_R_height_corr")
        if not deltas:
            a(f"| {label} | 0 | - | - |")
            continue
        mean_d, lo, hi = _bootstrap_ci(deltas, seed=seed)
        ci_str = f"[{lo:+.3f}, {hi:+.3f}]"
        sig = " 🟢" if (lo > 0 or hi < 0) else ""
        a(f"| {label} | {len(deltas)} | {mean_d:+.3f} | {ci_str}{sig} |")
    a("")
    a("### Gait — frac_both_swing delta (val)")
    a("")
    a("| comparison | n_paired | mean Δ | 95% CI |")
    a("| --- | ---: | ---: | --- |")
    for variant, ref, label in PAIRED_COMPARISONS:
        v_st = raw.get(variant, {}).get("gait_val")
        r_st = raw.get(ref, {}).get("gait_val")
        deltas = _paired_gait_delta(v_st, r_st, "frac_both_swing")
        if not deltas:
            a(f"| {label} | 0 | - | - |")
            continue
        mean_d, lo, hi = _bootstrap_ci(deltas, seed=seed)
        ci_str = f"[{lo:+.3f}, {hi:+.3f}]"
        sig = " 🟢" if (lo > 0 or hi < 0) else ""
        a(f"| {label} | {len(deltas)} | {mean_d:+.3f} | {ci_str}{sig} |")
    a("")


def _render_g1_soft_table(a: Any, g1_soft: dict) -> None:
    a("## G1 soft-stance diagnostic (val)")
    a("")
    a("Per prompt §4: G1 (and A0/A1/A2) only promote if soft-stance is healthy "
      "— not just aggregate gait stats. Flagged when constant_mid_rate is "
      "high, soft_alt_std is low, or soft_transition_density is far below GT.")
    a("")
    a("| variant | n_seg | pL mean | pR mean | soft_alt_std | soft_trans / GT_trans | constant_mid_rate | low_alt_amp_rate | low_trans_rate |")
    a("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    variants_with_g1 = [
        "r29_nb_g1_phasefree_gait_fixed",        # reference
        "r29_ns_a0_c41_g1_loss_s4",
        "r29_ns_a1_c41_s4_g1",
        "r29_ns_a2_c41_i5_g1",
    ]
    for v in variants_with_g1:
        s = (g1_soft.get(v, {}) or {}).get("val")
        if not s:
            a(f"| `{v}` | - | - | - | - | - | - | - | - |")
            continue
        agg = s.get("soft_aggregate", {})
        n_seg = agg.get("n_segments", 0)
        soft_trans = agg.get("soft_transition_density")
        gt_trans = agg.get("gt_transition_density")
        ratio = (soft_trans / gt_trans) if (soft_trans is not None and gt_trans not in (None, 0)) else None
        a(
            f"| `{v}` | {_fmt(n_seg, 0)} | "
            f"{_fmt(agg.get('pL_mean'), 3)} | "
            f"{_fmt(agg.get('pR_mean'), 3)} | "
            f"{_fmt(agg.get('soft_alt_std'), 3)} | "
            f"{_fmt(ratio, 3)} | "
            f"{_fmt(agg.get('constant_mid_rate'), 3)} | "
            f"{_fmt(agg.get('low_alt_amplitude_rate'), 3)} | "
            f"{_fmt(agg.get('low_transition_rate'), 3)} |"
        )
    a("")


def _render_repr_floor(a: Any, repr_floor: dict) -> None:
    a("## Motion representation round-trip floor")
    a("")
    a("FK(motion_135) vs raw `joints_22` from source NPZ. Measures the "
      "hard floor of the SMPL-pose-135 representation + FK reconstruction "
      "BEFORE any model error. A capacity sweep is only worthwhile if this "
      "floor is small relative to current model drift.")
    a("")
    for sub in SUBLABELS:
        s = repr_floor.get(sub)
        if not s:
            a(f"### {sub}")
            a("")
            a("(no `repr_floor_stats.json` found)")
            a("")
            continue
        agg = s.get("aggregate", {})
        interp = s.get("interpretation", {})
        a(f"### {sub} — verdict: **{interp.get('verdict', 'unknown')}**")
        a("")
        a(interp.get("reason", "(no interpretation)"))
        a("")
        per_joint = agg.get("per_joint", {})
        a("| joint | mean (cm) | p95 (cm) |")
        a("| --- | ---: | ---: |")
        for j in ("pelvis", "left_wrist", "right_wrist", "left_ankle",
                  "right_ankle", "left_foot", "right_foot", "neck"):
            if j not in per_joint:
                continue
            row = per_joint[j]
            a(
                f"| {j} | "
                f"{_fmt(row.get('mean_cm'), 2)} | "
                f"{_fmt(row.get('p95_cm'), 2)} |"
            )
        a("")


def _render_invalid_h1_warning(a: Any, results_root: Path) -> None:
    # If old H1 diag dirs exist, surface the invalid marker.
    old_h1_present = any(
        (results_root / f"round29_{INVALID_OLD_VARIANT}_diag_{kind}_{sub}").exists()
        for kind in KINDS for sub in SUBLABELS
    )
    if not old_h1_present:
        return
    a("## Invalid historical reference")
    a("")
    a(f"⚠️  `{INVALID_OLD_VARIANT}` diag stats are on disk but are NOT used "
      "as a valid decision reference in this report. The oracle-hint YAML "
      "keys (`use_oracle_interaction_hint`, `oracle_hint_dim`, "
      "`oracle_hint_injection_mode`) were not consumed by the current "
      "dataset/trainer/model — that variant is silently equivalent to R0 "
      "with dead YAML. Use the live `r29_ns_h1_i5_upper_bound` row instead.")
    a("")


def _gait_health_status(
    variant: str,
    rows: dict,
    g1_soft: dict,
    bucket: str = "val",
) -> dict[str, Any]:
    """Per Codex 2026-05-28 fix prompt §3: combine soft (G1 diag) and hard
    (round26 gait diag) gates into a single health verdict per variant.

    Returns dict with keys:
      - hard_status: "healthy" | "degenerate"
      - soft_status: "healthy" | "degenerate" | "missing"
      - hard_reasons: list[str] of failing gates (empty if healthy)
      - soft_reasons: list[str]
      - overall_status: "healthy" | "degenerate" | "missing"
    """
    out: dict[str, Any] = {
        "hard_status": "healthy", "soft_status": "missing",
        "hard_reasons": [], "soft_reasons": [],
        "overall_status": "missing",
    }
    # Hard gait gates from the gait_stats.json aggregate.
    gait = rows.get(variant, {}).get(bucket, {}).get("gait", {})
    fbs = gait.get("frac_both_swing")
    fbst = gait.get("frac_both_stance")
    tps = gait.get("transitions_per_sec")
    hard_reasons: list[str] = []
    try:
        if fbs is not None and float(fbs) > DEGEN_FRAC_BOTH_SWING_MAX:
            hard_reasons.append(
                f"frac_both_swing={float(fbs):.3f} > {DEGEN_FRAC_BOTH_SWING_MAX}"
            )
    except (TypeError, ValueError):
        pass
    try:
        if fbst is not None and float(fbst) < DEGEN_FRAC_BOTH_STANCE_MIN:
            hard_reasons.append(
                f"frac_both_stance={float(fbst):.3f} < {DEGEN_FRAC_BOTH_STANCE_MIN}"
            )
    except (TypeError, ValueError):
        pass
    try:
        if tps is not None and float(tps) < DEGEN_TRANS_PER_SEC_MIN:
            hard_reasons.append(
                f"transitions_per_sec={float(tps):.3f} < {DEGEN_TRANS_PER_SEC_MIN}"
            )
    except (TypeError, ValueError):
        pass
    out["hard_reasons"] = hard_reasons
    out["hard_status"] = "degenerate" if hard_reasons else "healthy"

    # Soft G1 gates from the G1 soft-stance diag.
    soft = (g1_soft.get(variant, {}) or {}).get(bucket)
    if soft:
        agg = soft.get("soft_aggregate") or {}
        soft_reasons: list[str] = []
        cmid = agg.get("constant_mid_rate")
        alt_std = agg.get("soft_alt_std")
        low_alt = agg.get("low_alt_amplitude_rate")
        low_trans = agg.get("low_transition_rate")
        try:
            if cmid is not None and float(cmid) > 0.40:
                soft_reasons.append(f"constant_mid_rate={float(cmid):.3f} > 0.40")
        except (TypeError, ValueError):
            pass
        try:
            if alt_std is not None and float(alt_std) < 0.15:
                soft_reasons.append(f"soft_alt_std={float(alt_std):.3f} < 0.15")
        except (TypeError, ValueError):
            pass
        try:
            if low_alt is not None and float(low_alt) > 0.50:
                soft_reasons.append(
                    f"low_alt_amplitude_rate={float(low_alt):.3f} > 0.50"
                )
        except (TypeError, ValueError):
            pass
        try:
            if low_trans is not None and float(low_trans) > 0.50:
                soft_reasons.append(
                    f"low_transition_rate={float(low_trans):.3f} > 0.50"
                )
        except (TypeError, ValueError):
            pass
        out["soft_reasons"] = soft_reasons
        out["soft_status"] = "degenerate" if soft_reasons else "healthy"
    # Overall: any degenerate → degenerate; missing soft + healthy hard → missing.
    if out["hard_status"] == "degenerate" or out["soft_status"] == "degenerate":
        out["overall_status"] = "degenerate"
    elif out["soft_status"] == "missing":
        out["overall_status"] = "missing"
    else:
        out["overall_status"] = "healthy"
    return out


def _render_gait_health_table(
    a: Any, rows: dict, g1_soft: dict,
) -> None:
    """Emit a compact hard+soft gait-health table for G1 + A0/A1/A2."""
    a("## Gait health gates (val)")
    a("")
    a("Hard gates from `gait_stats.json` aggregate; soft gates from G1 "
      "soft-stance diag. Per Codex fix prompt §3, a variant is `healthy` "
      "only if BOTH the hard and soft gates pass. Status `missing` means "
      "the soft-stance diag has no stats — promotion cannot be decided.")
    a("")
    a("Hard gates (any failure → degenerate): "
      "`frac_both_swing ≤ 0.70`, "
      "`frac_both_stance ≥ 0.02`, "
      "`transitions_per_sec ≥ 0.40`.")
    a("")
    a("Soft gates (any failure → degenerate): "
      "`constant_mid_rate ≤ 0.40`, "
      "`soft_alt_std ≥ 0.15`, "
      "`low_alt_amplitude_rate ≤ 0.50`, "
      "`low_transition_rate ≤ 0.50`.")
    a("")
    a("| variant | hard | soft | overall | reasons |")
    a("| --- | :---: | :---: | :---: | --- |")
    for v in (
        "r29_nb_g1_phasefree_gait_fixed",
        "r29_ns_a0_c41_g1_loss_s4",
        "r29_ns_a1_c41_s4_g1",
        "r29_ns_a2_c41_i5_g1",
    ):
        h = _gait_health_status(v, rows, g1_soft, bucket="val")
        reasons = h["hard_reasons"] + h["soft_reasons"]
        reasons_str = "; ".join(reasons) if reasons else "—"
        emoji = {
            "healthy": "✓", "degenerate": "⚠️", "missing": "—",
        }
        a(
            f"| `{v}` | "
            f"{emoji[h['hard_status']]} {h['hard_status']} | "
            f"{emoji[h['soft_status']]} {h['soft_status']} | "
            f"**{emoji[h['overall_status']]} {h['overall_status']}** | "
            f"{reasons_str} |"
        )
    a("")


def _render_decision_text(
    a: Any, rows: dict, g1_soft: dict, repr_floor: dict, raw: dict,
) -> None:
    a("## Decision verdict")
    a("")

    def _val(v: str, kind: str, key: str) -> float | None:
        x = rows.get(v, {}).get("val", {}).get(kind, {}).get(key)
        try:
            return float(x) if x is not None else None
        except (TypeError, ValueError):
            return None

    # Compute health statuses up front for all G1-loss variants.
    health = {
        v: _gait_health_status(v, rows, g1_soft, bucket="val")
        for v in (
            "r29_nb_g1_phasefree_gait_fixed",
            "r29_ns_a0_c41_g1_loss_s4",
            "r29_ns_a1_c41_s4_g1",
            "r29_ns_a2_c41_i5_g1",
        )
    }

    # A0 verdict.
    a("### A0 (`r29_ns_a0_c41_g1_loss_s4`) mainline verdict")
    a("")
    a("Promote A0 ONLY IF: contact/body near B1, gait near G1, AND both "
      "hard+soft gait health gates pass (see §Gait health gates).")
    a("")
    a0_drift = _val("r29_ns_a0_c41_g1_loss_s4", "sustained_contact", "drift_max_mean_cm")
    b1_drift = _val("r29_nb_b1_c41_only", "sustained_contact", "drift_max_mean_cm")
    a0_lr = _val("r29_ns_a0_c41_g1_loss_s4", "gait", "L_R_height_corr")
    g1_lr = _val("r29_nb_g1_phasefree_gait_fixed", "gait", "L_R_height_corr")
    a0_body = _val("r29_ns_a0_c41_g1_loss_s4", "body_action", "delta_err_cm_mean_overall")
    b1_body = _val("r29_nb_b1_c41_only", "body_action", "delta_err_cm_mean_overall")
    a(f"- A0 val drift {_fmt(a0_drift, 2)} vs B1 {_fmt(b1_drift, 2)} cm")
    a(f"- A0 val L_R_corr {_fmt(a0_lr, 3)} vs G1 {_fmt(g1_lr, 3)}")
    a(f"- A0 val body delta_err {_fmt(a0_body, 2)} vs B1 {_fmt(b1_body, 2)} cm")
    a0_h = health["r29_ns_a0_c41_g1_loss_s4"]
    a(f"- A0 hard gait: **{a0_h['hard_status']}**"
      + (f" ({'; '.join(a0_h['hard_reasons'])})" if a0_h['hard_reasons'] else ""))
    a(f"- A0 soft gait: **{a0_h['soft_status']}**"
      + (f" ({'; '.join(a0_h['soft_reasons'])})" if a0_h['soft_reasons'] else ""))
    if a0_h["overall_status"] == "degenerate":
        a("- ⚠️ **A0 cannot be promoted: gait degeneracy present** "
          "(failing gates listed above).")
    elif a0_h["overall_status"] == "missing":
        a("- ⚠️ A0 soft-stance diag missing — promotion gated until soft "
          "diag is generated.")
    else:
        a("- ✓ A0 passes both hard and soft gait gates.")
    a("")

    # A1 verdict.
    a("### A1 (`r29_ns_a1_c41_s4_g1`) S4-as-condition verdict")
    a("")
    a("If A1 >> A0 on gait without contact regression AND A1 passes both "
      "gait gates → Stage-1.5 should output an explicit support condition. "
      "If A1 ≈ A0 → S4 stays loss-only.")
    a("")
    a1_drift = _val("r29_ns_a1_c41_s4_g1", "sustained_contact", "drift_max_mean_cm")
    a1_lr = _val("r29_ns_a1_c41_s4_g1", "gait", "L_R_height_corr")
    a(f"- A1 val drift {_fmt(a1_drift, 2)} vs A0 {_fmt(a0_drift, 2)} cm")
    a(f"- A1 val L_R_corr {_fmt(a1_lr, 3)} vs A0 {_fmt(a0_lr, 3)}")
    a1_h = health["r29_ns_a1_c41_s4_g1"]
    a(f"- A1 hard gait: **{a1_h['hard_status']}**"
      + (f" ({'; '.join(a1_h['hard_reasons'])})" if a1_h['hard_reasons'] else ""))
    a(f"- A1 soft gait: **{a1_h['soft_status']}**"
      + (f" ({'; '.join(a1_h['soft_reasons'])})" if a1_h['soft_reasons'] else ""))
    if a1_h["overall_status"] == "degenerate":
        a("- ⚠️ **A1 cannot be promoted: gait degeneracy present.**")
    elif a1_h["overall_status"] == "missing":
        a("- ⚠️ A1 soft-stance diag missing.")
    a("")

    # A2 verdict.
    a("### A2 (`r29_ns_a2_c41_i5_g1`) combined-mainline verdict")
    a("")
    a("Promote A2 ONLY IF: contact ≥ H1, gait passes both gates, and no "
      "body regression vs A0/G1.")
    a("")
    a2_drift = _val("r29_ns_a2_c41_i5_g1", "sustained_contact", "drift_max_mean_cm")
    h1_drift = _val("r29_ns_h1_i5_upper_bound", "sustained_contact", "drift_max_mean_cm")
    a2_lr = _val("r29_ns_a2_c41_i5_g1", "gait", "L_R_height_corr")
    a(f"- A2 val drift {_fmt(a2_drift, 2)} vs H1 {_fmt(h1_drift, 2)} cm")
    a(f"- A2 val L_R_corr {_fmt(a2_lr, 3)} vs A0 {_fmt(a0_lr, 3)}")
    a2_h = health["r29_ns_a2_c41_i5_g1"]
    a(f"- A2 hard gait: **{a2_h['hard_status']}**"
      + (f" ({'; '.join(a2_h['hard_reasons'])})" if a2_h['hard_reasons'] else ""))
    a(f"- A2 soft gait: **{a2_h['soft_status']}**"
      + (f" ({'; '.join(a2_h['soft_reasons'])})" if a2_h['soft_reasons'] else ""))
    if a2_h["overall_status"] == "degenerate":
        a("- ⚠️ **A2 cannot be promoted: gait degeneracy present.**")
    elif a2_h["overall_status"] == "missing":
        a("- ⚠️ A2 soft-stance diag missing.")
    else:
        a("- ✓ A2 passes both hard and soft gait gates.")
    a("")

    # H1 verdict.
    a("### H1 (`r29_ns_h1_i5_upper_bound`) contact-content verdict")
    a("")
    a("If H1 improves contact AND representation floor is low → next "
      "architecture needs richer Stage-1.5 contact planning. If H1 does "
      "not improve AND representation floor is high → motion representation "
      "is on the critical path. Do NOT write 'contact bottleneck is not "
      "condition content' unless H1 actually does not improve.")
    a("")
    r0_drift = _val("r29_ft_r0_clean_a3_baseline", "sustained_contact", "drift_max_mean_cm")
    h1_lh = _val("r29_ns_h1_i5_upper_bound", "sustained_contact", "part_left_hand_drift_max_mean")
    r0_lh = _val("r29_ft_r0_clean_a3_baseline", "sustained_contact", "part_left_hand_drift_max_mean")
    h1_rh = _val("r29_ns_h1_i5_upper_bound", "sustained_contact", "part_right_hand_drift_max_mean")
    r0_rh = _val("r29_ft_r0_clean_a3_baseline", "sustained_contact", "part_right_hand_drift_max_mean")
    a(f"- H1 val drift {_fmt(h1_drift, 2)} vs R0 {_fmt(r0_drift, 2)} cm")
    a(f"- H1 val LH drift {_fmt(h1_lh, 2)} vs R0 {_fmt(r0_lh, 2)} cm")
    a(f"- H1 val RH drift {_fmt(h1_rh, 2)} vs R0 {_fmt(r0_rh, 2)} cm")
    floor_val = repr_floor.get("val") or {}
    floor_interp = floor_val.get("interpretation", {})
    a(f"- repr floor verdict (val): **{floor_interp.get('verdict', 'unknown')}**")
    a("")


# --------------------------------------------------------------------------- #
# Main report
# --------------------------------------------------------------------------- #

def _render_report(
    rows: dict, raw: dict, g1_soft: dict, repr_floor: dict,
    results_root: Path, missing: list[str],
) -> str:
    today = date.today().isoformat()
    L: list[str] = []
    a = L.append

    a("# Round-29 next-step ablation report")
    a("")
    a(f"**Date:** {today}")
    a("**Protocol:** FULL InterAct train set, 80 ep, heldout val,")
    a("from-scratch (no init_checkpoint), bs=32 / accum=1 (2× 5080).")
    a("R0 / B1 / G1 are existing references; A0/A1/H1/A2 are the 4 new")
    a("train variants in this matrix. The previous oracle-hint H1 is")
    a("invalid — see warning below.")
    a("")
    a("**Rows shown** (in display order):")
    a("")
    a(f"- **R0 (ref)** `{REFERENCE_VARIANTS[0]}`")
    a(f"- **B1 (ref)** `{REFERENCE_VARIANTS[1]}`")
    a(f"- **G1 (ref)** `{REFERENCE_VARIANTS[2]}`")
    a("- **A0** `r29_ns_a0_c41_g1_loss_s4` — C41 + G1 losses, S4 loss-only")
    a("- **A1** `r29_ns_a1_c41_s4_g1` — C41 + S4 consumed + G1 losses")
    a("- **H1** `r29_ns_h1_i5_upper_bound` — R0 cond with I3→I5 (live)")
    a("- **A2** `r29_ns_a2_c41_i5_g1` — C41 + I5 + G1 losses")
    a("")
    if missing:
        a("⚠️ **Partial report** — required stats are missing. "
          "**This report is NOT launch-decision-grade**; treat sections "
          "with placeholder dashes as unresolved. The missing slots are:")
        a("")
        for m in missing[:50]:
            a(f"- `{m}`")
        if len(missing) > 50:
            a(f"- ...and {len(missing) - 50} more (truncated)")
        a("")

    _render_invalid_h1_warning(a, results_root)

    for sub in SUBLABELS:
        _render_section_for_sublabel(a, rows, sub)

    _render_paired_bootstrap(a, raw)
    _render_g1_soft_table(a, g1_soft)
    _render_gait_health_table(a, rows, g1_soft)
    _render_repr_floor(a, repr_floor)
    _render_decision_text(a, rows, g1_soft, repr_floor, raw)

    return "\n".join(L) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize Round-29 next-step ablation (A0/A1/H1/A2) against "
            "R0/B1/G1 references; paired bootstrap CIs + G1 soft-stance "
            "diagnostic + motion-repr floor."
        ),
    )
    parser.add_argument(
        "--results-root", default=str(DEFAULT_RESULTS_ROOT),
        help="Directory containing the diag dirs.",
    )
    parser.add_argument(
        "--out", default=str(DEFAULT_REPORT_PATH),
        help="Output Markdown path.",
    )
    parser.add_argument(
        "--allow-partial", action="store_true",
        help="Render the report even when required stats are missing.",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root)
    rows, raw, g1_soft, repr_floor = _gather(results_root)
    missing = _missing_required_stats(rows, raw, g1_soft, repr_floor)
    if missing and not args.allow_partial:
        msg = (
            "FATAL: required diag stats are missing — cannot summarize.\n"
            "Missing slots:\n"
            + "\n".join(f"  {m}" for m in missing[:30])
            + (f"\n  ... ({len(missing) - 30} more)" if len(missing) > 30 else "")
            + "\n\nRun `--allow-partial` to render anyway (with placeholders)."
        )
        raise SystemExit(msg)

    report = _render_report(rows, raw, g1_soft, repr_floor, results_root, missing)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
