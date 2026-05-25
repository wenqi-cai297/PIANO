"""Round-29 ablation result summarizer (prompt §5.7, §8 + Codex post-review §P1).

Reads:
    analyses/round29_stage2_cond_ablation_manifest.json
    analyses/round29_<variant_id>_diag_sustained_contact/sustained_contact_stats.json
    analyses/round29_<variant_id>_diag_gait/gait_stats.json
    analyses/round29_<variant_id>_diag_body_action/body_action_stats.json
    runs/training/<output_dir>/metrics.jsonl                              (optional)
    analyses/round29_condition_stats.json                                 (optional)

Writes:
    analyses/round29_stage2_cond_ablation_summary.json
    analyses/round29_stage2_cond_ablation_summary.md

Codex review fixes (post-2026-05-26):
  * Load real diagnostic filenames (sustained_contact_stats.json /
    gait_stats.json / body_action_stats.json), NOT the placeholder
    {summary,diag,metrics}.json the original draft tried.
  * Flatten nested metrics into stable flat keys before scoring (see
    ``_flatten_metrics`` below).
  * Score `amp_ratio` by closeness to 1 (`abs(amp_ratio - 1)`), NOT as
    higher-is-better.
  * Regression gates use the flat keys and refuse to silently pass when
    all axis metrics are missing.
  * Baseline (`r29_f0_baseline`) missing -> "not_rankable", and the
    process exits non-zero unless `--allow-missing-results` is set.
  * Output includes a per-variant status (implemented / trained /
    diagnostics_complete / rankable / not_rankable_reason).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from piano.data.interaction_hint import (  # noqa: E402
    BODY_ACTION_KEY_JOINT_NAMES,
)


# Filenames produced by the three diagnostics.
DIAG_FILES: dict[str, str] = {
    "sustained_contact": "sustained_contact_stats.json",
    "gait":              "gait_stats.json",
    "body_action":       "body_action_stats.json",
}


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _safe_load_json(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _last_jsonl_record(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            last = None
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    last = json.loads(line)
                except json.JSONDecodeError:
                    continue
        return last
    except Exception:
        return None


def _as_float(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, bool):
        return None  # don't treat booleans as numbers
    if isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x)):
        return float(x)
    return None


# ---------------------------------------------------------------------------
# Metric flattening (the Codex fix at the heart of the summarizer)
# ---------------------------------------------------------------------------

def _flatten_sustained_contact(d: dict | None) -> dict[str, float | None]:
    """Flatten sustained_contact_stats.json into flat keys per Codex review."""
    if not d:
        return {}
    overall = d.get("overall", {})
    drift = overall.get("drift_max_cm", {}) if isinstance(overall, dict) else {}
    track = overall.get("tracking_fraction", {}) if isinstance(overall, dict) else {}
    n_segs = _as_float(overall.get("n_segments")) if isinstance(overall, dict) else None

    def _rate(n_key: str) -> float | None:
        n = _as_float(overall.get(n_key)) if isinstance(overall, dict) else None
        if n is None or n_segs is None or n_segs <= 0:
            return None
        return n / n_segs

    return {
        "drift_max_mean_cm":       _as_float(drift.get("mean")),
        "drift_max_p95_cm":        _as_float(drift.get("p95")),
        "pct_drift_gt_5cm":        _rate("n_drift_max_above_5cm"),
        "pct_drift_gt_10cm":       _rate("n_drift_max_above_10cm"),
        "tracking_fraction_mean":  _as_float(track.get("mean")),
        "tracking_fraction_lt_05": _as_float(track.get("rate_below_0.5")),
        "n_segments":              n_segs,
    }


def _flatten_gait(d: dict | None) -> dict[str, float | None]:
    if not d:
        return {}
    agg = d.get("pred_aggregate", {})
    if not isinstance(agg, dict):
        return {}
    both_swing = agg.get("frac_both_swing", {})
    both_stance = agg.get("frac_both_stance", {})
    trans_per_sec = agg.get("transitions_per_second", {})
    lr_corr = agg.get("L_R_height_corr", {})
    period = agg.get("step_period_frames", {})
    return {
        "frac_both_swing":      _as_float(both_swing.get("mean")) if isinstance(both_swing, dict) else None,
        "frac_both_stance":     _as_float(both_stance.get("mean")) if isinstance(both_stance, dict) else None,
        "transitions_per_second": _as_float(trans_per_sec.get("mean")) if isinstance(trans_per_sec, dict) else None,
        "lr_height_corr_mean":  _as_float(lr_corr.get("mean")) if isinstance(lr_corr, dict) else None,
        "step_period_rate":     _as_float(period.get("rate_with_period")) if isinstance(period, dict) else None,
        "n_walking_segments":   _as_float(d.get("n_walking_segments")),
    }


def _flatten_body_action(d: dict | None) -> dict[str, float | None]:
    if not d:
        return {}
    agg = d.get("aggregate", {})
    if not isinstance(agg, dict):
        return {}

    def _mean_over_joints(metric_key: str) -> float | None:
        vals: list[float] = []
        for joint in BODY_ACTION_KEY_JOINT_NAMES:
            joint_stats = agg.get(joint)
            if not isinstance(joint_stats, dict):
                continue
            v = _as_float(joint_stats.get(metric_key))
            if v is not None:
                vals.append(v)
        return float(sum(vals) / len(vals)) if vals else None

    delta_error_cm = _mean_over_joints("delta_error_cm_mean")
    direction_cos  = _mean_over_joints("direction_cosine_mean")
    active_frac    = _mean_over_joints("active_frame_frac_mean")

    # amp_ratio: score as |amp_ratio - 1| averaged across joints.
    amp_ratio_errs: list[float] = []
    for joint in BODY_ACTION_KEY_JOINT_NAMES:
        joint_stats = agg.get(joint)
        if not isinstance(joint_stats, dict):
            continue
        ar = _as_float(joint_stats.get("amp_ratio_mean"))
        if ar is not None:
            amp_ratio_errs.append(abs(ar - 1.0))
    amp_ratio_error = float(sum(amp_ratio_errs) / len(amp_ratio_errs)) if amp_ratio_errs else None

    return {
        "key_joint_delta_error_cm": delta_error_cm,
        "amp_ratio_error":          amp_ratio_error,
        "direction_cosine_mean":    direction_cos,
        "active_frame_frac_mean":   active_frac,
    }


def _flatten_train_metrics(d: dict | None) -> dict[str, float | None]:
    if not d:
        return {}
    return {
        "loss_anchor_joint_pos": _as_float(d.get("loss_anchor_joint_pos")),
    }


def _flatten_all(metrics_raw: dict[str, Any]) -> dict[str, float | None]:
    flat: dict[str, float | None] = {}
    flat.update(_flatten_sustained_contact(metrics_raw.get("diag_sustained_contact")))
    flat.update(_flatten_gait(metrics_raw.get("diag_gait")))
    flat.update(_flatten_body_action(metrics_raw.get("diag_body_action")))
    flat.update(_flatten_train_metrics(metrics_raw.get("train_last")))
    return flat


# ---------------------------------------------------------------------------
# Per-variant metric extraction
# ---------------------------------------------------------------------------

def _extract_metrics(variant: dict, diagnostics_root: Path) -> dict[str, Any]:
    """Pull per-variant raw blobs from disk. Flattening happens later.

    ``diagnostics_root`` is the directory under which the per-variant
    ``round29_<vid>_diag_<kind>/<file>.json`` paths live. Defaults to
    ``ROOT/"analyses"`` in production; tests point it at ``tmp_path`` so
    fake diagnostic JSONs don't pollute the real repo.
    """
    vid = variant["variant_id"]
    out: dict[str, Any] = {}

    for kind, filename in DIAG_FILES.items():
        diag_dir = diagnostics_root / f"round29_{vid}_diag_{kind}"
        j = _safe_load_json(diag_dir / filename)
        if j is not None:
            out[f"diag_{kind}"] = j

    train_metrics = _last_jsonl_record(ROOT / variant["output_dir"] / "metrics.jsonl")
    if train_metrics is not None:
        out["train_last"] = train_metrics

    return out


def _per_variant_status(
    variant: dict,
    raw_metrics: dict[str, Any],
    flat_metrics: dict[str, float | None],
) -> dict[str, Any]:
    """Per-variant status block (Codex P1 §5 — separate implemented /
    trained / diagnostics_complete / rankable / not_rankable_reason)."""
    output_dir = ROOT / variant["output_dir"]
    final_pt = output_dir / "final.pt"
    train_last = raw_metrics.get("train_last") is not None

    diags_present = {k: f"diag_{k}" in raw_metrics for k in DIAG_FILES}
    diagnostics_complete = all(diags_present.values())
    rankable_axes = [
        k for k in (
            "drift_max_mean_cm", "tracking_fraction_mean",
            "frac_both_swing", "frac_both_stance",
            "key_joint_delta_error_cm", "amp_ratio_error",
        )
        if flat_metrics.get(k) is not None
    ]
    rankable = len(rankable_axes) >= 2
    reason: str | None = None
    if not diagnostics_complete:
        missing = [k for k, present in diags_present.items() if not present]
        reason = f"missing diagnostics: {missing}"
    elif not rankable:
        reason = "fewer than 2 axes have usable metrics"

    return {
        "implemented": True,
        "trained": final_pt.exists() or train_last,
        "diagnostics_complete": diagnostics_complete,
        "diagnostics_present": diags_present,
        "rankable": rankable,
        "not_rankable_reason": reason,
        "n_axes_with_metric": len(rankable_axes),
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

AXES_LOWER_BETTER = [
    ("interaction", "drift_max_mean_cm"),
    ("interaction", "pct_drift_gt_5cm"),
    ("gait",        "frac_both_swing"),
    ("gait",        "frac_both_stance"),
    ("body_action", "key_joint_delta_error_cm"),
    ("body_action", "amp_ratio_error"),
    ("general",     "loss_anchor_joint_pos"),
]
AXES_HIGHER_BETTER = [
    ("interaction", "tracking_fraction_mean"),
    ("gait",        "transitions_per_second"),
    ("gait",        "step_period_rate"),
    ("body_action", "direction_cosine_mean"),
]


def _norm_lower(base: float | None, cand: float | None) -> float | None:
    if base is None or cand is None:
        return None
    if base == 0:
        return 0.0
    return float((base - cand) / abs(base))


def _norm_higher(base: float | None, cand: float | None) -> float | None:
    if base is None or cand is None:
        return None
    if base == 0:
        return 0.0
    return float((cand - base) / abs(base))


def _compute_scores(
    flat: dict[str, float | None],
    baseline_flat: dict[str, float | None] | None,
) -> dict[str, Any]:
    axis_scores: dict[str, list[float]] = {}
    axis_details: dict[str, dict[str, float | None]] = {}

    for axis, key in AXES_LOWER_BETTER:
        cand = flat.get(key)
        base = baseline_flat.get(key) if baseline_flat else None
        axis_details.setdefault(axis, {})[key] = cand
        s = _norm_lower(base, cand)
        if s is not None:
            axis_scores.setdefault(axis, []).append(s)
    for axis, key in AXES_HIGHER_BETTER:
        cand = flat.get(key)
        base = baseline_flat.get(key) if baseline_flat else None
        axis_details.setdefault(axis, {})[key] = cand
        s = _norm_higher(base, cand)
        if s is not None:
            axis_scores.setdefault(axis, []).append(s)

    per_axis = {
        axis: (sum(scores) / len(scores)) if scores else None
        for axis, scores in axis_scores.items()
    }
    valid_scores = [v for v in per_axis.values() if v is not None]
    composite = float(sum(valid_scores) / len(valid_scores)) if valid_scores else None
    return {
        "per_axis": per_axis,
        "axis_details": axis_details,
        "composite": composite,
        "n_axis_scores_used": len(valid_scores),
    }


def _passes_regression_gates(
    flat: dict[str, float | None],
    baseline_flat: dict[str, float | None] | None,
) -> tuple[bool | None, list[str]]:
    """Return (passes, fails). passes=None when there are no metrics at
    all on either side (so we can't even start to check); passes=False
    when at least one axis worsened ≥5% AND ≥0.01."""
    if baseline_flat is None:
        return None, ["no baseline metrics — cannot check gates"]
    fails: list[str] = []
    checked = 0

    def _check_lower(key: str) -> None:
        nonlocal checked
        cand = flat.get(key); base = baseline_flat.get(key)
        if cand is None or base is None:
            return
        checked += 1
        if cand > base * 1.05 and (cand - base) > 0.01:
            fails.append(f"{key}: {cand:.4f} > baseline {base:.4f}")

    def _check_higher(key: str) -> None:
        nonlocal checked
        cand = flat.get(key); base = baseline_flat.get(key)
        if cand is None or base is None:
            return
        checked += 1
        if cand < base * 0.95 and (base - cand) > 0.01:
            fails.append(f"{key}: {cand:.4f} < baseline {base:.4f}")

    for _, k in AXES_LOWER_BETTER:
        _check_lower(k)
    for _, k in AXES_HIGHER_BETTER:
        _check_higher(k)

    if checked == 0:
        # No overlap at all — gate decision is not safe.
        return None, ["no overlapping axis metrics between candidate and baseline"]
    return not fails, fails


def _family_count(variant: dict) -> int:
    dims = variant.get("expected_dense_dims", {})
    return sum(1 for v in dims.values() if int(v) > 0)


def _pareto_front(rows: list[dict[str, Any]]) -> list[str]:
    cands: list[tuple[str, tuple[float | None, ...]]] = []
    for r in rows:
        if not r.get("status", {}).get("rankable"):
            continue
        scores = r.get("scores", {}).get("per_axis", {}) or {}
        triple = (
            scores.get("interaction"),
            scores.get("gait"),
            scores.get("body_action"),
        )
        if any(t is None for t in triple):
            continue
        cands.append((r["variant_id"], triple))
    front: list[str] = []
    for vid, triple in cands:
        dominated = False
        for other_id, other_triple in cands:
            if other_id == vid:
                continue
            strictly_better = any(o > t for o, t in zip(other_triple, triple))
            not_worse = all(o >= t for o, t in zip(other_triple, triple))
            if strictly_better and not_worse:
                dominated = True
                break
        if not dominated:
            front.append(vid)
    return front


def _minimal_near_best(
    rows: list[dict[str, Any]],
    best_composite: float | None,
    tolerance: float = 0.05,
) -> str | None:
    if best_composite is None:
        return None
    target = best_composite - abs(best_composite) * tolerance
    eligible = [
        r for r in rows
        if r.get("status", {}).get("rankable")
        and r.get("scores", {}).get("composite") is not None
        and float(r["scores"]["composite"]) >= target
        and r.get("regression_gates", {}).get("passes") is True
    ]
    if not eligible:
        return None
    eligible.sort(key=lambda r: (_family_count(r["variant"]), -r["scores"]["composite"]))
    return eligible[0]["variant_id"]


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def _render_markdown(payload: dict[str, Any]) -> str:
    L: list[str] = []
    L.append("# Round-29 Stage-2 condition + injection ablation summary")
    L.append("")
    L.append(f"- Generated: {payload['generated_at']}")
    L.append(f"- Manifest: `{payload['manifest_path']}`")
    L.append(f"- Variants total / with diagnostics / rankable: "
             f"{payload['n_variants_total']} / "
             f"{payload['n_variants_with_diag']} / "
             f"{payload['n_variants_rankable']}")
    L.append(f"- Baseline metrics available: "
             f"{'yes' if payload['baseline_metrics_present'] else 'NO'}")
    L.append("")

    rec = payload.get("recommendation") or {}
    L.append("## Recommended Stage-2 interface")
    if not rec:
        reason = payload.get("not_computable_reason", "(no recommendation yet)")
        L.append(f"_Not yet computable: {reason}_")
    else:
        L.append(f"- **Best variant:** `{rec.get('best_variant')}`")
        L.append(f"- **Best injection mode:** `{rec.get('best_injection_mode')}`")
        L.append(f"- **Best condition content:** {json.dumps(rec.get('best_content'))}")
        L.append(f"- **Minimal-near-best variant:** `{rec.get('minimal_near_best_variant')}`")
        L.append(f"- **Pareto front:** {rec.get('pareto_front')}")
        L.append("")
        L.append("### Tentative responsibility assignment (per §8.4)")
        L.append(rec.get("responsibility_assignment_text", ""))
    L.append("")

    L.append("## Per-variant rows")
    L.append("")
    L.append(
        "| variant | group | C | I | S | B | inject | trained | diag | rankable | composite | passes_gates |"
    )
    L.append(
        "| --- | --- | --- | --- | --- | --- | --- | :---: | :---: | :---: | --- | --- |"
    )
    for r in payload["rows"]:
        v = r["variant"]
        comp = r.get("scores", {}).get("composite")
        comp_str = f"{comp:.4f}" if isinstance(comp, (int, float)) else "—"
        status = r.get("status", {})
        passes = r.get("regression_gates", {}).get("passes")
        passes_str = "✓" if passes is True else ("✗" if passes is False else "—")
        L.append(
            f"| {v['variant_id']} | {v['group']} | "
            f"{v['coarse_variant']} | {v['interaction_variant']} | "
            f"{v['support_variant']} | {v['body_variant']} | "
            f"{v['injection_mode']} | "
            f"{'✓' if status.get('trained') else ' '} | "
            f"{'✓' if status.get('diagnostics_complete') else ' '} | "
            f"{'✓' if status.get('rankable') else ' '} | "
            f"{comp_str} | {passes_str} |"
        )
    L.append("")

    L.append("## Failure modes (per regression gates)")
    any_fail = False
    for r in payload["rows"]:
        fails = r.get("regression_gates", {}).get("fails") or []
        if fails and r.get("regression_gates", {}).get("passes") is False:
            any_fail = True
            L.append(f"- `{r['variant']['variant_id']}`: {', '.join(fails)}")
    if not any_fail:
        L.append("- (none recorded — either no failing variants or no baseline yet)")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", default="analyses/round29_stage2_cond_ablation_manifest.json",
    )
    parser.add_argument(
        "--output-json", default="analyses/round29_stage2_cond_ablation_summary.json",
    )
    parser.add_argument(
        "--output-md", default="analyses/round29_stage2_cond_ablation_summary.md",
    )
    parser.add_argument(
        "--allow-missing-results", action="store_true",
        help="Don't exit nonzero when baseline/results are missing.",
    )
    parser.add_argument(
        "--diagnostics-root",
        default=str(ROOT / "analyses"),
        help="Root dir under which round29_<vid>_diag_<kind>/<file>.json live. "
             "Defaults to analyses/. Tests point this at tmp_path.",
    )
    args = parser.parse_args()
    diagnostics_root = Path(args.diagnostics_root)

    manifest_path = ROOT / args.manifest
    if not manifest_path.exists():
        print(
            f"[R29-summary] manifest missing at {manifest_path} — run "
            "round29_make_stage2_cond_ablation_configs.py first.",
            file=sys.stderr,
        )
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    rows: list[dict[str, Any]] = []
    for v in manifest["variants"]:
        raw = _extract_metrics(v, diagnostics_root)
        flat = _flatten_all(raw)
        status = _per_variant_status(v, raw, flat)
        rows.append({
            "variant_id": v["variant_id"],
            "variant": v,
            "raw_metrics": raw,
            "flat_metrics": flat,
            "status": status,
        })

    # Baseline (r29_f0_baseline) flat metrics, if usable.
    baseline_row = next(
        (r for r in rows if r["variant_id"] == "r29_f0_baseline"), None,
    )
    baseline_flat = None
    if baseline_row and baseline_row["status"]["rankable"]:
        baseline_flat = baseline_row["flat_metrics"]

    # Score and gate every row.
    for r in rows:
        r["scores"] = _compute_scores(r["flat_metrics"], baseline_flat)
        passes, fails = _passes_regression_gates(r["flat_metrics"], baseline_flat)
        r["regression_gates"] = {"passes": passes, "fails": fails}

    # Aggregate statuses.
    n_diag = sum(1 for r in rows if r["status"]["diagnostics_complete"])
    n_rankable = sum(1 for r in rows if r["status"]["rankable"])

    # Best by composite (passing gates).
    best_row = None
    for r in rows:
        composite = r["scores"].get("composite")
        if composite is None:
            continue
        if r["regression_gates"]["passes"] is not True:
            continue
        if best_row is None or composite > best_row["scores"]["composite"]:
            best_row = r

    front = _pareto_front(rows)
    minimal = _minimal_near_best(
        rows,
        best_composite=best_row["scores"]["composite"] if best_row else None,
    )

    not_computable_reason = None
    recommendation: dict[str, Any] = {}
    if baseline_flat is None:
        not_computable_reason = (
            "baseline (r29_f0_baseline) has no usable diagnostics; "
            "summarizer cannot score other variants."
        )
    elif n_rankable == 0:
        not_computable_reason = "no variant has rankable diagnostics yet."
    elif best_row is None:
        not_computable_reason = (
            "no variant beats baseline while passing all regression gates."
        )
    else:
        recommendation = {
            "best_variant": best_row["variant_id"],
            "best_injection_mode": best_row["variant"]["injection_mode"],
            "best_content": {
                "coarse": best_row["variant"]["coarse_variant"],
                "interaction": best_row["variant"]["interaction_variant"],
                "support": best_row["variant"]["support_variant"],
                "body": best_row["variant"]["body_variant"],
            },
            "minimal_near_best_variant": minimal,
            "pareto_front": front,
            "responsibility_assignment_text": (
                "Once Group A picks `best_injection_mode`, Stage-2's input "
                "interface is fixed. Group B/C/D/E then assign each content "
                "channel to either Stage-1 (cheaply predictable from the "
                "coarse plan) or Stage-1.5 (requires a richer detail head). "
                "Minimal-near-best identifies the smallest content set "
                "Stage-1.5 needs to predict; everything else can fall to "
                "Stage-1 or be deferred."
            ),
        }

    payload = {
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "manifest_path": str(manifest_path),
        "n_variants_total": len(rows),
        "n_variants_with_diag": n_diag,
        "n_variants_rankable": n_rankable,
        "baseline_variant_id": "r29_f0_baseline" if baseline_row else None,
        "baseline_metrics_present": baseline_flat is not None,
        "not_computable_reason": not_computable_reason,
        "rows": rows,
        "recommendation": recommendation,
    }

    out_json = ROOT / args.output_json
    out_md = ROOT / args.output_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    out_md.write_text(_render_markdown(payload), encoding="utf-8")
    print(f"[R29-summary] wrote {out_json}")
    print(f"[R29-summary] wrote {out_md}")
    print(
        f"[R29-summary] {n_diag}/{len(rows)} variants have full diagnostics; "
        f"{n_rankable} are rankable. "
        f"Best: {recommendation.get('best_variant', '(not computable)')}"
    )
    if recommendation:
        return 0
    if args.allow_missing_results:
        print(f"[R29-summary] note: {not_computable_reason}")
        return 0
    print(
        f"[R29-summary] FATAL: not rankable — {not_computable_reason}. "
        "Pass --allow-missing-results to suppress.",
        file=sys.stderr,
    )
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
