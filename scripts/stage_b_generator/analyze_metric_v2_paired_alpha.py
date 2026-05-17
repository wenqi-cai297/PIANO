"""Paired-delta + bootstrap-CI analyzer for metric-v2 alpha=0 vs alpha=1 runs.

Consumes the JSON output of
``replay_key_diagnostics_with_metric_v2.py`` (which now saves
``per_clip_rows[seed/alpha][clips[]]`` with v1/v2/dyn fields) and emits:

- per-seed value at each alpha
- per-seed paired Δ = alpha0 - alpha1
- mean Δ, std Δ
- sign consistency across seeds
- bootstrap 95% CI on per-clip paired Δ (resample clips with replacement
  within seed; events within clips are correlated, so we bootstrap at
  the clip level)
- safety-regression verdict

Sign convention:
- For M2/M3 onset direction / release direction / signed change scores
  POSITIVE Δ means alpha=0 IMPROVES transition direction (closer to GT).
- For far_unobs / anchor_realization / acc_p95 / jerk_p95 / body vel xGT
  the meaning is metric-specific; we report Δ and flag any safety
  bound violation explicitly.

Designed to be re-runnable on any later 6/8/12-seed v2 confirmation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _flatten_clips(replay_payload: dict[str, Any]) -> dict[tuple[int, float], list[dict[str, Any]]]:
    """Return {(seed, alpha): [clip_row, ...]} from the per_clip_rows section."""
    out: dict[tuple[int, float], list[dict[str, Any]]] = {}
    for entry in replay_payload.get("per_clip_rows", []):
        key = (int(entry["seed"]), float(entry["alpha_z_target"]))
        out[key] = list(entry["clips"])
    return out


def _per_clip_metric_value(clip_row: dict[str, Any], dotted_path: str) -> float:
    """Extract a numeric value via 'v2.M2_onset_direction_cm_per_frame_mean' style path.

    For per-clip rows, v2 fields are flat (mean across hand event types)
    so we read the corresponding field directly.
    """
    head, _, tail = dotted_path.partition(".")
    d = clip_row.get(head, {})
    if isinstance(d, dict):
        return float(d.get(tail, 0.0))
    return 0.0


def _seed_mean(rows: list[dict[str, Any]], path: str) -> float:
    return float(np.mean([_per_clip_metric_value(r, path) for r in rows])) if rows else 0.0


def _paired_per_clip_delta(
    rows_alpha0: list[dict[str, Any]], rows_alpha1: list[dict[str, Any]],
    path: str,
) -> list[float]:
    """Per-clip Δ at fixed seed, requires clips aligned by index."""
    aligned = list(zip(rows_alpha0, rows_alpha1))
    out = []
    for r0, r1 in aligned:
        v0 = _per_clip_metric_value(r0, path)
        v1 = _per_clip_metric_value(r1, path)
        out.append(float(v0 - v1))
    return out


def _bootstrap_ci(values: np.ndarray, n_boot: int = 2000, rng_seed: int = 0,
                  ci: tuple[float, float] = (2.5, 97.5)) -> tuple[float, float]:
    if values.size < 2:
        return (float(values.mean()) if values.size else 0.0,) * 2
    rng = np.random.RandomState(int(rng_seed))
    n = values.size
    boots = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boots.append(values[idx].mean())
    lo, hi = np.percentile(boots, ci)
    return float(lo), float(hi)


def _analyze_metric(
    paths_alpha0: dict[int, list[dict[str, Any]]],
    paths_alpha1: dict[int, list[dict[str, Any]]],
    *,
    metric_path: str,
    higher_is_better: bool | None,
    n_boot: int,
    rng_seed: int,
) -> dict[str, Any]:
    seeds = sorted(set(paths_alpha0.keys()) & set(paths_alpha1.keys()))
    per_seed = []
    all_clip_deltas: list[float] = []  # pooled across seeds (clip-level)
    for s in seeds:
        rows0 = paths_alpha0[s]
        rows1 = paths_alpha1[s]
        mean0 = _seed_mean(rows0, metric_path)
        mean1 = _seed_mean(rows1, metric_path)
        clip_deltas = _paired_per_clip_delta(rows0, rows1, metric_path)
        per_seed.append({
            "seed": int(s),
            "alpha0_mean": mean0,
            "alpha1_mean": mean1,
            "paired_delta_mean": float(np.mean(clip_deltas)) if clip_deltas else 0.0,
            "per_clip_deltas": clip_deltas,
        })
        all_clip_deltas.extend(clip_deltas)

    deltas_per_seed_means = np.array([p["paired_delta_mean"] for p in per_seed], dtype=np.float64)
    mean_delta = float(deltas_per_seed_means.mean()) if deltas_per_seed_means.size else 0.0
    std_delta = float(deltas_per_seed_means.std()) if deltas_per_seed_means.size else 0.0
    sign_pos = int(sum(1 for d in deltas_per_seed_means if d > 0))
    sign_neg = int(sum(1 for d in deltas_per_seed_means if d < 0))
    n_seeds = int(deltas_per_seed_means.size)

    # Bootstrap CI: pool per-clip deltas across seeds, resample clips with replacement.
    # This treats each (seed, clip) pair as a sample. Events within clips are
    # already pooled at the clip-row level, so this is clip-level bootstrap.
    pooled_arr = np.asarray(all_clip_deltas, dtype=np.float64)
    ci_lo, ci_hi = _bootstrap_ci(pooled_arr, n_boot=n_boot, rng_seed=rng_seed)

    verdict_excludes_zero = (ci_lo > 0.0 and ci_hi > 0.0) or (ci_lo < 0.0 and ci_hi < 0.0)

    return {
        "metric_path": metric_path,
        "higher_is_better": higher_is_better,
        "n_seeds": n_seeds,
        "per_seed": per_seed,
        "mean_delta_across_seeds": mean_delta,
        "std_delta_across_seeds": std_delta,
        "sign_positive_count": sign_pos,
        "sign_negative_count": sign_neg,
        "sign_consistency_str": f"{sign_pos}/{n_seeds} positive, {sign_neg}/{n_seeds} negative",
        "bootstrap_ci_lo": ci_lo,
        "bootstrap_ci_hi": ci_hi,
        "ci_excludes_zero": bool(verdict_excludes_zero),
        "n_clip_pairs_pooled": int(pooled_arr.size),
    }


def _summarize_verdict(metrics: dict[str, dict[str, Any]],
                       *, safety_threshold: float = 0.10) -> dict[str, Any]:
    """Apply round-8 decision rules to a set of analyzed metrics."""
    # Pull principal metrics
    m2_onset = metrics.get("v2.M2_onset_direction_cm_per_frame_mean", {})
    m2_release = metrics.get("v2.M2_release_direction_cm_per_frame_mean", {})
    m3_onset = metrics.get("v2.M3_onset_signed_cm_mean", {})
    m3_release = metrics.get("v2.M3_release_signed_cm_mean", {})
    body_vel = metrics.get("dyn.body_vel_xGT", {})
    hand_vel = metrics.get("dyn.hand_vel_xGT", {})
    acc_p95 = metrics.get("dyn.acc_p95_xGT", {})
    jerk_p95 = metrics.get("dyn.jerk_p95_xGT", {})

    n_seeds = int(m2_onset.get("n_seeds", 0))
    needed_sign = 4 if n_seeds == 6 else 5 if n_seeds == 8 else max(1, n_seeds * 2 // 3)
    onset_sign_ok = int(m2_onset.get("sign_positive_count", 0)) >= needed_sign
    release_sign_ok = int(m2_release.get("sign_positive_count", 0)) >= needed_sign
    onset_or_release_sign_ok = onset_sign_ok or release_sign_ok
    onset_ci_ok = bool(m2_onset.get("ci_excludes_zero", False) and m2_onset.get("bootstrap_ci_lo", 0.0) > 0)
    release_ci_ok = bool(m2_release.get("ci_excludes_zero", False) and m2_release.get("bootstrap_ci_lo", 0.0) > 0)
    onset_or_release_ci_ok = onset_ci_ok or release_ci_ok
    m3_supports = float(m3_onset.get("mean_delta_across_seeds", 0.0)) > 0 or \
                  float(m3_release.get("mean_delta_across_seeds", 0.0)) > 0

    # Safety: positive Δ in body_vel xGT or acc_p95 xGT above threshold = regression.
    safety_regress_reasons: list[str] = []
    if float(acc_p95.get("bootstrap_ci_hi", 0.0)) > safety_threshold:
        safety_regress_reasons.append(
            f"acc_p95 xGT CI upper {acc_p95['bootstrap_ci_hi']:+.3f} > +{safety_threshold:.2f}"
        )
    if float(jerk_p95.get("bootstrap_ci_hi", 0.0)) > safety_threshold:
        safety_regress_reasons.append(
            f"jerk_p95 xGT CI upper {jerk_p95['bootstrap_ci_hi']:+.3f} > +{safety_threshold:.2f}"
        )
    if abs(float(body_vel.get("mean_delta_across_seeds", 0.0))) > safety_threshold:
        safety_regress_reasons.append(
            f"body vel xGT mean Δ {body_vel['mean_delta_across_seeds']:+.3f} exceeds ±{safety_threshold:.2f}"
        )
    if abs(float(hand_vel.get("mean_delta_across_seeds", 0.0))) > safety_threshold:
        safety_regress_reasons.append(
            f"hand vel xGT mean Δ {hand_vel['mean_delta_across_seeds']:+.3f} exceeds ±{safety_threshold:.2f}"
        )

    case = "C"  # default underpowered
    reason = ""
    if onset_or_release_sign_ok and onset_or_release_ci_ok and m3_supports and not safety_regress_reasons:
        case = "A"
        reason = (
            f"Sign consistency >= {needed_sign}/{n_seeds} on M2; bootstrap CI excludes zero; "
            f"M3 supports direction; safety bounds OK."
        )
    elif safety_regress_reasons and (onset_or_release_sign_ok or onset_or_release_ci_ok):
        case = "E"
        reason = "Transition improves but safety regresses: " + "; ".join(safety_regress_reasons)
    elif not onset_or_release_sign_ok and not onset_or_release_ci_ok:
        case = "B"
        reason = (
            f"Sign consistency < {needed_sign}/{n_seeds} on both onset and release; "
            f"CI does not exclude zero."
        )
    else:
        case = "C"
        reason = "Direction is positive but variance protocol not fully met."

    return {
        "n_seeds": n_seeds,
        "needed_sign_threshold": int(needed_sign),
        "case": case,
        "reason": reason,
        "onset_sign_ok": bool(onset_sign_ok),
        "release_sign_ok": bool(release_sign_ok),
        "onset_ci_excludes_zero_positive": bool(onset_ci_ok),
        "release_ci_excludes_zero_positive": bool(release_ci_ok),
        "m3_supports_direction": bool(m3_supports),
        "safety_regression_reasons": safety_regress_reasons,
    }


METRIC_PATHS: list[tuple[str, bool | None]] = [
    # (dotted path, higher_is_better)
    # v2 transition metrics — higher M2/M3 direction score = better
    ("v2.M2_onset_direction_cm_per_frame_mean", True),
    ("v2.M2_release_direction_cm_per_frame_mean", True),
    ("v2.M3_onset_signed_cm_mean", True),
    ("v2.M3_release_signed_cm_mean", True),
    ("v2.M5_clip2cm_mean", True),
    ("v2.M5_clip5cm_mean", True),
    # safety
    ("dyn.body_vel_xGT", None),
    ("dyn.hand_vel_xGT", None),
    ("dyn.acc_p95_xGT", None),
    ("dyn.jerk_p95_xGT", None),
    # v1 for comparison only
    ("v1.onset_xGT", None),
    ("v1.release_xGT", None),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True,
                        help="JSON file written by replay_key_diagnostics_with_metric_v2.py")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--alpha-test", type=float, default=0.0)
    parser.add_argument("--alpha-baseline", type=float, default=1.0)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--safety-threshold", type=float, default=0.10,
                        help="Per-clip mean Δ safety bound on body/hand vel xGT and CI upper on acc/jerk p95.")
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    clips_by_key = _flatten_clips(payload)
    seeds_with_test = sorted({s for (s, a) in clips_by_key.keys() if abs(a - args.alpha_test) < 1e-6})
    seeds_with_base = sorted({s for (s, a) in clips_by_key.keys() if abs(a - args.alpha_baseline) < 1e-6})
    common_seeds = sorted(set(seeds_with_test) & set(seeds_with_base))
    if not common_seeds:
        raise SystemExit(
            f"No common seeds between alpha_test={args.alpha_test} and "
            f"alpha_baseline={args.alpha_baseline}. Test={seeds_with_test} Base={seeds_with_base}"
        )
    paths_test = {s: clips_by_key[(s, args.alpha_test)] for s in common_seeds}
    paths_base = {s: clips_by_key[(s, args.alpha_baseline)] for s in common_seeds}

    analyzed: dict[str, dict[str, Any]] = {}
    for metric_path, higher in METRIC_PATHS:
        analyzed[metric_path] = _analyze_metric(
            paths_test, paths_base,
            metric_path=metric_path, higher_is_better=higher,
            n_boot=int(args.n_boot), rng_seed=int(args.rng_seed),
        )

    verdict = _summarize_verdict(analyzed, safety_threshold=float(args.safety_threshold))

    out_payload = {
        "source_replay": str(args.input),
        "alpha_test": args.alpha_test,
        "alpha_baseline": args.alpha_baseline,
        "n_seeds": len(common_seeds),
        "seeds": common_seeds,
        "metrics": analyzed,
        "verdict": verdict,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out_payload, indent=2, default=float), encoding="utf-8")

    lines = [
        "# Metric-V2 Paired alpha=0.0 vs alpha=1.0 Variance Analysis (Round 8)",
        "",
        f"- Source replay: `{args.input}`",
        f"- alpha_test = {args.alpha_test}",
        f"- alpha_baseline = {args.alpha_baseline}",
        f"- Seeds: {common_seeds}  (n = {len(common_seeds)})",
        f"- Bootstrap iters: {args.n_boot}; clip-level paired delta resampling.",
        "",
        "## Per-metric summary",
        "",
        "| metric | n seeds | mean Δ across seeds | std Δ | sign pos/neg | CI 95% (clip-level) | CI excludes 0 |",
        "|--------|--------:|---------------------:|------:|:------------:|---------------------|:-------------:|",
    ]
    for path, _ in METRIC_PATHS:
        m = analyzed[path]
        lines.append(
            f"| `{path}` | {m['n_seeds']} | "
            f"{m['mean_delta_across_seeds']:+.4f} | "
            f"{m['std_delta_across_seeds']:.4f} | "
            f"{m['sign_positive_count']}/{m['sign_negative_count']} | "
            f"[{m['bootstrap_ci_lo']:+.4f}, {m['bootstrap_ci_hi']:+.4f}] | "
            f"{'YES' if m['ci_excludes_zero'] else 'no'} |"
        )

    lines += [
        "",
        "## Per-seed transition deltas (alpha=0.0 - alpha=1.0)",
        "",
        "| seed | M2 onset cm/f | M2 release cm/f | M3 onset cm | M3 release cm |",
        "|------|---------------|------------------|--------------|---------------|",
    ]
    for s in common_seeds:
        m2o_seed = next((p for p in analyzed["v2.M2_onset_direction_cm_per_frame_mean"]["per_seed"] if p["seed"] == s), None)
        m2r_seed = next((p for p in analyzed["v2.M2_release_direction_cm_per_frame_mean"]["per_seed"] if p["seed"] == s), None)
        m3o_seed = next((p for p in analyzed["v2.M3_onset_signed_cm_mean"]["per_seed"] if p["seed"] == s), None)
        m3r_seed = next((p for p in analyzed["v2.M3_release_signed_cm_mean"]["per_seed"] if p["seed"] == s), None)
        lines.append(
            f"| {s} | "
            f"{m2o_seed['paired_delta_mean']:+.4f} | "
            f"{m2r_seed['paired_delta_mean']:+.4f} | "
            f"{m3o_seed['paired_delta_mean']:+.3f} | "
            f"{m3r_seed['paired_delta_mean']:+.3f} |"
        )
    lines += [
        "",
        "## Per-seed safety deltas (alpha=0.0 - alpha=1.0)",
        "",
        "| seed | body vel xGT | hand vel xGT | acc p95 xGT | jerk p95 xGT |",
        "|------|--------------|--------------|--------------|--------------|",
    ]
    for s in common_seeds:
        bv = next((p for p in analyzed["dyn.body_vel_xGT"]["per_seed"] if p["seed"] == s), None)
        hv = next((p for p in analyzed["dyn.hand_vel_xGT"]["per_seed"] if p["seed"] == s), None)
        ap = next((p for p in analyzed["dyn.acc_p95_xGT"]["per_seed"] if p["seed"] == s), None)
        jp = next((p for p in analyzed["dyn.jerk_p95_xGT"]["per_seed"] if p["seed"] == s), None)
        lines.append(
            f"| {s} | "
            f"{bv['paired_delta_mean']:+.4f} | "
            f"{hv['paired_delta_mean']:+.4f} | "
            f"{ap['paired_delta_mean']:+.4f} | "
            f"{jp['paired_delta_mean']:+.4f} |"
        )

    lines += [
        "",
        "## Verdict",
        "",
        f"- Case: **{verdict['case']}**",
        f"- Reason: {verdict['reason']}",
        f"- Sign threshold ({verdict['n_seeds']} seeds): >= {verdict['needed_sign_threshold']} positive",
        f"- M2 onset sign OK: {verdict['onset_sign_ok']}",
        f"- M2 release sign OK: {verdict['release_sign_ok']}",
        f"- M2 onset CI excludes 0 positive: {verdict['onset_ci_excludes_zero_positive']}",
        f"- M2 release CI excludes 0 positive: {verdict['release_ci_excludes_zero_positive']}",
        f"- M3 supports direction (positive mean Δ on either onset or release): {verdict['m3_supports_direction']}",
    ]
    if verdict["safety_regression_reasons"]:
        lines.append("- Safety regression flags:")
        for r in verdict["safety_regression_reasons"]:
            lines.append(f"  - {r}")
    else:
        lines.append("- Safety bounds: OK")

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
