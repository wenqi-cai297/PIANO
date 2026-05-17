"""Round-14 paired-Δ bootstrap analyzer: S1-A (bidirectional) vs S1-B
(block-causal K=16).

Loads the 12 eval JSONs produced by
`scripts/stage_b_generator/eval_stage1_coarse_prior.py` against the
Round-9 fixed selection on `cache/stage1_coarse_v1_full`, pairs them
by ckpt seed (6 pairs: seed 42-47), and reports:

- Per-subset per-metric mean(xGT) for S1-A and S1-B.
- Per-subset per-metric mean(|xGT - 1|)  (= err vs GT).
- Per-subset per-metric paired Δerr  = err(S1A) - err(S1B), averaged
  over the 6 ckpt seed pairs (so positive Δerr means S1-B is closer
  to GT, negative means S1-A is closer).
- Bootstrap 95% CI on mean Δerr, resampling (clip, ckpt_seed) pairs
  with replacement (10 000 resamples).
- Sign consistency: count how many of the 6 ckpt seeds agree with
  the overall Δerr sign (Round-4 variance protocol: ≥4/6 required).
- Round-13 contract promotion verdict per subset and globally.

Within each (ckpt, clip), the 3 sampler seeds are averaged FIRST to
get one xGT per (mode, ckpt_seed, clip). Bootstrap then resamples
(clip, ckpt_seed) pairs.

The "velocity metrics" used for the |xGT − 1| < 0.15 promotion check
are (per Round-13 contract):

    root_vel_mean_abs
    yaw_vel_from_sincos_mean_abs
    pelvis_rot6d_vel_mean
    spine3_rot6d_vel_mean

Usage:

    $env:PYTHONIOENCODING="utf-8"
    conda run -n piano python scripts/stage_b_generator/analyze_round14_paired_s1a_vs_s1b.py
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


CKPT_SEEDS = (42, 43, 44, 45, 46, 47)
SAMPLER_SEEDS = (42, 43, 44)
SUBSETS = ("chairs", "imhd", "neuraldome", "omomo_correct_v2")
METRIC_KEYS = (
    "root_vel_mean_abs", "root_acc_p95", "root_jerk_p95",
    "yaw_range",
    "yaw_vel_from_sincos_mean_abs",
    "yaw_vel_stored_mean_abs",
    "yaw_vel_consistency_error_mean_abs",
    "pelvis_rot6d_vel_mean", "spine3_rot6d_vel_mean",
    "head_height_range", "head_height_vel_mean",
    "shoulder_height_range", "shoulder_height_vel_mean",
)
VELOCITY_METRICS_FOR_GATE = (
    "root_vel_mean_abs",
    "yaw_vel_from_sincos_mean_abs",
    "pelvis_rot6d_vel_mean",
    "spine3_rot6d_vel_mean",
)
PROMOTION_TOLERANCE = 0.15
N_BOOTSTRAP = 10_000
SIGN_CONSISTENCY_THRESHOLD = 4   # ≥4 of 6 ckpt seeds must agree

ANALYSES = Path("analyses")
DATE_TAG = "2026-05-22"   # eval script bakes this date into the JSON filename


def _load_eval(mode: str, ckpt_seed: int) -> dict[str, Any]:
    tag = f"round14_{mode}_ckptseed{ckpt_seed}"
    path = ANALYSES / f"{DATE_TAG}_stage1_eval_{tag}.json"
    if not path.exists():
        raise SystemExit(f"[analyze] missing eval JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _per_ckptseed_clip_xgt(
    eval_payload: dict[str, Any],
) -> dict[tuple[str, str], dict[str, float]]:
    """Average xGT over the 3 sampler seeds per (subset, seq_id) clip.

    Returns ``{(subset, seq_id) -> {metric: mean_xGT_over_sampler_seeds}}``.
    NaN xGT entries (GT denom near zero) are dropped from the average; if
    all 3 sampler seeds are NaN the metric is left as NaN.
    """
    rows = defaultdict(lambda: defaultdict(list))   # (sub, sid) -> metric -> [xGT...]
    for rec in eval_payload["per_clip"]:
        sub = rec["subset"]
        sid = rec["seq_id"]
        for k in METRIC_KEYS:
            x = rec["xGT"].get(f"xGT.{k}", float("nan"))
            if isinstance(x, (int, float)) and math.isfinite(x):
                rows[(sub, sid)][k].append(float(x))
    out: dict[tuple[str, str], dict[str, float]] = {}
    for key, m in rows.items():
        out[key] = {
            k: (float(np.mean(v)) if v else float("nan")) for k, v in m.items()
        }
    return out


def _bootstrap_mean_ci(
    deltas: np.ndarray, n_resamples: int = N_BOOTSTRAP, alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Bootstrap mean + (lo, hi) percentile CI from a 1-D array.

    Returns (mean, lo, hi). NaNs are filtered upstream.
    """
    rng = rng or np.random.default_rng(2026)
    n = len(deltas)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot_means = deltas[idx].mean(axis=1)
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return float(np.mean(deltas)), lo, hi


def main() -> int:
    # Load all 12 eval payloads + per-ckptseed per-clip xGT.
    per_ckptseed_xgt: dict[str, dict[int, dict[tuple[str, str], dict[str, float]]]] = {
        "s1a": {}, "s1b": {},
    }
    for mode in ("s1a", "s1b"):
        for cs in CKPT_SEEDS:
            payload = _load_eval(mode, cs)
            per_ckptseed_xgt[mode][cs] = _per_ckptseed_clip_xgt(payload)

    # Sanity: every ckpt seed × mode should match the same set of (sub, sid) keys.
    base_keys = set(per_ckptseed_xgt["s1a"][CKPT_SEEDS[0]].keys())
    for mode in ("s1a", "s1b"):
        for cs in CKPT_SEEDS:
            assert set(per_ckptseed_xgt[mode][cs].keys()) == base_keys, (
                f"clip-set drift on {mode}/seed{cs}"
            )
    print(f"[analyze] paired across {len(base_keys)} (subset, seq_id) clips × "
          f"{len(CKPT_SEEDS)} ckpt seeds × {len(SAMPLER_SEEDS)} sampler seeds")

    # Group clips by subset.
    by_subset: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for (sub, sid) in base_keys:
        by_subset[sub].append((sub, sid))

    rng = np.random.default_rng(2026)
    report: dict[str, Any] = {
        "n_ckpt_seeds": len(CKPT_SEEDS),
        "n_sampler_seeds": len(SAMPLER_SEEDS),
        "n_clips_total": len(base_keys),
        "n_bootstrap": N_BOOTSTRAP,
        "sign_consistency_threshold": SIGN_CONSISTENCY_THRESHOLD,
        "promotion_tolerance": PROMOTION_TOLERANCE,
        "per_subset": {},
    }

    for sub in SUBSETS:
        clips = by_subset.get(sub, [])
        if not clips:
            continue
        sub_report: dict[str, Any] = {"n_clips": len(clips), "metrics": {}}

        for metric in METRIC_KEYS:
            xgt_a_per_pair: list[float] = []   # per (clip, ckpt_seed)
            xgt_b_per_pair: list[float] = []
            err_a_per_pair: list[float] = []
            err_b_per_pair: list[float] = []
            delta_err_per_pair: list[float] = []
            per_seed_mean_delta: list[float] = []   # for sign consistency

            for cs in CKPT_SEEDS:
                seed_deltas: list[float] = []
                for clip in clips:
                    xa = per_ckptseed_xgt["s1a"][cs][clip].get(metric, float("nan"))
                    xb = per_ckptseed_xgt["s1b"][cs][clip].get(metric, float("nan"))
                    if not (math.isfinite(xa) and math.isfinite(xb)):
                        continue
                    ea = abs(xa - 1.0)
                    eb = abs(xb - 1.0)
                    de = ea - eb        # >0 ⇒ S1-B closer to GT ⇒ S1-B wins on this pair
                    xgt_a_per_pair.append(xa)
                    xgt_b_per_pair.append(xb)
                    err_a_per_pair.append(ea)
                    err_b_per_pair.append(eb)
                    delta_err_per_pair.append(de)
                    seed_deltas.append(de)
                if seed_deltas:
                    per_seed_mean_delta.append(float(np.mean(seed_deltas)))

            if not delta_err_per_pair:
                sub_report["metrics"][metric] = {"n_pairs": 0}
                continue

            arr = np.asarray(delta_err_per_pair, dtype=np.float64)
            mean_d, lo, hi = _bootstrap_mean_ci(arr, rng=rng)

            # Sign consistency: how many of the 6 per-seed mean deltas
            # agree with the sign of the overall mean.
            overall_sign = (mean_d > 0) - (mean_d < 0)
            n_agree = sum(
                ((d > 0) - (d < 0)) == overall_sign and overall_sign != 0
                for d in per_seed_mean_delta
            )

            sub_report["metrics"][metric] = {
                "n_pairs": len(delta_err_per_pair),
                "n_per_seed": len(per_seed_mean_delta),
                "mean_xGT_s1a":      float(np.mean(xgt_a_per_pair)),
                "mean_xGT_s1b":      float(np.mean(xgt_b_per_pair)),
                "mean_err_s1a":      float(np.mean(err_a_per_pair)),
                "mean_err_s1b":      float(np.mean(err_b_per_pair)),
                "mean_delta_err":    mean_d,
                "delta_err_ci_lo":   lo,
                "delta_err_ci_hi":   hi,
                "ci_excludes_zero":  (lo > 0) or (hi < 0),
                "per_seed_mean_delta": per_seed_mean_delta,
                "sign_consistency_count": int(n_agree),
                "sign_consistency_pass": bool(
                    n_agree >= SIGN_CONSISTENCY_THRESHOLD and overall_sign != 0
                ),
                "interpretation": (
                    "S1-B closer to GT"  if mean_d > 0 and lo > 0  else
                    "S1-A closer to GT"  if mean_d < 0 and hi < 0  else
                    "tie / inconclusive"
                ),
            }
        report["per_subset"][sub] = sub_report

    # ─── Per-subset promotion gate on velocity metrics ────────────────────────
    promo: dict[str, dict[str, Any]] = {}
    for sub in SUBSETS:
        if sub not in report["per_subset"]:
            continue
        m_block = report["per_subset"][sub]["metrics"]
        # S1-A passes outright iff every velocity metric has |xGT_A - 1| < 0.15
        # AND CI doesn't say S1-B is significantly closer.
        per_metric_s1a_within_tol: dict[str, bool] = {}
        per_metric_s1b_wins: dict[str, bool] = {}
        for vm in VELOCITY_METRICS_FOR_GATE:
            mm = m_block.get(vm, {})
            if not mm or mm.get("n_pairs", 0) == 0:
                continue
            per_metric_s1a_within_tol[vm] = (
                abs(mm["mean_xGT_s1a"] - 1.0) < PROMOTION_TOLERANCE
            )
            per_metric_s1b_wins[vm] = (
                mm["ci_excludes_zero"]
                and mm["sign_consistency_pass"]
                and mm["mean_delta_err"] > 0     # S1-B closer
            )
        all_a_pass = (
            len(per_metric_s1a_within_tol) > 0
            and all(per_metric_s1a_within_tol.values())
        )
        any_b_wins_with_a_failing = any(
            per_metric_s1b_wins.get(vm, False)
            and not per_metric_s1a_within_tol.get(vm, True)
            for vm in VELOCITY_METRICS_FOR_GATE
        )
        if all_a_pass:
            verdict = "ship_s1a"
        elif any_b_wins_with_a_failing:
            verdict = "ship_s1b"
        else:
            verdict = "both_fail_revisit"
        promo[sub] = {
            "per_metric_s1a_within_tol": per_metric_s1a_within_tol,
            "per_metric_s1b_wins_paired": per_metric_s1b_wins,
            "verdict": verdict,
        }
    report["promotion"] = promo

    # Global verdict: ship S1-A only if every subset is ship_s1a; ship S1-B
    # iff some subset says ship_s1b AND no subset says S1-A is clearly
    # superior beyond tolerance.
    subset_verdicts = [v["verdict"] for v in promo.values()]
    if all(v == "ship_s1a" for v in subset_verdicts):
        global_verdict = "ship_s1a"
    elif "ship_s1b" in subset_verdicts and "ship_s1a" not in subset_verdicts:
        global_verdict = "ship_s1b"
    elif "ship_s1a" in subset_verdicts and "ship_s1b" in subset_verdicts:
        global_verdict = "subset_split_decision_review"
    else:
        global_verdict = "both_fail_revisit"
    report["global_verdict"] = global_verdict

    # ─── Pretty-print + save ───
    out_path = ANALYSES / "2026-05-23_stage1_round14_paired_s1a_vs_s1b_analysis.json"
    out_path.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"[analyze] wrote {out_path}")

    print()
    print("Per-subset promotion verdicts:")
    for sub in SUBSETS:
        v = promo.get(sub, {}).get("verdict", "no_data")
        print(f"  {sub:>18s}  →  {v}")
    print(f"\nGlobal verdict: {global_verdict}\n")

    print("Per-subset, per-metric paired-Δerr (positive ⇒ S1-B closer to GT):")
    for sub in SUBSETS:
        sub_block = report["per_subset"].get(sub, {})
        if not sub_block.get("metrics"):
            continue
        print(f"\n  ── {sub} (n_clips = {sub_block['n_clips']}) ──")
        print(
            f"    {'metric':<38s}  {'xGT_A':>7s}  {'xGT_B':>7s}  "
            f"{'errA':>6s}  {'errB':>6s}  {'Δerr':>7s}  "
            f"{'CI':>21s}  {'signOK':>7s}  {'verdict':<22s}"
        )
        for metric in METRIC_KEYS:
            mm = sub_block["metrics"].get(metric, {})
            if not mm or mm.get("n_pairs", 0) == 0:
                continue
            ci_str = f"[{mm['delta_err_ci_lo']:+.3f}, {mm['delta_err_ci_hi']:+.3f}]"
            sign_str = f"{mm['sign_consistency_count']}/{mm['n_per_seed']}"
            print(
                f"    {metric:<38s}  "
                f"{mm['mean_xGT_s1a']:>7.3f}  {mm['mean_xGT_s1b']:>7.3f}  "
                f"{mm['mean_err_s1a']:>6.3f}  {mm['mean_err_s1b']:>6.3f}  "
                f"{mm['mean_delta_err']:>+7.3f}  "
                f"{ci_str:>21s}  {sign_str:>7s}  {mm['interpretation']:<22s}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
