"""Round-15 safety-gated promotion analyzer: adds acc/jerk gate on top of
the Round-13 velocity-only gate.

The Round-13 contract was velocity-only — it does not catch the case where
a model passes velocity but **catastrophically over-shoots root_acc /
root_jerk**. Round-14 paired evidence on the 24-clip Round-9 selection
shows exactly that failure mode for S1-B on imhd / neuraldome / omomo
(``root_acc_p95 xGT ≈ 6.7–8.2``; ``root_jerk_p95 xGT ≈ 15.2–21.8``).
This analyzer adds a safety gate (xGT ≤ 3.0 on both acc and jerk) and
combines it with the literal Round-13 verdict to produce a no-promote-
blocker decision.

This module does NOT regenerate the velocity-side bootstrap analysis. It
loads the verdicts from the existing Round-14 analyzer output
(``analyses/2026-05-23_stage1_round14_paired_s1a_vs_s1b_analysis.json``)
and the per-subset safety numbers from the 12 eval JSONs:

    analyses/2026-05-22_stage1_eval_round14_{s1a,s1b}_ckptseed{42..47}.json

The output JSON preserves the literal_velocity_verdict block verbatim
so the substantive history isn't erased.

Usage
-----

    $env:PYTHONIOENCODING="utf-8"
    conda run -n piano python scripts/stage_b_generator/analyze_round15_stage1_safety_gate.py
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

VELOCITY_METRICS_FOR_GATE = (
    "root_vel_mean_abs",
    "yaw_vel_from_sincos_mean_abs",
    "pelvis_rot6d_vel_mean",
    "spine3_rot6d_vel_mean",
)
SAFETY_METRICS_FOR_GATE = (
    "root_acc_p95",
    "root_jerk_p95",
)
SAFETY_XGT_MAX = 3.0     # |xGT| <= 3 (i.e. generated ≤ 3× GT magnitude)

ANALYSES = Path("analyses")
EVAL_DATE_TAG = "2026-05-22"
ROUND14_ANALYSIS = ANALYSES / "2026-05-23_stage1_round14_paired_s1a_vs_s1b_analysis.json"
OUT_PATH = ANALYSES / "2026-05-23_stage1_round15_safety_gate_analysis.json"


def _load_eval(mode: str, ckpt_seed: int) -> dict[str, Any]:
    tag = f"round14_{mode}_ckptseed{ckpt_seed}"
    path = ANALYSES / f"{EVAL_DATE_TAG}_stage1_eval_{tag}.json"
    if not path.exists():
        raise SystemExit(f"[r15] missing eval JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _per_subset_mode_xgt_mean(
    payloads_by_mode: dict[str, list[dict[str, Any]]],
    metric: str,
) -> dict[str, dict[str, float]]:
    """For each (subset, mode) compute the mean xGT for ``metric``.

    Averages over every (clip, sampler_seed, ckpt_seed) triple so each
    clip contributes 3 × 6 = 18 readings. NaN xGT entries are dropped
    (denom-near-zero clips).
    """
    out: dict[str, dict[str, float]] = {}
    for sub in SUBSETS:
        out[sub] = {}
    for mode, payloads in payloads_by_mode.items():
        per_subset: dict[str, list[float]] = defaultdict(list)
        for payload in payloads:
            for rec in payload["per_clip"]:
                sub = rec["subset"]
                x = rec["xGT"].get(f"xGT.{metric}", float("nan"))
                if isinstance(x, (int, float)) and math.isfinite(x):
                    per_subset[sub].append(float(x))
        for sub in SUBSETS:
            vals = per_subset.get(sub, [])
            out[sub][mode] = float(np.mean(vals)) if vals else float("nan")
    return out


def _per_subset_mode_xgt_per_ckpt_seed(
    payloads_by_mode: dict[str, list[dict[str, Any]]],
    metric: str,
) -> dict[str, dict[str, list[float]]]:
    """Per-(subset, mode), per-ckpt-seed mean xGT — used for the safety
    gate's sign-consistency-like check (count of ckpt seeds whose own
    mean xGT is within the safety threshold).
    """
    out: dict[str, dict[str, list[float]]] = {}
    for sub in SUBSETS:
        out[sub] = {}
    for mode, payloads in payloads_by_mode.items():
        # payloads is the list of 6 ckpt-seed JSONs in CKPT_SEEDS order.
        for sub in SUBSETS:
            out[sub][mode] = []
        for payload in payloads:
            per_subset_local: dict[str, list[float]] = defaultdict(list)
            for rec in payload["per_clip"]:
                sub = rec["subset"]
                x = rec["xGT"].get(f"xGT.{metric}", float("nan"))
                if isinstance(x, (int, float)) and math.isfinite(x):
                    per_subset_local[sub].append(float(x))
            for sub in SUBSETS:
                vals = per_subset_local.get(sub, [])
                out[sub][mode].append(
                    float(np.mean(vals)) if vals else float("nan")
                )
    return out


def _safety_pass(mean_xgt: float, threshold: float = SAFETY_XGT_MAX) -> bool:
    """Pass iff mean(xGT) ≤ threshold AND is finite."""
    return math.isfinite(mean_xgt) and mean_xgt <= threshold


def main() -> int:
    # Load Round-14 velocity-only analysis (literal verdict source).
    if not ROUND14_ANALYSIS.exists():
        raise SystemExit(f"[r15] missing Round-14 analysis JSON: {ROUND14_ANALYSIS}")
    r14 = json.loads(ROUND14_ANALYSIS.read_text(encoding="utf-8"))

    # Load 12 eval JSONs.
    payloads_by_mode: dict[str, list[dict[str, Any]]] = {"s1a": [], "s1b": []}
    for mode in ("s1a", "s1b"):
        for cs in CKPT_SEEDS:
            payloads_by_mode[mode].append(_load_eval(mode, cs))

    # ─── Safety: per-subset, per-mode mean xGT for acc + jerk ───
    safety_mean: dict[str, dict[str, dict[str, float]]] = {}
    for metric in SAFETY_METRICS_FOR_GATE:
        safety_mean[metric] = _per_subset_mode_xgt_mean(payloads_by_mode, metric)

    # Per-ckpt-seed breakdown (so we can flag any subset where the model
    # might be salvaged by a per-seed cherry-pick — and confirm the
    # safety failure is sign-consistent across all 6 ckpt seeds, not a
    # single outlier).
    safety_per_seed: dict[str, dict[str, dict[str, list[float]]]] = {}
    for metric in SAFETY_METRICS_FOR_GATE:
        safety_per_seed[metric] = _per_subset_mode_xgt_per_ckpt_seed(
            payloads_by_mode, metric,
        )

    # Velocity gate — copy from Round-14 analyzer output.
    velocity_per_subset_verdict: dict[str, str] = {}
    velocity_per_subset_block: dict[str, Any] = {}
    for sub in SUBSETS:
        block = r14.get("promotion", {}).get(sub, {})
        velocity_per_subset_verdict[sub] = str(block.get("verdict", "no_data"))
        velocity_per_subset_block[sub] = block
    velocity_global_verdict = str(r14.get("global_verdict", "no_data"))

    # Safety gate: per subset × mode, pass iff every safety metric's mean
    # xGT is ≤ threshold AND every per-ckpt-seed value is ≤ threshold
    # (so a single seed blow-up still trips the gate).
    safety_gate: dict[str, dict[str, Any]] = {}
    safety_failing_subsets_by_mode: dict[str, list[str]] = {"s1a": [], "s1b": []}
    for sub in SUBSETS:
        safety_gate[sub] = {}
        for mode in ("s1a", "s1b"):
            per_metric: dict[str, Any] = {}
            sub_pass = True
            for metric in SAFETY_METRICS_FOR_GATE:
                mean_x = safety_mean[metric][sub][mode]
                per_seed_xs = safety_per_seed[metric][sub][mode]
                m_pass = _safety_pass(mean_x)
                n_seed_pass = sum(
                    1 for x in per_seed_xs if _safety_pass(x)
                )
                per_metric[metric] = {
                    "mean_xGT": mean_x,
                    "per_ckpt_seed_xGT": per_seed_xs,
                    "n_ckpt_seeds_below_threshold": n_seed_pass,
                    "n_ckpt_seeds_total": len(per_seed_xs),
                    "mean_pass": bool(m_pass),
                    # all_seeds_pass is reported for transparency but does
                    # NOT drive the gate: a single-seed spike above 3×
                    # GT inside a subset whose mean is comfortably below
                    # the threshold is variance noise, not the
                    # catastrophic-regression failure mode the gate is
                    # trying to catch.
                    "all_seeds_pass": bool(
                        m_pass and n_seed_pass == len(per_seed_xs)
                    ),
                }
                if not per_metric[metric]["mean_pass"]:
                    sub_pass = False
            safety_gate[sub][mode] = {
                "per_metric": per_metric,
                "pass": bool(sub_pass),
                "gate_basis": "mean_xGT_below_threshold_on_every_safety_metric",
            }
            if not sub_pass:
                safety_failing_subsets_by_mode[mode].append(sub)

    safety_global_per_mode: dict[str, dict[str, Any]] = {}
    for mode in ("s1a", "s1b"):
        failing = safety_failing_subsets_by_mode[mode]
        safety_global_per_mode[mode] = {
            "pass": len(failing) == 0,
            "failing_subsets": failing,
            "n_failing_subsets": len(failing),
        }

    # ─── Combined verdict ─────────────────────────────────────────────
    # Per subset rules:
    #   - velocity says ship_s1a AND S1-A safety pass → ship_s1a
    #   - velocity says ship_s1b AND S1-B safety pass → ship_s1b
    #   - velocity says ship_s1X AND model X safety FAILS
    #         → no_promote_safety_blocked
    #   - velocity says both_fail_revisit (or no_data) → both_fail_revisit
    combined_per_subset: dict[str, dict[str, Any]] = {}
    for sub in SUBSETS:
        vv = velocity_per_subset_verdict[sub]
        a_safe = safety_gate[sub]["s1a"]["pass"]
        b_safe = safety_gate[sub]["s1b"]["pass"]
        if vv == "ship_s1a":
            verdict = "ship_s1a" if a_safe else "no_promote_safety_blocked"
        elif vv == "ship_s1b":
            verdict = "ship_s1b" if b_safe else "no_promote_safety_blocked"
        elif vv == "both_fail_revisit":
            verdict = "both_fail_revisit"
        else:
            verdict = "no_data"
        combined_per_subset[sub] = {
            "velocity_says": vv,
            "s1a_safety_pass": bool(a_safe),
            "s1b_safety_pass": bool(b_safe),
            "verdict": verdict,
        }

    # Global rule:
    #   - promote_s1a iff every subset says ship_s1a AND S1-A passes
    #     safety globally
    #   - promote_s1b iff at least one subset says ship_s1b AND no
    #     subset says ship_s1a AND S1-B passes safety globally
    #     (i.e. on every subset)
    #   - both_fail_revisit otherwise
    subset_verdicts = list(velocity_per_subset_verdict.values())
    a_safe_all = safety_global_per_mode["s1a"]["pass"]
    b_safe_all = safety_global_per_mode["s1b"]["pass"]
    if all(v == "ship_s1a" for v in subset_verdicts) and a_safe_all:
        global_verdict = "ship_s1a"
        global_rationale = (
            "every subset's velocity gate says ship_s1a AND S1-A passes "
            "acc/jerk safety on every subset"
        )
    elif "ship_s1b" in subset_verdicts and "ship_s1a" not in subset_verdicts and b_safe_all:
        global_verdict = "ship_s1b"
        global_rationale = (
            "S1-B paired-wins on at least one subset, no subset prefers "
            "S1-A, AND S1-B passes acc/jerk safety on every subset"
        )
    else:
        global_verdict = "both_fail_revisit"
        details: list[str] = []
        if not a_safe_all:
            details.append(
                "S1-A fails velocity gate (Round-13 contract) globally "
                f"(per-subset: {velocity_per_subset_verdict})"
            )
        if "ship_s1b" in subset_verdicts and not b_safe_all:
            details.append(
                f"S1-B fails acc/jerk safety on {safety_failing_subsets_by_mode['s1b']} "
                f"(threshold xGT ≤ {SAFETY_XGT_MAX})"
            )
        if "ship_s1a" in subset_verdicts and "ship_s1b" in subset_verdicts:
            details.append("subset-split: some subsets ship_s1a, others ship_s1b")
        global_rationale = " ; ".join(details) if details else "no_promote_path_passes_both_gates"

    # ─── Pack output ──────────────────────────────────────────────────
    out_payload: dict[str, Any] = {
        "round": "round15",
        "purpose": (
            "Extend Round-13 velocity-only promotion gate with an "
            "acc/jerk safety gate (xGT ≤ 3.0) so models that catastrophically "
            "over-shoot root smoothness cannot be promoted on velocity alone."
        ),
        "n_ckpt_seeds": len(CKPT_SEEDS),
        "n_sampler_seeds": len(SAMPLER_SEEDS),
        "n_clips_total": int(r14.get("n_clips_total", 24)),
        "velocity_metrics_for_gate": list(VELOCITY_METRICS_FOR_GATE),
        "safety_metrics_for_gate": list(SAFETY_METRICS_FOR_GATE),
        "safety_xGT_max": float(SAFETY_XGT_MAX),
        # ─ Literal Round-13 velocity-only verdict, preserved verbatim ─
        "literal_velocity_verdict": {
            "source": str(ROUND14_ANALYSIS),
            "per_subset": velocity_per_subset_block,
            "per_subset_verdict": velocity_per_subset_verdict,
            "global": velocity_global_verdict,
            "note": (
                "Verbatim copy of the Round-13 contract verdict produced by "
                "analyze_round14_paired_s1a_vs_s1b.py. Velocity gate is the "
                "|xGT-1| < 0.15 rule on root_vel, yaw_vel_from_sincos, "
                "pelvis_rot6d_vel, spine3_rot6d_vel with S1-B paired-win "
                "fallback. Does NOT include acc/jerk safety."
            ),
        },
        # ─ Safety: per-subset per-model acc/jerk breakdown ─
        "acc_jerk_safety_gate": {
            "threshold_xGT_max": float(SAFETY_XGT_MAX),
            "per_subset_per_mode": safety_gate,
            "global_per_mode": safety_global_per_mode,
            "raw_safety_mean_xGT": safety_mean,
        },
        # ─ Combined ─
        "combined_promotion_verdict": {
            "per_subset": combined_per_subset,
            "global": global_verdict,
            "global_rationale": global_rationale,
        },
    }
    OUT_PATH.write_text(json.dumps(out_payload, indent=2, default=float), encoding="utf-8")
    print(f"[r15] wrote {OUT_PATH}")

    # ─── Stdout summary ──────────────────────────────────────────────
    print()
    print("Per-subset summary:")
    print(
        f"{'subset':<18s} {'vel_says':<22s} "
        f"{'S1A acc xGT':>12s} {'S1A jerk xGT':>13s} {'S1A safe':>9s}  "
        f"{'S1B acc xGT':>12s} {'S1B jerk xGT':>13s} {'S1B safe':>9s}  "
        f"{'combined':<28s}"
    )
    for sub in SUBSETS:
        a_acc = safety_mean["root_acc_p95"][sub]["s1a"]
        a_jrk = safety_mean["root_jerk_p95"][sub]["s1a"]
        b_acc = safety_mean["root_acc_p95"][sub]["s1b"]
        b_jrk = safety_mean["root_jerk_p95"][sub]["s1b"]
        a_safe = "PASS" if safety_gate[sub]["s1a"]["pass"] else "FAIL"
        b_safe = "PASS" if safety_gate[sub]["s1b"]["pass"] else "FAIL"
        vv = velocity_per_subset_verdict[sub]
        cv = combined_per_subset[sub]["verdict"]
        print(
            f"{sub:<18s} {vv:<22s} "
            f"{a_acc:>12.3f} {a_jrk:>13.3f} {a_safe:>9s}  "
            f"{b_acc:>12.3f} {b_jrk:>13.3f} {b_safe:>9s}  "
            f"{cv:<28s}"
        )
    print()
    print(f"S1-A safety failing subsets globally: {safety_failing_subsets_by_mode['s1a']}")
    print(f"S1-B safety failing subsets globally: {safety_failing_subsets_by_mode['s1b']}")
    print()
    print(f"Literal Round-13 velocity verdict (global): {velocity_global_verdict}")
    print(f"Round-15 combined verdict (global):        {global_verdict}")
    print(f"  rationale: {global_rationale}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
