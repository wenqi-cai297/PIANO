"""Variance / bootstrap-CI analyzer for Stage B route-sensitivity diagnostics.

Reads one or more JSON outputs from
``condition_route_causal_sensitivity_diagnostic.py`` (any sampler / seed) and
produces a statistical reliability report:

* per-seed metric tables for each α
* across-seed mean ± std and sign-consistency counts
* if per-clip metrics are present (``alpha_z_target_level2_per_clip`` /
  ``alpha_hint_level2_per_clip``), bootstrap clips with replacement to get
  95 % CI on the paired Δ vs α=1 baseline for onset, release, far, anchor
* automatic verdict per (perturbation, metric) cell:
    - "actionable"        : CI excludes zero in the favorable direction AND
                            no safety metric regresses beyond threshold
    - "underpowered"      : sign is consistent but CI overlaps zero
    - "not_actionable"    : sign is inconsistent across seeds

Outputs JSON + Markdown.

This is a CPU-only post-hoc analyzer. It does NOT run the model. It does
not change the model. Inference-only α-scaling remains a diagnostic-only
probe; this analyzer just quantifies how much of an observed effect is
seed/clip noise vs route response.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

import numpy as np


# Metrics analyzed for actionability. Safety metrics are tracked but not
# subject to the "favorable direction" check.
PRIMARY_METRICS = ("onset_xgt", "release_xgt", "transition_relvel_xgt")
GEOM_METRICS = ("far_unobserved_error_cm", "anchor_realization_cm")
SAFETY_METRICS = ("body_acc_p95_over_gt", "body_jerk_p95_over_gt")
# A metric is "favorable" when it INCREASES (xGT ratios) except for cm
# distances which are favorable when they DECREASE.
HIGHER_IS_BETTER = set(PRIMARY_METRICS) | {"body_velocity_over_gt", "hand_velocity_over_gt"}
LOWER_IS_BETTER = set(GEOM_METRICS) | set(SAFETY_METRICS)


def _favorable_sign(metric: str) -> int:
    if metric in HIGHER_IS_BETTER:
        return +1
    if metric in LOWER_IS_BETTER:
        return -1
    return +1


def _stats(arr: np.ndarray) -> dict[str, float]:
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "n": 0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "sem": float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0,
        "n": int(arr.size),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _bootstrap_ci(
    paired_deltas: np.ndarray, n_resamples: int = 2000, seed: int = 12345,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Percentile bootstrap CI on the mean of paired Δ values."""
    paired_deltas = paired_deltas[np.isfinite(paired_deltas)]
    if paired_deltas.size == 0:
        return {"mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "p_sign": 1.0, "n": 0}
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(n_resamples), dtype=np.float64)
    n = paired_deltas.size
    for i in range(int(n_resamples)):
        idx = rng.integers(0, n, size=n)
        means[i] = paired_deltas[idx].mean()
    lo = float(np.percentile(means, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))
    # one-sided proportion in unfavorable direction (for diagnostic only):
    # this is NOT a formal p-value; just a sign-consistency indicator.
    mean_val = float(paired_deltas.mean())
    if mean_val >= 0:
        p_sign = float((means <= 0).mean())
    else:
        p_sign = float((means >= 0).mean())
    return {
        "mean": mean_val,
        "ci_lo": lo,
        "ci_hi": hi,
        "p_sign": p_sign,
        "n": int(n),
    }


def _verdict(
    metric: str,
    boot_lo: float,
    boot_hi: float,
    sign_count: int,
    sign_total: int,
    safety_violation: bool,
) -> str:
    """One-cell verdict per (metric, perturbation)."""
    fav = _favorable_sign(metric)
    if safety_violation:
        return "not_actionable_safety_regression"
    # CI on Δ excludes zero in the favorable direction
    if fav > 0 and boot_lo > 0:
        ci_signal = "actionable"
    elif fav < 0 and boot_hi < 0:
        ci_signal = "actionable"
    elif fav > 0 and boot_hi < 0:
        ci_signal = "harmful"
    elif fav < 0 and boot_lo > 0:
        ci_signal = "harmful"
    else:
        ci_signal = "underpowered"
    sign_consistent = sign_total > 0 and (sign_count / sign_total) >= 2.0 / 3.0
    if ci_signal == "actionable" and sign_consistent:
        return "actionable"
    if ci_signal == "harmful":
        return "harmful"
    if ci_signal == "underpowered" and sign_consistent:
        return "underpowered_consistent_sign"
    return "not_actionable"


def _load_runs(paths: list[Path]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        d["__path"] = str(p)
        out.append(d)
    return out


def _across_seed_table(
    runs: list[dict[str, Any]],
    level2_key: str,
    metrics: tuple[str, ...],
) -> dict[str, Any]:
    """Stack per-seed Level 2 aggregate rows by α. Returns a dict per α."""
    by_alpha: dict[str, dict[str, list[float]]] = {}
    seed_list: list[int] = []
    for run in runs:
        rows = run.get(level2_key, [])
        seed = int(run.get("seed", -1))
        seed_list.append(seed)
        for row in rows:
            a = f"{float(row['alpha']):.4f}"
            cell = by_alpha.setdefault(a, {})
            for m in metrics:
                cell.setdefault(m, []).append(float(row.get(m, np.nan)))
    out: dict[str, Any] = {"seeds": seed_list, "per_alpha": {}}
    for a, mdict in by_alpha.items():
        out["per_alpha"][a] = {
            m: _stats(np.asarray(v, dtype=np.float64)) for m, v in mdict.items()
        }
    return out


def _find_alpha_key(per_clip: dict[str, Any], target: float) -> str | None:
    """Find the per-clip dict key matching ``target`` alpha, tolerating
    ``str(0.0)='0.0'`` vs ``f"{0.0:.4f}"='0.0000'`` mismatches."""
    for k in per_clip.keys():
        try:
            if abs(float(k) - float(target)) < 1e-6:
                return k
        except ValueError:
            continue
    return None


def _paired_per_clip_deltas(
    runs: list[dict[str, Any]],
    per_clip_key: str,
    alpha_value: float,
    baseline_value: float,
    metric: str,
) -> np.ndarray:
    """Stack (seed × clip) paired Δ = metric(α) - metric(baseline)."""
    deltas: list[float] = []
    for run in runs:
        per_clip = run.get(per_clip_key, {})
        ka = _find_alpha_key(per_clip, alpha_value)
        kb = _find_alpha_key(per_clip, baseline_value)
        if ka is None or kb is None:
            continue
        rows_a = per_clip[ka]
        rows_b = per_clip[kb]
        if not rows_a or not rows_b:
            continue
        by_b = {row.get("seq_id", str(i)): row for i, row in enumerate(rows_b)}
        for row in rows_a:
            sid = row.get("seq_id", None)
            if sid is None or sid not in by_b:
                continue
            va = float(row.get(metric, np.nan))
            vb = float(by_b[sid].get(metric, np.nan))
            if np.isfinite(va) and np.isfinite(vb):
                deltas.append(va - vb)
    return np.asarray(deltas, dtype=np.float64)


def _sign_consistency(
    runs: list[dict[str, Any]],
    level2_key: str,
    alpha_value: float,
    baseline_value: float,
    metric: str,
) -> tuple[int, int]:
    """How many seeds show the favorable sign on (metric(α) - metric(baseline))."""
    fav = _favorable_sign(metric)
    good = 0
    total = 0
    for run in runs:
        rows = run.get(level2_key, [])
        v_a = v_b = None
        for row in rows:
            if abs(float(row["alpha"]) - alpha_value) < 1e-6:
                v_a = float(row.get(metric, np.nan))
            if abs(float(row["alpha"]) - baseline_value) < 1e-6:
                v_b = float(row.get(metric, np.nan))
        if v_a is None or v_b is None:
            continue
        if not (np.isfinite(v_a) and np.isfinite(v_b)):
            continue
        total += 1
        delta = v_a - v_b
        if fav > 0 and delta > 0:
            good += 1
        elif fav < 0 and delta < 0:
            good += 1
    return good, total


def _analyze_perturbation(
    runs: list[dict[str, Any]],
    *,
    label: str,
    level2_key: str,
    per_clip_key: str | None,
    baseline_alpha: float,
    target_alphas: list[float],
    n_bootstrap: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {"label": label, "by_alpha": {}}
    all_metrics = list(PRIMARY_METRICS) + list(GEOM_METRICS) + list(SAFETY_METRICS) + [
        "body_velocity_over_gt", "hand_velocity_over_gt",
    ]
    across = _across_seed_table(runs, level2_key, tuple(all_metrics))
    out["across_seeds"] = across

    for a in target_alphas:
        if abs(a - baseline_alpha) < 1e-9:
            continue
        block: dict[str, Any] = {"alpha": a, "baseline_alpha": baseline_alpha, "metrics": {}}
        for m in all_metrics:
            sign_count, sign_total = _sign_consistency(runs, level2_key, a, baseline_alpha, m)
            paired = (
                _paired_per_clip_deltas(runs, per_clip_key, a, baseline_alpha, m)
                if per_clip_key is not None
                else np.empty(0, dtype=np.float64)
            )
            boot = _bootstrap_ci(paired, n_resamples=int(n_bootstrap))
            block["metrics"][m] = {
                "sign_consistent_seeds": int(sign_count),
                "total_seeds": int(sign_total),
                "per_clip_n": int(boot["n"]),
                "paired_delta_mean": boot["mean"],
                "paired_delta_ci_lo": boot["ci_lo"],
                "paired_delta_ci_hi": boot["ci_hi"],
                "sign_consistency_p": boot["p_sign"],
            }
        # safety violation: any safety metric's CI > +0.10 (10% jerk/acc inflation)
        safety_violation = False
        for sm in SAFETY_METRICS:
            cell = block["metrics"].get(sm, {})
            ci_hi = float(cell.get("paired_delta_ci_hi", 0.0))
            if ci_hi > 0.10:
                safety_violation = True
                break
        # primary verdicts
        verdicts: dict[str, str] = {}
        for m in PRIMARY_METRICS + GEOM_METRICS:
            cell = block["metrics"][m]
            verdicts[m] = _verdict(
                m,
                boot_lo=float(cell["paired_delta_ci_lo"]),
                boot_hi=float(cell["paired_delta_ci_hi"]),
                sign_count=int(cell["sign_consistent_seeds"]),
                sign_total=int(cell["total_seeds"]),
                safety_violation=safety_violation,
            )
        block["verdicts"] = verdicts
        # overall actionability — actionable iff onset AND release are actionable
        onset_v = verdicts.get("onset_xgt", "not_actionable")
        release_v = verdicts.get("release_xgt", "not_actionable")
        if onset_v == "actionable" and release_v == "actionable":
            block["overall_verdict"] = "actionable"
        elif onset_v == "harmful" or release_v == "harmful":
            block["overall_verdict"] = "harmful"
        elif "underpowered" in onset_v or "underpowered" in release_v:
            block["overall_verdict"] = "underpowered"
        else:
            block["overall_verdict"] = "not_actionable"
        out["by_alpha"][f"{a:.4f}"] = block
    return out


def _format_block(block: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    lines.append(f"### α = {block['alpha']:.2f} vs baseline α = {block['baseline_alpha']:.2f}")
    lines.append("")
    lines.append("| metric | sign-consistent seeds | per-clip n | Δ mean | 95% CI | verdict |")
    lines.append("|---|---:|---:|---:|---|---|")
    for m, cell in block["metrics"].items():
        v = block["verdicts"].get(m, "—")
        lines.append(
            f"| {m} | {cell['sign_consistent_seeds']}/{cell['total_seeds']} | "
            f"{cell['per_clip_n']} | "
            f"{cell['paired_delta_mean']:+.4f} | "
            f"[{cell['paired_delta_ci_lo']:+.4f}, {cell['paired_delta_ci_hi']:+.4f}] | {v} |"
        )
    lines.append("")
    lines.append(f"**Overall verdict**: `{block['overall_verdict']}`")
    lines.append("")
    return lines


def _write_md(payload: dict[str, Any], path: Path) -> None:
    lines: list[str] = [
        "# Diagnostic Variance / Bootstrap-CI Analysis",
        "",
        f"- Input JSON files: {len(payload['inputs'])}",
        f"- Samplers seen: {sorted(set(payload['inputs_meta']['samplers']))}",
        f"- Seeds: {sorted(set(payload['inputs_meta']['seeds']))}",
        f"- Bootstrap resamples: {payload['n_bootstrap']}",
        "",
        "## Methodology",
        "",
        "For each perturbation (α_hint or α_z_target), the analyzer stacks per-seed "
        "Level-2 rollout aggregates and per-clip Level-2 metrics across input runs. "
        "Sign-consistency counts how many seeds show the favorable direction on a "
        "metric. The per-clip 95% percentile bootstrap CI is computed on the paired "
        "Δ = metric(α) − metric(baseline=1). A cell is `actionable` only if the CI "
        "excludes zero in the favorable direction AND at least 2/3 seeds agree on "
        "the sign AND no safety metric (acc p95 xGT, jerk p95 xGT) shows CI upper "
        "bound > 0.10.",
        "",
        "Caveat: inference-only α-scaling is a diagnostic probe; this analyzer "
        "quantifies whether observed effects survive seed/clip variance. It does "
        "not by itself justify training or inference changes.",
        "",
    ]
    for pert_label, pert_block in payload["perturbations"].items():
        lines.append(f"## Perturbation: {pert_label}")
        lines.append("")
        for _a_str, block in pert_block["by_alpha"].items():
            lines.extend(_format_block(block))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs", nargs="+", required=True,
        help="JSON files from condition_route_causal_sensitivity_diagnostic.py. "
             "Globs allowed.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--md", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    args = parser.parse_args()

    paths: list[Path] = []
    for pattern in args.inputs:
        matches = sorted(Path(p) for p in glob.glob(pattern))
        if matches:
            paths.extend(matches)
        else:
            p = Path(pattern)
            if p.exists():
                paths.append(p)
    if not paths:
        raise SystemExit(f"No input JSON files matched: {args.inputs}")

    runs = _load_runs(paths)

    samplers = [str(r.get("sampler", "ddpm")) for r in runs]
    seeds = [int(r.get("seed", -1)) for r in runs]

    # Detect α grids from the data
    z_alphas: set[float] = set()
    hint_alphas: set[float] = set()
    for r in runs:
        for row in r.get("alpha_z_target_level2", []):
            z_alphas.add(float(row["alpha"]))
        for row in r.get("alpha_hint_level2", []):
            hint_alphas.add(float(row["alpha"]))

    perturbations: dict[str, Any] = {}
    if z_alphas:
        perturbations["alpha_z_target_level2"] = _analyze_perturbation(
            runs,
            label="alpha_z_target_level2",
            level2_key="alpha_z_target_level2",
            per_clip_key="alpha_z_target_level2_per_clip",
            baseline_alpha=1.0,
            target_alphas=sorted(z_alphas),
            n_bootstrap=int(args.n_bootstrap),
        )
    if hint_alphas:
        perturbations["alpha_hint_level2"] = _analyze_perturbation(
            runs,
            label="alpha_hint_level2",
            level2_key="alpha_hint_level2",
            per_clip_key="alpha_hint_level2_per_clip",
            baseline_alpha=1.0,
            target_alphas=sorted(hint_alphas),
            n_bootstrap=int(args.n_bootstrap),
        )

    payload = {
        "inputs": [str(p) for p in paths],
        "inputs_meta": {"samplers": samplers, "seeds": seeds},
        "n_bootstrap": int(args.n_bootstrap),
        "perturbations": perturbations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    _write_md(payload, args.md)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.md}")


if __name__ == "__main__":
    main()
