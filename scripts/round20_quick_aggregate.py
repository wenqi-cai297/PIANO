"""Quick paired-delta aggregator for Round-20 eval JSONs.

Prints a compact table of:
  - root_traj_dtw_xz (primary direction, lower=better)
  - root_pos_err_xz (lower=better)
  - direction_alignment (higher=better)
  - endpoint_direction_cos (higher=better)
  - xGT.root_vel_mean_abs (legacy primary, higher closer to 1=better;
    delta computed as |1 - mean_xGT_S1O| - |1 - mean_xGT_PlanA| with
    'lower = closer to 1' convention; ship if delta < 0)
  - xGT.root_acc_p95 (safety gate; both must be <1.5 per Round-15)

Cluster-bootstrap CI by (subset, seq_id) — drops the 3 sampler-seed
inflation. tuple bootstrap CI also reported.

Usage:
    python scripts/round20_quick_aggregate.py
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

EVAL_DIR = Path("analyses/analyses/round20_eval")
FNAME_RE = re.compile(
    r"^stage1_(?P<mode>s1a_cmc|s1o)_round20_seed(?P<seed>\d+)"
    r"__(?P<ckpt>[^_]+(?:_[^_]+)*)__cfg(?P<cfg>[\d_]+)\.json$"
)

# (metric_path, direction, label)
METRICS = [
    ("paired|root_traj_dtw_xz", "lower", "root_traj_dtw_xz"),
    ("paired|root_pos_err_xz", "lower", "root_pos_err_xz"),
    ("paired|direction_alignment", "higher", "direction_alignment"),
    ("paired|endpoint_direction_cos", "higher", "endpoint_direction_cos"),
    ("xGT|xGT.root_vel_mean_abs", "closer_to_1", "xGT.root_vel_mean_abs"),
    ("xGT|xGT.root_acc_p95", "safety", "xGT.root_acc_p95"),
]


def get_metric(entry: dict, path: str) -> float:
    # "|" splits dict steps; "." kept literal so keys like "xGT.root_vel_mean_abs" work.
    parts = path.split("|")
    cur = entry
    for p in parts:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return float("nan")
    try:
        v = float(cur)
        return v
    except (TypeError, ValueError):
        return float("nan")


def bootstrap_ci(deltas: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 42):
    d = deltas[np.isfinite(deltas)]
    if d.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idxs = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idxs].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def load_eval(path: Path) -> tuple[dict, list[dict]]:
    payload = json.loads(path.read_text("utf-8"))
    return payload, payload["per_clip"]


def main():
    files = sorted(EVAL_DIR.glob("stage1_*round20_seed*__*.json"))
    by_key: dict[tuple, dict[str, list[dict]]] = defaultdict(lambda: {"s1a_cmc": [], "s1o": []})
    for f in files:
        m = FNAME_RE.match(f.name)
        if not m:
            continue
        key = (m["ckpt"], m["cfg"])
        _, per_clip = load_eval(f)
        by_key[key][m["mode"]] = per_clip

    print(f"# Round-20 paired delta — single seed 42, 32 clips × 3 sampler seeds = 96 paired")
    print()

    for (ckpt, cfg) in sorted(by_key.keys()):
        pair = by_key[(ckpt, cfg)]
        if not pair["s1a_cmc"] or not pair["s1o"]:
            print(f"  [SKIP {ckpt} cfg={cfg}] missing mode")
            continue
        cfg_disp = cfg.replace("_", ".")
        print(f"## {ckpt}  cfg={cfg_disp}")
        print(f"  | metric | Plan A mean | S1-O mean | delta | tuple CI95 | cluster CI95 | favors |")
        print(f"  |---|---:|---:|---:|---|---|---|")

        a_by_clip = {(e["subset"], e["seq_id"], e["seed"]): e for e in pair["s1a_cmc"]}
        o_by_clip = {(e["subset"], e["seq_id"], e["seed"]): e for e in pair["s1o"]}
        common = sorted(set(a_by_clip) & set(o_by_clip))
        if not common:
            print("  no paired entries")
            print()
            continue

        for metric_path, direction, label in METRICS:
            a_vals = np.array([get_metric(a_by_clip[k], metric_path) for k in common])
            o_vals = np.array([get_metric(o_by_clip[k], metric_path) for k in common])
            if direction == "closer_to_1":
                a_dev = np.abs(a_vals - 1.0)
                o_dev = np.abs(o_vals - 1.0)
                deltas = o_dev - a_dev
                fav_dir = "lower"
            elif direction == "safety":
                pa_pass = np.nanmean(a_vals < 1.5)
                so_pass = np.nanmean(o_vals < 1.5)
                a_mean = float(np.nanmean(a_vals))
                o_mean = float(np.nanmean(o_vals))
                print(f"  | {label} | {a_mean:.3f} | {o_mean:.3f} | — | — | — | Plan A pass={pa_pass:.2f}, S1-O pass={so_pass:.2f} |")
                continue
            else:
                deltas = o_vals - a_vals
                fav_dir = direction

            a_mean = float(np.nanmean(a_vals))
            o_mean = float(np.nanmean(o_vals))

            mean, lo, hi = bootstrap_ci(deltas)

            # cluster bootstrap: group 3 seeds per clip
            clip_groups: dict = defaultdict(list)
            for k, d in zip(common, deltas):
                clip_groups[(k[0], k[1])].append(d)
            cluster_means = np.array([np.nanmean(v) for v in clip_groups.values()])
            cmean, clo, chi = bootstrap_ci(cluster_means)

            if fav_dir == "lower":
                fav_s1o = lo < 0 and hi < 0
                fav_cluster = clo < 0 and chi < 0
            else:
                fav_s1o = lo > 0 and hi > 0
                fav_cluster = clo > 0 and chi > 0

            fav_str = "S1-O" if fav_cluster else ("S1-O*tuple" if fav_s1o else "—")
            print(
                f"  | {label} | {a_mean:.3f} | {o_mean:.3f} | {mean:+.4f} | "
                f"[{lo:+.3f}, {hi:+.3f}] | [{clo:+.3f}, {chi:+.3f}] | {fav_str} |"
            )
        print()


if __name__ == "__main__":
    main()
