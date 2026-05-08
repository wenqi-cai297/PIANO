"""Offline scan: compute per-group GT std for motion_263 feature groups.

Output: ``analyses/<date>_motion_feature_groups/group_stats.json`` with
per-group GT-std numbers used to:
  1. Initialize feature weights (inverse-variance).
  2. Normalize the per-group RMSE during the adaptive update so the
     "is this group lagging?" comparison is scale-invariant.

Usage:
    python scripts/stage_b_generator/compute_initial_feature_weights.py \\
        --config configs/training/anchordiff_v2_weighted.yaml \\
        --num-clips 500 \\
        --output analyses/2026-05-08_motion_feature_groups
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from piano.data.dataset import HOIDataset, collate_hoi
from piano.training.feature_groups import FEATURE_GROUPS as FEATURE_GROUP_DEFS


GROUPS = [(g.name, g.lo, g.hi) for g in FEATURE_GROUP_DEFS]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num-clips", type=int, default=500)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--w-min", type=float, default=0.5)
    parser.add_argument("--w-max", type=float, default=20.0)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    pseudo_label_subdir = cfg.data.get("pseudo_label_subdir", None)
    datasets = []
    for entry in cfg.data.datasets:
        sub_dir = (str(Path(entry.root) / pseudo_label_subdir)
                   if pseudo_label_subdir is not None else None)
        ds = HOIDataset(
            root=entry.root,
            pseudo_label_dir=sub_dir,
            max_seq_length=cfg.data.max_seq_length,
            augment=None,
            support_collapse_hand_support=True,
            surface_obj_pose=True,
        )
        datasets.append(ds)

    rng = random.Random(42)
    pool = []
    for ds_idx, ds in enumerate(datasets):
        for clip_idx in range(len(ds)):
            pool.append((ds_idx, clip_idx))
    rng.shuffle(pool)
    pool = pool[: min(args.num_clips, len(pool))]

    print(f"Scanning {len(pool)} clips...")

    # Accumulate per-group sum + sum-of-squares + count for std computation.
    group_sum = {n: 0.0 for n, *_ in GROUPS}
    group_sumsq = {n: 0.0 for n, *_ in GROUPS}
    group_count = {n: 0 for n, *_ in GROUPS}

    for k, (ds_idx, clip_idx) in enumerate(pool):
        if k % 50 == 0:
            print(f"  {k}/{len(pool)}")
        sample = datasets[ds_idx][clip_idx]
        seq_len = int(sample["seq_len"].item())
        motion = sample["motion"].numpy()[:seq_len]   # (T, 263)
        for name, lo, hi in GROUPS:
            grp = motion[:, lo:hi].astype(np.float64)
            group_sum[name] += grp.sum()
            group_sumsq[name] += (grp * grp).sum()
            group_count[name] += grp.size

    stats = {}
    for name, lo, hi in GROUPS:
        n = group_count[name]
        mean = group_sum[name] / max(n, 1)
        var = group_sumsq[name] / max(n, 1) - mean * mean
        std = float(np.sqrt(max(var, 0.0)))
        stats[name] = {
            "lo": lo, "hi": hi, "dim": hi - lo,
            "mean": float(mean),
            "std": std,
            "var": float(max(var, 0.0)),
        }

    # Initial inverse-variance weights, clamped + normalized to mean=1.
    inv_var = {n: 1.0 / max(stats[n]["var"], 1e-9) for n, *_ in GROUPS}
    mean_iv = sum(inv_var.values()) / len(inv_var)
    norm = {n: inv_var[n] / mean_iv for n in inv_var}
    clamped = {n: max(min(w, args.w_max), args.w_min) for n, w in norm.items()}
    mean_c = sum(clamped.values()) / len(clamped)
    init_weights = {n: clamped[n] / mean_c for n in clamped}

    summary = {
        "num_clips": len(pool),
        "groups": stats,
        "initial_weights_inverse_variance_clamped_normalized": init_weights,
        "w_min": args.w_min,
        "w_max": args.w_max,
    }
    out_json = out_dir / "group_stats.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nResults written to {out_json}")
    print()
    print(f"{'group':>20} | {'std':>10} | {'1/var':>14} | {'init weight':>12}")
    print("-" * 65)
    for name, *_ in GROUPS:
        s = stats[name]
        iv = inv_var[name]
        w = init_weights[name]
        print(f"{name:>20} | {s['std']:>10.5f} | {iv:>14.2f} | {w:>12.4f}")


if __name__ == "__main__":
    main()
