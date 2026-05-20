#!/usr/bin/env python3
"""Round-19 paired eval — build the fixed clip selection JSON.

All 12 paired runs (6 seeds × Plan A / S1-O) MUST evaluate on the
SAME clip set so per-run metric vectors are paired. This script
samples N clips per subset deterministically from the Stage-1 val
manifest and writes the Round-9-style selection JSON that
``eval_stage1_coarse_prior.py --selection-json ...`` consumes.

Usage
-----

    python scripts/stage_b_generator/build_round19_eval_selection.py \\
        --cache-root cache/stage1_coarse_v1_full \\
        --num-per-subset 8 \\
        --seed 42 \\
        --output analyses/2026-05-20_round19_eval_selection.json

The default 8 clips × 4 subsets = 32 clips, paired with 3 sample
seeds in the eval driver → 96 samples per ckpt per cfg. Bootstrap CI
of paired Δ across 6 training seeds × 32 clips × 3 sample seeds = 576
paired observations is enough for tight CI at the §9.5 metric scale.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-root", type=Path,
        default=Path("cache/stage1_coarse_v1_full"),
        help="Plan A cache root (S1-O cache must have same val manifest).",
    )
    parser.add_argument("--split", default="val", choices=["val", "train"])
    parser.add_argument(
        "--num-per-subset", type=int, default=8,
        help="Number of clips to sample per subset.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Selection RNG seed. The Round-19 paired eval should use 42.",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output JSON path.",
    )
    args = parser.parse_args()

    manifest_path = args.cache_root / f"manifest_{args.split}.jsonl"
    manifest = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"[selection] loaded {len(manifest)} clips from {manifest_path}")

    # Group by subset.
    by_subset: dict[str, list[dict]] = defaultdict(list)
    for r in manifest:
        by_subset[r["subset"]].append(r)
    rng = np.random.default_rng(args.seed)

    selected: list[dict] = []
    for subset in ("chairs", "imhd", "neuraldome", "omomo_correct_v2"):
        pool = by_subset.get(subset, [])
        if not pool:
            print(f"[selection] WARNING — no clips for subset {subset!r}")
            continue
        k = min(args.num_per_subset, len(pool))
        idxs = rng.choice(len(pool), size=k, replace=False)
        for i in idxs:
            r = pool[int(i)]
            selected.append({
                "subset": subset,
                "seq_id": r["seq_id"],
                "bucket": args.split,
                "T": int(r.get("seq_len", 0)),
                "text": r.get("text", "")[:80],
            })
        print(f"[selection] {subset}: {k}/{len(pool)} clips")

    payload = {
        "name": "round19_paired_eval_selection",
        "purpose": (
            "Fixed clip set for Round-19 paired Plan A vs S1-O eval. "
            "All 12 runs (6 seeds × 2 modes) evaluate on these clips. "
            "Per-clip pairs (Plan A, S1-O) at matched (training seed, "
            "clip, sample seed) provide the bootstrap-CI Δ basis."
        ),
        "bucket": args.split,
        "cache_root": str(args.cache_root),
        "selection_seed": int(args.seed),
        "num_per_subset": int(args.num_per_subset),
        "n_total": len(selected),
        "selected": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2), encoding="utf-8",
    )
    print(f"[selection] wrote {args.output}  (n={len(selected)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
