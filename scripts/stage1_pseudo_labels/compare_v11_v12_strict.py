"""Compare v11 vs v12_strict pseudo-label frame fractions.

Reads <subset>/pseudo_labels/<seq_id>.npz (v11) and
<subset>/pseudo_labels/v12_strict/<seq_id>.npz (v12 strict), reports
per-subset and per-body-part mean contact-frame-fraction comparisons.

This is the post-extraction sanity check: confirms v12 strict reduced
contact frame fraction substantially (predicted ~30-50% per subset
mesh-based, vs v11 60-95%) without dropping all contact signal.

Usage:
    python scripts/stage1_pseudo_labels/compare_v11_v12_strict.py \\
        --piano-root /media/.../InterAct/piano

    # Or local:
    python scripts/stage1_pseudo_labels/compare_v11_v12_strict.py \\
        --piano-root /path/to/local/piano
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from piano.utils.smpl_utils import BODY_PART_NAMES


def _scan_subset(subset_dir: Path) -> dict:
    v11_dir = subset_dir / "pseudo_labels"
    v12_dir = v11_dir / "v12_strict"
    if not v11_dir.exists() or not v12_dir.exists():
        return {"subset": subset_dir.name, "skip_reason": "missing v11 or v12_strict"}

    v11_files = sorted(v11_dir.glob("*.npz"))
    # Filter to ones for which v12 exists
    v12_files = sorted(v12_dir.glob("*.npz"))
    v12_names = {f.name for f in v12_files}

    n_clips = 0
    v11_contact_total = []
    v12_contact_total = []
    v11_per_part = []
    v12_per_part = []
    n_skipped = 0

    for v11_path in v11_files:
        if v11_path.name not in v12_names:
            continue
        v12_path = v12_dir / v11_path.name
        try:
            v11 = np.load(v11_path, allow_pickle=False)
            v12 = np.load(v12_path, allow_pickle=False)
            v11_c = v11["contact_state"]
            v12_c = v12["contact_state"]
        except Exception:
            n_skipped += 1
            continue
        if v11_c.shape != v12_c.shape:
            n_skipped += 1
            continue
        # Binary "in contact" frames
        v11_b = v11_c > 0.5
        v12_b = v12_c > 0.5
        v11_contact_total.append(float(v11_b.any(axis=1).mean()))
        v12_contact_total.append(float(v12_b.any(axis=1).mean()))
        v11_per_part.append([float(v11_b[:, b].mean()) for b in range(v11_b.shape[1])])
        v12_per_part.append([float(v12_b[:, b].mean()) for b in range(v12_b.shape[1])])
        n_clips += 1

    if n_clips == 0:
        return {"subset": subset_dir.name, "skip_reason": "no overlapping clips"}

    v11_arr = np.array(v11_contact_total)
    v12_arr = np.array(v12_contact_total)
    v11_pp = np.array(v11_per_part).mean(axis=0)
    v12_pp = np.array(v12_per_part).mean(axis=0)

    return {
        "subset": subset_dir.name,
        "n_clips": n_clips,
        "n_skipped": n_skipped,
        "v11_mean_contact_frac": float(v11_arr.mean()),
        "v12_mean_contact_frac": float(v12_arr.mean()),
        "reduction_pct": float(
            (1 - v12_arr.mean() / max(v11_arr.mean(), 1e-6)) * 100
        ),
        "v11_per_part": v11_pp.tolist(),
        "v12_per_part": v12_pp.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--piano-root", type=Path, required=True)
    parser.add_argument(
        "--subsets", nargs="+",
        default=["chairs", "imhd", "neuraldome", "omomo_correct_v2"],
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    results = []
    for subset in args.subsets:
        subset_dir = args.piano_root / subset
        r = _scan_subset(subset_dir)
        results.append(r)

    print(f"{'subset':22} {'n_clips':>8} {'v11_frac':>9} {'v12_frac':>9} {'reduction':>10}")
    print("-" * 65)
    for r in results:
        if "skip_reason" in r:
            print(f"  {r['subset']:20} SKIP: {r['skip_reason']}")
            continue
        print(
            f"  {r['subset']:20} {r['n_clips']:>8} "
            f"{r['v11_mean_contact_frac']*100:>7.2f}%  "
            f"{r['v12_mean_contact_frac']*100:>7.2f}%  "
            f"{r['reduction_pct']:>8.1f}%"
        )

    print()
    print("Per-body-part contact frac (mean over clips):")
    print(f"{'subset':22} {'part':12} {'v11':>7} {'v12':>7}")
    for r in results:
        if "skip_reason" in r:
            continue
        for b, name in enumerate(BODY_PART_NAMES):
            print(
                f"  {r['subset']:20} {name:12} "
                f"{r['v11_per_part'][b]*100:>6.2f}% "
                f"{r['v12_per_part'][b]*100:>6.2f}%"
            )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
