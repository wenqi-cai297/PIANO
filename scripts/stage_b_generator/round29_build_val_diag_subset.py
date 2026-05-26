"""Build the heldout-val 48-clip balanced selection used by R29 full-data diag.

R29 full-data ablations train on the full InterAct corpus from scratch
(no v27 warm-start). To measure generalization, every variant's diag
phase runs on TWO 48-clip subsets:

  1. The Round-27 train-bucket balanced subset (in-distribution sanity).
  2. This script's output: a heldout-val balanced subset matching the
     same selection protocol (cross-subset coverage + manipulation /
     walking / chair_contact category balance) but drawn from the val
     bucket.

Implementation: thin wrapper that re-uses round27_build_tier0_train_indices.py
with --bucket val and the canonical output path.

Usage on the server:
    python scripts/stage_b_generator/round29_build_val_diag_subset.py \
        --config configs/training/anchordiff_a2_full_data.yaml

(Any R29 full-data config works as long as it points at the four
InterAct subsets with the same subject_split — they all do.)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
ROOT = _SCRIPTS.parent.parent
DEFAULT_OUTPUT = ROOT / "analyses" / "round29_val_diag_indices_48_balanced.json"
BUILDER = _SCRIPTS / "round27_build_tier0_train_indices.py"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build heldout-val 48-clip balanced selection for R29 diag.",
    )
    parser.add_argument(
        "--config", type=Path, required=True,
        help="Any R29 full-data YAML (its datasets + subject_split drive the scan).",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Output JSON path. Default: {DEFAULT_OUTPUT.relative_to(ROOT)}",
    )
    parser.add_argument("--n-clips", type=int, default=48)
    parser.add_argument("--max-candidates-per-subset", type=int, default=600)
    args = parser.parse_args()

    if not args.config.exists():
        print(f"Config not found: {args.config}", file=sys.stderr)
        return 2

    cmd = [
        sys.executable,
        str(BUILDER),
        "--config", str(args.config),
        "--output", str(args.output),
        "--bucket", "val",
        "--n-clips", str(args.n_clips),
        "--max-candidates-per-subset", str(args.max_candidates_per_subset),
    ]
    print(f"[r29-val-subset] running: {' '.join(cmd)}")
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
