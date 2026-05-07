"""Apply motion_263 cumsum-drift exclusions to metadata_clean.json.

Reads ``analyses/<date>_motion_263_drift_scan/exclude_drifty_clips.json``
and removes those seq_ids from each subset's metadata_clean.json.

Usage:
    python scripts/stage1_pseudo_labels/apply_drift_exclusions.py \\
        --config configs/training/anchordiff_v1.yaml \\
        --exclusions analyses/2026-05-08_motion_263_drift_scan/exclude_drifty_clips.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from omegaconf import OmegaConf

from piano.utils.io_utils import load_json, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--exclusions", type=str, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    excl = load_json(Path(args.exclusions))

    by_subset_seq_ids: dict[str, set[str]] = defaultdict(set)
    for entry in excl["exclusions"]:
        by_subset_seq_ids[entry["subset"]].add(entry["seq_id"])

    print(f"Loaded {excl['n_excluded']} drifty seq_ids ({excl['threshold_m']*100:.0f} cm threshold)")
    print(f"Per subset: {{ {', '.join(f'{k}:{len(v)}' for k,v in by_subset_seq_ids.items())} }}")
    print()

    total_before = 0
    total_after = 0
    for entry in cfg.data.datasets:
        subset_root = Path(entry.root)
        meta_clean_path = subset_root / "metadata_clean.json"
        if not meta_clean_path.exists():
            print(f"  [{entry.name}] no metadata_clean.json, skipping")
            continue
        meta = load_json(meta_clean_path)
        before = len(meta)
        excl_ids = by_subset_seq_ids.get(entry.name, set())
        kept = [m for m in meta if m.get("seq_id") not in excl_ids]
        after = len(kept)
        removed = before - after
        total_before += before
        total_after += after
        print(f"  [{entry.name}] {before} → {after}  (removed {removed} drifty)")

        if not args.dry_run:
            save_json(meta_clean_path, kept)

    print()
    print(f"Total: {total_before} → {total_after}  (removed {total_before-total_after})")
    if args.dry_run:
        print("(dry run; no files modified)")


if __name__ == "__main__":
    main()
