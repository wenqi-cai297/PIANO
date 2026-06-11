"""Dump a full ``(subset, seq_id)`` selection JSON for one bucket.

Used by the R43 P0 pipeline to drive ``sample_substitute_conds_cli.py``
over the entire train or val split (the existing 48-clip diagnostic
selections are too small for Stage-1.5 training).

The output format matches the ``selected`` schema accepted by
``sample_substitute_conds._read_selection``:

    {"selected": [{"subset": "...", "seq_id": "..."}, ...]}

Usage
-----

    python scripts/stage_a_generator/dump_full_selection_json.py \\
        --config configs/training/stage1_r41_a2_world_vel.yaml \\
        --bucket train \\
        --out-json analyses/round43_full_selection_train.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

from piano.training.train_anchordiff import _build_dataset


def _enumerate_clips(cfg, bucket: str) -> list[dict[str, str]]:
    """Iterate the dataset for ``bucket`` and pull (subset, seq_id) pairs.

    Dataset items expose ``subset`` and ``seq_id`` as plain strings
    (dataset.py:608-609). We don't decode tensors — only string fields
    are needed.
    """
    dataset = _build_dataset(cfg, bucket=bucket, augment=False)
    print(f"[dump] bucket={bucket} dataset size={len(dataset)}", flush=True)

    seen: set[tuple[str, str]] = set()
    rows: list[dict[str, str]] = []
    n_dup = 0
    for i in range(len(dataset)):
        sample = dataset[i]
        subset = str(sample["subset"])
        seq_id = str(sample["seq_id"])
        key = (subset, seq_id)
        if key in seen:
            n_dup += 1
            continue
        seen.add(key)
        rows.append({"subset": subset, "seq_id": seq_id})
        if (i + 1) % 500 == 0:
            print(
                f"[dump]   visited {i + 1}/{len(dataset)}  unique={len(rows)}",
                flush=True,
            )
    if n_dup:
        print(
            f"[dump] de-duplicated {n_dup} entries within bucket={bucket} "
            "(legitimate: the dataset can re-emit the same clip when "
            "augmentation produces multiple views).",
            flush=True,
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Dump a full (subset, seq_id) selection JSON for one bucket. "
            "Output is consumed by sample_substitute_conds_cli.py "
            "--selection-json to sample the whole bucket."
        ),
    )
    ap.add_argument(
        "--config", type=Path, required=True,
        help=(
            "Any Stage-1 or Stage-1.5 cfg that resolves the dataset "
            "roots + subject split for this project. The data-only "
            "fields (datasets, subject_split, pseudo_label_subdir, "
            "max_seq_length) are read; model + loss sections are "
            "ignored. The R43 P0 pipeline passes the A2 Stage-1 cfg."
        ),
    )
    ap.add_argument(
        "--bucket", choices=["train", "val"], required=True,
    )
    ap.add_argument(
        "--out-json", type=Path, required=True,
    )
    ap.add_argument(
        "--limit", type=int, default=0,
        help="Optional: cap rows for debugging. 0 = no limit.",
    )
    args = ap.parse_args()

    if not args.config.is_file():
        print(f"[dump] FATAL: --config {args.config} not found", file=sys.stderr)
        return 1
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(str(args.config))
    rows = _enumerate_clips(cfg, args.bucket)
    if args.limit > 0:
        rows = rows[: args.limit]
        print(f"[dump] --limit {args.limit} applied; emitting {len(rows)} rows")

    payload = {"selected": rows}
    args.out_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"[dump] wrote {args.out_json} ({len(rows)} unique clips for "
        f"bucket={args.bucket}).",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
