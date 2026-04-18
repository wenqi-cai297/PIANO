"""Sanity check: verify HOIDataset can load preprocessed data.

Loads a batch via ``HOIDataset`` + ``collate_hoi`` and inspects shapes.
Catches mismatches between preprocessing output format and dataset loader.

Usage:
    python -m piano.checks.hoi_dataset --data-dir /path/to/piano/omomo
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from torch.utils.data import DataLoader

from piano.data.dataset import HOIDataset, collate_hoi
from piano.utils.io_utils import ensure_dir, save_json


def run_check(data_dir: Path, batch_size: int, max_seq_length: int, output_dir: Path) -> None:
    output_dir = ensure_dir(output_dir)

    print("=" * 72)
    print(f"HOIDataset sanity check: {data_dir}")
    print("=" * 72)

    dataset = HOIDataset(
        root=data_dir,
        max_seq_length=max_seq_length,
    )
    print(f"\nDataset size: {len(dataset)} sequences")

    # Load one sample directly
    sample = dataset[0]
    sample_shapes: dict[str, str] = {}
    print(f"\nSingle-sample keys: {sorted(sample.keys())}")
    for key, val in sample.items():
        if hasattr(val, "shape"):
            desc = f"shape={tuple(val.shape)} dtype={val.dtype}"
            print(f"  {key:20s} {desc}")
            sample_shapes[key] = desc
        else:
            desc = f"{type(val).__name__}"
            print(f"  {key:20s} = {val!r}"[:80])
            sample_shapes[key] = desc

    # Load a batch via DataLoader + collate_hoi
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_hoi, num_workers=0,
    )
    batch = next(iter(loader))
    batch_shapes: dict[str, str] = {}
    print(f"\nBatch (B={batch_size}) keys: {sorted(batch.keys())}")
    for key, val in batch.items():
        if hasattr(val, "shape"):
            desc = f"shape={tuple(val.shape)} dtype={val.dtype}"
            print(f"  {key:20s} {desc}")
            batch_shapes[key] = desc
        elif isinstance(val, list):
            preview = [str(v)[:40] for v in val[:3]]
            desc = f"list[{len(val)}] first items: {preview}"
            print(f"  {key:20s} {desc}")
            batch_shapes[key] = desc

    # Count metadata stats
    n_with_text = sum(1 for m in dataset.metadata if m.get("text"))
    splits = sorted({m.get("split") for m in dataset.metadata})
    print(f"\nMetadata: {len(dataset.metadata)} entries, {n_with_text} with non-empty text")
    print(f"Splits present: {splits}")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "data_dir": str(data_dir),
        "max_seq_length": max_seq_length,
        "dataset_size": len(dataset),
        "num_with_text": n_with_text,
        "splits": splits,
        "sample_shapes": sample_shapes,
        "batch_shapes": batch_shapes,
        "status": "success",
    }
    save_json(output_dir / "summary.json", summary)

    print("\n" + "=" * 72)
    print(f"SUCCESS. Summary: {output_dir / 'summary.json'}")
    print("=" * 72)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--data-dir", type=Path, required=True,
        help="PIANO-formatted dataset root (containing metadata.json, motions/, objects/)",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=196)
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: runs/checks/hoi_dataset/<timestamp>/)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        output_dir = Path("runs/checks/hoi_dataset") / timestamp
    else:
        output_dir = args.output_dir
    run_check(args.data_dir, args.batch_size, args.max_seq_length, output_dir)


if __name__ == "__main__":
    main()
