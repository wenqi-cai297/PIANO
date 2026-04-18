"""Sanity check: verify HOIDataset can load preprocessed data.

Loads a batch via ``HOIDataset`` + ``collate_hoi`` and inspects shapes.
Catches mismatches between preprocessing output format and dataset loader.

Usage:
    python -m piano.checks.hoi_dataset --data-dir /path/to/piano/omomo
"""
from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from piano.data.dataset import HOIDataset, collate_hoi


def run_check(data_dir: Path, batch_size: int, max_seq_length: int) -> None:
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
    print(f"\nSingle-sample keys: {sorted(sample.keys())}")
    for key, val in sample.items():
        if hasattr(val, "shape"):
            print(f"  {key:20s} shape={tuple(val.shape)} dtype={val.dtype}")
        else:
            print(f"  {key:20s} = {val!r}"[:80])

    # Load a batch via DataLoader + collate_hoi
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_hoi, num_workers=0,
    )
    batch = next(iter(loader))
    print(f"\nBatch (B={batch_size}) keys: {sorted(batch.keys())}")
    for key, val in batch.items():
        if hasattr(val, "shape"):
            print(f"  {key:20s} shape={tuple(val.shape)} dtype={val.dtype}")
        elif isinstance(val, list):
            preview = [str(v)[:40] for v in val[:3]]
            print(f"  {key:20s} list[{len(val)}] first items: {preview}")

    # Count metadata stats
    n_with_text = sum(1 for m in dataset.metadata if m.get("text"))
    print(f"\nMetadata: {len(dataset.metadata)} entries, {n_with_text} with non-empty text")
    splits = {m.get("split") for m in dataset.metadata}
    print(f"Splits present: {splits}")

    print("\n" + "=" * 72)
    print("SUCCESS: HOIDataset loads preprocessed OMOMO data cleanly.")
    print("=" * 72)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--data-dir", type=Path, required=True,
        help="PIANO-formatted dataset root (containing metadata.json, motions/, objects/)",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=196)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_check(args.data_dir, args.batch_size, args.max_seq_length)


if __name__ == "__main__":
    main()
