"""Inspect InterAct dataset format across all subsets.

InterAct organizes 4 core subsets (chairs, imhd, neuraldome, omomo_correct_v2)
plus annotation subtrees with a consistent schema:

    <subset>/
        objects/<obj_id>/          # Object meshes (.obj), sampled points (.npy)
        sequences_canonical/<seq_id>/
            human.npz              # SMPL-X body parameters
            object.npz             # Object pose trajectory
            joints.npy             # (optional) precomputed joint positions
            motion.npy             # (optional) possibly already HumanML3D-encoded
            markers.npy            # (optional) mocap markers
            text.txt               # Natural language description
            action.txt / .npy      # Action label

This script enumerates keys, shapes, and samples of every artifact so we
can decide how to map each subset into PIANO's format.

Usage:
    piano-check-interact [--data-dir <path>] [--output-dir <path>]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from piano.utils.io_utils import ensure_dir, save_json


SUBSETS: list[str] = ["chairs", "imhd", "neuraldome", "omomo_correct_v2"]
ANNOTATION_KINDS: list[str] = ["action", "natural", "raw", "change", "shorten"]


def inspect_npz(path: Path) -> dict:
    """Return keys + shapes of a .npz file."""
    if not path.exists():
        return {"exists": False, "path": str(path)}
    try:
        data = np.load(path, allow_pickle=True)
        fields: dict[str, str] = {}
        for k in data.files:
            try:
                arr = data[k]
                fields[k] = f"shape={tuple(arr.shape)} dtype={arr.dtype}"
            except Exception as e:
                fields[k] = f"<error: {e}>"
        return {"exists": True, "fields": fields}
    except Exception as e:
        return {"exists": True, "error": str(e)}


def inspect_npy(path: Path) -> dict:
    """Return shape + dtype of a .npy file."""
    if not path.exists():
        return {"exists": False}
    try:
        arr = np.load(path, allow_pickle=False)
        return {"exists": True, "shape": list(arr.shape), "dtype": str(arr.dtype)}
    except Exception as e:
        return {"exists": True, "error": str(e)}


def inspect_text(path: Path, max_chars: int = 300) -> dict:
    """Read first chunk of a text file."""
    if not path.exists():
        return {"exists": False}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        return {
            "exists": True,
            "length": len(text),
            "preview": text[:max_chars],
        }
    except Exception as e:
        return {"exists": True, "error": str(e)}


def inspect_sequence(seq_dir: Path) -> dict:
    """Inspect one sequence folder."""
    result = {"seq_id": seq_dir.name}
    for name in ("human.npz", "object.npz"):
        result[name] = inspect_npz(seq_dir / name)
    for name in ("joints.npy", "motion.npy", "markers.npy", "action.npy"):
        result[name] = inspect_npy(seq_dir / name)
    for name in ("text.txt", "action.txt"):
        result[name] = inspect_text(seq_dir / name, max_chars=250)
    return result


def inspect_object_dir(obj_dir: Path) -> dict:
    """List files in one object directory."""
    if not obj_dir.exists():
        return {"exists": False}
    entries: list[dict] = []
    for p in sorted(obj_dir.iterdir()):
        entries.append({
            "name": p.name,
            "size_bytes": p.stat().st_size,
        })
    return {"exists": True, "entries": entries}


def inspect_annotation_kind(ann_dir: Path) -> dict:
    """Inspect one annotation subdirectory (e.g. annotation/action/)."""
    if not ann_dir.exists():
        return {"exists": False, "path": str(ann_dir)}
    files = sorted(ann_dir.iterdir())
    result = {
        "exists": True,
        "path": str(ann_dir),
        "num_files": len(files),
    }
    if files:
        sample = files[0]
        result["sample_filename"] = sample.name
        # Try as text, fall back to numpy
        try:
            result["sample_text_preview"] = sample.read_text(encoding="utf-8", errors="replace")[:300]
        except Exception:
            try:
                arr = np.load(sample, allow_pickle=True)
                if hasattr(arr, "shape"):
                    result["sample_numpy_shape"] = list(arr.shape)
                    result["sample_numpy_dtype"] = str(arr.dtype)
            except Exception as e:
                result["sample_error"] = str(e)
    return result


def run_inspection(data_dir: Path, output_dir: Path) -> None:
    """Inspect InterAct dataset and write a structured summary."""
    output_dir = ensure_dir(output_dir)

    print("=" * 72)
    print(f"InterAct format inspection: {data_dir}")
    print("=" * 72)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    summary: dict = {
        "timestamp": datetime.now().isoformat(),
        "data_dir": str(data_dir),
        "subsets": {},
        "annotations": {},
    }

    # --- Per-subset inspection ---
    for subset in SUBSETS:
        subset_dir = data_dir / subset
        print(f"\n{'=' * 72}")
        print(f"Subset: {subset}")
        print("=" * 72)

        if not subset_dir.exists():
            print(f"  MISSING")
            summary["subsets"][subset] = {"exists": False}
            continue

        # Count sequences and objects
        seq_dir = subset_dir / "sequences_canonical"
        obj_dir = subset_dir / "objects"
        seqs = sorted(seq_dir.iterdir()) if seq_dir.exists() else []
        objs = sorted(obj_dir.iterdir()) if obj_dir.exists() else []

        print(f"  sequences_canonical: {len(seqs)} entries")
        print(f"  objects:             {len(objs)} entries")

        sub_summary: dict = {
            "exists": True,
            "num_sequences": len(seqs),
            "num_objects": len(objs),
            "object_names_preview": [o.name for o in objs[:10]],
        }

        # Inspect first sequence in detail
        if seqs:
            first_seq = inspect_sequence(seqs[0])
            sub_summary["sample_sequence"] = first_seq
            print(f"\n  Sample sequence: {first_seq['seq_id']}")
            for key in ("human.npz", "object.npz"):
                info = first_seq[key]
                if info.get("exists") and "fields" in info:
                    print(f"    {key}:")
                    for fname, fshape in info["fields"].items():
                        print(f"      {fname:22s} {fshape}")
            for key in ("joints.npy", "motion.npy", "markers.npy", "action.npy"):
                info = first_seq[key]
                if info.get("exists") and "shape" in info:
                    print(f"    {key:15s} shape={info['shape']} dtype={info['dtype']}")
                elif not info.get("exists"):
                    print(f"    {key:15s} (absent)")
            for key in ("text.txt", "action.txt"):
                info = first_seq[key]
                if info.get("exists"):
                    print(f"    {key}: {info.get('preview', '')[:100]!r}")

        # Inspect first object directory
        if objs:
            obj_info = inspect_object_dir(objs[0])
            sub_summary["sample_object"] = {
                "name": objs[0].name,
                "entries": obj_info.get("entries", []),
            }
            print(f"\n  Sample object: {objs[0].name}")
            for e in obj_info.get("entries", [])[:8]:
                print(f"    {e['name']:40s} {e['size_bytes']:>10} bytes")

        summary["subsets"][subset] = sub_summary

    # --- Annotation inspection ---
    print(f"\n{'=' * 72}")
    print("Annotations")
    print("=" * 72)
    for kind in ANNOTATION_KINDS:
        ann_dir = data_dir / "annotation" / kind
        info = inspect_annotation_kind(ann_dir)
        summary["annotations"][kind] = info
        if info.get("exists"):
            print(f"\n  {kind}/ ({info['num_files']} files)")
            print(f"    sample: {info.get('sample_filename', 'n/a')}")
            if "sample_text_preview" in info:
                preview = info["sample_text_preview"].replace("\n", " ")[:200]
                print(f"    preview: {preview!r}")
            if "sample_numpy_shape" in info:
                print(f"    numpy: shape={info['sample_numpy_shape']} dtype={info['sample_numpy_dtype']}")
        else:
            print(f"  {kind}/ MISSING")

    # --- Save summary ---
    save_json(output_dir / "summary.json", summary)
    print(f"\n{'=' * 72}")
    print(f"Inspection complete. Summary: {output_dir / 'summary.json'}")
    print("=" * 72)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path("/media/gpu-server-1/4TB_for_data/Cai/datasets/InterAct/InterAct"),
        help="Root InterAct directory (containing chairs/, imhd/, etc.)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: runs/checks/interact_format/<timestamp>/)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        output_dir = Path("runs/checks/interact_format") / timestamp
    else:
        output_dir = args.output_dir
    run_inspection(args.data_dir, output_dir)


if __name__ == "__main__":
    main()
