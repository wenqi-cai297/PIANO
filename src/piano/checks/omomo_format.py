"""Inspect OMOMO (CHOIS processed_data) format after download.

Verifies the downloaded data matches what our preprocessing pipeline expects:
    - Sequence pickle loads and has the expected keys
    - Object meshes exist
    - Text annotations are accessible
    - Contact labels present
    - Shapes and dtypes make sense

Usage:
    python -m piano.checks.omomo_format --data-dir data/omomo/processed_data
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


EXPECTED_SEQ_KEYS = {
    "seq_name", "betas", "gender", "trans", "root_orient", "pose_body",
    "obj_trans", "obj_rot", "obj_scale",
}


def inspect_sequence_pickle(pkl_path: Path) -> dict:
    """Load the per-sequence pickle and return a summary dict."""
    with pkl_path.open("rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise TypeError(f"Expected dict at {pkl_path}, got {type(data)}")

    n_seqs = len(data)
    sample_idx = next(iter(data))
    sample = data[sample_idx]

    missing_keys = EXPECTED_SEQ_KEYS - set(sample.keys())
    extra_keys = set(sample.keys()) - EXPECTED_SEQ_KEYS

    # Shapes for the first sample
    shape_report = {}
    for key, val in sample.items():
        if hasattr(val, "shape"):
            shape_report[key] = f"{type(val).__name__} {tuple(val.shape)} {val.dtype}"
        else:
            shape_report[key] = f"{type(val).__name__} = {val!r}"[:80]

    return {
        "num_sequences": n_seqs,
        "sample_seq_name": sample.get("seq_name", sample_idx),
        "missing_expected_keys": sorted(missing_keys),
        "extra_keys": sorted(extra_keys),
        "sample_shapes": shape_report,
    }


def inspect_objects(mesh_dir: Path) -> dict:
    """Count and list object mesh files."""
    if not mesh_dir.exists():
        return {"exists": False, "path": str(mesh_dir)}

    objs = sorted(mesh_dir.glob("*.obj"))
    plys = sorted(mesh_dir.glob("*.ply"))
    unique_names = {p.stem.replace("_cleaned_simplified", "").replace("_top", "").replace("_bottom", "")
                    for p in objs}

    return {
        "exists": True,
        "path": str(mesh_dir),
        "num_obj_files": len(objs),
        "num_ply_files": len(plys),
        "unique_object_names": sorted(unique_names),
    }


def inspect_text(text_dir: Path) -> dict:
    """Sample a text annotation JSON and report coverage."""
    if not text_dir.exists():
        return {"exists": False, "path": str(text_dir)}

    jsons = sorted(text_dir.glob("*.json"))
    if not jsons:
        return {"exists": True, "path": str(text_dir), "num_files": 0}

    sample = json.load(jsons[0].open())
    sample_values = list(sample.values())[:2] if isinstance(sample, dict) else sample

    return {
        "exists": True,
        "path": str(text_dir),
        "num_files": len(jsons),
        "sample_file": jsons[0].name,
        "sample_content": str(sample_values)[:200],
    }


def inspect_contacts(contact_dir: Path) -> dict:
    """Verify contact label files."""
    if not contact_dir.exists():
        return {"exists": False, "path": str(contact_dir)}

    npys = sorted(contact_dir.glob("*.npy"))
    if not npys:
        return {"exists": True, "path": str(contact_dir), "num_files": 0}

    sample = np.load(npys[0])
    return {
        "exists": True,
        "path": str(contact_dir),
        "num_files": len(npys),
        "sample_shape": tuple(sample.shape),
        "sample_dtype": str(sample.dtype),
        "value_range": (float(sample.min()), float(sample.max())),
    }


def run_inspection(data_dir: Path) -> None:
    """Run all format checks and print a report."""
    print("=" * 72)
    print(f"OMOMO format inspection: {data_dir}")
    print("=" * 72)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # --- Sequence pickles ---
    for split in ("train", "test"):
        pkl = data_dir / f"{split}_diffusion_manip_seq_joints24.p"
        print(f"\n[1.{split}] Sequence pickle: {pkl.name}")
        if not pkl.exists():
            print(f"  MISSING: {pkl}")
            continue
        report = inspect_sequence_pickle(pkl)
        print(f"  Number of sequences: {report['num_sequences']}")
        print(f"  Sample seq_name: {report['sample_seq_name']}")
        if report["missing_expected_keys"]:
            print(f"  WARN — missing expected keys: {report['missing_expected_keys']}")
        print(f"  Sample shapes:")
        for k, v in report["sample_shapes"].items():
            print(f"    {k:20s} {v}")
        if report["extra_keys"]:
            print(f"  Extra keys present: {report['extra_keys']}")

    # --- Object meshes ---
    print("\n[2] Object meshes")
    for sub in ("captured_objects", "rest_object_geo"):
        rpt = inspect_objects(data_dir / sub)
        print(f"  {sub}: ", end="")
        if not rpt["exists"]:
            print(f"MISSING ({rpt['path']})")
        else:
            print(f"{rpt['num_obj_files']} .obj, {rpt['num_ply_files']} .ply, {len(rpt['unique_object_names'])} unique objects")
            if rpt["unique_object_names"]:
                print(f"    Names: {', '.join(rpt['unique_object_names'][:10])}{'...' if len(rpt['unique_object_names']) > 10 else ''}")

    # --- Text annotations ---
    print("\n[3] Text annotations")
    rpt = inspect_text(data_dir / "omomo_text_anno_json_data")
    if not rpt["exists"]:
        print(f"  MISSING ({rpt['path']})")
    else:
        print(f"  {rpt['num_files']} JSON files in {rpt['path']}")
        print(f"  Sample: {rpt.get('sample_file', 'n/a')}")
        print(f"  Content: {rpt.get('sample_content', 'n/a')}")

    # --- Contact labels ---
    print("\n[4] Contact labels")
    rpt = inspect_contacts(data_dir / "contact_labels_w_semantics_npy_files")
    if not rpt["exists"]:
        print(f"  MISSING ({rpt['path']})")
    else:
        print(f"  {rpt['num_files']} .npy files")
        print(f"  Sample shape: {rpt['sample_shape']}, dtype: {rpt['sample_dtype']}, range: {rpt['value_range']}")

    print("\n" + "=" * 72)
    print("Inspection complete.")
    print("=" * 72)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/omomo/processed_data"),
        help="Path to OMOMO processed_data directory",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_inspection(args.data_dir)


if __name__ == "__main__":
    main()
