"""Full pseudo-label extraction pipeline.

Runs all four extraction steps (contact → target → phase → support)
sequentially for each motion sequence, and saves results as compressed
npz files.

Usage:
    piano-pseudo-labels --data-dir data/interact/processed --output-dir runs/pseudo_labels
    python -m piano.data.pseudo_labels.run_all --data-dir ... --output-dir ...
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from piano.data.pseudo_labels.extract_contact import ContactConfig, extract_contact_state
from piano.data.pseudo_labels.extract_phase import PhaseConfig, extract_interaction_phase
from piano.data.pseudo_labels.extract_support import SupportConfig, extract_support_state
from piano.data.pseudo_labels.extract_target import TargetConfig, extract_contact_target
from piano.data.pseudo_labels.refine_phase_hmm import (
    HMMConfig,
    build_phase_features,
    refine_phases_hmm,
)
from piano.utils.geometry import load_mesh
from piano.utils.io_utils import ensure_dir, load_json, save_npz


def process_sequence(
    joints: np.ndarray,
    object_mesh_path: str | Path,
    object_positions: np.ndarray | None = None,
    contact_config: ContactConfig | None = None,
    target_config: TargetConfig | None = None,
    phase_config: PhaseConfig | None = None,
    support_config: SupportConfig | None = None,
    hmm_config: HMMConfig | None = None,
    use_hmm_refinement: bool = True,
) -> dict[str, np.ndarray]:
    """Run the full pseudo-label extraction for one sequence.

    Parameters
    ----------
    joints : (T, 22, 3) — SMPL 22-joint positions
    object_mesh_path : path to object mesh file (obj, ply, etc.)
    object_positions : (T, 3) or None — object center per frame
    *_config : per-stage configuration (uses defaults if None)
    use_hmm_refinement : whether to refine phase labels with HMM

    Returns
    -------
    Dictionary containing:
        ``contact_state`` : (T, 5)
        ``contact_target`` : (T, 5, K)
        ``patch_centers`` : (K, 3)
        ``phase`` : (T,)
        ``support`` : (T,)
    """
    mesh = load_mesh(str(object_mesh_path))

    # Step 1: Contact state
    contact_state = extract_contact_state(joints, mesh, contact_config)

    # Step 2: Contact target (depends on contact_state)
    contact_target, patch_centers = extract_contact_target(
        joints, mesh, contact_state, target_config,
    )

    # Step 3: Interaction phase
    phase = extract_interaction_phase(
        joints, contact_state, object_positions, phase_config,
    )

    # Optional: HMM refinement
    if use_hmm_refinement:
        features = build_phase_features(
            joints, contact_state, object_positions,
            fps=(phase_config or PhaseConfig()).fps,
        )
        phase = refine_phases_hmm(features, phase, hmm_config)

    # Step 4: Support state (depends on contact_state)
    support = extract_support_state(contact_state, support_config)

    return {
        "contact_state": contact_state,
        "contact_target": contact_target,
        "patch_centers": patch_centers,
        "phase": phase,
        "support": support,
    }


def run_pipeline(
    data_dir: Path,
    output_dir: Path,
    mesh_dir: Path,
    metadata_path: Path | None = None,
    use_hmm: bool = True,
    mesh_suffixes: tuple[str, ...] = ("_cleaned_simplified", ""),
) -> None:
    """Batch pseudo-label extraction for all sequences.

    Expects preprocessed data at *data_dir*::

        data_dir/
            metadata.json            # list of {seq_id, object_id, ...}
            motions/<seq_id>.npz     # contains joints_22, object_positions

    Object meshes live at *mesh_dir* (typically the source dataset's
    captured_objects folder), as ``<obj_id><suffix>.{obj,ply,...}`` files.

    Parameters
    ----------
    data_dir : root of preprocessed (PIANO-format) dataset
    output_dir : where to write pseudo-label npz files
    mesh_dir : directory containing source object meshes
    metadata_path : override metadata.json location
    use_hmm : whether to refine phases with HMM
    mesh_suffixes : suffixes to try appending to object_id when searching
        for the mesh file (OMOMO uses ``_cleaned_simplified``).
    """
    data_dir = Path(data_dir)
    mesh_dir = Path(mesh_dir)
    output_dir = ensure_dir(output_dir)

    if metadata_path is None:
        metadata_path = data_dir / "metadata.json"
    metadata = load_json(metadata_path)

    print(f"Extracting pseudo-labels for {len(metadata)} sequences")
    print(f"  Data:   {data_dir}")
    print(f"  Meshes: {mesh_dir}")
    print(f"  Output: {output_dir}")

    # Cache mesh paths per object_id to avoid re-searching
    mesh_cache: dict[str, Path | None] = {}

    n_ok = 0
    n_skip = 0
    for entry in tqdm(metadata, desc="Pseudo-labels"):
        seq_id = entry["seq_id"]
        obj_id = entry["object_id"]

        # Load preprocessed motion
        motion_path = data_dir / "motions" / f"{seq_id}.npz"
        if not motion_path.exists():
            n_skip += 1
            continue
        motion_data = np.load(motion_path, allow_pickle=False)
        joints = motion_data["joints_22"]  # (T, 22, 3)

        # Find object mesh (cached)
        if obj_id not in mesh_cache:
            mesh_cache[obj_id] = _find_mesh(mesh_dir, obj_id, mesh_suffixes)
        mesh_path = mesh_cache[obj_id]
        if mesh_path is None:
            n_skip += 1
            continue

        # Object positions (from preprocessing)
        object_positions = motion_data.get("object_positions", None)

        try:
            labels = process_sequence(
                joints=joints,
                object_mesh_path=mesh_path,
                object_positions=object_positions,
                use_hmm_refinement=use_hmm,
            )
        except Exception as e:
            print(f"  [warn] {seq_id}: {e}")
            n_skip += 1
            continue

        save_npz(output_dir / f"{seq_id}.npz", **labels)
        n_ok += 1

    print(f"Done. {n_ok} labels written, {n_skip} skipped. Output: {output_dir}")


def _find_mesh(
    mesh_dir: Path,
    obj_id: str,
    suffixes: tuple[str, ...],
) -> Path | None:
    """Look up an object mesh file by id + suffix, trying common extensions."""
    extensions = (".obj", ".ply", ".stl", ".off")
    for suffix in suffixes:
        for ext in extensions:
            path = mesh_dir / f"{obj_id}{suffix}{ext}"
            if path.exists():
                return path
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract pseudo interaction labels from preprocessed HOI data",
    )
    parser.add_argument(
        "--data-dir", type=Path, required=True,
        help="Root of preprocessed PIANO dataset (contains motions/, metadata.json)",
    )
    parser.add_argument(
        "--mesh-dir", type=Path, required=True,
        help="Directory containing source object meshes (e.g. OMOMO captured_objects/)",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory to write pseudo-label npz files",
    )
    parser.add_argument(
        "--metadata", type=Path, default=None,
        help="Override metadata file path (default: <data-dir>/metadata.json)",
    )
    parser.add_argument(
        "--mesh-suffix", type=str, default="_cleaned_simplified",
        help="Filename suffix for mesh files (default: '_cleaned_simplified' for OMOMO)",
    )
    parser.add_argument(
        "--no-hmm", action="store_true",
        help="Skip HMM refinement for phase labels",
    )
    return parser


def main() -> None:
    """CLI entrypoint for ``piano-pseudo-labels``."""
    args = build_parser().parse_args()
    # Try the specified suffix first, then no suffix as fallback
    suffixes = (args.mesh_suffix, "") if args.mesh_suffix else ("",)
    run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        mesh_dir=args.mesh_dir,
        metadata_path=args.metadata,
        use_hmm=not args.no_hmm,
        mesh_suffixes=suffixes,
    )


if __name__ == "__main__":
    main()
