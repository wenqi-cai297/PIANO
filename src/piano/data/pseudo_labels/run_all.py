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
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

import hashlib

from piano.data.pseudo_labels.extract_contact import ContactConfig, extract_contact_state
from piano.data.pseudo_labels.extract_phase import PhaseConfig, extract_interaction_phase
from piano.data.pseudo_labels.extract_support import SupportConfig, extract_support_state
from piano.data.pseudo_labels.extract_target import TargetConfig, extract_contact_target
from piano.data.pseudo_labels.refine_phase_hmm import (
    HMMConfig,
    build_phase_features,
    refine_phases_hmm,
)
from piano.data.pseudo_labels.stats import (
    aggregate_stats,
    compute_seq_stats,
    make_quality_flags,
)
from piano.utils.geometry import cluster_surface_patches, load_mesh
from piano.utils.io_utils import ensure_dir, load_json, save_json, save_npz


DEFAULT_FPS: float = 20.0  # PIANO preprocessed data rate


def _object_patch_seed(obj_id: str) -> int:
    """Deterministic 32-bit seed derived from object id.

    Guarantees that the same object yields the same patch atlas across
    re-runs and machines. Different objects get independent seeds.
    """
    h = hashlib.md5(obj_id.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _resolve_fps(data_dir: Path, fps_override: float | None) -> float:
    """Find the fps of preprocessed motions.

    Preference order:
        1. Explicit ``fps_override`` from CLI.
        2. ``target_fps`` in ``<data_dir>/summary.json`` (per-subset summary
           written by newer preprocess_interact).
        3. ``config.target_fps`` in ``<data_dir>/../summary.json`` (top-level
           preprocess summary — covers datasets preprocessed before the
           per-subset fps field was added).
        4. ``DEFAULT_FPS`` with a warning.
    """
    if fps_override is not None:
        return float(fps_override)

    per_subset = data_dir / "summary.json"
    if per_subset.exists():
        try:
            s = load_json(per_subset)
            if "target_fps" in s:
                return float(s["target_fps"])
        except Exception:
            pass

    top_level = data_dir.parent / "summary.json"
    if top_level.exists():
        try:
            s = load_json(top_level)
            fps = s.get("config", {}).get("target_fps")
            if fps is not None:
                return float(fps)
        except Exception:
            pass

    print(f"  [warn] fps not found in {per_subset} or {top_level}; "
          f"defaulting to {DEFAULT_FPS}")
    return DEFAULT_FPS


def process_sequence(
    joints: np.ndarray,
    object_mesh: "trimesh.Trimesh | str | Path",
    object_positions: np.ndarray | None = None,
    object_rotations: np.ndarray | None = None,
    patch_centers: np.ndarray | None = None,
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
    joints : (T, 22, 3) — world-frame SMPL 22-joint positions
    object_mesh : a pre-loaded ``trimesh.Trimesh`` (preferred, enables caching
        across sequences with the same object) or a path to load. Mesh
        is in object-local frame.
    object_positions : (T, 3) or None — per-frame object translation in world
    object_rotations : (T, 3) or None — per-frame object axis-angle rotation.
        Needed for geometrically correct contact / target extraction. If
        None, rotation is treated as identity (still uses translation).
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
    if isinstance(object_mesh, (str, Path)):
        mesh = load_mesh(str(object_mesh))
    else:
        mesh = object_mesh

    # Step 1: Contact state — inverse-transforms joints to object-local
    contact_state = extract_contact_state(
        joints, mesh,
        object_positions=object_positions,
        object_rotations=object_rotations,
        config=contact_config,
    )

    # Step 2: Contact target (depends on contact_state) — same transform.
    # patch_centers is passed in from the caller's per-object atlas so that
    # patch ids are stable across every sequence of the same object.
    contact_target, patch_centers = extract_contact_target(
        joints, mesh, contact_state,
        object_positions=object_positions,
        object_rotations=object_rotations,
        config=target_config,
        patch_centers=patch_centers,
    )

    # Step 3: Interaction phase — rotation-aware so that rotation-only
    # manipulations (bat swing, chair rotate) reach manipulation instead
    # of collapsing to stable-contact.
    phase = extract_interaction_phase(
        joints, contact_state, object_positions, object_rotations, phase_config,
    )

    # Optional: HMM refinement
    if use_hmm_refinement:
        features = build_phase_features(
            joints, contact_state, object_positions, object_rotations,
            fps=(phase_config or PhaseConfig()).fps,
        )
        phase = refine_phases_hmm(features, phase, hmm_config)

    # Step 4: Support state — two gates on `sitting` beyond pelvis contact:
    #   (a) joints → pelvis XZ-speed < 0.15 m/s (rejects push/drag)
    #   (b) object_mesh + positions + rotations → geometric "object below
    #       pelvis" test (rejects standing-beside-object where the pelvis
    #       joint is within 20 cm of a backrest/leg but not *above* a seat)
    support = extract_support_state(
        contact_state,
        joints=joints,
        object_mesh=mesh,
        object_positions=object_positions,
        object_rotations=object_rotations,
        config=support_config,
    )

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
    fps: float | None = None,
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
    t_start = time.time()
    data_dir = Path(data_dir)
    mesh_dir = Path(mesh_dir)
    output_dir = ensure_dir(output_dir)
    atlas_dir = ensure_dir(output_dir / "patch_atlas")

    if metadata_path is None:
        metadata_path = data_dir / "metadata.json"
    metadata = load_json(metadata_path)

    resolved_fps = _resolve_fps(data_dir, fps)
    contact_cfg = ContactConfig(fps=resolved_fps)
    phase_cfg = PhaseConfig(fps=resolved_fps)
    target_cfg = TargetConfig()
    support_cfg = SupportConfig(fps=resolved_fps)

    print(f"Extracting pseudo-labels for {len(metadata)} sequences")
    print(f"  Data:   {data_dir}")
    print(f"  Meshes: {mesh_dir}")
    print(f"  Output: {output_dir}")
    print(f"  FPS:    {resolved_fps}  (used for velocity thresholds)")

    # Cache LOADED meshes (not just paths) so each object is loaded once
    # and its trimesh spatial index is reused across all sequences using
    # that object — critical for speed/memory on datasets with large meshes.
    import trimesh
    mesh_cache: dict[str, trimesh.Trimesh | None] = {}
    # Per-object deterministic patch atlas, cached on disk so re-runs and
    # separate machines produce identical patch ids.
    atlas_cache: dict[str, np.ndarray | None] = {}

    n_ok = 0
    n_skip = 0
    n_resume = 0
    skip_reasons: list[dict[str, str]] = []
    per_seq_stats = []
    for entry in tqdm(metadata, desc="Pseudo-labels"):
        seq_id = entry["seq_id"]
        obj_id = entry["object_id"]

        # Resume support: skip sequences we've already written
        out_path = output_dir / f"{seq_id}.npz"
        if out_path.exists():
            n_resume += 1
            continue

        # Load preprocessed motion
        motion_path = data_dir / "motions" / f"{seq_id}.npz"
        if not motion_path.exists():
            n_skip += 1
            skip_reasons.append({"seq_id": seq_id, "reason": "motion_file_missing"})
            continue
        motion_data = np.load(motion_path, allow_pickle=False)
        joints = motion_data["joints_22"]  # (T, 22, 3)

        # Lazily load and cache the mesh for this object_id
        if obj_id not in mesh_cache:
            mesh_path = _find_mesh(mesh_dir, obj_id, mesh_suffixes)
            if mesh_path is None:
                mesh_cache[obj_id] = None
            else:
                try:
                    mesh_cache[obj_id] = load_mesh(str(mesh_path))
                except Exception as e:
                    print(f"  [warn] failed to load mesh {mesh_path}: {e}")
                    mesh_cache[obj_id] = None
        mesh = mesh_cache[obj_id]
        if mesh is None:
            n_skip += 1
            skip_reasons.append({
                "seq_id": seq_id,
                "reason": f"mesh_not_found_or_failed_to_load (object_id={obj_id})",
            })
            continue

        # Per-object deterministic patch atlas (shared across all sequences
        # of this object). Disk-cached so re-runs stay consistent.
        if obj_id not in atlas_cache:
            atlas_path = atlas_dir / f"{obj_id}.npy"
            if atlas_path.exists():
                atlas_cache[obj_id] = np.load(atlas_path)
            else:
                atlas = cluster_surface_patches(
                    mesh,
                    num_patches=target_cfg.num_patches,
                    num_surface_samples=target_cfg.num_surface_samples,
                    seed=_object_patch_seed(obj_id),
                )
                np.save(atlas_path, atlas)
                atlas_cache[obj_id] = atlas
        patch_centers = atlas_cache[obj_id]

        # Object pose from preprocessing. object_rotations is only present
        # for data preprocessed with the updated preprocess_interact that
        # saves rotation. Older data will fall back to translation-only.
        files = set(motion_data.files)
        object_positions = motion_data["object_positions"] if "object_positions" in files else None
        object_rotations = motion_data["object_rotations"] if "object_rotations" in files else None

        try:
            labels = process_sequence(
                joints=joints,
                object_mesh=mesh,
                object_positions=object_positions,
                object_rotations=object_rotations,
                patch_centers=patch_centers,
                contact_config=contact_cfg,
                target_config=target_cfg,
                phase_config=phase_cfg,
                support_config=support_cfg,
                use_hmm_refinement=use_hmm,
            )
        except Exception as e:
            print(f"  [warn] {seq_id}: {e}")
            n_skip += 1
            skip_reasons.append({"seq_id": seq_id, "reason": f"exception: {e}"})
            continue

        save_npz(out_path, **labels)
        n_ok += 1

        # Accumulate quality stats for the summary. Uses the just-computed
        # labels in memory — no extra disk I/O and one set of traversals
        # per sequence (cheap compared to mesh distance queries).
        try:
            per_seq_stats.append(
                compute_seq_stats(
                    seq_id=seq_id,
                    labels=labels,
                    joints_22=joints,
                    object_positions=object_positions,
                )
            )
        except Exception as e:
            print(f"  [warn] stats failed for {seq_id}: {e}")

    elapsed = time.time() - t_start
    print(f"Done. {n_ok} labels written, {n_resume} resumed (already existed), "
          f"{n_skip} skipped. Output: {output_dir}")
    print(f"Elapsed: {elapsed:.1f}s  ({n_ok / max(elapsed, 1e-6):.1f} seq/s)")

    # --- Aggregate quality stats + derive readable flags ---
    subset_hint = data_dir.name or None
    stats_agg = aggregate_stats(per_seq_stats, num_patches=target_cfg.num_patches)
    quality_flags = make_quality_flags(stats_agg, subset_hint=subset_hint)

    # Summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "data_dir": str(data_dir),
        "mesh_dir": str(mesh_dir),
        "output_dir": str(output_dir),
        "subset": subset_hint,
        "fps": resolved_fps,
        "use_hmm": use_hmm,
        "mesh_suffixes": list(mesh_suffixes),
        "num_objects_with_atlas": len([k for k, v in atlas_cache.items() if v is not None]),
        "counts": {
            "num_in_metadata": len(metadata),
            "num_labels_written": n_ok,
            "num_resumed": n_resume,
            "num_skipped": n_skip,
        },
        "elapsed_sec": round(elapsed, 2),
        "throughput_seq_per_sec": round(n_ok / max(elapsed, 1e-6), 2),
        "skip_reasons": skip_reasons,
        "quality_flags": quality_flags,
        "stats": stats_agg,
    }
    save_json(output_dir / "summary.json", summary)

    if quality_flags:
        print(f"\nQuality flags ({len(quality_flags)}):")
        for f in quality_flags:
            print(f"  - {f}")
    else:
        print("\nQuality flags: none fired")


def _find_mesh(
    mesh_dir: Path,
    obj_id: str,
    suffixes: tuple[str, ...],
) -> Path | None:
    """Look up an object mesh file by id + suffix.

    Tries two layouts to support different upstream conventions:
        - Flat:    ``mesh_dir/<obj_id><suffix>.<ext>``         (OMOMO/CHOIS)
        - Nested:  ``mesh_dir/<obj_id>/<obj_id><suffix>.<ext>`` (InterAct)
    """
    extensions = (".obj", ".ply", ".stl", ".off")
    candidate_dirs = (mesh_dir, mesh_dir / obj_id)
    for d in candidate_dirs:
        for suffix in suffixes:
            for ext in extensions:
                path = d / f"{obj_id}{suffix}{ext}"
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
        "--mesh-suffixes", nargs="+", default=["_cleaned_simplified", ""],
        help="Filename suffixes to try in order when searching for the mesh. "
             "Empty string = bare filename. Default favors simplified variants: "
             "('_cleaned_simplified', '') for OMOMO. For InterAct subsets that "
             "ship simplified variants, pass '_face1000 _simplified \"\"'.",
    )
    parser.add_argument(
        "--no-hmm", action="store_true",
        help="Skip HMM refinement for phase labels",
    )
    parser.add_argument(
        "--fps", type=float, default=None,
        help="Override fps used for velocity thresholds. Default: read "
             "target_fps from <data-dir>/summary.json, else 20.",
    )
    return parser


def main() -> None:
    """CLI entrypoint for ``piano-pseudo-labels``."""
    args = build_parser().parse_args()
    run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        mesh_dir=args.mesh_dir,
        metadata_path=args.metadata,
        use_hmm=not args.no_hmm,
        mesh_suffixes=tuple(args.mesh_suffixes),
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
