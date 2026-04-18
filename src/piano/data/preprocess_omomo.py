"""Preprocess OMOMO (CHOIS format) → HumanML3D 263-dim for PIANO training.

Pipeline for each sequence:
    1. Load CHOIS joblib pickle entry (SMPL-X params + object pose)
    2. Run SMPL-X forward kinematics → 55 joint positions
    3. Extract SMPL 22 body joints (drop hand tip joints)
    4. Downsample 30 fps → 20 fps (matching MoMask/HumanML3D convention)
    5. Convert joints → HumanML3D 263-dim features
    6. Compute per-frame object world positions from obj_trans/obj_rot/obj_scale
    7. Save as compressed npz + build metadata.json

Output layout (matches what ``HOIDataset`` expects)::

    output_dir/
        metadata.json              # [{seq_id, text, object_id, gender, ...}, ...]
        motions/<seq_name>.npz     # motion_263, joints_22, object_positions
        objects/<object_id>.npy    # subsampled (N, 3) point cloud

Usage:
    python -m piano.data.preprocess_omomo \\
        --omomo-dir  /path/to/omomo/processed_data \\
        --smplx-dir  /path/to/smpl_x_v1.1/models/smplx \\
        --output-dir /path/to/piano_data/omomo \\
        [--device cuda]
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
import trimesh
from tqdm import tqdm

from piano.data.humanml3d_repr import joints_to_humanml3d
from piano.data.smplx_fk import load_smplx_model, run_smplx_fk
from piano.utils.io_utils import ensure_dir, save_json, save_npz


SOURCE_FPS: float = 30.0    # CHOIS/OMOMO mocap rate
TARGET_FPS: float = 20.0    # MoMask / HumanML3D convention
NUM_OBJECT_POINTS: int = 1024  # points sampled per object mesh


@dataclass(slots=True)
class PreprocessConfig:
    """Configuration for OMOMO preprocessing."""

    omomo_dir: Path
    smplx_dir: Path
    output_dir: Path
    num_betas: int = 16
    source_fps: float = SOURCE_FPS
    target_fps: float = TARGET_FPS
    num_object_points: int = NUM_OBJECT_POINTS
    device: str = "cpu"
    skip_objects: tuple[str, ...] = ("vacuum", "mop")  # CHOIS default: skip two-part objects


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def downsample_temporal(array: np.ndarray, src_fps: float, tgt_fps: float) -> np.ndarray:
    """Linearly resample ``array`` along axis 0 from src_fps to tgt_fps."""
    T = len(array)
    new_T = int(round(T * tgt_fps / src_fps))
    if new_T < 2:
        return array[: max(1, new_T)]

    src_times = np.arange(T, dtype=np.float32)
    tgt_times = np.linspace(0, T - 1, new_T, dtype=np.float32)

    flat = array.reshape(T, -1)                                    # (T, D)
    out = np.empty((new_T, flat.shape[1]), dtype=flat.dtype)
    for d in range(flat.shape[1]):
        out[:, d] = np.interp(tgt_times, src_times, flat[:, d])
    return out.reshape((new_T, *array.shape[1:]))


def compute_object_positions(
    obj_trans: np.ndarray, obj_rot: np.ndarray, obj_scale: np.ndarray,
) -> np.ndarray:
    """Return a simple per-frame object 3D position (the translation term).

    ``obj_trans`` has shape ``(T, 3, 1)`` in CHOIS; strip the trailing axis.
    (We ignore scale/rotation here since what we need is just a 3D point for
    pseudo-label extraction. For accurate contact queries on the mesh, the
    full transform is applied at label-extraction time.)
    """
    return obj_trans[:, :, 0].astype(np.float32)


def sample_object_point_cloud(
    mesh_path: Path, num_points: int, seed: int = 42,
) -> np.ndarray:
    """Uniformly sample ``num_points`` surface points from an object mesh."""
    mesh = trimesh.load(mesh_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Expected Trimesh at {mesh_path}, got {type(mesh)}")
    rng = np.random.default_rng(seed)
    with_random_state = trimesh.sample.sample_surface(mesh, num_points, seed=rng.integers(0, 2**31))
    points = with_random_state[0].astype(np.float32)
    return points


# ---------------------------------------------------------------------------
# Per-sequence preprocessing
# ---------------------------------------------------------------------------

def preprocess_sequence(
    seq: dict,
    smplx_models: dict[str, torch.nn.Module],
    config: PreprocessConfig,
) -> dict[str, np.ndarray] | None:
    """Convert one CHOIS sequence dict to our HumanML3D-compatible format.

    Returns None if the sequence should be skipped.
    """
    gender = str(seq["gender"])
    if gender not in smplx_models:
        # Unknown gender string — default to male
        gender = "male"
    model = smplx_models[gender]

    betas = seq["betas"].astype(np.float32)            # (1, 16)
    root_orient = seq["root_orient"].astype(np.float32)  # (T, 3)
    pose_body = seq["pose_body"].astype(np.float32)     # (T, 63)
    trans = seq["trans"].astype(np.float32)             # (T, 3)

    # Step 1: SMPL-X FK → (T, 55, 3) joints
    joints_smplx = run_smplx_fk(
        model, betas, root_orient, pose_body, trans, device=config.device,
    )

    # Step 2: keep first 22 body joints (SMPL standard)
    joints_22 = joints_smplx[:, :22, :]

    # Step 3: temporal downsample 30 → 20 fps
    joints_22_ds = downsample_temporal(joints_22, config.source_fps, config.target_fps)

    # Step 4: compute object world position per frame, downsampled
    obj_pos_src = compute_object_positions(
        seq["obj_trans"], seq["obj_rot"], seq["obj_scale"],
    )
    obj_pos_ds = downsample_temporal(obj_pos_src, config.source_fps, config.target_fps)

    # Step 5: HumanML3D 263-dim
    motion_263 = joints_to_humanml3d(joints_22_ds, fps=config.target_fps)

    return {
        "joints_22": joints_22_ds.astype(np.float32),
        "motion_263": motion_263.astype(np.float32),
        "object_positions": obj_pos_ds.astype(np.float32),
    }


def _parse_object_id(seq_name: str) -> str:
    """OMOMO seq_name pattern: ``sub{N}_{object}_{take}``. Return the object token."""
    parts = seq_name.split("_")
    # Join middle parts in case object name contains underscore
    if len(parts) < 3:
        raise ValueError(f"Unexpected seq_name format: {seq_name}")
    return "_".join(parts[1:-1])


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(config: PreprocessConfig) -> None:
    """Run the full OMOMO preprocessing pipeline."""
    omomo_dir = config.omomo_dir
    output_dir = ensure_dir(config.output_dir)
    motions_dir = ensure_dir(output_dir / "motions")
    objects_dir = ensure_dir(output_dir / "objects")

    # --- Load SMPL-X models (one per gender) ---
    print(f"Loading SMPL-X models from {config.smplx_dir} ...")
    smplx_models = {
        "male": load_smplx_model(config.smplx_dir, "male", config.num_betas, config.device),
        "female": load_smplx_model(config.smplx_dir, "female", config.num_betas, config.device),
        "neutral": load_smplx_model(config.smplx_dir, "neutral", config.num_betas, config.device),
    }

    # --- Load text annotations map ---
    text_dir = omomo_dir / "omomo_text_anno_json_data"
    text_map: dict[str, str] = {}
    if text_dir.exists():
        for jpath in text_dir.glob("*.json"):
            try:
                payload = json.load(jpath.open())
                # Format: { seq_name: description }
                if isinstance(payload, dict):
                    text_map.update({k: v for k, v in payload.items() if isinstance(v, str)})
            except Exception as e:
                print(f"  [warn] failed to parse {jpath}: {e}")
        print(f"Loaded {len(text_map)} text annotations")

    # --- Process object meshes (subsample to point clouds) ---
    meshes_dir = omomo_dir / "captured_objects"
    print(f"Sampling object point clouds from {meshes_dir} ...")
    object_names: set[str] = set()
    for obj_file in sorted(meshes_dir.glob("*_cleaned_simplified.obj")):
        name = obj_file.stem.replace("_cleaned_simplified", "").replace("_top", "").replace("_bottom", "")
        if name in config.skip_objects:
            continue
        if name in object_names:
            continue  # dedup top/bottom for multi-part objects we don't skip
        pc = sample_object_point_cloud(obj_file, config.num_object_points)
        np.save(objects_dir / f"{name}.npy", pc)
        object_names.add(name)
    print(f"Saved {len(object_names)} object point clouds")

    # --- Process each split ---
    metadata: list[dict] = []
    for split in ("train", "test"):
        pkl_path = omomo_dir / f"{split}_diffusion_manip_seq_joints24.p"
        if not pkl_path.exists():
            print(f"  [warn] {pkl_path.name} not found, skipping split")
            continue

        print(f"\nProcessing {split} split ...")
        data = joblib.load(pkl_path)
        print(f"  {len(data)} sequences in {split}")

        for key, seq in tqdm(data.items(), desc=f"  {split}"):
            seq_name = seq.get("seq_name", str(key))
            try:
                object_id = _parse_object_id(seq_name)
            except ValueError:
                continue

            if object_id in config.skip_objects or object_id not in object_names:
                continue

            try:
                processed = preprocess_sequence(seq, smplx_models, config)
            except Exception as e:
                print(f"  [warn] failed on {seq_name}: {e}")
                continue
            if processed is None:
                continue

            save_npz(motions_dir / f"{seq_name}.npz", **processed)

            metadata.append({
                "seq_id": seq_name,
                "split": split,
                "object_id": object_id,
                "gender": str(seq["gender"]),
                "text": text_map.get(seq_name, ""),
                "num_frames": int(len(processed["motion_263"])),
            })

    # --- Write metadata.json ---
    save_json(output_dir / "metadata.json", metadata)
    print(f"\nWrote metadata for {len(metadata)} sequences to {output_dir / 'metadata.json'}")
    print(f"Train sequences: {sum(1 for m in metadata if m['split'] == 'train')}")
    print(f"Test sequences:  {sum(1 for m in metadata if m['split'] == 'test')}")
    print(f"With text:       {sum(1 for m in metadata if m['text'])}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--omomo-dir", type=Path, required=True,
        help="OMOMO (CHOIS) processed_data directory",
    )
    parser.add_argument(
        "--smplx-dir", type=Path, required=True,
        help="SMPL-X models directory (contains SMPLX_*.npz)",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Output directory for PIANO-format data",
    )
    parser.add_argument("--num-betas", type=int, default=16)
    parser.add_argument("--source-fps", type=float, default=SOURCE_FPS)
    parser.add_argument("--target-fps", type=float, default=TARGET_FPS)
    parser.add_argument("--num-object-points", type=int, default=NUM_OBJECT_POINTS)
    parser.add_argument("--device", type=str, default=None,
                        help="Device for FK (default: cuda if available)")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    config = PreprocessConfig(
        omomo_dir=args.omomo_dir,
        smplx_dir=args.smplx_dir,
        output_dir=args.output_dir,
        num_betas=args.num_betas,
        source_fps=args.source_fps,
        target_fps=args.target_fps,
        num_object_points=args.num_object_points,
        device=device,
    )
    run_pipeline(config)


if __name__ == "__main__":
    main()
