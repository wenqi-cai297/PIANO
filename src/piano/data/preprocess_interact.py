"""Preprocess InterAct (CVPR 2025) subsets → PIANO HumanML3D format.

InterAct bundles 4 HOI subsets with a consistent schema. For each subset we
produce an independent PIANO data root (same layout as preprocess_omomo
output) so ``HOIDataset`` can combine them at training time.

Input layout (per InterAct):
    InterAct/
        chairs|imhd|neuraldome|omomo_correct_v2/
            objects/<obj_name>/<obj_name>.obj       # mesh
            sequences_canonical/<seq_id>/
                human.npz        # poses (T, 156), betas, trans, gender
                object.npz       # angles (T, 3), trans (T, 3), name
                text.txt         # natural language description

Output layout (per subset):
    <output>/<subset>/
        metadata.json
        motions/<seq_id>.npz     # motion_263, joints_22, object_positions
        objects/<obj_id>.npy     # (N, 3) point cloud
        summary.json

Usage:
    python -m piano.data.preprocess_interact \\
        --interact-dir /path/to/InterAct \\
        --smplx-dir /path/to/smplx \\
        --output-dir /path/to/output \\
        [--subset all|chairs|imhd|neuraldome|omomo_correct_v2] \\
        [--device cuda]
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from piano.data.humanml3d_encoder import HumanML3DEncoder
from piano.data.preprocess_omomo import (
    SOURCE_FPS,
    TARGET_FPS,
    NUM_OBJECT_POINTS,
    downsample_temporal,
    sample_object_point_cloud,
)
from piano.data.smplx_fk import load_smplx_model, run_smplx_fk
from piano.utils.io_utils import ensure_dir, save_json, save_npz


# SMPL-X pose splitting (total = 156 dims)
POSE_ROOT_SLICE = slice(0, 3)         # global_orient
POSE_BODY_SLICE = slice(3, 66)        # 21 body joints × 3

DEFAULT_SUBSETS: tuple[str, ...] = ("chairs", "imhd", "neuraldome", "omomo_correct_v2")


@dataclass(slots=True)
class InterActConfig:
    """Configuration for InterAct preprocessing."""

    interact_dir: Path
    smplx_dir: Path
    output_dir: Path
    subsets: tuple[str, ...] = DEFAULT_SUBSETS
    num_betas: int = 16                    # SMPL-X betas dim used by our model
    source_fps: float = SOURCE_FPS          # default 30 fps; may differ per subset
    target_fps: float = TARGET_FPS
    num_object_points: int = NUM_OBJECT_POINTS
    device: str = "cpu"
    num_samples_limit: int | None = None    # for smoke testing


# ---------------------------------------------------------------------------
# Per-sequence preprocessing
# ---------------------------------------------------------------------------

def _load_sequence_inputs(seq_dir: Path) -> dict | None:
    """Read human.npz + object.npz + text.txt from one InterAct sequence."""
    human_path = seq_dir / "human.npz"
    object_path = seq_dir / "object.npz"
    text_path = seq_dir / "text.txt"

    if not human_path.exists() or not object_path.exists():
        return None

    human = np.load(human_path, allow_pickle=True)
    obj = np.load(object_path, allow_pickle=True)

    poses = human["poses"].astype(np.float32)       # (T, 156)
    betas = human["betas"].astype(np.float32)       # (10,) or (16,)
    trans = human["trans"].astype(np.float32)       # (T, 3)
    gender = str(human["gender"])

    obj_angles = obj["angles"].astype(np.float32)   # (T, 3)
    obj_trans = obj["trans"].astype(np.float32)     # (T, 3)
    obj_name = str(obj["name"])

    # Natural language description — first line of text.txt before the first '#'
    text = ""
    if text_path.exists():
        raw = text_path.read_text(encoding="utf-8", errors="replace").strip()
        # InterAct stores "natural#postag#start#end"; keep only the natural text
        text = raw.split("#")[0].strip()

    return {
        "poses": poses,
        "betas": betas,
        "trans": trans,
        "gender": gender,
        "obj_angles": obj_angles,
        "obj_trans": obj_trans,
        "obj_name": obj_name,
        "text": text,
    }


def _pad_betas(betas: np.ndarray, target_dim: int) -> np.ndarray:
    """Pad body-shape coefficients to ``target_dim`` with zeros.

    Needed because chairs uses 10 betas while our SMPL-X model runs with 16.
    """
    if betas.ndim == 1:
        betas = betas[None, :]
    n = betas.shape[-1]
    if n >= target_dim:
        return betas[..., :target_dim]
    pad = np.zeros(betas.shape[:-1] + (target_dim - n,), dtype=betas.dtype)
    return np.concatenate([betas, pad], axis=-1)


def fk_and_downsample_interact(
    inputs: dict,
    smplx_models: dict[str, torch.nn.Module],
    config: InterActConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Run SMPL-X FK on one InterAct sequence, downsample to target_fps.

    Returns ``(joints_22_20fps, object_positions_20fps, object_rotations_20fps)``
    where object_rotations are axis-angle (T', 3). Returns ``None`` on failure.
    """
    gender = inputs["gender"]
    if gender not in smplx_models:
        gender = "male"
    model = smplx_models[gender]

    # SMPL-X pose splitting
    poses = inputs["poses"]
    root_orient = poses[:, POSE_ROOT_SLICE]
    pose_body = poses[:, POSE_BODY_SLICE]
    trans = inputs["trans"]
    betas = _pad_betas(inputs["betas"], config.num_betas)

    # SMPL-X FK → 55 joints
    joints_smplx = run_smplx_fk(
        model, betas, root_orient, pose_body, trans, device=config.device,
    )
    joints_22 = joints_smplx[:, :22, :]

    # Temporal downsample: joints + object translation + object rotation
    joints_22_ds = downsample_temporal(joints_22, config.source_fps, config.target_fps)
    obj_pos_ds = downsample_temporal(inputs["obj_trans"], config.source_fps, config.target_fps)
    obj_rot_ds = downsample_temporal(inputs["obj_angles"], config.source_fps, config.target_fps)

    return (
        joints_22_ds.astype(np.float32),
        obj_pos_ds.astype(np.float32),
        obj_rot_ds.astype(np.float32),
    )


def preprocess_sequence(
    inputs: dict,
    smplx_models: dict[str, torch.nn.Module],
    encoder: HumanML3DEncoder,
    config: InterActConfig,
) -> dict[str, np.ndarray] | None:
    """Convert one InterAct sequence to PIANO format (motion_263 + joints_22 + object_positions)."""
    result = fk_and_downsample_interact(inputs, smplx_models, config)
    if result is None:
        return None
    joints_raw, obj_pos_raw, obj_rot_raw = result

    # MoMask-compatible encoding (HumanML3D canonical + uniform skeleton)
    features, _aligned_joints = encoder.encode(joints_raw)   # (T-1, 263)

    # process_file drops one frame; align all arrays to T-1
    T_minus_1 = features.shape[0]
    joints_raw = joints_raw[:T_minus_1]
    obj_pos_raw = obj_pos_raw[:T_minus_1]
    obj_rot_raw = obj_rot_raw[:T_minus_1]

    return {
        "joints_22": joints_raw,
        "motion_263": features,
        "object_positions": obj_pos_raw,
        "object_rotations": obj_rot_raw,     # axis-angle, (T', 3)
    }


# ---------------------------------------------------------------------------
# Per-subset pipeline
# ---------------------------------------------------------------------------

def _process_subset(
    subset: str,
    config: InterActConfig,
    smplx_models: dict[str, torch.nn.Module],
) -> dict:
    """Process a single InterAct subset and write its output root."""
    t_start = time.time()
    subset_dir = config.interact_dir / subset
    seqs_dir = subset_dir / "sequences_canonical"
    objs_dir = subset_dir / "objects"

    if not seqs_dir.exists():
        print(f"  [skip] {subset}: sequences_canonical not found")
        return {"subset": subset, "num_sequences": 0, "error": "missing sequences_canonical"}

    out_root = ensure_dir(config.output_dir / subset)
    motions_out = ensure_dir(out_root / "motions")
    objects_out = ensure_dir(out_root / "objects")

    # --- Sample object point clouds (one per unique object) ---
    print(f"\n  [{subset}] sampling object point clouds ...")
    obj_names_saved: set[str] = set()
    for obj_dir in sorted(objs_dir.iterdir()):
        if not obj_dir.is_dir():
            continue
        name = obj_dir.name
        # Pick the first .obj file inside (most subsets have multiple variants;
        # prefer the un-suffixed canonical name if present)
        mesh_path = obj_dir / f"{name}.obj"
        if not mesh_path.exists():
            candidates = sorted(obj_dir.glob("*.obj"))
            if not candidates:
                continue
            mesh_path = candidates[0]
        pc = sample_object_point_cloud(mesh_path, config.num_object_points)
        np.save(objects_out / f"{name}.npy", pc)
        obj_names_saved.add(name)
    print(f"  [{subset}] saved {len(obj_names_saved)} object point clouds")

    # --- Initialize encoder with reference skeleton ---
    # Use first frame of first valid sequence as skeleton reference.
    seq_ids = sorted(p.name for p in seqs_dir.iterdir() if p.is_dir())
    if config.num_samples_limit is not None:
        seq_ids = seq_ids[: config.num_samples_limit]

    reference_joints: np.ndarray | None = None
    for sid in seq_ids:
        inputs = _load_sequence_inputs(seqs_dir / sid)
        if inputs is None or inputs["obj_name"] not in obj_names_saved:
            continue
        result = fk_and_downsample_interact(inputs, smplx_models, config)
        if result is None:
            continue
        reference_joints = result[0][0]
        print(f"  [{subset}] reference skeleton from {sid}")
        break

    if reference_joints is None:
        print(f"  [{subset}] no valid sequence found; skipping subset")
        return {"subset": subset, "num_sequences": 0, "error": "no valid reference"}

    encoder = HumanML3DEncoder(reference_joints=reference_joints, feet_thre=0.002)

    # --- Process sequences ---
    metadata: list[dict] = []
    n_ok = 0
    n_skip = 0
    for sid in tqdm(seq_ids, desc=f"  {subset}"):
        inputs = _load_sequence_inputs(seqs_dir / sid)
        if inputs is None:
            n_skip += 1
            continue

        obj_name = inputs["obj_name"]
        if obj_name not in obj_names_saved:
            n_skip += 1
            continue

        try:
            processed = preprocess_sequence(inputs, smplx_models, encoder, config)
        except Exception as e:
            print(f"  [warn] {subset}/{sid}: {e}")
            n_skip += 1
            continue
        if processed is None:
            n_skip += 1
            continue

        save_npz(motions_out / f"{sid}.npz", **processed)
        metadata.append({
            "seq_id": sid,
            "subset": subset,
            "split": "train",       # InterAct has no official split; default train
            "object_id": obj_name,
            "gender": inputs["gender"],
            "text": inputs["text"],
            "num_frames": int(len(processed["motion_263"])),
        })
        n_ok += 1

    save_json(out_root / "metadata.json", metadata)

    elapsed = time.time() - t_start
    summary = {
        "timestamp": datetime.now().isoformat(),
        "subset": subset,
        "num_sequences_total": len(seq_ids),
        "num_processed": n_ok,
        "num_skipped": n_skip,
        "num_objects": len(obj_names_saved),
        "num_with_text": sum(1 for m in metadata if m["text"]),
        "elapsed_sec": round(elapsed, 2),
    }
    save_json(out_root / "summary.json", summary)
    print(f"  [{subset}] done: {n_ok} sequences written, {n_skip} skipped "
          f"({elapsed:.1f}s)")
    return summary


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def run_pipeline(config: InterActConfig) -> None:
    t_start = time.time()
    output_root = ensure_dir(config.output_dir)

    print(f"InterAct → PIANO preprocessing")
    print(f"  Input:  {config.interact_dir}")
    print(f"  Output: {output_root}")
    print(f"  Subsets: {list(config.subsets)}")
    print(f"  Device: {config.device}")

    # Load SMPL-X models once (shared across subsets)
    print(f"\nLoading SMPL-X models from {config.smplx_dir} ...")
    smplx_models = {
        "male": load_smplx_model(config.smplx_dir, "male", config.num_betas, config.device),
        "female": load_smplx_model(config.smplx_dir, "female", config.num_betas, config.device),
        "neutral": load_smplx_model(config.smplx_dir, "neutral", config.num_betas, config.device),
    }

    subset_summaries: list[dict] = []
    for subset in config.subsets:
        summary = _process_subset(subset, config, smplx_models)
        subset_summaries.append(summary)

    # --- Top-level summary ---
    elapsed = time.time() - t_start
    top_summary = {
        "timestamp": datetime.now().isoformat(),
        "interact_dir": str(config.interact_dir),
        "output_dir": str(output_root),
        "smplx_dir": str(config.smplx_dir),
        "config": {
            "num_betas": config.num_betas,
            "source_fps": config.source_fps,
            "target_fps": config.target_fps,
            "num_object_points": config.num_object_points,
            "subsets": list(config.subsets),
        },
        "per_subset": subset_summaries,
        "totals": {
            "num_processed": sum(s.get("num_processed", 0) for s in subset_summaries),
            "num_skipped": sum(s.get("num_skipped", 0) for s in subset_summaries),
            "elapsed_sec": round(elapsed, 2),
        },
    }
    save_json(output_root / "summary.json", top_summary)

    print(f"\n{'=' * 60}")
    print(f"Done: {top_summary['totals']['num_processed']} sequences "
          f"across {len(subset_summaries)} subsets in {elapsed:.1f}s")
    print(f"Top-level summary: {output_root / 'summary.json'}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--interact-dir", type=Path, required=True,
        help="Root InterAct directory (contains chairs/, imhd/, etc.)",
    )
    parser.add_argument(
        "--smplx-dir", type=Path, required=True,
        help="SMPL-X model directory (contains SMPLX_*.npz)",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Output root — each subset becomes <output>/<subset>/",
    )
    parser.add_argument(
        "--subset", type=str, default="all",
        choices=("all", *DEFAULT_SUBSETS),
        help="Subset to process (default: all)",
    )
    parser.add_argument("--num-betas", type=int, default=16)
    parser.add_argument("--source-fps", type=float, default=SOURCE_FPS)
    parser.add_argument("--target-fps", type=float, default=TARGET_FPS)
    parser.add_argument("--num-object-points", type=int, default=NUM_OBJECT_POINTS)
    parser.add_argument("--num-samples-limit", type=int, default=None,
                        help="Process at most this many sequences per subset (smoke test)")
    parser.add_argument("--device", type=str, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    subsets = DEFAULT_SUBSETS if args.subset == "all" else (args.subset,)

    config = InterActConfig(
        interact_dir=args.interact_dir,
        smplx_dir=args.smplx_dir,
        output_dir=args.output_dir,
        subsets=subsets,
        num_betas=args.num_betas,
        source_fps=args.source_fps,
        target_fps=args.target_fps,
        num_object_points=args.num_object_points,
        num_samples_limit=args.num_samples_limit,
        device=device,
    )
    run_pipeline(config)


if __name__ == "__main__":
    main()
