"""Visualize the GT motion AnchorDiff anchor loss actually targets.

Renders two MP4s per clip:

    *_gt_recon.mp4 : body from recover_from_ric(motion_263) lifted to
                     world via per-clip (R_y, T_xz, T_y). This is the
                     "GT" the AnchorDiff training loss sees.
    *_raw_gt.mp4   : body from raw joints field. Ground truth.

Both overlay the world-frame object point cloud (rotated by
object_rotations and translated by object_positions per frame).

Visual side-by-side check before launching M1: if recon ≈ raw, the
anchor loss is targeting a faithful body reconstruction. If recon
diverges (cumsum drift, frame mismatch, etc.), expect that clip's
training contribution to be muted by max_distance_m clamp.

Usage:
    python scripts/stage_b_generator/visualize_anchordiff_gt_recon.py \\
        --config configs/training/anchordiff_v1.yaml \\
        --output runs/visualizations/anchordiff_gt_recon \\
        --clips chairs:0 chairs:Sub0008_Obj116_Seg0_330 \
                imhd:0 neuraldome:0 omomo_correct_v2:0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from piano.data.dataset import HOIDataset
from piano.inference.visualize_motion import render_motion_video
from piano.training.anchor_consistency_loss import lift_motion263_to_joints
from piano.utils.canonical_frame import (
    get_canonicalize_transform_from_clip,
    y_rotation_matrix,
)


def _resolve_clip(ds: HOIDataset, spec: str) -> int:
    """``spec`` is either an integer index or a seq_id substring."""
    try:
        return int(spec)
    except ValueError:
        pass
    for i in range(len(ds)):
        meta = ds.metadata[i] if hasattr(ds, "metadata") else None
        if meta is not None:
            if spec in str(meta.get("seq_id", "")):
                return i
        else:
            sample = ds[i]
            if spec in str(sample["seq_id"]):
                return i
    raise ValueError(f"clip '{spec}' not found in dataset")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--clips", type=str, nargs="+", required=True,
                        help="subset:index_or_seq_id, e.g. chairs:0 imhd:Sub0001")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--dpi", type=int, default=80)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    pseudo_label_subdir = cfg.data.get("pseudo_label_subdir", None)

    # Build per-subset HOIDataset (no augmentation, expose obj canonical fields).
    datasets: dict[str, HOIDataset] = {}
    for entry in cfg.data.datasets:
        sub_dir = (str(Path(entry.root) / pseudo_label_subdir)
                   if pseudo_label_subdir is not None else None)
        ds = HOIDataset(
            root=entry.root,
            pseudo_label_dir=sub_dir,
            max_seq_length=cfg.data.max_seq_length,
            augment=None,
            support_collapse_hand_support=True,
            surface_obj_pose=True,
        )
        datasets[entry.name] = ds

    for spec in args.clips:
        if ":" not in spec:
            print(f"Skip malformed spec: {spec}")
            continue
        subset, clip_spec = spec.split(":", 1)
        if subset not in datasets:
            print(f"Skip: unknown subset '{subset}'")
            continue
        ds = datasets[subset]
        try:
            clip_idx = _resolve_clip(ds, clip_spec)
        except ValueError as e:
            print(f"Skip: {e}")
            continue

        sample = ds[clip_idx]
        seq_id = sample["seq_id"]
        seq_len = int(sample["seq_len"].item())
        motion_263 = sample["motion"].numpy()[:seq_len]            # (T, 263)
        joints_world = sample["joints"].numpy()[:seq_len]          # (T, 22, 3)
        obj_pos = sample["object_positions"].numpy()[:seq_len]     # (T, 3)
        obj_rot = sample["object_rotations"].numpy()[:seq_len]     # (T, 3) axis-angle
        obj_pc = sample["object_pc"].numpy()                       # (N, 3) object-local

        # 1. GT recon: recover_from_ric(motion_263) -> uniform-skel canonical;
        #    lift to world via per-clip (R_y, T_xz, T_y).
        canon = lift_motion263_to_joints(
            torch.from_numpy(motion_263).float().unsqueeze(0)
        ).squeeze(0).cpu().numpy()                                  # (T, 22, 3)
        R_y, T_xz, T_y = get_canonicalize_transform_from_clip(joints_world, canon)
        R = y_rotation_matrix(R_y)
        joints_recon = canon @ R.T
        joints_recon[..., 0] += T_xz[0]
        joints_recon[..., 1] += T_y
        joints_recon[..., 2] += T_xz[1]

        title_recon = (
            f"{subset}/{seq_id}\n"
            f"GT RECON (motion_263 → world via R_y={np.degrees(R_y):.1f}°, "
            f"T_xz={T_xz[0]:.2f},{T_xz[1]:.2f}, T_y={T_y:.3f})"
        )
        title_raw = f"{subset}/{seq_id}\nRAW GT (joints field)"

        out_recon = out_dir / f"{subset}_{seq_id}_gt_recon.mp4"
        out_raw = out_dir / f"{subset}_{seq_id}_raw_gt.mp4"

        print(f"[{subset}/{seq_id}] T={seq_len}  rendering recon → {out_recon.name}")
        render_motion_video(
            joints=joints_recon,
            output_path=out_recon,
            fps=args.fps,
            title=title_recon,
            object_positions=obj_pos,
            object_rotations=obj_rot,
            object_pc=obj_pc,
            dpi=args.dpi,
        )
        print(f"[{subset}/{seq_id}] T={seq_len}  rendering raw   → {out_raw.name}")
        render_motion_video(
            joints=joints_world,
            output_path=out_raw,
            fps=args.fps,
            title=title_raw,
            object_positions=obj_pos,
            object_rotations=obj_rot,
            object_pc=obj_pc,
            dpi=args.dpi,
        )

    print(f"\nDone. Videos in {out_dir}/")


if __name__ == "__main__":
    main()
