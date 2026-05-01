"""Debug a specific clip's contact extraction: print per-frame distance,
kinematic_engagement, static_engagement, and decision for each body part.

Usage:
    python scripts/stage_b_generator/debug_strict_contact_clip.py \\
        --input-dir runs/eval/<...>_gt_roundtrip_80/gt_original \\
        --seq-id Sub1034_Obj96_Seg0_285
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import piano.models.backbones.momask_adapter  # noqa: F401
from utils.motion_process import recover_from_ric

from piano.data.pseudo_labels.extract_contact import (
    ContactConfig,
    DEFAULT_DISTANCE_THRESHOLDS,
    _kinematic_contact_score,
    _soft_sigmoid,
)
from piano.data.pseudo_labels.extract_strict_contact import (
    LOOSE_DISTANCE_THRESHOLDS,
    STRICT_DISTANCE_THRESHOLDS,
    StrictContactConfig,
    _static_engagement_score,
)
from piano.data.pseudo_labels._object_transform import world_to_object_local
from piano.utils.canonical_frame import y_rotation_matrix
from piano.utils.smpl_utils import BODY_PART_INDICES, BODY_PART_NAMES, NUM_BODY_PARTS


def _lift(joints_canon, R_y_angle, T_xz):
    R = y_rotation_matrix(float(R_y_angle))
    rotated = joints_canon @ R.T
    rotated[..., 0] += float(T_xz[0])
    rotated[..., 2] += float(T_xz[1])
    return rotated.astype(np.float32)


def _nearest_pc_dist(points_local, pc_local):
    diff = points_local[:, None, :] - pc_local[None, :, :]
    return np.linalg.norm(diff, axis=-1).min(axis=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--seq-id", required=True)
    parser.add_argument("--fps", type=float, default=20.0)
    args = parser.parse_args()

    npz = np.load(args.input_dir / "generated.npz")
    with open(args.input_dir / "summary.json") as f:
        meta = json.load(f)

    seq_ids = meta["seq_ids"]
    seq_lens = meta.get("seq_lens") or npz["seq_lens"].tolist()
    seq_lens = [int(L) for L in seq_lens]

    if args.seq_id not in seq_ids:
        print(f"seq_id {args.seq_id} not found. Available: {seq_ids[:5]}...")
        return
    i = seq_ids.index(args.seq_id)
    T = seq_lens[i]
    print(f"=== {args.seq_id}  T={T} ===")

    motion = torch.from_numpy(npz["motion_263"][i, :T]).float().unsqueeze(0)
    canon = recover_from_ric(motion, 22).squeeze(0).cpu().numpy().astype(np.float32)
    joints_world = _lift(canon, float(npz["world_R_y_angle"][i]), npz["world_T_xz"][i])

    object_pc = npz["object_pc"][i].astype(np.float32)
    object_positions = npz["object_positions"][i, :T].astype(np.float32)
    object_rotations = npz["object_rotations"][i, :T].astype(np.float32)

    pc_x_range = float(object_pc[:, 0].max() - object_pc[:, 0].min())
    pc_y_range = float(object_pc[:, 1].max() - object_pc[:, 1].min())
    pc_z_range = float(object_pc[:, 2].max() - object_pc[:, 2].min())
    print(f"  object_pc shape: {object_pc.shape}  range: x={pc_x_range:.3f} y={pc_y_range:.3f} z={pc_z_range:.3f}")
    obj_speed_world = np.zeros(T)
    obj_speed_world[1:] = np.linalg.norm(np.diff(object_positions, axis=0), axis=-1) * args.fps
    print(f"  object world speed: mean={obj_speed_world.mean():.4f} max={obj_speed_world.max():.4f} m/s")

    sc = StrictContactConfig(fps=float(args.fps))
    base = ContactConfig(fps=float(args.fps))

    ang_vel = np.zeros(T, dtype=np.float32)
    ang_vel[1:] = np.linalg.norm(np.diff(object_rotations, axis=0), axis=-1) * args.fps
    obj_speed_proxy = obj_speed_world + base.kin_radius_proxy * ang_vel

    kin_window = max(3, int(round(base.kin_window_sec * args.fps)))

    print(f"\n  Per-body-part summary (means over time):")
    print(f"  {'part':12} {'dist_m':>9} {'tight%':>7} {'loose%':>7} {'kin_engage':>11} {'static_engage':>14} {'case_kin%':>10} {'case_sta%':>10} {'OR%':>6}")
    for bp_idx, joint_idx in enumerate(BODY_PART_INDICES):
        bp_name = BODY_PART_NAMES[bp_idx]
        bp_world = joints_world[:, joint_idx, :]
        bp_local = world_to_object_local(bp_world, object_positions, object_rotations)
        distances = _nearest_pc_dist(bp_local, object_pc)

        tight_thr = STRICT_DISTANCE_THRESHOLDS[bp_name]
        loose_thr = LOOSE_DISTANCE_THRESHOLDS[bp_name]
        tight_score = _soft_sigmoid(distances, tight_thr, sc.distance_sigma)
        loose_score = _soft_sigmoid(distances, loose_thr, sc.loose_distance_sigma)
        kin_score = _kinematic_contact_score(bp_world, object_positions, object_rotations, base)
        static_score = _static_engagement_score(
            bp_local, obj_speed_proxy, kin_window=kin_window,
            eps_mps=sc.static_engagement_eps_mps,
            local_std_thresh=sc.static_engagement_local_std_m,
        )
        case_kin = kin_score * loose_score
        case_static_arr = static_score * tight_score
        contact_or = np.maximum(case_kin, case_static_arr)

        print(
            f"  {bp_name:12} "
            f"{distances.mean():>9.4f} "
            f"{tight_score.mean()*100:>6.1f}% "
            f"{loose_score.mean()*100:>6.1f}% "
            f"{kin_score.mean():>11.3f} "
            f"{static_score.mean():>14.3f} "
            f"{case_kin.mean()*100:>9.1f}% "
            f"{case_static_arr.mean()*100:>9.1f}% "
            f"{contact_or.mean()*100:>5.1f}%"
        )


if __name__ == "__main__":
    main()
