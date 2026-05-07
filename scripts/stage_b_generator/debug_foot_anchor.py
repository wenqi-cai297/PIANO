"""Inspect why the world-frame foot anchor distance is huge on
chairs/imhd/neuraldome but small on OMOMO.

For a few foot-contact frames in 1-2 chair clips, prints:
    - SMPL joint 10 (left_foot) and 11 (right_foot) world positions
    - SMPL joint 7 (left_ankle) and 8 (right_ankle) world positions
    - contact_target_xyz_local at that frame
    - target_world (lifted from object-local via obj_pose_world)
    - distance from each candidate joint to target_world
    - raw object_position_world / object_rotation_world

If the distance is closest with the ANKLE joint (7/8), then PART_TO_JOINT
is mapped to the wrong joint and we should remap foot → ankles.

If the distance is large for ALL candidate joints, then something in
the pseudo-label extraction or world-frame lifting is broken for
chairs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from piano.data.dataset import HOIDataset


def _aa_to_R(aa):
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)
    safe = np.where(theta < 1e-8, 1.0, theta)
    k = aa / safe
    cos = np.cos(theta)[..., None]
    sin = np.sin(theta)[..., None]
    K = np.zeros(aa.shape[:-1] + (3, 3))
    K[..., 0, 1] = -k[..., 2]; K[..., 0, 2] = k[..., 1]
    K[..., 1, 0] = k[..., 2];  K[..., 1, 2] = -k[..., 0]
    K[..., 2, 0] = -k[..., 1]; K[..., 2, 1] = k[..., 0]
    eye = np.broadcast_to(np.eye(3), K.shape)
    return eye + sin * K + (1 - cos) * (K @ K)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--subset", type=str, default="chairs")
    parser.add_argument("--num-clips", type=int, default=2)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    pseudo_label_subdir = cfg.data.get("pseudo_label_subdir", None)

    target_entry = next(
        (e for e in cfg.data.datasets if e.name == args.subset), None,
    )
    assert target_entry is not None, f"No {args.subset} entry"

    sub_dir = (str(Path(target_entry.root) / pseudo_label_subdir)
               if pseudo_label_subdir is not None else None)
    ds = HOIDataset(
        root=target_entry.root,
        pseudo_label_dir=sub_dir,
        max_seq_length=cfg.data.max_seq_length,
        augment=None,
        support_collapse_hand_support=True,
        surface_obj_pose=True,
    )

    found = 0
    for clip_idx in range(len(ds)):
        if found >= args.num_clips:
            break
        sample = ds[clip_idx]
        seq_len = int(sample["seq_len"].item())
        contact_state = sample["contact_state"].numpy()[:seq_len]   # (T, 5)
        # foot contact = part 2 (left_foot) or 3 (right_foot)
        any_foot = (contact_state[:, 2] >= 0.5) | (contact_state[:, 3] >= 0.5)
        if not any_foot.any():
            continue
        joints_world = sample["joints"].numpy()[:seq_len]
        contact_target = sample["contact_target_xyz"].numpy()[:seq_len]
        obj_pos = sample["object_positions"].numpy()[:seq_len]
        obj_rot_aa = sample["object_rotations"].numpy()[:seq_len]
        R_obj = _aa_to_R(obj_rot_aa)

        seq_id = sample["seq_id"]
        print(f"\n========== {seq_id} (T={seq_len}) ==========")
        # Pick 3 contact frames: first, mid, last
        contact_frames = np.flatnonzero(any_foot)
        chosen = [contact_frames[0],
                  contact_frames[len(contact_frames) // 2],
                  contact_frames[-1]]
        for t in chosen:
            print(f"\n--- frame t={t} ---")
            print(f"contact_state[t] = {contact_state[t]}")
            print(f"object_position_world = {obj_pos[t]}")
            print(f"object_rotation_world (aa) = {obj_rot_aa[t]}")
            for p_idx, p_name in [(2, "left_foot"), (3, "right_foot")]:
                if contact_state[t, p_idx] < 0.5:
                    continue
                tgt_local = contact_target[t, p_idx]
                tgt_world = R_obj[t] @ tgt_local + obj_pos[t]
                print(f"  {p_name}: contact_target_local = {tgt_local}")
                print(f"           target_world          = {tgt_world}")
                # Distance from each candidate joint to target_world
                for j_name, j_idx in [
                    ("pelvis (0)", 0),
                    ("l_knee (4)", 4), ("r_knee (5)", 5),
                    ("l_ankle (7)", 7), ("r_ankle (8)", 8),
                    ("l_foot/toe (10)", 10), ("r_foot/toe (11)", 11),
                ]:
                    j_world = joints_world[t, j_idx]
                    d = np.linalg.norm(j_world - tgt_world)
                    print(f"           dist({j_name})  = {d*100:6.2f} cm  joint_world={j_world}")
        found += 1


if __name__ == "__main__":
    main()
