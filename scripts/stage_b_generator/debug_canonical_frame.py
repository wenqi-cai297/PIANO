"""Pinpoint Bug 1: canonical-frame ↔ world-frame mismatch.

For each of N test clips, prints:
    - joints_world[0,0] (pelvis at frame 0, world)
    - canonical_joints[0,0] (pelvis at frame 0, after recover_from_ric)
    - canonical_joints[1,0] (pelvis at frame 1) — to detect first-frame-drop alignment
    - (R_y, T_xz) recovered by get_canonicalize_transform_from_clip
    - reconstruction error: world_pred = R_y(canonical) + T_xz vs actual joints_world

Then explicitly checks per-axis residuals to see whether Y-axis is the
unaccounted-for component.
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import torch
from omegaconf import OmegaConf

from piano.data.dataset import HOIDataset
from piano.utils.canonical_frame import (
    get_canonicalize_transform_from_clip,
    y_rotation_matrix,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num-clips", type=int, default=5)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    pseudo_label_subdir = cfg.data.get("pseudo_label_subdir", None)

    import piano.models.backbones.momask_adapter  # noqa: F401
    from utils.motion_process import recover_from_ric

    print("=" * 80)
    for entry in cfg.data.datasets[:1]:  # just chairs for speed
        from pathlib import Path
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

        for clip_idx in range(min(args.num_clips, len(ds))):
            sample = ds[clip_idx]
            seq_len = int(sample["seq_len"].item())
            joints_world = sample["joints"].numpy()[:seq_len]      # (T, 22, 3)
            motion_263 = sample["motion"].numpy()[:seq_len]        # (T, 263)

            canonical = recover_from_ric(
                torch.from_numpy(motion_263).float().unsqueeze(0),
                joints_num=22,
            ).squeeze(0).cpu().numpy()                              # (T, 22, 3)

            print(f"\n--- {sample['seq_id']}  T={seq_len} ---")
            print(f"world  pelvis @ t=0   = {joints_world[0, 0]}")
            print(f"canon  pelvis @ t=0   = {canonical[0, 0]}")
            print(f"canon  pelvis @ t=1   = {canonical[1, 0]}")
            print(f"motion_263[0, :4]     = {motion_263[0, :4]}")
            print(f"  (root_rot_vel, root_lin_vel_x, root_lin_vel_z, root_y)")

            # Hip + shoulder lines
            hip_w = joints_world[0, 2] - joints_world[0, 1]
            sdr_w = joints_world[0, 17] - joints_world[0, 16]
            across_w = hip_w + sdr_w
            hip_c = canonical[0, 2] - canonical[0, 1]
            sdr_c = canonical[0, 17] - canonical[0, 16]
            across_c = hip_c + sdr_c
            # forward = up × across = (across_z, 0, -across_x)
            fw_w_ang = math.atan2(across_w[2], -across_w[0])
            fw_c_ang = math.atan2(across_c[2], -across_c[0])
            print(f"world  hip={hip_w}  sdr={sdr_w}")
            print(f"world  across={across_w}  fwd_angle={math.degrees(fw_w_ang):+.2f}°")
            print(f"canon  hip={hip_c}  sdr={sdr_c}")
            print(f"canon  across={across_c}  fwd_angle={math.degrees(fw_c_ang):+.2f}°")

            # Run the full alignment
            R_y, T_xz, T_y = get_canonicalize_transform_from_clip(
                joints_world, canonical,
            )
            print(f"R_y_angle = {math.degrees(R_y):+.2f}°  T_xz = {T_xz}  T_y = {T_y:.4f}")

            # Reconstruct: world_pred = R_y(canon) + (T_xz[0], T_y, T_xz[1])
            R = y_rotation_matrix(R_y)
            recon_world = canonical @ R.T                            # (T, 22, 3)
            recon_world[..., 0] += T_xz[0]
            recon_world[..., 1] += T_y
            recon_world[..., 2] += T_xz[1]

            # Per-axis residuals at frame 0
            resid_t0 = joints_world[0] - recon_world[0]              # (22, 3)
            print(f"frame-0 reconstruction residual (world - recon):")
            print(f"  pelvis (joint 0)  = {resid_t0[0]}  |norm|={np.linalg.norm(resid_t0[0]):.4f}")
            print(f"  l_wrist (joint 20) = {resid_t0[20]} |norm|={np.linalg.norm(resid_t0[20]):.4f}")
            print(f"  per-axis MAE over 22 joints = "
                  f"X={np.abs(resid_t0[:, 0]).mean():.4f}  "
                  f"Y={np.abs(resid_t0[:, 1]).mean():.4f}  "
                  f"Z={np.abs(resid_t0[:, 2]).mean():.4f}")

            # Per-frame mean residual
            resid_all = joints_world - recon_world                    # (T, 22, 3)
            mean_per_frame = np.linalg.norm(resid_all, axis=-1).mean(axis=-1)  # (T,)
            print(f"per-frame mean |resid|: t=0:{mean_per_frame[0]:.4f}  "
                  f"t=T/2:{mean_per_frame[seq_len//2]:.4f}  "
                  f"t=T-1:{mean_per_frame[seq_len-1]:.4f}")

            # Per-frame pelvis Y in world vs canonical to see if Y diverges
            print(f"Per-frame pelvis Y (world vs canon vs delta):")
            for t in [0, seq_len // 4, seq_len // 2, 3 * seq_len // 4, seq_len - 1]:
                wy = joints_world[t, 0, 1]
                cy = canonical[t, 0, 1]
                print(f"  t={t:3d}  world_y={wy:.4f}  canon_y={cy:.4f}  delta={wy-cy:.4f}")

            # Lowest Y across frames (= what MoMask subtracts as floor_height
            # if applied to world joints).
            world_floor = joints_world.min(axis=0).min(axis=0)[1]
            canon_floor = canonical.min(axis=0).min(axis=0)[1]
            print(f"world clip-min Y (would-be floor_height) = {world_floor:.4f}")
            print(f"canon clip-min Y (should be ~0)           = {canon_floor:.4f}")


if __name__ == "__main__":
    main()
