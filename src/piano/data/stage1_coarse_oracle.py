"""Round-22 Stage-2 oracle Coarse-v1 adapter.

Extracts the Stage-1 23-D Coarse-v1 representation from a batch of GT
``motion_135`` for Stage-2 conditioning. The math is mirrored line-by-line
from the canonical extraction at:

    scripts/stage_b_generator/extract_coarse_motion_representation.py::extract_coarse_v0_v1

with the same constants (joint indices, channel order) and the same project
upstream APIs (``piano.training.smpl_kinematics.rotation_6d_to_matrix`` and
``fk_from_global_rotations``). A batched torch implementation is needed
because the Stage-2 trainer must extract Coarse-v1 inside ``step_fn`` without
a Python-level per-clip loop.

Equivalence is verified against ``extract_coarse_v0_v1`` on a single-clip
roundtrip — see ``tests/test_stage2_stage1_coarse_condition.py``.

Design source: ``analyses/2026-05-22_stage2_condition_reframe_and_next_plan.md`` §6.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from piano.training.smpl_kinematics import (
    fk_from_global_rotations,
    rotation_6d_to_matrix,
)


# Match the constants defined in extract_coarse_motion_representation.py:67-71.
J_PELVIS: int = 0
J_SPINE3: int = 9
J_HEAD: int = 15
J_L_SHOULDER: int = 16
J_R_SHOULDER: int = 17

COARSE_V0_DIM: int = 15
COARSE_V1_DIM: int = 23


def _unwrap_yaw_torch(yaw: Tensor) -> Tensor:
    """Batched equivalent of ``numpy.unwrap(yaw, axis=-1)``.

    yaw : (B, T) radians.
    Returns yaw_unwrapped : (B, T) where successive frame diffs are wrapped
    into ``[-pi, pi]`` before being re-accumulated. Matches NumPy's
    ``np.unwrap`` default (period=2*pi, discont=pi) on the trailing axis.
    """
    if yaw.shape[-1] < 2:
        return yaw
    diff = yaw[..., 1:] - yaw[..., :-1]
    pi = float(np.pi)
    # Wrap diff to [-pi, pi]
    diff = (diff + pi) % (2.0 * pi) - pi
    # Restore corner case: np.unwrap maps a +pi diff to +pi (not -pi).
    # Modulo above gives -pi for diff == pi, so adjust.
    diff = torch.where(
        (diff == -pi) & ((yaw[..., 1:] - yaw[..., :-1]) > 0),
        torch.full_like(diff, pi),
        diff,
    )
    out_diff = (yaw[..., 1:] - yaw[..., :-1]) - (
        ((yaw[..., 1:] - yaw[..., :-1]) - diff)
    )
    # Equivalent: out_diff = diff. Re-derive from cumsum.
    yaw_unwrapped = torch.cat(
        [yaw[..., :1], yaw[..., :1] + diff.cumsum(dim=-1)], dim=-1,
    )
    return yaw_unwrapped


def _facing_yaw_from_pelvis_rot6d_torch(pelvis_rot6d: Tensor) -> Tensor:
    """Compute facing yaw (radians) from pelvis global rot6d.

    pelvis_rot6d : (..., 6)
    Returns yaw : (...,) in radians.

    Mirrors ``extract_coarse_motion_representation._facing_yaw_from_pelvis_rot6d``
    (lines 92-119). Convention:
        - rot6d -> matrix via project upstream ``rotation_6d_to_matrix``
          (column-stacking convention; matches ``matrix_to_rotation_6d``).
        - body local forward = +Z_local; forward in world = R[..., :, 2]
        - yaw_world = atan2(forward_x, forward_z)
    """
    R = rotation_6d_to_matrix(pelvis_rot6d)             # (..., 3, 3)
    forward = R[..., :, 2]                              # (..., 3)
    fx = forward[..., 0]
    fz = forward[..., 2]
    return torch.atan2(fx, fz)


def extract_coarse_v1_batched(
    motion: Tensor,            # (B, T, 135)
    rest_offsets: Tensor,      # (B, 22, 3)
) -> Tensor:
    """Compute Coarse-v1 (B, T, 23) from a batch of motion_135 + rest_offsets.

    Channel layout (matches extract_coarse_v0_v1 in
    scripts/stage_b_generator/extract_coarse_motion_representation.py:122-203):

        [0]    root_local_x  = root_world_x - root_world_x[0]
        [1]    root_local_z  = root_world_z - root_world_z[0]
        [2]    root_local_y  = root_world_y - root_world_y[0]
        [3]    vel_x         = diff(root_world_x, prepend=root_world_x[0])
        [4]    vel_z         = ...
        [5]    vel_y         = ...
        [6]    yaw_sin       = sin(unwrap(yaw_from_pelvis_rot6d))
        [7]    yaw_cos       = cos(...)
        [8]    yaw_vel       = diff(unwrap(yaw), prepend=unwrap(yaw)[0])
        [9..14]  pelvis_rot6d (6)
        [15..20] spine3_rot6d (6)
        [21]   head_height_y                   from FK
        [22]   shoulder_center_height_y        from FK

    Frame convention: root_local is root0-relative world-axis (no rotation).
    This matches the S1-O ``obj_traj_root0_world`` frame, so the Stage-2
    motion frame (``force_world_frame: true``) needs no extra alignment.

    Pre-conditions
    --------------
    - ``motion`` must be valid for all (B, T) — padded frames are computed
      mechanically; the trainer is responsible for masking via seq_mask
      downstream.
    - ``rest_offsets`` must be the same per-clip rest_offsets as the v18
      Stage-2 batch dict (FK joints must use the SAME rest_offsets the
      trainer's anchor-loss FK uses, or velocity / shoulder height will
      diverge silently).

    Returns
    -------
    coarse_v1 : (B, T, 23) float32 tensor, same device as inputs.
    """
    if motion.dim() != 3 or motion.shape[-1] != 135:
        raise ValueError(
            f"motion must be (B, T, 135); got {tuple(motion.shape)}"
        )
    if rest_offsets.dim() != 3 or rest_offsets.shape[1:] != (22, 3):
        raise ValueError(
            f"rest_offsets must be (B, 22, 3); got {tuple(rest_offsets.shape)}"
        )
    if motion.shape[0] != rest_offsets.shape[0]:
        raise ValueError(
            f"batch dim mismatch: motion {motion.shape[0]} vs "
            f"rest_offsets {rest_offsets.shape[0]}"
        )

    B, T, _ = motion.shape
    rot6d = motion[..., :132].reshape(B, T, 22, 6).float()     # (B, T, 22, 6)
    root_world = motion[..., 132:135].float()                  # (B, T, 3)

    # Root local (relative to frame 0).
    root_local = root_world - root_world[:, :1, :]              # (B, T, 3)
    root_local_x = root_local[..., 0]
    root_local_y = root_local[..., 1]
    root_local_z = root_local[..., 2]

    # Per-frame velocity (diff with prepend == first-frame diff = 0).
    if T >= 2:
        vel_world = torch.cat(
            [torch.zeros_like(root_world[:, :1, :]),
             root_world[:, 1:, :] - root_world[:, :-1, :]],
            dim=1,
        )
    else:
        vel_world = torch.zeros_like(root_world)
    vel_x = vel_world[..., 0]
    vel_y = vel_world[..., 1]
    vel_z = vel_world[..., 2]

    # Facing yaw from pelvis (joint 0) global rot6d.
    pelvis_rot6d = rot6d[:, :, J_PELVIS, :]                     # (B, T, 6)
    yaw_raw = _facing_yaw_from_pelvis_rot6d_torch(pelvis_rot6d) # (B, T)
    yaw_unwrapped = _unwrap_yaw_torch(yaw_raw)                  # (B, T)
    yaw_sin = torch.sin(yaw_unwrapped)
    yaw_cos = torch.cos(yaw_unwrapped)
    if T >= 2:
        yaw_vel = torch.cat(
            [torch.zeros_like(yaw_unwrapped[:, :1]),
             yaw_unwrapped[:, 1:] - yaw_unwrapped[:, :-1]],
            dim=1,
        )
    else:
        yaw_vel = torch.zeros_like(yaw_unwrapped)

    # Coarse-v0 (T, 9) channels.
    coarse_v0_lin = torch.stack(
        [root_local_x, root_local_z, root_local_y,
         vel_x, vel_z, vel_y,
         yaw_sin, yaw_cos, yaw_vel],
        dim=-1,
    )                                                            # (B, T, 9)
    coarse_v0 = torch.cat([coarse_v0_lin, pelvis_rot6d], dim=-1) # (B, T, 15)

    # FK for head + shoulder heights.
    rot_mat = rotation_6d_to_matrix(rot6d)                       # (B, T, 22, 3, 3)
    rest_per_frame = rest_offsets.unsqueeze(1).expand(B, T, 22, 3).float()
    joints = fk_from_global_rotations(rot_mat, rest_per_frame, root_world)
    # joints: (B, T, 22, 3)
    spine3_rot6d = rot6d[:, :, J_SPINE3, :]                      # (B, T, 6)
    head_height = joints[..., J_HEAD, 1].unsqueeze(-1)           # (B, T, 1)
    shoulder_center_h = (
        (joints[..., J_L_SHOULDER, 1] + joints[..., J_R_SHOULDER, 1]) * 0.5
    ).unsqueeze(-1)                                              # (B, T, 1)

    coarse_v1_extra = torch.cat(
        [spine3_rot6d, head_height, shoulder_center_h], dim=-1,
    )                                                            # (B, T, 8)
    coarse_v1 = torch.cat([coarse_v0, coarse_v1_extra], dim=-1)  # (B, T, 23)
    return coarse_v1.float()


# ---------------------------------------------------------------------------
# Stage-1 normalization stats loader
# ---------------------------------------------------------------------------


def load_stage1_coarse_norm(
    cache_root: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load Stage-1 train normalization stats (mean, std) for the 23-D
    Coarse-v1 from the Stage-1 cache normalization JSON.

    Returns (mean, std_clamped) as float32 ndarrays of shape (23,).

    Path used:
        ``<cache_root>/normalization_train.json`` →
            global.mean (23 floats) + global.std_clamped (23 floats).

    Used by the Stage-2 trainer to z-score the oracle Coarse-v1 in
    consistent units with the Stage-1 S1-O ckpt. Switching to the S1-O
    sampler later does not require re-fitting these stats.
    """
    p = Path(cache_root) / "normalization_train.json"
    if not p.exists():
        raise FileNotFoundError(
            f"Stage-1 normalization stats missing at {p}. "
            "Run scripts/stage_b_generator/build_stage1_coarse_v1_cache.py "
            "to produce the cache."
        )
    payload: dict[str, Any] = json.loads(p.read_text("utf-8"))
    glob = payload["global"]
    mean = np.asarray(glob["mean"], dtype=np.float32)
    std = np.asarray(glob["std_clamped"], dtype=np.float32)
    if mean.shape != (COARSE_V1_DIM,) or std.shape != (COARSE_V1_DIM,):
        raise ValueError(
            f"Stage-1 norm stats expected shape ({COARSE_V1_DIM},); "
            f"got mean={mean.shape}, std={std.shape}"
        )
    return mean, std
