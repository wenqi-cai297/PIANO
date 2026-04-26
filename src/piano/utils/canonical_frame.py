"""Canonical-frame ↔ world-frame helpers for HumanML3D-style motion data.

Stage B v0.2 introduced object pose channels into ``z_int``. The PIANO
backbone (MoMask) consumes HumanML3D-canonicalized motion (frame-0
pelvis at XZ origin, frame-0 heading aligned with +Z), so for the new
object position channels to be in the **same frame as the body** they
must be expressed in the body's canonical frame too. Per
``analyses/2026-04-27_object_conditioning_review.md`` §5.2, this
deviates from the world-frame consensus of 7 surveyed methods, but
PIANO is the only entry whose body representation is canonical (the
7 references all use world-frame body), so "object frame == body frame"
is the actual transferable principle and PIANO needs canonical for both.

Two operations live here:

1. :func:`get_canonicalize_transform_from_clip` — given a preprocessed
   clip's ``joints_22`` (world frame) and ``motion_263`` (canonical
   frame), recover ``(R_y, T_xz)`` such that
   ``world = R_y @ canonical + T_xz``. Uses frame-0 pelvis match for T
   and hip-line direction match for R_y (verified to align with
   HumanML3D's canonicalization in
   :mod:`scripts.stage_b_generator.qual_eval`).
2. :func:`world_to_canonical_object_pose` — applies the inverse of the
   above to per-frame world-frame object position + axis-angle rotation,
   producing canonical-frame `(obj_com_canonical: (T, 3),
   obj_rot6d_canonical: (T, 6))`. The 6D rotation rep is Zhou et al.
   *On the Continuity of Rotation Representations in Neural Networks.*
   CVPR 2019. arXiv:1812.07035 — first 2 columns of the rotation matrix
   flattened, the standard for learning rotations.

These helpers are pure-numpy / pure-pytorch and do not depend on
the MoMask repo (so HOIDataset can call them at ``__getitem__`` time
without paying the MoMask sys.path import cost).
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor


# ============================================================================
# Forward / inverse Y-axis rotations + 6D rep
# ============================================================================

def axis_angle_to_matrix_np(aa: np.ndarray) -> np.ndarray:
    """Rodrigues: ``(..., 3)`` axis-angle → ``(..., 3, 3)`` rotation matrix.

    Vectorised over leading dims. NaN-safe at θ ≈ 0 (returns identity).
    """
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)         # (..., 1)
    safe_theta = np.where(theta < 1e-8, 1.0, theta)
    axis = aa / safe_theta                                     # (..., 3)
    cos_t = np.cos(theta)[..., None]                           # (..., 1, 1)
    sin_t = np.sin(theta)[..., None]
    one_m_cos = 1.0 - cos_t

    # Cross-product matrix K of axis.
    zeros = np.zeros_like(axis[..., 0])
    K = np.stack(
        [
            np.stack([zeros, -axis[..., 2], axis[..., 1]], axis=-1),
            np.stack([axis[..., 2], zeros, -axis[..., 0]], axis=-1),
            np.stack([-axis[..., 1], axis[..., 0], zeros], axis=-1),
        ],
        axis=-2,
    )                                                          # (..., 3, 3)
    eye = np.broadcast_to(np.eye(3, dtype=aa.dtype), K.shape)
    R = eye + sin_t * K + one_m_cos * (K @ K)                  # (..., 3, 3)
    # Identity at θ ≈ 0.
    is_small = theta < 1e-8                                    # (..., 1)
    R = np.where(is_small[..., None], eye, R)
    return R.astype(aa.dtype, copy=False)


def matrix_to_rotation_6d_np(R: np.ndarray) -> np.ndarray:
    """``(..., 3, 3)`` rotation matrix → ``(..., 6)`` Zhou-2019 6D rep.

    Takes the **first two columns** of R, flattened in row-major order
    (matching pytorch3d's ``matrix_to_rotation_6d`` and Zhou et al.'s
    paper Eq. 6).
    """
    return R[..., :, :2].reshape(*R.shape[:-2], 6).copy()


def y_rotation_matrix(angle: float) -> np.ndarray:
    """``(3, 3)`` rotation matrix around +Y by ``angle`` radians."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array(
        [[c, 0.0, s],
         [0.0, 1.0, 0.0],
         [-s, 0.0, c]],
        dtype=np.float32,
    )


# ============================================================================
# Canonicalization transform derivation
# ============================================================================

def _hip_line_angle_y(joints_t0: np.ndarray) -> float:
    """Y-axis angle of the right-hip → left-hip line at frame 0.

    Uses joint 1 (left hip) and joint 2 (right hip) in the SMPL-22
    convention. The hip line is roughly horizontal across the pelvis
    and rotates with the body's facing direction — exactly the signal
    HumanML3D canonicalization aligns to +Z. Returns the angle whose
    rotation around +Y maps canonical hip-line to world hip-line.
    """
    hip = joints_t0[2] - joints_t0[1]                          # (3,)
    return float(math.atan2(hip[0], hip[2]))


def get_canonicalize_transform_from_clip(
    joints_world: np.ndarray,
    canonical_joints: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Recover ``(R_y_angle, T_xz)`` mapping canonical → world for one clip.

    Parameters
    ----------
    joints_world : (T, 22, 3)
        World-frame SMPL-22 joints (PIANO ``joints_22`` field).
    canonical_joints : (T, 22, 3)
        Canonical-frame SMPL-22 joints (output of MoMask's
        ``recover_from_ric`` on ``motion_263``).

    Returns
    -------
    R_y_angle : float
        Rotation around +Y, in radians, that takes canonical → world.
    T_xz : (2,) float32
        XZ translation that takes canonical → world. Y is preserved by
        canonicalization, so no T_y component.
    """
    world_t0 = joints_world[0]
    canon_t0 = canonical_joints[0]
    R_y_angle = _hip_line_angle_y(world_t0) - _hip_line_angle_y(canon_t0)
    # After applying the rotation to the canonical pelvis, the
    # remaining gap to the world pelvis is the translation.
    R = y_rotation_matrix(R_y_angle)
    rotated_canon_pelvis = R @ canon_t0[0]
    T_xz = (world_t0[0, [0, 2]] - rotated_canon_pelvis[[0, 2]]).astype(np.float32)
    return R_y_angle, T_xz


# ============================================================================
# Object pose: world → canonical
# ============================================================================

def world_to_canonical_object_pose(
    obj_pos_world: np.ndarray,                  # (T, 3) per-frame COM
    obj_rot_world_axis_angle: np.ndarray,       # (T, 3) per-frame axis-angle
    R_y_angle: float,
    T_xz: np.ndarray,                           # (2,) — canonical→world
) -> tuple[np.ndarray, np.ndarray]:
    """Express world-frame object pose in body's canonical frame.

    Inverts ``(R_y, T_xz)`` to map world points back into canonical:
    ``canonical = R_y(-θ) @ (world - T_xz)``. Y is unchanged. The
    rotation portion of the object pose is similarly composed:
    ``R_obj_canonical = R_y(-θ) @ R_obj_world``, then converted to 6D.

    Returns
    -------
    obj_com_canonical : (T, 3) float32
    obj_rot6d_canonical : (T, 6) float32 (Zhou et al. 2019 6D rep)
    """
    obj_pos_world = obj_pos_world.astype(np.float32, copy=False)
    obj_rot_world_axis_angle = obj_rot_world_axis_angle.astype(np.float32, copy=False)

    # Inverse rotation matrix (rotation around +Y by -angle).
    R_inv = y_rotation_matrix(-R_y_angle)                      # (3, 3)

    # Position: subtract translation, then rotate.
    pos_centered = obj_pos_world.copy()
    pos_centered[..., 0] -= float(T_xz[0])
    pos_centered[..., 2] -= float(T_xz[1])
    obj_com_canonical = pos_centered @ R_inv.T                 # (T, 3)

    # Rotation: world axis-angle → world matrix → R_inv @ R → 6D.
    R_obj_world = axis_angle_to_matrix_np(obj_rot_world_axis_angle)   # (T, 3, 3)
    R_obj_canonical = np.einsum("ij,tjk->tik", R_inv, R_obj_world)
    obj_rot6d_canonical = matrix_to_rotation_6d_np(R_obj_canonical)   # (T, 6)

    return obj_com_canonical.astype(np.float32), obj_rot6d_canonical.astype(np.float32)


# ============================================================================
# Torch-side per-batch one-shot wrapper (used by tokenizer / step_fn)
# ============================================================================

def axis_angle_to_rotation_6d_torch(aa: Tensor) -> Tensor:
    """``(..., 3)`` axis-angle → ``(..., 6)`` 6D rotation rep, in torch.

    Same algorithm as the numpy variants above but stays on the input
    tensor's device + dtype so it can run inside the GPU step_fn
    without an extra CPU round-trip.
    """
    theta = aa.norm(dim=-1, keepdim=True).clamp(min=1e-8)              # (..., 1)
    axis = aa / theta
    cos_t = theta.cos().unsqueeze(-1)                                  # (..., 1, 1)
    sin_t = theta.sin().unsqueeze(-1)
    one_m_cos = 1.0 - cos_t

    zeros = torch.zeros_like(axis[..., 0])
    K = torch.stack(
        [
            torch.stack([zeros, -axis[..., 2], axis[..., 1]], dim=-1),
            torch.stack([axis[..., 2], zeros, -axis[..., 0]], dim=-1),
            torch.stack([-axis[..., 1], axis[..., 0], zeros], dim=-1),
        ],
        dim=-2,
    )
    eye = torch.eye(3, device=aa.device, dtype=aa.dtype).expand_as(K)
    R = eye + sin_t * K + one_m_cos * (K @ K)
    return R[..., :, :2].reshape(*aa.shape[:-1], 6).contiguous()
