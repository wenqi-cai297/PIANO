"""SMPL-22 forward / inverse kinematics utilities for AnchorDiff v5.

Ports the minimal subset of OMOMO's
``omomo_release/manip/data/hand_foot_dataset.py`` (the FK / IK / global
rotation chain functions) to a self-contained module that does not need
SMPL+H ``model.npz`` for the kintree (we hard-code the SMPL-22 parents
since SMPL / SMPL+H / SMPL-X all share the same first 22 joints).

The 6D rotation parameterisation follows Zhou et al. *On the Continuity
of Rotation Representations in Neural Networks* (CVPR 2019), via
``pytorch3d.transforms`` — same package OMOMO/CHOIS use.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch import Tensor

# SMPL-22 kinematic tree (parent index for each joint).
# Identical to omomo_release@manip/data/hand_foot_dataset.py:get_smpl_parents()
# (which reads it from SMPL+H model.npz). Hard-coded here so the module
# does not depend on a downloaded body-model file.
#
# Joint id -> name (SMPL convention):
#   0=pelvis, 1=L_hip, 2=R_hip, 3=spine1, 4=L_knee, 5=R_knee,
#   6=spine2, 7=L_ankle, 8=R_ankle, 9=spine3, 10=L_foot, 11=R_foot,
#   12=neck, 13=L_collar, 14=R_collar, 15=head, 16=L_shoulder,
#   17=R_shoulder, 18=L_elbow, 19=R_elbow, 20=L_wrist, 21=R_wrist
SMPL22_PARENTS: tuple[int, ...] = (
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19,
)


def get_smpl22_parents() -> np.ndarray:
    """Return the SMPL-22 parent table as a numpy array."""
    return np.asarray(SMPL22_PARENTS, dtype=np.int64)


def local2global_pose(local_rot_mat: Tensor) -> Tensor:
    """Chain SMPL-22 local rotations into per-joint global rotations.

    Mirrors omomo_release@manip/data/hand_foot_dataset.py:62-77
    ``local2global_pose`` exactly, except the kintree is hard-coded.

    Parameters
    ----------
    local_rot_mat : (..., 22, 3, 3)
        Per-joint local rotation matrix (rotation of joint i relative to
        joint parents[i]). Joint 0's "local" rotation is the global
        root orientation.

    Returns
    -------
    global_rot_mat : (..., 22, 3, 3)
        Per-joint orientation in the world frame.
    """
    parents = get_smpl22_parents()
    global_pose = local_rot_mat.clone()
    for j in range(1, len(parents)):
        p = int(parents[j])
        global_pose[..., j, :, :] = torch.matmul(
            global_pose[..., p, :, :], global_pose[..., j, :, :]
        )
    return global_pose


def global2local_pose(global_rot_mat: Tensor) -> Tensor:
    """Invert ``local2global_pose``.

    Mirrors ``quat_ik_torch`` in OMOMO but in matrix space:
        local[i] = global[parent[i]].T @ global[i]

    Parameters
    ----------
    global_rot_mat : (..., 22, 3, 3)

    Returns
    -------
    local_rot_mat : (..., 22, 3, 3)
    """
    parents = get_smpl22_parents()
    local_pose = global_rot_mat.clone()
    for j in range(1, len(parents)):
        p = int(parents[j])
        # local = parent_global^{-1} @ global = parent_global.T @ global
        local_pose[..., j, :, :] = torch.matmul(
            global_rot_mat[..., p, :, :].transpose(-1, -2),
            global_rot_mat[..., j, :, :],
        )
    return local_pose


def fk_from_global_rotations(
    global_rot_mat: Tensor,        # (..., 22, 3, 3)
    rest_offsets: Tensor,          # (..., 22, 3) — root row is ignored
    root_world_pos: Tensor,        # (..., 3)
) -> Tensor:
    """Forward kinematics that takes already-chained global rotations.

    Per joint j > 0:
        joint_pos[j] = joint_pos[parent[j]]
                       + global_rot_mat[parent[j]] @ rest_offset_local[j]

    where ``rest_offset_local[j] = T_pose_joint[j] - T_pose_joint[parent[j]]``
    (i.e. the bone vector from the parent to j in T-pose, in T-pose
    coordinates).

    Equivalent to OMOMO's ``quat_fk_torch`` but consumes global rotations
    directly. This avoids one IK pass when the network already predicts
    global rotations.

    Parameters
    ----------
    global_rot_mat : (..., 22, 3, 3)
    rest_offsets : (..., 22, 3) — bone offsets from parent in T-pose
        coordinates. Row 0 (root) is ignored; root position is set by
        ``root_world_pos``.
    root_world_pos : (..., 3) — root joint world position per frame.

    Returns
    -------
    joints_world : (..., 22, 3)
    """
    parents = get_smpl22_parents()
    n_joints = len(parents)
    out_shape = list(rest_offsets.shape)        # (..., 22, 3)
    joints = torch.empty(
        out_shape, dtype=rest_offsets.dtype, device=rest_offsets.device,
    )
    joints[..., 0, :] = root_world_pos
    for j in range(1, n_joints):
        p = int(parents[j])
        # global_rot_mat[parent] @ rest_offset[j]: (..., 3, 3) @ (..., 3) -> (..., 3)
        rotated = torch.einsum(
            "...ij,...j->...i",
            global_rot_mat[..., p, :, :],
            rest_offsets[..., j, :],
        )
        joints[..., j, :] = joints[..., p, :] + rotated
    return joints


def compute_rest_offsets_from_smplx_layer(
    betas: Tensor,                 # (B, 10)
    smplx_layer,                   # smplx.SMPLXLayer or smplx.create() output
) -> Tensor:
    """Run the SMPL-X body model with zero pose to get per-subject T-pose
    joint offsets for the first 22 joints.

    Parameters
    ----------
    betas : (B, 10)
    smplx_layer : a callable returning an object with .joints (B, J, 3).
        We use the first 22 joints (SMPL-22 layout); extra SMPL-X joints
        (jaw / eyes / fingers) are ignored.

    Returns
    -------
    rest_offsets : (B, 22, 3)
        ``rest_offsets[..., 0, :] = 0`` (root has no offset).
        ``rest_offsets[..., j, :] = T_joints[j] - T_joints[parents[j]]``
        for j > 0.
    """
    parents = get_smpl22_parents()
    out = smplx_layer(betas=betas)
    t_joints = out.joints[:, :22, :]            # (B, 22, 3)
    rest_offsets = torch.zeros_like(t_joints)
    for j in range(1, len(parents)):
        p = int(parents[j])
        rest_offsets[:, j, :] = t_joints[:, j, :] - t_joints[:, p, :]
    return rest_offsets


def axis_angle_to_matrix(aa: Tensor) -> Tensor:
    """Axis-angle (..., 3) -> rotation matrix (..., 3, 3) via Rodrigues.

    Self-contained implementation; piano env does not include pytorch3d.
    Numerically equivalent to pytorch3d.transforms.axis_angle_to_matrix
    that OMOMO/CHOIS use.
    """
    angle = aa.norm(dim=-1, keepdim=True)                   # (..., 1)
    axis = aa / angle.clamp_min(1e-8)                       # (..., 3)
    a0, a1, a2 = axis.unbind(dim=-1)
    zero = torch.zeros_like(a0)
    K = torch.stack(
        [
            torch.stack([zero, -a2, a1], dim=-1),
            torch.stack([a2, zero, -a0], dim=-1),
            torch.stack([-a1, a0, zero], dim=-1),
        ],
        dim=-2,
    )                                                       # (..., 3, 3)
    angle_unsq = angle.unsqueeze(-1)                        # (..., 1, 1)
    I = torch.eye(3, dtype=aa.dtype, device=aa.device).expand_as(K)
    return I + torch.sin(angle_unsq) * K + (1 - torch.cos(angle_unsq)) * (K @ K)


def matrix_to_rotation_6d(rot_mat: Tensor) -> Tensor:
    """(..., 3, 3) -> (..., 6) following Zhou et al. (CVPR 2019).

    The 6D rep stores the first two columns of the rotation matrix
    (the third can be recovered by cross product). pytorch3d's
    ``matrix_to_rotation_6d`` actually concatenates the first two ROWS
    (i.e. ``R[..., :2, :].reshape(..., 6)``); we follow the same
    convention so models trained against pytorch3d's helpers are
    compatible.
    """
    return rot_mat[..., :2, :].reshape(rot_mat.shape[:-2] + (6,)).contiguous()


def rotation_6d_to_matrix(d6: Tensor) -> Tensor:
    """(..., 6) -> (..., 3, 3) via Gram-Schmidt (Zhou et al. CVPR 2019).

    Inverse of ``matrix_to_rotation_6d``. Equivalent to
    ``pytorch3d.transforms.rotation_6d_to_matrix``.
    """
    a1 = d6[..., :3]
    a2 = d6[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.linalg.cross(b1, b2, dim=-1)
    # pytorch3d packs as rows ([b1; b2; b3]), giving R[i,:] = b_i.
    return torch.stack((b1, b2, b3), dim=-2)


def smplx_pose_to_global_rot_6d(
    smplx_pose_aa: Tensor,         # (B, T, 22, 3) axis-angle local rotations
                                   # (root_orient + body_pose for SMPL-22)
) -> Tuple[Tensor, Tensor]:
    """Convert SMPL-22 axis-angle local pose to (local_rot_mat, global_rot_6d).

    Parameters
    ----------
    smplx_pose_aa : (B, T, 22, 3)

    Returns
    -------
    local_rot_mat : (B, T, 22, 3, 3)
    global_rot_6d : (B, T, 22, 6)
    """
    local_rot_mat = axis_angle_to_matrix(smplx_pose_aa)
    # Flatten (B, T) so local2global_pose's loop is amortized.
    bsz, tsz = local_rot_mat.shape[:2]
    flat = local_rot_mat.reshape(bsz * tsz, 22, 3, 3)
    global_flat = local2global_pose(flat)
    global_rot_mat = global_flat.reshape(bsz, tsz, 22, 3, 3)
    global_rot_6d = matrix_to_rotation_6d(global_rot_mat)
    return local_rot_mat, global_rot_6d
