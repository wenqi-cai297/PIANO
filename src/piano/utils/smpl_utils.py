"""SMPL / SMPL-X body model utilities.

Handles the conversion from SMPL-X (with fingers) to SMPL 22-joint
representation used by HumanML3D, and provides joint name constants.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Joint name constants
# ---------------------------------------------------------------------------

# SMPL 22 joints used by HumanML3D (indices into SMPL joint regressor output)
SMPL_22_JOINT_NAMES: list[str] = [
    "pelvis",           # 0
    "left_hip",         # 1
    "right_hip",        # 2
    "spine1",           # 3
    "left_knee",        # 4
    "right_knee",       # 5
    "spine2",           # 6
    "left_ankle",       # 7
    "right_ankle",      # 8
    "spine3",           # 9
    "left_foot",        # 10
    "right_foot",       # 11
    "neck",             # 12
    "left_collar",      # 13
    "right_collar",     # 14
    "head",             # 15
    "left_shoulder",    # 16
    "right_shoulder",   # 17
    "left_elbow",       # 18
    "right_elbow",      # 19
    "left_wrist",       # 20
    "right_wrist",      # 21
]

# Body parts tracked for interaction pseudo-labels.
#
# Foot joints use SMPL idx 10/11 (left_foot/right_foot — mid-foot), NOT
# 7/8 (ankles). v1-v8 used ankles, which sit 8-10 cm above the sole. With
# threshold 0.06 that meant "ankle joint within 6 cm of mesh", which is
# only satisfied when the foot has penetrated the mesh — essentially
# never. The result was that ~99% of imhd/neuraldome/omomo clips had
# zero foot contact, and every "kick" / "use the foot to scoot" clip
# on omomo was dropped by cleaning (394/398 of omomo drops in v8 were
# foot-based actions). foot joints at idx 10/11 sit ~4-5 cm above sole,
# so threshold 0.06 becomes "sole within 1-2 cm of mesh" = a physically
# meaningful "foot-on-object" test. See
# analyses/2026-04-24_v9_kin_coupling.md for the data-driven evidence.
INTERACTION_BODY_PARTS: dict[str, int] = {
    "left_hand": 20,    # left_wrist joint index
    "right_hand": 21,   # right_wrist joint index
    "left_foot": 10,    # left_foot (mid-foot) — idx 7 is ankle, too far above sole
    "right_foot": 11,   # right_foot (mid-foot)
    "pelvis": 0,        # pelvis joint index
}

NUM_BODY_PARTS: int = len(INTERACTION_BODY_PARTS)
BODY_PART_NAMES: list[str] = list(INTERACTION_BODY_PARTS.keys())
BODY_PART_INDICES: list[int] = list(INTERACTION_BODY_PARTS.values())


# ---------------------------------------------------------------------------
# SMPL-X to SMPL 22-joint conversion
# ---------------------------------------------------------------------------

# SMPL-X has 55 joints (22 body + 30 hand + 3 jaw/eye).
# We keep only the first 22 body joints to match SMPL / HumanML3D.
SMPLX_TO_SMPL22_INDICES: list[int] = list(range(22))


def smplx_joints_to_smpl22(joints: np.ndarray) -> np.ndarray:
    """Extract SMPL 22 joints from SMPL-X joint positions.

    Parameters
    ----------
    joints : (..., J, 3) array where J >= 22 (SMPL-X joints)

    Returns
    -------
    smpl22 : (..., 22, 3) array
    """
    return joints[..., SMPLX_TO_SMPL22_INDICES, :]


# ---------------------------------------------------------------------------
# Basic kinematic helpers
# ---------------------------------------------------------------------------

def compute_joint_velocities(
    positions: np.ndarray,
    fps: float = 30.0,
) -> np.ndarray:
    """Compute per-joint velocities via finite difference.

    Parameters
    ----------
    positions : (T, J, 3) joint positions
    fps : frame rate

    Returns
    -------
    velocities : (T, J, 3) — first frame velocity is set to zero
    """
    dt = 1.0 / fps
    vel = np.zeros_like(positions)
    vel[1:] = (positions[1:] - positions[:-1]) / dt
    return vel


def compute_root_velocity(
    positions: np.ndarray,
    fps: float = 30.0,
) -> np.ndarray:
    """Compute root (pelvis) velocity on the ground plane (xz).

    Parameters
    ----------
    positions : (T, J, 3) joint positions — joint 0 is pelvis

    Returns
    -------
    root_vel : (T, 2) — [vx, vz] per frame, first frame is zero
    """
    root = positions[:, 0, :]  # (T, 3)
    dt = 1.0 / fps
    vel = np.zeros((len(root), 2))
    vel[1:, 0] = (root[1:, 0] - root[:-1, 0]) / dt  # x
    vel[1:, 1] = (root[1:, 2] - root[:-1, 2]) / dt  # z
    return vel


def estimate_foot_contact(
    positions: np.ndarray,
    height_threshold: float = 0.05,
    velocity_threshold: float = 0.5,
    fps: float = 30.0,
) -> np.ndarray:
    """Heuristic foot-ground contact detection.

    A foot is in contact if its height (y) is below *height_threshold*
    and its velocity magnitude is below *velocity_threshold*.

    Parameters
    ----------
    positions : (T, J, 3) joint positions
    height_threshold : meters above ground
    velocity_threshold : m/s

    Returns
    -------
    contact : (T, 2) binary array — [left_foot, right_foot]
    """
    left_ankle = positions[:, 7, :]   # (T, 3)
    right_ankle = positions[:, 8, :]  # (T, 3)

    vel = compute_joint_velocities(positions, fps)
    left_vel = np.linalg.norm(vel[:, 7, :], axis=-1)
    right_vel = np.linalg.norm(vel[:, 8, :], axis=-1)

    left_contact = (left_ankle[:, 1] < height_threshold) & (left_vel < velocity_threshold)
    right_contact = (right_ankle[:, 1] < height_threshold) & (right_vel < velocity_threshold)

    return np.stack([left_contact, right_contact], axis=-1).astype(np.float32)


def approximate_arm_length(positions: np.ndarray) -> float:
    """Estimate arm length from a single pose as shoulder-to-wrist distance.

    Uses the mean of left and right arms from the first frame.

    Parameters
    ----------
    positions : (T, 22, 3) or (22, 3) joint positions
    """
    if positions.ndim == 3:
        positions = positions[0]  # use first frame

    left_arm = np.linalg.norm(positions[20] - positions[16])   # wrist - shoulder
    right_arm = np.linalg.norm(positions[21] - positions[17])
    return float((left_arm + right_arm) / 2.0)
