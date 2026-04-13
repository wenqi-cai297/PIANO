"""HumanML3D 263-dimensional motion representation.

Converts SMPL 22-joint positions into the canonical 263-dim feature vector
used by MoMask, MLD, and other HumanML3D-ecosystem models.

The 263-dim vector per frame is composed of:
    - root angular velocity (1)
    - root linear velocity on xz plane (2)
    - root height (1)
    - joint positions relative to root, excluding root (21 × 3 = 63)
    - joint velocities (22 × 3 = 66)
    - joint rotations in 6D representation (21 × 6 = 126)
    - foot contact labels (4)
Total = 1 + 2 + 1 + 63 + 66 + 126 + 4 = 263

Reference: Guo et al., "Generating Diverse and Natural 3D Human Motions
from Text", CVPR 2022.
"""
from __future__ import annotations

import numpy as np

from piano.utils.smpl_utils import estimate_foot_contact


def joints_to_humanml3d(
    positions: np.ndarray,
    fps: float = 30.0,
) -> np.ndarray:
    """Convert joint positions to the HumanML3D 263-dim representation.

    This is a *simplified* version that computes the positional and velocity
    components.  The 6D rotation component requires SMPL body parameters
    (not just joint positions) and is handled separately when full SMPL
    parameters are available.

    Parameters
    ----------
    positions : (T, 22, 3) joint positions in world frame, y-up
    fps : capture frame rate

    Returns
    -------
    features : (T, 263) HumanML3D feature vector
    """
    T, J, _ = positions.shape
    assert J == 22, f"Expected 22 joints, got {J}"

    dt = 1.0 / fps

    # --- Root (pelvis) features ---
    root_pos = positions[:, 0, :]  # (T, 3)

    # Root height (y coordinate)
    root_height = root_pos[:, 1:2]  # (T, 1)

    # Root linear velocity on xz plane
    root_vel_xz = np.zeros((T, 2))
    root_vel_xz[1:] = (root_pos[1:, [0, 2]] - root_pos[:-1, [0, 2]]) / dt

    # Root angular velocity (approximate from facing direction change)
    # Use the cross product of consecutive facing directions projected to xz
    # Simplified: compute from spine direction
    spine_dir = positions[:, 6, :] - positions[:, 0, :]  # spine2 - pelvis
    facing = np.arctan2(spine_dir[:, 0], spine_dir[:, 2])  # angle in xz plane
    root_ang_vel = np.zeros((T, 1))
    root_ang_vel[1:, 0] = (facing[1:] - facing[:-1]) / dt
    # Handle angle wrapping
    root_ang_vel = np.where(
        np.abs(root_ang_vel) > np.pi / dt,
        root_ang_vel - np.sign(root_ang_vel) * 2 * np.pi / dt,
        root_ang_vel,
    )

    # --- Joint positions relative to root (exclude root itself) ---
    rel_positions = positions[:, 1:, :] - root_pos[:, None, :]  # (T, 21, 3)
    rel_positions_flat = rel_positions.reshape(T, -1)  # (T, 63)

    # --- Joint velocities (all 22 joints) ---
    joint_vel = np.zeros_like(positions)  # (T, 22, 3)
    joint_vel[1:] = (positions[1:] - positions[:-1]) / dt
    joint_vel_flat = joint_vel.reshape(T, -1)  # (T, 66)

    # --- 6D rotations placeholder ---
    # Full 6D rotation requires SMPL parameters (axis-angle or rotation matrices).
    # When only joint positions are available, we fill with zeros.
    # The proper pipeline should use SMPL forward kinematics output.
    rot_6d_flat = np.zeros((T, 126))  # (T, 21 × 6)

    # --- Foot contact ---
    foot_contact_lr = estimate_foot_contact(positions, fps=fps)  # (T, 2)
    # HumanML3D uses 4 contact labels: left_heel, left_toe, right_heel, right_toe
    # Simplified: duplicate ankle contact for heel and toe
    foot_contact = np.zeros((T, 4))
    foot_contact[:, 0] = foot_contact_lr[:, 0]  # left_heel ≈ left_ankle
    foot_contact[:, 1] = foot_contact_lr[:, 0]  # left_toe ≈ left_ankle
    foot_contact[:, 2] = foot_contact_lr[:, 1]  # right_heel
    foot_contact[:, 3] = foot_contact_lr[:, 1]  # right_toe

    # --- Concatenate ---
    features = np.concatenate([
        root_ang_vel,       # (T, 1)
        root_vel_xz,        # (T, 2)
        root_height,        # (T, 1)
        rel_positions_flat,  # (T, 63)
        joint_vel_flat,      # (T, 66)
        rot_6d_flat,         # (T, 126)
        foot_contact,        # (T, 4)
    ], axis=-1)

    assert features.shape == (T, 263), f"Expected (T, 263), got {features.shape}"
    return features


# ---------------------------------------------------------------------------
# Normalization using HumanML3D statistics
# ---------------------------------------------------------------------------

def normalize_motion(
    features: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Z-normalize motion features using dataset statistics.

    Parameters
    ----------
    features : (T, 263) or (B, T, 263)
    mean, std : (263,) — from HumanML3D dataset
    """
    return (features - mean) / np.clip(std, a_min=1e-8, a_max=None)


def denormalize_motion(
    features: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Reverse z-normalization."""
    return features * std + mean
