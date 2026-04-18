"""Extract contact state pseudo-labels from HOI motion data.

For each frame, computes whether each tracked body part (left_hand, right_hand,
left_foot, right_foot, pelvis) is in contact with the object surface.

Contact is detected by combining:
    1. Distance: body part joint is within ``distance_threshold`` of object surface
    2. Velocity: relative velocity between joint and object is below ``velocity_threshold``

The raw contact signal is temporally smoothed and filtered to remove
single-frame flickers.

Output: soft contact state array of shape ``(T, B)`` where B=5 body parts.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter

from piano.utils.geometry import points_to_mesh_distance
from piano.utils.smpl_utils import (
    BODY_PART_INDICES,
    NUM_BODY_PARTS,
    compute_joint_velocities,
)


@dataclass(slots=True)
class ContactConfig:
    """Configuration for contact state extraction."""

    distance_threshold: float = 0.02      # 2cm
    distance_sigma: float = 0.005         # sigmoid sharpness for distance
    velocity_threshold: float = 0.1       # 0.1 m/s
    velocity_sigma: float = 0.02          # sigmoid sharpness for velocity
    median_filter_size: int = 5           # temporal median filter window
    min_contact_duration: int = 3         # minimum consecutive frames for valid contact
    fps: float = 30.0


def _soft_sigmoid(x: np.ndarray, threshold: float, sigma: float) -> np.ndarray:
    """Smooth step function: 1 when x < threshold, 0 when x >> threshold.

    Uses ``scipy.special.expit`` (numerically stable logistic) instead of
    the naive ``1 / (1 + exp(...))`` to avoid overflow warnings when the
    argument is large (distances much farther than the threshold).
    """
    from scipy.special import expit
    return expit(-(x - threshold) / sigma)


def extract_contact_state(
    joints: np.ndarray,
    object_mesh: "trimesh.Trimesh",
    config: ContactConfig | None = None,
) -> np.ndarray:
    """Extract per-frame, per-body-part contact state.

    Parameters
    ----------
    joints : (T, 22, 3) — SMPL 22-joint positions
    object_mesh : trimesh.Trimesh — object mesh (static or per-frame)
    config : extraction parameters

    Returns
    -------
    contact : (T, 5) — soft contact probability for each body part
    """
    if config is None:
        config = ContactConfig()

    T = len(joints)
    contact = np.zeros((T, NUM_BODY_PARTS), dtype=np.float32)

    # Compute joint velocities for relative velocity check
    joint_vel = compute_joint_velocities(joints, fps=config.fps)

    for bp_idx, joint_idx in enumerate(BODY_PART_INDICES):
        bp_positions = joints[:, joint_idx, :]      # (T, 3)
        bp_velocities = joint_vel[:, joint_idx, :]  # (T, 3)
        bp_speed = np.linalg.norm(bp_velocities, axis=-1)  # (T,)

        # Distance to object surface
        distances, _ = points_to_mesh_distance(bp_positions, object_mesh)  # (T,)

        # Soft contact: high when close AND slow
        dist_score = _soft_sigmoid(distances, config.distance_threshold, config.distance_sigma)
        vel_score = _soft_sigmoid(bp_speed, config.velocity_threshold, config.velocity_sigma)
        contact[:, bp_idx] = dist_score * vel_score

    # Temporal smoothing
    for bp_idx in range(NUM_BODY_PARTS):
        contact[:, bp_idx] = median_filter(contact[:, bp_idx], size=config.median_filter_size)

    # Remove short contact events (below min_contact_duration)
    contact = _filter_short_contacts(contact, config.min_contact_duration)

    return contact


def _filter_short_contacts(
    contact: np.ndarray,
    min_duration: int,
    threshold: float = 0.5,
) -> np.ndarray:
    """Zero out contact segments shorter than *min_duration* frames.

    Operates on each body part independently. Only suppresses
    segments where the mean soft contact exceeds *threshold*.
    """
    result = contact.copy()
    T, B = contact.shape

    for bp in range(B):
        binary = contact[:, bp] > threshold
        # Find contiguous True segments
        changes = np.diff(binary.astype(np.int8), prepend=0, append=0)
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]

        for s, e in zip(starts, ends):
            if (e - s) < min_duration:
                result[s:e, bp] = 0.0

    return result
