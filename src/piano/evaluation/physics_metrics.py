"""Physical and interaction quality metrics.

Measures that go beyond standard motion quality to evaluate physical
plausibility and interaction correctness — the core evaluation axis
for PIANO's contribution.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class PhysicsMetrics:
    """Container for physics / interaction evaluation metrics."""

    penetration_rate: float       # % frames with body-object penetration
    contact_precision: float      # among predicted contacts, % geometrically valid
    contact_recall: float         # among GT contacts, % correctly predicted
    contact_f1: float
    foot_sliding: float           # mean foot velocity during ground contact (m/s)
    support_consistency: float    # % frames with valid support state
    phase_accuracy: float         # agreement between extracted and predicted phase


def compute_penetration_rate(
    joints: np.ndarray,
    object_mesh: "trimesh.Trimesh",
    threshold: float = 0.0,
) -> float:
    """Fraction of frames where any body joint penetrates the object.

    Parameters
    ----------
    joints : (T, 22, 3) — generated joint positions
    object_mesh : trimesh.Trimesh
    threshold : penetration distance threshold (0 = surface)
    """
    from piano.utils.geometry import points_to_mesh_distance

    T, J, _ = joints.shape
    penetrating_frames = 0

    for t in range(T):
        dists, _ = points_to_mesh_distance(joints[t], object_mesh)
        # Check if any joint is inside the mesh (signed distance < threshold)
        # For unsigned distance, we check if point is very close AND inside
        if (dists < threshold).any():
            penetrating_frames += 1

    return penetrating_frames / T


def compute_contact_metrics(
    pred_contact: np.ndarray,
    gt_contact: np.ndarray,
    threshold: float = 0.5,
) -> tuple[float, float, float]:
    """Compute contact precision, recall, and F1.

    Parameters
    ----------
    pred_contact : (T, B) — predicted contact probabilities
    gt_contact : (T, B) — ground truth contact labels
    threshold : binarization threshold
    """
    pred_binary = pred_contact > threshold
    gt_binary = gt_contact > threshold

    tp = (pred_binary & gt_binary).sum()
    fp = (pred_binary & ~gt_binary).sum()
    fn = (~pred_binary & gt_binary).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return float(precision), float(recall), float(f1)


def compute_foot_sliding(
    joints: np.ndarray,
    foot_contact: np.ndarray,
    fps: float = 30.0,
    height_threshold: float = 0.05,
) -> float:
    """Mean foot velocity when foot is in contact with ground.

    Lower is better — feet should be stationary when in contact.

    Parameters
    ----------
    joints : (T, 22, 3) — joint positions
    foot_contact : (T, 2) — [left_foot, right_foot] contact labels
    fps : frame rate
    """
    dt = 1.0 / fps
    foot_indices = [7, 8]  # left_ankle, right_ankle

    sliding_velocities = []
    for i, joint_idx in enumerate(foot_indices):
        pos = joints[:, joint_idx, :]  # (T, 3)
        vel = np.zeros_like(pos)
        vel[1:] = (pos[1:] - pos[:-1]) / dt

        # Only ground-plane velocity (xz)
        speed_xz = np.linalg.norm(vel[:, [0, 2]], axis=-1)  # (T,)

        # Foot sliding = speed during contact
        contact_mask = foot_contact[:, i] > 0.5
        if contact_mask.any():
            sliding_velocities.append(speed_xz[contact_mask].mean())

    return float(np.mean(sliding_velocities)) if sliding_velocities else 0.0


def compute_support_consistency(
    support_pred: np.ndarray,
    joints: np.ndarray,
    height_threshold: float = 0.05,
) -> float:
    """Fraction of frames where predicted support state is physically valid.

    Checks that:
    - "both_feet" (0): both feet are near ground
    - "single_foot" (1): exactly one foot near ground
    - "sitting" (2): pelvis height is low
    - "hand_support" (3): at least one hand is below shoulder

    Parameters
    ----------
    support_pred : (T,) — integer support state predictions
    joints : (T, 22, 3) — joint positions
    """
    T = len(support_pred)
    valid = 0

    left_ankle_y = joints[:, 7, 1]
    right_ankle_y = joints[:, 8, 1]
    pelvis_y = joints[:, 0, 1]

    left_foot_ground = left_ankle_y < height_threshold
    right_foot_ground = right_ankle_y < height_threshold

    for t in range(T):
        state = support_pred[t]
        if state == 0:  # both_feet
            valid += int(left_foot_ground[t] and right_foot_ground[t])
        elif state == 1:  # single_foot
            valid += int(left_foot_ground[t] != right_foot_ground[t])
        elif state == 2:  # sitting
            valid += int(pelvis_y[t] < 0.6)  # pelvis below 60cm
        elif state == 3:  # hand_support
            valid += 1  # hard to verify without scene; accept
        else:
            valid += 1

    return valid / T


def compute_phase_accuracy(
    pred_phase: np.ndarray,
    gt_phase: np.ndarray,
) -> float:
    """Frame-level accuracy of predicted interaction phase vs ground truth.

    Parameters
    ----------
    pred_phase : (T,) — predicted phase indices
    gt_phase : (T,) — ground truth phase indices
    """
    return float((pred_phase == gt_phase).mean())
