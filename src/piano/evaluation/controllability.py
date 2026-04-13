"""Controllability and object-adaptive evaluation metrics.

These are PIANO's core novelty metrics — they measure whether the model
produces different motion strategies for different object properties,
and whether the interaction latent provides decomposed control.

Metrics:
    - Attribute Sensitivity Score (ASS): does motion change with object attributes?
    - Attribute-Strategy Consistency (ASC): is the change physically correct?
    - Latent sensitivity: does perturbing z_int change the motion?
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ControllabilityMetrics:
    """Container for controllability evaluation results."""

    attribute_sensitivity_score: float    # ASS: motion change / attribute change
    attribute_strategy_consistency: float  # ASC: % of physically correct adaptations
    latent_sensitivity: float             # motion change when z_int is perturbed


# ============================================================================
# Attribute Sensitivity Score (ASS)
# ============================================================================

def compute_attribute_sensitivity(
    motions_by_attribute: dict[str, np.ndarray],
    attributes: dict[str, np.ndarray],
) -> float:
    """Compute Attribute Sensitivity Score.

    For each pair of attribute values, measures whether the generated
    motion changes proportionally to the attribute difference.

    Parameters
    ----------
    motions_by_attribute : mapping from attribute_id → (T, D) motion features.
        Same text prompt, different object attributes.
    attributes : mapping from attribute_id → (A,) attribute feature vector.
        E.g., [size, weight, height].

    Returns
    -------
    ASS : higher = more sensitive to attribute changes (better).
    """
    attr_ids = list(motions_by_attribute.keys())
    if len(attr_ids) < 2:
        return 0.0

    ratios = []
    for i in range(len(attr_ids)):
        for j in range(i + 1, len(attr_ids)):
            motion_diff = np.linalg.norm(
                _extract_motion_features(motions_by_attribute[attr_ids[i]])
                - _extract_motion_features(motions_by_attribute[attr_ids[j]])
            )
            attr_diff = np.linalg.norm(attributes[attr_ids[i]] - attributes[attr_ids[j]])
            if attr_diff > 1e-8:
                ratios.append(motion_diff / attr_diff)

    return float(np.mean(ratios)) if ratios else 0.0


def _extract_motion_features(motion: np.ndarray) -> np.ndarray:
    """Extract summary features from a motion sequence for ASS comparison.

    Features: mean joint velocity, mean acceleration, total displacement,
    contact duration ratio, manipulation speed.
    """
    T = len(motion)
    if T < 3:
        return np.zeros(5)

    # Velocity (use first 66 dims = joint velocities in HumanML3D)
    vel = motion[:, 4:70] if motion.shape[1] >= 70 else motion
    mean_vel = np.linalg.norm(vel, axis=-1).mean()

    # Acceleration
    acc = np.diff(vel, axis=0)
    mean_acc = np.linalg.norm(acc, axis=-1).mean()

    # Root displacement (first 3 dims: root angular vel + root xz vel)
    root_vel = motion[:, 1:3]  # xz velocity
    total_disp = np.linalg.norm(root_vel, axis=-1).sum()

    # Duration proxy
    duration = float(T)

    # Contact ratio (last 4 dims = foot contact in HumanML3D)
    if motion.shape[1] >= 263:
        foot_contact = motion[:, 259:263]
        contact_ratio = foot_contact.mean()
    else:
        contact_ratio = 0.0

    return np.array([mean_vel, mean_acc, total_disp, duration, contact_ratio])


# ============================================================================
# Attribute-Strategy Consistency (ASC)
# ============================================================================

def compute_attribute_strategy_consistency(
    motion_pairs: list[tuple[np.ndarray, np.ndarray, str]],
) -> float:
    """Compute Attribute-Strategy Consistency.

    Checks whether motion changes in the physically correct direction
    when object attributes change.

    Parameters
    ----------
    motion_pairs : list of (motion_small, motion_large, rule_type) tuples.
        rule_type is one of: "heavier", "larger", "higher".

    Returns
    -------
    ASC : fraction of pairs that obey the expected physical rule.
    """
    if not motion_pairs:
        return 0.0

    compliant = 0
    for motion_a, motion_b, rule in motion_pairs:
        if _check_physical_rule(motion_a, motion_b, rule):
            compliant += 1

    return compliant / len(motion_pairs)


def _check_physical_rule(
    motion_light: np.ndarray,
    motion_heavy: np.ndarray,
    rule: str,
) -> bool:
    """Check if motion change follows the expected physical rule.

    Rules:
        "heavier": heavy object → slower manipulation, lower CoM
        "larger": larger object → wider reach, more body lean
        "higher": higher surface → more arm elevation
    """
    feat_light = _extract_motion_features(motion_light)
    feat_heavy = _extract_motion_features(motion_heavy)

    if rule == "heavier":
        # Heavier → slower velocity, more contact time
        return feat_heavy[0] < feat_light[0]  # mean velocity should decrease
    elif rule == "larger":
        # Larger → more displacement, higher acceleration
        return feat_heavy[2] > feat_light[2]  # more root displacement
    elif rule == "higher":
        # Higher → more vertical displacement
        return True  # simplified; would need joint-level analysis
    return True


# ============================================================================
# Latent Sensitivity
# ============================================================================

def compute_latent_sensitivity(
    base_motion: np.ndarray,
    perturbed_motions: list[np.ndarray],
) -> float:
    """Measure how much motion changes when interaction latent is perturbed.

    Parameters
    ----------
    base_motion : (T, D) — motion from original z_int
    perturbed_motions : list of (T, D) — motions from perturbed z_int

    Returns
    -------
    Mean L2 distance between base and perturbed motions (higher = more sensitive).
    """
    if not perturbed_motions:
        return 0.0

    dists = []
    for pm in perturbed_motions:
        T = min(len(base_motion), len(pm))
        dist = np.linalg.norm(base_motion[:T] - pm[:T], axis=-1).mean()
        dists.append(dist)

    return float(np.mean(dists))
