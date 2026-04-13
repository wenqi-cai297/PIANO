"""Extract support state pseudo-labels from HOI motion data.

Classifies each frame into one of four body support configurations
based on foot and pelvis contact patterns.

Support states:
    0 = both_feet    — both feet on ground
    1 = single_foot  — only one foot on ground
    2 = sitting      — pelvis contacting a surface
    3 = hand_support — hands providing primary support (e.g., leaning)

Output: integer support array of shape ``(T,)``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter


# Support state constants
SUPPORT_BOTH_FEET = 0
SUPPORT_SINGLE_FOOT = 1
SUPPORT_SITTING = 2
SUPPORT_HAND = 3

SUPPORT_NAMES: list[str] = [
    "both_feet",
    "single_foot",
    "sitting",
    "hand_support",
]
NUM_SUPPORT_STATES: int = len(SUPPORT_NAMES)


@dataclass(slots=True)
class SupportConfig:
    """Configuration for support state extraction."""

    contact_threshold: float = 0.5  # binarization threshold for contact scores
    median_filter_size: int = 7     # temporal smoothing window


def extract_support_state(
    contact_state: np.ndarray,
    config: SupportConfig | None = None,
) -> np.ndarray:
    """Extract per-frame support state from contact pseudo-labels.

    Parameters
    ----------
    contact_state : (T, 5) — soft contact for
        [left_hand, right_hand, left_foot, right_foot, pelvis]
    config : extraction parameters

    Returns
    -------
    support : (T,) — integer support state per frame
    """
    if config is None:
        config = SupportConfig()

    T = len(contact_state)
    tau = config.contact_threshold

    # Binarize contacts
    left_hand = contact_state[:, 0] > tau
    right_hand = contact_state[:, 1] > tau
    left_foot = contact_state[:, 2] > tau
    right_foot = contact_state[:, 3] > tau
    pelvis = contact_state[:, 4] > tau

    support = np.full(T, SUPPORT_BOTH_FEET, dtype=np.int64)

    for t in range(T):
        if pelvis[t]:
            support[t] = SUPPORT_SITTING
        elif (left_hand[t] or right_hand[t]) and not (left_foot[t] and right_foot[t]):
            # Hands active, not both feet grounded → hand support
            support[t] = SUPPORT_HAND
        elif left_foot[t] and right_foot[t]:
            support[t] = SUPPORT_BOTH_FEET
        elif left_foot[t] or right_foot[t]:
            support[t] = SUPPORT_SINGLE_FOOT
        else:
            # Airborne or ambiguous — default to both_feet (most common)
            support[t] = SUPPORT_BOTH_FEET

    # Temporal smoothing
    support = median_filter(support, size=config.median_filter_size).astype(np.int64)

    return support
