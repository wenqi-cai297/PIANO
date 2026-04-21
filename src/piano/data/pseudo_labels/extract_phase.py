"""Extract interaction phase pseudo-labels from HOI motion data.

Assigns each frame a coarse interaction phase based on heuristic rules
derived from hand-object distance, any-body-part contact, and object
motion (translation + rotation).

Phases:
    0 = approach       — moving toward object, no contact
    1 = pre-contact    — close to object, about to make contact
    2 = stable-contact — in contact, object stationary
    3 = manipulation   — in contact, object moving (translation OR rotation)
    4 = release        — contact just ended or ending

Two design choices worth calling out:

    * ``is_contact`` uses the max over all tracked body parts, not just
      hands. Chair sitting sequences have pelvis contact with hands idle;
      an earlier hand-only definition kept every sitting frame stuck in
      ``approach``.

    * Object motion combines translational and rotational velocity. An
      in-place bat swing or rotating a chair leaves ``object_positions``
      roughly static — rotation must also be observed to reach
      ``manipulation``.

The heuristic assignment is optionally refined using an HMM
(see ``refine_phase_hmm.py``).

Output: integer phase array of shape ``(T,)``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter

from piano.utils.smpl_utils import BODY_PART_INDICES


# Phase label constants
PHASE_APPROACH = 0
PHASE_PRE_CONTACT = 1
PHASE_STABLE_CONTACT = 2
PHASE_MANIPULATION = 3
PHASE_RELEASE = 4

PHASE_NAMES: list[str] = [
    "approach",
    "pre-contact",
    "stable-contact",
    "manipulation",
    "release",
]
NUM_PHASES: int = len(PHASE_NAMES)


@dataclass(slots=True)
class PhaseConfig:
    """Configuration for interaction phase extraction."""

    far_threshold: float = 0.5             # meters — beyond this is "approach"
    near_threshold: float = 0.1            # meters — within this is "pre-contact"
    translational_velocity_eps: float = 0.02  # m/s
    rotational_velocity_eps: float = 0.3      # rad/s (~17 deg/s)
    contact_threshold: float = 0.5         # any body part contact score threshold
    release_window: int = 10               # frames after contact loss to label "release"
    median_filter_size: int = 7            # temporal smoothing window
    fps: float = 30.0


def extract_interaction_phase(
    joints: np.ndarray,
    contact_state: np.ndarray,
    object_positions: np.ndarray | None = None,
    object_rotations: np.ndarray | None = None,
    config: PhaseConfig | None = None,
) -> np.ndarray:
    """Extract per-frame interaction phase labels.

    Parameters
    ----------
    joints : (T, 22, 3) — SMPL 22-joint positions
    contact_state : (T, 5) — soft contact state from ``extract_contact``
    object_positions : (T, 3) or None — object center position per frame.
        If None, object is assumed static at the mean closest-hand position.
    object_rotations : (T, 3) or None — per-frame axis-angle rotation. When
        available, its finite difference contributes to the stable/manipulation
        decision so that rotation-only motions (bat swing, chair rotate) don't
        collapse to stable-contact.
    config : extraction parameters

    Returns
    -------
    phase : (T,) — integer phase label per frame
    """
    if config is None:
        config = PhaseConfig()

    T = len(joints)
    phase = np.full(T, PHASE_APPROACH, dtype=np.int64)

    # Hand joint positions — used only for the approach/pre-contact hand-
    # leading proxy. The contact decision itself comes from all tracked parts.
    left_hand = joints[:, BODY_PART_INDICES[0], :]   # (T, 3)
    right_hand = joints[:, BODY_PART_INDICES[1], :]   # (T, 3)

    # Any-body-part contact — drives stable/manipulation branches.
    # Hand-only was wrong for sitting: pelvis contacts chair but hands idle,
    # so the whole sitting stretch was mislabeled as approach.
    any_contact_score = contact_state.max(axis=-1)                     # (T,)
    is_contact = any_contact_score > config.contact_threshold           # (T,)

    # Fallback object-position estimation. Prefer the closest body-part
    # position during contact (hand or otherwise) — generalises the
    # previous hand-only guess.
    if object_positions is None:
        if is_contact.any():
            # Use pelvis position during contact as a coarse anchor; any
            # tracked part is fine since this branch is a fallback only.
            pelvis_contact_frames = contact_state[:, 4] > config.contact_threshold
            anchor_frames = pelvis_contact_frames if pelvis_contact_frames.any() else is_contact
            object_positions = np.tile(
                joints[anchor_frames, 0, :].mean(axis=0), (T, 1),
            )
        else:
            return phase
    elif object_positions.ndim == 1:
        object_positions = np.tile(object_positions, (T, 1))

    # Distance from each hand to object center (still hand-led because the
    # approach/pre-contact distinction tracks the reaching hand).
    dist_left = np.linalg.norm(left_hand - object_positions, axis=-1)
    dist_right = np.linalg.norm(right_hand - object_positions, axis=-1)
    hand_obj_dist = np.minimum(dist_left, dist_right)  # (T,)

    # Object translational velocity.
    trans_vel = np.zeros(T)
    if T > 1:
        trans_vel[1:] = np.linalg.norm(
            np.diff(object_positions, axis=0), axis=-1,
        ) * config.fps

    # Object angular velocity via axis-angle finite difference. The
    # axis-angle diff overestimates for large rotations (it ignores the
    # group structure) but is accurate enough at per-frame Δt (20 fps →
    # Δangle typically << 0.5 rad) to serve as a manipulation cue.
    ang_vel = np.zeros(T)
    if object_rotations is not None and T > 1:
        ang_vel[1:] = np.linalg.norm(
            np.diff(object_rotations, axis=0), axis=-1,
        ) * config.fps

    is_moving = (
        (trans_vel > config.translational_velocity_eps)
        | (ang_vel > config.rotational_velocity_eps)
    )

    # --- Heuristic state machine ---
    for t in range(T):
        if is_contact[t]:
            phase[t] = PHASE_MANIPULATION if is_moving[t] else PHASE_STABLE_CONTACT
        elif hand_obj_dist[t] < config.near_threshold:
            phase[t] = PHASE_PRE_CONTACT
        else:
            phase[t] = PHASE_APPROACH

    # --- Mark release: frames right after contact ends ---
    _mark_release(phase, is_contact, config.release_window)

    # --- Temporal smoothing ---
    phase = median_filter(phase, size=config.median_filter_size).astype(np.int64)

    return phase


def _mark_release(
    phase: np.ndarray,
    is_contact: np.ndarray,
    release_window: int,
) -> None:
    """In-place: mark frames as RELEASE for *release_window* frames
    after each contact→no-contact transition."""
    T = len(phase)
    contact_diff = np.diff(is_contact.astype(np.int8), prepend=0)
    release_starts = np.where(contact_diff == -1)[0]  # contact just ended

    for start in release_starts:
        end = min(start + release_window, T)
        phase[start:end] = PHASE_RELEASE
