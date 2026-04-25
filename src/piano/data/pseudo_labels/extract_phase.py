"""Extract interaction phase pseudo-labels from HOI motion data.

Assigns each frame a coarse interaction phase based on heuristic rules
derived from any-body-part contact and object motion (translation +
rotation).

Phases (3-class as of v5 / 2026-04-25 redesign):
    0 = non_contact     — body parts not in contact with object
                          (covers approach, transition, release —
                           spatially indistinguishable, temporal order
                           is implicit in text)
    1 = stable_contact  — in contact, object stationary
    2 = manipulation    — in contact, object moving (translation OR rotation)

Why 3 classes (and not the 5 used through v4):

* Approach + pre_contact + release all label "person not in contact"
  configurations. The spatial signal is identical across them; only
  temporal ordering differs (before / between / after contact). That
  ordering is already implicit in the text prompt, so a phase head
  encoding it is redundant with the generator's text conditioning.
* `pre_contact` (5-class id 1) was 0.51% of frames, structurally narrow
  by the `near_threshold=0.1m AND not in contact` definition — a
  labelling artifact, not a semantic class. v4 with Logit Adjustment
  (Menon ICLR'21) catastrophically over-corrected on this class:
  predicted pre_contact for ~95% of frames at inference because raw
  logits couldn't be calibrated down to the 0.5% prior on a small
  dataset.
* All HOI generation papers in the field (CG-HOI CVPR'24, HOI-Diff,
  Text2HOI CVPR'24, Move-as-You-Say CVPR'24, ContactGen ICCV'23) use
  binary contact + spatial contact map only — no multi-class phase head.
  The 5-class scheme came from analysis datasets like GRAB ECCV'20
  where humans annotate phases for behavioural analysis, not generation.

The 3-class scheme keeps the genuinely-useful temporal signal
(`stable_contact` vs `manipulation`, i.e. "is the object moving while
in contact"), which is information neither contact_state nor text
reliably carries.

Two design choices worth calling out:

    * ``is_contact`` uses the max over all tracked body parts, not just
      hands. Chair sitting sequences have pelvis contact with hands idle;
      an earlier hand-only definition kept every sitting frame stuck in
      ``non_contact``.

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


# Phase label constants (3-class)
PHASE_NON_CONTACT = 0
PHASE_STABLE_CONTACT = 1
PHASE_MANIPULATION = 2

PHASE_NAMES: list[str] = [
    "non_contact",
    "stable_contact",
    "manipulation",
]
NUM_PHASES: int = len(PHASE_NAMES)


@dataclass(slots=True)
class PhaseConfig:
    """Configuration for interaction phase extraction."""

    translational_velocity_eps: float = 0.02  # m/s
    rotational_velocity_eps: float = 0.3      # rad/s (~17 deg/s)
    contact_threshold: float = 0.5         # any body part contact score threshold
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
    phase : (T,) — integer phase label per frame, values in {0, 1, 2}
    """
    if config is None:
        config = PhaseConfig()

    T = len(joints)
    phase = np.full(T, PHASE_NON_CONTACT, dtype=np.int64)

    # Any-body-part contact — drives stable/manipulation branches.
    # Hand-only was wrong for sitting: pelvis contacts chair but hands idle,
    # so the whole sitting stretch was mislabeled as non_contact.
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

    # --- Heuristic state machine (3-class) ---
    # Non-contact frames stay at PHASE_NON_CONTACT (the array's init value),
    # so we only need to assign during the in-contact branch.
    for t in range(T):
        if is_contact[t]:
            phase[t] = PHASE_MANIPULATION if is_moving[t] else PHASE_STABLE_CONTACT

    # --- Temporal smoothing ---
    phase = median_filter(phase, size=config.median_filter_size).astype(np.int64)

    return phase
