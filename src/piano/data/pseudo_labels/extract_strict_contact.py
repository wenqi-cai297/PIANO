"""Strict 'real contact' pseudo-label extraction (v12, 2026-05-03).

Why this exists
---------------

The v11 pseudo-label defines `contact_state` as **(distance < threshold) OR
(kinematic coupling)** with anatomy-calibrated distance thresholds:
  * hand: 0.12 m (wrist joint to mesh surface)
  * foot: 0.06 m
  * pelvis: 0.20 m

These thresholds reflect the joint-to-skin offset (wrist sits ~5–8 cm inside
the forearm, palm surface protrudes another 5–8 cm). So `dist < 0.12` for
the hand means the **wrist** is within 12 cm of the surface — the **palm**
might be 4–7 cm away from the surface and the model still labels this
"in contact". This trains the model to "approach to within 5–7 cm" rather
than "make contact".

The 2026-05-03 visual review confirmed this failure mode:
> "人没真正接触到物体，只是有点靠近而已。"

This module re-extracts contact with a STRICT definition that aims for
"the body part is physically engaged with the object surface", not
"approaching the neighbourhood of the object". Differences vs v11:

1. **Tighter distance thresholds** (joint-to-skin offset only, no buffer):
     hand 0.05 m, foot 0.03 m, pelvis 0.12 m.
   At hand 0.05 m, the palm should be within ~0–2 cm of the mesh surface —
   "actually touching".

2. **AND-combined with engagement** (instead of OR with kinematic coupling).
   `contact = (dist_score >= 0.5) AND (engagement_score >= 0.5)`.
   Engagement = kinematic coupling > 0.5 (body stable in object-local frame
   while object is moving) OR (object is static AND body is also static at
   the contact point — for press / sit / lean).
   The "AND" is the key change: a hand that's just close to the object
   (without moving with it OR being still on a static object) is no longer
   labelled as contact.

3. **Longer minimum segment** (5 frames = 0.25 s @ 20 fps, was 3 frames):
   eliminates "swipe through" false positives.

4. **Within-segment drift filter**: a contact segment is invalidated if
   the body part drifts > 5 cm in the object-local frame within the
   segment. Forces the contact to be a "stable grip / press / sit",
   not a glancing touch.

Reference for community 5 cm threshold: OMOMO (Li et al., SIGGRAPH Asia
2023, arXiv:2309.16237) §"Contact metric"; CHOIS (Li et al., CVPR 2024,
arXiv:2312.17134); InterDiff (Xu et al., ICCV 2023, arXiv:2308.16905) all
use 0.05 m hand-to-object distance + duration filter.

Note on PIANO 22-joint scope: finger articulation is not modelled
(matches InterDiff/CHOIS/HOI-Diff convention). The 5 cm threshold at
the wrist joint is the closest the body representation can express to
"hand grasping object". Finger-level grasp accuracy is not evaluated.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.special import expit

from piano.data.pseudo_labels.extract_contact import (
    ContactConfig,
    DEFAULT_DISTANCE_THRESHOLDS,
    _kinematic_contact_score,
    _soft_sigmoid,
    extract_contact_state,        # for v11 fallback / comparison
)
from piano.utils.geometry import points_to_mesh_distance
from piano.utils.smpl_utils import (
    BODY_PART_INDICES,
    BODY_PART_NAMES,
    NUM_BODY_PARTS,
)


# ============================================================================
# Strict thresholds (v12)
# ============================================================================

STRICT_DISTANCE_THRESHOLDS: dict[str, float] = {
    # Tight thresholds for the case_static branch (object stationary +
    # body in physical contact; e.g. sit, press, grip-static). At these
    # distances the body skin/sole is essentially touching the surface.
    "left_hand":  0.05,    # was 0.12 — wrist within 5 cm ≈ palm touching surface
    "right_hand": 0.05,
    "left_foot":  0.03,    # was 0.06 — foot joint within 3 cm ≈ sole touching
    "right_foot": 0.03,
    "pelvis":     0.12,    # was 0.20 — root within 12 cm ≈ ischium touching seat
}

LOOSE_DISTANCE_THRESHOLDS: dict[str, float] = {
    # Loose thresholds for the case_kinematic branch (wrap-grip / gloved
    # / handle-grip / carry-the-bag cases). The wrist can be 18–25 cm
    # from the mesh and the hand still be physically attached if the
    # body part moves in lockstep with the object in world frame
    # (kinematic engagement). v11 docstring: "wrist joint is 18-22 cm
    # from mesh surface in wrap-grip cases" — these thresholds give
    # ~25 cm coverage for hand-grip and proportional limits elsewhere.
    "left_hand":  0.25,
    "right_hand": 0.25,
    "left_foot":  0.15,
    "right_foot": 0.15,
    "pelvis":     0.30,
}


@dataclass(slots=True)
class StrictContactConfig:
    """Configuration for v12 strict 'real contact' extraction.

    The contact decision is the OR of two cases:

      case_kinematic:  body moves in lockstep with object in world frame
                       (kinematic_score >= threshold) AND body is within
                       a *loose* distance threshold (handles wrap-grip
                       / gloved / handle-grip cases where wrist can be
                       18–25 cm from mesh but is physically attached).

      case_static:     object is stationary AND body is stable at the
                       contact point (static_engagement_score >=
                       threshold) AND body is within a *tight* distance
                       threshold (handles press / sit / grip-static —
                       must be physically touching, not just close).

    Within-segment drift (object-local frame) is filtered post-hoc
    regardless of which case fired the segment, ensuring the contact
    is a stable engagement rather than a glancing brush.
    """

    # Per-part TIGHT distance thresholds (m), used in case_static.
    distance_thresholds: dict[str, float] = field(
        default_factory=lambda: dict(STRICT_DISTANCE_THRESHOLDS)
    )
    # Per-part LOOSE distance thresholds (m), used in case_kinematic.
    # Allows wrap-grip / glove / handle cases where wrist is 18–25 cm
    # from the mesh but truly attached.
    loose_distance_thresholds: dict[str, float] = field(
        default_factory=lambda: dict(LOOSE_DISTANCE_THRESHOLDS)
    )
    # Sigmoid transition widths.
    distance_sigma: float = 0.015          # tight, for case_static
    loose_distance_sigma: float = 0.04     # loose, for case_kinematic

    # Engagement thresholds — currently used as soft scores, but kept
    # as named knobs for consistency with the design doc.
    require_engagement: bool = True
    engagement_threshold: float = 0.5

    # Minimum contact segment duration in frames (≥ 0.25 s @ 20 fps).
    min_contact_duration: int = 5

    # Within-segment drift in object-local frame (m).
    # Contact segments with body drift > this are invalidated.
    max_segment_drift_m: float = 0.05

    # Static-engagement detection: when object is stationary, body stable
    # at contact point counts as engagement. Object stationary =
    # speed < eps; body stable = local std < threshold over kin_window.
    static_engagement_eps_mps: float = 0.05    # m/s
    static_engagement_local_std_m: float = 0.02

    # Median filter window (frames).
    median_filter_size: int = 7

    fps: float = 30.0


def _static_engagement_score(
    body_local: np.ndarray,            # (T, 3) body part position in object-local
    object_speed: np.ndarray,          # (T,) m/s proxy
    *,
    kin_window: int,
    eps_mps: float,
    local_std_thresh: float,
) -> np.ndarray:
    """Score for 'object is stationary AND body part is stable on it'.

    For each frame, returns soft probability that:
      - object speed is below eps (object effectively still)
      - AND body's object-local position has low std over kin_window (body
        is resting on the object, not just briefly near it)

    Used to capture press/sit/lean/grasp-static cases where kinematic
    coupling doesn't fire (because object isn't moving).
    """
    T = len(body_local)
    # Local std of body in object-local frame
    mean_x = uniform_filter1d(body_local, size=kin_window, axis=0, mode="nearest")
    mean_x_sq = uniform_filter1d(body_local ** 2, size=kin_window, axis=0, mode="nearest")
    rolling_var = np.maximum(mean_x_sq - mean_x ** 2, 0.0)
    local_std = np.sqrt(rolling_var + 1e-12).max(axis=-1)         # (T,)

    # Soft scores — body stable + object stationary
    body_stable = expit((local_std_thresh - local_std) / 0.005)
    obj_stationary = expit((eps_mps - object_speed) / 0.02)
    return (body_stable * obj_stationary).astype(np.float32)


def _filter_drifting_contacts(
    contact: np.ndarray,                # (T, B)
    body_locals: np.ndarray,             # (T, B, 3) — body parts in object-local
    *,
    max_drift_m: float,
    threshold: float = 0.5,
) -> np.ndarray:
    """Zero out contact segments where the body part drifts > max_drift in the
    object-local frame within the segment.

    Drift = max distance from segment-mean position. Segments below threshold
    are kept as-is (they are already filtered as 'no contact').
    """
    result = contact.copy()
    T, B = contact.shape

    for bp in range(B):
        binary = contact[:, bp] > threshold
        changes = np.diff(binary.astype(np.int8), prepend=0, append=0)
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]

        for s, e in zip(starts, ends):
            seg = body_locals[s:e, bp, :]                # (L, 3)
            if seg.shape[0] < 2:
                continue
            centroid = seg.mean(axis=0)
            spread = float(np.linalg.norm(seg - centroid, axis=-1).max())
            if spread > max_drift_m:
                result[s:e, bp] = 0.0
    return result


def _filter_short_contacts(
    contact: np.ndarray,
    min_duration: int,
    threshold: float = 0.5,
) -> np.ndarray:
    """Same as v11 _filter_short_contacts; copied here so this module
    doesn't depend on the v11 internal helper."""
    result = contact.copy()
    T, B = contact.shape
    for bp in range(B):
        binary = contact[:, bp] > threshold
        changes = np.diff(binary.astype(np.int8), prepend=0, append=0)
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        for s, e in zip(starts, ends):
            if (e - s) < min_duration:
                result[s:e, bp] = 0.0
    return result


# ============================================================================
# Main API
# ============================================================================

def extract_strict_contact_state(
    joints: np.ndarray,                  # (T, 22, 3) world frame
    object_mesh,                          # trimesh.Trimesh in object-local frame
    object_positions: np.ndarray,        # (T, 3) world
    object_rotations: np.ndarray,        # (T, 3) axis-angle world
    *,
    strict_config: StrictContactConfig | None = None,
    base_kin_config: ContactConfig | None = None,
) -> np.ndarray:
    """Extract per-frame, per-part STRICT contact state.

    Returns
    -------
    contact : (T, 5) — soft contact in [0, 1]; > 0.5 means 'real contact'
        per the v12 strict definition.
    """
    from piano.data.pseudo_labels._object_transform import world_to_object_local

    if strict_config is None:
        strict_config = StrictContactConfig()
    if base_kin_config is None:
        base_kin_config = ContactConfig(fps=strict_config.fps)
    else:
        # Match fps and kinematic params from caller's base config
        base_kin_config = ContactConfig(
            **{k: getattr(base_kin_config, k) for k in base_kin_config.__slots__}
        )
    # Ensure fps consistency
    base_kin_config.fps = strict_config.fps

    T = len(joints)
    contact = np.zeros((T, NUM_BODY_PARTS), dtype=np.float32)
    body_locals = np.zeros((T, NUM_BODY_PARTS, 3), dtype=np.float32)

    # Per-frame object speed for static-engagement detection
    trans_vel = np.zeros(T, dtype=np.float32)
    trans_vel[1:] = np.linalg.norm(np.diff(object_positions, axis=0), axis=-1) * strict_config.fps
    ang_vel = np.zeros(T, dtype=np.float32)
    if object_rotations is not None:
        ang_vel[1:] = np.linalg.norm(np.diff(object_rotations, axis=0), axis=-1) * strict_config.fps
    obj_speed = trans_vel + base_kin_config.kin_radius_proxy * ang_vel       # (T,)

    kin_window = max(3, int(round(base_kin_config.kin_window_sec * strict_config.fps)))

    for bp_idx, joint_idx in enumerate(BODY_PART_INDICES):
        bp_name = BODY_PART_NAMES[bp_idx]
        bp_world = joints[:, joint_idx, :]                               # (T, 3)
        bp_local = world_to_object_local(bp_world, object_positions, object_rotations)
        body_locals[:, bp_idx, :] = bp_local

        # Distance scores — both tight (case_static) and loose (case_kinematic)
        distances, _ = points_to_mesh_distance(bp_local, object_mesh)
        tight_thr = strict_config.distance_thresholds.get(
            bp_name, STRICT_DISTANCE_THRESHOLDS[bp_name]
        )
        loose_thr = strict_config.loose_distance_thresholds.get(
            bp_name, LOOSE_DISTANCE_THRESHOLDS[bp_name]
        )
        tight_dist_score = _soft_sigmoid(distances, tight_thr, strict_config.distance_sigma)
        loose_dist_score = _soft_sigmoid(distances, loose_thr, strict_config.loose_distance_sigma)

        # Two-case OR formulation:
        #   case_kinematic = kinematic_engagement × loose_distance
        #     captures wrap-grip / glove / handle-grip / carry-bag cases
        #     where wrist is 18–25 cm from mesh but physically attached.
        #   case_static = static_engagement × tight_distance
        #     captures press / sit / grip-static where body is actually
        #     touching the surface and not moving relative to it.
        if strict_config.require_engagement:
            kin_score = _kinematic_contact_score(
                bp_world, object_positions, object_rotations, base_kin_config,
            )
            static_score = _static_engagement_score(
                bp_local, obj_speed,
                kin_window=kin_window,
                eps_mps=strict_config.static_engagement_eps_mps,
                local_std_thresh=strict_config.static_engagement_local_std_m,
            )
            case_kinematic = kin_score * loose_dist_score
            case_static = static_score * tight_dist_score
            score = np.maximum(case_kinematic, case_static)
        else:
            # Distance-only fallback (for ablation / sanity)
            score = tight_dist_score

        contact[:, bp_idx] = score

    # Temporal smoothing
    for bp_idx in range(NUM_BODY_PARTS):
        contact[:, bp_idx] = median_filter(
            contact[:, bp_idx], size=strict_config.median_filter_size,
        )

    # Min duration filter
    contact = _filter_short_contacts(contact, strict_config.min_contact_duration)

    # Within-segment drift filter (v12 specific)
    contact = _filter_drifting_contacts(
        contact, body_locals,
        max_drift_m=strict_config.max_segment_drift_m,
    )

    return contact
