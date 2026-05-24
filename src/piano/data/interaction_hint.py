"""Per-frame oracle interaction hint for Tier-0A diagnostic (Round-27+).

Builds a compact ``(T, D)`` per-frame hint that summarises the GT-derived
hand-object relation and foot-ground support state.

Used as a direct conditioning channel by ``MotionAnchorDiff`` in Tier-0
oracle-hint overfit experiments. The goal is to test whether Stage-2 can
consume an explicit interaction-state condition at all, before committing
to a Stage-1.5 interaction planner.

Variants
--------

- ``"hand"`` (D=8):
    [:2]   hand_contact_mask                (L, R)
    [2:8]  hand_object_local_offset         (L_xyz, R_xyz) in object frame

- ``"foot"`` (D=5):
    [:2]   foot_stance_probability          (L, R)
    [2:4]  ankle_height_norm                (L, R)
    [4:5]  walking_mask

- ``"full"`` (D=13): concatenation of hand (8) and foot (5).

Conventions
-----------

- SMPL-22 joint indices: pelvis=0, L_ankle=7, R_ankle=8, L_wrist=20,
  R_wrist=21 (from ``piano.utils.smpl_utils``).
- ``contact_state`` ordering: ``[L_hand, R_hand, L_foot, R_foot, pelvis]``
  (from ``piano.data.dataset._BODY_PART_LR_PAIRS`` doc).
- Up axis = Y (joint y-coordinate is height; see
  ``smpl_utils.estimate_foot_contact``).
- ``object_rotations`` is axis-angle ``(T, 3)`` in world frame
  (see ``dataset._compute_canonical_object_pose`` docstring and
  ``contact_guidance.py`` usage).
- Foot stance is derived from GT ankle (joint 7/8) — NOT from the
  InterAct foot-object pseudo-label, which is tied to mid-foot / knee
  markers and is not reliable for foot-ground support (roadmap §6.6,
  §16-3).

References
----------
``piano_stage2_full_architecture_roadmap.md`` §6 (oracle interaction
hint) and §16 (coding constraints). Reviewer-produced 2026-05-25.
"""
from __future__ import annotations

import numpy as np

from piano.utils.canonical_frame import axis_angle_to_matrix_np


# SMPL-22 joint indices used by this module. Verified against
# ``piano.utils.smpl_utils.SMPL_22_JOINT_NAMES``.
LEFT_WRIST_IDX: int = 20
RIGHT_WRIST_IDX: int = 21
LEFT_ANKLE_IDX: int = 7
RIGHT_ANKLE_IDX: int = 8
ROOT_IDX: int = 0

# ``contact_state`` column indices (``piano.utils.smpl_utils
# .INTERACTION_BODY_PARTS`` ordering, surfaced in
# ``dataset._BODY_PART_LR_PAIRS`` comment).
CONTACT_LEFT_HAND_COL: int = 0
CONTACT_RIGHT_HAND_COL: int = 1

# Hint dimensionalities.
HINT_DIM_HAND: int = 8
HINT_DIM_FOOT: int = 5
HINT_DIM_FULL: int = HINT_DIM_HAND + HINT_DIM_FOOT  # 13

# Default project-wide FPS (matches every other ``fps`` default in the
# repo — see ``contact_eval.py`` / ``visualize_motion.py`` etc.).
DEFAULT_FPS: float = 20.0


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def derive_walking_mask_from_gt(
    joints_22: np.ndarray,
    fps: float = DEFAULT_FPS,
    root_idx: int = ROOT_IDX,
    speed_threshold_mps: float = 0.10,
) -> np.ndarray:
    """Boolean per-frame walking mask from GT root XZ speed.

    A frame is "walking" when the root horizontal speed exceeds
    ``speed_threshold_mps``. Default 0.10 m/s ≈ 0.5 cm/frame at 20 fps
    (roadmap §6.8).

    Parameters
    ----------
    joints_22 : (T, 22, 3) float, world-frame joint positions in metres.
    fps : float, frame rate of ``joints_22``.
    root_idx : int, SMPL-22 root joint index.
    speed_threshold_mps : float, walking threshold in m/s.

    Returns
    -------
    walking_mask : (T, 1) float32 in {0, 1}.
    """
    T = int(joints_22.shape[0])
    root_xz = joints_22[:, root_idx, [0, 2]].astype(np.float32)        # (T, 2)
    diff = np.zeros_like(root_xz)
    diff[1:] = root_xz[1:] - root_xz[:-1]
    speed = np.linalg.norm(diff, axis=-1) * float(fps)                  # (T,)
    walking = (speed > float(speed_threshold_mps)).astype(np.float32)
    return walking.reshape(T, 1)


def derive_foot_stance_from_gt(
    joints_22: np.ndarray,
    fps: float = DEFAULT_FPS,
    left_ankle_idx: int = LEFT_ANKLE_IDX,
    right_ankle_idx: int = RIGHT_ANKLE_IDX,
    ankle_height_clamp_m: float = 0.5,
    floor_quantile: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Soft foot-ground stance + normalised ankle height from GT.

    Stance is derived from GT ankle height and horizontal velocity
    (roadmap §6.6). Each ankle gets a soft stance score in [0, 1]:

        floor_y = quantile(all_ankle_y, 0.05)
        height = ankle_y - floor_y
        height_score = sigmoid((0.10 - height) / 0.03)
        vel_score    = sigmoid((0.25 - ankle_xz_speed_mps) / 0.08)
        stance = height_score * vel_score

    The 5% quantile gives a sample-specific floor estimate that is
    robust to outliers (a single stomp / kick) and does not assume the
    global y=0 plane. Both ankles share the same floor (the body has one
    centre of support).

    Parameters
    ----------
    joints_22 : (T, 22, 3) float, world-frame joint positions in metres.
    fps : float, frame rate.
    left_ankle_idx, right_ankle_idx : int, SMPL-22 ankle joint indices.
    ankle_height_clamp_m : float, normaliser for ``ankle_height_norm``.
    floor_quantile : float in (0, 1), quantile used for floor estimate.

    Returns
    -------
    foot_stance : (T, 2) float32 in [0, 1] — left, right.
    ankle_height_norm : (T, 2) float32 in [0, 1] — clamped ankle height.
    """
    T = int(joints_22.shape[0])
    l_ankle = joints_22[:, left_ankle_idx, :].astype(np.float32)        # (T, 3)
    r_ankle = joints_22[:, right_ankle_idx, :].astype(np.float32)

    # Sample-specific floor — use both ankles together.
    all_ankle_y = np.concatenate([l_ankle[:, 1], r_ankle[:, 1]], axis=0)
    floor_y = float(np.quantile(all_ankle_y, float(floor_quantile)))

    l_height = np.maximum(l_ankle[:, 1] - floor_y, 0.0)                 # (T,)
    r_height = np.maximum(r_ankle[:, 1] - floor_y, 0.0)

    # XZ velocity magnitude in m/s.
    def _xz_speed(ankle_pos: np.ndarray) -> np.ndarray:
        diff = np.zeros_like(ankle_pos[:, [0, 2]])
        diff[1:] = ankle_pos[1:, [0, 2]] - ankle_pos[:-1, [0, 2]]
        return np.linalg.norm(diff, axis=-1) * float(fps)               # (T,)

    l_speed = _xz_speed(l_ankle)
    r_speed = _xz_speed(r_ankle)

    def _sigmoid(x: np.ndarray) -> np.ndarray:
        # Numerically stable sigmoid that avoids overflow in both
        # branches: clip the exponent argument before exp.
        x = np.clip(x, -60.0, 60.0)
        return np.where(
            x >= 0,
            1.0 / (1.0 + np.exp(-x)),
            np.exp(x) / (1.0 + np.exp(x)),
        )

    l_h_score = _sigmoid((0.10 - l_height) / 0.03)
    r_h_score = _sigmoid((0.10 - r_height) / 0.03)
    l_v_score = _sigmoid((0.25 - l_speed) / 0.08)
    r_v_score = _sigmoid((0.25 - r_speed) / 0.08)

    foot_stance = np.stack(
        [l_h_score * l_v_score, r_h_score * r_v_score], axis=-1,
    ).astype(np.float32)                                                # (T, 2)

    clamp = float(ankle_height_clamp_m)
    ankle_height_norm = np.stack(
        [
            np.clip(l_height, 0.0, clamp) / clamp,
            np.clip(r_height, 0.0, clamp) / clamp,
        ],
        axis=-1,
    ).astype(np.float32)                                                # (T, 2)

    return foot_stance, ankle_height_norm


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_oracle_interaction_hint(
    joints_22: np.ndarray,
    object_positions: np.ndarray,
    object_rotations: np.ndarray,
    contact_state: np.ndarray,
    variant: str = "full",
    fps: float = DEFAULT_FPS,
    hand_offset_clamp_m: float = 2.0,
) -> np.ndarray:
    """Build a per-frame oracle interaction hint of shape ``(T, D)``.

    Parameters
    ----------
    joints_22 : (T, 22, 3) world-frame SMPL-22 joints, metres.
    object_positions : (T, 3) world-frame object centre position, metres.
    object_rotations : (T, 3) world-frame object axis-angle rotation.
        (Format verified — see ``dataset._compute_canonical_object_pose``
        and ``contact_guidance.py`` usage.)
    contact_state : (T, 5) pseudo-label contact probability per part,
        ordering ``[L_hand, R_hand, L_foot, R_foot, pelvis]``.
    variant : {"hand", "foot", "full"}.
    fps : frame rate used for stance/walking velocity scoring.
    hand_offset_clamp_m : range used to scale object-local hand offset
        into ``[-1, 1]``.

    Returns
    -------
    hint : (T, D) float32. D is 8 / 5 / 13 for hand / foot / full.

    Notes
    -----
    * Hand offset is computed in **object-local** coordinates:
      ``r = R_obj.T @ (wrist_world - obj_pos_world)`` (roadmap §6.5).
      This avoids the joint-centre vs object-surface offset that broke
      the legacy absolute ``contact_target_xyz`` route (Round-24 metric
      diagnostic).
    * Foot stance is **derived from GT ankle**, not InterAct foot-object
      pseudo-label (roadmap §6.6, §16-3).
    """
    T = int(joints_22.shape[0])

    if variant not in {"hand", "foot", "full"}:
        raise ValueError(
            f"variant must be one of {{'hand', 'foot', 'full'}}, got {variant!r}"
        )

    if joints_22.shape != (T, 22, 3):
        raise ValueError(
            f"joints_22 must be (T, 22, 3); got {joints_22.shape!r}"
        )
    if object_positions.shape != (T, 3):
        raise ValueError(
            f"object_positions must be (T, 3); got {object_positions.shape!r}"
        )
    if object_rotations.shape != (T, 3):
        raise ValueError(
            f"object_rotations must be (T, 3) axis-angle; got "
            f"{object_rotations.shape!r}"
        )
    if contact_state.shape != (T, 5):
        raise ValueError(
            f"contact_state must be (T, 5); got {contact_state.shape!r}"
        )

    # --- Hand-side hint ------------------------------------------------
    # hand_contact_mask: (T, 2) in {0, 1} for [L, R].
    hand_contact = contact_state[
        :, [CONTACT_LEFT_HAND_COL, CONTACT_RIGHT_HAND_COL]
    ].astype(np.float32)                                                # (T, 2)
    # Pseudo-labels are 0/1 but allow soft values too — clamp for safety.
    hand_contact = np.clip(hand_contact, 0.0, 1.0)

    # Object-local frame. Project convention (see ``contact_guidance.py``
    # L258 ``pc_world = einsum("tij,nj->tni", R_obj, pc_local)``) is
    # ``world = R_obj @ local``, so ``local = R_obj.T @ world``.
    R_obj = axis_angle_to_matrix_np(
        object_rotations.astype(np.float32)
    )                                                                   # (T, 3, 3)
    R_obj_T = R_obj.transpose(0, 2, 1)                                  # (T, 3, 3)
    wrist_world = np.stack(
        [
            joints_22[:, LEFT_WRIST_IDX, :],
            joints_22[:, RIGHT_WRIST_IDX, :],
        ],
        axis=1,
    ).astype(np.float32)                                                # (T, 2, 3)
    obj_pos = object_positions[:, None, :].astype(np.float32)           # (T, 1, 3)
    rel = np.einsum(
        "tij,thj->thi",
        R_obj_T,
        wrist_world - obj_pos,
    ).astype(np.float32)                                                # (T, 2, 3)
    # Mask non-contact frames so the network does not see a target
    # outside contact segments (roadmap §6.5 final block).
    rel = rel * hand_contact[:, :, None]
    clamp = float(hand_offset_clamp_m)
    rel = np.clip(rel, -clamp, clamp) / clamp                            # (T, 2, 3)

    hand_hint = np.concatenate(
        [hand_contact, rel.reshape(T, 6)],
        axis=-1,
    ).astype(np.float32)                                                # (T, 8)

    # --- Foot-side hint ------------------------------------------------
    foot_stance, ankle_height_norm = derive_foot_stance_from_gt(
        joints_22, fps=fps,
        left_ankle_idx=LEFT_ANKLE_IDX,
        right_ankle_idx=RIGHT_ANKLE_IDX,
    )                                                                   # (T, 2), (T, 2)
    walking_mask = derive_walking_mask_from_gt(
        joints_22, fps=fps, root_idx=ROOT_IDX,
    )                                                                   # (T, 1)
    foot_hint = np.concatenate(
        [foot_stance, ankle_height_norm, walking_mask],
        axis=-1,
    ).astype(np.float32)                                                # (T, 5)

    if variant == "hand":
        out = hand_hint
    elif variant == "foot":
        out = foot_hint
    else:  # "full"
        out = np.concatenate([hand_hint, foot_hint], axis=-1)            # (T, 13)

    if not np.isfinite(out).all():
        raise FloatingPointError(
            "oracle_interaction_hint contains non-finite values — check "
            "joints_22 / object_positions / object_rotations for NaN/Inf "
            "and verify object_rotations is axis-angle (not rot6d)."
        )
    return out


def hint_dim(variant: str) -> int:
    """Return the expected output dim of ``build_oracle_interaction_hint``."""
    return {"hand": HINT_DIM_HAND, "foot": HINT_DIM_FOOT, "full": HINT_DIM_FULL}[
        variant
    ]
