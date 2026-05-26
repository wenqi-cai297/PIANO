"""GT-derived foot stance + walking helpers used by R29 oracle conditions.

The R29 Tier-1 cleanup removed the R27 / R28 oracle interaction-hint and
body-action-hint builders. The remaining symbols here are imported by
``stage2_oracle_conditions`` and the temporal-interaction loss tests.

Conventions
-----------

- SMPL-22 joint indices: pelvis=0, L_ankle=7, R_ankle=8, L_wrist=20,
  R_wrist=21 (from ``piano.utils.smpl_utils``).
- Up axis = Y (joint y-coordinate is height).
- Foot stance is derived from GT ankle (joint 7/8) — NOT from the
  InterAct foot-object pseudo-label.
"""
from __future__ import annotations

import numpy as np


# SMPL-22 joint indices used by this module.
LEFT_WRIST_IDX: int = 20
RIGHT_WRIST_IDX: int = 21
LEFT_ANKLE_IDX: int = 7
RIGHT_ANKLE_IDX: int = 8
ROOT_IDX: int = 0

# Body-action key joints — used by R29 stage2_oracle_conditions.
LEFT_KNEE_IDX: int = 4
RIGHT_KNEE_IDX: int = 5
NECK_IDX: int = 12

# Body-action key joint order. Pelvis is intentionally LAST so the
# "non-pelvis" slice [:5] aligns with the pelvis-local delta path and the
# pelvis index [5] uses the global root-frame delta path.
BODY_ACTION_KEY_JOINT_INDICES: tuple[int, ...] = (
    LEFT_WRIST_IDX,
    RIGHT_WRIST_IDX,
    LEFT_KNEE_IDX,
    RIGHT_KNEE_IDX,
    NECK_IDX,
    ROOT_IDX,            # pelvis — must be LAST
)
BODY_ACTION_KEY_JOINT_NAMES: tuple[str, ...] = (
    "left_wrist", "right_wrist", "left_knee", "right_knee", "neck", "pelvis",
)
NUM_BODY_ACTION_JOINTS: int = len(BODY_ACTION_KEY_JOINT_INDICES)  # 6

# Default project-wide FPS.
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

