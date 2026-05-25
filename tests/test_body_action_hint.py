"""Unit tests for ``build_body_action_oracle_hint`` (Round-28 Commit 1).

Verifies (prompt §5.6):

1. Output shape == (T, 24).
2. Output is finite for sensible inputs.
3. Pelvis delta is NOT all zero when pelvis moves (special case — pelvis
   uses displacement from frame 0, NOT pelvis-relative which would be 0).
4. ``all_on`` mask mode produces an all-ones joint mask.
5. ``energy`` mask mode activates only joints whose mean delta magnitude
   crosses the threshold.
6. Body-action delta equals zero at t=0 by construction (frame-0 anchor).
7. Root-yaw-canonical frame invariance: rotating the whole clip around
   +Y at frame 0 does NOT change the per-clip deltas (since the canonical
   frame moves with the body).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from piano.data.interaction_hint import (
    BODY_ACTION_KEY_JOINT_INDICES,
    BODY_ACTION_KEY_JOINT_NAMES,
    HINT_DIM_BODY_ACTION,
    LEFT_KNEE_IDX,
    LEFT_WRIST_IDX,
    NECK_IDX,
    NUM_BODY_ACTION_JOINTS,
    RIGHT_KNEE_IDX,
    RIGHT_WRIST_IDX,
    ROOT_IDX,
    build_body_action_oracle_hint,
)


def _make_sensible_clip(T: int = 20) -> np.ndarray:
    """SMPL-22 joints with shoulders + hips at sensible positions so
    ``_facing_angle_y`` returns yaw=0 (body facing +Z).

    HumanML3D convention: ``across = (sdr_R - sdr_L) + (hip_R - hip_L)``,
    ``forward = up x across``. For yaw=0 we need ``forward = +Z``, i.e.
    ``across`` points in -X (so left-side joints are at +X, right-side
    at -X — actually the opposite: ``sdr_R - sdr_L`` needs to be -X*2
    so right shoulder at -X, left shoulder at +X). With shoulders at
    L=(+0.20,...) R=(-0.20,...), we get across_x = -0.20 - 0.20 = -0.4,
    forward_x = 0, forward_z = -0.4 → yaw = atan2(0, -0.4) = pi (facing -Z).

    To face +Z, flip: L shoulder at -X, R shoulder at +X. Then across_x
    = +0.20 - (-0.20) = +0.4, forward_x = 0, forward_z = +0.4 → yaw = 0.
    """
    joints = np.zeros((T, 22, 3), dtype=np.float32)
    # Left-side at -X, right-side at +X — so forward = +Z, yaw = 0.
    joints[:, 1] = np.array([-0.12, 0.95, 0.0])   # left_hip
    joints[:, 2] = np.array([+0.12, 0.95, 0.0])   # right_hip
    joints[:, 16] = np.array([-0.20, 1.40, 0.0])  # left_shoulder
    joints[:, 17] = np.array([+0.20, 1.40, 0.0])  # right_shoulder
    # Other key joints at neutral positions.
    joints[:, ROOT_IDX] = np.array([0.0, 0.90, 0.0])
    joints[:, LEFT_KNEE_IDX] = np.array([-0.10, 0.50, 0.02])
    joints[:, RIGHT_KNEE_IDX] = np.array([+0.10, 0.50, 0.02])
    joints[:, NECK_IDX] = np.array([0.0, 1.55, 0.0])
    joints[:, LEFT_WRIST_IDX] = np.array([-0.30, 1.20, 0.20])
    joints[:, RIGHT_WRIST_IDX] = np.array([+0.30, 1.20, 0.20])
    return joints


def test_indices_constants_match_smpl22():
    """Key joints must match piano.utils.smpl_utils.SMPL_22_JOINT_NAMES."""
    from piano.utils.smpl_utils import SMPL_22_JOINT_NAMES
    assert SMPL_22_JOINT_NAMES[LEFT_WRIST_IDX] == "left_wrist"
    assert SMPL_22_JOINT_NAMES[RIGHT_WRIST_IDX] == "right_wrist"
    assert SMPL_22_JOINT_NAMES[LEFT_KNEE_IDX] == "left_knee"
    assert SMPL_22_JOINT_NAMES[RIGHT_KNEE_IDX] == "right_knee"
    assert SMPL_22_JOINT_NAMES[NECK_IDX] == "neck"
    assert SMPL_22_JOINT_NAMES[ROOT_IDX] == "pelvis"
    # Pelvis MUST be the LAST entry — implementation relies on this.
    assert BODY_ACTION_KEY_JOINT_INDICES[-1] == ROOT_IDX
    assert BODY_ACTION_KEY_JOINT_NAMES[-1] == "pelvis"
    assert NUM_BODY_ACTION_JOINTS == 6
    assert HINT_DIM_BODY_ACTION == 24


def test_shape_and_finite():
    joints = _make_sensible_clip(T=18)
    h = build_body_action_oracle_hint(joints, mask_mode="all_on")
    assert h.shape == (18, HINT_DIM_BODY_ACTION)
    assert np.isfinite(h).all()


def test_all_on_mask_is_ones():
    joints = _make_sensible_clip(T=10)
    h = build_body_action_oracle_hint(joints, mask_mode="all_on")
    np.testing.assert_array_equal(h[:, :6], 1.0)


def test_frame_zero_delta_is_zero():
    """The frame-0 anchor → delta[0, :, :] must be exactly 0 for ALL six joints
    (including pelvis, which uses pelvis_world - pelvis_world[0])."""
    joints = _make_sensible_clip(T=15)
    # Add some movement to make the rest of the trajectory non-trivial.
    joints[:, LEFT_WRIST_IDX, 0] += np.linspace(0, 0.4, 15).astype(np.float32)
    joints[:, ROOT_IDX, 0] += np.linspace(0, 0.3, 15).astype(np.float32)
    h = build_body_action_oracle_hint(joints, mask_mode="all_on")
    delta = h[:, 6:].reshape(15, 6, 3)
    np.testing.assert_allclose(delta[0], 0.0, atol=1e-6)


def test_pelvis_delta_nonzero_when_pelvis_moves():
    """Pelvis MUST use displacement-from-frame-0, not pelvis-relative
    (which would force pelvis delta = 0 trivially). When the pelvis
    translates by 1 m in X, the pelvis delta should be ~1 m."""
    T = 12
    joints = _make_sensible_clip(T=T)
    # Translate the entire body (including pelvis) by +1 m along X from frame 5.
    joints[5:, :, 0] += 1.0
    h = build_body_action_oracle_hint(joints, mask_mode="all_on")
    delta = h[:, 6:].reshape(T, 6, 3)
    # Pelvis is the LAST joint (index 5 in the key-joint axis).
    pelvis_delta_frame10 = delta[10, 5]
    # Yaw at frame 0 derived from the symmetric shoulders/hips lies near 0,
    # so the canonical frame is essentially the world frame here. Pelvis
    # displacement frame 0 -> frame 10 should be (+1, 0, 0) ± small numeric.
    assert abs(pelvis_delta_frame10[0] - 1.0) < 1e-4, (
        f"pelvis X delta should be ~1.0, got {pelvis_delta_frame10!r}"
    )
    assert abs(pelvis_delta_frame10[1]) < 1e-4
    assert abs(pelvis_delta_frame10[2]) < 1e-4


def test_non_pelvis_delta_zero_under_rigid_translation():
    """If the WHOLE body rigidly translates, non-pelvis joints' pelvis-relative
    deltas remain zero (only the pelvis trace records the translation)."""
    T = 10
    joints = _make_sensible_clip(T=T)
    joints[5:, :, 0] += 0.7
    h = build_body_action_oracle_hint(joints, mask_mode="all_on")
    delta = h[:, 6:].reshape(T, 6, 3)
    # Joints 0..4 are non-pelvis. Their pelvis-relative position never
    # changes when the whole body translates rigidly.
    np.testing.assert_allclose(delta[:, :5, :], 0.0, atol=1e-5)


def test_energy_mask_activates_moving_joints():
    """Only joints whose mean delta magnitude exceeds the threshold should
    have mask==1."""
    T = 30
    joints = _make_sensible_clip(T=T)
    # Move ONLY the left wrist — large amplitude (0.3 m) across the clip.
    joints[:, LEFT_WRIST_IDX, 1] += np.linspace(0, 0.3, T).astype(np.float32)
    h = build_body_action_oracle_hint(
        joints, mask_mode="energy", energy_threshold=0.05,
    )
    mask = h[0, :6]
    # Left wrist is index 0 in BODY_ACTION_KEY_JOINT_INDICES.
    assert mask[0] == 1.0, f"left wrist should be active, got mask={mask}"
    # All other joints had no motion → energy ≈ 0 → mask 0.
    np.testing.assert_array_equal(mask[1:], 0.0)


def test_root_yaw_canonical_invariance():
    """Rotating the clip globally around +Y at frame 0 should NOT change the
    body-action deltas (since the per-clip canonical frame rotates with the
    body). Pelvis displacement in canonical frame is unchanged too."""
    T = 12
    joints = _make_sensible_clip(T=T)
    # Add some intra-body motion so deltas are non-trivial.
    joints[:, LEFT_WRIST_IDX, 0] += np.linspace(0, 0.25, T).astype(np.float32)
    joints[:, ROOT_IDX, 2] += np.linspace(0, 0.5, T).astype(np.float32)

    h0 = build_body_action_oracle_hint(joints, mask_mode="all_on")

    # Rotate the whole clip by +30 degrees around +Y.
    theta = math.radians(30.0)
    c, s = math.cos(theta), math.sin(theta)
    R = np.array(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32,
    )
    joints_rot = joints @ R.T   # rotates each joint
    h_rot = build_body_action_oracle_hint(joints_rot, mask_mode="all_on")

    # The delta block (and mask block) should match.
    np.testing.assert_allclose(h0, h_rot, atol=1e-4)


def test_invalid_mask_mode_raises():
    joints = _make_sensible_clip(T=5)
    with pytest.raises(ValueError):
        build_body_action_oracle_hint(joints, mask_mode="not_a_mode")


def test_invalid_shape_raises():
    bad = np.zeros((10, 21, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        build_body_action_oracle_hint(bad, mask_mode="all_on")


def test_nan_input_raises():
    joints = _make_sensible_clip(T=8)
    joints[3, LEFT_WRIST_IDX, 0] = np.nan
    with pytest.raises(FloatingPointError):
        build_body_action_oracle_hint(joints, mask_mode="all_on")
