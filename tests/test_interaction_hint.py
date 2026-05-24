"""Unit tests for ``piano.data.interaction_hint`` (Tier-0A Commit 1).

Synthesised inputs verify:

1. Hand offset is computed in the **object-local** frame (rotating the
   object by 90° around Y rotates the wrist offset components
   correspondingly, instead of leaving them in world frame).
2. Hand offset is zero on non-contact frames (masking by
   ``hand_contact``).
3. Walking mask fires when the root XZ speed exceeds threshold and is
   zero when the body is stationary.
4. Foot stance is high when ankle is on the floor with zero velocity,
   and low when ankle is high in the air.
5. ``hint_dim`` matches the produced tensor's last dimension for each
   variant.
6. NaN / Inf in input would raise (sanity).
"""
from __future__ import annotations

import numpy as np
import pytest

from piano.data.interaction_hint import (
    HINT_DIM_FOOT,
    HINT_DIM_FULL,
    HINT_DIM_HAND,
    LEFT_ANKLE_IDX,
    LEFT_WRIST_IDX,
    RIGHT_ANKLE_IDX,
    RIGHT_WRIST_IDX,
    build_oracle_interaction_hint,
    derive_foot_stance_from_gt,
    derive_walking_mask_from_gt,
    hint_dim,
)


def _make_clip(T: int = 30) -> dict:
    """Build a minimal synthetic clip with all required arrays."""
    joints = np.zeros((T, 22, 3), dtype=np.float32)
    # Default ankles + wrists at sensible heights.
    joints[:, LEFT_ANKLE_IDX] = np.array([+0.1, 0.0, 0.0])
    joints[:, RIGHT_ANKLE_IDX] = np.array([-0.1, 0.0, 0.0])
    joints[:, LEFT_WRIST_IDX] = np.array([+0.3, 1.2, 0.2])
    joints[:, RIGHT_WRIST_IDX] = np.array([-0.3, 1.2, 0.2])

    object_positions = np.zeros((T, 3), dtype=np.float32)
    object_positions[:] = np.array([0.0, 1.0, 0.5])
    object_rotations = np.zeros((T, 3), dtype=np.float32)  # identity
    contact_state = np.zeros((T, 5), dtype=np.float32)  # all not in contact
    return dict(
        joints=joints,
        object_positions=object_positions,
        object_rotations=object_rotations,
        contact_state=contact_state,
    )


def test_hand_offset_object_local_under_90deg_rotation():
    """A 90° rotation around Y must rotate the object-local wrist offset."""
    clip = _make_clip(T=10)
    # Force left-hand contact for all 10 frames.
    clip["contact_state"][:, 0] = 1.0

    # Frame-by-frame world wrist is (+0.3, 1.2, 0.2); world obj is
    # (0, 1, 0.5). World offset = wrist - obj = (+0.3, 0.2, -0.3).
    # Object-local frame for identity rotation = world frame.
    hint0 = build_oracle_interaction_hint(
        joints_22=clip["joints"],
        object_positions=clip["object_positions"],
        object_rotations=clip["object_rotations"],
        contact_state=clip["contact_state"],
        variant="hand",
    )
    # Hand offset slice for left hand is hint[:, 2:5]. Stored after
    # clamping to [-2, 2] then dividing by 2 → divide world offset by 2.
    L_off_identity = hint0[0, 2:5]
    expected_identity = np.array([+0.3, 0.2, -0.3], dtype=np.float32) / 2.0
    np.testing.assert_allclose(L_off_identity, expected_identity, atol=1e-5)

    # Now rotate the object by +90° around Y. World wrist unchanged;
    # object-local offset = R_obj.T @ world_offset.
    # R_y(+90°) = [[0,0,1],[0,1,0],[-1,0,0]]; R_y.T = R_y(-90°) =
    # [[0,0,-1],[0,1,0],[+1,0,0]]. So
    # R_y.T @ (0.3, 0.2, -0.3) = (-(-0.3), 0.2, +0.3) = (+0.3, 0.2, +0.3).
    rot = clip["object_rotations"].copy()
    rot[:, 1] = np.pi / 2.0  # rotate around Y axis
    hint90 = build_oracle_interaction_hint(
        joints_22=clip["joints"],
        object_positions=clip["object_positions"],
        object_rotations=rot,
        contact_state=clip["contact_state"],
        variant="hand",
    )
    L_off_90 = hint90[0, 2:5]
    expected_90 = np.array([+0.3, 0.2, +0.3], dtype=np.float32) / 2.0
    np.testing.assert_allclose(L_off_90, expected_90, atol=1e-5)


def test_hand_offset_zero_when_no_contact():
    """Wrist offset must be zeroed on non-contact frames."""
    clip = _make_clip(T=10)
    # contact_state already all-zero in _make_clip
    hint = build_oracle_interaction_hint(
        joints_22=clip["joints"],
        object_positions=clip["object_positions"],
        object_rotations=clip["object_rotations"],
        contact_state=clip["contact_state"],
        variant="hand",
    )
    # Hand contact (first 2 dims) must be zero.
    np.testing.assert_array_equal(hint[:, :2], 0.0)
    # All 6 offset dims must be zero (masked).
    np.testing.assert_array_equal(hint[:, 2:8], 0.0)


def test_walking_mask_threshold():
    """Walking mask is 1 when root XZ speed > threshold, 0 when stationary."""
    T = 20
    joints = np.zeros((T, 22, 3), dtype=np.float32)
    # Move root +0.5 m along X in 1 frame at 20 fps → 10 m/s, well above
    # the 0.10 m/s default. First frame is always 0 by construction.
    joints[:, 0, 0] = 0.0
    joints[5:, 0, 0] = 0.5
    mask = derive_walking_mask_from_gt(joints, fps=20.0)
    assert mask.shape == (T, 1)
    assert mask[0, 0] == 0.0  # first frame always 0 (no prev)
    assert mask[5, 0] == 1.0  # 0 -> 0.5 m
    assert mask[6, 0] == 0.0  # 0.5 -> 0.5 m, stationary
    assert float(mask.mean()) == pytest.approx(1.0 / T, abs=1e-6)


def test_foot_stance_grounded_vs_airborne():
    """Stance ≈ 1 when ankle on floor + still; ≈ 0 when airborne."""
    T = 10
    joints = np.zeros((T, 22, 3), dtype=np.float32)
    # Left ankle on the floor and motionless.
    joints[:, LEFT_ANKLE_IDX, :] = np.array([0.1, 0.0, 0.0])
    # Right ankle 50 cm in the air.
    joints[:, RIGHT_ANKLE_IDX, :] = np.array([-0.1, 0.5, 0.0])
    stance, ankle_h = derive_foot_stance_from_gt(joints, fps=20.0)
    assert stance.shape == (T, 2)
    assert ankle_h.shape == (T, 2)
    # Use a static-frame check (avoid first-frame velocity = 0 ambiguity).
    assert stance[5, 0] > 0.9, f"L stance on floor should be high, got {stance[5, 0]}"
    assert stance[5, 1] < 0.1, f"R stance airborne should be low, got {stance[5, 1]}"
    # Ankle height normalisation: airborne foot at 50 cm → 0.5/0.5 = 1.0.
    assert ankle_h[5, 1] == pytest.approx(1.0, abs=1e-5)
    assert ankle_h[5, 0] == pytest.approx(0.0, abs=1e-5)


def test_hint_dim_matches_variants():
    for v, expected in (("hand", HINT_DIM_HAND), ("foot", HINT_DIM_FOOT),
                       ("full", HINT_DIM_FULL)):
        assert hint_dim(v) == expected
        clip = _make_clip(T=12)
        h = build_oracle_interaction_hint(
            joints_22=clip["joints"],
            object_positions=clip["object_positions"],
            object_rotations=clip["object_rotations"],
            contact_state=clip["contact_state"],
            variant=v,
        )
        assert h.shape == (12, expected), (v, h.shape)


def test_hint_finite_check_raises_on_nan_input():
    """NaN in joints must propagate to a clear FloatingPointError."""
    clip = _make_clip(T=8)
    clip["joints"][3, LEFT_WRIST_IDX, 0] = np.nan
    clip["contact_state"][:, 0] = 1.0  # force computation through the offset path
    with pytest.raises(FloatingPointError):
        build_oracle_interaction_hint(
            joints_22=clip["joints"],
            object_positions=clip["object_positions"],
            object_rotations=clip["object_rotations"],
            contact_state=clip["contact_state"],
            variant="hand",
        )
