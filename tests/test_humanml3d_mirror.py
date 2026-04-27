"""Tests for piano.utils.humanml3d_mirror.

CPU-only — uses only numpy. No torch / trimesh dependency, so this test
runs in the lightweight local dev env per restart_prompt §"Local dev
env".
"""
from __future__ import annotations

import numpy as np

from piano.utils.humanml3d_mirror import (
    _round_trip_max_error,
    mirror_motion_263,
    mirror_object_world_pose,
)


# ---------------------------------------------------------------------------
# motion_263 mirror
# ---------------------------------------------------------------------------

def test_motion_263_round_trip_is_identity() -> None:
    """Mirror is its own inverse (involution): m == mirror(mirror(m))."""
    rng = np.random.default_rng(42)
    m = rng.standard_normal((50, 263)).astype(np.float32)
    err = _round_trip_max_error(m)
    # Pure index permutation + sign flips; no float ops introduce error.
    assert err == 0.0, f"round-trip should be bit-exact, got {err}"


def test_motion_263_single_mirror_changes_input() -> None:
    """Sanity: a single mirror produces non-trivially different output."""
    rng = np.random.default_rng(7)
    m = rng.standard_normal((30, 263)).astype(np.float32)
    m1 = mirror_motion_263(m)
    diff = float(np.abs(m - m1).max())
    assert diff > 0.5, f"mirror was a near-noop on random input (max diff {diff})"


def test_motion_263_root_features() -> None:
    """[0]: y-rotation vel flips. [1]: x-vel flips. [2]: z-vel unchanged.
    [3]: y-height unchanged."""
    rng = np.random.default_rng(13)
    m = rng.standard_normal((20, 263)).astype(np.float32)
    m1 = mirror_motion_263(m)
    np.testing.assert_allclose(m[..., 0], -m1[..., 0])
    np.testing.assert_allclose(m[..., 1], -m1[..., 1])
    np.testing.assert_allclose(m[..., 2],  m1[..., 2])
    np.testing.assert_allclose(m[..., 3],  m1[..., 3])


def test_motion_263_ric_swaps_lr_pairs() -> None:
    """Hands (joints 20, 21) and feet (joints 10, 11) swap under mirror."""
    m = np.zeros((1, 263), dtype=np.float32)
    # ric_data covers joints 1..21 at offset 4..67. Joint i lives at
    # offset 4 + 3 * (i - 1).
    # Joint 20 (left wrist) at offset 4 + 3*19 = 61, 62, 63.
    # Joint 21 (right wrist) at offset 4 + 3*20 = 64, 65, 66.
    m[0,  61:64] = [1.0, 2.0, 3.0]   # left wrist xyz (canonical-frame)
    m[0,  64:67] = [4.0, 5.0, 6.0]   # right wrist xyz
    m1 = mirror_motion_263(m)
    # After mirror: left ↔ right swap, plus x flips.
    np.testing.assert_allclose(m1[0, 61:64], [-4.0, 5.0, 6.0])
    np.testing.assert_allclose(m1[0, 64:67], [-1.0, 2.0, 3.0])


def test_motion_263_feet_contact_swaps() -> None:
    """foot_contact [259:263] = [l_ankle, l_toe, r_ankle, r_toe] →
    [r_ankle, r_toe, l_ankle, l_toe]."""
    m = np.zeros((1, 263), dtype=np.float32)
    m[0, 259:263] = [0.1, 0.2, 0.7, 0.9]
    m1 = mirror_motion_263(m)
    np.testing.assert_allclose(m1[0, 259:263], [0.7, 0.9, 0.1, 0.2])


def test_motion_263_cont6d_signs() -> None:
    """For each cont6d block (b1, b2): mirror produces signs
    [+, -, -, -, +, +]."""
    m = np.zeros((1, 263), dtype=np.float32)
    # rot_data starts at offset 67. Joint 1 (idx 0 in 21-joint subset)
    # cont6d at offset 67..73.
    m[0, 67:73] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    # Note: joint 1 is left_hip; mirror swaps with joint 2 (right_hip)
    # at offset 67+6 = 73..79. So we also need joint 2's value to test
    # the sign-flip in isolation. Set joint 2 to zeros for that.
    m1 = mirror_motion_263(m)
    # After mirror: joint 1's cont6d position now holds joint 2's
    # mirrored values (which were zero), so post-mirror joint 1 should
    # be zeros.
    np.testing.assert_allclose(m1[0, 67:73], [0, 0, 0, 0, 0, 0])
    # The mirror of joint 1's original (which moved to joint 2 slot at
    # 73..79) should be [+1, -2, -3, -4, +5, +6]:
    np.testing.assert_allclose(m1[0, 73:79], [1.0, -2.0, -3.0, -4.0, 5.0, 6.0])


def test_motion_263_wrong_dim_raises() -> None:
    import pytest
    with pytest.raises(ValueError, match="263"):
        mirror_motion_263(np.zeros((10, 100), dtype=np.float32))


# ---------------------------------------------------------------------------
# Object world-pose mirror
# ---------------------------------------------------------------------------

def test_object_pose_round_trip() -> None:
    rng = np.random.default_rng(101)
    pos = rng.standard_normal((40, 3)).astype(np.float32)
    rot = rng.standard_normal((40, 3)).astype(np.float32) * 0.5
    pos_m, rot_m = mirror_object_world_pose(pos, rot)
    pos_back, rot_back = mirror_object_world_pose(pos_m, rot_m)
    assert float(np.abs(pos - pos_back).max()) == 0.0
    assert float(np.abs(rot - rot_back).max()) == 0.0


def test_object_pose_x_flip() -> None:
    pos = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    rot = np.array([[0.5, 0.6, 0.7]], dtype=np.float32)
    pos_m, rot_m = mirror_object_world_pose(pos, rot)
    np.testing.assert_allclose(pos_m, [[-1.0, 2.0, 3.0]])
    # axis-angle: (v_x, v_y, v_z) → (v_x, -v_y, -v_z)
    np.testing.assert_allclose(rot_m, [[0.5, -0.6, -0.7]])


def test_object_rotation_around_y_inverts() -> None:
    """A pure rotation around +Y (axis-angle (0, θ, 0)) under x-mirror
    becomes rotation around +Y by -θ — i.e., axis-angle (0, -θ, 0).

    This matches physical intuition: looking at someone spinning right,
    if you mirror them through a vertical plane, they're now spinning
    left.
    """
    theta = 0.7
    rot = np.array([[0.0, theta, 0.0]], dtype=np.float32)
    _, rot_m = mirror_object_world_pose(np.zeros((1, 3), dtype=np.float32), rot)
    np.testing.assert_allclose(rot_m, [[0.0, -theta, 0.0]])


def test_object_rotation_around_x_unchanged() -> None:
    """Pure rotation around +X axis (axis-angle (θ, 0, 0)) — axis is
    in the mirror plane (x-axis), so the rotation should be unchanged
    under x-mirror."""
    theta = 0.7
    rot = np.array([[theta, 0.0, 0.0]], dtype=np.float32)
    _, rot_m = mirror_object_world_pose(np.zeros((1, 3), dtype=np.float32), rot)
    np.testing.assert_allclose(rot_m, [[theta, 0.0, 0.0]])
