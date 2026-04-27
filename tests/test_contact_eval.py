"""Smoke tests for the contact-eval geometric helpers.

The full ``build_contact_eval_fn`` requires MoMask + CLIP and is
server-only; here we test the pure-numpy geometric pieces:

- ``_lift_canonical_to_world``: round-trip with known (R_y, T_xz) on a
  hand-built joint set.
- ``_per_frame_body_to_object_distance``: distance to a 1-point object
  PC equals ``|body - obj|`` for each (T, n_parts) pair.
- ``compute_clip_contact_distance``: smoke-test end-to-end on a
  synthetic clip, asserting shape + a known relationship (no NaN /
  inf, distance >= 0).

These pin the math contracts. Anything that breaks them breaks
``best_contact.pt`` selection.
"""
from __future__ import annotations

import numpy as np
import pytest


def test_lift_canonical_to_world_zero_transform_is_identity():
    from piano.training.contact_eval import _lift_canonical_to_world

    canon = np.array(
        [[[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]],   # (T=1, J=2, 3)
        dtype=np.float32,
    )
    world = _lift_canonical_to_world(canon, R_y_angle=0.0, T_xz=np.array([0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(world, canon, atol=1e-6)


def test_lift_canonical_to_world_pure_translation():
    from piano.training.contact_eval import _lift_canonical_to_world

    canon = np.zeros((1, 1, 3), dtype=np.float32)
    canon[0, 0] = [1.0, 2.0, 3.0]
    world = _lift_canonical_to_world(
        canon, R_y_angle=0.0, T_xz=np.array([10.0, -5.0], dtype=np.float32),
    )
    # Rotation is identity. X gets +10, Y unchanged, Z gets -5.
    np.testing.assert_allclose(world[0, 0], [11.0, 2.0, -2.0], atol=1e-6)


def test_lift_canonical_to_world_pure_y_rotation():
    from piano.training.contact_eval import _lift_canonical_to_world

    # Point at (1, 0, 0) rotated +90° around Y should land at (0, 0, -1).
    canon = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32)
    world = _lift_canonical_to_world(
        canon, R_y_angle=np.pi / 2, T_xz=np.array([0.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_allclose(world[0, 0], [0.0, 0.0, -1.0], atol=1e-5)


def test_per_frame_body_to_object_distance_shape_and_values():
    from piano.training.contact_eval import _per_frame_body_to_object_distance

    # T=2 frames, n_parts=3, single-point PC at world origin.
    body = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    pc = np.zeros((1, 3), dtype=np.float32)
    obj_pos = np.zeros((2, 3), dtype=np.float32)
    obj_rot = np.zeros((2, 3), dtype=np.float32)   # axis-angle 0 = identity

    d = _per_frame_body_to_object_distance(body, pc, obj_pos, obj_rot)
    assert d.shape == (2, 3)
    np.testing.assert_allclose(d[0], [1.0, 2.0, 3.0], atol=1e-6)
    np.testing.assert_allclose(d[1], [0.0, 4.0, 0.0], atol=1e-6)


def test_per_frame_body_to_object_distance_object_translation():
    from piano.training.contact_eval import _per_frame_body_to_object_distance

    # Body fixed at origin; object slides from (0,0,0) → (1,0,0).
    body = np.zeros((2, 1, 3), dtype=np.float32)
    pc = np.zeros((1, 3), dtype=np.float32)
    obj_pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    obj_rot = np.zeros((2, 3), dtype=np.float32)

    d = _per_frame_body_to_object_distance(body, pc, obj_pos, obj_rot)
    np.testing.assert_allclose(d[0, 0], 0.0, atol=1e-6)
    np.testing.assert_allclose(d[1, 0], 1.0, atol=1e-6)


def test_per_frame_body_to_object_distance_takes_min_over_pc():
    """Distance is min over PC points, not mean."""
    from piano.training.contact_eval import _per_frame_body_to_object_distance

    body = np.zeros((1, 1, 3), dtype=np.float32)        # body at origin
    # PC has 2 points: one at distance 1, one at distance 100.
    pc = np.array([[1.0, 0.0, 0.0], [100.0, 0.0, 0.0]], dtype=np.float32)
    obj_pos = np.zeros((1, 3), dtype=np.float32)
    obj_rot = np.zeros((1, 3), dtype=np.float32)

    d = _per_frame_body_to_object_distance(body, pc, obj_pos, obj_rot)
    np.testing.assert_allclose(d[0, 0], 1.0, atol=1e-6)


def test_compute_clip_contact_distance_handles_decoded_shorter_than_seq_len():
    """Regression for 2026-04-28 server-training crash.

    MoMask VQ-VAE has total stride 4. For seq_len=186 the generator
    outputs ``(186 // 4) * 4 = 184`` frames. ``compute_clip_contact_distance``
    received ``seq_len=186`` and tried to index source object trajectories
    by 186 while body had 184 → broadcast error (184,5,1,3) vs (186,1,N,3).
    Fix: ``T = min(seq_len, motion_gen.shape[0])``. This test pins the
    contract.
    """
    try:
        import piano.models.backbones.momask_adapter  # noqa: F401
        from utils.motion_process import recover_from_ric
    except Exception:
        pytest.skip("MoMask repo not on sys.path (server-only test)")

    from piano.training.contact_eval import compute_clip_contact_distance

    T_gen = 184          # what the model actually produces
    T_src = 186          # source clip's seq_len (≠ multiple of 4)

    motion = np.zeros((T_gen, 263), dtype=np.float32)
    motion[:, 3] = 1.0   # root height

    pc = np.random.RandomState(0).randn(64, 3).astype(np.float32)
    obj_pos = np.zeros((T_src, 3), dtype=np.float32)
    obj_pos[:, 0] = np.linspace(0.0, 1.0, T_src)
    obj_rot = np.zeros((T_src, 3), dtype=np.float32)

    d = compute_clip_contact_distance(
        motion_263_generated=motion,
        R_y_angle=0.0,
        T_xz=np.zeros(2, dtype=np.float32),
        object_pc_local=pc,
        object_positions=obj_pos,
        object_rotations=obj_rot,
        seq_len=T_src,                              # callers pass source seq_len
        recover_from_ric_fn=recover_from_ric,
    )
    assert isinstance(d, float)
    assert d >= 0.0
    assert np.isfinite(d)


def test_compute_clip_contact_distance_does_not_crash_and_is_nonneg():
    """End-to-end synthetic — exercises recover_from_ric + lift + distance.

    Skipped if MoMask repo isn't on sys.path (CPU-only dev env).
    """
    try:
        import piano.models.backbones.momask_adapter  # noqa: F401
        from utils.motion_process import recover_from_ric
    except Exception:
        pytest.skip("MoMask repo not on sys.path (server-only test)")

    from piano.training.contact_eval import compute_clip_contact_distance

    # Build a synthetic stationary motion_263: small std, zero velocities.
    T = 16
    motion = np.zeros((T, 263), dtype=np.float32)
    motion[:, 3] = 1.0   # root height = 1m

    pc = np.random.RandomState(0).randn(64, 3).astype(np.float32)
    obj_pos = np.zeros((T, 3), dtype=np.float32)
    obj_pos[:, 0] = np.linspace(0.0, 1.0, T)
    obj_rot = np.zeros((T, 3), dtype=np.float32)

    d = compute_clip_contact_distance(
        motion_263_generated=motion,
        R_y_angle=0.0,
        T_xz=np.zeros(2, dtype=np.float32),
        object_pc_local=pc,
        object_positions=obj_pos,
        object_rotations=obj_rot,
        seq_len=T,
        recover_from_ric_fn=recover_from_ric,
    )
    assert isinstance(d, float)
    assert d >= 0.0
    assert np.isfinite(d)
