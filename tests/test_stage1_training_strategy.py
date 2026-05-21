from pathlib import Path

import numpy as np
import torch

from piano.training.train_coarse_prior import (
    mirror_coarse_v1,
    mirror_obj_traj_root0_world,
    resolve_best_val_checkpoint,
)
from piano.training.smpl_kinematics import (
    matrix_to_rotation_6d as _matrix_to_rotation_6d_rows,
)
from piano.utils.canonical_frame import (
    matrix_to_rotation_6d_np as _matrix_to_rotation_6d_canonical_frame,
)


# ----------------------------------------------------------------------
# Helpers for ground-truth rot6d derivation under reflection.
# ----------------------------------------------------------------------

# X-axis reflection. R' = M R M for a global rotation under simultaneous
# mirror of both the world frame and the body's internal frame.
_MIRROR_X = np.diag([-1.0, 1.0, 1.0]).astype(np.float32)


def _R_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32,
    )


def _R_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32,
    )


def _R_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32,
    )


def _cont6d_rows(R: np.ndarray) -> np.ndarray:
    """Pack R via the smpl_kinematics ROWS convention (motion_135 layout)."""
    return _matrix_to_rotation_6d_rows(torch.from_numpy(R)).numpy()


def _cont6d_canonical_frame(R: np.ndarray) -> np.ndarray:
    """Pack R via canonical_frame's COLS-row-interleaved convention (obj_traj layout)."""
    return _matrix_to_rotation_6d_canonical_frame(R)


def test_mirror_coarse_v1_is_involution_and_flips_expected_channels():
    x = np.arange(46, dtype=np.float32).reshape(2, 23)
    mirrored = mirror_coarse_v1(x)
    round_trip = mirror_coarse_v1(mirrored)

    np.testing.assert_allclose(round_trip, x)
    np.testing.assert_allclose(mirrored[:, 0], -x[:, 0])
    np.testing.assert_allclose(mirrored[:, 3], -x[:, 3])
    np.testing.assert_allclose(mirrored[:, 6], -x[:, 6])
    np.testing.assert_allclose(mirrored[:, 8], -x[:, 8])
    np.testing.assert_allclose(mirrored[:, 1], x[:, 1])
    np.testing.assert_allclose(mirrored[:, 2], x[:, 2])
    np.testing.assert_allclose(mirrored[:, 7], x[:, 7])


def test_mirror_obj_traj_root0_world_is_involution():
    x = np.arange(18, dtype=np.float32).reshape(2, 9)
    mirrored = mirror_obj_traj_root0_world(x)
    round_trip = mirror_obj_traj_root0_world(mirrored)

    np.testing.assert_allclose(round_trip, x)
    np.testing.assert_allclose(mirrored[:, 0], -x[:, 0])
    np.testing.assert_allclose(mirrored[:, 1:3], x[:, 1:3])


def test_mirror_coarse_v1_pelvis_rot6d_yields_M_R_M_under_reflection():
    """End-to-end correctness for the rot6d block: pack pelvis/spine3 R
    into Coarse-v1 layout via the smpl_kinematics ROWS convention (the
    actual convention motion_135 uses — Round-12 Codex review confirmed
    this), apply mirror_coarse_v1, then verify the mirrored cont6d is
    bit-exact equal to the cont6d of (M R M).

    This catches sign-pattern bugs the involution test would miss (any
    invertible sign mask round-trips, but only the correct pattern
    reproduces the reflection identity).
    """
    rng = np.random.default_rng(0)
    yaws = [0.0, 0.7, -1.3, 2.4]
    pitches = [0.0, 0.4, -0.6, 1.1]
    rolls = [0.0, -0.5, 0.9, 1.7]
    T = len(yaws)
    coarse = np.zeros((T, 23), dtype=np.float32)
    coarse[:, 0:3] = rng.normal(size=(T, 3)).astype(np.float32)
    coarse[:, 3:6] = rng.normal(size=(T, 3)).astype(np.float32)
    coarse[:, 21:23] = rng.normal(size=(T, 2)).astype(np.float32)
    pelvis_R = np.stack([_R_y(y) @ _R_x(p) @ _R_z(r) for y, p, r in zip(yaws, pitches, rolls)]).astype(np.float32)
    spine3_R = np.stack([_R_z(r) @ _R_y(y) @ _R_x(p) for y, p, r in zip(yaws, pitches, rolls)]).astype(np.float32)
    coarse[:, 9:15] = _cont6d_rows(pelvis_R)
    coarse[:, 15:21] = _cont6d_rows(spine3_R)
    # Yaw sin/cos packed consistent with pelvis_R's forward column (col 2),
    # per extract_coarse_motion_representation.py.
    forward = pelvis_R[:, :, 2]
    coarse_yaw = np.arctan2(forward[:, 0], forward[:, 2])
    coarse[:, 6] = np.sin(coarse_yaw)
    coarse[:, 7] = np.cos(coarse_yaw)
    coarse[:, 8] = rng.normal(size=(T,)).astype(np.float32)

    mirrored = mirror_coarse_v1(coarse)

    pelvis_R_expected = _MIRROR_X @ pelvis_R @ _MIRROR_X
    spine3_R_expected = _MIRROR_X @ spine3_R @ _MIRROR_X
    np.testing.assert_allclose(
        mirrored[:, 9:15], _cont6d_rows(pelvis_R_expected.astype(np.float32)),
        atol=1e-5,
        err_msg="mirror_coarse_v1 pelvis_rot6d does not satisfy cont6d(M R M)",
    )
    np.testing.assert_allclose(
        mirrored[:, 15:21], _cont6d_rows(spine3_R_expected.astype(np.float32)),
        atol=1e-5,
        err_msg="mirror_coarse_v1 spine3_rot6d does not satisfy cont6d(M R M)",
    )

    # Yaw consistency: yaw' = -yaw → sin/cos must agree with mirrored
    # pelvis_R's forward column.
    forward_mirror_expected = pelvis_R_expected[:, :, 2]
    yaw_mirror_expected = np.arctan2(forward_mirror_expected[:, 0], forward_mirror_expected[:, 2])
    np.testing.assert_allclose(mirrored[:, 6], np.sin(yaw_mirror_expected), atol=1e-5)
    np.testing.assert_allclose(mirrored[:, 7], np.cos(yaw_mirror_expected), atol=1e-5)
    np.testing.assert_allclose(mirrored[:, 8], -coarse[:, 8])
    np.testing.assert_allclose(mirrored[:, 0], -coarse[:, 0])
    np.testing.assert_allclose(mirrored[:, 1:3], coarse[:, 1:3])
    np.testing.assert_allclose(mirrored[:, 3], -coarse[:, 3])
    np.testing.assert_allclose(mirrored[:, 4:6], coarse[:, 4:6])
    np.testing.assert_allclose(mirrored[:, 21:23], coarse[:, 21:23])


def test_mirror_obj_traj_root0_world_rot6d_yields_M_R_M_under_reflection():
    """Same R' = M R M check on obj_rot6d (dims [3:9]) under X-mirror.

    obj_rot6d is stored via canonical_frame's COLS-row-interleaved
    cont6d convention (NOT the smpl_kinematics ROWS convention used by
    Coarse-v1). The two conventions have DIFFERENT X-mirror sign
    patterns; the original Round-20 implementation used the ROWS pattern
    here by mistake, producing wrong values on cont6d positions 3 and 4.
    """
    R_obj = np.stack([
        _R_y(0.0), _R_z(0.5), _R_y(1.3) @ _R_x(-0.4), _R_x(2.0) @ _R_z(-0.7),
    ]).astype(np.float32)
    T = R_obj.shape[0]
    obj_traj = np.zeros((T, 9), dtype=np.float32)
    rng = np.random.default_rng(1)
    obj_traj[:, 0:3] = rng.normal(size=(T, 3)).astype(np.float32)
    obj_traj[:, 3:9] = _cont6d_canonical_frame(R_obj)

    mirrored = mirror_obj_traj_root0_world(obj_traj)

    np.testing.assert_allclose(mirrored[:, 0], -obj_traj[:, 0])
    np.testing.assert_allclose(mirrored[:, 1:3], obj_traj[:, 1:3])

    R_obj_expected = _MIRROR_X @ R_obj @ _MIRROR_X
    np.testing.assert_allclose(
        mirrored[:, 3:9],
        _cont6d_canonical_frame(R_obj_expected.astype(np.float32)),
        atol=1e-5,
        err_msg="mirror_obj_traj_root0_world obj_rot6d does not satisfy cont6d(M R M)",
    )


def test_resolve_best_val_checkpoint_exact_periodic(tmp_path: Path):
    exact = tmp_path / "ckpt-030000.pt"
    exact.write_bytes(b"")

    info = resolve_best_val_checkpoint(tmp_path, 30000)

    assert info["best_val_ckpt_path"] == str(exact)
    assert info["best_val_ckpt_step"] == 30000
    assert info["best_val_ckpt_exact"] is True
    assert info["best_val_nearest_ckpt_path"] == str(exact)


def test_resolve_best_val_checkpoint_nearest_when_non_exact(tmp_path: Path):
    nearest = tmp_path / "ckpt-030000.pt"
    nearest.write_bytes(b"")

    info = resolve_best_val_checkpoint(tmp_path, 35000)

    assert info["best_val_ckpt_path"] is None
    assert info["best_val_ckpt_step"] is None
    assert info["best_val_ckpt_exact"] is False
    assert info["best_val_nearest_ckpt_path"] == str(nearest)
    assert info["best_val_nearest_ckpt_step"] == 30000


def test_resolve_best_val_checkpoint_final_exact(tmp_path: Path):
    final = tmp_path / "final.pt"
    final.write_bytes(b"")

    info = resolve_best_val_checkpoint(
        tmp_path,
        40000,
        final_ckpt_path=final,
        final_step=40000,
    )

    assert info["best_val_ckpt_path"] == str(final)
    assert info["best_val_ckpt_step"] == 40000
    assert info["best_val_ckpt_exact"] is True
