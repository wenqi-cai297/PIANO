"""Helper for transforming world-frame points into object-local frame.

The pseudo-label extractors query distance against a *static* object mesh
loaded from disk — but the object is in fact moving in world space per
frame (via object_positions + object_rotations). So we must inverse-
transform the world-frame joint positions into the object's local frame
before querying the static mesh.

World → Local:  local = R(angles)^T @ (world - trans)

If object_rotations is None, the rotation is treated as identity (pure
translation). This is a useful fallback for older preprocessed data that
only stores object_positions.
"""
from __future__ import annotations

import numpy as np


def axis_angle_to_rotmat(aa: np.ndarray) -> np.ndarray:
    """Rodrigues formula: (3,) axis-angle → (3, 3) rotation matrix."""
    theta = float(np.linalg.norm(aa))
    if theta < 1e-8:
        return np.eye(3, dtype=np.float32)
    axis = aa / theta
    K = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ], dtype=np.float32)
    return (np.eye(3, dtype=np.float32)
            + np.sin(theta) * K
            + (1.0 - np.cos(theta)) * (K @ K))


def world_to_object_local(
    points_world: np.ndarray,
    object_positions: np.ndarray,
    object_rotations: np.ndarray | None,
) -> np.ndarray:
    """Transform per-frame world points into object-local frame.

    Parameters
    ----------
    points_world : (T, 3) — world-frame point positions, one per frame.
    object_positions : (T, 3) — per-frame object translation.
    object_rotations : (T, 3) or None — per-frame object axis-angle rotation.
        If None, treated as identity (translation-only inverse).

    Returns
    -------
    points_local : (T, 3) — same points in the object's local frame.
    """
    translated = points_world - object_positions
    if object_rotations is None:
        return translated.astype(np.float32)

    T = len(translated)
    out = np.empty_like(translated, dtype=np.float32)
    for t in range(T):
        R = axis_angle_to_rotmat(object_rotations[t].astype(np.float32))
        out[t] = R.T @ translated[t]
    return out


def world_points_batch_to_local(
    points_world: np.ndarray,
    object_positions: np.ndarray,
    object_rotations: np.ndarray | None,
) -> np.ndarray:
    """Same as ``world_to_object_local`` but for a batch of (T, K, 3) points.

    Used when transforming multiple patch centers per frame.
    """
    assert points_world.shape[0] == object_positions.shape[0]
    translated = points_world - object_positions[:, None, :]
    if object_rotations is None:
        return translated.astype(np.float32)

    T, K, _ = translated.shape
    out = np.empty_like(translated, dtype=np.float32)
    for t in range(T):
        R = axis_angle_to_rotmat(object_rotations[t].astype(np.float32))
        out[t] = translated[t] @ R   # (K, 3) @ (3, 3) — note transpose via R not R.T
    return out
