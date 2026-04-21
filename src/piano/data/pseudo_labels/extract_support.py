"""Extract support state pseudo-labels from HOI motion data.

Classifies each frame into one of four body support configurations
based on foot and pelvis contact patterns.

Support states:
    0 = both_feet    — both feet on ground
    1 = single_foot  — only one foot on ground
    2 = sitting      — pelvis contacts object, body is stationary, AND
                       the object is below the pelvis (geometric test)
    3 = hand_support — hands providing primary support (e.g., leaning)

``sitting`` uses two disambiguating conditions on top of pelvis contact:

1. **Pelvis stationary** (XZ-plane speed < 0.15 m/s, 1 s moving average).
   A seated person is stationary; someone pushing a chair walks or
   shuffles at 0.2-0.5 m/s. Rejects neuraldome `subject01_chair_0` style
   false positives where the user stands behind the chair and pushes it.

2. **Upward-facing mesh surface sits within a cylinder below the pelvis**
   (XZ radius ~0.15 m, extending ~0.30 m downward; candidate surface
   has a face normal within ~45° of +Y). The physical signature of
   sitting is that a seat surface is directly under the pelvis and can
   support body weight; a person standing beside a chair still
   triggers pelvis contact (joint within 20 cm of backrest) but has no
   seat-like surface beneath them. Rejects neuraldome
   `subject01_bigsofa_330` (push) and keeps neuraldome
   `subject01_bigsofa_1310` (sit on sofa edge with pelvis offset
   toward the armrest — closest point lies on the armrest
   horizontally, but the seat sits directly below pelvis so the
   cylinder-and-normal test still opens).

Both conditions are conjunctions — either being false rejects sitting.
If ``joints`` or ``object_mesh`` is unavailable, the corresponding gate
defaults to open, preserving legacy behaviour at the cost of more FP.

Output: integer support array of shape ``(T,)``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Support state constants
SUPPORT_BOTH_FEET = 0
SUPPORT_SINGLE_FOOT = 1
SUPPORT_SITTING = 2
SUPPORT_HAND = 3

SUPPORT_NAMES: list[str] = [
    "both_feet",
    "single_foot",
    "sitting",
    "hand_support",
]
NUM_SUPPORT_STATES: int = len(SUPPORT_NAMES)


@dataclass(slots=True)
class SupportConfig:
    """Configuration for support state extraction."""

    contact_threshold: float = 0.5  # binarization threshold for contact scores
    smoothing_window: int = 7       # temporal majority-filter window
    fps: float = 30.0               # used to turn per-frame diffs into m/s
    # Moving-average horizontal pelvis speed (m/s) above which a frame
    # with pelvis contact is *not* classified as sitting. Reference:
    # seated person's small shifts produce <0.10 m/s; slow object push
    # is 0.2-0.5 m/s; walking is >1 m/s.
    sitting_max_pelvis_horz_speed: float = 0.15
    sitting_velocity_window_sec: float = 1.0
    # "Object below pelvis" gate parameters. Replaces the earlier
    # closest-point-direction gate, which mis-fired for sitting-on-sofa-
    # edge poses where the closest mesh point is on a nearby armrest
    # (direction horizontal) even though a seat surface lies directly
    # below the pelvis. The new gate inspects a thin cylinder below the
    # pelvis and requires an upward-facing surface inside it.
    sitting_below_horz_radius: float = 0.15     # cylinder radius (m)
    sitting_below_vert_gate: float = 0.30       # cylinder height below pelvis (m)
    # Minimum +Y component of face normal for a surface to count as
    # "seat-like". 0.7 ≈ within 45° of vertical. Filters out backrest
    # / leg / armrest side faces that happen to intersect the cylinder.
    sitting_below_upward_normal_threshold: float = 0.7


def _majority_filter(labels: np.ndarray, size: int) -> np.ndarray:
    """Temporal majority filter for categorical labels.

    A median filter is wrong on support ids {0=both_feet, 1=single_foot,
    2=sitting, 3=hand_support}: those ids have no ordinal meaning, so
    ``median([single_foot, sitting, hand_support]) == sitting`` is
    arbitrary. Mode returns the most frequent label in the window, which
    is the right generalisation of median to unordered categories.

    Edge frames are padded by replication (same as scipy's ``mode="edge"``).
    """
    if size <= 1:
        return labels
    pad = size // 2
    padded = np.pad(labels, pad, mode="edge")
    out = np.empty_like(labels)
    for i in range(len(labels)):
        window = padded[i : i + size]
        out[i] = int(np.bincount(window).argmax())
    return out


def extract_support_state(
    contact_state: np.ndarray,
    joints: np.ndarray | None = None,
    object_mesh=None,
    object_positions: np.ndarray | None = None,
    object_rotations: np.ndarray | None = None,
    config: SupportConfig | None = None,
) -> np.ndarray:
    """Extract per-frame support state from contact pseudo-labels.

    Parameters
    ----------
    contact_state : (T, 5) — soft contact for
        [left_hand, right_hand, left_foot, right_foot, pelvis]
    joints : (T, 22, 3) or None — SMPL joint positions in world frame.
        Required for pelvis-velocity gate on ``sitting``.
    object_mesh : ``trimesh.Trimesh`` or None — object mesh in its local
        frame. Required for the geometric "object-below-pelvis" gate on
        ``sitting``.
    object_positions : (T, 3) or None — per-frame object translation.
    object_rotations : (T, 3) or None — per-frame axis-angle rotation.
    config : extraction parameters

    Returns
    -------
    support : (T,) — integer support state per frame
    """
    if config is None:
        config = SupportConfig()

    T = len(contact_state)
    tau = config.contact_threshold

    # Binarize contacts
    left_hand = contact_state[:, 0] > tau
    right_hand = contact_state[:, 1] > tau
    left_foot = contact_state[:, 2] > tau
    right_foot = contact_state[:, 3] > tau
    pelvis = contact_state[:, 4] > tau

    pelvis_stationary = _pelvis_stationary_mask(joints, config, T)
    pelvis_object_below = _pelvis_object_below_mask(
        joints, object_mesh, object_positions, object_rotations, config, T,
    )

    support = np.full(T, SUPPORT_BOTH_FEET, dtype=np.int64)

    for t in range(T):
        if pelvis[t] and pelvis_stationary[t] and pelvis_object_below[t]:
            support[t] = SUPPORT_SITTING
        elif (left_hand[t] or right_hand[t]) and not (left_foot[t] and right_foot[t]):
            # Hands active, not both feet grounded → hand support
            support[t] = SUPPORT_HAND
        elif left_foot[t] and right_foot[t]:
            support[t] = SUPPORT_BOTH_FEET
        elif left_foot[t] or right_foot[t]:
            support[t] = SUPPORT_SINGLE_FOOT
        else:
            # Airborne or ambiguous — default to both_feet (most common)
            support[t] = SUPPORT_BOTH_FEET

    # Temporal smoothing — majority (mode), not median: support ids are
    # categorical, so median has no meaning on mixed windows.
    support = _majority_filter(support, size=config.smoothing_window).astype(np.int64)

    return support


def _pelvis_stationary_mask(
    joints: np.ndarray | None,
    config: SupportConfig,
    T: int,
) -> np.ndarray:
    """Per-frame boolean: pelvis horizontal speed below sitting threshold.

    Uses XZ-plane finite differences and a 1-second moving average so a
    single jittery frame does not flip sitting on and off. If ``joints``
    is ``None`` the mask is all-True (legacy behaviour — pelvis contact
    alone triggers sitting).
    """
    if joints is None or T <= 1:
        return np.ones(T, dtype=bool)

    from scipy.ndimage import uniform_filter1d

    pelvis_xz = joints[:, 0, :][:, [0, 2]]                    # (T, 2)
    step = np.zeros(T, dtype=np.float64)
    step[1:] = np.linalg.norm(np.diff(pelvis_xz, axis=0), axis=-1)
    horiz_speed = step * config.fps                            # m/s

    window = max(1, int(round(config.sitting_velocity_window_sec * config.fps)))
    smoothed = uniform_filter1d(horiz_speed, size=window, mode="nearest")
    return smoothed < config.sitting_max_pelvis_horz_speed


def _pelvis_object_below_mask(
    joints: np.ndarray | None,
    object_mesh,
    object_positions: np.ndarray | None,
    object_rotations: np.ndarray | None,
    config: SupportConfig,
    T: int,
) -> np.ndarray:
    """Per-frame boolean: an upward-facing mesh surface sits within a
    thin vertical cylinder below the pelvis.

    Sitting requires a seat-like surface (upward-facing, i.e. normal
    close to +Y) inside a pelvis-sized cylinder (XZ radius
    ``sitting_below_horz_radius``) extending down by
    ``sitting_below_vert_gate``. Backrests, chair legs, and armrests
    have horizontal or downward normals and are filtered out even if
    they happen to intersect the cylinder; they are not sittable.

    This replaces a prior closest-point-direction gate that mis-fired
    on sitting-at-sofa-edge, where the nearest mesh point is on the
    armrest (direction horizontal) but a seat surface still lies
    directly below pelvis.

    Returns all-True when inputs are unavailable (keeps legacy
    behaviour; the velocity gate is the only remaining disambiguator).
    """
    if (
        joints is None
        or object_mesh is None
        or object_positions is None
    ):
        return np.ones(T, dtype=bool)

    import trimesh

    from piano.data.pseudo_labels._object_transform import world_to_object_local

    pelvis_world = joints[:, 0, :]
    pelvis_local = world_to_object_local(
        pelvis_world, object_positions, object_rotations,
    )

    # Sample surface points and keep only those on upward-facing faces.
    # Raw mesh.vertices can be too sparse on low-poly meshes (a cube has
    # only 8 corner vertices), so sampling the surface gives uniform
    # coverage for the proximity test. Upward = face normal within ~45°
    # of +Y in the object-local frame; the filter drops backrests,
    # side faces, and inverted faces.
    n_samples = min(3000, max(500, 4 * len(object_mesh.vertices)))
    surface_pts, face_idx = trimesh.sample.sample_surface(object_mesh, n_samples)
    surface_normals = object_mesh.face_normals[face_idx]
    upward = surface_normals[:, 1] > config.sitting_below_upward_normal_threshold
    seat_pts = surface_pts[upward].astype(np.float32)

    if len(seat_pts) == 0:
        # No upward-facing surface → object cannot support a seated pose.
        return np.zeros(T, dtype=bool)

    # Check per-frame: does any seat-like point fall inside the cylinder
    # below the pelvis? (T, V) broadcasting on a small number of points.
    dx = pelvis_local[:, None, 0] - seat_pts[None, :, 0]
    dz = pelvis_local[:, None, 2] - seat_pts[None, :, 2]
    xz_dist2 = dx * dx + dz * dz
    in_xz = xz_dist2 < config.sitting_below_horz_radius ** 2

    dy = pelvis_local[:, None, 1] - seat_pts[None, :, 1]
    y_below = (dy > 0) & (dy < config.sitting_below_vert_gate)

    return (in_xz & y_below).any(axis=1)
