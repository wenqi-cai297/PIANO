"""Extract support state pseudo-labels from HOI motion data.

Classifies each frame into one of four body support configurations
based on foot / pelvis / hand contact patterns.

Support states:
    0 = both_feet    — both feet on ground (default when no other signal fires)
    1 = single_foot  — only one foot on the object (rare: step stool pose)
    2 = sitting      — pelvis contacts object, body is stationary, AND
                       the object is below the pelvis (geometric test)
    3 = hand_support — body supported by hand (leaning on table etc.);
                       NOT "hand carrying an object"

``sitting`` uses two disambiguating conditions on top of pelvis contact:

1. **Pelvis stationary** (XZ-plane speed < 0.15 m/s, 1 s moving average).
   A seated person is stationary; someone pushing a chair walks or
   shuffles at 0.2-0.5 m/s. Rejects neuraldome `subject01_chair_0` style
   false positives where the user stands behind the chair and pushes it.

2. **Upward-facing mesh surface sits within a cylinder below the pelvis**.
   The upward direction defaults to +Y, with an explicit whitelist for
   known Z-up objects (`bigsofa`, `smallsofa`; see
   ``OBJECT_UP_AXIS_OVERRIDES``). The gate opens if any seat-candidate
   face (normal within ~45° of the up axis) falls inside a cylinder
   (radius 0.15 m, height 0.30 m) extending opposite to the up
   direction from the pelvis. Backrests / legs / armrests have
   off-axis normals and get filtered out even when they intersect the
   cylinder.

``hand_support`` is narrowly defined as body-on-hand (leaning, bracing),
NOT hand-on-object-carrying. Requires a conjunction of:

1. **Hand contact with object** (one or both).
2. **Pelvis stationary** — walking while holding an object is foot-
   supported, not hand-supported.
3. **phase == stable-contact** — object is not moving relative to the
   scene. Manipulation / approach / release frames mean the hand is
   applying force to the object, not the other way round.

Without gates 2+3, the v1-v7 pipeline mislabelled every "carry object
while walking" sequence as hand_support: omomo clothesstand / largebox
/ whitechair clips hit 61-89% hand_support, imhd bat-swing / bat-hit
clips hit 42-86%. The gates eliminate the FP class while preserving
genuine leaning poses (static body + static object + hand contact).

If ``joints``, ``phase``, or ``object_mesh`` is unavailable, the
corresponding gate defaults to open, preserving legacy behaviour at
the cost of more FP. In the real extraction pipeline all three are
always provided.

Output: integer support array of shape ``(T,)``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from piano.data.pseudo_labels.extract_phase import PHASE_STABLE_CONTACT


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
    object_id: str | None = None,
    phase: np.ndarray | None = None,
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
        object_id=object_id,
    )

    # Phase gate for hand_support: only stable-contact frames qualify
    # (carrying / manipulating object is NOT body supported by hand).
    # If phase is unavailable, default the gate open for back-compat.
    if phase is not None:
        phase_stable = (phase == PHASE_STABLE_CONTACT)
    else:
        phase_stable = np.ones(T, dtype=bool)

    support = np.full(T, SUPPORT_BOTH_FEET, dtype=np.int64)

    for t in range(T):
        if pelvis[t] and pelvis_stationary[t] and pelvis_object_below[t]:
            support[t] = SUPPORT_SITTING
        elif (
            (left_hand[t] or right_hand[t])
            and pelvis_stationary[t]
            and phase_stable[t]
        ):
            # Hand on object + body static + object static → plausibly
            # body-on-hand support (leaning). Walking while holding
            # (pelvis not stationary) and manipulating (object moving →
            # phase != stable-contact) are filtered to both_feet.
            support[t] = SUPPORT_HAND
        elif left_foot[t] and right_foot[t]:
            # Rare: both feet on the object (step stool / standing on chair).
            support[t] = SUPPORT_BOTH_FEET
        elif left_foot[t] or right_foot[t]:
            support[t] = SUPPORT_SINGLE_FOOT
        else:
            # Default: person standing on floor. We have no direct
            # floor-contact signal so this covers walking / standing /
            # carrying objects / airborne — all collapsed to both_feet.
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


# Object-local up axis per InterAct mesh. Default is +Y (the prevalent
# authoring convention across chairs / imhd / omomo / most neuraldome
# meshes); a small whitelist overrides this for known Z-up objects.
#
# The previous face-area-argmax heuristic was abandoned after the v5
# probe (``runs/checks/up_axis_probe/2026-04-22_101850/probe.json``)
# showed it picks a non-Y axis on 21/60 chairs and 8/10 imhd objects.
# That drove v5 chairs sitting from 49.6% → 39.5% (-10 pp) and gave
# imhd a 9.5 pp false-positive sitting (impossible on bats / brooms /
# dumbbells). Chair meshes rarely have face-area dominance on the
# true +Y: chair 116 has +Y 0.79 vs +Z 0.69 — ratio 1.15, well below
# any usable dominance threshold. Hardcoded +Y with named exceptions
# is the minimum-regression fix.
OBJECT_UP_AXIS_OVERRIDES: dict[str, str] = {
    # Confirmed Z-up by the 2026-04-22 probe and by the server-side
    # face-normal dump that drove ``edf2bb3``:
    #   bigsofa face normals +X 41234 / +Y 22833 / +Z 48901 (dominant)
    "bigsofa": "+Z",
    # smallsofa extents [0.89, 0.86, 0.71]: Z is the short/vertical
    # axis (height), consistent with Z-up authoring. Face-area argmax
    # mis-picked +X (big armrest faces), which is why we can't trust
    # auto-detect here either.
    "smallsofa": "+Z",
}

_AXIS_TO_INDEX: dict[str, int] = {"+X": 0, "+Y": 1, "+Z": 2}


def _detect_mesh_up_axis(
    mesh,
    object_id: str | None = None,
    threshold: float = 0.7,  # noqa: ARG001  kept for call-site compat
) -> np.ndarray:
    """Return the object-local +axis that should be treated as "up".

    Defaults to +Y. ``object_id`` is looked up in
    ``OBJECT_UP_AXIS_OVERRIDES`` to opt specific meshes into a
    non-default axis. The ``mesh`` and ``threshold`` arguments are kept
    so existing call sites (including the ``probe_mesh_up_axis`` tool)
    don't need to change, but neither is used anymore — face-area
    auto-detection proved too unreliable on InterAct's mixed
    authoring conventions.
    """
    axis_name = OBJECT_UP_AXIS_OVERRIDES.get(object_id, "+Y") if object_id else "+Y"
    up = np.zeros(3, dtype=np.float32)
    up[_AXIS_TO_INDEX[axis_name]] = 1.0
    return up


def _pelvis_object_below_mask(
    joints: np.ndarray | None,
    object_mesh,
    object_positions: np.ndarray | None,
    object_rotations: np.ndarray | None,
    config: SupportConfig,
    T: int,
    object_id: str | None = None,
) -> np.ndarray:
    """Per-frame boolean: a seat-like mesh surface sits inside a
    cylinder extending below the pelvis along the mesh's up axis.

    The up axis defaults to +Y, with explicit overrides for known
    Z-up meshes via ``OBJECT_UP_AXIS_OVERRIDES`` (see
    ``_detect_mesh_up_axis``). "Below" is measured along that axis,
    and the cylinder radius is the perpendicular-to-up distance.
    Filtering by normal alignment with the up axis drops backrests /
    legs / armrests whose normals point sideways or downward.

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

    up_local = _detect_mesh_up_axis(
        object_mesh,
        object_id=object_id,
        threshold=config.sitting_below_upward_normal_threshold,
    )

    # Sample surface points + face normals. Low-poly meshes (8-vertex
    # primitive boxes) don't give uniform coverage via mesh.vertices,
    # so sample_surface gives the cylinder test a fair density.
    n_samples = min(3000, max(500, 4 * len(object_mesh.vertices)))
    surface_pts, face_idx = trimesh.sample.sample_surface(object_mesh, n_samples)
    surface_normals = object_mesh.face_normals[face_idx]

    # "Seat-like" = normal aligned with the detected up axis.
    alignment = surface_normals @ up_local
    upward = alignment > config.sitting_below_upward_normal_threshold
    seat_pts = surface_pts[upward].astype(np.float32)

    if len(seat_pts) == 0:
        # No upward-facing surface in this axis → cannot support a seated pose.
        return np.zeros(T, dtype=bool)

    # Cylinder axis = up_local. Decompose offsets (seat − pelvis) into
    # axial (along up) and radial (perpendicular) components.
    offsets = seat_pts[None, :, :] - pelvis_local[:, None, :]  # (T, N, 3)
    axial = (offsets * up_local).sum(axis=-1)                  # (T, N)
    radial_sq = (offsets ** 2).sum(axis=-1) - axial ** 2       # (T, N)

    in_radius = radial_sq < config.sitting_below_horz_radius ** 2
    below = (axial < 0) & (axial > -config.sitting_below_vert_gate)

    return (in_radius & below).any(axis=1)
