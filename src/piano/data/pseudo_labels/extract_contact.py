"""Extract contact state pseudo-labels from HOI motion data.

For each frame, computes whether each tracked body part (left_hand, right_hand,
left_foot, right_foot, pelvis) is in contact with the object surface.

Contact is detected from the distance between the body-part *joint* and the
mesh surface, using per-part anatomy-calibrated thresholds:

    SMPL joint centers sit inside the body, not on the skin. A wrist joint is
    ~5-8 cm from the palm surface; an ankle joint is ~7-10 cm above the sole;
    the SMPL pelvis root is ~15-20 cm from the seat surface during sitting.
    A single tight threshold (e.g. 2 cm) measures joint penetration into the
    mesh, which almost never happens, and suppresses nearly all real contact.
    So thresholds are set per body part to reflect joint-to-skin offset.

Velocity gating is off by default. Early runs with strict velocity gating
(v < 0.1 m/s) erased the remaining contact signal during manipulation, and
world-frame speed on a moving object is the wrong quantity anyway. Phase
extraction (which does care about motion vs. rest) is the right place for
the "stable vs. moving" distinction. Can be re-enabled via
``use_velocity_gating=True`` for ablations.

The raw contact signal is temporally smoothed and filtered to remove
single-frame flickers.

Output: soft contact state array of shape ``(T, B)`` where B=5 body parts.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import median_filter

from piano.utils.geometry import points_to_mesh_distance
from piano.utils.smpl_utils import (
    BODY_PART_INDICES,
    BODY_PART_NAMES,
    NUM_BODY_PARTS,
    compute_joint_velocities,
)


# Per-body-part joint-to-contact-surface offsets (meters), used as the
# distance-threshold midpoints in the soft sigmoid. Derived from anatomy
# then verified (and adjusted) against the full-dataset sweep
# (runs/threshold_sweep/2026-04-20_193818/):
#   * left/right_hand: wrist joint sits inside the forearm; palm surface
#     is 5-8 cm out, and when the hand *wraps* a handle / edge / bat
#     grip the wrist ends up 10-15 cm from the mesh. v1-v6 used 0.08 m
#     ("palm just touching edge"), which undercounted every gripping
#     pose. v6 hand seq_reached was only 38-63% across subsets even
#     though these datasets are explicitly hand-object interaction.
#     Sweep shows seq_reached curves elbow at 0.12:
#         chairs  L 61→75% / R 59→73% (at 0.08 → 0.12)
#         imhd    L 63→75% / R 58→69%
#         neural. L 38→47% / R 40→50%
#         omomo   L 58→74% / R 63→79%
#     Above 0.14 gains diminish to < 4 pp and risk "hand near but not
#     touching" FPs. 0.12 is the tuned value. FP risk is tempered by
#     min_contact_duration=3 and median_filter_size=5 — isolated
#     approach frames don't register. See
#     2026-04-22_hand_threshold_bump.md.
#   * left/right_foot: the tracked joint is the ankle (SMPL idx 7/8),
#     ~8-10 cm above the sole. An anatomy-only guess of 0.12 was LOOSE
#     here because our mesh is the OBJECT, not the ground: a foot on the
#     floor next to a chair leg sits ~5-10 cm from the chair mesh without
#     actually contacting it. Sweep showed chairs 0.12 gave 48%
#     seq_reached (false positives) while 0.06 gave 12% (genuine
#     foot-object contact rate expected for chairs). 0.06 is the tuned
#     value.
#   * pelvis: SMPL root is inside the hip; during sitting the ischium is
#     ~15 cm below + buttock flesh ~5 cm, so 0.20 m covers seat contact.
#     Sweep confirmed: chairs 0.20 gives 93% seq_reached, saturating at
#     the elbow of the curve.
DEFAULT_DISTANCE_THRESHOLDS: dict[str, float] = {
    "left_hand":  0.12,
    "right_hand": 0.12,
    "left_foot":  0.06,
    "right_foot": 0.06,
    "pelvis":     0.20,
}


@dataclass(slots=True)
class ContactConfig:
    """Configuration for contact state extraction."""

    distance_thresholds: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_DISTANCE_THRESHOLDS)
    )
    distance_sigma: float = 0.03            # sigmoid transition width (m)
    use_velocity_gating: bool = False       # disabled by default; see module docstring
    velocity_threshold: float = 0.5         # m/s — only used when gating is on
    velocity_sigma: float = 0.2             # m/s
    median_filter_size: int = 5             # temporal median filter window
    min_contact_duration: int = 3           # minimum consecutive frames for valid contact
    fps: float = 30.0


def _soft_sigmoid(x: np.ndarray, threshold: float, sigma: float) -> np.ndarray:
    """Smooth step function: 1 when x < threshold, 0 when x >> threshold.

    Uses ``scipy.special.expit`` (numerically stable logistic) instead of
    the naive ``1 / (1 + exp(...))`` to avoid overflow warnings when the
    argument is large (distances much farther than the threshold).
    """
    from scipy.special import expit
    return expit(-(x - threshold) / sigma)


def extract_contact_state(
    joints: np.ndarray,
    object_mesh: "trimesh.Trimesh",
    object_positions: np.ndarray | None = None,
    object_rotations: np.ndarray | None = None,
    config: ContactConfig | None = None,
) -> np.ndarray:
    """Extract per-frame, per-body-part contact state.

    The object mesh is loaded in its *local frame* (template at origin).
    Joints are in the world frame. To query correct distances we must
    inverse-transform joints into the object's per-frame local frame:

        joint_local[t] = R(obj_rot[t])^T @ (joint_world[t] - obj_pos[t])

    If ``object_positions`` is None, we fall back to treating the object
    as static at the origin — correct only if that's actually the case.

    Parameters
    ----------
    joints : (T, 22, 3) — world-frame SMPL 22-joint positions
    object_mesh : trimesh.Trimesh — object mesh in object-local frame
    object_positions : (T, 3) — per-frame object translation in world frame
    object_rotations : (T, 3) — per-frame object axis-angle rotation
    config : extraction parameters

    Returns
    -------
    contact : (T, 5) — soft contact probability for each body part
    """
    from piano.data.pseudo_labels._object_transform import world_to_object_local

    if config is None:
        config = ContactConfig()

    T = len(joints)
    contact = np.zeros((T, NUM_BODY_PARTS), dtype=np.float32)

    # Velocities are only needed when the optional velocity gate is on
    joint_vel = (
        compute_joint_velocities(joints, fps=config.fps)
        if config.use_velocity_gating else None
    )

    for bp_idx, joint_idx in enumerate(BODY_PART_INDICES):
        bp_name = BODY_PART_NAMES[bp_idx]
        bp_positions_world = joints[:, joint_idx, :]      # (T, 3)

        # Inverse-transform joint positions into the object-local frame
        if object_positions is not None:
            bp_positions_local = world_to_object_local(
                bp_positions_world, object_positions, object_rotations,
            )
        else:
            bp_positions_local = bp_positions_world

        # Distance to object surface (in object-local frame, matching mesh)
        distances, _ = points_to_mesh_distance(bp_positions_local, object_mesh)

        # Soft contact: per-part anatomy-calibrated distance threshold
        threshold = config.distance_thresholds.get(
            bp_name, DEFAULT_DISTANCE_THRESHOLDS[bp_name]
        )
        dist_score = _soft_sigmoid(distances, threshold, config.distance_sigma)

        if joint_vel is not None:
            bp_velocities = joint_vel[:, joint_idx, :]
            bp_speed = np.linalg.norm(bp_velocities, axis=-1)
            vel_score = _soft_sigmoid(
                bp_speed, config.velocity_threshold, config.velocity_sigma,
            )
            contact[:, bp_idx] = dist_score * vel_score
        else:
            contact[:, bp_idx] = dist_score

    # Temporal smoothing
    for bp_idx in range(NUM_BODY_PARTS):
        contact[:, bp_idx] = median_filter(contact[:, bp_idx], size=config.median_filter_size)

    # Remove short contact events (below min_contact_duration)
    contact = _filter_short_contacts(contact, config.min_contact_duration)

    return contact


def _filter_short_contacts(
    contact: np.ndarray,
    min_duration: int,
    threshold: float = 0.5,
) -> np.ndarray:
    """Zero out contact segments shorter than *min_duration* frames.

    Operates on each body part independently. Only suppresses
    segments where the mean soft contact exceeds *threshold*.
    """
    result = contact.copy()
    T, B = contact.shape

    for bp in range(B):
        binary = contact[:, bp] > threshold
        # Find contiguous True segments
        changes = np.diff(binary.astype(np.int8), prepend=0, append=0)
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]

        for s, e in zip(starts, ends):
            if (e - s) < min_duration:
                result[s:e, bp] = 0.0

    return result
