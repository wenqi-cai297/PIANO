"""Extract contact state pseudo-labels from HOI motion data.

For each frame, computes whether each tracked body part (left_hand, right_hand,
left_foot, right_foot, pelvis) is in contact with the object surface.

Two orthogonal contact signals are OR-combined:

    1. **Distance signal**: per-part anatomy-calibrated threshold on the
       joint-to-mesh distance in the object-local frame. Catches static
       contact (holding a stationary cup, sitting, etc.). SMPL joint
       centers sit inside the body, not on the skin — wrist ~5-8 cm from
       palm surface, foot joint ~4-5 cm above sole, pelvis root ~15-20 cm
       from seat surface during sitting. Thresholds reflect the joint-
       to-skin offset.

    2. **Kinematic coupling signal** (v9, 2026-04-24): per-part soft score
       that fires when the body part's position in the *object-local
       frame* is stable over a ~0.5 s window AND the object is translating
       or rotating in the world. This is the standard rigid-coupling
       test — a hand wrapping a bat / carrying a case / holding a flower
       in a walking subject is geometrically "attached" to the object
       even if the wrist joint is 18-22 cm from the mesh surface (too far
       for the distance threshold). The v8 cleaning pass showed this is
       the dominant failure mode on neuraldome (large wrap-grip objects:
       cases, flowers, boxes, rackets — 500+ dropped clips of 624 total
       drops).

       The "object must be moving" gate is essential: a stationary scene
       trivially has every body part stationary in the object frame, so
       without it the signal would fire on everything. See
       ``analyses/2026-04-24_v9_kin_coupling.md``.

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

# Index of pelvis in BODY_PART_NAMES (= 4 in current layout); exported
# for v19 directional gating in run_all.py:process_sequence.
_PELVIS_BP_IDX: int = BODY_PART_NAMES.index("pelvis")


# Per-body-part joint-to-contact-surface offsets (meters), used as the
# distance-threshold midpoints in the soft sigmoid. Derived from anatomy
# then verified (and adjusted) against the full-dataset sweep
# (runs/threshold_sweep/2026-04-20_193818/):
#
#   * left/right_hand: wrist joint sits inside the forearm; palm surface
#     is 5-8 cm out. v1-v6 used 0.08 (too tight); v7 bumped to 0.12 at
#     sweep elbow. v8 pushed to 0.16 to catch wrap-grip cases where wrist
#     is 10-15 cm from mesh — but data showed (a) v8 0.16 still missed
#     the 18-22 cm wrap-grip cases (neuraldome 45% left-hand
#     seq_without_contact) and (b) 0.16 was at the edge of "hand reaching
#     toward object" FPs per pipeline doc.
#     v9 rolls back to 0.12 and delegates the wrap-grip class to the
#     **kinematic coupling signal** (see ContactConfig). Wrap-grip is
#     always rigid coupling (hand moves with object), so kin signal
#     handles it; 0.12 keeps the distance signal tight and avoids FP.
#     See ``analyses/2026-04-24_v9_kin_coupling.md``.
#
#   * left/right_foot: tracked joint is now SMPL idx 10/11 (mid-foot),
#     NOT 7/8 (ankle). v1-v8 used ankle which is 8-10 cm above sole;
#     threshold 0.06 = "ankle inside mesh" which essentially never
#     happens. Result: ~99% feet-seq-without-contact on imhd/neural/
#     omomo, and every "Kick the X" / "scoot with foot" clip on omomo
#     dropped by cleaning (394/398 of omomo drops).
#     Foot joint at mid-foot sits ~4-5 cm above sole; threshold 0.06
#     now means "sole within 1-2 cm of mesh", a genuine foot-on-object
#     contact test. Threshold stays at 0.06 — the joint swap closes
#     the 4-5 cm gap implicitly.
#
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

    # --- v19 directional gate for pelvis (2026-05-09) ---
    # The default v18 pelvis contact uses pure 3D Euclidean distance,
    # which gives false positives during dynamic motions where an object
    # passes near pelvis (e.g. bat handle 15-20 cm to the side during
    # a swing). Sitting / lying are the only physical situations where
    # pelvis contact is real, and they all share the same geometric
    # signature: an upward-facing surface is below the pelvis.
    #
    # When ``use_directional_pelvis_gate`` is True, the pelvis contact
    # score (both distance- and kinematic-coupling-based) is multiplied
    # by ``_pelvis_object_below_mask`` from extract_support.py — the
    # exact same cylinder + upward-normal helper used to disambiguate
    # the "sitting" support label. Reusing the helper means v19
    # automatically inherits sitting's tested gate parameters and its
    # per-mesh seat-points cache.
    use_directional_pelvis_gate: bool = False  # opt-in (False = v18 behavior)
    pelvis_below_horz_radius: float = 0.15     # cylinder XZ radius (m)
    pelvis_below_vert_gate: float = 0.30       # cylinder height below pelvis (m)
    pelvis_below_upward_normal_threshold: float = 0.5
                                                # face-normal · up_axis threshold
                                                # (0.5 ≈ 60° from vertical, matches
                                                #  extract_support.py:109 default)
    # Temporal median filter window, in frames. v1-v7 used 5 (0.25s at
    # 20fps). v7 vis surfaced heavy frame-to-frame contact flicker that
    # drove phase transitions every 10 frames on some imhd clips. v8
    # widens the smoothing window to 7 (0.35s) so that single borderline
    # frames are out-voted by neighbours. Cost: phase boundaries drift
    # by at most 2 frames (~0.1s) — well below what the HMM refinement
    # would capture anyway.
    median_filter_size: int = 7
    min_contact_duration: int = 3           # minimum consecutive frames for valid contact
    fps: float = 30.0

    # --- Kinematic coupling signal (v9) ---
    # Detects contact via rigid attachment to a moving object. Soft-ORs
    # with the distance signal above. See module docstring for rationale.
    use_kinematic_coupling: bool = True
    kin_window_sec: float = 0.5             # rolling window for local-frame std (s)
    kin_local_sigma: float = 0.03           # m — "rigid" if per-axis local std < this
    kin_local_transition: float = 0.015     # m — softness of the "rigid" sigmoid
    kin_world_eps: float = 0.15             # m/s — object-world speed gate midpoint
    kin_world_sigma: float = 0.04           # m/s — softness of world gate. Tight so
                                            # a static object (speed ≈ 0) gives
                                            # world_score ≈ 0.02 (cleanly off) and
                                            # moving object (>= 0.3 m/s) saturates.
    kin_radius_proxy: float = 0.3           # m — converts ang_vel (rad/s) to surface speed


def _soft_sigmoid(x: np.ndarray, threshold: float, sigma: float) -> np.ndarray:
    """Smooth step function: 1 when x < threshold, 0 when x >> threshold.

    Uses ``scipy.special.expit`` (numerically stable logistic) instead of
    the naive ``1 / (1 + exp(...))`` to avoid overflow warnings when the
    argument is large (distances much farther than the threshold).
    """
    from scipy.special import expit
    return expit(-(x - threshold) / sigma)


def _kinematic_contact_score(
    part_world: np.ndarray,
    object_positions: np.ndarray | None,
    object_rotations: np.ndarray | None,
    config: ContactConfig,
) -> np.ndarray:
    """Per-frame soft score for rigid-coupling-based contact detection.

    High when:
        (a) the body part's position in the object-local frame has low
            variance over a ~``kin_window_sec`` window (= the part is
            rigidly attached to the object over that window), AND
        (b) the object is actually translating or rotating in world
            (otherwise "stationary in object frame" is trivially true
            for any body part and the signal is meaningless).

    Returns a zero array when inputs are insufficient (no object pose,
    T < 2, or kinematic gating disabled by config).

    Parameters
    ----------
    part_world : (T, 3) — body part world position per frame.
    object_positions : (T, 3) or None — per-frame object translation.
    object_rotations : (T, 3) or None — per-frame axis-angle rotation.
        If None, rotation is treated as identity and only translation
        contributes to object motion + local transformation.
    config : contact extraction parameters (kin_* fields).

    Returns
    -------
    score : (T,) float32 in [0, 1].
    """
    from scipy.ndimage import uniform_filter1d
    from scipy.special import expit

    from piano.data.pseudo_labels._object_transform import world_to_object_local

    T = len(part_world)
    if (
        not config.use_kinematic_coupling
        or object_positions is None
        or T < 2
    ):
        return np.zeros(T, dtype=np.float32)

    # --- Stability in object-local frame ---
    part_local = world_to_object_local(
        part_world, object_positions, object_rotations,
    )                                                     # (T, 3)

    window = max(3, int(round(config.kin_window_sec * config.fps)))
    # Rolling variance per xyz axis via the identity Var[X] = E[X^2] - E[X]^2,
    # both taken over the ``window``-frame neighbourhood. DO NOT filter
    # ``(x - rolling_mean)^2`` directly — that measures deviation from
    # the LOCAL rolling mean (a low-pass signal that tracks x), which is
    # near-zero for any slowly-varying signal including a wide orbit.
    # We want "does x stay near some constant over the window" = true
    # window variance.
    mean_x = uniform_filter1d(part_local, size=window, axis=0, mode="nearest")
    mean_x_sq = uniform_filter1d(part_local ** 2, size=window, axis=0, mode="nearest")
    rolling_var = np.maximum(mean_x_sq - mean_x ** 2, 0.0)   # clip numerical neg
    local_std = np.sqrt(rolling_var + 1e-12).max(axis=-1)     # (T,)  worst-axis std

    # Low local std => rigid coupling. Sigmoid centred at ``kin_local_sigma``.
    local_score = expit(
        (config.kin_local_sigma - local_std) / max(config.kin_local_transition, 1e-6)
    )                                                     # (T,)

    # --- Object world speed (translational + angular surface proxy) ---
    trans_vel = np.zeros(T)
    trans_vel[1:] = np.linalg.norm(
        np.diff(object_positions, axis=0), axis=-1
    ) * config.fps

    ang_vel = np.zeros(T)
    if object_rotations is not None:
        ang_vel[1:] = np.linalg.norm(
            np.diff(object_rotations, axis=0), axis=-1
        ) * config.fps

    obj_speed = trans_vel + config.kin_radius_proxy * ang_vel   # (T,)  m/s proxy

    world_score = expit(
        (obj_speed - config.kin_world_eps) / max(config.kin_world_sigma, 1e-6)
    )                                                     # (T,)

    return (local_score * world_score).astype(np.float32)


def extract_contact_state(
    joints: np.ndarray,
    object_mesh: "trimesh.Trimesh",
    object_positions: np.ndarray | None = None,
    object_rotations: np.ndarray | None = None,
    config: ContactConfig | None = None,
    object_id: str | None = None,
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

    # v19 directional gate is applied at the process_sequence level
    # (after extract_contact_state's mesh-distance signal is max-combined
    # with the official semantic-marker contact prior), so it filters
    # both signal sources at once. See run_all.py:process_sequence.

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

        # v9 kinematic coupling: OR-combine with the rigid-coupling
        # signal. Catches wrap-grip cases where the wrist joint is
        # 18-22 cm from the mesh surface (too far for the distance
        # threshold) but the hand is clearly "attached" to a moving
        # object. See module docstring.
        kin_score = _kinematic_contact_score(
            bp_positions_world, object_positions, object_rotations, config,
        )
        score = np.maximum(dist_score, kin_score)


        if joint_vel is not None:
            bp_velocities = joint_vel[:, joint_idx, :]
            bp_speed = np.linalg.norm(bp_velocities, axis=-1)
            vel_score = _soft_sigmoid(
                bp_speed, config.velocity_threshold, config.velocity_sigma,
            )
            contact[:, bp_idx] = score * vel_score
        else:
            contact[:, bp_idx] = score

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
