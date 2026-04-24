"""Regression tests for the four 2026-04-21 pseudo-label P0 fixes.

Each test targets one design bug found during the v2 extraction review:

1. ``test_phase_sitting_enters_stable_contact`` — any-body-part contact
   replaces hand-only contact in extract_phase. Before the fix, chair
   sitting sequences stayed stuck in approach because hand_contact was 0.

2. ``test_phase_rotation_only_enters_manipulation`` — object motion now
   combines translational + angular velocity. Before the fix, an
   in-place bat swing had obj_vel = 0 and was labelled stable-contact.

3. ``test_hmm_state_ids_preserve_phase_semantics`` — HMM parameters are
   frozen during fit (params=""), so state id k remains aligned with
   phase constant k. Before the fix, EM could permute state ids.

4. ``test_support_majority_filter_rejects_ordinal_median`` — support
   uses majority (mode) smoothing instead of median. Before the fix,
   median on categorical ids could invent semantically meaningless
   in-between values.
"""
from __future__ import annotations

import numpy as np

from piano.data.pseudo_labels.extract_contact import (
    ContactConfig,
    _kinematic_contact_score,
    extract_contact_state,
)
from piano.data.pseudo_labels.extract_phase import (
    PHASE_MANIPULATION,
    PHASE_STABLE_CONTACT,
    PhaseConfig,
    extract_interaction_phase,
)
from piano.data.pseudo_labels.extract_support import (
    SUPPORT_BOTH_FEET,
    SUPPORT_HAND,
    SUPPORT_SINGLE_FOOT,
    SUPPORT_SITTING,
    SupportConfig,
    _majority_filter,
    extract_support_state,
)
from piano.data.pseudo_labels.refine_phase_hmm import (
    HMMConfig,
    build_phase_features,
    refine_phases_hmm,
)
from piano.data.pseudo_labels.extract_phase import PHASE_APPROACH
from piano.data.pseudo_labels.extract_target import TargetConfig
from piano.utils.geometry import soft_patch_assignment


# ---------------------------------------------------------------------------
# Phase fixes
# ---------------------------------------------------------------------------

def test_phase_sitting_enters_stable_contact() -> None:
    """Pelvis-only contact (no hand contact), object static → stable-contact.

    Reproduces the chair-sitting scenario. Pre-fix this stayed in approach
    because is_contact was hand-only.
    """
    T = 30
    joints = np.zeros((T, 22, 3), dtype=np.float32)
    # Pelvis near origin, hands far away — mirrors a seated pose with hands
    # resting in the lap far from the chair surface.
    joints[:, 0] = [0.0, 0.0, 0.0]                 # pelvis
    joints[:, 20] = [2.0, 1.0, 2.0]                # left wrist
    joints[:, 21] = [2.0, 1.0, 2.0]                # right wrist

    contact = np.zeros((T, 5), dtype=np.float32)
    contact[:, 4] = 1.0                            # pelvis contact only

    object_positions = np.zeros((T, 3), dtype=np.float32)   # static at origin

    cfg = PhaseConfig(fps=20.0, median_filter_size=3, contact_threshold=0.5)
    phase = extract_interaction_phase(
        joints, contact, object_positions, None, cfg,
    )

    # Majority of the sequence must have reached stable-contact. Edge
    # frames can be consumed by the release_window, that's fine.
    assert (phase == PHASE_STABLE_CONTACT).sum() > T * 0.5, (
        f"expected mostly stable-contact, got histogram "
        f"{np.bincount(phase, minlength=5).tolist()}"
    )


def test_phase_rotation_only_enters_manipulation() -> None:
    """Contact active + object rotating (translation 0) → manipulation.

    Reproduces a bat-swing scenario: hands grip the bat at origin, bat
    center doesn't translate but rotates in place. Pre-fix this was
    stable-contact because obj_vel only measured translation.
    """
    T = 30
    joints = np.zeros((T, 22, 3), dtype=np.float32)
    # Both hands at object centre (gripping).
    joints[:, 20] = [0.0, 0.0, 0.0]
    joints[:, 21] = [0.0, 0.0, 0.0]

    contact = np.zeros((T, 5), dtype=np.float32)
    contact[:, 0] = 1.0                            # left_hand contact
    contact[:, 1] = 1.0                            # right_hand contact

    object_positions = np.zeros((T, 3), dtype=np.float32)    # no translation
    # Axis-angle rotating about z at ~2 rad/s — well above rotational_eps.
    object_rotations = np.zeros((T, 3), dtype=np.float32)
    object_rotations[:, 2] = np.arange(T) * (2.0 / 20.0)

    cfg = PhaseConfig(fps=20.0, median_filter_size=3, contact_threshold=0.5)
    phase = extract_interaction_phase(
        joints, contact, object_positions, object_rotations, cfg,
    )

    # Most contact frames must be manipulation (not stable-contact).
    assert (phase == PHASE_MANIPULATION).sum() > T * 0.5, (
        f"expected mostly manipulation, got histogram "
        f"{np.bincount(phase, minlength=5).tolist()}"
    )


def test_phase_rotation_only_is_stable_without_rotation_signal() -> None:
    """Same rotating bat, but caller didn't pass object_rotations — the
    signal is unavailable, so we must fall back to stable-contact rather
    than silently mislabel.
    """
    T = 30
    joints = np.zeros((T, 22, 3), dtype=np.float32)
    joints[:, 20] = 0.0
    joints[:, 21] = 0.0

    contact = np.zeros((T, 5), dtype=np.float32)
    contact[:, 0] = 1.0
    contact[:, 1] = 1.0

    object_positions = np.zeros((T, 3), dtype=np.float32)    # static trans

    cfg = PhaseConfig(fps=20.0, median_filter_size=3, contact_threshold=0.5)
    phase = extract_interaction_phase(
        joints, contact, object_positions, None, cfg,
    )

    # Without rotation signal, translation-only velocity = 0, so
    # stable-contact is the honest label.
    assert (phase == PHASE_STABLE_CONTACT).sum() > T * 0.5


# ---------------------------------------------------------------------------
# HMM fix: state id must stay bound to phase constant
# ---------------------------------------------------------------------------

def test_hmm_state_ids_preserve_phase_semantics() -> None:
    """With ``params=""`` the HMM cannot drift means/covars; state k
    stays phase k. We construct three clearly-separable regimes and
    verify the refined labels keep the same id assignment as the input.
    """
    # 3 regimes, each with a distinctive feature signature:
    # [hand_obj_dist, hand_contact, trans_vel, ang_vel]
    approach = np.tile([1.5, 0.0, 0.0, 0.0], (10, 1))
    stable = np.tile([0.05, 1.0, 0.0, 0.0], (10, 1))
    manipulation = np.tile([0.05, 1.0, 0.5, 1.2], (10, 1))
    features = np.concatenate([approach, stable, manipulation], axis=0).astype(np.float64)
    # Add tiny jitter to avoid zero-variance covars during HMM init.
    rng = np.random.default_rng(0)
    features = features + rng.normal(scale=1e-3, size=features.shape)

    initial_phases = np.concatenate([
        np.zeros(10, dtype=np.int64),              # approach = PHASE_APPROACH = 0
        np.full(10, PHASE_STABLE_CONTACT),
        np.full(10, PHASE_MANIPULATION),
    ])

    refined = refine_phases_hmm(features, initial_phases, HMMConfig(n_iter=1))

    # After Viterbi decode with frozen params, most frames should keep
    # their initial id. A few boundary frames can flip without harming
    # semantics.
    agreement = (refined == initial_phases).mean()
    assert agreement >= 0.8, (
        f"state ids drifted: refined={refined.tolist()}, initial={initial_phases.tolist()}"
    )


def test_build_phase_features_shape_with_rotation() -> None:
    """build_phase_features now produces 4-dim features when object_rotations
    is supplied — the HMM sees angular velocity as its 4th component."""
    T = 20
    joints = np.zeros((T, 22, 3), dtype=np.float32)
    contact = np.zeros((T, 5), dtype=np.float32)
    obj_pos = np.zeros((T, 3), dtype=np.float32)
    obj_rot = np.zeros((T, 3), dtype=np.float32)
    obj_rot[:, 2] = np.linspace(0, 1.0, T)

    feats = build_phase_features(joints, contact, obj_pos, obj_rot, fps=20.0)
    assert feats.shape == (T, 4)
    # Angular velocity column should be non-zero (rotation changes).
    assert feats[1:, 3].max() > 0.0


# ---------------------------------------------------------------------------
# Support fix: majority filter over categorical ids
# ---------------------------------------------------------------------------

def test_support_majority_filter_no_ordinal_artifacts() -> None:
    """A window ``[single_foot, sitting, hand_support]`` has no well-
    defined median — the three ids are not ordered. The majority filter
    returns the most-frequent value, which for any single window is the
    unique value (ties broken by lower id via np.bincount+argmax).
    """
    labels = np.array([
        SUPPORT_SINGLE_FOOT, SUPPORT_SITTING, 3,        # hand_support = 3
        SUPPORT_SITTING, SUPPORT_SITTING, SUPPORT_SITTING,
        SUPPORT_BOTH_FEET, SUPPORT_BOTH_FEET, SUPPORT_SINGLE_FOOT,
    ], dtype=np.int64)

    smoothed = _majority_filter(labels, size=3)

    # No new class invented.
    assert set(smoothed.tolist()) <= set(labels.tolist())
    # The sitting run is preserved.
    assert (smoothed[3:6] == SUPPORT_SITTING).all()


def test_support_extraction_sitting_sequence() -> None:
    """Full extract_support_state on a clean sitting sequence produces
    all-sitting labels — the new smoother doesn't corrupt a pure run.
    """
    T = 20
    contact = np.zeros((T, 5), dtype=np.float32)
    contact[:, 4] = 1.0   # pelvis contact → sitting
    support = extract_support_state(contact, config=SupportConfig(smoothing_window=3))
    assert (support == SUPPORT_SITTING).all()


def test_support_push_object_not_classified_as_sitting() -> None:
    """A person pushing or dragging a chair/sofa walks in the XZ plane
    while their pelvis joint is often within 20cm of the object mesh
    (standing right next to it). Pure pelvis-contact → sitting put
    bigsofa_330 (push) and chair_0 (pull) under the sitting label in v2.
    After the pelvis-velocity gate, these frames must not be sitting.
    Under the v8 hand_support tightening, they also cannot be
    hand_support (pelvis is moving) — they collapse to both_feet.
    """
    T = 60
    # Pelvis walks 3 m over 3 seconds → 1 m/s horizontal speed (>> 0.15)
    joints = np.zeros((T, 22, 3), dtype=np.float32)
    joints[:, 0, 0] = np.linspace(0.0, 3.0, T)  # x moves
    joints[:, 0, 1] = 1.0                        # y=1m (typical pelvis height)

    contact = np.zeros((T, 5), dtype=np.float32)
    contact[:, 4] = 1.0   # pelvis contact (standing close to object)
    contact[:, 0] = 1.0   # left hand contact (pushing)
    contact[:, 1] = 1.0   # right hand contact (pushing)

    cfg = SupportConfig(fps=20.0, smoothing_window=3)
    support = extract_support_state(contact, joints=joints, config=cfg)

    # Sitting must not dominate — the whole sequence is walking with hands
    # on the object, which collapses to both_feet under the v8 logic.
    sitting_frac = (support == SUPPORT_SITTING).sum() / T
    assert sitting_frac < 0.2, (
        f"push/drag sequence still mostly sitting ({sitting_frac:.2%}) — "
        f"pelvis-velocity gate not working"
    )


def test_support_stationary_sitting_still_classified_as_sitting() -> None:
    """Sanity: actual sitting (pelvis contact + stationary body) must
    stay sitting after the velocity gate is added."""
    T = 30
    joints = np.zeros((T, 22, 3), dtype=np.float32)
    joints[:, 0, 1] = 0.5     # sit height — y axis is vertical

    contact = np.zeros((T, 5), dtype=np.float32)
    contact[:, 4] = 1.0       # pelvis only

    cfg = SupportConfig(fps=20.0, smoothing_window=3)
    support = extract_support_state(contact, joints=joints, config=cfg)

    # Stationary pelvis → gate open → sitting dominates.
    assert (support == SUPPORT_SITTING).sum() > T * 0.9


def test_support_rejects_sitting_when_pelvis_far_above_seat() -> None:
    """Pelvis well above a seat surface (more than
    ``sitting_below_vert_gate``): no seat point fits inside the cylinder
    extending only ~0.30 m below the pelvis. Gate closes.

    Replaces an earlier "pelvis beside a tall box" test whose synthetic
    mesh had no clear up-axis — a standalone tall Box is neither a
    chair nor a backrest, and with per-mesh up-axis auto-detection the
    test became ambiguous.
    """
    import trimesh

    # Wide, flat "stool top" slab: dominant face area is ±Y (1 m² each),
    # so auto-detect picks +Y as up.
    mesh = trimesh.primitives.Box(extents=[1.0, 0.1, 1.0])   # top at y=0.05

    T = 30
    joints = np.zeros((T, 22, 3), dtype=np.float32)
    joints[:, 0] = [0.0, 0.55, 0.0]   # 0.50 m above seat top (> 0.30 gate)

    contact = np.zeros((T, 5), dtype=np.float32)
    contact[:, 4] = 1.0

    object_positions = np.zeros((T, 3), dtype=np.float32)
    object_rotations = np.zeros((T, 3), dtype=np.float32)

    cfg = SupportConfig(fps=20.0, smoothing_window=3)
    support = extract_support_state(
        contact,
        joints=joints,
        object_mesh=mesh,
        object_positions=object_positions,
        object_rotations=object_rotations,
        config=cfg,
    )

    sitting_frac = (support == SUPPORT_SITTING).sum() / T
    assert sitting_frac < 0.2, (
        f"pelvis far above seat still classified as sitting ({sitting_frac:.2%}) — "
        f"vertical gate not working"
    )


def test_support_allows_sitting_when_object_below_pelvis() -> None:
    """Pelvis just above a wide flat seat → auto-detect picks +Y as up,
    the seat surface sits inside the cylinder below pelvis, gate opens.
    """
    import trimesh

    # Flat slab with Y-dominant area: top/bottom faces 1 m² each,
    # side faces 0.1 m² each → +Y auto-detected as up.
    mesh = trimesh.primitives.Box(extents=[1.0, 0.1, 1.0])   # top at y=0.05

    T = 30
    joints = np.zeros((T, 22, 3), dtype=np.float32)
    joints[:, 0] = [0.0, 0.15, 0.0]   # 0.10 m above top face

    contact = np.zeros((T, 5), dtype=np.float32)
    contact[:, 4] = 1.0

    object_positions = np.zeros((T, 3), dtype=np.float32)
    object_rotations = np.zeros((T, 3), dtype=np.float32)

    cfg = SupportConfig(fps=20.0, smoothing_window=3)
    support = extract_support_state(
        contact,
        joints=joints,
        object_mesh=mesh,
        object_positions=object_positions,
        object_rotations=object_rotations,
        config=cfg,
    )

    assert (support == SUPPORT_SITTING).sum() > T * 0.8


def test_support_up_axis_override_unlocks_z_up_mesh() -> None:
    """Regression for neuraldome/bigsofa: mesh authored Z-up in its own
    local frame. v4 hard-coded normal.Y > 0.7 and filtered every seat
    face out, rejecting all sitting frames. v5's face-area argmax
    auto-detect picked +Z for bigsofa but also mis-picked non-Y axes
    for 21/60 chairs and 8/10 imhd objects (see
    ``runs/checks/up_axis_probe/2026-04-22_101850/probe.json``).
    The fix is a small whitelist: default +Y, override per object_id.

    This test verifies the override branch — passing
    ``object_id="bigsofa"`` unlocks a Z-up slab even though the
    default would have said +Y.
    """
    import trimesh

    mesh = trimesh.primitives.Box(extents=[1.0, 1.0, 0.1])   # top at z=0.05

    T = 30
    joints = np.zeros((T, 22, 3), dtype=np.float32)
    joints[:, 0] = [0.0, 0.0, 0.15]   # 0.10 m above the +Z face

    contact = np.zeros((T, 5), dtype=np.float32)
    contact[:, 4] = 1.0

    object_positions = np.zeros((T, 3), dtype=np.float32)
    object_rotations = np.zeros((T, 3), dtype=np.float32)

    cfg = SupportConfig(fps=20.0, smoothing_window=3)
    support = extract_support_state(
        contact,
        joints=joints,
        object_mesh=mesh,
        object_positions=object_positions,
        object_rotations=object_rotations,
        object_id="bigsofa",   # whitelisted as +Z-up
        config=cfg,
    )

    assert (support == SUPPORT_SITTING).sum() > T * 0.8, (
        f"Z-up mesh below-gate failed to open under bigsofa override; "
        f"got {dict((SUPPORT_NAMES[s], int((support == s).sum())) for s in range(4))}"
    )


def test_support_default_up_axis_rejects_z_up_mesh_without_override() -> None:
    """Complement to the override test: the same Z-up slab must be
    rejected when no ``object_id`` is supplied. This guards against a
    regression where auto-detect gets re-enabled and starts picking
    non-Y axes on meshes that should default to +Y (the imhd bat /
    broom / kettlebell false-positive class from v5).
    """
    import trimesh

    mesh = trimesh.primitives.Box(extents=[1.0, 1.0, 0.1])   # top at z=0.05

    T = 30
    joints = np.zeros((T, 22, 3), dtype=np.float32)
    joints[:, 0] = [0.0, 0.0, 0.15]   # above +Z face but default up=+Y

    contact = np.zeros((T, 5), dtype=np.float32)
    contact[:, 4] = 1.0

    object_positions = np.zeros((T, 3), dtype=np.float32)
    object_rotations = np.zeros((T, 3), dtype=np.float32)

    cfg = SupportConfig(fps=20.0, smoothing_window=3)
    support = extract_support_state(
        contact,
        joints=joints,
        object_mesh=mesh,
        object_positions=object_positions,
        object_rotations=object_rotations,
        # no object_id → default +Y → cylinder sampled along +Y, no
        # seat face in that direction under the slab, gate stays shut.
        config=cfg,
    )

    sitting_frac = (support == SUPPORT_SITTING).sum() / T
    assert sitting_frac < 0.2, (
        f"default +Y should reject Z-up mesh without override; "
        f"sitting_frac={sitting_frac:.2%}"
    )


def test_support_carrying_object_while_walking_is_both_feet() -> None:
    """A person walking while holding an object with their hand must
    classify as both_feet, NOT hand_support. In v1-v7 any hand-object
    contact with feet off the object collapsed to hand_support,
    including all lift/carry/push-while-walking patterns — this
    flooded imhd (60-86% FP hand_support), omomo (61-89%) and neuraldome
    with the wrong body-support label. The v8 tightening requires
    pelvis stationary AND phase == stable-contact before hand_support
    can fire; walking obviously fails the stationarity gate.
    """
    T = 60
    joints = np.zeros((T, 22, 3), dtype=np.float32)
    # Pelvis walks 3 m over 3 seconds → 1 m/s XZ speed (>> 0.15 gate)
    joints[:, 0, 0] = np.linspace(0.0, 3.0, T)
    joints[:, 0, 1] = 1.0

    contact = np.zeros((T, 5), dtype=np.float32)
    contact[:, 1] = 1.0   # right hand holding a carried object

    # Phase = manipulation (object translating with the person while held)
    phase = np.full(T, PHASE_MANIPULATION, dtype=np.int64)

    cfg = SupportConfig(fps=20.0, smoothing_window=3)
    support = extract_support_state(
        contact, joints=joints, phase=phase, config=cfg,
    )

    both_feet_frac = (support == SUPPORT_BOTH_FEET).sum() / T
    hand_frac = (support == SUPPORT_HAND).sum() / T
    assert both_feet_frac > 0.8, (
        f"walking-carry should collapse to both_feet; got "
        f"both_feet={both_feet_frac:.2%}, hand={hand_frac:.2%}"
    )
    assert hand_frac < 0.05


def test_support_leaning_on_stationary_object_is_hand_support() -> None:
    """Complement to the carry test: a static person with hand on a
    static object (leaning on a table / bracing against a wall / using
    a chair to stand up) must still classify as hand_support. This is
    the genuine "body supported by hand" semantics we want to preserve
    after the v8 tightening.
    """
    T = 30
    joints = np.zeros((T, 22, 3), dtype=np.float32)
    joints[:, 0] = [0.0, 1.0, 0.0]   # pelvis stationary at standing height

    contact = np.zeros((T, 5), dtype=np.float32)
    contact[:, 0] = 1.0   # left hand contact (leaning)

    # Phase = stable-contact (object isn't moving; hand is just resting)
    phase = np.full(T, PHASE_STABLE_CONTACT, dtype=np.int64)

    cfg = SupportConfig(fps=20.0, smoothing_window=3)
    support = extract_support_state(
        contact, joints=joints, phase=phase, config=cfg,
    )

    hand_frac = (support == SUPPORT_HAND).sum() / T
    assert hand_frac > 0.8, (
        f"static leaning should classify as hand_support; got hand={hand_frac:.2%}"
    )


def test_support_manipulation_phase_blocks_hand_support() -> None:
    """Even when pelvis is stationary, if the object is moving
    (phase == manipulation) the hand is applying force to the object
    — not the other way round. This is the bat-swing-in-place case:
    user rooted in one spot, swinging a bat. Not hand_support.
    """
    T = 30
    joints = np.zeros((T, 22, 3), dtype=np.float32)
    joints[:, 0] = [0.0, 1.0, 0.0]   # pelvis planted

    contact = np.zeros((T, 5), dtype=np.float32)
    contact[:, 1] = 1.0   # right hand on bat

    phase = np.full(T, PHASE_MANIPULATION, dtype=np.int64)

    cfg = SupportConfig(fps=20.0, smoothing_window=3)
    support = extract_support_state(
        contact, joints=joints, phase=phase, config=cfg,
    )

    hand_frac = (support == SUPPORT_HAND).sum() / T
    both_feet_frac = (support == SUPPORT_BOTH_FEET).sum() / T
    assert hand_frac < 0.05 and both_feet_frac > 0.8, (
        f"stationary-user in manipulation phase must be both_feet, "
        f"not hand_support; got hand={hand_frac:.2%}, both_feet={both_feet_frac:.2%}"
    )


def test_support_allows_sitting_when_pelvis_offset_toward_armrest() -> None:
    """Regression for neuraldome/subject01_bigsofa_1310 and
    subject02_bigsofa_0: the person sits near one edge of a sofa seat
    with pelvis offset toward the armrest. The closest mesh point is
    on the armrest (horizontal direction), but the seat surface still
    sits directly below the pelvis. The old closest-point-direction
    gate rejected such frames; the new cylinder-and-normal gate must
    keep them as sitting.
    """
    import trimesh

    # Sofa proxy: wide flat seat + tall thin left armrest.
    seat = trimesh.primitives.Box(extents=[1.5, 0.1, 0.5])
    seat.apply_translation([0.0, 0.25, 0.0])        # seat top face at y=0.3
    armrest = trimesh.primitives.Box(extents=[0.1, 0.5, 0.5])
    armrest.apply_translation([-0.7, 0.55, 0.0])    # armrest left of seat, tall
    sofa = trimesh.util.concatenate([seat, armrest])

    T = 30
    joints = np.zeros((T, 22, 3), dtype=np.float32)
    # Pelvis above seat but offset left toward the armrest. Closest mesh
    # point is on the armrest side face (horizontal direction); seat is
    # directly below pelvis at y=0.3.
    joints[:, 0] = [-0.55, 0.35, 0.0]

    contact = np.zeros((T, 5), dtype=np.float32)
    contact[:, 4] = 1.0

    object_positions = np.zeros((T, 3), dtype=np.float32)
    object_rotations = np.zeros((T, 3), dtype=np.float32)

    cfg = SupportConfig(fps=20.0, smoothing_window=3)
    support = extract_support_state(
        contact,
        joints=joints,
        object_mesh=sofa,
        object_positions=object_positions,
        object_rotations=object_rotations,
        config=cfg,
    )

    sitting_frac = (support == SUPPORT_SITTING).sum() / T
    assert sitting_frac > 0.8, (
        f"offset-sitting mis-classified ({sitting_frac:.2%} sitting) — "
        f"cylinder-and-normal gate should accept seat directly below pelvis "
        f"even when armrest is horizontally closer"
    )


# ---------------------------------------------------------------------------
# Target soft-assign sigma tuned against v2 entropy stats
# ---------------------------------------------------------------------------

def test_target_sigma_default_yields_soft_distribution() -> None:
    """Default sigma must produce a *soft* distribution at typical InterAct
    patch spacing (~0.25 m between neighbours). v2 used sigma=0.05 and
    measured chairs entropy_mean = 0.26 / 2.77 max — the softmax was still
    effectively argmax. Bump ensures a neighbour contributes non-trivially.
    """
    # Two patch centres 0.25 m apart; query right at centre 0.
    patch_centers = np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]], dtype=np.float32)
    query = np.zeros(3, dtype=np.float32)

    # At the new default sigma, the ratio weight[0] / weight[1] should not
    # be pathologically large. "Soft" here = far-patch mass >= 5%.
    weights = soft_patch_assignment(query, patch_centers, sigma=TargetConfig().soft_sigma)
    assert weights[1] >= 0.05, (
        f"neighbour patch mass {weights[1]:.4f} still near-zero at "
        f"sigma={TargetConfig().soft_sigma} — kernel still too sharp"
    )

    # Sanity: the old sigma was demonstrably too sharp.
    weights_old = soft_patch_assignment(query, patch_centers, sigma=0.05)
    assert weights_old[1] < 1e-3, (
        "regression: sigma=0.05 should collapse to near-argmax per v2 stats"
    )


# ---------------------------------------------------------------------------
# v9 — Kinematic coupling contact signal
# ---------------------------------------------------------------------------

def test_kin_coupling_fires_when_rigidly_attached_to_moving_object() -> None:
    """Hand's position in the object's local frame is constant while the
    object translates in world — the textbook rigid-coupling setup that
    distance thresholds miss on wrap-grip. kin score must approach 1.
    """
    T = 40
    fps = 20.0
    # Object translates at 0.5 m/s along +x (>> kin_world_eps=0.15).
    obj_pos = np.zeros((T, 3), dtype=np.float32)
    obj_pos[:, 0] = np.linspace(0.0, 1.0, T)
    obj_rot = np.zeros((T, 3), dtype=np.float32)

    # Hand is rigidly attached at local offset (0.2, 0, 0) — so in world
    # it moves identically with the object. Distance to mesh would be 20 cm,
    # beyond any sensible threshold, but kin signal should fire.
    hand_world = obj_pos + np.array([0.2, 0.0, 0.0], dtype=np.float32)

    cfg = ContactConfig(fps=fps)
    score = _kinematic_contact_score(hand_world, obj_pos, obj_rot, cfg)

    # Middle of the window (avoid edges with boundary padding bias) must
    # be well above the downstream binarization threshold (0.5). Perfect
    # rigid + fully-moving product saturates at ~0.86 because each sigmoid
    # factor is < 1 individually — that's the design (leave headroom so
    # partial signal still scores below full).
    mid = T // 2
    assert score[mid] > 0.7, (
        f"rigid-coupled + moving object should fire kin contact; "
        f"got score[{mid}]={score[mid]:.3f}, full={score.tolist()}"
    )


def test_kin_coupling_silent_in_static_scene() -> None:
    """Object not moving in world → kin signal must NOT fire (otherwise
    it would flag every static frame as contact, which is the whole
    reason for the world-speed gate). Hand itself is also stationary here,
    which is a realistic "hand resting at rest" scenario.
    """
    T = 40
    fps = 20.0
    obj_pos = np.zeros((T, 3), dtype=np.float32)
    obj_rot = np.zeros((T, 3), dtype=np.float32)
    hand_world = np.tile(np.array([0.2, 0.0, 0.0], dtype=np.float32), (T, 1))

    cfg = ContactConfig(fps=fps)
    score = _kinematic_contact_score(hand_world, obj_pos, obj_rot, cfg)

    assert score.max() < 0.05, (
        f"static scene must not fire kin contact; got max={score.max():.3f}"
    )


def test_kin_coupling_silent_when_hand_orbits_moving_object() -> None:
    """Hand circles the object while the object translates. Object-local
    hand position varies large-amplitude → local_score near 0 → kin silent.
    This guards against the "hand near but not attached" FP class.
    """
    T = 40
    fps = 20.0
    # Object translates +x at 0.5 m/s
    obj_pos = np.zeros((T, 3), dtype=np.float32)
    obj_pos[:, 0] = np.linspace(0.0, 1.0, T)
    obj_rot = np.zeros((T, 3), dtype=np.float32)

    # Hand follows the object but also orbits at radius 0.3 m in xz plane.
    theta = np.linspace(0, 2 * np.pi, T)
    hand_world = obj_pos.copy()
    hand_world[:, 0] += 0.3 * np.cos(theta)
    hand_world[:, 2] += 0.3 * np.sin(theta)

    cfg = ContactConfig(fps=fps)
    score = _kinematic_contact_score(hand_world, obj_pos, obj_rot, cfg)

    # The local xyz std over a 0.5s window is ~0.15 m (well above 0.03),
    # so local_score ≈ 0 even though world_score ≈ 1.
    assert score.max() < 0.2, (
        f"orbiting hand must not fire kin contact; got max={score.max():.3f}"
    )


def test_kin_coupling_recovers_wrap_grip_in_extract_contact_state() -> None:
    """End-to-end: a hand 18 cm from the mesh surface (too far for the
    0.12 m distance threshold) but rigidly attached to a moving object
    must still be labelled as contact after v9. This is the neuraldome
    wrap-grip class that dominated v8's 624 drops.
    """
    import trimesh

    # Small box mesh: 10 cm cube, so distance from a point 18 cm away
    # is ~13 cm — well beyond hand threshold 0.12 m.
    mesh = trimesh.primitives.Box(extents=[0.1, 0.1, 0.1])

    T = 40
    fps = 20.0
    joints = np.zeros((T, 22, 3), dtype=np.float32)
    obj_pos = np.zeros((T, 3), dtype=np.float32)
    obj_pos[:, 0] = np.linspace(0.0, 1.0, T)   # box translates +x at 0.5 m/s
    obj_rot = np.zeros((T, 3), dtype=np.float32)

    # Right wrist (idx 21) rigidly offset 18 cm from object centre.
    joints[:, 21, :] = obj_pos + np.array([0.18, 0.0, 0.0], dtype=np.float32)

    cfg = ContactConfig(fps=fps, median_filter_size=3)
    contact = extract_contact_state(joints, mesh, obj_pos, obj_rot, cfg)
    # Body part 1 = right_hand. Middle window must be labelled contact.
    right_hand_contact = contact[:, 1]
    mid = T // 2
    assert right_hand_contact[mid] > 0.5, (
        f"wrap-grip at 18 cm not recovered by kin coupling; "
        f"contact[{mid}]={right_hand_contact[mid]:.3f}"
    )


def test_kin_coupling_disabled_flag_switches_off() -> None:
    """``use_kinematic_coupling=False`` must restore v8 pure-distance
    behaviour — kin score returns all zeros regardless of input."""
    T = 20
    fps = 20.0
    obj_pos = np.zeros((T, 3), dtype=np.float32)
    obj_pos[:, 0] = np.linspace(0.0, 1.0, T)
    obj_rot = np.zeros((T, 3), dtype=np.float32)
    hand_world = obj_pos + np.array([0.2, 0.0, 0.0], dtype=np.float32)

    cfg = ContactConfig(fps=fps, use_kinematic_coupling=False)
    score = _kinematic_contact_score(hand_world, obj_pos, obj_rot, cfg)

    assert (score == 0).all()


# ---------------------------------------------------------------------------
# HMM fallback on degenerate input
# ---------------------------------------------------------------------------

def test_hmm_falls_back_to_initial_on_bad_features() -> None:
    """v2 saw 5/8475 sequences abort with ``startprob_ must sum to 1
    (got nan)`` — NaN-tainted features crashed hmmlearn. Refinement is a
    smoothing step, not a hard dependency, so the right degradation is
    to keep the heuristic labels rather than lose the whole sequence.
    """
    T = 10
    features = np.zeros((T, 4), dtype=np.float64)
    features[:, 0] = 0.5
    features[5, 0] = np.nan   # poisons fit()

    initial_phases = np.full(T, PHASE_APPROACH, dtype=np.int64)
    initial_phases[5:] = PHASE_STABLE_CONTACT

    refined = refine_phases_hmm(features, initial_phases, HMMConfig(n_iter=1))

    # Fallback returns the heuristic labels unchanged (int64, identical).
    assert refined.dtype == np.int64
    assert (refined == initial_phases).all(), (
        f"expected unchanged heuristic labels, got {refined.tolist()}"
    )


if __name__ == "__main__":
    tests = [
        test_phase_sitting_enters_stable_contact,
        test_phase_rotation_only_enters_manipulation,
        test_phase_rotation_only_is_stable_without_rotation_signal,
        test_hmm_state_ids_preserve_phase_semantics,
        test_build_phase_features_shape_with_rotation,
        test_support_majority_filter_no_ordinal_artifacts,
        test_support_extraction_sitting_sequence,
        test_support_push_object_not_classified_as_sitting,
        test_support_stationary_sitting_still_classified_as_sitting,
        test_support_rejects_sitting_when_pelvis_far_above_seat,
        test_support_allows_sitting_when_object_below_pelvis,
        test_support_allows_sitting_when_pelvis_offset_toward_armrest,
        test_support_up_axis_override_unlocks_z_up_mesh,
        test_support_default_up_axis_rejects_z_up_mesh_without_override,
        test_support_carrying_object_while_walking_is_both_feet,
        test_support_leaning_on_stationary_object_is_hand_support,
        test_support_manipulation_phase_blocks_hand_support,
        test_target_sigma_default_yields_soft_distribution,
        test_kin_coupling_fires_when_rigidly_attached_to_moving_object,
        test_kin_coupling_silent_in_static_scene,
        test_kin_coupling_silent_when_hand_orbits_moving_object,
        test_kin_coupling_recovers_wrap_grip_in_extract_contact_state,
        test_kin_coupling_disabled_flag_switches_off,
        test_hmm_falls_back_to_initial_on_bad_features,
    ]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {t.__name__}\n      {e}")
        except Exception as e:
            failures += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    raise SystemExit(failures)
