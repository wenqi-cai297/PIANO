"""Tests for ``piano.data.stage2_oracle_conditions`` (Round-29 prompt §9)."""
from __future__ import annotations

import numpy as np
import pytest

from piano.data.stage2_oracle_conditions import (
    BODY_VARIANT_DIMS,
    COARSE_KEY_JOINT_INDICES,
    COARSE_VARIANT_DIMS,
    INTERACTION_VARIANT_DIMS,
    SUPPORT_VARIANT_DIMS,
    build_body_refinement_condition,
    build_coarse_condition,
    build_interaction_condition,
    build_stage2_condition_bundle,
    build_support_condition,
)


@pytest.fixture
def synth_joints() -> np.ndarray:
    T = 32
    rng = np.random.default_rng(0)
    base = rng.normal(scale=0.05, size=(22, 3)).astype(np.float32)
    base[0] = [0.0, 0.9, 0.0]
    base[16] = [-0.2, 1.3, 0.0]
    base[17] = [+0.2, 1.3, 0.0]
    base[1] = [-0.1, 0.8, 0.0]
    base[2] = [+0.1, 0.8, 0.0]
    base[7] = [-0.1, 0.05, 0.0]
    base[8] = [+0.1, 0.05, 0.0]
    base[20] = [-0.4, 1.2, 0.1]
    base[21] = [+0.4, 1.2, 0.1]
    base[12] = [0.0, 1.5, 0.0]
    joints = np.zeros((T, 22, 3), dtype=np.float32)
    for t in range(T):
        joints[t] = base + rng.normal(scale=0.02, size=(22, 3)).astype(np.float32) * (t / T)
        joints[t, 0, 0] = 0.02 * t
    return joints


@pytest.fixture
def synth_object() -> tuple[np.ndarray, np.ndarray]:
    T = 32
    rng = np.random.default_rng(7)
    obj_pos = rng.normal(scale=0.5, size=(T, 3)).astype(np.float32) + np.array([0.0, 0.5, 0.5], dtype=np.float32)
    obj_rot = rng.normal(scale=0.1, size=(T, 3)).astype(np.float32)
    return obj_pos, obj_rot


@pytest.fixture
def synth_contact() -> np.ndarray:
    T = 32
    rng = np.random.default_rng(13)
    return (rng.random((T, 5)) > 0.5).astype(np.float32)


# ---------------- Coarse ----------------

def test_coarse_variant_dims(synth_joints: np.ndarray) -> None:
    for cv, expected in COARSE_VARIANT_DIMS.items():
        arr, info = build_coarse_condition(synth_joints, cv)
        assert arr.shape == (synth_joints.shape[0], expected), (cv, arr.shape, expected)
        assert info["finite_frac"] == 1.0


def test_coarse_t0_key_joint_deltas_are_zero(synth_joints: np.ndarray) -> None:
    for cv in ("C38-current", "C41-current", "C38-root0", "C41-root0"):
        arr, _ = build_coarse_condition(synth_joints, cv)
        # First 15 channels are the 5-joint deltas; t=0 must be zero by construction.
        np.testing.assert_allclose(arr[0, :15], np.zeros(15), atol=1e-5)


def test_coarse_pelvis_delta_is_nonzero_when_pelvis_moves(synth_joints: np.ndarray) -> None:
    # synth_joints has pelvis x += 0.02 * t, so pelvis_delta != 0 by t=1.
    arr, _ = build_coarse_condition(synth_joints, "C41-current")
    pelvis_delta = arr[:, 15:18]
    # First-frame pelvis delta should be 0; later frames nonzero.
    np.testing.assert_allclose(pelvis_delta[0], np.zeros(3), atol=1e-5)
    assert np.abs(pelvis_delta[10]).max() > 0.01


def test_coarse_root0_invariant_under_global_yaw(synth_joints: np.ndarray) -> None:
    """root0-yaw coarse channel should be the same for two clips that
    differ only by an applied global Y rotation at frame 0 (re-applied
    consistently across all frames)."""
    import math

    angle = 0.4
    c, s = math.cos(angle), math.sin(angle)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    rotated = synth_joints @ R.T
    arr_a, _ = build_coarse_condition(synth_joints, "C38-root0")
    arr_b, _ = build_coarse_condition(rotated, "C38-root0")
    # The 5-joint deltas in root0 frame are invariant to a global Y rotation.
    np.testing.assert_allclose(arr_a, arr_b, atol=1e-4)


# ---------------- Interaction ----------------

def test_interaction_variant_dims(
    synth_joints: np.ndarray,
    synth_object: tuple[np.ndarray, np.ndarray],
    synth_contact: np.ndarray,
) -> None:
    obj_pos, obj_rot = synth_object
    for iv, expected in INTERACTION_VARIANT_DIMS.items():
        arr, info = build_interaction_condition(
            synth_joints, obj_pos, obj_rot, synth_contact, variant=iv,
        )
        assert arr.shape == (synth_joints.shape[0], expected), (iv, arr.shape, expected)
        assert info["finite_frac"] == 1.0


def test_interaction_i2_masks_offset_to_zero_when_not_in_contact(
    synth_joints: np.ndarray,
    synth_object: tuple[np.ndarray, np.ndarray],
) -> None:
    obj_pos, obj_rot = synth_object
    # All-zero contact -> masked offset is identically 0.
    contact = np.zeros((synth_joints.shape[0], 5), dtype=np.float32)
    arr, _ = build_interaction_condition(
        synth_joints, obj_pos, obj_rot, contact, variant="I2-offset-masked",
    )
    assert np.abs(arr).max() == 0.0


def test_interaction_i4_unmasked_keeps_offset(
    synth_joints: np.ndarray,
    synth_object: tuple[np.ndarray, np.ndarray],
) -> None:
    obj_pos, obj_rot = synth_object
    contact = np.zeros((synth_joints.shape[0], 5), dtype=np.float32)
    arr, _ = build_interaction_condition(
        synth_joints, obj_pos, obj_rot, contact, variant="I4-contact-offset-unmasked",
    )
    # I4 offset survives even when contact mask is zero.
    assert np.abs(arr[:, 2:]).max() > 0.01


def test_interaction_requires_object_when_not_i0(
    synth_joints: np.ndarray,
    synth_contact: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        build_interaction_condition(
            synth_joints, None, None, synth_contact, variant="I3-contact-offset-masked",
        )


# ---------------- I5 all-part (R29 failure-targeted ablation R5) ----------------


def test_interaction_i5_shape_and_layout(
    synth_joints: np.ndarray,
    synth_object: tuple[np.ndarray, np.ndarray],
    synth_contact: np.ndarray,
) -> None:
    """I5 must be 20D = 5 contact + 5 parts × 3 = 5 + 15."""
    obj_pos, obj_rot = synth_object
    T = synth_joints.shape[0]
    arr, info = build_interaction_condition(
        synth_joints, obj_pos, obj_rot, synth_contact,
        variant="I5-allpart-contact-offset-masked",
    )
    assert arr.shape == (T, 20)
    # First 5 channels are the per-part contact (clipped from contact_state).
    assert np.allclose(arr[:, 0:5], np.clip(synth_contact[:, 0:5], 0.0, 1.0))
    # Remaining 15 channels are masked object-local offset for 5 parts × 3.
    assert np.isfinite(arr).all()
    assert "left_foot_contact_frac" in info
    assert "right_foot_contact_frac" in info
    assert "pelvis_contact_frac" in info


def test_interaction_i5_offset_masked_to_zero_when_not_in_contact(
    synth_joints: np.ndarray,
    synth_object: tuple[np.ndarray, np.ndarray],
) -> None:
    """All-zero contact_state ⇒ all 15 offset channels identically zero."""
    obj_pos, obj_rot = synth_object
    contact = np.zeros((synth_joints.shape[0], 5), dtype=np.float32)
    arr, _ = build_interaction_condition(
        synth_joints, obj_pos, obj_rot, contact,
        variant="I5-allpart-contact-offset-masked",
    )
    # First 5 (contact) are zero, last 15 (masked offset) must also be zero.
    assert np.abs(arr).max() == 0.0


def test_interaction_i5_contact_order_matches_contact_state(
    synth_joints: np.ndarray,
    synth_object: tuple[np.ndarray, np.ndarray],
) -> None:
    """Per prompt §R5: contact channel order must match contact_state
    column order — 0 L_hand, 1 R_hand, 2 L_foot, 3 R_foot, 4 pelvis."""
    obj_pos, obj_rot = synth_object
    T = synth_joints.shape[0]
    contact = np.zeros((T, 5), dtype=np.float32)
    # Set only column 2 (left_foot) to 1.0 across all frames.
    contact[:, 2] = 1.0
    arr, _ = build_interaction_condition(
        synth_joints, obj_pos, obj_rot, contact,
        variant="I5-allpart-contact-offset-masked",
    )
    # Only L_foot's offset slice should be non-zero. Slice for part i is
    # arr[:, 5 + 3i : 5 + 3i + 3]. L_foot is part 2 → slice [11:14].
    for part_i in range(5):
        s = slice(5 + 3 * part_i, 5 + 3 * (part_i + 1))
        if part_i == 2:
            # L_foot must be non-zero on at least some frames (object-local
            # offset between left ankle and the object).
            assert np.abs(arr[:, s]).max() > 0.0, "L_foot offset zeroed wrongly"
        else:
            assert np.abs(arr[:, s]).max() == 0.0, (
                f"part {part_i} leaked offset under L_foot-only contact"
            )


def test_interaction_i5_dim_in_registry() -> None:
    """Sanity: INTERACTION_VARIANT_DIMS must register I5 at 20D."""
    assert INTERACTION_VARIANT_DIMS["I5-allpart-contact-offset-masked"] == 20


def test_interaction_i5_requires_object_and_contact(
    synth_joints: np.ndarray,
    synth_contact: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        build_interaction_condition(
            synth_joints, None, None, synth_contact,
            variant="I5-allpart-contact-offset-masked",
        )


# ---------------- Support ----------------

def test_support_variant_dims(synth_joints: np.ndarray) -> None:
    for sv, expected in SUPPORT_VARIANT_DIMS.items():
        arr, info = build_support_condition(synth_joints, variant=sv)
        assert arr.shape == (synth_joints.shape[0], expected), (sv, arr.shape, expected)
        assert info["finite_frac"] == 1.0


# ---------------- Body refinement ----------------

def test_body_variant_dims(synth_joints: np.ndarray) -> None:
    for bv, expected in BODY_VARIANT_DIMS.items():
        arr, info = build_body_refinement_condition(synth_joints, variant=bv)
        assert arr.shape == (synth_joints.shape[0], expected), (bv, arr.shape, expected)
        assert info["finite_frac"] == 1.0


def test_body_b3_residual_is_smaller_than_b2_absolute(synth_joints: np.ndarray) -> None:
    arr_abs, _ = build_body_refinement_condition(synth_joints, variant="B2-absolute-delta")
    arr_res, _ = build_body_refinement_condition(synth_joints, variant="B3-lowpass-residual")
    # The low-pass residual should have smaller energy than the absolute delta.
    assert np.abs(arr_res).mean() <= np.abs(arr_abs).mean() + 1e-6


def test_body_b1_mask_only_is_constant_per_clip(synth_joints: np.ndarray) -> None:
    arr, _ = build_body_refinement_condition(synth_joints, variant="B1-mask-only")
    # The mask broadcasts the per-joint active flag across T → row-equal.
    for t in range(arr.shape[0]):
        np.testing.assert_allclose(arr[t], arr[0])


# ---------------- Bundle ----------------

def test_bundle_full_dense(
    synth_joints: np.ndarray,
    synth_object: tuple[np.ndarray, np.ndarray],
    synth_contact: np.ndarray,
) -> None:
    obj_pos, obj_rot = synth_object
    bundle = build_stage2_condition_bundle(
        synth_joints,
        coarse_variant="C41-current",
        interaction_variant="I3-contact-offset-masked",
        support_variant="S4-S1-phase-footstep",
        body_variant="B4-lowpass-residual-mask",
        object_positions=obj_pos,
        object_rotations=obj_rot,
        contact_state=synth_contact,
    )
    assert bundle.coarse_extra is not None and bundle.coarse_extra.shape[-1] == 18
    assert bundle.interaction is not None and bundle.interaction.shape[-1] == 8
    assert bundle.support is not None and bundle.support.shape[-1] == 13
    assert bundle.body_refine is not None and bundle.body_refine.shape[-1] == 20
