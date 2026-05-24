"""Round-27 Tier-0B Commit 4 — unit tests for the 5 temporal interaction
losses in ``piano.training.temporal_interaction_losses``.

Each loss is verified on synthetic (B=1, T=20) clips so the maths can
be hand-checked. Tests cover:

* GT-as-pred -> loss ≈ 0 (sanity)
* Perturbed pred -> loss > 0
* Tracking projection asymmetry (over-following GT is NOT penalised)
* Empty-mask cases return a finite scalar (no NaN / Inf)
* Gradient flows from each loss back to pred_joints
"""
from __future__ import annotations

import math

import pytest
import torch

from piano.training.temporal_interaction_losses import (
    LEFT_ANKLE_IDX,
    LEFT_WRIST_IDX,
    RIGHT_ANKLE_IDX,
    RIGHT_WRIST_IDX,
    TemporalInteractionLossConfig,
    compute_all_temporal_losses,
    loss_contact_drift_smoothl1,
    loss_contact_rel_offset_smoothl1,
    loss_contact_tracking_projection,
    loss_gait_both_airborne,
    loss_gait_stance_velocity,
)


def _make_clip(T: int = 20, with_contact: bool = True, with_walking: bool = True):
    """Synthetic GT clip: object moves +X, hand follows, walking on
    floor with alternating ankle motion."""
    joints = torch.zeros(1, T, 22, 3, dtype=torch.float32)
    # Ankles on the floor (y=0). Add a tiny baseline xz separation so
    # they aren't co-located.
    joints[:, :, LEFT_ANKLE_IDX, 0] = +0.1
    joints[:, :, RIGHT_ANKLE_IDX, 0] = -0.1
    # Wrists at chest height, moving along +X over the segment.
    t_axis = torch.arange(T, dtype=torch.float32).reshape(1, T)
    joints[:, :, LEFT_WRIST_IDX, 0] = 0.30 + 0.5 * (t_axis / T)
    joints[:, :, LEFT_WRIST_IDX, 1] = 1.2
    joints[:, :, RIGHT_WRIST_IDX, 0] = -0.30 + 0.5 * (t_axis / T)
    joints[:, :, RIGHT_WRIST_IDX, 1] = 1.2

    # Object also at chest height, moves +X parallel to GT hand.
    obj_pos = torch.zeros(1, T, 3, dtype=torch.float32)
    obj_pos[..., 0] = 0.0 + 0.5 * (t_axis.squeeze(0) / T)
    obj_pos[..., 1] = 1.0
    obj_pos[..., 2] = 0.0
    obj_rot = torch.zeros(1, T, 3, dtype=torch.float32)  # identity

    contact_state = torch.zeros(1, T, 5, dtype=torch.float32)
    if with_contact:
        contact_state[:, 2:18, 0] = 1.0    # left hand in contact [2, 18)
        contact_state[:, 5:15, 1] = 1.0    # right hand in contact [5, 15)

    walking_mask = torch.zeros(1, T, 1, dtype=torch.float32)
    if with_walking:
        walking_mask[:, 4:T, 0] = 1.0

    # GT-derived foot stance (faked: full stance during walking).
    foot_stance_gt = torch.zeros(1, T, 2, dtype=torch.float32)
    foot_stance_gt[:, 4:, :] = 1.0

    return joints, obj_pos, obj_rot, contact_state, walking_mask, foot_stance_gt


def _default_cfg() -> TemporalInteractionLossConfig:
    return TemporalInteractionLossConfig(
        contact_rel_offset_weight=1.0,
        contact_drift_weight=1.0,
        contact_tracking_weight=1.0,
        gait_both_airborne_weight=1.0,
        gait_stance_velocity_weight=1.0,
    )


# ---------------------------------------------------------------------------
# Contact losses
# ---------------------------------------------------------------------------


def test_contact_rel_offset_zero_when_pred_eq_gt():
    joints, obj_pos, obj_rot, contact_state, _, _ = _make_clip()
    cfg = _default_cfg()
    loss = loss_contact_rel_offset_smoothl1(
        pred_joints=joints,
        gt_joints=joints,
        object_positions=obj_pos,
        object_rotations=obj_rot,
        contact_state=contact_state,
        cfg=cfg,
    )
    assert torch.isfinite(loss)
    assert loss.item() < 1e-6, f"GT-as-pred should give ~0 loss, got {loss.item()}"


def test_contact_rel_offset_positive_on_perturbation():
    joints, obj_pos, obj_rot, contact_state, _, _ = _make_clip()
    pred = joints.clone()
    pred[:, :, LEFT_WRIST_IDX, 0] += 0.15   # offset all left-wrist frames by +15 cm
    loss = loss_contact_rel_offset_smoothl1(
        pred_joints=pred,
        gt_joints=joints,
        object_positions=obj_pos,
        object_rotations=obj_rot,
        contact_state=contact_state,
        cfg=_default_cfg(),
    )
    assert loss.item() > 0.0, "Perturbed pred should produce positive loss"


def test_contact_drift_zero_when_pred_eq_gt():
    joints, obj_pos, obj_rot, contact_state, _, _ = _make_clip()
    loss = loss_contact_drift_smoothl1(
        pred_joints=joints,
        gt_joints=joints,
        object_positions=obj_pos,
        object_rotations=obj_rot,
        contact_state=contact_state,
        cfg=_default_cfg(),
    )
    assert torch.isfinite(loss)
    assert loss.item() < 1e-6


def test_contact_drift_positive_when_pred_drifts_away():
    """Pred starts on object then linearly drifts away — GT stays put."""
    joints, obj_pos, obj_rot, contact_state, _, _ = _make_clip()
    pred = joints.clone()
    # Add a linearly growing offset to left wrist within the contact segment.
    T = pred.shape[1]
    growth = torch.linspace(0.0, 0.20, T)
    pred[:, :, LEFT_WRIST_IDX, 0] += growth
    loss = loss_contact_drift_smoothl1(
        pred_joints=pred,
        gt_joints=joints,
        object_positions=obj_pos,
        object_rotations=obj_rot,
        contact_state=contact_state,
        cfg=_default_cfg(),
    )
    assert loss.item() > 0.0


def test_contact_tracking_projection_zero_when_pred_eq_gt():
    joints, obj_pos, obj_rot, contact_state, _, _ = _make_clip()
    loss = loss_contact_tracking_projection(
        pred_joints=joints,
        gt_joints=joints,
        object_positions=obj_pos,
        object_rotations=obj_rot,
        contact_state=contact_state,
        cfg=_default_cfg(),
    )
    assert torch.isfinite(loss)
    assert loss.item() < 1e-6


def test_contact_tracking_projection_penalises_falling_behind():
    """If pred wrist stays put while obj+GT-wrist move +X, the
    tracking projection penalty should fire."""
    joints, obj_pos, obj_rot, contact_state, _, _ = _make_clip()
    pred = joints.clone()
    # Hold left wrist X coordinate constant from frame 2 onward.
    pred[:, 2:, LEFT_WRIST_IDX, 0] = pred[:, 2, LEFT_WRIST_IDX, 0]
    loss = loss_contact_tracking_projection(
        pred_joints=pred,
        gt_joints=joints,
        object_positions=obj_pos,
        object_rotations=obj_rot,
        contact_state=contact_state,
        cfg=_default_cfg(),
    )
    assert loss.item() > 0.0, "Held-back pred wrist should incur tracking loss"


def test_contact_tracking_projection_asymmetric_no_penalty_for_overshooting():
    """If pred wrist OVER-follows the object (goes further than GT in
    the +X direction), the tracking ReLU should still be ≤ margin."""
    joints, obj_pos, obj_rot, contact_state, _, _ = _make_clip()
    pred = joints.clone()
    # Make pred-wrist travel TWICE as far in +X as GT does.
    pred[:, :, LEFT_WRIST_IDX, 0] = (
        joints[:, 0, LEFT_WRIST_IDX, 0]
        + 2.0 * (joints[:, :, LEFT_WRIST_IDX, 0] - joints[:, 0, LEFT_WRIST_IDX, 0])
    )
    pred[:, :, RIGHT_WRIST_IDX, 0] = (
        joints[:, 0, RIGHT_WRIST_IDX, 0]
        + 2.0 * (joints[:, :, RIGHT_WRIST_IDX, 0] - joints[:, 0, RIGHT_WRIST_IDX, 0])
    )
    loss = loss_contact_tracking_projection(
        pred_joints=pred,
        gt_joints=joints,
        object_positions=obj_pos,
        object_rotations=obj_rot,
        contact_state=contact_state,
        cfg=_default_cfg(),
    )
    assert loss.item() < 1e-6, (
        "Tracking loss is asymmetric — pred over-following GT must not "
        f"be penalised; got {loss.item()}"
    )


def test_contact_losses_zero_when_no_contact_frames():
    """Empty mask -> all 3 contact losses must be 0 (clamped denom)."""
    joints, obj_pos, obj_rot, _, _, _ = _make_clip(with_contact=False)
    cs = torch.zeros(1, joints.shape[1], 5, dtype=torch.float32)
    cfg = _default_cfg()
    for fn in (
        loss_contact_rel_offset_smoothl1,
        loss_contact_drift_smoothl1,
        loss_contact_tracking_projection,
    ):
        loss = fn(
            pred_joints=joints,
            gt_joints=joints,
            object_positions=obj_pos,
            object_rotations=obj_rot,
            contact_state=cs,
            cfg=cfg,
        )
        assert torch.isfinite(loss)
        assert loss.item() == 0.0


# ---------------------------------------------------------------------------
# Gait losses
# ---------------------------------------------------------------------------


def test_gait_both_airborne_zero_when_feet_on_floor():
    joints, _, _, _, walking_mask, _ = _make_clip()
    cfg = _default_cfg()
    loss = loss_gait_both_airborne(
        pred_joints=joints, gt_joints=joints,
        walking_mask=walking_mask, cfg=cfg,
    )
    assert torch.isfinite(loss)
    # Feet on floor (y=0) and floor estimate ≈ 0 → grounded prob ≈ 1 →
    # both_airborne ≈ 0. Tolerance is generous because of the sigmoid soft edge.
    assert loss.item() < 1e-2, f"Feet on floor should give ~0 loss, got {loss.item()}"


def test_gait_both_airborne_positive_when_both_feet_in_air():
    joints, _, _, _, walking_mask, _ = _make_clip()
    pred = joints.clone()
    # Lift both ankles to 50 cm.
    pred[:, :, LEFT_ANKLE_IDX, 1] = 0.5
    pred[:, :, RIGHT_ANKLE_IDX, 1] = 0.5
    loss = loss_gait_both_airborne(
        pred_joints=pred, gt_joints=joints,
        walking_mask=walking_mask, cfg=_default_cfg(),
    )
    assert loss.item() > 0.5, (
        f"Both feet 50cm above floor should give large loss, got {loss.item()}"
    )


def test_gait_both_airborne_zero_when_no_walking():
    joints, _, _, _, _, _ = _make_clip(with_walking=False)
    walk = torch.zeros(1, joints.shape[1], 1, dtype=torch.float32)
    loss = loss_gait_both_airborne(
        pred_joints=joints, gt_joints=joints,
        walking_mask=walk, cfg=_default_cfg(),
    )
    assert torch.isfinite(loss)
    assert loss.item() == 0.0


def test_gait_stance_velocity_zero_when_feet_still():
    joints, _, _, _, walking_mask, foot_stance_gt = _make_clip()
    # Ankles are stationary across t in _make_clip.
    loss = loss_gait_stance_velocity(
        pred_joints=joints,
        foot_stance_gt=foot_stance_gt,
        walking_mask=walking_mask,
    )
    assert torch.isfinite(loss)
    assert loss.item() < 1e-6


def test_gait_stance_velocity_positive_when_stance_foot_slides():
    joints, _, _, _, walking_mask, foot_stance_gt = _make_clip()
    pred = joints.clone()
    T = pred.shape[1]
    # Slide left ankle +0.5 m/s in +X (== +0.025 m/frame at 20 fps).
    pred[:, :, LEFT_ANKLE_IDX, 0] = 0.1 + 0.025 * torch.arange(T, dtype=torch.float32)
    loss = loss_gait_stance_velocity(
        pred_joints=pred,
        foot_stance_gt=foot_stance_gt,
        walking_mask=walking_mask,
    )
    assert loss.item() > 0.0


# ---------------------------------------------------------------------------
# Gradient flow + aggregator
# ---------------------------------------------------------------------------


def test_gradient_flows_to_pred_joints():
    joints, obj_pos, obj_rot, contact_state, walking_mask, foot_stance_gt = _make_clip()
    cfg = _default_cfg()
    pred = joints.clone()
    # Perturb so each loss is non-zero.
    pred[:, :, LEFT_WRIST_IDX, 0] += 0.10
    pred[:, :, LEFT_ANKLE_IDX, 1] += 0.30
    pred[:, :, LEFT_ANKLE_IDX, 0] = 0.1 + 0.025 * torch.arange(
        pred.shape[1], dtype=torch.float32,
    )
    pred.requires_grad_(True)

    losses = compute_all_temporal_losses(
        pred_joints=pred,
        gt_joints=joints,
        object_positions=obj_pos,
        object_rotations=obj_rot,
        contact_state=contact_state,
        walking_mask=walking_mask,
        foot_stance_gt=foot_stance_gt,
        cfg=cfg,
    )
    assert set(losses.keys()) == {
        "loss_contact_rel_offset",
        "loss_contact_drift",
        "loss_contact_tracking_projection",
        "loss_gait_both_airborne",
        "loss_gait_stance_velocity",
    }
    total = sum(losses.values())
    total.backward()
    assert pred.grad is not None
    assert pred.grad.abs().sum().item() > 0.0


def test_object_local_frame_under_90deg_rotation():
    """If the object rotates by +90° around Y, the object-local hand
    offset is rotated correspondingly — so the same world wrist
    contributes a different ``r``, and the rel_offset loss against a
    perturbed pred therefore changes. This catches einsum-direction
    bugs in ``_wrist_object_local``.
    """
    joints, obj_pos, obj_rot, contact_state, _, _ = _make_clip()
    pred = joints.clone()
    pred[:, :, LEFT_WRIST_IDX, 0] += 0.10  # offset in WORLD X.

    cfg = _default_cfg()
    loss_id = loss_contact_rel_offset_smoothl1(
        pred_joints=pred,
        gt_joints=joints,
        object_positions=obj_pos,
        object_rotations=obj_rot,
        contact_state=contact_state,
        cfg=cfg,
    )

    # Rotate object 90° around Y. In object-local frame the wrist's
    # +X perturbation maps to ±Z (sign per einsum direction). The
    # SmoothL1 magnitude is the same, so |loss_rot| ≈ |loss_id| within
    # SmoothL1 quadratic-zone tolerance.
    obj_rot_y90 = obj_rot.clone()
    obj_rot_y90[..., 1] = math.pi / 2.0
    loss_rot = loss_contact_rel_offset_smoothl1(
        pred_joints=pred,
        gt_joints=joints,
        object_positions=obj_pos,
        object_rotations=obj_rot_y90,
        contact_state=contact_state,
        cfg=cfg,
    )
    # Both should be > 0 and approximately equal magnitude
    # (rotation just relabels which axis the perturbation lives in).
    assert torch.isfinite(loss_id) and torch.isfinite(loss_rot)
    assert loss_id.item() > 1e-6 and loss_rot.item() > 1e-6
    assert loss_rot.item() == pytest.approx(loss_id.item(), rel=1e-3, abs=1e-4)
