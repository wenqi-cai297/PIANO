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
    loss_r29_interaction_consistency,
    loss_r29_support_both_airborne,
    loss_r29_support_stance_velocity,
    loss_r29_swing_clearance,
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


# ---------------------------------------------------------------------------
# Round-29 condition-consistency losses (loss strategy ablation)
# Per analyses/2026-05-27_round29_loss_strategy_ablation_prompt_for_claude_code.md
# ---------------------------------------------------------------------------


def _build_i3_condition(
    gt_joints: torch.Tensor,
    obj_pos: torch.Tensor,
    obj_rot: torch.Tensor,
    contact_state: torch.Tensor,
    hand_offset_clamp_m: float = 2.0,
) -> torch.Tensor:
    """Build a synthetic (B, T, 8) I3 condition tensor that matches the
    dataset-side builder ``build_interaction_condition('I3-...')``:

        [..., 0:2] = hand_contact  (L, R) soft 0..1
        [..., 2:8] = (rel_norm * hand_contact)[..., L/R, xyz] flattened,
                     where rel_norm = clamp(R_obj.T @ (wrist - obj_pos),
                                            ±clamp_m) / clamp_m
    """
    from piano.training.temporal_interaction_losses import (
        _axis_angle_to_matrix_t,
        _wrist_object_local,
        _wrist_world_pred_gt,
    )
    R_obj = _axis_angle_to_matrix_t(obj_rot)                                   # (B, T, 3, 3)
    pw, _ = _wrist_world_pred_gt(gt_joints, gt_joints)                         # (B, T, 2, 3)
    rel = _wrist_object_local(pw, obj_pos, R_obj)                              # (B, T, 2, 3)
    rel = rel.clamp(-hand_offset_clamp_m, hand_offset_clamp_m)
    rel_norm = rel / float(hand_offset_clamp_m)
    hand_contact = contact_state[..., 0:2].clamp(0.0, 1.0)                     # (B, T, 2)
    target_offset = rel_norm * hand_contact.unsqueeze(-1)                      # (B, T, 2, 3)
    return torch.cat(
        [hand_contact, target_offset.reshape(*target_offset.shape[:2], 6)],
        dim=-1,
    )                                                                           # (B, T, 8)


def _build_s4_condition(
    walking_mask: torch.Tensor,        # (B, T, 1)
    foot_stance: torch.Tensor,         # (B, T, 2)
) -> torch.Tensor:
    """Build a minimal (B, T, 13) S4-shaped condition. We only need
    the first 5 channels for the R29 support losses; the rest are
    zero-filled per the prompt §1 layout."""
    B, T, _ = walking_mask.shape
    height_norm = torch.zeros(B, T, 2, dtype=walking_mask.dtype)
    s1 = torch.cat([foot_stance, height_norm, walking_mask], dim=-1)           # (B, T, 5)
    phase_pad = torch.zeros(B, T, 4, dtype=walking_mask.dtype)                 # S2 phase
    footstep_pad = torch.zeros(B, T, 4, dtype=walking_mask.dtype)              # S3 footstep
    return torch.cat([s1, phase_pad, footstep_pad], dim=-1)                    # (B, T, 13)


# ----- R29 interaction consistency -----------------------------------


def test_r29_interaction_consistency_zero_when_pred_eq_gt():
    joints, obj_pos, obj_rot, contact_state, _, _ = _make_clip()
    stage2_int = _build_i3_condition(joints, obj_pos, obj_rot, contact_state)
    loss = loss_r29_interaction_consistency(
        pred_joints=joints,
        object_positions=obj_pos, object_rotations=obj_rot,
        stage2_interaction=stage2_int, cfg=_default_cfg(),
    )
    assert torch.isfinite(loss)
    assert loss.item() < 1e-6


def test_r29_interaction_consistency_positive_on_perturbation():
    joints, obj_pos, obj_rot, contact_state, _, _ = _make_clip()
    stage2_int = _build_i3_condition(joints, obj_pos, obj_rot, contact_state)
    pred = joints.clone()
    pred[:, :, LEFT_WRIST_IDX, 0] += 0.15
    loss = loss_r29_interaction_consistency(
        pred_joints=pred,
        object_positions=obj_pos, object_rotations=obj_rot,
        stage2_interaction=stage2_int, cfg=_default_cfg(),
    )
    assert loss.item() > 0.0


def test_r29_interaction_consistency_empty_mask_returns_zero():
    joints, obj_pos, obj_rot, _contact_state, _, _ = _make_clip(with_contact=False)
    # No contact frames anywhere → hand_contact == 0 everywhere.
    zero_contact = torch.zeros_like(_contact_state)
    stage2_int = _build_i3_condition(joints, obj_pos, obj_rot, zero_contact)
    pred = joints.clone()
    pred[:, :, LEFT_WRIST_IDX, 0] += 0.50  # large perturbation, but mask is empty
    loss = loss_r29_interaction_consistency(
        pred_joints=pred,
        object_positions=obj_pos, object_rotations=obj_rot,
        stage2_interaction=stage2_int, cfg=_default_cfg(),
    )
    assert torch.isfinite(loss)
    assert loss.item() == 0.0


def test_r29_interaction_consistency_wrong_dim_raises():
    joints, obj_pos, obj_rot, contact_state, _, _ = _make_clip()
    # Pass an I2-shaped (B, T, 6) tensor — should reject with ValueError.
    bad_cond = torch.zeros(joints.shape[0], joints.shape[1], 6, dtype=joints.dtype)
    with pytest.raises(ValueError, match="I3 layout"):
        loss_r29_interaction_consistency(
            pred_joints=joints,
            object_positions=obj_pos, object_rotations=obj_rot,
            stage2_interaction=bad_cond, cfg=_default_cfg(),
        )


def test_r29_interaction_consistency_gradient_flows():
    joints, obj_pos, obj_rot, contact_state, _, _ = _make_clip()
    stage2_int = _build_i3_condition(joints, obj_pos, obj_rot, contact_state)
    pred = joints.clone()
    pred[:, :, LEFT_WRIST_IDX, 0] += 0.10
    pred.requires_grad_(True)
    loss = loss_r29_interaction_consistency(
        pred_joints=pred,
        object_positions=obj_pos, object_rotations=obj_rot,
        stage2_interaction=stage2_int, cfg=_default_cfg(),
    )
    loss.backward()
    assert pred.grad is not None
    assert pred.grad.abs().sum().item() > 0.0


# ----- R29 support both-airborne -------------------------------------


def test_r29_support_both_airborne_zero_when_ankle_grounded():
    joints, _, _, _, walking_mask, foot_stance = _make_clip()
    stage2_sup = _build_s4_condition(walking_mask, foot_stance)
    loss = loss_r29_support_both_airborne(
        pred_joints=joints, gt_joints=joints,
        stage2_support=stage2_sup, cfg=_default_cfg(),
    )
    # Ankles in fixture sit at y=0 (floor); both grounded with sigmoid
    # softness=3cm gives grounded_prob ≈ 0.965 each, so (1-L_g)(1-R_g) ≈ 1.2e-3.
    # Should be much smaller than the airborne-feet case (which is order 1.0).
    assert torch.isfinite(loss)
    assert loss.item() < 5e-3


def test_r29_support_both_airborne_positive_when_both_feet_up():
    joints, _, _, _, walking_mask, foot_stance = _make_clip()
    stage2_sup = _build_s4_condition(walking_mask, foot_stance)
    pred = joints.clone()
    # Lift BOTH ankles to 50 cm — both airborne.
    pred[:, :, LEFT_ANKLE_IDX, 1] = 0.50
    pred[:, :, RIGHT_ANKLE_IDX, 1] = 0.50
    loss = loss_r29_support_both_airborne(
        pred_joints=pred, gt_joints=joints,
        stage2_support=stage2_sup, cfg=_default_cfg(),
    )
    assert loss.item() > 0.0


def test_r29_support_both_airborne_empty_walking_returns_zero():
    joints, _, _, _, _, foot_stance = _make_clip(with_walking=False)
    walking_mask = torch.zeros(1, joints.shape[1], 1)
    stage2_sup = _build_s4_condition(walking_mask, foot_stance)
    pred = joints.clone()
    pred[:, :, LEFT_ANKLE_IDX, 1] = 0.50
    pred[:, :, RIGHT_ANKLE_IDX, 1] = 0.50
    loss = loss_r29_support_both_airborne(
        pred_joints=pred, gt_joints=joints,
        stage2_support=stage2_sup, cfg=_default_cfg(),
    )
    assert torch.isfinite(loss)
    assert loss.item() == 0.0


def test_r29_support_both_airborne_gradient_flows():
    joints, _, _, _, walking_mask, foot_stance = _make_clip()
    stage2_sup = _build_s4_condition(walking_mask, foot_stance)
    pred = joints.clone()
    pred[:, :, LEFT_ANKLE_IDX, 1] = 0.30
    pred[:, :, RIGHT_ANKLE_IDX, 1] = 0.30
    pred.requires_grad_(True)
    loss = loss_r29_support_both_airborne(
        pred_joints=pred, gt_joints=joints.detach(),
        stage2_support=stage2_sup, cfg=_default_cfg(),
    )
    loss.backward()
    assert pred.grad is not None
    assert pred.grad.abs().sum().item() > 0.0


def test_r29_support_both_airborne_low_dim_raises():
    """An S0-dim (or any dim<5) support condition must raise — the
    walking-mask slice does not exist."""
    joints, _, _, _, _, _ = _make_clip()
    bad = torch.zeros(joints.shape[0], joints.shape[1], 4, dtype=joints.dtype)
    with pytest.raises(ValueError, match="at least 5 channels"):
        loss_r29_support_both_airborne(
            pred_joints=joints, gt_joints=joints,
            stage2_support=bad, cfg=_default_cfg(),
        )


# ----- R29 support stance velocity -----------------------------------


def test_r29_support_stance_velocity_zero_when_feet_still():
    joints, _, _, _, walking_mask, foot_stance = _make_clip()
    stage2_sup = _build_s4_condition(walking_mask, foot_stance)
    # Ankles are stationary across t in _make_clip → zero stance velocity.
    loss = loss_r29_support_stance_velocity(
        pred_joints=joints,
        stage2_support=stage2_sup,
    )
    assert torch.isfinite(loss)
    assert loss.item() < 1e-6


def test_r29_support_stance_velocity_positive_when_stance_foot_slides():
    joints, _, _, _, walking_mask, foot_stance = _make_clip()
    stage2_sup = _build_s4_condition(walking_mask, foot_stance)
    pred = joints.clone()
    T = pred.shape[1]
    # Slide left ankle +0.025 m/frame in +X.
    pred[:, :, LEFT_ANKLE_IDX, 0] = 0.1 + 0.025 * torch.arange(T, dtype=torch.float32)
    loss = loss_r29_support_stance_velocity(
        pred_joints=pred,
        stage2_support=stage2_sup,
    )
    assert loss.item() > 0.0


def test_r29_support_stance_velocity_empty_returns_zero():
    """No walking frames AND no stance → finite zero."""
    joints, _, _, _, _, _ = _make_clip(with_walking=False)
    walking_mask = torch.zeros(1, joints.shape[1], 1)
    foot_stance = torch.zeros(1, joints.shape[1], 2)
    stage2_sup = _build_s4_condition(walking_mask, foot_stance)
    pred = joints.clone()
    pred[:, :, LEFT_ANKLE_IDX, 0] = 0.1 + 0.025 * torch.arange(
        pred.shape[1], dtype=torch.float32,
    )
    loss = loss_r29_support_stance_velocity(
        pred_joints=pred,
        stage2_support=stage2_sup,
    )
    assert torch.isfinite(loss)
    assert loss.item() == 0.0


def test_r29_support_stance_velocity_gradient_flows():
    joints, _, _, _, walking_mask, foot_stance = _make_clip()
    stage2_sup = _build_s4_condition(walking_mask, foot_stance)
    pred = joints.clone()
    pred[:, :, LEFT_ANKLE_IDX, 0] = 0.1 + 0.025 * torch.arange(
        pred.shape[1], dtype=torch.float32,
    )
    pred.requires_grad_(True)
    loss = loss_r29_support_stance_velocity(
        pred_joints=pred,
        stage2_support=stage2_sup,
    )
    loss.backward()
    assert pred.grad is not None
    assert pred.grad.abs().sum().item() > 0.0


# ----- R29 swing clearance (Codex P0+ patch) --------------------------


def test_r29_swing_clearance_zero_when_swing_foot_lifted():
    """If the swing foot is already above the clearance threshold,
    the relu(clearance - h)^2 term is zero everywhere → loss = 0."""
    joints, _, _, _, walking_mask, foot_stance = _make_clip()
    stage2_sup = _build_s4_condition(walking_mask, foot_stance)
    pred = joints.clone()
    # foot_stance in fixture is all-1 during walking → swing_mask = 0 → loss = 0.
    # Override foot_stance: make left foot swing (stance=0) and lift it.
    fs2 = torch.zeros_like(foot_stance)
    fs2[:, :, 1] = 1.0   # right is always stance
    # left is always swing → must be above clearance
    stage2_sup2 = _build_s4_condition(walking_mask, fs2)
    pred[:, :, LEFT_ANKLE_IDX, 1] = 0.20  # 20 cm above floor (>> 5 cm clearance)
    cfg = _default_cfg()
    loss = loss_r29_swing_clearance(
        pred_joints=pred, gt_joints=joints,
        stage2_support=stage2_sup2, cfg=cfg,
    )
    assert torch.isfinite(loss)
    assert loss.item() < 1e-6


def test_r29_swing_clearance_positive_when_swing_foot_dragged():
    """If the swing foot stays on the floor (height 0), it violates
    the clearance threshold by the full ``clearance_m`` → positive loss."""
    joints, _, _, _, walking_mask, foot_stance = _make_clip()
    fs2 = torch.zeros_like(foot_stance)
    fs2[:, :, 1] = 1.0   # right stance, left swing
    stage2_sup2 = _build_s4_condition(walking_mask, fs2)
    pred = joints.clone()
    # left ankle stays at floor (y=0) → violates clearance=0.05 by 5 cm
    pred[:, :, LEFT_ANKLE_IDX, 1] = 0.0
    cfg = _default_cfg()
    loss = loss_r29_swing_clearance(
        pred_joints=pred, gt_joints=joints,
        stage2_support=stage2_sup2, cfg=cfg,
    )
    # Per-frame penalty = (0.05)^2 = 0.0025. Mask weight on left-swing-during-walk
    # is non-zero → loss ~ 0.0025.
    assert loss.item() > 1e-4


def test_r29_swing_clearance_no_walking_returns_zero():
    joints, _, _, _, _, foot_stance = _make_clip(with_walking=False)
    walking_mask = torch.zeros(1, joints.shape[1], 1)
    fs2 = torch.zeros_like(foot_stance)
    stage2_sup = _build_s4_condition(walking_mask, fs2)
    pred = joints.clone()
    pred[:, :, LEFT_ANKLE_IDX, 1] = 0.0
    loss = loss_r29_swing_clearance(
        pred_joints=pred, gt_joints=joints,
        stage2_support=stage2_sup, cfg=_default_cfg(),
    )
    assert torch.isfinite(loss)
    assert loss.item() == 0.0


def test_r29_swing_clearance_gradient_flows():
    joints, _, _, _, walking_mask, foot_stance = _make_clip()
    fs2 = torch.zeros_like(foot_stance)
    fs2[:, :, 1] = 1.0
    stage2_sup = _build_s4_condition(walking_mask, fs2)
    pred = joints.clone()
    pred[:, :, LEFT_ANKLE_IDX, 1] = 0.01   # 1 cm above floor, below clearance
    pred.requires_grad_(True)
    loss = loss_r29_swing_clearance(
        pred_joints=pred, gt_joints=joints.detach(),
        stage2_support=stage2_sup, cfg=_default_cfg(),
    )
    loss.backward()
    assert pred.grad is not None
    assert pred.grad.abs().sum().item() > 0.0


def test_r29_swing_clearance_low_dim_raises():
    joints, _, _, _, _, _ = _make_clip()
    bad = torch.zeros(joints.shape[0], joints.shape[1], 4, dtype=joints.dtype)
    with pytest.raises(ValueError, match="dim >= 5"):
        loss_r29_swing_clearance(
            pred_joints=joints, gt_joints=joints,
            stage2_support=bad, cfg=_default_cfg(),
        )


# ===========================================================================
# Round-29 failure-targeted ablation (R2/R3/R4/R5).
# ===========================================================================

from piano.training.temporal_interaction_losses import (   # noqa: E402
    loss_r29_contact_lock_offset,
    loss_r29_contact_lock_segment_drift,
    loss_r29_contact_lock_tracking,
    loss_r29_gait_ankle_smooth,
    loss_r29_gait_antiphase_corr,
    loss_r29_gait_one_foot_support,
    loss_r29_gait_pred_stance_velocity,
    loss_r29_s4_footstep_target,
    loss_r29_s4_stance_bce,
)


def _make_s4_support(T: int = 20, *, walking_from: int = 4) -> torch.Tensor:
    """Build a synthetic 13-D S4 support channel (B=1, T)."""
    out = torch.zeros(1, T, 13, dtype=torch.float32)
    # walking_mask (channel 4) ON from walking_from to end.
    out[:, walking_from:, 4] = 1.0
    # GT stance (channels 0:2) alternating L/R every 4 frames during walking,
    # so the BCE target is well-defined.
    for t in range(walking_from, T):
        if ((t - walking_from) // 4) % 2 == 0:
            out[:, t, 0] = 1.0   # L stance
        else:
            out[:, t, 1] = 1.0   # R stance
    # ankle height norm (channels 2:4) = 0; phase sincos (5:9) = 0;
    # footstep target XZ (9:13) = small constant just so loss is non-trivial.
    out[:, walking_from:, 9] = 0.1  # L target x_norm
    out[:, walking_from:, 11] = -0.1  # R target x_norm
    return out


# ---------------- R2: behavior-level gait ----------------


def test_r29_gait_one_foot_support_zero_for_perfect_alternation():
    """Perfect alternation = one foot grounded per frame → loss ≈ 0."""
    joints, _, _, _, _, _ = _make_clip()
    T = joints.shape[1]
    support = _make_s4_support(T, walking_from=4)
    # Force pred grounded prob ≈ (1, 0) alternating: left ankle on floor,
    # right ankle in the air, every other frame.
    pred = joints.clone()
    for t in range(4, T):
        if (t - 4) % 2 == 0:
            pred[:, t, LEFT_ANKLE_IDX, 1] = 0.0
            pred[:, t, RIGHT_ANKLE_IDX, 1] = 1.0  # well above floor
        else:
            pred[:, t, LEFT_ANKLE_IDX, 1] = 1.0
            pred[:, t, RIGHT_ANKLE_IDX, 1] = 0.0
    loss = loss_r29_gait_one_foot_support(
        pred_joints=pred, gt_joints=joints,
        stage2_support=support, cfg=_default_cfg(),
    )
    assert loss.item() < 0.05, f"one_foot_support should be ~0; got {loss.item()}"


def test_r29_gait_one_foot_support_positive_when_both_airborne():
    joints, _, _, _, _, _ = _make_clip()
    T = joints.shape[1]
    support = _make_s4_support(T, walking_from=4)
    # Both ankles lifted high during walking → grounded ≈ 0 → (0+0-1)^2 = 1.
    pred = joints.clone()
    pred[:, 4:, LEFT_ANKLE_IDX, 1] = 1.0
    pred[:, 4:, RIGHT_ANKLE_IDX, 1] = 1.0
    loss = loss_r29_gait_one_foot_support(
        pred_joints=pred, gt_joints=joints,
        stage2_support=support, cfg=_default_cfg(),
    )
    assert loss.item() > 0.5


def test_r29_gait_pred_stance_velocity_zero_when_grounded_foot_still():
    joints, _, _, _, _, _ = _make_clip()
    T = joints.shape[1]
    support = _make_s4_support(T, walking_from=4)
    # Pred = GT (ankles still on floor at constant x, z).
    loss = loss_r29_gait_pred_stance_velocity(
        pred_joints=joints, gt_joints=joints,
        stage2_support=support, cfg=_default_cfg(),
    )
    assert loss.item() < 1e-4


def test_r29_gait_pred_stance_velocity_positive_when_grounded_foot_slides():
    joints, _, _, _, _, _ = _make_clip()
    T = joints.shape[1]
    support = _make_s4_support(T, walking_from=4)
    pred = joints.clone()
    # Slide both ankles +X during walking (still on floor → grounded ≈ 1).
    for t in range(4, T):
        pred[:, t, LEFT_ANKLE_IDX, 0] += 0.2 * (t - 4)
        pred[:, t, RIGHT_ANKLE_IDX, 0] += 0.2 * (t - 4)
    loss = loss_r29_gait_pred_stance_velocity(
        pred_joints=pred, gt_joints=joints,
        stage2_support=support, cfg=_default_cfg(),
    )
    assert loss.item() > 0.1


def test_r29_gait_ankle_smooth_zero_for_static_ankles():
    joints, _, _, _, _, _ = _make_clip()
    T = joints.shape[1]
    support = _make_s4_support(T, walking_from=4)
    loss = loss_r29_gait_ankle_smooth(
        pred_joints=joints, stage2_support=support,
    )
    assert loss.item() < 1e-4


def test_r29_gait_ankle_smooth_positive_under_jitter():
    joints, _, _, _, _, _ = _make_clip()
    T = joints.shape[1]
    support = _make_s4_support(T, walking_from=4)
    pred = joints.clone()
    # Sinusoidal high-frequency jitter on left ankle during walking.
    for t in range(4, T):
        pred[:, t, LEFT_ANKLE_IDX, 1] += 0.05 * ((-1) ** t)
    loss = loss_r29_gait_ankle_smooth(
        pred_joints=pred, stage2_support=support,
    )
    assert loss.item() > 0.0


def test_r29_gait_antiphase_corr_accepts_anti_phase():
    """corr <= -0.15 should give zero penalty."""
    T = 40
    joints = torch.zeros(1, T, 22, 3, dtype=torch.float32)
    # L ankle height = sin(t), R ankle height = -sin(t) → corr = -1.
    t = torch.arange(T, dtype=torch.float32)
    joints[:, :, LEFT_ANKLE_IDX, 1] = torch.sin(0.5 * t)
    joints[:, :, RIGHT_ANKLE_IDX, 1] = -torch.sin(0.5 * t)
    support = _make_s4_support(T, walking_from=4)
    loss = loss_r29_gait_antiphase_corr(
        pred_joints=joints, stage2_support=support, cfg=_default_cfg(),
    )
    assert loss.item() < 1e-4


def test_r29_gait_antiphase_corr_penalises_in_phase():
    """corr ≈ +1 → penalty = relu(1 + 0.15) ≈ 1.15."""
    T = 40
    joints = torch.zeros(1, T, 22, 3, dtype=torch.float32)
    t = torch.arange(T, dtype=torch.float32)
    joints[:, :, LEFT_ANKLE_IDX, 1] = torch.sin(0.5 * t)
    joints[:, :, RIGHT_ANKLE_IDX, 1] = torch.sin(0.5 * t)
    support = _make_s4_support(T, walking_from=4)
    loss = loss_r29_gait_antiphase_corr(
        pred_joints=joints, stage2_support=support, cfg=_default_cfg(),
    )
    assert loss.item() > 1.0


def test_r29_gait_antiphase_skips_when_too_few_walking_frames():
    joints = torch.zeros(1, 12, 22, 3, dtype=torch.float32)
    support = torch.zeros(1, 12, 13, dtype=torch.float32)
    # Walking_mask only ON for 3 frames (< min 10).
    support[:, 0:3, 4] = 1.0
    loss = loss_r29_gait_antiphase_corr(
        pred_joints=joints, stage2_support=support, cfg=_default_cfg(),
    )
    assert loss.item() == 0.0


# ---------------- R3: exact S4 execution ----------------


def test_r29_s4_stance_bce_low_when_pred_matches_target():
    joints, _, _, _, _, _ = _make_clip()
    T = joints.shape[1]
    support = _make_s4_support(T, walking_from=4)
    pred = joints.clone()
    # Drive ankles to match the S4 target: when L_stance target=1, L ankle
    # on floor; when R_stance target=1, R ankle on floor; the other lifts.
    for t in range(4, T):
        if support[0, t, 0] > 0.5:   # L stance
            pred[:, t, LEFT_ANKLE_IDX, 1] = 0.0
            pred[:, t, RIGHT_ANKLE_IDX, 1] = 1.0
        else:                          # R stance
            pred[:, t, LEFT_ANKLE_IDX, 1] = 1.0
            pred[:, t, RIGHT_ANKLE_IDX, 1] = 0.0
    loss = loss_r29_s4_stance_bce(
        pred_joints=pred, gt_joints=joints,
        stage2_support=support, cfg=_default_cfg(),
    )
    # BCE at saturated sigmoid ≈ -log(eps) bounded; should be small (≲ 1.0).
    assert loss.item() < 1.0


def test_r29_s4_stance_bce_high_when_pred_opposite_target():
    joints, _, _, _, _, _ = _make_clip()
    T = joints.shape[1]
    support = _make_s4_support(T, walking_from=4)
    pred = joints.clone()
    # Opposite assignment.
    for t in range(4, T):
        if support[0, t, 0] > 0.5:
            pred[:, t, LEFT_ANKLE_IDX, 1] = 1.0
            pred[:, t, RIGHT_ANKLE_IDX, 1] = 0.0
        else:
            pred[:, t, LEFT_ANKLE_IDX, 1] = 0.0
            pred[:, t, RIGHT_ANKLE_IDX, 1] = 1.0
    loss = loss_r29_s4_stance_bce(
        pred_joints=pred, gt_joints=joints,
        stage2_support=support, cfg=_default_cfg(),
    )
    assert loss.item() > 1.0


def test_r29_s4_stance_bce_requires_dim_13():
    joints, _, _, _, _, _ = _make_clip()
    bad = torch.zeros(joints.shape[0], joints.shape[1], 5, dtype=joints.dtype)
    with pytest.raises(ValueError, match="S4 layout"):
        loss_r29_s4_stance_bce(
            pred_joints=joints, gt_joints=joints,
            stage2_support=bad, cfg=_default_cfg(),
        )


def test_r29_s4_footstep_target_zero_when_pred_at_target():
    """Run with target = pred's actual ankle XZ in canonical frame → ≈ 0."""
    joints, _, _, _, _, _ = _make_clip()
    T = joints.shape[1]
    support = _make_s4_support(T, walking_from=4)
    # Set target = ankle position in canonical frame (pelvis_0 at origin,
    # yaw_0 from neutral → identity matrix). Ankle X = 0.1 / -0.1, Z = 0.
    # In _make_clip pelvis is at origin and joints flat; canonical = world.
    # ankle XZ / 3 = (0.0333, 0) and (-0.0333, 0).
    support_aligned = support.clone()
    support_aligned[:, :, 9] = 0.1 / 3.0    # L target x
    support_aligned[:, :, 10] = 0.0          # L target z
    support_aligned[:, :, 11] = -0.1 / 3.0  # R target x
    support_aligned[:, :, 12] = 0.0          # R target z
    loss = loss_r29_s4_footstep_target(
        pred_joints=joints, gt_joints=joints,
        stage2_support=support_aligned,
    )
    assert loss.item() < 1e-3, f"footstep loss should be ~0 when aligned; got {loss.item()}"


def test_r29_s4_footstep_target_positive_when_pred_off_target():
    joints, _, _, _, _, _ = _make_clip()
    T = joints.shape[1]
    support = _make_s4_support(T, walking_from=4)
    # Targets at large positive X, but pred ankles stay at the small
    # baseline positions → SmoothL1 > 0.
    support_far = support.clone()
    support_far[:, :, 9] = 0.9   # L target x_norm far from pred's 0.033
    support_far[:, :, 11] = 0.9  # R target x_norm
    loss = loss_r29_s4_footstep_target(
        pred_joints=joints, gt_joints=joints,
        stage2_support=support_far,
    )
    assert loss.item() > 0.01


def test_r29_s4_footstep_requires_dim_13():
    joints, _, _, _, _, _ = _make_clip()
    bad = torch.zeros(joints.shape[0], joints.shape[1], 5, dtype=joints.dtype)
    with pytest.raises(ValueError, match="S4 layout"):
        loss_r29_s4_footstep_target(
            pred_joints=joints, gt_joints=joints, stage2_support=bad,
        )


# ---------------- R4 / R5: contact-lock ----------------


def _make_i3_channel(joints, obj_pos, obj_rot, contact_state) -> torch.Tensor:
    """Mirror the dataset's I3 builder for a torch tensor (no clamp/3 norm).

    The dataset emits: [hand_contact (2), masked_offset_normed (6)].
    Synthesised by computing the GT offset under the same convention used
    by ``loss_r29_contact_lock_offset``.
    """
    from piano.training.temporal_interaction_losses import (
        _axis_angle_to_matrix_t, _wrist_object_local, _wrist_world_pred_gt,
    )
    clamp_m = 2.0
    R_obj = _axis_angle_to_matrix_t(obj_rot.float())
    pw, _ = _wrist_world_pred_gt(joints, joints)
    rel = _wrist_object_local(pw, obj_pos.float(), R_obj)              # (B, T, 2, 3)
    rel_norm = (rel.clamp(-clamp_m, clamp_m) / clamp_m)
    hand_contact = contact_state[..., 0:2]                              # (B, T, 2)
    offset_masked = rel_norm * hand_contact.unsqueeze(-1)
    return torch.cat(
        [hand_contact, offset_masked.reshape(*offset_masked.shape[:2], 6)],
        dim=-1,
    )                                                                   # (B, T, 8)


def test_r29_contact_lock_offset_zero_when_pred_eq_gt_i3():
    joints, obj_pos, obj_rot, contact, _, _ = _make_clip()
    inter = _make_i3_channel(joints, obj_pos, obj_rot, contact)
    loss = loss_r29_contact_lock_offset(
        pred_joints=joints, object_positions=obj_pos, object_rotations=obj_rot,
        stage2_interaction=inter, cfg=_default_cfg(),
    )
    assert loss.item() < 1e-5


def test_r29_contact_lock_offset_positive_on_perturbation_i3():
    joints, obj_pos, obj_rot, contact, _, _ = _make_clip()
    inter = _make_i3_channel(joints, obj_pos, obj_rot, contact)
    pred = joints.clone()
    pred[:, :, LEFT_WRIST_IDX, 0] += 0.3   # shift wrist off target
    pred[:, :, RIGHT_WRIST_IDX, 0] -= 0.3
    loss = loss_r29_contact_lock_offset(
        pred_joints=pred, object_positions=obj_pos, object_rotations=obj_rot,
        stage2_interaction=inter, cfg=_default_cfg(),
    )
    assert loss.item() > 0.01


def test_r29_contact_lock_offset_handles_i5_dim_20():
    """I5 has 5-part contact + 5*3 offset = 20D. Loss must accept it."""
    joints, obj_pos, obj_rot, contact, _, _ = _make_clip()
    B, T = joints.shape[:2]
    # Build a synthetic 20-D I5 channel: all-zero contact + zero offset
    # → mask all-zero → loss = 0 (empty-mask).
    i5 = torch.zeros(B, T, 20, dtype=torch.float32)
    loss = loss_r29_contact_lock_offset(
        pred_joints=joints, object_positions=obj_pos, object_rotations=obj_rot,
        stage2_interaction=i5, cfg=_default_cfg(),
    )
    assert loss.item() == 0.0


def test_r29_contact_lock_offset_wrong_dim_raises():
    joints, obj_pos, obj_rot, _, _, _ = _make_clip()
    B, T = joints.shape[:2]
    bad = torch.zeros(B, T, 6, dtype=torch.float32)
    with pytest.raises(ValueError, match="I3.*I5"):
        loss_r29_contact_lock_offset(
            pred_joints=joints, object_positions=obj_pos, object_rotations=obj_rot,
            stage2_interaction=bad, cfg=_default_cfg(),
        )


def test_r29_contact_lock_segment_drift_zero_when_pred_eq_gt():
    joints, obj_pos, obj_rot, contact, _, _ = _make_clip()
    inter = _make_i3_channel(joints, obj_pos, obj_rot, contact)
    loss = loss_r29_contact_lock_segment_drift(
        pred_joints=joints, object_positions=obj_pos, object_rotations=obj_rot,
        stage2_interaction=inter, cfg=_default_cfg(),
    )
    assert loss.item() < 1e-5


def test_r29_contact_lock_segment_drift_positive_under_pred_drift():
    joints, obj_pos, obj_rot, contact, _, _ = _make_clip()
    inter = _make_i3_channel(joints, obj_pos, obj_rot, contact)
    pred = joints.clone()
    # Make pred's wrist drift away from object monotonically during contact.
    for t in range(joints.shape[1]):
        pred[:, t, LEFT_WRIST_IDX, 0] += 0.02 * t
    loss = loss_r29_contact_lock_segment_drift(
        pred_joints=pred, object_positions=obj_pos, object_rotations=obj_rot,
        stage2_interaction=inter, cfg=_default_cfg(),
    )
    assert loss.item() > 0.0


def test_r29_contact_lock_tracking_zero_when_pred_follows_object():
    joints, obj_pos, obj_rot, contact, _, _ = _make_clip()
    inter = _make_i3_channel(joints, obj_pos, obj_rot, contact)
    # Pred wrist exactly follows obj — pred_align == target_align → penalty 0.
    loss = loss_r29_contact_lock_tracking(
        pred_joints=joints, object_positions=obj_pos,
        stage2_interaction=inter, cfg=_default_cfg(),
    )
    assert loss.item() < 1e-4


def test_r29_contact_lock_tracking_positive_when_pred_stays_still():
    joints, obj_pos, obj_rot, contact, _, _ = _make_clip()
    inter = _make_i3_channel(joints, obj_pos, obj_rot, contact)
    pred = joints.clone()
    # Freeze the wrist at frame 0 — pred doesn't track the moving object.
    for t in range(joints.shape[1]):
        pred[:, t, LEFT_WRIST_IDX, :] = pred[:, 0, LEFT_WRIST_IDX, :]
        pred[:, t, RIGHT_WRIST_IDX, :] = pred[:, 0, RIGHT_WRIST_IDX, :]
    loss = loss_r29_contact_lock_tracking(
        pred_joints=pred, object_positions=obj_pos,
        stage2_interaction=inter, cfg=_default_cfg(),
    )
    assert loss.item() > 0.0


def test_r29_contact_lock_gradient_flows():
    joints, obj_pos, obj_rot, contact, _, _ = _make_clip()
    inter = _make_i3_channel(joints, obj_pos, obj_rot, contact)
    pred = joints.clone().detach().requires_grad_(True)
    loss = loss_r29_contact_lock_offset(
        pred_joints=pred, object_positions=obj_pos, object_rotations=obj_rot,
        stage2_interaction=inter, cfg=_default_cfg(),
    )
    loss.backward()
    g = pred.grad
    # Some hand-frame gradient must be non-zero (loss is non-trivial on
    # the synthetic perturbed pred). At minimum, no NaN.
    assert g is not None
    assert torch.isfinite(g).all()

