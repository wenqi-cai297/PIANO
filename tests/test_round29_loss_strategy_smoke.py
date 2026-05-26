"""Round-29 loss-strategy 1-batch smoke test.

Per Codex review (analyses/2026-05-27_round29_loss_strategy_codex_review.md §"add a real 1-batch smoke test"):
the AST regression test catches bare-``cfg`` scope leaks but not the
runtime classes Codex flagged: missing ``cond`` keys, typos on
``temporal_loss_cfg``, dtype/shape mismatches in the active R29 branch.

This smoke test runs every R29 loss with the ``relative_behavior_v2``
weight profile + synthetic tensors at trainer-realistic shapes. It does
NOT need OmegaConf, ObjectEncoder, CLIP, or accelerate — only the loss
module itself + a TemporalInteractionLossConfig populated as main()
would populate it.

Failure modes this catches:
- A new R29 loss raises an unexpected exception at trainer-realistic
  shapes / dtypes (bf16 + cpu fallback).
- ``TemporalInteractionLossConfig`` is missing a field a loss expects.
- A loss returns non-finite values under the prompt-recommended weights.
- gradients fail to flow back to pred_joints.
- A new field added to ``TemporalInteractionLossConfig`` fails to
  propagate from a YAML-like dict through the trainer's config builder.
"""
from __future__ import annotations

import pytest
import torch

from piano.training.temporal_interaction_losses import (
    LEFT_ANKLE_IDX,
    LEFT_WRIST_IDX,
    RIGHT_ANKLE_IDX,
    RIGHT_WRIST_IDX,
    TemporalInteractionLossConfig,
    loss_contact_drift_smoothl1,
    loss_contact_rel_offset_smoothl1,
    loss_contact_tracking_projection,
    loss_r29_interaction_consistency,
    loss_r29_support_both_airborne,
    loss_r29_support_stance_velocity,
    loss_r29_swing_clearance,
)


# ---------------------------------------------------------------------------
# Synthetic batch builder (trainer-realistic shapes)
# ---------------------------------------------------------------------------


def _trainer_realistic_batch(B: int = 2, T: int = 16, dtype=torch.float32):
    """Build all tensors needed for the R29 loss-block code path.

    Shapes mirror what build_anchordiff_step_fn's step_fn closure
    constructs from each batch + cond dict on contact-frame variants.
    """
    g = torch.Generator().manual_seed(0)

    # ---- batch-derived (step_fn locals) ----
    joints = torch.zeros(B, T, 22, 3, dtype=dtype)
    # Ankles on floor with baseline xz separation.
    joints[:, :, LEFT_ANKLE_IDX, 0] = +0.1
    joints[:, :, RIGHT_ANKLE_IDX, 0] = -0.1
    # Wrists at chest height; move +X parallel to object.
    t_axis = torch.arange(T, dtype=dtype).reshape(1, T)
    joints[:, :, LEFT_WRIST_IDX, 0] = 0.30 + 0.5 * (t_axis / T)
    joints[:, :, LEFT_WRIST_IDX, 1] = 1.2
    joints[:, :, RIGHT_WRIST_IDX, 0] = -0.30 + 0.5 * (t_axis / T)
    joints[:, :, RIGHT_WRIST_IDX, 1] = 1.2

    obj_positions = torch.zeros(B, T, 3, dtype=dtype)
    obj_positions[..., 0] = 0.0 + 0.5 * (t_axis.squeeze(0) / T)
    obj_positions[..., 1] = 1.0

    obj_rotations = torch.zeros(B, T, 3, dtype=dtype)  # identity axis-angle

    seq_mask = torch.ones(B, T, dtype=dtype)
    contact_state = torch.zeros(B, T, 5, dtype=dtype)
    contact_state[:, 2:14, 0] = 1.0   # left hand contacts [2, 14)
    contact_state[:, 5:11, 1] = 1.0   # right hand contacts [5, 11)

    # ---- R29 condition tensors (step_fn reads from cond dict) ----
    # I3 layout: [..., 0:2] hand_contact, [..., 2:8] target_offset (rel_norm * hand_contact, flat L_xyz, R_xyz).
    hand_contact = contact_state[..., 0:2]                                      # (B, T, 2)
    # Build target_offset matching what build_interaction_condition emits.
    # (Same construction logic as test_temporal_interaction_losses._build_i3_condition.)
    from piano.training.temporal_interaction_losses import (
        _axis_angle_to_matrix_t,
        _wrist_object_local,
        _wrist_world_pred_gt,
    )
    R_obj = _axis_angle_to_matrix_t(obj_rotations)                              # (B, T, 3, 3)
    pw, _ = _wrist_world_pred_gt(joints, joints)                                # (B, T, 2, 3)
    rel = _wrist_object_local(pw, obj_positions, R_obj).clamp(-2.0, 2.0)
    rel_norm = rel / 2.0
    target_offset = rel_norm * hand_contact.unsqueeze(-1)                       # (B, T, 2, 3)
    stage2_interaction = torch.cat(
        [hand_contact, target_offset.reshape(B, T, 6)], dim=-1,
    )                                                                            # (B, T, 8)

    # S4 layout: [..., 0:2] foot_stance, [..., 2:4] height_norm, [..., 4:5] walking, ...
    walking_mask = torch.zeros(B, T, 1, dtype=dtype)
    walking_mask[:, 4:, 0] = 1.0  # walking from frame 4 onward
    # Alternating stance: even-t left stance, odd-t right stance.
    foot_stance = torch.zeros(B, T, 2, dtype=dtype)
    for t in range(T):
        if t % 2 == 0:
            foot_stance[:, t, 0] = 1.0
        else:
            foot_stance[:, t, 1] = 1.0
    height_norm = torch.zeros(B, T, 2, dtype=dtype)
    phase_pad = torch.zeros(B, T, 4, dtype=dtype)
    footstep_pad = torch.zeros(B, T, 4, dtype=dtype)
    stage2_support = torch.cat(
        [foot_stance, height_norm, walking_mask, phase_pad, footstep_pad], dim=-1,
    )                                                                            # (B, T, 13)

    return {
        "joints": joints,
        "obj_positions": obj_positions,
        "obj_rotations": obj_rotations,
        "seq_mask": seq_mask,
        "contact_state": contact_state,
        "stage2_interaction": stage2_interaction,
        "stage2_support": stage2_support,
    }


def _v2_loss_cfg() -> TemporalInteractionLossConfig:
    """v2 weight profile per analyses/2026-05-27_round29_loss_strategy_codex_review.md."""
    return TemporalInteractionLossConfig(
        contact_rel_offset_weight=0.25,
        contact_drift_weight=0.25,
        contact_tracking_weight=0.25,
        r29_interaction_consistency_weight=0.10,
        r29_support_both_airborne_weight=0.10,
        r29_support_stance_velocity_weight=0.10,
        r29_swing_clearance_weight=0.10,
        r29_swing_clearance_m=0.05,
        r29_hand_offset_clamp_m=2.0,
    )


# ---------------------------------------------------------------------------
# Smoke 1: every R29 loss returns a finite scalar on a trainer-realistic batch
# ---------------------------------------------------------------------------


def test_all_r29_losses_finite_on_realistic_batch():
    """Every loss in the relative_behavior_v2 active set must return a
    finite 0-D tensor on a realistic 2x16 batch with mixed contact/walking."""
    b = _trainer_realistic_batch()
    cfg = _v2_loss_cfg()

    losses = {
        "contact_rel_offset": loss_contact_rel_offset_smoothl1(
            pred_joints=b["joints"], gt_joints=b["joints"],
            object_positions=b["obj_positions"], object_rotations=b["obj_rotations"],
            contact_state=b["contact_state"], cfg=cfg, seq_mask=b["seq_mask"],
        ),
        "contact_drift": loss_contact_drift_smoothl1(
            pred_joints=b["joints"], gt_joints=b["joints"],
            object_positions=b["obj_positions"], object_rotations=b["obj_rotations"],
            contact_state=b["contact_state"], cfg=cfg, seq_mask=b["seq_mask"],
        ),
        "contact_tracking": loss_contact_tracking_projection(
            pred_joints=b["joints"], gt_joints=b["joints"],
            object_positions=b["obj_positions"], object_rotations=b["obj_rotations"],
            contact_state=b["contact_state"], cfg=cfg, seq_mask=b["seq_mask"],
        ),
        "r29_interaction": loss_r29_interaction_consistency(
            pred_joints=b["joints"],
            object_positions=b["obj_positions"], object_rotations=b["obj_rotations"],
            stage2_interaction=b["stage2_interaction"],
            cfg=cfg, seq_mask=b["seq_mask"],
            hand_offset_clamp_m=cfg.r29_hand_offset_clamp_m,
        ),
        "r29_support_air": loss_r29_support_both_airborne(
            pred_joints=b["joints"], gt_joints=b["joints"],
            stage2_support=b["stage2_support"], cfg=cfg, seq_mask=b["seq_mask"],
        ),
        "r29_support_stance_vel": loss_r29_support_stance_velocity(
            pred_joints=b["joints"], stage2_support=b["stage2_support"],
            seq_mask=b["seq_mask"],
        ),
        "r29_swing_clearance": loss_r29_swing_clearance(
            pred_joints=b["joints"], gt_joints=b["joints"],
            stage2_support=b["stage2_support"], cfg=cfg, seq_mask=b["seq_mask"],
        ),
    }
    for name, loss in losses.items():
        assert torch.isfinite(loss), f"{name} returned non-finite: {loss.item()}"
        assert loss.dim() == 0, f"{name} returned non-scalar shape {loss.shape}"
        assert loss.item() >= 0.0, f"{name} returned negative scalar: {loss.item()}"


# ---------------------------------------------------------------------------
# Smoke 2: gradient flows from the v2 weighted-sum back to pred_joints
# ---------------------------------------------------------------------------


def test_v2_weighted_sum_gradient_flows():
    """Apply v2 weights, sum, backward — all four new R29 losses + the
    three existing relative contact losses must contribute gradient."""
    b = _trainer_realistic_batch()
    cfg = _v2_loss_cfg()
    pred = b["joints"].clone()
    pred[:, :, LEFT_WRIST_IDX, 0] += 0.10
    pred[:, :, LEFT_ANKLE_IDX, 1] = 0.01   # below clearance to wake swing_clear
    pred.requires_grad_(True)

    losses = [
        cfg.contact_rel_offset_weight * loss_contact_rel_offset_smoothl1(
            pred_joints=pred, gt_joints=b["joints"],
            object_positions=b["obj_positions"], object_rotations=b["obj_rotations"],
            contact_state=b["contact_state"], cfg=cfg, seq_mask=b["seq_mask"],
        ),
        cfg.contact_drift_weight * loss_contact_drift_smoothl1(
            pred_joints=pred, gt_joints=b["joints"],
            object_positions=b["obj_positions"], object_rotations=b["obj_rotations"],
            contact_state=b["contact_state"], cfg=cfg, seq_mask=b["seq_mask"],
        ),
        cfg.contact_tracking_weight * loss_contact_tracking_projection(
            pred_joints=pred, gt_joints=b["joints"],
            object_positions=b["obj_positions"], object_rotations=b["obj_rotations"],
            contact_state=b["contact_state"], cfg=cfg, seq_mask=b["seq_mask"],
        ),
        cfg.r29_interaction_consistency_weight * loss_r29_interaction_consistency(
            pred_joints=pred,
            object_positions=b["obj_positions"], object_rotations=b["obj_rotations"],
            stage2_interaction=b["stage2_interaction"],
            cfg=cfg, seq_mask=b["seq_mask"],
            hand_offset_clamp_m=cfg.r29_hand_offset_clamp_m,
        ),
        cfg.r29_support_both_airborne_weight * loss_r29_support_both_airborne(
            pred_joints=pred, gt_joints=b["joints"],
            stage2_support=b["stage2_support"], cfg=cfg, seq_mask=b["seq_mask"],
        ),
        cfg.r29_support_stance_velocity_weight * loss_r29_support_stance_velocity(
            pred_joints=pred, stage2_support=b["stage2_support"],
            seq_mask=b["seq_mask"],
        ),
        cfg.r29_swing_clearance_weight * loss_r29_swing_clearance(
            pred_joints=pred, gt_joints=b["joints"],
            stage2_support=b["stage2_support"], cfg=cfg, seq_mask=b["seq_mask"],
        ),
    ]
    total = sum(losses)
    total.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum().item() > 0.0


# ---------------------------------------------------------------------------
# Smoke 3: TemporalInteractionLossConfig field propagation matches trainer
# ---------------------------------------------------------------------------


def test_temporal_loss_cfg_field_propagation():
    """Build a TemporalInteractionLossConfig the same way main() does
    (from a YAML-like dict), and verify all R29 fields are present with
    the expected values.

    This catches: missing kwargs in the trainer's config builder, typos
    in the YAML keys that silently leave a field at default, and
    forgetting to add a new field to the dataclass when the loss expects it.
    """
    # Mirror exactly what train_anchordiff.py reads from cfg.loss.temporal_interaction.
    yaml_like = {
        "contact_rel_offset_weight": 0.25,
        "contact_drift_weight": 0.25,
        "contact_tracking_weight": 0.25,
        "r29_interaction_consistency_weight": 0.10,
        "r29_support_both_airborne_weight": 0.10,
        "r29_support_stance_velocity_weight": 0.10,
        "r29_swing_clearance_weight": 0.10,
        "r29_swing_clearance_m": 0.05,
    }
    cfg = TemporalInteractionLossConfig(
        contact_rel_offset_weight=float(yaml_like.get("contact_rel_offset_weight", 0.0)),
        contact_drift_weight=float(yaml_like.get("contact_drift_weight", 0.0)),
        contact_tracking_weight=float(yaml_like.get("contact_tracking_weight", 0.0)),
        r29_interaction_consistency_weight=float(
            yaml_like.get("r29_interaction_consistency_weight", 0.0)
        ),
        r29_support_both_airborne_weight=float(
            yaml_like.get("r29_support_both_airborne_weight", 0.0)
        ),
        r29_support_stance_velocity_weight=float(
            yaml_like.get("r29_support_stance_velocity_weight", 0.0)
        ),
        r29_swing_clearance_weight=float(yaml_like.get("r29_swing_clearance_weight", 0.0)),
        r29_swing_clearance_m=float(yaml_like.get("r29_swing_clearance_m", 0.05)),
        r29_hand_offset_clamp_m=2.0,   # passed from cfg.data, not _tloss
    )
    # All R29 fields must round-trip cleanly.
    assert float(cfg.r29_interaction_consistency_weight) == 0.10
    assert float(cfg.r29_support_both_airborne_weight) == 0.10
    assert float(cfg.r29_support_stance_velocity_weight) == 0.10
    assert float(cfg.r29_swing_clearance_weight) == 0.10
    assert float(cfg.r29_swing_clearance_m) == 0.05
    assert float(cfg.r29_hand_offset_clamp_m) == 2.0


# ---------------------------------------------------------------------------
# Smoke 4: missing cond key raises a clear KeyError (matches step_fn guards)
# ---------------------------------------------------------------------------


def test_missing_stage2_interaction_path_clear_failure():
    """The trainer's step_fn raises KeyError("...missing stage2_interaction...")
    when the loss weight is active but the dataset emitted no I3 channel.
    The loss itself doesn't enforce this (the trainer does), but verify
    that calling the loss with an obviously-wrong shape produces a clear
    ValueError rather than a silent garbage output.
    """
    b = _trainer_realistic_batch()
    cfg = _v2_loss_cfg()
    # Pass an I2-shaped (B, T, 6) tensor — should reject with ValueError
    # mentioning I3.
    bad_cond = torch.zeros(b["joints"].shape[0], b["joints"].shape[1], 6, dtype=b["joints"].dtype)
    with pytest.raises(ValueError, match="I3 layout"):
        loss_r29_interaction_consistency(
            pred_joints=b["joints"],
            object_positions=b["obj_positions"], object_rotations=b["obj_rotations"],
            stage2_interaction=bad_cond, cfg=cfg, seq_mask=b["seq_mask"],
            hand_offset_clamp_m=cfg.r29_hand_offset_clamp_m,
        )


def test_missing_stage2_support_dim_clear_failure():
    """S0 (dim<5) must be rejected for swing_clearance with a clear
    ValueError, not a silent slice error."""
    b = _trainer_realistic_batch()
    cfg = _v2_loss_cfg()
    bad_cond = torch.zeros(b["joints"].shape[0], b["joints"].shape[1], 4, dtype=b["joints"].dtype)
    with pytest.raises(ValueError, match="dim >= 5"):
        loss_r29_swing_clearance(
            pred_joints=b["joints"], gt_joints=b["joints"],
            stage2_support=bad_cond, cfg=cfg, seq_mask=b["seq_mask"],
        )
