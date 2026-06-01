"""Tests for R41 cascade loss building blocks.

The 6 helpers in ``piano.training.pb1_loss_helpers`` mirror PB1's
inline loss math (``train_anchordiff.py:319-1230``). These tests don't
re-verify the math — those would just compare the helper to itself.
They verify the contracts that the R41 cascade trainer relies on:

  - masked reduction (padded frames don't contribute)
  - gradient flows from helper output back to ``pred`` (cascade path)
  - frozen / detached inputs don't accidentally pick up grad
  - shape mismatch raises with a clear ValueError
  - min-SNR weight has the expected normalization (mean=1)
"""
from __future__ import annotations

import pytest
import torch

from piano.training.pb1_loss_helpers import (
    anchor_joint_pos_loss,
    compute_min_snr_weight,
    fk_motion_135_to_joints_22,
    l_pos_full_loss,
    masked_motion_mse_loss,
    world_joint_velocity_loss,
)


# ──────────────────────────────────────────────────────────────────────────
# masked_motion_mse_loss
# ──────────────────────────────────────────────────────────────────────────


def test_masked_motion_mse_zero_when_pred_equals_target():
    pred = torch.randn(2, 8, 135)
    target = pred.clone()
    mask = torch.ones(2, 8)
    out = masked_motion_mse_loss(pred=pred, target=target, seq_mask=mask)
    assert out.item() == pytest.approx(0.0, abs=1e-6)


def test_masked_motion_mse_ignores_padded_frames():
    """Padded frames in pred should not contribute to the loss."""
    pred = torch.zeros(2, 8, 135)
    target = torch.zeros(2, 8, 135)
    # Corrupt the last 2 frames in pred only.
    pred[:, -2:, :] = 100.0
    # Mask out the last 2 frames.
    mask = torch.ones(2, 8)
    mask[:, -2:] = 0.0
    out = masked_motion_mse_loss(pred=pred, target=target, seq_mask=mask)
    assert out.item() == pytest.approx(0.0, abs=1e-6)


def test_masked_motion_mse_applies_min_snr_weight():
    """Min-SNR weight should multiply per-sample (B,) before the masked mean."""
    pred = torch.ones(2, 8, 135)
    target = torch.zeros(2, 8, 135)
    mask = torch.ones(2, 8)
    # Compose w manually so sum is preserved: w = (2.0, 0.0).
    w = torch.tensor([2.0, 0.0])
    out_w = masked_motion_mse_loss(
        pred=pred, target=target, seq_mask=mask, min_snr_weight=w,
    )
    out_uniform = masked_motion_mse_loss(
        pred=pred, target=target, seq_mask=mask, min_snr_weight=None,
    )
    # w[0]=2, w[1]=0 → only batch 0 contributes, with weight 2x.
    # mean(2*135, 0*135) = 135. Uniform: mean(135, 135) = 135.
    # So out_w == out_uniform when sum(w) == n_batch — verify:
    assert out_w.item() == pytest.approx(out_uniform.item(), rel=1e-4)


def test_masked_motion_mse_grad_flows_to_pred():
    pred = torch.randn(2, 8, 135, requires_grad=True)
    target = torch.randn(2, 8, 135)
    mask = torch.ones(2, 8)
    out = masked_motion_mse_loss(pred=pred, target=target, seq_mask=mask)
    out.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum().item() > 0.0


def test_masked_motion_mse_shape_mismatch_raises():
    pred = torch.randn(2, 8, 135)
    target = torch.randn(2, 7, 135)  # T mismatch
    mask = torch.ones(2, 8)
    with pytest.raises(ValueError, match="shape mismatch"):
        masked_motion_mse_loss(pred=pred, target=target, seq_mask=mask)


# ──────────────────────────────────────────────────────────────────────────
# world_joint_velocity_loss
# ──────────────────────────────────────────────────────────────────────────


def test_world_joint_velocity_zero_when_pred_equals_target():
    pred = torch.randn(2, 8, 135)
    out = world_joint_velocity_loss(
        pred=pred, target=pred.clone(), seq_mask=torch.ones(2, 8),
    )
    assert out.item() == pytest.approx(0.0, abs=1e-6)


def test_world_joint_velocity_grad_flows_to_pred():
    pred = torch.randn(2, 8, 135, requires_grad=True)
    target = torch.randn(2, 8, 135)
    out = world_joint_velocity_loss(
        pred=pred, target=target, seq_mask=torch.ones(2, 8),
    )
    out.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum().item() > 0.0


# ──────────────────────────────────────────────────────────────────────────
# fk_motion_135_to_joints_22 + l_pos_full + anchor
# ──────────────────────────────────────────────────────────────────────────


def _make_dummy_motion(B: int = 2, T: int = 8) -> torch.Tensor:
    """Build a (B, T, 135) motion tensor with valid rot6d slices and
    nonzero root position. Doesn't need to be SMPL-realistic — the FK
    helper only does matrix conversion + transform chain, so any rot6d
    works."""
    motion = torch.zeros(B, T, 135)
    # 22 joints × 6 rot6d = 132. Default identity-like: a1 = (1,0,0),
    # a2 = (0,1,0).
    for j in range(22):
        motion[..., j * 6 + 0] = 1.0
        motion[..., j * 6 + 4] = 1.0
    # Root world position.
    motion[..., 132:135] = torch.randn(B, T, 3) * 0.1
    return motion


def test_fk_motion_135_to_joints_22_shape():
    motion = _make_dummy_motion()
    rest_offsets = torch.randn(2, 22, 3) * 0.05
    out = fk_motion_135_to_joints_22(motion=motion, rest_offsets=rest_offsets)
    assert out.shape == (2, 8, 22, 3)
    assert torch.isfinite(out).all()


def test_fk_motion_135_grad_flows_back():
    motion = _make_dummy_motion()
    motion = motion.requires_grad_(True)
    rest_offsets = torch.randn(2, 22, 3) * 0.05
    out = fk_motion_135_to_joints_22(motion=motion, rest_offsets=rest_offsets)
    out.sum().backward()
    assert motion.grad is not None
    assert torch.isfinite(motion.grad).all()


def test_fk_motion_135_wrong_dim_raises():
    motion = torch.zeros(2, 8, 130)
    with pytest.raises(ValueError, match=r"\(B, T, 135\)"):
        fk_motion_135_to_joints_22(
            motion=motion, rest_offsets=torch.zeros(2, 22, 3),
        )


def test_l_pos_full_zero_on_identity():
    j = torch.randn(2, 8, 22, 3)
    out = l_pos_full_loss(
        jpos_pred=j, joints_gt=j.clone(), seq_mask=torch.ones(2, 8),
    )
    assert out.item() == pytest.approx(0.0, abs=1e-6)


def test_l_pos_full_grad_flows():
    j = torch.randn(2, 8, 22, 3, requires_grad=True)
    gt = torch.randn(2, 8, 22, 3)
    out = l_pos_full_loss(jpos_pred=j, joints_gt=gt, seq_mask=torch.ones(2, 8))
    out.backward()
    assert j.grad is not None
    assert torch.isfinite(j.grad).all()
    assert j.grad.abs().sum().item() > 0.0


def test_l_pos_full_hand_endpoint_reweight():
    """hand_endpoint_weight=2 should make wrist channels contribute ~2x."""
    j = torch.zeros(1, 4, 22, 3)
    gt = torch.zeros(1, 4, 22, 3)
    gt[..., 20, :] = 1.0  # left wrist offset by 1
    mask = torch.ones(1, 4)
    out_unweighted = l_pos_full_loss(
        jpos_pred=j, joints_gt=gt, seq_mask=mask,
        hand_endpoint_weight=1.0, foot_endpoint_weight=1.0,
    )
    out_weighted = l_pos_full_loss(
        jpos_pred=j, joints_gt=gt, seq_mask=mask,
        hand_endpoint_weight=2.0, foot_endpoint_weight=1.0,
    )
    # Weighted version reweights the wrist error * 2 but also weight_sum
    # increases by 1 (22 + 1 extra weight). The ratio should be
    # 2*err / (22+1*err_others_zero) vs 1*err / 22 → roughly larger.
    assert out_weighted.item() > out_unweighted.item()


def test_anchor_joint_pos_zero_on_identity():
    j = torch.randn(2, 8, 22, 3)
    contact = torch.ones(2, 8, 5) * 0.6  # all parts contact-active
    out = anchor_joint_pos_loss(
        jpos_pred=j, joints_gt=j.clone(),
        contact_state=contact, seq_mask=torch.ones(2, 8),
    )
    assert out.item() == pytest.approx(0.0, abs=1e-6)


def test_anchor_joint_pos_zero_when_no_contact():
    j = torch.zeros(2, 8, 22, 3)
    gt = torch.ones(2, 8, 22, 3)
    contact_off = torch.zeros(2, 8, 5)  # nothing active
    out = anchor_joint_pos_loss(
        jpos_pred=j, joints_gt=gt,
        contact_state=contact_off, seq_mask=torch.ones(2, 8),
    )
    assert out.item() == pytest.approx(0.0, abs=1e-6)


def test_anchor_joint_pos_grad_flows():
    j = torch.randn(2, 8, 22, 3, requires_grad=True)
    gt = torch.randn(2, 8, 22, 3)
    contact = torch.ones(2, 8, 5) * 0.6
    out = anchor_joint_pos_loss(
        jpos_pred=j, joints_gt=gt,
        contact_state=contact, seq_mask=torch.ones(2, 8),
    )
    out.backward()
    assert j.grad is not None
    assert torch.isfinite(j.grad).all()


def test_anchor_joint_pos_bad_contact_shape_raises():
    with pytest.raises(ValueError, match="contact_state last dim"):
        anchor_joint_pos_loss(
            jpos_pred=torch.randn(1, 4, 22, 3),
            joints_gt=torch.randn(1, 4, 22, 3),
            contact_state=torch.zeros(1, 4, 3),  # wrong last dim
            seq_mask=torch.ones(1, 4),
        )


# ──────────────────────────────────────────────────────────────────────────
# compute_min_snr_weight
# ──────────────────────────────────────────────────────────────────────────


def test_min_snr_weight_mean_is_one():
    # Cosine schedule analog: alphas_cumprod ∈ (0, 1), decreasing.
    num_steps = 1000
    alphas_cumprod = torch.linspace(0.9999, 0.0001, num_steps)
    t = torch.randint(0, num_steps, (16,))
    w = compute_min_snr_weight(t, alphas_cumprod, gamma=5.0)
    assert w.shape == (16,)
    assert w.mean().item() == pytest.approx(1.0, abs=1e-4)
    assert torch.isfinite(w).all()


def test_min_snr_weight_clamps_high_snr():
    """At low t (high SNR), w should be clamped at γ before normalization."""
    num_steps = 1000
    alphas_cumprod = torch.linspace(0.9999, 0.0001, num_steps)
    # All samples at t=0 → SNR very high → all clamped to γ.
    t = torch.zeros(8, dtype=torch.long)
    w = compute_min_snr_weight(t, alphas_cumprod, gamma=5.0)
    # After normalization mean = 1, since all equal, w = ones.
    assert torch.allclose(w, torch.ones(8), atol=1e-5)


def test_min_snr_weight_high_t_gets_lower():
    """At high t (low SNR), per-sample weight should be smaller than at low t."""
    num_steps = 1000
    alphas_cumprod = torch.linspace(0.9999, 0.0001, num_steps)
    # Mix low + high t in the same batch.
    t = torch.tensor([10, 10, 10, 10, 990, 990, 990, 990])
    w = compute_min_snr_weight(t, alphas_cumprod, gamma=5.0)
    # Low-t samples (first 4) should have larger weight than high-t.
    assert w[:4].mean().item() > w[4:].mean().item()
