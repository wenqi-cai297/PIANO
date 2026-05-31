"""Tests for stage1p5_losses helpers (R32 V7 anti-bug matrix).

Each helper tested for: zero on canonical input, positive on perturbation,
correct gradient flow, mask behaviour, and any helper-specific invariants.
"""
from __future__ import annotations

import math

import pytest
import torch

from piano.training.stage1p5_losses import (
    CH_C41_WRIST,
    CH_S4_PHASE_L_COS,
    CH_S4_PHASE_L_SIN,
    CH_S4_PHASE_R_COS,
    CH_S4_PHASE_R_SIN,
    C41_DIM,
    R34_C41_WRIST_SLICE,
    R37_C41_ALL_SLICE,
    R37_C41_KNEE_SLICE,
    R37_C41_LEFT_KNEE_SLICE,
    R37_C41_LEFT_WRIST_SLICE,
    R37_C41_NECK_SLICE,
    R37_C41_PELVIS_SLICE,
    R37_C41_RIGHT_KNEE_SLICE,
    R37_C41_RIGHT_WRIST_SLICE,
    R37_C41_WRIST_SLICE,
    S4_DIM,
    TOTAL_DIM,
    apply_stage1_coarse_cond_aug,
    build_r37_group_masks,
    c41_contact_window_wrist_loss,
    c41_smoothl1_finite_diff_cm,
    c41_speed_moment_cm,
    c41_temporal_derivative_loss,
    c41_wrist_frame0_consistency_loss,
    phase_unit_circle_loss,
    wrist_lowband_rfft_loss,
)


# ──────────────────────────────────────────────────────────────────────────
# Sanity: layout constants match the audit's binding
# ──────────────────────────────────────────────────────────────────────────


def test_layout_constants_match_stage1p5_design():
    """Constants are S4-LOCAL indices; the global-layout audit doc
    reports the same channels at offsets 23..26 (= 18 + these)."""
    assert C41_DIM == 18
    assert S4_DIM == 13
    assert TOTAL_DIM == 31
    assert CH_C41_WRIST == slice(0, 6)
    # Phase L is the 6th S4 channel (sin, cos), Phase R the 8th.
    assert CH_S4_PHASE_L_SIN == 5
    assert CH_S4_PHASE_L_COS == 6
    assert CH_S4_PHASE_R_SIN == 7
    assert CH_S4_PHASE_R_COS == 8
    # Their global positions in the 31-D x0 are 23..26.
    assert CH_S4_PHASE_L_SIN + C41_DIM == 23


# ──────────────────────────────────────────────────────────────────────────
# V7-B: phase_unit_circle_loss
# ──────────────────────────────────────────────────────────────────────────


def _make_phase_s4(B: int, T: int, ang_L: torch.Tensor, ang_R: torch.Tensor) -> torch.Tensor:
    """Helper: build a (B, T, 13) S4 tensor with given L+R angles, other
    channels random."""
    s4 = torch.randn(B, T, 13) * 0.1
    s4[..., CH_S4_PHASE_L_SIN] = ang_L.sin()
    s4[..., CH_S4_PHASE_L_COS] = ang_L.cos()
    s4[..., CH_S4_PHASE_R_SIN] = ang_R.sin()
    s4[..., CH_S4_PHASE_R_COS] = ang_R.cos()
    return s4


def test_phase_unit_circle_zero_when_pred_equals_gt():
    """Identical (sin, cos) → 0 unit-norm dev + 0 angle dev."""
    B, T = 2, 8
    ang_L = torch.linspace(0.0, 2 * math.pi, T).unsqueeze(0).expand(B, T)
    ang_R = ang_L + 1.0
    s4 = _make_phase_s4(B, T, ang_L, ang_R)
    mask = torch.ones(B, T)
    loss = phase_unit_circle_loss(s4, s4, mask)
    assert loss.item() < 1e-6


def test_phase_unit_circle_zero_when_pred_matches_angle_and_unit_norm():
    """Same angle, different sin/cos magnitudes scaled by 1 → 0 loss."""
    B, T = 2, 5
    ang = torch.zeros(B, T)
    s4_gt = _make_phase_s4(B, T, ang, ang)
    s4_pred = s4_gt.clone()
    loss = phase_unit_circle_loss(s4_pred, s4_gt, torch.ones(B, T))
    assert loss.item() < 1e-6


def test_phase_unit_circle_positive_on_unit_norm_violation():
    """Pred (sin, cos) scaled by 2 (off unit circle) → unit-norm term fires."""
    B, T = 2, 5
    ang = torch.linspace(0.0, math.pi, T).unsqueeze(0).expand(B, T)
    s4_gt = _make_phase_s4(B, T, ang, ang)
    s4_pred = s4_gt.clone()
    # Scale phase channels by 2 — sin² + cos² = 4 instead of 1.
    s4_pred[..., CH_S4_PHASE_L_SIN] *= 2.0
    s4_pred[..., CH_S4_PHASE_L_COS] *= 2.0
    s4_pred[..., CH_S4_PHASE_R_SIN] *= 2.0
    s4_pred[..., CH_S4_PHASE_R_COS] *= 2.0
    loss = phase_unit_circle_loss(
        s4_pred, s4_gt, torch.ones(B, T),
        unit_norm_weight=1.0, angle_weight=0.0,
    )
    # (4 - 1)² = 9 per leg, averaged across legs and frames.
    assert abs(loss.item() - 9.0) < 1e-3


def test_phase_unit_circle_positive_on_angle_offset():
    """Same unit norm but rotated 90° → angle term fires."""
    B, T = 2, 5
    ang_gt = torch.zeros(B, T)
    ang_pred = ang_gt + math.pi / 2.0
    s4_gt = _make_phase_s4(B, T, ang_gt, ang_gt)
    s4_pred = _make_phase_s4(B, T, ang_pred, ang_pred)
    loss = phase_unit_circle_loss(
        s4_pred, s4_gt, torch.ones(B, T),
        unit_norm_weight=0.0, angle_weight=1.0,
    )
    # cos(π/2) = 0 → 1 - 0 = 1 per leg, average = 1.
    assert abs(loss.item() - 1.0) < 1e-3


def test_phase_unit_circle_respects_mask():
    """Masked frames don't pollute the loss."""
    B, T = 2, 6
    ang_gt = torch.zeros(B, T)
    s4_gt = _make_phase_s4(B, T, ang_gt, ang_gt)
    # Pred matches GT on first 3 frames, big perturbation on last 3.
    s4_pred = s4_gt.clone()
    s4_pred[:, 3:, CH_S4_PHASE_L_SIN] *= 5.0
    mask = torch.zeros(B, T); mask[:, :3] = 1.0
    loss = phase_unit_circle_loss(s4_pred, s4_gt, mask)
    assert loss.item() < 1e-5


def test_phase_unit_circle_weights_zero_disable_term():
    """unit_norm_weight=0 AND angle_weight=0 → 0 loss even on perturbed input."""
    B, T = 2, 5
    ang = torch.zeros(B, T)
    s4_gt = _make_phase_s4(B, T, ang, ang)
    s4_pred = s4_gt + torch.randn_like(s4_gt) * 0.5
    loss = phase_unit_circle_loss(
        s4_pred, s4_gt, torch.ones(B, T),
        unit_norm_weight=0.0, angle_weight=0.0,
    )
    assert loss.item() == 0.0


def test_phase_unit_circle_gradient_flow():
    """Gradient flows into pred phase channels."""
    B, T = 2, 5
    ang_gt = torch.zeros(B, T)
    s4_gt = _make_phase_s4(B, T, ang_gt, ang_gt)
    s4_pred = (s4_gt + 0.1).requires_grad_(True)
    loss = phase_unit_circle_loss(s4_pred, s4_gt, torch.ones(B, T))
    loss.backward()
    assert s4_pred.grad is not None
    # Gradient should be non-zero on phase channels (S4-local 5..8).
    phase_g = s4_pred.grad[..., 5:9].abs().sum().item()
    assert phase_g > 0.0


# ──────────────────────────────────────────────────────────────────────────
# V7-D: c41_wrist_frame0_consistency_loss
# ──────────────────────────────────────────────────────────────────────────


def test_c41_frame0_zero_when_t0_wrist_is_zero():
    """C41 wrist channels at t=0 = 0 (by construction) → 0 loss."""
    B, T = 2, 5
    c41 = torch.randn(B, T, 18)
    c41[:, 0, CH_C41_WRIST] = 0.0
    loss = c41_wrist_frame0_consistency_loss(c41)
    assert loss.item() == 0.0


def test_c41_frame0_positive_when_t0_wrist_offset():
    """5 cm offset on lw at t=0 → loss = (0.05)² = 0.0025 per channel."""
    B, T = 2, 5
    c41 = torch.zeros(B, T, 18)
    c41[:, 0, 0] = 0.05      # lw_dx = 5 cm
    loss = c41_wrist_frame0_consistency_loss(c41)
    # mean over (B=2, 6 wrist channels) = (B*1) * (0.05)² / (B*6) = 0.05²/6.
    expected = 0.05 ** 2 / 6.0
    assert abs(loss.item() - expected) < 1e-6


def test_c41_frame0_only_penalises_t0():
    """Big offsets on t>0 wrist must NOT be penalised."""
    B, T = 2, 5
    c41 = torch.zeros(B, T, 18)
    c41[:, 1:, :6] = 0.5    # huge offsets but only on t>=1
    loss = c41_wrist_frame0_consistency_loss(c41)
    assert loss.item() == 0.0


def test_c41_frame0_only_penalises_wrist_channels():
    """Big offsets at t=0 on knee/neck/pelvis (channels [6:18]) → no loss."""
    B, T = 2, 5
    c41 = torch.zeros(B, T, 18)
    c41[:, 0, 6:] = 1.0    # huge offset on non-wrist at t=0
    loss = c41_wrist_frame0_consistency_loss(c41)
    assert loss.item() == 0.0


def test_c41_frame0_gradient_flow():
    """Gradient flows back through the t=0 wrist channels."""
    B, T = 2, 5
    c41 = torch.randn(B, T, 18, requires_grad=True)
    loss = c41_wrist_frame0_consistency_loss(c41)
    loss.backward()
    assert c41.grad is not None
    # Grad should be non-zero only on (b, t=0, ch in [0:6]).
    g = c41.grad
    assert g[:, 0, :6].abs().sum().item() > 0.0
    assert g[:, 1:, :].abs().sum().item() == 0.0
    assert g[:, 0, 6:].abs().sum().item() == 0.0


# ──────────────────────────────────────────────────────────────────────────
# V7 integration via channel_moment_match_loss (re-uses R31 helper)
# ──────────────────────────────────────────────────────────────────────────


def test_moment_match_handles_full_31d_output():
    """V7-A applies R31's channel_moment_match_loss to the full 31-D
    Stage-1.5 output. Confirm the helper works at that shape."""
    from piano.training.stage1_losses import channel_moment_match_loss
    B, T = 4, 50
    x0_gt = torch.randn(B, T, TOTAL_DIM) * 0.3
    x0_pred = x0_gt * 0.5     # std collapse to half
    mask = torch.ones(B, T)
    loss = channel_moment_match_loss(
        x0_pred, x0_gt, mask,
        velocity_match=True, value_match=False,
        channel_subset=None,
        normalize_by_gt_std=True,
    )
    assert loss.item() > 0.0
    assert torch.isfinite(loss)


def test_moment_match_zero_on_31d_when_pred_equals_gt():
    from piano.training.stage1_losses import channel_moment_match_loss
    B, T = 4, 20
    x0_gt = torch.randn(B, T, TOTAL_DIM)
    loss = channel_moment_match_loss(
        x0_gt, x0_gt, torch.ones(B, T),
        velocity_match=True, value_match=True,
    )
    assert loss.item() < 1e-10


# ──────────────────────────────────────────────────────────────────────────
# R34 — wrist low-band rFFT loss
# ──────────────────────────────────────────────────────────────────────────


def test_r34_wrist_lowband_slice_matches_audit():
    """R34_C41_WRIST_SLICE must match the wrist channel locations used
    by the Phase 1 audit (channels [0:6] = left+right wrist Δxyz)."""
    assert R34_C41_WRIST_SLICE == slice(0, 6)


def test_r34_wrist_lowband_zero_when_pred_equals_gt():
    """rFFT MSE on identical pred + gt should be exactly 0."""
    B, T = 2, 196
    torch.manual_seed(0)
    c41 = torch.randn(B, T, 18)
    mask = torch.ones(B, T)
    loss = wrist_lowband_rfft_loss(c41, c41, mask, fps=20.0, cutoff_hz=1.0)
    assert loss.item() < 1e-7


def test_r34_wrist_lowband_positive_on_random_perturbation():
    B, T = 2, 196
    torch.manual_seed(0)
    pred = torch.randn(B, T, 18)
    gt = torch.randn(B, T, 18)
    mask = torch.ones(B, T)
    loss = wrist_lowband_rfft_loss(pred, gt, mask, fps=20.0, cutoff_hz=1.0)
    assert loss.item() > 0.0
    assert torch.isfinite(loss)


def test_r34_wrist_lowband_only_grad_on_wrist_channels():
    """Gradient must NOT leak into non-wrist channels [6:18]."""
    B, T = 2, 196
    torch.manual_seed(0)
    pred = torch.randn(B, T, 18, requires_grad=True)
    gt = torch.randn(B, T, 18)
    mask = torch.ones(B, T)
    loss = wrist_lowband_rfft_loss(pred, gt, mask, fps=20.0, cutoff_hz=1.0)
    loss.backward()
    assert pred.grad is not None
    wrist_grad = pred.grad[..., 0:6].abs().sum().item()
    nonwrist_grad = pred.grad[..., 6:18].abs().sum().item()
    assert wrist_grad > 0.0
    assert nonwrist_grad == 0.0


def test_r34_wrist_lowband_only_targets_low_band():
    """Pred = GT in low band but different in high band → loss ~= 0.

    Construct a signal whose 0-1 Hz DFT bins match GT exactly, and
    differ only at higher freqs. The low-band rFFT MSE should be 0.
    """
    B, T, C = 1, 196, 18
    fps = 20.0
    cutoff = 1.0
    torch.manual_seed(0)
    gt = torch.randn(B, T, C)
    # Add a pure 5 Hz sinusoid to the wrist channels only, in pred only,
    # so the low-band rFFT MSE on the wrist [0:6] is unchanged.
    t = torch.arange(T, dtype=torch.float32) / fps
    high_freq = torch.sin(2 * math.pi * 5.0 * t).view(1, T, 1).expand(B, T, 6)
    pred = gt.clone()
    pred[..., 0:6] = pred[..., 0:6] + 10.0 * high_freq
    mask = torch.ones(B, T)
    loss = wrist_lowband_rfft_loss(pred, gt, mask, fps=fps, cutoff_hz=cutoff)
    # 5 Hz is well above the 1 Hz cutoff; the low-band MSE must be ~0.
    assert loss.item() < 1e-4, (
        f"high-frequency perturbation should not enter low-band loss, "
        f"got {loss.item()}"
    )


def test_r34_wrist_lowband_mask_zeros_padding():
    """seq_mask must zero out padding so the FFT spectra match between
    pred and gt in the padding region. Use mask that zeros half the
    sequence — the in-window content should drive the loss only."""
    B, T = 1, 100
    torch.manual_seed(0)
    pred = torch.randn(B, T, 18)
    gt = torch.randn(B, T, 18)
    mask = torch.zeros(B, T)
    mask[:, :50] = 1.0   # valid first half only
    loss_masked = wrist_lowband_rfft_loss(pred, gt, mask, fps=20.0, cutoff_hz=1.0)
    # Also compute against a version where pred[..., 0:6][:, 50:] is set to gt's
    # value (so the masked region contributes equally in both cases): the
    # loss should be unchanged.
    pred_eq_pad = pred.clone()
    pred_eq_pad[:, 50:, :] = gt[:, 50:, :]
    loss_eq_pad = wrist_lowband_rfft_loss(pred_eq_pad, gt, mask, fps=20.0, cutoff_hz=1.0)
    assert abs(loss_masked.item() - loss_eq_pad.item()) < 1e-6


def test_r34_wrist_lowband_wrong_shape_raises():
    with pytest.raises(ValueError):
        wrist_lowband_rfft_loss(
            torch.zeros(2, 196, 18), torch.zeros(2, 196, 17),
            torch.ones(2, 196),
        )
    with pytest.raises(ValueError):
        wrist_lowband_rfft_loss(
            torch.zeros(2, 196, 18, 1), torch.zeros(2, 196, 18, 1),
            torch.ones(2, 196),
        )


# ──────────────────────────────────────────────────────────────────────────
# R34 — conditioning augmentation
# ──────────────────────────────────────────────────────────────────────────


def test_r34_cond_aug_identity_when_sigma_max_zero():
    z = torch.randn(4, 50, 23)
    out = apply_stage1_coarse_cond_aug(z, sigma_max=0.0, training=True)
    assert torch.equal(z, out)


def test_r34_cond_aug_identity_in_eval():
    z = torch.randn(4, 50, 23)
    out = apply_stage1_coarse_cond_aug(z, sigma_max=0.5, training=False)
    assert torch.equal(z, out)


def test_r34_cond_aug_per_sample_sigma_at_most_sigma_max():
    """σ ~ U[0, sigma_max] per batch item; noise std per sample ≤ sigma_max."""
    B, T, C = 64, 100, 23
    z = torch.zeros(B, T, C)   # zero base so out - z = noise
    torch.manual_seed(0)
    out = apply_stage1_coarse_cond_aug(z, sigma_max=0.1, training=True)
    noise = out - z
    per_sample_std = noise.flatten(1).std(dim=1)
    # With T*C=2300 samples per row, the empirical std should be very
    # close to the row's σ ≤ sigma_max=0.1; allow 5% finite-sample slack.
    assert per_sample_std.max().item() <= 0.105


def test_r34_cond_aug_changes_per_batch_item():
    """σ is sampled per batch item, so two rows in the same forward pass
    must (almost surely) have different noise variance."""
    B, T, C = 4, 100, 23
    z = torch.zeros(B, T, C)
    torch.manual_seed(0)
    out = apply_stage1_coarse_cond_aug(z, sigma_max=0.5, training=True)
    per_row_std = (out - z).flatten(1).std(dim=1)
    # All 4 rows having exactly equal std is probability zero for σ~U[0, 0.5].
    assert per_row_std.unique().numel() == B


def test_r34_cond_aug_wrong_shape_raises():
    with pytest.raises(ValueError):
        apply_stage1_coarse_cond_aug(
            torch.zeros(4, 100), sigma_max=0.1, training=True,
        )


def test_r34_cond_aug_return_sigma_shape_and_values():
    """return_sigma=True returns (out, sigma) where sigma is (B,) in [0, sigma_max]."""
    B, T, C = 8, 50, 23
    torch.manual_seed(0)
    z = torch.randn(B, T, C)
    out, sigma = apply_stage1_coarse_cond_aug(
        z, sigma_max=0.2, training=True, return_sigma=True,
    )
    assert out.shape == z.shape
    assert sigma.shape == (B,)
    assert (sigma >= 0).all()
    assert (sigma <= 0.2 + 1e-6).all()


def test_r34_cond_aug_return_sigma_zero_in_eval():
    """In eval mode, sigma is all zeros (no augmentation)."""
    B = 4
    z = torch.zeros(B, 50, 23)
    out, sigma = apply_stage1_coarse_cond_aug(
        z, sigma_max=0.5, training=False, return_sigma=True,
    )
    assert torch.equal(out, z)
    assert torch.equal(sigma, torch.zeros(B))


def test_r34_cond_aug_return_sigma_zero_when_sigma_max_zero():
    """sigma_max=0 → sigma is all zeros (identity path)."""
    B = 4
    z = torch.zeros(B, 50, 23)
    out, sigma = apply_stage1_coarse_cond_aug(
        z, sigma_max=0.0, training=True, return_sigma=True,
    )
    assert torch.equal(out, z)
    assert torch.equal(sigma, torch.zeros(B))


def test_r34_cond_aug_back_compat_default_return():
    """Default return_sigma=False returns plain tensor (back-compat with prior callers)."""
    z = torch.randn(4, 50, 23)
    out = apply_stage1_coarse_cond_aug(z, sigma_max=0.1, training=True)
    # Not a tuple
    assert isinstance(out, torch.Tensor)
    assert out.shape == z.shape


def test_c41_temporal_derivative_zero_when_pred_equals_gt():
    c41 = torch.randn(2, 8, C41_DIM)
    mask = torch.ones(2, 8)
    vel = c41_temporal_derivative_loss(c41, c41, mask, order=1)
    acc = c41_temporal_derivative_loss(c41, c41, mask, order=2)
    assert vel.item() < 1e-10
    assert acc.item() < 1e-10


def test_c41_temporal_derivative_acceleration_positive_on_kink():
    gt = torch.zeros(1, 5, C41_DIM)
    pred = gt.clone()
    pred[0, 2, 0] = 1.0
    loss = c41_temporal_derivative_loss(
        pred,
        gt,
        torch.ones(1, 5),
        order=2,
        channel_subset=(0,),
        normalize_by_gt_std=False,
    )
    assert loss.item() > 0.0


def test_c41_temporal_derivative_respects_mask_and_grad():
    gt = torch.zeros(1, 5, C41_DIM)
    pred = gt.clone().requires_grad_(True)
    pred.data[0, 4, 0] = 100.0
    mask = torch.ones(1, 5)
    mask[0, 3:] = 0.0
    loss = c41_temporal_derivative_loss(
        pred,
        gt,
        mask,
        order=1,
        channel_subset=(0,),
        normalize_by_gt_std=False,
    )
    assert loss.item() == 0.0
    loss.backward()
    assert pred.grad is not None


# ──────────────────────────────────────────────────────────────────────────
# R37: Stage-2-style C41 dynamics losses
# ──────────────────────────────────────────────────────────────────────────


def test_r37_slice_constants():
    assert R37_C41_LEFT_WRIST_SLICE == slice(0, 3)
    assert R37_C41_RIGHT_WRIST_SLICE == slice(3, 6)
    assert R37_C41_WRIST_SLICE == slice(0, 6)
    assert R37_C41_LEFT_KNEE_SLICE == slice(6, 9)
    assert R37_C41_RIGHT_KNEE_SLICE == slice(9, 12)
    assert R37_C41_KNEE_SLICE == slice(6, 12)
    assert R37_C41_NECK_SLICE == slice(12, 15)
    assert R37_C41_PELVIS_SLICE == slice(15, 18)
    assert R37_C41_ALL_SLICE == slice(0, 18)


def test_r37_smoothl1_zero_when_pred_equals_gt():
    """All three orders return ~0 when pred == gt across the full mask."""
    c41 = torch.randn(2, 10, C41_DIM)
    mask = torch.ones(2, 10)
    for order in (1, 2, 3):
        loss = c41_smoothl1_finite_diff_cm(
            c41, c41, mask, order=order,
            group_slice=R37_C41_WRIST_SLICE, beta=1.0,
        )
        assert loss.item() < 1e-10, f"order={order} loss = {loss.item()}"


def test_r37_smoothl1_positive_on_kink():
    """A predicted 1-frame spike on one channel produces positive vel/acc."""
    gt = torch.zeros(1, 6, C41_DIM)
    pred = gt.clone()
    pred[0, 3, 15] = 0.01      # 1 cm spike on pelvis_x (m → cm = 1)
    mask = torch.ones(1, 6)
    vel = c41_smoothl1_finite_diff_cm(
        pred, gt, mask, order=1,
        group_slice=R37_C41_PELVIS_SLICE,
    )
    acc = c41_smoothl1_finite_diff_cm(
        pred, gt, mask, order=2,
        group_slice=R37_C41_PELVIS_SLICE,
    )
    assert vel.item() > 0
    assert acc.item() > 0


def test_r37_smoothl1_cm_scale_outlier_capped_by_smoothl1():
    """SmoothL1 with beta=1 caps the loss linearly above ±1 cm.

    A 10 cm spike should produce loss between 1 (purely linear,
    if mean reduction ignored outlier scale) and 10 (purely quadratic).
    With ~6 axes contributing per joint over the masked window, we
    check the per-element growth is sub-quadratic.
    """
    gt = torch.zeros(1, 4, C41_DIM)
    pred_small = gt.clone()
    pred_large = gt.clone()
    # 1 cm spike vs 10 cm spike — same channel, same window.
    pred_small[0, 2, 15] = 0.01     # 1 cm
    pred_large[0, 2, 15] = 0.10     # 10 cm
    mask = torch.ones(1, 4)
    small = c41_smoothl1_finite_diff_cm(
        pred_small, gt, mask, order=1, group_slice=R37_C41_PELVIS_SLICE,
    ).item()
    large = c41_smoothl1_finite_diff_cm(
        pred_large, gt, mask, order=1, group_slice=R37_C41_PELVIS_SLICE,
    ).item()
    # If it were pure MSE the ratio would be ~100; SmoothL1 caps it.
    assert large / max(small, 1e-12) < 50


def test_r37_smoothl1_zero_T_le_order():
    c41 = torch.randn(1, 2, C41_DIM, requires_grad=True)
    mask = torch.ones(1, 2)
    # T = 2, order = 2 → too short, must return safe-grad zero.
    loss = c41_smoothl1_finite_diff_cm(
        c41, c41, mask, order=2, group_slice=R37_C41_WRIST_SLICE,
    )
    assert loss.item() == 0.0
    loss.backward()
    assert c41.grad is not None


def test_r37_smoothl1_mask_shrinks_through_order():
    """When all but `order` frames are masked out, loss is 0 (no valid
    derivative pair)."""
    gt = torch.zeros(1, 5, C41_DIM)
    pred = torch.zeros(1, 5, C41_DIM)
    pred[0, 4, 15] = 100.0
    mask = torch.ones(1, 5)
    mask[0, 3:] = 0.0    # only frames 0,1,2 valid
    # order=2 needs three consecutive valid frames; 0,1,2 are valid so
    # one derivative-2 sample at t=2 exists, but pred[0,2,15] = 0 so
    # contribution is 0.
    loss = c41_smoothl1_finite_diff_cm(
        pred, gt, mask, order=2, group_slice=R37_C41_PELVIS_SLICE,
    )
    assert loss.item() == 0.0


def test_r37_smoothl1_invalid_order_raises():
    c41 = torch.randn(1, 5, C41_DIM)
    mask = torch.ones(1, 5)
    with pytest.raises(ValueError):
        c41_smoothl1_finite_diff_cm(
            c41, c41, mask, order=4, group_slice=R37_C41_WRIST_SLICE,
        )


def test_r37_smoothl1_shape_check():
    c41 = torch.randn(1, 5, C41_DIM)
    mask = torch.ones(1, 5)
    with pytest.raises(ValueError):
        c41_smoothl1_finite_diff_cm(
            c41, c41[:, :, :10], mask, order=1, group_slice=R37_C41_WRIST_SLICE,
        )


def test_r37_speed_moment_zero_when_identical():
    c41 = torch.randn(2, 10, C41_DIM)
    mask = torch.ones(2, 10)
    loss = c41_speed_moment_cm(c41, c41, mask)
    assert loss.item() < 1e-10


def test_r37_speed_moment_positive_on_scale_mismatch():
    """Pred 2× faster than GT should produce non-zero moment loss."""
    torch.manual_seed(0)
    gt = torch.randn(2, 20, C41_DIM) * 0.01      # ~ 1 cm/frame motion
    pred = gt * 2.0                              # double speed
    mask = torch.ones(2, 20)
    loss = c41_speed_moment_cm(pred, gt, mask)
    assert loss.item() > 0.01


def test_r37_speed_moment_invalid_slice_raises():
    c41 = torch.randn(1, 4, C41_DIM)
    mask = torch.ones(1, 4)
    with pytest.raises(ValueError):
        # Non-multiple-of-3 slice — speed_moment requires Δxyz joints.
        c41_speed_moment_cm(c41, c41, mask, group_slice=slice(0, 5))


def test_r37_build_group_masks_with_contact_state():
    """Mask construction:
      - foot_stance L (S4 ch 0) = 1 on frames 5-12
      - walking_mask (S4 ch 4) = 1 on frames 8-15
      - contact_state L_hand (ch 0) = 1 on frames 6-10
      - pelvis_contact (ch 4) = 1 on frames 0-7
    """
    B, T = 1, 20
    seq_mask = torch.ones(B, T)
    s4 = torch.zeros(B, T, S4_DIM)
    s4[0, 5:13, 0] = 1.0       # foot_stance L
    s4[0, 8:16, 4] = 1.0       # walking_mask
    cs = torch.zeros(B, T, 5)
    cs[0, 6:11, 0] = 1.0       # L_hand
    cs[0, 0:8, 4] = 1.0        # pelvis_contact
    masks = build_r37_group_masks(
        seq_mask, s4, cs, erode_half=0,
    )
    # neck = full
    assert torch.equal(masks["neck"], seq_mask)
    # knee = foot_stance OR pre-erode = frames 5-12
    # erode_half=0 means no erosion, so should equal S4 stance.
    assert masks["knee"][0, 5].item() == 1.0
    assert masks["knee"][0, 12].item() == 1.0
    assert masks["knee"][0, 4].item() == 0.0
    assert masks["knee"][0, 13].item() == 0.0
    # pelvis stable = NOT walking AND any-stance.
    # Walking is 8-15; stance is 5-12. Intersection of "not walking" and
    # "any-stance" = 5,6,7.
    assert masks["pelvis"][0, 5].item() == 1.0
    assert masks["pelvis"][0, 7].item() == 1.0
    assert masks["pelvis"][0, 8].item() == 0.0
    # wrist = (L_hand OR R_hand OR pelvis_contact). L_hand 6-10, pelvis 0-7.
    # Union = 0..10.
    assert masks["wrist"][0, 0].item() == 1.0
    assert masks["wrist"][0, 10].item() == 1.0
    assert masks["wrist"][0, 11].item() == 0.0


def test_r37_build_group_masks_without_contact_state():
    """When contact_state is None, wrist mask falls back to full seq_mask."""
    B, T = 1, 10
    seq_mask = torch.zeros(B, T)
    seq_mask[0, :7] = 1.0       # padding past frame 7
    s4 = torch.zeros(B, T, S4_DIM)
    s4[0, 2:5, 0] = 1.0
    masks = build_r37_group_masks(
        seq_mask, s4, contact_state=None, erode_half=0,
    )
    # Padded frames (7-9) must be 0 in every mask.
    for key in ("wrist", "knee", "pelvis", "neck"):
        assert masks[key][0, 7:].sum().item() == 0.0
    # Wrist fallback = seq_mask exactly.
    assert torch.equal(masks["wrist"], seq_mask)


def test_r37_build_group_masks_erosion_shrinks_edges():
    """erode_half=2 should zero 2 frames on each side of a stance run."""
    B, T = 1, 20
    seq_mask = torch.ones(B, T)
    s4 = torch.zeros(B, T, S4_DIM)
    s4[0, 5:15, 0] = 1.0      # foot_stance L, span 5-14 (10 frames)
    masks = build_r37_group_masks(
        seq_mask, s4, contact_state=None, erode_half=2,
    )
    # Edges 5-6 and 13-14 should be eroded.
    assert masks["knee"][0, 5].item() == 0.0
    assert masks["knee"][0, 6].item() == 0.0
    assert masks["knee"][0, 7].item() == 1.0
    assert masks["knee"][0, 12].item() == 1.0
    assert masks["knee"][0, 13].item() == 0.0
    assert masks["knee"][0, 14].item() == 0.0


def test_r37_build_group_masks_shape_checks():
    seq_mask = torch.ones(1, 10)
    bad_s4 = torch.zeros(1, 10, 12)        # wrong S4_DIM
    with pytest.raises(ValueError):
        build_r37_group_masks(seq_mask, bad_s4, contact_state=None)
    good_s4 = torch.zeros(1, 10, S4_DIM)
    bad_cs = torch.zeros(1, 10, 3)         # wrong last dim
    with pytest.raises(ValueError):
        build_r37_group_masks(seq_mask, good_s4, contact_state=bad_cs)


def test_r37_smoothl1_grad_flow_through_pred():
    """Loss must produce non-trivial gradient on pred."""
    torch.manual_seed(1)
    gt = torch.zeros(2, 8, C41_DIM)
    pred = (torch.randn(2, 8, C41_DIM) * 0.05).requires_grad_(True)
    mask = torch.ones(2, 8)
    loss = c41_smoothl1_finite_diff_cm(
        pred, gt, mask, order=1, group_slice=R37_C41_WRIST_SLICE,
    )
    loss.backward()
    assert pred.grad is not None
    # Gradient should be non-zero on the wrist slice.
    assert pred.grad[..., R37_C41_WRIST_SLICE].abs().sum().item() > 0
    # No gradient should flow to non-wrist channels since the slice
    # excludes them.
    assert pred.grad[..., 6:].abs().sum().item() == 0.0


# ──────────────────────────────────────────────────────────────────────────
# R38: contact-window weighted wrist value MSE
# ──────────────────────────────────────────────────────────────────────────


def _make_contact_state(B: int, T: int, l_seg: slice, r_seg: slice) -> torch.Tensor:
    """Helper: build a (B, T, 5) contact_state with L/R hand contact
    set on the given frame slices, foot/pelvis zero."""
    cs = torch.zeros(B, T, 5)
    cs[:, l_seg, 0] = 1.0      # L_hand
    cs[:, r_seg, 1] = 1.0      # R_hand
    return cs


def test_r38_contact_wrist_zero_when_pred_equals_gt():
    c41 = torch.randn(2, 20, C41_DIM)
    seq_mask = torch.ones(2, 20)
    cs = _make_contact_state(2, 20, slice(5, 10), slice(8, 15))
    loss = c41_contact_window_wrist_loss(c41, c41, seq_mask, cs, erode_half=0)
    assert loss.item() < 1e-10


def test_r38_contact_wrist_positive_on_wrist_offset_in_contact():
    gt = torch.zeros(1, 10, C41_DIM)
    pred = gt.clone()
    # Inject 1 cm error on wrist L channel only at a contact frame.
    pred[0, 5, 0] = 0.01
    seq_mask = torch.ones(1, 10)
    cs = _make_contact_state(1, 10, slice(4, 8), slice(20, 21))
    loss = c41_contact_window_wrist_loss(
        gt, pred, seq_mask, cs, erode_half=0,
    )
    # 0.01^2 = 1e-4 per frame; denominator is # of contact frames inside
    # the mask. With erode_half=0, contact frames = [4, 5, 6, 7] = 4
    # frames. Loss = 1e-4 / 4 = 2.5e-5 ish.
    assert loss.item() > 0
    assert loss.item() < 1e-3


def test_r38_contact_wrist_ignores_non_wrist_channels():
    """Error on non-wrist channels should NOT contribute."""
    gt = torch.zeros(1, 10, C41_DIM)
    pred = gt.clone()
    pred[0, 5, 7] = 100.0      # left_knee
    pred[0, 5, 13] = 100.0     # neck
    seq_mask = torch.ones(1, 10)
    cs = _make_contact_state(1, 10, slice(4, 8), slice(20, 21))
    loss = c41_contact_window_wrist_loss(gt, pred, seq_mask, cs, erode_half=0)
    assert loss.item() == 0.0


def test_r38_contact_wrist_only_active_frames():
    """Error outside contact window should NOT contribute."""
    gt = torch.zeros(1, 10, C41_DIM)
    pred = gt.clone()
    pred[0, 0, 0] = 1.0        # frame 0 - no contact
    pred[0, 9, 0] = 1.0        # frame 9 - no contact
    seq_mask = torch.ones(1, 10)
    cs = _make_contact_state(1, 10, slice(4, 8), slice(20, 21))
    loss = c41_contact_window_wrist_loss(gt, pred, seq_mask, cs, erode_half=0)
    assert loss.item() == 0.0


def test_r38_contact_wrist_erosion_shrinks_contact_window():
    """With erode_half=1, the contact frames at the edges are excluded."""
    gt = torch.zeros(1, 10, C41_DIM)
    pred = gt.clone()
    # Inject 1 cm on the edge frame of the L_hand contact (4 and 7).
    pred[0, 4, 0] = 1.0
    pred[0, 7, 0] = 1.0
    seq_mask = torch.ones(1, 10)
    cs = _make_contact_state(1, 10, slice(4, 8), slice(20, 21))
    # erode_half=1 removes the edge frames 4 and 7 from the mask.
    # Interior frames 5 and 6 have pred=gt=0 → loss=0.
    loss = c41_contact_window_wrist_loss(gt, pred, seq_mask, cs, erode_half=1)
    assert loss.item() == 0.0
    # Without erosion, the edge errors DO contribute.
    loss_no_erode = c41_contact_window_wrist_loss(
        gt, pred, seq_mask, cs, erode_half=0,
    )
    assert loss_no_erode.item() > 0


def test_r38_contact_wrist_safe_zero_on_no_contact():
    """All-zero contact_state must return a gradient-safe zero (not NaN
    or KeyError)."""
    pred = torch.randn(1, 10, C41_DIM, requires_grad=True)
    gt = torch.zeros(1, 10, C41_DIM)
    seq_mask = torch.ones(1, 10)
    cs = torch.zeros(1, 10, 5)
    loss = c41_contact_window_wrist_loss(pred, gt, seq_mask, cs)
    assert loss.item() == 0.0
    loss.backward()
    assert pred.grad is not None


def test_r38_contact_wrist_seq_mask_excludes_padded_frames():
    """A padded frame even if contact_state says contact should NOT
    contribute (seq_mask wins)."""
    gt = torch.zeros(1, 10, C41_DIM)
    pred = gt.clone()
    pred[0, 8, 0] = 1.0        # would be a contact frame, but padded
    seq_mask = torch.ones(1, 10)
    seq_mask[0, 8:] = 0.0
    cs = _make_contact_state(1, 10, slice(7, 10), slice(20, 21))
    loss = c41_contact_window_wrist_loss(gt, pred, seq_mask, cs, erode_half=0)
    # Frame 7 has contact and is valid; pred=gt=0 there. Frames 8, 9 are
    # padded (seq_mask=0) so even though contact_state says contact,
    # they are excluded. Loss = 0.
    assert loss.item() == 0.0


def test_r38_contact_wrist_per_sample_loss_weighting_b():
    """The optional min-SNR weight should scale the per-frame loss
    consistently."""
    gt = torch.zeros(2, 10, C41_DIM)
    pred = gt.clone()
    pred[0, 5, 0] = 0.01       # batch 0 has an error
    seq_mask = torch.ones(2, 10)
    cs = _make_contact_state(2, 10, slice(5, 6), slice(20, 21))
    # Unweighted: both batches use weight 1.
    loss_uniform = c41_contact_window_wrist_loss(
        gt, pred, seq_mask, cs, erode_half=0,
    )
    # Weighted with w=(2, 1): batch 0's contribution is doubled.
    w_b = torch.tensor([2.0, 1.0])
    loss_weighted = c41_contact_window_wrist_loss(
        gt, pred, seq_mask, cs, erode_half=0,
        per_sample_loss_weighting_b=w_b,
    )
    # Batch 1 has no error → contributes 0. Loss is just batch 0's
    # contribution divided by total active frames.
    # The denom (sum of mask) is the SAME in both cases (the mask
    # doesn't see w_b), so loss_weighted = 2 * loss_uniform.
    assert loss_weighted.item() == pytest.approx(
        2 * loss_uniform.item(), rel=1e-5,
    )


def test_r38_contact_wrist_shape_check_pred_gt():
    bad = torch.randn(1, 10, 17)   # wrong last dim
    ok = torch.randn(1, 10, C41_DIM)
    seq_mask = torch.ones(1, 10)
    cs = torch.zeros(1, 10, 5)
    with pytest.raises(ValueError):
        c41_contact_window_wrist_loss(bad, ok, seq_mask, cs)


def test_r38_contact_wrist_shape_check_contact_state():
    c41 = torch.randn(1, 10, C41_DIM)
    seq_mask = torch.ones(1, 10)
    bad_cs = torch.zeros(1, 10, 3)
    with pytest.raises(ValueError):
        c41_contact_window_wrist_loss(c41, c41, seq_mask, bad_cs)


# ──────────────────────────────────────────────────────────────────────────
# R38: Stage-1.5 denoiser init_pose injection (integration)
# ──────────────────────────────────────────────────────────────────────────


def test_r38_stage1p5_denoiser_init_pose_forward():
    """End-to-end: enabling init_pose_dim=135 changes the forward output
    deterministically, but starts identical at init (zero-init Linear)."""
    from piano.models.stage1p5_interaction import (
        Stage1p5Denoiser, Stage1p5DenoiserConfig,
    )

    torch.manual_seed(0)
    base_cfg = Stage1p5DenoiserConfig(
        d_model=64, n_layers=2, n_heads=2, ff_mult=2, dropout=0.0,
        max_seq_length=32, object_num_tokens=4, object_token_dim=32,
        text_dim=0,                     # disable text path
        use_text=False,
    )
    init_cfg = Stage1p5DenoiserConfig(
        d_model=64, n_layers=2, n_heads=2, ff_mult=2, dropout=0.0,
        max_seq_length=32, object_num_tokens=4, object_token_dim=32,
        text_dim=0, use_text=False,
        init_pose_dim=135,
    )

    torch.manual_seed(123)
    base = Stage1p5Denoiser(base_cfg)
    torch.manual_seed(123)
    init = Stage1p5Denoiser(init_cfg)

    # Same RNG state was used for both; the only difference is the
    # init_pose Linear which is zero-init. Confirm step-0 output equality.
    B, T = 2, 8
    x_t = torch.randn(B, T, base_cfg.motion_dim)
    t = torch.randint(0, 1000, (B,))
    cond_base = {
        "object_world_traj": torch.randn(B, T, 9),
        "object_tokens": torch.randn(B, 4, 32),
        "stage1_coarse": torch.randn(B, T, 23),
    }
    cond_init = dict(cond_base)
    cond_init["init_pose"] = torch.randn(B, 135)

    with torch.no_grad():
        out_base = base(x_t, t, cond_base, cond_drop_mask=None)
        out_init = init(x_t, t, cond_init, cond_drop_mask=None)

    assert out_base.shape == (B, T, base_cfg.motion_dim)
    assert out_init.shape == out_base.shape
    # init_pose_proj is zero-init → adds 0 → outputs should match.
    diff = (out_base - out_init).abs().max().item()
    assert diff < 1e-5, (
        f"Zero-init init_pose_proj should not change step-0 output; "
        f"max diff = {diff:.2e}"
    )


def test_r38_stage1p5_denoiser_init_pose_missing_raises():
    """When init_pose_dim>0 but cond['init_pose'] is missing, forward
    must raise (clear error vs silent bug)."""
    from piano.models.stage1p5_interaction import (
        Stage1p5Denoiser, Stage1p5DenoiserConfig,
    )

    cfg = Stage1p5DenoiserConfig(
        d_model=64, n_layers=2, n_heads=2, ff_mult=2, dropout=0.0,
        max_seq_length=32, object_num_tokens=4, object_token_dim=32,
        text_dim=0, use_text=False,
        init_pose_dim=135,
    )
    model = Stage1p5Denoiser(cfg)

    B, T = 2, 8
    x_t = torch.randn(B, T, cfg.motion_dim)
    t = torch.randint(0, 1000, (B,))
    cond = {
        "object_world_traj": torch.randn(B, T, 9),
        "object_tokens": torch.randn(B, 4, 32),
        "stage1_coarse": torch.randn(B, T, 23),
        # NO init_pose — should raise.
    }
    with pytest.raises(KeyError):
        model(x_t, t, cond, cond_drop_mask=None)
