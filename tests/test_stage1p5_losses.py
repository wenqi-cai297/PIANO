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
    S4_DIM,
    TOTAL_DIM,
    apply_stage1_coarse_cond_aug,
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
