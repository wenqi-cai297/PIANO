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
    S4_DIM,
    TOTAL_DIM,
    c41_wrist_frame0_consistency_loss,
    phase_unit_circle_loss,
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
