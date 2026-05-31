"""Stage-1.5 R32 V7 anti-bug loss helpers.

The R32 Phase 1 audit (analyses/round32_phase1_dyn_audit_20260530_121621/
audit_report.md) found Stage-1.5 V0 is NOT mode-collapsed (std ratios
mostly in [0.83, 1.29], unlike Stage-1 V0's [0.24, 0.46]). Instead it
has 5 distinct localised failure modes:

  B1: wrist + footstep velocity under-articulation
      vel_rms ratio 0.70 (wrist), 0.47 (footstep); PSD mid-band 0.23.
  B2: same as B1 with footstep specifically.
  B3: phase unit-circle violation
      pred (sin² + cos²) − 1 = 0.027 / 0.030 per leg (L/R).
      V0 has w_s4_phase=0.05·((sin²+cos²)−1)² but it's clearly too weak.
  B4: stance + walking_mask BCE not driving saturation
      foot_stance mean ≈ 0.51 std ≈ 0.38 — no logit at all in the
      |x| > 2 region. BCE weight is too low.
  B5: C41 wrist frame-0 invariant violated
      rms_at_t0 median 5.3 cm; should be 0 by construction.

This module collects the helpers V7 needs (B1+B2 reuse R31's
channel_moment_match_loss; B3 + B5 need new helpers; B4 is a pure
config-weight change).

Channel layout (matches train_stage1p5 + stage2_oracle_conditions):

    C41 (18) — pelvis-local current-yaw delta against frame 0
      [0:3]   left_wrist  Δxyz
      [3:6]   right_wrist Δxyz
      [6:9]   left_knee   Δxyz
      [9:12]  right_knee  Δxyz
      [12:15] neck        Δxyz
      [15:18] pelvis      Δxzy

    S4 (13)
      [18:20] foot_stance L, R          (BCE; logits)
      [20:22] ankle_height_norm L, R
      [22:23] walking_mask              (BCE; logits)
      [23:27] phase_sin/cos L, phase_sin/cos R
      [27:31] footstep_x/z L, footstep_x/z R

All helpers operate on RAW Stage-1.5 outputs (no z-scoring).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


# ─── Channel-layout constants ──────────────────────────────────────────
# C41 wrist channels (B5 frame-0 invariant target). Indices into the
# C41 sub-tensor (B, T, 18).
CH_C41_WRIST = slice(0, 6)

# S4 phase (sin, cos) per leg (B3). Indices into the S4 sub-tensor
# (B, T, 13). The trainer splits x0_pred into c41_pred[..., :18] and
# s4_pred[..., 18:], so helpers receive S4 separately and use
# S4-LOCAL indices (NOT the 23..26 indices the Phase 1 audit reports,
# which were global to the 31-D x0).
CH_S4_PHASE_L_SIN = 5     # global 23 - 18
CH_S4_PHASE_L_COS = 6     # global 24 - 18
CH_S4_PHASE_R_SIN = 7     # global 25 - 18
CH_S4_PHASE_R_COS = 8     # global 26 - 18

# C41 + S4 dims (sanity).
C41_DIM = 18
S4_DIM = 13
TOTAL_DIM = C41_DIM + S4_DIM   # 31


# ──────────────────────────────────────────────────────────────────────────
# V7-B: phase unit-circle / angle consistency
# ──────────────────────────────────────────────────────────────────────────


def phase_unit_circle_loss(
    s4_pred: Tensor,                  # (B, T, 13)
    s4_gt: Tensor,                    # (B, T, 13)
    seq_mask: Tensor,                 # (B, T)
    *,
    unit_norm_weight: float = 1.0,
    angle_weight: float = 0.5,
) -> Tensor:
    """V7-B — phase (sin, cos) channels need TWO things V0's
    ``((sin² + cos²) − 1)²`` term doesn't enforce:

    1. **Unit-norm**: the GT term V0 already has, but with w_s4_phase=0.05
       it's too weak (audit shows pred (sin² + cos²) - 1 ≈ 0.03 still).

    2. **Angle consistency**: even if `sin² + cos² = 1` (lazy solution),
       the angle `atan2(sin, cos)` can be arbitrary. We need pred angle
       to match GT angle.

    Component 1 (unit-norm penalty) on L and R legs, mean-aggregated:

        ||sin² + cos² - 1||₂² , averaged over (B, T, 2 legs).

    Component 2 (angle consistency via complex inner product):

        1 - (sin_p · sin_g + cos_p · cos_g)
          = 1 - cos(angle_p - angle_g)
          ∈ [0, 2], zero iff angles equal.

    Returns
    -------
    Scalar: unit_norm_weight · L1 + angle_weight · L2.
    Suggested weights match V7-B variant (1.0, 0.5).
    """
    sin_p_L = s4_pred[..., CH_S4_PHASE_L_SIN]
    cos_p_L = s4_pred[..., CH_S4_PHASE_L_COS]
    sin_p_R = s4_pred[..., CH_S4_PHASE_R_SIN]
    cos_p_R = s4_pred[..., CH_S4_PHASE_R_COS]

    sin_g_L = s4_gt[..., CH_S4_PHASE_L_SIN]
    cos_g_L = s4_gt[..., CH_S4_PHASE_L_COS]
    sin_g_R = s4_gt[..., CH_S4_PHASE_R_SIN]
    cos_g_R = s4_gt[..., CH_S4_PHASE_R_COS]

    mask = seq_mask                                            # (B, T)
    denom = mask.sum().clamp_min(1.0)

    # Unit-norm penalty (mean over valid frames + L/R).
    if unit_norm_weight > 0:
        un_L = ((sin_p_L.pow(2) + cos_p_L.pow(2)) - 1.0).pow(2)
        un_R = ((sin_p_R.pow(2) + cos_p_R.pow(2)) - 1.0).pow(2)
        un = ((un_L + un_R) * 0.5 * mask).sum() / denom
    else:
        un = s4_pred.new_zeros(())

    # Angle consistency (cos angle diff).
    if angle_weight > 0:
        dot_L = sin_p_L * sin_g_L + cos_p_L * cos_g_L
        dot_R = sin_p_R * sin_g_R + cos_p_R * cos_g_R
        ang = (((1.0 - dot_L) + (1.0 - dot_R)) * 0.5 * mask).sum() / denom
    else:
        ang = s4_pred.new_zeros(())

    return unit_norm_weight * un + angle_weight * ang


# ──────────────────────────────────────────────────────────────────────────
# V7-D: C41 wrist frame-0 invariant
# ──────────────────────────────────────────────────────────────────────────


def c41_wrist_frame0_consistency_loss(
    c41_pred: Tensor,                 # (B, T, 18)
) -> Tensor:
    """V7-D — Stage-1.5 C41 channels [0:6] are pelvis-local Δxyz against
    frame 0, so at t=0 they are exactly 0 by construction. Stage-1.5 V0
    violates this — audit shows median rms_at_t0 = 5.3 cm.

    Loss = MSE between pred_c41[:, 0, 0:6] and 0.

    Returns
    -------
    Scalar (m²). Suggested weight 1.0 in V7-D config.
    """
    frame0_wrist = c41_pred[:, 0, CH_C41_WRIST]                # (B, 6)
    return frame0_wrist.pow(2).mean()


def c41_temporal_derivative_loss(
    c41_pred: Tensor,
    c41_gt: Tensor,
    seq_mask: Tensor,
    *,
    order: int = 1,
    channel_subset: tuple[int, ...] | None = None,
    normalize_by_gt_std: bool = True,
) -> Tensor:
    """Masked C41 velocity/acceleration MSE.

    ``order=1`` compares C41 velocity and ``order=2`` compares C41
    acceleration. C41 is raw metric-space delta xyz, so this complements the
    per-frame MSE with explicit temporal dynamics supervision.
    """
    if c41_pred.shape != c41_gt.shape:
        raise ValueError(
            f"shape mismatch: pred {c41_pred.shape} vs gt {c41_gt.shape}"
        )
    if c41_pred.ndim != 3 or c41_pred.shape[-1] != C41_DIM:
        raise ValueError(f"expected (B, T, {C41_DIM}), got {c41_pred.shape}")
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order}")

    B, T, D = c41_pred.shape
    if seq_mask.shape != (B, T):
        raise ValueError(
            f"seq_mask shape {tuple(seq_mask.shape)} != (B, T) = {(B, T)}"
        )
    if T <= order:
        return c41_pred.sum() * 0.0

    if channel_subset is None:
        pred = c41_pred
        gt = c41_gt
        n_ch = D
    else:
        idx = torch.tensor(channel_subset, device=c41_pred.device, dtype=torch.long)
        pred = c41_pred.index_select(-1, idx)
        gt = c41_gt.index_select(-1, idx)
        n_ch = int(idx.numel())
        if n_ch == 0:
            return c41_pred.sum() * 0.0

    d_pred = pred
    d_gt = gt
    valid = seq_mask.float()
    for _ in range(order):
        d_pred = d_pred[:, 1:] - d_pred[:, :-1]
        d_gt = d_gt[:, 1:] - d_gt[:, :-1]
        valid = valid[:, 1:] * valid[:, :-1]

    err = (d_pred - d_gt).pow(2)
    if normalize_by_gt_std:
        flat_gt = d_gt.reshape(-1, n_ch)
        flat_valid = valid.reshape(-1).to(dtype=flat_gt.dtype)
        w_sum = flat_valid.sum().clamp_min(1.0)
        mean_gt = (flat_gt * flat_valid.unsqueeze(-1)).sum(0) / w_sum
        var_gt = (
            (flat_gt - mean_gt).pow(2) * flat_valid.unsqueeze(-1)
        ).sum(0) / w_sum
        err = err / var_gt.clamp_min(1e-6).view(1, 1, n_ch)

    denom = valid.sum().clamp_min(1.0) * float(n_ch)
    return (err * valid.unsqueeze(-1).to(err.dtype)).sum() / denom


# ──────────────────────────────────────────────────────────────────────────
# R34: targeted wrist low-band rFFT loss
# ──────────────────────────────────────────────────────────────────────────

# C41 wrist channels (R34 low-band loss target) — same slice as CH_C41_WRIST.
# Re-aliased here so call sites can be explicit about R34 vs V7-D usage.
R34_C41_WRIST_SLICE = slice(0, 6)


def wrist_lowband_rfft_loss(
    c41_pred: Tensor,                 # (B, T, 18)
    c41_gt: Tensor,                   # (B, T, 18)
    seq_mask: Tensor,                 # (B, T) float {0, 1}
    *,
    fps: float = 20.0,
    cutoff_hz: float = 1.0,
) -> Tensor:
    """R34 — low-frequency complex FFT MSE on C41 wrist channels.

    Implementation per brief §7.2:

        L = MSE(rFFT(pred_wrist).real[mask], rFFT(gt_wrist).real[mask])
          + MSE(rFFT(pred_wrist).imag[mask], rFFT(gt_wrist).imag[mask])

    where ``mask`` selects frequencies ``freqs <= cutoff_hz`` along the
    time axis at the given fps. Operates only on C41 wrist channels
    ``[0:6]`` so this loss does not interfere with non-wrist channels.

    Args:
        c41_pred: (B, T, 18) — predicted C41 (raw, NOT z-scored).
        c41_gt:   (B, T, 18) — GT C41.
        seq_mask: (B, T)  — 1.0 inside the valid window, 0.0 in padding.

    Returns
    -------
    Scalar (float). The rFFT is taken over the full T axis (including
    padding), so the mask is used to zero out padding before FFT, mirroring
    the standard "convolve with valid window" trick. We zero the padding
    region of BOTH pred and GT so the FFT spectra match in the same
    domain.
    """
    if c41_pred.shape != c41_gt.shape:
        raise ValueError(
            f"shape mismatch: pred {c41_pred.shape} vs gt {c41_gt.shape}"
        )
    if c41_pred.ndim != 3:
        raise ValueError(f"expected (B, T, C); got {c41_pred.shape}")
    B, T, _ = c41_pred.shape
    if seq_mask.shape != (B, T):
        raise ValueError(
            f"seq_mask shape {tuple(seq_mask.shape)} != (B, T) = {(B, T)}"
        )
    pw = c41_pred[..., R34_C41_WRIST_SLICE]                          # (B, T, 6)
    gw = c41_gt[..., R34_C41_WRIST_SLICE]
    mask = seq_mask.unsqueeze(-1).to(pw.dtype)                       # (B, T, 1)
    pw = pw * mask
    gw = gw * mask

    freqs = torch.fft.rfftfreq(T, d=1.0 / fps).to(pw.device)         # (n_freqs,)
    low_mask = freqs <= cutoff_hz                                    # bool (n_freqs,)

    pred_fft = torch.fft.rfft(pw.float(), dim=1)                     # (B, n_freqs, 6)
    gt_fft = torch.fft.rfft(gw.float(), dim=1)

    pred_low = pred_fft[:, low_mask, :]
    gt_low = gt_fft[:, low_mask, :]

    loss_real = F.mse_loss(pred_low.real, gt_low.real)
    loss_imag = F.mse_loss(pred_low.imag, gt_low.imag)
    return loss_real + loss_imag


# ──────────────────────────────────────────────────────────────────────────
# R34: conditioning augmentation on z-scored stage1_coarse
# ──────────────────────────────────────────────────────────────────────────


def apply_stage1_coarse_cond_aug(
    stage1_coarse_z: Tensor,
    sigma_max: float = 0.0,
    *,
    training: bool = True,
    return_sigma: bool = False,
) -> "Tensor | tuple[Tensor, Tensor]":
    """R34 — per-sample Gaussian noise on z-scored stage1_coarse cond.

    Implementation per brief §7.3:

        sigma ~ U[0, sigma_max]    (one σ per batch item, broadcast over T, C)
        coarse_aug = coarse_z + sigma * eps,   eps ~ N(0, I)

    Sigma is NOT passed to the model (first-day non-truncated mode from
    Ho 2021 §3.3); architecture is unchanged. When training is False or
    sigma_max <= 0, this is the identity.

    Args:
        stage1_coarse_z: (B, T, 23) — already z-scored.
        sigma_max: float >= 0 — upper bound of per-batch σ.
        training: bool — when False, no aug.
        return_sigma: bool — when True, returns (out, sigma) where sigma
            is a (B,) tensor of the per-batch-item sampled σ. Useful for
            train-time logging of the actually-sampled sigma distribution.
            When False (default), returns the augmented tensor directly
            (back-compat with prior API).

    Returns
    -------
    Tensor of the same shape as ``stage1_coarse_z``, or
    (augmented_tensor, sigma) if ``return_sigma=True``. ``sigma`` is a
    ``(B,)`` tensor (zeros when augmentation is off).
    """
    if sigma_max <= 0.0 or not training:
        if return_sigma:
            B = stage1_coarse_z.shape[0]
            zero_sigma = torch.zeros(
                (B,), device=stage1_coarse_z.device, dtype=stage1_coarse_z.dtype,
            )
            return stage1_coarse_z, zero_sigma
        return stage1_coarse_z
    if stage1_coarse_z.ndim != 3:
        raise ValueError(
            f"expected (B, T, C); got {stage1_coarse_z.shape}"
        )
    B = stage1_coarse_z.shape[0]
    sigma = torch.rand(
        (B, 1, 1),
        device=stage1_coarse_z.device,
        dtype=stage1_coarse_z.dtype,
    ) * float(sigma_max)
    noise = torch.randn_like(stage1_coarse_z) * sigma
    out = stage1_coarse_z + noise
    if return_sigma:
        return out, sigma.squeeze(-1).squeeze(-1)   # (B,)
    return out


# ──────────────────────────────────────────────────────────────────────────
# Convenience re-export for the trainer
# ──────────────────────────────────────────────────────────────────────────

# V7-A reuses R31's channel_moment_match_loss; the trainer imports it
# directly from piano.training.stage1_losses to avoid duplicating
# code.
