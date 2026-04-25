"""Loss functions for all PIANO training stages.

Stage A (Predictor): pseudo-label supervision losses
Stage B (Generator): masked token prediction loss
Stage C (Joint):     predictor + generator + consistency loss
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ============================================================================
# Kendall et al. CVPR 2018 — homoscedastic uncertainty multi-task weighting
# ============================================================================

class KendallTaskWeights(nn.Module):
    """Learnable per-task log-variance weights (Kendall et al. CVPR 2018).

    Total loss is::

        L_total = Σ_i  exp(-s_i) · L_i  +  0.5 · s_i

    where ``s_i = log σ_i^2`` is a learnable scalar per task. At
    optimum ``∂L/∂s_i = 0`` gives ``exp(-s_i) · L_i = 0.5`` so each
    task contributes 0.5 to the total — the optimiser automatically
    rebalances tasks of very different scales (our problem: train
    target loss ≈ 0.018 vs train contact loss ≈ 0.43, a 24× scale gap
    that no fixed weight could compensate cleanly).

    At init ``s_i = 0`` → all tasks are weighted ×1 (which is equal
    to the manual unit-weight starting point). The optimiser learns
    s_i over the first few epochs.

    Reference: Kendall, Gal, Cipolla. "Multi-Task Learning Using
    Uncertainty to Weigh Losses for Scene Geometry and Semantics."
    CVPR 2018.
    """

    def __init__(self, task_names: tuple[str, ...]) -> None:
        super().__init__()
        self.task_names = tuple(task_names)
        # One nn.Parameter per task. Stored as a flat ParameterDict so
        # they're picked up by .parameters() and DDP synchronises them.
        self.log_vars = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(())) for name in task_names}
        )

    def forward(self, raw_losses: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        """Combine per-task losses with learned uncertainty weights.

        Returns ``(total, log_dict)`` where log_dict carries the
        learned weights and log-variances for wandb.
        """
        device = next(self.log_vars.values()).device
        total = torch.zeros((), device=device)
        log: dict[str, Tensor] = {}
        for name in self.task_names:
            if name not in raw_losses:
                continue
            s = self.log_vars[name]
            li = raw_losses[name]
            # Numerically more stable than literal exp(-s) for very
            # negative s: clamp the exponent.
            inv_var = torch.exp(-s.clamp(min=-10.0, max=10.0))
            total = total + inv_var * li + 0.5 * s
            log[f"weight_{name}"] = inv_var.detach()
            log[f"log_var_{name}"] = s.detach()
        return total, log


# ============================================================================
# Stage A: Interaction Predictor losses
# ============================================================================

class PredictorLoss(nn.Module):
    """Combined loss for the Interaction Predictor.

    Supervises contact state, contact target (xyz), interaction phase,
    and support state against pseudo-labels.

    **Target head is xyz regression, not patch classification.** The
    contact-gate is still applied: for (t, body_part) cells where GT
    contact < threshold, the body part isn't touching anything and the
    regression target is undefined — we zero out the per-cell loss and
    normalise by the count of gated cells.

    **Focal loss** on the phase + support CE heads (Lin et al.,
    RetinaNet / ICCV 2017). Our pseudo-label class frequencies are
    severely skewed — phase `pre_contact` is 0.4% of frames, support
    `hand_support` 3%. With naive CE the model ignored rare classes
    entirely (F1 = 0 even on the training set). The focal factor
    ``(1 - p_t)^γ`` down-weights confident / easy frames so gradient
    mass stays on the hard / rare-class ones.
    """

    def __init__(
        self,
        contact_weight: float = 1.0,
        target_weight: float = 0.5,
        phase_weight: float = 0.5,
        support_weight: float = 0.5,
        contact_threshold: float = 0.5,
        label_smoothing: float = 0.0,
        focal_gamma: float = 0.0,
        use_kendall_weights: bool = False,
        logit_adjust_phase: Tensor | None = None,
        logit_adjust_support: Tensor | None = None,
        logit_adjust_tau: float = 1.0,
    ) -> None:
        super().__init__()
        self.contact_weight = contact_weight
        self.target_weight = target_weight
        self.phase_weight = phase_weight
        self.support_weight = support_weight
        self.contact_threshold = contact_threshold
        # Label smoothing on phase / support CE (ViT / T5 / PointNeXt /
        # MoMask convention). Stops the model from over-confidently
        # fitting ~10%-noisy pseudo-labels.
        self.label_smoothing = label_smoothing
        # Focal loss gamma. 0 disables. RetinaNet default is 2.0.
        self.focal_gamma = focal_gamma
        # Kendall et al. CVPR'18 multi-task uncertainty weighting.
        # When True, the static {contact,target,phase,support}_weight
        # values above are IGNORED and the optimiser learns per-task
        # log-variances that auto-balance scale differences. v2 found
        # that fixed weights left the target term contributing only
        # ~2% of total gradient because target loss ≈ 0.018 while
        # contact loss ≈ 0.43; Kendall weights remove that hand-tuning
        # by construction.
        self.use_kendall_weights = use_kendall_weights
        if use_kendall_weights:
            self.kendall = KendallTaskWeights(
                task_names=("contact", "target", "phase", "support"),
            )
        else:
            self.kendall = None
        # Logit Adjustment (Menon et al. ICLR'21) for long-tailed
        # categorical heads — Bayes-optimal under a known class prior
        # at extreme imbalance. v3 with focal γ=2 alone left
        # `pre_contact` (0.4%) F1 = 0 and `single_foot` (5%) F1 = 0
        # even on train; LDAM (Cao NeurIPS'19) and Menon ICLR'21
        # both show focal underperforms margin/logit-based methods
        # at ratios >100:1. Add ``τ × log π_y`` to the logits at
        # train time only — at inference the raw logits are used.
        # ``logit_adjust_*`` is a 1D tensor of class log-priors
        # (== log(N_y / Σ N_y)). Pass None to disable per-head.
        self.logit_adjust_tau = float(logit_adjust_tau)
        if logit_adjust_phase is not None:
            self.register_buffer("logit_adjust_phase", logit_adjust_phase.float())
        else:
            self.logit_adjust_phase = None  # type: ignore[assignment]
        if logit_adjust_support is not None:
            self.register_buffer("logit_adjust_support", logit_adjust_support.float())
        else:
            self.logit_adjust_support = None  # type: ignore[assignment]

    @staticmethod
    def _focal_weight(logits: Tensor, gt: Tensor, gamma: float) -> Tensor:
        """(1 - p_t)^gamma per-element, where p_t is the predicted
        probability of the true class. logits shape (N, C), gt shape (N,)."""
        with torch.no_grad():
            p = F.softmax(logits, dim=-1)                            # (N, C)
            p_t = p.gather(-1, gt.unsqueeze(-1)).squeeze(-1)         # (N,)
            return (1.0 - p_t).clamp(min=0.0).pow(gamma)

    def forward(
        self,
        pred: dict[str, Tensor],
        gt_contact: Tensor,
        gt_target: Tensor,
        gt_phase: Tensor,
        gt_support: Tensor,
        mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute all predictor losses.

        Parameters
        ----------
        pred : output dict from InteractionPredictor. Must contain
            ``contact_logits``, ``contact_target_xyz``, ``phase_logits``,
            ``support_logits``.
        gt_contact : (B, T, 5) — soft contact pseudo-labels in [0, 1]
        gt_target : (B, T, 5, 3) — xyz target in object-local frame
            (zero / arbitrary where gt_contact < threshold — gated out)
        gt_phase : (B, T) — integer phase labels
        gt_support : (B, T) — integer support labels
        mask : (B, T) — True for valid (non-padded) frames

        Returns
        -------
        Dictionary with individual losses and total.
        """
        # Contact: BCE on soft labels
        loss_contact = F.binary_cross_entropy_with_logits(
            pred["contact_logits"], gt_contact, reduction="none",
        )  # (B, T, 5)

        # Target: smooth-L1 (Huber) on xyz in object-local frame. Summed
        # over the coord dim so loss scales like a single L1 distance,
        # which is more interpretable than an L2 squared-distance.
        loss_target = F.smooth_l1_loss(
            pred["contact_target_xyz"], gt_target, reduction="none",
        ).sum(dim=-1)  # (B, T, 5)

        # Phase: CE on integer labels + label smoothing + optional
        # logit-adjustment (Menon ICLR'21) + optional focal.
        B, T, P = pred["phase_logits"].shape
        phase_logits_flat = pred["phase_logits"].reshape(-1, P)
        phase_gt_flat = gt_phase.reshape(-1)
        if self.logit_adjust_phase is not None:
            phase_logits_flat = phase_logits_flat + (
                self.logit_adjust_tau * self.logit_adjust_phase
            )
        loss_phase_flat = F.cross_entropy(
            phase_logits_flat, phase_gt_flat,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        if self.focal_gamma > 0:
            loss_phase_flat = loss_phase_flat * self._focal_weight(
                phase_logits_flat, phase_gt_flat, self.focal_gamma,
            )
        loss_phase = loss_phase_flat.reshape(B, T)  # (B, T)

        # Support: CE on integer labels + label smoothing + optional
        # logit-adjustment + optional focal.
        S = pred["support_logits"].shape[-1]
        support_logits_flat = pred["support_logits"].reshape(-1, S)
        support_gt_flat = gt_support.reshape(-1)
        if self.logit_adjust_support is not None:
            support_logits_flat = support_logits_flat + (
                self.logit_adjust_tau * self.logit_adjust_support
            )
        loss_support_flat = F.cross_entropy(
            support_logits_flat, support_gt_flat,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        if self.focal_gamma > 0:
            loss_support_flat = loss_support_flat * self._focal_weight(
                support_logits_flat, support_gt_flat, self.focal_gamma,
            )
        loss_support = loss_support_flat.reshape(B, T)  # (B, T)

        # Build frame mask and contact gate
        if mask is not None:
            frame_mask = mask.float()                             # (B, T)
        else:
            frame_mask = torch.ones(
                gt_contact.shape[:2],
                device=gt_contact.device, dtype=torch.float32,
            )
        # Contact gate: only supervise contact_target where the GT
        # contact label crosses threshold — outside contact the pseudo-
        # target is uninformative.
        contact_gate = (
            (gt_contact > self.contact_threshold).float()
            * frame_mask.unsqueeze(-1)
        )                                                         # (B, T, 5)

        n_frames = frame_mask.sum() + 1e-8
        n_contact = contact_gate.sum() + 1e-8
        num_parts = gt_contact.shape[-1]

        loss_contact = (
            (loss_contact * frame_mask.unsqueeze(-1)).sum()
            / (n_frames * num_parts)
        )
        loss_target = (loss_target * contact_gate).sum() / n_contact
        loss_phase = (loss_phase * frame_mask).sum() / n_frames
        loss_support = (loss_support * frame_mask).sum() / n_frames

        if self.kendall is not None:
            # Kendall et al. CVPR'18 multi-task uncertainty weighting.
            # Static *_weight values are ignored. The optimiser learns
            # per-task log-variances; combined loss equalises gradient
            # contribution by construction.
            total, kendall_log = self.kendall({
                "contact": loss_contact,
                "target": loss_target,
                "phase": loss_phase,
                "support": loss_support,
            })
        else:
            total = (
                self.contact_weight * loss_contact
                + self.target_weight * loss_target
                + self.phase_weight * loss_phase
                + self.support_weight * loss_support
            )
            kendall_log = {}

        # Unweighted sum of raw per-task losses — the supervision
        # signal that actually tracks model quality, with no Kendall
        # combinator and no `+0.5·s_i` term. v4 found that selecting
        # best_val on the Kendall-combined ``loss`` was broken: as
        # log_var_target descended (good), the `0.5·s` contribution
        # made total ``loss`` decrease monotonically regardless of
        # supervision quality, so best_val.pt got saved at epoch 4
        # (before Kendall took off) and final.pt was 3× better on
        # phase macro-F1. Use this key for ``val_best_key`` instead.
        loss_unweighted = loss_contact + loss_target + loss_phase + loss_support

        return {
            "loss": total,
            "loss_unweighted": loss_unweighted,
            "loss_contact": loss_contact,
            "loss_target": loss_target,
            "loss_phase": loss_phase,
            "loss_support": loss_support,
            "n_contact_frames": n_contact.detach(),
            **kendall_log,
        }


# ============================================================================
# Stage B: Generator losses
# ============================================================================

class GeneratorLoss(nn.Module):
    """Loss for the Motion Generator (MoMask masked transformer).

    The primary loss is masked token prediction CE (computed inside the
    MaskedTransformer forward). This module adds optional velocity
    smoothness regularization on the decoded motion.
    """

    def __init__(self, velocity_smoothness_weight: float = 0.01) -> None:
        super().__init__()
        self.velocity_smoothness_weight = velocity_smoothness_weight

    def forward(
        self,
        mask_pred_loss: Tensor,
        decoded_motion: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute generator loss.

        Parameters
        ----------
        mask_pred_loss : scalar — CE loss on masked tokens (from MaskedTransformer.forward)
        decoded_motion : (B, T, 263) — optionally decoded motion for smoothness
        """
        total = mask_pred_loss

        losses = {
            "loss_mask_pred": mask_pred_loss,
        }

        if decoded_motion is not None and self.velocity_smoothness_weight > 0:
            loss_smooth = velocity_smoothness_loss(decoded_motion)
            total = total + self.velocity_smoothness_weight * loss_smooth
            losses["loss_smoothness"] = loss_smooth

        losses["loss"] = total
        return losses


# ============================================================================
# Stage C: Consistency loss
# ============================================================================

class ConsistencyLoss(nn.Module):
    """Consistency loss between input interaction latent and extracted latent.

    Ensures the generator doesn't ignore the interaction conditioning by
    requiring that the generated motion, when passed through the extractor,
    recovers the original interaction labels.
    """

    def __init__(
        self,
        contact_weight: float = 1.0,
        phase_weight: float = 0.5,
        support_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.contact_weight = contact_weight
        self.phase_weight = phase_weight
        self.support_weight = support_weight

    def forward(
        self,
        extracted: dict[str, Tensor],
        original: dict[str, Tensor],
        mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute consistency between extracted and original interaction labels.

        Parameters
        ----------
        extracted : output from InteractionExtractor on generated motion
        original : output from InteractionPredictor (the conditioning labels)
        mask : (B, T) — True for valid frames
        """
        # Contact consistency (BCE between two soft predictions)
        loss_contact = F.mse_loss(
            extracted["contact_state"], original["contact_state"].detach(),
            reduction="none",
        )

        # Phase consistency (KL between two distributions)
        loss_phase = F.kl_div(
            F.log_softmax(extracted["phase_logits"], dim=-1),
            original["phase"].detach(),
            reduction="none",
        ).sum(dim=-1)

        # Support consistency
        loss_support = F.kl_div(
            F.log_softmax(extracted["support_logits"], dim=-1),
            original["support"].detach(),
            reduction="none",
        ).sum(dim=-1)

        if mask is not None:
            frame_mask = mask.float()
            loss_contact = (loss_contact * frame_mask.unsqueeze(-1)).sum() / (frame_mask.sum() * 5 + 1e-8)
            loss_phase = (loss_phase * frame_mask).sum() / (frame_mask.sum() + 1e-8)
            loss_support = (loss_support * frame_mask).sum() / (frame_mask.sum() + 1e-8)
        else:
            loss_contact = loss_contact.mean()
            loss_phase = loss_phase.mean()
            loss_support = loss_support.mean()

        total = (
            self.contact_weight * loss_contact
            + self.phase_weight * loss_phase
            + self.support_weight * loss_support
        )

        return {
            "loss": total,
            "loss_consistency_contact": loss_contact,
            "loss_consistency_phase": loss_phase,
            "loss_consistency_support": loss_support,
        }


# ============================================================================
# Helpers
# ============================================================================

def velocity_smoothness_loss(motion: Tensor) -> Tensor:
    """Penalize high acceleration (second-order finite difference).

    Parameters
    ----------
    motion : (B, T, D) — motion feature sequence

    Returns
    -------
    Scalar loss — mean squared acceleration.
    """
    # velocity: (B, T-1, D)
    vel = motion[:, 1:, :] - motion[:, :-1, :]
    # acceleration: (B, T-2, D)
    acc = vel[:, 1:, :] - vel[:, :-1, :]
    return acc.pow(2).mean()
