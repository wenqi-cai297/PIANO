"""Loss functions for all PIANO training stages.

Stage A (Predictor): pseudo-label supervision losses
Stage B (Generator): masked token prediction loss
Stage C (Joint):     predictor + generator + consistency loss
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ============================================================================
# Stage A: Interaction Predictor losses
# ============================================================================

class PredictorLoss(nn.Module):
    """Combined loss for the Interaction Predictor.

    Supervises contact state, contact target, interaction phase, and
    support state against pseudo-labels.
    """

    def __init__(
        self,
        contact_weight: float = 1.0,
        target_weight: float = 0.5,
        phase_weight: float = 0.5,
        support_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.contact_weight = contact_weight
        self.target_weight = target_weight
        self.phase_weight = phase_weight
        self.support_weight = support_weight

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
        pred : output dict from InteractionPredictor (contains *_logits keys)
        gt_contact : (B, T, 5) — soft contact pseudo-labels
        gt_target : (B, T, 5, K) — soft target pseudo-labels
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

        # Target: CE on soft labels (use KL divergence for soft targets)
        pred_target_log = F.log_softmax(pred["target_logits"], dim=-1)  # (B, T, 5, K)
        loss_target = F.kl_div(
            pred_target_log, gt_target, reduction="none",
        ).sum(dim=-1)  # (B, T, 5)

        # Phase: CE on integer labels
        B, T, P = pred["phase_logits"].shape
        loss_phase = F.cross_entropy(
            pred["phase_logits"].reshape(-1, P),
            gt_phase.reshape(-1),
            reduction="none",
        ).reshape(B, T)  # (B, T)

        # Support: CE on integer labels
        S = pred["support_logits"].shape[-1]
        loss_support = F.cross_entropy(
            pred["support_logits"].reshape(-1, S),
            gt_support.reshape(-1),
            reduction="none",
        ).reshape(B, T)  # (B, T)

        # Apply frame mask (ignore padded frames)
        if mask is not None:
            frame_mask = mask.float()  # (B, T)
            loss_contact = (loss_contact * frame_mask.unsqueeze(-1)).sum() / (frame_mask.sum() * 5 + 1e-8)
            loss_target = (loss_target * frame_mask.unsqueeze(-1)).sum() / (frame_mask.sum() * 5 + 1e-8)
            loss_phase = (loss_phase * frame_mask).sum() / (frame_mask.sum() + 1e-8)
            loss_support = (loss_support * frame_mask).sum() / (frame_mask.sum() + 1e-8)
        else:
            loss_contact = loss_contact.mean()
            loss_target = loss_target.mean()
            loss_phase = loss_phase.mean()
            loss_support = loss_support.mean()

        total = (
            self.contact_weight * loss_contact
            + self.target_weight * loss_target
            + self.phase_weight * loss_phase
            + self.support_weight * loss_support
        )

        return {
            "loss": total,
            "loss_contact": loss_contact,
            "loss_target": loss_target,
            "loss_phase": loss_phase,
            "loss_support": loss_support,
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
