"""Weak physical priors for interaction predictor regularization.

These losses constrain the predicted interaction latents to be
physically plausible without requiring a full physics simulator.
Applied during Stage A (predictor training) and Stage C (joint finetune).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class PhysicalPriors(nn.Module):
    """Combined weak physical prior losses.

    Parameters
    ----------
    reachability_weight : penalize contacts beyond arm reach
    contact_persistence_weight : penalize single-frame contact flickers
    support_smoothness_weight : penalize rapid support state changes
    phase_monotonicity_weight : penalize backward phase transitions
    """

    def __init__(
        self,
        reachability_weight: float = 0.1,
        contact_persistence_weight: float = 0.1,
        support_smoothness_weight: float = 0.05,
        phase_monotonicity_weight: float = 0.05,
    ) -> None:
        super().__init__()
        self.reachability_weight = reachability_weight
        self.contact_persistence_weight = contact_persistence_weight
        self.support_smoothness_weight = support_smoothness_weight
        self.phase_monotonicity_weight = phase_monotonicity_weight

    def forward(
        self,
        pred: dict[str, Tensor],
        joints: Tensor | None = None,
        arm_length: float = 0.6,
        mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute all physical prior losses.

        Parameters
        ----------
        pred : output from InteractionPredictor
        joints : (B, T, 22, 3) — joint positions (for reachability)
        arm_length : approximate max arm reach in meters
        mask : (B, T) — True for valid frames
        """
        losses: dict[str, Tensor] = {}
        total = torch.tensor(0.0, device=pred["contact_state"].device)

        # Contact persistence: penalize frame-to-frame flickers
        if self.contact_persistence_weight > 0:
            loss = contact_persistence_loss(pred["contact_state"], mask)
            losses["loss_persistence"] = loss
            total = total + self.contact_persistence_weight * loss

        # Support smoothness: penalize rapid support state changes
        if self.support_smoothness_weight > 0:
            loss = support_smoothness_loss(pred["support"], mask)
            losses["loss_support_smooth"] = loss
            total = total + self.support_smoothness_weight * loss

        # Phase monotonicity: penalize backward transitions
        if self.phase_monotonicity_weight > 0:
            loss = phase_monotonicity_loss(pred["phase"], mask)
            losses["loss_phase_mono"] = loss
            total = total + self.phase_monotonicity_weight * loss

        # Reachability: penalize hand contacts beyond arm reach
        if self.reachability_weight > 0 and joints is not None:
            loss = reachability_loss(pred["contact_state"], joints, arm_length, mask)
            losses["loss_reachability"] = loss
            total = total + self.reachability_weight * loss

        losses["loss"] = total
        return losses


# ============================================================================
# Individual prior losses
# ============================================================================

def contact_persistence_loss(
    contact_state: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """Penalize frame-to-frame contact flickers.

    Real contacts are temporally persistent — they don't appear and
    disappear within a single frame.

    Parameters
    ----------
    contact_state : (B, T, 5) — predicted soft contact probabilities
    mask : (B, T) — True for valid frames
    """
    # Temporal difference: (B, T-1, 5)
    diff = (contact_state[:, 1:, :] - contact_state[:, :-1, :]).abs()

    if mask is not None:
        # Both frames must be valid
        pair_mask = (mask[:, 1:] & mask[:, :-1]).float().unsqueeze(-1)
        return (diff * pair_mask).sum() / (pair_mask.sum() * 5 + 1e-8)
    return diff.mean()


def support_smoothness_loss(
    support: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """Penalize rapid support state oscillation.

    Support transitions (e.g., standing → sitting) should be smooth,
    not flickering between states.

    Parameters
    ----------
    support : (B, T, S) — predicted support state probabilities
    mask : (B, T) — True for valid frames
    """
    # Temporal difference of probability distributions: (B, T-1, S)
    diff = (support[:, 1:, :] - support[:, :-1, :]).abs().sum(dim=-1)  # (B, T-1)

    if mask is not None:
        pair_mask = (mask[:, 1:] & mask[:, :-1]).float()
        return (diff * pair_mask).sum() / (pair_mask.sum() + 1e-8)
    return diff.mean()


def phase_monotonicity_loss(
    phase: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """Penalize backward phase transitions.

    The interaction phase should progress mostly forward:
    approach → pre-contact → stable-contact → manipulation → release.
    Backward jumps (e.g., manipulation → approach) are penalized.

    Parameters
    ----------
    phase : (B, T, P) — predicted phase probabilities (softmax output)
    mask : (B, T) — True for valid frames
    """
    # Expected phase index (soft): weighted sum of phase indices
    P = phase.shape[-1]
    phase_indices = torch.arange(P, device=phase.device, dtype=torch.float)
    expected_phase = (phase * phase_indices).sum(dim=-1)  # (B, T)

    # Penalize when expected phase decreases
    diff = expected_phase[:, :-1] - expected_phase[:, 1:]  # positive = backward
    backward = torch.relu(diff)  # (B, T-1)

    if mask is not None:
        pair_mask = (mask[:, 1:] & mask[:, :-1]).float()
        return (backward * pair_mask).sum() / (pair_mask.sum() + 1e-8)
    return backward.mean()


def reachability_loss(
    contact_state: Tensor,
    joints: Tensor,
    arm_length: float = 0.6,
    mask: Tensor | None = None,
) -> Tensor:
    """Penalize predicted hand contacts that are beyond arm reach.

    If the predictor says "hand is in contact" but the hand joint is
    farther than arm_length from the shoulder, that's physically
    implausible.

    Parameters
    ----------
    contact_state : (B, T, 5) — predicted contact probabilities
        indices 0,1 are left_hand, right_hand
    joints : (B, T, 22, 3) — joint positions
    arm_length : max reach in meters
    mask : (B, T) — True for valid frames
    """
    # Left hand: wrist (20) to shoulder (16)
    left_dist = torch.norm(joints[:, :, 20, :] - joints[:, :, 16, :], dim=-1)  # (B, T)
    # Right hand: wrist (21) to shoulder (17)
    right_dist = torch.norm(joints[:, :, 21, :] - joints[:, :, 17, :], dim=-1)  # (B, T)

    # Penalty: contact_probability * max(0, distance - arm_length)
    left_violation = contact_state[:, :, 0] * torch.relu(left_dist - arm_length)
    right_violation = contact_state[:, :, 1] * torch.relu(right_dist - arm_length)
    violation = left_violation + right_violation  # (B, T)

    if mask is not None:
        return (violation * mask.float()).sum() / (mask.float().sum() + 1e-8)
    return violation.mean()
