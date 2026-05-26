"""Geometric auxiliary loss helpers for PIANO-AnchorDiff (R29 minimal set).

After the R29 Tier-1 cleanup only ``feature_velocity_loss`` survives;
``compute_motion_geometric_losses`` and ``MotionGeometricLossConfig``
were motion_263-only and removed.
"""
from __future__ import annotations

import torch
from torch import Tensor


def _zero_like_loss(x: Tensor) -> Tensor:
    return x.sum() * 0.0


def feature_velocity_loss(
    x0_pred: Tensor,
    x0_target: Tensor,
    seq_mask: Tensor,
) -> Tensor:
    """MSE on temporal differences of the full motion feature vector."""
    if x0_pred.shape[1] < 2:
        return _zero_like_loss(x0_pred)
    valid_pair = (seq_mask[:, 1:] * seq_mask[:, :-1]).float()
    pred_v = x0_pred[:, 1:] - x0_pred[:, :-1]
    target_v = x0_target[:, 1:] - x0_target[:, :-1]
    sq = (pred_v - target_v).pow(2).sum(dim=-1)
    return (sq * valid_pair).sum() / valid_pair.sum().clamp_min(1.0)
