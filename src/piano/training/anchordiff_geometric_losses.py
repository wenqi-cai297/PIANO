"""Decoded-space auxiliary losses for PIANO-AnchorDiff.

The base AnchorDiff objective predicts HumanML3D/MoMask ``motion_263``.
Root yaw and root XZ are decoded through cumulative sums, so a local
feature MSE can look acceptable while decoded late-frame joints drift.
These losses compare the decoded motion that downstream visualization and
contact metrics actually use.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from piano.training.anchor_consistency_loss import lift_motion263_to_joints


@dataclass(frozen=True, slots=True)
class MotionGeometricLossConfig:
    """Weights and thresholds for decoded motion geometry losses."""

    enabled: bool = False
    pos_weight: float = 0.0
    vel_weight: float = 0.0
    foot_weight: float = 0.0
    foot_contact_threshold: float = 0.5
    foot_velocity_threshold: float = 0.01
    foot_joint_indices: tuple[int, int, int, int] = (7, 10, 8, 11)


def _zero_like_loss(x: Tensor) -> Tensor:
    return x.sum() * 0.0


def decoded_joint_position_loss(
    joints_pred: Tensor,
    joints_target: Tensor,
    seq_mask: Tensor,
) -> Tensor:
    """MSE on decoded canonical 22-joint positions.

    ``joints_*`` are shaped ``(B, T, J, 3)``. We sum XYZ per joint, then
    average over valid frames and joints. This mirrors the task-space
    error users see after ``recover_from_ric`` integrates root channels.
    """
    valid = seq_mask.float().unsqueeze(-1)  # (B, T, 1)
    sq = (joints_pred - joints_target).pow(2).sum(dim=-1)
    denom = (valid.sum() * joints_pred.shape[2]).clamp_min(1.0)
    return (sq * valid).sum() / denom


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


def foot_zero_velocity_loss(
    joints_pred: Tensor,
    joints_target: Tensor,
    foot_contact_target: Tensor,
    seq_mask: Tensor,
    cfg: MotionGeometricLossConfig,
) -> Tensor:
    """MDM-style foot loss: predicted foot velocity is zero on GT-static feet."""
    if joints_pred.shape[1] < 2:
        return _zero_like_loss(joints_pred)

    foot_idx = torch.tensor(
        cfg.foot_joint_indices,
        device=joints_pred.device,
        dtype=torch.long,
    )
    pred_feet = joints_pred.index_select(2, foot_idx)
    target_feet = joints_target.index_select(2, foot_idx)

    pred_vel = pred_feet[:, 1:] - pred_feet[:, :-1]
    target_vel = target_feet[:, 1:] - target_feet[:, :-1]
    target_speed = target_vel.norm(dim=-1)

    valid_pair = (seq_mask[:, 1:] * seq_mask[:, :-1]).bool()
    contact = foot_contact_target[:, :-1] >= cfg.foot_contact_threshold
    static = target_speed <= cfg.foot_velocity_threshold
    mask = (valid_pair.unsqueeze(-1) & contact & static).float()

    sq = pred_vel.pow(2).sum(dim=-1)
    return (sq * mask).sum() / mask.sum().clamp_min(1.0)


def compute_motion_geometric_losses(
    x0_pred: Tensor,
    x0_target: Tensor,
    seq_mask: Tensor,
    cfg: MotionGeometricLossConfig,
) -> dict[str, Tensor]:
    """Return weighted total plus unweighted decoded geometry components."""
    zero = _zero_like_loss(x0_pred)
    if not cfg.enabled:
        return {
            "loss_geometric": zero,
            "loss_pos": zero,
            "loss_vel": zero,
            "loss_foot": zero,
        }

    x0_pred_f = x0_pred.float()
    x0_target_f = x0_target.float()
    seq_mask_f = seq_mask.float()

    need_joints = cfg.pos_weight > 0.0 or cfg.foot_weight > 0.0
    if need_joints:
        joints_pred = lift_motion263_to_joints(x0_pred_f)
        joints_target = lift_motion263_to_joints(x0_target_f)
    else:
        joints_pred = joints_target = None

    pos = (
        decoded_joint_position_loss(joints_pred, joints_target, seq_mask_f)
        if cfg.pos_weight > 0.0 and joints_pred is not None
        else zero
    )
    vel = (
        feature_velocity_loss(x0_pred_f, x0_target_f, seq_mask_f)
        if cfg.vel_weight > 0.0
        else zero
    )
    foot = (
        foot_zero_velocity_loss(
            joints_pred=joints_pred,
            joints_target=joints_target,
            foot_contact_target=x0_target_f[..., 259:263],
            seq_mask=seq_mask_f,
            cfg=cfg,
        )
        if cfg.foot_weight > 0.0 and joints_pred is not None
        else zero
    )

    total = (
        cfg.pos_weight * pos
        + cfg.vel_weight * vel
        + cfg.foot_weight * foot
    )
    return {
        "loss_geometric": total,
        "loss_pos": pos,
        "loss_vel": vel,
        "loss_foot": foot,
    }
