"""Inference-time anchor corrections for AnchorDiff samples."""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from piano.training.anchor_consistency_loss import (
    PART_TO_JOINT,
    lift_object_local_to_world,
)


def translate_world_joints_to_active_anchors(
    joints_world: Tensor,
    contact_state: Tensor,
    contact_target_xyz_local: Tensor,
    object_positions: Tensor,
    object_rotations: Tensor,
    *,
    strength: float = 1.0,
    contact_threshold: float = 0.5,
    smooth_window: int = 9,
    max_offset_m: float = 0.75,
    part_weights: tuple[float, ...] = (1.0, 1.0, 0.0, 0.0, 1.0),
    seq_mask: Tensor | None = None,
) -> tuple[Tensor, dict[str, Any]]:
    """Translate each frame's whole body toward its active object anchors.

    This is an inference-only correction for world-joint AnchorDiff outputs.
    It estimates one translation offset per frame from active body-part anchors
    and applies that offset to all joints, preserving the generated skeleton
    shape. It intentionally does not move individual hands independently; that
    would hide contact errors by stretching limbs.
    """
    if strength <= 0.0:
        return joints_world, {"enabled": False}

    target_world = lift_object_local_to_world(
        contact_target_xyz_local,
        object_positions,
        object_rotations,
    )
    part_idx = torch.tensor(PART_TO_JOINT, device=joints_world.device, dtype=torch.long)
    pred_part = joints_world.index_select(2, part_idx)

    part_w = torch.tensor(part_weights, device=joints_world.device, dtype=joints_world.dtype)
    mask = (contact_state >= contact_threshold).to(joints_world.dtype) * part_w.view(1, 1, -1)
    if seq_mask is not None:
        mask = mask * seq_mask.unsqueeze(-1).to(joints_world.dtype)

    denom = mask.sum(dim=-1, keepdim=True)
    valid = denom > 0.0
    offset = ((target_world - pred_part) * mask.unsqueeze(-1)).sum(dim=2)
    offset = offset / denom.clamp_min(1.0)
    offset = torch.where(valid, offset, torch.zeros_like(offset))

    if max_offset_m > 0.0:
        norm = offset.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        offset = offset * (float(max_offset_m) / norm).clamp(max=1.0)

    smooth_window = int(max(1, smooth_window))
    if smooth_window > 1:
        radius = smooth_window // 2
        smoothed = torch.zeros_like(offset)
        valid_f = valid.to(joints_world.dtype)
        for t in range(offset.shape[1]):
            lo = max(0, t - radius)
            hi = min(offset.shape[1], t + radius + 1)
            win_valid = valid_f[:, lo:hi, :]
            win_denom = win_valid.sum(dim=1).clamp_min(1.0)
            smoothed[:, t, :] = (offset[:, lo:hi, :] * win_valid).sum(dim=1) / win_denom
        offset = smoothed

    adjusted = joints_world + float(strength) * offset.unsqueeze(2)

    with torch.no_grad():
        active_frames = int(valid.sum().item())
        active_parts = float(mask.sum().item())
        offset_norm = offset.norm(dim=-1)
        active_norm = offset_norm[valid.squeeze(-1)]
        mean_offset = float(active_norm.mean().item()) if active_norm.numel() else 0.0
        max_offset = float(active_norm.max().item()) if active_norm.numel() else 0.0

    return adjusted, {
        "enabled": True,
        "strength": float(strength),
        "active_frames": active_frames,
        "active_parts": active_parts,
        "mean_offset_m": mean_offset,
        "max_offset_m": max_offset,
    }
