"""Decoded-space contact auxiliary loss for Stage B C2.

The base Stage B objective is MoMask's masked CE on base RVQ tokens. C2 adds
one geometric term by decoding a relaxed base-token distribution through the
frozen RVQ-VAE decoder, recovering SMPL-22 joints with MoMask's upstream
``recover_from_ric``, and measuring body-to-object distance in the same
HumanML3D canonical frame as ``z_int``.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from piano.inference.contact_guidance import _decode_relaxed_base
from piano.utils.smpl_utils import BODY_PART_INDICES


def rotation_6d_to_matrix_torch(d6: Tensor) -> Tensor:
    """Convert Zhou-2019 first-two-columns 6D rotations to matrices.

    ``piano.utils.canonical_frame.matrix_to_rotation_6d_np`` stores
    ``R[..., :, :2].reshape(..., 6)``. Inverting that layout requires
    reshaping to ``(..., 3, 2)`` before Gram-Schmidt orthonormalization.
    """
    if d6.shape[-1] != 6:
        raise ValueError(f"rotation_6d_to_matrix_torch expects last dim 6, got {d6.shape[-1]}")

    cols = d6.reshape(*d6.shape[:-1], 3, 2)
    a1 = cols[..., :, 0]
    a2 = cols[..., :, 1]

    b1 = F.normalize(a1, dim=-1, eps=1e-6)
    a2_orth = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(a2_orth, dim=-1, eps=1e-6)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def object_pc_to_canonical_torch(
    object_pc_local: Tensor,       # (B, N, 3)
    obj_com_canonical: Tensor,     # (B, T, 3)
    obj_rot6d_canonical: Tensor,   # (B, T, 6)
    *,
    num_points: int | None = None,
) -> Tensor:
    """Lift object-local point clouds to the body-canonical frame."""
    if num_points is not None and num_points > 0:
        object_pc_local = object_pc_local[:, :num_points]

    R = rotation_6d_to_matrix_torch(obj_rot6d_canonical)             # (B, T, 3, 3)
    pc = torch.einsum("btij,bnj->btni", R, object_pc_local)          # (B, T, N, 3)
    return pc + obj_com_canonical[:, :, None, :]


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    mask_f = mask.to(device=values.device, dtype=values.dtype)
    denom = mask_f.sum().clamp(min=1.0)
    return (values * mask_f).sum() / denom


def _eval_metric_loss_canonical(
    body_canonical: Tensor,        # (B, T, P, 3)
    pc_canonical: Tensor,          # (B, T, N, 3)
    frame_mask: Tensor,            # (B, T)
) -> tuple[Tensor, Tensor]:
    """Differentiable canonical-frame version of mean min contact distance."""
    diff = body_canonical[:, :, :, None, :] - pc_canonical[:, :, None, :, :]
    d = torch.linalg.vector_norm(diff, dim=-1)                       # (B, T, P, N)
    d_min_pc = d.min(dim=-1).values                                  # (B, T, P)
    d_min_parts = d_min_pc.min(dim=-1).values                        # (B, T)
    loss = _masked_mean(d_min_parts, frame_mask)
    return loss, d_min_parts


def _base_logits_to_bsv(base_logits: Tensor, token_count: int) -> Tensor:
    if base_logits.ndim != 3:
        raise ValueError(f"base_logits must be rank-3, got shape {tuple(base_logits.shape)}")
    if base_logits.shape[1] == token_count:
        return base_logits
    if base_logits.shape[2] == token_count:
        return base_logits.transpose(1, 2).contiguous()
    raise ValueError(
        "base_logits must be shaped (B, S, V) or (B, V, S); "
        f"token_count={token_count}, shape={tuple(base_logits.shape)}",
    )


def _valid_residual_ids(all_indices: Tensor, m_lens_tok: Tensor) -> Tensor:
    residual_ids = all_indices[..., 1:].long()
    _, S, _ = residual_ids.shape
    token_mask = (
        torch.arange(S, device=residual_ids.device).unsqueeze(0)
        < m_lens_tok.to(device=residual_ids.device).long().clamp(min=1).unsqueeze(1)
    )
    return torch.where(token_mask[:, :, None], residual_ids, torch.zeros_like(residual_ids))


def decoded_contact_aux_loss(
    *,
    base_logits: Tensor,
    all_indices: Tensor,
    vq_model: nn.Module,
    motion_mean: Tensor,
    motion_std: Tensor,
    batch: Mapping[str, Any],
    m_lens_tok: Tensor,
    num_object_points: int = 256,
    temperature: float = 1.0,
    mode: str = "metric",
    body_part_indices: Sequence[int] = tuple(BODY_PART_INDICES),
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute C2 decoded contact auxiliary loss.

    ``mode="metric"`` matches the offline contact-distance selector: for each
    frame, take the closest object point to each candidate body part, then the
    closest body part, then average over valid frames.
    """
    if mode != "metric":
        raise ValueError(f"decoded contact aux currently supports mode='metric', got {mode!r}")

    required = ("object_pc", "obj_com_canonical", "obj_rot6d_canonical", "seq_len")
    missing = [k for k in required if k not in batch]
    if missing:
        raise KeyError(f"decoded contact aux requires batch keys {missing}")

    device = base_logits.device
    dtype = torch.float32
    S = int(all_indices.shape[1])
    base_logits_bsv = _base_logits_to_bsv(base_logits, token_count=S)
    token_mask = (
        torch.arange(S, device=device).unsqueeze(0)
        < m_lens_tok.to(device=device).long().clamp(min=1).unsqueeze(1)
    )
    pad_logits = torch.full_like(base_logits_bsv, -30.0)
    pad_logits[..., 0] = 30.0
    base_logits_bsv = torch.where(token_mask[:, :, None], base_logits_bsv, pad_logits)
    residual_ids = _valid_residual_ids(all_indices.to(device), m_lens_tok.to(device))

    motion_norm = _decode_relaxed_base(
        base_logits=base_logits_bsv,
        residual_ids=residual_ids,
        vq_model=vq_model,
        temperature=temperature,
    )
    motion = (
        motion_norm.float() * motion_std.to(device=device, dtype=dtype).view(1, 1, -1)
        + motion_mean.to(device=device, dtype=dtype).view(1, 1, -1)
    )

    import piano.models.backbones.momask_adapter  # noqa: F401
    from utils.motion_process import recover_from_ric

    joints_canonical = recover_from_ric(motion, joints_num=22).float()      # (B, T, 22, 3)
    body_idx = torch.as_tensor(body_part_indices, device=device, dtype=torch.long)
    body = joints_canonical.index_select(dim=2, index=body_idx)             # (B, T, P, 3)

    T_dec = int(body.shape[1])
    object_pc = batch["object_pc"].to(device=device, dtype=body.dtype)
    obj_com = batch["obj_com_canonical"].to(device=device, dtype=body.dtype)[:, :T_dec]
    obj_rot6d = batch["obj_rot6d_canonical"].to(device=device, dtype=body.dtype)[:, :T_dec]
    pc = object_pc_to_canonical_torch(
        object_pc,
        obj_com,
        obj_rot6d,
        num_points=num_object_points,
    )

    T = min(int(body.shape[1]), int(pc.shape[1]))
    body = body[:, :T]
    pc = pc[:, :T]
    frames_per_token = max(1, T_dec // max(S, 1))
    decoded_valid_frames = (
        m_lens_tok.to(device=device).long().clamp(min=1) * frames_per_token
    )
    seq_len = batch["seq_len"].to(device=device).long()
    valid_frames = torch.minimum(seq_len, decoded_valid_frames).clamp(min=1, max=T)
    frame_mask = torch.arange(T, device=device).unsqueeze(0) < valid_frames.unsqueeze(1)

    loss, _per_frame = _eval_metric_loss_canonical(body, pc, frame_mask)
    return loss, {
        "decoded_contact_aux_mean_min_dist": loss.detach(),
        "decoded_contact_aux_valid_frames": frame_mask.float().sum().detach(),
    }
