"""Decoded-space contact auxiliary loss for Stage B C2/C2b.

The base Stage B objective is MoMask's masked CE on base RVQ tokens. C2 adds
one geometric term by decoding a relaxed base-token distribution through the
frozen RVQ-VAE decoder, recovering SMPL-22 joints with MoMask's upstream
``recover_from_ric``, and measuring body-to-object distance in the same
HumanML3D canonical frame as ``z_int``.

C2b extends that path through the residual transformer: soft base logits are
rolled through MoMask's residual RVQ predictor layer by layer, and the full
relaxed RVQ stack is decoded before measuring contact.
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


def body_canonical_to_object_local_torch(
    body_canonical: Tensor,       # (B, T, P, 3)
    obj_com_canonical: Tensor,    # (B, T, 3)
    obj_rot6d_canonical: Tensor,  # (B, T, 6)
) -> Tensor:
    """Transform canonical-frame body points into the object's local frame."""
    R = rotation_6d_to_matrix_torch(obj_rot6d_canonical)             # (B, T, 3, 3)
    centered = body_canonical - obj_com_canonical[:, :, None, :]
    return torch.einsum("btij,btpj->btpi", R.transpose(-1, -2), centered)


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    mask_f = mask.to(device=values.device, dtype=values.dtype)
    denom = mask_f.sum().clamp(min=1.0)
    return (values * mask_f).sum() / denom


def _weighted_mean(values: Tensor, weights: Tensor) -> Tensor:
    weights = weights.to(device=values.device, dtype=values.dtype)
    denom = weights.sum().clamp(min=1e-6)
    return (values * weights).sum() / denom


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


def _object_motion_speed_from_canonical(
    obj_com_canonical: Tensor,     # (B, T, 3)
    obj_rot6d_canonical: Tensor,   # (B, T, 6)
    *,
    fps: float,
    radius_proxy: float,
) -> Tensor:
    """Object speed proxy matching pseudo-label kinematic coupling."""
    B, T, _ = obj_com_canonical.shape
    speed = torch.zeros(B, T, device=obj_com_canonical.device, dtype=obj_com_canonical.dtype)
    if T <= 1:
        return speed

    trans_speed = torch.linalg.vector_norm(
        obj_com_canonical[:, 1:] - obj_com_canonical[:, :-1],
        dim=-1,
    ) * float(fps)

    R = rotation_6d_to_matrix_torch(obj_rot6d_canonical)
    R_rel = torch.matmul(R[:, 1:], R[:, :-1].transpose(-1, -2))
    trace = R_rel.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cos = ((trace - 1.0) * 0.5).clamp(min=-1.0, max=1.0)
    ang_speed = torch.acos(cos) * float(fps)

    speed[:, 1:] = trans_speed + float(radius_proxy) * ang_speed
    return speed


def _target_trajectory_loss_canonical(
    *,
    body_canonical: Tensor,        # (B, T, P, 3)
    obj_com_canonical: Tensor,     # (B, T, 3)
    obj_rot6d_canonical: Tensor,   # (B, T, 6)
    contact_state: Tensor,         # (B, T, P)
    contact_target_xyz: Tensor,    # (B, T, P, 3), object-local
    frame_mask: Tensor,            # (B, T)
    position_weight: float,
    velocity_weight: float,
    metric_loss: Tensor,
    metric_weight: float,
    moving_frame_extra_weight: float,
    contact_threshold: float,
    use_soft_contact_weights: bool,
    velocity_moving_only: bool,
    fps: float,
    moving_speed_threshold: float,
    kin_radius_proxy: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Part-specific contact-target and object-local trajectory objective.

    Unlike the legacy metric loss, this loss never minimises over arbitrary
    body parts or arbitrary object points. It supervises the body part named by
    ``contact_state`` against that part's object-local ``contact_target_xyz``,
    and it adds a local-frame velocity term on moving-object contact frames.
    """
    dtype = body_canonical.dtype
    body_local = body_canonical_to_object_local_torch(
        body_canonical,
        obj_com_canonical,
        obj_rot6d_canonical,
    )
    contact = contact_state.to(device=body_canonical.device, dtype=dtype).clamp(0.0, 1.0)
    target = contact_target_xyz.to(device=body_canonical.device, dtype=dtype)
    frame_mask_f = frame_mask.to(device=body_canonical.device, dtype=dtype)
    contact_binary = contact >= float(contact_threshold)

    obj_speed = _object_motion_speed_from_canonical(
        obj_com_canonical,
        obj_rot6d_canonical,
        fps=float(fps),
        radius_proxy=float(kin_radius_proxy),
    )
    moving = obj_speed >= float(moving_speed_threshold)
    moving_f = moving.to(dtype=dtype)

    contact_strength = contact if use_soft_contact_weights else contact_binary.to(dtype=dtype)
    valid_part = contact_binary.to(dtype=dtype) * frame_mask_f[:, :, None]
    moving_boost = 1.0 + float(moving_frame_extra_weight) * moving_f[:, :, None]
    pos_weights = valid_part * contact_strength * moving_boost

    pos_dist = torch.linalg.vector_norm(body_local - target, dim=-1)
    target_position = _weighted_mean(pos_dist, pos_weights)

    if body_local.shape[1] > 1:
        pred_delta = body_local[:, 1:] - body_local[:, :-1]
        target_delta = target[:, 1:] - target[:, :-1]
        vel_dist = torch.linalg.vector_norm(pred_delta - target_delta, dim=-1)
        pair_frame = (frame_mask[:, 1:] & frame_mask[:, :-1]).to(dtype=dtype)
        pair_contact = torch.minimum(contact_strength[:, 1:], contact_strength[:, :-1])
        pair_binary = contact_binary[:, 1:] & contact_binary[:, :-1]
        vel_weights = pair_frame[:, :, None] * pair_binary.to(dtype=dtype) * pair_contact
        if velocity_moving_only:
            pair_moving = moving[:, 1:] | moving[:, :-1]
            vel_weights = vel_weights * pair_moving.to(dtype=dtype)[:, :, None]
        target_velocity = _weighted_mean(vel_dist, vel_weights)
    else:
        vel_weights = torch.zeros_like(pos_weights[:, :0])
        target_velocity = body_canonical.new_zeros(())

    total = (
        float(position_weight) * target_position
        + float(velocity_weight) * target_velocity
        + float(metric_weight) * metric_loss
    )

    valid_frames = frame_mask_f.sum().clamp(min=1.0)
    valid_part_slots = (frame_mask_f[:, :, None].expand_as(contact)).sum().clamp(min=1.0)
    contact_part_count = valid_part.sum()
    moving_frames = (moving_f * frame_mask_f).sum()
    moving_contact_parts = (valid_part * moving_f[:, :, None]).sum()
    moving_pos_loss = _weighted_mean(
        pos_dist,
        valid_part * contact_strength * moving_f[:, :, None],
    )
    static_pos_loss = _weighted_mean(
        pos_dist,
        valid_part * contact_strength * (1.0 - moving_f[:, :, None]),
    )

    metrics = {
        "decoded_contact_aux_target_position": target_position.detach(),
        "decoded_contact_aux_target_velocity": target_velocity.detach(),
        "decoded_contact_aux_metric_component": metric_loss.detach(),
        "decoded_contact_aux_target_position_moving": moving_pos_loss.detach(),
        "decoded_contact_aux_target_position_static": static_pos_loss.detach(),
        "decoded_contact_aux_contact_part_frac": (
            contact_part_count / valid_part_slots
        ).detach(),
        "decoded_contact_aux_moving_frame_frac": (
            moving_frames / valid_frames
        ).detach(),
        "decoded_contact_aux_moving_contact_part_frac": (
            moving_contact_parts / contact_part_count.clamp(min=1.0)
        ).detach(),
        "decoded_contact_aux_target_position_weight_sum": pos_weights.sum().detach(),
        "decoded_contact_aux_target_velocity_weight_sum": vel_weights.sum().detach(),
        "decoded_contact_aux_object_speed_mean": _masked_mean(obj_speed, frame_mask).detach(),
    }
    return total, metrics


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


def _logits_to_codebook_vocab(logits_bsv: Tensor, vocab_size: int) -> Tensor:
    """Trim model logits to the RVQ codebook vocabulary.

    MoMask's residual transformer predicts one extra pad token
    (``opt.num_tokens``), but the RVQ decoder codebooks contain only the real
    token ids ``[0, vocab_size)``. Dropping the pad column mirrors generation,
    where pad is masked out before decode.
    """
    if logits_bsv.shape[-1] < vocab_size:
        raise ValueError(
            f"logits vocab {logits_bsv.shape[-1]} is smaller than codebook "
            f"vocab {vocab_size}",
        )
    return logits_bsv[..., :vocab_size]


def _force_pad_to_token_zero(logits_bsv: Tensor, token_mask: Tensor) -> Tensor:
    pad_logits = torch.full_like(logits_bsv, -30.0)
    pad_logits[..., 0] = 30.0
    return torch.where(token_mask[:, :, None], logits_bsv, pad_logits)


def _valid_residual_ids(all_indices: Tensor, m_lens_tok: Tensor) -> Tensor:
    residual_ids = all_indices[..., 1:].long()
    _, S, _ = residual_ids.shape
    token_mask = (
        torch.arange(S, device=residual_ids.device).unsqueeze(0)
        < m_lens_tok.to(device=residual_ids.device).long().clamp(min=1).unsqueeze(1)
    )
    return torch.where(token_mask[:, :, None], residual_ids, torch.zeros_like(residual_ids))


def _decode_relaxed_full_rvq_prediction(
    *,
    base_logits: Tensor,
    residual_transformer: nn.Module,
    vq_model: nn.Module,
    text: Any,
    m_lens_tok: Tensor,
    int_kv: Tensor | None = None,
    int_padding_mask: Tensor | None = None,
    temperature: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Decode soft base logits through a differentiable residual RVQ rollout.

    This mirrors ``ResidualTransformer.generate`` but replaces argmax/Gumbel
    ids with soft codebook expectations. Two embedding spaces are involved:

    - residual-transformer inputs use MoMask's learned
      ``token_embed_weight`` (same as upstream residual training/generation);
    - RVQ-VAE decoding uses the frozen VQ quantizer codebooks.
    """
    quantizer = vq_model.quantizer
    codebooks = quantizer.codebooks
    Q, V = int(codebooks.shape[0]), int(codebooks.shape[1])

    B, S = int(base_logits.shape[0]), int(base_logits.shape[1])
    base_logits_bsv = _base_logits_to_bsv(base_logits, token_count=S)
    base_logits_bsv = _logits_to_codebook_vocab(base_logits_bsv, V)
    token_mask = (
        torch.arange(S, device=base_logits.device).unsqueeze(0)
        < m_lens_tok.to(device=base_logits.device).long().clamp(min=1).unsqueeze(1)
    )
    base_logits_bsv = _force_pad_to_token_zero(base_logits_bsv, token_mask)

    r = getattr(residual_transformer, "residual", residual_transformer)
    if not hasattr(residual_transformer, "trans_forward_with_int"):
        raise TypeError(
            "rvq_path='full_prediction' requires ResidualTransformerWithInteraction "
            "or a compatible module exposing trans_forward_with_int.",
        )
    r.process_embed_proj_weight()

    # Text encoding is frozen in MoMask. Keep it out of autograd, exactly as
    # upstream residual CE training does.
    if r.cond_mode == "text":
        with torch.no_grad():
            cond_vector = r.encode_text(text)
    elif r.cond_mode == "action":
        cond_vector = r.enc_action(text).to(base_logits.device).float()
    elif r.cond_mode == "uncond":
        cond_vector = torch.zeros(B, r.latent_dim, device=base_logits.device)
    else:
        raise NotImplementedError(f"Unsupported cond_mode {r.cond_mode!r}")
    cond_vector = cond_vector.to(base_logits.device).float()

    base_probs = F.softmax(base_logits_bsv / max(temperature, 1e-6), dim=-1)
    decode_emb_sum = base_probs @ codebooks[0]

    token_embed_weight = r.token_embed_weight
    if int(token_embed_weight.shape[0]) < Q - 1:
        raise ValueError(
            f"residual token_embed_weight has {token_embed_weight.shape[0]} layers, "
            f"but VQ has {Q} quantizers",
        )
    if int(token_embed_weight.shape[1]) < V:
        raise ValueError(
            f"residual token_embed_weight vocab {token_embed_weight.shape[1]} "
            f"is smaller than VQ vocab {V}",
        )

    history_sum = base_probs @ token_embed_weight[0, :V]
    padding_mask = ~token_mask
    residual_logits: list[Tensor] = []

    for q in range(1, Q):
        qids = torch.full((B,), q, dtype=torch.long, device=base_logits.device)
        logits_code = residual_transformer.trans_forward_with_int(
            history_sum,
            qids,
            cond_vector,
            padding_mask,
            int_kv=int_kv,
            int_padding_mask=int_padding_mask,
        )
        logits_bvs = r.output_project(logits_code, qids - 1)
        logits_bsv = _base_logits_to_bsv(logits_bvs, token_count=S)
        logits_bsv = _logits_to_codebook_vocab(logits_bsv, V)
        logits_bsv = _force_pad_to_token_zero(logits_bsv, token_mask)
        residual_logits.append(logits_bsv)

        probs = F.softmax(logits_bsv / max(temperature, 1e-6), dim=-1)
        decode_emb_sum = decode_emb_sum + probs @ codebooks[q]
        if q < Q - 1:
            history_sum = history_sum + probs @ token_embed_weight[q, :V]

    motion_norm = vq_model.decoder(decode_emb_sum.permute(0, 2, 1))
    metrics = {
        "decoded_contact_aux_residual_layers": torch.as_tensor(
            len(residual_logits),
            device=base_logits.device,
            dtype=base_logits.dtype,
        ),
    }
    return motion_norm, metrics


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
    rvq_path: str = "base_gt_residual",
    target_position_weight: float = 1.0,
    target_velocity_weight: float = 0.5,
    metric_weight: float = 0.0,
    moving_frame_extra_weight: float = 2.0,
    contact_threshold: float = 0.5,
    use_soft_contact_weights: bool = True,
    velocity_moving_only: bool = True,
    fps: float = 20.0,
    moving_speed_threshold: float = 0.15,
    kin_radius_proxy: float = 0.3,
    residual_transformer: nn.Module | None = None,
    text: Any = None,
    int_kv: Tensor | None = None,
    int_padding_mask: Tensor | None = None,
    body_part_indices: Sequence[int] = tuple(BODY_PART_INDICES),
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute C2 decoded contact auxiliary loss.

    ``mode="metric"`` matches the legacy offline contact-distance selector: for
    each frame, take the closest object point to each candidate body part, then
    the closest body part, then average over valid frames.

    ``mode="target_trajectory"`` is the v0.13 objective. It supervises the
    predicted body part in object-local coordinates against that exact part's
    ``contact_target_xyz`` trajectory under the ``contact_state`` mask, with an
    optional moving-object local-velocity term. This removes the old shortcut
    where any body point could satisfy the loss by approaching any object point.
    """
    supported_modes = {"metric", "target_trajectory"}
    if mode not in supported_modes:
        raise ValueError(
            f"decoded contact aux supports modes {sorted(supported_modes)}, got {mode!r}",
        )

    required = ("object_pc", "obj_com_canonical", "obj_rot6d_canonical", "seq_len")
    if mode == "target_trajectory":
        required = required + ("contact_state", "contact_target_xyz")
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
    base_logits_bsv = _force_pad_to_token_zero(base_logits_bsv, token_mask)

    extra_metrics: dict[str, Tensor] = {}
    if rvq_path == "base_gt_residual":
        residual_ids = _valid_residual_ids(all_indices.to(device), m_lens_tok.to(device))
        motion_norm = _decode_relaxed_base(
            base_logits=base_logits_bsv,
            residual_ids=residual_ids,
            vq_model=vq_model,
            temperature=temperature,
        )
    elif rvq_path == "full_prediction":
        if residual_transformer is None:
            raise ValueError("rvq_path='full_prediction' requires residual_transformer")
        residual_core = getattr(residual_transformer, "residual", residual_transformer)
        if text is None and getattr(residual_core, "cond_mode", None) == "text":
            raise ValueError("rvq_path='full_prediction' requires text")
        motion_norm, extra_metrics = _decode_relaxed_full_rvq_prediction(
            base_logits=base_logits_bsv,
            residual_transformer=residual_transformer,
            vq_model=vq_model,
            text=text,
            m_lens_tok=m_lens_tok,
            int_kv=int_kv,
            int_padding_mask=int_padding_mask,
            temperature=temperature,
        )
    else:
        raise ValueError(
            "decoded contact aux rvq_path must be 'base_gt_residual' "
            f"or 'full_prediction', got {rvq_path!r}",
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
    obj_com = obj_com[:, :T]
    obj_rot6d = obj_rot6d[:, :T]
    frames_per_token = max(1, T_dec // max(S, 1))
    decoded_valid_frames = (
        m_lens_tok.to(device=device).long().clamp(min=1) * frames_per_token
    )
    seq_len = batch["seq_len"].to(device=device).long()
    valid_frames = torch.minimum(seq_len, decoded_valid_frames).clamp(min=1, max=T)
    frame_mask = torch.arange(T, device=device).unsqueeze(0) < valid_frames.unsqueeze(1)

    metric_loss, _per_frame = _eval_metric_loss_canonical(body, pc, frame_mask)
    if mode == "metric":
        loss = metric_loss
        metrics = {
            "decoded_contact_aux_mean_min_dist": metric_loss.detach(),
            "decoded_contact_aux_valid_frames": frame_mask.float().sum().detach(),
        }
    else:
        contact_state = batch["contact_state"].to(device=device, dtype=body.dtype)[:, :T]
        contact_target_xyz = batch["contact_target_xyz"].to(device=device, dtype=body.dtype)[:, :T]
        loss, metrics = _target_trajectory_loss_canonical(
            body_canonical=body,
            obj_com_canonical=obj_com,
            obj_rot6d_canonical=obj_rot6d,
            contact_state=contact_state,
            contact_target_xyz=contact_target_xyz,
            frame_mask=frame_mask,
            position_weight=target_position_weight,
            velocity_weight=target_velocity_weight,
            metric_loss=metric_loss,
            metric_weight=metric_weight,
            moving_frame_extra_weight=moving_frame_extra_weight,
            contact_threshold=contact_threshold,
            use_soft_contact_weights=use_soft_contact_weights,
            velocity_moving_only=velocity_moving_only,
            fps=fps,
            moving_speed_threshold=moving_speed_threshold,
            kin_radius_proxy=kin_radius_proxy,
        )
        metrics.update({
            "decoded_contact_aux_mean_min_dist": metric_loss.detach(),
        })
    metrics.update({
        "decoded_contact_aux_valid_frames": frame_mask.float().sum().detach(),
    })
    metrics.update({k: v.detach() for k, v in extra_metrics.items()})
    return loss, metrics
