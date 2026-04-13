"""Mask scheduling and iterative unmasking utilities for MoMask-style generation.

MoMask uses a cosine schedule to control the fraction of tokens that
remain masked at each iteration during generation. This module provides:
    - ``cosine_schedule``: mask ratio as a function of progress
    - ``mask_by_confidence``: re-mask lowest-confidence tokens
    - ``sample_from_logits``: top-k filtered sampling
"""
from __future__ import annotations

import math

import torch
from torch import Tensor


def cosine_schedule(t: float | Tensor, total_steps: int | None = None) -> Tensor:
    """Cosine mask ratio schedule.

    Parameters
    ----------
    t : progress in [0, 1] (0 = start, fully masked; 1 = end, fully unmasked).
        Can also be a scalar or tensor.
    total_steps : unused, kept for API compat

    Returns
    -------
    mask_ratio : fraction of tokens that should remain masked at progress *t*.
        Goes from 1.0 (fully masked) at t=0 to 0.0 (fully unmasked) at t=1.
    """
    if not isinstance(t, Tensor):
        t = torch.tensor(t)
    return torch.cos(t * math.pi / 2)


def mask_by_confidence(
    token_ids: Tensor,
    confidence: Tensor,
    num_to_mask: int,
    mask_id: int,
) -> Tensor:
    """Re-mask the *num_to_mask* least confident tokens.

    Parameters
    ----------
    token_ids : (B, S) — current token predictions
    confidence : (B, S) — confidence scores per token
    num_to_mask : number of tokens to mask
    mask_id : the mask token index

    Returns
    -------
    masked_ids : (B, S) — tokens with lowest-confidence positions masked
    """
    B, S = token_ids.shape
    if num_to_mask <= 0:
        return token_ids
    if num_to_mask >= S:
        return torch.full_like(token_ids, mask_id)

    # Find positions with lowest confidence
    _, indices = confidence.topk(S - num_to_mask, dim=-1, largest=True)
    # Build mask: True = keep, False = re-mask
    keep_mask = torch.zeros(B, S, dtype=torch.bool, device=token_ids.device)
    keep_mask.scatter_(1, indices, True)

    result = token_ids.clone()
    result[~keep_mask] = mask_id
    return result


def sample_from_logits(
    logits: Tensor,
    temperature: float = 1.0,
    topk_filter_thres: float = 0.9,
) -> tuple[Tensor, Tensor]:
    """Sample token IDs from logits with top-k filtering.

    Parameters
    ----------
    logits : (B, S, V) — per-position logits over vocabulary
    temperature : sampling temperature (lower = more greedy)
    topk_filter_thres : keep only top fraction of vocabulary

    Returns
    -------
    sampled_ids : (B, S) — sampled token indices
    confidence : (B, S) — confidence (probability) of sampled tokens
    """
    B, S, V = logits.shape

    # Temperature scaling
    if temperature != 1.0:
        logits = logits / temperature

    # Top-k filtering
    if topk_filter_thres < 1.0:
        k = max(1, int(V * (1 - topk_filter_thres)))
        val, _ = logits.topk(k, dim=-1)
        threshold = val[..., -1:]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    # Sample
    probs = torch.softmax(logits, dim=-1)
    flat_probs = probs.reshape(-1, V)
    sampled = torch.multinomial(flat_probs, num_samples=1).reshape(B, S)

    # Confidence: probability of the sampled token
    confidence = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)

    return sampled, confidence
