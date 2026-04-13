"""Interaction cross-attention layer for injection into MoMask's masked transformer.

This module provides a drop-in cross-attention layer that allows the masked
transformer to attend to interaction tokens (contact, target, phase, support).
It includes:
    1. A temporal alignment conv that downsamples frame-level interaction
       latents to match VQ token-level temporal resolution.
    2. An MLP that projects raw interaction labels into embedding space.
    3. A standard multi-head cross-attention with zero-initialized output
       projection (to preserve pretrained weights at init).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class InteractionTokenizer(nn.Module):
    """Convert raw interaction pseudo-labels into embedding tokens.

    Takes the concatenated interaction labels (contact_state, contact_target,
    phase, support) and projects them into a continuous embedding space,
    then temporally downsamples to match VQ token resolution.
    """

    def __init__(
        self,
        contact_dim: int = 5,         # B body parts
        target_dim: int = 80,         # B * K (5 * 16)
        phase_dim: int = 5,           # P phases (one-hot)
        support_dim: int = 4,         # S support states (one-hot)
        d_model: int = 512,
        temporal_stride: int = 4,     # VQ temporal downsampling factor
    ) -> None:
        super().__init__()
        input_dim = contact_dim + target_dim + phase_dim + support_dim

        self.project = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Learnable temporal downsampling to align with VQ token resolution
        self.temporal_conv = nn.Conv1d(
            d_model, d_model,
            kernel_size=temporal_stride,
            stride=temporal_stride,
            padding=0,
        )

    def forward(
        self,
        contact_state: Tensor,    # (B, T, 5)
        contact_target: Tensor,   # (B, T, 5, K) or (B, T, 80)
        phase: Tensor,            # (B, T, P) one-hot
        support: Tensor,          # (B, T, S) one-hot
    ) -> Tensor:
        """Project and temporally align interaction labels.

        Returns
        -------
        tokens : (B, S, d_model) where S = T // temporal_stride
        """
        # Flatten contact_target if needed
        if contact_target.ndim == 4:
            B, T = contact_target.shape[:2]
            contact_target = contact_target.reshape(B, T, -1)

        # Concatenate all interaction labels
        x = torch.cat([contact_state, contact_target, phase, support], dim=-1)  # (B, T, input_dim)

        # Project to embedding space
        x = self.project(x)  # (B, T, d_model)

        # Temporal downsampling: (B, T, d) -> conv1d -> (B, S, d)
        x = x.permute(0, 2, 1)  # (B, d, T)
        x = self.temporal_conv(x)  # (B, d, S)
        x = x.permute(0, 2, 1)  # (B, S, d)

        return x


class InteractionCrossAttention(nn.Module):
    """Cross-attention layer from motion tokens to interaction tokens.

    Designed to be inserted into each MoMask masked transformer block,
    after the existing text cross-attention and before the feedforward.

    Uses **zero-initialized output projection** so that at initialization,
    this layer is a no-op and the pretrained transformer is unaffected.
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        if zero_init:
            # Zero-init the output projection so this layer starts as identity
            nn.init.zeros_(self.cross_attn.out_proj.weight)
            nn.init.zeros_(self.cross_attn.out_proj.bias)

    def forward(
        self,
        x: Tensor,
        interaction_tokens: Tensor,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Apply interaction cross-attention.

        Parameters
        ----------
        x : (B, S, d_model) — motion token embeddings (query)
        interaction_tokens : (B, S_int, d_model) — interaction tokens (key/value)
        key_padding_mask : (B, S_int) — True for padded positions

        Returns
        -------
        x : (B, S, d_model) — with interaction information added
        """
        residual = x
        x = self.norm(x)
        x, _ = self.cross_attn(
            query=x,
            key=interaction_tokens,
            value=interaction_tokens,
            key_padding_mask=key_padding_mask,
        )
        return residual + x
