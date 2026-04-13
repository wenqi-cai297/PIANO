"""Interaction Predictor: predicts structured interaction latents from text + object + pose.

A Temporal Transformer Decoder that takes text, object geometry, and initial
body pose as input, and outputs per-frame interaction labels:
    - contact_state  (T, B)    — which body parts contact the object
    - contact_target (T, B, K) — which object surface patch is contacted
    - phase          (T, P)    — interaction phase (approach/pre-contact/...)
    - support        (T, S)    — body support configuration

The architecture uses:
    - Learnable time tokens with positional encoding (query)
    - Cross-attention to object tokens from PointNet++ (geometry awareness)
    - AdaLN conditioning on text embedding (semantic guidance)
    - Initial pose injection on the first time token
    - Block Attention Residuals (Block AttnRes) from MoonshotAI/Attention-Residuals
      for selective depth-wise information aggregation
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


# ============================================================================
# RMSNorm (used by AttnRes for key normalization)
# ============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, d_model: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


# ============================================================================
# Block Attention Residuals — faithful implementation from
# https://github.com/MoonshotAI/Attention-Residuals
# ============================================================================

def block_attn_res(
    blocks: list[Tensor],
    partial_block: Tensor,
    proj: nn.Linear,
    norm: RMSNorm,
) -> Tensor:
    """Compute Block Attention Residuals.

    Attends over completed block representations + current partial block
    using a learned pseudo-query vector, with softmax over the depth axis.

    Follows the official implementation exactly:
        V = stack(blocks + [partial_block])      # [N+1, B, T, D]
        K = norm(V)                               # RMSNorm over D
        logits = einsum('d, n b t d -> n b t', proj.weight, K)
        h = einsum('n b t, n b t d -> b t d', softmax(logits, dim=0), V)

    Parameters
    ----------
    blocks : list of (B, T, D) tensors — completed block representations
    partial_block : (B, T, D) — current intra-block partial sum
    proj : nn.Linear — pseudo-query, only .weight (shape [1, D]) is used
    norm : RMSNorm — applied to keys before attention

    Returns
    -------
    h : (B, T, D) — selectively aggregated representation
    """
    V = torch.stack(blocks + [partial_block])  # (N+1, B, T, D)
    K = norm(V)
    logits = torch.einsum("d, n b t d -> n b t", proj.weight.squeeze(), K)
    h = torch.einsum("n b t, n b t d -> b t d", logits.softmax(0), V)
    return h


# ============================================================================
# AdaLN (Adaptive Layer Norm) — conditions norm on a global vector
# ============================================================================

class AdaLN(nn.Module):
    """Adaptive Layer Normalization conditioned on a global vector."""

    def __init__(self, d_model: int, cond_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.scale_shift = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * d_model),
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """x: (B, T, d), cond: (B, d_cond)."""
        scale_shift = self.scale_shift(cond).unsqueeze(1)  # (B, 1, 2*d)
        scale, shift = scale_shift.chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


# ============================================================================
# Transformer Block with object cross-attention, AdaLN, and Block AttnRes
# ============================================================================

class PredictorBlock(nn.Module):
    """Single Transformer layer of the Interaction Predictor.

    Each layer has 3 sublayers: self-attention, object cross-attention, FFN.
    Each sublayer has its own Block AttnRes components (proj + norm).

    The block boundary logic (when to seal a block and start a new one)
    is handled by the parent ``InteractionPredictor``, not by this module.
    """

    def __init__(
        self,
        d_model: int = 384,
        num_heads: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        text_dim: int = 512,
    ) -> None:
        super().__init__()

        # --- Self-attention sublayer ---
        self.adaln_sa = AdaLN(d_model, text_dim)
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True,
        )
        # AttnRes for self-attention sublayer
        self.sa_attn_res_proj = nn.Linear(d_model, 1, bias=False)
        self.sa_attn_res_norm = RMSNorm(d_model)

        # --- Cross-attention sublayer (to object tokens) ---
        self.adaln_ca = AdaLN(d_model, text_dim)
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True,
        )
        # AttnRes for cross-attention sublayer
        self.ca_attn_res_proj = nn.Linear(d_model, 1, bias=False)
        self.ca_attn_res_norm = RMSNorm(d_model)

        # --- FFN sublayer ---
        self.adaln_ff = AdaLN(d_model, text_dim)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        # AttnRes for FFN sublayer
        self.ff_attn_res_proj = nn.Linear(d_model, 1, bias=False)
        self.ff_attn_res_norm = RMSNorm(d_model)

    def forward(
        self,
        blocks: list[Tensor],
        partial_block: Tensor,
        object_tokens: Tensor,
        text_emb: Tensor,
        is_block_boundary: bool = False,
    ) -> tuple[list[Tensor], Tensor]:
        """Forward pass with Block AttnRes.

        Parameters
        ----------
        blocks : list of (B, T, d) — sealed block representations
        partial_block : (B, T, d) — current intra-block partial sum
        object_tokens : (B, M, d) — object feature tokens
        text_emb : (B, d_text) — global text embedding
        is_block_boundary : if True, seal partial_block before this layer

        Returns
        -------
        blocks : updated list (may have new entry if boundary)
        partial_block : updated intra-block partial sum
        """
        # --- Block boundary: seal completed block ---
        if is_block_boundary and partial_block is not None:
            blocks = blocks + [partial_block]
            partial_block = None

        # --- Self-attention sublayer ---
        # When partial_block is None (start of a new block), use the last
        # sealed block as the AttnRes target. The sublayer output then
        # becomes the new partial_block (no additive residual from prior).
        if partial_block is not None:
            h = block_attn_res(blocks, partial_block,
                               self.sa_attn_res_proj, self.sa_attn_res_norm)
        else:
            h = block_attn_res(blocks[:-1], blocks[-1],
                               self.sa_attn_res_proj, self.sa_attn_res_norm) if len(blocks) > 1 else blocks[-1]
        h_norm = self.adaln_sa(h, text_emb)
        sa_out, _ = self.self_attn(h_norm, h_norm, h_norm)
        partial_block = partial_block + sa_out if partial_block is not None else sa_out

        # --- Cross-attention sublayer ---
        h = block_attn_res(blocks, partial_block,
                           self.ca_attn_res_proj, self.ca_attn_res_norm)
        h_norm = self.adaln_ca(h, text_emb)
        ca_out, _ = self.cross_attn(h_norm, object_tokens, object_tokens)
        partial_block = partial_block + ca_out

        # --- FFN sublayer ---
        h = block_attn_res(blocks, partial_block,
                           self.ff_attn_res_proj, self.ff_attn_res_norm)
        h_norm = self.adaln_ff(h, text_emb)
        ff_out = self.ffn(h_norm)
        partial_block = partial_block + ff_out

        return blocks, partial_block


# ============================================================================
# Full Interaction Predictor
# ============================================================================

class InteractionPredictor(nn.Module):
    """Predicts structured interaction latents from text + object + initial pose.

    Scaled architecture with Block AttnRes:
        - 10 layers, d_model=384, heads=6, ffn=1024 (~30M params)
        - Block AttnRes with block_size=2 (5 blocks for 10 layers)
        - Deeper but narrower than the original 6-layer d=512 design

    Parameters
    ----------
    d_model : hidden dimension (384 to match MoMask latent_dim)
    num_layers : number of transformer blocks (10 for deeper temporal/geometric reasoning)
    num_heads : attention heads per block
    dim_feedforward : FFN hidden dimension
    dropout : dropout rate
    text_dim : CLIP text embedding dimension (512 for ViT-B/32)
    pose_dim : initial pose feature dimension (HumanML3D 263-dim)
    max_seq_length : maximum number of output frames
    block_size : number of layers per AttnRes block
    num_body_parts : B — number of tracked body parts
    num_object_patches : K — number of object surface patches
    num_phases : P — number of interaction phases
    num_support_states : S — number of support states
    """

    def __init__(
        self,
        d_model: int = 384,
        num_layers: int = 10,
        num_heads: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        text_dim: int = 512,
        pose_dim: int = 263,
        max_seq_length: int = 196,
        block_size: int = 2,
        num_body_parts: int = 5,
        num_object_patches: int = 16,
        num_phases: int = 5,
        num_support_states: int = 4,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.num_body_parts = num_body_parts
        self.num_object_patches = num_object_patches
        self.num_layers = num_layers
        self.block_size = block_size

        # Learnable time tokens
        self.time_tokens = nn.Parameter(torch.randn(1, max_seq_length, d_model) * 0.02)
        self.pos_encoding = self._sinusoidal_encoding(max_seq_length, d_model)

        # Initial pose projection
        self.pose_proj = nn.Linear(pose_dim, d_model)

        # Transformer layers
        self.layers = nn.ModuleList([
            PredictorBlock(d_model, num_heads, dim_feedforward, dropout, text_dim)
            for _ in range(num_layers)
        ])

        # Output heads
        self.contact_head = nn.Linear(d_model, num_body_parts)
        self.target_head = nn.Linear(d_model, num_body_parts * num_object_patches)
        self.phase_head = nn.Linear(d_model, num_phases)
        self.support_head = nn.Linear(d_model, num_support_states)

    def forward(
        self,
        text_emb: Tensor,
        object_tokens: Tensor,
        init_pose: Tensor,
        seq_length: int | None = None,
    ) -> dict[str, Tensor]:
        """Predict interaction latents.

        Parameters
        ----------
        text_emb : (B, text_dim) — CLIP text embedding
        object_tokens : (B, M, d_model) — from ObjectEncoder
        init_pose : (B, pose_dim) — initial pose features
        seq_length : output length (defaults to max_seq_length)

        Returns
        -------
        Dictionary with keys:
            contact_state : (B, T, num_body_parts) — sigmoid probabilities
            contact_target : (B, T, num_body_parts, num_object_patches) — softmax
            phase : (B, T, num_phases) — softmax
            support : (B, T, num_support_states) — softmax
        """
        B = text_emb.shape[0]
        T = seq_length or self.max_seq_length
        device = text_emb.device

        # Initialize time tokens with positional encoding
        x = self.time_tokens[:, :T, :].expand(B, -1, -1).clone()  # (B, T, d)
        pos_enc = self.pos_encoding[:T, :].unsqueeze(0).to(device)
        x = x + pos_enc

        # Inject initial pose into first token
        pose_emb = self.pose_proj(init_pose)  # (B, d)
        x[:, 0, :] = x[:, 0, :] + pose_emb

        # --- Block AttnRes forward ---
        # blocks[0] = token embedding (before any transformer layer)
        # partial_block starts as None — the first sublayer output begins a new block
        blocks: list[Tensor] = [x]
        partial_block: Tensor | None = None

        for layer_idx, layer in enumerate(self.layers):
            # Block boundary: every block_size layers (except layer 0)
            # At boundary, seal partial_block into blocks and reset
            is_boundary = (layer_idx > 0) and (layer_idx % self.block_size == 0)

            blocks, partial_block = layer(
                blocks, partial_block, object_tokens, text_emb,
                is_block_boundary=is_boundary,
            )

        # Final output is the last partial_block
        x = partial_block

        # Output heads
        contact_logits = self.contact_head(x)           # (B, T, B_parts)
        target_logits = self.target_head(x)             # (B, T, B_parts * K)
        phase_logits = self.phase_head(x)               # (B, T, P)
        support_logits = self.support_head(x)           # (B, T, S)

        # Reshape target: (B, T, B_parts*K) -> (B, T, B_parts, K)
        target_logits = target_logits.reshape(
            B, T, self.num_body_parts, self.num_object_patches,
        )

        return {
            "contact_state": torch.sigmoid(contact_logits),
            "contact_target": torch.softmax(target_logits, dim=-1),
            "phase": torch.softmax(phase_logits, dim=-1),
            "support": torch.softmax(support_logits, dim=-1),
            # Raw logits for loss computation
            "contact_logits": contact_logits,
            "target_logits": target_logits,
            "phase_logits": phase_logits,
            "support_logits": support_logits,
        }

    @staticmethod
    def _sinusoidal_encoding(length: int, d_model: int) -> Tensor:
        """Generate sinusoidal positional encoding (not learnable)."""
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
