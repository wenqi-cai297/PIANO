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
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# AdaLN (Adaptive Layer Norm) — conditions norm on a global vector
# ---------------------------------------------------------------------------

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
        """Apply adaptive layer norm.

        Parameters
        ----------
        x : (B, T, d) — sequence to normalize
        cond : (B, d_cond) — conditioning vector (e.g., text embedding)
        """
        scale_shift = self.scale_shift(cond).unsqueeze(1)  # (B, 1, 2*d)
        scale, shift = scale_shift.chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


# ---------------------------------------------------------------------------
# Transformer Decoder Block with object cross-attention and AdaLN
# ---------------------------------------------------------------------------

class PredictorBlock(nn.Module):
    """Single block of the Interaction Predictor.

    Order: AdaLN → Self-Attention → AdaLN → Object Cross-Attention → FFN
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        text_dim: int = 768,
    ) -> None:
        super().__init__()

        # Self-attention
        self.adaln_sa = AdaLN(d_model, text_dim)
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True,
        )

        # Cross-attention to object tokens
        self.adaln_ca = AdaLN(d_model, text_dim)
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True,
        )

        # Feedforward
        self.adaln_ff = AdaLN(d_model, text_dim)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: Tensor,
        object_tokens: Tensor,
        text_emb: Tensor,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : (B, T, d) — time tokens
        object_tokens : (B, M, d) — object feature tokens
        text_emb : (B, d_text) — global text embedding
        attn_mask : optional causal mask for self-attention
        """
        # Self-attention with AdaLN
        residual = x
        x = self.adaln_sa(x, text_emb)
        x, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = residual + x

        # Cross-attention to object tokens
        residual = x
        x = self.adaln_ca(x, text_emb)
        x, _ = self.cross_attn(x, object_tokens, object_tokens)
        x = residual + x

        # Feedforward
        residual = x
        x = self.adaln_ff(x, text_emb)
        x = self.ffn(x)
        x = residual + x

        return x


# ---------------------------------------------------------------------------
# Full Interaction Predictor
# ---------------------------------------------------------------------------

class InteractionPredictor(nn.Module):
    """Predicts structured interaction latents from text + object + initial pose.

    Parameters
    ----------
    d_model : hidden dimension
    num_layers : number of transformer blocks
    num_heads : attention heads per block
    dim_feedforward : FFN hidden dimension
    dropout : dropout rate
    text_dim : CLIP text embedding dimension
    pose_dim : initial pose feature dimension (HumanML3D 263-dim)
    max_seq_length : maximum number of output frames
    num_body_parts : B — number of tracked body parts
    num_object_patches : K — number of object surface patches
    num_phases : P — number of interaction phases
    num_support_states : S — number of support states
    """

    def __init__(
        self,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        text_dim: int = 768,
        pose_dim: int = 263,
        max_seq_length: int = 196,
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

        # Learnable time tokens
        self.time_tokens = nn.Parameter(torch.randn(1, max_seq_length, d_model) * 0.02)
        self.pos_encoding = self._sinusoidal_encoding(max_seq_length, d_model)

        # Initial pose projection
        self.pose_proj = nn.Linear(pose_dim, d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
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
        x = self.time_tokens[:, :T, :].expand(B, -1, -1)  # (B, T, d)
        pos_enc = self.pos_encoding[:T, :].unsqueeze(0).to(device)
        x = x + pos_enc

        # Inject initial pose into first token
        pose_emb = self.pose_proj(init_pose)  # (B, d)
        x[:, 0, :] = x[:, 0, :] + pose_emb

        # Transformer blocks
        for block in self.blocks:
            x = block(x, object_tokens, text_emb)

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
            # Also return raw logits for loss computation
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
