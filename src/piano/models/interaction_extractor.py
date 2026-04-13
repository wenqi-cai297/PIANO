"""Lightweight Interaction Extractor for consistency loss.

Given a generated motion sequence, extracts predicted interaction labels.
Used during joint finetuning (Stage C) to enforce that the motion generator
actually respects the interaction latent it was conditioned on:

    z_int → Generator → motion → Extractor → z_int_predicted
    L_consistency = ||z_int - z_int_predicted||

This is a smaller, simpler transformer than the Interaction Predictor —
it only needs to read a motion sequence and classify per-frame labels,
not predict from text/object.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class InteractionExtractor(nn.Module):
    """Extract interaction labels from a motion sequence.

    Parameters
    ----------
    motion_dim : input motion feature dimension (HumanML3D 263-dim)
    d_model : hidden dimension
    num_layers : number of transformer encoder layers
    num_heads : attention heads
    dropout : dropout rate
    num_body_parts : B
    num_object_patches : K
    num_phases : P
    num_support_states : S
    """

    def __init__(
        self,
        motion_dim: int = 263,
        d_model: int = 256,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_seq_length: int = 196,
        num_body_parts: int = 5,
        num_object_patches: int = 16,
        num_phases: int = 5,
        num_support_states: int = 4,
    ) -> None:
        super().__init__()
        self.num_body_parts = num_body_parts
        self.num_object_patches = num_object_patches

        # Input projection
        self.input_proj = nn.Linear(motion_dim, d_model)

        # Positional encoding
        self.pos_encoding = self._sinusoidal_encoding(max_seq_length, d_model)

        # Transformer encoder (reads motion, outputs per-frame features)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output heads (same structure as predictor)
        self.contact_head = nn.Linear(d_model, num_body_parts)
        self.target_head = nn.Linear(d_model, num_body_parts * num_object_patches)
        self.phase_head = nn.Linear(d_model, num_phases)
        self.support_head = nn.Linear(d_model, num_support_states)

    def forward(self, motion: Tensor) -> dict[str, Tensor]:
        """Extract interaction labels from motion.

        Parameters
        ----------
        motion : (B, T, 263) — HumanML3D motion features

        Returns
        -------
        Same structure as InteractionPredictor output (logits + probabilities).
        """
        B, T, _ = motion.shape
        device = motion.device

        # Project and add positional encoding
        x = self.input_proj(motion)  # (B, T, d_model)
        pos_enc = self.pos_encoding[:T, :].unsqueeze(0).to(device)
        x = x + pos_enc

        # Encode
        x = self.encoder(x)  # (B, T, d_model)

        # Output heads
        contact_logits = self.contact_head(x)
        target_logits = self.target_head(x).reshape(
            B, T, self.num_body_parts, self.num_object_patches,
        )
        phase_logits = self.phase_head(x)
        support_logits = self.support_head(x)

        return {
            "contact_state": torch.sigmoid(contact_logits),
            "contact_target": torch.softmax(target_logits, dim=-1),
            "phase": torch.softmax(phase_logits, dim=-1),
            "support": torch.softmax(support_logits, dim=-1),
            "contact_logits": contact_logits,
            "target_logits": target_logits,
            "phase_logits": phase_logits,
            "support_logits": support_logits,
        }

    @staticmethod
    def _sinusoidal_encoding(length: int, d_model: int) -> Tensor:
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
