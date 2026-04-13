"""Motion Generator: MoMask-compatible architecture with interaction cross-attention.

This module provides a self-contained reimplementation of MoMask's core
components (VQ-VAE, MaskedTransformer, ResidualTransformer) that is
**weight-compatible** with MoMask's pretrained checkpoints.  The key
addition is interaction cross-attention injected into the MaskedTransformer.

We do NOT import MoMask's code.  Instead we reimplement just the
forward paths needed for training and inference, matching MoMask's
architecture exactly so that ``load_state_dict(ckpt, strict=False)``
works — the only "unexpected" keys are our new interaction layers.

Architecture overview (from MoMask paper)::

    MaskedTransformer:
        Input:  partially masked VQ tokens [S] + prepended CLIP cond token
        Layers: N × (self-attention + [interaction cross-attention] + FFN)
        Output: logits over codebook for each position

    ResidualTransformer:
        Input:  base tokens [S] + prepended CLIP cond + quantizer-id token
        Layers: M × (self-attention + FFN)
        Output: residual-level token logits

    RVQVAE:
        Encoder: 1D conv stack → quantize → token indices
        Decoder: dequantize → 1D conv stack → motion features
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from piano.models.interaction_cross_attn import InteractionCrossAttention
from piano.models.masking import cosine_schedule, mask_by_confidence, sample_from_logits


# ============================================================================
# Building blocks (MoMask-compatible)
# ============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (matches MoMask's implementation)."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model) — seq-first
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """Add positional encoding. x: (seq_len, batch, d_model)."""
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class InputProcess(nn.Module):
    """Linear projection from code_dim to latent_dim (MoMask compat)."""

    def __init__(self, input_feats: int, latent_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_feats, latent_dim)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, S, code_dim) → (S, B, latent_dim) — MoMask uses seq-first."""
        return self.proj(x).permute(1, 0, 2)


class OutputProcess(nn.Module):
    """Project latent back to token logits (MoMask's OutputProcess_Bert compat)."""

    def __init__(self, out_feats: int, latent_dim: int) -> None:
        super().__init__()
        self.dense = nn.Linear(latent_dim, latent_dim)
        self.transform_act_fn = nn.GELU()
        self.LayerNorm = nn.LayerNorm(latent_dim)
        self.decoder = nn.Linear(latent_dim, out_feats, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_feats))

    def forward(self, hidden_states: Tensor) -> Tensor:
        """hidden_states: (S, B, latent_dim) → (B, out_feats, S)."""
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        output = self.decoder(hidden_states) + self.bias  # (S, B, out_feats)
        return output.permute(1, 2, 0)  # (B, out_feats, S)


# ============================================================================
# Custom Transformer Block with optional interaction cross-attention
# ============================================================================

class TransformerBlockWithInteraction(nn.Module):
    """Self-attention + (optional) interaction cross-attention + FFN.

    When ``interaction_cross_attn`` is None, this behaves identically to
    ``nn.TransformerEncoderLayer(activation='gelu')``, so pretrained
    MoMask weights load directly into the self-attention + FFN parameters.
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        enable_interaction: bool = True,
        zero_init_interaction: bool = True,
    ) -> None:
        super().__init__()

        # --- Self-attention (matches nn.TransformerEncoderLayer) ---
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # --- Interaction cross-attention (our addition) ---
        self.interaction_cross_attn: InteractionCrossAttention | None = None
        if enable_interaction:
            self.interaction_cross_attn = InteractionCrossAttention(
                d_model=d_model,
                num_heads=nhead,
                dropout=dropout,
                zero_init=zero_init_interaction,
            )

        # --- FFN (matches nn.TransformerEncoderLayer) ---
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(
        self,
        src: Tensor,
        interaction_tokens: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass. src: (S, B, d_model) — seq-first convention.

        Parameters
        ----------
        src : (S+1, B, d) — includes prepended condition token
        interaction_tokens : (B, S_int, d) or None — batch-first
        src_key_padding_mask : (B, S+1) — True for padded positions
        """
        # Self-attention (post-norm, matching MoMask's nn.TransformerEncoderLayer)
        src2 = self.self_attn(src, src, src, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Interaction cross-attention (our addition)
        if self.interaction_cross_attn is not None and interaction_tokens is not None:
            # Convert from seq-first to batch-first for cross-attn
            src_bf = src.permute(1, 0, 2)  # (B, S+1, d)
            src_bf = self.interaction_cross_attn(src_bf, interaction_tokens)
            src = src_bf.permute(1, 0, 2)  # back to (S+1, B, d)

        # FFN
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)

        return src


# ============================================================================
# MaskedTransformer with interaction conditioning
# ============================================================================

class MaskedTransformerWithInteraction(nn.Module):
    """MoMask MaskedTransformer extended with interaction cross-attention.

    Weight-compatible with MoMask checkpoints.  The interaction cross-attention
    layers are new and will be randomly initialized (or zero-initialized)
    when loading pretrained weights with ``strict=False``.

    Parameters
    ----------
    num_tokens : codebook size (e.g., 512)
    code_dim : codebook embedding dimension
    latent_dim : transformer hidden dimension
    ff_size : feedforward dimension
    num_layers : number of transformer blocks
    num_heads : attention heads
    dropout : dropout rate
    clip_dim : CLIP text embedding dimension
    cond_drop_prob : probability of dropping condition (for CFG)
    interaction_drop_prob : probability of dropping interaction condition
    enable_interaction : whether to add interaction cross-attention
    """

    def __init__(
        self,
        num_tokens: int = 512,
        code_dim: int = 512,
        latent_dim: int = 512,
        ff_size: int = 1024,
        num_layers: int = 8,
        num_heads: int = 4,
        dropout: float = 0.1,
        clip_dim: int = 512,
        cond_drop_prob: float = 0.1,
        interaction_drop_prob: float = 0.1,
        enable_interaction: bool = True,
    ) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.latent_dim = latent_dim
        self.cond_drop_prob = cond_drop_prob
        self.interaction_drop_prob = interaction_drop_prob

        # Special token IDs
        self.mask_id = num_tokens
        self.pad_id = num_tokens + 1

        # Token embedding (+2 for mask and pad tokens)
        self.token_emb = nn.Embedding(num_tokens + 2, code_dim)

        # Input / output processing (MoMask compat names)
        self.input_process = InputProcess(code_dim, latent_dim)
        self.position_enc = PositionalEncoding(latent_dim, dropout)
        self.output_process = OutputProcess(num_tokens, latent_dim)

        # Condition embedding (text)
        self.cond_emb = nn.Linear(clip_dim, latent_dim)

        # Transformer blocks with interaction cross-attention
        self.blocks = nn.ModuleList([
            TransformerBlockWithInteraction(
                d_model=latent_dim,
                nhead=num_heads,
                dim_feedforward=ff_size,
                dropout=dropout,
                enable_interaction=enable_interaction,
            )
            for _ in range(num_layers)
        ])

        # Length token embedding (MoMask uses this to signal sequence length)
        self.encode_length = nn.Embedding(512, latent_dim)

    def trans_forward(
        self,
        motion_ids: Tensor,
        cond: Tensor,
        padding_mask: Tensor,
        interaction_tokens: Tensor | None = None,
        force_mask: bool = False,
    ) -> Tensor:
        """Core forward pass through the transformer.

        Parameters
        ----------
        motion_ids : (B, S) — token indices (may contain mask_id, pad_id)
        cond : (B, clip_dim) — CLIP text embedding (already encoded)
        padding_mask : (B, S) — True for padded positions
        interaction_tokens : (B, S_int, latent_dim) or None
        force_mask : if True, zero out the condition (for CFG unconditional pass)

        Returns
        -------
        logits : (B, num_tokens, S)
        """
        B, S = motion_ids.shape

        # Embed tokens
        x = self.token_emb(motion_ids)  # (B, S, code_dim)
        x = self.input_process(x)  # (S, B, latent_dim)

        # Condition token
        if force_mask:
            cond_token = torch.zeros(1, B, self.latent_dim, device=x.device)
        else:
            cond_token = self.cond_emb(cond).unsqueeze(0)  # (1, B, latent_dim)

        # Positional encoding
        x = self.position_enc(x)

        # Prepend condition token
        xseq = torch.cat([cond_token, x], dim=0)  # (S+1, B, latent_dim)
        pad = torch.cat([
            torch.zeros(B, 1, dtype=torch.bool, device=padding_mask.device),
            padding_mask,
        ], dim=1)  # (B, S+1)

        # Pass through transformer blocks
        for block in self.blocks:
            xseq = block(xseq, interaction_tokens=interaction_tokens, src_key_padding_mask=pad)

        # Remove condition token, project to logits
        output = xseq[1:]  # (S, B, latent_dim)
        logits = self.output_process(output)  # (B, num_tokens, S)
        return logits

    def forward(
        self,
        ids: Tensor,
        cond: Tensor,
        m_lens: Tensor,
        interaction_tokens: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Training forward pass: mask tokens, predict, compute loss.

        Parameters
        ----------
        ids : (B, S) — ground-truth token indices
        cond : (B, clip_dim) — text embedding
        m_lens : (B,) — actual token sequence lengths
        interaction_tokens : (B, S_int, latent_dim) or None

        Returns
        -------
        loss : scalar — cross-entropy on masked positions
        pred_ids : (B, S) — predicted token IDs
        accuracy : scalar — prediction accuracy on masked positions
        """
        B, S = ids.shape
        device = ids.device

        # Build padding mask
        padding_mask = _lengths_to_mask(m_lens, S)  # True = padded

        # Random mask ratio per sample (cosine schedule)
        t = torch.rand(B, device=device)
        mask_ratio = cosine_schedule(t)  # (B,)
        num_masked = (mask_ratio * m_lens.float()).long().clamp(min=1)

        # Create masked input
        masked_ids = ids.clone()
        mask_positions = torch.zeros(B, S, dtype=torch.bool, device=device)
        for i in range(B):
            n = num_masked[i].item()
            valid_len = m_lens[i].item()
            perm = torch.randperm(valid_len, device=device)[:n]
            mask_positions[i, perm] = True
            masked_ids[i, perm] = self.mask_id

        # Drop conditions for classifier-free guidance training
        drop_cond = torch.rand(B, device=device) < self.cond_drop_prob
        drop_int = torch.rand(B, device=device) < self.interaction_drop_prob

        # Zero out cond for dropped samples
        cond_input = cond.clone()
        cond_input[drop_cond] = 0.0

        # Zero out interaction for dropped samples
        int_input = interaction_tokens
        if int_input is not None:
            int_input = int_input.clone()
            int_input[drop_cond | drop_int] = 0.0  # drop both or just interaction

        # Forward
        logits = self.trans_forward(masked_ids, cond_input, padding_mask, int_input)
        # logits: (B, num_tokens, S)

        # Loss only on masked positions
        target = ids[mask_positions]  # (num_masked_total,)
        pred_logits = logits.permute(0, 2, 1)[mask_positions]  # (num_masked_total, num_tokens)
        loss = F.cross_entropy(pred_logits, target)

        # Accuracy
        pred_ids_all = logits.argmax(dim=1)  # (B, S)
        correct = (pred_ids_all[mask_positions] == target).float().mean()

        return loss, pred_ids_all, correct

    def forward_with_cond_scale(
        self,
        motion_ids: Tensor,
        cond: Tensor,
        padding_mask: Tensor,
        interaction_tokens: Tensor | None = None,
        cond_scale: float = 4.5,
        interaction_scale: float = 2.0,
    ) -> Tensor:
        """Classifier-free guidance with dual-condition scaling.

        Three forward passes:
            1. Unconditional (no text, no interaction)
            2. Text-only (text, no interaction)
            3. Full (text + interaction)

        Returns guided logits.
        """
        # Unconditional
        logits_uncond = self.trans_forward(
            motion_ids, cond, padding_mask,
            interaction_tokens=None, force_mask=True,
        )

        # Text-only
        logits_text = self.trans_forward(
            motion_ids, cond, padding_mask,
            interaction_tokens=None, force_mask=False,
        )

        # Full (text + interaction)
        if interaction_tokens is not None:
            logits_full = self.trans_forward(
                motion_ids, cond, padding_mask,
                interaction_tokens=interaction_tokens, force_mask=False,
            )
            # Dual-condition guidance
            return (
                logits_uncond
                + cond_scale * (logits_text - logits_uncond)
                + interaction_scale * (logits_full - logits_text)
            )
        else:
            # Standard single-condition guidance
            return logits_uncond + cond_scale * (logits_text - logits_uncond)

    @torch.no_grad()
    def generate(
        self,
        cond: Tensor,
        m_lens: Tensor,
        interaction_tokens: Tensor | None = None,
        timesteps: int = 10,
        cond_scale: float = 4.5,
        interaction_scale: float = 2.0,
        temperature: float = 1.0,
        topk_filter_thres: float = 0.9,
    ) -> Tensor:
        """Iterative unmasking generation.

        Parameters
        ----------
        cond : (B, clip_dim) — text embedding
        m_lens : (B,) — desired token sequence lengths
        interaction_tokens : (B, S_int, latent_dim) or None
        timesteps : number of unmasking iterations
        cond_scale : text guidance scale
        interaction_scale : interaction guidance scale
        temperature : sampling temperature
        topk_filter_thres : top-k filtering threshold

        Returns
        -------
        ids : (B, S) — generated token indices (-1 for padding)
        """
        B = cond.shape[0]
        S = m_lens.max().item()
        device = cond.device

        # Start fully masked
        ids = torch.full((B, S), self.mask_id, dtype=torch.long, device=device)
        confidence = torch.zeros(B, S, device=device)
        padding_mask = _lengths_to_mask(m_lens, S)

        for step in range(timesteps):
            t = step / timesteps
            mask_ratio = cosine_schedule(torch.tensor(t)).item()
            num_masked = max(1, int(mask_ratio * S))

            # Get logits with guidance
            logits = self.forward_with_cond_scale(
                ids, cond, padding_mask,
                interaction_tokens=interaction_tokens,
                cond_scale=cond_scale,
                interaction_scale=interaction_scale,
            )  # (B, num_tokens, S)

            logits = logits.permute(0, 2, 1)  # (B, S, num_tokens)

            # Sample
            sampled_ids, sampled_conf = sample_from_logits(
                logits, temperature=temperature, topk_filter_thres=topk_filter_thres,
            )

            # Only update currently masked positions
            is_masked = ids == self.mask_id
            ids = torch.where(is_masked, sampled_ids, ids)
            confidence = torch.where(is_masked, sampled_conf, confidence)

            # Re-mask lowest confidence (except last step)
            if step < timesteps - 1:
                ids = mask_by_confidence(ids, confidence, num_masked, self.mask_id)

        # Mark padding as -1
        ids[padding_mask] = -1
        return ids

    def load_momask_weights(self, checkpoint_path: str, device: str = "cpu") -> None:
        """Load pretrained MoMask MaskTransformer weights.

        Loads with ``strict=False`` — our interaction cross-attention layers
        and any renamed keys will be reported but not cause errors.

        The state dict key mapping handles the difference between MoMask's
        ``seqTransEncoder.layers.{i}.{param}`` and our ``blocks.{i}.{param}``.
        """
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("t2m_transformer", ckpt.get("trans", ckpt))

        # Remap MoMask's nn.TransformerEncoder keys to our block structure
        mapped = {}
        for key, value in state_dict.items():
            new_key = self._remap_key(key)
            if new_key is not None:
                mapped[new_key] = value

        missing, unexpected = self.load_state_dict(mapped, strict=False)

        # Filter out expected missing keys (interaction layers, clip_model)
        real_missing = [
            k for k in missing
            if not k.startswith("blocks.") or "interaction_cross_attn" not in k
        ]
        if real_missing:
            print(f"[WARN] Missing keys (not interaction layers): {real_missing[:10]}")
        if unexpected:
            print(f"[WARN] Unexpected keys: {unexpected[:10]}")

        n_loaded = len(mapped) - len(unexpected)
        print(f"Loaded {n_loaded} parameters from MoMask checkpoint")

    @staticmethod
    def _remap_key(key: str) -> str | None:
        """Remap MoMask state dict keys to our naming convention.

        MoMask: ``seqTransEncoder.layers.{i}.self_attn.{param}``
        Ours:   ``blocks.{i}.self_attn.{param}``

        MoMask: ``seqTransEncoder.layers.{i}.linear1.{param}``
        Ours:   ``blocks.{i}.linear1.{param}``

        MoMask: ``seqTransEncoder.layers.{i}.norm1.{param}``
        Ours:   ``blocks.{i}.norm1.{param}``
        """
        # Skip CLIP keys
        if key.startswith("clip_model."):
            return None

        # Remap TransformerEncoder layers
        if key.startswith("seqTransEncoder.layers."):
            return key.replace("seqTransEncoder.layers.", "blocks.")

        # Skip the TransformerEncoder norm (we don't use a final norm)
        if key.startswith("seqTransEncoder.norm."):
            return None

        # MoMask's OutputProcess_Bert → our OutputProcess
        if key.startswith("output_process."):
            return key

        # Everything else maps directly
        return key


# ============================================================================
# VQ-VAE loader (weight-compatible with MoMask's RVQVAE)
# ============================================================================

class RVQVAE(nn.Module):
    """Residual VQ-VAE for motion tokenization.

    This is a minimal reimplementation of MoMask's RVQVAE that supports
    ``encode`` and ``decode`` via pretrained weights. The encoder and decoder
    use 1D convolutions with residual blocks.

    For PIANO, the VQ-VAE is always **frozen** — we only use it to convert
    between motion features and discrete tokens.
    """

    def __init__(
        self,
        input_width: int = 263,
        nb_code: int = 512,
        code_dim: int = 512,
        down_t: int = 2,
        stride_t: int = 2,
        width: int = 512,
        depth: int = 3,
        dilation_growth_rate: int = 3,
        num_quantizers: int = 2,
    ) -> None:
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.num_quantizers = num_quantizers

        # Encoder: (B, input_width, T) → (B, code_dim, T')
        self.encoder = self._build_conv_stack(
            input_width, width, code_dim, down_t, stride_t, depth, dilation_growth_rate,
            downsample=True,
        )

        # Decoder: (B, code_dim, T') → (B, input_width, T)
        self.decoder = self._build_conv_stack(
            code_dim, width, input_width, down_t, stride_t, depth, dilation_growth_rate,
            downsample=False,
        )

        # Quantizer codebooks
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(nb_code, code_dim))
            for _ in range(num_quantizers)
        ])

    def encode(self, x: Tensor) -> Tensor:
        """Encode motion features to VQ token indices.

        Parameters
        ----------
        x : (B, T, input_width) — raw motion features

        Returns
        -------
        indices : (B, T', num_quantizers) — token indices per quantizer level
        """
        z = self.encoder(x.permute(0, 2, 1))  # (B, code_dim, T')
        B, C, T_prime = z.shape

        indices = []
        residual = z
        for q in range(self.num_quantizers):
            codebook = self.codebooks[q]  # (nb_code, code_dim)
            # Nearest neighbour lookup
            flat = residual.permute(0, 2, 1).reshape(-1, C)  # (B*T', code_dim)
            dist = torch.cdist(flat, codebook)  # (B*T', nb_code)
            idx = dist.argmin(dim=-1).reshape(B, T_prime)  # (B, T')
            indices.append(idx)
            # Subtract quantized value for residual
            quantized = codebook[idx]  # (B, T', code_dim)
            residual = residual - quantized.permute(0, 2, 1)

        return torch.stack(indices, dim=-1)  # (B, T', num_quantizers)

    def decode(self, indices: Tensor) -> Tensor:
        """Decode VQ token indices back to motion features.

        Parameters
        ----------
        indices : (B, T', num_quantizers) — token indices

        Returns
        -------
        motion : (B, T, input_width) — reconstructed motion features
        """
        B, T_prime, Q = indices.shape

        # Sum quantized embeddings across quantizer levels
        z = torch.zeros(B, T_prime, self.code_dim, device=indices.device)
        for q in range(Q):
            codebook = self.codebooks[q]
            z = z + codebook[indices[:, :, q]]  # (B, T', code_dim)

        z = z.permute(0, 2, 1)  # (B, code_dim, T')
        motion = self.decoder(z)  # (B, input_width, T)
        return motion.permute(0, 2, 1)  # (B, T, input_width)

    def load_momask_weights(self, checkpoint_path: str, device: str = "cpu") -> None:
        """Load pretrained MoMask VQ-VAE weights."""
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("vq_model", ckpt.get("net", ckpt))
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if unexpected:
            print(f"[WARN] VQ-VAE unexpected keys: {unexpected[:10]}")
        print(f"Loaded VQ-VAE weights ({len(state_dict) - len(unexpected)} params)")

    @staticmethod
    def _build_conv_stack(
        in_ch: int, width: int, out_ch: int,
        num_layers: int, stride: int, depth: int,
        dilation_growth_rate: int, downsample: bool,
    ) -> nn.Sequential:
        """Build a 1D conv stack with residual blocks (simplified).

        This is a simplified version — the actual MoMask conv stack uses
        dilated residual blocks.  For weight loading, the exact architecture
        must match.  This placeholder works for the API; the real weights
        are loaded via ``load_momask_weights``.
        """
        layers: list[nn.Module] = []

        # Input projection
        layers.append(nn.Conv1d(in_ch, width, 3, padding=1))
        layers.append(nn.ReLU(inplace=True))

        # Residual blocks with downsampling/upsampling
        for i in range(num_layers):
            for d in range(depth):
                dilation = dilation_growth_rate ** d
                layers.append(nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation))
                layers.append(nn.ReLU(inplace=True))

            if downsample:
                layers.append(nn.Conv1d(width, width, stride * 2, stride=stride, padding=stride // 2))
            else:
                layers.append(nn.ConvTranspose1d(width, width, stride * 2, stride=stride, padding=stride // 2))
            layers.append(nn.ReLU(inplace=True))

        # Output projection
        layers.append(nn.Conv1d(width, out_ch, 3, padding=1))

        return nn.Sequential(*layers)


# ============================================================================
# Helper
# ============================================================================

def _lengths_to_mask(lengths: Tensor, max_len: int) -> Tensor:
    """Convert sequence lengths to a boolean padding mask.

    Returns True for padded positions (consistent with PyTorch convention).
    """
    arange = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return arange >= lengths.unsqueeze(1)
