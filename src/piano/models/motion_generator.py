"""Motion Generator: wraps MoMask backbone with interaction cross-attention.

Instead of reimplementing MoMask's architecture, we:
    1. Import MoMask's original ``MaskTransformer`` via ``momask_adapter``
    2. Patch its ``seqTransEncoder`` to inject interaction cross-attention
    3. Override ``trans_forward`` to pass interaction tokens through

This guarantees 100% weight compatibility — we load the exact same model
and only add new parameters (interaction cross-attention layers).

Usage::

    from piano.models.motion_generator import InteractionMaskTransformer

    model = InteractionMaskTransformer.from_pretrained(
        "checkpoints/momask/t2m_.../model/latest.tar"
    )
    # model.mask_transformer is MoMask's original MaskTransformer
    # model.interaction_blocks are the new cross-attention layers
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from piano.models.interaction_cross_attn import InteractionCrossAttention, InteractionTokenizer
from piano.models.masking import cosine_schedule, mask_by_confidence, sample_from_logits


class InteractionTransformerBlock(nn.Module):
    """Wraps a single ``nn.TransformerEncoderLayer`` and adds interaction cross-attention.

    During forward: runs the original encoder layer first, then applies
    interaction cross-attention. Zero-init ensures this is a no-op at start.
    """

    def __init__(
        self,
        original_layer: nn.TransformerEncoderLayer,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.original_layer = original_layer
        self.interaction_attn = InteractionCrossAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            zero_init=True,
        )

    def forward(
        self,
        src: Tensor,
        interaction_tokens: Tensor | None = None,
        src_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Run original layer, then interaction cross-attention.

        Parameters
        ----------
        src : (S, B, d) — seq-first (MoMask convention)
        interaction_tokens : (B, S_int, d) — batch-first, or None
        """
        # Original TransformerEncoderLayer forward
        src = self.original_layer(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        # Interaction cross-attention (if tokens provided)
        if interaction_tokens is not None:
            # Convert seq-first → batch-first for cross-attn
            src_bf = src.permute(1, 0, 2)  # (B, S, d)
            src_bf = self.interaction_attn(src_bf, interaction_tokens)
            src = src_bf.permute(1, 0, 2)  # (S, B, d)

        return src


class InteractionTransformerEncoder(nn.Module):
    """Drop-in replacement for ``nn.TransformerEncoder`` that supports interaction tokens.

    Wraps each original ``TransformerEncoderLayer`` with an
    ``InteractionTransformerBlock``, preserving all original weights.
    """

    def __init__(
        self,
        original_encoder: nn.TransformerEncoder,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            InteractionTransformerBlock(layer, d_model, num_heads, dropout)
            for layer in original_encoder.layers
        ])
        self.norm = original_encoder.norm  # may be None

    def forward(
        self,
        src: Tensor,
        interaction_tokens: Tensor | None = None,
        mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward through all layers with interaction conditioning."""
        output = src
        for layer in self.layers:
            output = layer(output, interaction_tokens, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class InteractionMaskTransformer(nn.Module):
    """MoMask MaskTransformer with interaction cross-attention injection.

    This class:
    1. Holds the original MoMask MaskTransformer (loaded from checkpoint)
    2. Replaces its ``seqTransEncoder`` with ``InteractionTransformerEncoder``
    3. Provides interaction-aware ``trans_forward``, ``forward``, and ``generate``

    The original model's weights are fully preserved. Only the new
    interaction cross-attention layers are added (zero-initialized).
    """

    def __init__(
        self,
        mask_transformer: nn.Module,
        interaction_tokenizer: InteractionTokenizer,
        interaction_drop_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self.mask_transformer = mask_transformer
        self.interaction_tokenizer = interaction_tokenizer
        self.interaction_drop_prob = interaction_drop_prob

        # Patch: replace seqTransEncoder with interaction-aware version
        mt = self.mask_transformer
        d_model = mt.latent_dim
        num_heads = mt.seqTransEncoder.layers[0].self_attn.num_heads
        dropout = mt.dropout

        mt.seqTransEncoder = InteractionTransformerEncoder(
            original_encoder=mt.seqTransEncoder,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

    @classmethod
    def from_pretrained(
        cls,
        transformer_checkpoint: str | Path,
        interaction_tokenizer: InteractionTokenizer | None = None,
        interaction_drop_prob: float = 0.1,
        device: str = "cpu",
        **kwargs,
    ) -> "InteractionMaskTransformer":
        """Load pretrained MoMask MaskTransformer and wrap with interaction layers.

        Parameters
        ----------
        transformer_checkpoint : path to MoMask MaskTransformer checkpoint
        interaction_tokenizer : InteractionTokenizer instance (created if None)
        """
        from piano.models.backbones.momask_adapter import load_momask_mask_transformer

        mask_transformer = load_momask_mask_transformer(
            transformer_checkpoint, device=device, **kwargs,
        )

        if interaction_tokenizer is None:
            interaction_tokenizer = InteractionTokenizer(
                d_model=mask_transformer.latent_dim,
                temporal_stride=4,
            )

        return cls(mask_transformer, interaction_tokenizer, interaction_drop_prob)

    def trans_forward(
        self,
        motion_ids: Tensor,
        cond: Tensor,
        padding_mask: Tensor,
        interaction_tokens: Tensor | None = None,
        force_mask: bool = False,
    ) -> Tensor:
        """Interaction-aware version of MoMask's ``trans_forward``.

        Mirrors the original logic but passes interaction_tokens through
        the patched ``seqTransEncoder``.
        """
        mt = self.mask_transformer

        # Condition masking (CFG)
        cond = mt.mask_cond(cond, force_mask=force_mask)

        # Token embedding → positional encoding
        x = mt.token_emb(motion_ids)
        x = mt.input_process(x)
        cond_emb = mt.cond_emb(cond).unsqueeze(0)  # (1, B, d)
        x = mt.position_enc(x)
        xseq = torch.cat([cond_emb, x], dim=0)  # (S+1, B, d)

        # Padding mask (prepend False for cond token)
        padding_mask = torch.cat([
            torch.zeros_like(padding_mask[:, 0:1]), padding_mask,
        ], dim=1)

        # Forward through patched encoder (with interaction tokens)
        output = mt.seqTransEncoder(
            xseq,
            interaction_tokens=interaction_tokens,
            src_key_padding_mask=padding_mask,
        )[1:]  # drop cond token

        logits = mt.output_process(output)  # (B, num_tokens, S)
        return logits

    def forward(
        self,
        ids: Tensor,
        cond: Tensor,
        m_lens: Tensor,
        interaction_labels: dict[str, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Training forward: mask, predict, compute loss.

        Parameters
        ----------
        ids : (B, S) — ground-truth VQ token indices
        cond : (B, clip_dim) — CLIP text embedding (already encoded)
        m_lens : (B,) — token sequence lengths
        interaction_labels : dict with contact_state, contact_target, phase, support
            If None, trains without interaction conditioning.

        Returns
        -------
        loss, pred_ids, accuracy
        """
        from models.mask_transformer.tools import lengths_to_mask, get_mask_subset_prob, cal_performance, uniform

        mt = self.mask_transformer
        B, S = ids.shape
        device = ids.device

        # Build interaction tokens
        interaction_tokens = None
        if interaction_labels is not None:
            interaction_tokens = self.interaction_tokenizer(
                interaction_labels["contact_state"],
                interaction_labels["contact_target"],
                interaction_labels["phase"],
                interaction_labels["support"],
            )

            # Drop interaction for CFG training
            if self.training:
                drop_mask = torch.rand(B, device=device) < self.interaction_drop_prob
                if drop_mask.any():
                    interaction_tokens = interaction_tokens.clone()
                    interaction_tokens[drop_mask] = 0.0

        # --- Original MoMask masking logic (from MaskTransformer.forward) ---
        non_pad_mask = lengths_to_mask(m_lens, S)
        ids = torch.where(non_pad_mask, ids, mt.pad_id)

        # Random mask
        rand_time = uniform((B,), device=device)
        rand_mask_probs = cosine_schedule(rand_time)
        num_token_masked = (S * rand_mask_probs).round().clamp(min=1)

        batch_randperm = torch.rand((B, S), device=device).argsort(dim=-1)
        mask = batch_randperm < num_token_masked.unsqueeze(-1)
        mask &= non_pad_mask

        labels = torch.where(mask, ids, mt.mask_id)

        x_ids = ids.clone()
        # 10% replace with random token
        mask_rid = get_mask_subset_prob(mask, 0.1)
        rand_id = torch.randint_like(x_ids, high=mt.opt.num_tokens)
        x_ids = torch.where(mask_rid, rand_id, x_ids)
        # 90% × 88% replace with mask token
        mask_mid = get_mask_subset_prob(mask & ~mask_rid, 0.88)
        x_ids = torch.where(mask_mid, mt.mask_id, x_ids)

        # Forward with interaction tokens
        logits = self.trans_forward(x_ids, cond, ~non_pad_mask, interaction_tokens)
        ce_loss, pred_id, acc = cal_performance(logits, labels, ignore_index=mt.mask_id)

        return ce_loss, pred_id, acc

    def forward_with_cond_scale(
        self,
        motion_ids: Tensor,
        cond: Tensor,
        padding_mask: Tensor,
        interaction_tokens: Tensor | None = None,
        cond_scale: float = 4.5,
        interaction_scale: float = 2.0,
    ) -> Tensor:
        """Dual-condition classifier-free guidance."""
        # Unconditional
        logits_uncond = self.trans_forward(
            motion_ids, cond, padding_mask,
            interaction_tokens=None, force_mask=True,
        )

        if cond_scale == 1 and interaction_tokens is None:
            return logits_uncond

        # Text-only
        logits_text = self.trans_forward(
            motion_ids, cond, padding_mask,
            interaction_tokens=None, force_mask=False,
        )

        if interaction_tokens is None:
            # Standard single-condition CFG
            return logits_uncond + cond_scale * (logits_text - logits_uncond)

        # Full (text + interaction)
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
        """Iterative unmasking generation with interaction guidance.

        Parameters
        ----------
        cond : (B, clip_dim) — CLIP text embedding
        m_lens : (B,) — desired token sequence lengths
        interaction_tokens : (B, S_int, d) or None
        timesteps : number of unmasking iterations

        Returns
        -------
        ids : (B, S) — generated token indices, -1 for padding
        """
        mt = self.mask_transformer
        device = cond.device
        B = cond.shape[0]
        S = int(m_lens.max().item())

        from models.mask_transformer.tools import lengths_to_mask

        non_pad_mask = lengths_to_mask(m_lens, S)
        padding_mask = ~non_pad_mask

        # Start fully masked
        ids = torch.where(padding_mask, mt.pad_id, mt.mask_id)
        scores = torch.where(padding_mask, 1e5, 0.0)

        for timestep, steps_until_x0 in zip(
            torch.linspace(0, 1, timesteps, device=device),
            reversed(range(timesteps)),
        ):
            rand_mask_prob = cosine_schedule(timestep)
            num_mask = (rand_mask_prob * m_lens.float()).long().clamp(min=1)

            # Re-mask lowest confidence tokens (except first step)
            if timestep > 0:
                scores_for_mask = scores.clone()
                scores_for_mask[padding_mask] = 1e5
                for i in range(B):
                    n = num_mask[i].item()
                    if n >= m_lens[i].item():
                        ids[i, :m_lens[i]] = mt.mask_id
                    else:
                        _, low_idx = scores_for_mask[i, :m_lens[i]].topk(n, largest=False)
                        ids[i, low_idx] = mt.mask_id

            # Get logits with guidance
            logits = self.forward_with_cond_scale(
                ids, cond, padding_mask,
                interaction_tokens=interaction_tokens,
                cond_scale=cond_scale,
                interaction_scale=interaction_scale,
            ).permute(0, 2, 1)  # (B, S, V)

            # Adjust temperature
            filtered_logits = logits / max(temperature, 1e-3)

            # Top-k filtering
            if topk_filter_thres < 1.0:
                V = filtered_logits.shape[-1]
                k = max(1, int(V * (1 - topk_filter_thres)))
                val, _ = filtered_logits.topk(k, dim=-1)
                filtered_logits = filtered_logits.masked_fill(
                    filtered_logits < val[..., -1:], float("-inf"),
                )

            # Sample
            probs = F.softmax(filtered_logits, dim=-1)
            sampled = probs.reshape(-1, probs.shape[-1]).multinomial(1).reshape(B, S)
            sampled_scores = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)

            # Update only masked positions
            is_masked = ids == mt.mask_id
            ids = torch.where(is_masked, sampled, ids)
            scores = torch.where(is_masked, sampled_scores, scores)

        # Mark padding as -1
        ids[padding_mask] = -1
        return ids

    def encode_text(self, raw_text: list[str]) -> Tensor:
        """Encode raw text strings via MoMask's CLIP model."""
        return self.mask_transformer.encode_text(raw_text)

    def interaction_parameters(self) -> list[nn.Parameter]:
        """Return only the new interaction cross-attention parameters (for separate LR)."""
        params = []
        for name, p in self.mask_transformer.seqTransEncoder.named_parameters():
            if "interaction_attn" in name:
                params.append(p)
        params.extend(self.interaction_tokenizer.parameters())
        return params

    def backbone_parameters(self) -> list[nn.Parameter]:
        """Return only the original MoMask parameters (for lower LR finetuning)."""
        params = []
        for name, p in self.mask_transformer.named_parameters():
            if "interaction_attn" not in name and "clip_model" not in name:
                params.append(p)
        return params
