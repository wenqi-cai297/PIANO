"""Interaction Predictor: predicts structured interaction latents from text + object + pose.

A Transformer stack that maps (text, object_tokens, init_pose) to per-frame
interaction labels:

    - contact_state      (T, B)    — which body parts contact the object
    - contact_target_xyz (T, B, 3) — where on the object surface, in
                                     object-local coords (regression)
    - phase              (T, P)    — interaction phase (approach/.../release)
    - support            (T, S)    — body support configuration

Each transformer block has four sublayers, each pre-norm + residual:

    self-attn          — over time tokens (incl. a prepended [POSE] token)
    text-cross-attn    — over CLIP per-token features (77 tokens)
    object-cross-attn  — over PointNet++ object tokens (128 tokens)
    FFN                — standard 2-layer MLP

Design notes (2026-04-25 rewrite — target head re-architected):

    * Target is **continuous xyz regression** in the object's local
      frame, not a softmax over K per-object FPS patches. The earlier
      16-way classification assigned patch IDs independently per object
      (hash-seeded FPS per ``object_id``), so "patch 3" on chair A and
      "patch 3" on chair B referred to different surface locations.
      This made train↔val patch semantics incompatible: the first Stage
      A training had val target top-1 = 7.6% (chance is 1/16 = 6.25%).
      All major HOI-generation papers (ContactGen ICCV'23, HOI-Diff,
      CG-HOI CVPR'24, Text2HOI CVPR'24, CHOIS ECCV'24, GenHOI 2025)
      avoid per-object fixed indices — they use either per-point
      heatmaps over the object PC, or continuous xyz regression.
      HOI-Diff ``y^o ∈ R^{8×3}`` is the closest precedent for ours.

    * CLIP conditioning uses the **per-token** sequence (B, 77, d_text),
      not just the pooled CLS/EOT vector. Pooled-only AdaLN discards the
      verb/noun/modifier structure that disambiguates "push" / "pull" /
      "sit on" / "lift" on the same object (see SALAD CVPR'25 ablation).
      This matches MoMask (our Stage B backbone), which also conditions
      via per-token CLIP cross-attention.

    * Initial pose is a **dedicated [POSE] token** prepended to the time
      tokens, not injected only at t=0. Self-attention then propagates
      pose information to every frame without depth-dilution. Input is
      SMPL-22 joint positions (66-d), not HumanML3D 263-d — frame-0
      velocities in 263-d are undefined after MoMask's process_file
      drops the first frame.

    * **Object tokens** are 128 per sequence (from PointNet++), up from
      the earlier 16. KV count is decoupled from any label-space count.

    * No Block Attention Residuals. MoonshotAI's block-AttnRes was
      validated on 3B-48B LLMs; at 10 layers / ~30M params the depth-
      dilution it targets doesn't exist.

    * No AdaLN. With per-token text cross-attn + the [POSE] token +
      object cross-attn carrying all the conditioning, AdaLN on pooled
      summaries becomes redundant.

    * **Temporal refinement** (2026-04-25, post v2 review): a
      depthwise-separable 1D conv block applied between the
      Transformer output and the per-frame heads. Adds explicit
      temporal-locality inductive bias on top of full self-attention
      — addresses the v2 failure where intra-clip predictions were
      noisy frame-to-frame even though attention saw the full
      sequence. Per the literature, "per-frame structured prediction
      over long motion sequences" tasks (MS-TCN++ CVPR'20, ASFormer
      BMVC'21, VideoPose3D CVPR'19) all use TCN or local attention
      because pure full-self-attention lacks locality bias for
      densely-labelled frame outputs.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


# ============================================================================
# Transformer block: self-attn → text-xattn → object-xattn → FFN
# ============================================================================

class PredictorBlock(nn.Module):
    """Single Transformer layer of the Interaction Predictor.

    Standard pre-norm + residual around each of four sublayers. No AdaLN,
    no depth-wise AttnRes — the earlier design added both without an
    empirical win at this scale.
    """

    def __init__(
        self,
        d_model: int = 384,
        num_heads: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Self-attention over (time tokens + [POSE] token)
        self.norm_sa = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True,
        )

        # Cross-attention to CLIP per-token features
        self.norm_tx = nn.LayerNorm(d_model)
        self.text_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True,
        )

        # Cross-attention to PointNet++ object tokens
        self.norm_ox = nn.LayerNorm(d_model)
        self.object_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True,
        )

        # Feedforward
        self.norm_ff = nn.LayerNorm(d_model)
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
        text_kv: Tensor,
        object_kv: Tensor,
        text_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Apply one predictor block.

        Parameters
        ----------
        x : (B, T+1, d) — [POSE] token at index 0 + time tokens
        text_kv : (B, 77, d) — projected CLIP per-token features
        object_kv : (B, M, d) — object tokens
        text_key_padding_mask : (B, 77) — True for padded CLIP positions

        Returns
        -------
        x : (B, T+1, d)
        """
        h = self.norm_sa(x)
        sa_out, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + sa_out

        h = self.norm_tx(x)
        tx_out, _ = self.text_attn(
            h, text_kv, text_kv,
            key_padding_mask=text_key_padding_mask, need_weights=False,
        )
        x = x + tx_out

        h = self.norm_ox(x)
        ox_out, _ = self.object_attn(h, object_kv, object_kv, need_weights=False)
        x = x + ox_out

        h = self.norm_ff(x)
        x = x + self.ffn(h)
        return x


# ============================================================================
# Temporal refinement — depthwise-separable 1D conv on the time axis
# ============================================================================

class TemporalRefineBlock(nn.Module):
    """Pre-norm depthwise-separable 1D conv with residual.

    Applied AFTER ``final_norm`` and AFTER the [POSE] token is stripped,
    so the conv operates only on the per-frame sequence (B, T, d). The
    output keeps shape (B, T, d) thanks to ``padding = kernel_size // 2``.

    Depthwise-separable structure (Chollet, *Xception*, CVPR 2017;
    used by VideoPose3D CVPR'19 for temporal pose refinement) splits
    the d×d×k full conv into a (1×k) per-channel temporal pass plus
    a (d×d×1) channel-mixing pass — same receptive field, ~k× fewer
    params than a vanilla d×d×k conv.

    For our config (d=384, k=5): depthwise has 384×5 = 1920 weights,
    pointwise has 384×384 = 147 K. Total ~150 K params (~0.5% of the
    main predictor), negligible.
    """

    def __init__(
        self,
        d_model: int = 384,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=d_model,
        )
        self.pointwise = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T, d) → (B, T, d). Pre-norm + residual."""
        h = self.norm(x).transpose(1, 2)            # (B, d, T)
        h = self.depthwise(h)
        h = self.pointwise(h)
        h = self.act(h)
        h = self.drop(h)
        return x + h.transpose(1, 2)                # (B, T, d)


# ============================================================================
# v8 helper: cross-attention that returns ONLY the attention weights
# ============================================================================
#
# `nn.MultiheadAttention` always computes V × output_projection even when
# the caller only consumes ``attn_weights`` from ``need_weights=True``.
# Under DDP, that leaves V's ``out_proj.weight`` and ``out_proj.bias``
# without gradients, which trips ``find_unused_parameters`` and crashes
# at the first backward step. The legacy fix is to set
# ``find_unused_parameters=True``, but that adds DDP comms overhead and
# masks future bugs of this class.
#
# Cleaner: a Q/K-only attention module. We emit the multi-head softmax
# weights averaged across heads and skip the V path entirely. No wasted
# compute, no unused params, no DDP find-unused workaround needed.

class CrossAttentionWeightsOnly(nn.Module):
    """Multi-head cross-attention emitting only the (averaged) attention map.

    Output shape (B, Lq, Lk). No value projection, no output projection
    — the affordance head consumes the soft distribution over keys
    directly as its prediction.

    Parameters
    ----------
    output : "softmax" (default — v8 behaviour, returns probability
        distribution that sums to 1 across keys) or "logits" (v8.1
        behaviour — returns raw QK^T/sqrt(d_h) scores averaged across
        heads, suitable for per-key sigmoid + multi-hot binary GT
        following EgoChoir / Text2HOI HOI affordance literature).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        output: str = "softmax",
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )
        if output not in ("softmax", "logits"):
            raise ValueError(
                f"output must be 'softmax' or 'logits', got {output!r}"
            )
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self._scale = 1.0 / math.sqrt(self.head_dim)
        self.output = output

    def forward(self, q: Tensor, k: Tensor) -> Tensor:
        """Compute (B, Lq, Lk) attention weights, averaged across heads."""
        B, Lq, _ = q.shape
        Lk = k.shape[1]
        q = self.q_proj(q).reshape(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).reshape(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        # (B, h, Lq, Lk)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self._scale
        if self.output == "softmax":
            return torch.softmax(scores, dim=-1).mean(dim=1)
        # logits: average across heads, return raw scores for sigmoid
        return scores.mean(dim=1)


# ============================================================================
# v9.4 helper: 3D positional encoding for object tokens (NeRF-style)
# ============================================================================
#
# Standard cross-attention over object tokens uses feature-only inner
# products (q·k). Positional structure must be inferred from the
# features alone, which conflicts with PointNet++ SA's 30 cm-radius
# ball-query smoothing — adjacent tokens (~9 cm apart) have highly
# overlapping receptive fields → similar features → low spatial
# discriminability for the dot-product mask.
#
# DETR (Carion ECCV 2020, arXiv:2005.12872) and Mask3D (Schult ICRA
# 2023, arXiv:2210.03105) explicitly add positional encoding to keys
# before cross-attn. We do the same here using a NeRF-style
# (Mildenhall ECCV 2020, arXiv:2003.08934) sinusoidal expansion of the
# 3D centroid xyz, projected to d_model via a learned linear.
#
# Frequencies: 2^k for k in [0, num_frequencies-1], scaled by π. With
# objects in object-local frame (~ [-0.5, 0.5] m), 6 frequencies give
# wavelengths from 2 m down to ~6 cm — covers from "which side of the
# object" to "this token vs. its neighbour".

class PositionalEncoding3D(nn.Module):
    """NeRF-style sinusoidal positional encoding for 3D coordinates.

    Maps (B, ..., 3) xyz coordinates to (B, ..., d_model) embeddings
    via 2*F sinusoidal basis functions per axis followed by a learned
    linear projection. F = num_frequencies.
    """

    def __init__(
        self,
        d_model: int,
        num_frequencies: int = 6,
        coord_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_frequencies = int(num_frequencies)
        self.coord_scale = float(coord_scale)
        # Frequency bank: [2^0, 2^1, ..., 2^(F-1)] × π, registered as
        # buffer so it follows .to(device) but isn't trainable.
        freqs = (2.0 ** torch.arange(num_frequencies, dtype=torch.float32)) * math.pi
        self.register_buffer("freqs", freqs, persistent=False)
        # Input dim for the projection: 3 coords × F freqs × 2 (sin, cos)
        in_dim = 3 * num_frequencies * 2
        self.proj = nn.Linear(in_dim, d_model)

    def forward(self, xyz: Tensor) -> Tensor:
        """Encode xyz coordinates to d_model embeddings.

        Parameters
        ----------
        xyz : (..., 3) — 3D coordinates in object-local frame.

        Returns
        -------
        pe : (..., d_model) — additive positional encoding.
        """
        x = xyz * self.coord_scale                                       # (..., 3)
        # Expand: (..., 3, F) by broadcasting
        scaled = x.unsqueeze(-1) * self.freqs                            # (..., 3, F)
        sin_emb = torch.sin(scaled)
        cos_emb = torch.cos(scaled)
        # Stack sin/cos along last axis, then flatten the last 3 dims.
        pe = torch.stack([sin_emb, cos_emb], dim=-1)                     # (..., 3, F, 2)
        pe = pe.reshape(*xyz.shape[:-1], -1)                             # (..., 3*F*2)
        return self.proj(pe)                                             # (..., d_model)


# ============================================================================
# v9 helper: Mask3D-style multi-layer mask decoder for affordance
# ============================================================================
#
# v8.1.1 measured topk3_iou plateauing at 0.13 on hand/foot moving
# contact, ~ 2.5× below SOTA HOI affordance F1 (~ 0.45 EgoChoir).
# Diagnosis: a single Q/K-only cross-attn layer (CrossAttentionWeightsOnly)
# does not have enough capacity to disambiguate adjacent tokens for
# moving contact targets — pelvis (stationary) works because the target
# is fixed; hand/foot (moving) need per-frame query refinement.
#
# Direct precedent: InteractVLM (Dwivedi et al., CVPR 2025,
# arXiv:2504.05303) replaced DECO's single-layer body-part attention
# with a SAM-style mask decoder + focal+L1 loss → DAMON contact F1
# 55.0% → 75.6% (+20pp). Mask3D (Schult et al., ICRA 2023,
# arXiv:2210.03105) is the architectural reference for query decoders
# over point-cloud-derived tokens.
#
# This module: N TransformerDecoder layers (self-attn over per-frame
# per-part queries → cross-attn to obj_tokens → FFN), then a final mask
# logit via dot product Q·K^T between projected queries and projected
# keys. Same Q/K-only output as v8.1 (no V projection / out_proj — still
# DDP-safe, no unused params).

class AffordanceMaskDecoder(nn.Module):
    """v9 mask decoder: stacked decoder layers + dot-product mask output.

    Parameters
    ----------
    d_model : int
    num_body_parts : int
    num_layers : decoder depth (4 by default — Mask3D uses 6, InteractVLM
        uses 6; 4 is a budget-conscious choice for our 30M predictor).
    num_heads : multi-head attention heads.
    dim_feedforward : FFN hidden dim.
    dropout : applies to FFN + self-attn dropout, NOT to the final
        Q·K^T mask logit (keeps the affordance heatmap output clean).

    Output shape: (B, T, P, M) raw logits. Caller applies sigmoid +
    focal/dice loss (training) or sigmoid + threshold/top-K (eval).
    """

    def __init__(
        self,
        d_model: int,
        num_body_parts: int,
        num_layers: int = 4,
        num_heads: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        # v9.4 (2026-05-04): NeRF-style positional encoding on object
        # token keys. Off by default for backward compat. When True,
        # forward() requires object_xyz to be passed.
        use_positional_encoding: bool = False,
        num_pe_frequencies: int = 6,
        pe_coord_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_body_parts = num_body_parts
        # Per-body-part learnable query token (5 distinct queries).
        self.part_queries = nn.Parameter(
            torch.randn(num_body_parts, d_model) * 0.02
        )
        # v9.4: positional encoding for object token keys. Added to
        # both the cross-attention input AND the mask-projection key
        # input — DETR/Mask3D pattern. Optional via flag.
        self.use_positional_encoding = bool(use_positional_encoding)
        if self.use_positional_encoding:
            self.pe3d = PositionalEncoding3D(
                d_model=d_model,
                num_frequencies=num_pe_frequencies,
                coord_scale=pe_coord_scale,
            )
        # Frame feature → query base (mixed with part query inside forward).
        # Caller is expected to pass already-contact-emb-conditioned x
        # via target_query_proj — see StructuredHead.
        # Decoder stack: TransformerDecoderLayer = self-attn(q) +
        # cross-attn(q, kv) + FFN. Pre-norm for training stability.
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        # Final mask projections. Project queries and keys into a
        # shared mask space, then dot-product. Mask3D / InteractVLM
        # pattern. d_model // 2 keeps the projections compact.
        mask_dim = d_model // 2
        self.q_mask_proj = nn.Linear(d_model, mask_dim)
        self.k_mask_proj = nn.Linear(d_model, mask_dim)
        self._mask_scale = 1.0 / math.sqrt(mask_dim)

    def forward(
        self,
        frame_q: Tensor,                  # (B, T, d_model) — query base from frame feature
        object_tokens: Tensor,            # (B, M, d_model)
        object_xyz: Tensor | None = None, # (B, M, 3) — required iff use_positional_encoding
    ) -> Tensor:
        """Return (B, T, P, M) raw mask logits.

        Caller is responsible for sigmoid + loss/threshold.
        """
        B, T, _ = frame_q.shape
        M = object_tokens.shape[1]
        P = self.num_body_parts
        # v9.4: positional encoding on keys. Object xyz is required at
        # this point — guard explicitly so misconfiguration surfaces
        # at the first forward, not at a downstream shape mismatch.
        if self.use_positional_encoding:
            if object_xyz is None:
                raise ValueError(
                    "AffordanceMaskDecoder built with "
                    "use_positional_encoding=True requires object_xyz "
                    "to be passed to forward()."
                )
            pe = self.pe3d(object_xyz.to(object_tokens.dtype))           # (B, M, d)
            object_tokens_pe = object_tokens + pe                         # (B, M, d)
        else:
            object_tokens_pe = object_tokens
        # Build per-(frame, part) queries: q[b, t, p] = frame_q[b, t] + part_q[p]
        q = frame_q.unsqueeze(2) + self.part_queries.view(1, 1, P, -1)  # (B, T, P, d)
        q_flat = q.reshape(B, T * P, -1)                                 # (B, T*P, d)
        # Decoder stack: queries refine across layers via self-attn +
        # cross-attn to object features (with PE if enabled).
        q_refined = self.decoder(q_flat, object_tokens_pe)               # (B, T*P, d)
        # Mask projection + dot product. Use PE-augmented keys here too
        # (Mask3D pattern: PE participates in both cross-attn and
        # mask-out projection so the dot product carries position
        # discrimination consistent with the cross-attn).
        q_proj = self.q_mask_proj(q_refined)                             # (B, T*P, mask_dim)
        k_proj = self.k_mask_proj(object_tokens_pe)                      # (B, M, mask_dim)
        # Logits: (B, T*P, mask_dim) @ (B, mask_dim, M) → (B, T*P, M)
        logits = torch.bmm(q_proj, k_proj.transpose(1, 2)) * self._mask_scale
        return logits.reshape(B, T, P, M)


# ============================================================================
# v8 StructuredHead — affordance-style target attention + DAG conditioning
# ============================================================================
#
# Replaces the four parallel ``nn.Linear`` heads in v7-fix with sequential
# conditioning that mirrors the pseudo-label extraction DAG:
#
#     contact ──┬──▶ target  (soft attention over 128 object tokens)
#               ├──▶ phase
#               └──▶ support  (also conditioned on phase)
#
# The target head is now an attention readout over object tokens
# (Move-as-You-Say CVPR'24 style affordance heatmap) instead of direct
# xyz regression, fixing W1 (head too thin) + W2 (head loses object
# identity) in one stroke. See
# ``analyses/2026-05-05_predictor_v8_design.md`` Section 3.
#
# Backward-compatible xyz output is retained as
# ``contact_target_xyz = einsum('btpk,bkc->btpc', attn, object_xyz)``
# (attention-weighted token positions). This keeps the existing Stage B
# inference path working without modification; v8.5 will migrate that
# path to consume ``contact_target_attn`` directly.

class StructuredHead(nn.Module):
    """v8 head: DAG-ordered conditioning + affordance-style target attention.

    Parameters
    ----------
    d_model : trunk hidden dim (matches encoder feature_dim)
    num_body_parts : 5 — left/right hand/foot + pelvis
    num_phases : 3 — non_contact / stable_contact / manipulation
    num_support_states : 4 — both_feet / single_foot / sitting / hand_support
    d_emb : dimension of the contact/phase one-hot-style embedding fed
        into downstream heads. 64 is a sensible default — small enough
        to act as a "context channel" without competing with x's
        bandwidth.
    head_hidden : MLP hidden dim for contact/phase/support 2-layer heads.
    num_attn_heads : multi-head attention heads for target xattn.
    dropout : dropout in MLP heads.
    """

    def __init__(
        self,
        d_model: int = 384,
        num_body_parts: int = 5,
        num_phases: int = 3,
        num_support_states: int = 4,
        d_emb: int = 64,
        head_hidden: int = 256,
        num_attn_heads: int = 6,
        dropout: float = 0.1,
        # v8.1 (2026-05-05): random masking replaces scheduled-sampling
        # teacher forcing (Bengio NeurIPS 2015), which was proven non-
        # consistent (Huszár arXiv:1511.05101). Following MoMask
        # (Guo et al. CVPR 2024, arXiv:2312.00063): per-batch sample
        # mask_ratio ~ Uniform[0, 1], Bernoulli-mask GT-vs-pred for the
        # downstream conditioning input, train head on every mix
        # simultaneously.
        downstream_mode: str = "tf",  # "tf" (v8) | "mask" (v8.1)
        # v8.1: target attention emits per-token sigmoid logits (multi-
        # hot binary GT) instead of softmax. Following EgoChoir
        # (NeurIPS 2024, arXiv:2405.13659) and Text2HOI (CVPR 2024,
        # arXiv:2404.00562) — HOI affordance literature consensus.
        target_attn_output: str = "softmax",  # "softmax" (v8) | "logits" (v8.1)
        # v9 (2026-05-03): replace single Q/K cross-attn with a Mask3D-
        # style multi-layer mask decoder. Direct precedent: InteractVLM
        # CVPR 2025 +20pp on DAMON contact F1.
        target_attn_kind: str = "single_layer",  # "single_layer" (v8/v8.1) | "mask_decoder" (v9)
        target_decoder_layers: int = 4,
        target_decoder_ffn: int = 1024,
        # v9.4 (2026-05-04): NeRF-style positional encoding on object
        # tokens (DETR/Mask3D pattern). Only used under
        # target_attn_kind="mask_decoder". Off for back-compat.
        target_pos_enc: bool = False,
        target_pos_enc_frequencies: int = 6,
        target_pos_enc_coord_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_body_parts = num_body_parts
        self.num_phases = num_phases
        self.num_support_states = num_support_states
        self.d_emb = d_emb
        if downstream_mode not in ("tf", "mask"):
            raise ValueError(
                f"downstream_mode must be 'tf' or 'mask', got {downstream_mode!r}"
            )
        self.downstream_mode = downstream_mode
        if target_attn_output not in ("softmax", "logits"):
            raise ValueError(
                f"target_attn_output must be 'softmax' or 'logits', "
                f"got {target_attn_output!r}"
            )
        self.target_attn_output = target_attn_output
        if target_attn_kind not in ("single_layer", "mask_decoder"):
            raise ValueError(
                f"target_attn_kind must be 'single_layer' or 'mask_decoder', "
                f"got {target_attn_kind!r}"
            )
        if target_attn_kind == "mask_decoder" and target_attn_output != "logits":
            raise ValueError(
                "target_attn_kind='mask_decoder' must pair with "
                "target_attn_output='logits' (sigmoid + multi-hot binary GT)."
            )
        self.target_attn_kind = target_attn_kind

        # ── Level 0: contact (base) ──────────────────────────────
        self.contact_head = nn.Sequential(
            nn.Linear(d_model, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, num_body_parts),
        )
        # Project the per-frame contact prob (B, T, num_body_parts) into
        # a context embedding consumed by all downstream heads. Scaled
        # init so contact_emb starts ~ 0 and downstream heads behave
        # like the legacy independent-head setup at epoch 0.
        self.contact_to_emb = nn.Linear(num_body_parts, d_emb)
        nn.init.normal_(self.contact_to_emb.weight, std=0.02)
        nn.init.zeros_(self.contact_to_emb.bias)

        # ── Level 1a: target (attention over object tokens) ──────
        # Per-body-part learnable query token (separate computational
        # path per part — fixes the v7-fix W1 "single Linear cant
        # span 5-part output" issue). Distinct queries per part so
        # each part attends differently to object tokens.
        #
        # Owned by StructuredHead under target_attn_kind="single_layer";
        # under "mask_decoder" the AffordanceMaskDecoder owns its own
        # part_queries — declaring them here would leave them unused
        # under DDP and trip the unused-parameter check at backward.
        if target_attn_kind == "single_layer":
            self.part_queries = nn.Parameter(
                torch.randn(num_body_parts, d_model) * 0.02
            )
        # Frame feature + contact context → query base. Always defined
        # (used by both single_layer and mask_decoder paths).
        self.target_query_proj = nn.Linear(d_model + d_emb, d_model)
        # Target attention. Two implementations:
        # - "single_layer" (v8, v8.1, v8.1.1): CrossAttentionWeightsOnly
        #   — single Q/K-only attention. Output is (B, T*P, M)
        #   {softmax | logits} via head-averaging.
        # - "mask_decoder" (v9): AffordanceMaskDecoder — N-layer
        #   TransformerDecoder (self-attn + cross-attn + FFN) followed
        #   by Q·K^T mask projection. Mask3D / InteractVLM style.
        if target_attn_kind == "single_layer":
            self.target_attn = CrossAttentionWeightsOnly(
                d_model=d_model, num_heads=num_attn_heads,
                output=target_attn_output,
            )
        else:
            self.target_attn = AffordanceMaskDecoder(
                d_model=d_model,
                num_body_parts=num_body_parts,
                num_layers=target_decoder_layers,
                num_heads=num_attn_heads,
                dim_feedforward=target_decoder_ffn,
                dropout=dropout,
                use_positional_encoding=bool(target_pos_enc),
                num_pe_frequencies=int(target_pos_enc_frequencies),
                pe_coord_scale=float(target_pos_enc_coord_scale),
            )

        # ── Level 1b: phase (cond on contact) ────────────────────
        self.phase_head = nn.Sequential(
            nn.Linear(d_model + d_emb, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, num_phases),
        )
        self.phase_to_emb = nn.Linear(num_phases, d_emb)
        nn.init.normal_(self.phase_to_emb.weight, std=0.02)
        nn.init.zeros_(self.phase_to_emb.bias)

        # ── Level 2: support (cond on contact + phase) ───────────
        self.support_head = nn.Sequential(
            nn.Linear(d_model + 2 * d_emb, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, num_support_states),
        )

    def forward(
        self,
        x: Tensor,
        object_tokens: Tensor,
        object_xyz: Tensor,
        gt_contact: Tensor | None = None,
        gt_phase: Tensor | None = None,
        teacher_forcing: bool = False,
    ) -> dict[str, Tensor]:
        """Run the structured head.

        Parameters
        ----------
        x : (B, T, d_model) — per-frame trunk features (POSE token already stripped)
        object_tokens : (B, M=128, d_model) — object encoder features
        object_xyz : (B, M=128, 3) — object encoder centroid positions,
            in object-local frame (consistent with the GT extraction's
            object-local frame).
        gt_contact : (B, T, num_body_parts) — GT contact prob in
            {0., 1.} from the pseudo-labels. Required when
            ``teacher_forcing=True``.
        gt_phase : (B, T) long — GT phase ids in [0, num_phases).
            Required when ``teacher_forcing=True``.
        teacher_forcing : if True (training only), feed GT contact +
            GT phase as the conditioning input to downstream heads.
            If False, feed model predictions (sigmoid / softmax). The
            decision is made per-batch by the caller (scheduled
            sampling), not per-frame.

        Returns
        -------
        Dict with keys:
            contact_logits      : (B, T, num_body_parts)
            contact_state       : sigmoid of the above
            contact_target_attn : (B, T, num_body_parts, M) softmax
                                  over object tokens — the primary v8
                                  affordance output
            contact_target_xyz  : (B, T, num_body_parts, 3) — back-compat
                                  attention-weighted token xyz
            phase_logits        : (B, T, num_phases)
            phase               : softmax
            support_logits      : (B, T, num_support_states)
            support             : softmax
        """
        B, T, _ = x.shape
        M = object_tokens.shape[1]
        P = self.num_body_parts

        # ── Level 0: contact ─────────────────────────────────────
        contact_logits = self.contact_head(x)                      # (B, T, P)
        contact_prob = torch.sigmoid(contact_logits)               # (B, T, P)
        contact_for_downstream = self._mix_with_gt(
            pred=contact_prob, gt=gt_contact,
            teacher_forcing=teacher_forcing,
            training=self.training,
        )
        contact_emb = self.contact_to_emb(contact_for_downstream)  # (B, T, d_emb)

        # ── Level 1a: target attention ───────────────────────────
        # frame_q encodes frame feature + contact context per (B, T)
        x_with_c = torch.cat([x, contact_emb], dim=-1)             # (B, T, d + d_emb)
        frame_q = self.target_query_proj(x_with_c)                 # (B, T, d)
        if self.target_attn_kind == "single_layer":
            # v8 / v8.1 path: per-(frame, part) query → single Q/K cross-attn.
            q = frame_q.unsqueeze(2) + self.part_queries.view(1, 1, P, -1)  # (B, T, P, d)
            q_flat = q.reshape(B, T * P, -1)                                 # (B, T*P, d)
            target_attn_raw = self.target_attn(q_flat, object_tokens)        # (B, T*P, M)
            target_attn_raw = target_attn_raw.reshape(B, T, P, M)            # (B, T, P, M)
        else:
            # v9 path: AffordanceMaskDecoder builds queries internally
            # (frame_q + part_query) and runs the N-layer decoder. Output
            # is (B, T, P, M) raw logits. v9.4: pass object_xyz so the
            # decoder can add positional encoding to keys when enabled.
            target_attn_raw = self.target_attn(
                frame_q, object_tokens, object_xyz=object_xyz,
            )

        out: dict[str, Tensor] = {
            "contact_logits": contact_logits,
            "contact_state": contact_prob,
        }
        if self.target_attn_output == "softmax":
            # v8 path: target_attn_raw is a probability distribution.
            # Emit it directly + an attention-weighted xyz back-compat
            # output for Stage B's existing xyz consumption.
            out["contact_target_attn"] = target_attn_raw
            out["contact_target_xyz"] = torch.einsum(
                "btpk,bkc->btpc", target_attn_raw, object_xyz,
            )
        else:
            # v8.1 path: target_attn_raw is logits. Stage B v8.1b will
            # consume the per-token sigmoid mask directly. No
            # back-compat xyz emitted (Path B).
            out["contact_target_attn_logits"] = target_attn_raw
            out["contact_target_attn"] = torch.sigmoid(target_attn_raw)

        # ── Level 1b: phase ──────────────────────────────────────
        phase_logits = self.phase_head(x_with_c)                    # (B, T, num_phases)
        phase_prob = torch.softmax(phase_logits, dim=-1)            # (B, T, num_phases)
        # Build one-hot from gt_phase if available (used only when feeding
        # GT downstream — either TF=True or in mask mode where some
        # cells are GT-fed).
        if gt_phase is not None:
            gt_phase_one_hot = torch.zeros_like(phase_prob)
            gt_phase_one_hot.scatter_(
                -1, gt_phase.long().unsqueeze(-1).clamp_(0, self.num_phases - 1), 1.0,
            )
        else:
            gt_phase_one_hot = None
        phase_for_downstream = self._mix_with_gt(
            pred=phase_prob, gt=gt_phase_one_hot,
            teacher_forcing=teacher_forcing,
            training=self.training,
        )
        phase_emb = self.phase_to_emb(phase_for_downstream)         # (B, T, d_emb)

        # ── Level 2: support ─────────────────────────────────────
        x_full = torch.cat([x, contact_emb, phase_emb], dim=-1)    # (B, T, d + 2*d_emb)
        support_logits = self.support_head(x_full)                  # (B, T, num_support)

        out["phase_logits"] = phase_logits
        out["phase"] = phase_prob
        out["support_logits"] = support_logits
        out["support"] = torch.softmax(support_logits, dim=-1)
        return out

    def _mix_with_gt(
        self,
        pred: Tensor,
        gt: Tensor | None,
        teacher_forcing: bool,
        training: bool,
    ) -> Tensor:
        """Mix model predictions with GT for downstream conditioning.

        Three modes:
        - eval / inference: always return ``pred`` (model never sees GT
          at test time)
        - training, ``downstream_mode == "tf"``: return ``gt`` if
          ``teacher_forcing`` else ``pred`` (Bengio NeurIPS 2015
          scheduled sampling — v8 behaviour)
        - training, ``downstream_mode == "mask"``: per-batch sample
          ``r ~ Uniform[0, 1]``, draw a Bernoulli mask of that ratio,
          mix ``r * gt + (1-r) * pred`` element-wise (MoMask CVPR 2024
          random masking — v8.1 behaviour). The model sees every
          information mix, never an extreme it wasn't trained for.
        """
        if not training or gt is None:
            return pred
        gt = gt.to(pred.dtype)
        if self.downstream_mode == "tf":
            return gt if teacher_forcing else pred
        # mask mode: per-batch ratio, Bernoulli mask per (B, T, *) cell
        # Shape match: gt.shape == pred.shape (we built phase one-hot above).
        mask_ratio = torch.rand((), device=pred.device).item()
        # Mask drawn over leading dims (B, T) but broadcast across the
        # last (num_classes / num_parts) so the entire prediction for a
        # cell is either GT or pred — keeps the per-cell distribution
        # internally consistent.
        leading = pred.shape[:-1]
        mask = torch.bernoulli(
            torch.full(leading, mask_ratio, device=pred.device, dtype=pred.dtype)
        ).unsqueeze(-1)
        return mask * gt + (1.0 - mask) * pred


# ============================================================================
# Full Interaction Predictor
# ============================================================================

class InteractionPredictor(nn.Module):
    """Predicts structured interaction latents from text + object + initial pose.

    Parameters
    ----------
    d_model : hidden dimension (384 matches MoMask's latent_dim)
    num_layers : number of transformer blocks
    num_heads : attention heads per block
    dim_feedforward : FFN hidden dimension
    dropout : dropout rate
    text_dim : CLIP text embedding dimension (512 for ViT-B/32)
    pose_dim : initial pose feature dimension (66 = 22 joints × 3)
    max_seq_length : maximum number of output frames
    num_body_parts : B — number of tracked body parts
    target_coord_dim : output dim of the contact-target regression head
        (3 for xyz in object-local frame). ``num_object_patches`` (legacy
        name for the discarded K-way classification head) is still
        accepted for config back-compat but silently remapped to 3.
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
        pose_dim: int = 66,
        max_seq_length: int = 196,
        num_body_parts: int = 5,
        target_coord_dim: int = 3,
        num_phases: int = 5,
        num_support_states: int = 4,
        # Temporal refinement (post-Transformer, pre-heads). Default on.
        temporal_refine_enabled: bool = True,
        temporal_refine_kernel_size: int = 5,
        temporal_refine_dropout: float = 0.1,
        # v8 (2026-05-05): structured head with DAG conditioning + affordance-
        # style target attention. Default off for backward compat with
        # v6 / v7 / v7-fix configs. When enabled, the four parallel
        # nn.Linear heads are replaced with the StructuredHead module
        # above. See analyses/2026-05-05_predictor_v8_design.md.
        structured_head: bool = False,
        structured_head_d_emb: int = 64,
        structured_head_hidden: int = 256,
        structured_head_attn_heads: int = 6,
        # v8.1 (2026-05-05): "tf" preserves v8 behaviour (caller decides
        # teacher_forcing per batch). "mask" replaces TF with per-batch
        # random Bernoulli mask between GT and pred (MoMask CVPR 2024).
        structured_head_downstream_mode: str = "tf",
        # v8.1: "softmax" preserves v8 KL-on-softmax target output;
        # "logits" emits raw scores for sigmoid + multi-hot binary GT
        # supervision (EgoChoir / Text2HOI HOI affordance literature).
        structured_head_target_attn_output: str = "softmax",
        # v9 (2026-05-03): "single_layer" preserves v8/v8.1 single
        # Q/K cross-attn; "mask_decoder" replaces it with Mask3D-style
        # multi-layer decoder (Schult ICRA 2023, Dwivedi CVPR 2025
        # InteractVLM). Pairs with target_attn_output="logits".
        structured_head_target_attn_kind: str = "single_layer",
        structured_head_target_decoder_layers: int = 4,
        structured_head_target_decoder_ffn: int = 1024,
        # v9.4 (2026-05-04): NeRF-style positional encoding on object
        # tokens for the mask decoder. Off for back-compat.
        structured_head_target_pos_enc: bool = False,
        structured_head_target_pos_enc_frequencies: int = 6,
        structured_head_target_pos_enc_coord_scale: float = 1.0,
        # v9.2 (2026-05-03): motion-aware trunk with MoMask-style random
        # masking. When enabled, per-frame body kinematics
        # (joints_per_frame: B, T, 22, 3) are projected and added to
        # time tokens as an extra signal. Random mask ratio per batch
        # ~ Uniform[0, 1] (consistent estimator, Huszár 2015) handles
        # train-test asymmetry: at inference, ``joints_per_frame=None``
        # → all-mask path matches training's r=1 distribution.
        motion_aware_trunk: bool = False,
        motion_input_dim: int = 66,           # 22 SMPL joints × 3
        # Legacy alias: older configs pass ``num_object_patches=16``;
        # ignored since the target head is now an xyz regressor.
        num_object_patches: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.num_body_parts = num_body_parts
        self.target_coord_dim = target_coord_dim
        self.num_layers = num_layers
        if num_object_patches is not None and num_object_patches != target_coord_dim:
            # Back-compat: don't crash older configs, but don't honour
            # the discarded classification shape.
            pass

        # Learnable time-token bank + fixed sinusoidal positions
        self.time_tokens = nn.Parameter(torch.randn(1, max_seq_length, d_model) * 0.02)
        self.register_buffer(
            "pos_encoding",
            self._sinusoidal_encoding(max_seq_length, d_model),
            persistent=False,
        )

        # v9.2 (2026-05-03): motion-aware trunk. Per-frame joints get
        # projected into d_model and added to time tokens during forward.
        # Train-test asymmetry handled via MoMask CVPR 2024 random
        # masking: each training batch samples mask_ratio ~ Uniform[0,1]
        # and Bernoulli-masks per-frame joints with that ratio. At
        # inference, joints_per_frame=None → mask_ratio=1 (all masked,
        # matches a strict subset of the training distribution). The
        # ``joint_mask_emb`` learnable [MASK] embedding fills masked
        # positions — same primitive as our v8.1 contact / phase mask
        # downstream conditioning.
        #
        # Reference: Guo et al., MoMask CVPR 2024, arXiv:2312.00063.
        # Adaptation note: MoMask uses cosine schedule + top-k masking
        # for discrete iterative generation; for continuous joint xyz
        # features we use uniform mask ratio + per-frame Bernoulli
        # (simpler, gives uniform train coverage, no iterative inference
        # to bias toward).
        self.motion_aware_trunk = bool(motion_aware_trunk)
        if motion_aware_trunk:
            self.joint_proj = nn.Linear(motion_input_dim, d_model)
            self.joint_mask_emb = nn.Parameter(
                torch.randn(1, 1, d_model) * 0.02
            )

        # [POSE] token: pose projected into model space, carries initial
        # body state as its own sequence position (index 0)
        self.pose_proj = nn.Linear(pose_dim, d_model)

        # Project CLIP per-token features (512) into model space once.
        # All blocks share this projection — saves compute vs. each MHA
        # re-projecting (B, 77, 512) → (B, 77, d_model) with its own K/V
        # weights.
        self.text_proj = nn.Linear(text_dim, d_model)

        # Transformer stack
        self.layers = nn.ModuleList([
            PredictorBlock(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # Optional temporal refinement before the heads — gives the
        # heads a smoothed per-frame embedding with explicit local
        # bias. See TemporalRefineBlock docstring.
        self.temporal_refine_enabled = temporal_refine_enabled
        if temporal_refine_enabled:
            self.temporal_refine = TemporalRefineBlock(
                d_model=d_model,
                kernel_size=temporal_refine_kernel_size,
                dropout=temporal_refine_dropout,
            )

        # Output heads. Two modes:
        #
        # (a) Legacy independent heads (v6 / v7 / v7-fix): four parallel
        #     ``nn.Linear`` projections of the same trunk feature. No
        #     cross-head information flow. ``contact_target_xyz`` is
        #     regressed directly in the object-local frame.
        #
        # (b) StructuredHead (v8+): DAG-ordered conditioning that mirrors
        #     the pseudo-label extraction order (contact → {target,
        #     phase} → support), with affordance-style target attention
        #     over the 128 object tokens. Requires the ObjectEncoder to
        #     also pass token xyz into ``forward``.
        self.structured_head = structured_head
        if structured_head:
            self.head = StructuredHead(
                d_model=d_model,
                num_body_parts=num_body_parts,
                num_phases=num_phases,
                num_support_states=num_support_states,
                d_emb=structured_head_d_emb,
                head_hidden=structured_head_hidden,
                num_attn_heads=structured_head_attn_heads,
                dropout=dropout,
                downstream_mode=structured_head_downstream_mode,
                target_attn_output=structured_head_target_attn_output,
                target_attn_kind=structured_head_target_attn_kind,
                target_decoder_layers=structured_head_target_decoder_layers,
                target_decoder_ffn=structured_head_target_decoder_ffn,
                target_pos_enc=structured_head_target_pos_enc,
                target_pos_enc_frequencies=structured_head_target_pos_enc_frequencies,
                target_pos_enc_coord_scale=structured_head_target_pos_enc_coord_scale,
            )
        else:
            self.contact_head = nn.Linear(d_model, num_body_parts)
            self.target_head = nn.Linear(d_model, num_body_parts * target_coord_dim)
            self.phase_head = nn.Linear(d_model, num_phases)
            self.support_head = nn.Linear(d_model, num_support_states)

    def forward(
        self,
        text_tokens: Tensor,
        object_tokens: Tensor,
        init_pose: Tensor,
        seq_length: int | None = None,
        text_key_padding_mask: Tensor | None = None,
        object_xyz: Tensor | None = None,
        gt_contact: Tensor | None = None,
        gt_phase: Tensor | None = None,
        teacher_forcing: bool = False,
        joints_per_frame: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Predict interaction latents.

        Parameters
        ----------
        text_tokens : (B, 77, text_dim) — CLIP per-token features.
            Use the output of ``encode_text`` up to and including
            ``ln_final`` — i.e. before the pooled EOT projection.
        object_tokens : (B, M, d_model) — from ObjectEncoder
        init_pose : (B, pose_dim) — initial pose features (joint xyz)
        seq_length : output length (defaults to max_seq_length)
        text_key_padding_mask : (B, 77) — True for padded positions. Safe
            to leave None for CLIP ViT-B/32 — its fixed context length
            and learned padding usually don't poison cross-attention,
            but pass the mask when available (use tokenizer's attention
            mask: True where padded).

        Returns
        -------
        Dictionary with keys (all on time positions, [POSE] stripped):
            contact_state      : (B, T, num_body_parts) — sigmoid probs
            contact_target_xyz : (B, T, num_body_parts, 3) — xyz in
                object-local frame (regression, no activation)
            phase              : (B, T, num_phases) — softmax
            support            : (B, T, num_support_states) — softmax
            contact_logits, phase_logits, support_logits
                — raw logits for loss computation
        """
        B = text_tokens.shape[0]
        T = seq_length or self.max_seq_length

        # Time tokens with sinusoidal positional encoding
        time_x = self.time_tokens[:, :T, :].expand(B, -1, -1).contiguous()
        time_x = time_x + self.pos_encoding[:T, :].unsqueeze(0)

        # v9.2: motion-aware trunk. Per-frame joints (when provided)
        # are projected to d_model and added to time tokens. Random
        # masking handles train-test asymmetry.
        if self.motion_aware_trunk:
            time_x = time_x + self._build_joint_signal(
                joints_per_frame=joints_per_frame, B=B, T=T,
                device=time_x.device, dtype=time_x.dtype,
            )

        # [POSE] token at index 0 — gets no positional offset (it's not a
        # frame). Self-attn propagates pose info to all time tokens.
        pose_emb = self.pose_proj(init_pose).unsqueeze(1)   # (B, 1, d)
        x = torch.cat([pose_emb, time_x], dim=1)            # (B, T+1, d)

        # Project CLIP text features once
        text_kv = self.text_proj(text_tokens)               # (B, 77, d)

        for layer in self.layers:
            x = layer(
                x, text_kv, object_tokens,
                text_key_padding_mask=text_key_padding_mask,
            )
        x = self.final_norm(x)

        # Drop [POSE] position before emitting per-frame predictions
        x = x[:, 1:, :]                                     # (B, T, d)

        # Temporal refinement on the per-frame sequence (POSE token
        # already stripped). Pre-norm + residual; preserves shape.
        if self.temporal_refine_enabled:
            x = self.temporal_refine(x)

        if self.structured_head:
            if object_xyz is None:
                raise ValueError(
                    "structured_head=True requires object_xyz "
                    "(pass ObjectEncoder(pc, return_xyz=True))."
                )
            return self.head(
                x=x,
                object_tokens=object_tokens,
                object_xyz=object_xyz,
                gt_contact=gt_contact,
                gt_phase=gt_phase,
                teacher_forcing=teacher_forcing,
            )

        # Legacy independent-heads path (v6 / v7 / v7-fix)
        contact_logits = self.contact_head(x)               # (B, T, num_body_parts)
        target_xyz = self.target_head(x)                    # (B, T, num_body_parts * 3)
        phase_logits = self.phase_head(x)                   # (B, T, P)
        support_logits = self.support_head(x)               # (B, T, S)

        target_xyz = target_xyz.reshape(
            B, T, self.num_body_parts, self.target_coord_dim,
        )

        return {
            "contact_state": torch.sigmoid(contact_logits),
            "contact_target_xyz": target_xyz,
            "phase": torch.softmax(phase_logits, dim=-1),
            "support": torch.softmax(support_logits, dim=-1),
            "contact_logits": contact_logits,
            "phase_logits": phase_logits,
            "support_logits": support_logits,
        }

    def _build_joint_signal(
        self,
        joints_per_frame: Tensor | None,
        B: int,
        T: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """v9.2 motion-aware time-token signal with random masking.

        Three modes:
        - inference (joints_per_frame is None): all-mask. Every time
          token gets the [MASK] embedding. Equivalent to the
          training distribution at mask_ratio = 1, so no distribution
          shift.
        - training with joints (and self.training): Bernoulli mask
          with per-batch ratio r ~ Uniform[0, 1]. Each (b, t) cell
          either keeps its projected joint or gets [MASK]. The model
          is trained on every information mix, including r=1
          (matches inference) and r=0 (full info regularizer).
        - eval with joints (joints provided + self.eval()): no
          masking, full info — used by Stage A independent eval if
          we ever want to measure the "with privileged info" upper
          bound; not the v18 production path.

        Following Guo et al., MoMask (CVPR 2024, arXiv:2312.00063)
        adapted for continuous joint xyz features:
        - Per-batch uniform mask ratio (simpler than cosine schedule;
          appropriate when there is no iterative inference)
        - Per-(B, T) Bernoulli mask (simpler than top-k; we don't
          require an exact masked-cell count)
        - Single learnable [MASK] embedding fills masked positions
        """
        if joints_per_frame is None:
            # Inference path: all-mask. Broadcast [MASK] embedding.
            return self.joint_mask_emb.to(device=device, dtype=dtype).expand(B, T, -1)

        if joints_per_frame.dim() == 4:
            joints_flat = joints_per_frame.reshape(B, T, -1)
        else:
            joints_flat = joints_per_frame
        if joints_flat.shape[-1] != self.joint_proj.in_features:
            raise ValueError(
                f"joints_per_frame last-dim must be {self.joint_proj.in_features} "
                f"(matches motion_input_dim); got {joints_flat.shape[-1]}"
            )
        joints_emb = self.joint_proj(joints_flat.to(dtype))  # (B, T, d)
        mask_emb = self.joint_mask_emb.to(device=device, dtype=dtype)

        if self.training:
            # Per-batch uniform mask ratio in [0, 1]; a (B, T, 1)
            # Bernoulli mask gates GT joints vs [MASK].
            mask_ratio = torch.rand((), device=device, dtype=dtype).item()
            keep = torch.bernoulli(
                torch.full((B, T, 1), 1.0 - mask_ratio, device=device, dtype=dtype)
            )
            return keep * joints_emb + (1.0 - keep) * mask_emb
        # Eval with joints provided: no masking.
        return joints_emb

    @staticmethod
    def _sinusoidal_encoding(length: int, d_model: int) -> Tensor:
        """Standard sinusoidal positional encoding (not learnable)."""
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
