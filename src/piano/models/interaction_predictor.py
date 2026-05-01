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
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_body_parts = num_body_parts
        self.num_phases = num_phases
        self.num_support_states = num_support_states
        self.d_emb = d_emb

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
        self.part_queries = nn.Parameter(
            torch.randn(num_body_parts, d_model) * 0.02
        )
        # Frame feature + contact context → query base. The attention
        # sums frame_q + part_query[p] for each (frame, part) cell.
        self.target_query_proj = nn.Linear(d_model + d_emb, d_model)
        # Cross-attention over object tokens. need_weights=True at
        # forward time to expose the attention map as the affordance
        # output — the soft distribution over 128 tokens *is* the
        # primary v8 target prediction.
        #
        # dropout=0 here is critical: torch's MultiheadAttention applies
        # dropout to the post-softmax attention weights, then returns
        # those (rescaled) weights via need_weights. With dropout > 0
        # the returned weights no longer sum to 1, which breaks the
        # KL-divergence loss formulation. The MLP heads carry the
        # regularisation budget for v8 instead.
        self.target_attn = nn.MultiheadAttention(
            d_model, num_attn_heads, dropout=0.0, batch_first=True,
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
        if teacher_forcing:
            assert gt_contact is not None, "teacher_forcing requires gt_contact"
            contact_for_downstream = gt_contact.to(contact_prob.dtype)
        else:
            contact_for_downstream = contact_prob
        contact_emb = self.contact_to_emb(contact_for_downstream)  # (B, T, d_emb)

        # ── Level 1a: target attention ───────────────────────────
        # frame_q encodes frame feature + contact context per (B, T)
        x_with_c = torch.cat([x, contact_emb], dim=-1)             # (B, T, d + d_emb)
        frame_q = self.target_query_proj(x_with_c)                 # (B, T, d)
        # Broadcast: per-part query = frame_q + part_query[p]
        # frame_q.unsqueeze(2): (B, T, 1, d); part_queries: (P, d)
        q = frame_q.unsqueeze(2) + self.part_queries.view(1, 1, P, -1)  # (B, T, P, d)
        q_flat = q.reshape(B, T * P, -1)                            # (B, T*P, d)
        # Cross-attention. average_attn_weights=False to expose per-head
        # weights, then mean across heads to get a single affordance map.
        _, attn_weights = self.target_attn(
            q_flat, object_tokens, object_tokens,
            need_weights=True, average_attn_weights=False,
        )
        # attn_weights: (B, num_attn_heads, T*P, M) → mean across heads → (B, T*P, M)
        target_attn = attn_weights.mean(dim=1).reshape(B, T, P, M)  # (B, T, P, M)

        # Back-compat xyz output: attention-weighted centroid positions.
        # (B, T, P, M) @ (B, M, 3) → (B, T, P, 3)
        contact_target_xyz = torch.einsum(
            "btpk,bkc->btpc", target_attn, object_xyz,
        )                                                           # (B, T, P, 3)

        # ── Level 1b: phase ──────────────────────────────────────
        phase_logits = self.phase_head(x_with_c)                    # (B, T, num_phases)
        phase_prob = torch.softmax(phase_logits, dim=-1)            # (B, T, num_phases)
        if teacher_forcing:
            assert gt_phase is not None, "teacher_forcing requires gt_phase"
            phase_one_hot = torch.zeros_like(phase_prob)
            phase_one_hot.scatter_(
                -1, gt_phase.long().unsqueeze(-1).clamp_(0, self.num_phases - 1), 1.0,
            )
            phase_for_downstream = phase_one_hot
        else:
            phase_for_downstream = phase_prob
        phase_emb = self.phase_to_emb(phase_for_downstream)         # (B, T, d_emb)

        # ── Level 2: support ─────────────────────────────────────
        x_full = torch.cat([x, contact_emb, phase_emb], dim=-1)    # (B, T, d + 2*d_emb)
        support_logits = self.support_head(x_full)                  # (B, T, num_support)

        return {
            "contact_logits": contact_logits,
            "contact_state": contact_prob,
            "contact_target_attn": target_attn,
            "contact_target_xyz": contact_target_xyz,
            "phase_logits": phase_logits,
            "phase": phase_prob,
            "support_logits": support_logits,
            "support": torch.softmax(support_logits, dim=-1),
        }


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
