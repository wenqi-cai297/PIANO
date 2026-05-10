"""Encode an InteractionPlan into a sequence of plan tokens for Stage B.

Per the reframe (analyses/piano_interaction_plan_pipeline_reframe_for_claude_code.md
§5), Stage B should not consume the plan as a dense per-frame channel —
that would recreate the dense-conditioning trap. Instead, anchors and
segments are encoded into discrete tokens; motion tokens cross-attend to
plan tokens.

This module provides:

- ``InteractionPlanEncoderConfig`` — dimensions and toggles.
- ``InteractionPlanEncoder`` — produces (plan_tokens, plan_mask) given a
  batched plan dict and the motion sequence length T.
- ``compute_plan_context_hint`` — per-frame relative-temporal feature that
  is concatenated with the noisy motion at input projection (§5.5). Sits
  alongside the cross-attention mechanism, not in place of it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from piano.data.interaction_plan_compiler import (
    NUM_ANCHOR_TYPES,
    NUM_PARTS_DEFAULT,
)


@dataclass(slots=True)
class InteractionPlanEncoderConfig:
    """Hyperparameters for the plan encoder.

    The defaults assume the compiler defaults (5 parts, 3 phase classes,
    3 support classes, K_MAX=12, S_MAX=12). When changing the compiler
    config, the encoder dims must follow.
    """
    d_model: int = 512
    num_parts: int = NUM_PARTS_DEFAULT
    num_anchor_types: int = NUM_ANCHOR_TYPES
    num_phase_classes: int = 3
    num_support_classes: int = 3
    k_max: int = 12
    s_max: int = 12

    # Whether to encode segment tokens. Per spec §13 ("Anchor tokens only,
    # no segment tokens initially"), the v10 mainline starts with anchors
    # only and adds segments once anchor routing works.
    use_segment_tokens: bool = False

    # Per-frame plan-context hint (§5.5). When True, the encoder also
    # produces a (B, T, d_hint) tensor that the denoiser concatenates with
    # the motion input projection. Pure local temporal hint, doesn't
    # replace cross-attention.
    use_plan_context_hint: bool = True
    d_hint: int = 32

    # v11 per-(anchor, part) tokenization (per
    # analyses/claude_code_v10_after_fkposfix_strategy.md §7).
    # When True, the encoder emits one token for every active
    # (anchor_idx, body_part_idx) pair instead of one token per anchor
    # with parts as a multi-hot vector. Token count goes from K_MAX to
    # at most K_MAX × P. Each token has an explicit part_id embedding
    # and a dedicated geometric target MLP, so body-part identity and
    # per-part target geometry are routed individually rather than
    # averaged into a single anchor representation.
    per_part_tokens: bool = False

    # v11 context-hint mode (§7.5). Choices:
    #   "time_only"    : original v10 hint (time/type/phase/support of
    #                    nearest anchor); does not encode target xyz or
    #                    body part — likely the cause of weak target /
    #                    part sensitivity in v10.
    #   "off"          : disable the per-frame hint entirely (rely
    #                    purely on cross-attention into plan tokens).
    #   "target_aware" : extend "time_only" with nearest-anchor
    #                    target_world (averaged over active parts),
    #                    target_world − root, and dominant part one-hot.
    context_hint_mode: str = "time_only"

    # Time-embedding dimension (sinusoidal). Half-and-half embed for the
    # normalised anchor / segment time. Must be even.
    d_time_embed: int = 64

    def __post_init__(self) -> None:  # validate
        if self.context_hint_mode not in {"time_only", "off", "target_aware"}:
            raise ValueError(
                f"context_hint_mode must be 'time_only' | 'off' | "
                f"'target_aware'; got {self.context_hint_mode!r}"
            )
        # Backwards-compat: ``use_plan_context_hint`` is the v10 toggle.
        # If it's False, force mode to "off" so old configs keep working.
        if not self.use_plan_context_hint:
            object.__setattr__(self, "context_hint_mode", "off")


# ---------------------------------------------------------------------------
# Sinusoidal time embedding for the normalised time
# ---------------------------------------------------------------------------


def _sinusoidal_time_embed(time_norm: Tensor, dim: int) -> Tensor:
    """``time_norm`` ∈ [0, 1]; output (..., dim) sinusoidal features.

    The embedding range corresponds to ~1k positional bands at the
    standard MDM / DDPM frequency, the same scheme used for diffusion
    timesteps elsewhere in this codebase.
    """
    if dim % 2 != 0:
        raise ValueError("d_time_embed must be even")
    half = dim // 2
    device = time_norm.device
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    args = time_norm.float().unsqueeze(-1) * freqs            # broadcast
    return torch.cat([args.sin(), args.cos()], dim=-1)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class InteractionPlanEncoder(nn.Module):
    """Encode anchor + (optional) segment slots into plan tokens.

    The token sequence is::

        [anchor_token_0, anchor_token_1, ..., anchor_token_{K_MAX-1},
         segment_token_0, ..., segment_token_{S_MAX-1}]    # if enabled

    Padding (mask=False) slots are embedded normally and masked out by
    the cross-attention's ``key_padding_mask``. We do not skip the
    forward pass for padded slots — keeping shapes static is friendlier
    for ``nn.MultiheadAttention``.
    """

    def __init__(self, cfg: InteractionPlanEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # ------------- Anchor token features -------------
        # The anchor input vector (per slot) concatenates:
        #   - sinusoidal time embedding                          (d_time_embed)
        #   - parts multi-hot                                     (num_parts)
        #   - target_local flattened                              (num_parts*3)
        #   - target_world flattened                              (num_parts*3)
        #   - anchor_type embedding                               (d_model//8)
        #   - phase embedding                                     (d_model//8)
        #   - support embedding                                   (d_model//8)
        #   - confidence scalar                                   (1)
        emb_small = max(cfg.d_model // 8, 16)
        self.emb_small = emb_small
        self.anchor_type_embed = nn.Embedding(cfg.num_anchor_types, emb_small)
        self.anchor_phase_embed = nn.Embedding(cfg.num_phase_classes, emb_small)
        self.anchor_support_embed = nn.Embedding(cfg.num_support_classes, emb_small)

        # ------------- v10 per-anchor token (multi-hot parts) -------------
        anchor_in_dim = (
            cfg.d_time_embed
            + cfg.num_parts
            + cfg.num_parts * 3
            + cfg.num_parts * 3
            + emb_small * 3
            + 1
        )
        self.anchor_proj = nn.Sequential(
            nn.Linear(anchor_in_dim, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

        # ------------- v11 per-(anchor, part) token -------------
        # When ``per_part_tokens`` is enabled the encoder emits one token
        # per active (anchor_idx, body_part_idx) pair. Each token has:
        #   - sinusoidal time embedding                          (d_time_embed)
        #   - body_part embedding                                (emb_small)
        #   - target_local[3]                                     (3)
        #   - target_world[3]                                     (3)
        #   - anchor_type / phase / support embeddings           (emb_small × 3)
        #   - confidence scalar                                   (1)
        # Note: target_local/world go through a small MLP first so the
        # network has a dedicated geometric-target encoder (§7.3) rather
        # than concatenating raw 3-vectors.
        if cfg.per_part_tokens:
            self.part_embed = nn.Embedding(cfg.num_parts, emb_small)
            target_mlp_in = 6  # target_local 3 + target_world 3
            target_mlp_out = emb_small
            self.target_mlp = nn.Sequential(
                nn.Linear(target_mlp_in, target_mlp_out),
                nn.GELU(),
                nn.Linear(target_mlp_out, target_mlp_out),
            )
            per_part_in_dim = (
                cfg.d_time_embed
                + emb_small      # part
                + target_mlp_out
                + emb_small * 3  # type, phase, support
                + 1              # conf
            )
            self.per_part_proj = nn.Sequential(
                nn.Linear(per_part_in_dim, cfg.d_model),
                nn.GELU(),
                nn.Linear(cfg.d_model, cfg.d_model),
            )

        # ------------- Segment token features -------------
        # The segment input vector concatenates:
        #   - sinusoidal start_time embedding                     (d_time_embed)
        #   - sinusoidal end_time   embedding                     (d_time_embed)
        #   - sinusoidal duration embedding                       (d_time_embed)
        #   - active_parts multi-hot                              (num_parts)
        #   - target_summary_local flattened                      (num_parts*3)
        #   - phase embedding                                     (emb_small)
        #   - support embedding                                   (emb_small)
        #   - confidence scalar                                   (1)
        if cfg.use_segment_tokens:
            self.segment_phase_embed = nn.Embedding(cfg.num_phase_classes, emb_small)
            self.segment_support_embed = nn.Embedding(cfg.num_support_classes, emb_small)
            segment_in_dim = (
                cfg.d_time_embed * 3
                + cfg.num_parts
                + cfg.num_parts * 3
                + emb_small * 2
                + 1
            )
            self.segment_proj = nn.Sequential(
                nn.Linear(segment_in_dim, cfg.d_model),
                nn.GELU(),
                nn.Linear(cfg.d_model, cfg.d_model),
            )

        # ------------- Plan-context hint (per-frame) -------------
        # Per-frame relative-temporal features (§5.5).
        #
        # ``time_only`` (v10 default): distance_to_prev/next_anchor / T,
        # tau, prev/next anchor-type embedding, active-segment phase/support
        # one-hot. Does NOT encode target xyz or body-part identity — was
        # likely the cause of weak target/part propagation in v10.
        #
        # ``target_aware`` (v11 §7.5): adds the dominant active part
        # one-hot at prev/next anchors and the relative target_world
        # vector (target_world − pelvis-at-prev/next) so the per-frame
        # hint encodes "where the next contact is" not just "when".
        if cfg.context_hint_mode != "off":
            base_dim = (
                3                                                # dist_prev, dist_next, tau
                + emb_small * 2                                  # prev/next type
                + cfg.num_phase_classes
                + cfg.num_support_classes
            )
            extra_dim = 0
            if cfg.context_hint_mode == "target_aware":
                extra_dim = (
                    cfg.num_parts * 2     # prev/next dominant-part one-hot
                    + 3 * 2               # prev/next target_world (averaged over active parts)
                )
            hint_in_dim = base_dim + extra_dim
            self.hint_mlp = nn.Sequential(
                nn.Linear(hint_in_dim, cfg.d_hint),
                nn.GELU(),
                nn.Linear(cfg.d_hint, cfg.d_hint),
            )

    # -------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------

    def forward(
        self,
        plan: dict[str, Tensor],
        T: int,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Encode a batched plan dict.

        Parameters
        ----------
        plan : dict
            Keys: ``anchor_*`` and (optionally) ``segment_*``. Each value
            has a leading batch dim ``B`` followed by the shapes
            documented in :mod:`piano.data.interaction_plan_compiler`.
        T : int
            Motion sequence length (used to normalise anchor times to
            [0, 1] and to compute the per-frame plan-context hint).

        Returns
        -------
        plan_tokens : (B, N_plan, d_model)
            Concatenation of [anchor tokens, segment tokens] (segments
            only if enabled).
        plan_mask : (B, N_plan) bool
            True at valid slots, False at padding. Use as
            ``key_padding_mask = ~plan_mask`` in MultiheadAttention.
        plan_context_hint : (B, T, d_hint) | None
            Per-frame temporal hint. ``None`` if disabled.
        """
        cfg = self.cfg
        device = plan["anchor_time"].device
        B = plan["anchor_time"].shape[0]

        if cfg.per_part_tokens:
            anchor_tokens, anchor_mask = self._build_per_part_tokens(plan, T, B)
        else:
            # v10 default: one token per anchor with parts as multi-hot.
            a_time_norm = plan["anchor_time"].float() / max(T, 1)
            a_time_emb = _sinusoidal_time_embed(a_time_norm, cfg.d_time_embed)  # (B, K, dt)
            a_parts = plan["anchor_part"].float()                               # (B, K, P)
            a_tloc = plan["anchor_target_local"].float().reshape(
                B, cfg.k_max, cfg.num_parts * 3
            )
            a_twrl = plan["anchor_target_world"].float().reshape(
                B, cfg.k_max, cfg.num_parts * 3
            )
            a_type = self.anchor_type_embed(plan["anchor_type"].long())         # (B, K, e)
            a_phase = self.anchor_phase_embed(plan["anchor_phase"].long())
            a_support = self.anchor_support_embed(plan["anchor_support"].long())
            a_conf = plan["anchor_conf"].float().unsqueeze(-1)                  # (B, K, 1)
            a_in = torch.cat(
                [a_time_emb, a_parts, a_tloc, a_twrl, a_type, a_phase, a_support, a_conf],
                dim=-1,
            )
            anchor_tokens = self.anchor_proj(a_in)                              # (B, K, D)
            anchor_mask = plan["anchor_mask"].bool()                            # (B, K)

        # --- Segment tokens (optional) ---
        if cfg.use_segment_tokens:
            s_start_norm = plan["segment_start"].float() / max(T, 1)
            s_end_norm = plan["segment_end"].float() / max(T, 1)
            s_dur_norm = (
                plan["segment_end"].float() - plan["segment_start"].float() + 1.0
            ).clamp_min(0.0) / max(T, 1)
            s_start_emb = _sinusoidal_time_embed(s_start_norm, cfg.d_time_embed)
            s_end_emb = _sinusoidal_time_embed(s_end_norm, cfg.d_time_embed)
            s_dur_emb = _sinusoidal_time_embed(s_dur_norm, cfg.d_time_embed)
            s_parts = plan["segment_part"].float()
            s_tloc = plan["segment_target_summary_local"].float().reshape(
                B, cfg.s_max, cfg.num_parts * 3
            )
            s_phase = self.segment_phase_embed(plan["segment_phase"].long())
            s_support = self.segment_support_embed(plan["segment_support"].long())
            s_conf = plan["segment_conf"].float().unsqueeze(-1)
            s_in = torch.cat(
                [s_start_emb, s_end_emb, s_dur_emb, s_parts, s_tloc,
                 s_phase, s_support, s_conf],
                dim=-1,
            )
            segment_tokens = self.segment_proj(s_in)                        # (B, S, D)
            segment_mask = plan["segment_mask"].bool()                      # (B, S)
            plan_tokens = torch.cat([anchor_tokens, segment_tokens], dim=1)
            plan_mask = torch.cat([anchor_mask, segment_mask], dim=1)
        else:
            plan_tokens = anchor_tokens
            plan_mask = anchor_mask

        # --- Per-frame plan-context hint ---
        plan_context_hint = None
        if cfg.context_hint_mode != "off":
            plan_context_hint = self._compute_plan_context_hint(plan, T, B)

        return plan_tokens, plan_mask, plan_context_hint

    # -------------------------------------------------------------------
    # v11 per-(anchor, part) token builder
    # -------------------------------------------------------------------

    def _build_per_part_tokens(
        self, plan: dict[str, Tensor], T: int, B: int,
    ) -> tuple[Tensor, Tensor]:
        """Emit one token per active (anchor_idx, body_part_idx) pair.

        Total token count = K_max × P (60 with the defaults). Mask is
        True only where both the anchor slot is valid AND the part is
        active (anchor_part > 0). Padded slots get a zero token; their
        contribution is masked out by the cross-attention's
        ``key_padding_mask``.

        Body-part identity is encoded by an explicit ``part_embed``
        rather than by a multi-hot vector — the spec §7's load-bearing
        change. Targets pass through a small MLP first (§7.3).
        """
        cfg = self.cfg
        K, P = cfg.k_max, cfg.num_parts

        a_time = plan["anchor_time"].long()                                  # (B, K)
        a_mask = plan["anchor_mask"].bool()                                  # (B, K)
        a_part = plan["anchor_part"].float()                                 # (B, K, P)
        a_tloc = plan["anchor_target_local"].float()                         # (B, K, P, 3)
        a_twrl = plan["anchor_target_world"].float()                         # (B, K, P, 3)
        a_type = plan["anchor_type"].long()                                  # (B, K)
        a_phase = plan["anchor_phase"].long()                                # (B, K)
        a_support = plan["anchor_support"].long()                            # (B, K)
        a_conf = plan["anchor_conf"].float()                                 # (B, K)

        # Time / type / phase / support / conf are anchor-level; broadcast
        # across the P-axis when building per-part tokens.
        time_norm = a_time.float() / max(T, 1)                               # (B, K)
        time_emb = _sinusoidal_time_embed(time_norm, cfg.d_time_embed)       # (B, K, dt)
        type_emb = self.anchor_type_embed(a_type)                            # (B, K, e)
        phase_emb = self.anchor_phase_embed(a_phase)                         # (B, K, e)
        support_emb = self.anchor_support_embed(a_support)                   # (B, K, e)

        # Expand to (B, K, P, *) for fusion
        time_emb_p = time_emb.unsqueeze(2).expand(B, K, P, cfg.d_time_embed)
        type_emb_p = type_emb.unsqueeze(2).expand(B, K, P, self.emb_small)
        phase_emb_p = phase_emb.unsqueeze(2).expand(B, K, P, self.emb_small)
        support_emb_p = support_emb.unsqueeze(2).expand(B, K, P, self.emb_small)
        conf_p = a_conf.view(B, K, 1, 1).expand(B, K, P, 1)

        # Per-part part embedding
        part_idx = torch.arange(P, device=a_time.device)                     # (P,)
        part_emb = self.part_embed(part_idx)                                 # (P, e)
        part_emb_p = part_emb.view(1, 1, P, self.emb_small).expand(B, K, P, self.emb_small)

        # Target geometry MLP — concat (target_local, target_world) per (anchor, part)
        target_in = torch.cat([a_tloc, a_twrl], dim=-1)                      # (B, K, P, 6)
        target_emb_p = self.target_mlp(target_in)                            # (B, K, P, e)

        # Fuse and project
        per_part_in = torch.cat(
            [time_emb_p, part_emb_p, target_emb_p,
             type_emb_p, phase_emb_p, support_emb_p, conf_p],
            dim=-1,
        )                                                                     # (B, K, P, in_dim)
        per_part_in = per_part_in.reshape(B, K * P, -1)                      # (B, K*P, in_dim)
        tokens = self.per_part_proj(per_part_in)                             # (B, K*P, D)

        # Mask: True iff the anchor slot is valid AND this body-part is active.
        # ``a_part > 0`` handles both binary (active=1) and multi-hot continuous
        # part vectors that the compiler may emit. Cast to bool.
        token_mask = (
            a_mask.view(B, K, 1) & (a_part > 0.0).view(B, K, P)
        ).reshape(B, K * P)                                                   # (B, K*P)
        return tokens, token_mask

    # -------------------------------------------------------------------
    # Plan-context hint (per-frame)
    # -------------------------------------------------------------------

    def _compute_plan_context_hint(
        self, plan: dict[str, Tensor], T: int, B: int,
    ) -> Tensor:
        """Per-frame relative-temporal feature (§5.5)."""
        cfg = self.cfg
        device = plan["anchor_time"].device

        anchor_time = plan["anchor_time"].long()                            # (B, K)
        anchor_mask = plan["anchor_mask"].bool()                            # (B, K)
        anchor_type = plan["anchor_type"].long()                            # (B, K)

        # For each frame t in [0, T), find the nearest valid anchor before
        # and after in time. Use a vectorised scan: for each (B, T) we
        # build the diff matrix (T, K) and reduce.
        t_grid = torch.arange(T, device=device)                             # (T,)
        # (B, T, K)
        diff = t_grid.view(1, T, 1) - anchor_time.view(B, 1, cfg.k_max)
        invalid = ~anchor_mask.view(B, 1, cfg.k_max)
        # prev: max(diff) over diff>=0 and valid; nan if none
        diff_prev = diff.float().clone()
        diff_prev[invalid.expand(-1, T, -1)] = float("-inf")
        diff_prev[diff_prev < 0] = float("-inf")
        prev_diff_val, prev_diff_idx = diff_prev.max(dim=-1)                # (B, T)

        # next: min(-diff) over diff<=0 and valid
        neg_diff = -diff.float().clone()
        neg_diff[invalid.expand(-1, T, -1)] = float("-inf")
        neg_diff[neg_diff < 0] = float("-inf")     # diff > 0 means anchor was before
        next_diff_val, next_diff_idx = neg_diff.max(dim=-1)                 # (B, T)

        # Replace -inf with 0 distance and a sentinel index 0 (will be
        # masked by the type embedding lookup below). Distance is the
        # absolute frame gap; the model will see "dist_prev/next" as the
        # raw closeness signal.
        no_prev = torch.isinf(prev_diff_val)
        no_next = torch.isinf(next_diff_val)
        prev_dist = torch.where(no_prev, torch.zeros_like(prev_diff_val), prev_diff_val)
        next_dist = torch.where(no_next, torch.zeros_like(next_diff_val), next_diff_val)

        # tau ∈ [0, 1]: position between prev and next anchor. If only one
        # side is valid, fall back to 0 (left edge) / 1 (right edge).
        denom = (prev_dist + next_dist).clamp_min(1.0)
        tau = prev_dist / denom
        tau = torch.where(no_prev & ~no_next, torch.zeros_like(tau), tau)
        tau = torch.where(~no_prev & no_next, torch.ones_like(tau), tau)

        prev_dist_n = prev_dist / max(T, 1)
        next_dist_n = next_dist / max(T, 1)

        # Type embedding lookup at prev / next indices (masked).
        # When no_prev / no_next, use type 0 — the mask channel will tell
        # the model this is invalid (we encode no_prev as a non-zero
        # tau == 0 / dist == 0 combination, no extra channel needed).
        prev_idx_safe = torch.where(no_prev, torch.zeros_like(prev_diff_idx), prev_diff_idx)
        next_idx_safe = torch.where(no_next, torch.zeros_like(next_diff_idx), next_diff_idx)
        prev_type = torch.gather(anchor_type, 1, prev_idx_safe)              # (B, T)
        next_type = torch.gather(anchor_type, 1, next_idx_safe)
        prev_type_emb = self.anchor_type_embed(prev_type)                    # (B, T, e)
        next_type_emb = self.anchor_type_embed(next_type)

        # Active-segment phase / support: for each frame find the segment
        # whose [start, end] contains it. Pick first match (earliest
        # start). Default to phase=0, support=0 when no match. We expose
        # them as one-hot to keep the hint vector simple.
        seg_start = plan["segment_start"].long()                             # (B, S)
        seg_end = plan["segment_end"].long()
        seg_mask = plan["segment_mask"].bool()
        seg_phase_idx = plan["segment_phase"].long()
        seg_support_idx = plan["segment_support"].long()

        # (B, T, S) mask: which (frame, segment) pairs are active
        in_seg = (
            (t_grid.view(1, T, 1) >= seg_start.view(B, 1, cfg.s_max))
            & (t_grid.view(1, T, 1) <= seg_end.view(B, 1, cfg.s_max))
            & seg_mask.view(B, 1, cfg.s_max)
        )
        first_active = in_seg.float().argmax(dim=-1)                         # (B, T)
        any_active = in_seg.any(dim=-1)                                      # (B, T)
        active_phase = torch.where(
            any_active,
            torch.gather(seg_phase_idx, 1, first_active),
            torch.zeros_like(first_active),
        )
        active_support = torch.where(
            any_active,
            torch.gather(seg_support_idx, 1, first_active),
            torch.zeros_like(first_active),
        )
        phase_oh = F.one_hot(active_phase, num_classes=cfg.num_phase_classes).float()
        sup_oh = F.one_hot(active_support, num_classes=cfg.num_support_classes).float()

        hint_components = [
            prev_dist_n.unsqueeze(-1),
            next_dist_n.unsqueeze(-1),
            tau.unsqueeze(-1),
            prev_type_emb,
            next_type_emb,
            phase_oh,
            sup_oh,
        ]

        if cfg.context_hint_mode == "target_aware":
            # v11 §7.5: extend the hint with prev/next anchor body-part
            # identity (dominant active part one-hot) and prev/next
            # target_world (averaged over active parts at that anchor).
            # This gives the per-frame hint explicit "where the next
            # contact is" rather than only "when".
            anchor_part = plan["anchor_part"].float()                       # (B, K, P)
            anchor_target_world = plan["anchor_target_world"].float()       # (B, K, P, 3)

            # Dominant part one-hot per anchor: argmax of anchor_part along P.
            # Inactive anchors will produce 0-vector below via the mask.
            dom_part_idx = anchor_part.argmax(dim=-1)                       # (B, K)
            dom_part_oh = F.one_hot(
                dom_part_idx, num_classes=cfg.num_parts,
            ).float()                                                        # (B, K, P)
            # Average target_world over active parts at each anchor.
            denom_p = anchor_part.sum(dim=-1, keepdim=True).clamp_min(1e-6)  # (B, K, 1)
            target_world_avg = (
                (anchor_target_world * anchor_part.unsqueeze(-1)).sum(dim=-2)
                / denom_p
            )                                                                # (B, K, 3)

            # Gather at prev / next anchor indices.
            prev_part_oh = torch.gather(
                dom_part_oh, 1,
                prev_idx_safe.unsqueeze(-1).expand(-1, -1, cfg.num_parts),
            )                                                                # (B, T, P)
            next_part_oh = torch.gather(
                dom_part_oh, 1,
                next_idx_safe.unsqueeze(-1).expand(-1, -1, cfg.num_parts),
            )
            prev_target = torch.gather(
                target_world_avg, 1,
                prev_idx_safe.unsqueeze(-1).expand(-1, -1, 3),
            )                                                                # (B, T, 3)
            next_target = torch.gather(
                target_world_avg, 1,
                next_idx_safe.unsqueeze(-1).expand(-1, -1, 3),
            )

            # Zero out where the anchor is invalid so the model knows
            # there's no nearest anchor on that side.
            prev_target = torch.where(
                no_prev.unsqueeze(-1), torch.zeros_like(prev_target), prev_target,
            )
            next_target = torch.where(
                no_next.unsqueeze(-1), torch.zeros_like(next_target), next_target,
            )
            prev_part_oh = torch.where(
                no_prev.unsqueeze(-1), torch.zeros_like(prev_part_oh), prev_part_oh,
            )
            next_part_oh = torch.where(
                no_next.unsqueeze(-1), torch.zeros_like(next_part_oh), next_part_oh,
            )
            hint_components.extend([prev_part_oh, next_part_oh, prev_target, next_target])

        hint_in = torch.cat(hint_components, dim=-1)                         # (B, T, H_in)
        return self.hint_mlp(hint_in)                                        # (B, T, d_hint)


# ---------------------------------------------------------------------------
# Cross-attention block — usable as a drop-in inside the denoiser
# ---------------------------------------------------------------------------


class PlanCrossAttentionBlock(nn.Module):
    """One block of "motion tokens cross-attend to plan tokens" + FFN.

    Intentionally lightweight: a single MultiheadAttention layer + a small
    feed-forward, with pre-norm. This block is meant to be inserted ONCE
    after the input projection and before the main self-attention encoder
    (per spec §5.4 starting recommendation). If anchor routing improves
    on the diagnostic but unobs error doesn't drop enough, we move to a
    per-layer variant.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Linear(d_model * ff_mult, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        motion: Tensor,            # (B, T, D)
        plan_tokens: Tensor,       # (B, N, D)
        plan_mask: Tensor,         # (B, N) True = valid
    ) -> Tensor:
        q = self.norm_q(motion)
        kv = self.norm_kv(plan_tokens)
        # MHA wants True = "attend to" but key_padding_mask flips: True =
        # ignore. So we negate the validity mask.
        attn, _ = self.attn(
            q, kv, kv,
            key_padding_mask=~plan_mask,
            need_weights=False,
        )
        h = motion + attn
        h = h + self.ff(self.norm_ff(h))
        return h
