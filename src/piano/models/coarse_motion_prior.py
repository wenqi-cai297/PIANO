"""Stage-1 Coarse-v1 motion prior — fresh denoiser for vNext two-stage gen.

This module is intentionally object-free / plan-free / contact-free /
hand-foot-free. It does NOT import any of:

- ``piano.models.interaction_plan_encoder``
- ``piano.models.object_encoder``
- ``piano.models.dit_blocks.ConditionedEncoderLayer``  (plan-bound)
- ``piano.models.motion_anchordiff.AnchorDenoiser``    (anchor/plan-bound)

It DOES reuse generic local utilities:

- ``GaussianDiffusion`` math + ``cosine_beta_schedule`` + ``_extract``
  from ``motion_anchordiff``.
- ``SinusoidalTimestepEmbed`` and ``PositionalEncoding`` from
  ``motion_anchordiff``.
- ``modulate`` (AdaLN canonical helper) from ``dit_blocks``.

The denoiser supports two attention modes via the ``attention_mode``
config field:

- ``"none"``         — bidirectional self-attention (S1-A).
- ``"block_causal"`` — block-causal self-attention with configurable
                       ``block_size`` (S1-B).

S1-A and S1-B are otherwise identical: same input projection, same
AdaLN-Zero blocks, same final layer, same loss, same data, same
optimizer.

Design source-of-truth:
    analyses/2026-05-21_stage1_coarse_prior_literature_code_audit.md
    analyses/2026-05-21_codex_stage1_coarse_prior_design_review.md

Conventions
-----------

- Input ``x_t``: ``(B, T, coarse_dim)`` — normalized noisy Coarse-v1.
- ``t``: ``(B,)`` long, diffusion-step indices in ``[0, num_steps)``.
- ``cond["text_pool"]``: ``(B, text_dim)`` — pooled CLIP EOT feature.
- ``cond["init_coarse"]``: ``(B, coarse_dim)`` — normalized
  ``coarse_v1[..., 0, :]`` (i.e. frame 0).
- ``cond["valid_mask"]``: ``(B, T)`` bool — True at real frames.
  Used as ``key_padding_mask = ~valid_mask``.
- ``cond_drop_mask``: ``(B,) bool`` — True samples have text replaced
  by a learned null embedding. Init pose is NEVER dropped (per Codex
  §6.5).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from piano.models.motion_anchordiff import (
    DiffusionConfig, GaussianDiffusion,
    PositionalEncoding, SinusoidalTimestepEmbed,
)
from piano.models.dit_blocks import modulate


# ============================================================================
# Attention-mask helpers
# ============================================================================


def make_block_causal_bool_mask(
    seq_len: int, block_size: int, device: torch.device,
) -> Tensor:
    """Return a (T, T) bool attention mask where True = disallow.

    Inside a block of size ``block_size``, attention is bidirectional;
    across blocks, future blocks cannot be attended to by past blocks.
    Specifically, position i may attend to position j iff
    ``(i // block_size) >= (j // block_size)`` — i.e. j's block index
    is <= i's block index.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}")
    idx = torch.arange(seq_len, device=device)
    block_id = idx // int(block_size)
    # mask[i, j] = True if j's block is AFTER i's block (disallow).
    return block_id.unsqueeze(0) > block_id.unsqueeze(1)


# ============================================================================
# Model config
# ============================================================================


@dataclass(slots=True)
class CoarsePriorDenoiserConfig:
    coarse_dim: int = 23
    text_dim: int = 512
    init_pose_dim: int = 23                    # init Coarse-v1 row (= coarse_dim by default)
    d_model: int = 384
    n_layers: int = 4
    n_heads: int = 4
    ff_mult: int = 4
    dropout: float = 0.1
    max_seq_length: int = 256
    attention_mode: str = "none"               # "none" | "block_causal"
    block_size: int = 16                       # only used if attention_mode == "block_causal"


# ============================================================================
# AdaLN-Zero block (no plan cross-attn, no register tokens)
# ============================================================================


class CoarseAdaLNBlock(nn.Module):
    """DiT-style AdaLN-Zero block: self-attn + MLP, both modulated.

    Differences from ``ConditionedEncoderLayer`` (the v12 block in
    ``dit_blocks.py``):

    - No plan cross-attention. Self-attn + MLP only.
    - No temporal-conv branch.
    - Accepts an attention mask + key-padding mask via forward kwargs.
    """

    def __init__(
        self, d_model: int, n_heads: int, ff_mult: int, dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(approximate="tanh"),
            nn.Linear(d_model * ff_mult, d_model),
        )
        # AdaLN-Zero modulation MLP: 6 outputs per layer.
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model, bias=True),
        )

    def forward(
        self,
        x: Tensor,                                  # (B, T, D)
        c: Tensor,                                  # (B, D)
        attn_mask_bool: Tensor | None = None,       # (T, T) bool
        key_padding_mask: Tensor | None = None,     # (B, T) bool — True = pad
    ) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        # (1) Self-attention with AdaLN-Zero modulation
        h = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.self_attn(
            h, h, h,
            attn_mask=attn_mask_bool,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + gate_msa.unsqueeze(1) * attn_out
        # (2) MLP with AdaLN-Zero modulation
        h = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(h)
        return x


class CoarseFinalLayer(nn.Module):
    """Same pattern as v12 ``V12FinalLayer``: AdaLN-Zero shift/scale +
    zero-init Linear to output dim."""

    def __init__(self, d_model: int, out_dim: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(d_model, out_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model, bias=True),
        )

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


# ============================================================================
# Denoiser
# ============================================================================


class CoarsePriorDenoiser(nn.Module):
    """Fresh Stage-1 Coarse-v1 denoiser.

    Forward signature is intentionally narrow: a small ``cond`` dict
    with pooled text + init pose + valid mask. No object / plan /
    contact / object-token / hand / foot fields.
    """

    def __init__(self, cfg: CoarsePriorDenoiserConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.attention_mode not in ("none", "block_causal"):
            raise ValueError(
                f"attention_mode must be 'none' or 'block_causal', got {cfg.attention_mode!r}",
            )

        # Per-frame input projection: motion only. No aux channels.
        self.in_proj = nn.Linear(cfg.coarse_dim, cfg.d_model)

        # Timestep / text / init-pose -> AdaLN cond.
        self.time_embed = nn.Sequential(
            SinusoidalTimestepEmbed(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.text_proj = nn.Linear(cfg.text_dim, cfg.d_model)
        self.init_proj = nn.Linear(cfg.init_pose_dim, cfg.d_model)

        # Null embedding for text-only CFG dropout.
        self.null_text = nn.Parameter(torch.zeros(cfg.text_dim))

        # Positional encoding only on the T tokens — NO prefix/register tokens
        # so attention masks remain exactly (T, T).
        self.pos_enc = PositionalEncoding(cfg.d_model, max_len=cfg.max_seq_length)

        # Stack of AdaLN-Zero blocks.
        self.blocks = nn.ModuleList(
            [
                CoarseAdaLNBlock(
                    d_model=cfg.d_model,
                    n_heads=cfg.n_heads,
                    ff_mult=cfg.ff_mult,
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.n_layers)
            ]
        )
        self.final_layer = CoarseFinalLayer(cfg.d_model, cfg.coarse_dim)

        # Cached attention mask for block-causal (built lazily on first
        # forward with a given (T, device)).
        self._cached_mask_key: tuple[int, int, torch.device] | None = None
        self._cached_mask: Tensor | None = None

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        # Input projection: xavier on weights, zero on bias.
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.zeros_(self.in_proj.bias)
        # Per-block AdaLN-Zero: the final Linear in adaLN_modulation is zeroed
        # so every gate/shift/scale starts at 0 -> blocks are identity at init.
        for blk in self.blocks:
            nn.init.zeros_(blk.adaLN_modulation[-1].weight)
            nn.init.zeros_(blk.adaLN_modulation[-1].bias)
        # Final layer: AdaLN zeroed + readout linear zeroed -> model predicts 0
        # at init. Pairs with x0-prediction so the initial gradient is large
        # and well-conditioned.
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)
        # Text / init projections: standard small init so the cond signal is
        # non-trivial from step 0 (AdaLN-Zero gates handle the "no condition
        # at start" guarantee, not these MLPs).
        for m in (self.text_proj, self.init_proj):
            nn.init.normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)
        # Timestep embed MLPs: standard normal init.
        for m in self.time_embed:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    # --------------------------- attention mask --------------------------- #

    def _attn_mask(self, seq_len: int, device: torch.device) -> Tensor | None:
        if self.cfg.attention_mode == "none":
            return None
        key = (self.cfg.block_size, seq_len, device)
        if self._cached_mask_key != key or self._cached_mask is None:
            self._cached_mask = make_block_causal_bool_mask(
                seq_len=seq_len, block_size=self.cfg.block_size, device=device,
            )
            self._cached_mask_key = key
        return self._cached_mask

    # ------------------------------ forward ------------------------------- #

    @staticmethod
    def _broadcast_drop(
        mask: Tensor | None, x: Tensor, null_value: Tensor,
    ) -> Tensor:
        """If ``mask[b]`` is True, replace ``x[b]`` with the null embedding."""
        if mask is None:
            return x
        null = null_value.expand_as(x)
        m = mask.view(-1, *([1] * (x.dim() - 1)))
        return torch.where(m, null, x)

    def forward(
        self,
        x_t: Tensor,                     # (B, T, coarse_dim)
        t: Tensor,                       # (B,) long
        cond: dict,
        cond_drop_mask: Tensor | None = None,    # (B,) bool — True drops text
    ) -> Tensor:
        cfg = self.cfg
        B, T, D_in = x_t.shape
        if D_in != cfg.coarse_dim:
            raise ValueError(
                f"x_t last dim {D_in} != coarse_dim {cfg.coarse_dim}"
            )
        if T > cfg.max_seq_length:
            raise ValueError(
                f"seq_len {T} > max_seq_length {cfg.max_seq_length}"
            )

        text_pool: Tensor = cond["text_pool"]            # (B, text_dim)
        init_pose: Tensor = cond["init_coarse"]          # (B, init_pose_dim)
        valid_mask: Tensor | None = cond.get("valid_mask", None)  # (B, T) bool

        # CFG drop: text only — init pose is deterministic scene fact, kept.
        text_pool_eff = self._broadcast_drop(cond_drop_mask, text_pool, self.null_text)

        # Per-frame projection + positional encoding.
        h = self.in_proj(x_t)                            # (B, T, D)
        h = self.pos_enc(h)

        # AdaLN cond vector.
        t_emb = self.time_embed(t)                       # (B, D)
        c = t_emb + self.text_proj(text_pool_eff) + self.init_proj(init_pose)  # (B, D)

        # Attention masks.
        attn_mask_bool = self._attn_mask(T, x_t.device)              # (T, T) bool or None
        key_padding_mask = None
        if valid_mask is not None:
            # PyTorch convention: True = ignore that key.
            key_padding_mask = ~valid_mask.bool()

        for blk in self.blocks:
            h = blk(
                h, c,
                attn_mask_bool=attn_mask_bool,
                key_padding_mask=key_padding_mask,
            )

        x0 = self.final_layer(h, c)                      # (B, T, coarse_dim)
        return x0


# ============================================================================
# Top-level "model + diffusion" wrapper
# ============================================================================


@dataclass(slots=True)
class CoarsePriorConfig:
    diffusion: DiffusionConfig
    denoiser: CoarsePriorDenoiserConfig


class CoarsePriorDiff(nn.Module):
    """Diffusion + Coarse-v1 denoiser bundle.

    Mirrors the ``MotionAnchorDiff`` packaging style but with a much
    smaller surface area.
    """

    def __init__(self, cfg: CoarsePriorConfig) -> None:
        super().__init__()
        if cfg.diffusion.prediction_target != "x0":
            raise ValueError(
                "CoarsePriorDiff requires diffusion.prediction_target='x0' "
                "(Round-12 spec). Got "
                f"{cfg.diffusion.prediction_target!r}.",
            )
        if cfg.diffusion.objective != "ddpm":
            raise ValueError(
                "CoarsePriorDiff requires diffusion.objective='ddpm' "
                "(Round-12 spec).",
            )
        self.cfg = cfg
        self.diffusion = GaussianDiffusion(cfg.diffusion)
        self.denoiser = CoarsePriorDenoiser(cfg.denoiser)

    # ---------- training-time helpers ---------- #

    def q_sample(self, x_start: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        return self.diffusion.q_sample(x_start, t, noise)

    def forward_x0(
        self,
        x_t: Tensor,
        t: Tensor,
        cond: dict,
        cond_drop_mask: Tensor | None = None,
    ) -> Tensor:
        return self.denoiser(x_t, t, cond, cond_drop_mask=cond_drop_mask)

    # ---------- sampling ---------- #

    @torch.no_grad()
    def sample(
        self,
        shape: tuple[int, int, int],
        cond: dict,
        cfg_scale: float = 1.0,
        device: torch.device | None = None,
        sampler: str = "ddpm",
    ) -> Tensor:
        """Reverse-diffusion sample loop for Coarse-v1.

        Mirrors ``GaussianDiffusion.p_sample_loop`` but specialised to
        this denoiser's narrow ``cond`` schema. No ``cond_motion`` /
        ``replacement`` / ``output_skip`` — those were CondMDI-era v9
        features that don't apply to Stage-1.
        """
        device = device or self.diffusion.betas.device
        B, T, D = shape
        x = torch.randn(shape, device=device)
        from piano.models.motion_anchordiff import _extract

        for t_int in reversed(range(self.diffusion.num_steps)):
            t = torch.full((B,), t_int, device=device, dtype=torch.long)
            x0_cond = self.denoiser(x, t, cond, cond_drop_mask=None)
            if cfg_scale != 1.0:
                drop = torch.ones(B, dtype=torch.bool, device=device)
                x0_uncond = self.denoiser(x, t, cond, cond_drop_mask=drop)
                x0 = x0_uncond + cfg_scale * (x0_cond - x0_uncond)
            else:
                x0 = x0_cond

            mean = self.diffusion.posterior_mean_from_x0(x0, x, t)
            if sampler == "ddpm_det" or t_int == 0:
                x = mean
            else:
                noise = torch.randn_like(x)
                log_var = _extract(
                    self.diffusion.posterior_log_variance_clipped, t, x.shape,
                )
                x = mean + (0.5 * log_var).exp() * noise
        return x


__all__ = [
    "CoarsePriorConfig",
    "CoarsePriorDenoiser",
    "CoarsePriorDenoiserConfig",
    "CoarsePriorDiff",
    "CoarseAdaLNBlock",
    "CoarseFinalLayer",
    "make_block_causal_bool_mask",
]
