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
    # Round-18: optional object-trajectory hint channel. 0 = disabled (object-free,
    # back-compat with Round-12/14 checkpoints). 9 = obj_com (3) + obj_rot6d (6)
    # in the frame documented by the active cache contract (Round-18-fix
    # uses `obj_pos_root0_world + obj_rot6d_world`, matching Coarse-v1
    # exactly), injected via a CMC-style zero-init-last-linear HintBlock
    # (additive to the motion input embedding after positional encoding).
    obj_traj_dim: int = 0
    obj_traj_hint_hidden_mult: int = 1         # hidden dim = d_model * mult; default = d_model


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


class ObjTrajHintBlock(nn.Module):
    """Per-frame additive spatial-cue MLP for the object-trajectory channel.

    Mirrors the CMC HintBlock pattern (external/CMC/models/omnimdm_spatial.py:167-185):
    a 4-layer MLP with **zero-init last linear** so the obj_traj contribution is
    exactly zero at step 0, and the gate opens during training. This degenerates
    the S1-O model to the object-free S1-A baseline at random init, providing
    the cleanest possible "fair starting point" for a paired ablation.

    Input shape : (B, T, obj_traj_dim)
    Output shape: (B, T, d_model)  — added per-frame to the motion input embedding.
    """

    def __init__(self, obj_traj_dim: int, d_model: int, hidden_mult: int = 1) -> None:
        super().__init__()
        if obj_traj_dim <= 0:
            raise ValueError(f"obj_traj_dim must be > 0, got {obj_traj_dim}")
        hidden = d_model * max(1, hidden_mult)
        self.layers = nn.ModuleList(
            [
                nn.Linear(obj_traj_dim, hidden),
                nn.Linear(hidden, hidden),
                nn.Linear(hidden, d_model),
                nn.Linear(d_model, d_model),       # last layer; zero-init below
            ]
        )
        self.act = nn.SiLU()
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        # First three linears: small normal init so the hidden activations
        # are non-trivial as soon as the gate opens.
        for layer in self.layers[:-1]:
            nn.init.normal_(layer.weight, std=0.02)
            nn.init.zeros_(layer.bias)
        # Last linear: zero-init so step-0 output is exactly 0 → S1-O ==
        # S1-A at init under matched seed.
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, obj_traj: Tensor) -> Tensor:
        h = obj_traj
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.act(h)
        return h


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

        # Round-18 follow-up: HintBlock instantiation MOVED to the END of
        # __init__ (after blocks + final_layer + _initialize_weights). This
        # guarantees that under the same `torch.manual_seed`, Plan A
        # (obj_traj_dim=0) and S1-O (obj_traj_dim=9) consume IDENTICAL RNG
        # for the shared `in_proj / time_embed / text_proj / init_proj /
        # blocks / final_layer` parameters → bitwise-equal shared weights.
        # The HintBlock + null_obj_traj attributes are placeholder-set here
        # so attribute access in forward() doesn't AttributeError before the
        # post-init step assigns them.
        self.obj_traj_hint = None
        self.register_parameter("null_obj_traj", None)

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

        # Round-18 follow-up fairness: build HintBlock + null_obj_traj
        # ONLY AFTER all shared parameters are created + re-initialised by
        # `_initialize_weights()`. This means the RNG state that the
        # HintBlock consumes does NOT perturb the base denoiser's weights.
        # Under matched seed, Plan A and S1-O have bitwise-equal shared
        # params; S1-O has extra params only for the HintBlock + null
        # embedding.
        if cfg.obj_traj_dim > 0:
            self.obj_traj_hint = ObjTrajHintBlock(
                obj_traj_dim=cfg.obj_traj_dim,
                d_model=cfg.d_model,
                hidden_mult=cfg.obj_traj_hint_hidden_mult,
            )
            self.null_obj_traj = nn.Parameter(torch.zeros(cfg.obj_traj_dim))

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
        cond_drop_mask: Tensor | None = None,    # (B,) bool — True drops TEXT (back-compat alias)
        obj_traj_drop_mask: Tensor | None = None,   # (B,) bool — True drops obj_traj (Round-18)
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
        # The legacy cond_drop_mask continues to drop text only (back-compat
        # with Round-12/14 checkpoints + the old single-channel CFG policy).
        text_pool_eff = self._broadcast_drop(cond_drop_mask, text_pool, self.null_text)

        # Per-frame projection + positional encoding.
        h = self.in_proj(x_t)                            # (B, T, D)
        h = self.pos_enc(h)

        # Round-18: object-trajectory HintBlock (additive, zero-init at start).
        # Only consumed when the model was constructed with obj_traj_dim > 0.
        # cond key is `obj_traj` (frame-agnostic — the cache contract
        # documents the frame: Round-18-fix uses `obj_pos_root0_world +
        # obj_rot6d_world`, matching Coarse-v1's frame exactly).
        if self.obj_traj_hint is not None:
            if "obj_traj" not in cond:
                raise KeyError(
                    "model has obj_traj_dim > 0 but cond is missing "
                    "'obj_traj' field"
                )
            obj_traj: Tensor = cond["obj_traj"]            # (B, T, obj_traj_dim)
            if obj_traj.shape[-1] != cfg.obj_traj_dim:
                raise ValueError(
                    f"obj_traj last dim {obj_traj.shape[-1]} != "
                    f"obj_traj_dim {cfg.obj_traj_dim}"
                )
            # Independent CFG dropout for obj_traj. Falls back to no-drop if
            # the trainer hasn't passed an obj_traj_drop_mask.
            obj_traj_eff = self._broadcast_drop(
                obj_traj_drop_mask, obj_traj, self.null_obj_traj,
            )
            h = h + self.obj_traj_hint(obj_traj_eff)        # (B, T, D)

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
        obj_traj_drop_mask: Tensor | None = None,
    ) -> Tensor:
        return self.denoiser(
            x_t, t, cond,
            cond_drop_mask=cond_drop_mask,
            obj_traj_drop_mask=obj_traj_drop_mask,
        )

    # ---------- sampling ---------- #

    @torch.no_grad()
    def sample(
        self,
        shape: tuple[int, int, int],
        cond: dict,
        cfg_scale: float = 1.0,
        device: torch.device | None = None,
        sampler: str = "ddpm",
        inpaint_frame0: bool = False,
    ) -> Tensor:
        """Reverse-diffusion sample loop for Coarse-v1.

        Mirrors ``GaussianDiffusion.p_sample_loop`` but specialised to
        this denoiser's narrow ``cond`` schema. No ``cond_motion`` /
        ``replacement`` / ``output_skip`` — those were CondMDI-era v9
        features that don't apply to Stage-1.

        Round-18 additions:

        - obj_traj is read from ``cond["obj_traj"]`` automatically
          when the model has ``obj_traj_dim > 0`` (no API change here).
        - ``inpaint_frame0=True`` enables RePaint-style hard-clamping of
          frame 0 to the conditioned init pose at every reverse step. At
          each timestep t (and t-1) the known init pose is forward-diffused
          to the correct noise level and substituted into ``x[:, 0, :]``.
          The model's predicted x0 also has frame 0 forced to ``init_coarse``
          before forming the posterior. This is the canonical "force known
          frames" pattern (CMC's SIM applied to a single-frame mask).
        """
        device = device or self.diffusion.betas.device
        B, T, D = shape
        x = torch.randn(shape, device=device)
        from piano.models.motion_anchordiff import _extract

        # Frame-0 inpainting needs the conditioned init pose as (B, D).
        init_for_inpaint: Tensor | None = None
        if inpaint_frame0:
            init_for_inpaint = cond.get("init_coarse", None)
            if init_for_inpaint is None:
                raise KeyError(
                    "inpaint_frame0=True but cond is missing 'init_coarse' field"
                )
            if init_for_inpaint.dim() == 1:
                # Allow a single-clip (D,) tensor; promote to (B, D).
                init_for_inpaint = init_for_inpaint.unsqueeze(0).expand(B, -1)
            if init_for_inpaint.shape != (B, D):
                raise ValueError(
                    f"init_coarse shape {tuple(init_for_inpaint.shape)} != ({B}, {D})"
                )
            # Replace the random frame-0 with the GT init pose forward-diffused
            # to the start-of-chain noise level (timestep T-1).
            t_max = torch.full((B,), self.diffusion.num_steps - 1, device=device, dtype=torch.long)
            noise0 = torch.randn(B, D, device=device)
            x_init0 = self.diffusion.q_sample(
                init_for_inpaint.unsqueeze(1).contiguous(),         # (B, 1, D)
                t_max,
                noise0.unsqueeze(1),                                  # (B, 1, D)
            ).squeeze(1)
            x[:, 0, :] = x_init0

        for t_int in reversed(range(self.diffusion.num_steps)):
            t = torch.full((B,), t_int, device=device, dtype=torch.long)
            x0_cond = self.denoiser(
                x, t, cond, cond_drop_mask=None, obj_traj_drop_mask=None,
            )
            if cfg_scale != 1.0:
                # CFG-uncond branch drops TEXT only by default; obj_traj is
                # treated as a deterministic spatial cue at sampling time
                # (matches CMC Stage-1 sample.py — Stage-1 doesn't CFG the
                # spatial hint either). If a future round wants to CFG
                # obj_traj, add an `obj_traj_cfg_scale` arg here.
                drop = torch.ones(B, dtype=torch.bool, device=device)
                x0_uncond = self.denoiser(
                    x, t, cond, cond_drop_mask=drop, obj_traj_drop_mask=None,
                )
                x0 = x0_uncond + cfg_scale * (x0_cond - x0_uncond)
            else:
                x0 = x0_cond

            if inpaint_frame0 and init_for_inpaint is not None:
                # Force x0_pred at frame 0 to the GT init pose. This means
                # the posterior mean at frame 0 is the q_posterior of the
                # GT init pose, so frame 0 of the next x_{t-1} stays
                # consistent with the conditioned init.
                x0 = x0.clone()
                x0[:, 0, :] = init_for_inpaint

            mean = self.diffusion.posterior_mean_from_x0(x0, x, t)
            if sampler == "ddpm_det" or t_int == 0:
                x = mean
            else:
                noise = torch.randn_like(x)
                log_var = _extract(
                    self.diffusion.posterior_log_variance_clipped, t, x.shape,
                )
                x = mean + (0.5 * log_var).exp() * noise

            if inpaint_frame0 and init_for_inpaint is not None and t_int > 0:
                # Re-inject the GT init pose at the new noise level for x_{t-1}.
                t_prev = torch.full((B,), t_int - 1, device=device, dtype=torch.long)
                noise_prev = torch.randn(B, D, device=device)
                x_init_prev = self.diffusion.q_sample(
                    init_for_inpaint.unsqueeze(1).contiguous(),
                    t_prev,
                    noise_prev.unsqueeze(1),
                ).squeeze(1)
                x[:, 0, :] = x_init_prev

        # Final pass: lock frame 0 to GT init pose exactly (no diffusion noise
        # since we're at t=0). This guarantees the returned sample has
        # x[:, 0, :] == init_coarse to numerical precision.
        if inpaint_frame0 and init_for_inpaint is not None:
            x = x.clone()
            x[:, 0, :] = init_for_inpaint
        return x


__all__ = [
    "CoarsePriorConfig",
    "CoarsePriorDenoiser",
    "CoarsePriorDenoiserConfig",
    "CoarsePriorDiff",
    "CoarseAdaLNBlock",
    "CoarseFinalLayer",
    "ObjTrajHintBlock",
    "make_block_causal_bool_mask",
]
