"""Stage-1.5 — Interaction Plan Refiner.

Produces two cond tensors that Stage-2 PB1 consumes:

  - ``stage2_coarse_extra`` (C41) — (B, T, 18)
        5 key joints × Δxyz + pelvis Δxzy (layout from
        ``src/piano/data/stage2_oracle_conditions.py:74-93``).
  - ``stage2_support`` (S4) — (B, T, 13)
        foot_stance L/R + ankle_height_norm L/R + walking_mask +
        phase sin/cos L/R + footstep x/z L/R.

Inputs (cond):
  - ``stage1_coarse`` (B, T, 23)  — from Stage-1 (at inference) or oracle
    (at training). Consumed via the existing
    ``V12InputProjection.stage1_coarse_proj`` zero-init lane.
  - ``object_world_traj`` (B, T, 9)
  - ``object_tokens`` (B, N_obj, D_obj)
  - ``text`` (B, N_text, D_text)

Architecture: same DiT-Zero + PixArt cross-attn family as Stage-2,
slightly bigger than Stage-1 (d_model=384, n_layers=6). Split readout
heads emit C41 and S4 separately so per-family gradient flow stays
unmixed at the final layer.

Design source: ``analyses/2026-05-29_stage1_and_stage1_5_design.md``.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from piano.models.dit_blocks import (
    ConditionedEncoderLayer,
    GlobalCondSummary,
    V12FinalLayer,
    V12InputProjection,
    initialize_weights_v12,
    modulate,
)
from piano.models.motion_anchordiff import (
    PositionalEncoding,
    SinusoidalTimestepEmbed,
)


# Output dims are fixed by Stage-2 PB1's input contract.
STAGE1P5_C41_DIM: int = 18      # cond["stage2_coarse_extra"]
STAGE1P5_S4_DIM: int = 13       # cond["stage2_support"]
STAGE1P5_TOTAL_DIM: int = STAGE1P5_C41_DIM + STAGE1P5_S4_DIM   # 31


@dataclass(slots=True)
class Stage1p5DenoiserConfig:
    """Stage-1.5 denoiser config.

    Output dim is fixed at 31 = C41(18) + S4(13). Stage-1 23-D output is
    the most informative cond.
    """
    motion_dim: int = STAGE1P5_TOTAL_DIM
    stage1_coarse_dim: int = 23      # Stage-1's output, consumed here
    object_traj_dim: int = 9
    text_dim: int = 512
    object_token_dim: int = 256
    object_num_tokens: int = 128

    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 4
    ff_mult: int = 4
    dropout: float = 0.1
    max_seq_length: int = 256

    use_text: bool = True

    # R33 — per-block object cross-attention. When True, each
    # ConditionedEncoderLayer gets an AdaLN-Zero cross-attn sub-layer
    # over ``cond["object_tokens"]`` between self-attn and MLP. The
    # end-of-encoder ``obj_xattn`` is kept (we don't strip it) so the
    # only change vs the V0/V7 architecture is the per-block channel.
    # Zero-init AdaLN means step-0 forward is bit-identical to V0/V7.
    enable_per_block_obj_xattn: bool = False

    # R38 — frame-0 anchor injection via zero-init Linear into the
    # input-add lane (same mechanism Stage-1 uses; see
    # ``Stage1DenoiserConfig.init_pose_dim``). 0 = OFF (V0/V7/R33/R34
    # baseline). 135 = full motion_135[:, 0, :] slice (F1 mode). 66 =
    # SMPL-22 joint world positions reshaped to (B, 22*3) at frame 0.
    # The R38 cfg gen uses F1 (135) because the inference path already
    # supplies frame-0 motion (sample_substitute_conds and diagnostic
    # paths take it from ``batch["motion"][:, 0, :]``). Zero-init Linear
    # → step-0 output is bit-identical to the baseline so a non-init
    # variant of an R34 V2-A ckpt can be re-used as warm start, although
    # the R38 cells train from scratch for cleanliness.
    init_pose_dim: int = 0


class _SplitReadout(nn.Module):
    """Final readout that emits (C41, S4) separately, both AdaLN-Zero gated.

    Mirrors ``V12FinalLayer`` shape but with two output Linears (zero-init).
    """

    def __init__(self, d_model: int, c41_dim: int, s4_dim: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(
            d_model, elementwise_affine=False, eps=1e-6,
        )
        self.linear_c41 = nn.Linear(d_model, c41_dim, bias=True)
        self.linear_s4 = nn.Linear(d_model, s4_dim, bias=True)
        # AdaLN modulation (shift+scale only, no gate — matches V12FinalLayer).
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model, bias=True),
        )
        # Zero-init both output heads + the AdaLN final Linear.
        nn.init.zeros_(self.linear_c41.weight)
        nn.init.zeros_(self.linear_c41.bias)
        nn.init.zeros_(self.linear_s4.weight)
        nn.init.zeros_(self.linear_s4.bias)
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: Tensor, c: Tensor) -> tuple[Tensor, Tensor]:
        """x: (B, T, D); c: (B, D). Returns (C41, S4) of (B, T, 18) and (B, T, 13)."""
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        h = modulate(self.norm_final(x), shift, scale)
        return self.linear_c41(h), self.linear_s4(h)


class Stage1p5Denoiser(nn.Module):
    """x₀-prediction denoiser for the (C41, S4) interaction plan.

    Forward signature is ``(x_t, t, cond, cond_drop_mask) → x0_pred``
    matching Stage-1 / Stage-2, so the same ``GaussianDiffusion.sample``
    loop can drive it. The returned ``x0_pred`` is a single (B, T, 31)
    tensor; split into (C41, S4) by ``x0_pred[..., :18]`` and
    ``x0_pred[..., 18:]``.

    Cond keys consumed:
      - ``stage1_coarse``      : (B, T, 23) — required when stage1_coarse_dim>0
      - ``object_world_traj``  : (B, T, 9)  — never CFG-dropped
      - ``object_tokens``      : (B, N_obj, object_token_dim)
      - ``text``               : (B, N_text, text_dim) (only if use_text)
    """

    def __init__(self, cfg: Stage1p5DenoiserConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.time_embed = nn.Sequential(
            SinusoidalTimestepEmbed(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

        # ---- Text (optional) ----
        self.use_text = bool(cfg.use_text) and cfg.text_dim > 0
        if self.use_text:
            self.text_proj = nn.Linear(cfg.text_dim, cfg.d_model)
            self.text_xattn = nn.MultiheadAttention(
                cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True,
            )
            self.text_norm = nn.LayerNorm(cfg.d_model)
            self.null_text = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        else:
            self.text_proj = None
            self.text_xattn = None
            self.text_norm = None
            self.register_parameter("null_text", None)

        # ---- Object cross-attention ----
        self.object_proj = nn.Linear(cfg.object_token_dim, cfg.d_model)
        self.obj_xattn = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True,
        )
        self.obj_norm = nn.LayerNorm(cfg.d_model)
        self.null_obj_tokens = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        # NOTE: obj_traj is NEVER CFG-dropped (same as Stage-1) — no
        # null_obj_traj parameter, see Stage1Denoiser for the rationale.

        # ---- V12 core (input proj + cond summary + DiT stack) ----
        self.v12_input_proj = V12InputProjection(
            motion_dim=cfg.motion_dim,
            obj_traj_dim=cfg.object_traj_dim,
            d_model=cfg.d_model,
            stage1_coarse_dim=cfg.stage1_coarse_dim,
            init_pose_dim=int(cfg.init_pose_dim),
        )
        self.use_init_pose = int(cfg.init_pose_dim) > 0
        self.v12_cond_summary = GlobalCondSummary(
            d_model=cfg.d_model, use_cond_summary_mlp=False,
        )
        self.v12_blocks = nn.ModuleList([
            ConditionedEncoderLayer(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                ff_mult=cfg.ff_mult,
                dropout=cfg.dropout,
                enable_obj_xattn=bool(cfg.enable_per_block_obj_xattn),
            )
            for _ in range(cfg.n_layers)
        ])
        self.use_per_block_obj_xattn = bool(cfg.enable_per_block_obj_xattn)
        # Use a stand-in V12FinalLayer just so initialize_weights_v12
        # can zero-init the AdaLN of every DiT block + input proj. We
        # then DROP it and use the split readout instead.
        self._dummy_final = V12FinalLayer(
            d_model=cfg.d_model, motion_dim=cfg.motion_dim,
        )
        initialize_weights_v12(
            input_proj=self.v12_input_proj,
            blocks=self.v12_blocks,
            final_layer=self._dummy_final,
            cond_summary=self.v12_cond_summary,
        )
        # Drop dummy; replace with the real split readout.
        del self._dummy_final
        self.v12_final = _SplitReadout(
            d_model=cfg.d_model,
            c41_dim=STAGE1P5_C41_DIM,
            s4_dim=STAGE1P5_S4_DIM,
        )

        self.pos_enc = PositionalEncoding(
            cfg.d_model, max_len=cfg.max_seq_length + 2,
        )

    @staticmethod
    def _broadcast_drop(
        mask: Tensor | None, x: Tensor, null_value: Tensor,
    ) -> Tensor:
        if mask is None:
            return x
        null = null_value.expand_as(x)
        m = mask.view(-1, *([1] * (x.dim() - 1)))
        return torch.where(m, null, x)

    def forward(
        self,
        x_t: Tensor,                    # (B, T, 31)  noisy (C41 ∥ S4)
        t: Tensor,                      # (B,) long
        cond: dict,                     # see docstring
        cond_drop_mask: Tensor | None,  # (B,) bool — True = drop conditioning
    ) -> Tensor:
        """Returns x0_pred (B, T, 31), with channels [0:18]=C41, [18:31]=S4."""
        B, T, _ = x_t.shape

        obj_traj: Tensor = cond["object_world_traj"]
        obj_tok: Tensor = cond["object_tokens"]
        stage1_coarse_eff: Tensor | None = None
        if self.cfg.stage1_coarse_dim > 0:
            if "stage1_coarse" not in cond:
                raise KeyError(
                    "Stage1p5Denoiser.stage1_coarse_dim>0 but "
                    "cond['stage1_coarse'] is missing."
                )
            stage1_coarse_eff = cond["stage1_coarse"]
        init_pose_eff: Tensor | None = (
            cond.get("init_pose") if self.use_init_pose else None
        )
        text_tok: Tensor | None = (
            cond.get("text") if self.use_text else None
        )

        # Timestep embedding.
        t_emb = self.time_embed(t)

        # Input projection. stage1_coarse and init_pose never CFG-dropped
        # (per Stage-1 R31 V8 design and V12InputProjection contract).
        h = self.v12_input_proj(
            x_t=x_t, obj_traj=obj_traj, stage1_coarse=stage1_coarse_eff,
            init_pose=init_pose_eff,
        )

        # Global cond summary (timestep only).
        c = self.v12_cond_summary(t_emb=t_emb)

        # Positional encoding.
        seq = self.pos_enc(h)

        # R33 — pre-compute object key/value once (shared between
        # per-block obj_xattn and end-of-encoder obj_xattn). The
        # ``object_proj`` Linear is identical for both, so this is a
        # cheap one-time projection. CFG drop applies uniformly.
        obj_kv = self.object_proj(obj_tok)
        obj_kv = self._broadcast_drop(
            cond_drop_mask, obj_kv, self.null_obj_tokens,
        )

        # DiT stack.
        for block in self.v12_blocks:
            if self.use_per_block_obj_xattn:
                seq = block(seq, c, obj_kv=obj_kv)
            else:
                seq = block(seq, c)

        # Cross-attention: text + object (end-of-encoder).
        if self.use_text:
            if text_tok is None:
                raise KeyError(
                    "Stage1p5Denoiser.use_text=True but cond['text'] is missing."
                )
            text_kv = self.text_proj(text_tok)
            text_kv = self._broadcast_drop(
                cond_drop_mask, text_kv, self.null_text,
            )
            text_attn, _ = self.text_xattn(
                seq, text_kv, text_kv, need_weights=False,
            )
            seq = self.text_norm(seq + text_attn)

        obj_attn, _ = self.obj_xattn(seq, obj_kv, obj_kv, need_weights=False)
        seq = self.obj_norm(seq + obj_attn)

        # Split readout.
        c41_pred, s4_pred = self.v12_final(seq, c)
        return torch.cat([c41_pred, s4_pred], dim=-1)
