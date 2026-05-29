"""Stage-1 — Trajectory & Orientation Generator.

Outputs ``stage1_coarse`` (B, T, 23) — the same 23-D layout consumed
by Stage-2 PB1 via ``cond["stage1_coarse"]``. Layout matches
``src/piano/data/stage1_coarse_oracle.py:191-214`` byte-for-byte:

    [ 0: 3] root_local_x, root_local_z, root_local_y
    [ 3: 6] vel_x, vel_z, vel_y
    [ 6: 9] yaw_sin, yaw_cos, yaw_vel
    [ 9:15] pelvis_rot6d
    [15:21] spine3_rot6d
    [21:22] head_height
    [22:23] shoulder_center_h

Architecture mirrors the Stage-2 ``AnchorDenoiser`` family (DiT-Zero +
PixArt cross-attn). Differences from Stage-2:

  - motion_dim = 23 (not 135).
  - No ``init_pose`` token (Stage-1 generates from frame 0).
  - No R29 typed cond machinery.
  - No ``stage1_coarse`` cond key (this stage *produces* it).

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
)
from piano.models.motion_anchordiff import (
    PositionalEncoding,
    SinusoidalTimestepEmbed,
)


# Output dim is fixed by the Stage-2 PB1 input contract.
STAGE1_COARSE_DIM: int = 23


@dataclass(slots=True)
class Stage1DenoiserConfig:
    """Stage-1 denoiser config.

    Output dim (``motion_dim``) is fixed at 23 by the Stage-2 input contract.
    Object encoder + CLIP text encoder are assumed pre-trained and
    frozen; this denoiser only handles the diffusion side.
    """
    motion_dim: int = STAGE1_COARSE_DIM
    object_traj_dim: int = 9       # 3 pos + 6 rot6d
    text_dim: int = 512            # CLIP ViT-B/32
    object_token_dim: int = 256
    object_num_tokens: int = 128

    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    ff_mult: int = 4
    dropout: float = 0.1
    max_seq_length: int = 256

    use_text: bool = True


class Stage1Denoiser(nn.Module):
    """x₀-prediction DiT-style denoiser for the 23-D ``stage1_coarse``.

    Forward signature matches Stage-2's ``AnchorDenoiser.forward`` so the
    same ``GaussianDiffusion.sample`` loop can drive it.

    Cond keys consumed:

      - ``object_world_traj`` : (B, T, 9)  — never CFG-dropped
      - ``object_tokens``     : (B, N_obj, object_token_dim)
      - ``text``              : (B, N_text, text_dim)  (only if use_text)

    Cond keys NOT consumed (raise nothing — silently ignored):
      ``init_pose``, ``stage1_coarse``, ``stage2_*`` — kept out of this
      denoiser intentionally per the design doc.
    """

    def __init__(self, cfg: Stage1DenoiserConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # ---- Embeddings ----
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

        # ``obj_traj`` is NEVER CFG-dropped (design §"CFG dropout"). Without
        # it the model cannot decide trajectory. We still expose a buffer
        # for shape compatibility with the AnchorDenoiser interface but
        # never use it.
        self.null_obj_traj = nn.Parameter(torch.zeros(cfg.object_traj_dim))

        # ---- V12 core (input proj + cond summary + DiT stack + final) ----
        self.v12_input_proj = V12InputProjection(
            motion_dim=cfg.motion_dim,
            obj_traj_dim=cfg.object_traj_dim,
            d_model=cfg.d_model,
            stage1_coarse_dim=0,                       # Stage-1 does not take it as cond
        )
        self.v12_cond_summary = GlobalCondSummary(
            d_model=cfg.d_model, use_cond_summary_mlp=False,
        )
        self.v12_blocks = nn.ModuleList([
            ConditionedEncoderLayer(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                ff_mult=cfg.ff_mult,
                dropout=cfg.dropout,
            )
            for _ in range(cfg.n_layers)
        ])
        self.v12_final_layer = V12FinalLayer(
            d_model=cfg.d_model,
            motion_dim=cfg.motion_dim,
        )
        self.pos_enc = PositionalEncoding(
            cfg.d_model, max_len=cfg.max_seq_length + 2,
        )

        initialize_weights_v12(
            input_proj=self.v12_input_proj,
            blocks=self.v12_blocks,
            final_layer=self.v12_final_layer,
            cond_summary=self.v12_cond_summary,
        )

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
        x_t: Tensor,                    # (B, T, 23)  noisy stage1_coarse
        t: Tensor,                      # (B,) long, diffusion step
        cond: dict,                     # see docstring
        cond_drop_mask: Tensor | None,  # (B,) bool — True = drop conditioning
    ) -> Tensor:
        """Returns x0_pred (B, T, 23)."""
        B, T, _ = x_t.shape

        obj_traj: Tensor = cond["object_world_traj"]      # (B, T, 9)
        obj_tok: Tensor = cond["object_tokens"]           # (B, N_obj, D_obj)
        text_tok: Tensor | None = (
            cond.get("text") if self.use_text else None
        )

        # ─── CFG drop: obj_tokens + text only. obj_traj NEVER dropped. ───
        # (design doc §"CFG dropout": without obj_traj the model cannot
        # decide trajectory; we never train it with a null obj_traj.)
        # Timestep embedding (B, D)
        t_emb = self.time_embed(t)

        # Input projection (B, T, D). stage1_coarse is intentionally absent.
        h = self.v12_input_proj(x_t=x_t, obj_traj=obj_traj)

        # Global cond summary (timestep only for Stage-1).
        c = self.v12_cond_summary(t_emb=t_emb)            # (B, D)

        # Positional encoding.
        seq = self.pos_enc(h)                              # (B, T, D)
        motion_token_start = 0

        # DiT stack.
        for block in self.v12_blocks:
            seq = block(seq, c)

        # End-of-encoder cross-attn: text then object tokens.
        if self.use_text:
            if text_tok is None:
                raise KeyError(
                    "Stage1Denoiser.use_text=True but cond['text'] is missing. "
                    "Trainer must populate text tokens (CLIP ViT-B/32 features)."
                )
            text_kv = self.text_proj(text_tok)
            text_kv = self._broadcast_drop(
                cond_drop_mask, text_kv, self.null_text,
            )
            text_attn, _ = self.text_xattn(
                seq, text_kv, text_kv, need_weights=False,
            )
            seq = self.text_norm(seq + text_attn)

        obj_kv = self.object_proj(obj_tok)
        obj_kv = self._broadcast_drop(
            cond_drop_mask, obj_kv, self.null_obj_tokens,
        )
        obj_attn, _ = self.obj_xattn(seq, obj_kv, obj_kv, need_weights=False)
        seq = self.obj_norm(seq + obj_attn)

        # Final readout.
        h_motion = seq[:, motion_token_start:, :]          # (B, T, D)
        x0 = self.v12_final_layer(h_motion, c)             # (B, T, 23)
        return x0


