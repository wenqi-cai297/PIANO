"""DiT/InterGen-style conditional transformer building blocks for PIANO v12 (+v13).

Per the v12 design doc (analyses/2026-05-11_v12_architecture_design_doc.md), this
module provides:

  modulate(x, shift, scale)            — DiT canonical AdaLN modulation helper
  V12InputProjection                   — separate per-channel input projections (motion / z_int / obj_traj / plan_hint), summed
  GlobalCondSummary                    — per-sample (B, D) condition vector for AdaLN
  ConditionedEncoderLayer              — DiT/InterGen-style block: AdaLN-Zero self-attn + unmodulated plan cross-attn + AdaLN-Zero MLP
  V12FinalLayer                        — final readout with AdaLN-Zero + zero-init linear

Per the v13 design doc (analyses/2026-05-11_v13_dynhead_temporalconv_design.md):
  TemporalConvResidual                 — depthwise Conv1D residual with zero-init gate (Conformer/ConvNeXt-style local temporal bias)
  V13DynamicsHead                      — base + integrated-velocity residual head

All modules implement the *initialize_weights_v12* convention (called externally by
AnchorDenoiser.initialize_weights when use_dit_block=True). Zero-init is critical
for the AdaLN-Zero training stability guarantee.

References (verbatim source in analyses/2026-05-11_v12_dit_pixart_reference_code.md):
- DiT: facebookresearch/DiT@models.py
- PixArt-α: PixArt-alpha/PixArt-alpha@diffusion/model/nets/{PixArt,PixArt_blocks}.py
- InterGen: tr3e/InterGen@models/{layers,blocks,nets}.py (closest motion-domain analog)
- Conformer (Gulati et al. Interspeech 2020) for v13 local temporal coupling.
- ConvNeXt (Liu et al. CVPR 2022) for v13 depthwise+pointwise block pattern.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# AdaLN modulation helper (DiT canonical form)
# ---------------------------------------------------------------------------


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """Per-sample AdaLN modulation: x * (1 + scale) + shift.

    x:     (B, T, D)
    shift: (B, D)        -> broadcast to (B, 1, D) via unsqueeze(1)
    scale: (B, D)        -> broadcast to (B, 1, D)

    Returns (B, T, D). All T tokens get the same modulation within a sample.

    Source: facebookresearch/DiT@models.py:19-20 (verbatim).
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ---------------------------------------------------------------------------
# §4.3 — Input projection (separate per channel, summed)
# ---------------------------------------------------------------------------


class V12InputProjection(nn.Module):
    """Per-channel input projection: motion / z_int / obj_traj each get their
    own Linear(in_dim_i, d_model), summed.

    Aux projections (zint, obj, stage1_coarse) are zero-init'd so step-0
    output equals motion_proj(x_t) only.
    """

    def __init__(
        self,
        motion_dim: int,
        zint_dim: int,
        obj_traj_dim: int,
        d_model: int,
        stage1_coarse_dim: int = 0,
    ) -> None:
        super().__init__()
        self.stage1_coarse_dim = int(stage1_coarse_dim)
        self.motion_proj = nn.Linear(motion_dim, d_model)
        self.zint_proj = nn.Linear(zint_dim, d_model)
        self.obj_proj = nn.Linear(obj_traj_dim, d_model)
        if self.stage1_coarse_dim > 0:
            self.stage1_coarse_proj = nn.Linear(self.stage1_coarse_dim, d_model)
        else:
            self.stage1_coarse_proj = None
        # Init in `initialize_weights_v12` (called by AnchorDenoiser).

    def forward(
        self,
        x_t: Tensor,
        z_int: Tensor,
        obj_traj: Tensor,
        stage1_coarse: Tensor | None = None,
    ) -> Tensor:
        """All inputs (B, T, *). Output (B, T, d_model)."""
        h = (
            self.motion_proj(x_t)
            + self.zint_proj(z_int)
            + self.obj_proj(obj_traj)
        )
        if self.stage1_coarse_proj is not None:
            if stage1_coarse is None:
                raise KeyError(
                    "V12InputProjection.stage1_coarse_dim>0 but "
                    "stage1_coarse cond tensor was not provided. The trainer "
                    "must populate cond['stage1_coarse'] (B, T, stage1_coarse_dim)."
                )
            h = h + self.stage1_coarse_proj(stage1_coarse)
        return h


# ---------------------------------------------------------------------------
# §4.4 — Global condition vector for AdaLN (per-sample, InterGen pattern)
# ---------------------------------------------------------------------------


class GlobalCondSummary(nn.Module):
    """Produces a single per-sample (B, D) condition vector for AdaLN.

    After the R29 cleanup the only input is the diffusion timestep
    embedding (the plan-pool branch was removed alongside the interaction
    plan tokens). Per-frame information flows through V12InputProjection.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        # No parameters — kept as a Module for state_dict compatibility
        # with the historical name.
        self.plan_pool_mlp = None

    def forward(self, t_emb: Tensor) -> Tensor:
        """t_emb: (B, D). Returns (B, D)."""
        return t_emb


# ---------------------------------------------------------------------------
# §4.5 — ConditionedEncoderLayer (DiT 6-output AdaLN-Zero + PixArt cross-attn)
# ---------------------------------------------------------------------------


class ConditionedEncoderLayer(nn.Module):
    """DiT/InterGen-style encoder block with per-block plan cross-attn.

    Three (or four) sub-blocks per layer:
        (1) self-attn,        modulated by AdaLN-Zero (shift_msa, scale_msa, gate_msa)
        (1.5) [v13] temporal Conv1D residual, zero-gated (use_temporal_conv=True)
        (2) plan cross-attn,  UNMODULATED, output proj zero-init
        (3) MLP,              modulated by AdaLN-Zero (shift_mlp, scale_mlp, gate_mlp)

    AdaLN MLP per-block (DiT pattern): Linear(D, 6*D) producing
    (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp), each (B, D).

    At zero-init: all 6 outputs are 0
        -> shift = scale = 0   (modulate is identity)
        -> gate = 0            (residual update is killed)
        -> block is exact identity at step 0
    Combined with V12FinalLayer's zero-init linear, model predicts 0 at step 0.

    v13 addition (per analyses/2026-05-11_v13_dynhead_temporalconv_design.md §3.2):
    when use_temporal_conv=True, a TemporalConvResidual is applied to motion
    tokens (skipping the prepended init_pose prefix token at index 0). The
    residual gate is zero-init, so step-0 behavior is unchanged from v12.

    Source: facebookresearch/DiT@models.py:101-122 (DiTBlock, 6-output AdaLN)
            +  PixArt-alpha@diffusion/model/nets/PixArt.py:25-54 (cross-attn placement)
            +  InterGen@models/blocks.py + layers.py (motion-domain validation)
            +  Conformer / ConvNeXt for v13 local temporal coupling.
    """

    def __init__(
        self, d_model: int, n_heads: int, ff_mult: int = 4, dropout: float = 0.1,
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
        # Per-block AdaLN-Zero MLP (DiT models.py:110-113).
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model, bias=True),
        )

    def forward(
        self,
        x: Tensor,                      # (B, T, D)  motion tokens (including prepended init_pose_tok)
        c: Tensor,                      # (B, D)     global AdaLN condition vector (per sample)
    ) -> Tensor:
        """DiT-style block: AdaLN-Zero self-attn + AdaLN-Zero MLP."""
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )  # each (B, D)

        # (1) Self-attention with AdaLN-Zero
        h = modulate(self.norm1(x), shift_msa, scale_msa)            # (B, T, D)
        attn_out, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # (2) MLP with AdaLN-Zero
        h = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(h)
        return x


# ---------------------------------------------------------------------------
# §4.6 — Final readout layer with AdaLN-Zero
# ---------------------------------------------------------------------------


class V12FinalLayer(nn.Module):
    """Final layer: AdaLN-Zero shift+scale modulation + Linear to motion_dim.

    Source: facebookresearch/DiT@models.py:125-142 (FinalLayer), with the patch
    unflatten dropped (motion domain has no spatial unpatchify).

    At zero-init: shift = scale = 0 -> modulate is identity; linear weights = 0
    -> output is exactly 0. Combined with all ConditionedEncoderLayers also at
    identity, model predicts zero motion at step 0.
    """

    def __init__(self, d_model: int, motion_dim: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(d_model, motion_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model, bias=True),
        )

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        """
        x: (B, T, d_model)
        c: (B, d_model)
        Returns (B, T, motion_dim).
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)   # (B, D) each
        x = modulate(self.norm_final(x), shift, scale)             # (B, T, D)
        return self.linear(x)                                       # (B, T, motion_dim)


# ---------------------------------------------------------------------------
# §4.8 — Initialization recipe for v12 modules
# ---------------------------------------------------------------------------


def initialize_weights_v12(
    input_proj: V12InputProjection,
    blocks: nn.ModuleList,                  # of ConditionedEncoderLayer
    final_layer: V12FinalLayer,
    cond_summary: GlobalCondSummary | None = None,
) -> None:
    """Apply the v12 zero-init recipe.

    Per design doc §4.8 this guarantees a step-0 identity forward: all AdaLN
    gates = 0 -> blocks are identity; aux input proj = 0 -> only motion
    contributes at input; final layer = 0 -> output is exactly 0.

    Components NOT touched here: timestep_embed MLP, plan_encoder, text_proj,
    object_proj, end-of-encoder text_xattn / obj_xattn — those are initialized
    by AnchorDenoiser's own existing init.
    """
    # 1. Xavier-uniform on input_proj.motion_proj (motion MUST flow at init).
    nn.init.xavier_uniform_(input_proj.motion_proj.weight)
    nn.init.zeros_(input_proj.motion_proj.bias)

    # 2. Aux input projections zero-init (bandwidth allocation starts cold).
    aux_projs = [input_proj.zint_proj, input_proj.obj_proj]
    if getattr(input_proj, "stage1_coarse_proj", None) is not None:
        aux_projs.append(input_proj.stage1_coarse_proj)
    for proj in aux_projs:
        nn.init.zeros_(proj.weight)
        nn.init.zeros_(proj.bias)

    # 3. Per-block AdaLN-Zero: final Linear in each block's adaLN_modulation zeroed.
    for block in blocks:
        nn.init.zeros_(block.adaLN_modulation[-1].weight)
        nn.init.zeros_(block.adaLN_modulation[-1].bias)

    # 5. Final layer: AdaLN-Zero + zero-init linear -> step-0 output is 0.
    nn.init.zeros_(final_layer.adaLN_modulation[-1].weight)
    nn.init.zeros_(final_layer.adaLN_modulation[-1].bias)
    nn.init.zeros_(final_layer.linear.weight)
    nn.init.zeros_(final_layer.linear.bias)
