"""DiT/InterGen-style conditional transformer building blocks for PIANO v12 (+v13).

Per the v12 design doc (analyses/2026-05-11_v12_architecture_design_doc.md), this
module provides:

  modulate(x, shift, scale)            — DiT canonical AdaLN modulation helper
  V12InputProjection                   — separate per-channel input projections (motion / obj_traj), summed
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
    """Per-channel input projection: motion / obj_traj each get their
    own Linear(in_dim_i, d_model), summed.

    Aux projections (obj, stage1_coarse, init_pose) are zero-init'd so
    step-0 output equals motion_proj(x_t) only.
    """

    def __init__(
        self,
        motion_dim: int,
        obj_traj_dim: int,
        d_model: int,
        stage1_coarse_dim: int = 0,
        init_pose_dim: int = 0,
    ) -> None:
        super().__init__()
        self.stage1_coarse_dim = int(stage1_coarse_dim)
        self.init_pose_dim = int(init_pose_dim)
        self.motion_proj = nn.Linear(motion_dim, d_model)
        self.obj_proj = nn.Linear(obj_traj_dim, d_model)
        if self.stage1_coarse_dim > 0:
            self.stage1_coarse_proj = nn.Linear(self.stage1_coarse_dim, d_model)
        else:
            self.stage1_coarse_proj = None
        if self.init_pose_dim > 0:
            # init_pose is (B, init_pose_dim); broadcast across T inside
            # forward(). Per-token information about the starting pose
            # arrives at every timestep through this zero-init Linear.
            self.init_pose_proj = nn.Linear(self.init_pose_dim, d_model)
        else:
            self.init_pose_proj = None
        # Init in `initialize_weights_v12` (called by AnchorDenoiser).

    def forward(
        self,
        x_t: Tensor,
        obj_traj: Tensor,
        stage1_coarse: Tensor | None = None,
        init_pose: Tensor | None = None,
    ) -> Tensor:
        """All sequence inputs (B, T, *) → output (B, T, d_model).

        ``init_pose`` is (B, init_pose_dim) — a per-sample pose summary
        that is broadcast across the T axis before projection. (No per-
        frame variation; this is a frame-0 anchor signal.)
        """
        h = self.motion_proj(x_t) + self.obj_proj(obj_traj)
        if self.stage1_coarse_proj is not None:
            if stage1_coarse is None:
                raise KeyError(
                    "V12InputProjection.stage1_coarse_dim>0 but "
                    "stage1_coarse cond tensor was not provided. The trainer "
                    "must populate cond['stage1_coarse'] (B, T, stage1_coarse_dim)."
                )
            h = h + self.stage1_coarse_proj(stage1_coarse)
        if self.init_pose_proj is not None:
            if init_pose is None:
                raise KeyError(
                    "V12InputProjection.init_pose_dim>0 but init_pose "
                    "tensor was not provided. The trainer must populate "
                    "cond['init_pose'] (B, init_pose_dim)."
                )
            # Broadcast (B, D_in) → (B, T, D_in) before projection.
            T = x_t.shape[1]
            ip_btd = init_pose.unsqueeze(1).expand(-1, T, -1)
            h = h + self.init_pose_proj(ip_btd)
        return h


# ---------------------------------------------------------------------------
# §4.4 — Global condition vector for AdaLN (per-sample, InterGen pattern)
# ---------------------------------------------------------------------------


class GlobalCondSummary(nn.Module):
    """Produces a single per-sample (B, D) condition vector for AdaLN.

    Two modes:

    1. **R28 / pre-PB1 default** (``use_cond_summary_mlp=False``): the only
       input is the diffusion timestep embedding (R29 cleanup removed the
       plan-pool branch). ``forward(t_emb)`` returns ``t_emb`` unchanged.
       Per-frame information flows through ``V12InputProjection``.

    2. **PB1 path** (``use_cond_summary_mlp=True``): the parent passes a
       pooled R29 cond summary (B, D) alongside ``t_emb``; we project it
       through a SiLU + Linear(D, D) MLP whose final Linear is
       zero-initialised and add to ``t_emb`` before returning. Zero-init
       guarantees that step-0 output equals ``t_emb`` exactly — so the
       PB1 model with this branch ON has the same step-0 forward as a
       PB1 model with the branch OFF (which itself equals the A1
       baseline forward when both are trained from scratch with the
       same seed).

    State-dict compatibility: ``cond_summary_mlp`` is only created when
    ``use_cond_summary_mlp=True``. When False (the historical / R28
    default), the module has no parameters and loads cleanly from old
    ckpts that never saw this branch.
    """

    def __init__(self, d_model: int, *, use_cond_summary_mlp: bool = False) -> None:
        super().__init__()
        # Historical placeholder; kept for state_dict compatibility with
        # the pre-PB1 ckpts (which had ``plan_pool_mlp = None`` as a
        # registered attribute).
        self.plan_pool_mlp = None
        self.use_cond_summary_mlp = bool(use_cond_summary_mlp)
        if self.use_cond_summary_mlp:
            # 2-layer MLP. Final Linear zero-init -> the cond_summary
            # branch contributes 0 at init (PB1 invariant).
            self.cond_summary_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(d_model, d_model),
            )
            nn.init.zeros_(self.cond_summary_mlp[-1].weight)
            nn.init.zeros_(self.cond_summary_mlp[-1].bias)
        else:
            self.cond_summary_mlp = None

    def forward(
        self, t_emb: Tensor, cond_summary: Tensor | None = None,
    ) -> Tensor:
        """t_emb: (B, D). cond_summary: (B, D) or None. Returns (B, D)."""
        if cond_summary is None or self.cond_summary_mlp is None:
            return t_emb
        return t_emb + self.cond_summary_mlp(cond_summary)


# ---------------------------------------------------------------------------
# §4.5 — ConditionedEncoderLayer (DiT 6-output AdaLN-Zero + PixArt cross-attn)
# ---------------------------------------------------------------------------


class ConditionedEncoderLayer(nn.Module):
    """DiT encoder block: AdaLN-Zero self-attn + AdaLN-Zero MLP.

    Two sub-blocks per layer (DiT default):
        (1) self-attn,  modulated by AdaLN-Zero (shift_msa, scale_msa, gate_msa)
        (2) MLP,        modulated by AdaLN-Zero (shift_mlp, scale_mlp, gate_mlp)

    AdaLN MLP per-block (DiT pattern): Linear(D, 6*D) producing
    (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp), each (B, D).

    At zero-init: all 6 outputs are 0
        -> shift = scale = 0   (modulate is identity)
        -> gate = 0            (residual update is killed)
        -> block is exact identity at step 0
    Combined with V12FinalLayer's zero-init linear, model predicts 0 at step 0.

    Source: facebookresearch/DiT@models.py:101-122 (DiTBlock, 6-output AdaLN).

    R33 — optional per-block cross-attention sub-layer
    ---------------------------------------------------
    When ``enable_obj_xattn=True``, a 3rd sub-block is inserted between
    self-attn and MLP::

        (1) self-attn,   AdaLN-Zero on c
        (1.5) obj_xattn, AdaLN-Zero on c  — cross-attn over object_tokens
        (2) MLP,         AdaLN-Zero on c

    DiT-XL ICCV-2023 style (Peebles & Xie). A SECOND AdaLN MLP
    (``adaLN_modulation_xattn``) supplies (shift, scale, gate) for the
    cross-attn sub-block, separately initialised to zero so the
    cross-attn contribution starts at 0 (model is exact-identity at
    step 0). The original 6-output ``adaLN_modulation`` keeps its
    state-dict layout unchanged, so V0/V7/V8 ckpts trained without
    ``enable_obj_xattn`` still load cleanly into the same class.
    """

    def __init__(
        self, d_model: int, n_heads: int, ff_mult: int = 4, dropout: float = 0.1,
        *,
        enable_obj_xattn: bool = False,
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

        # R33 — per-block cross-attention sub-layer.
        self.enable_obj_xattn = bool(enable_obj_xattn)
        if self.enable_obj_xattn:
            self.norm_xattn = nn.LayerNorm(
                d_model, elementwise_affine=False, eps=1e-6,
            )
            self.obj_xattn = nn.MultiheadAttention(
                d_model, n_heads, dropout=dropout, batch_first=True,
            )
            # AdaLN-Zero modulation for the cross-attn sub-block
            # (separate from the 6-output one to preserve state_dict
            # of variants trained without obj_xattn).
            self.adaLN_modulation_xattn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(d_model, 3 * d_model, bias=True),
            )
            # The final Linear of the xattn AdaLN is zero-init'd in
            # initialize_weights_v12.
        else:
            self.norm_xattn = None
            self.obj_xattn = None
            self.adaLN_modulation_xattn = None

    def forward(
        self,
        x: Tensor,                      # (B, T, D)  motion tokens
        c: Tensor,                      # (B, D)     global AdaLN condition vector
        obj_kv: Tensor | None = None,    # (B, N_obj, D) object key/value (R33)
    ) -> Tensor:
        """DiT-style block. With ``enable_obj_xattn``, an AdaLN-Zero
        cross-attn sub-block over ``obj_kv`` is inserted between self-attn
        and MLP. ``obj_kv`` is required when ``enable_obj_xattn`` is True.
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )  # each (B, D)

        # (1) Self-attention with AdaLN-Zero
        h = modulate(self.norm1(x), shift_msa, scale_msa)            # (B, T, D)
        attn_out, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # (1.5) Cross-attention with AdaLN-Zero (R33 only)
        if self.enable_obj_xattn:
            if obj_kv is None:
                raise ValueError(
                    "ConditionedEncoderLayer.enable_obj_xattn=True but "
                    "obj_kv was not supplied to forward()."
                )
            shift_x, scale_x, gate_x = (
                self.adaLN_modulation_xattn(c).chunk(3, dim=-1)
            )
            h_x = modulate(self.norm_xattn(x), shift_x, scale_x)
            xattn_out, _ = self.obj_xattn(h_x, obj_kv, obj_kv, need_weights=False)
            x = x + gate_x.unsqueeze(1) * xattn_out

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
    aux_projs = [input_proj.obj_proj]
    if getattr(input_proj, "stage1_coarse_proj", None) is not None:
        aux_projs.append(input_proj.stage1_coarse_proj)
    if getattr(input_proj, "init_pose_proj", None) is not None:
        aux_projs.append(input_proj.init_pose_proj)
    for proj in aux_projs:
        nn.init.zeros_(proj.weight)
        nn.init.zeros_(proj.bias)

    # 3. Per-block AdaLN-Zero: final Linear in each block's adaLN_modulation zeroed.
    for block in blocks:
        nn.init.zeros_(block.adaLN_modulation[-1].weight)
        nn.init.zeros_(block.adaLN_modulation[-1].bias)
        # R33 — zero-init the optional per-block cross-attn AdaLN so its
        # contribution starts at 0 (step-0 identity invariant preserved).
        if getattr(block, "adaLN_modulation_xattn", None) is not None:
            nn.init.zeros_(block.adaLN_modulation_xattn[-1].weight)
            nn.init.zeros_(block.adaLN_modulation_xattn[-1].bias)

    # 5. Final layer: AdaLN-Zero + zero-init linear -> step-0 output is 0.
    nn.init.zeros_(final_layer.adaLN_modulation[-1].weight)
    nn.init.zeros_(final_layer.adaLN_modulation[-1].bias)
    nn.init.zeros_(final_layer.linear.weight)
    nn.init.zeros_(final_layer.linear.bias)
