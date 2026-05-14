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
    """Per-channel input projection: motion / z_int / obj_traj / plan_hint each
    get their own Linear(in_dim_i, d_model), summed.

    Decouples gradient through each channel (vs v11's single Linear(217, 512)
    which had the dense_channel_audit-confirmed bandwidth-bottleneck issue).

    Aux projections (zint, obj, hint) are zero-init'd so step-0 output equals
    motion_proj(x_t) only — preserving v11-like initial behavior. Aux channels
    activate as their projections learn non-zero weights.

    Bandwidth per channel (effective rank bounded by input dim):
      motion (135-d) -> up to 135 of 512 (26.4%)
      zint   (26-d)  -> up to 26 of 512  (5.1%)
      obj    (24-d)  -> up to 24 of 512  (4.7%)
      hint   (32-d)  -> up to 32 of 512  (6.3%)
    Total occupied: 217/512 (42.4%). Remaining 295/512 is filled by encoder
    self-attention higher-order interactions.
    """

    def __init__(
        self,
        motion_dim: int,
        zint_dim: int,
        obj_traj_dim: int,
        hint_dim: int,
        d_model: int,
        use_self_conditioning: bool = False,
        self_conditioning_zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.use_self_conditioning = bool(use_self_conditioning)
        self.self_conditioning_zero_init = bool(self_conditioning_zero_init)
        self.motion_proj = nn.Linear(motion_dim, d_model)
        self.zint_proj = nn.Linear(zint_dim, d_model)
        self.obj_proj = nn.Linear(obj_traj_dim, d_model)
        self.hint_proj = nn.Linear(hint_dim, d_model)
        if self.use_self_conditioning:
            self.self_cond_proj = nn.Linear(motion_dim, d_model)
            self.self_cond_gate = nn.Parameter(torch.zeros(()))
        else:
            self.self_cond_proj = None
            self.register_parameter("self_cond_gate", None)
        # Init in `initialize_weights_v12` (called by AnchorDenoiser).

    def forward(
        self,
        x_t: Tensor,
        z_int: Tensor,
        obj_traj: Tensor,
        plan_hint: Tensor,
        self_cond: Tensor | None = None,
    ) -> Tensor:
        """All inputs (B, T, *). Output (B, T, d_model)."""
        h = (
            self.motion_proj(x_t)
            + self.zint_proj(z_int)
            + self.obj_proj(obj_traj)
            + self.hint_proj(plan_hint)
        )
        if self.use_self_conditioning:
            if self_cond is None:
                self_cond = torch.zeros_like(x_t)
            h = h + self.self_cond_gate * self.self_cond_proj(self_cond)
        return h


# ---------------------------------------------------------------------------
# §4.4 — Global condition vector for AdaLN (per-sample, InterGen pattern)
# ---------------------------------------------------------------------------


class GlobalCondSummary(nn.Module):
    """Produces a single per-sample (B, D) condition vector for AdaLN.

    Two modes (`use_plan_pool` toggle):
      - True (v12 default, InterGen-style): c = t_emb + plan_pool_emb
        Where plan_pool_emb = MLP(masked_mean(plan_tokens)).
      - False (v12-A1, post 2026-05-11 cond audit): c = t_emb only.
        Forces ALL plan information to flow through per-layer plan cross-attn,
        which preserves per-anchor spatial detail (the masked-mean pool was
        destroying it — see analyses/2026-05-11_cond_diversity_audit.md §4.5).

    Per-frame information does NOT enter here in either mode — it flows
    through V12InputProjection's residual stream and per-layer plan cross-attn.
    """

    def __init__(self, d_model: int, use_plan_pool: bool = True) -> None:
        super().__init__()
        self.use_plan_pool = use_plan_pool
        if use_plan_pool:
            self.plan_pool_mlp = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
            )
        else:
            self.plan_pool_mlp = None

    def forward(
        self, t_emb: Tensor, plan_tokens: Tensor, plan_mask: Tensor,
    ) -> Tensor:
        """
        t_emb:       (B, D)
        plan_tokens: (B, K, D)  (ignored when use_plan_pool=False)
        plan_mask:   (B, K) bool — True at valid plan positions  (ignored when use_plan_pool=False)

        Returns (B, D).
        """
        if not self.use_plan_pool:
            return t_emb                                            # (B, D)
        mask = plan_mask.float().unsqueeze(-1)                      # (B, K, 1)
        denom = mask.sum(dim=1).clamp_min(1.0)                      # (B, 1)
        plan_pool = (plan_tokens * mask).sum(dim=1) / denom         # (B, D)
        plan_pool_emb = self.plan_pool_mlp(plan_pool)               # (B, D)
        return t_emb + plan_pool_emb                                # (B, D)


# ---------------------------------------------------------------------------
# §4.5 — ConditionedEncoderLayer (DiT 6-output AdaLN-Zero + PixArt cross-attn)
# ---------------------------------------------------------------------------


class TemporalConvResidual(nn.Module):
    """v13 P2 (per stageB_frozen_motion_diagnosis_and_fix_plan.md §9.1):
    depthwise temporal Conv1D residual with zero-init gate.

    Inserted between self-attn and plan cross-attn in each ConditionedEncoderLayer
    when use_temporal_conv=True. Adds local temporal inductive bias (±k//2 frames)
    on top of global self-attention. Applied only to motion tokens — the prepended
    init_pose prefix is excluded by the caller.

    Recipe (Conformer / ConvNeXt-style):
        LayerNorm  →  Conv1d depthwise (k=5, groups=D)  →  GELU  →
        Conv1d pointwise (1×1)  →  gate × output  +  residual

    At init: gate = 0 → branch is identity → preserves v12 step-0 behavior.

    Param count for d_model=512, k=5: 512*5 (dw) + 512*512 (pw) = 264_704 per
    layer ≈ 0.27 M. For 8 layers: 2.1 M total (~0.8 % of model). Cheaper than
    a full Conv1d (which would be 512*5*512 = 1.31 M per layer).
    """

    def __init__(self, dim: int, kernel_size: int = 5) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.dw_conv = nn.Conv1d(
            dim, dim, kernel_size,
            padding=kernel_size // 2, groups=dim,
        )
        self.act = nn.GELU(approximate="tanh")
        self.pw_conv = nn.Conv1d(dim, dim, kernel_size=1)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T, D). Returns (B, T, D) — same shape as input."""
        h = self.norm(x)
        h = h.transpose(1, 2)                       # (B, D, T)
        h = self.dw_conv(h)
        h = self.act(h)
        h = self.pw_conv(h)
        h = h.transpose(1, 2)                       # (B, T, D)
        return x + self.gate * h


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
        use_temporal_conv: bool = False, temporal_conv_kernel: int = 5,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.use_temporal_conv = use_temporal_conv
        if use_temporal_conv:
            self.temporal_conv = TemporalConvResidual(
                d_model, kernel_size=temporal_conv_kernel,
            )
        self.plan_xattn = nn.MultiheadAttention(
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
        plan_kv: Tensor,                # (B, K, D)  plan tokens
        plan_key_padding_mask: Tensor,  # (B, K) bool, True at padded positions (PyTorch convention)
        motion_token_start: int = 1,    # index where actual motion tokens begin (after prefix)
    ) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )  # each (B, D)

        # (1) Self-attention with AdaLN-Zero
        h = modulate(self.norm1(x), shift_msa, scale_msa)            # (B, T, D)
        attn_out, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # (1.5) [v13] Temporal Conv1D residual on motion tokens only —
        # the prefix init_pose token is a scene fact, not a motion frame.
        if self.use_temporal_conv:
            prefix = x[:, :motion_token_start]
            motion_tokens = x[:, motion_token_start:]
            motion_tokens = self.temporal_conv(motion_tokens)
            x = torch.cat([prefix, motion_tokens], dim=1)

        # (2) Plan cross-attention, UNMODULATED (PixArt placement)
        xattn_out, _ = self.plan_xattn(
            x, plan_kv, plan_kv,
            key_padding_mask=plan_key_padding_mask,
            need_weights=False,
        )
        x = x + xattn_out

        # (3) MLP with AdaLN-Zero
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
# v13 — Base + integrated-velocity-residual head (P1)
# ---------------------------------------------------------------------------


class V13DynamicsHead(nn.Module):
    """v13 P1: base + cumsum-integrated velocity residual final layer.

    Replaces V12FinalLayer when use_v13_dynhead=True. Two AdaLN-Zero branches
    consume the same h_motion: a base branch (low-frequency pose) and a delta
    branch whose output is integrated via cumsum to inject explicit temporal
    dynamics into the prediction path.

        x_base = base_linear(modulate(norm_base(h), shift_b, scale_b))
        delta  = delta_linear(modulate(norm_delta(h), shift_d, scale_d))
        delta[:, 0] = 0                     # no "prior-frame velocity" for frame 0
        x_dyn = cumsum(delta, dim=1)
        x_dyn = x_dyn - mean(x_dyn, dim=1)  # remove DC (absolute level lives in x_base)
        x0_pred = x_base + gamma * x_dyn

    At zero-init: both base_linear and delta_linear are zero → x0_pred = 0,
    matching v12 step-0 behavior exactly. γ is a learnable scalar initialized
    to 0.1 (or fixed if learnable_gamma=False). The optimizer picks how much
    dynamics flows through the delta branch.

    For motion_135 = [rot_6d_22, root_world_3], the cumsum integration is
    physically meaningful for root translation (Δposition ≈ velocity) and
    acts as a temporal-residual regularizer for the rot_6d block. Magnitude
    is controlled by γ (default 0.1) so the rot_6d component remains a small
    perturbation on x_base.

    Source: Stage B v13 design doc §3.1, with the integrator pattern adapted
    from MotionDiffuser (Jiang et al. CVPR 2023) trajectory-residual diffusion.
    """

    def __init__(
        self,
        d_model: int,
        motion_dim: int,
        gamma_init: float = 0.1,
        learnable_gamma: bool = True,
    ) -> None:
        super().__init__()
        # Base branch (same recipe as V12FinalLayer)
        self.norm_base = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.base_linear = nn.Linear(d_model, motion_dim, bias=True)
        self.adaLN_base = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model, bias=True),
        )
        # Delta branch (same recipe — independent norm + AdaLN MLP + linear)
        self.norm_delta = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.delta_linear = nn.Linear(d_model, motion_dim, bias=True)
        self.adaLN_delta = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model, bias=True),
        )
        if learnable_gamma:
            self.gamma = nn.Parameter(torch.tensor(gamma_init, dtype=torch.float32))
        else:
            self.register_buffer(
                "gamma", torch.tensor(gamma_init, dtype=torch.float32),
            )

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        """x: (B, T, d_model), c: (B, d_model). Returns (B, T, motion_dim)."""
        # Base branch
        shift_b, scale_b = self.adaLN_base(c).chunk(2, dim=-1)
        x_base = self.base_linear(modulate(self.norm_base(x), shift_b, scale_b))

        # Delta branch
        shift_d, scale_d = self.adaLN_delta(c).chunk(2, dim=-1)
        delta = self.delta_linear(modulate(self.norm_delta(x), shift_d, scale_d))

        # Zero out first-frame delta. Use masking (non-in-place) for clean autograd.
        mask = torch.ones_like(delta)
        mask[:, 0] = 0.0
        delta = delta * mask

        # Cumulative-integrate + mean-center (remove DC; absolute level is in x_base)
        x_dyn = torch.cumsum(delta, dim=1)
        x_dyn = x_dyn - x_dyn.mean(dim=1, keepdim=True)

        return x_base + self.gamma * x_dyn


# ---------------------------------------------------------------------------
# §4.8 — Initialization recipe for v12 / v13 modules
# ---------------------------------------------------------------------------


def initialize_weights_v12(
    input_proj: V12InputProjection,
    blocks: nn.ModuleList,                  # of ConditionedEncoderLayer
    final_layer: V12FinalLayer | "V13DynamicsHead",
    cond_summary: GlobalCondSummary | None = None,
) -> None:
    """Apply the v12 / v13 zero-init recipe.

    Per design doc §4.8 (v12) + v13 design doc §3.1, this guarantees a step-0
    identity forward: all AdaLN gates = 0 -> blocks are identity; aux input
    proj = 0 -> only motion contributes at input; final layer = 0 -> output
    is exactly 0.

    Handles both V12FinalLayer (single head) and V13DynamicsHead (base + delta).
    The temporal-conv branch inside ConditionedEncoderLayer is zero-gated by
    construction at TemporalConvResidual.__init__, so no extra init needed here.

    Components NOT touched here: timestep_embed MLP, plan_encoder, text_proj,
    object_proj, end-of-encoder text_xattn / obj_xattn — those are initialized
    by AnchorDenoiser's own existing init (matches v11 behavior).
    """
    # 1. Xavier-uniform on input_proj.motion_proj (motion MUST flow at init).
    nn.init.xavier_uniform_(input_proj.motion_proj.weight)
    nn.init.zeros_(input_proj.motion_proj.bias)

    # 2. Aux input projections zero-init (bandwidth allocation starts cold).
    for proj in (input_proj.zint_proj, input_proj.obj_proj, input_proj.hint_proj):
        nn.init.zeros_(proj.weight)
        nn.init.zeros_(proj.bias)
    if getattr(input_proj, "self_cond_proj", None) is not None:
        # Option B for v22 self-conditioning: initialize the projection
        # normally and zero only the scalar gate. Zeroing both projection and
        # gate would make the branch dead at initialization because neither
        # side receives a useful gradient.
        nn.init.xavier_uniform_(input_proj.self_cond_proj.weight)
        nn.init.zeros_(input_proj.self_cond_proj.bias)
        if getattr(input_proj, "self_cond_gate", None) is not None:
            nn.init.zeros_(input_proj.self_cond_gate)

    # 3. Per-block AdaLN-Zero: final Linear in each block's adaLN_modulation zeroed.
    for block in blocks:
        nn.init.zeros_(block.adaLN_modulation[-1].weight)
        nn.init.zeros_(block.adaLN_modulation[-1].bias)
        # 4. Per-block cross-attn output projection zero-init.
        nn.init.zeros_(block.plan_xattn.out_proj.weight)
        nn.init.zeros_(block.plan_xattn.out_proj.bias)
        # 4.5 [v13] Temporal-conv branch is already zero-gated at construction
        # (gate = nn.Parameter(zeros(1))). No extra init needed.

    # 5. Final layer: AdaLN-Zero + zero-init linear -> step-0 output is 0.
    if isinstance(final_layer, V13DynamicsHead):
        # v13: zero both base and delta branches' AdaLN modulation + Linear.
        nn.init.zeros_(final_layer.adaLN_base[-1].weight)
        nn.init.zeros_(final_layer.adaLN_base[-1].bias)
        nn.init.zeros_(final_layer.base_linear.weight)
        nn.init.zeros_(final_layer.base_linear.bias)
        nn.init.zeros_(final_layer.adaLN_delta[-1].weight)
        nn.init.zeros_(final_layer.adaLN_delta[-1].bias)
        nn.init.zeros_(final_layer.delta_linear.weight)
        nn.init.zeros_(final_layer.delta_linear.bias)
        # gamma remains at its init value (default 0.1).
    else:
        nn.init.zeros_(final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(final_layer.linear.weight)
        nn.init.zeros_(final_layer.linear.bias)

    # 6. cond_summary MLP: standard small-normal init (NOT zero — c needs to be
    # non-trivial from step 0; AdaLN zero-init handles the "no condition at start"
    # guarantee via the adaLN_modulation, not via cond_summary).
    # Skipped entirely when use_plan_pool=False (cond_summary.plan_pool_mlp is None).
    if cond_summary is not None and cond_summary.plan_pool_mlp is not None:
        for m in cond_summary.plan_pool_mlp:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
