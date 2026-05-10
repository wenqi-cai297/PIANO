"""PIANO-AnchorDiff: anchor-conditioned continuous motion diffusion.

OMOMO-style anchor conditioning (z_int as primary per-frame channel)
on top of an MDM-style transformer encoder denoiser, trained with
classifier-free guidance dropout. Operates on HumanML3D motion_263.

Prediction parameterisation: **x₀-prediction** (sample-prediction).
MDM, OMOMO, HOI-Dyn all use x₀; ε-prediction would force the anchor
consistency loss through a 1/√ᾱ_t derivation that explodes at high
noise levels. See ``analyses/2026-05-08_diffusion_prediction_target_review.md``
for the full rationale and source citations.

Design source of truth:
    analyses/2026-05-08_piano_anchordiff_design.md

Self-contained for M0. We may refactor against
``src/external/motion-diffusion-model`` once the smoke test passes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ============================================================================
# Diffusion math (DDPM, cosine schedule, ε-prediction)
# ============================================================================


def cosine_beta_schedule(num_steps: int, s: float = 0.008) -> Tensor:
    """Nichol & Dhariwal cosine schedule. Same as MDM/OMOMO/CHOIS use."""
    steps = num_steps + 1
    x = torch.linspace(0, num_steps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0, 0.999).float()


def _extract(a: Tensor, t: Tensor, x_shape: torch.Size) -> Tensor:
    out = a.gather(0, t)
    return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))


@dataclass(slots=True)
class DiffusionConfig:
    num_steps: int = 1000
    schedule: str = "cosine"
    # "x0" = denoiser predicts clean x_0 (MDM/OMOMO style).
    # "v"  = denoiser predicts v = sqrt(ᾱ)·ε - sqrt(1-ᾱ)·x_0 (Salimans & Ho 2022).
    #        Hybrid noise+clean target; recommended by Back-to-Basics for
    #        better dynamics under rotation reps. x_0 is recovered via
    #        x_0 = sqrt(ᾱ)·x_t - sqrt(1-ᾱ)·v inside the sampler / training.
    prediction_target: str = "x0"


class GaussianDiffusion(nn.Module):
    """ε-prediction DDPM with cosine β-schedule.

    All buffers are float32 and registered, so they move with the module
    under Accelerate / .to(device).
    """

    def __init__(self, cfg: DiffusionConfig) -> None:
        super().__init__()
        self.num_steps = cfg.num_steps
        self.prediction_target = cfg.prediction_target
        if cfg.prediction_target not in ("x0", "v"):
            raise ValueError(f"prediction_target must be 'x0' or 'v', got {cfg.prediction_target!r}")
        if cfg.schedule == "cosine":
            betas = cosine_beta_schedule(cfg.num_steps)
        else:
            raise NotImplementedError(f"schedule={cfg.schedule!r}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0))
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )

    def q_sample(self, x_start: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        """Forward diffusion: x_t = √ᾱ x_0 + √(1-ᾱ) ε."""
        return (
            _extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def predict_x0_from_eps(self, x_t: Tensor, t: Tensor, eps: Tensor) -> Tensor:
        return (
            _extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def predict_x0_from_v(self, x_t: Tensor, t: Tensor, v: Tensor) -> Tensor:
        """Recover x_0 from v-prediction:  x_0 = sqrt(ᾱ)·x_t - sqrt(1-ᾱ)·v."""
        return (
            _extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - _extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def v_target(self, x_start: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        """v_target = sqrt(ᾱ)·ε - sqrt(1-ᾱ)·x_0 (Salimans & Ho 2022)."""
        return (
            _extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise
            - _extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def posterior_mean_from_x0(
        self, x_start: Tensor, x_t: Tensor, t: Tensor,
    ) -> Tensor:
        """Closed-form posterior mean μ(x_{t-1} | x_t, x_0) from DDPM:

            μ = (√ᾱ_{t-1} β_t / (1 - ᾱ_t)) x_0
              + (√α_t (1 - ᾱ_{t-1}) / (1 - ᾱ_t)) x_t
        """
        coef_x0 = (
            self.alphas_cumprod_prev.sqrt() * self.betas
            / (1 - self.alphas_cumprod)
        )
        coef_xt = (
            (1 - self.alphas_cumprod_prev) * (1 - self.betas).sqrt()
            / (1 - self.alphas_cumprod)
        )
        return (
            _extract(coef_x0, t, x_t.shape) * x_start
            + _extract(coef_xt, t, x_t.shape) * x_t
        )

    @torch.no_grad()
    def p_sample_loop(
        self,
        denoiser: "AnchorDenoiser",
        shape: tuple[int, ...],
        cond: dict,
        cfg_scale: float = 1.0,
        device: torch.device | None = None,
        replacement: str = "none",
        output_skip: bool = False,
        sampler: str = "ddpm",
    ) -> Tensor:
        """Reverse-diffusion sampling with optional classifier-free guidance,
        operating on x₀-prediction outputs.

        ``cond`` follows the same dict spec as ``AnchorDenoiser.forward``.
        ``cfg_scale > 1`` enables CFG: x_0 = x_0_uncond + s·(x_0_cond - x_0_uncond).

        ``replacement`` options (per claude_code_v9_condmdi_diagnostic_next_steps.md §8.2):
            "none"        : default DDPM trajectory.
            "x0"          : after each network call, replace predicted x_0
                            at observed frames with cond_motion (so the
                            posterior mean uses GT keyframe at obs frames).
            "x_t"         : at each step, replace x_t at observed frames
                            with sqrt(α̅_t)·cond_motion + sqrt(1-α̅_t)·noise
                            BEFORE the network call. Forces the trajectory
                            through GT keyframes at every step.

        ``sampler`` options (per claude_code_v10_after_fkposfix_strategy.md §6 —
        the residual-jitter diagnostic):
            "ddpm"        : default ancestral DDPM. Stochastic — adds posterior
                            variance noise at every step except t=0.
            "ddim_eta0"   : standard DDIM update (Song et al. 2021) with η=0.
                            Deterministic given a fixed initial x_T. Tests whether
                            visible per-frame jitter comes from sampler stochasticity
                            or from the learned model.
            "ddpm_det"    : same DDPM posterior mean as "ddpm" but skip the
                            noise injection at t > 0. Cheaper deterministic
                            variant; not exactly DDIM but isolates the same
                            "is the noise injection responsible" question.
        """
        device = device or self.betas.device
        x = torch.randn(shape, device=device)

        # If replacement OR output_skip is enabled, extract cond_motion
        # and obs_mask from the CondMDI cond_motion_input.
        cond_motion = obs_mask = None
        if replacement != "none" or output_skip:
            if "cond_motion_input" not in cond:
                raise ValueError(
                    f"replacement={replacement!r}, output_skip={output_skip} "
                    "requires cond['cond_motion_input']; model was not "
                    "trained with the CondMDI inpainting channel."
                )
            cmi = cond["cond_motion_input"]                         # (B, T, motion_dim+1)
            motion_dim = shape[-1]
            cond_motion = cmi[..., :motion_dim]                     # (B, T, motion_dim)
            obs_mask = cmi[..., motion_dim:motion_dim + 1]          # (B, T, 1)

        for t_int in reversed(range(self.num_steps)):
            t = torch.full((shape[0],), t_int, device=device, dtype=torch.long)

            # x_t replacement: BEFORE the network call at each step. Forces
            # the trajectory's noisy state at observed frames to match the
            # forward-diffusion of cond_motion. This is the standard
            # "RePaint"-style inpainting trick (Lugmayr et al. CVPR 2022).
            if replacement == "x_t" and cond_motion is not None:
                noise_obs = torch.randn_like(x)
                x_obs = (
                    _extract(self.sqrt_alphas_cumprod, t, x.shape) * cond_motion
                    + _extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * noise_obs
                )
                x = obs_mask * x_obs + (1.0 - obs_mask) * x

            pred_cond = denoiser(x, t, cond, cond_drop_mask=None)
            if cfg_scale != 1.0:
                drop = torch.ones(shape[0], dtype=torch.bool, device=device)
                pred_uncond = denoiser(x, t, cond, cond_drop_mask=drop)
            else:
                pred_uncond = None

            # Convert raw network output → x0 (CFG blend always done in x0-space)
            if self.prediction_target == "v":
                x0_cond = self.predict_x0_from_v(x, t, pred_cond)
                x0_uncond = (
                    self.predict_x0_from_v(x, t, pred_uncond)
                    if pred_uncond is not None else None
                )
            else:
                x0_cond = pred_cond
                x0_uncond = pred_uncond

            if x0_uncond is not None:
                x0 = x0_uncond + cfg_scale * (x0_cond - x0_uncond)
            else:
                x0 = x0_cond

            # v9_4 hard observation output skip — applied at every DDPM
            # step BEFORE the posterior mean. Identical to the training-
            # time skip in MotionAnchorDiff.training_step, so the model
            # the sampler operates on is the same model the loss saw.
            if output_skip and cond_motion is not None:
                x0 = obs_mask * cond_motion + (1.0 - obs_mask) * x0

            # Legacy v9_3 sampler-only x0 replacement (kept for ablation
            # comparison). When v9_4 output_skip is on, this is a no-op
            # because x0 already matches cond_motion at observed frames.
            if replacement == "x0" and cond_motion is not None:
                x0 = obs_mask * cond_motion + (1.0 - obs_mask) * x0

            if sampler == "ddim_eta0":
                # DDIM update with η = 0 (deterministic) — Song et al.,
                # "Denoising Diffusion Implicit Models" (ICLR 2021),
                # arXiv:2010.02502 Eq. 12.
                #
                #   x_{t-1} = √ᾱ_{t-1} · x0 + √(1 - ᾱ_{t-1}) · ε
                # where ε is recovered from x_t and x0:
                #   ε = (x_t - √ᾱ_t · x0) / √(1 - ᾱ_t)
                if t_int == 0:
                    x = x0
                else:
                    eps = (
                        x - _extract(self.sqrt_alphas_cumprod, t, x.shape) * x0
                    ) / _extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
                    t_prev = (t - 1).clamp(min=0)
                    sqrt_a_prev = _extract(
                        self.sqrt_alphas_cumprod, t_prev, x.shape,
                    )
                    sqrt_om_a_prev = _extract(
                        self.sqrt_one_minus_alphas_cumprod, t_prev, x.shape,
                    )
                    x = sqrt_a_prev * x0 + sqrt_om_a_prev * eps
            else:
                mean = self.posterior_mean_from_x0(x0, x, t)
                if sampler == "ddpm_det" or t_int == 0:
                    x = mean
                else:
                    noise = torch.randn_like(x)
                    log_var = _extract(
                        self.posterior_log_variance_clipped, t, x.shape,
                    )
                    x = mean + (0.5 * log_var).exp() * noise
        return x


# ============================================================================
# v9_4 hard-observation helpers
# ============================================================================
#
# Per claude_code_v9_4_hard_observation_execution_plan.md §6.2: split the
# CondMDI side-channel into (cond_motion, obs_mask) and apply a hard
# observed-frame skip on the network output. Used both in training_step
# (so loss + downstream geometric losses see the skipped output) and in
# p_sample_loop (so posterior mean uses the skipped output).


def _split_cond_motion_input(
    cond: dict, motion_dim: int,
) -> tuple[Tensor | None, Tensor | None]:
    """Split cond["cond_motion_input"] of shape (B, T, motion_dim+1) into
    (cond_motion (B,T,motion_dim), obs_mask (B,T,1)). Returns (None, None)
    if the channel is absent."""
    cmi = cond.get("cond_motion_input", None)
    if cmi is None:
        return None, None
    cond_motion = cmi[..., :motion_dim]
    obs_mask = cmi[..., motion_dim:motion_dim + 1]
    return cond_motion, obs_mask


def _apply_observed_x0_skip(
    x0_raw: Tensor, cond: dict, motion_dim: int, enabled: bool,
) -> Tensor:
    """If `enabled`, return x0 = obs_mask·cond_motion + (1-obs_mask)·x0_raw.
    Else return x0_raw unchanged. The skip is a hard injection at observed
    frames — the model is no longer asked to learn to reproduce known
    keyframes, only to interpolate around them."""
    if not enabled:
        return x0_raw
    cond_motion, obs_mask = _split_cond_motion_input(cond, motion_dim)
    if cond_motion is None:
        return x0_raw
    return obs_mask * cond_motion + (1.0 - obs_mask) * x0_raw


# ============================================================================
# Embeddings + small modules
# ============================================================================


class SinusoidalTimestepEmbed(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float()[:, None] * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:  # (B, T, D)
        return x + self.pe[: x.shape[1]]


# ============================================================================
# Anchor (z_int) conditioning channel
# ============================================================================


@dataclass(slots=True)
class ZIntDims:
    """Per-frame channels of the z_int conditioning signal.

    Default lays out 5 + 15 + 3 + 3 = 26 dims:
        contact_state            : 5     (sigmoid)
        contact_target_xyz_local : 5*3   (object-local coords)
        phase                    : 3     (softmax)
        support                  : 3     (softmax, hand_support collapsed)
    """

    num_parts: int = 5
    phase_classes: int = 3
    support_classes: int = 3

    @property
    def total(self) -> int:
        return self.num_parts + self.num_parts * 3 + self.phase_classes + self.support_classes


def pack_z_int(
    contact_state: Tensor,
    contact_target_xyz: Tensor,
    phase_logits: Tensor,
    support_logits: Tensor,
    dims: ZIntDims,
) -> Tensor:
    """Pack Stage A outputs (or GT labels) into a flat (B, T, ZIntDims.total) tensor.

    Inputs may be the raw GT one-hot/integer labels or sigmoid/softmax
    outputs of Stage A — both shapes are supported as long as the last
    dims match ``dims``.
    """
    B, T, _ = contact_state.shape
    cs = contact_state.float().view(B, T, dims.num_parts)
    cx = contact_target_xyz.float().view(B, T, dims.num_parts * 3)
    ph = phase_logits.float().view(B, T, dims.phase_classes)
    sp = support_logits.float().view(B, T, dims.support_classes)
    return torch.cat([cs, cx, ph, sp], dim=-1)


# ============================================================================
# Denoiser
# ============================================================================


@dataclass(slots=True)
class AnchorDenoiserConfig:
    motion_dim: int = 263
    z_int: ZIntDims = ZIntDims()
    object_traj_dim: int = 9     # 3 (pos) + 6 (rot6d)
    init_pose_dim: int = 22 * 3  # SMPL-22 joints
    text_dim: int = 512          # CLIP per-token feature dim
    object_token_dim: int = 256  # Stage A's object encoder output dim
    object_num_tokens: int = 128
    # CondMDI keyframe-inpainting channel: per-frame clean motion (zeroed
    # at non-observed frames) + 1-D observation mask, concatenated to the
    # noisy motion before the input projection. 0 = disabled.
    # For v9 with motion_dim=135: cond_motion_dim = 135 + 1 = 136.
    cond_motion_dim: int = 0
    # v9_4 hard-observation flags (per claude_code_v9_4_hard_observation_execution_plan.md §4):
    #   cond_motion_output_skip : if True, replace x0_pred at observed frames
    #       with cond_motion BOTH at training and sampling. Removes the
    #       "model must learn to copy" job entirely; the model's capacity is
    #       routed to unobserved-frame interpolation only.
    #   cfg_drop_cond_motion    : default False. When False, CFG dropout
    #       does NOT drop the CondMDI channel — so cfg=0/1/3 sweeps measure
    #       semantic/object/text condition strength only, not keyframe
    #       condition strength. Set True only for explicit condition-
    #       sensitivity ablations.
    #   cond_motion_xt_inject   : sampler-side hint kept for v9_3-style
    #       observed-frame x_t replacement; reported here for completeness
    #       but actually consumed via the sample(replacement=...) arg.
    cond_motion_output_skip: bool = False
    cfg_drop_cond_motion: bool = False
    cond_motion_xt_inject: bool = False

    # v10 InteractionPlan tokens (per
    # analyses/piano_interaction_plan_pipeline_reframe_for_claude_code.md
    # §5). When ``use_interaction_plan=True``, the denoiser instantiates
    # an ``InteractionPlanEncoder`` that turns the batched plan dict into
    # plan tokens, and adds a cross-attention block from motion tokens to
    # plan tokens (§5.4). ``plan_use_context_hint`` adds a per-frame
    # relative-temporal hint that is concatenated to the per-frame input
    # projection (§5.5) — sits alongside cross-attention, doesn't replace
    # it. ``cfg_drop_plan`` (default False) controls whether the plan
    # branch participates in CFG dropout for an explicit plan-condition
    # ablation; default is False so cfg=0/1/3 sweeps measure semantic
    # condition strength only.
    use_interaction_plan: bool = False
    plan_k_max: int = 12
    plan_s_max: int = 12
    plan_num_anchor_types: int = 5
    plan_num_parts: int = 5
    plan_use_segment_tokens: bool = False
    plan_use_context_hint: bool = True
    plan_d_hint: int = 32
    plan_d_time_embed: int = 64
    cfg_drop_plan: bool = False
    # v11 (per claude_code_v10_after_fkposfix_strategy.md §7):
    #   plan_per_part_tokens: emit one cross-attention token per active
    #     (anchor, body_part) pair instead of one per anchor.
    #   plan_context_hint_mode: "time_only" (v10 default) | "off" |
    #     "target_aware" (v11 hint with target_world + dominant part).
    plan_per_part_tokens: bool = False
    plan_context_hint_mode: str = "time_only"

    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 4
    ff_mult: int = 4
    dropout: float = 0.1
    max_seq_length: int = 256


class AnchorDenoiser(nn.Module):
    """x₀-prediction transformer-encoder denoiser with anchor conditioning.

    Output is the predicted clean motion ``x_0``, not ε. This matches MDM /
    OMOMO / HOI-Dyn and lets geometric losses (anchor consistency, future
    foot contact) attach directly to the network output without going
    through a 1/√ᾱ_t derivation. See
    ``analyses/2026-05-08_diffusion_prediction_target_review.md``.

    Per-frame conditioning (concatenated to the projected motion before
    transformer):
        - z_int (anchor)        : ZIntDims.total ≈ 26 dims
        - object_world_traj     : 9 dims (pos + rot6d)
        - cond_motion_input     : (motion_dim + 1) dims, optional (CondMDI
                                  keyframe-inpainting channel: clean motion
                                  zeroed at non-observed frames + 1-D mask)
    Sequence-level conditioning (cross-attention K/V):
        - text                  : CLIP per-token (B, 77, text_dim)
        - object_pc tokens      : (B, 128, object_token_dim)
    Special tokens prepended to the sequence:
        - timestep token (sinusoidal embedding -> Linear projection)
        - init_pose token (Linear projection of frame-0 SMPL-22 joints)

    CFG dropout is applied independently per conditioning channel via
    ``cond_drop_mask`` (a (B,) bool tensor): if True for a sample,
    z_int + object_traj + text + object_pc tokens are masked to a learned
    null embedding for that sample. ``init_pose`` is preserved (it is a
    deterministic scene fact).
    """

    def __init__(self, cfg: AnchorDenoiserConfig) -> None:
        super().__init__()
        self.cfg = cfg

        per_frame_in = (
            cfg.motion_dim
            + cfg.cond_motion_dim
            + cfg.z_int.total
            + cfg.object_traj_dim
        )
        # v10 plan-context hint widens the per-frame input projection. The
        # hint is encoded by the InteractionPlanEncoder (§5.5) and gives
        # the motion-token branch explicit "where am I relative to the
        # nearest anchor" features without replacing cross-attention.
        if cfg.use_interaction_plan and cfg.plan_use_context_hint:
            per_frame_in += cfg.plan_d_hint
        self.in_proj = nn.Linear(per_frame_in, cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.motion_dim)

        self.time_embed = nn.Sequential(
            SinusoidalTimestepEmbed(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.pose_proj = nn.Linear(cfg.init_pose_dim, cfg.d_model)
        self.text_proj = nn.Linear(cfg.text_dim, cfg.d_model)
        self.object_proj = nn.Linear(cfg.object_token_dim, cfg.d_model)

        # Learned null embeddings for CFG dropout (one per channel).
        self.null_zint = nn.Parameter(torch.zeros(cfg.z_int.total))
        self.null_obj_traj = nn.Parameter(torch.zeros(cfg.object_traj_dim))
        self.null_text = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.null_obj_tokens = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        # v9_4 §5.2: null for cond_motion_input, used only when
        # cfg_drop_cond_motion=True (explicit keyframe-condition ablation).
        # Stored unconditionally for state_dict stability; ignored otherwise.
        if cfg.cond_motion_dim > 0:
            self.null_cond_motion_input = nn.Parameter(
                torch.zeros(cfg.cond_motion_dim)
            )
        else:
            self.register_parameter("null_cond_motion_input", None)

        self.pos_enc = PositionalEncoding(cfg.d_model, max_len=cfg.max_seq_length + 2)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * cfg.ff_mult,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)

        # Cross-attn for text / object tokens — applied AFTER encoder.
        # Simpler than alternating self/cross like MDM.trans_dec; still
        # gives sequence-level conditioning a path into per-frame outputs.
        self.text_xattn = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True
        )
        self.text_norm = nn.LayerNorm(cfg.d_model)
        self.obj_xattn = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True
        )
        self.obj_norm = nn.LayerNorm(cfg.d_model)

        # v10 InteractionPlan branch. Lazy-import the encoder + xattn
        # block to avoid a circular import (interaction_plan_encoder
        # imports from data.interaction_plan_compiler, both pure-python
        # imports, no actual cycle but keeping the heavy module
        # off the always-loaded path keeps `from piano.models.motion_anchordiff
        # import ...` cheap when plan tokens are off).
        if cfg.use_interaction_plan:
            from piano.models.interaction_plan_encoder import (
                InteractionPlanEncoder,
                InteractionPlanEncoderConfig,
                PlanCrossAttentionBlock,
            )
            plan_enc_cfg = InteractionPlanEncoderConfig(
                d_model=cfg.d_model,
                num_parts=cfg.plan_num_parts,
                num_anchor_types=cfg.plan_num_anchor_types,
                num_phase_classes=cfg.z_int.phase_classes,
                num_support_classes=cfg.z_int.support_classes,
                k_max=cfg.plan_k_max,
                s_max=cfg.plan_s_max,
                use_segment_tokens=cfg.plan_use_segment_tokens,
                use_plan_context_hint=cfg.plan_use_context_hint,
                d_hint=cfg.plan_d_hint,
                d_time_embed=cfg.plan_d_time_embed,
                per_part_tokens=cfg.plan_per_part_tokens,
                context_hint_mode=cfg.plan_context_hint_mode,
            )
            self.plan_encoder = InteractionPlanEncoder(plan_enc_cfg)
            self.plan_xattn = PlanCrossAttentionBlock(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                ff_mult=cfg.ff_mult,
                dropout=cfg.dropout,
            )
            # Null plan tokens for explicit plan-CFG ablation. Stored
            # unconditionally (size 1×D) for state_dict stability; only
            # consumed when ``cfg_drop_plan=True``. The value broadcasts
            # to every plan slot when the dropout mask fires for a sample.
            self.null_plan_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
            # Null hint vector, used when CFG drops the plan branch and
            # ``plan_use_context_hint=True``.
            if cfg.plan_use_context_hint:
                self.null_plan_hint = nn.Parameter(torch.zeros(1, 1, cfg.plan_d_hint))
        else:
            self.plan_encoder = None
            self.plan_xattn = None

    @staticmethod
    def _broadcast_drop(mask: Tensor | None, x: Tensor, null_value: Tensor) -> Tensor:
        """If ``mask[b]`` is True, replace ``x[b]`` with the null embedding."""
        if mask is None:
            return x
        # null_value broadcasting handles per-frame channels (1D) and
        # per-token sequences (3D).
        null = null_value.expand_as(x)
        m = mask.view(-1, *([1] * (x.dim() - 1)))
        return torch.where(m, null, x)

    def forward(
        self,
        x_t: Tensor,          # (B, T, motion_dim) noisy motion
        t: Tensor,            # (B,) long, diffusion step
        cond: dict,           # see top-level docstring
        cond_drop_mask: Tensor | None,  # (B,) bool — True = drop conditioning
    ) -> Tensor:
        B, T, _ = x_t.shape
        cfg = self.cfg

        z_int: Tensor = cond["z_int"]                  # (B, T, zint_total)
        obj_traj: Tensor = cond["object_world_traj"]   # (B, T, 9)
        init_pose: Tensor = cond["init_pose"]          # (B, init_pose_dim)
        text_tok: Tensor = cond["text"]                # (B, L_text, text_dim)
        obj_tok: Tensor = cond["object_tokens"]        # (B, N_obj, object_token_dim)

        # --- CFG drop: replace conditioning channels with null embeddings ---
        z_int_eff = self._broadcast_drop(cond_drop_mask, z_int, self.null_zint)
        obj_traj_eff = self._broadcast_drop(cond_drop_mask, obj_traj, self.null_obj_traj)

        # --- Per-frame projection: concat motion + (cond_motion) + z_int + object_traj ---
        # CondMDI inpainting channel: caller has already concatenated
        # [clean_motion_at_observed_frames, observation_mask] along the
        # feature dim.
        #
        # v9_4 §5.2: by default (cfg_drop_cond_motion=False) cond_motion
        # is NOT dropped under CFG — so cfg=0/1/3 sweeps measure semantic
        # condition strength only, not keyframe condition strength. To do
        # an explicit condition-sensitivity ablation, set
        # cfg_drop_cond_motion=True so the unconditional branch sees the
        # null cond_motion embedding.
        if cfg.cond_motion_dim > 0:
            cond_motion: Tensor = cond["cond_motion_input"]   # (B, T, cond_motion_dim)
            if cfg.cfg_drop_cond_motion and self.null_cond_motion_input is not None:
                cond_motion = self._broadcast_drop(
                    cond_drop_mask, cond_motion, self.null_cond_motion_input,
                )
            per_frame = torch.cat(
                [x_t, cond_motion, z_int_eff, obj_traj_eff], dim=-1
            )
        else:
            per_frame = torch.cat([x_t, z_int_eff, obj_traj_eff], dim=-1)

        # v10 plan tokens — encode the InteractionPlan and compute the
        # per-frame hint that goes into the input projection. The plan
        # token branch participates in CFG dropout only when
        # ``cfg_drop_plan=True`` (ablation flag); otherwise the plan is
        # carried through both conditional and unconditional CFG branches
        # so cfg sweeps measure semantic strength only.
        plan_tokens = None
        plan_mask = None
        if cfg.use_interaction_plan and self.plan_encoder is not None:
            plan_dict = cond.get("interaction_plan", None)
            if plan_dict is None:
                raise KeyError(
                    "use_interaction_plan=True requires "
                    "cond['interaction_plan'] dict; trainer must pass "
                    "the compiled plan through to the denoiser."
                )
            plan_tokens, plan_mask, plan_hint = self.plan_encoder(plan_dict, T)
            if cfg.cfg_drop_plan:
                plan_tokens = self._broadcast_drop(
                    cond_drop_mask, plan_tokens, self.null_plan_token,
                )
                if plan_hint is not None and hasattr(self, "null_plan_hint"):
                    plan_hint = self._broadcast_drop(
                        cond_drop_mask, plan_hint, self.null_plan_hint,
                    )
            if plan_hint is not None:
                per_frame = torch.cat([per_frame, plan_hint], dim=-1)

        h = self.in_proj(per_frame)                                      # (B, T, D)

        # --- Prepend timestep token + init-pose token ---
        t_tok = self.time_embed(t).unsqueeze(1)                          # (B, 1, D)
        pose_tok = self.pose_proj(init_pose).unsqueeze(1)                # (B, 1, D)
        seq = torch.cat([t_tok, pose_tok, h], dim=1)                     # (B, T+2, D)
        seq = self.pos_enc(seq)

        # --- Self-attention encoder (motion-side) ---
        seq = self.encoder(seq)                                          # (B, T+2, D)

        # --- v10 plan-token cross-attention ---
        # Apply once after the self-attention encoder, before text /
        # object cross-attn. Motion tokens query plan tokens directly so
        # the network has an explicit position-aware path to read
        # interaction-program information. Per
        # piano_interaction_plan_pipeline_reframe_for_claude_code.md §5.4
        # we start with a single block; if anchor-routing diagnostics
        # show partial improvement we move to per-layer cross-attn.
        if (
            cfg.use_interaction_plan
            and self.plan_xattn is not None
            and plan_tokens is not None
        ):
            seq = self.plan_xattn(seq, plan_tokens, plan_mask)

        # --- Cross-attn over text + object tokens (with CFG drop) ---
        text_q = seq
        text_kv = self.text_proj(text_tok)
        text_kv = self._broadcast_drop(cond_drop_mask, text_kv, self.null_text)
        text_attn, _ = self.text_xattn(text_q, text_kv, text_kv, need_weights=False)
        seq = self.text_norm(seq + text_attn)

        obj_kv = self.object_proj(obj_tok)
        obj_kv = self._broadcast_drop(cond_drop_mask, obj_kv, self.null_obj_tokens)
        obj_attn, _ = self.obj_xattn(seq, obj_kv, obj_kv, need_weights=False)
        seq = self.obj_norm(seq + obj_attn)

        # --- Drop the two prepended tokens; project to motion_dim ---
        h_out = seq[:, 2:, :]                                            # (B, T, D)
        x0 = self.out_proj(h_out)                                        # (B, T, motion_dim)
        return x0


# ============================================================================
# Public bundle
# ============================================================================


@dataclass(slots=True)
class AnchorDiffConfig:
    diffusion: DiffusionConfig
    denoiser: AnchorDenoiserConfig
    cfg_drop_prob: float = 0.15

    @classmethod
    def default(cls) -> "AnchorDiffConfig":
        return cls(
            diffusion=DiffusionConfig(),
            denoiser=AnchorDenoiserConfig(),
        )


class MotionAnchorDiff(nn.Module):
    """Wraps the diffusion process + denoiser. Use ``training_step`` for
    training and ``sample`` for inference."""

    def __init__(self, cfg: AnchorDiffConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.diffusion = GaussianDiffusion(cfg.diffusion)
        self.denoiser = AnchorDenoiser(cfg.denoiser)

    def training_step(self, x_start: Tensor, cond: dict) -> dict:
        """One forward pass: sample t, sample noise, run denoiser,
        return prediction + targets in BOTH x_0-space and the network's
        native parameterisation.

        Returned dict:
            - "x0_pred"     : clean-motion prediction (recovered from v
                              if prediction_target=='v'). Anchor loss /
                              FK loss / L_pos all attach here.
            - "x0_target"   : clean-motion ground truth (= x_start).
            - "diff_pred"   : raw network output (== x0_pred for "x0",
                              == v_pred for "v"). MSE diffusion loss
                              should be MSE(diff_pred, diff_target).
            - "diff_target" : MSE target matching diff_pred.
            - "x_t", "t"    : carried for downstream losses if needed.
        """
        B, T, _ = x_start.shape
        device = x_start.device
        t = torch.randint(0, self.diffusion.num_steps, (B,), device=device)
        noise = torch.randn_like(x_start)
        x_t = self.diffusion.q_sample(x_start, t, noise)

        # CFG dropout: per-sample bernoulli over the full conditioning.
        drop_mask = torch.rand(B, device=device) < self.cfg.cfg_drop_prob
        net_out = self.denoiser(x_t, t, cond, cond_drop_mask=drop_mask)

        target = self.cfg.diffusion.prediction_target
        denoiser_cfg = self.cfg.denoiser
        output_skip = bool(denoiser_cfg.cond_motion_output_skip)

        if target == "v":
            if output_skip:
                # v9_4 §6.3: output skip uses x_0-pred natively. Mixing v-pred
                # with hard observation makes the recovery formula and the
                # MSE target inconsistent.
                raise ValueError(
                    "cond_motion_output_skip=True requires prediction_target='x0'."
                )
            v_target = self.diffusion.v_target(x_start, t, noise)
            x0_raw = self.diffusion.predict_x0_from_v(x_t, t, net_out)
            x0_pred = x0_raw
            diff_pred = net_out
            diff_target = v_target
        else:  # "x0"
            x0_raw = net_out
            x0_pred = _apply_observed_x0_skip(
                x0_raw, cond,
                motion_dim=denoiser_cfg.motion_dim,
                enabled=output_skip,
            )
            # When skip is on, the diffusion target is matched against
            # x0_pred (= GT at obs frames + raw at unobs frames). The
            # caller can mask the loss to unobs frames only — see
            # train_anchordiff.py step_fn.
            diff_pred = x0_pred
            diff_target = x_start

        return {
            "x0_raw": x0_raw,
            "x0_pred": x0_pred,
            "x0_target": x_start,
            "diff_pred": diff_pred,
            "diff_target": diff_target,
            "x_t": x_t,
            "t": t,
        }

    @torch.no_grad()
    def sample(
        self,
        cond: dict,
        seq_length: int,
        cfg_scale: float = 1.0,
        replacement: str = "none",
        output_skip: bool | None = None,
        sampler: str = "ddpm",
    ) -> Tensor:
        """Generate motion from conditioning.

        replacement (legacy v9_3 ablation, per claude_code_v9_condmdi_diagnostic_next_steps.md §8.2):
          "none" : standard DDPM trajectory.
          "x0"   : replace predicted x_0 at observed frames with cond_motion.
          "x_t"  : replace x_t at observed frames before each network call.

        output_skip (v9_4, per claude_code_v9_4_hard_observation_execution_plan.md §6.4):
          If None, defaults to denoiser_cfg.cond_motion_output_skip (so the
          sampler matches what the model was trained with).
          If True/False, overrides — useful for ablations on a v9_4 ckpt.
        """
        B = cond["z_int"].shape[0]
        shape = (B, seq_length, self.cfg.denoiser.motion_dim)
        if output_skip is None:
            output_skip = bool(self.cfg.denoiser.cond_motion_output_skip)
        return self.diffusion.p_sample_loop(
            self.denoiser, shape, cond, cfg_scale=cfg_scale,
            replacement=replacement, output_skip=output_skip,
            sampler=sampler,
        )
