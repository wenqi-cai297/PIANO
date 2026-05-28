"""PIANO-AnchorDiff: anchor-conditioned continuous motion diffusion.

MDM-style transformer encoder denoiser conditioned on object trajectory,
object point-cloud tokens, text (CLIP), initial pose, and (optionally)
Stage-1 Coarse-v1 + Round-29 typed C/I/S/B condition extras. Trained
with classifier-free guidance dropout. Operates on HumanML3D motion_263.

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
    objective: str = "ddpm"  # "ddpm" | "rectified_flow"
    # "x0" = denoiser predicts clean x_0 (MDM/OMOMO style).
    # "v"  = denoiser predicts v = sqrt(ᾱ)·ε - sqrt(1-ᾱ)·x_0 (Salimans & Ho 2022).
    #        Hybrid noise+clean target; recommended by Back-to-Basics for
    #        better dynamics under rotation reps. x_0 is recovered via
    #        x_0 = sqrt(ᾱ)·x_t - sqrt(1-ᾱ)·v inside the sampler / training.
    prediction_target: str = "x0"
    # Rectified-flow / flow-matching ablation. Time convention follows ELF:
    # rf_t=0 is pure noise, rf_t=1 is clean data. We map RF time to the
    # existing DDPM-style timestep embedding as noisedness:
    # rf_t=0 -> num_steps-1, rf_t=1 -> 0.
    rf_eps_time: float = 0.05
    rf_time_schedule: str = "uniform"  # "uniform" | "logit_normal"
    rf_denoiser_p_mean: float = -1.5
    rf_denoiser_p_std: float = 0.8
    rf_denoiser_noise_scale: float = 1.0
    rf_num_sampling_steps: int = 100
    rf_sampler_type: str = "rectified_flow_ode"  # "rectified_flow_ode" | "rectified_flow_sde"
    rf_sde_gamma: float = 0.0


class GaussianDiffusion(nn.Module):
    """ε-prediction DDPM with cosine β-schedule.

    All buffers are float32 and registered, so they move with the module
    under Accelerate / .to(device).
    """

    def __init__(self, cfg: DiffusionConfig) -> None:
        super().__init__()
        self.num_steps = cfg.num_steps
        self.objective = cfg.objective
        if cfg.objective not in ("ddpm", "rectified_flow"):
            raise ValueError(
                f"objective must be 'ddpm' or 'rectified_flow', got {cfg.objective!r}"
            )
        self.prediction_target = cfg.prediction_target
        if cfg.prediction_target not in ("x0", "v"):
            raise ValueError(f"prediction_target must be 'x0' or 'v', got {cfg.prediction_target!r}")
        self.rf_eps_time = float(cfg.rf_eps_time)
        self.rf_time_schedule = str(cfg.rf_time_schedule)
        self.rf_denoiser_p_mean = float(cfg.rf_denoiser_p_mean)
        self.rf_denoiser_p_std = float(cfg.rf_denoiser_p_std)
        self.rf_denoiser_noise_scale = float(cfg.rf_denoiser_noise_scale)
        self.rf_num_sampling_steps = int(cfg.rf_num_sampling_steps)
        self.rf_sampler_type = str(cfg.rf_sampler_type)
        self.rf_sde_gamma = float(cfg.rf_sde_gamma)
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

    def sample_rf_t(self, batch_size: int, device: torch.device) -> Tensor:
        """Sample rectified-flow times in (0, 1), where 0=noise and 1=data."""
        eps = float(self.rf_eps_time)
        if self.rf_time_schedule == "uniform":
            t = torch.rand(batch_size, device=device)
        elif self.rf_time_schedule == "logit_normal":
            z = (
                torch.randn(batch_size, device=device) * float(self.rf_denoiser_p_std)
                + float(self.rf_denoiser_p_mean)
            )
            t = torch.sigmoid(z)
        else:
            raise ValueError(f"Unknown rf_time_schedule={self.rf_time_schedule!r}")
        return t.clamp(min=eps, max=1.0 - eps)

    def rf_time_to_index(self, t_rf: Tensor) -> Tensor:
        """Map RF time (0=noise, 1=data) to existing DDPM noisedness index."""
        t_idx = torch.round((1.0 - t_rf) * float(self.num_steps - 1))
        return t_idx.clamp(0, self.num_steps - 1).long()

    def rf_interpolate(self, x_start: Tensor, t_rf: Tensor, noise: Tensor) -> Tensor:
        """Rectified-flow path z_t = t*x0 + (1-t)*noise."""
        t = t_rf.reshape(t_rf.shape[0], *((1,) * (x_start.ndim - 1)))
        return t * x_start + (1.0 - t) * noise

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
        sampler: str = "ddpm",
    ) -> Tensor:
        """Reverse-diffusion sampling with optional classifier-free guidance,
        operating on x₀-prediction outputs.

        ``cond`` follows the same dict spec as ``AnchorDenoiser.forward``.
        ``cfg_scale > 1`` enables CFG: x_0 = x_0_uncond + s·(x_0_cond - x_0_uncond).

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

        for t_int in reversed(range(self.num_steps)):
            t = torch.full((shape[0],), t_int, device=device, dtype=torch.long)

            pred_cond = denoiser(
                x, t, cond, cond_drop_mask=None,
            )
            if cfg_scale != 1.0:
                drop = torch.ones(shape[0], dtype=torch.bool, device=device)
                pred_uncond = denoiser(
                    x, t, cond, cond_drop_mask=drop,
                )
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

    @torch.no_grad()
    def rf_sample_loop(
        self,
        denoiser: "AnchorDenoiser",
        shape: tuple[int, ...],
        cond: dict,
        cfg_scale: float = 1.0,
        device: torch.device | None = None,
        num_steps: int | None = None,
        sampler_type: str | None = None,
        time_schedule: str | None = None,
        sde_gamma: float | None = None,
        return_intermediates: tuple[float, ...] | None = None,
    ) -> Tensor | tuple[Tensor, dict[float, Tensor]]:
        """Rectified-flow ODE/SDE sampler adapted from ELF's continuous flow."""
        device = device or self.betas.device
        steps = int(num_steps or self.rf_num_sampling_steps)
        sampler = str(sampler_type or self.rf_sampler_type)
        schedule = str(time_schedule or self.rf_time_schedule)
        gamma = float(self.rf_sde_gamma if sde_gamma is None else sde_gamma)
        eps_time = float(self.rf_eps_time)
        if steps <= 0:
            raise ValueError(f"rf num_steps must be positive, got {steps}")
        if sampler not in ("rectified_flow_ode", "rectified_flow_sde"):
            raise ValueError(f"Unknown RF sampler_type={sampler!r}")

        if schedule == "uniform":
            times = torch.linspace(0.0, 1.0, steps + 1, device=device)
        elif schedule == "logit_normal":
            inner = torch.sigmoid(
                torch.randn(steps - 1, device=device) * float(self.rf_denoiser_p_std)
                + float(self.rf_denoiser_p_mean)
            )
            times = torch.cat([
                torch.zeros(1, device=device),
                inner.sort().values,
                torch.ones(1, device=device),
            ])
        else:
            raise ValueError(f"Unknown RF sampling schedule={schedule!r}")

        z = torch.randn(shape, device=device) * float(self.rf_denoiser_noise_scale)
        logs: dict[float, Tensor] = {}
        log_targets = tuple(float(v) for v in (return_intermediates or ()))
        next_log_idx = 0

        def _predict_clean(z_in: Tensor, t_rf_scalar: Tensor) -> Tensor:
            t_batch_rf = t_rf_scalar.expand(shape[0])
            t_idx = self.rf_time_to_index(t_batch_rf)
            pred_cond = denoiser(z_in, t_idx, cond, cond_drop_mask=None)
            if cfg_scale != 1.0:
                drop = torch.ones(shape[0], dtype=torch.bool, device=device)
                pred_uncond = denoiser(z_in, t_idx, cond, cond_drop_mask=drop)
                x0 = pred_uncond + float(cfg_scale) * (pred_cond - pred_uncond)
            else:
                x0 = pred_cond
            return x0

        for i in range(steps):
            t_cur = times[i]
            t_next = times[i + 1]
            z_in = z
            t_model = t_cur
            if sampler == "rectified_flow_sde" and gamma > 0.0:
                h = t_next - t_cur
                alpha = torch.clamp(1.0 - gamma * h, min=0.0, max=1.0)
                t_model = alpha * t_cur
                eps = torch.randn_like(z) * float(self.rf_denoiser_noise_scale)
                z_in = alpha * z + (1.0 - alpha) * eps

            x0 = _predict_clean(z_in, t_model.reshape(()))
            v = (x0 - z_in) / torch.clamp(1.0 - t_model, min=eps_time)
            z = z_in + (t_next - t_model) * v

            while next_log_idx < len(log_targets) and float(t_next.item()) >= log_targets[next_log_idx]:
                logs[log_targets[next_log_idx]] = z.detach().clone()
                next_log_idx += 1

        if return_intermediates is not None:
            if 1.0 not in logs:
                logs[1.0] = z.detach().clone()
            return z, logs
        return z


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
# Denoiser
# ============================================================================


@dataclass(slots=True)
class AnchorDenoiserConfig:
    motion_dim: int = 263
    object_traj_dim: int = 9     # 3 (pos) + 6 (rot6d)
    init_pose_dim: int = 22 * 3  # SMPL-22 joints
    text_dim: int = 512          # CLIP per-token feature dim
    object_token_dim: int = 256  # Stage A's object encoder output dim
    object_num_tokens: int = 128

    # Stage-1 Coarse-v1 (23-D route/backbone) condition branch.
    # When ``stage1_coarse_dim > 0``: V12InputProjection instantiates a
    # zero-init ``stage1_coarse_proj``; the trainer must populate
    # ``cond["stage1_coarse"]`` of shape (B, T, stage1_coarse_dim).
    # Frame convention: Coarse-v1 root_local_trans is root0-relative
    # world axis (matches S1-O obj_traj_root0_world frame). Stage-1
    # output is treated as deterministic and is never CFG-dropped.
    stage1_coarse_dim: int = 0

    # Round-29 Stage-2 condition injection (per analyses/
    # 2026-05-26_stage2_cond_injection_ablation_claude_code_prompt.md).
    # Opt-in alternative to R28 oracle_interaction_hint + body_action_hint
    # paths: typed per-family projections for the four condition families
    # (coarse_extra / interaction / support / body_refine), with one of
    # five injection modes (input_add, gated_input, adapter_only,
    # input_add_adapter, typed). When ALL four family dims are 0 OR
    # use_round29_cond_injection=False, this branch is fully bypassed and
    # the model behaves identically to R28.
    use_round29_cond_injection: bool = False
    r29_coarse_extra_dim: int = 0
    r29_interaction_dim: int = 0
    r29_support_dim: int = 0
    r29_body_refine_dim: int = 0
    r29_injection_mode: str = "input_add"
    r29_gate_bias_init: float = -1.0
    # Typed per-family modes (J4). None = uniform mode from
    # r29_injection_mode; dict = per-family override.
    r29_per_family_modes: dict | None = None
    r29_zero_init_adapters: bool = True

    # PB1 — AdaLN-cond branch (per Codex review §4.3 / §4.4 of
    # analyses/2026-05-29_round29_cond_injection_prior_codex_review_for_claude_code.md).
    # When ``r29_use_cond_adaln=True``, GlobalCondSummary gains a
    # zero-init Linear that adds a pooled R29 cond summary to the AdaLN
    # control vector. ``r29_adaln_families`` selects which active
    # families feed the pool; ``r29_adaln_pool`` chooses the pooling
    # method:
    #   - "mean":                 mean over T, then mean across families.
    #   - "support_walking_mean": walking_mask-weighted mean of the
    #                             support family (S4 dim 4 = walking_mask).
    # Phase 0 verdict (analyses/2026-05-29_round29_cond_usage_verdict.md):
    # A1 uses S4 actively but non-temporally with sub-linear scale
    # response (lin = 0.70). AdaLN-S4 with support_walking_mean is the
    # textbook fix for this regime. C41 deliberately stays OUT of the
    # pool — Codex §4.4 — because it is a spatial scaffold and pooling
    # destroys spatial structure.
    r29_use_cond_adaln: bool = False
    r29_adaln_families: tuple | list | None = None
    r29_adaln_pool: str = "mean"

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
        - object_world_traj     : 9 dims (pos + rot6d)
        - stage1_coarse         : 23 dims, optional (Stage-1 Coarse-v1)
        - r29 typed extras      : C/I/S/B per-family (optional)
    Sequence-level conditioning (cross-attention K/V):
        - text                  : CLIP per-token (B, 77, text_dim)
        - object_pc tokens      : (B, 128, object_token_dim)
    Special tokens prepended to the sequence:
        - timestep token (sinusoidal embedding -> Linear projection)
        - init_pose token (Linear projection of frame-0 SMPL-22 joints)

    CFG dropout is applied independently per conditioning channel via
    ``cond_drop_mask`` (a (B,) bool tensor): if True for a sample,
    object_traj + text + object_pc tokens are masked to a learned null
    embedding for that sample. ``init_pose`` and ``stage1_coarse`` are
    preserved (deterministic scene facts).
    """

    def __init__(self, cfg: AnchorDenoiserConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.time_embed = nn.Sequential(
            SinusoidalTimestepEmbed(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        # init_pose_dim=0 disables the frame-0 SMPL-22 init-pose token
        # entirely (Tier-2 ablation: is frame-0 GT a useful redundant
        # signal on top of motion[0] + stage1_coarse[0], or just a
        # train/inference distribution leak?).
        self.use_init_pose = cfg.init_pose_dim > 0
        if self.use_init_pose:
            self.pose_proj = nn.Linear(cfg.init_pose_dim, cfg.d_model)
        else:
            self.pose_proj = None
        # text_dim=0 disables the CLIP text condition + text_xattn entirely
        # (Tier-2 ablation: does the InterAct text label add anything that
        # object_tokens doesn't already encode?).
        self.use_text = cfg.text_dim > 0
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
        self.object_proj = nn.Linear(cfg.object_token_dim, cfg.d_model)

        # Learned null embeddings for CFG dropout. Live channels are
        # object_world_traj and object_tokens (and text when on).
        # stage1_coarse is never CFG-dropped (it is a deterministic
        # Stage-1 output).
        self.null_obj_traj = nn.Parameter(torch.zeros(cfg.object_traj_dim))
        self.null_obj_tokens = nn.Parameter(torch.zeros(1, 1, cfg.d_model))

        self.pos_enc = PositionalEncoding(cfg.d_model, max_len=cfg.max_seq_length + 2)

        self.obj_xattn = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True
        )
        self.obj_norm = nn.LayerNorm(cfg.d_model)

        # v12 modules (per analyses/2026-05-11_v12_architecture_design_doc.md):
        # V12InputProjection + GlobalCondSummary + ConditionedEncoderLayer ×
        # n_layers + V12FinalLayer.
        from piano.models.dit_blocks import (
            V12InputProjection,
            GlobalCondSummary,
            ConditionedEncoderLayer,
            V12FinalLayer,
            initialize_weights_v12,
        )
        self.v12_input_proj = V12InputProjection(
            motion_dim=cfg.motion_dim,
            obj_traj_dim=cfg.object_traj_dim,
            d_model=cfg.d_model,
            stage1_coarse_dim=cfg.stage1_coarse_dim,
        )
        self.v12_cond_summary = GlobalCondSummary(
            d_model=cfg.d_model,
            use_cond_summary_mlp=bool(cfg.r29_use_cond_adaln),
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
        initialize_weights_v12(
            input_proj=self.v12_input_proj,
            blocks=self.v12_blocks,
            final_layer=self.v12_final_layer,
            cond_summary=self.v12_cond_summary,
        )

        # Round-29 condition injection module (opt-in).
        if cfg.use_round29_cond_injection:
            from piano.models.round29_cond_injection import (
                Round29CondInjectionConfig,
                Round29CondInjectionModule,
                coerce_per_family_modes,
            )
            r29_cfg = Round29CondInjectionConfig(
                coarse_extra_dim=int(cfg.r29_coarse_extra_dim),
                interaction_dim=int(cfg.r29_interaction_dim),
                support_dim=int(cfg.r29_support_dim),
                body_refine_dim=int(cfg.r29_body_refine_dim),
                injection_mode=str(cfg.r29_injection_mode),
                gate_bias_init=float(cfg.r29_gate_bias_init),
                per_family_modes=coerce_per_family_modes(
                    cfg.r29_per_family_modes,
                ),
                zero_init_adapters=bool(cfg.r29_zero_init_adapters),
            )
            self.r29_inject = Round29CondInjectionModule(
                r29_cfg, d_model=cfg.d_model,
            )
            self.r29_inject.configure_adapter_layers(int(cfg.n_layers))
        else:
            self.r29_inject = None

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

        obj_traj: Tensor = cond["object_world_traj"]   # (B, T, 9)
        obj_tok: Tensor = cond["object_tokens"]        # (B, N_obj, object_token_dim)
        # init_pose and text are optional in R29 ablations.
        init_pose: Tensor | None = cond.get("init_pose") if self.use_init_pose else None
        text_tok: Tensor | None = cond.get("text") if self.use_text else None

        # --- CFG drop: replace conditioning channels with null embeddings ---
        # stage1_coarse is treated as a deterministic Stage-1 output and is
        # never CFG-dropped (it must always reach the denoiser at inference).
        obj_traj_eff = self._broadcast_drop(cond_drop_mask, obj_traj, self.null_obj_traj)

        # ─── Timestep embedding (B, D) ───
        t_emb = self.time_embed(t)                                       # (B, D)

        # ─── §4.3 Input projection (per-channel summed, residual stream) ───
        stage1_coarse_eff: Tensor | None = None
        if cfg.stage1_coarse_dim > 0:
            if "stage1_coarse" not in cond:
                raise KeyError(
                    "stage1_coarse_dim > 0 requires cond['stage1_coarse'] "
                    "(B, T, stage1_coarse_dim). The trainer must populate "
                    "this from oracle GT extraction or the S1-O sampler."
                )
            stage1_coarse_eff = cond["stage1_coarse"]
        h = self.v12_input_proj(
            x_t=x_t,
            obj_traj=obj_traj_eff,
            stage1_coarse=stage1_coarse_eff,
        )                                                                # (B, T, D)

        # ─── PB1: precompute per-family embeddings so AdaLN pool + input-add
        #         lane share one projection pass. No-op when r29_inject is
        #         off; cheap when on (it is what apply_input_injection used
        #         to do internally on first call).
        r29_cond_summary: Tensor | None = None
        if self.r29_inject is not None:
            self.r29_inject.compute_family_embeddings(cond)
            if cfg.r29_use_cond_adaln:
                families = cfg.r29_adaln_families or ()
                r29_cond_summary = self.r29_inject.pool_cond_summary(
                    families=families,
                    cond=cond,
                    pool=cfg.r29_adaln_pool,
                )                                                        # (B, D)

        # ─── §4.4 Global condition vector for AdaLN (per-sample) ───
        # When PB1 is off this is t_emb unchanged (R28 / A1 behaviour).
        # When PB1 is on this is t_emb + cond_summary_mlp(r29_cond_summary),
        # with the MLP final Linear zero-init so the contribution starts at 0.
        c = self.v12_cond_summary(t_emb=t_emb, cond_summary=r29_cond_summary)  # (B, D)

        # ─── Round-29: typed condition family input injection ───
        # apply_input_injection now reuses the cache populated above
        # (no double projection pass).
        if self.r29_inject is not None:
            h = self.r29_inject.apply_input_injection(h, cond, c_summary=c)

        # ─── Prepend init_pose token (time_tok dropped — timestep is in AdaLN) ───
        # When init_pose_dim=0 the model skips the prefix token entirely;
        # the motion-token slice starts at index 0 instead of 1.
        if self.use_init_pose:
            pose_tok = self.pose_proj(init_pose).unsqueeze(1)            # (B, 1, D)
            seq = torch.cat([pose_tok, h], dim=1)                        # (B, T+1, D)
            motion_token_start = 1
        else:
            seq = h                                                       # (B, T, D)
            motion_token_start = 0
        seq = self.pos_enc(seq)

        # ─── §4.5 ConditionedEncoderLayer × n_layers ───
        for layer_idx, block in enumerate(self.v12_blocks):
            seq = block(seq, c)
            # Round-29: optional per-family per-layer adapters. No-op
            # unless ``r29_inject`` is present and at least one active
            # family uses an adapter-mode.
            if self.r29_inject is not None:
                seq = self.r29_inject.apply_per_layer_adapter(
                    seq, layer_idx=layer_idx,
                    motion_token_start=motion_token_start,
                )

        # ─── Text cross-attn (optional) + obj cross-attn at end of encoder ───
        if self.use_text:
            text_kv = self.text_proj(text_tok)
            text_kv = self._broadcast_drop(cond_drop_mask, text_kv, self.null_text)
            text_attn, _ = self.text_xattn(seq, text_kv, text_kv, need_weights=False)
            seq = self.text_norm(seq + text_attn)

        obj_kv = self.object_proj(obj_tok)
        obj_kv = self._broadcast_drop(cond_drop_mask, obj_kv, self.null_obj_tokens)
        obj_attn, _ = self.obj_xattn(seq, obj_kv, obj_kv, need_weights=False)
        seq = self.obj_norm(seq + obj_attn)

        # ─── §4.6 Final layer (drop pose prefix token if present, AdaLN-Zero readout) ───
        h_motion = seq[:, motion_token_start:, :]                        # (B, T, D)
        x0 = self.v12_final_layer(h_motion, c)                           # (B, T, motion_dim)
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

    def forward(self, x_start: Tensor, cond: dict) -> dict:
        """DDP-compatible entry point that delegates to ``training_step``.

        PyTorch DistributedDataParallel intercepts gradient sync via
        ``Module.__call__`` (the path Python takes when you write
        ``model(x_start, cond)``). Calling a custom method like
        ``training_step`` directly bypasses ``__call__`` so DDP never
        gets a chance to reset its per-iteration reducer — under
        multi-process training this manifests as either AttributeError
        on the DDP wrapper (it does not forward arbitrary attribute
        lookups to ``.module``) or as stale reducer state on iteration
        boundaries.

        Trainers (``train_anchordiff.py::step_fn``) should call
        ``out = model(motion, cond)`` so DDP's ``__call__`` runs.
        ``training_step`` remains the canonical implementation and is
        still callable directly on the unwrapped model for inference /
        debugging utilities.
        """
        return self.training_step(x_start, cond)

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

        if self.diffusion.objective == "rectified_flow":
            if self.cfg.diffusion.prediction_target != "x0":
                raise ValueError("rectified_flow objective currently requires prediction_target='x0'")
            t_rf = self.diffusion.sample_rf_t(B, device=device)
            t_idx = self.diffusion.rf_time_to_index(t_rf)
            noise = torch.randn_like(x_start) * float(self.diffusion.rf_denoiser_noise_scale)
            z_t = self.diffusion.rf_interpolate(x_start, t_rf, noise)

            # CFG dropout: per-sample bernoulli over the full conditioning.
            drop_mask = torch.rand(B, device=device) < self.cfg.cfg_drop_prob
            x0_raw = self.denoiser(
                z_t, t_idx, cond, cond_drop_mask=drop_mask,
            )
            x0_pred = x0_raw
            denom = (1.0 - t_rf).view(B, 1, 1).clamp_min(float(self.diffusion.rf_eps_time))
            v_pred = (x0_pred - z_t) / denom
            v_target = x_start - noise
            return {
                "x0_raw": x0_raw,
                "x0_pred": x0_pred,
                "x0_target": x_start,
                "diff_pred": v_pred,
                "diff_target": v_target,
                "x_t": z_t,
                "t": t_idx,
                "rf_t": t_rf,
                "rf_v_pred_norm": v_pred.detach().pow(2).mean().sqrt(),
                "rf_v_target_norm": v_target.detach().pow(2).mean().sqrt(),
                "rf_x0_pred_norm": x0_pred.detach().pow(2).mean().sqrt(),
            }

        t = torch.randint(0, self.diffusion.num_steps, (B,), device=device)
        noise = torch.randn_like(x_start)
        x_t = self.diffusion.q_sample(x_start, t, noise)

        # CFG dropout: per-sample bernoulli over the full conditioning.
        drop_mask = torch.rand(B, device=device) < self.cfg.cfg_drop_prob
        target = self.cfg.diffusion.prediction_target
        denoiser_cfg = self.cfg.denoiser

        net_out = self.denoiser(
            x_t, t, cond, cond_drop_mask=drop_mask,
        )

        if target == "v":
            v_target = self.diffusion.v_target(x_start, t, noise)
            x0_raw = self.diffusion.predict_x0_from_v(x_t, t, net_out)
            x0_pred = x0_raw
            diff_pred = net_out
            diff_target = v_target
        else:  # "x0"
            x0_raw = net_out
            x0_pred = x0_raw
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
        sampler: str = "ddpm",
    ) -> Tensor:
        """Generate motion from conditioning."""
        B = cond["object_world_traj"].shape[0]
        shape = (B, seq_length, self.cfg.denoiser.motion_dim)
        if self.diffusion.objective == "rectified_flow" or sampler.startswith("rectified_flow"):
            return self.diffusion.rf_sample_loop(
                self.denoiser, shape, cond, cfg_scale=cfg_scale,
                device=cond["object_world_traj"].device,
                sampler_type=sampler if sampler.startswith("rectified_flow") else None,
            )
        return self.diffusion.p_sample_loop(
            self.denoiser, shape, cond, cfg_scale=cfg_scale,
            sampler=sampler,
        )
