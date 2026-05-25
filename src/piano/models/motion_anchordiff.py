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
        denoiser_cfg = getattr(denoiser, "cfg", None)
        use_self_cond = bool(getattr(denoiser_cfg, "use_self_conditioning", False))
        self_cond_mode = str(getattr(denoiser_cfg, "self_conditioning_mode", "standard"))
        self_cond_t_max = int(getattr(denoiser_cfg, "self_conditioning_t_max", 700))
        self_cond: Tensor | None = None

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
            if not use_self_cond:
                curr_self_cond = None
            elif self_cond_mode == "standard":
                curr_self_cond = self_cond
            elif self_cond_mode == "late_start":
                curr_self_cond = self_cond if t_int <= self_cond_t_max else None
            else:
                raise ValueError(
                    "self_conditioning_mode must be 'standard' or 'late_start', "
                    f"got {self_cond_mode!r}"
                )

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

            pred_cond = denoiser(
                x, t, cond, cond_drop_mask=None, self_cond=curr_self_cond,
            )
            if cfg_scale != 1.0:
                drop = torch.ones(shape[0], dtype=torch.bool, device=device)
                pred_uncond = denoiser(
                    x, t, cond, cond_drop_mask=drop, self_cond=curr_self_cond,
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

            if use_self_cond:
                self_cond = x0.detach()

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
        output_skip: bool = False,
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

        cond_motion = obs_mask = None
        if output_skip:
            if "cond_motion_input" not in cond:
                raise ValueError("output_skip=True requires cond['cond_motion_input']")
            cmi = cond["cond_motion_input"]
            motion_dim = shape[-1]
            cond_motion = cmi[..., :motion_dim]
            obs_mask = cmi[..., motion_dim:motion_dim + 1]

        def _predict_clean(z_in: Tensor, t_rf_scalar: Tensor) -> Tensor:
            t_batch_rf = t_rf_scalar.expand(shape[0])
            t_idx = self.rf_time_to_index(t_batch_rf)
            pred_cond = denoiser(z_in, t_idx, cond, cond_drop_mask=None, self_cond=None)
            if cfg_scale != 1.0:
                drop = torch.ones(shape[0], dtype=torch.bool, device=device)
                pred_uncond = denoiser(z_in, t_idx, cond, cond_drop_mask=drop, self_cond=None)
                x0 = pred_uncond + float(cfg_scale) * (pred_cond - pred_uncond)
            else:
                x0 = pred_cond
            if output_skip and cond_motion is not None and obs_mask is not None:
                x0 = obs_mask * cond_motion + (1.0 - obs_mask) * x0
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

    # v12 DiT/InterGen-style conditional transformer (per
    # analyses/2026-05-11_v12_architecture_design_doc.md).
    # When ``use_dit_block=True``: separate per-channel input projections
    # summed (V12InputProjection) + per-block AdaLN-Zero + per-block plan
    # cross-attn (PixArt placement) + AdaLN final layer. When False:
    # legacy v11 path (concat + nn.TransformerEncoder + end-of-encoder
    # plan_xattn x 1). Default False = back-compat preserved.
    use_dit_block: bool = False

    # v12-A1 flag (post 2026-05-11 cond_diversity_audit):
    # when True (v12 default): AdaLN cond = t_emb + plan_pool_emb (InterGen).
    # when False (A1): AdaLN cond = t_emb only; all plan info must flow
    # through per-layer plan cross-attn. Forces the model to use per-anchor
    # spatial detail rather than the masked-mean pool that washes it out.
    # See analyses/2026-05-11_cond_diversity_audit.md §4 + v12 round report §7.
    dit_block_use_plan_pool_in_cond: bool = True

    # v13 (per analyses/2026-05-11_v13_dynhead_temporalconv_design.md):
    #   use_v13_dynhead: replace V12FinalLayer with V13DynamicsHead
    #     (base + cumsum-integrated delta residual, learnable γ scale).
    #   use_v13_temporal_conv: insert depthwise temporal Conv1D residual
    #     (zero-gated) between self-attn and plan-xattn inside every block.
    #   Both flags require use_dit_block=True. Defaults preserve v12 A1.
    use_v13_dynhead: bool = False
    v13_dynhead_gamma_init: float = 0.1
    v13_dynhead_learnable_gamma: bool = True
    use_v13_temporal_conv: bool = False
    v13_temporal_conv_kernel: int = 5

    # v22 self-conditioning ablation. When enabled, the DiT input stream
    # receives a zero-gated projection of an x0 prediction. During training it
    # is generated by a no-grad first pass at the same x_t and t; during
    # sampling it is the previous reverse step's x0 prediction.
    use_self_conditioning: bool = False
    self_conditioning_prob: float = 0.0
    self_conditioning_mode: str = "standard"  # "standard" | "late_start"
    self_conditioning_t_max: int = 700
    self_conditioning_zero_init: bool = True

    # Round-22: Stage-1 Coarse-v1 (23-D route/backbone) condition branch.
    # When ``stage1_coarse_dim > 0``: V12InputProjection instantiates a
    # zero-init ``stage1_coarse_proj`` so the step-0 output is bit-exact
    # equal to the pre-R22 path; the trainer must populate
    # ``cond["stage1_coarse"]`` of shape (B, T, stage1_coarse_dim).
    # ``cfg_drop_stage1_coarse`` controls whether the branch participates
    # in CFG dropout (default False — Round-22 smokes don't perturb the
    # route stream under cfg sweeps).
    # Frame convention: Coarse-v1 root_local_trans is root0-relative
    # world axis (matches S1-O obj_traj_root0_world frame). See
    # ``analyses/2026-05-22_stage2_condition_reframe_and_next_plan.md`` §6.
    # Currently requires ``use_dit_block=True`` — v11 legacy path does
    # not support this branch.
    stage1_coarse_dim: int = 0
    cfg_drop_stage1_coarse: bool = False

    # Round-23: ``plan_tokens_force_null`` is the no-plan ablation flag.
    # When True, the v12 forward replaces ``plan_tokens`` with the learned
    # ``null_plan_token`` (broadcast) and ``plan_hint`` with
    # ``null_plan_hint`` AFTER the plan encoder forward, regardless of
    # ``cond_drop_mask``. The interaction-plan condition is fully removed
    # from the per-frame residual stream and the per-block cross-attention.
    # Used to test whether plan information is load-bearing at full scale
    # (paired ablation against the with-plan run). Must be combined with
    # zeroed plan-aware loss weights (plan_anchor_weight=0, etc.) in the
    # YAML config for a clean no-plan comparison.
    plan_tokens_force_null: bool = False

    # Round-23: ALiBi-style relative-time bias on the per-block plan
    # cross-attention. The attention-inspection diagnostic (see
    # ``analyses/2026-05-22_round22_tier_b_v18_baseline_diagnostic_report.md``
    # §5 + ``scripts/stage_b_generator/plan_cross_attention_inspector.py``)
    # showed that v18 + R22 both produce vertical-band attention patterns
    # — motion-tokens-at-all-frames attend to the same 1-2 plan tokens
    # regardless of motion-frame time (top1=nearest-anchor ≈ 5-30%, near
    # chance). The fix adds an additive pre-softmax bias to plan
    # cross-attention proportional to ``-|motion_time - plan_token_time|``
    # so the model has an explicit inductive bias toward temporally
    # nearby anchors. Slopes are learnable per (layer, head).
    #
    # When ``plan_xattn_relative_time_bias=False`` (default), the
    # cross-attention behaves identically to v18/R22 — bit-exact
    # equivalent forward, no new parameters added. When ``True``, slopes
    # initialized to ``plan_xattn_time_bias_init`` (default 0.5 — non-
    # trivial inductive bias at init that the model can refine; set
    # explicitly to 0.0 to preserve strict zero-init invariant for ckpt
    # loadability tests).
    plan_xattn_relative_time_bias: bool = False
    plan_xattn_time_bias_init: float = 0.5

    # Round-27 Tier-0A: per-frame oracle interaction-hint branch
    # (roadmap §6.12). When ``use_oracle_interaction_hint=True``,
    # ``cond["oracle_interaction_hint"]`` of shape (B, T,
    # oracle_hint_dim) is projected via a 2-layer MLP and ADDED into the
    # per-frame motion-token embedding ``h`` right after
    # ``v12_input_proj``. The hint is intentionally injected directly
    # (not via cross-attention, not via plan tokens) to give Stage-2 an
    # upper-bound diagnostic on whether explicit per-frame interaction
    # state helps sustained contact / gait. v12-only — v11 legacy path
    # does not support this branch.
    use_oracle_interaction_hint: bool = False
    oracle_hint_dim: int = 0

    # Round-28 Stage-2 oracle interface refinement (analyses/
    # 2026-05-25_round28_*.md). Two orthogonal extensions on top of the
    # Round-27 input-add baseline:
    #
    # 1. Body-action hint branch — a separate 24D channel projected
    #    through its OWN MLP (not merged into a monolithic 37D MLP).
    # 2. Injection mode — `input_add` (R27 baseline, kept), `gated_input`
    #    (separate per-branch sigmoid gate driven by the AdaLN summary
    #    c-vector + branch embedding), `per_layer_adapter` (zero-init
    #    per-DiT-block adapters added ON TOP of input_add), or
    #    `adapter_only` (Round-28 A2b ablation: pure per-layer adapters,
    #    no input-token add — isolates the adapter contribution).
    #
    # When `separate_hint_branches=False` and only interaction_hint is
    # enabled, the behavior is bit-exact equivalent to R27 input-add.
    use_body_action_hint: bool = False
    body_action_hint_dim: int = 0
    # `input_add` / `gated_input` / `per_layer_adapter` / `adapter_only`.
    oracle_hint_injection_mode: str = "input_add"
    # Initial sigmoid gate bias for ``gated_input``. -3.0 is the
    # conservative A1 setting; -1.0 is the fair-gate A1b ablation.
    oracle_hint_gate_bias_init: float = -3.0
    # When True, interaction & body-action hints have their own
    # projections + (if gated/adapter) their own gates/adapters.
    separate_hint_branches: bool = True
    # When True, every newly-added adapter / second-projection-layer is
    # zero-initialized so step-0 forward matches the no-hint baseline.
    zero_init_hint_adapters: bool = True

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

        # Round-22 guard: Stage-1 coarse branch is v12-only.
        if cfg.stage1_coarse_dim > 0 and not cfg.use_dit_block:
            raise ValueError(
                "stage1_coarse_dim > 0 requires use_dit_block=True (v12). "
                "The v11 legacy concat-then-Linear path does not support the "
                "Stage-1 Coarse-v1 branch. Set use_dit_block=true in the config."
            )

        # v11 path: concat-then-Linear input projection. Skipped under v12.
        if not cfg.use_dit_block:
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

        # Round-22: Stage-1 Coarse-v1 null embedding for cfg_drop_stage1_coarse=True.
        # Stored unconditionally for state_dict stability; consumed only when the
        # branch is active (stage1_coarse_dim > 0) AND cfg_drop_stage1_coarse=True.
        if cfg.stage1_coarse_dim > 0:
            self.null_stage1_coarse = nn.Parameter(
                torch.zeros(cfg.stage1_coarse_dim)
            )
        else:
            self.register_parameter("null_stage1_coarse", None)

        self.pos_enc = PositionalEncoding(cfg.d_model, max_len=cfg.max_seq_length + 2)

        if cfg.use_self_conditioning and not cfg.use_dit_block:
            raise ValueError(
                "Self-conditioning is implemented for the v12 DiT path only; "
                "set use_dit_block=True or disable use_self_conditioning."
            )

        # v11 path: vanilla self-attn encoder. v12 uses ConditionedEncoderLayer.
        if not cfg.use_dit_block:
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
            # v11 path: single end-of-encoder plan cross-attn block.
            # v12 path: per-block plan cross-attn lives inside each
            # ConditionedEncoderLayer (built below); skip this single block.
            if not cfg.use_dit_block:
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

        # ─── v12 modules (per analyses/2026-05-11_v12_architecture_design_doc.md) ───
        # When use_dit_block=True, replace v11's concat-input + vanilla
        # encoder + end-of-encoder plan_xattn with:
        #   - V12InputProjection: separate per-channel projections summed
        #   - GlobalCondSummary: per-sample (B, D) cond for AdaLN
        #   - ConditionedEncoderLayer × n_layers: AdaLN-Zero + plan xattn per block
        #   - V12FinalLayer: AdaLN-Zero + zero-init linear readout
        # text + obj cross-attn (above) stay at end-of-encoder (v11 carryover).
        if cfg.use_dit_block:
            from piano.models.dit_blocks import (
                V12InputProjection,
                GlobalCondSummary,
                ConditionedEncoderLayer,
                V12FinalLayer,
                V13DynamicsHead,
                initialize_weights_v12,
            )
            if not cfg.use_interaction_plan:
                raise ValueError(
                    "use_dit_block=True requires use_interaction_plan=True "
                    "(GlobalCondSummary pools plan tokens and per-block plan "
                    "cross-attn is the core of the v12 design)."
                )
            if not cfg.plan_use_context_hint:
                raise ValueError(
                    "use_dit_block=True requires plan_use_context_hint=True "
                    "(v12 InputProjection includes the plan_hint channel)."
                )
            if cfg.cond_motion_dim > 0:
                raise ValueError(
                    "use_dit_block=True is not compatible with cond_motion_dim>0 "
                    "(CondMDI inpainting channel was a v9 feature; v12 dropped it). "
                    "Use cond_motion_dim=0."
                )
            # Round-22: Stage-1 coarse branch is v12-only — v11 legacy path
            # has no V12InputProjection to extend, and adding it there would
            # invalidate the bandwidth-bottleneck audit.
            # (No-op when stage1_coarse_dim == 0; guard fires only if a caller
            # tries to enable it on the v11 path — see _build_v11_input below.)
            if cfg.use_self_conditioning and cfg.self_conditioning_mode not in (
                "standard", "late_start",
            ):
                raise ValueError(
                    "self_conditioning_mode must be 'standard' or 'late_start', "
                    f"got {cfg.self_conditioning_mode!r}"
                )
            self.v12_input_proj = V12InputProjection(
                motion_dim=cfg.motion_dim,
                zint_dim=cfg.z_int.total,
                obj_traj_dim=cfg.object_traj_dim,
                hint_dim=cfg.plan_d_hint,
                d_model=cfg.d_model,
                use_self_conditioning=cfg.use_self_conditioning,
                self_conditioning_zero_init=cfg.self_conditioning_zero_init,
                stage1_coarse_dim=cfg.stage1_coarse_dim,
            )
            self.v12_cond_summary = GlobalCondSummary(
                d_model=cfg.d_model,
                use_plan_pool=cfg.dit_block_use_plan_pool_in_cond,
            )
            self.v12_blocks = nn.ModuleList([
                ConditionedEncoderLayer(
                    d_model=cfg.d_model,
                    n_heads=cfg.n_heads,
                    ff_mult=cfg.ff_mult,
                    dropout=cfg.dropout,
                    use_temporal_conv=cfg.use_v13_temporal_conv,
                    temporal_conv_kernel=cfg.v13_temporal_conv_kernel,
                )
                for _ in range(cfg.n_layers)
            ])
            if cfg.use_v13_dynhead:
                self.v12_final_layer = V13DynamicsHead(
                    d_model=cfg.d_model,
                    motion_dim=cfg.motion_dim,
                    gamma_init=cfg.v13_dynhead_gamma_init,
                    learnable_gamma=cfg.v13_dynhead_learnable_gamma,
                )
            else:
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

            # Round-23: ALiBi-style relative-time bias slopes (per layer × head).
            if cfg.plan_xattn_relative_time_bias:
                self.plan_xattn_time_bias_slopes = nn.Parameter(
                    torch.full(
                        (cfg.n_layers, cfg.n_heads),
                        float(cfg.plan_xattn_time_bias_init),
                    )
                )
            else:
                self.register_parameter("plan_xattn_time_bias_slopes", None)

            # Round-27/28 oracle hint branches. Two independent branches:
            # interaction (13D / 8 / 5) and body-action (24D). Each gets
            # its own 2-layer MLP projection; second Linear is
            # zero-initialized so step-0 forward equals the no-hint
            # baseline (same pattern as self_conditioning / stage1_coarse
            # branches in V12InputProjection).
            #
            # Naming kept as `oracle_hint_proj` for backward-compatible
            # checkpoint loading from R27 T0-A3 (interaction-only) runs.
            self._build_oracle_hint_branches(cfg)

    # ------------------------------------------------------------------
    # Round-27/28 oracle hint branch builder + helpers
    # ------------------------------------------------------------------

    def _build_oracle_hint_branches(self, cfg) -> None:
        """Construct interaction + body-action hint MLPs, gates, and
        optional per-layer adapters. Names are arranged so that legacy
        R27 T0-A3 checkpoints (which only have ``oracle_hint_proj.*``)
        still load correctly via ``partial_init_allow_shape_mismatch``.
        """
        # --- Interaction-hint branch ----------------------------------
        if cfg.use_oracle_interaction_hint:
            if cfg.oracle_hint_dim <= 0:
                raise ValueError(
                    "use_oracle_interaction_hint=True requires "
                    f"oracle_hint_dim > 0; got {cfg.oracle_hint_dim}"
                )
            self.oracle_hint_proj = nn.Sequential(
                nn.Linear(cfg.oracle_hint_dim, cfg.d_model),
                nn.SiLU(),
                nn.Linear(cfg.d_model, cfg.d_model),
            )
            nn.init.zeros_(self.oracle_hint_proj[-1].weight)
            nn.init.zeros_(self.oracle_hint_proj[-1].bias)
        else:
            self.oracle_hint_proj = None

        # --- Body-action hint branch (Round-28) -----------------------
        if cfg.use_body_action_hint:
            if cfg.body_action_hint_dim <= 0:
                raise ValueError(
                    "use_body_action_hint=True requires "
                    f"body_action_hint_dim > 0; got {cfg.body_action_hint_dim}"
                )
            self.body_action_hint_proj = nn.Sequential(
                nn.Linear(cfg.body_action_hint_dim, cfg.d_model),
                nn.SiLU(),
                nn.Linear(cfg.d_model, cfg.d_model),
            )
            nn.init.zeros_(self.body_action_hint_proj[-1].weight)
            nn.init.zeros_(self.body_action_hint_proj[-1].bias)
        else:
            self.body_action_hint_proj = None

        # --- Gated-input injection gates ------------------------------
        # Sigmoid gate driven by [c_summary (D), hint_emb (D)] -> 1 scalar
        # gate per (B, T, 1). When ``oracle_hint_injection_mode ==
        # "gated_input"``, the gate modulates the additive injection.
        # Historical default -3.0 is conservative/nearly closed;
        # A1b uses -1.0 for a less cold-started gate.
        if cfg.oracle_hint_injection_mode == "gated_input":
            gate_bias = float(cfg.oracle_hint_gate_bias_init)
            if self.oracle_hint_proj is not None:
                self.interaction_gate = nn.Linear(2 * cfg.d_model, 1)
                if cfg.zero_init_hint_adapters:
                    nn.init.zeros_(self.interaction_gate.weight)
                    nn.init.constant_(self.interaction_gate.bias, gate_bias)
            else:
                self.interaction_gate = None
            if self.body_action_hint_proj is not None:
                self.body_action_gate = nn.Linear(2 * cfg.d_model, 1)
                if cfg.zero_init_hint_adapters:
                    nn.init.zeros_(self.body_action_gate.weight)
                    nn.init.constant_(self.body_action_gate.bias, gate_bias)
            else:
                self.body_action_gate = None
        else:
            self.interaction_gate = None
            self.body_action_gate = None

        # --- Per-layer adapters (Commit 3) ----------------------------
        # One small adapter per DiT block per branch. Each is
        # zero-initialized at the final Linear so the per-layer
        # contribution starts at 0 (preserves R27 ckpt forward).
        if cfg.oracle_hint_injection_mode in ("per_layer_adapter", "adapter_only"):
            n_layers = int(cfg.n_layers)
            if self.oracle_hint_proj is not None:
                self.interaction_adapters = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(cfg.d_model, cfg.d_model),
                        nn.SiLU(),
                        nn.Linear(cfg.d_model, cfg.d_model),
                    )
                    for _ in range(n_layers)
                ])
                if cfg.zero_init_hint_adapters:
                    for ad in self.interaction_adapters:
                        nn.init.zeros_(ad[-1].weight)
                        nn.init.zeros_(ad[-1].bias)
            else:
                self.interaction_adapters = None
            if self.body_action_hint_proj is not None:
                self.body_action_adapters = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(cfg.d_model, cfg.d_model),
                        nn.SiLU(),
                        nn.Linear(cfg.d_model, cfg.d_model),
                    )
                    for _ in range(n_layers)
                ])
                if cfg.zero_init_hint_adapters:
                    for ad in self.body_action_adapters:
                        nn.init.zeros_(ad[-1].weight)
                        nn.init.zeros_(ad[-1].bias)
            else:
                self.body_action_adapters = None
        else:
            self.interaction_adapters = None
            self.body_action_adapters = None

        # Cached most-recent hint embeddings (B, T, D), populated by the
        # input-injection helper for use by per-layer adapters. Reset to
        # None at the start of every forward pass.
        self._interaction_hint_emb_cache = None
        self._body_action_hint_emb_cache = None
        # Scalar diagnostics from the most recent forward. The trainer
        # reads this after ``model(...)`` and writes the values to
        # metrics.jsonl / wandb. Values are detached tensors on-device.
        self._last_oracle_hint_stats: dict[str, Tensor] = {}

    @staticmethod
    def _hint_scalar(x: Tensor) -> Tensor:
        return x.detach().float()

    @staticmethod
    def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
        """Mean of ``values`` over a boolean (B,T) mask, 0 when empty."""
        values_f = values.detach().float()
        mask_f = mask.detach().float()
        denom = mask_f.sum().clamp_min(1.0)
        return (values_f * mask_f).sum() / denom

    @staticmethod
    def _mean_norm(x: Tensor) -> Tensor:
        return x.detach().float().norm(dim=-1).mean()

    def _set_oracle_hint_stat(self, key: str, value: Tensor) -> None:
        self._last_oracle_hint_stats[f"r28_{key}"] = self._hint_scalar(value)

    def _apply_oracle_hint_input_injection(
        self,
        h: Tensor,
        cond: dict,
        c_summary: Tensor | None = None,
    ) -> Tensor:
        """Compute interaction + body-action hint embeddings and inject
        them into the per-frame embedding ``h`` per the configured
        ``oracle_hint_injection_mode``. Caches the embeddings for
        per-layer adapter consumption.

        ``c_summary`` is the AdaLN cond summary (B, D). Required only
        when ``injection_mode == "gated_input"`` (used as gate input
        alongside the hint embedding).
        """
        cfg = self.cfg
        self._interaction_hint_emb_cache = None
        self._body_action_hint_emb_cache = None
        self._last_oracle_hint_stats = {}

        # --- Interaction hint ----------------------------------------
        interaction_emb = None
        interaction_hint = None
        if self.oracle_hint_proj is not None:
            if "oracle_interaction_hint" not in cond:
                raise KeyError(
                    "use_oracle_interaction_hint=True but "
                    "cond['oracle_interaction_hint'] is missing. The "
                    "trainer must populate this from "
                    "batch['oracle_interaction_hint'] (set "
                    "data.use_oracle_interaction_hint=true in the config)."
                )
            ih = cond["oracle_interaction_hint"]                          # (B, T, D_hint)
            interaction_hint = ih
            interaction_emb = self.oracle_hint_proj(ih)                   # (B, T, D)
            self._interaction_hint_emb_cache = interaction_emb
            self._set_oracle_hint_stat(
                "interaction_hint_norm", self._mean_norm(ih),
            )
            self._set_oracle_hint_stat(
                "interaction_emb_norm", self._mean_norm(interaction_emb),
            )
            if ih.shape[-1] >= 2:
                contact_mask = ih[..., :2].amax(dim=-1) > 0.5
                self._set_oracle_hint_stat(
                    "interaction_contact_frame_frac",
                    contact_mask.float().mean(),
                )
            if ih.shape[-1] >= 13:
                walking_mask = ih[..., 12] > 0.5
                self._set_oracle_hint_stat(
                    "interaction_walking_frame_frac",
                    walking_mask.float().mean(),
                )

        # --- Body-action hint (R28) ----------------------------------
        body_emb = None
        body_hint = None
        if self.body_action_hint_proj is not None:
            if "body_action_hint" not in cond:
                raise KeyError(
                    "use_body_action_hint=True but "
                    "cond['body_action_hint'] is missing. The trainer "
                    "must populate this from batch['body_action_hint'] "
                    "(set data.use_body_action_hint=true in the config)."
                )
            bh = cond["body_action_hint"]                                  # (B, T, 24)
            body_hint = bh
            body_emb = self.body_action_hint_proj(bh)                      # (B, T, D)
            self._body_action_hint_emb_cache = body_emb
            self._set_oracle_hint_stat(
                "body_action_hint_norm", self._mean_norm(bh),
            )
            self._set_oracle_hint_stat(
                "body_action_emb_norm", self._mean_norm(body_emb),
            )
            if bh.shape[-1] >= 24:
                joint_mask = bh[..., :6]
                body_delta = bh[..., 6:24].reshape(*bh.shape[:2], 6, 3)
                active_frame = body_delta.detach().float().norm(dim=-1).amax(dim=-1) > 0.01
                self._set_oracle_hint_stat(
                    "body_action_joint_mask_rate", joint_mask.float().mean(),
                )
                self._set_oracle_hint_stat(
                    "body_action_active_frame_frac", active_frame.float().mean(),
                )

        if interaction_emb is None and body_emb is None:
            return h

        mode = cfg.oracle_hint_injection_mode
        if mode in ("input_add", "per_layer_adapter"):
            # ``per_layer_adapter`` still does the (zero-init) input
            # injection so behavior matches input_add at step 0 — the
            # adapters add their own per-layer contributions on top.
            if interaction_emb is not None:
                h = h + interaction_emb
            if body_emb is not None:
                h = h + body_emb
            return h
        if mode == "adapter_only":
            # Pure per-layer-adapter injection (Round-28 A2b ablation).
            # NO input-token addition; the hint reaches the residual
            # stream only via per-layer zero-init adapters. This isolates
            # the adapter contribution from the input-add baseline so A2
            # vs A2b separates "adapter helps when added on top of
            # input_add" from "adapter alone is sufficient".
            return h
        if mode == "gated_input":
            if interaction_emb is not None and self.interaction_gate is not None:
                gate_input = torch.cat(
                    [interaction_emb, h], dim=-1,
                )                                                        # (B, T, 2D)
                g = torch.sigmoid(self.interaction_gate(gate_input))     # (B, T, 1)
                g_bt = g.squeeze(-1)
                self._set_oracle_hint_stat("interaction_gate_mean", g_bt.mean())
                self._set_oracle_hint_stat(
                    "interaction_gate_std", g_bt.float().std(unbiased=False),
                )
                if interaction_hint is not None and interaction_hint.shape[-1] >= 2:
                    contact_mask = interaction_hint[..., :2].amax(dim=-1) > 0.5
                    self._set_oracle_hint_stat(
                        "interaction_gate_contact_mean",
                        self._masked_mean(g_bt, contact_mask),
                    )
                    self._set_oracle_hint_stat(
                        "interaction_gate_noncontact_mean",
                        self._masked_mean(g_bt, ~contact_mask),
                    )
                if interaction_hint is not None and interaction_hint.shape[-1] >= 13:
                    walking_mask = interaction_hint[..., 12] > 0.5
                    self._set_oracle_hint_stat(
                        "interaction_gate_walking_mean",
                        self._masked_mean(g_bt, walking_mask),
                    )
                    self._set_oracle_hint_stat(
                        "interaction_gate_nonwalking_mean",
                        self._masked_mean(g_bt, ~walking_mask),
                    )
                h = h + g * interaction_emb
            if body_emb is not None and self.body_action_gate is not None:
                gate_input = torch.cat(
                    [body_emb, h], dim=-1,
                )                                                        # (B, T, 2D)
                g = torch.sigmoid(self.body_action_gate(gate_input))     # (B, T, 1)
                g_bt = g.squeeze(-1)
                self._set_oracle_hint_stat("body_action_gate_mean", g_bt.mean())
                self._set_oracle_hint_stat(
                    "body_action_gate_std", g_bt.float().std(unbiased=False),
                )
                if body_hint is not None and body_hint.shape[-1] >= 24:
                    body_delta = body_hint[..., 6:24].reshape(*body_hint.shape[:2], 6, 3)
                    active_frame = body_delta.detach().float().norm(dim=-1).amax(dim=-1) > 0.01
                    self._set_oracle_hint_stat(
                        "body_action_gate_active_mean",
                        self._masked_mean(g_bt, active_frame),
                    )
                    self._set_oracle_hint_stat(
                        "body_action_gate_inactive_mean",
                        self._masked_mean(g_bt, ~active_frame),
                    )
                h = h + g * body_emb
            return h
        raise ValueError(
            f"oracle_hint_injection_mode={mode!r} not in "
            "{'input_add', 'gated_input', 'per_layer_adapter', 'adapter_only'}"
        )

    def _apply_oracle_hint_per_layer_adapter(
        self,
        seq: Tensor,
        layer_idx: int,
        motion_token_start: int,
    ) -> Tensor:
        """Add interaction + body-action adapter outputs to the motion-token
        slice of ``seq`` at the given DiT block index. No-op when adapter
        mode is disabled or no hint is active. Called inside the DiT
        encoder loop after each block.

        ``motion_token_start`` is the index of the first motion frame in
        ``seq`` (init_pose prefix occupies positions [0:motion_token_start)).
        """
        if self.cfg.oracle_hint_injection_mode not in (
            "per_layer_adapter", "adapter_only",
        ):
            return seq
        added = False
        if (
            self.interaction_adapters is not None
            and self._interaction_hint_emb_cache is not None
        ):
            delta = self.interaction_adapters[layer_idx](
                self._interaction_hint_emb_cache,
            )                                                              # (B, T, D)
            self._set_oracle_hint_stat(
                f"interaction_adapter_norm_layer{layer_idx}",
                self._mean_norm(delta),
            )
            seq = seq.clone()
            seq[:, motion_token_start:, :] = (
                seq[:, motion_token_start:, :] + delta
            )
            added = True
        if (
            self.body_action_adapters is not None
            and self._body_action_hint_emb_cache is not None
        ):
            delta = self.body_action_adapters[layer_idx](
                self._body_action_hint_emb_cache,
            )
            self._set_oracle_hint_stat(
                f"body_action_adapter_norm_layer{layer_idx}",
                self._mean_norm(delta),
            )
            if not added:
                seq = seq.clone()
            seq[:, motion_token_start:, :] = (
                seq[:, motion_token_start:, :] + delta
            )
        return seq

    def _compute_plan_xattn_dist_norm(
        self,
        plan_dict: dict,
        plan_tokens_shape: tuple,           # (B, K_total, D)
        T: int,
        motion_token_start: int,
        seq_total_len: int,
        device: torch.device,
    ) -> Tensor:
        """Build the (B, T_q, K_total) per-token motion-time-distance
        tensor used by the Round-23 ALiBi-style plan cross-attention
        bias. Returns distances normalized to [0, 1] by T.

        Padded plan-token slots get distance=0 (they are masked out by
        ``key_padding_mask`` anyway, so the value doesn't affect the
        softmax). The init_pose prefix token (position 0 in the motion
        sequence) gets distance=0 to all anchors (it's a scene fact,
        not a time-indexed motion frame).
        """
        cfg = self.cfg
        B = plan_tokens_shape[0]
        K_total = plan_tokens_shape[1]
        K_max = int(cfg.plan_k_max)
        P = int(cfg.plan_num_parts)
        S_max = int(cfg.plan_s_max)

        # Plan-token times: anchor part first (per_part_tokens or per-anchor),
        # then segment tokens if enabled.
        anchor_time = plan_dict["anchor_time"].to(device).float()        # (B, K_max)
        if cfg.plan_per_part_tokens:
            # Row-major (k, p) flatten; all P parts share anchor k's time.
            anchor_token_time = (
                anchor_time.view(B, K_max, 1).expand(B, K_max, P).reshape(B, K_max * P)
            )                                                            # (B, K_max*P)
        else:
            anchor_token_time = anchor_time                              # (B, K_max)

        if cfg.plan_use_segment_tokens:
            # Use segment start as the representative time.
            segment_time = plan_dict["segment_start"].to(device).float() # (B, S_max)
            plan_token_time = torch.cat([anchor_token_time, segment_time], dim=1)
        else:
            plan_token_time = anchor_token_time

        # Sanity: shape should match K_total from the encoder output.
        if plan_token_time.shape[1] != K_total:
            raise ValueError(
                f"plan_token_time width {plan_token_time.shape[1]} != "
                f"K_total {K_total} (encoder layout mismatch)"
            )

        # Motion-time for each query position. The sequence is
        # [init_pose_tok, motion_frame_0, motion_frame_1, ...]. Init_pose
        # uses motion_time=0 (treated as "no temporal preference"); each
        # motion frame uses its frame index.
        motion_time = torch.zeros(seq_total_len, device=device, dtype=torch.float32)
        motion_time[motion_token_start:] = torch.arange(
            T, device=device, dtype=torch.float32,
        )                                                                # (T_q,)

        # Distance |t_q - plan_time_k| / T, broadcast.
        # motion_time: (T_q,) → (1, T_q, 1)
        # plan_token_time: (B, K) → (B, 1, K)
        dist = (
            motion_time.view(1, seq_total_len, 1)
            - plan_token_time.view(B, 1, K_total)
        ).abs() / max(T, 1)                                              # (B, T_q, K)

        # Zero out the init_pose prefix row so it doesn't get any
        # time-distance preference (it's a scene fact, not a frame).
        init_pose_mask = torch.zeros(seq_total_len, device=device, dtype=torch.bool)
        init_pose_mask[:motion_token_start] = True
        dist = torch.where(
            init_pose_mask.view(1, seq_total_len, 1),
            torch.zeros_like(dist), dist,
        )
        return dist

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
        self_cond: Tensor | None = None,
    ) -> Tensor:
        B, T, _ = x_t.shape
        cfg = self.cfg
        if self_cond is not None and self_cond.shape != x_t.shape:
            raise ValueError(
                f"self_cond shape {tuple(self_cond.shape)} must match x_t "
                f"shape {tuple(x_t.shape)}"
            )

        z_int: Tensor = cond["z_int"]                  # (B, T, zint_total)
        obj_traj: Tensor = cond["object_world_traj"]   # (B, T, 9)
        init_pose: Tensor = cond["init_pose"]          # (B, init_pose_dim)
        text_tok: Tensor = cond["text"]                # (B, L_text, text_dim)
        obj_tok: Tensor = cond["object_tokens"]        # (B, N_obj, object_token_dim)

        # --- CFG drop: replace conditioning channels with null embeddings ---
        z_int_eff = self._broadcast_drop(cond_drop_mask, z_int, self.null_zint)
        obj_traj_eff = self._broadcast_drop(cond_drop_mask, obj_traj, self.null_obj_traj)

        # ─── v12 path (use_dit_block=True): DiT/InterGen-style forward ───
        if cfg.use_dit_block:
            return self._forward_v12(
                x_t=x_t, t=t, z_int_eff=z_int_eff, obj_traj_eff=obj_traj_eff,
                init_pose=init_pose, text_tok=text_tok, obj_tok=obj_tok,
                cond=cond, cond_drop_mask=cond_drop_mask,
                self_cond=self_cond, T=T,
            )

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

    def _forward_v12(
        self,
        x_t: Tensor,
        t: Tensor,
        z_int_eff: Tensor,
        obj_traj_eff: Tensor,
        init_pose: Tensor,
        text_tok: Tensor,
        obj_tok: Tensor,
        cond: dict,
        cond_drop_mask: Tensor | None,
        self_cond: Tensor | None,
        T: int,
    ) -> Tensor:
        """v12 forward path — per analyses/2026-05-11_v12_architecture_design_doc.md §5."""
        cfg = self.cfg
        B = x_t.shape[0]

        # ─── Plan encoder (carryover from v11) ───
        plan_dict = cond.get("interaction_plan", None)
        if plan_dict is None:
            raise KeyError(
                "v12 (use_dit_block=True) requires cond['interaction_plan']"
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
        if cfg.plan_tokens_force_null:
            # Round-23 no-plan ablation: unconditionally replace plan signals
            # with their learned null embeddings. Cross-attn key_padding_mask
            # stays valid so MHA doesn't degenerate to NaN (the constant K/V
            # yields a constant attention output, contributing zero per-frame
            # discrimination — equivalent to removing the plan condition).
            plan_tokens = self.null_plan_token.expand_as(plan_tokens)
            if plan_hint is not None and hasattr(self, "null_plan_hint"):
                plan_hint = self.null_plan_hint.expand_as(plan_hint)

        # ─── Timestep embedding (B, D) ───
        t_emb = self.time_embed(t)                                       # (B, D)

        # ─── §4.3 Input projection (per-channel summed, residual stream) ───
        # Per-frame conditions flow through here; v11's bandwidth bottleneck
        # (single Linear(217,512)) is replaced by 4 (Round-22: 5) projections.
        stage1_coarse_eff: Tensor | None = None
        if cfg.stage1_coarse_dim > 0:
            if "stage1_coarse" not in cond:
                raise KeyError(
                    "stage1_coarse_dim > 0 requires cond['stage1_coarse'] "
                    "(B, T, stage1_coarse_dim). The trainer must populate "
                    "this from oracle GT extraction or the S1-O sampler."
                )
            stage1_coarse_eff = cond["stage1_coarse"]
            if cfg.cfg_drop_stage1_coarse:
                stage1_coarse_eff = self._broadcast_drop(
                    cond_drop_mask, stage1_coarse_eff, self.null_stage1_coarse,
                )
        h = self.v12_input_proj(
            x_t=x_t,
            z_int=z_int_eff,
            obj_traj=obj_traj_eff,
            plan_hint=plan_hint,
            self_cond=self_cond,
            stage1_coarse=stage1_coarse_eff,
        )                                                                # (B, T, D)

        # ─── Round-27/28: oracle hint input injection ───
        # The Round-27 input-add path is preserved via injection_mode
        # ="input_add". Round-28 adds gated_input + per_layer_adapter
        # modes and the separate body-action branch. The hint MLPs are
        # zero-initialized at their last layer so the step-0 forward is
        # bit-exact equal to the no-hint baseline.
        h = self._apply_oracle_hint_input_injection(h, cond, c_summary=None)

        # ─── §4.4 Global condition vector for AdaLN (per-sample) ───
        # c = t_emb + plan_pool_emb. Drives AdaLN modulation in every block
        # and in the final layer. Per-sample = same modulation across all T frames.
        c = self.v12_cond_summary(
            t_emb=t_emb, plan_tokens=plan_tokens, plan_mask=plan_mask,
        )                                                                # (B, D)

        # ─── Prepend init_pose token (time_tok dropped — timestep is in AdaLN) ───
        pose_tok = self.pose_proj(init_pose).unsqueeze(1)                # (B, 1, D)
        seq = torch.cat([pose_tok, h], dim=1)                            # (B, T+1, D)
        seq = self.pos_enc(seq)

        # ─── §4.5 ConditionedEncoderLayer × n_layers ───
        # PyTorch nn.MultiheadAttention key_padding_mask: True = ignore.
        # plan_mask is True at valid positions → invert.
        plan_kpm = ~plan_mask.bool()                                     # (B, K)

        # Round-23: optional ALiBi-style relative-time bias for plan
        # cross-attention. See AnchorDenoiserConfig.plan_xattn_relative_time_bias.
        # Computed once per forward (independent of layer) modulo the
        # per-layer learnable slopes.
        if cfg.plan_xattn_relative_time_bias:
            dist_norm = self._compute_plan_xattn_dist_norm(
                plan_dict=plan_dict,
                plan_tokens_shape=plan_tokens.shape,
                T=T,
                motion_token_start=1,
                seq_total_len=seq.shape[1],
                device=seq.device,
            )                                                            # (B, T_q, K) in [0, 1]
        else:
            dist_norm = None

        n_heads_for_bias = int(cfg.n_heads)
        for layer_idx, block in enumerate(self.v12_blocks):
            if dist_norm is not None:
                slopes_L = self.plan_xattn_time_bias_slopes[layer_idx]     # (n_heads,)
                # bias: (B, n_heads, T_q, K) → reshape to (B*n_heads, T_q, K)
                bias = (
                    -slopes_L.view(1, n_heads_for_bias, 1, 1)
                    * dist_norm.unsqueeze(1)                                # (B, 1, T_q, K)
                )                                                           # (B, n_heads, T_q, K)
                B_bias, _, Tq_bias, K_bias = bias.shape
                bias = bias.reshape(B_bias * n_heads_for_bias, Tq_bias, K_bias)
            else:
                bias = None
            seq = block(
                seq, c, plan_tokens, plan_kpm,
                plan_xattn_attn_bias=bias,
            )
            # Round-28: optional per-layer zero-init adapters on the
            # interaction + body-action hint embeddings. No-op when
            # injection_mode != "per_layer_adapter". motion_token_start=1
            # because the sequence is [pose_tok, frame_0, frame_1, ...].
            seq = self._apply_oracle_hint_per_layer_adapter(
                seq, layer_idx=layer_idx, motion_token_start=1,
            )

        # ─── Text + obj cross-attn at end of encoder (v11 carryover) ───
        text_kv = self.text_proj(text_tok)
        text_kv = self._broadcast_drop(cond_drop_mask, text_kv, self.null_text)
        text_attn, _ = self.text_xattn(seq, text_kv, text_kv, need_weights=False)
        seq = self.text_norm(seq + text_attn)

        obj_kv = self.object_proj(obj_tok)
        obj_kv = self._broadcast_drop(cond_drop_mask, obj_kv, self.null_obj_tokens)
        obj_attn, _ = self.obj_xattn(seq, obj_kv, obj_kv, need_weights=False)
        seq = self.obj_norm(seq + obj_attn)

        # ─── §4.6 Final layer (drop pose prefix token, AdaLN-Zero readout) ───
        h_motion = seq[:, 1:, :]                                         # (B, T, D)
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
                z_t, t_idx, cond, cond_drop_mask=drop_mask, self_cond=None,
            )
            denoiser_cfg = self.cfg.denoiser
            x0_pred = _apply_observed_x0_skip(
                x0_raw, cond,
                motion_dim=denoiser_cfg.motion_dim,
                enabled=bool(denoiser_cfg.cond_motion_output_skip),
            )
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
                "self_conditioning_used_fraction": torch.zeros((), device=device),
                "self_conditioning_allowed_fraction": torch.zeros((), device=device),
            }

        t = torch.randint(0, self.diffusion.num_steps, (B,), device=device)
        noise = torch.randn_like(x_start)
        x_t = self.diffusion.q_sample(x_start, t, noise)

        # CFG dropout: per-sample bernoulli over the full conditioning.
        drop_mask = torch.rand(B, device=device) < self.cfg.cfg_drop_prob
        target = self.cfg.diffusion.prediction_target
        denoiser_cfg = self.cfg.denoiser
        self_cond: Tensor | None = None
        sc_used_mask = torch.zeros(B, device=device, dtype=torch.bool)
        sc_allowed_mask = torch.zeros(B, device=device, dtype=torch.bool)
        if (
            bool(denoiser_cfg.use_self_conditioning)
            and float(denoiser_cfg.self_conditioning_prob) > 0.0
        ):
            mode = str(denoiser_cfg.self_conditioning_mode)
            if mode == "standard":
                sc_allowed_mask = torch.ones(B, device=device, dtype=torch.bool)
            elif mode == "late_start":
                sc_allowed_mask = t <= int(denoiser_cfg.self_conditioning_t_max)
            else:
                raise ValueError(
                    "self_conditioning_mode must be 'standard' or 'late_start', "
                    f"got {mode!r}"
                )
            sc_used_mask = (
                torch.rand(B, device=device) < float(denoiser_cfg.self_conditioning_prob)
            ) & sc_allowed_mask
            if sc_used_mask.any():
                # Training self_cond is generated by a no-grad first pass at
                # the same x_t and t. Inference self_cond comes from the
                # previous reverse step, so this is standard self-conditioning
                # but only an approximation to true rollout conditioning.
                with torch.no_grad():
                    sc_raw = self.denoiser(
                        x_t, t, cond, cond_drop_mask=None, self_cond=None,
                    )
                    if target == "v":
                        sc_x0 = self.diffusion.predict_x0_from_v(x_t, t, sc_raw)
                    else:
                        sc_x0 = sc_raw
                    self_cond = torch.zeros_like(x_start)
                    self_cond[sc_used_mask] = sc_x0.detach()[sc_used_mask]

        net_out = self.denoiser(
            x_t, t, cond, cond_drop_mask=drop_mask, self_cond=self_cond,
        )

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
            "self_conditioning_used_fraction": sc_used_mask.float().mean(),
            "self_conditioning_allowed_fraction": sc_allowed_mask.float().mean(),
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
        if self.diffusion.objective == "rectified_flow" or sampler.startswith("rectified_flow"):
            return self.diffusion.rf_sample_loop(
                self.denoiser, shape, cond, cfg_scale=cfg_scale,
                device=cond["z_int"].device, output_skip=output_skip,
                sampler_type=sampler if sampler.startswith("rectified_flow") else None,
            )
        return self.diffusion.p_sample_loop(
            self.denoiser, shape, cond, cfg_scale=cfg_scale,
            replacement=replacement, output_skip=output_skip,
            sampler=sampler,
        )
