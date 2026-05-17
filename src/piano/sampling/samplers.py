"""Config-driven samplers for Stage-B AnchorDiff models.

The model interface is x0-prediction (or v-prediction converted to x0), so the
samplers here adapt DDIM/DDPM and a small k-diffusion-inspired sigma-space
family to the current VP cosine buffers. External repos are reference-only:
no code is imported from them at runtime.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from piano.models.motion_anchordiff import _extract


@dataclass(frozen=True, slots=True)
class SamplerConfig:
    name: str
    sampler_type: str = "ddpm"
    num_sampling_steps: int = 1000
    sampling_schedule: str = "cosine_default"
    ddim_eta: float = 0.0
    karras_rho: float = 7.0
    sde_noise_scale: float = 1.0
    seed: int = 1234


def default_sampler_sweep(include_dpmpp: bool = True, minimum_only: bool = False) -> list[SamplerConfig]:
    """Return the named sampler variants for the v18 refinement sweep."""
    variants = [
        SamplerConfig("A0_ddpm_1000_cosine_default", "ddpm", 1000, "cosine_default", 1.0),
        SamplerConfig("A1_ddim_eta0_250_logit_normal", "ddim", 250, "logit_normal", 0.0),
        SamplerConfig("A2_ddim_eta0_250_logsnr_uniform", "ddim", 250, "logsnr_uniform", 0.0),
        SamplerConfig("A3_ddim_eta0_500_logit_normal", "ddim", 500, "logit_normal", 0.0),
        SamplerConfig("A4_ddim_eta0_500_logsnr_uniform", "ddim", 500, "logsnr_uniform", 0.0),
        SamplerConfig("A5_ddim_eta0_250_mild_logit_normal", "ddim", 250, "mild_logit_normal", 0.0),
        SamplerConfig("A6_ddim_eta0_500_mild_logit_normal", "ddim", 500, "mild_logit_normal", 0.0),
        SamplerConfig("A7_ddim_eta0_250_cosine_subsample", "ddim", 250, "cosine_subsample", 0.0),
        SamplerConfig("A8_ddim_eta0_500_cosine_subsample", "ddim", 500, "cosine_subsample", 0.0),
        SamplerConfig("A9_ddim_eta0p2_250_logsnr_uniform", "ddim", 250, "logsnr_uniform", 0.2),
        SamplerConfig("A10_ddim_eta0p5_250_logsnr_uniform", "ddim", 250, "logsnr_uniform", 0.5),
        SamplerConfig("A11_ddim_eta0p2_500_logsnr_uniform", "ddim", 500, "logsnr_uniform", 0.2),
        SamplerConfig("A12_ddim_eta0p5_500_logsnr_uniform", "ddim", 500, "logsnr_uniform", 0.5),
    ]
    if include_dpmpp:
        variants.extend([
            SamplerConfig("A13_dpmpp_2m_sde_250_karras", "dpmpp_2m_sde", 250, "karras", 0.0, sde_noise_scale=0.3),
            SamplerConfig("A14_dpmpp_2m_sde_500_karras", "dpmpp_2m_sde", 500, "karras", 0.0, sde_noise_scale=0.3),
            SamplerConfig("A15_dpmpp_2m_250_karras", "dpmpp_2m", 250, "karras", 0.0),
            SamplerConfig("A16_dpmpp_2m_500_karras", "dpmpp_2m", 500, "karras", 0.0),
        ])
    if not minimum_only:
        return variants
    keep = {
        "A0_ddpm_1000_cosine_default",
        "A1_ddim_eta0_250_logit_normal",
        "A2_ddim_eta0_250_logsnr_uniform",
        "A3_ddim_eta0_500_logit_normal",
        "A5_ddim_eta0_250_mild_logit_normal",
        "A7_ddim_eta0_250_cosine_subsample",
        "A9_ddim_eta0p2_250_logsnr_uniform",
        "A13_dpmpp_2m_sde_250_karras",
    }
    return [v for v in variants if v.name in keep]


def _normal_icdf_quantiles(n: int) -> np.ndarray:
    ps = torch.linspace(0.5 / n, 1.0 - 0.5 / n, n, dtype=torch.float64)
    return torch.distributions.Normal(0.0, 1.0).icdf(ps).cpu().numpy()


def _unique_descending_timesteps(raw: np.ndarray, max_t: int) -> list[int]:
    idx = [int(np.clip(round(float(v)), 0, max_t)) for v in raw]
    idx = sorted(set(idx), reverse=True)
    if not idx or idx[0] != max_t:
        idx.insert(0, max_t)
    if idx[-1] != 0:
        idx.append(0)
    return idx


def make_timestep_list(
    diffusion: Any,
    config: SamplerConfig,
    *,
    seed: int | None = None,
) -> list[int]:
    """Build descending discrete DDPM timestep indices for a sampler config."""
    max_t = int(diffusion.num_steps) - 1
    steps = max(2, int(config.num_sampling_steps))
    schedule = str(config.sampling_schedule)
    if config.sampler_type == "ddpm" and schedule == "cosine_default" and steps >= diffusion.num_steps:
        return list(range(max_t, -1, -1))

    if schedule in ("cosine_default", "cosine_subsample", "uniform"):
        raw = np.linspace(max_t, 0, steps)
    elif schedule == "low_noise_dense":
        u = np.linspace(0.0, 1.0, steps)
        raw = max_t * (1.0 - u) ** 2
    elif schedule == "high_noise_dense":
        u = np.linspace(0.0, 1.0, steps)
        raw = max_t * (1.0 - u ** 2)
    elif schedule == "logsnr_uniform":
        alpha = diffusion.alphas_cumprod.detach().cpu().numpy().astype(np.float64)
        logsnr = np.log(np.maximum(alpha, 1e-30)) - np.log(np.maximum(1.0 - alpha, 1e-30))
        targets = np.linspace(logsnr[max_t], logsnr[0], steps)
        raw = np.array([int(np.argmin(np.abs(logsnr - v))) for v in targets], dtype=np.int64)
    elif schedule in ("logit_normal", "mild_logit_normal"):
        if schedule == "logit_normal":
            mean, std = -1.5, 0.8
        else:
            mean, std = -1.0, 0.55
        z = _normal_icdf_quantiles(max(2, steps - 2))
        inner = 1.0 / (1.0 + np.exp(-(mean + std * z)))
        raw = np.concatenate([[max_t], np.sort((1.0 - inner) * max_t)[::-1], [0]])
    elif schedule == "karras":
        sigmas = make_sigma_schedule(diffusion, config, device=diffusion.betas.device)
        raw = sigma_to_timestep(diffusion, sigmas).detach().cpu().numpy()
    else:
        raise ValueError(f"Unknown sampling_schedule={schedule!r}")
    return _unique_descending_timesteps(raw, max_t)


def alpha_to_sigma(alpha: Tensor) -> Tensor:
    return ((1.0 - alpha).clamp_min(0.0) / alpha.clamp_min(1e-12)).sqrt()


def sigma_to_alpha(sigma: Tensor) -> Tensor:
    return 1.0 / (1.0 + sigma.square())


def sigma_to_timestep(diffusion: Any, sigma: Tensor) -> Tensor:
    sigma_table = alpha_to_sigma(diffusion.alphas_cumprod).to(device=sigma.device, dtype=sigma.dtype)
    flat = sigma.reshape(-1)
    idx = torch.argmin((sigma_table.view(1, -1) - flat.view(-1, 1)).abs(), dim=1)
    return idx.reshape(sigma.shape).long()


def make_sigma_schedule(diffusion: Any, config: SamplerConfig, *, device: torch.device) -> Tensor:
    """Karras-like sigma schedule for VP buffers, ending at sigma=0."""
    n = max(2, int(config.num_sampling_steps))
    alpha = diffusion.alphas_cumprod.detach().to(device=device).float()
    sigma_table = alpha_to_sigma(alpha)
    sigma_max = float(sigma_table[-1].item())
    # The first training step has tiny but nonzero noise. DPM++ final denoise
    # is cleaner with an explicit zero appended, as in k-diffusion.
    sigma_min = float(max(sigma_table[0].item(), 1e-4))
    ramp = torch.linspace(0.0, 1.0, n, device=device)
    rho = float(config.karras_rho)
    min_inv = sigma_min ** (1.0 / rho)
    max_inv = sigma_max ** (1.0 / rho)
    sigmas = (max_inv + ramp * (min_inv - max_inv)) ** rho
    return torch.cat([sigmas, sigmas.new_zeros(1)])


def _predict_x0(
    diffusion: Any,
    denoiser: Any,
    x_t: Tensor,
    t: Tensor,
    cond: dict[str, Any],
    cfg_scale: float,
) -> Tensor:
    pred_cond = denoiser(x_t, t, cond, cond_drop_mask=None, self_cond=None)
    if float(cfg_scale) != 1.0:
        drop = torch.ones(x_t.shape[0], dtype=torch.bool, device=x_t.device)
        pred_uncond = denoiser(x_t, t, cond, cond_drop_mask=drop, self_cond=None)
    else:
        pred_uncond = None

    if diffusion.prediction_target == "v":
        x0_cond = diffusion.predict_x0_from_v(x_t, t, pred_cond)
        x0_uncond = diffusion.predict_x0_from_v(x_t, t, pred_uncond) if pred_uncond is not None else None
    else:
        x0_cond = pred_cond
        x0_uncond = pred_uncond
    if x0_uncond is None:
        return x0_cond
    return x0_uncond + float(cfg_scale) * (x0_cond - x0_uncond)


def _record_nearest_log(
    logs: dict[int, tuple[int, Tensor]],
    log_timesteps: set[int],
    t_int: int,
    x0: Tensor,
) -> None:
    for target in log_timesteps:
        diff = abs(int(t_int) - int(target))
        if target not in logs or diff < logs[target][0]:
            logs[int(target)] = (diff, x0.detach().clone())


@torch.no_grad()
def _sample_ddpm_or_ddim(
    diffusion: Any,
    denoiser: Any,
    shape: tuple[int, ...],
    cond: dict[str, Any],
    config: SamplerConfig,
    *,
    cfg_scale: float,
    device: torch.device,
    log_timesteps: tuple[int, ...],
) -> tuple[Tensor, dict[int, Tensor], dict[str, Any]]:
    x = torch.randn(shape, device=device)
    timesteps = make_timestep_list(diffusion, config)
    best_logs: dict[int, tuple[int, Tensor]] = {}
    log_set = {int(t) for t in log_timesteps}

    if config.sampler_type == "ddpm" and len(timesteps) >= int(diffusion.num_steps):
        for t_int in timesteps:
            t = torch.full((shape[0],), int(t_int), device=device, dtype=torch.long)
            x0 = _predict_x0(diffusion, denoiser, x, t, cond, cfg_scale)
            _record_nearest_log(best_logs, log_set, int(t_int), x0)
            mean = diffusion.posterior_mean_from_x0(x0, x, t)
            if int(t_int) == 0:
                x = mean
            else:
                noise = torch.randn_like(x)
                log_var = _extract(diffusion.posterior_log_variance_clipped, t, x.shape)
                x = mean + (0.5 * log_var).exp() * noise
        logs = {int(k): v for k, (_d, v) in best_logs.items()}
        logs[0] = x.detach().clone()
        return x, logs, {"actual_steps": len(timesteps), "timesteps": timesteps}

    eta = float(config.ddim_eta)
    for i, t_int in enumerate(timesteps[:-1]):
        prev_int = int(timesteps[i + 1])
        t = torch.full((shape[0],), int(t_int), device=device, dtype=torch.long)
        prev = torch.full((shape[0],), prev_int, device=device, dtype=torch.long)
        x0 = _predict_x0(diffusion, denoiser, x, t, cond, cfg_scale)
        _record_nearest_log(best_logs, log_set, int(t_int), x0)
        alpha_t = _extract(diffusion.alphas_cumprod, t, x.shape)
        alpha_prev = _extract(diffusion.alphas_cumprod, prev, x.shape)
        eps = (x - alpha_t.sqrt() * x0) / (1.0 - alpha_t).sqrt().clamp_min(1e-8)
        sigma = (
            eta
            * ((1.0 - alpha_prev) / (1.0 - alpha_t)).sqrt()
            * (1.0 - alpha_t / alpha_prev).clamp_min(0.0).sqrt()
        )
        dir_scale = (1.0 - alpha_prev - sigma.square()).clamp_min(0.0).sqrt()
        noise = torch.randn_like(x) if eta > 0.0 and prev_int > 0 else torch.zeros_like(x)
        x = alpha_prev.sqrt() * x0 + dir_scale * eps + sigma * noise
    logs = {int(k): v for k, (_d, v) in best_logs.items()}
    logs[0] = x.detach().clone()
    return x, logs, {"actual_steps": len(timesteps), "timesteps": timesteps}


def _vp_denoised_from_sigma(
    diffusion: Any,
    denoiser: Any,
    y: Tensor,
    sigma: Tensor,
    cond: dict[str, Any],
    cfg_scale: float,
) -> tuple[Tensor, Tensor]:
    """Predict x0 for sigma-space state y = x_t / sqrt(alpha_t)."""
    sigma_batch = sigma.expand(y.shape[0]).to(device=y.device, dtype=y.dtype)
    alpha = sigma_to_alpha(sigma_batch)
    x_t = y * alpha.sqrt().reshape(y.shape[0], *((1,) * (y.ndim - 1)))
    t = sigma_to_timestep(diffusion, sigma_batch)
    x0 = _predict_x0(diffusion, denoiser, x_t, t, cond, cfg_scale)
    return x0, t


@torch.no_grad()
def _sample_sigma_space(
    diffusion: Any,
    denoiser: Any,
    shape: tuple[int, ...],
    cond: dict[str, Any],
    config: SamplerConfig,
    *,
    cfg_scale: float,
    device: torch.device,
    log_timesteps: tuple[int, ...],
) -> tuple[Tensor, dict[int, Tensor], dict[str, Any]]:
    sigmas = make_sigma_schedule(diffusion, config, device=device)
    y = torch.randn(shape, device=device) * sigmas[0]
    best_logs: dict[int, tuple[int, Tensor]] = {}
    log_set = {int(t) for t in log_timesteps}

    if config.sampler_type == "heun":
        for i in range(len(sigmas) - 1):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            denoised, t = _vp_denoised_from_sigma(diffusion, denoiser, y, sigma.reshape(()), cond, cfg_scale)
            _record_nearest_log(best_logs, log_set, int(t[0].item()), denoised)
            if float(sigma.item()) <= 0.0:
                y = denoised
                continue
            d = (y - denoised) / sigma.clamp_min(1e-8)
            dt = sigma_next - sigma
            if float(sigma_next.item()) <= 0.0:
                y = y + d * dt
            else:
                y_2 = y + d * dt
                denoised_2, _ = _vp_denoised_from_sigma(
                    diffusion, denoiser, y_2, sigma_next.reshape(()), cond, cfg_scale,
                )
                d_2 = (y_2 - denoised_2) / sigma_next.clamp_min(1e-8)
                y = y + 0.5 * (d + d_2) * dt
        logs = {int(k): v for k, (_d, v) in best_logs.items()}
        logs[0] = y.detach().clone()
        return y, logs, {"actual_steps": int(len(sigmas) - 1), "sigmas": sigmas.detach().cpu().tolist()}

    if config.sampler_type not in ("dpmpp_2m", "dpmpp_2m_sde"):
        raise ValueError(f"Unsupported sigma-space sampler_type={config.sampler_type!r}")

    old_denoised: Tensor | None = None
    h_last: Tensor | None = None
    for i in range(len(sigmas) - 1):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        denoised, t = _vp_denoised_from_sigma(diffusion, denoiser, y, sigma.reshape(()), cond, cfg_scale)
        _record_nearest_log(best_logs, log_set, int(t[0].item()), denoised)

        if float(sigma_next.item()) <= 0.0:
            y = denoised
        elif config.sampler_type == "dpmpp_2m":
            t_cur = -sigma.log()
            t_next = -sigma_next.log()
            h = t_next - t_cur
            if old_denoised is None:
                y = (sigma_next / sigma) * y - torch.expm1(-h) * denoised
            else:
                assert h_last is not None
                r = h_last / h
                denoised_d = (1.0 + 1.0 / (2.0 * r)) * denoised - (1.0 / (2.0 * r)) * old_denoised
                y = (sigma_next / sigma) * y - torch.expm1(-h) * denoised_d
            h_last = h
        else:
            t_cur = -sigma.log()
            t_next = -sigma_next.log()
            h = t_next - t_cur
            eta_h = float(config.sde_noise_scale) * h
            y = sigma_next / sigma * torch.exp(-eta_h) * y + (-h - eta_h).expm1().neg() * denoised
            if old_denoised is not None:
                assert h_last is not None
                r = h_last / h
                y = y + 0.5 * (-h - eta_h).expm1().neg() * (1.0 / r) * (denoised - old_denoised)
            if float(config.sde_noise_scale) > 0.0:
                noise_scale = sigma_next * (-2.0 * eta_h).expm1().neg().clamp_min(0.0).sqrt()
                y = y + torch.randn_like(y) * noise_scale
            h_last = h
        old_denoised = denoised

    logs = {int(k): v for k, (_d, v) in best_logs.items()}
    logs[0] = y.detach().clone()
    return y, logs, {"actual_steps": int(len(sigmas) - 1), "sigmas": sigmas.detach().cpu().tolist()}


@torch.no_grad()
def sample_with_config(
    model: Any,
    cond: dict[str, Any],
    seq_length: int,
    config: SamplerConfig,
    *,
    cfg_scale: float = 1.0,
    seed: int | None = None,
    log_timesteps: tuple[int, ...] = (),
) -> tuple[Tensor, dict[int, Tensor], dict[str, Any]]:
    """Sample a Stage-B motion sequence with a registry sampler config.

    Returns final motion, intermediate x0 logs keyed by nearest DDPM timestep,
    and sampler metadata. The function changes no model weights.
    """
    run_seed = int(config.seed if seed is None else seed)
    torch.manual_seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)
    device = cond["z_int"].device
    shape = (cond["z_int"].shape[0], int(seq_length), model.cfg.denoiser.motion_dim)
    sampler_type = str(config.sampler_type)
    if sampler_type in ("ddpm", "ddim"):
        final, logs, meta = _sample_ddpm_or_ddim(
            model.diffusion,
            model.denoiser,
            shape,
            cond,
            config,
            cfg_scale=float(cfg_scale),
            device=device,
            log_timesteps=tuple(int(t) for t in log_timesteps),
        )
    elif sampler_type in ("dpmpp_2m", "dpmpp_2m_sde", "heun"):
        final, logs, meta = _sample_sigma_space(
            model.diffusion,
            model.denoiser,
            shape,
            cond,
            config,
            cfg_scale=float(cfg_scale),
            device=device,
            log_timesteps=tuple(int(t) for t in log_timesteps),
        )
    else:
        raise ValueError(f"Unknown sampler_type={sampler_type!r}")

    meta = {
        **meta,
        "name": config.name,
        "sampler_type": config.sampler_type,
        "num_sampling_steps": int(config.num_sampling_steps),
        "sampling_schedule": config.sampling_schedule,
        "ddim_eta": float(config.ddim_eta),
        "karras_rho": float(config.karras_rho),
        "sde_noise_scale": float(config.sde_noise_scale),
        "seed": run_seed,
    }
    return final, logs, meta
