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


class GaussianDiffusion(nn.Module):
    """ε-prediction DDPM with cosine β-schedule.

    All buffers are float32 and registered, so they move with the module
    under Accelerate / .to(device).
    """

    def __init__(self, cfg: DiffusionConfig) -> None:
        super().__init__()
        self.num_steps = cfg.num_steps
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
    ) -> Tensor:
        """Ancestral sampling with optional classifier-free guidance,
        operating on x₀-prediction outputs.

        ``cond`` follows the same dict spec as ``AnchorDenoiser.forward``.
        ``cfg_scale > 1`` enables CFG: x_0 = x_0_uncond + s·(x_0_cond - x_0_uncond).
        """
        device = device or self.betas.device
        x = torch.randn(shape, device=device)
        for t_int in reversed(range(self.num_steps)):
            t = torch.full((shape[0],), t_int, device=device, dtype=torch.long)
            x0_cond = denoiser(x, t, cond, cond_drop_mask=None)
            if cfg_scale != 1.0:
                drop = torch.ones(shape[0], dtype=torch.bool, device=device)
                x0_uncond = denoiser(x, t, cond, cond_drop_mask=drop)
                x0 = x0_uncond + cfg_scale * (x0_cond - x0_uncond)
            else:
                x0 = x0_cond

            mean = self.posterior_mean_from_x0(x0, x, t)
            if t_int == 0:
                x = mean
            else:
                noise = torch.randn_like(x)
                log_var = _extract(self.posterior_log_variance_clipped, t, x.shape)
                x = mean + (0.5 * log_var).exp() * noise
        return x


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

        per_frame_in = cfg.motion_dim + cfg.z_int.total + cfg.object_traj_dim
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

        # --- Per-frame projection: concat motion + z_int + object_traj ---
        per_frame = torch.cat([x_t, z_int_eff, obj_traj_eff], dim=-1)   # (B, T, in_dim)
        h = self.in_proj(per_frame)                                      # (B, T, D)

        # --- Prepend timestep token + init-pose token ---
        t_tok = self.time_embed(t).unsqueeze(1)                          # (B, 1, D)
        pose_tok = self.pose_proj(init_pose).unsqueeze(1)                # (B, 1, D)
        seq = torch.cat([t_tok, pose_tok, h], dim=1)                     # (B, T+2, D)
        seq = self.pos_enc(seq)

        # --- Self-attention encoder (motion-side) ---
        seq = self.encoder(seq)                                          # (B, T+2, D)

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
        """One forward pass: sample t, sample noise, predict x_0, return
        prediction + ground-truth target (caller computes MSE + anchor loss).

        Under x₀-prediction the denoiser's output IS the predicted
        clean motion. No `predict_x0_from_eps` derivation is needed,
        which means the anchor consistency loss applied on `x0_pred`
        flows directly through the network without the 1/√ᾱ_t
        Jacobian explosion at high noise levels.
        """
        B, T, _ = x_start.shape
        device = x_start.device
        t = torch.randint(0, self.diffusion.num_steps, (B,), device=device)
        noise = torch.randn_like(x_start)
        x_t = self.diffusion.q_sample(x_start, t, noise)

        # CFG dropout: per-sample bernoulli over the full conditioning.
        drop_mask = torch.rand(B, device=device) < self.cfg.cfg_drop_prob
        x0_pred = self.denoiser(x_t, t, cond, cond_drop_mask=drop_mask)

        return {
            "x0_pred": x0_pred,
            "x0_target": x_start,
            "x_t": x_t,
            "t": t,
        }

    @torch.no_grad()
    def sample(
        self,
        cond: dict,
        seq_length: int,
        cfg_scale: float = 1.0,
    ) -> Tensor:
        """Generate motion_263 from conditioning. Used at eval time."""
        B = cond["z_int"].shape[0]
        shape = (B, seq_length, self.cfg.denoiser.motion_dim)
        return self.diffusion.p_sample_loop(
            self.denoiser, shape, cond, cfg_scale=cfg_scale
        )
