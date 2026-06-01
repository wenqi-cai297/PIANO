"""Round-41 Stage-1 cascade training — pre-implementation P0 diagnostic.

R41 plans to train Stage-1 with a frozen PB1 in the loop, so Stage-1
finally receives motion-space supervision from the model that scores
it downstream. Before writing the cascade trainer (Stage 2 of the
R41 plan), this script answers every "must verify before we touch
``train_stage1.py``" question we identified in the design discussion.

The script does NOT modify production code. It instantiates Stage-1
and PB1 from existing trainer machinery, threads them through one
single-step teacher-forced cascade forward, and probes everything
that would silently kill R41 if wrong:

  1. Batch contract — R41 cfg actually surfaces stage2_coarse_extra
     (18-D) and stage2_support (13-D) from the dataset.
  2. Stage-1 ckpt round-trip — V8 V6 warm-start loads cleanly via the
     R41 helper that landed in commit 565b867.
  3. PB1 ckpt round-trip — denoiser + own object_encoder load, freeze
     correctly, forward works.
  4. End-to-end cascade forward — Stage-1 single-step x0-pred
     → frozen PB1 single-step x0-pred → motion-space MSE, all
     intermediates finite, all shapes correct.
  5. Cascade gradient path — cascade-only backward populates Stage-1
     denoiser grads with finite non-zero values; PB1 + PB1's
     object_encoder remain grad-free.
  6. Gradient scale — grad-norm(Stage-1 self loss) vs grad-norm(cascade
     motion MSE at w=1) to inform initial ``w_motion_mse``.
  7. Gradient by t_pb1 bucket — cascade grad norm at t∈[0,100],
     [400,500], [900,1000] to test whether low-t bias gives a
     stronger signal (input-add path is per-frame so this could
     matter even with PB1's S4-AdaLN being weak).
  8. Distribution alignment — V8 V6 generated stage1_coarse (z-scored)
     vs GT-derived z-scored vs GT + σ=0.05 noise (which is what PB1
     was actually trained against); per-channel mean/std table.
  9. Memory + wallclock — peak GPU MB at bs=64 cascade forward + the
     avg ms/step over 5 batches to predict R41 training cost.

Outputs:
  <out_dir>/p0_stats.json   — machine-readable, every probe in one dict
  <out_dir>/p0_summary.md   — human-readable, ordered by check #

Run on the server (env has omegaconf + torch + GPU + PB1 ckpt):

  python -u scripts/stage_a_generator/round41_stage1_cascade_p0_diag.py \\
      --stage1-config configs/training/stage1_v8_v6_full_f1.yaml \\
      --stage1-ckpt   runs/training/stage1_v8_v6_full_f1/final.pt \\
      --pb1-config    configs/training/anchordiff_r29_pb_a1_adaln_s4.yaml \\
      --pb1-ckpt      runs/training/stageB_anchordiff_r29_pb_a1_adaln_s4/final.pt \\
      --stage1-v8v6-substitute-cache analyses/round31_stage1_substitute_conds_v8_stage1_v8_v6_full_f1 \\
      --out-dir       analyses/round41_p0_cascade_diag

If --stage1-v8v6-substitute-cache is not provided, check 8 falls back
to "GT vs GT+noise" only (still informative).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from piano.data.dataset import collate_hoi
from piano.data.stage1_coarse_oracle import (
    extract_coarse_v1_batched,
    load_stage1_coarse_norm,
)
from piano.models.motion_anchordiff import (
    AnchorDenoiserConfig,
    AnchorDiffConfig,
    DiffusionConfig,
    GaussianDiffusion,
    MotionAnchorDiff,
)
from piano.models.object_encoder import ObjectEncoder
from piano.models.stage1_trajectory import (
    STAGE1_COARSE_DIM,
    Stage1Denoiser,
    Stage1DenoiserConfig,
)
from piano.training.pb1_loss_helpers import (
    anchor_joint_pos_loss,
    compute_min_snr_weight,
    fk_motion_135_to_joints_22,
    l_pos_full_loss,
    masked_motion_mse_loss,
    world_joint_velocity_loss,
)
from piano.training.stage1_losses import build_init_pose_f1
from piano.training.train_anchordiff import _build_dataset
from piano.training.train_stage1 import (
    _maybe_load_stage1_init_checkpoint,
)
from piano.utils.clip_utils import encode_text_per_token, load_clip_text_encoder


PB1_STAGE1_COARSE_NOISE_STD = 0.05  # from PB1 cfg training.stage1_coarse_noise_std
T_BUCKETS = ((0, 100), (400, 500), (900, 1000))


# ──────────────────────────────────────────────────────────────────────────
# Module builders — mirror the trainers' main() construction so the
# tests exercise the exact same code path R41 will use at training time.
# ──────────────────────────────────────────────────────────────────────────


def _build_stage1(cfg, device: torch.device) -> tuple[Stage1Denoiser, ObjectEncoder]:
    d = cfg.model.denoiser
    denoiser_cfg = Stage1DenoiserConfig(
        motion_dim=int(d.motion_dim),
        object_traj_dim=int(d.object_traj_dim),
        text_dim=int(d.text_dim),
        object_token_dim=int(d.object_token_dim),
        object_num_tokens=int(d.object_num_tokens),
        d_model=int(d.d_model),
        n_layers=int(d.n_layers),
        n_heads=int(d.n_heads),
        ff_mult=int(d.ff_mult),
        dropout=float(d.dropout),
        max_seq_length=int(cfg.data.max_seq_length),
        use_text=bool(d.get("use_text", True)),
        init_pose_dim=int(d.get("init_pose_dim", 0)),
    )
    model = Stage1Denoiser(denoiser_cfg).to(device)
    encoder = ObjectEncoder(
        num_input_points=int(cfg.model.object_encoder.num_input_points),
        num_output_tokens=int(cfg.model.object_encoder.num_output_tokens),
        feature_dim=int(cfg.model.object_encoder.feature_dim),
    ).to(device)
    return model, encoder


def _build_pb1(cfg, device: torch.device) -> tuple[MotionAnchorDiff, ObjectEncoder]:
    """Construct PB1 MotionAnchorDiff + ObjectEncoder from an OmegaConf cfg.

    This mirrors the construction in ``train_anchordiff.py::main`` lines
    1549-1630 (read but not refactored into a public helper — R41 Stage 2
    should pull this out of the trainer's main() and have both the
    trainer and this P0 script call the same builder).
    """
    d = cfg.model.denoiser
    denoiser_cfg = AnchorDenoiserConfig(
        motion_dim=int(d.motion_dim),
        object_traj_dim=int(d.object_traj_dim),
        init_pose_dim=int(d.init_pose_dim),
        text_dim=int(d.text_dim),
        object_token_dim=int(d.object_token_dim),
        object_num_tokens=int(d.object_num_tokens),
        stage1_coarse_dim=int(d.get("stage1_coarse_dim", 0)),
        use_round29_cond_injection=bool(d.get("use_round29_cond_injection", False)),
        r29_coarse_extra_dim=int(d.get("r29_coarse_extra_dim", 0)),
        r29_interaction_dim=int(d.get("r29_interaction_dim", 0)),
        r29_support_dim=int(d.get("r29_support_dim", 0)),
        r29_body_refine_dim=int(d.get("r29_body_refine_dim", 0)),
        r29_injection_mode=str(d.get("r29_injection_mode", "input_add")),
        r29_gate_bias_init=float(d.get("r29_gate_bias_init", -1.0)),
        r29_per_family_modes=(
            dict(d.get("r29_per_family_modes"))
            if d.get("r29_per_family_modes") is not None
            else None
        ),
        r29_zero_init_adapters=bool(d.get("r29_zero_init_adapters", True)),
        r29_use_cond_adaln=bool(d.get("r29_use_cond_adaln", False)),
        r29_adaln_families=(
            list(d.get("r29_adaln_families"))
            if d.get("r29_adaln_families") is not None
            else None
        ),
        r29_adaln_pool=str(d.get("r29_adaln_pool", "mean")),
        d_model=int(d.d_model),
        n_layers=int(d.n_layers),
        n_heads=int(d.n_heads),
        ff_mult=int(d.ff_mult),
        dropout=float(d.dropout),
        max_seq_length=int(cfg.data.max_seq_length),
    )
    diff_cfg = DiffusionConfig(
        num_steps=int(cfg.model.diffusion.num_steps),
        schedule=str(cfg.model.diffusion.schedule),
        objective=str(cfg.model.diffusion.get("objective", "ddpm")),
        prediction_target=str(cfg.model.diffusion.get("prediction_target", "x0")),
    )
    model = MotionAnchorDiff(
        AnchorDiffConfig(
            diffusion=diff_cfg,
            denoiser=denoiser_cfg,
            cfg_drop_prob=float(cfg.model.cfg_drop_prob),
        )
    ).to(device)
    encoder = ObjectEncoder(
        num_input_points=int(cfg.model.object_encoder.num_input_points),
        num_output_tokens=int(cfg.model.object_encoder.num_output_tokens),
        feature_dim=int(cfg.model.object_encoder.feature_dim),
    ).to(device)
    return model, encoder


def _load_pb1_state(
    model: MotionAnchorDiff,
    encoder: ObjectEncoder,
    ckpt_path: Path,
) -> dict[str, Any]:
    """Load PB1 ckpt into both modules + report what was found.

    Returns a dict the caller can drop into the JSON output.
    """
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    top_keys = list(state.keys())

    # Denoiser load (try both wrapped and flat).
    model_state = state.get("model", state)
    model.load_state_dict(model_state, strict=True)

    # Object encoder under one of two save formats.
    enc_loaded_from = None
    if "object_encoder" in state:
        encoder.load_state_dict(state["object_encoder"], strict=True)
        enc_loaded_from = "object_encoder"
    elif (
        isinstance(state.get("extra_modules"), dict)
        and "object_encoder" in state["extra_modules"]
    ):
        encoder.load_state_dict(
            state["extra_modules"]["object_encoder"], strict=True,
        )
        enc_loaded_from = "extra_modules.object_encoder"
    else:
        raise KeyError(
            f"PB1 ckpt {ckpt_path} has no object_encoder state under "
            f"either top-level 'object_encoder' or 'extra_modules.object_encoder'. "
            f"Top-level keys: {top_keys}"
        )
    return {
        "top_level_keys": top_keys,
        "object_encoder_loaded_from": enc_loaded_from,
    }


def _freeze(modules: list[torch.nn.Module]) -> None:
    for m in modules:
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)


def _count_trainable(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def _count_total(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def _grad_norm(module: torch.nn.Module) -> float:
    """L2 norm over finite grads of all trainable params. Returns 0 when
    no param has a grad. Inf/NaN grads are excluded from the norm but
    their presence is reported separately."""
    total = 0.0
    n = 0
    for p in module.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        finite_mask = torch.isfinite(g)
        if not finite_mask.all():
            # at least one element non-finite — count finite portion only
            g = torch.where(finite_mask, g, torch.zeros_like(g))
        total += float((g.float() ** 2).sum().item())
        n += 1
    return float(total ** 0.5) if n > 0 else 0.0


def _has_any_finite_grad(module: torch.nn.Module) -> bool:
    for p in module.parameters():
        if p.grad is None:
            continue
        if torch.isfinite(p.grad).any() and float(p.grad.abs().sum().item()) > 0.0:
            return True
    return False


def _clear_grads(modules: list[torch.nn.Module]) -> None:
    for m in modules:
        for p in m.parameters():
            if p.grad is not None:
                p.grad = None


# ──────────────────────────────────────────────────────────────────────────
# Single cascade forward — pure function so each check can call it with
# different timesteps, dtypes, etc. without copy-pasting plumbing.
# ──────────────────────────────────────────────────────────────────────────


def _cascade_forward(
    *,
    batch: dict,
    device: torch.device,
    stage1: Stage1Denoiser,
    stage1_object_encoder: ObjectEncoder,
    pb1: MotionAnchorDiff,
    pb1_object_encoder: ObjectEncoder,
    clip_model: torch.nn.Module | None,
    stage1_coarse_mean: torch.Tensor,
    stage1_coarse_std: torch.Tensor,
    t_s1_low: int = 0,
    t_s1_high: int = 1000,
    t_pb1_low: int = 0,
    t_pb1_high: int = 1000,
    add_stage1_coarse_noise: bool = True,
    pb1_cfg_drop_disabled: bool = True,
) -> dict[str, Any]:
    """One R41-style cascade forward step. Returns intermediate tensors
    + scalar losses so each P0 check can drill into the part it needs.

    Implementation note: this mirrors what Stage 2 of the R41 plan will
    inline into train_stage1.step_fn, with the same key choices —
    GT-derived C41/S4 from the batch, separate object_encoders, PB1
    CFG drop forced off, σ=0.05 noise on the cascade stage1_coarse
    cond to match PB1 training distribution.
    """
    motion = batch["motion"].to(device).float()          # (B, T, 135)
    joints = batch["joints"].to(device).float()          # (B, T, 22, 3)
    rest_offsets = batch["rest_offsets"].to(device).float()
    object_pc = batch["object_pc"].to(device).float()
    obj_com = batch["obj_com_canonical"].to(device).float()        # (B, T, 3)
    obj_rot6d = batch["obj_rot6d_canonical"].to(device).float()    # (B, T, 6)
    seq_len = batch["seq_len"].to(device)                # (B,)

    B, T, _ = motion.shape
    seq_idx = torch.arange(T, device=device).unsqueeze(0)
    seq_mask = (seq_idx < seq_len.unsqueeze(1)).float()  # (B, T)

    # ─── Object trajectory (9-D) — identical for Stage-1 and PB1 ─────────
    object_traj = torch.cat([obj_com, obj_rot6d], dim=-1)  # (B, T, 9)

    # ─── Object tokens — TWO separate encoders ───────────────────────────
    # Stage-1's own object_encoder (trainable in R41 training; here we just
    # need a forward).
    obj_tokens_stage1 = stage1_object_encoder(object_pc)   # (B, N, D_obj)
    # PB1's frozen object_encoder. R41 must use this for PB1 forward; using
    # Stage-1's would be the silent failure mode Codex §3.2 flagged.
    with torch.no_grad():
        obj_tokens_pb1 = pb1_object_encoder(object_pc)

    # ─── Text features (shared CLIP encoder) ─────────────────────────────
    if clip_model is not None and "text" in batch:
        text_feats, _ = encode_text_per_token(clip_model, batch["text"], device)
        text_feats = text_feats.float()
    else:
        text_feats = None

    # ─── GT-derived 23-D stage1_coarse (z-scored) ────────────────────────
    coarse_v1_raw = extract_coarse_v1_batched(
        motion=motion, rest_offsets=rest_offsets,
    )                                                      # (B, T, 23) raw
    coarse_v1_z = (coarse_v1_raw - stage1_coarse_mean) / stage1_coarse_std

    # ─── Stage-1 cond ────────────────────────────────────────────────────
    stage1_cond: dict[str, torch.Tensor] = {
        "object_world_traj": object_traj,
        "object_tokens": obj_tokens_stage1,
    }
    if text_feats is not None:
        stage1_cond["text"] = text_feats
    # init_pose F1 — 135-D (V8 V6 setting). build_init_pose_f1 is just
    # motion[:, 0, :] but go through the helper so we mirror trainer code.
    stage1_cond["init_pose"] = build_init_pose_f1(motion)

    # ─── Stage-1 single-step teacher-forced x0-pred ──────────────────────
    t_s1 = torch.randint(t_s1_low, t_s1_high, (B,), device=device, dtype=torch.long)
    noise_s1 = torch.randn_like(coarse_v1_z)
    diff = pb1.diffusion                                   # share diffusion schedule (PB1 is also cosine 1000)
    # NOTE: Stage-1's own DiffusionConfig is also cosine 1000 + x0 per
    # the V8 V6 yaml, so PB1's diffusion noise schedule is the same as
    # Stage-1's. Confirmed against
    # configs/training/stage1_v8_v6_full_f1.yaml model.diffusion.
    stage1_x_t = diff.q_sample(coarse_v1_z, t_s1, noise_s1)
    stage1_x0_pred = stage1(stage1_x_t, t_s1, stage1_cond, cond_drop_mask=None)

    # ─── Stage-1 self loss (z-scored MSE, masked) ────────────────────────
    stage1_self_per_dim = (stage1_x0_pred - coarse_v1_z) ** 2
    stage1_self_loss = (
        (stage1_self_per_dim.sum(-1) * seq_mask).sum()
        / seq_mask.sum().clamp_min(1.0)
    )

    # ─── Build PB1 cond — Stage-1 prediction + GT-derived C41/S4 ─────────
    # Match PB1 training-time stage1_coarse_noise (σ=0.05). Noise is
    # detached so cascade grad flows only through stage1_x0_pred.
    if add_stage1_coarse_noise:
        stage1_for_pb1 = stage1_x0_pred + torch.randn_like(
            stage1_x0_pred
        ).detach() * PB1_STAGE1_COARSE_NOISE_STD
    else:
        stage1_for_pb1 = stage1_x0_pred

    pb1_cond: dict[str, torch.Tensor] = {
        "object_world_traj": object_traj,
        "object_tokens": obj_tokens_pb1,
        "stage1_coarse": stage1_for_pb1,
        "stage2_coarse_extra": batch["stage2_coarse_extra"].to(device).float(),
        "stage2_support": batch["stage2_support"].to(device).float(),
        "init_pose": joints[:, 0, :, :].reshape(B, -1),  # PB1 uses 66-D
    }
    if text_feats is not None:
        pb1_cond["text"] = text_feats

    # ─── Disable PB1 CFG dropout for the cascade loop ────────────────────
    saved_cfg_drop = None
    if pb1_cfg_drop_disabled:
        saved_cfg_drop = float(pb1.cfg.cfg_drop_prob)
        pb1.cfg.cfg_drop_prob = 0.0

    # ─── PB1 single-step teacher-forced — frozen but grad-enabled path ───
    # PB1.training_step samples t internally + does q_sample. We want
    # explicit t bucket control for check 7, so override after the fact
    # by re-running the inner steps. For checks 4-6 we just call
    # training_step.
    if t_pb1_low == 0 and t_pb1_high == 1000:
        out = pb1.training_step(motion, pb1_cond)
        pb1_x0_pred = out["x0_pred"]
        t_pb1_used = out["t"]
    else:
        # Custom t bucket — manually q_sample + forward.
        t_pb1 = torch.randint(
            t_pb1_low, t_pb1_high, (B,), device=device, dtype=torch.long,
        )
        noise_pb1 = torch.randn_like(motion)
        motion_noisy = pb1.diffusion.q_sample(motion, t_pb1, noise_pb1)
        # Match training_step: zero drop mask (already saved cfg above).
        drop_mask = torch.zeros(B, device=device, dtype=torch.bool)
        net_out = pb1.denoiser(motion_noisy, t_pb1, pb1_cond, cond_drop_mask=drop_mask)
        # PB1 uses prediction_target "x0" per cfg, so net_out == x0_pred.
        pb1_x0_pred = net_out
        t_pb1_used = t_pb1

    # Restore PB1 cfg_drop.
    if saved_cfg_drop is not None:
        pb1.cfg.cfg_drop_prob = saved_cfg_drop

    # ─── Cascade motion-space MSE (masked) ───────────────────────────────
    motion_per_dim = (pb1_x0_pred - motion) ** 2
    cascade_loss = (
        (motion_per_dim.sum(-1) * seq_mask).sum()
        / seq_mask.sum().clamp_min(1.0)
    )

    return {
        "stage1_x0_pred": stage1_x0_pred,
        "stage1_self_loss": stage1_self_loss,
        "pb1_x0_pred": pb1_x0_pred,
        "cascade_loss": cascade_loss,
        "t_s1": t_s1,
        "t_pb1": t_pb1_used,
        "seq_mask": seq_mask,
        "coarse_v1_raw": coarse_v1_raw,
        "coarse_v1_z": coarse_v1_z,
        "stage1_for_pb1": stage1_for_pb1,
        "B": B,
        "T": T,
    }


# ──────────────────────────────────────────────────────────────────────────
# Individual checks. Each returns a dict to merge into p0_stats.json.
# ──────────────────────────────────────────────────────────────────────────


def check_1_batch_contract(batch: dict) -> dict[str, Any]:
    """1. Batch surface stage2_coarse_extra (18) + stage2_support (13)."""
    out: dict[str, Any] = {"name": "batch_contract"}
    for key, expected_dim in (
        ("stage2_coarse_extra", 18),
        ("stage2_support", 13),
    ):
        present = key in batch
        out[f"{key}_present"] = present
        if present:
            v = batch[key]
            out[f"{key}_shape"] = list(v.shape)
            out[f"{key}_last_dim"] = int(v.shape[-1])
            out[f"{key}_dim_ok"] = bool(v.shape[-1] == expected_dim)
            out[f"{key}_any_nan"] = bool(torch.isnan(v).any())
            out[f"{key}_any_inf"] = bool(torch.isinf(v).any())
            out[f"{key}_min"] = float(v.float().min())
            out[f"{key}_max"] = float(v.float().max())
            out[f"{key}_mean"] = float(v.float().mean())
            out[f"{key}_std"] = float(v.float().std())
    # Core motion / joints sanity (R41 will rely on these in step_fn).
    out["motion_shape"] = list(batch["motion"].shape)
    out["joints_shape"] = list(batch["joints"].shape)
    out["motion_last_dim_ok"] = bool(batch["motion"].shape[-1] == 135)
    out["joints_shape_ok"] = bool(batch["joints"].shape[-2:] == (22, 3))
    out["pass"] = bool(
        out.get("stage2_coarse_extra_dim_ok", False)
        and out.get("stage2_support_dim_ok", False)
        and out["motion_last_dim_ok"]
        and out["joints_shape_ok"]
        and not out.get("stage2_coarse_extra_any_nan", True)
        and not out.get("stage2_support_any_nan", True)
    )
    return out


def check_2_stage1_warm_start(
    stage1_cfg, stage1_ckpt_path: Path, device: torch.device,
) -> dict[str, Any]:
    """2. Stage-1 V8 V6 ckpt round-trip via the R41 warm-start helper."""
    out: dict[str, Any] = {"name": "stage1_warm_start"}
    stage1, stage1_encoder = _build_stage1(stage1_cfg, device)
    # Snapshot pre-load.
    fresh_denoiser_param_sum = float(
        sum(p.detach().abs().sum().item() for p in stage1.parameters())
    )
    fresh_encoder_param_sum = float(
        sum(p.detach().abs().sum().item() for p in stage1_encoder.parameters())
    )
    try:
        _maybe_load_stage1_init_checkpoint(
            model=stage1, object_encoder=stage1_encoder,
            ckpt_path=str(stage1_ckpt_path), strict=True,
        )
        out["loaded_ok"] = True
    except Exception as exc:
        out["loaded_ok"] = False
        out["error"] = repr(exc)
        return out
    out["denoiser_param_sum_before"] = fresh_denoiser_param_sum
    out["denoiser_param_sum_after"] = float(
        sum(p.detach().abs().sum().item() for p in stage1.parameters())
    )
    out["denoiser_changed"] = bool(
        abs(out["denoiser_param_sum_after"] - fresh_denoiser_param_sum) > 1e-3
    )
    out["encoder_param_sum_before"] = fresh_encoder_param_sum
    out["encoder_param_sum_after"] = float(
        sum(p.detach().abs().sum().item() for p in stage1_encoder.parameters())
    )
    out["encoder_changed"] = bool(
        abs(out["encoder_param_sum_after"] - fresh_encoder_param_sum) > 1e-3
    )
    out["pass"] = bool(out["loaded_ok"] and out["denoiser_changed"] and out["encoder_changed"])
    return out


def check_3_pb1_ckpt(
    pb1_cfg, pb1_ckpt_path: Path, device: torch.device,
) -> tuple[MotionAnchorDiff, ObjectEncoder, dict[str, Any]]:
    """3. PB1 ckpt load + freeze + 1 dry forward."""
    out: dict[str, Any] = {"name": "pb1_ckpt"}
    pb1, pb1_encoder = _build_pb1(pb1_cfg, device)
    try:
        load_info = _load_pb1_state(pb1, pb1_encoder, pb1_ckpt_path)
        out.update(load_info)
    except Exception as exc:
        out["error"] = repr(exc)
        out["pass"] = False
        return pb1, pb1_encoder, out
    _freeze([pb1, pb1_encoder])
    out["pb1_trainable_params"] = _count_trainable(pb1)
    out["pb1_total_params"] = _count_total(pb1)
    out["pb1_encoder_trainable_params"] = _count_trainable(pb1_encoder)
    out["pb1_encoder_total_params"] = _count_total(pb1_encoder)
    out["pass"] = bool(
        out["pb1_trainable_params"] == 0
        and out["pb1_encoder_trainable_params"] == 0
        and out["pb1_total_params"] > 0
    )
    return pb1, pb1_encoder, out


def check_4_cascade_forward(
    *, fwd_args: dict, batches: list[dict],
) -> dict[str, Any]:
    """4. End-to-end cascade forward — shapes + finite checks across N batches."""
    out: dict[str, Any] = {"name": "cascade_forward"}
    rows = []
    for i, batch in enumerate(batches):
        r = _cascade_forward(**{**fwd_args, "batch": batch})
        rows.append({
            "batch_idx": i,
            "B": r["B"],
            "T": r["T"],
            "stage1_x0_pred_shape": list(r["stage1_x0_pred"].shape),
            "stage1_x0_pred_finite": bool(torch.isfinite(r["stage1_x0_pred"]).all()),
            "stage1_self_loss": float(r["stage1_self_loss"].item()),
            "pb1_x0_pred_shape": list(r["pb1_x0_pred"].shape),
            "pb1_x0_pred_finite": bool(torch.isfinite(r["pb1_x0_pred"]).all()),
            "cascade_loss": float(r["cascade_loss"].item()),
            "cascade_loss_finite": bool(torch.isfinite(r["cascade_loss"])),
            "t_pb1_mean": float(r["t_pb1"].float().mean()),
        })
    out["per_batch"] = rows
    out["pass"] = all(
        r["stage1_x0_pred_finite"]
        and r["pb1_x0_pred_finite"]
        and r["cascade_loss_finite"]
        for r in rows
    )
    return out


def check_5_grad_path(
    *, fwd_args: dict, batch: dict, stage1: Stage1Denoiser,
    stage1_encoder: ObjectEncoder, pb1: MotionAnchorDiff,
    pb1_encoder: ObjectEncoder,
) -> dict[str, Any]:
    """5. Cascade-only backward — Stage-1 receives finite non-zero grad,
    PB1 receives no grad."""
    out: dict[str, Any] = {"name": "grad_path"}
    _clear_grads([stage1, stage1_encoder, pb1, pb1_encoder])
    r = _cascade_forward(**{**fwd_args, "batch": batch})
    r["cascade_loss"].backward()

    out["stage1_denoiser_has_finite_grad"] = _has_any_finite_grad(stage1)
    out["stage1_denoiser_grad_norm"] = _grad_norm(stage1)
    out["stage1_encoder_has_finite_grad"] = _has_any_finite_grad(stage1_encoder)
    out["stage1_encoder_grad_norm"] = _grad_norm(stage1_encoder)
    out["pb1_has_grad"] = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in pb1.parameters()
    )
    out["pb1_encoder_has_grad"] = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in pb1_encoder.parameters()
    )
    _clear_grads([stage1, stage1_encoder, pb1, pb1_encoder])
    out["pass"] = bool(
        out["stage1_denoiser_has_finite_grad"]
        and out["stage1_denoiser_grad_norm"] > 0.0
        and not out["pb1_has_grad"]
        and not out["pb1_encoder_has_grad"]
    )
    return out


def check_6_grad_scale(
    *, fwd_args: dict, batch: dict, stage1: Stage1Denoiser,
    stage1_encoder: ObjectEncoder, pb1: MotionAnchorDiff,
    pb1_encoder: ObjectEncoder,
) -> dict[str, Any]:
    """6. Grad norm of (stage1 self) vs (cascade motion mse at w=1)."""
    out: dict[str, Any] = {"name": "grad_scale"}
    _clear_grads([stage1, stage1_encoder, pb1, pb1_encoder])
    r1 = _cascade_forward(**{**fwd_args, "batch": batch})
    r1["stage1_self_loss"].backward(retain_graph=False)
    out["grad_norm_stage1_self"] = _grad_norm(stage1)

    _clear_grads([stage1, stage1_encoder, pb1, pb1_encoder])
    r2 = _cascade_forward(**{**fwd_args, "batch": batch})
    r2["cascade_loss"].backward(retain_graph=False)
    out["grad_norm_cascade_w1"] = _grad_norm(stage1)
    _clear_grads([stage1, stage1_encoder, pb1, pb1_encoder])

    self_norm = out["grad_norm_stage1_self"]
    casc_norm = out["grad_norm_cascade_w1"]
    out["ratio_cascade_over_self"] = (
        casc_norm / self_norm if self_norm > 0 else float("inf")
    )
    # Decision rule per Codex §4 + my design notes.
    ratio = out["ratio_cascade_over_self"]
    if ratio > 10.0:
        rec = "w_motion_mse <= 0.05  (cascade grad too strong)"
    elif ratio > 1.0:
        rec = "w_motion_mse in [0.1, 0.5]"
    elif ratio > 0.1:
        rec = "w_motion_mse in [0.5, 2.0]"
    else:
        rec = "w_motion_mse >= 5.0 OR bias t_pb1 to low (cascade grad weak)"
    out["recommended_initial_w_motion_mse"] = rec
    out["pass"] = bool(self_norm > 0 and casc_norm > 0)
    return out


def _compute_actual_cascade_loss(
    *,
    fwd_result: dict,
    batch: dict,
    device: torch.device,
    cascade_weights: dict[str, float],
    cascade_extras: dict[str, Any],
    pb1: MotionAnchorDiff,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Reconstruct the same cascade loss the trainer's step_fn computes,
    using the helpers in ``pb1_loss_helpers``.

    Returns: (cascade_loss_weighted, component_dict)

    The component_dict records the unweighted loss values for the
    calibration report (informational).
    """
    motion = batch["motion"].to(device).float()
    joints = batch["joints"].to(device).float()
    rest_offsets = batch["rest_offsets"].to(device).float()
    seq_mask = fwd_result["seq_mask"]
    pb1_x0_pred = fwd_result["pb1_x0_pred"]
    t_pb1 = fwd_result["t_pb1"]

    w_motion = float(cascade_weights.get("w_motion_mse", 0.0))
    w_world_vel = float(cascade_weights.get("w_world_joint_vel", 0.0))
    w_lpos = float(cascade_weights.get("w_l_pos_full", 0.0))
    w_anchor = float(cascade_weights.get("w_anchor_joint_pos", 0.0))
    w_total = float(cascade_weights.get("w_total", 1.0))

    # min-SNR weight on motion MSE only (matches trainer).
    min_snr_w = None
    if bool(cascade_extras.get("use_min_snr", True)):
        _pb1_diff = (
            pb1.diffusion.module if hasattr(pb1.diffusion, "module")
            else pb1.diffusion
        )
        min_snr_w = compute_min_snr_weight(
            t_pb1, _pb1_diff.alphas_cumprod,
            gamma=float(cascade_extras.get("min_snr_gamma", 5.0)),
        )

    components: dict[str, float] = {}
    cascade_raw = torch.zeros((), device=device, dtype=motion.dtype)

    if w_motion > 0:
        l = masked_motion_mse_loss(
            pred=pb1_x0_pred, target=motion, seq_mask=seq_mask,
            min_snr_weight=min_snr_w,
        )
        components["motion_mse"] = float(l.item())
        cascade_raw = cascade_raw + w_motion * l

    if w_world_vel > 0:
        l = world_joint_velocity_loss(
            pred=pb1_x0_pred, target=motion, seq_mask=seq_mask,
        )
        components["world_joint_vel"] = float(l.item())
        cascade_raw = cascade_raw + w_world_vel * l

    if w_lpos > 0 or w_anchor > 0:
        jpos_pred = fk_motion_135_to_joints_22(
            motion=pb1_x0_pred, rest_offsets=rest_offsets,
        )
        if w_lpos > 0:
            l = l_pos_full_loss(
                jpos_pred=jpos_pred, joints_gt=joints, seq_mask=seq_mask,
                hand_endpoint_weight=float(
                    cascade_extras.get("l_pos_hand_endpoint_weight", 2.0)
                ),
                foot_endpoint_weight=float(
                    cascade_extras.get("l_pos_foot_endpoint_weight", 2.0)
                ),
            )
            components["l_pos_full"] = float(l.item())
            cascade_raw = cascade_raw + w_lpos * l
        if w_anchor > 0:
            if "contact_state" not in batch:
                raise KeyError(
                    "cascade_w_anchor_joint_pos > 0 but batch missing "
                    "contact_state — confirm dataset has R29 pseudo-labels."
                )
            contact = batch["contact_state"].to(device).float()
            l = anchor_joint_pos_loss(
                jpos_pred=jpos_pred, joints_gt=joints,
                contact_state=contact, seq_mask=seq_mask,
                part_weights=tuple(
                    float(w) for w in cascade_extras.get(
                        "anchor_part_weights",
                        (2.0, 2.0, 0.0, 0.0, 0.5),
                    )
                ),
                contact_threshold=float(
                    cascade_extras.get("anchor_contact_threshold", 0.5)
                ),
            )
            components["anchor_joint_pos"] = float(l.item())
            cascade_raw = cascade_raw + w_anchor * l

    return w_total * cascade_raw, components


def check_10_grad_scale_actual_stack(
    *, fwd_args: dict, batch: dict, stage1: Stage1Denoiser,
    stage1_encoder: ObjectEncoder, pb1: MotionAnchorDiff,
    pb1_encoder: ObjectEncoder,
    cascade_weights: dict[str, float],
    cascade_extras: dict[str, Any],
) -> dict[str, Any]:
    """10. Actual cascade stack gradient ratio (drives R41 calibration).

    Unlike check 6 which measures grad scale of motion-MSE only at w=1,
    this check rebuilds the *actual* cascade loss the trainer would
    compute for this specific cfg (with its current cascade_w_*
    weights and w_total), backprops it, and reports the resulting
    Stage-1 grad norm vs Stage-1 self loss grad norm.

    This is what calibration should use to recommend w_total: loss
    ratio is a misleading proxy when PB1's Jacobian is in the chain
    (Codex code-review blocker §2).
    """
    out: dict[str, Any] = {"name": "grad_scale_actual_stack"}
    out["cascade_weights"] = dict(cascade_weights)

    # 1) self loss only.
    _clear_grads([stage1, stage1_encoder, pb1, pb1_encoder])
    r1 = _cascade_forward(**{**fwd_args, "batch": batch})
    r1["stage1_self_loss"].backward()
    self_norm = _grad_norm(stage1)
    out["grad_norm_stage1_self"] = self_norm

    # 2) actual weighted cascade stack only.
    _clear_grads([stage1, stage1_encoder, pb1, pb1_encoder])
    r2 = _cascade_forward(**{**fwd_args, "batch": batch})
    cascade_weighted, components = _compute_actual_cascade_loss(
        fwd_result=r2, batch=batch, device=fwd_args["device"],
        cascade_weights=cascade_weights,
        cascade_extras=cascade_extras,
        pb1=pb1,
    )
    out["component_loss_values"] = components
    out["cascade_weighted_value"] = float(cascade_weighted.item())
    if cascade_weighted.requires_grad:
        cascade_weighted.backward()
    casc_norm = _grad_norm(stage1)
    out["grad_norm_actual_cascade_weighted"] = casc_norm
    _clear_grads([stage1, stage1_encoder, pb1, pb1_encoder])

    ratio = casc_norm / self_norm if self_norm > 0 else float("inf")
    out["ratio_actual_cascade_over_self"] = ratio
    # Linear recommendation: scale w_total so ratio = 1.0.
    current_w_total = float(cascade_weights.get("w_total", 1.0))
    if ratio > 0:
        out["recommended_w_total_for_ratio_1"] = (
            current_w_total * 1.0 / ratio
        )
    else:
        out["recommended_w_total_for_ratio_1"] = current_w_total
    out["pass"] = bool(self_norm > 0 and casc_norm > 0)
    return out


def check_7_grad_by_t_bucket(
    *, fwd_args: dict, batch: dict, stage1: Stage1Denoiser,
    stage1_encoder: ObjectEncoder, pb1: MotionAnchorDiff,
    pb1_encoder: ObjectEncoder,
) -> dict[str, Any]:
    """7. Cascade grad norm by t_pb1 bucket."""
    out: dict[str, Any] = {"name": "grad_by_t_bucket"}
    out["buckets"] = []
    norms_by_bucket = {}
    for (t_low, t_high) in T_BUCKETS:
        _clear_grads([stage1, stage1_encoder, pb1, pb1_encoder])
        local = dict(fwd_args)
        local["batch"] = batch
        local["t_pb1_low"] = t_low
        local["t_pb1_high"] = t_high
        r = _cascade_forward(**local)
        r["cascade_loss"].backward()
        gn = _grad_norm(stage1)
        norms_by_bucket[f"[{t_low},{t_high}]"] = gn
        out["buckets"].append({
            "t_low": t_low,
            "t_high": t_high,
            "grad_norm": gn,
            "cascade_loss": float(r["cascade_loss"].item()),
            "pb1_x0_finite": bool(torch.isfinite(r["pb1_x0_pred"]).all()),
        })
    _clear_grads([stage1, stage1_encoder, pb1, pb1_encoder])

    # Pretty cross-bucket comparison.
    low_norm = out["buckets"][0]["grad_norm"]
    high_norm = out["buckets"][-1]["grad_norm"]
    out["low_vs_high_ratio"] = (
        low_norm / high_norm if high_norm > 0 else float("inf")
    )
    if out["low_vs_high_ratio"] > 5.0:
        out["recommendation"] = (
            "low-t bias variant (A2) likely informative — "
            "low-t cascade grad is >> high-t"
        )
    elif out["low_vs_high_ratio"] > 1.5:
        out["recommendation"] = (
            "low-t bias may help — modest improvement over uniform"
        )
    else:
        out["recommendation"] = (
            "uniform t_pb1 (A1) sufficient — low-t does not concentrate signal"
        )
    out["pass"] = bool(low_norm > 0 and high_norm > 0)
    return out


def check_8_distribution_alignment(
    *,
    stage1_cfg,
    device: torch.device,
    stage1_coarse_mean: torch.Tensor,
    stage1_coarse_std: torch.Tensor,
    v8v6_cache_dir: Path | None,
    batch_iter,
    selection_json: Path | None = None,
    n_clips_target: int = 32,
) -> dict[str, Any]:
    """8. Distribution alignment between V8 V6 generated stage1_coarse,
    GT-derived z-scored, and GT + σ=0.05 noise.

    All numbers in z-scored space (which is what PB1 sees).
    """
    out: dict[str, Any] = {"name": "distribution_alignment"}

    # Always compute GT and GT+noise from a few batches.
    gt_chunks: list[np.ndarray] = []
    n_seen = 0
    for batch in batch_iter:
        motion = batch["motion"].to(device).float()
        rest_offsets = batch["rest_offsets"].to(device).float()
        coarse_raw = extract_coarse_v1_batched(
            motion=motion, rest_offsets=rest_offsets,
        )
        coarse_z = (coarse_raw - stage1_coarse_mean) / stage1_coarse_std
        # Drop padded frames using seq_len.
        seq_len = batch["seq_len"].to(device)
        B, T, _ = coarse_z.shape
        seq_idx = torch.arange(T, device=device).unsqueeze(0)
        mask = (seq_idx < seq_len.unsqueeze(1))
        gt_chunks.append(coarse_z[mask].detach().cpu().numpy())
        n_seen += B
        if n_seen >= n_clips_target:
            break
    gt_all = np.concatenate(gt_chunks, axis=0).astype(np.float32)
    gt_plus_noise = (
        gt_all + np.random.RandomState(0).randn(*gt_all.shape).astype(np.float32)
        * PB1_STAGE1_COARSE_NOISE_STD
    )

    out["gt_n_frames"] = int(gt_all.shape[0])
    out["gt_per_channel_mean"] = gt_all.mean(axis=0).tolist()
    out["gt_per_channel_std"] = gt_all.std(axis=0).tolist()
    out["gt_plus_noise_per_channel_mean"] = gt_plus_noise.mean(axis=0).tolist()
    out["gt_plus_noise_per_channel_std"] = gt_plus_noise.std(axis=0).tolist()

    # V8 V6 generated (from cache) — optional.
    if v8v6_cache_dir is not None and v8v6_cache_dir.exists():
        v8v6_chunks = []
        cache_root = v8v6_cache_dir / "val" if (v8v6_cache_dir / "val").is_dir() else v8v6_cache_dir
        for npz_path in sorted(cache_root.glob("**/*.npz"))[:n_clips_target]:
            data = np.load(npz_path)
            if "stage1_coarse" not in data.files:
                continue
            arr = data["stage1_coarse"].astype(np.float32)  # already z-scored
            vt = int(data["valid_T"]) if "valid_T" in data.files else arr.shape[0]
            v8v6_chunks.append(arr[:vt])
        if v8v6_chunks:
            v8v6_all = np.concatenate(v8v6_chunks, axis=0).astype(np.float32)
            out["v8v6_n_frames"] = int(v8v6_all.shape[0])
            out["v8v6_per_channel_mean"] = v8v6_all.mean(axis=0).tolist()
            out["v8v6_per_channel_std"] = v8v6_all.std(axis=0).tolist()
            # Per-channel mean diff vs GT+noise (the actual PB1 train distrib).
            mean_gap = np.abs(v8v6_all.mean(axis=0) - gt_plus_noise.mean(axis=0))
            std_ratio = v8v6_all.std(axis=0) / np.maximum(gt_plus_noise.std(axis=0), 1e-6)
            out["v8v6_vs_pb1_train_mean_gap_per_channel"] = mean_gap.tolist()
            out["v8v6_vs_pb1_train_std_ratio_per_channel"] = std_ratio.tolist()
            out["v8v6_vs_pb1_train_max_mean_gap"] = float(mean_gap.max())
            out["v8v6_vs_pb1_train_max_std_dev_from_1"] = float(
                np.abs(std_ratio - 1.0).max()
            )
        else:
            out["v8v6_n_frames"] = 0
            out["note"] = (
                f"V8 V6 cache dir {v8v6_cache_dir} exists but contains no npz with 'stage1_coarse'"
            )
    else:
        out["v8v6_n_frames"] = 0
        out["note"] = "V8 V6 cache dir not provided or missing — skipping V8V6 distribution comparison"

    out["pass"] = True  # informational only
    return out


def check_9_memory_wallclock(
    *, fwd_args: dict, batch: dict, n_iters: int = 5,
) -> dict[str, Any]:
    """9. Memory + wallclock probe at the trainer's batch size."""
    out: dict[str, Any] = {"name": "memory_wallclock"}
    device = fwd_args["device"]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    times = []
    for i in range(n_iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = _cascade_forward(**{**fwd_args, "batch": batch})
        # full backward to capture the realistic memory profile
        (r["stage1_self_loss"] + r["cascade_loss"]).backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    out["batch_size_observed"] = int(batch["motion"].shape[0])
    out["seq_len_observed"] = int(batch["motion"].shape[1])
    out["avg_step_seconds"] = float(np.mean(times))
    out["min_step_seconds"] = float(np.min(times))
    out["max_step_seconds"] = float(np.max(times))
    if device.type == "cuda":
        out["peak_gpu_mb"] = float(
            torch.cuda.max_memory_allocated() / (1024 ** 2)
        )
    else:
        out["peak_gpu_mb"] = None
    out["pass"] = bool(out["avg_step_seconds"] > 0)
    return out


# ──────────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────────


def _make_p0_cfg(stage1_cfg, pb1_cfg):
    """Build an R41-style hybrid cfg that uses Stage-1's data/model
    block but switches r29 variants to PB1's so the dataloader populates
    stage2_coarse_extra + stage2_support."""
    p0_cfg = OmegaConf.create(OmegaConf.to_container(stage1_cfg, resolve=True))
    p0_cfg.data.r29_coarse_variant = pb1_cfg.data.r29_coarse_variant
    p0_cfg.data.r29_support_variant = pb1_cfg.data.r29_support_variant
    p0_cfg.data.r29_interaction_variant = pb1_cfg.data.get(
        "r29_interaction_variant", "I0",
    )
    p0_cfg.data.r29_body_variant = pb1_cfg.data.get("r29_body_variant", "B0")
    return p0_cfg


def _write_summary_md(stats: dict[str, Any], out_md: Path) -> None:
    lines = [
        "# Round-41 Stage-1 cascade — P0 diagnostic summary",
        "",
        f"- stage1_config: `{stats['inputs']['stage1_config']}`",
        f"- stage1_ckpt:   `{stats['inputs']['stage1_ckpt']}`",
        f"- pb1_config:    `{stats['inputs']['pb1_config']}`",
        f"- pb1_ckpt:      `{stats['inputs']['pb1_ckpt']}`",
        f"- v8v6_cache:    `{stats['inputs'].get('v8v6_cache_dir') or '(none)'}`",
        f"- device:        `{stats['inputs']['device']}`",
        f"- batch_size:    {stats['inputs']['batch_size']}",
        "",
        "## Pass / fail per check",
        "",
        "| # | check | pass |",
        "|---:|---|---:|",
    ]
    for i, name in enumerate(stats["check_order"], start=1):
        c = stats["checks"][name]
        lines.append(f"| {i} | {name} | {'✓' if c.get('pass') else '✗'} |")
    lines.append("")

    # Headline numbers per check (we surface the ones that matter for
    # the next design step; full detail is in p0_stats.json).
    c1 = stats["checks"].get("batch_contract", {})
    lines.extend([
        "## 1. Batch contract",
        "",
        f"- `stage2_coarse_extra` present: {c1.get('stage2_coarse_extra_present')}, "
        f"shape={c1.get('stage2_coarse_extra_shape')}",
        f"- `stage2_support` present: {c1.get('stage2_support_present')}, "
        f"shape={c1.get('stage2_support_shape')}",
        "",
    ])

    c2 = stats["checks"].get("stage1_warm_start", {})
    lines.extend([
        "## 2. Stage-1 warm-start (V8 V6 round-trip)",
        "",
        f"- loaded_ok: {c2.get('loaded_ok')}",
        f"- denoiser_changed: {c2.get('denoiser_changed')}",
        f"- encoder_changed: {c2.get('encoder_changed')}",
        "",
    ])

    c3 = stats["checks"].get("pb1_ckpt", {})
    lines.extend([
        "## 3. PB1 ckpt + freeze",
        "",
        f"- object_encoder_loaded_from: `{c3.get('object_encoder_loaded_from')}`",
        f"- pb1 trainable params: {c3.get('pb1_trainable_params')} "
        f"(of {c3.get('pb1_total_params')})",
        f"- pb1_encoder trainable params: {c3.get('pb1_encoder_trainable_params')} "
        f"(of {c3.get('pb1_encoder_total_params')})",
        "",
    ])

    c4 = stats["checks"].get("cascade_forward", {})
    if "per_batch" in c4:
        lines.append("## 4. Cascade forward")
        lines.append("")
        lines.append("| batch | stage1_self | cascade_mse | t_pb1_mean | finite |")
        lines.append("|---:|---:|---:|---:|---:|")
        for r in c4["per_batch"]:
            ok = r["stage1_x0_pred_finite"] and r["pb1_x0_pred_finite"] and r["cascade_loss_finite"]
            lines.append(
                f"| {r['batch_idx']} | {r['stage1_self_loss']:.4f} | "
                f"{r['cascade_loss']:.4f} | {r['t_pb1_mean']:.1f} | "
                f"{'✓' if ok else '✗'} |"
            )
        lines.append("")

    c5 = stats["checks"].get("grad_path", {})
    lines.extend([
        "## 5. Cascade grad path",
        "",
        f"- stage1 denoiser has finite grad: {c5.get('stage1_denoiser_has_finite_grad')}",
        f"- stage1 denoiser grad norm: {c5.get('stage1_denoiser_grad_norm')}",
        f"- stage1 encoder has finite grad: {c5.get('stage1_encoder_has_finite_grad')}",
        f"- stage1 encoder grad norm: {c5.get('stage1_encoder_grad_norm')}",
        f"- pb1 has grad: {c5.get('pb1_has_grad')} (must be False)",
        f"- pb1_encoder has grad: {c5.get('pb1_encoder_has_grad')} (must be False)",
        "",
    ])

    c6 = stats["checks"].get("grad_scale", {})
    lines.extend([
        "## 6. Grad scale",
        "",
        f"- grad_norm_stage1_self: {c6.get('grad_norm_stage1_self'):.4e}"
        if isinstance(c6.get('grad_norm_stage1_self'), float)
        else f"- grad_norm_stage1_self: {c6.get('grad_norm_stage1_self')}",
        f"- grad_norm_cascade_w1: {c6.get('grad_norm_cascade_w1'):.4e}"
        if isinstance(c6.get('grad_norm_cascade_w1'), float)
        else f"- grad_norm_cascade_w1: {c6.get('grad_norm_cascade_w1')}",
        f"- ratio cascade/self: {c6.get('ratio_cascade_over_self')}",
        f"- **recommendation**: {c6.get('recommended_initial_w_motion_mse')}",
        "",
    ])

    c7 = stats["checks"].get("grad_by_t_bucket", {})
    if "buckets" in c7:
        lines.append("## 7. Cascade grad by t_pb1 bucket")
        lines.append("")
        lines.append("| t range | grad norm | cascade loss |")
        lines.append("|---|---:|---:|")
        for b in c7["buckets"]:
            lines.append(
                f"| [{b['t_low']},{b['t_high']}] | {b['grad_norm']:.4e} | "
                f"{b['cascade_loss']:.4f} |"
            )
        lines.append("")
        lines.append(f"- low/high ratio: {c7.get('low_vs_high_ratio')}")
        lines.append(f"- **recommendation**: {c7.get('recommendation')}")
        lines.append("")

    c8 = stats["checks"].get("distribution_alignment", {})
    lines.append("## 8. Distribution alignment")
    lines.append("")
    lines.append(f"- gt_n_frames: {c8.get('gt_n_frames')}")
    lines.append(f"- v8v6_n_frames: {c8.get('v8v6_n_frames')}")
    if c8.get("v8v6_n_frames", 0) > 0:
        lines.append(
            f"- max |V8V6 mean − (GT+noise) mean| over 23 channels: "
            f"{c8.get('v8v6_vs_pb1_train_max_mean_gap')}"
        )
        lines.append(
            f"- max |V8V6 std / (GT+noise) std − 1| over 23 channels: "
            f"{c8.get('v8v6_vs_pb1_train_max_std_dev_from_1')}"
        )
    if "note" in c8:
        lines.append(f"- note: {c8['note']}")
    lines.append("")

    c9 = stats["checks"].get("memory_wallclock", {})
    lines.extend([
        "## 9. Memory + wallclock",
        "",
        f"- batch size observed: {c9.get('batch_size_observed')}",
        f"- seq_len observed:    {c9.get('seq_len_observed')}",
        f"- avg step seconds:    {c9.get('avg_step_seconds')}",
        f"- peak GPU MB:         {c9.get('peak_gpu_mb')}",
        "",
    ])

    lines.extend([
        "## Next-step decision",
        "",
        "1. If any of 1-5 fails: **fix the implementation** before launching R41 training.",
        "2. Use check 6's `recommended_initial_w_motion_mse` for the first training matrix.",
        "3. Use check 7's recommendation to decide whether to include an A2 low-t variant.",
        "4. Use check 8 to confirm σ=0.05 noise injection brings V8 V6 generated cond "
        "into PB1's training-distribution range.",
        "5. Use check 9 to set `batch_size` / `gradient_accumulation_steps` in R41 cfg.",
        "",
    ])

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1-config", type=Path, required=True)
    ap.add_argument("--stage1-ckpt", type=Path, required=True)
    ap.add_argument("--pb1-config", type=Path, required=True)
    ap.add_argument("--pb1-ckpt", type=Path, required=True)
    ap.add_argument(
        "--stage1-v8v6-substitute-cache", type=Path, default=None,
        help="Optional V8 V6 generated stage1_coarse cache "
             "(e.g. analyses/round31_stage1_substitute_conds_v8_stage1_v8_v6_full_f1). "
             "Used for check 8 distribution alignment.",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--bucket", choices=["train", "val"], default="val",
        help="Dataset bucket for the dataloader. val is cheaper.",
    )
    ap.add_argument(
        "--batch-size", type=int, default=16,
        help=(
            "P0 batch size. Defaults to 16 because cascade forward "
            "(Stage-1 encoder + Stage-1 denoiser + PB1 encoder + PB1 denoiser "
            "+ Stage-1 grad activations + PB1 forward activations for grad "
            "backprop) at bs=64 OOM's a single 5080 16 GB GPU. "
            "Per-batch grad scale / cascade-loss / t-bucket gradient norm "
            "are bs-invariant (loss is masked mean, grad is normalized) — "
            "P0 numbers transfer to any bs used at training time. "
            "Use a larger value only if peak GPU usage probe (check 9) is "
            "the only quantity of interest."
        ),
    )
    ap.add_argument(
        "--n-mem-iters", type=int, default=5,
        help="How many forward+backward iters to time for check 9.",
    )
    ap.add_argument(
        "--calibration-only", action="store_true",
        help=(
            "Fast calibration mode for round41_cascade_calibration.py. "
            "Runs only checks 1 (batch contract), 2 (Stage-1 warm-start), "
            "3 (PB1 ckpt), 5 (grad path), 10 (actual cascade stack grad "
            "scale). Skips check 4 (3-batch forward), 6 (motion-mse-only "
            "grad), 7 (t-bucket), 8 (distribution alignment), 9 (memory)."
        ),
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[p0] device={device}")

    # Load configs.
    stage1_cfg = OmegaConf.load(str(args.stage1_config))
    pb1_cfg = OmegaConf.load(str(args.pb1_config))

    # Build P0 hybrid cfg with PB1's r29 variants so the dataloader emits
    # stage2_coarse_extra (18-D) and stage2_support (13-D).
    p0_cfg = _make_p0_cfg(stage1_cfg, pb1_cfg)
    p0_cfg.training.batch_size = int(args.batch_size)
    print(
        f"[p0] dataloader cfg: r29_coarse_variant={p0_cfg.data.r29_coarse_variant}, "
        f"r29_support_variant={p0_cfg.data.r29_support_variant}"
    )

    # Dataset + 1-batch dataloader (cheap — we only need a handful of batches).
    dataset = _build_dataset(p0_cfg, bucket=args.bucket, augment=False)
    print(f"[p0] dataset[{args.bucket}]: {len(dataset)} clips")
    loader = DataLoader(
        dataset, batch_size=int(args.batch_size),
        num_workers=0, shuffle=False,
        collate_fn=collate_hoi, pin_memory=False,
    )
    loader_iter = iter(loader)

    # Stats accumulator + check order so the md report is stable.
    stats: dict[str, Any] = {
        "inputs": {
            "stage1_config": str(args.stage1_config),
            "stage1_ckpt": str(args.stage1_ckpt),
            "pb1_config": str(args.pb1_config),
            "pb1_ckpt": str(args.pb1_ckpt),
            "v8v6_cache_dir": (
                str(args.stage1_v8v6_substitute_cache)
                if args.stage1_v8v6_substitute_cache else None
            ),
            "device": str(device),
            "batch_size": int(args.batch_size),
            "bucket": args.bucket,
        },
        "check_order": [],
        "checks": {},
    }

    def _record(name: str, payload: dict[str, Any]) -> None:
        stats["check_order"].append(name)
        stats["checks"][name] = payload
        print(f"[p0] check '{name}': pass={payload.get('pass')}")

    # ─── Check 1: batch contract ────────────────────────────────────────
    first_batch = next(loader_iter)
    _record("batch_contract", check_1_batch_contract(first_batch))

    # If batch_contract failed, the rest of the checks will also fail
    # because PB1 forward needs those keys. Still run them so we get
    # a complete error picture.

    # ─── Check 2: Stage-1 warm-start ────────────────────────────────────
    c2 = check_2_stage1_warm_start(p0_cfg, args.stage1_ckpt, device)
    _record("stage1_warm_start", c2)
    # Reuse the freshly-loaded stage1 + encoder for the rest.
    stage1, stage1_encoder = _build_stage1(p0_cfg, device)
    _maybe_load_stage1_init_checkpoint(
        model=stage1, object_encoder=stage1_encoder,
        ckpt_path=str(args.stage1_ckpt), strict=True,
    )
    # Stage-1 is trainable; do NOT freeze.

    # ─── Check 3: PB1 ckpt + freeze ─────────────────────────────────────
    pb1, pb1_encoder, c3 = check_3_pb1_ckpt(pb1_cfg, args.pb1_ckpt, device)
    _record("pb1_ckpt", c3)

    # Shared CLIP (Stage-1 and PB1 both use 512-D ViT-B/32 per cfg).
    if int(p0_cfg.model.denoiser.text_dim) > 0:
        clip_model = load_clip_text_encoder(
            device=device,
            model_name=str(p0_cfg.model.text_encoder.clip_version),
            download_root=str(
                p0_cfg.model.text_encoder.get("download_root", "cache/clip"),
            ),
        )
    else:
        clip_model = None

    # Stage-1 coarse norm stats (z-score parameters).
    mean_np, std_np = load_stage1_coarse_norm(str(p0_cfg.data.stage1_coarse_cache_root))
    stage1_coarse_mean = torch.from_numpy(mean_np).to(device).float()
    stage1_coarse_std = torch.from_numpy(std_np).to(device).float()

    # ─── Shared fwd_args ────────────────────────────────────────────────
    base_fwd_args = dict(
        device=device,
        stage1=stage1,
        stage1_object_encoder=stage1_encoder,
        pb1=pb1,
        pb1_object_encoder=pb1_encoder,
        clip_model=clip_model,
        stage1_coarse_mean=stage1_coarse_mean,
        stage1_coarse_std=stage1_coarse_std,
        t_pb1_low=0, t_pb1_high=1000,
        add_stage1_coarse_noise=True,
        pb1_cfg_drop_disabled=True,
    )

    # ─── Check 4: cascade forward (1 batch in calibration-only mode) ─
    sample_batches = [first_batch]
    if not args.calibration_only:
        for _ in range(2):
            try:
                sample_batches.append(next(loader_iter))
            except StopIteration:
                break
    _record("cascade_forward", check_4_cascade_forward(
        fwd_args=base_fwd_args, batches=sample_batches,
    ))

    # ─── Check 5: cascade grad path ─────────────────────────────────────
    _record("grad_path", check_5_grad_path(
        fwd_args=base_fwd_args, batch=sample_batches[0],
        stage1=stage1, stage1_encoder=stage1_encoder,
        pb1=pb1, pb1_encoder=pb1_encoder,
    ))

    # ─── Check 10: actual cascade stack grad ratio (calibration core) ─
    # Reads the target stage1 cfg's cascade block and rebuilds the
    # actual cascade loss the trainer would compute. Runs in both
    # full and calibration-only modes because it is the load-bearing
    # number for round41_cascade_calibration.py's recommendation.
    _target_cascade = (
        OmegaConf.load(str(args.stage1_config)).get("cascade", None)
    )
    if _target_cascade is None or not bool(_target_cascade.get("enabled", False)):
        # Control cell (no cascade); check 10 is informational only.
        cascade_weights = {
            "w_motion_mse": 0.0, "w_world_joint_vel": 0.0,
            "w_l_pos_full": 0.0, "w_anchor_joint_pos": 0.0,
            "w_total": 1.0,
        }
        cascade_extras = {
            "use_min_snr": True, "min_snr_gamma": 5.0,
            "l_pos_hand_endpoint_weight": 2.0,
            "l_pos_foot_endpoint_weight": 2.0,
            "anchor_part_weights": (2.0, 2.0, 0.0, 0.0, 0.5),
            "anchor_contact_threshold": 0.5,
        }
    else:
        cascade_weights = {
            "w_motion_mse": float(_target_cascade.get("w_motion_mse", 0.0)),
            "w_world_joint_vel": float(
                _target_cascade.get("w_world_joint_vel", 0.0)
            ),
            "w_l_pos_full": float(_target_cascade.get("w_l_pos_full", 0.0)),
            "w_anchor_joint_pos": float(
                _target_cascade.get("w_anchor_joint_pos", 0.0)
            ),
            "w_total": float(_target_cascade.get("w_total", 1.0)),
        }
        cascade_extras = {
            "use_min_snr": bool(_target_cascade.get("use_min_snr", True)),
            "min_snr_gamma": float(_target_cascade.get("min_snr_gamma", 5.0)),
            "l_pos_hand_endpoint_weight": float(
                _target_cascade.get("l_pos_hand_endpoint_weight", 2.0)
            ),
            "l_pos_foot_endpoint_weight": float(
                _target_cascade.get("l_pos_foot_endpoint_weight", 2.0)
            ),
            "anchor_part_weights": tuple(
                float(w) for w in _target_cascade.get(
                    "anchor_part_weights", [2.0, 2.0, 0.0, 0.0, 0.5],
                )
            ),
            "anchor_contact_threshold": float(
                _target_cascade.get("anchor_contact_threshold", 0.5)
            ),
        }
    _record("grad_scale_actual_stack", check_10_grad_scale_actual_stack(
        fwd_args=base_fwd_args, batch=sample_batches[0],
        stage1=stage1, stage1_encoder=stage1_encoder,
        pb1=pb1, pb1_encoder=pb1_encoder,
        cascade_weights=cascade_weights,
        cascade_extras=cascade_extras,
    ))

    if not args.calibration_only:
        # ─── Check 6: grad scale (motion-MSE-only, legacy) ──────────────
        _record("grad_scale", check_6_grad_scale(
            fwd_args=base_fwd_args, batch=sample_batches[0],
            stage1=stage1, stage1_encoder=stage1_encoder,
            pb1=pb1, pb1_encoder=pb1_encoder,
        ))

        # ─── Check 7: grad by t_pb1 bucket ──────────────────────────────
        _record("grad_by_t_bucket", check_7_grad_by_t_bucket(
            fwd_args=base_fwd_args, batch=sample_batches[0],
            stage1=stage1, stage1_encoder=stage1_encoder,
            pb1=pb1, pb1_encoder=pb1_encoder,
        ))

        # ─── Check 8: distribution alignment ────────────────────────────
        def _fresh_batch_iter():
            for b in sample_batches:
                yield b
            for b in loader_iter:
                yield b
        _record("distribution_alignment", check_8_distribution_alignment(
            stage1_cfg=p0_cfg, device=device,
            stage1_coarse_mean=stage1_coarse_mean,
            stage1_coarse_std=stage1_coarse_std,
            v8v6_cache_dir=args.stage1_v8v6_substitute_cache,
            batch_iter=_fresh_batch_iter(),
        ))

        # ─── Check 9: memory + wallclock ────────────────────────────────
        _record("memory_wallclock", check_9_memory_wallclock(
            fwd_args=base_fwd_args, batch=sample_batches[0],
            n_iters=int(args.n_mem_iters),
        ))

    # ─── Write outputs ──────────────────────────────────────────────────
    out_json = args.out_dir / "p0_stats.json"
    out_md = args.out_dir / "p0_summary.md"
    out_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    _write_summary_md(stats, out_md)

    all_pass = all(
        stats["checks"][n].get("pass", False) for n in stats["check_order"]
    )
    print(f"[p0] wrote {out_md}")
    print(f"[p0] wrote {out_json}")
    print(f"[p0] OVERALL pass={all_pass}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
