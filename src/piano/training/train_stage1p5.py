"""Stage-1.5 (Interaction Plan) training entry.

Trains ``Stage1p5Denoiser`` (in ``piano.models.stage1p5_interaction``)
to predict the (C41, S4) interaction-plan cond tensors that Stage-2
PB1 consumes.

Reuses Stage-2's infrastructure:

  - ``_build_dataset`` from ``train_anchordiff`` (PIANO clip loader).
  - ``GaussianDiffusion`` / ``DiffusionConfig`` from
    ``piano.models.motion_anchordiff``.
  - ``ObjectEncoder`` and CLIP text encoder.
  - ``run_training_loop`` from ``piano.training.trainer``.
  - Stage-2's oracle ``Stage2ConditionBundle`` is surfaced by the
    dataset directly as ``batch["stage2_coarse_extra"]`` (C41) and
    ``batch["stage2_support"]`` (S4) when the right data variants are
    configured.

Design source: ``analyses/2026-05-29_stage1_and_stage1_5_design.md``.
"""
from __future__ import annotations

import argparse
from typing import Any

import torch
import torch.nn.functional as F
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader

from piano.data.dataset import collate_hoi
from piano.models.object_encoder import ObjectEncoder
from piano.data.stage1_coarse_oracle import (
    extract_coarse_v1_batched,
    load_stage1_coarse_norm,
)
from piano.models.motion_anchordiff import (
    DiffusionConfig,
    GaussianDiffusion,
    _extract,
)
from piano.models.stage1p5_interaction import (
    STAGE1P5_C41_DIM,
    STAGE1P5_S4_DIM,
    STAGE1P5_TOTAL_DIM,
    Stage1p5Denoiser,
    Stage1p5DenoiserConfig,
)
from piano.training.train_anchordiff import _build_dataset
from piano.training.trainer import (
    build_scheduler,
    run_training_loop,
)
from piano.utils.clip_utils import encode_text_per_token, load_clip_text_encoder


def build_stage1p5_step_fn(
    model: Stage1p5Denoiser,
    diffusion: GaussianDiffusion,
    object_encoder: ObjectEncoder,
    clip_model: torch.nn.Module | None,
    device: torch.device,
    stage1_coarse_mean_t: Tensor,
    stage1_coarse_std_t: Tensor,
    *,
    cfg_drop_prob: float = 0.15,
    w_x0_c41: float = 1.0,
    w_x0_s4: float = 1.0,
    w_c41_jl: float = 0.1,
    c41_joint_limit_m: float = 1.5,
    w_s4_stance: float = 0.5,
    w_s4_phase: float = 0.05,
    w_s4_walking: float = 0.5,
    use_min_snr_weighting: bool = True,
    min_snr_gamma: float = 5.0,
):
    """Stage-1.5 step_fn closure.

    Total loss per design doc §"Training loss":
      L = w_x0_c41 * MSE(c41_pred, c41_gt)
        + w_x0_s4  * MSE(s4_pred, s4_gt)
        + w_c41_jl * mean(relu(||c41_delta|| − R_joint))
        + w_s4_stance * BCE(s4[:, :2], gt[:, :2])              # foot stance
        + w_s4_phase  * ((s4_phase_sin² + s4_phase_cos²) − 1)²
        + w_s4_walking * BCE(s4[:, 4], gt[:, 4])               # walking_mask
    """
    _diff_for_read = (
        diffusion.module if hasattr(diffusion, "module") else diffusion
    )

    def step_fn(_model, batch: dict, global_step: int = 0) -> dict[str, Tensor]:
        # ─── Required keys ───
        if "stage2_coarse_extra" not in batch:
            raise KeyError(
                "Stage-1.5 training requires batch['stage2_coarse_extra']. "
                "Configure data.r29_coarse_variant=C41-current."
            )
        if "stage2_support" not in batch:
            raise KeyError(
                "Stage-1.5 training requires batch['stage2_support']. "
                "Configure data.r29_support_variant=S4-S1-phase-footstep."
            )

        motion = batch["motion"].to(device)                          # (B, T, 135)
        rest_offsets = batch["rest_offsets"].to(device).float()      # (B, 22, 3)
        object_pc = batch["object_pc"].to(device)
        c41_gt = batch["stage2_coarse_extra"].to(device).float()     # (B, T, 18)
        s4_gt = batch["stage2_support"].to(device).float()           # (B, T, 13)
        seq_len = batch["seq_len"].to(device)

        B, T, _ = motion.shape
        seq_idx = torch.arange(T, device=device).unsqueeze(0)
        seq_mask = (seq_idx < seq_len.unsqueeze(1)).float()          # (B, T)

        if c41_gt.shape[-1] != STAGE1P5_C41_DIM:
            raise RuntimeError(
                f"stage2_coarse_extra dim {c41_gt.shape[-1]} != "
                f"{STAGE1P5_C41_DIM}; check r29_coarse_variant."
            )
        if s4_gt.shape[-1] != STAGE1P5_S4_DIM:
            raise RuntimeError(
                f"stage2_support dim {s4_gt.shape[-1]} != "
                f"{STAGE1P5_S4_DIM}; check r29_support_variant."
            )

        # ─── Object trajectory (canonical 9-D, matches Stage-2) ───
        obj_com = batch["obj_com_canonical"].to(device)               # (B, T, 3)
        obj_rot6d = batch["obj_rot6d_canonical"].to(device)          # (B, T, 6)
        object_traj = torch.cat([obj_com, obj_rot6d], dim=-1)

        # ─── Object tokens ───
        obj_tokens = object_encoder(object_pc)

        # ─── Text features ───
        if clip_model is not None and "text" in batch:
            text_features, _text_mask = encode_text_per_token(
                clip_model, batch["text"], device,
            )
            text_features = text_features.float()
        else:
            text_features = None

        # ─── Oracle Stage-1 coarse_v1 (z-scored), passed as cond ───
        # During training we use the same oracle Stage-1 output Stage-2
        # consumes. At inference we'll substitute Stage-1's prediction.
        coarse_v1_raw = extract_coarse_v1_batched(
            motion=motion, rest_offsets=rest_offsets,
        )
        coarse_v1 = (coarse_v1_raw - stage1_coarse_mean_t) / stage1_coarse_std_t

        cond: dict[str, Tensor] = {
            "object_world_traj": object_traj,
            "object_tokens": obj_tokens,
            "stage1_coarse": coarse_v1,
        }
        if text_features is not None:
            cond["text"] = text_features

        # ─── Diffusion training step ───
        x0 = torch.cat([c41_gt, s4_gt], dim=-1)                     # (B, T, 31)
        t = torch.randint(
            0, _diff_for_read.num_steps, (B,), device=device, dtype=torch.long,
        )
        noise = torch.randn_like(x0)
        sqrt_a = _extract(_diff_for_read.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_om = _extract(_diff_for_read.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        x_t = sqrt_a * x0 + sqrt_om * noise

        if cfg_drop_prob > 0 and _model.training:
            cond_drop_mask = (
                torch.rand((B,), device=device) < cfg_drop_prob
            )
        else:
            cond_drop_mask = None

        x0_pred = _model(x_t, t, cond, cond_drop_mask=cond_drop_mask)  # (B, T, 31)
        c41_pred = x0_pred[..., :STAGE1P5_C41_DIM]
        s4_pred = x0_pred[..., STAGE1P5_C41_DIM:]

        # ─── min-SNR-γ weight ───
        if use_min_snr_weighting:
            alpha_bar = _diff_for_read.alphas_cumprod.gather(0, t)
            snr = alpha_bar / (1.0 - alpha_bar + 1e-8)
            snr_clamped = torch.clamp_max(snr, float(min_snr_gamma))
            w_b = snr_clamped
            w_b_norm = (w_b / w_b.mean().clamp_min(1e-8)).view(-1, 1)
        else:
            w_b_norm = torch.ones(B, 1, device=device)

        # ─── Loss 1: C41 MSE ───
        c41_per_frame = (c41_pred - c41_gt).pow(2).sum(-1) * w_b_norm  # (B, T)
        mse_c41 = (c41_per_frame * seq_mask).sum() / seq_mask.sum().clamp_min(1.0)

        # ─── Loss 2: S4 MSE ───
        s4_per_frame = (s4_pred - s4_gt).pow(2).sum(-1) * w_b_norm
        mse_s4 = (s4_per_frame * seq_mask).sum() / seq_mask.sum().clamp_min(1.0)

        # ─── Loss 3: C41 joint-limit (clip per-key-joint Δ at R_joint) ───
        # C41 layout: [0:15] = 5 joints × Δxyz; [15:18] = pelvis Δxzy.
        if w_c41_jl > 0:
            c41_joints = c41_pred[..., :15].view(B, T, 5, 3)
            joint_norm = c41_joints.norm(dim=-1)                       # (B, T, 5)
            over = F.relu(joint_norm - c41_joint_limit_m)
            jl_per_frame = over.mean(-1)                               # (B, T)
            jl = (jl_per_frame * seq_mask).sum() / seq_mask.sum().clamp_min(1.0)
        else:
            jl = torch.zeros((), device=device, dtype=mse_c41.dtype)

        # ─── Loss 4: S4 stance BCE (channels 0, 1) ───
        # The model outputs raw logits via Linear (no sigmoid); use
        # binary_cross_entropy_with_logits on logits + clip GT to [0, 1].
        if w_s4_stance > 0:
            stance_pred_logits = s4_pred[..., :2]                     # (B, T, 2)
            stance_gt = s4_gt[..., :2].clamp(0.0, 1.0)                # (B, T, 2)
            stance_bce = F.binary_cross_entropy_with_logits(
                stance_pred_logits, stance_gt, reduction="none",
            ).mean(-1)                                                 # (B, T)
            stance_bce = (
                stance_bce * seq_mask
            ).sum() / seq_mask.sum().clamp_min(1.0)
        else:
            stance_bce = torch.zeros((), device=device, dtype=mse_s4.dtype)

        # ─── Loss 5: S4 phase unit-norm violation (channels 5-9) ───
        # phase_sin_L, phase_cos_L, phase_sin_R, phase_cos_R.
        if w_s4_phase > 0:
            phase = s4_pred[..., 5:9]                                  # (B, T, 4)
            r2_l = phase[..., 0].pow(2) + phase[..., 1].pow(2)
            r2_r = phase[..., 2].pow(2) + phase[..., 3].pow(2)
            unit_violation = ((r2_l - 1).pow(2) + (r2_r - 1).pow(2))   # (B, T)
            phase_unit = (
                unit_violation * seq_mask
            ).sum() / seq_mask.sum().clamp_min(1.0)
        else:
            phase_unit = torch.zeros((), device=device, dtype=mse_s4.dtype)

        # ─── Loss 6: walking_mask BCE (channel 4) ───
        if w_s4_walking > 0:
            walking_pred_logits = s4_pred[..., 4]                      # (B, T)
            walking_gt = s4_gt[..., 4].clamp(0.0, 1.0)                # (B, T)
            walking_bce = F.binary_cross_entropy_with_logits(
                walking_pred_logits, walking_gt, reduction="none",
            )                                                          # (B, T)
            walking_bce = (
                walking_bce * seq_mask
            ).sum() / seq_mask.sum().clamp_min(1.0)
        else:
            walking_bce = torch.zeros((), device=device, dtype=mse_s4.dtype)

        loss = (
            w_x0_c41 * mse_c41
            + w_x0_s4 * mse_s4
            + w_c41_jl * jl
            + w_s4_stance * stance_bce
            + w_s4_phase * phase_unit
            + w_s4_walking * walking_bce
        )

        return {
            "loss": loss,
            "mse_c41": mse_c41.detach(),
            "mse_s4": mse_s4.detach(),
            "c41_joint_limit": jl.detach(),
            "s4_stance_bce": stance_bce.detach(),
            "s4_phase_unit": phase_unit.detach(),
            "s4_walking_bce": walking_bce.detach(),
        }

    return step_fn


def _make_dataloader(
    dataset, batch_size: int, num_workers: int, shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=True,
        drop_last=shuffle,
        collate_fn=collate_hoi,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    # find_unused_parameters=True: matches Stage-2 trainer (defensive).
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.get(
            "gradient_accumulation_steps", 1,
        ),
        mixed_precision=cfg.training.get("mixed_precision", "bf16"),
        kwargs_handlers=[ddp_kwargs],
    )
    set_seed(int(cfg.training.get("seed", 42)))
    device = accelerator.device

    accelerator.print("===== Stage-1.5 (Interaction Plan) training =====")
    accelerator.print(f"output_dir = {cfg.output_dir}")
    accelerator.print(f"smoke_test = {args.smoke_test}")

    train_dataset = _build_dataset(cfg, bucket="train", augment=True)
    val_dataset = None
    if int(cfg.training.get("val_every_epochs", 0)) > 0:
        val_dataset = _build_dataset(cfg, bucket="val", augment=False)
    accelerator.print(f"Train: {len(train_dataset)} clips")
    if val_dataset is not None:
        accelerator.print(f"Val:   {len(val_dataset)} clips")

    train_loader = _make_dataloader(
        train_dataset,
        batch_size=int(cfg.training.batch_size),
        num_workers=int(cfg.training.get("num_workers", 4)),
        shuffle=True,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = _make_dataloader(
            val_dataset,
            batch_size=int(cfg.training.batch_size),
            num_workers=int(cfg.training.get("num_workers", 4)),
            shuffle=False,
        )

    denoiser_cfg = Stage1p5DenoiserConfig(
        motion_dim=int(cfg.model.denoiser.motion_dim),
        stage1_coarse_dim=int(cfg.model.denoiser.stage1_coarse_dim),
        object_traj_dim=int(cfg.model.denoiser.object_traj_dim),
        text_dim=int(cfg.model.denoiser.text_dim),
        object_token_dim=int(cfg.model.denoiser.object_token_dim),
        object_num_tokens=int(cfg.model.denoiser.object_num_tokens),
        d_model=int(cfg.model.denoiser.d_model),
        n_layers=int(cfg.model.denoiser.n_layers),
        n_heads=int(cfg.model.denoiser.n_heads),
        ff_mult=int(cfg.model.denoiser.ff_mult),
        dropout=float(cfg.model.denoiser.dropout),
        max_seq_length=int(cfg.data.max_seq_length),
        use_text=bool(cfg.model.denoiser.get("use_text", True)),
    )
    if denoiser_cfg.motion_dim != STAGE1P5_TOTAL_DIM:
        raise ValueError(
            f"Stage-1.5 motion_dim must be {STAGE1P5_TOTAL_DIM} "
            f"(={STAGE1P5_C41_DIM}+{STAGE1P5_S4_DIM}); got "
            f"{denoiser_cfg.motion_dim}."
        )
    model = Stage1p5Denoiser(denoiser_cfg)

    diff_cfg = DiffusionConfig(
        num_steps=int(cfg.model.diffusion.num_steps),
        schedule=str(cfg.model.diffusion.schedule),
        objective=str(cfg.model.diffusion.get("objective", "ddpm")),
        prediction_target=str(
            cfg.model.diffusion.get("prediction_target", "x0"),
        ),
    )
    diffusion = GaussianDiffusion(diff_cfg)

    object_encoder = ObjectEncoder(
        num_input_points=int(cfg.model.object_encoder.num_input_points),
        num_output_tokens=int(cfg.model.object_encoder.num_output_tokens),
        feature_dim=int(cfg.model.object_encoder.feature_dim),
    )

    if int(cfg.model.denoiser.text_dim) > 0:
        clip_model = load_clip_text_encoder(
            device=device,
            model_name=str(cfg.model.text_encoder.clip_version),
            download_root=str(
                cfg.model.text_encoder.get("download_root", "cache/clip"),
            ),
        )
    else:
        clip_model = None

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(object_encoder.parameters()),
        lr=float(cfg.training.optimizer.lr),
        weight_decay=float(cfg.training.optimizer.weight_decay),
        betas=tuple(cfg.training.optimizer.get("betas", [0.9, 0.999])),
    )
    total_steps = int(
        cfg.training.num_epochs * len(train_loader)
        // int(cfg.training.get("gradient_accumulation_steps", 1))
    )
    scheduler = build_scheduler(
        optimizer,
        warmup_steps=int(cfg.training.scheduler.get("warmup_steps", 500)),
        total_steps=max(total_steps, 1),
    )

    norm_mean_np, norm_std_np = load_stage1_coarse_norm(
        str(cfg.data.stage1_coarse_cache_root)
    )
    stage1_coarse_mean_t = (
        torch.from_numpy(norm_mean_np).to(device).float()
    )
    stage1_coarse_std_t = (
        torch.from_numpy(norm_std_np).to(device).float()
    )

    (
        model, object_encoder, optimizer, train_loader, scheduler,
    ) = accelerator.prepare(
        model, object_encoder, optimizer, train_loader, scheduler,
    )
    if val_loader is not None:
        val_loader = accelerator.prepare(val_loader)
    diffusion = diffusion.to(device)

    step_fn = build_stage1p5_step_fn(
        model=model,
        diffusion=diffusion,
        object_encoder=object_encoder,
        clip_model=clip_model,
        device=device,
        stage1_coarse_mean_t=stage1_coarse_mean_t,
        stage1_coarse_std_t=stage1_coarse_std_t,
        cfg_drop_prob=float(cfg.model.get("cfg_drop_prob", 0.15)),
        w_x0_c41=float(cfg.loss.w_x0_c41),
        w_x0_s4=float(cfg.loss.w_x0_s4),
        w_c41_jl=float(cfg.loss.w_c41_jl),
        c41_joint_limit_m=float(cfg.loss.get("c41_joint_limit_m", 1.5)),
        w_s4_stance=float(cfg.loss.w_s4_stance),
        w_s4_phase=float(cfg.loss.w_s4_phase),
        w_s4_walking=float(cfg.loss.w_s4_walking),
        use_min_snr_weighting=bool(
            cfg.loss.get("use_min_snr_weighting", True),
        ),
        min_snr_gamma=float(cfg.loss.get("min_snr_gamma", 5.0)),
    )

    if args.smoke_test:
        accelerator.print("Smoke test: running one batch.")
        batch = next(iter(train_loader))
        out = step_fn(model, batch, global_step=0)
        accelerator.print(
            f"loss = {out['loss'].item():.4f}  "
            f"mse_c41 = {out['mse_c41'].item():.4f}  "
            f"mse_s4 = {out['mse_s4'].item():.4f}"
        )
        accelerator.backward(out["loss"])
        accelerator.print("Smoke test backward OK.")
        return

    run_training_loop(
        accelerator=accelerator,
        model=model,
        dataloader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        step_fn=step_fn,
        num_epochs=int(cfg.training.num_epochs),
        output_dir=cfg.output_dir,
        log_every=int(cfg.logging.get("log_every_n_steps", 50)),
        save_every_epochs=int(cfg.logging.get("save_every_n_epochs", 10)),
        max_grad_norm=float(cfg.training.get("max_grad_norm", 1.0)),
        extra_modules={"object_encoder": object_encoder},
        val_dataloader=val_loader,
        val_every_epochs=int(cfg.training.get("val_every_epochs", 0)),
        val_best_key=str(cfg.training.get("val_best_key", "loss")),
    )


if __name__ == "__main__":
    main()
