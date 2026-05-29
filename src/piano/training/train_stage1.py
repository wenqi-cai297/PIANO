"""Stage-1 (Trajectory & Orientation) training entry.

Trains ``Stage1Denoiser`` (in ``piano.models.stage1_trajectory``) to
predict the 23-D ``stage1_coarse`` representation that Stage-2 PB1
consumes via ``cond["stage1_coarse"]``.

Reuses Stage-2's infrastructure:

  - ``_build_dataset`` from ``train_anchordiff`` (PIANO clip loader).
  - ``GaussianDiffusion`` + ``DiffusionConfig`` from
    ``piano.models.motion_anchordiff`` (cosine schedule, x0-prediction).
  - ``ObjectEncoder`` (PointNet++) and CLIP text encoder.
  - ``run_training_loop`` from ``piano.training.trainer``.
  - GT 23-D extraction from
    ``piano.data.stage1_coarse_oracle.extract_coarse_v1_batched``.

Design source: ``analyses/2026-05-29_stage1_and_stage1_5_design.md``.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
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
from piano.models.stage1_trajectory import (
    STAGE1_COARSE_DIM,
    Stage1Denoiser,
    Stage1DenoiserConfig,
)
from piano.training.train_anchordiff import _build_dataset
from piano.training.trainer import (
    build_scheduler,
    run_training_loop,
)
from piano.utils.clip_utils import encode_text_per_token, load_clip_text_encoder


# ----------------------------------------------------------------------------
# Step function
# ----------------------------------------------------------------------------


def build_stage1_step_fn(
    model: Stage1Denoiser,
    diffusion: GaussianDiffusion,
    object_encoder: ObjectEncoder,
    clip_model: torch.nn.Module | None,
    device: torch.device,
    stage1_coarse_mean_t: Tensor,
    stage1_coarse_std_t: Tensor,
    *,
    cfg_drop_prob: float = 0.15,
    w_x0: float = 1.0,
    w_vel: float = 1.0,
    w_yaw_smooth: float = 0.02,
    use_min_snr_weighting: bool = True,
    min_snr_gamma: float = 5.0,
):
    """Return the Stage-1 ``step_fn(model, batch, global_step)`` closure.

    Training target is the **z-scored** 23-D stage1_coarse — same
    normalisation Stage-2 PB1 was trained against. This way Stage-1's
    output drops directly into Stage-2's cond[\"stage1_coarse\"] at
    inference with no extra re-scaling.

    Per design doc §"Training loss":
      L = w_x0 * MSE(x0_pred, x0_gt_normed)
        + w_vel * MSE(vel(x0_pred), vel(x0_gt_normed))
        + w_yaw_smooth * mean(|Δ²yaw_unwrapped(raw_x0_pred)|)
    """
    # Unwrap DDP for read-only diffusion access.
    _diff_for_read = (
        diffusion.module if hasattr(diffusion, "module") else diffusion
    )

    def step_fn(_model, batch: dict, global_step: int = 0) -> dict[str, Tensor]:
        motion = batch["motion"].to(device)                    # (B, T, 135)
        rest_offsets = batch["rest_offsets"].to(device).float()  # (B, 22, 3)
        object_pc = batch["object_pc"].to(device)
        seq_len = batch["seq_len"].to(device)                  # (B,)

        B, T, _ = motion.shape
        seq_idx = torch.arange(T, device=device).unsqueeze(0)
        seq_mask = (seq_idx < seq_len.unsqueeze(1)).float()    # (B, T)

        # ─── Build object trajectory (3 pos + 6 rot6d = 9) ───
        # Match Stage-2's convention exactly: use the CANONICAL COM and
        # CANONICAL rot6d. The Stage-2 trainer at train_anchordiff.py:327
        # uses obj_com_canonical for this same 9-D channel.
        obj_com = batch["obj_com_canonical"].to(device)        # (B, T, 3)
        obj_rot6d = batch["obj_rot6d_canonical"].to(device)    # (B, T, 6)
        object_traj = torch.cat([obj_com, obj_rot6d], dim=-1)  # (B, T, 9)

        # ─── Object tokens ───
        obj_tokens = object_encoder(object_pc)                 # (B, N_obj, D_obj)

        # ─── Text features ───
        if clip_model is not None and "text" in batch:
            text_features, _text_mask = encode_text_per_token(
                clip_model, batch["text"], device,
            )
            text_features = text_features.float()
        else:
            text_features = None

        # ─── GT target: z-scored 23-D stage1_coarse ───
        # Stage-2 PB1 was trained against this same z-scoring; Stage-1
        # learns the normalised output so its inference drops directly
        # into Stage-2's cond["stage1_coarse"] with no re-scaling.
        coarse_v1_raw = extract_coarse_v1_batched(
            motion=motion, rest_offsets=rest_offsets,
        )                                                       # (B, T, 23)
        if coarse_v1_raw.shape[-1] != STAGE1_COARSE_DIM:
            raise RuntimeError(
                f"extract_coarse_v1_batched returned {coarse_v1_raw.shape[-1]}D; "
                f"expected {STAGE1_COARSE_DIM}D."
            )
        coarse_v1 = (coarse_v1_raw - stage1_coarse_mean_t) / stage1_coarse_std_t

        # ─── Build cond dict for the denoiser ───
        cond: dict[str, Tensor] = {
            "object_world_traj": object_traj,
            "object_tokens": obj_tokens,
        }
        if text_features is not None:
            cond["text"] = text_features

        # ─── Diffusion forward (x₀-prediction) ───
        # x_t = sqrt(α_t) * x_0 + sqrt(1-α_t) * noise; predict x_0.
        x0 = coarse_v1                                          # (B, T, 23)
        t = torch.randint(
            0, _diff_for_read.num_steps, (B,), device=device, dtype=torch.long,
        )
        noise = torch.randn_like(x0)
        sqrt_a = _extract(_diff_for_read.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_om = _extract(_diff_for_read.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        x_t = sqrt_a * x0 + sqrt_om * noise

        # CFG drop mask on text + obj_tokens (never on obj_traj).
        if cfg_drop_prob > 0 and _model.training:
            cond_drop_mask = (
                torch.rand((B,), device=device) < cfg_drop_prob
            )
        else:
            cond_drop_mask = None

        x0_pred = _model(x_t, t, cond, cond_drop_mask=cond_drop_mask)  # (B, T, 23)

        # ─── Loss 1: MSE on x0 with optional min-SNR-γ weighting ───
        mse_per_dim = (x0_pred - x0).pow(2)                     # (B, T, 23)
        per_frame = mse_per_dim.sum(-1)                         # (B, T)

        if use_min_snr_weighting:
            alpha_bar = _diff_for_read.alphas_cumprod.gather(0, t)
            snr = alpha_bar / (1.0 - alpha_bar + 1e-8)
            snr_clamped = torch.clamp_max(snr, float(min_snr_gamma))
            w_b = snr_clamped                                    # x0-pred form
            w_b_norm = w_b / w_b.mean().clamp_min(1e-8)
            per_frame = per_frame * w_b_norm.view(-1, 1)

        mse_x0 = (per_frame * seq_mask).sum() / seq_mask.sum().clamp_min(1.0)
        mse_x0_unweighted = (
            mse_per_dim.sum(-1) * seq_mask
        ).sum() / seq_mask.sum().clamp_min(1.0)

        # ─── Loss 2: velocity consistency ───
        # 1-frame finite diff on the FIRST 9 channels (root_local xzy + vel xzy
        # + yaw sin/cos/vel — i.e. the kinematic block that matters most).
        if T >= 2 and w_vel > 0:
            vel_pred = x0_pred[:, 1:] - x0_pred[:, :-1]         # (B, T-1, 23)
            vel_gt = x0[:, 1:] - x0[:, :-1]
            vel_mask = seq_mask[:, 1:] * seq_mask[:, :-1]       # (B, T-1)
            vel_mse = ((vel_pred - vel_gt).pow(2).sum(-1) * vel_mask).sum() / vel_mask.sum().clamp_min(1.0)
        else:
            vel_mse = torch.zeros((), device=device, dtype=mse_x0.dtype)

        # ─── Loss 3: yaw 2nd-derivative smoothness on PRED ───
        # Pred yaw is at channels [6: 8] (sin, cos). We compute the angle via
        # atan2 and penalise its 2nd diff. Note: yaw_vel itself is channel 8,
        # but here we smooth the derived angle, not the predicted vel channel
        # (the vel-MSE term already constrains channel 8 directly).
        if T >= 3 and w_yaw_smooth > 0:
            yaw_pred = torch.atan2(x0_pred[..., 6], x0_pred[..., 7])  # (B, T)
            yaw_d1 = yaw_pred[:, 1:] - yaw_pred[:, :-1]
            # Wrap to [-π, π] to handle the atan2 discontinuity.
            yaw_d1 = (yaw_d1 + 3.14159265) % (2 * 3.14159265) - 3.14159265
            yaw_d2 = yaw_d1[:, 1:] - yaw_d1[:, :-1]                   # (B, T-2)
            yaw_mask = (
                seq_mask[:, 2:] * seq_mask[:, 1:-1] * seq_mask[:, :-2]
            )
            yaw_sm = (yaw_d2.abs() * yaw_mask).sum() / yaw_mask.sum().clamp_min(1.0)
        else:
            yaw_sm = torch.zeros((), device=device, dtype=mse_x0.dtype)

        loss = (
            w_x0 * mse_x0
            + w_vel * vel_mse
            + w_yaw_smooth * yaw_sm
        )

        return {
            "loss": loss,
            "mse_x0": mse_x0.detach(),
            "mse_x0_unweighted": mse_x0_unweighted.detach(),
            "vel_mse": vel_mse.detach(),
            "yaw_smooth": yaw_sm.detach(),
        }

    return step_fn


# ----------------------------------------------------------------------------
# Main entry
# ----------------------------------------------------------------------------


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
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run a single batch + backward to verify wiring; do not save.",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.get(
            "gradient_accumulation_steps", 1,
        ),
        mixed_precision=cfg.training.get("mixed_precision", "bf16"),
        kwargs_handlers=[ddp_kwargs],
    )
    set_seed(int(cfg.training.get("seed", 42)))
    device = accelerator.device

    accelerator.print("===== Stage-1 (Trajectory) training =====")
    accelerator.print(f"output_dir = {cfg.output_dir}")
    accelerator.print(f"smoke_test = {args.smoke_test}")

    # ─── Datasets ───
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

    # ─── Model ───
    denoiser_cfg = Stage1DenoiserConfig(
        motion_dim=int(cfg.model.denoiser.motion_dim),
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
    if denoiser_cfg.motion_dim != STAGE1_COARSE_DIM:
        raise ValueError(
            f"Stage-1 motion_dim must be {STAGE1_COARSE_DIM}; got "
            f"{denoiser_cfg.motion_dim}."
        )
    model = Stage1Denoiser(denoiser_cfg)

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

    # ─── Optimizer + scheduler ───
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

    # ─── Load Stage-1 norm stats so the GT target is z-scored ──────────
    # Stage-2 PB1 was trained against (raw - mean)/std using the same
    # cache; we match its target so Stage-1's output drops in directly.
    norm_mean_np, norm_std_np = load_stage1_coarse_norm(
        str(cfg.data.stage1_coarse_cache_root)
    )
    stage1_coarse_mean_t = (
        torch.from_numpy(norm_mean_np).to(device).float()
    )
    stage1_coarse_std_t = (
        torch.from_numpy(norm_std_np).to(device).float()
    )

    # ─── Prepare with accelerator ───
    (
        model, object_encoder, optimizer, train_loader, scheduler,
    ) = accelerator.prepare(
        model, object_encoder, optimizer, train_loader, scheduler,
    )
    if val_loader is not None:
        val_loader = accelerator.prepare(val_loader)
    diffusion = diffusion.to(device)

    step_fn = build_stage1_step_fn(
        model=model,
        diffusion=diffusion,
        object_encoder=object_encoder,
        clip_model=clip_model,
        device=device,
        stage1_coarse_mean_t=stage1_coarse_mean_t,
        stage1_coarse_std_t=stage1_coarse_std_t,
        cfg_drop_prob=float(cfg.model.get("cfg_drop_prob", 0.15)),
        w_x0=float(cfg.loss.w_x0),
        w_vel=float(cfg.loss.w_vel),
        w_yaw_smooth=float(cfg.loss.w_yaw_smooth),
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
            f"loss = {out['loss'].item():.4f}  mse_x0 = {out['mse_x0'].item():.4f}  "
            f"vel = {out['vel_mse'].item():.4f}  yaw_sm = {out['yaw_smooth'].item():.4e}"
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
        val_best_key=str(cfg.training.get("val_best_key", "mse_x0")),
    )


if __name__ == "__main__":
    main()
