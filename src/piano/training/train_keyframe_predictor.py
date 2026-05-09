"""Stage 1 of AnchorDiff v8: train the keyframe position predictor.

Deterministic regression: minimize MSE on (K_MAX, 6, 3) keyframe
positions vs the offline-precomputed targets from
``piano.data.keyframe_extraction``. Padded slots are masked.

Conditions are the same as Stage B v7's:
  - text (CLIP per-token)
  - object trajectory (24-D)
  - z_int (26-D = contact_state 5 + contact_target_xyz 15 + phase 3
    + support 3) — full v18 z_int including spatial anchor
  - init_pose (frame-0 SMPL-22 joints flattened, 66-D)

Stage 1 is small (~5M params) and converges quickly (~30 epochs).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from accelerate import Accelerator
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from piano.data.dataset import HOIDataset, AugmentConfig, collate_hoi
from piano.models.keyframe_predictor import (
    KeyframePredictor, KeyframePredictorConfig, keyframe_l2_loss,
)
from piano.models.motion_anchordiff import ZIntDims, pack_z_int
from piano.models.object_encoder import ObjectEncoder
from piano.training.anchor_consistency_loss import lift_object_local_to_world
from piano.training.train_anchordiff import (
    _phase_to_softmax, _support_to_softmax, _resolve_subject_split,
)


def _build_object_traj_24(
    obj_com: torch.Tensor,
    obj_rot6d: torch.Tensor,
    contact_target_xyz: torch.Tensor,
    obj_pos_world: torch.Tensor,
    obj_rot_world: torch.Tensor,
) -> torch.Tensor:
    """Stage 1 uses 24-D object_traj (COM 3 + rot6d 6 + 5 anchors world * 3)."""
    target_world = lift_object_local_to_world(
        contact_target_xyz, obj_pos_world, obj_rot_world,
    ).reshape(obj_com.shape[0], obj_com.shape[1], -1)
    return torch.cat([obj_com, obj_rot6d, target_world], dim=-1)
from piano.training.trainer import (
    build_optimizer_with_decay_groups, build_scheduler,
)
from piano.utils.clip_utils import encode_text_per_token, load_clip_text_encoder
from piano.utils.io_utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    accelerator = Accelerator(mixed_precision=cfg.training.get("mixed_precision", "bf16"))
    device = accelerator.device

    output_dir = ensure_dir(Path(cfg.output_dir))
    accelerator.print(f"===== Stage 1: keyframe predictor =====")
    accelerator.print(f"output_dir = {output_dir}")
    accelerator.print(f"smoke_test = {args.smoke_test}")

    # --- Datasets (no augment for keyframed rep) ---
    train_subj = _resolve_subject_split(cfg, "train")
    val_subj = _resolve_subject_split(cfg, "val")

    pseudo_label_subdir = cfg.data.get("pseudo_label_subdir", None)
    train_datasets = []
    val_datasets = []
    for entry in cfg.data.datasets:
        sub = (str(Path(entry.root) / pseudo_label_subdir)
               if pseudo_label_subdir is not None else None)
        common = dict(
            root=entry.root,
            pseudo_label_dir=sub,
            max_seq_length=cfg.data.max_seq_length,
            augment=AugmentConfig(enabled=False),
            surface_obj_pose=True,
            support_collapse_hand_support=bool(
                cfg.data.get("support_collapse_hand_support", True)
            ),
            motion_representation="smpl_pose_135_keyframed",
        )
        train_datasets.append(HOIDataset(subject_id_filter=train_subj, **common))
        val_datasets.append(HOIDataset(subject_id_filter=val_subj, **common))
    train_ds = torch.utils.data.ConcatDataset(train_datasets)
    val_ds = torch.utils.data.ConcatDataset(val_datasets)
    accelerator.print(f"Train: {len(train_ds)} clips  Val: {len(val_ds)} clips")

    train_loader = DataLoader(
        train_ds, batch_size=int(cfg.training.batch_size),
        shuffle=True, num_workers=int(cfg.training.num_workers),
        collate_fn=collate_hoi, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=int(cfg.training.batch_size),
        shuffle=False, num_workers=int(cfg.training.num_workers),
        collate_fn=collate_hoi, drop_last=False,
    )

    # --- Model + condition encoders ---
    z_dims = ZIntDims(
        num_parts=int(cfg.model.z_int.num_parts),
        phase_classes=int(cfg.model.z_int.phase_classes),
        support_classes=int(cfg.model.z_int.support_classes),
    )
    object_encoder = ObjectEncoder(
        num_input_points=int(cfg.model.object_encoder.num_input_points),
        num_output_tokens=int(cfg.model.object_encoder.num_output_tokens),
        feature_dim=int(cfg.model.object_encoder.feature_dim),
    )
    # NOTE: object_encoder unused by Stage 1 (we use object_traj only,
    # not point cloud) — kept here for future use; could be removed.

    clip_model = load_clip_text_encoder(
        device=device,
        model_name=str(cfg.model.text_encoder.clip_version),
        download_root=str(cfg.model.text_encoder.download_root),
    )

    pred_cfg = KeyframePredictorConfig(
        object_traj_dim=int(cfg.model.predictor.object_traj_dim),
        z_int_dim=int(cfg.model.predictor.z_int_dim),
        text_dim=int(cfg.model.predictor.text_dim),
        init_pose_dim=int(cfg.model.predictor.init_pose_dim),
        d_model=int(cfg.model.predictor.d_model),
        n_layers=int(cfg.model.predictor.n_layers),
        n_heads=int(cfg.model.predictor.n_heads),
        ff_mult=int(cfg.model.predictor.ff_mult),
        dropout=float(cfg.model.predictor.dropout),
        num_keyjoints=int(cfg.model.predictor.num_keyjoints),
        k_max=int(cfg.model.predictor.k_max),
    )
    model = KeyframePredictor(pred_cfg)
    n_params = sum(p.numel() for p in model.parameters())
    accelerator.print(f"KeyframePredictor params: {n_params/1e6:.2f}M")

    # --- Optimizer + scheduler ---
    optimizer = build_optimizer_with_decay_groups(
        modules=[model],
        lr=float(cfg.training.optimizer.lr),
        weight_decay=float(cfg.training.optimizer.weight_decay),
        betas=tuple(cfg.training.optimizer.betas),
    )
    accum = int(cfg.training.get("gradient_accumulation_steps", 1))
    steps_per_epoch = max(1, len(train_loader) // accum)
    total_steps = steps_per_epoch * int(cfg.training.num_epochs)
    scheduler = build_scheduler(
        optimizer,
        int(cfg.training.scheduler.warmup_steps),
        total_steps,
    )

    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler,
    )
    val_loader = accelerator.prepare(val_loader)

    # --- Step ---
    def step_fn(batch: dict) -> dict[str, torch.Tensor]:
        contact_state = batch["contact_state"].to(device)              # (B, T, 5)
        contact_target_xyz = batch["contact_target_xyz"].to(device)    # (B, T, 5, 3)
        phase = batch["phase"].to(device)                              # (B, T)
        support = batch["support"].to(device)                          # (B, T)
        obj_com = batch["obj_com_canonical"].to(device)                # (B, T, 3)
        obj_rot6d = batch["obj_rot6d_canonical"].to(device)            # (B, T, 6)
        obj_pos_world = batch["object_positions"].to(device)           # (B, T, 3)
        obj_rot_world = batch["object_rotations"].to(device)           # (B, T, 3)
        joints = batch["joints"].to(device)                            # (B, T, 22, 3)
        seq_len = batch["seq_len"].to(device)                          # (B,)
        kf_indices = batch["keyframe_indices"].to(device)              # (B, K_MAX)
        kf_targets = batch["keyframe_targets"].to(device).float()      # (B, K_MAX, 6, 3)
        kf_mask = batch["keyframe_mask"].to(device).float()            # (B, K_MAX)
        text = batch["text"]                                           # list[str]

        B, T = contact_state.shape[:2]
        seq_idx_b = torch.arange(T, device=device).unsqueeze(0)
        seq_mask = (seq_idx_b < seq_len.unsqueeze(1)).float()           # (B, T)

        phase_soft = _phase_to_softmax(phase, z_dims.phase_classes)
        support_soft = _support_to_softmax(support, z_dims.support_classes)
        z_int = pack_z_int(
            contact_state, contact_target_xyz, phase_soft, support_soft, z_dims,
        )                                                              # (B, T, 26)

        obj_traj = _build_object_traj_24(
            obj_com=obj_com, obj_rot6d=obj_rot6d,
            contact_target_xyz=contact_target_xyz,
            obj_pos_world=obj_pos_world, obj_rot_world=obj_rot_world,
        )                                                              # (B, T, 24)

        init_pose = joints[:, 0, :, :].reshape(B, -1)                  # (B, 66)
        text_features, _ = encode_text_per_token(clip_model, text, device)

        pred = model(
            obj_traj=obj_traj, z_int=z_int, init_pose=init_pose,
            text=text_features.float(),
            keyframe_indices=kf_indices,
            seq_mask=seq_mask,
        )                                                              # (B, K_MAX, 6, 3)
        loss = keyframe_l2_loss(pred, kf_targets, kf_mask)
        return {"loss": loss}

    # --- Smoke test ---
    if args.smoke_test:
        accelerator.print("Running smoke test...")
        batch = next(iter(train_loader))
        out = step_fn(batch)
        accelerator.print(f"  loss={out['loss'].item():.4f}")
        accelerator.backward(out["loss"])
        optimizer.step()
        accelerator.print("Smoke test PASSED.")
        return

    # --- Train loop ---
    import json, time
    metrics_path = Path(output_dir) / "metrics.jsonl"
    metrics_f = open(metrics_path, "a", encoding="utf-8")

    best_val_loss = float("inf")
    global_step = 0
    for epoch in range(int(cfg.training.num_epochs)):
        model.train()
        t_epoch_start = time.time()
        epoch_losses = []
        for i, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                out = step_fn(batch)
                accelerator.backward(out["loss"])
                accelerator.clip_grad_norm_(model.parameters(),
                                            float(cfg.training.max_grad_norm))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            epoch_losses.append(float(out["loss"].item()))
            global_step += 1
            if global_step % int(cfg.logging.log_every_n_steps) == 0:
                metrics_f.write(json.dumps({
                    "event": "step", "epoch": epoch, "global_step": global_step,
                    "loss": float(out["loss"].item()),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "time": time.time(),
                }) + "\n")
                metrics_f.flush()

        epoch_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        epoch_time = time.time() - t_epoch_start
        accelerator.print(f"ep{epoch:3d} train_loss={epoch_loss:.4f} time={epoch_time:.1f}s")
        metrics_f.write(json.dumps({
            "event": "epoch", "epoch": epoch, "global_step": global_step,
            "loss": epoch_loss, "epoch_time_sec": epoch_time, "time": time.time(),
        }) + "\n")
        metrics_f.flush()

        # Val + ckpt every val_every_epochs.
        if (epoch + 1) % int(cfg.training.val_every_epochs) == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for vb in val_loader:
                    vout = step_fn(vb)
                    val_losses.append(float(vout["loss"].item()))
            val_loss = sum(val_losses) / max(len(val_losses), 1)
            accelerator.print(f"  VAL ep{epoch:3d} loss={val_loss:.4f}")
            metrics_f.write(json.dumps({
                "event": "val", "epoch": epoch, "global_step": global_step,
                "val_loss": val_loss, "time": time.time(),
            }) + "\n")
            metrics_f.flush()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "model": accelerator.unwrap_model(model).state_dict(),
                    "epoch": epoch, "val_loss": val_loss, "config": OmegaConf.to_container(cfg),
                }, output_dir / "best_val.pt")
                accelerator.print(f"  saved best_val.pt (val_loss={val_loss:.4f})")

            torch.save({
                "model": accelerator.unwrap_model(model).state_dict(),
                "epoch": epoch, "val_loss": val_loss, "config": OmegaConf.to_container(cfg),
            }, output_dir / f"epoch_{epoch+1:04d}.pt")

    # Final ckpt
    torch.save({
        "model": accelerator.unwrap_model(model).state_dict(),
        "epoch": int(cfg.training.num_epochs) - 1,
        "config": OmegaConf.to_container(cfg),
    }, output_dir / "final.pt")
    metrics_f.close()
    accelerator.print("Stage 1 training complete.")


if __name__ == "__main__":
    main()
