"""Stage B (new): Train PIANO-AnchorDiff.

Anchor-conditioned continuous motion diffusion. Replaces the closed
MoMask Stage B track. Trained with GT ``z_int`` + classifier-free
guidance dropout; inference uses Stage A v10 predicted ``z_int``.

Design source of truth:
    analyses/2026-05-08_piano_anchordiff_design.md

Usage:
    accelerate launch --config_file configs/accelerate_config.yaml \\
        -m piano.training.train_anchordiff \\
        --config configs/training/anchordiff_v1.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader

from piano.data.dataset import (
    AugmentConfig,
    HOIDataset,
    build_subject_split,
    collate_hoi,
    extract_subject_id,
)
from piano.models.motion_anchordiff import (
    AnchorDenoiserConfig,
    AnchorDiffConfig,
    DiffusionConfig,
    MotionAnchorDiff,
    ZIntDims,
    pack_z_int,
)
from piano.models.object_encoder import ObjectEncoder
from piano.training.anchor_consistency_loss import (
    AnchorConsistencyConfig,
    anchor_consistency_loss,
)
from piano.training.trainer import (
    build_optimizer_with_decay_groups,
    build_scheduler,
    run_training_loop,
)
from piano.utils.clip_utils import encode_text_per_token, load_clip_text_encoder
from piano.utils.io_utils import load_json


# ---------------------------------------------------------------------------
# Dataset assembly (subject-split path, mirrors train_predictor.py)
# ---------------------------------------------------------------------------


def _read_metadata(roots: list) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    for entry in roots:
        root = Path(entry.root)
        meta_path = root / "metadata_clean.json"
        if not meta_path.exists():
            meta_path = root / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata not found in {root}")
        for m in load_json(meta_path):
            out.append((root.name, m))
    return out


def _resolve_subject_split(cfg, bucket: str) -> set | None:
    subj_cfg = cfg.data.get("subject_split")
    if subj_cfg is None or not subj_cfg.get("enabled", False):
        return None
    keys = sorted({
        (subset, extract_subject_id(subset, m.get("seq_id", "")))
        for subset, m in _read_metadata(cfg.data.datasets)
        if extract_subject_id(subset, m.get("seq_id", "")) is not None
    })
    splits = build_subject_split(
        keys,
        train_pct=subj_cfg.train_pct,
        val_pct=subj_cfg.val_pct,
        seed=subj_cfg.seed,
    )
    if bucket == "all":
        return None
    return splits[bucket]


def _build_dataset(cfg, bucket: str = "train", augment: bool = True) -> ConcatDataset:
    subj_filter = _resolve_subject_split(cfg, bucket)

    aug_cfg = cfg.data.get("augmentation", None)
    augment_obj = None
    if augment and aug_cfg is not None and aug_cfg.get("enabled", False):
        augment_obj = AugmentConfig(
            enabled=True,
            mirror_prob=float(aug_cfg.get("mirror_prob", 0.0)),
            mirror_duplicate=bool(aug_cfg.get("mirror_duplicate", False)),
            rotate_around_y_prob=float(aug_cfg.get("rotate_around_y_prob", 0.0)),
            pc_jitter_std=float(aug_cfg.get("pc_jitter_std", 0.0)),
        )

    pseudo_label_subdir = cfg.data.get("pseudo_label_subdir", None)
    pseudo_label_dir = cfg.data.get("pseudo_label_dir", None)

    datasets = []
    for entry in cfg.data.datasets:
        if pseudo_label_dir is not None:
            sub_dir = pseudo_label_dir
        elif pseudo_label_subdir is not None:
            sub_dir = str(Path(entry.root) / pseudo_label_subdir)
        else:
            sub_dir = None
        ds = HOIDataset(
            root=entry.root,
            pseudo_label_dir=sub_dir,
            max_seq_length=cfg.data.max_seq_length,
            subject_id_filter=subj_filter,
            augment=augment_obj,
            support_collapse_hand_support=bool(
                cfg.data.get("support_collapse_hand_support", True)
            ),
            surface_obj_pose=True,
        )
        datasets.append(ds)
    return ConcatDataset(datasets)


# ---------------------------------------------------------------------------
# Step function
# ---------------------------------------------------------------------------


def _phase_to_softmax(phase_idx: Tensor, num_classes: int) -> Tensor:
    """One-hot phase ids → soft 'softmax' representation we feed into z_int.

    Stage A produces softmax; GT during training is an integer index.
    We one-hot it so the input shape is consistent with predicted-z_int
    inference. (Identity at training time, soft at inference time.)
    """
    onehot = F.one_hot(phase_idx.clamp_min(0).long(), num_classes=num_classes)
    return onehot.float()


def _support_to_softmax(support_idx: Tensor, num_classes: int) -> Tensor:
    return _phase_to_softmax(support_idx, num_classes)


def _per_clip_canon_transform(
    joints_world: Tensor,          # (B, T, 22, 3)
    motion_263: Tensor,            # (B, T, 263)
    seq_len: Tensor,               # (B,)
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute per-clip canonical→world (R_y, T_xz, T_y) on-device.

    Mirrors :func:`piano.utils.canonical_frame.get_canonicalize_transform_from_clip`
    but in torch and batched. Cheap: one frame-0 reduction per clip.
    """
    from piano.training.anchor_consistency_loss import lift_motion263_to_joints
    canon = lift_motion263_to_joints(motion_263)                   # (B, T, 22, 3)
    B = joints_world.shape[0]
    R_y_list, T_xz_list, T_y_list = [], [], []
    for b in range(B):
        T_b = max(int(seq_len[b].item()), 1)
        wt0 = joints_world[b, 0]                                   # (22, 3)
        ct0 = canon[b, 0]
        across_w = (wt0[17] - wt0[16]) + (wt0[2] - wt0[1])
        across_c = (ct0[17] - ct0[16]) + (ct0[2] - ct0[1])
        fw_w = torch.atan2(across_w[2], -across_w[0])
        fw_c = torch.atan2(across_c[2], -across_c[0])
        R_y = fw_w - fw_c
        # XZ translation: world_pelvis_t0 - R_y(canon_pelvis_t0) on XZ
        cos_, sin_ = R_y.cos(), R_y.sin()
        Rmat = torch.stack([
            torch.stack([cos_, torch.zeros_like(cos_), sin_]),
            torch.stack([torch.zeros_like(cos_), torch.ones_like(cos_), torch.zeros_like(cos_)]),
            torch.stack([-sin_, torch.zeros_like(cos_), cos_]),
        ])
        rot_pelv = Rmat @ ct0[0]
        T_xz = torch.stack([wt0[0, 0] - rot_pelv[0], wt0[0, 2] - rot_pelv[2]])
        T_y = wt0[0, 1] - ct0[0, 1]
        R_y_list.append(R_y)
        T_xz_list.append(T_xz)
        T_y_list.append(T_y)
    return (
        torch.stack(R_y_list),                                     # (B,)
        torch.stack(T_xz_list),                                    # (B, 2)
        torch.stack(T_y_list),                                     # (B,)
    )


def build_anchordiff_step_fn(
    model: MotionAnchorDiff,
    object_encoder: ObjectEncoder,
    clip_model: torch.nn.Module,
    anchor_cfg: AnchorConsistencyConfig,
    z_dims: ZIntDims,
    device: torch.device,
):
    def step_fn(_model, batch: dict, global_step: int = 0) -> dict[str, Tensor]:
        motion = batch["motion"].to(device)                       # (B, T, 263)
        joints = batch["joints"].to(device)                       # (B, T, 22, 3)
        object_pc = batch["object_pc"].to(device)
        contact_state = batch["contact_state"].to(device)         # (B, T, 5)
        contact_target_xyz = batch["contact_target_xyz"].to(device)  # (B, T, 5, 3)
        phase = batch["phase"].to(device)                         # (B, T)
        support = batch["support"].to(device)                     # (B, T)
        obj_com = batch["obj_com_canonical"].to(device)           # (B, T, 3)
        obj_rot6d = batch["obj_rot6d_canonical"].to(device)       # (B, T, 6)
        obj_pos_world = batch["object_positions"].to(device)      # (B, T, 3)
        obj_rot_world = batch["object_rotations"].to(device)      # (B, T, 3) axis-angle
        seq_len = batch["seq_len"].to(device)                     # (B,)

        B, T, _ = motion.shape
        seq_idx = torch.arange(T, device=device).unsqueeze(0)
        seq_mask = (seq_idx < seq_len.unsqueeze(1)).float()        # (B, T)

        # Per-clip canonical→world transform (cheap; one frame-0 op per clip).
        with torch.no_grad():
            R_y, T_xz_canon, T_y_canon = _per_clip_canon_transform(
                joints, motion, seq_len,
            )

        # --- Pack z_int (training: GT contact + GT target + GT phase/support) ---
        phase_soft = _phase_to_softmax(phase, z_dims.phase_classes)
        support_soft = _support_to_softmax(support, z_dims.support_classes)
        z_int = pack_z_int(
            contact_state, contact_target_xyz, phase_soft, support_soft, z_dims,
        )                                                          # (B, T, total)

        # --- Object world trajectory channel: use canonical object pose
        # (matches motion_263 frame), 3 (com) + 6 (rot6d) = 9 dims/frame.
        object_traj = torch.cat([obj_com, obj_rot6d], dim=-1)      # (B, T, 9)

        # --- Init pose: SMPL-22 frame 0 ---
        init_pose = joints[:, 0, :, :].reshape(B, -1)              # (B, 66)

        # --- Text features via CLIP per-token ---
        text_features, _text_mask = encode_text_per_token(
            clip_model, batch["text"], device,
        )                                                          # (B, L, text_dim)

        # --- Object tokens via PointNet++ encoder ---
        obj_tokens = object_encoder(object_pc)                     # (B, N, obj_dim)

        cond = {
            "z_int": z_int,
            "object_world_traj": object_traj,
            "init_pose": init_pose,
            "text": text_features.float(),
            "object_tokens": obj_tokens,
        }

        # --- Diffusion training step (x₀-prediction) ---
        out = model.training_step(motion, cond)
        x0_pred = out["x0_pred"]
        x0_target = out["x0_target"]

        # x₀ MSE — masked to valid frames (padded frames have undefined
        # motion, would bias the gradient).
        mse = (x0_pred - x0_target).pow(2).sum(-1)                  # (B, T)
        mse = (mse * seq_mask).sum() / seq_mask.sum().clamp_min(1.0)

        # Anchor consistency in WORLD frame (uniform-skel canonical lifted
        # to world via per-clip (R_y, T_xz, T_y); see
        # analyses/2026-05-08_anchordiff_frame_bug_fix.md).
        anchor = anchor_consistency_loss(
            x0_pred=x0_pred,
            contact_state_gt=contact_state,
            contact_target_xyz_local=contact_target_xyz,
            object_positions=obj_pos_world,
            object_rotations=obj_rot_world,
            R_y=R_y,
            T_xz=T_xz_canon,
            T_y=T_y_canon,
            cfg=anchor_cfg,
            seq_mask=seq_mask.bool(),
        )

        total = mse + anchor
        return {
            "loss": total,
            "mse_x0": mse.detach(),
            "anchor_l2": anchor.detach(),
        }

    return step_fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run a single batch + backward to verify wiring; do not save.")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.get("gradient_accumulation_steps", 1),
        mixed_precision=cfg.training.get("mixed_precision", "bf16"),
    )
    set_seed(int(cfg.training.get("seed", 42)))
    device = accelerator.device

    accelerator.print("===== PIANO-AnchorDiff training =====")
    accelerator.print(f"output_dir = {cfg.output_dir}")
    accelerator.print(f"smoke_test = {args.smoke_test}")

    # --- Build dataset ---
    train_dataset = _build_dataset(cfg, bucket="train", augment=True)
    accelerator.print(f"Train dataset: {len(train_dataset)} clips")

    val_dataset = None
    if int(cfg.training.get("val_every_epochs", 0)) > 0:
        val_dataset = _build_dataset(cfg, bucket="val", augment=False)
        accelerator.print(f"Val dataset:   {len(val_dataset)} clips")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=collate_hoi,
        num_workers=int(cfg.training.get("num_workers", 4)),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            collate_fn=collate_hoi,
            num_workers=int(cfg.training.get("num_workers", 4)),
            pin_memory=True,
            drop_last=False,
        )

    # --- Build model ---
    z_dims = ZIntDims(
        num_parts=int(cfg.model.z_int.num_parts),
        phase_classes=int(cfg.model.z_int.phase_classes),
        support_classes=int(cfg.model.z_int.support_classes),
    )
    denoiser_cfg = AnchorDenoiserConfig(
        motion_dim=int(cfg.model.denoiser.motion_dim),
        z_int=z_dims,
        object_traj_dim=int(cfg.model.denoiser.object_traj_dim),
        init_pose_dim=int(cfg.model.denoiser.init_pose_dim),
        text_dim=int(cfg.model.denoiser.text_dim),
        object_token_dim=int(cfg.model.denoiser.object_token_dim),
        object_num_tokens=int(cfg.model.denoiser.object_num_tokens),
        d_model=int(cfg.model.denoiser.d_model),
        n_layers=int(cfg.model.denoiser.n_layers),
        n_heads=int(cfg.model.denoiser.n_heads),
        ff_mult=int(cfg.model.denoiser.ff_mult),
        dropout=float(cfg.model.denoiser.dropout),
        max_seq_length=int(cfg.data.max_seq_length),
    )
    diff_cfg = DiffusionConfig(
        num_steps=int(cfg.model.diffusion.num_steps),
        schedule=str(cfg.model.diffusion.schedule),
    )
    model = MotionAnchorDiff(
        AnchorDiffConfig(
            diffusion=diff_cfg,
            denoiser=denoiser_cfg,
            cfg_drop_prob=float(cfg.model.cfg_drop_prob),
        )
    )

    object_encoder = ObjectEncoder(
        num_input_points=int(cfg.model.object_encoder.num_input_points),
        num_output_tokens=int(cfg.model.object_encoder.num_output_tokens),
        feature_dim=int(cfg.model.object_encoder.feature_dim),
    )

    clip_model = load_clip_text_encoder(
        device=device,
        model_name=str(cfg.model.text_encoder.clip_version),
        download_root=str(cfg.model.text_encoder.get("download_root", "cache/clip")),
    )

    anchor_cfg = AnchorConsistencyConfig(
        weight=float(cfg.loss.anchor_weight),
        contact_threshold=float(cfg.loss.contact_threshold),
    )

    # --- Optimizer + scheduler ---
    accum = int(cfg.training.get("gradient_accumulation_steps", 1))
    steps_per_epoch = max(1, len(train_loader) // accum)
    total_steps = steps_per_epoch * int(cfg.training.num_epochs)
    optimizer = build_optimizer_with_decay_groups(
        modules=[model, object_encoder],
        lr=float(cfg.training.optimizer.lr),
        weight_decay=float(cfg.training.optimizer.weight_decay),
        betas=tuple(cfg.training.optimizer.betas),
    )
    scheduler = build_scheduler(
        optimizer,
        int(cfg.training.scheduler.warmup_steps),
        total_steps,
    )

    model, object_encoder, optimizer, train_loader, scheduler = accelerator.prepare(
        model, object_encoder, optimizer, train_loader, scheduler,
    )
    if val_loader is not None:
        val_loader = accelerator.prepare(val_loader)

    step_fn = build_anchordiff_step_fn(
        model=model,
        object_encoder=object_encoder,
        clip_model=clip_model,
        anchor_cfg=anchor_cfg,
        z_dims=z_dims,
        device=device,
    )

    # --- Smoke test: one batch through forward + backward ---
    if args.smoke_test:
        accelerator.print("Running smoke test (1 batch)...")
        batch = next(iter(train_loader))
        out = step_fn(model, batch, global_step=0)
        accelerator.print(
            f"  loss={out['loss'].item():.4f}  "
            f"mse_x0={out['mse_x0'].item():.4f}  "
            f"anchor_l2={out['anchor_l2'].item():.4f}"
        )
        accelerator.backward(out["loss"])
        optimizer.step()
        accelerator.print("Smoke test PASSED.")
        return

    # --- Wandb (optional) ---
    wandb_run = None
    if accelerator.is_main_process:
        try:
            import wandb
            wandb_run = wandb.init(
                project=cfg.logging.project, name=cfg.logging.run_name,
            )
        except ImportError:
            pass

    run_training_loop(
        accelerator=accelerator,
        model=model,
        dataloader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        step_fn=step_fn,
        num_epochs=int(cfg.training.num_epochs),
        output_dir=cfg.output_dir,
        log_every=int(cfg.logging.log_every_n_steps),
        save_every_epochs=int(cfg.logging.save_every_n_epochs),
        max_grad_norm=float(cfg.training.max_grad_norm),
        wandb_run=wandb_run,
        extra_modules={"object_encoder": object_encoder},
        val_dataloader=val_loader,
        val_every_epochs=int(cfg.training.get("val_every_epochs", 0)),
    )


if __name__ == "__main__":
    main()
