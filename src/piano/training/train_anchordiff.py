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
import json
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
from piano.training.feature_groups import FEATURE_GROUPS
from piano.training.feature_weight_state import FeatureWeightState
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
    feature_weight_state: FeatureWeightState,
):
    """Build the AnchorDiff step_fn closure.

    The step_fn reads `feature_weight_state.current` every batch via
    `to_per_frame_tensor` (cheap; lazily refreshed when state has changed).
    Dynamic-update mode: an external epoch hook calls
    `feature_weight_state.update(...)` after every K epochs; the next
    batch's `cache.get()` automatically picks up the new weights.
    """

    class _WeightCache:
        def __init__(self) -> None:
            self._t = feature_weight_state.to_per_frame_tensor(device)
            self._ver = feature_weight_state.last_update_epoch

        def get(self) -> Tensor:
            if feature_weight_state.last_update_epoch != self._ver:
                self._t = feature_weight_state.to_per_frame_tensor(device)
                self._ver = feature_weight_state.last_update_epoch
            return self._t

    cache = _WeightCache()
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

        # x₀ MSE — masked to valid frames. FEATURE-WEIGHTED via the
        # FeatureWeightState (static or dynamic). See feature_groups.py
        # for the per-group layout and feature_weight_state.py for the
        # update logic.
        mse_per_dim = (x0_pred - x0_target).pow(2)                  # (B, T, 263)
        weighted = (mse_per_dim * cache.get()).sum(-1)              # (B, T)
        mse = (weighted * seq_mask).sum() / seq_mask.sum().clamp_min(1.0)

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

    # --- FeatureWeightState (static v2 / dynamic v2.1) ---
    static_w_cfg = (
        OmegaConf.to_container(cfg.loss.motion_feature_weights, resolve=True)
        if "motion_feature_weights" in cfg.loss else None
    )
    dyn_cfg = cfg.loss.get("dynamic_metric", None)
    if dyn_cfg is not None and dyn_cfg.get("enabled", False):
        feature_weight_state = FeatureWeightState.dynamic_from_config(
            static_weights_cfg=static_w_cfg,
            geometry_prior_path=str(dyn_cfg.geometry_prior_path),
            update_every_epochs=int(dyn_cfg.get("update_every_epochs", 5)),
            ema_beta=float(dyn_cfg.get("ema_beta", 0.2)),
            residual_alpha=float(dyn_cfg.get("residual_alpha", 0.5)),
            clamp_min=float(dyn_cfg.get("clamp_min", 0.25)),
            clamp_max=float(dyn_cfg.get("clamp_max", 150.0)),
        )
        accelerator.print(
            f"FeatureWeightState DYNAMIC mode: update every "
            f"{feature_weight_state.update_every_epochs} epochs "
            f"(β={feature_weight_state.ema_beta}, α={feature_weight_state.residual_alpha})"
        )
    else:
        feature_weight_state = FeatureWeightState.static_from_config(static_w_cfg)
        accelerator.print("FeatureWeightState STATIC mode")

    step_fn = build_anchordiff_step_fn(
        model=model,
        object_encoder=object_encoder,
        clip_model=clip_model,
        anchor_cfg=anchor_cfg,
        z_dims=z_dims,
        device=device,
        feature_weight_state=feature_weight_state,
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

    # --- Calibration data + dynamic-weight update hook (v2.1 only) ---
    epoch_end_hook = None
    if feature_weight_state.enable_dynamic:
        from piano.data.dataset import HOIDataset
        from piano.training.feature_groups import FEATURE_GROUPS as _FG
        from piano.utils.io_utils import load_json as _load_json

        cal_path = Path(dyn_cfg["calibration_clips"])
        cal_manifest = _load_json(cal_path)
        accelerator.print(
            f"Loaded calibration manifest: {cal_manifest['num_clips']} clips from {cal_path}"
        )

        # Pre-build calibration HOIDatasets per subset.
        cal_datasets: dict[str, HOIDataset] = {}
        pseudo_label_subdir_local = cfg.data.get("pseudo_label_subdir", None)
        for entry in cfg.data.datasets:
            sub_dir = (str(Path(entry.root) / pseudo_label_subdir_local)
                       if pseudo_label_subdir_local is not None else None)
            cal_datasets[entry.name] = HOIDataset(
                root=entry.root,
                pseudo_label_dir=sub_dir,
                max_seq_length=cfg.data.max_seq_length,
                augment=None,
                support_collapse_hand_support=True,
                surface_obj_pose=True,
            )

        # Use a fixed seed for the timestep / noise sampling at every
        # update so the residual is stable across epochs (only the
        # model state changes).
        cal_seed = int(dyn_cfg.get("calibration_seed", 12345))
        cal_t = int(dyn_cfg.get("fixed_timestep", 200))     # mid-noise level

        def _calibration_eval(
            unwrapped_model: torch.nn.Module,
            unwrapped_obj_enc: torch.nn.Module,
        ) -> tuple[dict[str, float], dict[str, float]]:
            """Single-step denoising on calibration clips.

            Returns (per_group_RMSE, per_group_GT_std).
            """
            unwrapped_model.eval()
            unwrapped_obj_enc.eval()
            torch.manual_seed(cal_seed)

            n_groups = len(_FG)
            sum_se = {g.name: 0.0 for g in _FG}
            sum_gt = {g.name: 0.0 for g in _FG}
            sum_gt_sq = {g.name: 0.0 for g in _FG}
            count_dim = {g.name: 0 for g in _FG}

            for clip_meta in cal_manifest["clips"]:
                ds = cal_datasets[clip_meta["subset"]]
                # Locate by seq_id (clip_idx may shift if dataset filter changed)
                target_id = clip_meta["seq_id"]
                idx = clip_meta.get("clip_idx_in_filtered_dataset", -1)
                sample = None
                if 0 <= idx < len(ds):
                    cand = ds[idx]
                    if str(cand["seq_id"]) == target_id:
                        sample = cand
                if sample is None:
                    for cand_idx in range(len(ds)):
                        cand = ds[cand_idx]
                        if str(cand["seq_id"]) == target_id:
                            sample = cand
                            break
                if sample is None:
                    continue
                seq_len_s = int(sample["seq_len"].item())
                if seq_len_s < 5:
                    continue

                batch = collate_hoi([sample])
                motion_b = batch["motion"].to(device)
                joints_b = batch["joints"].to(device)
                cs_b = batch["contact_state"].to(device)
                ctx_b = batch["contact_target_xyz"].to(device)
                ph_b = batch["phase"].to(device)
                sp_b = batch["support"].to(device)
                obj_com_b = batch["obj_com_canonical"].to(device)
                obj_rot6d_b = batch["obj_rot6d_canonical"].to(device)
                object_pc_b = batch["object_pc"].to(device)

                Tlen = motion_b.shape[1]
                seq_idx_b = torch.arange(Tlen, device=device).unsqueeze(0)
                seq_mask_b = (seq_idx_b < torch.tensor([seq_len_s], device=device).unsqueeze(1)).float()

                phase_soft_b = F.one_hot(ph_b.clamp_min(0).long(), z_dims.phase_classes).float()
                support_soft_b = F.one_hot(sp_b.clamp_min(0).long(), z_dims.support_classes).float()
                z_int_b = pack_z_int(cs_b, ctx_b, phase_soft_b, support_soft_b, z_dims)
                obj_traj_b = torch.cat([obj_com_b, obj_rot6d_b], dim=-1)
                init_pose_b = joints_b[:, 0, :, :].reshape(1, -1)
                text_features_b, _ = encode_text_per_token(clip_model, batch["text"], device)
                obj_tokens_b = unwrapped_obj_enc(object_pc_b)

                cond_b = {
                    "z_int": z_int_b,
                    "object_world_traj": obj_traj_b,
                    "init_pose": init_pose_b,
                    "text": text_features_b.float(),
                    "object_tokens": obj_tokens_b,
                }
                # Single-step denoising at fixed t (cheaper than full DDPM)
                t_tensor = torch.full((1,), cal_t, device=device, dtype=torch.long)
                noise = torch.randn_like(motion_b)
                with torch.no_grad():
                    x_t = unwrapped_model.diffusion.q_sample(motion_b, t_tensor, noise)
                    x0_pred = unwrapped_model.denoiser(x_t, t_tensor, cond_b, cond_drop_mask=None)

                # Per-group RMSE numerator + denominator
                err_sq = (x0_pred - motion_b).pow(2)            # (1, T, 263)
                gt_sq = motion_b.pow(2)
                gt_v = motion_b
                mask3 = seq_mask_b.unsqueeze(-1)                # (1, T, 1)
                for g in _FG:
                    e_grp = err_sq[..., g.lo:g.hi]
                    gt_grp = gt_v[..., g.lo:g.hi]
                    gt_sq_grp = gt_sq[..., g.lo:g.hi]
                    se = (e_grp * mask3).sum().item()
                    s = (gt_grp * mask3).sum().item()
                    s_sq = (gt_sq_grp * mask3).sum().item()
                    sum_se[g.name] += se
                    sum_gt[g.name] += s
                    sum_gt_sq[g.name] += s_sq
                    count_dim[g.name] += int(mask3.sum().item() * (g.hi - g.lo))

            unwrapped_model.train()
            unwrapped_obj_enc.train()

            rmse = {n: (sum_se[n] / max(count_dim[n], 1)) ** 0.5 for n in sum_se}
            mean_gt = {n: sum_gt[n] / max(count_dim[n], 1) for n in sum_gt}
            var_gt = {n: max(sum_gt_sq[n] / max(count_dim[n], 1) - mean_gt[n] ** 2, 0.0)
                      for n in sum_gt}
            std_gt = {n: var_gt[n] ** 0.5 for n in var_gt}
            return rmse, std_gt

        def epoch_end_hook(epoch, accelerator, model, global_step, output_dir,
                           metrics_appender, wandb_run):
            if not feature_weight_state.should_update(epoch):
                return
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_obj_enc = accelerator.unwrap_model(object_encoder)
            rmse, gt_std = _calibration_eval(unwrapped_model, unwrapped_obj_enc)
            log = feature_weight_state.update(
                epoch=epoch, group_rmse=rmse, group_gt_std=gt_std,
            )
            feature_weight_state.broadcast(accelerator)
            # Log to metrics + wandb + JSON snapshot
            if accelerator.is_main_process:
                metrics_appender("dynamic_weight_update", {
                    "epoch": epoch,
                    "global_step": global_step,
                    **{f"weight_{k}": v for k, v in log["new_w_g"].items()},
                    **{f"rmse_{k}": rmse[k] for k in rmse},
                    **{f"norm_err_{k}": log["norm_err_g"][k] for k in log["norm_err_g"]},
                })
                snap_path = Path(output_dir) / f"feature_weights_epoch_{epoch:04d}.json"
                snap_path.write_text(json.dumps(log, indent=2))
                if wandb_run is not None:
                    flat = {}
                    for k, v in log["new_w_g"].items():
                        flat[f"weight/{k}"] = v
                    for k, v in rmse.items():
                        flat[f"calib_rmse/{k}"] = v
                    wandb_run.log(flat, step=epoch)

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
        epoch_end_hook=epoch_end_hook,
        val_dataloader=val_loader,
        val_every_epochs=int(cfg.training.get("val_every_epochs", 0)),
    )


if __name__ == "__main__":
    main()
