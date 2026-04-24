"""Stage A: Train the Interaction Predictor.

Trains the predictor to map (text, object, init_pose) → interaction latent,
supervised by pseudo-labels extracted from HOI data.

Usage:
    accelerate launch --config_file configs/accelerate_config.yaml \\
        -m piano.training.train_predictor \\
        --config configs/training/predictor.yaml
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader

from piano.data.dataset import (
    AugmentConfig,
    HOIDataset,
    build_object_split,
    collate_hoi,
)
from piano.models.interaction_predictor import InteractionPredictor
from piano.models.object_encoder import ObjectEncoder
from piano.training.losses import PredictorLoss
from piano.training.priors import PhysicalPriors
from piano.training.trainer import (
    build_optimizer_with_decay_groups,
    build_scheduler,
    run_training_loop,
)
from piano.utils.clip_utils import encode_text_per_token, load_clip_text_encoder
from piano.utils.io_utils import load_json


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------

def _collect_object_ids(roots: list) -> list[str]:
    """Collect the union of object_ids across all dataset roots.

    Reads each root's ``metadata_clean.json`` (or ``metadata.json``
    fallback — the clean variant is what HOIDataset prefers at train
    time, so the split should be computed on the same universe). Returns
    object_ids sorted for deterministic iteration across processes.
    """
    from pathlib import Path

    seen: set[str] = set()
    for entry in roots:
        root = Path(entry.root)
        meta_path = root / "metadata_clean.json"
        if not meta_path.exists():
            meta_path = root / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata not found in {root}")
        for m in load_json(meta_path):
            obj_id = m.get("object_id")
            if obj_id is not None:
                seen.add(obj_id)
    return sorted(seen)


def _build_dataset(
    cfg,
    split_override: str | None = None,
    enable_augment: bool = True,
) -> ConcatDataset:
    """Build a Stage A dataset from the 4 InterAct subset roots.

    Applies the object-id split (H5). Caller decides which bucket via
    ``split_override`` (``train`` / ``val`` / ``test`` / ``val+test``
    / ``all``) or lets the config default take effect. Augmentation is
    toggleable independent of the split so the val loader can share
    this builder with augmentation disabled.
    """
    # Object-id split — deterministic hash, identical across ranks.
    split_cfg = cfg.data.object_split
    allowed_object_ids: set[str] | None = None
    if split_cfg.enabled:
        object_ids = _collect_object_ids(cfg.data.datasets)
        splits = build_object_split(
            object_ids,
            train_pct=split_cfg.train_pct,
            val_pct=split_cfg.val_pct,
            test_pct=split_cfg.test_pct,
            seed=split_cfg.seed,
        )
        bucket = split_override or split_cfg.get("split", "train")
        if bucket == "val+test":
            allowed_object_ids = splits["val"] | splits["test"]
        elif bucket == "all":
            allowed_object_ids = None   # no filter
        elif bucket in splits:
            allowed_object_ids = splits[bucket]
        else:
            raise ValueError(f"unknown object_split bucket: {bucket!r}")

    # Augmentation — mirror + Y-rotation + pc jitter (train only by default).
    aug_cfg = cfg.data.get("augmentation", None)
    augment = None
    if enable_augment and aug_cfg is not None and aug_cfg.get("enabled", False):
        augment = AugmentConfig(
            enabled=True,
            mirror_prob=float(aug_cfg.get("mirror_prob", 0.0)),
            rotate_around_y_prob=float(aug_cfg.get("rotate_around_y_prob", 0.0)),
            pc_jitter_std=float(aug_cfg.get("pc_jitter_std", 0.0)),
        )

    # Per-subset HOIDataset instances, concatenated. pseudo_label_dir
    # defaults to <root>/pseudo_labels inside HOIDataset when the top-
    # level config leaves it null — matching the v9 per-subset layout.
    pseudo_label_dir = cfg.data.get("pseudo_label_dir", None)
    datasets = []
    for entry in cfg.data.datasets:
        ds = HOIDataset(
            root=entry.root,
            pseudo_label_dir=pseudo_label_dir,
            max_seq_length=cfg.data.max_seq_length,
            object_id_filter=allowed_object_ids,
            augment=augment,
        )
        datasets.append(ds)

    return ConcatDataset(datasets)


# ---------------------------------------------------------------------------
# Step function
# ---------------------------------------------------------------------------

def build_predictor_step_fn(
    predictor: InteractionPredictor,
    object_encoder: ObjectEncoder,
    clip_model: nn.Module,
    criterion: PredictorLoss,
    priors: PhysicalPriors,
    device: torch.device,
    prior_warmup_steps: int = 0,
):
    """Build the step function for predictor training.

    The returned callable takes (model, batch, global_step=...) and
    returns a loss dict. ``global_step`` is the optimizer-step counter
    fed in by ``run_training_loop`` — we use it to linearly ramp the
    physical prior contribution from 0 to full weight over the first
    ``prior_warmup_steps`` calls (PhysDiff / CG-HOI convention).
    """
    def step_fn(
        _model: nn.Module,
        batch: dict,
        global_step: int = 0,
    ) -> dict[str, Tensor]:
        # Text: CLIP per-token features + padding mask. Returned in
        # CLIP's native dtype (typically fp16 on GPU); the bf16 autocast
        # context handles casting through the predictor's Linear layers.
        text_features, text_mask = encode_text_per_token(
            clip_model, batch["text"], device,
        )

        # Object tokens
        obj_tokens = object_encoder(batch["object_pc"])         # (B, M, d)

        # Initial pose: first-frame SMPL-22 joint positions (66-d).
        # HumanML3D 263-d frame 0 has undefined velocities (process_file
        # drops the first frame for velocity computation), so we use
        # the raw joint positions instead.
        B = batch["joints"].shape[0]
        init_pose = batch["joints"][:, 0, :, :].reshape(B, -1)  # (B, 66)

        seq_len = batch["seq_len"]
        max_T = batch["motion"].shape[1]

        # Predict interaction latent
        pred = predictor(
            text_features, obj_tokens, init_pose,
            seq_length=max_T,
            text_key_padding_mask=text_mask,
        )

        # Frame mask (True for valid, non-padded frames)
        frame_mask = (
            torch.arange(max_T, device=seq_len.device).unsqueeze(0)
            < seq_len.unsqueeze(1)
        )

        # Supervision loss. Target is now xyz regression in object-
        # local frame (not patch-softmax), fed from HOIDataset's
        # contact_target_xyz = softmax-weighted patch-centroid.
        loss_dict = criterion(
            pred,
            gt_contact=batch["contact_state"],
            gt_target=batch["contact_target_xyz"],
            gt_phase=batch["phase"].long(),
            gt_support=batch["support"].long(),
            mask=frame_mask,
        )

        # Physical prior regularization, linearly warmed up from 0. On
        # a random-init predictor, prior gradients would otherwise
        # dominate the first few hundred steps and pull the model away
        # from fitting the pseudo-labels. Ramping lets the data fit lead.
        joints = batch.get("joints")
        prior_dict = priors(pred, joints=joints, mask=frame_mask)
        if prior_warmup_steps > 0:
            prior_scale = min(1.0, float(global_step) / float(prior_warmup_steps))
        else:
            prior_scale = 1.0
        loss_dict["loss_priors"] = prior_dict["loss"]
        loss_dict["prior_scale"] = torch.tensor(
            prior_scale, device=prior_dict["loss"].device,
        )
        loss_dict["loss"] = loss_dict["loss"] + prior_scale * prior_dict["loss"]

        return loss_dict

    return step_fn


# ---------------------------------------------------------------------------
# Training entrypoint
# ---------------------------------------------------------------------------

def run(config_path: str) -> None:
    """Run Stage A training."""
    cfg = OmegaConf.load(config_path)
    set_seed(42)

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        mixed_precision="bf16",
    )
    device = accelerator.device

    # Sub-configs. These are the *model* configs referenced by the
    # training yaml; we read them explicitly so all hyperparameters are
    # auditable from the top-level config tree.
    model_cfg = OmegaConf.load(cfg.model.config)
    obj_cfg = OmegaConf.load(cfg.model.object_encoder_config)

    # Models
    predictor = InteractionPredictor(
        d_model=model_cfg.encoder.d_model,
        num_layers=model_cfg.encoder.num_layers,
        num_heads=model_cfg.encoder.num_heads,
        dim_feedforward=model_cfg.encoder.dim_feedforward,
        dropout=model_cfg.encoder.dropout,
        text_dim=model_cfg.input.text_dim,
        pose_dim=model_cfg.input.pose_dim,
        max_seq_length=model_cfg.sequence.max_length,
        num_body_parts=model_cfg.output.num_body_parts,
        target_coord_dim=model_cfg.output.get("target_coord_dim", 3),
        num_phases=model_cfg.output.num_phases,
        num_support_states=model_cfg.output.num_support_states,
    )
    object_encoder = ObjectEncoder(
        num_input_points=obj_cfg.pointnet.num_input_points,
        num_output_tokens=obj_cfg.pointnet.num_output_tokens,
        feature_dim=obj_cfg.pointnet.feature_dim,
    )

    # SyncBatchNorm on the object encoder under multi-GPU DDP. The
    # PointNet++ SA layers use BatchNorm1d; per-rank running stats
    # would otherwise diverge across A6000 cards. Only valid when
    # there is actually more than one process — the in-place conversion
    # inserts collective comms that fail on single-GPU runs.
    if accelerator.num_processes > 1:
        object_encoder = nn.SyncBatchNorm.convert_sync_batchnorm(object_encoder)

    # CLIP text encoder (frozen). Kept OUT of accelerator.prepare() so
    # it doesn't get wrapped by DDP — it has no trainable parameters.
    # HF Accelerate's recommended pattern for frozen sub-modules.
    clip_model = load_clip_text_encoder(
        device=device,
        model_name=cfg.model.get("text_encoder", "ViT-B/32"),
    )

    # Loss and priors. With Kendall multi-task weights enabled, the
    # static contact/target/phase/support weights are ignored and the
    # optimiser learns per-task log-variances (one extra scalar param
    # per task) that auto-balance loss-scale differences.
    criterion = PredictorLoss(
        contact_weight=cfg.loss.contact_weight,
        target_weight=cfg.loss.target_weight,
        phase_weight=cfg.loss.phase_weight,
        support_weight=cfg.loss.support_weight,
        label_smoothing=cfg.loss.get("label_smoothing", 0.0),
        focal_gamma=cfg.loss.get("focal_gamma", 0.0),
        use_kendall_weights=cfg.loss.get("use_kendall_weights", False),
    )
    criterion = criterion.to(device)
    priors = PhysicalPriors(
        reachability_weight=cfg.priors.reachability_weight,
        contact_persistence_weight=cfg.priors.contact_persistence_weight,
        support_smoothness_weight=cfg.priors.support_smoothness_weight,
        phase_monotonicity_weight=cfg.priors.phase_monotonicity_weight,
    )

    # Data — multi-root ConcatDataset + object-id split + augmentation.
    dataset = _build_dataset(cfg)
    accelerator.print(
        f"Train dataset: {len(dataset)} clips across {len(cfg.data.datasets)} roots "
        f"(split={cfg.data.object_split.get('split', 'train')})",
    )
    dataloader = DataLoader(
        dataset, batch_size=cfg.training.batch_size,
        shuffle=True, collate_fn=collate_hoi, num_workers=4,
        pin_memory=True, drop_last=True,
    )

    # Val dataloader — same builder with split=val+test, augmentation
    # disabled. Used by the in-training keep-best-val loop in
    # run_training_loop. Skipped when val_every_epochs <= 0.
    val_dataloader = None
    val_every_epochs = int(cfg.training.get("val_every_epochs", 0))
    if val_every_epochs > 0:
        val_dataset = _build_dataset(
            cfg, split_override="val+test", enable_augment=False,
        )
        accelerator.print(
            f"Val dataset:   {len(val_dataset)} clips "
            f"(split=val+test, augmentation disabled)",
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=cfg.training.batch_size,
            shuffle=False, collate_fn=collate_hoi, num_workers=4,
            pin_memory=True, drop_last=False,
        )

    # Optimizer: AdamW with ViT/T5-style weight-decay groups (no decay
    # on biases, LayerNorm / BatchNorm weights, positional embeddings).
    # When Kendall multi-task weights are active, criterion holds 4
    # learnable scalars — include it in the optimised modules so they
    # actually update (otherwise they sit at 0 forever, equivalent to
    # all-ones static weights).
    optim_modules = [predictor, object_encoder]
    if criterion.use_kendall_weights:
        optim_modules.append(criterion)
    optimizer = build_optimizer_with_decay_groups(
        modules=optim_modules,
        lr=cfg.training.optimizer.lr,
        weight_decay=cfg.training.optimizer.weight_decay,
        betas=tuple(cfg.training.optimizer.betas),
    )
    # Scheduler total_steps measured in optimizer steps — Accelerate's
    # prepared scheduler handles the accumulation skip, so we feed it
    # (len(dataloader) / accum) × epochs.
    accum = cfg.training.gradient_accumulation_steps
    steps_per_epoch = max(1, len(dataloader) // accum)
    total_steps = steps_per_epoch * cfg.training.num_epochs
    scheduler = build_scheduler(
        optimizer, cfg.training.scheduler.warmup_steps, total_steps,
    )

    # Prepare trainables with accelerator. When Kendall is on, the
    # criterion holds 4 learnable scalars; under DDP these need grad
    # sync across ranks (otherwise each rank drifts to a different
    # value and the loss formula on rank 0 ≠ rank 1, breaking gradient
    # consistency on the predictor itself). prepare() wraps the
    # criterion in DDP so its parameter grads are all-reduced.
    if criterion.use_kendall_weights:
        predictor, object_encoder, criterion, optimizer, dataloader, scheduler = accelerator.prepare(
            predictor, object_encoder, criterion, optimizer, dataloader, scheduler,
        )
    else:
        predictor, object_encoder, optimizer, dataloader, scheduler = accelerator.prepare(
            predictor, object_encoder, optimizer, dataloader, scheduler,
        )
    if val_dataloader is not None:
        val_dataloader = accelerator.prepare(val_dataloader)

    step_fn = build_predictor_step_fn(
        predictor, object_encoder, clip_model, criterion, priors, device,
        prior_warmup_steps=cfg.priors.get("prior_warmup_steps", 0),
    )

    # Wandb (optional)
    wandb_run = None
    if accelerator.is_main_process:
        try:
            import wandb
            wandb_run = wandb.init(project=cfg.logging.project, name=cfg.logging.run_name)
        except ImportError:
            pass

    run_training_loop(
        accelerator=accelerator,
        model=predictor,
        dataloader=dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        step_fn=step_fn,
        num_epochs=cfg.training.num_epochs,
        output_dir=cfg.output_dir,
        log_every=cfg.logging.log_every_n_steps,
        save_every_epochs=cfg.logging.save_every_n_epochs,
        max_grad_norm=cfg.training.max_grad_norm,
        wandb_run=wandb_run,
        # Persist the object encoder's weights into every checkpoint
        # — predictor alone can't run inference; the object cross-
        # attention KV tokens come from this encoder and its weights
        # are trained from scratch (no pretrained fallback exists).
        extra_modules={"object_encoder": object_encoder},
        # Keep-best-val: re-run the step_fn on val_dataloader every N
        # epochs, save a best_val.pt when total val loss improves.
        # Does not interrupt training — best checkpoint is kept in
        # parallel to the final one.
        val_dataloader=val_dataloader,
        val_every_epochs=val_every_epochs,
    )


def main() -> None:
    """CLI entry point for ``piano-train-predictor`` (Stage A)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/training/predictor.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
