"""Stage A: Train the Interaction Predictor.

Trains the predictor to map (text, object, init_pose) → interaction latent,
supervised by pseudo-labels extracted from HOI data.

Usage:
    accelerate launch -m piano.training.train_predictor --config configs/training/predictor.yaml
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader

from piano.data.dataset import HOIDataset, collate_hoi
from piano.models.interaction_predictor import InteractionPredictor
from piano.models.object_encoder import ObjectEncoder
from piano.training.losses import PredictorLoss
from piano.training.priors import PhysicalPriors
from piano.training.trainer import (
    build_optimizer,
    build_scheduler,
    run_training_loop,
)
from piano.utils.clip_utils import encode_text_per_token, load_clip_text_encoder


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
):
    """Build the step function for predictor training.

    Returns a callable(model_unused, batch) -> loss_dict. The actual
    models are captured in the closure.
    """
    def step_fn(_model: nn.Module, batch: dict) -> dict[str, Tensor]:
        # Text: CLIP per-token features + padding mask
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

        # Supervision loss
        loss_dict = criterion(
            pred,
            gt_contact=batch["contact_state"],
            gt_target=batch["contact_target"],
            gt_phase=batch["phase"].long(),
            gt_support=batch["support"].long(),
            mask=frame_mask,
        )

        # Physical prior regularization
        joints = batch.get("joints")
        prior_dict = priors(pred, joints=joints, mask=frame_mask)
        loss_dict["loss_priors"] = prior_dict["loss"]
        loss_dict["loss"] = loss_dict["loss"] + prior_dict["loss"]

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
        num_object_patches=model_cfg.output.num_object_patches,
        num_phases=model_cfg.output.num_phases,
        num_support_states=model_cfg.output.num_support_states,
    )
    object_encoder = ObjectEncoder(
        num_input_points=obj_cfg.pointnet.num_input_points,
        num_output_tokens=obj_cfg.pointnet.num_output_tokens,
        feature_dim=obj_cfg.pointnet.feature_dim,
    )

    # CLIP text encoder (frozen)
    clip_model = load_clip_text_encoder(
        device=device,
        model_name=cfg.model.get("text_encoder", "ViT-B/32"),
    )

    # Loss and priors
    criterion = PredictorLoss(
        contact_weight=cfg.loss.contact_weight,
        target_weight=cfg.loss.target_weight,
        phase_weight=cfg.loss.phase_weight,
        support_weight=cfg.loss.support_weight,
    )
    priors = PhysicalPriors(
        reachability_weight=cfg.priors.reachability_weight,
        contact_persistence_weight=cfg.priors.contact_persistence_weight,
        support_smoothness_weight=cfg.priors.support_smoothness_weight,
        phase_monotonicity_weight=cfg.priors.phase_monotonicity_weight,
    )

    # Data
    dataset = HOIDataset(
        root=cfg.data.datasets[0].root,
        pseudo_label_dir=cfg.data.pseudo_label_dir,
        max_seq_length=cfg.data.max_seq_length,
    )
    dataloader = DataLoader(
        dataset, batch_size=cfg.training.batch_size,
        shuffle=True, collate_fn=collate_hoi, num_workers=4,
    )

    # Optimizer & scheduler (over trainable modules only — CLIP is frozen)
    params = list(predictor.parameters()) + list(object_encoder.parameters())
    optimizer = build_optimizer(params, lr=cfg.training.optimizer.lr)
    total_steps = len(dataloader) * cfg.training.num_epochs
    scheduler = build_scheduler(optimizer, cfg.training.scheduler.warmup_steps, total_steps)

    # Prepare trainables with accelerator
    predictor, object_encoder, optimizer, dataloader, scheduler = accelerator.prepare(
        predictor, object_encoder, optimizer, dataloader, scheduler,
    )

    step_fn = build_predictor_step_fn(
        predictor, object_encoder, clip_model, criterion, priors, device,
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
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/training/predictor.yaml")
    args = parser.parse_args()
    run(args.config)
