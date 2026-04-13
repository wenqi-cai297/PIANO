"""Stage A: Train the Interaction Predictor.

Trains the predictor to map (text, object, init_pose) → interaction latent,
supervised by pseudo-labels extracted from HOI data.

Usage:
    accelerate launch -m piano.training.train_predictor --config configs/training/predictor.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf
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


def build_predictor_step_fn(
    predictor: InteractionPredictor,
    object_encoder: ObjectEncoder,
    clip_model: torch.nn.Module,
    criterion: PredictorLoss,
    priors: PhysicalPriors,
):
    """Build the step function for predictor training.

    Returns a callable(model_unused, batch) -> loss_dict.
    The actual models are captured in the closure.
    """
    def step_fn(_model: torch.nn.Module, batch: dict) -> dict[str, torch.Tensor]:
        # Encode text
        with torch.no_grad():
            text_emb = clip_model.encode_text(batch["text"])  # (B, clip_dim)

        # Encode object
        obj_tokens = object_encoder(batch["object_pc"])  # (B, M, d)

        # Initial pose (first frame of motion)
        init_pose = batch["motion"][:, 0, :]  # (B, 263)
        seq_len = batch["seq_len"]

        # Predict interaction latent
        pred = predictor(text_emb, obj_tokens, init_pose, seq_length=batch["motion"].shape[1])

        # Frame mask (True for valid, non-padded frames)
        max_T = batch["motion"].shape[1]
        frame_mask = torch.arange(max_T, device=seq_len.device).unsqueeze(0) < seq_len.unsqueeze(1)

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


def run(config_path: str) -> None:
    """Run Stage A training."""
    cfg = OmegaConf.load(config_path)
    set_seed(42)

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        mixed_precision="bf16",
    )

    # Models
    predictor = InteractionPredictor(
        d_model=384, num_layers=10, num_heads=6,
        dim_feedforward=1024, text_dim=512, pose_dim=263,
        block_size=2,
    )
    object_encoder = ObjectEncoder(num_output_tokens=16, feature_dim=384)

    # CLIP text encoder (frozen) — loaded via OpenAI CLIP
    # In practice: clip.load("ViT-B/32") → clip_model
    # Placeholder: will be initialized properly when CLIP is available
    clip_model = None  # TODO: load CLIP model on server

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

    # Optimizer & scheduler
    params = list(predictor.parameters()) + list(object_encoder.parameters())
    optimizer = build_optimizer(params, lr=cfg.training.optimizer.lr)
    total_steps = len(dataloader) * cfg.training.num_epochs
    scheduler = build_scheduler(optimizer, cfg.training.scheduler.warmup_steps, total_steps)

    # Prepare with accelerator
    predictor, object_encoder, optimizer, dataloader, scheduler = accelerator.prepare(
        predictor, object_encoder, optimizer, dataloader, scheduler,
    )

    # Wrap models into a single module for the training loop
    # (the step_fn closure captures the individual models)
    step_fn = build_predictor_step_fn(predictor, object_encoder, clip_model, criterion, priors)

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
