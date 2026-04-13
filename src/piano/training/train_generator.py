"""Stage B: Finetune the Motion Generator with interaction conditioning.

Loads pretrained MoMask weights into InteractionMaskTransformer,
freezes the VQ-VAE, and trains the masked transformer (with new interaction
cross-attention layers) conditioned on GT pseudo-labels.

Usage:
    accelerate launch -m piano.training.train_generator --config configs/training/generator.yaml
"""
from __future__ import annotations

import argparse

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from piano.data.dataset import HOIDataset, collate_hoi
from piano.models.interaction_cross_attn import InteractionTokenizer
from piano.models.backbones.momask_adapter import load_momask_vqvae
from piano.models.motion_generator import InteractionMaskTransformer
from piano.training.losses import GeneratorLoss
from piano.training.trainer import (
    build_optimizer,
    build_scheduler,
    run_training_loop,
)


def build_generator_step_fn(
    transformer: InteractionMaskTransformer,
    vq_vae: RVQVAE,
    interaction_tokenizer: InteractionTokenizer,
    clip_model: torch.nn.Module,
    criterion: GeneratorLoss,
):
    """Build step function for generator training.

    Uses GT pseudo-labels as interaction condition (not predicted).
    """
    def step_fn(_model: torch.nn.Module, batch: dict) -> dict[str, torch.Tensor]:
        # Encode text
        with torch.no_grad():
            text_emb = clip_model.encode_text(batch["text"])  # (B, clip_dim)

        # Encode motion to VQ tokens (frozen VQ-VAE)
        with torch.no_grad():
            token_indices = vq_vae.encode(batch["motion"])  # (B, S, Q)
            base_tokens = token_indices[:, :, 0]  # (B, S) — base level only

        # Build interaction tokens from GT pseudo-labels
        interaction_tokens = interaction_tokenizer(
            contact_state=batch["contact_state"],
            contact_target=batch["contact_target"],
            phase=batch["phase"],
            support=batch["support"],
        )  # (B, S_int, d_model)

        # Token sequence lengths (after VQ downsampling)
        seq_len = batch["seq_len"]
        token_lens = (seq_len / 4).long().clamp(min=1)  # VQ temporal downsample factor

        # Forward: masked prediction with interaction conditioning
        mask_pred_loss, pred_ids, accuracy = transformer(
            ids=base_tokens,
            cond=text_emb,
            m_lens=token_lens,
            interaction_tokens=interaction_tokens,
        )

        # Compute generator loss
        loss_dict = criterion(mask_pred_loss)
        loss_dict["accuracy"] = accuracy

        return loss_dict

    return step_fn


def run(config_path: str) -> None:
    """Run Stage B training."""
    cfg = OmegaConf.load(config_path)
    set_seed(42)

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        mixed_precision="bf16",
    )

    # Load MoMask VQ-VAE (frozen, with pretrained weights)
    vq_vae = load_momask_vqvae(cfg.model.momask_checkpoint, device="cpu")

    # Load MoMask MaskTransformer + wrap with interaction cross-attention
    transformer = InteractionMaskTransformer.from_pretrained(
        cfg.model.momask_checkpoint,
        interaction_drop_prob=0.1,
        device="cpu",
    )
    # CLIP is loaded inside MoMask's MaskTransformer constructor (frozen)

    # Loss
    criterion = GeneratorLoss(
        velocity_smoothness_weight=cfg.loss.velocity_smoothness_weight,
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

    # Optimizer: different LR for backbone vs new interaction layers
    optimizer = build_optimizer(
        [
            {"params": transformer.backbone_parameters(), "lr": cfg.training.optimizer.lr},
            {"params": transformer.interaction_parameters(), "lr": cfg.training.optimizer.lr * 2},
        ],
        lr=cfg.training.optimizer.lr,
    )

    total_steps = len(dataloader) * cfg.training.num_epochs
    scheduler = build_scheduler(optimizer, cfg.training.scheduler.warmup_steps, total_steps)

    # Prepare
    transformer, optimizer, dataloader, scheduler = accelerator.prepare(
        transformer, optimizer, dataloader, scheduler,
    )
    vq_vae = vq_vae.to(accelerator.device)

    step_fn = build_generator_step_fn(
        transformer, vq_vae, interaction_tokenizer, clip_model, criterion,
    )

    # Wandb
    wandb_run = None
    if accelerator.is_main_process:
        try:
            import wandb
            wandb_run = wandb.init(project=cfg.logging.project, name=cfg.logging.run_name)
        except ImportError:
            pass

    run_training_loop(
        accelerator=accelerator,
        model=transformer,
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
    parser.add_argument("--config", type=str, default="configs/training/generator.yaml")
    args = parser.parse_args()
    run(args.config)
