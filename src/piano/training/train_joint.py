"""Stage C: Joint finetune predictor + generator with consistency loss.

Loads trained predictor and generator checkpoints, unfreezes all trainable
parameters, and jointly optimizes with:
    - Predictor supervision loss (pseudo-labels)
    - Generator masked prediction loss
    - Consistency loss (extractor verifies generator uses z_int)

Usage:
    accelerate launch -m piano.training.train_joint --config configs/training/joint_finetune.yaml
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
from piano.models.interaction_extractor import InteractionExtractor
from piano.models.interaction_predictor import InteractionPredictor
from piano.models.backbones.momask_adapter import load_momask_vqvae
from piano.models.motion_generator import InteractionMaskTransformer
from piano.models.object_encoder import ObjectEncoder
from piano.training.losses import ConsistencyLoss, GeneratorLoss, PredictorLoss
from piano.training.priors import PhysicalPriors
from piano.training.trainer import (
    build_optimizer,
    build_scheduler,
    run_training_loop,
)


def build_joint_step_fn(
    predictor: InteractionPredictor,
    object_encoder: ObjectEncoder,
    transformer: InteractionMaskTransformer,
    vq_vae: RVQVAE,
    interaction_tokenizer: InteractionTokenizer,
    extractor: InteractionExtractor,
    clip_model: torch.nn.Module,
    pred_criterion: PredictorLoss,
    gen_criterion: GeneratorLoss,
    cons_criterion: ConsistencyLoss,
    priors: PhysicalPriors,
    consistency_weight: float = 0.5,
):
    """Build step function for joint finetuning."""

    def step_fn(_model: torch.nn.Module, batch: dict) -> dict[str, torch.Tensor]:
        # Encode text
        with torch.no_grad():
            text_emb = clip_model.encode_text(batch["text"])

        # Encode object
        obj_tokens = object_encoder(batch["object_pc"])
        init_pose = batch["motion"][:, 0, :]
        max_T = batch["motion"].shape[1]
        seq_len = batch["seq_len"]
        frame_mask = torch.arange(max_T, device=seq_len.device).unsqueeze(0) < seq_len.unsqueeze(1)

        # --- Predictor forward ---
        pred = predictor(text_emb, obj_tokens, init_pose, seq_length=max_T)

        pred_loss = pred_criterion(
            pred,
            gt_contact=batch["contact_state"],
            gt_target=batch["contact_target"],
            gt_phase=batch["phase"].long(),
            gt_support=batch["support"].long(),
            mask=frame_mask,
        )
        prior_loss = priors(pred, joints=batch.get("joints"), mask=frame_mask)

        # --- Generator forward ---
        # Use PREDICTED interaction latent (not GT) for joint training
        interaction_tokens = interaction_tokenizer(
            pred["contact_state"],
            pred["contact_target"],
            pred["phase"],
            pred["support"],
        )

        with torch.no_grad():
            token_indices = vq_vae.encode(batch["motion"])
            base_tokens = token_indices[:, :, 0]
        token_lens = (seq_len / 4).long().clamp(min=1)

        mask_pred_loss, pred_ids, accuracy = transformer(
            ids=base_tokens, cond=text_emb, m_lens=token_lens,
            interaction_tokens=interaction_tokens,
        )
        gen_loss = gen_criterion(mask_pred_loss)

        # --- Consistency loss ---
        # Decode generated tokens to motion, then extract interaction labels
        with torch.no_grad():
            # Create full token indices for decoding (base only, pad residual with 0)
            full_indices = torch.zeros_like(token_indices)
            full_indices[:, :, 0] = pred_ids
            generated_motion = vq_vae.decode(full_indices)

        extracted = extractor(generated_motion)
        cons_loss = cons_criterion(extracted, pred, mask=frame_mask)

        # --- Total ---
        total_loss = (
            pred_loss["loss"]
            + prior_loss["loss"]
            + gen_loss["loss"]
            + consistency_weight * cons_loss["loss"]
        )

        return {
            "loss": total_loss,
            "loss_pred": pred_loss["loss"],
            "loss_priors": prior_loss["loss"],
            "loss_gen": gen_loss["loss"],
            "loss_consistency": cons_loss["loss"],
            "accuracy": accuracy,
        }

    return step_fn


def run(config_path: str) -> None:
    """Run Stage C training."""
    cfg = OmegaConf.load(config_path)
    set_seed(42)

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        mixed_precision="bf16",
    )

    # --- Build models ---
    predictor = InteractionPredictor(d_model=384, num_layers=10, num_heads=6, dim_feedforward=1024, text_dim=512, pose_dim=263, block_size=2)
    object_encoder = ObjectEncoder(num_output_tokens=16, feature_dim=384)

    # Load MoMask VQ-VAE (frozen) and MaskTransformer (with interaction layers)
    vq_vae = load_momask_vqvae(cfg.model.vq_vae_checkpoint, device="cpu")
    transformer = InteractionMaskTransformer.from_pretrained(
        cfg.model.generator_checkpoint, device="cpu",
    )
    extractor = InteractionExtractor(motion_dim=263, d_model=256, num_layers=3)

    # Load Stage A/B checkpoints for predictor and object_encoder
    # TODO: load from cfg.model.predictor_checkpoint / cfg.model.object_encoder_checkpoint

    # CLIP is already loaded inside transformer.mask_transformer

    # --- Losses ---
    pred_criterion = PredictorLoss(
        contact_weight=cfg.loss.contact_weight, target_weight=cfg.loss.target_weight,
        phase_weight=cfg.loss.phase_weight, support_weight=cfg.loss.support_weight,
    )
    gen_criterion = GeneratorLoss(velocity_smoothness_weight=cfg.loss.velocity_smoothness_weight)
    cons_criterion = ConsistencyLoss()
    priors = PhysicalPriors(
        reachability_weight=cfg.priors.reachability_weight,
        contact_persistence_weight=cfg.priors.contact_persistence_weight,
        support_smoothness_weight=cfg.priors.support_smoothness_weight,
        phase_monotonicity_weight=cfg.priors.phase_monotonicity_weight,
    )

    # --- Data ---
    dataset = HOIDataset(
        root=cfg.data.datasets[0].root,
        pseudo_label_dir=cfg.data.pseudo_label_dir,
        max_seq_length=cfg.data.max_seq_length,
    )
    dataloader = DataLoader(
        dataset, batch_size=cfg.training.batch_size,
        shuffle=True, collate_fn=collate_hoi, num_workers=4,
    )

    # --- Optimizer (all trainable params, reduced LR) ---
    all_params = (
        list(predictor.parameters())
        + list(object_encoder.parameters())
        + list(transformer.parameters())
        + list(interaction_tokenizer.parameters())
        + list(extractor.parameters())
    )
    optimizer = build_optimizer(all_params, lr=cfg.training.optimizer.lr)
    total_steps = len(dataloader) * cfg.training.num_epochs
    scheduler = build_scheduler(optimizer, cfg.training.scheduler.warmup_steps, total_steps)

    # --- Prepare ---
    (predictor, object_encoder, transformer, interaction_tokenizer, extractor,
     optimizer, dataloader, scheduler) = accelerator.prepare(
        predictor, object_encoder, transformer, interaction_tokenizer, extractor,
        optimizer, dataloader, scheduler,
    )
    vq_vae = vq_vae.to(accelerator.device)

    step_fn = build_joint_step_fn(
        predictor, object_encoder, transformer, vq_vae, interaction_tokenizer,
        extractor, clip_model, pred_criterion, gen_criterion, cons_criterion,
        priors, consistency_weight=cfg.loss.consistency_weight,
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
        model=predictor,  # used for checkpoint saving (main model)
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
    parser.add_argument("--config", type=str, default="configs/training/joint_finetune.yaml")
    args = parser.parse_args()
    run(args.config)
