"""Probe: can the local GPU run a Stage B v18 train step?

Loads the v18 generator (MoMask masked + residual transformer + RVQ-VAE
+ InteractionMaskTransformer + InteractionTokenizer + decoded_contact_aux),
runs a single forward + backward at the configured batch size in bf16,
and prints peak VRAM.

PREREQ: MoMask checkpoints must be present at the paths in the cfg's
``model.checkpoints.{vq_vae, masked_transformer, residual_transformer}``.
Run AFTER finishing the MoMask weight download.

Usage:
    python scripts/stage_b_generator/probe_local_gpu.py \\
        --config configs/training/generator_v18_v12strict_local.yaml \\
        --batch-size 8

The probe is informative for tuning batch_size / grad_accum_steps in
the local yaml. Server uses batch=32 + grad_accum=2 across 2× A6000
(effective 128); local single-GPU 4070 12 GB needs smaller batch.
Typical results to expect (rough):
    batch=4  → ~5-7 GB peak reserved (lots of headroom)
    batch=8  → ~8-10 GB peak reserved (probably fits)
    batch=16 → likely OOM
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from piano.utils.clip_utils import set_clip_cache_root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=Path("configs/training/generator_v18_v12strict_local.yaml"))
    parser.add_argument("--batch-size", type=int, default=None,
                        help="override cfg.training.batch_size")
    parser.add_argument("--seq-len", type=int, default=196)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available")
        return 1

    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"Total VRAM: {total_mem:.2f} GB")

    cfg = OmegaConf.load(str(args.config))
    model_cfg = OmegaConf.load(cfg.model.config)

    clip_dl_root = cfg.model.get("clip_download_root", None)
    if clip_dl_root is not None:
        set_clip_cache_root(clip_dl_root)
        print(f"CLIP cache redirected to: {clip_dl_root}")

    # Verify MoMask ckpts exist before the heavy load.
    ckpts = cfg.model.checkpoints
    for key in ("vq_vae", "masked_transformer", "residual_transformer"):
        p = Path(ckpts[key])
        if not p.exists():
            print(
                f"\nERROR: missing MoMask ckpt: {p}\n"
                f"  → finish downloading MoMask weights from\n"
                f"    https://github.com/EricGuo5513/momask-codes\n"
                f"  → place under checkpoints/momask/t2m/<run>/model/<file>.tar"
            )
            return 1

    batch_size = args.batch_size or int(cfg.training.batch_size)

    torch.cuda.reset_peak_memory_stats()
    print(f"\nLoading MoMask backbones...")
    from piano.models.backbones.momask_adapter import (
        load_momask_mask_transformer,
        load_momask_residual_transformer,
        load_momask_vqvae,
    )
    vq_vae = load_momask_vqvae(ckpts.vq_vae, device=device)
    mask_transformer = load_momask_mask_transformer(
        ckpts.masked_transformer, device=device,
    )
    residual_transformer = load_momask_residual_transformer(
        ckpts.residual_transformer, device=device,
    )

    from piano.models.motion_generator import (
        InteractionMaskTransformer, ResidualTransformerWithInteraction,
    )
    from piano.models.interaction_tokenizer import InteractionTokenizer

    # Build the same wrappers train_generator builds. Defaults match the
    # v18 yaml; if the cfg overrides any of these, the matching value
    # is used. Tokenizer's d_model must equal mask_transformer.latent_dim
    # (384 for HumanML3D MoMask).
    tokenizer = InteractionTokenizer(
        d_model=mask_transformer.latent_dim,
        max_seq_length=args.seq_len,
    ).to(device)

    rint_cfg = cfg.model.get("residual_int_xattn", {})
    transformer = InteractionMaskTransformer(
        mask_transformer=mask_transformer,
        interaction_tokenizer=tokenizer,
        interaction_drop_prob=float(model_cfg.get("interaction_cross_attn", {})
                                    .get("interaction_drop_prob", 0.1)),
        zero_init_gamma=True,
        gamma_kind=cfg.model.get("gamma_kind", "per_head"),
        wrapper_kind=str(cfg.model.get("wrapper_kind", "v0.6")),
    ).to(device)

    residual_wrapper = ResidualTransformerWithInteraction(
        residual_transformer=residual_transformer,
        d_model=mask_transformer.latent_dim,
        num_heads=mask_transformer.seqTransEncoder.layers[0].self_attn.num_heads
                  if hasattr(mask_transformer.seqTransEncoder, "layers")
                  else 6,
        dropout=float(rint_cfg.get("dropout", 0.1)),
        zero_init_gamma=bool(rint_cfg.get("zero_init_gamma", True)),
        gamma_kind=str(rint_cfg.get("gamma_kind", "per_head")),
    ).to(device)

    n_train = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    n_train += sum(p.numel() for p in residual_wrapper.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in transformer.parameters())
    n_total += sum(p.numel() for p in residual_wrapper.parameters())
    n_total += sum(p.numel() for p in vq_vae.parameters())
    print(f"Trainable params: {n_train/1e6:.2f} M of {n_total/1e6:.2f} M total")

    # Synthesize a single batch and run forward + backward through the
    # base path only (skip decoded_contact_aux for the probe — that path
    # adds graph state but is gated, the base path dominates VRAM).
    B, T = batch_size, args.seq_len
    print(f"\nForward+backward probe: B={B}, T={T}, bf16=True")
    base_ids = torch.randint(0, 512, (B, T // 4), device=device)
    cond_vector = torch.randn(B, 512, device=device)
    m_lens_tok = torch.full((B,), T // 4, dtype=torch.long, device=device)
    contact_state = torch.rand(B, T, 5, device=device).round()
    contact_target_xyz = torch.randn(B, T, 5, 3, device=device) * 0.3
    phase = torch.randint(0, 3, (B, T), device=device)
    support = torch.randint(0, 4, (B, T), device=device)
    obj_com_canonical = torch.randn(B, T, 3, device=device) * 0.5
    obj_rot6d_canonical = torch.randn(B, T, 6, device=device) * 0.5
    seq_len = torch.full((B,), T, dtype=torch.long, device=device)

    optimizer = torch.optim.AdamW(
        [p for p in transformer.parameters() if p.requires_grad]
        + [p for p in residual_wrapper.parameters() if p.requires_grad],
        lr=1e-4,
    )

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        int_tokens_bf, int_pad_mask_bf = transformer.interaction_tokenizer(
            contact_state=contact_state,
            contact_target_xyz=contact_target_xyz,
            phase=phase, support=support,
            obj_com_canonical=obj_com_canonical,
            obj_rot6d_canonical=obj_rot6d_canonical,
            seq_lens=seq_len,
        )
        out = transformer(
            ids=base_ids,
            cond_vector=cond_vector,
            m_lens_tok=m_lens_tok,
            int_tokens_bf=int_tokens_bf,
            int_padding_mask_bf=int_pad_mask_bf,
            cfg_drop_buckets={"drop_both": 0.1, "drop_int_only": 0.1, "drop_text_only": 0.05},
            return_logits=False,
        )

    loss = out["loss"] if "loss" in out else (
        out.get("loss_base", torch.zeros((), device=device))
        + out.get("loss_residual", torch.zeros((), device=device))
    )
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    peak = torch.cuda.max_memory_allocated() / 1024**3
    reserved = torch.cuda.max_memory_reserved() / 1024**3
    print(f"\nPeak allocated: {peak:.2f} GB")
    print(f"Peak reserved:  {reserved:.2f} GB")
    print(f"Total VRAM:     {total_mem:.2f} GB")
    print(f"Headroom:       {total_mem - reserved:.2f} GB")
    print(f"\nLoss: {loss.item():.4f}")

    # Recommendations.
    if reserved < total_mem * 0.6:
        print(f"\n→ Comfortable. Try a larger batch_size to reduce grad_accum.")
    elif reserved < total_mem * 0.85:
        print(f"\n→ Acceptable headroom. Current batch_size={B} is reasonable.")
    else:
        print(f"\n→ TIGHT. Consider batch_size={max(B // 2, 1)} with 2× grad_accum.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
