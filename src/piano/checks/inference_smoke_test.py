"""End-to-end inference smoke test with untrained PIANO components.

Verifies that the full generation pipeline runs without errors:
    1. Load pretrained MoMask (VQ-VAE, MaskTransformer, ResidualTransformer)
    2. Wrap MaskTransformer with our InteractionMaskTransformer (zero-init)
    3. Build untrained InteractionPredictor + ObjectEncoder + InteractionTokenizer
    4. Load a few real samples from preprocessed OMOMO
    5. Run the full pipeline: text + object + init_pose → motion tokens → motion
    6. Report output shapes and basic statistics

Because the interaction cross-attention is zero-initialized, the output
at this stage should be identical to pure MoMask text-only generation
(i.e., the object and predicted z_int have no effect yet). This is the
correct "baseline" behavior before finetuning.

Usage:
    python -m piano.checks.inference_smoke_test \\
        --data-dir /path/to/omomo/piano \\
        --momask-dir /path/to/momask/t2m \\
        [--num-samples 4] [--device cuda]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from piano.data.dataset import HOIDataset, collate_hoi
from piano.models.backbones.momask_adapter import (
    load_momask_mask_transformer,
    load_momask_residual_transformer,
    load_momask_vqvae,
)
from piano.models.interaction_cross_attn import InteractionTokenizer
from piano.models.interaction_predictor import InteractionPredictor
from piano.models.motion_generator import InteractionMaskTransformer
from piano.models.object_encoder import ObjectEncoder


def run_smoke_test(
    data_dir: Path,
    momask_dir: Path,
    num_samples: int,
    device: str,
    max_seq_length: int,
) -> None:
    """Load a few samples and run through the full inference pipeline."""
    # ---------------------------------------------------------------
    # 1. Load pretrained MoMask components
    # ---------------------------------------------------------------
    print("=" * 72)
    print("[1/5] Loading pretrained MoMask ...")
    print("=" * 72)

    vq_vae = load_momask_vqvae(
        momask_dir / "rvq_nq6_dc512_nc512_noshare_qdp0.2" / "model" / "net_best_fid.tar",
        device=device,
    )
    mask_ckpt_path = momask_dir / "t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns" / "model" / "latest.tar"
    res_trans = load_momask_residual_transformer(
        momask_dir / "tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw" / "model" / "net_best_fid.tar",
        device=device,
    )

    # ---------------------------------------------------------------
    # 2. Wrap MaskTransformer with our interaction cross-attn (zero-init)
    # ---------------------------------------------------------------
    print()
    print("=" * 72)
    print("[2/5] Wrapping MaskTransformer with InteractionMaskTransformer ...")
    print("=" * 72)

    interaction_tokenizer = InteractionTokenizer(d_model=384, temporal_stride=4)
    interaction_tokenizer.to(device).eval()

    mask_trans = InteractionMaskTransformer.from_pretrained(
        mask_ckpt_path,
        interaction_tokenizer=interaction_tokenizer,
        device=device,
    )
    mask_trans.eval()

    n_interaction = sum(p.numel() for p in mask_trans.interaction_parameters())
    n_backbone = sum(p.numel() for p in mask_trans.backbone_parameters())
    print(f"  Interaction params (new, zero-init): {n_interaction/1e6:.2f}M")
    print(f"  MoMask backbone params (pretrained): {n_backbone/1e6:.2f}M")

    # ---------------------------------------------------------------
    # 3. Build untrained PIANO components
    # ---------------------------------------------------------------
    print()
    print("=" * 72)
    print("[3/5] Building untrained PIANO components ...")
    print("=" * 72)

    object_encoder = ObjectEncoder(
        num_input_points=1024, num_output_tokens=16, feature_dim=384,
    ).to(device).eval()

    predictor = InteractionPredictor(
        d_model=384, num_layers=10, num_heads=6,
        dim_feedforward=1024, text_dim=512, pose_dim=263,
        max_seq_length=max_seq_length, block_size=2,
    ).to(device).eval()

    print(f"  ObjectEncoder:       {sum(p.numel() for p in object_encoder.parameters())/1e6:.2f}M")
    print(f"  InteractionPredictor: {sum(p.numel() for p in predictor.parameters())/1e6:.2f}M")

    # ---------------------------------------------------------------
    # 4. Load a few real samples
    # ---------------------------------------------------------------
    print()
    print("=" * 72)
    print(f"[4/5] Loading {num_samples} samples from {data_dir} ...")
    print("=" * 72)

    dataset = HOIDataset(root=data_dir, max_seq_length=max_seq_length)
    # Build a small batch from the first N samples
    samples = [dataset[i] for i in range(num_samples)]
    batch = collate_hoi(samples)

    print(f"  Sample texts:")
    for i, t in enumerate(batch["text"]):
        print(f"    [{i}] {t[:80]}")
    print(f"  Object PC shape:  {tuple(batch['object_pc'].shape)}")
    print(f"  Motion shape:     {tuple(batch['motion'].shape)}")
    print(f"  Seq lengths:      {batch['seq_len'].tolist()}")

    # ---------------------------------------------------------------
    # 5. Run full inference pipeline
    # ---------------------------------------------------------------
    print()
    print("=" * 72)
    print("[5/5] Running full inference pipeline ...")
    print("=" * 72)

    object_pc = batch["object_pc"].to(device)
    init_pose = batch["motion"][:, 0, :].to(device)           # first-frame features

    # Text encoding via MoMask's CLIP
    with torch.no_grad():
        text_emb = mask_trans.encode_text(batch["text"])       # (B, 512)
        print(f"\n  CLIP text_emb:        {tuple(text_emb.shape)}")

        # Object encoding
        obj_tokens = object_encoder(object_pc)                 # (B, 16, 384)
        print(f"  Object tokens:        {tuple(obj_tokens.shape)}")

        # Interaction prediction (untrained — random output)
        pred = predictor(text_emb, obj_tokens, init_pose, seq_length=max_seq_length)
        print(f"  z_int contact_state:  {tuple(pred['contact_state'].shape)}")
        print(f"  z_int phase:          {tuple(pred['phase'].shape)}")
        print(f"  z_int support:        {tuple(pred['support'].shape)}")

        # Interaction tokenize (downsample to VQ-token resolution)
        interaction_tokens = interaction_tokenizer(
            pred["contact_state"],
            pred["contact_target"],
            pred["phase"],
            pred["support"],
        )
        print(f"  Interaction tokens:   {tuple(interaction_tokens.shape)} (temporal: {max_seq_length} → {interaction_tokens.shape[1]})")

        # Generate motion tokens
        # Token length = max_seq_length / 4 (VQ downsample factor)
        token_len = max_seq_length // 4
        m_lens = torch.full((len(samples),), token_len, dtype=torch.long, device=device)
        gen_token_ids = mask_trans.generate(
            cond=text_emb,
            m_lens=m_lens,
            interaction_tokens=interaction_tokens,
            timesteps=10,
            cond_scale=4.5,
            interaction_scale=2.0,
            temperature=1.0,
        )
        print(f"  Generated base tokens: {tuple(gen_token_ids.shape)}  range=[{gen_token_ids.min().item()}, {gen_token_ids.max().item()}]")

        # Residual transformer to refine tokens
        # (MoMask's ResidualTransformer expects the base token ids)
        valid_mask = gen_token_ids >= 0
        # Replace -1 padding with 0 for decoding
        base_tokens = torch.where(valid_mask, gen_token_ids, torch.zeros_like(gen_token_ids))
        # Use the MoMask generate() of residual transformer to fill Q2..QN
        all_indices = res_trans.generate(
            base_tokens,
            batch["text"],
            m_lens,
            temperature=1.0,
            cond_scale=5,
        )
        print(f"  Full token indices:   {tuple(all_indices.shape)}")

        # Decode to motion features via VQ-VAE
        motion_263 = vq_vae.forward_decoder(all_indices)      # (B, T, 263)
        print(f"  Decoded motion_263:   {tuple(motion_263.shape)}")

        # Basic output sanity
        finite = torch.isfinite(motion_263).all()
        print(f"  Output finite:        {bool(finite)}")
        print(f"  Output mean/std:      {motion_263.mean().item():.3f} / {motion_263.std().item():.3f}")

    print()
    print("=" * 72)
    print("SUCCESS: end-to-end inference pipeline runs cleanly.")
    print("=" * 72)
    print()
    print("Note: with all PIANO components untrained and interaction")
    print("cross-attn zero-initialized, the output at this stage is equivalent")
    print("to pure MoMask text-only generation. This is the expected baseline")
    print("before any finetuning — it proves the plumbing works.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--data-dir", type=Path, default=Path("/media/gpu-server-1/4TB_for_data/Cai/datasets/omomo/piano"),
        help="Preprocessed PIANO dataset directory",
    )
    parser.add_argument(
        "--momask-dir", type=Path, default=Path("checkpoints/momask/t2m"),
        help="MoMask pretrained weights root",
    )
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=196)
    parser.add_argument("--device", type=str, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_smoke_test(
        data_dir=args.data_dir,
        momask_dir=args.momask_dir,
        num_samples=args.num_samples,
        device=device,
        max_seq_length=args.max_seq_length,
    )


if __name__ == "__main__":
    main()
