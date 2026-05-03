"""Quick probe: can the local GPU run the v9.5 predictor + encoder?

Builds the predictor + 256-token object encoder, runs a single forward
+ backward pass at the configured batch size in bf16, and prints peak
VRAM. No training, no dataloader — just a memory feasibility check.
"""
from __future__ import annotations

import argparse
import sys

import torch
import torch.nn.functional as F

from piano.models.interaction_predictor import InteractionPredictor
from piano.models.object_encoder import ObjectEncoder
from piano.training.losses import PredictorLoss


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=196)
    parser.add_argument("--num-tokens", type=int, default=256)
    parser.add_argument("--bf16", action="store_true", default=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available")
        return 1

    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"Total VRAM: {total_mem:.2f} GB")

    torch.cuda.reset_peak_memory_stats()

    predictor = InteractionPredictor(
        d_model=384, num_layers=10, num_heads=6, dim_feedforward=1024,
        text_dim=512, pose_dim=66, max_seq_length=args.seq_len,
        num_body_parts=5, num_phases=3, num_support_states=3,
        structured_head=True,
        structured_head_d_emb=64, structured_head_hidden=256,
        structured_head_attn_heads=6,
        structured_head_downstream_mode="mask",
        structured_head_target_attn_output="logits",
        structured_head_target_attn_kind="hierarchical_mask_decoder",
        structured_head_target_decoder_layers=4,
        structured_head_target_decoder_ffn=1024,
        structured_head_target_pos_enc=True,
        structured_head_target_num_patches=16,
    ).to(device)

    enc = ObjectEncoder(
        num_input_points=1024, num_output_tokens=args.num_tokens,
        feature_dim=384, sa2_radius=0.15, sa2_num_samples=32,
    ).to(device)

    criterion = PredictorLoss(
        contact_weight=2.0, target_weight=5.0, phase_weight=0.3, support_weight=0.1,
        target_loss_kind="focal_dice",
        target_topk_min_positives=1,
        target_aux_xyz_weight=0.3,
        target_patch_weight=0.3,
        # contact_pos_weight=None implicit (default); skips prior-scan,
        # doesn't affect memory probe.
    ).to(device)

    n_params = sum(p.numel() for p in predictor.parameters()) + sum(
        p.numel() for p in enc.parameters()
    )
    print(f"Trainable params: {n_params/1e6:.2f} M")

    B, T = args.batch_size, args.seq_len
    pc = torch.randn(B, 1024, 3, device=device) * 0.4
    text = torch.randn(B, 77, 512, device=device)
    init_pose = torch.randn(B, 66, device=device) * 0.3
    gt_contact = (torch.rand(B, T, 5, device=device) > 0.7).float()
    gt_target = torch.randn(B, T, 5, 3, device=device) * 0.3
    gt_phase = torch.zeros(B, T, dtype=torch.long, device=device)
    gt_support = torch.zeros(B, T, dtype=torch.long, device=device)

    optimizer = torch.optim.AdamW(
        list(predictor.parameters()) + list(enc.parameters()), lr=1e-4,
    )

    print(f"\nForward+backward probe: B={B}, T={T}, M={args.num_tokens}, bf16={args.bf16}")

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.bf16):
        obj_xyz, obj_tokens = enc(pc, return_xyz=True)
        out = predictor(
            text, obj_tokens, init_pose, seq_length=T,
            object_xyz=obj_xyz, gt_contact=gt_contact,
            gt_phase=gt_phase, teacher_forcing=False,
        )
        out_fp32 = {
            k: (v.float() if isinstance(v, torch.Tensor) else v)
            for k, v in out.items()
        }
        loss_dict = criterion(
            out_fp32, gt_contact=gt_contact, gt_target=gt_target,
            gt_phase=gt_phase, gt_support=gt_support,
            mask=None, object_xyz=obj_xyz.float(),
        )

    loss_dict["loss"].backward()
    optimizer.step()
    optimizer.zero_grad()

    peak = torch.cuda.max_memory_allocated() / 1024**3
    reserved = torch.cuda.max_memory_reserved() / 1024**3
    print(f"Peak allocated: {peak:.2f} GB")
    print(f"Peak reserved:  {reserved:.2f} GB")
    print(f"Total VRAM:     {total_mem:.2f} GB")
    print(f"Headroom:       {total_mem - reserved:.2f} GB")
    print(f"\nFinal loss: {loss_dict['loss'].item():.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
