"""Sanity check: verify MoMask pretrained weights can be loaded via our adapter.

This is the single most important pre-training check. If weight loading fails
here, the entire Stage B / Stage C pipeline will not work.

Verifies:
    1. MoMask source code is reachable (backbones/momask/ is cloned)
    2. momask_adapter's sys.path injection works
    3. VQ-VAE, MaskTransformer, ResidualTransformer all load with matching keys
    4. CLIP (loaded inside MoMask's MaskTransformer constructor) is available

Usage:
    python -m piano.checks.momask_weights \\
        [--ckpt-root checkpoints/momask/t2m] \\
        [--device cuda]
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch

from piano.models.backbones.momask_adapter import (
    load_momask_mask_transformer,
    load_momask_residual_transformer,
    load_momask_vqvae,
)
from piano.utils.io_utils import ensure_dir, save_json


def run_check(ckpt_root: Path, device: str, output_dir: Path) -> None:
    """Load all three MoMask models and report status, saving a summary."""
    output_dir = ensure_dir(output_dir)

    vq_ckpt = ckpt_root / "rvq_nq6_dc512_nc512_noshare_qdp0.2" / "model" / "net_best_fid.tar"
    mask_ckpt = ckpt_root / "t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns" / "model" / "latest.tar"
    res_ckpt = ckpt_root / "tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw" / "model" / "net_best_fid.tar"

    for name, path in [("VQ-VAE", vq_ckpt), ("MaskTransformer", mask_ckpt), ("ResidualTransformer", res_ckpt)]:
        if not path.exists():
            raise FileNotFoundError(f"{name} checkpoint not found at {path}")

    print("=" * 70)
    print("Test 1/3: Load MoMask VQ-VAE")
    print("=" * 70)
    vq_vae = load_momask_vqvae(vq_ckpt, device=device)
    vq_params = sum(p.numel() for p in vq_vae.parameters())
    print(f"  Params: {vq_params / 1e6:.1f}M")

    print()
    print("=" * 70)
    print("Test 2/3: Load MoMask MaskTransformer (includes CLIP)")
    print("=" * 70)
    mask_trans = load_momask_mask_transformer(mask_ckpt, device=device)
    mask_params = sum(p.numel() for p in mask_trans.parameters())
    print(f"  Params: {mask_params / 1e6:.1f}M")

    print()
    print("=" * 70)
    print("Test 3/3: Load MoMask ResidualTransformer (includes CLIP)")
    print("=" * 70)
    res_trans = load_momask_residual_transformer(res_ckpt, device=device)
    res_params = sum(p.numel() for p in res_trans.parameters())
    print(f"  Params: {res_params / 1e6:.1f}M")

    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "device": device,
        "ckpt_root": str(ckpt_root),
        "checkpoints": {
            "vq_vae": str(vq_ckpt),
            "mask_transformer": str(mask_ckpt),
            "residual_transformer": str(res_ckpt),
        },
        "params": {
            "vq_vae_M": round(vq_params / 1e6, 2),
            "mask_transformer_M": round(mask_params / 1e6, 2),
            "residual_transformer_M": round(res_params / 1e6, 2),
            "total_M": round((vq_params + mask_params + res_params) / 1e6, 2),
        },
        "status": "success",
    }
    save_json(output_dir / "summary.json", summary)

    print()
    print("=" * 70)
    print(f"SUCCESS: all MoMask weights loaded. Summary: {output_dir / 'summary.json'}")
    print("=" * 70)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--ckpt-root", type=Path, default=Path("checkpoints/momask/t2m"),
        help="Root directory containing MoMask checkpoint folders",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to load onto (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: runs/checks/momask_weights/<timestamp>/)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        output_dir = Path("runs/checks/momask_weights") / timestamp
    else:
        output_dir = args.output_dir

    run_check(args.ckpt_root, device, output_dir)


if __name__ == "__main__":
    main()
