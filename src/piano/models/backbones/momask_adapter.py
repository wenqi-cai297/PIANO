"""MoMask adapter: imports MoMask's original classes and provides PIANO-friendly wrappers.

This module handles the sys.path setup to import from the cloned MoMask repo
(``backbones/momask/``), and exposes the key classes with a clean interface:
    - ``load_momask_vqvae``: load pretrained VQ-VAE for motion tokenization
    - ``load_momask_mask_transformer``: load pretrained MaskTransformer
    - ``load_momask_residual_transformer``: load pretrained ResidualTransformer

We import MoMask's classes directly — no reimplementation, no weight mismatch.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Set up import path for MoMask's flat repo structure
# ---------------------------------------------------------------------------

_MOMASK_ROOT = Path(__file__).parent / "momask"
if not _MOMASK_ROOT.exists():
    raise FileNotFoundError(
        f"MoMask repo not found at {_MOMASK_ROOT}. "
        f"Run: git clone https://github.com/EricGuo5513/momask-codes.git {_MOMASK_ROOT}"
    )

# Add MoMask root to sys.path so its internal imports work
# (e.g., ``from models.mask_transformer.tools import *``)
_momask_path = str(_MOMASK_ROOT)
if _momask_path not in sys.path:
    sys.path.insert(0, _momask_path)


# ---------------------------------------------------------------------------
# Lazy imports from MoMask (deferred to avoid import-time CLIP loading)
# ---------------------------------------------------------------------------

def _import_mask_transformer():
    """Import MoMask's MaskTransformer class."""
    from models.mask_transformer.transformer import MaskTransformer
    return MaskTransformer


def _import_residual_transformer():
    """Import MoMask's ResidualTransformer class."""
    from models.mask_transformer.transformer import ResidualTransformer
    return ResidualTransformer


def _import_rvqvae():
    """Import MoMask's RVQVAE class."""
    from models.vq.model import RVQVAE
    return RVQVAE


def _import_length_estimator():
    """Import MoMask's LengthEstimator class."""
    from models.vq.model import LengthEstimator
    return LengthEstimator


# ---------------------------------------------------------------------------
# Opt (options) builder — MoMask models expect an ``opt`` namespace
# ---------------------------------------------------------------------------

def build_momask_opt(
    num_tokens: int = 512,
    num_quantizers: int = 6,
    shared_codebook: bool = False,
    quantize_dropout_prob: float = 0.2,
    mu: float = 0.99,
    device: str = "cpu",
) -> SimpleNamespace:
    """Build a minimal ``opt`` namespace that MoMask constructors expect.

    Default values mirror MoMask's ``options/vq_option.py`` defaults so
    pretrained checkpoints load without mismatch.
    """
    return SimpleNamespace(
        num_tokens=num_tokens,
        num_quantizers=num_quantizers,
        shared_codebook=shared_codebook,
        quantize_dropout_prob=quantize_dropout_prob,
        mu=mu,                          # EMA decay for codebook (MoMask default: 0.99)
        device=device,
    )


# ---------------------------------------------------------------------------
# VQ-VAE loader
# ---------------------------------------------------------------------------

def load_momask_vqvae(
    checkpoint_path: str | Path,
    input_width: int = 263,
    nb_code: int = 512,
    code_dim: int = 512,
    output_emb_width: int = 512,
    down_t: int = 2,
    stride_t: int = 2,
    width: int = 512,
    depth: int = 3,
    dilation_growth_rate: int = 3,
    num_quantizers: int = 6,
    device: str = "cpu",
) -> nn.Module:
    """Load a pretrained MoMask RVQVAE.

    Parameters
    ----------
    checkpoint_path : path to ``.tar`` checkpoint file
    Other args match MoMask's RVQVAE constructor defaults for HumanML3D.

    Returns
    -------
    Frozen RVQVAE model ready for encode/decode.
    """
    RVQVAE = _import_rvqvae()

    args = build_momask_opt(
        num_tokens=nb_code,
        num_quantizers=num_quantizers,
    )

    model = RVQVAE(
        args,
        input_width=input_width,
        nb_code=nb_code,
        code_dim=code_dim,
        output_emb_width=output_emb_width,
        down_t=down_t,
        stride_t=stride_t,
        width=width,
        depth=depth,
        dilation_growth_rate=dilation_growth_rate,
    )

    # Load weights
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("vq_model", ckpt.get("net", ckpt))
    model.load_state_dict(state_dict, strict=True)

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    print(f"Loaded MoMask VQ-VAE from {checkpoint_path}")
    return model.to(device)


# ---------------------------------------------------------------------------
# MaskTransformer loader
# ---------------------------------------------------------------------------

def load_momask_mask_transformer(
    checkpoint_path: str | Path,
    code_dim: int = 512,
    latent_dim: int = 384,
    ff_size: int = 1024,
    num_layers: int = 8,
    num_heads: int = 6,
    dropout: float = 0.1,
    clip_dim: int = 512,
    clip_version: str = "ViT-B/32",
    cond_drop_prob: float = 0.1,
    num_tokens: int = 512,
    device: str = "cpu",
) -> nn.Module:
    """Load a pretrained MoMask MaskTransformer.

    Parameters match MoMask's HumanML3D checkpoint:
    ``t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns``

    Returns
    -------
    MaskTransformer model with loaded weights.
    Note: CLIP model inside is loaded and frozen by MoMask's constructor.
    """
    MaskTransformer = _import_mask_transformer()

    opt = build_momask_opt(num_tokens=num_tokens)
    opt.device = device

    model = MaskTransformer(
        code_dim=code_dim,
        cond_mode="text",
        latent_dim=latent_dim,
        ff_size=ff_size,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        clip_dim=clip_dim,
        cond_drop_prob=cond_drop_prob,
        clip_version=clip_version,
        opt=opt,
    )

    # Load weights (skip CLIP keys — they're loaded separately)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("t2m_transformer", ckpt.get("trans", ckpt))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    # All missing keys should be CLIP-related (loaded in constructor)
    real_missing = [k for k in missing if not k.startswith("clip_model.")]
    if real_missing:
        print(f"[WARN] Missing non-CLIP keys: {real_missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected[:10]}")

    print(f"Loaded MoMask MaskTransformer from {checkpoint_path}")
    return model.to(device)


# ---------------------------------------------------------------------------
# ResidualTransformer loader
# ---------------------------------------------------------------------------

def load_momask_residual_transformer(
    checkpoint_path: str | Path,
    code_dim: int = 512,
    latent_dim: int = 384,
    ff_size: int = 1024,
    num_layers: int = 8,
    num_heads: int = 6,
    dropout: float = 0.1,
    clip_dim: int = 512,
    clip_version: str = "ViT-B/32",
    cond_drop_prob: float = 0.2,
    num_tokens: int = 512,
    num_quantizers: int = 6,
    shared_codebook: bool = False,
    share_weight: bool = True,
    device: str = "cpu",
) -> nn.Module:
    """Load a pretrained MoMask ResidualTransformer.

    Defaults match MoMask's HumanML3D checkpoint
    ``tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw`` (``_sw`` = share_weight=True).
    """
    ResidualTransformer = _import_residual_transformer()

    opt = build_momask_opt(
        num_tokens=num_tokens,
        num_quantizers=num_quantizers,
        shared_codebook=shared_codebook,
    )
    opt.device = device

    model = ResidualTransformer(
        code_dim=code_dim,
        cond_mode="text",
        latent_dim=latent_dim,
        ff_size=ff_size,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        clip_dim=clip_dim,
        cond_drop_prob=cond_drop_prob,
        clip_version=clip_version,
        share_weight=share_weight,
        shared_codebook=shared_codebook,
        opt=opt,
    )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("res_transformer", ckpt.get("trans", ckpt))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    real_missing = [k for k in missing if not k.startswith("clip_model.")]
    if real_missing:
        print(f"[WARN] Missing non-CLIP keys: {real_missing}")

    print(f"Loaded MoMask ResidualTransformer from {checkpoint_path}")
    return model.to(device)
