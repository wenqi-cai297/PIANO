"""CLIP text encoder helpers for PIANO.

Extracts per-token text features (before the EOT-gather + text projection)
for use as cross-attention K/V. OpenAI CLIP's ``encode_text`` only returns
the pooled vector; the 77-token sequence carries the verb/noun/modifier
structure that disambiguates "push" vs "pull", "small box" vs "large box",
etc. Using the pooled vector alone discards that information.

This helper works with both a standalone ``clip.load("ViT-B/32")`` model
and MoMask's bundled CLIP (``mask_transformer.clip_model``), since both
are the same underlying OpenAI CLIP module.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def encode_text_per_token(
    clip_model: nn.Module,
    text_list: list[str],
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Return per-token CLIP features + key-padding mask.

    Replicates ``clip.model.encode_text`` up to (but not including) the
    EOT-gather and text projection.

    Parameters
    ----------
    clip_model : an OpenAI CLIP model (or any module exposing
        ``token_embedding``, ``positional_embedding``, ``transformer``,
        ``ln_final``, and ``dtype``).
    text_list : list of raw prompt strings.
    device : target device for the tokenizer output.

    Returns
    -------
    features : (B, 77, D) — per-token features in CLIP's native dtype
        (typically fp16 on GPU). The downstream Linear layer in the
        predictor runs under bf16 autocast, which casts inputs as
        needed; forcing ``.float()`` here would have triggered an
        unnecessary fp32 path and lost the autocast efficiency.
        D is the CLIP text transformer width (512 for ViT-B/32).
    key_padding_mask : (B, 77) bool — True where the position is past
        each row's EOT (MultiheadAttention's ``key_padding_mask`` convention).
    """
    import clip

    token_ids = clip.tokenize(text_list, truncate=True).to(device)  # (B, 77)

    with torch.no_grad():
        x = clip_model.token_embedding(token_ids).type(clip_model.dtype)
        x = x + clip_model.positional_embedding.type(clip_model.dtype)
        x = x.permute(1, 0, 2)                                # NLD -> LND
        x = clip_model.transformer(x)
        x = x.permute(1, 0, 2)                                # LND -> NLD
        x = clip_model.ln_final(x).type(clip_model.dtype)     # (B, 77, D)

    # Positions past EOT are padding. EOT is the max token-id in the row.
    eot_pos = token_ids.argmax(dim=-1)                        # (B,)
    positions = torch.arange(token_ids.shape[1], device=device)
    key_padding_mask = positions.unsqueeze(0) > eot_pos.unsqueeze(1)

    return x, key_padding_mask


def set_clip_cache_root(path: str) -> None:
    """Monkeypatch ``clip.load``'s default ``download_root`` to ``path``.

    OpenAI CLIP's ``load(name, device, jit, download_root=None)`` defaults
    download_root to ``~/.cache/clip`` (hardcoded). Some upstream
    dependencies (notably MoMask's ``load_and_freeze_clip`` in
    ``backbones/momask/models/mask_transformer/transformer.py:178``)
    call ``clip.load`` directly without exposing the kwarg, so they
    write to the user-home cache.

    This helper monkeypatches ``clip.load`` at the module level so all
    subsequent calls — even from third-party code — use the supplied
    cache root. Idempotent: safe to call multiple times. The original
    function is preserved on a private attribute so unit tests can
    restore it if needed.

    Use case: keep all model weights co-located with the project
    workspace so the project directory is self-contained.

    Parameters
    ----------
    path : str
        Directory to use as ``download_root``. Created on demand by
        ``clip.load`` if missing.
    """
    import os
    import clip

    target = os.path.abspath(os.path.expanduser(path))
    os.makedirs(target, exist_ok=True)

    # Save original on first patch only (so repeated calls don't
    # re-wrap the patched version into infinite recursion).
    if not hasattr(clip, "_piano_orig_load"):
        clip._piano_orig_load = clip.load

    orig = clip._piano_orig_load

    def _patched_load(name, device="cpu", jit=False, download_root=None):
        if download_root is None:
            download_root = target
        return orig(name, device=device, jit=jit, download_root=download_root)

    clip.load = _patched_load


def load_clip_text_encoder(
    device: torch.device,
    model_name: str = "ViT-B/32",
    download_root: str | None = None,
) -> nn.Module:
    """Load OpenAI CLIP, freeze all parameters, and move to *device*.

    Only the text tower is used downstream, but ``clip.load`` returns the
    full model — the visual tower stays on device with no gradients and
    no forward passes.

    Parameters
    ----------
    download_root : optional override for where CLIP weights are
        downloaded / loaded from. Defaults to ``~/.cache/clip`` (OpenAI
        CLIP default). Pass a workspace-local path (e.g. ``./cache/clip``)
        to keep the ~340 MB ``ViT-B-32.pt`` weights inside the project
        directory rather than the user-home cache.
    """
    try:
        import clip
    except ImportError as e:  # pragma: no cover — server-only path
        raise ImportError(
            "OpenAI CLIP is required. Install via "
            "`pip install ftfy regex git+https://github.com/openai/CLIP.git` "
            "or the PyPI mirror `pip install ftfy regex clip-anytorch`."
        ) from e

    # jit=False keeps Python-level access to the internals we need for
    # per-token feature extraction.
    model, _ = clip.load(
        model_name, device=device, jit=False,
        download_root=download_root,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
