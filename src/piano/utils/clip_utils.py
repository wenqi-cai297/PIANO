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
    features : (B, 77, D) float32 — per-token features. D is the CLIP
        text transformer width (512 for ViT-B/32).
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

    return x.float(), key_padding_mask


def load_clip_text_encoder(
    device: torch.device,
    model_name: str = "ViT-B/32",
) -> nn.Module:
    """Load OpenAI CLIP, freeze all parameters, and move to *device*.

    Only the text tower is used downstream, but ``clip.load`` returns the
    full model — the visual tower stays on device with no gradients and
    no forward passes.
    """
    try:
        import clip
    except ImportError as e:  # pragma: no cover — server-only path
        raise ImportError(
            "OpenAI CLIP is required. Install via "
            "`pip install ftfy regex git+https://github.com/openai/CLIP.git`."
        ) from e

    # jit=False keeps Python-level access to the internals we need for
    # per-token feature extraction.
    model, _ = clip.load(model_name, device=device, jit=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
