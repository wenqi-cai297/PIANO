"""Deprecation shim — replaced by :mod:`piano.models.interaction_tokenizer`
and the per-block IntXAttn sublayers in
:mod:`piano.models.motion_generator`.

The original implementation in this file (Stage A pre-launch
scaffolding, 2026-04-23) was the wrong shape for v11 pseudo-labels:
``contact_target`` was K=16 classification (since dropped in favour of
xyz regression in v10/v3), ``phase`` was 5-class (since collapsed to
3-class in v5/v11), and the cross-attention sublayer it defined sat in
the wrong place (``[SelfAttn → FFN → IntXAttn]`` instead of
``[SelfAttn → IntXAttn → FFN]``) and lacked the per-layer γ_int gate
that makes zero-init byte-identity work.

Replaced 2026-04-26 evening when Stage B implementation landed. The
new modules:

- :class:`piano.models.interaction_tokenizer.InteractionTokenizer` —
  v11 channel layout (5 + 5×3 + 3 + 4 = 27), 1D-conv stride-4 to
  match MoMask's ``unit_length=4`` (verified from
  ``backbones/momask/options/base_option.py``).
- :class:`piano.models.motion_generator.MaskTransformerBlockWithInteraction` —
  per-block IntXAttn at the right position with γ_int gate.

This file remains as a re-export so legacy imports
(``from piano.models.interaction_cross_attn import InteractionTokenizer``
in stale Stage C / inference / smoke skeletons) keep parsing. Calls to
the legacy ``InteractionTokenizer(...)`` constructor with the old
positional kwargs (``contact_dim=5, target_dim=80, ...``) will raise
``TypeError`` because the new tokenizer's signature is incompatible —
that's the right behaviour: it forces the caller to update to the
v11 channel layout.
"""
from __future__ import annotations

import warnings

from piano.models.interaction_tokenizer import InteractionTokenizer

warnings.warn(
    "piano.models.interaction_cross_attn is deprecated; import "
    "InteractionTokenizer from piano.models.interaction_tokenizer "
    "instead. The new tokenizer has v11 channel layout and is wired "
    "into per-block IntXAttn sublayers via "
    "piano.models.motion_generator.InteractionMaskTransformer.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["InteractionTokenizer"]
