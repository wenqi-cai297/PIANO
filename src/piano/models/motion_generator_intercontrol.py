"""B2 / v0.3-δ scaffold — InterControl-style trainable-copy controllable
MoMask, per arXiv:2311.15864 §3.2.

Architecture
------------
- **Main branch**: original MoMask MaskTransformer's ``seqTransEncoder``,
  permanently FROZEN at the pretrained weights.
- **Control branch**: a deep-copy of the same encoder (same architecture,
  same init weights at start of training), trainable. Each control-branch
  layer is wrapped with an IntXAttn sublayer (per-head γ gate, same as
  v0.6) so z_int can flow through the control branch via cross-attention.
- **Per-layer zero-init linear connectors**: ``nn.Linear(d, d)`` (weight
  AND bias zero-initialised) merge ctrl features into main's residual
  stream. At step 0 the connectors output zero → ctrl branch contributes
  zero → main branch produces byte-identical output to the un-controlled
  MoMask. Training learns to grow the connectors' weights as controllable
  generation rewards them.

This is the **only motion-domain validated** recipe for backbone-pretrained
controllable generation (InterControl NeurIPS'24 / OmniControl ICLR'24 /
MotionLCM ECCV'24 all use this exact pattern). PIANO v0.1-v0.7 sit in
the alternative regime (γ-gate over a fine-tuned backbone), which is
strictly weaker per the source-level audit in
``analyses/2026-04-27_adapter_source_review.md``.

Status: SCAFFOLD ONLY (2026-04-27 night). The encoder class +
byte-identity smoke test are in place; full integration with
``InteractionMaskTransformer`` / ``train_generator`` / ``qual_eval``
follows in the next commit pending v0.7 (mirror-aug) results. If v0.7
closes the bilateral gap, this becomes moot; if not, this is the next
fire.

Sources
-------
- Wang, Z. et al. *InterControl: Zero-shot Human Interaction Generation
  by Controlling Every Joint.* **NeurIPS 2024**. arXiv:2311.15864 §3.2.
- Xie, Y. et al. *OmniControl: Control Any Joint at Any Time for Human
  Motion Generation.* **ICLR 2024**. arXiv:2310.08580 §3.2 + §4.3
  ablation (removing trainable copy → 4× control-error worse).
- Dai, W. et al. *MotionLCM: Real-time Controllable Motion Generation
  via Latent Consistency Model.* **ECCV 2024**. arXiv:2404.19759.
- Zhang, L. & Agrawala, M. *Adding Conditional Control to Text-to-Image
  Diffusion Models (ControlNet).* **ICCV 2023**. arXiv:2302.05543 — the
  recipe template (locked-copy + trainable-copy + zero-conv connectors).
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
from torch import Tensor

from piano.models.motion_generator import MaskTransformerBlockWithInteraction


# ============================================================================
# Trainable-copy encoder
# ============================================================================

class InterControlTransformerEncoder(nn.Module):
    """Drop-in replacement for ``seqTransEncoder`` (v0.3-δ pattern).

    Holds a frozen reference to the original MoMask ``nn.TransformerEncoder``'s
    layers (main branch) AND a trainable deep-copy wrapped with IntXAttn
    sublayers (control branch). Per-layer zero-init linear connectors
    merge ctrl features into the main residual stream.

    Forward signature matches v0.6's :class:`MaskTransformerEncoderWithInteraction`
    so the wrapping :class:`piano.models.motion_generator.InteractionMaskTransformer`
    can swap encoders without touching ``trans_forward``.

    Parameters
    ----------
    original_encoder
        The pretrained ``nn.TransformerEncoder`` from MoMask. Its
        ``.layers`` are stored by reference (NOT copied) for the main
        branch — caller is responsible for setting ``requires_grad=False``
        on those parameters.
    d_model, num_heads, dropout
        Pulled from the original encoder; passed through to wrap each
        control-branch layer with IntXAttn.
    gamma_kind
        ``"scalar"`` (1 dof per ctrl layer) or ``"per_head"`` (n_heads
        dof per ctrl layer). Default ``"per_head"`` matches v0.6.
    """

    def __init__(
        self,
        original_encoder: nn.TransformerEncoder,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        gamma_kind: str = "per_head",
    ) -> None:
        super().__init__()

        # Main branch: hold by reference (no copy). Caller freezes via
        # ``InterControlTransformerEncoder.freeze_main_branch()`` so
        # gradient updates skip these.
        self.main_layers = original_encoder.layers     # nn.ModuleList of nn.TransformerEncoderLayer
        self.main_norm = original_encoder.norm         # typically None for MoMask

        # Control branch: deepcopy each layer so trainable updates don't
        # leak into the frozen main branch. Wrap with IntXAttn for the
        # cross-attention path that carries z_int.
        ctrl_raw_layers = nn.ModuleList(
            [copy.deepcopy(layer) for layer in original_encoder.layers],
        )
        self.ctrl_layers = nn.ModuleList([
            MaskTransformerBlockWithInteraction(
                original_layer=layer,
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
                zero_init_gamma=True,
                gamma_kind=gamma_kind,
            )
            for layer in ctrl_raw_layers
        ])

        # Per-layer zero-init connectors. ControlNet-style "zero conv"
        # but for transformer features: nn.Linear(d, d) with both
        # weight AND bias initialised to zero. At step 0 the ctrl
        # branch contributes zero to main → the encoder is byte-identical
        # to base MoMask. Training grows the connector weights as
        # controllable generation rewards them.
        self.connectors = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in original_encoder.layers
        ])
        for conn in self.connectors:
            nn.init.zeros_(conn.weight)
            nn.init.zeros_(conn.bias)

        # Compatibility alias: the v0.6 ``MaskTransformerEncoderWithInteraction``
        # exposes ``.layers`` as the IntXAttn-bearing blocks (each has
        # ``.gamma_int``). Several call sites — train_generator's
        # γ_int_abs_mean diagnostic, measure_effect_size's hook attach,
        # tests/test_motion_generator.py param-group test — iterate
        # ``encoder.layers`` to find the per-block γ. In v0.3-δ those
        # γ live on the **control branch** (the trainable copy with
        # IntXAttn). Aliasing ``layers`` to ``ctrl_layers`` lets the
        # existing diagnostics work transparently. Main branch layers
        # (frozen, no γ) stay accessible as ``main_layers``.
        self.layers = self.ctrl_layers

    def freeze_main_branch(self) -> None:
        """Set ``requires_grad=False`` on all main-branch layer params.

        Idempotent. Call after construction (and after loading
        pretrained weights, if relevant) so the optimizer skips these.
        Per InterControl §3.2 + OmniControl §3.2 the main branch must
        stay frozen for the trainable-copy regime to give controllable
        generation without forgetting pretraining.
        """
        for p in self.main_layers.parameters():
            p.requires_grad = False
        if self.main_norm is not None:
            for p in self.main_norm.parameters():
                p.requires_grad = False

    def forward(
        self,
        src: Tensor,
        int_kv: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        int_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Run main + ctrl branches in lockstep with per-layer merge.

        At init (zero-init connectors), the ctrl branch contributes
        zero → output is byte-identical to ``original_encoder(src)``.
        """
        h_main = src
        h_ctrl = src                 # ctrl branch starts from the same input as main

        for L, (main_layer, ctrl_block, conn) in enumerate(
            zip(self.main_layers, self.ctrl_layers, self.connectors),
        ):
            # ---- Main branch: stock nn.TransformerEncoderLayer.forward
            #      Frozen at pretrained weights; no IntXAttn injection.
            h_main_out = main_layer(
                h_main,
                src_mask=None,
                src_key_padding_mask=src_key_padding_mask,
            )

            # ---- Ctrl branch: trainable deepcopy + IntXAttn (γ-gated)
            h_ctrl_out = ctrl_block(
                h_ctrl,
                int_kv=int_kv,
                src_key_padding_mask=src_key_padding_mask,
                int_key_padding_mask=int_key_padding_mask,
            )

            # ---- Merge: zero-init linear pulls ctrl features into
            #      main residual stream. At init this adds 0 → main
            #      output is unchanged.
            h_main = h_main_out + conn(h_ctrl_out)
            h_ctrl = h_ctrl_out

        if self.main_norm is not None:
            h_main = self.main_norm(h_main)
        return h_main


# ============================================================================
# Helper: count trainable vs frozen params after wrapping.
# ============================================================================

def count_trainable_vs_frozen(encoder: InterControlTransformerEncoder) -> dict[str, int]:
    """Return ``{trainable: N, frozen: N, total: N}`` over the encoder
    submodule. Useful as a sanity-check post-construction +
    ``freeze_main_branch()`` to verify the optimizer's param group is
    pulling the right tensors.
    """
    trainable = 0
    frozen = 0
    for p in encoder.parameters():
        if p.requires_grad:
            trainable += int(p.numel())
        else:
            frozen += int(p.numel())
    return {
        "trainable": trainable,
        "frozen": frozen,
        "total": trainable + frozen,
    }
