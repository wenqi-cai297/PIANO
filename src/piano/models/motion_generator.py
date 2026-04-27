"""Motion Generator: MoMask MaskTransformer with per-block interaction cross-attention.

Stage B finetunes MoMask's pretrained MaskTransformer to consume the
per-frame interaction latent ``z_int`` (predicted by Stage A or, at
training time, the v11 GT pseudo-labels). The modification is
**surgical**: we keep all 8 of MoMask's ``nn.TransformerEncoderLayer``
sublayers — including their pretrained weights — and insert a new
interaction cross-attention sublayer between each block's self-attention
and feedforward. The new sublayer is gated by a per-layer learnable
scalar ``γ_int`` initialised to 0, so at step 0 the wrapped model is
**byte-identical** to the pretrained MoMask checkpoint (ControlNet /
LLaMA-Adapter zero-init convention).

Architecture (verified from
``src/piano/models/backbones/momask/models/mask_transformer/transformer.py``):

    MoMask original block (post-norm, n_layers=8):
        h = norm1(h + dropout(self_attn(h)))
        h = norm2(h + dropout(ffn(h)))

    PIANO-modified block (this file):
        h = norm1(h + dropout(self_attn(h)))                           # MoMask original
        h = h + γ_int · dropout(int_attn(LayerNorm(h), int_kv))         # NEW; γ_int = 0 at init
        h = norm2(h + dropout(ffn(h)))                                  # MoMask original

The new sublayer uses pre-norm + γ-gate + residual (LLaMA-Adapter /
ControlNet structure) rather than the post-norm wrapping the
``analyses/2026-04-26_stageB_design.md §1.3`` sketch shows. Reason:
post-norm with a residual would still invoke ``LayerNorm(h_self + 0)``
at γ=0, which differs from ``h_self`` for non-Gaussian inputs and
breaks byte-identity. Pre-norm + γ-gate gives an **exact** zero-init
identity, which is the whole point of the technique. ControlNet ICCV'23
mixes pre-norm zero-conv branches into a post-norm SD UNet for the same
reason. The flagged design uncertainty in §6.5 is resolved this way.

The text condition stays exactly where MoMask put it: encoded by a
frozen CLIP ViT-B/32, projected via ``cond_emb: Linear(512, 384)``, and
**prepended as token 0** of the motion sequence. We do not move text
into a per-block xattn path — that would change weight identity vs.
the pretrained checkpoint. (The original SPEC §7.2 sketch was wrong
about MoMask having per-block text xattn; correction lives in the
2026-04-26 Stage B design doc.)

Compositional dual-CFG (Liu et al. ECCV'22 eq. 5):

    logits = logits_uncond
           + w_text · (logits_text_only - logits_uncond)
           + w_int  · (logits_full       - logits_text_only)

3 forward passes per denoising step: uncond / text-only / full. Each
sample's "drop interaction" branch substitutes a learnable ``null_int_kv``
(S, d) tensor — broadcast across the batch — to avoid the
softmax-over-all-zero numerical footgun that ``int_kv = 0`` would create
in the new IntXAttn (per design §6.3). At training time, per-sample
4-bucket categorical drops (drop both / drop int only / drop text only /
keep both) populate the equation's branches with the right marginal
probabilities (10% / 10% / 5% / 75%, per design §2.2).

Citations
---------
- Guo et al. *MoMask.* CVPR 2024 Highlight. arXiv:2312.00063.
- Tevet et al. *Human Motion Diffusion Model.* ICLR 2023. arXiv:2209.14916.
- Liu et al. *Compositional Visual Generation with Composable Diffusion
  Models.* ECCV 2022. arXiv:2206.01714. Eq. 5 (multi-condition CFG).
- Zhang & Agrawala. *Adding Conditional Control to Text-to-Image
  Diffusion Models (ControlNet).* ICCV 2023. arXiv:2302.05543.
  Zero-init convention.
- Zhang et al. *LLaMA-Adapter.* arXiv:2303.16199. Per-layer γ-gate.
- Diller & Dai. *CG-HOI.* CVPR 2024. arXiv:2311.16097. Per-block xattn
  precedent for HOI.
- Wang et al. *Move as You Say, Interact as You Can.* CVPR 2024
  Highlight. arXiv:2403.18036. Per-block affordance xattn precedent.
- Ho & Salimans. *Classifier-Free Diffusion Guidance.*
  NeurIPS 2021 Workshop. arXiv:2207.12598.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from piano.models.interaction_tokenizer import InteractionTokenizer


# ============================================================================
# Single-block wrapper: [SelfAttn (MoMask) → IntXAttn(γ·proj) → FFN (MoMask)]
# ============================================================================

class MaskTransformerBlockWithInteraction(nn.Module):
    """One MoMask encoder layer with an injected interaction cross-attn sublayer.

    Holds a reference to the original ``nn.TransformerEncoderLayer`` so
    its self-attn and FFN submodules — and their pretrained weights —
    are reused unchanged. The new IntXAttn sublayer + per-layer γ
    scalar are the only additions.
    """

    def __init__(
        self,
        original_layer: nn.TransformerEncoderLayer,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        zero_init_gamma: bool = True,
        gamma_kind: str = "scalar",
    ) -> None:
        super().__init__()
        self.layer = original_layer
        self.num_heads = int(num_heads)
        if d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model {d_model} must be divisible by num_heads {self.num_heads} "
                f"for per-head gamma to reshape cleanly",
            )
        self.head_dim = d_model // self.num_heads

        # Pre-norm on the cross-attn input. Initialised at PyTorch
        # default (weight=1, bias=0) so the very first forward sees
        # `LayerNorm(h)` as a unit-variance projection of h. The γ gate
        # below makes this layer's contribution exactly zero at init
        # regardless of LayerNorm details.
        self.norm_int = nn.LayerNorm(d_model)

        # Standard multi-head cross-attention. ``batch_first=False``
        # to match MoMask's seq-first convention `(S, B, d)` so we
        # don't have to permute on every block.
        self.int_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False,
        )
        self.dropout_int = nn.Dropout(dropout)

        # γ_int: learnable gate, shape depends on ``gamma_kind``. Zero-init
        # makes the new sublayer an *exact* no-op at step 0, so the wrapped
        # block is byte-identical to the pretrained MoMask block.
        #
        # - "scalar"   : single scalar per layer (8 dof total, v0.1-v0.5
        #                ControlNet-flavour). Stored as ``(1,)`` so
        #                ``.mean()``/``.abs()`` are well-defined.
        # - "per_head" : one scalar per attention head (n_heads dof per
        #                layer = 48 total, v0.6+, LLaMA-Adapter pattern
        #                ICLR'24 — `OpenGVLab/LLaMA-Adapter@alpaca_finetuning_v1/llama/model.py`).
        #                Each head gates independently, allowing some heads
        #                to attend to z_int while others ignore it.
        self.gamma_kind = str(gamma_kind)
        init_value = 0.0 if zero_init_gamma else 1.0
        if self.gamma_kind == "scalar":
            self.gamma_int = nn.Parameter(torch.full((1,), init_value))
        elif self.gamma_kind == "per_head":
            self.gamma_int = nn.Parameter(torch.full((self.num_heads,), init_value))
        else:
            raise ValueError(
                f"gamma_kind must be 'scalar' or 'per_head', got {self.gamma_kind!r}",
            )

    def _apply_gamma(self, x: Tensor) -> Tensor:
        """Multiply ``(S, B, d)`` interaction output by the γ gate.

        Scalar path is plain broadcast multiply (preserves v0.1-v0.5
        behaviour bytewise). Per-head path reshapes the channel axis
        to ``(num_heads, head_dim)``, multiplies each head by its scalar
        gate, then folds back.
        """
        if self.gamma_kind == "scalar":
            return self.gamma_int * x
        # per-head
        S, B, d = x.shape
        x_h = x.view(S, B, self.num_heads, self.head_dim)
        x_h = x_h * self.gamma_int.view(1, 1, self.num_heads, 1)
        return x_h.reshape(S, B, d)

    def forward(
        self,
        src: Tensor,                          # (S+1, B, d) — text token + motion tokens
        int_kv: Tensor | None,                # (S_int, B, d) — interaction K/V
        src_key_padding_mask: Tensor | None = None,   # (B, S+1) True = padded
        int_key_padding_mask: Tensor | None = None,   # (B, S_int) True = padded
    ) -> Tensor:
        layer = self.layer

        # ---- MoMask original: SelfAttn + post-norm + residual ----
        # Replicate ``nn.TransformerEncoderLayer.forward`` (post-norm
        # branch) using the same submodules. We don't call the original
        # layer's ``forward`` directly because we need to insert
        # IntXAttn between SelfAttn and FFN, not before/after the whole
        # block.
        attn_out, _ = layer.self_attn(
            src, src, src,
            attn_mask=None,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
        )
        src = layer.norm1(src + layer.dropout1(attn_out))

        # ---- NEW: IntXAttn pre-norm + γ-gated residual ----
        # Skip entirely when no interaction tokens are provided
        # (matches the design's "interaction-dropped" branches in
        # the compositional CFG arithmetic). At γ_int=0 with int_kv
        # supplied, the contribution is also exactly zero — both
        # paths are byte-identical at init.
        if int_kv is not None:
            q = self.norm_int(src)
            int_out, _ = self.int_attn(
                query=q,
                key=int_kv,
                value=int_kv,
                key_padding_mask=int_key_padding_mask,
                need_weights=False,
            )
            src = src + self._apply_gamma(self.dropout_int(int_out))

        # ---- MoMask original: FFN + post-norm + residual ----
        ff_out = layer.linear2(
            layer.dropout(layer.activation(layer.linear1(src)))
        )
        src = layer.norm2(src + layer.dropout2(ff_out))
        return src


class MaskTransformerEncoderWithInteraction(nn.Module):
    """Drop-in replacement for ``nn.TransformerEncoder`` carrying interaction K/V.

    Wraps every layer in the original encoder with
    :class:`MaskTransformerBlockWithInteraction`, preserving the original
    weights and the optional final ``norm`` (typically None for MoMask,
    which uses post-norm inside each layer).
    """

    def __init__(
        self,
        original_encoder: nn.TransformerEncoder,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        zero_init_gamma: bool = True,
        gamma_kind: str = "scalar",
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            MaskTransformerBlockWithInteraction(
                layer, d_model=d_model, num_heads=num_heads,
                dropout=dropout, zero_init_gamma=zero_init_gamma,
                gamma_kind=gamma_kind,
            )
            for layer in original_encoder.layers
        ])
        # MoMask's ``nn.TransformerEncoder`` has ``norm = None`` (post-
        # norm is inside each layer), but keep this for API parity
        # with PyTorch's stock encoder.
        self.norm = original_encoder.norm

    def forward(
        self,
        src: Tensor,
        int_kv: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        int_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        h = src
        for blk in self.layers:
            h = blk(
                h, int_kv,
                src_key_padding_mask=src_key_padding_mask,
                int_key_padding_mask=int_key_padding_mask,
            )
        if self.norm is not None:
            h = self.norm(h)
        return h


# ============================================================================
# CFG drop bucketing helper (training time)
# ============================================================================

def sample_cfg_buckets(
    batch_size: int,
    p_drop_both: float = 0.10,
    p_drop_int_only: float = 0.10,
    p_drop_text_only: float = 0.05,
    *,
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Sample per-sample drop masks from the design §2.2 4-bucket categorical.

    Mutually exclusive buckets:
      - p_drop_both         → drop_text=True,  drop_int=True
      - p_drop_int_only     → drop_text=False, drop_int=True
      - p_drop_text_only    → drop_text=True,  drop_int=False
      - 1 − sum(above)      → drop_text=False, drop_int=False (keep both)

    Default probabilities (10/10/5/75) match the design — text-only
    is over-sampled vs. interaction-only because text-only is the
    "safer fallback" path the model already trained on, and the
    asymmetry encourages the new IntXAttn weights to become useful
    rather than be merely additive. The 5% text-only-drop is the
    minimum needed to make the ``logits_int_only`` branch in
    compositional CFG well-defined.

    Returns
    -------
    drop_text_mask : (B,) bool — True ⇒ replace text cond with zeros
    drop_int_mask  : (B,) bool — True ⇒ replace int K/V with null_int_kv
    """
    total = p_drop_both + p_drop_int_only + p_drop_text_only
    if total > 1.0 + 1e-6:
        raise ValueError(
            f"CFG bucket probabilities sum to {total:.4f} > 1.0",
        )
    u = torch.rand(batch_size, device=device, generator=generator)
    in_drop_both = u < p_drop_both
    in_drop_int_only = (u >= p_drop_both) & (u < p_drop_both + p_drop_int_only)
    in_drop_text_only = (
        (u >= p_drop_both + p_drop_int_only)
        & (u < p_drop_both + p_drop_int_only + p_drop_text_only)
    )
    drop_text_mask = in_drop_both | in_drop_text_only
    drop_int_mask = in_drop_both | in_drop_int_only
    return drop_text_mask, drop_int_mask


# ============================================================================
# Full Stage B model
# ============================================================================

class InteractionMaskTransformer(nn.Module):
    """MoMask MaskTransformer + per-block interaction cross-attention.

    Composition:
        - ``self.mask_transformer`` — the original MoMask MaskTransformer
          loaded via :func:`piano.models.backbones.momask_adapter.load_momask_mask_transformer`.
          Its 8-layer ``seqTransEncoder`` is **replaced in-place** with
          :class:`MaskTransformerEncoderWithInteraction` so every block
          gains an IntXAttn sublayer while keeping its self-attn + FFN
          weights.
        - ``self.interaction_tokenizer`` — projects per-frame z_int
          into the K/V tensor shared across all 8 IntXAttn sublayers.
        - ``self.null_int_kv`` — learnable (S_max, d) null token bank
          used when interaction is dropped (CFG inference + training
          drop). At init it's zeros, but the model can grow it during
          training; per design §6.3, learnable null beats zero K/V
          because zero K/V causes the IntXAttn softmax to be
          undefined when V is all-zero.

    Parameters
    ----------
    mask_transformer
        A pretrained MoMask MaskTransformer (already loaded). Its
        ``seqTransEncoder`` will be patched in this constructor.
    interaction_tokenizer
        Pre-built tokenizer module (must have ``d_model`` matching
        ``mask_transformer.latent_dim``).
    interaction_drop_prob
        Per-sample probability of dropping the interaction K/V at
        training time, used by the simple ``mask_int`` path. Ignored
        when the caller passes explicit per-sample drop masks (the
        recommended path). Kept for backward compat.
    """

    def __init__(
        self,
        mask_transformer: nn.Module,
        interaction_tokenizer: InteractionTokenizer,
        interaction_drop_prob: float = 0.1,
        zero_init_gamma: bool = True,
        max_token_seq_length: int = 49,
        gamma_kind: str = "scalar",
        wrapper_kind: str = "v0.6",
    ) -> None:
        super().__init__()
        self.mask_transformer = mask_transformer
        self.interaction_tokenizer = interaction_tokenizer
        self.interaction_drop_prob = float(interaction_drop_prob)
        self.gamma_kind = str(gamma_kind)
        # ``wrapper_kind`` selects which encoder swap-in to use:
        #   - "v0.6"       : MaskTransformerEncoderWithInteraction (per-block IntXAttn
        #                    on the same layers, backbone fine-tuned). v0.1-v0.7 default.
        #   - "v0.3-delta" : InterControlTransformerEncoder (trainable deepcopy
        #                    of seqTransEncoder + per-layer zero-init Linear
        #                    connectors + frozen main branch). InterControl
        #                    NeurIPS'24 / OmniControl ICLR'24 / MotionLCM
        #                    ECCV'24 recipe.
        self.wrapper_kind = str(wrapper_kind)
        if self.wrapper_kind not in ("v0.6", "v0.3-delta"):
            raise ValueError(
                f"wrapper_kind must be 'v0.6' or 'v0.3-delta', got {self.wrapper_kind!r}",
            )

        # Sanity: tokenizer d_model must match MoMask latent_dim so K/V
        # can cross-attend without an extra projection.
        if interaction_tokenizer.d_model != mask_transformer.latent_dim:
            raise ValueError(
                "InteractionTokenizer.d_model "
                f"({interaction_tokenizer.d_model}) must equal "
                f"MaskTransformer.latent_dim ({mask_transformer.latent_dim})",
            )

        # Patch: replace seqTransEncoder with our interaction-aware
        # version. The new wrapper holds references to the original
        # nn.TransformerEncoderLayer instances, so MoMask's pretrained
        # self-attn + FFN weights are preserved.
        d_model = mask_transformer.latent_dim
        num_heads = mask_transformer.seqTransEncoder.layers[0].self_attn.num_heads
        dropout = mask_transformer.dropout

        if self.wrapper_kind == "v0.6":
            mask_transformer.seqTransEncoder = MaskTransformerEncoderWithInteraction(
                original_encoder=mask_transformer.seqTransEncoder,
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
                zero_init_gamma=zero_init_gamma,
                gamma_kind=self.gamma_kind,
            )
        else:  # "v0.3-delta" — InterControl trainable-copy
            from piano.models.motion_generator_intercontrol import (
                InterControlTransformerEncoder,
            )
            mask_transformer.seqTransEncoder = InterControlTransformerEncoder(
                original_encoder=mask_transformer.seqTransEncoder,
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
                gamma_kind=self.gamma_kind,
            )
            # Freeze main branch (the deepcopy-source frozen layers).
            mask_transformer.seqTransEncoder.freeze_main_branch()
            # Per InterControl §3.2: freeze ALL of MoMask's pretrained
            # components (token_emb, input_process, position_enc,
            # cond_emb, output_process). Only the trainable-copy ctrl
            # branch + zero-init connectors + tokenizer + null_int_kv
            # learn. The seqTransEncoder.main_layers were already frozen
            # by freeze_main_branch above.
            for name, p in mask_transformer.named_parameters():
                if name.startswith("clip_model."):
                    continue   # already frozen by load_and_freeze_clip
                if name.startswith("seqTransEncoder."):
                    continue   # main_layers handled by freeze_main_branch;
                               # ctrl_layers + connectors stay trainable
                p.requires_grad = False

        # Disable MoMask's own internal text-drop. We handle CFG
        # bucketed drops at the wrapper level so per-sample masks for
        # text and interaction stay aligned; if MoMask still ran
        # ``mask_cond`` with cond_drop_prob=0.1, we'd get extra
        # uncoordinated text drops on top of our explicit ones.
        self.mask_transformer.cond_drop_prob = 0.0

        # Learnable null interaction K/V tokens (per design §6.3).
        # Shape (S_max, 1, d_model) — broadcast across batch when used.
        # Init from N(0, 0.02) (same as MoMask's __init_weights) — the
        # model can learn what "no interaction" means in K/V space.
        self.null_int_kv = nn.Parameter(
            torch.empty(max_token_seq_length, 1, d_model).normal_(0.0, 0.02),
        )

        # Cache the model's max token seq length so we can slice
        # null_int_kv for shorter sequences without recomputing.
        self.max_token_seq_length = max_token_seq_length

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        transformer_checkpoint: str | Path,
        interaction_tokenizer: InteractionTokenizer | None = None,
        interaction_drop_prob: float = 0.1,
        zero_init_gamma: bool = True,
        device: str | torch.device = "cpu",
        max_token_seq_length: int = 49,
        gamma_kind: str = "scalar",
        wrapper_kind: str = "v0.6",
        **mask_transformer_kwargs: Any,
    ) -> "InteractionMaskTransformer":
        """Load pretrained MoMask MaskTransformer and wrap with interaction layers.

        Forwards ``mask_transformer_kwargs`` to
        :func:`piano.models.backbones.momask_adapter.load_momask_mask_transformer`
        — useful if the checkpoint comes from a non-default config
        (different latent_dim, num_layers, etc.).
        """
        from piano.models.backbones.momask_adapter import load_momask_mask_transformer

        mask_transformer = load_momask_mask_transformer(
            transformer_checkpoint, device=device, **mask_transformer_kwargs,
        )
        if interaction_tokenizer is None:
            interaction_tokenizer = InteractionTokenizer(
                d_model=mask_transformer.latent_dim,
                token_stride=4,
            )
        wrapper = cls(
            mask_transformer=mask_transformer,
            interaction_tokenizer=interaction_tokenizer,
            interaction_drop_prob=interaction_drop_prob,
            zero_init_gamma=zero_init_gamma,
            max_token_seq_length=max_token_seq_length,
            gamma_kind=gamma_kind,
            wrapper_kind=wrapper_kind,
        )
        # Move EVERYTHING (not just the loaded MoMask) to the target
        # device. The newly-created wrapper layers + tokenizer +
        # null_int_kv start on CPU.
        wrapper.to(device)
        return wrapper

    # ------------------------------------------------------------------
    # Forward primitives
    # ------------------------------------------------------------------

    def _broadcast_null_kv(self, batch_size: int, S: int, dtype: torch.dtype, device: torch.device) -> Tensor:
        """Slice and broadcast the learnable null K/V to (S, B, d)."""
        if S > self.null_int_kv.shape[0]:
            raise ValueError(
                f"requested null_int_kv length {S} exceeds "
                f"max_token_seq_length {self.null_int_kv.shape[0]}; "
                f"raise max_token_seq_length when constructing the model",
            )
        null = self.null_int_kv[:S].to(dtype=dtype, device=device)   # (S, 1, d)
        return null.expand(-1, batch_size, -1).contiguous()

    def _build_int_kv(
        self,
        int_tokens_bf: Tensor | None,                  # (B, S_int, d) batch-first or None
        int_padding_mask_bf: Tensor | None,            # (B, S_int) bool or None
        drop_int_mask: Tensor | None,                  # (B,) bool or None
        batch_size: int,
        seq_S: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[Tensor | None, Tensor | None]:
        """Materialise the K/V tensor (and optional padding mask) the
        IntXAttn sublayers consume, applying per-sample CFG drops.

        Returns ``(int_kv_seq_first, int_kv_padding_mask)`` where
        ``int_kv_seq_first`` is ``(S_int, B, d)`` ready to feed into a
        ``nn.MultiheadAttention(batch_first=False)``.

        The three calling conventions from compositional CFG inference:
        1. ``int_tokens_bf=None`` → no interaction at all → returns
           ``(None, None)``. IntXAttn sublayers skip entirely.
        2. ``int_tokens_bf=given``, ``drop_int_mask=None`` → all samples
           use real interaction tokens.
        3. ``int_tokens_bf=given``, ``drop_int_mask=given`` → per-sample
           replacement: samples with True get null_int_kv, others get
           their real tokens.
        """
        if int_tokens_bf is None:
            # Pure text-only or unconditional path. No K/V to attend over;
            # the wrapped block will skip IntXAttn entirely.
            return None, None

        # (B, S_int, d) → (S_int, B, d) for MoMask's seq-first convention.
        int_kv = int_tokens_bf.transpose(0, 1).contiguous()
        S_int = int_kv.shape[0]

        if drop_int_mask is not None and drop_int_mask.any():
            # Replace dropped samples' K/V with the broadcast null bank.
            null_kv = self._broadcast_null_kv(batch_size, S_int, dtype, device)
            # drop_int_mask is (B,). Index along the batch axis (dim=1).
            replace = drop_int_mask.to(device=int_kv.device).view(1, -1, 1)
            int_kv = torch.where(replace, null_kv, int_kv)
            # Padding for the null branch: null tokens are valid (the
            # model learns what they mean), so dropped samples don't
            # need any positions masked. We zero the padding mask for
            # those samples. Otherwise the original mask is preserved.
            if int_padding_mask_bf is not None:
                pad = int_padding_mask_bf.clone()
                pad[drop_int_mask.to(device=pad.device)] = False
                int_padding_mask_out = pad
            else:
                int_padding_mask_out = None
        else:
            int_padding_mask_out = int_padding_mask_bf

        return int_kv, int_padding_mask_out

    def trans_forward(
        self,
        motion_ids: Tensor,                            # (B, S) token indices
        cond_vector: Tensor,                           # (B, clip_dim) raw CLIP (pooled)
        token_padding_mask: Tensor,                    # (B, S) bool, True = padded motion-token
        int_tokens_bf: Tensor | None = None,           # (B, S_int, d) or None
        int_padding_mask_bf: Tensor | None = None,     # (B, S_int) bool or None
        drop_text_mask: Tensor | None = None,          # (B,) bool — True ⇒ zero text vec
        drop_int_mask: Tensor | None = None,           # (B,) bool — True ⇒ null_int_kv
    ) -> Tensor:
        """Interaction-aware version of MoMask's ``trans_forward``.

        Mirrors the MoMask source at
        ``backbones/momask/models/mask_transformer/transformer.py:210-240``,
        with two changes:

        1. The CFG drop is per-sample (a (B,) bool mask) instead of
           MoMask's per-batch Bernoulli, so the wrapper can coordinate
           drops across text and interaction in compositional CFG.
        2. The encoder is our interaction-aware version, fed
           ``int_kv`` + ``int_padding_mask`` from ``_build_int_kv``.

        Returns
        -------
        logits : (B, num_tokens, S) — same shape as MoMask original.
        """
        mt = self.mask_transformer
        B = motion_ids.shape[0]

        # ---- Token embedding + InputProcess ----
        x = mt.token_emb(motion_ids)                         # (B, S, code_dim)
        x = mt.input_process(x)                              # (S, B, latent_dim)

        # ---- Text condition (per-sample drop) ----
        if drop_text_mask is not None and drop_text_mask.any():
            cond_in = cond_vector.clone()
            cond_in[drop_text_mask.to(cond_in.device)] = 0.0
        else:
            cond_in = cond_vector
        cond_emb = mt.cond_emb(cond_in).unsqueeze(0)         # (1, B, d)

        # ---- Positional encoding + prepend cond token ----
        x = mt.position_enc(x)                               # (S, B, d)
        xseq = torch.cat([cond_emb, x], dim=0)               # (S+1, B, d)

        # Padding mask — prepend False for the cond token (always valid).
        full_pad_mask = torch.cat(
            [torch.zeros_like(token_padding_mask[:, 0:1]), token_padding_mask],
            dim=1,
        )                                                    # (B, S+1)

        # ---- Interaction K/V (with per-sample CFG drops) ----
        int_kv, int_pad_mask = self._build_int_kv(
            int_tokens_bf=int_tokens_bf,
            int_padding_mask_bf=int_padding_mask_bf,
            drop_int_mask=drop_int_mask,
            batch_size=B,
            seq_S=x.shape[0],
            dtype=x.dtype,
            device=x.device,
        )

        # ---- Forward through interaction-aware encoder ----
        output = mt.seqTransEncoder(
            xseq,
            int_kv=int_kv,
            src_key_padding_mask=full_pad_mask,
            int_key_padding_mask=int_pad_mask,
        )                                                    # (S+1, B, d)

        # Drop the cond token before the output head (MoMask convention).
        output = output[1:]                                  # (S, B, d)
        logits = mt.output_process(output)                   # (B, num_tokens, S)
        return logits

    def forward(
        self,
        ids: Tensor,                                # (B, S) GT VQ tokens
        cond_vector: Tensor,                        # (B, clip_dim) CLIP pooled
        m_lens_tok: Tensor,                         # (B,) token-space lengths (= seq_len // 4)
        int_tokens_bf: Tensor | None = None,        # (B, S_int, d) or None
        int_padding_mask_bf: Tensor | None = None,  # (B, S_int) or None
        drop_text_mask: Tensor | None = None,       # (B,) bool — explicit text drops
        drop_int_mask: Tensor | None = None,        # (B,) bool — explicit interaction drops
        cfg_drop_buckets: tuple[float, float, float] | None = None,
    ) -> dict[str, Tensor]:
        """Training forward: BERT-style mask + masked-CE loss.

        Mirrors MoMask's ``MaskTransformer.forward`` at
        ``transformer.py:242-304`` exactly (cosine schedule, BERT 88/10/2
        token corruption split, ``cal_performance`` over masked positions
        with ``ignore_index=mask_id``). The only changes:

        - Per-sample CFG drop masks are populated from
          ``cfg_drop_buckets`` (training default: 10/10/5) when explicit
          masks are not provided. ``cfg_drop_buckets=None`` disables
          drops entirely (used by val).
        - Each block's encoder forward consumes interaction K/V via the
          patched encoder.

        Returns
        -------
        Dict with scalar-tensor entries only (so the trainer's per-step
        ``.item()`` accumulation in :func:`piano.training.trainer.run_training_loop`
        works without per-element shape gymnastics):

          - ``loss``: scalar masked-CE loss (only at masked positions)
          - ``acc``: scalar mean accuracy at masked positions

        The full ``pred_id`` tensor (B, S) is NOT returned — it's
        not consumed by the training loop, and including non-scalar
        tensors in this dict would crash the per-step logger.
        """
        # MoMask's own helpers — re-imported here to keep the wrapper
        # self-contained (the tools module path is set up by
        # ``momask_adapter`` at import time).
        from models.mask_transformer.tools import (
            cal_performance,
            cosine_schedule,
            get_mask_subset_prob,
            lengths_to_mask,
            uniform,
        )

        mt = self.mask_transformer
        B, S = ids.shape
        device = ids.device

        # ---- Per-sample padding mask ----
        non_pad_mask = lengths_to_mask(m_lens_tok, S)        # (B, S) — True = valid
        ids = torch.where(non_pad_mask, ids, mt.pad_id)

        # ---- BERT-style random masking ----
        rand_time = uniform((B,), device=device)
        rand_mask_probs = cosine_schedule(rand_time)
        num_token_masked = (S * rand_mask_probs).round().clamp(min=1)
        batch_randperm = torch.rand((B, S), device=device).argsort(dim=-1)
        mask = batch_randperm < num_token_masked.unsqueeze(-1)
        mask &= non_pad_mask
        labels = torch.where(mask, ids, mt.mask_id)          # GT only at masked positions

        # 88/10/2 BERT corruption split at masked positions.
        x_ids = ids.clone()
        mask_rid = get_mask_subset_prob(mask, 0.1)
        rand_id = torch.randint_like(x_ids, high=mt.opt.num_tokens)
        x_ids = torch.where(mask_rid, rand_id, x_ids)
        mask_mid = get_mask_subset_prob(mask & ~mask_rid, 0.88)
        x_ids = torch.where(mask_mid, mt.mask_id, x_ids)

        # ---- CFG drop masks ----
        if drop_text_mask is None and drop_int_mask is None and cfg_drop_buckets is not None:
            p_both, p_int_only, p_text_only = cfg_drop_buckets
            drop_text_mask, drop_int_mask = sample_cfg_buckets(
                B, p_drop_both=p_both, p_drop_int_only=p_int_only,
                p_drop_text_only=p_text_only, device=device,
            )

        # ---- Forward through interaction-aware MaskTransformer ----
        logits = self.trans_forward(
            motion_ids=x_ids,
            cond_vector=cond_vector,
            token_padding_mask=~non_pad_mask,
            int_tokens_bf=int_tokens_bf,
            int_padding_mask_bf=int_padding_mask_bf,
            drop_text_mask=drop_text_mask,
            drop_int_mask=drop_int_mask,
        )                                                    # (B, num_tokens, S)

        ce_loss, _pred_id, acc = cal_performance(
            logits, labels, ignore_index=mt.mask_id,
        )
        return {
            "loss": ce_loss,
            "acc": torch.tensor(acc, device=device),
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def forward_with_cond_scale(
        self,
        motion_ids: Tensor,
        cond_vector: Tensor,
        token_padding_mask: Tensor,
        int_tokens_bf: Tensor | None = None,
        int_padding_mask_bf: Tensor | None = None,
        w_text: float = 4.0,
        w_int: float = 2.0,
    ) -> Tensor:
        """Compositional dual-condition CFG (Liu et al. ECCV'22 eq. 5).

        Three forward passes:
          1. ``logits_uncond``       — text dropped, interaction dropped
          2. ``logits_text_only``    — text given,   interaction dropped (null_int_kv)
          3. ``logits_full``         — text given,   interaction given

        Combined as::

            logits = logits_uncond
                   + w_text · (logits_text_only - logits_uncond)
                   + w_int  · (logits_full       - logits_text_only)

        When ``int_tokens_bf`` is None or ``w_int == 0``, the equation
        collapses to MoMask's standard 2-pass single-condition CFG.

        Parameters
        ----------
        w_text
            Text guidance weight (≈ MoMask's ``cond_scale - 1``).
            Default 4 — slightly higher than MoMask's default 2 because
            text now competes with interaction for control authority.
        w_int
            Interaction guidance weight. Sweep on val per design §2.2;
            default 2 is the SPEC starting point.
        """
        B = motion_ids.shape[0]
        device = motion_ids.device

        # 1. Unconditional: drop everything.
        drop_all = torch.ones(B, dtype=torch.bool, device=device)
        logits_uncond = self.trans_forward(
            motion_ids=motion_ids,
            cond_vector=cond_vector,
            token_padding_mask=token_padding_mask,
            int_tokens_bf=int_tokens_bf,
            int_padding_mask_bf=int_padding_mask_bf,
            drop_text_mask=drop_all,
            drop_int_mask=drop_all if int_tokens_bf is not None else None,
        )

        # No-condition shortcut: no text scale and no interaction.
        if w_text == 0 and (int_tokens_bf is None or w_int == 0):
            return logits_uncond

        # 2. Text-only: text given, interaction dropped (or absent).
        keep_text = torch.zeros(B, dtype=torch.bool, device=device)
        logits_text_only = self.trans_forward(
            motion_ids=motion_ids,
            cond_vector=cond_vector,
            token_padding_mask=token_padding_mask,
            int_tokens_bf=int_tokens_bf,
            int_padding_mask_bf=int_padding_mask_bf,
            drop_text_mask=keep_text,
            drop_int_mask=drop_all if int_tokens_bf is not None else None,
        )

        if int_tokens_bf is None or w_int == 0:
            # Single-condition CFG — same arithmetic as MoMask's
            # ``forward_with_cond_scale`` but with our per-sample
            # bookkeeping. ``w_text=4`` corresponds to MoMask
            # ``cond_scale=5``.
            return logits_uncond + w_text * (logits_text_only - logits_uncond)

        # 3. Full: text given, interaction given.
        logits_full = self.trans_forward(
            motion_ids=motion_ids,
            cond_vector=cond_vector,
            token_padding_mask=token_padding_mask,
            int_tokens_bf=int_tokens_bf,
            int_padding_mask_bf=int_padding_mask_bf,
            drop_text_mask=keep_text,
            drop_int_mask=keep_text,    # all-False ⇒ keep interaction
        )

        return (
            logits_uncond
            + w_text * (logits_text_only - logits_uncond)
            + w_int * (logits_full - logits_text_only)
        )

    @torch.no_grad()
    def generate(
        self,
        cond_vector: Tensor,            # (B, clip_dim)
        m_lens_tok: Tensor,             # (B,) token-space lengths
        int_tokens_bf: Tensor | None = None,
        int_padding_mask_bf: Tensor | None = None,
        timesteps: int = 10,
        w_text: float = 4.0,
        w_int: float = 2.0,
        temperature: float = 1.0,
        topk_filter_thres: float = 0.9,
    ) -> Tensor:
        """Iterative parallel-decoding generation (MoMask MaskGIT-style)
        with compositional dual CFG.

        Mirrors MoMask's ``MaskTransformer.generate`` (10 unmasking
        iterations, cosine remask schedule, gumbel sampling) but
        replaces the inner ``forward_with_cond_scale`` call with the
        3-pass compositional version.

        Returns
        -------
        ids : (B, S) — generated VQ-VAE base-layer token IDs.
                      Padded positions are -1 (matches MoMask convention).
        """
        from models.mask_transformer.tools import (
            cosine_schedule, gumbel_sample, lengths_to_mask, top_k,
        )

        mt = self.mask_transformer
        device = next(self.parameters()).device
        B = cond_vector.shape[0]
        S = int(m_lens_tok.max().item())

        non_pad_mask = lengths_to_mask(m_lens_tok, S)        # (B, S)
        padding_mask = ~non_pad_mask

        # Start fully masked (except padded positions which are pad_id).
        ids = torch.where(padding_mask, mt.pad_id, mt.mask_id)
        scores = torch.where(padding_mask, torch.full_like(ids, 1e5, dtype=torch.float), torch.zeros_like(ids, dtype=torch.float))
        starting_temperature = temperature

        for timestep in torch.linspace(0, 1, timesteps, device=device):
            rand_mask_prob = cosine_schedule(timestep)        # scalar tensor
            num_token_masked = torch.round(rand_mask_prob * m_lens_tok.float()).clamp(min=1)

            # Re-mask the lowest-confidence tokens at this iteration.
            sorted_indices = scores.argsort(dim=1)
            ranks = sorted_indices.argsort(dim=1)
            is_mask = ranks < num_token_masked.unsqueeze(-1)
            ids = torch.where(is_mask, mt.mask_id, ids)

            # 3-pass compositional CFG.
            logits = self.forward_with_cond_scale(
                motion_ids=ids,
                cond_vector=cond_vector,
                token_padding_mask=padding_mask,
                int_tokens_bf=int_tokens_bf,
                int_padding_mask_bf=int_padding_mask_bf,
                w_text=w_text,
                w_int=w_int,
            )
            logits = logits.permute(0, 2, 1)                  # (B, S, V)

            filtered_logits = top_k(logits, topk_filter_thres, dim=-1)

            # Gumbel sample (matches MoMask's default ``gsample=False``
            # would use multinomial; we follow the gumbel branch since
            # it's the cleaner implementation and equivalent in
            # expectation).
            pred_ids = gumbel_sample(
                filtered_logits, temperature=starting_temperature, dim=-1,
            )
            ids = torch.where(is_mask, pred_ids, ids)

            # Update scores: prob assigned to the sampled token.
            probs = logits.softmax(dim=-1)
            tok_scores = probs.gather(2, pred_ids.unsqueeze(-1)).squeeze(-1)
            scores = torch.where(is_mask, tok_scores, scores)
            scores = scores.masked_fill(~is_mask, 1e5)

        ids = torch.where(padding_mask, torch.full_like(ids, -1), ids)
        return ids

    def encode_text(self, raw_text: list[str]) -> Tensor:
        """Pass-through to MoMask's frozen CLIP text encoder."""
        return self.mask_transformer.encode_text(raw_text)

    # ------------------------------------------------------------------
    # Parameter groups (for the trainer's two-LR optimiser)
    # ------------------------------------------------------------------

    def new_parameters(self) -> list[nn.Parameter]:
        """Trained-from-scratch params: tokenizer + IntXAttn + γ + null_int_kv
        (+ control branch + zero-init connectors when wrapper_kind=v0.3-delta).

        These need a higher LR than the MoMask backbone finetune. Per
        ControlNet ICCV'23 + LLaMA-Adapter convention, "new" weights
        train at full LR while pretrained weights move at a much lower
        rate. The trainer routes these into a separate AdamW group.

        Layout per wrapper_kind:
          - "v0.6"       : tokenizer + null_int_kv + per-block IntXAttn
                           (the new sublayers added on top of frozen
                           ``blk.layer`` references inside
                           ``MaskTransformerEncoderWithInteraction``).
          - "v0.3-delta" : tokenizer + null_int_kv + ctrl_layers
                           (deepcopy of seqTransEncoder layers wrapped
                           with IntXAttn) + per-layer zero-init
                           connectors. Main layers (frozen at
                           pretrained MoMask weights) NOT included.
        """
        params: list[nn.Parameter] = []
        # Common: tokenizer + null_int_kv.
        params.extend(self.interaction_tokenizer.parameters())
        params.append(self.null_int_kv)

        encoder = self.mask_transformer.seqTransEncoder
        if self.wrapper_kind == "v0.6":
            # Per-block IntXAttn weights + γ + norm_int (anything that
            # lives under MaskTransformerEncoderWithInteraction and
            # ISN'T part of the original MoMask
            # ``nn.TransformerEncoderLayer`` referenced by ``blk.layer``).
            for blk in encoder.layers:
                for name, p in blk.named_parameters():
                    if name.startswith("layer."):
                        continue
                    params.append(p)
        else:  # "v0.3-delta" — InterControlTransformerEncoder
            # Trainable-copy ctrl branch (deepcopy + IntXAttn wrap)
            # + per-layer zero-init Linear connectors. main_layers and
            # main_norm are frozen and excluded.
            params.extend(encoder.ctrl_layers.parameters())
            params.extend(encoder.connectors.parameters())
        return params

    def backbone_parameters(self) -> list[nn.Parameter]:
        """Pretrained MoMask params (excluding CLIP, which stays frozen).

        Layout per wrapper_kind:
          - "v0.6": token_emb + input_process + position_enc +
                    output_process + cond_emb + each ``blk.layer``
                    (original nn.TransformerEncoderLayer with self-attn
                    + FFN weights). All trained at backbone_lr.
          - "v0.3-delta": empty list. Per InterControl §3.2, the main
                          branch (including embeddings + projections)
                          stays frozen at pretrained MoMask weights.
                          Optimiser uses one effective LR group (new_lr).

        Always excludes:
          - clip_model.* (frozen by MoMask's ``load_and_freeze_clip``)
          - Anything counted by ``new_parameters`` (no double-count)
        """
        if self.wrapper_kind == "v0.3-delta":
            # Main branch frozen by InterControl convention; no
            # backbone-LR group.
            return []
        new_param_ids = {id(p) for p in self.new_parameters()}
        backbone: list[nn.Parameter] = []
        for name, p in self.mask_transformer.named_parameters():
            if name.startswith("clip_model."):
                continue
            if id(p) in new_param_ids:
                continue
            backbone.append(p)
        return backbone
