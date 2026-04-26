"""Interaction tokenizer: per-frame z_int (T, 27) → token-level K/V (S, d).

Stage B encodes the predictor's per-frame interaction latent into the
shared K/V tensor that every IntXAttn sublayer in
:class:`InteractionMaskTransformer` cross-attends to. Two operations:

1. **Project** the per-frame structured 4-tuple
   ``{contact_state(5), contact_target_xyz(5,3), phase(3), support(4)}``
   (concatenated to (T, 27)) into the model's hidden dimension via a
   2-layer MLP (Linear-GELU-Linear).
2. **Temporally downsample** with a 1D conv of kernel=stride=4 to align
   with MoMask's VQ-VAE token sequence (T=196 frames → S=49 tokens),
   matching the unit_length=4 quantizer (verified from
   ``backbones/momask/options/base_option.py``: ``--unit_length``).

Design rationale (analyses/2026-04-26_stageB_design.md §3):

- 1D conv stride-4 over a Perceiver-style learned-query bottleneck:
  per-frame z_int is small (27 channels) and already aligned with motion
  frames; a learnable bottleneck adds parameters without solving any
  problem.
- Pooling lossy: closest-point GT was chosen precisely because *which
  frame* matters; pooling 4 frames flattens that.
- ``d_model=384`` to match MoMask's ``latent_dim=384`` (verified default
  in ``options/base_option.py``).
- K/V is shared across all 8 IntXAttn sublayers (text-encoded-once
  pattern; MoMask follows the same convention for its cond token).

The tokenizer also produces a per-sample **token-space padding mask**
``(B, S)`` so the IntXAttn sublayers can mask out padded interaction
tokens with the same convention MoMask uses for its motion sequence.

Citations
---------
- Guo et al. *MoMask: Generative Masked Modeling of 3D Human Motions.*
  CVPR 2024 Highlight. arXiv:2312.00063. ``unit_length=4`` quantizer.
- Diller & Dai. *CG-HOI: Contact-Guided 3D Human-Object Interaction
  Generation.* CVPR 2024. arXiv:2311.16097. Per-frame contact-marker
  embedding precedent.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# Per-frame z_int channel layout (must match
# ``InteractionPredictor`` outputs and ``HOIDataset`` pseudo-label fields).
# Numbers are post-v5/v11 (3-class phase, xyz regression target).
DEFAULT_NUM_BODY_PARTS = 5      # L_hand / R_hand / L_foot / R_foot / pelvis
DEFAULT_TARGET_COORD_DIM = 3    # xyz in object-local frame
DEFAULT_NUM_PHASES = 3          # non_contact / stable_contact / manipulation
DEFAULT_NUM_SUPPORT = 4         # both_feet / single_foot / sitting / hand_support
DEFAULT_TOKEN_STRIDE = 4        # MoMask VQ-VAE temporal downsample factor
# v0.2 (analyses/2026-04-27_object_conditioning_review.md): per-frame
# object pose channels added to z_int. Position 3-d, 6D rotation 6-d
# (Zhou et al. CVPR'19, arXiv:1812.07035), in body-canonical frame —
# same frame as contact_target_xyz, so the IntXAttn K/V is internally
# frame-consistent. Disable via ``num_obj_pose_channels=0`` to recover
# the v0.1 27-d behaviour.
DEFAULT_NUM_OBJ_POSE_CHANNELS = 9   # 3 (com) + 6 (rot6d)


def z_int_input_dim(
    num_body_parts: int = DEFAULT_NUM_BODY_PARTS,
    target_coord_dim: int = DEFAULT_TARGET_COORD_DIM,
    num_phases: int = DEFAULT_NUM_PHASES,
    num_support: int = DEFAULT_NUM_SUPPORT,
    num_obj_pose_channels: int = DEFAULT_NUM_OBJ_POSE_CHANNELS,
) -> int:
    """Total per-frame z_int width after concat.

    With v11 + v0.2 defaults: 5 + 5×3 + 3 + 4 + 9 = 36.
    Pass ``num_obj_pose_channels=0`` to recover v0.1 27-d behaviour.
    """
    return (
        num_body_parts                          # contact_state
        + num_body_parts * target_coord_dim     # contact_target_xyz flattened
        + num_phases                            # phase one-hot
        + num_support                           # support one-hot
        + num_obj_pose_channels                 # v0.2: obj_com (3) + obj_rot6d (6)
    )


class InteractionTokenizer(nn.Module):
    """Encode per-frame z_int into MoMask-token-space K/V.

    Parameters
    ----------
    d_model
        Hidden dimension. Must equal MoMask's ``latent_dim`` (384) so
        K/V can be cross-attended directly.
    num_body_parts, target_coord_dim, num_phases, num_support
        Channel counts of the four z_int components — see module
        docstring for v11 defaults.
    token_stride
        Temporal downsample factor (kernel size = stride; non-overlapping
        windows). Must equal MoMask's ``unit_length`` (4) so the output
        sequence length S matches the motion-token sequence length.
    mlp_hidden
        Hidden width of the per-frame projection MLP. Default
        ``d_model`` keeps the param count light; can be widened if
        Stage C ablation finds the projection underfits.
    dropout
        Dropout inside the MLP (applied between the two Linear layers).
        0.0 by default — z_int is a low-noise discrete signal so we
        don't need dropout regularisation here.
    """

    def __init__(
        self,
        d_model: int = 384,
        num_body_parts: int = DEFAULT_NUM_BODY_PARTS,
        target_coord_dim: int = DEFAULT_TARGET_COORD_DIM,
        num_phases: int = DEFAULT_NUM_PHASES,
        num_support: int = DEFAULT_NUM_SUPPORT,
        token_stride: int = DEFAULT_TOKEN_STRIDE,
        mlp_hidden: int | None = None,
        dropout: float = 0.0,
        max_seq_length: int = 196,
        num_obj_pose_channels: int = DEFAULT_NUM_OBJ_POSE_CHANNELS,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_body_parts = num_body_parts
        self.target_coord_dim = target_coord_dim
        self.num_phases = num_phases
        self.num_support = num_support
        self.token_stride = token_stride
        self.num_obj_pose_channels = num_obj_pose_channels

        in_dim = z_int_input_dim(
            num_body_parts, target_coord_dim, num_phases, num_support,
            num_obj_pose_channels,
        )
        hidden = mlp_hidden if mlp_hidden is not None else d_model

        # Per-frame projection: (B, T, in_dim) → (B, T, d_model).
        self.project = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )

        # Temporal downsample: kernel=stride, non-overlapping. Maps
        # (B, T, d) → (B, T/stride, d). Matches MoMask's VQ encoder
        # which is also stride-4 (down_t=2, stride_t=2 → 2² = 4).
        self.temporal_conv = nn.Conv1d(
            d_model, d_model,
            kernel_size=token_stride,
            stride=token_stride,
            padding=0,
        )

        # Token-space sinusoidal positional encoding. Buffer (not
        # learnable) so its weight count doesn't churn when the
        # downstream attention learns to attend to specific positions
        # via the gradient flowing into the conv kernel above.
        s_max = max_seq_length // token_stride
        self.register_buffer(
            "pos_encoding",
            self._sinusoidal_encoding(s_max, d_model),
            persistent=False,
        )

        self._init_weights()

    @staticmethod
    def _sinusoidal_encoding(length: int, d_model: int) -> Tensor:
        """Standard sinusoidal positional encoding (Vaswani NeurIPS'17)."""
        import math

        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def _init_weights(self) -> None:
        """ViT/T5/MoMask convention: Normal(0, 0.02) on Linear/Conv,
        zero bias, ones LayerNorm. Matches MoMask's ``__init_weights``
        so the new tokenizer parameters mix coherently with the
        preserved transformer weights at init."""
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)

    def forward(
        self,
        contact_state: Tensor,             # (B, T, num_body_parts)
        contact_target_xyz: Tensor,        # (B, T, num_body_parts, 3)
        phase: Tensor,                     # (B, T) int OR (B, T, num_phases) one-hot
        support: Tensor,                   # (B, T) int OR (B, T, num_support) one-hot
        obj_com_canonical: Tensor | None = None,    # (B, T, 3)  body-canonical
        obj_rot6d_canonical: Tensor | None = None,  # (B, T, 6)  body-canonical
        seq_lens: Tensor | None = None,    # (B,) frame-space lengths
    ) -> tuple[Tensor, Tensor | None]:
        """Encode z_int into K/V tokens.

        Parameters
        ----------
        contact_state : (B, T, B_parts) float — soft contact probabilities
        contact_target_xyz : (B, T, B_parts, 3) float — closest-surface-point
            xyz in object-local frame (the v10/v11 GT field).
        phase : either int (B, T) or one-hot (B, T, num_phases)
        support : either int (B, T) or one-hot (B, T, num_support)
        obj_com_canonical : (B, T, 3) — per-frame object COM in BODY-CANONICAL
            frame (v0.2). Required when ``num_obj_pose_channels > 0``;
            ignored otherwise. Caller must apply
            :func:`piano.utils.canonical_frame.world_to_canonical_object_pose`
            (or its torch counterpart) to lift world-frame InterAct
            object_positions into the body's canonical frame BEFORE
            calling this.
        obj_rot6d_canonical : (B, T, 6) — per-frame 6D rotation
            (Zhou et al. CVPR'19) in body-canonical frame.
        seq_lens : optional (B,) frame-space sequence lengths. When
            provided, the returned padding mask covers token-space
            (lengths // token_stride). When None, no mask is returned
            (caller should treat all S positions as valid).

        Returns
        -------
        kv : (B, S, d_model) — interaction K/V (sequence-batch order
            matches what MoMask uses internally; we provide
            batch-first here and swap inside the wrapper as needed).
        kv_padding_mask : (B, S) bool or None — True for padded
            positions (matches PyTorch ``key_padding_mask`` convention).
        """
        if contact_target_xyz.dim() != 4:
            raise ValueError(
                "contact_target_xyz must be (B, T, num_body_parts, 3); got "
                f"shape {tuple(contact_target_xyz.shape)}",
            )
        B, T, P, _ = contact_target_xyz.shape

        # Promote phase / support to one-hot if they came in as ints.
        # The dataloader ships them as int64 (HOIDataset preserves the
        # extractor's dtype) — the tokenizer one-hots inside so callers
        # don't have to remember.
        phase_oh = self._as_one_hot(phase, self.num_phases)
        support_oh = self._as_one_hot(support, self.num_support)

        # Flatten contact_target_xyz to (B, T, P*3) so concat works.
        ctx_flat = contact_target_xyz.reshape(B, T, P * self.target_coord_dim)

        components: list[Tensor] = [contact_state, ctx_flat, phase_oh, support_oh]

        # v0.2: append per-frame object pose channels (canonical frame).
        # Required when the tokenizer was constructed with
        # ``num_obj_pose_channels > 0``. We keep it strict (no silent
        # zero fill-in) so that misconfiguration is caught early
        # rather than producing a model that silently ignores object
        # pose at training time.
        if self.num_obj_pose_channels > 0:
            if obj_com_canonical is None or obj_rot6d_canonical is None:
                raise ValueError(
                    "tokenizer was built with num_obj_pose_channels="
                    f"{self.num_obj_pose_channels}; obj_com_canonical and "
                    "obj_rot6d_canonical must both be provided.",
                )
            if obj_com_canonical.shape != (B, T, 3):
                raise ValueError(
                    f"obj_com_canonical must be (B, T, 3); got "
                    f"{tuple(obj_com_canonical.shape)}",
                )
            if obj_rot6d_canonical.shape != (B, T, 6):
                raise ValueError(
                    f"obj_rot6d_canonical must be (B, T, 6); got "
                    f"{tuple(obj_rot6d_canonical.shape)}",
                )
            components.append(obj_com_canonical.to(contact_state.dtype))
            components.append(obj_rot6d_canonical.to(contact_state.dtype))

        # Concatenate all components along the feature axis.
        z = torch.cat(components, dim=-1)
        # Sanity check: concatenated width matches z_int_input_dim().
        # Guards against silent shape-mismatches when callers pass
        # off-spec channel counts.
        expected = z_int_input_dim(
            self.num_body_parts, self.target_coord_dim,
            self.num_phases, self.num_support,
            self.num_obj_pose_channels,
        )
        if z.shape[-1] != expected:
            raise ValueError(
                f"z_int concat width mismatch: got {z.shape[-1]}, "
                f"expected {expected}",
            )

        # Per-frame projection.
        z = self.project(z)                          # (B, T, d)

        # Temporal downsample. Conv1d expects (B, C, L) so transpose.
        # Truncate T to a multiple of token_stride to keep the conv
        # output deterministic — MoMask's preprocess already pads to
        # exact 196 frames so this should be a no-op in practice.
        T_aligned = (T // self.token_stride) * self.token_stride
        z = z[:, :T_aligned, :]
        z = z.transpose(1, 2)                        # (B, d, T_aligned)
        z = self.temporal_conv(z)                    # (B, d, S)
        z = z.transpose(1, 2)                        # (B, S, d)

        # Add token-space sinusoidal positional encoding.
        S = z.shape[1]
        z = z + self.pos_encoding[:S, :].unsqueeze(0).to(z.dtype)

        # Build token-space padding mask (PyTorch convention: True =
        # padded). Both the conv stride and the predictor's pad-to-T
        # behaviour mean a valid frame at index t produces a valid
        # token at index t // token_stride.
        kv_padding_mask = None
        if seq_lens is not None:
            # Token-space length = ceil(seq_len / stride) — but we
            # truncated to multiples of stride above, so floor matches.
            token_lens = (seq_lens // self.token_stride).clamp(min=0, max=S)
            arange = torch.arange(S, device=z.device).unsqueeze(0).expand(B, S)
            kv_padding_mask = arange >= token_lens.unsqueeze(1)

        return z, kv_padding_mask

    @staticmethod
    def _as_one_hot(labels: Tensor, num_classes: int) -> Tensor:
        """Promote int labels to float one-hot. Pass-through if already
        floating-point with the right channel dim — keeps the tokenizer
        composable with predictor outputs (which emit softmax probs of
        shape (B, T, K))."""
        if labels.dtype.is_floating_point:
            if labels.dim() == 3 and labels.shape[-1] == num_classes:
                return labels
            raise ValueError(
                f"float-typed phase/support must be (B, T, {num_classes}); "
                f"got shape {tuple(labels.shape)}",
            )
        if labels.dim() != 2:
            raise ValueError(
                f"int-typed phase/support must be (B, T); got shape "
                f"{tuple(labels.shape)}",
            )
        return F.one_hot(labels.long(), num_classes=num_classes).float()
