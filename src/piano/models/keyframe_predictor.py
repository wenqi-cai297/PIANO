"""Stage 1 of AnchorDiff v8: deterministic keyframe predictor.

Input: per-frame conditioning (text features, object trajectory, z_int)
       + keyframe frame indices (where the keyframes should land).
Output: 6-keyjoint world XYZ per keyframe slot.

Architecture: small transformer encoder over the T-frame conditioning
sequence, then ``index_select`` at the keyframe indices to extract
per-keyframe features, then an MLP head to (6 joints × 3 = 18-D) per
slot. Padded slots are masked out in the loss.

The 6 keyjoints (per ``keyframe_extraction.KEYJOINT_INDICES``):
    root (0), L_hand (20), R_hand (21), L_foot (10), R_foot (11), head (15)

Key design choices:
- **Deterministic** (not diffusion): keyframes are sparse (K ≤ 12) and
  the conditioning is already strong (Stage A v10 z_int), so a regression
  model is sufficient. Diffusion would add ~5x training cost without
  clear benefit. v8 design §2 + §6 details the trade-off.
- **No CLS token / no pooling**: the per-frame encoder is reused as a
  feature extractor; we ``gather`` features at the K_MAX keyframe indices.
- **Variable K via mask**: outputs always (K_MAX, 6, 3); ``keyframe_mask``
  zeros out padding slots. K_MAX=12 from v8 design.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass(slots=True)
class KeyframePredictorConfig:
    object_traj_dim: int = 24
    z_int_dim: int = 26
    text_dim: int = 512
    init_pose_dim: int = 66
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    ff_mult: int = 4
    dropout: float = 0.1
    num_keyjoints: int = 6
    k_max: int = 12


class KeyframePredictor(nn.Module):
    """Per-frame transformer encoder + per-keyframe MLP head.

    Input
    -----
    obj_traj : (B, T, 24)
    z_int    : (B, T, 26)
    init_pose : (B, 66)            — flattened SMPL-22 frame-0 joints
    text     : (B, L, 512)         — CLIP per-token features
    keyframe_indices : (B, K_MAX)  — int64, frame index for each slot;
                                     padding slot uses 0 (will be masked)
    keyframe_mask    : (B, K_MAX)  — bool/float, 1 for valid slot

    Output
    ------
    keyframe_pred : (B, K_MAX, num_keyjoints, 3) — world XYZ per slot
    """

    def __init__(self, cfg: KeyframePredictorConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        self.obj_proj = nn.Linear(cfg.object_traj_dim, d)
        self.z_proj = nn.Linear(cfg.z_int_dim, d)
        self.init_pose_proj = nn.Linear(cfg.init_pose_dim, d)
        self.text_proj = nn.Linear(cfg.text_dim, d)
        # Positional embedding (learned, simple).
        # Max length 256 (motion clips are <= 196 in our data).
        self.pos_embed = nn.Embedding(256, d)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg.n_heads,
            dim_feedforward=int(d * cfg.ff_mult),
            dropout=cfg.dropout,
            activation="gelu",
            norm_first=True,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)

        # Per-keyframe MLP head: (d) → (num_keyjoints * 3)
        self.head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, cfg.num_keyjoints * 3),
        )

    def forward(
        self,
        obj_traj: Tensor,             # (B, T, 24)
        z_int: Tensor,                # (B, T, 26)
        init_pose: Tensor,            # (B, 66)
        text: Tensor,                 # (B, L, 512)
        keyframe_indices: Tensor,     # (B, K_MAX) int64
        seq_mask: Tensor | None = None,  # (B, T) optional, 1 valid
    ) -> Tensor:
        B, T, _ = obj_traj.shape
        K = keyframe_indices.shape[1]

        # Project per-frame conditions.
        h_obj = self.obj_proj(obj_traj)                       # (B, T, d)
        h_z = self.z_proj(z_int)                              # (B, T, d)
        h_init = self.init_pose_proj(init_pose).unsqueeze(1)  # (B, 1, d)
        # Sum (instead of cat) keeps d_model unchanged; matches MDM
        # conditioning pattern.
        h = h_obj + h_z                                       # (B, T, d)

        # Add positional embedding.
        pos_ids = torch.arange(T, device=obj_traj.device)
        pos_e = self.pos_embed(pos_ids).unsqueeze(0)          # (1, T, d)
        h = h + pos_e

        # Prepend text features (project + concat): treats text as L
        # extra "tokens" before the T motion frames. The encoder's
        # self-attention then mixes text and motion freely.
        h_text = self.text_proj(text)                         # (B, L, d)
        # Init-pose token gets added to the first motion frame (no
        # position offset needed; it fuses into frame-0).
        h[:, 0, :] = h[:, 0, :] + h_init.squeeze(1)
        seq = torch.cat([h_text, h], dim=1)                   # (B, L+T, d)

        # Build attention mask if seq_mask given.
        # nn.TransformerEncoder uses src_key_padding_mask: True = ignore.
        attn_pad: Tensor | None = None
        if seq_mask is not None:
            text_len = h_text.shape[1]
            text_pad = torch.zeros(B, text_len, dtype=torch.bool, device=seq.device)
            motion_pad = ~seq_mask.bool()                     # True where invalid
            attn_pad = torch.cat([text_pad, motion_pad], dim=1)

        out = self.encoder(seq, src_key_padding_mask=attn_pad)  # (B, L+T, d)

        # Discard text part, keep motion features.
        text_len = h_text.shape[1]
        motion_feats = out[:, text_len:, :]                   # (B, T, d)

        # Gather keyframe features by index. Pad slots use idx=0; the
        # caller is responsible for masking out the loss on padding.
        idx = keyframe_indices.clamp_min(0).unsqueeze(-1).expand(-1, -1, motion_feats.shape[-1])
        kf_feats = torch.gather(motion_feats, dim=1, index=idx)  # (B, K_MAX, d)

        # Per-keyframe MLP head → world XYZ per joint.
        out_flat = self.head(kf_feats)                        # (B, K_MAX, 6*3)
        return out_flat.reshape(B, K, self.cfg.num_keyjoints, 3)


def keyframe_l2_loss(
    pred: Tensor,                   # (B, K_MAX, 6, 3)
    target: Tensor,                 # (B, K_MAX, 6, 3)
    keyframe_mask: Tensor,          # (B, K_MAX), 1 valid 0 pad
) -> Tensor:
    """Masked L2 (squared) on keyframe positions."""
    err = (pred - target).pow(2).sum(dim=-1)                  # (B, K_MAX, 6)
    mask3 = keyframe_mask.unsqueeze(-1).float()               # (B, K_MAX, 1)
    denom = (keyframe_mask.float().sum() * pred.shape[2]).clamp_min(1.0)
    return (err * mask3).sum() / denom
