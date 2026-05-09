"""Adaptive post-processing for v18 pseudo-label contact_state.

Two rules applied to make contact_state robust against incidental /
flickering contacts in passive scenarios (sitting on sofa with hands
optionally on armrest, etc.):

Rule C — sitting + pelvis-stable suppression (zero-out):
    When support==sitting AND pelvis has stable contact, hand contact
    is forced to 0 regardless of pseudo-label value. Rationale:
    physically the hands need not be on the seat during sitting; the
    pseudo label captures incidental armrest contact that flickers
    as the hands move freely.

Rule A — keyframe-side stability gate (NOT applied here; applied in
    keyframe_extraction.select_keyframes via uniform_filter1d on the
    final contact_state). This module only handles rule C.

These are applied at TWO points to keep training/inference consistent:
- Offline: keyframe_extraction.extract_for_subset before keyframe selection
- Online: HOIDataset.__getitem__ after loading pseudo labels

Both call ``suppress_sitting_hand_contact`` so the same rule fires.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d


# Support label mapping (v8: sitting=2 after support_collapse_hand_support).
# See extract_support.py for the canonical schema.
SUPPORT_SITTING_ID: int = 2

# Body part indices in contact_state[:, :] order
# (extract_contact.BODY_PART_NAMES = ["left_hand", "right_hand",
#  "left_foot", "right_foot", "pelvis"])
BP_LEFT_HAND: int = 0
BP_RIGHT_HAND: int = 1
BP_PELVIS: int = 4

# Pelvis-stable detection: rolling-mean window length and threshold.
# 15 frames @ 30fps = 0.5s. Threshold 0.7 means at least 70% of the
# window must be in contact.
PELVIS_STABILITY_WINDOW: int = 15
PELVIS_STABILITY_THRESHOLD: float = 0.7


def suppress_sitting_hand_contact(
    contact_state: np.ndarray,        # (T, 5) float, in-place safe
    support: np.ndarray | None,       # (T,) int support label
    contact_threshold: float = 0.5,
) -> np.ndarray:
    """Zero out hand contact at frames where support==sitting AND
    pelvis has STABLE contact (per rolling-mean smoothing).

    Returns a new (or in-place modified) array.

    If support is None or unavailable, this is a no-op.
    """
    if support is None or contact_state.shape[0] == 0:
        return contact_state

    T = contact_state.shape[0]

    # Detect stable pelvis contact via rolling mean.
    pelvis_raw = (contact_state[:, BP_PELVIS] >= contact_threshold).astype(np.float32)
    pelvis_smooth = uniform_filter1d(
        pelvis_raw, size=PELVIS_STABILITY_WINDOW, mode="nearest"
    )
    pelvis_stable = pelvis_smooth >= PELVIS_STABILITY_THRESHOLD     # (T,)

    # Sitting frames.
    sitting = (support == SUPPORT_SITTING_ID)                       # (T,)

    # Combined suppression mask: sitting AND pelvis-stable.
    suppress = sitting & pelvis_stable                              # (T,)

    # Zero hand contact (both hands) at these frames.
    if suppress.any():
        contact_state = contact_state.copy()
        contact_state[suppress, BP_LEFT_HAND] = 0.0
        contact_state[suppress, BP_RIGHT_HAND] = 0.0
    return contact_state


def compute_contact_stability_mask_torch(
    contact_state,              # (B, T, 5) torch tensor float
    window: int = 15,
):
    """Per-frame per-bodypart contact stability factor in [0, 1].

    Used by Stage 2 anchor loss to downweight anchor at frames where
    contact_state is ambiguous (e.g. flickering between 0 and 1).

    factor = (|smoothed(contact_state) - 0.5| * 2).clamp(0, 1)
    - 1.0 when smoothed is consistently 0 or consistently 1
    - 0.0 when smoothed ≈ 0.5 (flickering / transient)

    Mirrors numpy logic for offline use; this version operates on torch
    tensors via avg_pool1d for batched temporal smoothing.
    """
    import torch
    import torch.nn.functional as F

    B, T, P = contact_state.shape
    cs = (contact_state >= 0.5).float()                             # (B, T, P)
    pad = window // 2
    # avg_pool1d: input (N, C, L) — treat each (B, P) as a separate
    # 1D series. Reshape to (B*P, 1, T).
    cs_perm = cs.permute(0, 2, 1).reshape(B * P, 1, T)              # (B*P, 1, T)
    cs_smooth = F.avg_pool1d(cs_perm, kernel_size=window, stride=1, padding=pad)
    cs_smooth = cs_smooth[:, :, :T]                                  # crop padding asymmetry
    cs_smooth = cs_smooth.reshape(B, P, T).permute(0, 2, 1)          # (B, T, P)

    # Stability factor.
    factor = ((cs_smooth - 0.5).abs() * 2.0).clamp(0.0, 1.0)         # (B, T, P)
    return factor
