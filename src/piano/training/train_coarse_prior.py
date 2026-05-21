"""Stage-1 Coarse-v1 prior trainer (Round 12).

Object-free / plan-free / contact-free trainer for the Coarse-v1
denoiser. Reads from a pre-built cache at
``cache/stage1_coarse_v1_round12`` (or wherever the config points).
Trains S1-A (bidirectional) and S1-B (block-causal) with identical
hyperparameters except for ``model.denoiser.attention_mode``.

Loss decomposition (per Codex review §5):

- ``L_mse_weighted``: weighted MSE on all 23 normalized dims, per-frame.
  Channel-group weights configurable.
- ``L_state_vel``: finite-difference velocity MSE on STATE-LIKE dims
  only (``[0:3]`` root local trans, ``[9:15]`` pelvis rot6d, ``[15:21]``
  spine3 rot6d, and ``[21:23]`` head/shoulder height).
  Stored-velocity dims (``[3:6]`` root_vel, ``[8]`` yaw_vel) are
  deliberately EXCLUDED from this term to avoid double-counting (else
  it becomes an acceleration loss on those channels — see Codex §5.3).
- Stored velocity channels are emphasised via the per-channel weights
  in ``L_mse_weighted``, not via a separate diff-loss.

This module is a fresh, self-contained trainer. It does NOT reuse
``train_anchordiff.py`` — different data, different loss surface,
different conditioning. Patterns borrowed: Accelerate + OmegaConf,
gradient accumulation, optimizer/scheduler builders.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from piano.data.dataset import _swap_left_right_in_text
from piano.models.coarse_motion_prior import (
    CoarsePriorConfig, CoarsePriorDenoiserConfig, CoarsePriorDiff,
)
from piano.models.motion_anchordiff import DiffusionConfig


# ============================================================================
# Channel layout — sync with Round-10 contract
# ============================================================================

# Coarse-v1 channel groups (per Round-10 contract):
#   [0:3]   root_local_trans (xz + y)               — STATE
#   [3:6]   root_vel (xz + y)                       — STORED VELOCITY
#   [6:8]   facing_yaw (sin, cos)                   — STATE
#   [8]     facing_yaw_velocity                     — STORED VELOCITY
#   [9:15]  pelvis_rot6d                            — STATE
#   [15:21] spine3_rot6d                            — STATE
#   [21:22] head_height                             — STATE
#   [22:23] shoulder_center_height                  — STATE

STATE_LIKE_DIMS = (
    list(range(0, 3))                  # root_local_trans
    + list(range(9, 15))               # pelvis_rot6d
    + list(range(15, 21))              # spine3_rot6d
    + list(range(21, 23))              # head + shoulder heights
)
STORED_VEL_DIMS = list(range(3, 6)) + [8]   # root_vel + yaw_vel

# Cont6d X-reflection sign patterns — two conventions are in use in this repo
# and they have DIFFERENT X-mirror sign patterns. Verified by R' = M R M
# derivation on R_y(π/2):
#
# Convention 1 — ROWS (smpl_kinematics / pytorch3d): cont6d = first two rows
#   of R, row-major flatten = [R00, R01, R02, R10, R11, R12]. Used by
#   motion_135 (pelvis_rot6d + spine3_rot6d in Coarse-v1).
# Convention 3 — CANONICAL_FRAME (first two columns of R, row-major flatten
#   of (3,2)): cont6d = [R00, R01, R10, R11, R20, R21]. Used by
#   build_stage1_coarse_v1_objtraj_root0_world_cache.py (obj_rot6d_world).
#
# DO NOT confuse the two — using the wrong sign pattern silently produces
# a cont6d that decodes to a different rotation than M R M.
_ROT6D_ROWS_MIRROR_SIGNS = np.asarray([1.0, -1.0, -1.0, -1.0, 1.0, 1.0], dtype=np.float32)
_ROT6D_CANONICAL_FRAME_MIRROR_SIGNS = np.asarray([1.0, -1.0, -1.0, 1.0, -1.0, 1.0], dtype=np.float32)
# Back-compat alias: pre-existing callers expecting `_ROT6D_MIRROR_SIGNS`
# get the smpl_kinematics ROWS pattern (the one used by Coarse-v1).
_ROT6D_MIRROR_SIGNS = _ROT6D_ROWS_MIRROR_SIGNS


def mirror_coarse_v1(coarse: np.ndarray) -> np.ndarray:
    """Mirror a Coarse-v1 sequence through world X=0.

    Coarse-v1 stores root translation as (x, z, y); pelvis/spine3 rot6d
    live at dims [9:15] / [15:21] in the smpl_kinematics ROWS cont6d
    convention (the Round-12 `_facing_yaw_from_pelvis_rot6d` fix relies
    on this — see `extract_coarse_motion_representation.py`). Reflection
    uses R' = M R M ⇒ cont6d sign pattern [+, -, -, -, +, +] for the ROWS
    convention.
    """
    if coarse.shape[-1] != 23:
        raise ValueError(f"mirror_coarse_v1 expects last dim 23, got {coarse.shape[-1]}")
    out = np.asarray(coarse, dtype=np.float32).copy()
    out[..., 0] *= -1.0   # root_local_trans_x
    out[..., 3] *= -1.0   # root_vel_x
    out[..., 6] *= -1.0   # yaw_sin, since yaw -> -yaw under X reflection
    out[..., 8] *= -1.0   # yaw velocity
    out[..., 9:15] *= _ROT6D_ROWS_MIRROR_SIGNS
    out[..., 15:21] *= _ROT6D_ROWS_MIRROR_SIGNS
    return out


def mirror_obj_traj_root0_world(obj_traj: np.ndarray) -> np.ndarray:
    """Mirror obj_traj_root0_world = [obj_pos_xyz, obj_rot6d_canonical_frame]
    through world X=0.

    obj_rot6d at dims [3:9] is stored via
    ``piano.utils.canonical_frame.matrix_to_rotation_6d_np`` which uses
    the COLUMN-extracted, row-major-flattened layout
    ``[R00, R01, R10, R11, R20, R21]`` — DIFFERENT from the smpl_kinematics
    ROWS convention used by Coarse-v1's pelvis/spine3 rot6d. Under R' = M R M
    the canonical_frame convention's X-mirror sign pattern is
    ``[+, -, -, +, -, +]`` — verified by direct M R M derivation on
    R_y(π/2). Using the ROWS pattern here would produce wrong values on
    dims 3, 4 (the bug Codex's Round-20 implementation originally had).
    """
    if obj_traj.shape[-1] != 9:
        raise ValueError(
            f"mirror_obj_traj_root0_world expects last dim 9, got {obj_traj.shape[-1]}",
        )
    out = np.asarray(obj_traj, dtype=np.float32).copy()
    out[..., 0] *= -1.0
    out[..., 3:9] *= _ROT6D_CANONICAL_FRAME_MIRROR_SIGNS
    return out


def _parse_periodic_ckpt_step(path: Path) -> int | None:
    name = path.name
    if not (name.startswith("ckpt-") and name.endswith(".pt")):
        return None
    try:
        return int(name[len("ckpt-"):-len(".pt")])
    except ValueError:
        return None


def resolve_best_val_checkpoint(
    out_dir: Path,
    best_step: int | None,
    *,
    final_ckpt_path: Path | None = None,
    final_step: int | None = None,
) -> dict[str, Any]:
    """Resolve exact and nearest checkpoint paths for the true best-val step."""
    if best_step is None or int(best_step) < 0:
        return {
            "best_val_ckpt_path": None,
            "best_val_ckpt_step": None,
            "best_val_ckpt_exact": False,
            "best_val_nearest_ckpt_path": None,
            "best_val_nearest_ckpt_step": None,
        }

    step_i = int(best_step)
    exact = Path(out_dir) / f"ckpt-{step_i:06d}.pt"
    if exact.exists():
        return {
            "best_val_ckpt_path": str(exact),
            "best_val_ckpt_step": step_i,
            "best_val_ckpt_exact": True,
            "best_val_nearest_ckpt_path": str(exact),
            "best_val_nearest_ckpt_step": step_i,
        }

    if (
        final_ckpt_path is not None
        and final_step is not None
        and int(final_step) == step_i
        and Path(final_ckpt_path).exists()
    ):
        return {
            "best_val_ckpt_path": str(final_ckpt_path),
            "best_val_ckpt_step": step_i,
            "best_val_ckpt_exact": True,
            "best_val_nearest_ckpt_path": str(final_ckpt_path),
            "best_val_nearest_ckpt_step": step_i,
        }

    candidates: list[tuple[int, Path]] = []
    for path in Path(out_dir).glob("ckpt-*.pt"):
        ckpt_step = _parse_periodic_ckpt_step(path)
        if ckpt_step is not None and ckpt_step <= step_i:
            candidates.append((ckpt_step, path))
    if candidates:
        nearest_step, nearest_path = max(candidates, key=lambda item: item[0])
        nearest_path_s = str(nearest_path)
    else:
        nearest_step = None
        nearest_path_s = None
    return {
        "best_val_ckpt_path": None,
        "best_val_ckpt_step": None,
        "best_val_ckpt_exact": False,
        "best_val_nearest_ckpt_path": nearest_path_s,
        "best_val_nearest_ckpt_step": nearest_step,
    }


# ============================================================================
# Cache-backed dataset
# ============================================================================


@dataclass
class CoarsePriorBatch:
    coarse_v1_norm: Tensor              # (B, T_max, 23) normalized
    init_coarse_norm: Tensor            # (B, 23)
    text_pool: Tensor                   # (B, text_dim)
    valid_mask: Tensor                  # (B, T_max) bool
    seq_len: Tensor                     # (B,) long
    subsets: list[str]
    seq_ids: list[str]
    # Round-18 + Round-18-fix: optional object-trajectory channel,
    # present iff the active cache stores obj_traj per clip. The active
    # Round-19 cache field name is `obj_traj_root0_world` (root0-relative
    # world-axis); the legacy Round-18 cache field name was
    # `obj_traj_canonical` (body-canonical with Y-rotation + MoMask
    # floor-Y; superseded — kept on disk for forensic comparison).
    # Shape (B, T_max, 9) normalized. None = object-free mode.
    obj_traj_norm: Tensor | None = None


class Stage1CacheDataset(Dataset):
    """Loads Coarse-v1 .npz clips + manifest + CLIP text embedding cache.

    No HOIDataset, no plan compiler, no object loader is touched.
    """

    def __init__(
        self,
        cache_root: Path,
        split: str,
        max_seq_length: int,
        *,
        augmentation: dict[str, Any] | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        self.cache_root = Path(cache_root)
        self.split = split
        self.max_seq_length = int(max_seq_length)
        aug = augmentation or {}
        self.augment_enabled = bool(aug.get("enabled", False)) and split == "train"
        self.mirror_prob = float(aug.get("mirror_prob", 0.0)) if self.augment_enabled else 0.0
        self.mirror_duplicate = bool(aug.get("mirror_duplicate", False)) and self.augment_enabled
        self.require_mirrored_text_embeddings = bool(
            aug.get("require_mirrored_text_embeddings", True)
        )
        self.augmentation_seed = int(aug.get("seed", seed * 10_000 + 5))
        if self.mirror_prob < 0.0 or self.mirror_prob > 1.0:
            raise ValueError(f"augmentation.mirror_prob must be in [0, 1], got {self.mirror_prob}")

        manifest_path = self.cache_root / f"manifest_{split}.jsonl"
        self.records: list[dict[str, Any]] = []
        for line in manifest_path.read_text("utf-8").splitlines():
            if line.strip():
                self.records.append(json.loads(line))
        if not self.records:
            raise SystemExit(f"[stage1-data] empty manifest at {manifest_path}")

        # Load CLIP text embeddings.
        clip_npz = np.load(
            self.cache_root / "text_embeddings_clip_vit_b32.npz",
            allow_pickle=True,
        )
        self.clip_embeddings = clip_npz["embeddings"]                      # (N_unique, 512)
        idx_payload = json.loads(
            (self.cache_root / "text_embeddings_index.json").read_text("utf-8")
        )
        self.text_index: dict[str, int] = idx_payload["index"]
        self.text_dim: int = int(idx_payload["dim"])

        # Load normalization (used by collate).
        norm = json.loads((self.cache_root / "normalization_train.json").read_text("utf-8"))
        self.norm_mean = np.asarray(norm["global"]["mean"], dtype=np.float32)
        self.norm_std = np.asarray(norm["global"]["std_clamped"], dtype=np.float32)
        self.norm_dim = int(norm["n_dims"])
        # Round-18 + Round-18-fix: optional obj_traj normalization stats.
        # Look up by the new field name first (`obj_traj_root0_world`,
        # Round-18-fix), then the old name (`obj_traj_canonical`,
        # original Round-18) as a fallback so the trainer still loads
        # the legacy cache cleanly even though configs no longer point
        # at it. We DON'T accept both at once (mixing schemas would be
        # a bug).
        global_block = norm.get("global", {})
        obj_block = global_block.get("obj_traj_root0_world", None)
        self.obj_traj_field_name: str | None = None
        if obj_block is not None:
            self.obj_traj_field_name = "obj_traj_root0_world"
        else:
            obj_block = global_block.get("obj_traj_canonical", None)
            if obj_block is not None:
                self.obj_traj_field_name = "obj_traj_canonical"
        if obj_block is not None:
            self.obj_traj_norm_mean = np.asarray(obj_block["mean"], dtype=np.float32)
            self.obj_traj_norm_std = np.asarray(obj_block["std_clamped"], dtype=np.float32)
            self.obj_traj_dim = int(self.obj_traj_norm_mean.shape[0])
        else:
            self.obj_traj_norm_mean = None
            self.obj_traj_norm_std = None
            self.obj_traj_dim = 0

    def __len__(self) -> int:
        n = len(self.records)
        if self.mirror_duplicate:
            n *= 2
        return n

    def __getitem__(self, idx: int) -> dict[str, Any]:
        force_mirror = False
        base_idx = int(idx)
        if self.mirror_duplicate:
            force_mirror = bool(base_idx % 2)
            base_idx = base_idx // 2
        r = self.records[base_idx]
        npz = np.load(self.cache_root / r["npz_path"], allow_pickle=False)
        coarse = npz["coarse_v1"].astype(np.float32)                   # (T, 23)
        init = npz["init_coarse_v1"].astype(np.float32)                # (23,)
        T = min(int(r["seq_len"]), self.max_seq_length, coarse.shape[0])
        coarse = coarse[:T]
        text = r.get("text", "")

        obj_traj: np.ndarray | None = None
        if (
            self.obj_traj_dim > 0
            and self.obj_traj_field_name is not None
            and self.obj_traj_field_name in npz.files
        ):
            obj_traj = npz[self.obj_traj_field_name].astype(np.float32)[:T]  # (T, 9)

        if self.augment_enabled and not force_mirror and self.mirror_prob > 0.0:
            rng = np.random.default_rng(self.augmentation_seed + int(idx))
            force_mirror = bool(rng.random() < self.mirror_prob)
        if force_mirror:
            coarse = mirror_coarse_v1(coarse)
            init = coarse[0].astype(np.float32)
            text = _swap_left_right_in_text(text)
            if obj_traj is not None and self.obj_traj_field_name == "obj_traj_root0_world":
                obj_traj = mirror_obj_traj_root0_world(obj_traj)
            elif obj_traj is not None:
                raise RuntimeError(
                    "Stage-1 mirror augmentation requires obj_traj_root0_world. "
                    f"Found {self.obj_traj_field_name!r}; rebuild/use the Round-18-fix cache."
                )
        # z-score normalize
        coarse_norm = (coarse - self.norm_mean) / self.norm_std
        init_norm = (init - self.norm_mean) / self.norm_std
        text_row = self.text_index.get(text, None)
        if text_row is None:
            if force_mirror and self.require_mirrored_text_embeddings:
                raise KeyError(
                    "mirrored text embedding is missing from text_embeddings_index.json. "
                    "Re-run cache_stage1_clip_text_embeddings.py with "
                    "--include-mirrored-texts before enabling Stage-1 mirror augmentation. "
                    f"Missing text: {text!r}"
                )
            # Should not happen since we cached every manifest text, but
            # guard with a zero pool feature.
            text_pool = np.zeros((self.text_dim,), dtype=np.float32)
        else:
            text_pool = self.clip_embeddings[int(text_row)].astype(np.float32)
        sample = {
            "coarse_norm": coarse_norm,
            "init_norm": init_norm,
            "text_pool": text_pool,
            "seq_len": T,
            "subset": r["subset"],
            "seq_id": r["seq_id"],
        }
        # Round-18 + Round-18-fix: load obj_traj when both the cache
        # exposes the normalization stats AND the clip npz has the field.
        # New cache (Round-18-fix) stores `obj_traj_root0_world` at
        # `(seq_len, 9)`. Old cache (Round-18 original) stored
        # `obj_traj_canonical` at `(196, 9)` (padded). The trainer trims
        # to T regardless via `[:T]`. Z-score normalize on the same
        # train-only stats. Padding to T_pad happens in the collate fn.
        if obj_traj is not None:
            obj_traj_norm = (obj_traj - self.obj_traj_norm_mean) / self.obj_traj_norm_std
            sample["obj_traj_norm"] = obj_traj_norm
        return sample


def coarse_prior_collate(
    samples: list[dict[str, Any]], *, T_pad: int,
) -> CoarsePriorBatch:
    B = len(samples)
    D = samples[0]["coarse_norm"].shape[1]
    coarse_buf = np.zeros((B, T_pad, D), dtype=np.float32)
    valid_mask = np.zeros((B, T_pad), dtype=bool)
    init_buf = np.zeros((B, D), dtype=np.float32)
    text_buf = np.zeros((B, samples[0]["text_pool"].shape[0]), dtype=np.float32)
    seq_lens = np.zeros((B,), dtype=np.int64)
    subsets: list[str] = []
    seq_ids: list[str] = []
    # Round-18: pad obj_traj iff every sample in the batch has it. Mixed
    # batches (some with, some without) are not supported — the cache is
    # either obj-traj-enabled or not, per the schema.
    has_obj_traj = "obj_traj_norm" in samples[0]
    obj_traj_buf: np.ndarray | None = None
    if has_obj_traj:
        Dobj = samples[0]["obj_traj_norm"].shape[1]
        obj_traj_buf = np.zeros((B, T_pad, Dobj), dtype=np.float32)
    for i, s in enumerate(samples):
        T = int(s["seq_len"])
        coarse_buf[i, :T] = s["coarse_norm"]
        valid_mask[i, :T] = True
        init_buf[i] = s["init_norm"]
        text_buf[i] = s["text_pool"]
        seq_lens[i] = T
        subsets.append(s["subset"])
        seq_ids.append(s["seq_id"])
        if obj_traj_buf is not None:
            obj_traj_buf[i, :T] = s["obj_traj_norm"]
    return CoarsePriorBatch(
        coarse_v1_norm=torch.from_numpy(coarse_buf),
        init_coarse_norm=torch.from_numpy(init_buf),
        text_pool=torch.from_numpy(text_buf),
        valid_mask=torch.from_numpy(valid_mask),
        seq_len=torch.from_numpy(seq_lens),
        subsets=subsets,
        seq_ids=seq_ids,
        obj_traj_norm=torch.from_numpy(obj_traj_buf) if obj_traj_buf is not None else None,
    )


# ============================================================================
# Loss
# ============================================================================


@dataclass
class CoarsePriorLossWeights:
    # Per-channel-group MSE weights (applied to z-score-normalized targets).
    root_local_trans: float = 1.5         # dims [0:3]
    root_vel: float = 2.5                 # dims [3:6]  (stored vel)
    yaw_sincos: float = 1.0               # dims [6:8]
    yaw_vel: float = 2.5                  # dim [8]     (stored vel)
    pelvis_rot6d: float = 1.0             # dims [9:15]
    spine3_rot6d: float = 1.0             # dims [15:21]
    head_height: float = 0.5              # dim [21]
    shoulder_height: float = 0.5          # dim [22]
    # State-velocity loss (finite-difference) weight (Codex §5.3).
    state_vel: float = 1.0
    # Round-18 (per Round-17 §7.4 + Round-15 safety-gate evidence):
    # root-only acc² + jerk² MSE on root_local_trans dims [0:3]. Default 0.0
    # for back-compat with Round-12/14 configs; new Round-18 configs set it
    # to 0.1 per the SUGGESTION.md guidance. Operates on z-score-normalized
    # root channels — adjust weight to taste after observing the relative
    # magnitudes of l_mse / l_state_vel / l_root_acc_jerk in smoke logs.
    root_acc_jerk: float = 0.0

    def per_dim_weights(self, n_dims: int = 23) -> Tensor:
        w = torch.zeros(n_dims, dtype=torch.float32)
        w[0:3] = self.root_local_trans
        w[3:6] = self.root_vel
        w[6:8] = self.yaw_sincos
        w[8] = self.yaw_vel
        w[9:15] = self.pelvis_rot6d
        w[15:21] = self.spine3_rot6d
        w[21] = self.head_height
        w[22] = self.shoulder_height
        return w


def masked_weighted_mse(
    pred: Tensor,                # (B, T, D)
    target: Tensor,              # (B, T, D)
    valid_mask: Tensor,          # (B, T) bool
    per_dim_weights: Tensor,     # (D,)
) -> Tensor:
    """Per-frame MSE weighted by channel; masked + averaged over valid frames."""
    sq = (pred - target).pow(2)                                # (B, T, D)
    w = per_dim_weights.to(sq.device).view(1, 1, -1)
    weighted = (sq * w).sum(dim=-1)                            # (B, T)
    mask = valid_mask.to(sq.device).float()                    # (B, T)
    weighted = weighted * mask
    denom = mask.sum().clamp_min(1.0)
    return weighted.sum() / denom


def masked_weighted_mse_with_sample_weights(
    pred: Tensor,
    target: Tensor,
    valid_mask: Tensor,
    per_dim_weights: Tensor,
    sample_weights: Tensor | None = None,
) -> Tensor:
    """Channel-weighted MSE, optionally reweighted per batch item."""
    sq = (pred - target).pow(2)
    w = per_dim_weights.to(sq.device).view(1, 1, -1)
    frame_loss = (sq * w).sum(dim=-1)
    if sample_weights is not None:
        frame_loss = frame_loss * sample_weights.to(sq.device).view(-1, 1)
    mask = valid_mask.to(sq.device).float()
    denom = mask.sum().clamp_min(1.0)
    return (frame_loss * mask).sum() / denom


def min_snr_x0_sample_weights(diff, t: Tensor, gamma: float) -> tuple[Tensor, dict[str, Tensor]]:
    """Min-SNR-gamma weights for x0-prediction, normalized to mean 1."""
    alpha_bar = diff.alphas_cumprod.to(device=t.device).gather(0, t)
    snr = alpha_bar / (1.0 - alpha_bar + 1e-8)
    weights = torch.clamp_max(snr, float(gamma))
    stats = {
        "mean": weights.mean().detach(),
        "min": weights.min().detach(),
        "max": weights.max().detach(),
    }
    weights = weights / weights.mean().clamp_min(1e-8)
    return weights, stats


def masked_state_velocity_loss(
    pred: Tensor,                # (B, T, D)
    target: Tensor,              # (B, T, D)
    valid_mask: Tensor,          # (B, T) bool
    state_dims: list[int],
) -> Tensor:
    """Finite-difference MSE on a subset of dims, over valid frame-pairs.

    A frame pair (t, t+1) is "valid" iff both frames are real (not padding).
    Excludes stored-velocity dims to avoid double-counting (Codex §5.3).
    """
    # Slice the state-like channels.
    dims_t = torch.tensor(state_dims, dtype=torch.long, device=pred.device)
    p = pred.index_select(dim=-1, index=dims_t)
    g = target.index_select(dim=-1, index=dims_t)
    if p.shape[1] < 2:
        return p.sum() * 0.0
    pd = p[:, 1:] - p[:, :-1]                                  # (B, T-1, Dsub)
    gd = g[:, 1:] - g[:, :-1]
    vmask = valid_mask.to(pred.device).float()
    pair_mask = vmask[:, 1:] * vmask[:, :-1]                   # (B, T-1)
    sq = (pd - gd).pow(2).sum(dim=-1)                          # (B, T-1)
    sq = sq * pair_mask
    denom = pair_mask.sum().clamp_min(1.0)
    return sq.sum() / denom


def masked_root_acc_jerk_loss(
    pred: Tensor,                # (B, T, D)
    target: Tensor,              # (B, T, D)
    valid_mask: Tensor,          # (B, T) bool
    *,
    root_dims: tuple[int, int] = (0, 3),
) -> tuple[Tensor, Tensor]:
    """Finite-difference acceleration² + jerk² MSE on root_local_trans only.

    A frame triple (t-1, t, t+1) is "valid" iff all three frames are real
    (not padding); jerk uses a 4-frame window. Returns ``(acc_term, jerk_term)``
    each as a scalar Tensor on the same device. Operates on z-score-normalized
    inputs (caller passes already-normalized x0 and x0_pred), so the magnitude
    of the term is dimensionless and directly comparable to the channel-weighted
    MSE.

    Round-18 introduction. Per Round-17 §7.4 + Round-15 safety-gate evidence:
    S1-B's catastrophic root_jerk_p95 (15-22× GT on imhd / neur / omo) is
    exactly the failure mode an acc/jerk regulariser would penalise. Scoped
    to root channels [0:3] only so chairs's under-jittery root_jerk (xGT 0.58)
    isn't over-corrected.
    """
    lo, hi = root_dims
    if pred.shape[-1] < hi or target.shape[-1] < hi:
        raise ValueError(
            f"root_dims=[{lo}:{hi}] exceeds last dim "
            f"pred={pred.shape[-1]} target={target.shape[-1]}"
        )
    p_root = pred[..., lo:hi]                                    # (B, T, 3)
    g_root = target[..., lo:hi]                                  # (B, T, 3)
    vmask = valid_mask.to(pred.device).float()                   # (B, T)
    # Acceleration: Δ² over (t-1, t, t+1) windows. Length T-2.
    if pred.shape[1] >= 3:
        p_acc = p_root[:, 2:] - 2.0 * p_root[:, 1:-1] + p_root[:, :-2]      # (B, T-2, 3)
        g_acc = g_root[:, 2:] - 2.0 * g_root[:, 1:-1] + g_root[:, :-2]
        triple_mask = vmask[:, 2:] * vmask[:, 1:-1] * vmask[:, :-2]         # (B, T-2)
        sq_acc = (p_acc - g_acc).pow(2).sum(dim=-1)                          # (B, T-2)
        sq_acc = sq_acc * triple_mask
        denom_acc = triple_mask.sum().clamp_min(1.0)
        acc_term = sq_acc.sum() / denom_acc
    else:
        acc_term = pred.sum() * 0.0
    # Jerk: Δ³ over (t-1, t, t+1, t+2) windows. Length T-3.
    if pred.shape[1] >= 4:
        p_jerk = (
            p_root[:, 3:]
            - 3.0 * p_root[:, 2:-1]
            + 3.0 * p_root[:, 1:-2]
            - p_root[:, :-3]
        )                                                                    # (B, T-3, 3)
        g_jerk = (
            g_root[:, 3:]
            - 3.0 * g_root[:, 2:-1]
            + 3.0 * g_root[:, 1:-2]
            - g_root[:, :-3]
        )
        quad_mask = (
            vmask[:, 3:] * vmask[:, 2:-1] * vmask[:, 1:-2] * vmask[:, :-3]
        )                                                                    # (B, T-3)
        sq_jerk = (p_jerk - g_jerk).pow(2).sum(dim=-1)                       # (B, T-3)
        sq_jerk = sq_jerk * quad_mask
        denom_jerk = quad_mask.sum().clamp_min(1.0)
        jerk_term = sq_jerk.sum() / denom_jerk
    else:
        jerk_term = pred.sum() * 0.0
    return acc_term, jerk_term


def masked_per_dim_mse(
    pred: Tensor,                # (B, T, D)
    target: Tensor,              # (B, T, D)
    valid_mask: Tensor,          # (B, T) bool
) -> Tensor:
    """Per-dim mean squared error over valid frames. Returns (D,).

    Numerically identical to the per-frame masked MSE used by
    ``masked_weighted_mse`` once weighted-summed across dims:
        l_mse = (masked_per_dim_mse(...) * per_dim_w).sum()
    is equal to ``masked_weighted_mse(pred, target, valid_mask, per_dim_w)``.

    Exposing the per-dim vector lets the trainer log a per-channel-group
    breakdown to wandb without changing the backprop semantics.
    """
    sq = (pred - target).pow(2)                                # (B, T, D)
    mask = valid_mask.to(sq.device).float().unsqueeze(-1)      # (B, T, 1)
    sq = sq * mask
    denom = mask.squeeze(-1).sum().clamp_min(1.0)              # total valid (B, T) frames
    return sq.sum(dim=(0, 1)) / denom                          # (D,)


# Per-channel-group dim indices (matches CoarsePriorLossWeights groups). Used
# only for wandb logging — does not affect the loss math.
CHANNEL_GROUP_DIMS: dict[str, tuple[int, ...]] = {
    "root_local_trans": (0, 1, 2),
    "root_vel":         (3, 4, 5),
    "yaw_sincos":       (6, 7),
    "yaw_vel":          (8,),
    "pelvis_rot6d":     (9, 10, 11, 12, 13, 14),
    "spine3_rot6d":     (15, 16, 17, 18, 19, 20),
    "head_height":      (21,),
    "shoulder_height":  (22,),
}


# ============================================================================
# Trainer
# ============================================================================


def build_model(cfg: DictConfig) -> CoarsePriorDiff:
    diff_cfg = DiffusionConfig(
        num_steps=int(cfg.model.diffusion.num_steps),
        schedule=str(cfg.model.diffusion.schedule),
        objective="ddpm",
        prediction_target="x0",
    )
    den_cfg = CoarsePriorDenoiserConfig(
        coarse_dim=int(cfg.model.denoiser.coarse_dim),
        text_dim=int(cfg.model.denoiser.text_dim),
        init_pose_dim=int(cfg.model.denoiser.init_pose_dim),
        d_model=int(cfg.model.denoiser.d_model),
        n_layers=int(cfg.model.denoiser.n_layers),
        n_heads=int(cfg.model.denoiser.n_heads),
        ff_mult=int(cfg.model.denoiser.ff_mult),
        dropout=float(cfg.model.denoiser.dropout),
        max_seq_length=int(cfg.model.denoiser.max_seq_length),
        attention_mode=str(cfg.model.denoiser.attention_mode),
        block_size=int(cfg.model.denoiser.get("block_size", 16)),
        # Round-18: obj_traj plumbing. Default 0 preserves Round-12/14
        # behavior (object-free); Plan C / S1-O configs override to 9.
        obj_traj_dim=int(cfg.model.denoiser.get("obj_traj_dim", 0)),
        obj_traj_hint_hidden_mult=int(
            cfg.model.denoiser.get("obj_traj_hint_hidden_mult", 1),
        ),
    )
    return CoarsePriorDiff(CoarsePriorConfig(diffusion=diff_cfg, denoiser=den_cfg))


def build_loss_weights(cfg: DictConfig) -> CoarsePriorLossWeights:
    L = cfg.get("loss", {})
    w = L.get("channel_weights", {})
    return CoarsePriorLossWeights(
        root_local_trans=float(w.get("root_local_trans", 1.5)),
        root_vel=float(w.get("root_vel", 2.5)),
        yaw_sincos=float(w.get("yaw_sincos", 1.0)),
        yaw_vel=float(w.get("yaw_vel", 2.5)),
        pelvis_rot6d=float(w.get("pelvis_rot6d", 1.0)),
        spine3_rot6d=float(w.get("spine3_rot6d", 1.0)),
        head_height=float(w.get("head_height", 0.5)),
        shoulder_height=float(w.get("shoulder_height", 0.5)),
        state_vel=float(L.get("state_velocity_weight", 1.0)),
        # Round-18: optional root-only acc² + jerk² MSE weight. 0.0 = disabled
        # (back-compat with Round-12/14 configs). New Round-18 configs set
        # this to 0.1 per the SUGGESTION.md guidance; trainer logs the
        # relative magnitudes of l_mse / l_state_vel / l_root_acc_jerk so
        # the user can tune the weight after observing smoke runs.
        root_acc_jerk=float(L.get("root_acc_jerk_weight", 0.0)),
    )


# ============================================================================
# EMA helper (mirrors MDM-family TrainLoop.update_average_model pattern)
# ============================================================================


class EMAState:
    """Maintains an exponentially-decayed copy of model parameters.

    Mirrors ``avg_model_beta`` from the MDM-family canonical TrainLoop
    (external/guided-motion-diffusion/train/training_loop.py:347-358). At each
    optimizer step, ``ema_param = decay * ema_param + (1-decay) * live_param``.

    Sampling-time consumers should use ``apply_to(model)`` (and restore via
    ``restore(model)``) to temporarily swap the live model parameters with
    the EMA ones — this preserves the live optimizer state for continued
    training.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = float(decay)
        # Store EMA copies as detached float32 tensors on the SAME device as
        # the source parameters. Accelerate's model.prepare() will place the
        # live model on cuda before this is called; we mirror that placement.
        self._ema: dict[str, Tensor] = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self._ema[name] = p.detach().clone().float()
        # Track buffers separately so things like positional encoding pe
        # buffers don't accidentally get EMA-treated; we just snapshot them.
        self._buffers_snapshot: dict[str, Tensor] = {}
        for name, b in model.named_buffers():
            self._buffers_snapshot[name] = b.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        decay = self.decay
        for name, p in model.named_parameters():
            if not p.requires_grad or name not in self._ema:
                continue
            ema_p = self._ema[name]
            if ema_p.device != p.device:
                ema_p = ema_p.to(p.device)
                self._ema[name] = ema_p
            # ema = decay * ema + (1 - decay) * p
            ema_p.mul_(decay).add_(p.detach().float(), alpha=1.0 - decay)

    @torch.no_grad()
    def state_dict(self) -> dict[str, Tensor]:
        return {k: v.detach().clone().cpu() for k, v in self._ema.items()}

    @torch.no_grad()
    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        for k, v in state.items():
            if k in self._ema:
                self._ema[k] = v.detach().clone()

    # ---- swap helpers for eval-time use ----

    @torch.no_grad()
    def apply_to(self, model: nn.Module) -> dict[str, Tensor]:
        """Replace live model params with EMA params; return backup dict so
        the caller can ``restore`` afterwards."""
        backup: dict[str, Tensor] = {}
        for name, p in model.named_parameters():
            if not p.requires_grad or name not in self._ema:
                continue
            backup[name] = p.detach().clone()
            p.data.copy_(self._ema[name].to(p.device, p.dtype))
        return backup

    @torch.no_grad()
    def restore(self, model: nn.Module, backup: dict[str, Tensor]) -> None:
        for name, p in model.named_parameters():
            if name in backup:
                p.data.copy_(backup[name].to(p.device, p.dtype))


def make_dataloader(
    cfg: DictConfig, split: str, *, max_seq_length: int, shuffle: bool,
    shuffle_generator: torch.Generator | None = None,
) -> DataLoader:
    aug_cfg = cfg.data.get("augmentation", None)
    ds = Stage1CacheDataset(
        cache_root=Path(cfg.data.cache_root),
        split=split,
        max_seq_length=max_seq_length,
        augmentation=OmegaConf.to_container(aug_cfg, resolve=True) if aug_cfg is not None else None,
        seed=int(cfg.training.get("seed", 42)),
    )
    bs = int(cfg.training.batch_size)
    def _collate(samples):
        return coarse_prior_collate(samples, T_pad=max_seq_length)
    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=shuffle,
        # Round-18-fix: explicit `generator` for shuffle when provided.
        # Decouples DataLoader shuffle RNG from the global RNG so model-
        # construction RNG consumption (e.g. HintBlock init) doesn't
        # perturb the train batch order across Plan A and S1-O.
        generator=shuffle_generator if shuffle else None,
        drop_last=False,
        collate_fn=_collate,
        num_workers=int(cfg.training.get("num_workers", 0)),
        pin_memory=True,
    )


@torch.no_grad()
def _run_validation_pass(
    *,
    model: nn.Module,
    ema_state: "EMAState | None",
    val_loader: DataLoader,
    accelerator: Accelerator,
    diff,
    num_steps: int,
    per_dim_w: Tensor,
    loss_w: CoarsePriorLossWeights,
    model_obj_traj_dim: int,
    use_min_snr_weighting: bool = False,
    min_snr_gamma: float = 5.0,
    val_max_batches: int = 0,
    val_diff_seed: int = 0,
) -> dict[str, float]:
    """Run one validation pass under both live AND EMA weights and report
    the same loss the trainer optimizes. ``val_max_batches=0`` = full pass.

    Returns a dict with ``loss_live``, ``loss_ema`` (NaN if EMA disabled),
    ``n_batches``, and per-component breakdowns. Restores the live model
    after the EMA pass so training can continue uninterrupted.

    Round-18 final polish: validation diffusion RNG is now deterministic
    from ``val_diff_seed`` (caller passes a function of ``(seed, step)``).
    Under matched seed, Plan A and S1-O see IDENTICAL `t` and `noise` in
    every val batch — removes the spurious noise that biased best-EMA
    selection across paired runs. The LIVE and EMA passes share the SAME
    val_diff_seed (so they evaluate on bit-exact identical noise levels),
    which is what we want when comparing live vs EMA loss within a single
    val invocation. No validation draw is taken from the global RNG, so
    validation does NOT perturb the training-side RNG stream.
    """
    model.eval()
    device = accelerator.device

    def _loss_one_pass(label: str) -> tuple[float, float, float, float, int]:
        total_loss = 0.0
        total_mse = 0.0
        total_sv = 0.0
        total_rj = 0.0
        n_b = 0
        val_diff_rng = torch.Generator(device=device)
        val_diff_rng.manual_seed(int(val_diff_seed))
        for vi, batch in enumerate(val_loader):
            if val_max_batches > 0 and vi >= val_max_batches:
                break
            x0 = batch.coarse_v1_norm.to(device)
            B = x0.shape[0]
            t = torch.randint(
                0, num_steps, (B,), device=device, generator=val_diff_rng,
            )
            noise = torch.randn(
                x0.shape, device=device, dtype=x0.dtype, generator=val_diff_rng,
            )
            x_t = diff.q_sample(x0, t, noise)
            cond = {
                "text_pool": batch.text_pool.to(device),
                "init_coarse": batch.init_coarse_norm.to(device),
                "valid_mask": batch.valid_mask.to(device),
            }
            if model_obj_traj_dim > 0:
                if batch.obj_traj_norm is None:
                    raise RuntimeError(
                        "val: model has obj_traj_dim > 0 but val batch lacks obj_traj_norm"
                    )
                cond["obj_traj"] = batch.obj_traj_norm.to(device)
            x0_pred = accelerator.unwrap_model(model).forward_x0(
                x_t, t, cond, cond_drop_mask=None, obj_traj_drop_mask=None,
            )
            valid_mask = batch.valid_mask.to(device)
            per_dim_mse = masked_per_dim_mse(x0_pred, x0, valid_mask)
            sample_weights = None
            if use_min_snr_weighting:
                sample_weights, _ = min_snr_x0_sample_weights(diff, t, min_snr_gamma)
            l_mse = masked_weighted_mse_with_sample_weights(
                x0_pred, x0, valid_mask, per_dim_w, sample_weights,
            )
            l_sv = masked_state_velocity_loss(
                x0_pred, x0, valid_mask, state_dims=STATE_LIKE_DIMS,
            )
            if float(loss_w.root_acc_jerk) > 0.0:
                a, j = masked_root_acc_jerk_loss(x0_pred, x0, valid_mask)
                l_rj = a + j
            else:
                l_rj = x0.sum() * 0.0
            loss = (
                l_mse
                + float(loss_w.state_vel) * l_sv
                + float(loss_w.root_acc_jerk) * l_rj
            )
            total_loss += float(loss.detach().item())
            total_mse += float(l_mse.detach().item())
            total_sv += float(l_sv.detach().item())
            total_rj += float(l_rj.detach().item())
            n_b += 1
        if n_b == 0:
            return float("nan"), float("nan"), float("nan"), float("nan"), 0
        return (
            total_loss / n_b,
            total_mse / n_b,
            total_sv / n_b,
            total_rj / n_b,
            n_b,
        )

    # Live pass.
    live_loss, live_mse, live_sv, live_rj, n_b = _loss_one_pass("live")

    # EMA pass (if available): swap, evaluate, restore.
    if ema_state is not None:
        backup = ema_state.apply_to(accelerator.unwrap_model(model))
        try:
            ema_loss, ema_mse, ema_sv, ema_rj, _ = _loss_one_pass("ema")
        finally:
            ema_state.restore(accelerator.unwrap_model(model), backup)
    else:
        ema_loss = float("nan")
        ema_mse = float("nan")
        ema_sv = float("nan")
        ema_rj = float("nan")

    model.train()
    return {
        "loss_live": float(live_loss),
        "loss_ema": float(ema_loss),
        "mse_live": float(live_mse),
        "mse_ema": float(ema_mse),
        "state_vel_live": float(live_sv),
        "state_vel_ema": float(ema_sv),
        "root_acc_jerk_live": float(live_rj),
        "root_acc_jerk_ema": float(ema_rj),
        "n_batches": int(n_b),
    }


def cosine_lr(step: int, *, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * float(step + 1) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-train-steps", type=int, default=None,
                        help="If set, overrides training.total_steps (used for smoke runs).")
    parser.add_argument("--overfit-n-clips", type=int, default=None,
                        help="If set, restrict trainer to first N train clips (tiny overfit test).")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    # Round-13 prep-cleanup: smoke runs must be able to distinguish their
    # checkpoint from a real "final.pt" promotion artefact.
    parser.add_argument(
        "--checkpoint-name", type=str, default="final.pt",
        help="Filename for the final checkpoint written under output_dir. "
             "Use 'smoke_final.pt' (or similar) for smoke / sanity runs so "
             "the artefact is not mistaken for a promotion checkpoint.",
    )
    parser.add_argument(
        "--no-save-final", action="store_true",
        help="Skip writing the final checkpoint entirely (smoke argument-only test).",
    )
    # Round-13 follow-up: official S1-A/S1-B variance protocol needs to
    # vary cache_root + seed per invocation without editing the YAML.
    parser.add_argument(
        "--cache-root", type=Path, default=None,
        help="If set, overrides cfg.data.cache_root. Use "
             "cache/stage1_coarse_v1_full for the official variance "
             "protocol; the YAML default points at the Round-12 smoke "
             "cache for short tests only.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="If set, overrides cfg.training.seed. The official paired "
             "S1-A/S1-B sweep must pass --seed explicitly per run so "
             "six runs do not all share seed 42.",
    )
    # Round-18 trainer extensions. Most live in the YAML and can be CLI-
    # overridden for smoke-time tuning.
    parser.add_argument(
        "--save-every-n-steps", type=int, default=None,
        help="If set, overrides cfg.training.save_every_n_steps. Smoke runs "
             "typically pass a large number to skip periodic ckpts.",
    )
    parser.add_argument(
        "--val-every-n-steps", type=int, default=None,
        help="If set, overrides cfg.training.val_every_n_steps. 0 disables "
             "validation entirely.",
    )
    parser.add_argument(
        "--no-ema", action="store_true",
        help="Disable EMA even if cfg.training.ema_decay > 0. Smoke shortcut.",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.output_dir is not None:
        cfg.output_dir = str(args.output_dir)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve cache_root (CLI > YAML default).
    if args.cache_root is not None:
        cfg.data.cache_root = str(args.cache_root)
    resolved_cache_root = str(cfg.data.cache_root)

    # Resolve seed (CLI > YAML default). Write the resolved value back
    # to cfg.training.seed so the serialized config inside the
    # checkpoint payload agrees with the top-level `seed` field
    # (Round-13 final polish — seed provenance consistency).
    seed = int(args.seed) if args.seed is not None else int(cfg.training.seed)
    cfg.training.seed = seed
    set_seed(seed)

    accelerator = Accelerator(
        mixed_precision=str(cfg.training.get("mixed_precision", "no")),
        gradient_accumulation_steps=int(cfg.training.get("gradient_accumulation_steps", 1)),
    )

    model = build_model(cfg)
    loss_w = build_loss_weights(cfg)
    per_dim_w = loss_w.per_dim_weights(int(cfg.model.denoiser.coarse_dim))

    base_lr = float(cfg.training.optimizer.lr)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=float(cfg.training.optimizer.weight_decay),
        betas=tuple(cfg.training.optimizer.betas),
    )

    max_seq = int(cfg.model.denoiser.max_seq_length)
    # Use the cache's max_seq_length (Coarse-v1 capped at this length).
    # Round-18-fix: pass an explicit generator to the train DataLoader so
    # shuffle order is deterministic from `seed` alone (NOT affected by
    # how much RNG model construction consumed). This guarantees same-seed
    # Plan A and S1-O see the same batch order across steps.
    loader_rng = torch.Generator()
    loader_rng.manual_seed(seed * 10_000 + 2)
    train_loader = make_dataloader(
        cfg, "train", max_seq_length=max_seq, shuffle=True,
        shuffle_generator=loader_rng,
    )
    val_loader = make_dataloader(cfg, "val", max_seq_length=max_seq, shuffle=False)

    # Optional sub-set for tiny-overfit test.
    if args.overfit_n_clips is not None and args.overfit_n_clips > 0:
        from torch.utils.data import Subset
        n_keep = min(int(args.overfit_n_clips), len(train_loader.dataset))
        train_loader = DataLoader(
            Subset(train_loader.dataset, list(range(n_keep))),
            batch_size=int(cfg.training.batch_size),
            shuffle=True,
            drop_last=False,
            collate_fn=train_loader.collate_fn,
            num_workers=0,
            pin_memory=True,
        )
        accelerator.print(f"[stage1] overfit mode: using {n_keep} train clips only")

    total_steps = (
        int(args.max_train_steps)
        if args.max_train_steps is not None
        else int(cfg.training.get("total_steps", 1000))
    )
    warmup_steps = int(cfg.training.scheduler.get("warmup_steps", 100))
    use_min_snr_weighting = bool(cfg.loss.get("use_min_snr_weighting", False))
    min_snr_gamma = float(cfg.loss.get("min_snr_gamma", 5.0))

    use_wandb = (not args.no_wandb) and bool(cfg.logging.get("wandb", False))
    # Only main process logs to wandb under multi-GPU; single-GPU is the
    # default on the local box but this guard keeps DDP usage clean.
    if use_wandb and not accelerator.is_main_process:
        use_wandb = False
    if use_wandb:
        try:
            import wandb
            base_name = str(cfg.logging.get("run_name", out_dir.name))
            # Suffix the wandb run name with the resolved seed so 6
            # paired-seed runs of the same mode are distinct on the UI
            # (the YAML run_name alone is identical across seeds).
            full_run_name = f"{base_name}_seed{seed}"
            # Build tag set so the wandb UI can filter by mode / mask.
            attn_mode = str(cfg.model.denoiser.get("attention_mode", "none"))
            base_tags = list(cfg.logging.get("tags", ["round14", "stage1"]))
            if attn_mode == "block_causal":
                base_tags = base_tags + ["s1b", "block_causal"]
            else:
                base_tags = base_tags + ["s1a", "bidirectional"]
            wandb.init(
                project=str(cfg.logging.get("project", "piano")),
                name=full_run_name,
                group=str(cfg.logging.get("group", "stage1_paired_s1a_vs_s1b")),
                tags=base_tags,
                config=OmegaConf.to_container(cfg, resolve=True),
                # All wandb-local artefacts inside the per-run output_dir
                # so the project root stays tidy under runs/training/.
                dir=str(out_dir),
            )
            accelerator.print(
                f"[stage1] wandb run = {full_run_name}  "
                f"group=stage1_paired_s1a_vs_s1b  tags={base_tags}"
            )
        except Exception as e:
            accelerator.print(f"[stage1] wandb init failed ({e}); continuing without it")
            use_wandb = False

    model, optim, train_loader, val_loader = accelerator.prepare(
        model, optim, train_loader, val_loader,
    )
    per_dim_w = per_dim_w.to(accelerator.device)

    n_params = sum(p.numel() for p in model.parameters())
    accelerator.print(
        f"[stage1] model params = {n_params:,} | "
        f"attention_mode = {cfg.model.denoiser.attention_mode} "
        f"block_size = {cfg.model.denoiser.get('block_size', 16)} | "
        f"total_steps = {total_steps} warmup = {warmup_steps} base_lr = {base_lr}"
    )
    # Round-13 follow-up: print resolved cache_root + seed so log scraping
    # can confirm an official run isn't silently using a smoke cache or
    # the YAML-default seed.
    accelerator.print(f"[stage1] cache_root = {resolved_cache_root}")
    accelerator.print(f"[stage1] seed = {seed}")
    accelerator.print(
        f"[stage1] min_snr_weighting = {use_min_snr_weighting} "
        f"(gamma={min_snr_gamma})"
    )

    # ---------------- Round-18 trainer extensions ---------------- #
    # Independent CFG dropout probabilities; back-compat to cfg_drop_prob.
    _legacy_drop = float(cfg.training.get("cfg_drop_prob", 0.1))
    cfg_drop_prob_text = float(cfg.training.get("cfg_drop_prob_text", _legacy_drop))
    cfg_drop_prob_obj_traj = float(cfg.training.get("cfg_drop_prob_obj_traj", _legacy_drop))
    # Whether the active model expects obj_traj (depends on the loaded config).
    model_obj_traj_dim = int(cfg.model.denoiser.get("obj_traj_dim", 0))

    # EMA setup. Disabled by default for back-compat with Round-14 (which
    # had no EMA); new Round-18 configs set ema_decay > 0. CLI --no-ema
    # forces off regardless of config.
    ema_decay = float(cfg.training.get("ema_decay", 0.0))
    use_ema = bool(ema_decay > 0.0 and not args.no_ema and accelerator.is_main_process)
    ema_state: EMAState | None = None
    if use_ema:
        ema_state = EMAState(
            accelerator.unwrap_model(model), decay=ema_decay,
        )
        accelerator.print(f"[stage1] EMA enabled, decay = {ema_decay}")

    # Periodic ckpt + val cadence.
    save_every_n_steps = int(
        args.save_every_n_steps
        if args.save_every_n_steps is not None
        else cfg.training.get("save_every_n_steps", 0)
    )
    val_every_n_steps = int(
        args.val_every_n_steps
        if args.val_every_n_steps is not None
        else cfg.training.get("val_every_n_steps", 0)
    )
    val_max_batches = int(cfg.training.get("val_max_batches", 0))  # 0 = full pass

    # Best-val tracking (used to pick which periodic ckpt to mark BEST).
    best_val_loss = float("inf")
    best_val_step = -1
    best_val_path: str | None = None

    # Round-18-fix: dedicated `torch.Generator` for diffusion-step + noise
    # sampling so the timestep `t` and noise sequence are IDENTICAL across
    # Plan A (obj_traj_dim=0) and S1-O (obj_traj_dim=9) at the same seed.
    # The generator is initialised on the same device as the model so
    # randint/randn don't host-roundtrip.
    diff_rng = torch.Generator(device=accelerator.device)
    diff_rng.manual_seed(seed * 10_000 + 1)

    # Round-18 final polish: dedicated generators for text + obj_traj
    # CFG dropout. Decoupling these from the GLOBAL RNG ensures that:
    # (1) Plan A and S1-O see the SAME text-dropout mask sequence at the
    #     same seed (so the "text-conditioning signal" being dropped is
    #     paired-fair), and
    # (2) S1-O's extra obj_traj-dropout draws DON'T shift any other
    #     stream (diffusion, text, data loader, model construction).
    # All three RNG streams (diff, text, obj) are independent of each
    # other and of the global RNG.
    text_drop_rng = torch.Generator(device=accelerator.device)
    text_drop_rng.manual_seed(seed * 10_000 + 3)
    obj_drop_rng = torch.Generator(device=accelerator.device)
    obj_drop_rng.manual_seed(seed * 10_000 + 4)

    # ---------------- training loop ---------------- #
    step = 0
    train_iter = iter(train_loader)
    loss_log: list[dict[str, float]] = []
    val_log: list[dict[str, float]] = []
    diff = accelerator.unwrap_model(model).diffusion
    num_steps = int(cfg.model.diffusion.num_steps)
    t_start = time.time()

    while step < total_steps:
        try:
            batch: CoarsePriorBatch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        with accelerator.accumulate(model):
            # Set per-step LR
            lr_now = cosine_lr(step, total_steps=total_steps,
                               warmup_steps=warmup_steps, base_lr=base_lr)
            for g in optim.param_groups:
                g["lr"] = lr_now

            x0 = batch.coarse_v1_norm.to(accelerator.device)              # (B, T, 23)
            B = x0.shape[0]
            # Round-18-fix: `t` and `noise` use the dedicated `diff_rng`
            # generator — identical across Plan A and S1-O at same seed.
            t = torch.randint(
                0, num_steps, (B,), device=accelerator.device, generator=diff_rng,
            )
            noise = torch.randn(
                x0.shape, device=accelerator.device, dtype=x0.dtype, generator=diff_rng,
            )
            x_t = diff.q_sample(x0, t, noise)

            cond = {
                "text_pool": batch.text_pool.to(accelerator.device),
                "init_coarse": batch.init_coarse_norm.to(accelerator.device),
                "valid_mask": batch.valid_mask.to(accelerator.device),
            }
            # Round-18-fix: obj_traj is in the batch iff (a) the active cache
            # exposes it AND (b) the active model expects it. Mismatches
            # are caught here loud so we don't silently train a model that
            # ignores its obj_traj_dim slot. Cond key is `obj_traj`
            # (frame-agnostic; the cache contract documents the frame).
            if model_obj_traj_dim > 0:
                if batch.obj_traj_norm is None:
                    raise RuntimeError(
                        "model has obj_traj_dim > 0 but the active cache "
                        "does not expose obj_traj. Check that data.cache_root "
                        "points at an objtraj cache."
                    )
                cond["obj_traj"] = batch.obj_traj_norm.to(accelerator.device)
            # Round-18 final polish: independent dedicated generators for
            # text and obj_traj CFG dropout. The text dropout mask is now
            # IDENTICAL across Plan A and S1-O at the same seed (both draw
            # from text_drop_rng). S1-O's obj_traj dropout uses its own
            # obj_drop_rng so it doesn't perturb the text stream OR the
            # diffusion-side `t`/`noise`. Plan A doesn't touch obj_drop_rng
            # so it's idle there; not a fairness concern.
            cond_drop_mask = (
                torch.rand(B, device=accelerator.device, generator=text_drop_rng)
                    < cfg_drop_prob_text
                if cfg_drop_prob_text > 0 else None
            )
            obj_traj_drop_mask = None
            if model_obj_traj_dim > 0 and cfg_drop_prob_obj_traj > 0:
                obj_traj_drop_mask = (
                    torch.rand(B, device=accelerator.device, generator=obj_drop_rng)
                        < cfg_drop_prob_obj_traj
                )

            x0_pred = accelerator.unwrap_model(model).forward_x0(
                x_t, t, cond,
                cond_drop_mask=cond_drop_mask,
                obj_traj_drop_mask=obj_traj_drop_mask,
            )

            valid_mask = batch.valid_mask.to(accelerator.device)
            # Compute per-dim MSE once and aggregate to the weighted total.
            # The per-dim vector is reused below for the wandb breakdown;
            # Min-SNR, when enabled, applies only to the diffusion MSE term.
            per_dim_mse = masked_per_dim_mse(x0_pred, x0, valid_mask)        # (D,)
            min_snr_weight_mean = x0.sum() * 0.0
            min_snr_weight_min = x0.sum() * 0.0
            min_snr_weight_max = x0.sum() * 0.0
            sample_weights = None
            if use_min_snr_weighting:
                sample_weights, min_snr_stats = min_snr_x0_sample_weights(
                    diff, t, min_snr_gamma,
                )
                min_snr_weight_mean = min_snr_stats["mean"]
                min_snr_weight_min = min_snr_stats["min"]
                min_snr_weight_max = min_snr_stats["max"]
            l_mse = masked_weighted_mse_with_sample_weights(
                x0_pred, x0, valid_mask, per_dim_w, sample_weights,
            )
            l_state_vel = masked_state_velocity_loss(
                x0_pred, x0, valid_mask, state_dims=STATE_LIKE_DIMS,
            )
            # Round-18: root-only acc/jerk MSE. Disabled (weight 0.0) by
            # default for back-compat. Smoke runs log component magnitudes
            # so the operator can see the relative scale before choosing a
            # production weight.
            if float(loss_w.root_acc_jerk) > 0.0:
                l_root_acc, l_root_jerk = masked_root_acc_jerk_loss(
                    x0_pred, x0, valid_mask, root_dims=(0, 3),
                )
                l_root_acc_jerk = l_root_acc + l_root_jerk
            else:
                l_root_acc = x0.sum() * 0.0
                l_root_jerk = x0.sum() * 0.0
                l_root_acc_jerk = x0.sum() * 0.0
            loss = (
                l_mse
                + float(loss_w.state_vel) * l_state_vel
                + float(loss_w.root_acc_jerk) * l_root_acc_jerk
            )

            accelerator.backward(loss)
            grad_norm_t: Tensor | None = None
            if accelerator.sync_gradients:
                grad_norm_t = accelerator.clip_grad_norm_(
                    model.parameters(), max_norm=1.0,
                )
            optim.step()
            optim.zero_grad()

            # Round-18: EMA update once per real optimizer step (gated by
            # sync_gradients so micro-batches under grad accumulation don't
            # produce N updates). Main-process only — EMA lives outside the
            # accelerator.unwrap_model state.
            if (
                ema_state is not None
                and accelerator.sync_gradients
                and accelerator.is_main_process
            ):
                ema_state.update(accelerator.unwrap_model(model))

        # Logging
        if accelerator.sync_gradients:
            loss_v = loss.detach().item()
            mse_v = l_mse.detach().item()
            sv_v = l_state_vel.detach().item()
            l_root_acc_v = float(l_root_acc.detach().item())
            l_root_jerk_v = float(l_root_jerk.detach().item())
            l_root_acc_jerk_total = float(l_root_acc_jerk.detach().item())
            min_snr_weight_mean_v = float(min_snr_weight_mean.detach().item())
            min_snr_weight_min_v = float(min_snr_weight_min.detach().item())
            min_snr_weight_max_v = float(min_snr_weight_max.detach().item())
            grad_norm_v = (
                float(grad_norm_t.detach().item())
                if grad_norm_t is not None else float("nan")
            )
            x0_std_v = float(x0_pred.detach().std().item())
            x0_abs_mean_v = float(x0_pred.detach().abs().mean().item())
            # Per-channel-group weighted contribution to l_mse. The sum of
            # all group contributions equals `mse_v` by construction
            # (per_dim_mse * per_dim_w summed over disjoint dim groups).
            per_dim_mse_d = per_dim_mse.detach()
            per_dim_w_d = per_dim_w.detach()
            grp_contrib: dict[str, float] = {}
            for grp_name, dim_idxs in CHANNEL_GROUP_DIMS.items():
                dims_t = torch.tensor(dim_idxs, device=per_dim_mse_d.device)
                grp_per_dim = per_dim_mse_d.index_select(0, dims_t)
                grp_per_w = per_dim_w_d.index_select(0, dims_t)
                grp_contrib[grp_name] = float(
                    (grp_per_dim * grp_per_w).sum().item()
                )

            if step % int(cfg.logging.get("log_every_n_steps", 50)) == 0 or step == total_steps - 1:
                accelerator.print(
                    f"[stage1] step {step:5d}  lr={lr_now:.2e}  "
                    f"loss={loss_v:.4f}  mse={mse_v:.4f}  "
                    f"state_vel={sv_v:.4f}  "
                    f"root_acc_jerk={l_root_acc_jerk_total:.4f} "
                    f"(acc={l_root_acc_v:.4f} jerk={l_root_jerk_v:.4f})  "
                    f"gnorm={grad_norm_v:.3f}  "
                    f"elapsed={time.time() - t_start:.1f}s"
                )
                loss_log.append({
                    "step": int(step),
                    "loss": float(loss_v),
                    "mse": float(mse_v),
                    "state_vel": float(sv_v),
                    "root_acc": l_root_acc_v,
                    "root_jerk": l_root_jerk_v,
                    "root_acc_jerk_total": l_root_acc_jerk_total,
                    "min_snr_weight_mean": min_snr_weight_mean_v,
                    "min_snr_weight_min": min_snr_weight_min_v,
                    "min_snr_weight_max": min_snr_weight_max_v,
                    "lr": float(lr_now),
                    "grad_norm": grad_norm_v,
                    "x0_pred_std": x0_std_v,
                    "x0_pred_abs_mean": x0_abs_mean_v,
                    "mse_grp": grp_contrib,
                })
                if use_wandb:
                    import wandb
                    log_payload = {
                        # Basic training scalars (the user-facing essentials).
                        "train/loss":       float(loss_v),
                        "train/mse":        float(mse_v),
                        "train/state_vel":  float(sv_v),
                        "train/root_acc":   l_root_acc_v,
                        "train/root_jerk":  l_root_jerk_v,
                        "train/root_acc_jerk_total": l_root_acc_jerk_total,
                        "train/min_snr_weight_mean": min_snr_weight_mean_v,
                        "train/min_snr_weight_min": min_snr_weight_min_v,
                        "train/min_snr_weight_max": min_snr_weight_max_v,
                        "train/lr":         float(lr_now),
                        # Health metrics — cheap to compute, very useful for
                        # spotting collapse / blow-up runs.
                        "train/grad_norm":  grad_norm_v,
                        "train/x0_pred_std":      x0_std_v,
                        "train/x0_pred_abs_mean": x0_abs_mean_v,
                        # Wall-clock for cross-run comparison.
                        "train/elapsed_seconds": float(time.time() - t_start),
                    }
                    # Per-channel-group weighted MSE contribution. Sum of these
                    # 8 fields equals "train/mse" by construction.
                    for grp_name, grp_val in grp_contrib.items():
                        log_payload[f"train/mse_grp/{grp_name}"] = grp_val
                    wandb.log(log_payload, step=step)

            # ─── Round-18: periodic checkpoint + validation ──────────
            if (
                accelerator.is_main_process
                and save_every_n_steps > 0
                and (step + 1) % save_every_n_steps == 0
                and (step + 1) < total_steps
            ):
                ckpt_step_path = out_dir / f"ckpt-{step + 1:06d}.pt"
                _payload = {
                    "model": accelerator.unwrap_model(model).state_dict(),
                    "config": OmegaConf.to_container(cfg, resolve=True),
                    "step": int(step + 1),
                    "checkpoint_name": ckpt_step_path.name,
                    "cache_root": resolved_cache_root,
                    "seed": seed,
                }
                if ema_state is not None:
                    _payload["ema"] = ema_state.state_dict()
                    _payload["ema_decay"] = ema_decay
                accelerator.save(_payload, ckpt_step_path)
                accelerator.print(f"[stage1] periodic ckpt → {ckpt_step_path}")

            if (
                accelerator.is_main_process
                and val_every_n_steps > 0
                and (step + 1) % val_every_n_steps == 0
            ):
                # Round-18 final polish: val_diff_seed deterministic from
                # (seed, step+1) so under matched training seed the
                # validation `t`/`noise` are bit-exact across Plan A and
                # S1-O at every val invocation. This removes a spurious
                # source of variance in best-EMA selection across paired
                # runs.
                val_metrics = _run_validation_pass(
                    model=model,
                    ema_state=ema_state,
                    val_loader=val_loader,
                    accelerator=accelerator,
                    diff=diff,
                    num_steps=num_steps,
                    per_dim_w=per_dim_w,
                    loss_w=loss_w,
                    model_obj_traj_dim=model_obj_traj_dim,
                    use_min_snr_weighting=use_min_snr_weighting,
                    min_snr_gamma=min_snr_gamma,
                    val_max_batches=val_max_batches,
                    val_diff_seed=seed * 1_000_000 + (step + 1),
                )
                val_metrics["step"] = int(step + 1)
                val_log.append(val_metrics)
                ema_val = val_metrics.get("loss_ema", float("nan"))
                live_val = val_metrics.get("loss_live", float("nan"))
                accelerator.print(
                    f"[stage1] val step {step + 1}  "
                    f"loss_live={live_val:.4f}  loss_ema={ema_val:.4f}  "
                    f"n_batches={val_metrics.get('n_batches', 0)}"
                )
                if use_wandb:
                    import wandb
                    wandb.log({
                        "val/loss_live": float(live_val),
                        "val/loss_ema": float(ema_val),
                        "val/n_batches": int(val_metrics.get("n_batches", 0)),
                    }, step=step + 1)
                # Track best val (prefer EMA, fall back to live).
                cmp_val = ema_val if math.isfinite(ema_val) else live_val
                if math.isfinite(cmp_val) and cmp_val < best_val_loss:
                    best_val_loss = float(cmp_val)
                    best_val_step = int(step + 1)
                    cand = out_dir / f"ckpt-{step + 1:06d}.pt"
                    best_val_path = str(cand) if cand.exists() else None
                    best_info_now = resolve_best_val_checkpoint(out_dir, best_val_step)
                    accelerator.print(
                        f"[stage1] new best val loss = {best_val_loss:.4f} "
                        f"at step {best_val_step}  "
                        f"(exact_ckpt: {best_val_path or 'none'}; "
                        f"nearest_ckpt: {best_info_now['best_val_nearest_ckpt_path'] or 'none'})"
                    )
            step += 1

    # Save final state.
    if accelerator.is_main_process:
        # Round-13: --no-save-final disables checkpoint write entirely
        # (used by smoke argument tests). --checkpoint-name lets smoke
        # runs avoid clobbering / impersonating a real "final.pt".
        if args.no_save_final:
            accelerator.print(
                "[stage1] --no-save-final set: skipping checkpoint write"
            )
            ckpt_path = None
        else:
            ckpt_path = out_dir / str(args.checkpoint_name)
            _final_payload = {
                "model": accelerator.unwrap_model(model).state_dict(),
                "config": OmegaConf.to_container(cfg, resolve=True),
                "step": step,
                "checkpoint_name": str(args.checkpoint_name),
                # Round-13 follow-up: persist the resolved runtime knobs
                # so the eval / analysis stage can spot a smoke-cache
                # or wrong-seed checkpoint after the fact.
                "cache_root": resolved_cache_root,
                "seed": seed,
            }
            # Round-18: include EMA state when enabled. Sampling tools
            # should load the "ema" entry preferentially for best-quality
            # generation.
            if ema_state is not None:
                _final_payload["ema"] = ema_state.state_dict()
                _final_payload["ema_decay"] = ema_decay
            accelerator.save(_final_payload, ckpt_path)
            accelerator.print(f"[stage1] wrote {ckpt_path}")
        best_ckpt_info = resolve_best_val_checkpoint(
            out_dir,
            int(best_val_step) if best_val_step >= 0 else None,
            final_ckpt_path=ckpt_path,
            final_step=step,
        )
        # Loss log is always written (it does not impersonate a model
        # checkpoint and is useful for the round report).
        (out_dir / "loss_log.json").write_text(
            json.dumps({
                "steps": loss_log,
                "total_steps": total_steps,
                "checkpoint_name": (
                    str(args.checkpoint_name) if not args.no_save_final else None
                ),
                "cache_root": resolved_cache_root,
                "seed": seed,
            }, indent=2),
            encoding="utf-8",
        )
        # Round-18: emit a separate val + best-ckpt summary so post-hoc
        # selection has the canonical record. Always written even if val
        # never ran (the file then contains empty val_log + sentinel).
        (out_dir / "training_summary.json").write_text(
            json.dumps({
                "total_steps": total_steps,
                "cache_root": resolved_cache_root,
                "seed": seed,
                "ema_enabled": bool(ema_state is not None),
                "ema_decay": ema_decay if ema_state is not None else 0.0,
                "save_every_n_steps": save_every_n_steps,
                "val_every_n_steps": val_every_n_steps,
                "val_log": val_log,
                "best_val_loss": float(best_val_loss) if best_val_loss != float("inf") else None,
                "best_val_step": int(best_val_step) if best_val_step >= 0 else None,
                "best_val_ckpt_path": best_ckpt_info["best_val_ckpt_path"],
                "best_val_ckpt_step": best_ckpt_info["best_val_ckpt_step"],
                "best_val_ckpt_exact": best_ckpt_info["best_val_ckpt_exact"],
                "best_val_nearest_ckpt_path": best_ckpt_info["best_val_nearest_ckpt_path"],
                "best_val_nearest_ckpt_step": best_ckpt_info["best_val_nearest_ckpt_step"],
                "model_obj_traj_dim": model_obj_traj_dim,
                "cfg_drop_prob_text": cfg_drop_prob_text,
                "cfg_drop_prob_obj_traj": cfg_drop_prob_obj_traj,
                "use_min_snr_weighting": use_min_snr_weighting,
                "min_snr_gamma": min_snr_gamma,
            }, indent=2),
            encoding="utf-8",
        )
    # Cleanly close the wandb run so the next paired invocation can call
    # wandb.init() without interference.
    if use_wandb:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
