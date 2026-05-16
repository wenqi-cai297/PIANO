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
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

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


class Stage1CacheDataset(Dataset):
    """Loads Coarse-v1 .npz clips + manifest + CLIP text embedding cache.

    No HOIDataset, no plan compiler, no object loader is touched.
    """

    def __init__(
        self,
        cache_root: Path,
        split: str,
        max_seq_length: int,
    ) -> None:
        super().__init__()
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        self.cache_root = Path(cache_root)
        self.split = split
        self.max_seq_length = int(max_seq_length)

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

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        r = self.records[idx]
        npz = np.load(self.cache_root / r["npz_path"], allow_pickle=False)
        coarse = npz["coarse_v1"].astype(np.float32)                   # (T, 23)
        init = npz["init_coarse_v1"].astype(np.float32)                # (23,)
        T = min(int(r["seq_len"]), self.max_seq_length, coarse.shape[0])
        coarse = coarse[:T]
        # z-score normalize
        coarse_norm = (coarse - self.norm_mean) / self.norm_std
        init_norm = (init - self.norm_mean) / self.norm_std
        text = r.get("text", "")
        text_row = self.text_index.get(text, None)
        if text_row is None:
            # Should not happen since we cached every manifest text, but
            # guard with a zero pool feature.
            text_pool = np.zeros((self.text_dim,), dtype=np.float32)
        else:
            text_pool = self.clip_embeddings[int(text_row)].astype(np.float32)
        return {
            "coarse_norm": coarse_norm,
            "init_norm": init_norm,
            "text_pool": text_pool,
            "seq_len": T,
            "subset": r["subset"],
            "seq_id": r["seq_id"],
        }


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
    for i, s in enumerate(samples):
        T = int(s["seq_len"])
        coarse_buf[i, :T] = s["coarse_norm"]
        valid_mask[i, :T] = True
        init_buf[i] = s["init_norm"]
        text_buf[i] = s["text_pool"]
        seq_lens[i] = T
        subsets.append(s["subset"])
        seq_ids.append(s["seq_id"])
    return CoarsePriorBatch(
        coarse_v1_norm=torch.from_numpy(coarse_buf),
        init_coarse_norm=torch.from_numpy(init_buf),
        text_pool=torch.from_numpy(text_buf),
        valid_mask=torch.from_numpy(valid_mask),
        seq_len=torch.from_numpy(seq_lens),
        subsets=subsets,
        seq_ids=seq_ids,
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
    )


def make_dataloader(
    cfg: DictConfig, split: str, *, max_seq_length: int, shuffle: bool,
) -> DataLoader:
    ds = Stage1CacheDataset(
        cache_root=Path(cfg.data.cache_root),
        split=split,
        max_seq_length=max_seq_length,
    )
    bs = int(cfg.training.batch_size)
    def _collate(samples):
        return coarse_prior_collate(samples, T_pad=max_seq_length)
    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=shuffle,
        drop_last=False,
        collate_fn=_collate,
        num_workers=int(cfg.training.get("num_workers", 0)),
        pin_memory=True,
    )


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
    train_loader = make_dataloader(cfg, "train", max_seq_length=max_seq, shuffle=True)
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

    use_wandb = (not args.no_wandb) and bool(cfg.logging.get("wandb", False))
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=str(cfg.logging.get("project", "piano")),
                name=str(cfg.logging.get("run_name", out_dir.name)),
                config=OmegaConf.to_container(cfg, resolve=True),
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

    # ---------------- training loop ---------------- #
    step = 0
    train_iter = iter(train_loader)
    loss_log: list[dict[str, float]] = []
    cfg_drop_prob = float(cfg.training.get("cfg_drop_prob", 0.1))
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
            t = torch.randint(0, num_steps, (B,), device=accelerator.device)
            noise = torch.randn_like(x0)
            x_t = diff.q_sample(x0, t, noise)

            cond = {
                "text_pool": batch.text_pool.to(accelerator.device),
                "init_coarse": batch.init_coarse_norm.to(accelerator.device),
                "valid_mask": batch.valid_mask.to(accelerator.device),
            }
            cond_drop_mask = (
                torch.rand(B, device=accelerator.device) < cfg_drop_prob
                if cfg_drop_prob > 0 else None
            )

            x0_pred = accelerator.unwrap_model(model).forward_x0(
                x_t, t, cond, cond_drop_mask=cond_drop_mask,
            )

            valid_mask = batch.valid_mask.to(accelerator.device)
            l_mse = masked_weighted_mse(x0_pred, x0, valid_mask, per_dim_w)
            l_state_vel = masked_state_velocity_loss(
                x0_pred, x0, valid_mask, state_dims=STATE_LIKE_DIMS,
            )
            loss = l_mse + float(loss_w.state_vel) * l_state_vel

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            optim.zero_grad()

        # Logging
        if accelerator.sync_gradients:
            loss_v = loss.detach().item()
            mse_v = l_mse.detach().item()
            sv_v = l_state_vel.detach().item()
            if step % int(cfg.logging.get("log_every_n_steps", 50)) == 0 or step == total_steps - 1:
                accelerator.print(
                    f"[stage1] step {step:5d}  lr={lr_now:.2e}  "
                    f"loss={loss_v:.4f}  mse={mse_v:.4f}  "
                    f"state_vel={sv_v:.4f}  "
                    f"elapsed={time.time() - t_start:.1f}s"
                )
                loss_log.append({
                    "step": int(step),
                    "loss": float(loss_v),
                    "mse": float(mse_v),
                    "state_vel": float(sv_v),
                    "lr": float(lr_now),
                })
                if use_wandb:
                    import wandb
                    wandb.log({
                        "train/loss": float(loss_v),
                        "train/mse": float(mse_v),
                        "train/state_vel": float(sv_v),
                        "train/lr": float(lr_now),
                    }, step=step)
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
            accelerator.save({
                "model": accelerator.unwrap_model(model).state_dict(),
                "config": OmegaConf.to_container(cfg, resolve=True),
                "step": step,
                "checkpoint_name": str(args.checkpoint_name),
                # Round-13 follow-up: persist the resolved runtime knobs
                # so the eval / analysis stage can spot a smoke-cache
                # or wrong-seed checkpoint after the fact.
                "cache_root": resolved_cache_root,
                "seed": seed,
            }, ckpt_path)
            accelerator.print(f"[stage1] wrote {ckpt_path}")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
