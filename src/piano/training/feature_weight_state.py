"""FeatureWeightState — per-group MSE weights with optional dynamic update.

Per PLAN.md M1.5 Step 4-7:

- Holds per-group group weights (one float per FEATURE_GROUPS entry).
- Expands to a (1, 1, MOTION_DIM) tensor consumed by per-frame MSE.
- Supports OFFLINE-INIT (static, v2-style) and DYNAMIC (v2.1-style) modes.
- Dynamic update is called at epoch boundaries from the trainer's
  callback hook; never inside per-batch step_fn.
- Combines `geometry_prior_g`, `residual_factor_g(epoch)`, EMA smoothing,
  clamp, and total-MSE-scale renormalisation.

Save/load via state_dict for ckpt round-trip.

DDP: in DDP, only rank 0 should update; broadcast new group weights to
all ranks before the next epoch.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from piano.training.feature_groups import (
    FEATURE_GROUPS, MOTION_DIM, ROOT_MOTION_GROUPS, FeatureGroup,
)


@dataclass(slots=True)
class FeatureWeightState:
    """Mutable state for per-group feature weights.

    Default behaviour (no dynamic): static weights from `static_weights`.
    When `enable_dynamic=True`, supports update via :meth:`update`.
    """

    # Per-group static fallback (v2-style hand-tuned values).
    # Used as base_prior_floor under v2.1 dynamic combine, AND as
    # initial seed before any dynamic updates.
    static_weights: dict[str, float]

    # Per-group geometry prior, computed offline once. Set to None when
    # running in pure static mode (v2 fallback).
    geometry_prior: dict[str, float] | None = None

    # Current effective per-group weights. Initialized in __post_init__.
    current: dict[str, float] = field(default_factory=dict)

    # Dynamic-mode parameters
    enable_dynamic: bool = False
    update_every_epochs: int = 5
    ema_beta: float = 0.2
    residual_alpha: float = 0.5
    clamp_min: float = 0.25
    clamp_max: float = 150.0
    target_mse_scale: float | None = None    # if None, computed from static_weights

    # Bookkeeping
    last_update_epoch: int = -1

    def __post_init__(self) -> None:
        for g in FEATURE_GROUPS:
            if g.name not in self.static_weights:
                raise KeyError(f"Missing static weight for group {g.name!r}")
        if not self.current:
            self.current = dict(self.static_weights)
        # target MSE scale: Σ_g (n_dims_g * static_weight_g)
        if self.target_mse_scale is None:
            self.target_mse_scale = float(sum(
                g.n_dims * self.static_weights[g.name] for g in FEATURE_GROUPS
            ))

    # ------------------------------------------------------------------
    # Tensor expansion (used by step_fn every batch)
    # ------------------------------------------------------------------
    def to_per_frame_tensor(self, device: torch.device) -> Tensor:
        """Expand current per-group weights to a (1, 1, MOTION_DIM) tensor."""
        w = torch.ones(MOTION_DIM, device=device, dtype=torch.float32)
        for g in FEATURE_GROUPS:
            w[g.lo:g.hi] = self.current[g.name]
        return w.view(1, 1, -1)

    # ------------------------------------------------------------------
    # Dynamic update (Step 4-5)
    # ------------------------------------------------------------------
    def should_update(self, epoch: int) -> bool:
        if not self.enable_dynamic:
            return False
        if epoch <= 0:
            return False
        if (epoch % self.update_every_epochs) != 0:
            return False
        if epoch == self.last_update_epoch:
            return False
        return True

    def update(
        self,
        epoch: int,
        group_rmse: dict[str, float],
        group_gt_std: dict[str, float],
    ) -> dict[str, dict[str, float]]:
        """Recompute group weights from current residuals.

        Parameters
        ----------
        group_rmse : measured per-group RMSE on the calibration split,
            single-step denoising residual (PLAN.md Step 4).
        group_gt_std : per-group GT std on the calibration split (used
            to scale-normalise the residuals).

        Returns
        -------
        log : dict of intermediates for logging
            {
                "norm_err_g": {...},
                "relative_err_g": {...},
                "residual_factor_g": {...},
                "raw_w_g": {...},
                "estimated_w_g": {...},
                "new_w_g": {...},
            }
        """
        if not self.enable_dynamic:
            raise RuntimeError("update() called on a static state")

        log: dict[str, dict[str, float]] = {}

        # --- norm_err and relative_err (Step 4) ---
        norm_err = {
            n: group_rmse[n] / max(group_gt_std[n], 1e-9)
            for n in self.current
        }
        # Mean weighted by group dim count (per Step 4 spec)
        total_w = float(sum(g.n_dims for g in FEATURE_GROUPS))
        mean_norm = sum(
            FEATURE_GROUPS[i].n_dims * norm_err[FEATURE_GROUPS[i].name]
            for i in range(len(FEATURE_GROUPS))
        ) / max(total_w, 1.0)
        relative_err = {
            n: norm_err[n] / max(mean_norm, 1e-9) for n in norm_err
        }
        residual_factor = {
            n: max(relative_err[n], 1e-9) ** self.residual_alpha
            for n in relative_err
        }

        # --- raw_w_g (Step 5): geometry_prior * residual_factor, with
        # static_weights as a base floor for groups whose geometry prior
        # is ~0 (e.g. joint_rot_6d, joint_velocity, foot_contact don't
        # affect recover_from_ric so their geometry_prior is 0). ---
        raw_w: dict[str, float] = {}
        prior = self.geometry_prior or {}
        for g in FEATURE_GROUPS:
            n = g.name
            base = max(prior.get(n, 0.0), self.static_weights[n])
            raw_w[n] = base * residual_factor[n]

        # Clamp
        clamped = {n: min(max(w, self.clamp_min), self.clamp_max)
                   for n, w in raw_w.items()}
        # Root-motion floor: never below static_weights for these
        for n in ROOT_MOTION_GROUPS:
            clamped[n] = max(clamped[n], self.static_weights[n])

        # Renormalise to keep target MSE scale (Step 5)
        weighted_sum = float(sum(g.n_dims * clamped[g.name] for g in FEATURE_GROUPS))
        scale = self.target_mse_scale / max(weighted_sum, 1e-9)
        estimated = {n: clamped[n] * scale for n in clamped}

        # EMA smoothing (Step 5)
        new_w = {
            n: (1 - self.ema_beta) * self.current[n]
            + self.ema_beta * estimated[n]
            for n in self.current
        }

        # Final clamp again after EMA (defensive, EMA shouldn't push out)
        new_w = {n: min(max(w, self.clamp_min), self.clamp_max)
                 for n, w in new_w.items()}
        for n in ROOT_MOTION_GROUPS:
            new_w[n] = max(new_w[n], self.static_weights[n])

        log = {
            "epoch": epoch,
            "geometry_prior": dict(prior) if prior else {n: 0.0 for n in self.current},
            "norm_err_g": norm_err,
            "relative_err_g": relative_err,
            "residual_factor_g": residual_factor,
            "raw_w_g": raw_w,
            "estimated_w_g": estimated,
            "old_w_g": dict(self.current),
            "new_w_g": new_w,
        }

        self.current = new_w
        self.last_update_epoch = epoch
        return log

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        return {
            "current": dict(self.current),
            "static_weights": dict(self.static_weights),
            "geometry_prior": dict(self.geometry_prior) if self.geometry_prior else None,
            "enable_dynamic": self.enable_dynamic,
            "update_every_epochs": self.update_every_epochs,
            "ema_beta": self.ema_beta,
            "residual_alpha": self.residual_alpha,
            "clamp_min": self.clamp_min,
            "clamp_max": self.clamp_max,
            "target_mse_scale": self.target_mse_scale,
            "last_update_epoch": self.last_update_epoch,
        }

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        self.current = dict(sd["current"])
        self.static_weights = dict(sd["static_weights"])
        self.geometry_prior = dict(sd["geometry_prior"]) if sd["geometry_prior"] else None
        self.enable_dynamic = bool(sd["enable_dynamic"])
        self.update_every_epochs = int(sd["update_every_epochs"])
        self.ema_beta = float(sd["ema_beta"])
        self.residual_alpha = float(sd["residual_alpha"])
        self.clamp_min = float(sd["clamp_min"])
        self.clamp_max = float(sd["clamp_max"])
        self.target_mse_scale = float(sd["target_mse_scale"])
        self.last_update_epoch = int(sd["last_update_epoch"])

    # ------------------------------------------------------------------
    # DDP broadcast (called from rank 0 after update())
    # ------------------------------------------------------------------
    def broadcast(self, accelerator) -> None:
        """If running in DDP, broadcast `current` from main process to others."""
        if not accelerator.use_distributed:
            return
        # Pack current dict into a fixed-order tensor on device
        device = accelerator.device
        ordered = [self.current[g.name] for g in FEATURE_GROUPS]
        t = torch.tensor(ordered, device=device, dtype=torch.float32)
        torch.distributed.broadcast(t, src=0)
        for i, g in enumerate(FEATURE_GROUPS):
            self.current[g.name] = float(t[i].item())

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def static_from_config(cls, weights_cfg: dict | None) -> "FeatureWeightState":
        """v2-style fixed weights, no dynamic update."""
        defaults = {
            "root_rot_vel": 100.0,
            "root_lin_vel": 30.0,
            "root_height_y": 30.0,
            "joint_pos_local": 1.0,
            "joint_rot_6d": 1.0,
            "joint_velocity": 1.0,
            "foot_contact": 5.0,
        }
        if weights_cfg:
            defaults.update(dict(weights_cfg))
        return cls(static_weights=defaults, enable_dynamic=False)

    @classmethod
    def dynamic_from_config(
        cls,
        static_weights_cfg: dict | None,
        geometry_prior_path: str,
        update_every_epochs: int = 5,
        ema_beta: float = 0.2,
        residual_alpha: float = 0.5,
        clamp_min: float = 0.25,
        clamp_max: float = 150.0,
    ) -> "FeatureWeightState":
        """v2.1-style dynamic group metric. Loads geometry_prior from JSON."""
        # Static seed (used as base_prior_floor + initial weights)
        static = {
            "root_rot_vel": 100.0,
            "root_lin_vel": 30.0,
            "root_height_y": 30.0,
            "joint_pos_local": 1.0,
            "joint_rot_6d": 1.0,
            "joint_velocity": 1.0,
            "foot_contact": 5.0,
        }
        if static_weights_cfg:
            static.update(dict(static_weights_cfg))

        prior_data = json.loads(Path(geometry_prior_path).read_text())
        norm_prior = prior_data["normalized_geometry_prior"]
        for g in FEATURE_GROUPS:
            if g.name not in norm_prior:
                raise KeyError(f"geometry_prior.json missing group {g.name!r}")

        return cls(
            static_weights=static,
            geometry_prior=norm_prior,
            enable_dynamic=True,
            update_every_epochs=update_every_epochs,
            ema_beta=ema_beta,
            residual_alpha=residual_alpha,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
        )
