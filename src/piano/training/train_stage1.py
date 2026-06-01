"""Stage-1 (Trajectory & Orientation) training entry.

Trains ``Stage1Denoiser`` (in ``piano.models.stage1_trajectory``) to
predict the 23-D ``stage1_coarse`` representation that Stage-2 PB1
consumes via ``cond["stage1_coarse"]``.

Reuses Stage-2's infrastructure:

  - ``_build_dataset`` from ``train_anchordiff`` (PIANO clip loader).
  - ``GaussianDiffusion`` + ``DiffusionConfig`` from
    ``piano.models.motion_anchordiff`` (cosine schedule, x0-prediction).
  - ``ObjectEncoder`` (PointNet++) and CLIP text encoder.
  - ``run_training_loop`` from ``piano.training.trainer``.
  - GT 23-D extraction from
    ``piano.data.stage1_coarse_oracle.extract_coarse_v1_batched``.

Design source: ``analyses/2026-05-29_stage1_and_stage1_5_design.md``.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader

from piano.data.dataset import collate_hoi
from piano.data.stage1_coarse_oracle import (
    extract_coarse_v1_batched,
    load_stage1_coarse_norm,
)
from piano.models.motion_anchordiff import (
    DiffusionConfig,
    GaussianDiffusion,
    _extract,
)
from piano.models.object_encoder import ObjectEncoder
from piano.models.stage1_trajectory import (
    STAGE1_COARSE_DIM,
    Stage1Denoiser,
    Stage1DenoiserConfig,
)
from piano.training.stage1_losses import (
    CH_HEAD_HEIGHT,
    CH_PELVIS_ROT6D,
    CH_SHOULDER_H,
    CH_SPINE3_ROT6D,
    INIT_POSE_F2_DIM,
    build_channel_weight_tensor,
    build_init_pose_f1,
    build_init_pose_f2,
    channel_moment_match_loss,
    fk_height_consistency_loss,
    fk_pelvis_spine_pos_loss,
    fk_pelvis_spine_pos_loss_cm,
    frame0_consistency_loss,
    kinematic_self_consistency_loss,
    rot6d_ortho_loss,
    stage1_plan_invariant_loss,
    temporal_derivative_mse_loss,
    wrist_fk_supervision_loss,
    yaw_aggregate_match_loss,
)
from piano.training.smpl_kinematics import (
    fk_from_global_rotations,
    rotation_6d_to_matrix,
)
from piano.training.train_anchordiff import _build_dataset
from piano.training.trainer import (
    build_scheduler,
    run_training_loop,
)
from piano.utils.clip_utils import encode_text_per_token, load_clip_text_encoder


# ----------------------------------------------------------------------------
# Warm-start (R41 prereq) — load a prior Stage-1 ckpt into a fresh trainer
# ----------------------------------------------------------------------------


def _maybe_load_stage1_init_checkpoint(
    *,
    model: "Stage1Denoiser",
    object_encoder: "ObjectEncoder",
    ckpt_path: str | None,
    strict: bool = True,
) -> None:
    """Optionally warm-start Stage-1 training from a prior checkpoint.

    R41 cascade fine-tunes V8 V6 (drift_max 17.43 cm on R31 V8 matrix)
    by adding a frozen-PB1 motion-space loss on top of the existing
    Stage-1 training objective. Without this loader, ``training.init_checkpoint``
    in the config would be silently ignored and the run would start from
    a fresh init — which would confound R41 results with from-scratch
    training. See ``analyses/2026-06-02_r41_stage1_cascade_experiment_plan_for_claude.md``
    §3.1.

    Parameters
    ----------
    model
        Pre-built ``Stage1Denoiser`` (not yet wrapped by Accelerator).
    object_encoder
        Pre-built ``ObjectEncoder`` (not yet wrapped). Loading the
        object encoder state is required: leaving it at random init
        means the object features fed to Stage-1 disagree with whatever
        the original ckpt was trained against, which silently breaks
        fine-tuning.
    ckpt_path
        Path to the ckpt to load. ``None`` or empty string → no-op
        (preserves from-scratch behavior bit-for-bit when the cfg key
        is absent).
    strict
        Forwarded to ``load_state_dict``. Default True: any key
        mismatch raises so we never silently partial-load and ship a
        broken warm-start.

    Raises
    ------
    FileNotFoundError
        If ``ckpt_path`` is non-empty but the file does not exist.
    KeyError
        If the ckpt has no ``object_encoder`` state under either the
        top-level key or the ``extra_modules`` nested key (matches the
        two save formats ``sample_substitute_conds.py`` supports).
    """
    if not ckpt_path:
        return

    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(
            f"training.init_checkpoint does not exist: {path}"
        )

    state = torch.load(str(path), map_location="cpu", weights_only=False)

    # Denoiser. The trainer's save format wraps the model state under
    # ``state["model"]``; fall back to a flat state_dict for legacy ckpts.
    model_state = state.get("model", state)
    model.load_state_dict(model_state, strict=strict)

    # Object encoder. Two save formats coexist in the repo (matches
    # ``sample_substitute_conds.py:143-156``):
    #   1) state["object_encoder"]                          — newer
    #   2) state["extra_modules"]["object_encoder"]         — older
    if "object_encoder" in state:
        object_encoder.load_state_dict(
            state["object_encoder"], strict=strict,
        )
    elif (
        isinstance(state.get("extra_modules"), dict)
        and "object_encoder" in state["extra_modules"]
    ):
        object_encoder.load_state_dict(
            state["extra_modules"]["object_encoder"], strict=strict,
        )
    else:
        raise KeyError(
            f"training.init_checkpoint {path} has no object_encoder "
            "state. Cannot warm-start without it: the object features "
            "would reset to random init while the denoiser is loaded "
            "from the ckpt, silently breaking fine-tuning."
        )


# ----------------------------------------------------------------------------
# Step function
# ----------------------------------------------------------------------------


def build_stage1_step_fn(
    model: Stage1Denoiser,
    diffusion: GaussianDiffusion,
    object_encoder: ObjectEncoder,
    clip_model: torch.nn.Module | None,
    device: torch.device,
    stage1_coarse_mean_t: Tensor,
    stage1_coarse_std_t: Tensor,
    *,
    cfg_drop_prob: float = 0.15,
    w_x0: float = 1.0,
    w_vel: float = 1.0,
    w_yaw_smooth: float = 0.02,
    # R31 V2 ablation losses (default 0 = OFF, V0 baseline behaviour).
    w_rot6d_ortho: float = 0.0,
    w_fk_pos: float = 0.0,
    w_height_fk: float = 0.0,
    w_self_consistency: float = 0.0,
    # R31 V7 anti-mode-collapse losses (default 0 = OFF, V0 baseline).
    # V7-A: per-channel (mean,std) matching of finite-diff magnitudes.
    w_moment_velocity: float = 0.0,
    # V7-A': per-channel (mean,std) matching of raw values.
    w_moment_value: float = 0.0,
    # V7-B: aggregate yaw transition rate + cumulative range matching.
    w_yaw_aggregate: float = 0.0,
    # V7-C: cm-space SmoothL1 FK pos (PB1 L_pos scale; do NOT combine with
    # the L2 ``w_fk_pos`` term in the same run, they shadow each other).
    w_fk_pos_cm: float = 0.0,
    fk_pos_cm_beta: float = 1.0,
    # R31 V8 — wrist FK supervision (extends V7-C's target chain to wrist).
    w_wrist_fk_pos: float = 0.0,
    wrist_fk_target_joints: tuple[int, ...] | None = None,
    wrist_fk_joint_weights: tuple[float, ...] | None = None,
    wrist_fk_contact_mode: str = "off",          # off | reweight | hard
    wrist_fk_contact_weight: float = 4.0,
    wrist_fk_add_velocity: bool = False,
    wrist_fk_velocity_weight: float = 0.5,
    wrist_fk_beta_cm: float = 1.0,
    # R31 V8 — frame-0 consistency loss (applied on Stage-1 t=0 prediction).
    # Used together with denoiser config init_pose_dim > 0 (F1 or F2 mode).
    w_init_pose_consistency: float = 0.0,
    # R36 raw-space temporal dynamics. Defaults off for back-compat.
    w_r36_raw_velocity: float = 0.0,
    w_r36_raw_acceleration: float = 0.0,
    r36_raw_dynamics_channel_subset: tuple[int, ...] | None = None,
    r36_raw_dynamics_normalize_by_gt_std: bool = True,
    # R31 V8 — denoiser-injected init_pose mode (0 = OFF, 14 = F2,
    # 135 = F1). Must match cfg.model.denoiser.init_pose_dim.
    init_pose_dim: int = 0,
    # rot6d-weighted velocity loss: multiplies channels [9:21] of the
    # vel-MSE term by ``vel_rot6d_weight`` (default 1.0 = baseline).
    vel_rot6d_weight: float = 1.0,
    # R40 — per-channel weighting for the exact-GT MSE terms.
    # Empty / None means "all ones" → identical to pre-R40 behavior.
    # When provided, each list must have length 23 (one per stage1_coarse
    # channel) and is applied multiplicatively on top of the existing
    # ``vel_rot6d_weight`` for the velocity term. Lets configs reduce
    # GT pressure on under-determined channels (root, vel, yaw, pelvis_rot6d)
    # so the plan-invariant loss can dominate.
    x0_channel_weights: tuple[float, ...] | list[float] | None = None,
    vel_channel_weights: tuple[float, ...] | list[float] | None = None,
    # R40 — plan-invariant ("plan-energy") loss. Default 0 = OFF.
    w_r40_plan_invariant: float = 0.0,
    r40_plan_beta: float = 1.0,
    r40_plan_component_weights: dict[str, float] | None = None,
    use_min_snr_weighting: bool = True,
    min_snr_gamma: float = 5.0,
):
    """Return the Stage-1 ``step_fn(model, batch, global_step)`` closure.

    Training target is the **z-scored** 23-D stage1_coarse — same
    normalisation Stage-2 PB1 was trained against. This way Stage-1's
    output drops directly into Stage-2's cond[\"stage1_coarse\"] at
    inference with no extra re-scaling.

    Per design doc §"Training loss":
      L = w_x0 * MSE(x0_pred, x0_gt_normed)
        + w_vel * MSE(vel(x0_pred), vel(x0_gt_normed))
        + w_yaw_smooth * mean(|Δ²yaw_unwrapped(raw_x0_pred)|)
    """
    # Unwrap DDP for read-only diffusion access.
    _diff_for_read = (
        diffusion.module if hasattr(diffusion, "module") else diffusion
    )

    # R40 — materialise per-channel weight tensors once. ``None`` means
    # all ones; the step_fn skips the multiplication branch entirely so
    # default behavior is bit-identical to pre-R40.
    _w_dtype = stage1_coarse_mean_t.dtype
    x0_channel_w = build_channel_weight_tensor(
        x0_channel_weights, expected_dim=STAGE1_COARSE_DIM,
        device=device, dtype=_w_dtype, name="x0_channel_weights",
    )
    vel_channel_w = build_channel_weight_tensor(
        vel_channel_weights, expected_dim=STAGE1_COARSE_DIM,
        device=device, dtype=_w_dtype, name="vel_channel_weights",
    )

    def step_fn(_model, batch: dict, global_step: int = 0) -> dict[str, Tensor]:
        motion = batch["motion"].to(device)                    # (B, T, 135)
        rest_offsets = batch["rest_offsets"].to(device).float()  # (B, 22, 3)
        gt_joints = batch["joints"].to(device).float()         # (B, T, 22, 3)
        object_pc = batch["object_pc"].to(device)
        seq_len = batch["seq_len"].to(device)                  # (B,)

        B, T, _ = motion.shape
        seq_idx = torch.arange(T, device=device).unsqueeze(0)
        seq_mask = (seq_idx < seq_len.unsqueeze(1)).float()    # (B, T)

        # ─── Build object trajectory (3 pos + 6 rot6d = 9) ───
        # Match Stage-2's convention exactly: use the CANONICAL COM and
        # CANONICAL rot6d. The Stage-2 trainer at train_anchordiff.py:327
        # uses obj_com_canonical for this same 9-D channel.
        obj_com = batch["obj_com_canonical"].to(device)        # (B, T, 3)
        obj_rot6d = batch["obj_rot6d_canonical"].to(device)    # (B, T, 6)
        object_traj = torch.cat([obj_com, obj_rot6d], dim=-1)  # (B, T, 9)

        # ─── Object tokens ───
        obj_tokens = object_encoder(object_pc)                 # (B, N_obj, D_obj)

        # ─── Text features ───
        if clip_model is not None and "text" in batch:
            text_features, _text_mask = encode_text_per_token(
                clip_model, batch["text"], device,
            )
            text_features = text_features.float()
        else:
            text_features = None

        # ─── GT target: z-scored 23-D stage1_coarse ───
        # Stage-2 PB1 was trained against this same z-scoring; Stage-1
        # learns the normalised output so its inference drops directly
        # into Stage-2's cond["stage1_coarse"] with no re-scaling.
        coarse_v1_raw = extract_coarse_v1_batched(
            motion=motion, rest_offsets=rest_offsets,
        )                                                       # (B, T, 23)
        if coarse_v1_raw.shape[-1] != STAGE1_COARSE_DIM:
            raise RuntimeError(
                f"extract_coarse_v1_batched returned {coarse_v1_raw.shape[-1]}D; "
                f"expected {STAGE1_COARSE_DIM}D."
            )
        coarse_v1 = (coarse_v1_raw - stage1_coarse_mean_t) / stage1_coarse_std_t

        # ─── Build cond dict for the denoiser ───
        cond: dict[str, Tensor] = {
            "object_world_traj": object_traj,
            "object_tokens": obj_tokens,
        }
        if text_features is not None:
            cond["text"] = text_features

        # R31 V8 — init_pose injection (F1 = 135-D raw, F2 = 14-D z-scored).
        init_pose_targets_z: Tensor | None = None
        if init_pose_dim == 135:
            cond["init_pose"] = build_init_pose_f1(motion)            # (B, 135)
        elif init_pose_dim == 14:
            cond["init_pose"] = build_init_pose_f2(
                coarse_v1_raw, stage1_coarse_mean_t, stage1_coarse_std_t,
            )                                                          # (B, 14)
            init_pose_targets_z = cond["init_pose"]
        elif init_pose_dim != 0:
            raise ValueError(
                f"init_pose_dim must be 0, 14, or 135; got {init_pose_dim}."
            )
        # When init_pose_dim == 14 but cfg didn't enable consistency loss,
        # init_pose_targets_z is still useful (computed once). When the
        # weight is 0 the loss term short-circuits.
        if init_pose_dim == 0 and w_init_pose_consistency > 0:
            # User error: consistency loss requires F2 targets. Compute
            # them from coarse_v1_raw without registering them in cond.
            init_pose_targets_z = build_init_pose_f2(
                coarse_v1_raw, stage1_coarse_mean_t, stage1_coarse_std_t,
            )

        # ─── Diffusion forward (x₀-prediction) ───
        # x_t = sqrt(α_t) * x_0 + sqrt(1-α_t) * noise; predict x_0.
        x0 = coarse_v1                                          # (B, T, 23)
        t = torch.randint(
            0, _diff_for_read.num_steps, (B,), device=device, dtype=torch.long,
        )
        noise = torch.randn_like(x0)
        sqrt_a = _extract(_diff_for_read.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_om = _extract(_diff_for_read.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        x_t = sqrt_a * x0 + sqrt_om * noise

        # CFG drop mask on text + obj_tokens (never on obj_traj).
        if cfg_drop_prob > 0 and _model.training:
            cond_drop_mask = (
                torch.rand((B,), device=device) < cfg_drop_prob
            )
        else:
            cond_drop_mask = None

        x0_pred = _model(x_t, t, cond, cond_drop_mask=cond_drop_mask)  # (B, T, 23)

        # ─── Loss 1: MSE on x0 with optional min-SNR-γ weighting ───
        mse_per_dim = (x0_pred - x0).pow(2)                     # (B, T, 23)
        # R40 — apply per-channel weighting before summing. Unweighted
        # version is preserved as an audit metric.
        if x0_channel_w is not None:
            mse_per_dim_weighted = mse_per_dim * x0_channel_w
        else:
            mse_per_dim_weighted = mse_per_dim
        per_frame = mse_per_dim_weighted.sum(-1)                # (B, T)

        if use_min_snr_weighting:
            alpha_bar = _diff_for_read.alphas_cumprod.gather(0, t)
            snr = alpha_bar / (1.0 - alpha_bar + 1e-8)
            snr_clamped = torch.clamp_max(snr, float(min_snr_gamma))
            w_b = snr_clamped                                    # x0-pred form
            w_b_norm = w_b / w_b.mean().clamp_min(1e-8)
            per_frame = per_frame * w_b_norm.view(-1, 1)

        mse_x0 = (per_frame * seq_mask).sum() / seq_mask.sum().clamp_min(1.0)
        mse_x0_unweighted = (
            mse_per_dim.sum(-1) * seq_mask
        ).sum() / seq_mask.sum().clamp_min(1.0)

        # ─── Loss 2: velocity consistency ───
        # 1-frame finite diff on the FIRST 9 channels (root_local xzy + vel xzy
        # + yaw sin/cos/vel — i.e. the kinematic block that matters most).
        if T >= 2 and w_vel > 0:
            vel_pred = x0_pred[:, 1:] - x0_pred[:, :-1]         # (B, T-1, 23)
            vel_gt = x0[:, 1:] - x0[:, :-1]
            vel_mask = seq_mask[:, 1:] * seq_mask[:, :-1]       # (B, T-1)
            vel_per_dim_raw = (vel_pred - vel_gt).pow(2)        # (B, T-1, 23)
            # R40 audit metric — unweighted per-dim sum, no rot6d boost.
            vel_mse_unweighted = (
                vel_per_dim_raw.sum(-1) * vel_mask
            ).sum() / vel_mask.sum().clamp_min(1.0)
            vel_per_dim = vel_per_dim_raw
            if vel_rot6d_weight != 1.0:
                # Re-weight the rot6d channels [9:21] in the velocity MSE.
                channel_w = torch.ones(
                    23, device=device, dtype=vel_per_dim.dtype,
                )
                channel_w[CH_PELVIS_ROT6D] = float(vel_rot6d_weight)
                channel_w[CH_SPINE3_ROT6D] = float(vel_rot6d_weight)
                vel_per_dim = vel_per_dim * channel_w.view(1, 1, -1)
            # R40 — per-channel velocity weights, composed multiplicatively
            # on top of vel_rot6d_weight. Lets configs downweight exact-GT
            # frame-to-frame velocity on under-determined channels.
            if vel_channel_w is not None:
                vel_per_dim = vel_per_dim * vel_channel_w
            vel_mse = (vel_per_dim.sum(-1) * vel_mask).sum() / vel_mask.sum().clamp_min(1.0)
        else:
            vel_mse = torch.zeros((), device=device, dtype=mse_x0.dtype)
            vel_mse_unweighted = torch.zeros((), device=device, dtype=mse_x0.dtype)

        # ─── Loss 3: yaw 2nd-derivative smoothness on PRED ───
        # Pred yaw is at channels [6: 8] (sin, cos). We compute the angle via
        # atan2 and penalise its 2nd diff. Note: yaw_vel itself is channel 8,
        # but here we smooth the derived angle, not the predicted vel channel
        # (the vel-MSE term already constrains channel 8 directly).
        if T >= 3 and w_yaw_smooth > 0:
            yaw_pred = torch.atan2(x0_pred[..., 6], x0_pred[..., 7])  # (B, T)
            yaw_d1 = yaw_pred[:, 1:] - yaw_pred[:, :-1]
            # Wrap to [-π, π] to handle the atan2 discontinuity.
            yaw_d1 = (yaw_d1 + 3.14159265) % (2 * 3.14159265) - 3.14159265
            yaw_d2 = yaw_d1[:, 1:] - yaw_d1[:, :-1]                   # (B, T-2)
            yaw_mask = (
                seq_mask[:, 2:] * seq_mask[:, 1:-1] * seq_mask[:, :-2]
            )
            yaw_sm = (yaw_d2.abs() * yaw_mask).sum() / yaw_mask.sum().clamp_min(1.0)
        else:
            yaw_sm = torch.zeros((), device=device, dtype=mse_x0.dtype)

        # ─── Loss 4-7 (R31 V2 ablation losses) + V7 anti-collapse losses ─
        # All operate on the RAW (un-z-scored) prediction (and, for V7-A,
        # the matching raw GT). Compute once (avoid re-allocating per
        # loss term).
        need_raw = (
            w_rot6d_ortho > 0 or w_fk_pos > 0 or w_height_fk > 0
            or w_self_consistency > 0
            or w_moment_velocity > 0 or w_moment_value > 0
            or w_yaw_aggregate > 0 or w_fk_pos_cm > 0
            or w_wrist_fk_pos > 0
            or w_r36_raw_velocity > 0 or w_r36_raw_acceleration > 0
            or w_r40_plan_invariant > 0
        )
        if need_raw:
            x0_raw = x0_pred * stage1_coarse_std_t + stage1_coarse_mean_t   # (B, T, 23)
            x0_gt_raw = coarse_v1_raw                                        # (B, T, 23) un-z-scored GT
        else:
            x0_raw = None
            x0_gt_raw = None

        # L1: rot6d orthogonality.
        if w_rot6d_ortho > 0 and x0_raw is not None:
            mask_3d = seq_mask                                              # (B, T)
            pelvis_rot6d = x0_raw[..., CH_PELVIS_ROT6D]                     # (B, T, 6)
            spine3_rot6d = x0_raw[..., CH_SPINE3_ROT6D]                     # (B, T, 6)
            ortho_loss = (
                rot6d_ortho_loss(pelvis_rot6d, mask=mask_3d)
                + rot6d_ortho_loss(spine3_rot6d, mask=mask_3d)
            )
        else:
            ortho_loss = torch.zeros((), device=device, dtype=mse_x0.dtype)

        # L2: FK position loss on neck/head/shoulders driven by pelvis +
        # spine3 predicted rotations. Requires raw root_world (= root_local
        # + GT frame-0 root_world).
        if w_fk_pos > 0 and x0_raw is not None:
            root_local_pred = x0_raw[..., :3]                               # (B, T, 3)
            root_world_t0 = motion[:, :1, 132:135].float()                  # (B, 1, 3)
            # root_local channel order is (x, z, y) per the oracle convention.
            # motion[..., 132:135] is also (x, y, z). The oracle stores the
            # channels in oracle-order; we need to reconstruct the world
            # position by adding the same-order frame-0. Since both are
            # frame-0-relative + frame-0 absolute in the same channel order
            # respectively, the absolute pelvis world is consistent.
            #
            # CAUTION: stage1_coarse_oracle.py stores channels as
            # (x, z, y) but motion[..., 132:135] is (x, y, z). To recover
            # world pelvis we map (x_local, z_local, y_local) → (x, y, z)
            # and add to motion[..., 132:135].
            root_local_world_order = torch.stack(
                [
                    root_local_pred[..., 0],
                    root_local_pred[..., 2],   # y is at slot 2 in oracle
                    root_local_pred[..., 1],   # z is at slot 1 in oracle
                ], dim=-1,
            )                                                                # (B, T, 3) in (x, y, z) world order
            root_world_pred = root_local_world_order + root_world_t0
            fk_pos = fk_pelvis_spine_pos_loss(
                pelvis_rot6d_pred=x0_raw[..., CH_PELVIS_ROT6D],
                spine3_rot6d_pred=x0_raw[..., CH_SPINE3_ROT6D],
                root_world_pred=root_world_pred,
                gt_motion_135=motion,
                rest_offsets=rest_offsets,
                gt_joints=gt_joints,
                seq_mask=seq_mask,
            )
        else:
            fk_pos = torch.zeros((), device=device, dtype=mse_x0.dtype)

        # L3: height-FK consistency.
        if w_height_fk > 0 and x0_raw is not None:
            root_local_pred = x0_raw[..., :3]
            root_world_t0 = motion[:, :1, 132:135].float()
            root_local_world_order = torch.stack(
                [
                    root_local_pred[..., 0],
                    root_local_pred[..., 2],
                    root_local_pred[..., 1],
                ], dim=-1,
            )
            root_world_pred = root_local_world_order + root_world_t0
            height_fk = fk_height_consistency_loss(
                head_height_pred=x0_raw[..., CH_HEAD_HEIGHT],
                shoulder_h_pred=x0_raw[..., CH_SHOULDER_H],
                pelvis_rot6d_pred=x0_raw[..., CH_PELVIS_ROT6D],
                spine3_rot6d_pred=x0_raw[..., CH_SPINE3_ROT6D],
                root_world_pred=root_world_pred,
                gt_motion_135=motion,
                rest_offsets=rest_offsets,
                seq_mask=seq_mask,
            )
        else:
            height_fk = torch.zeros((), device=device, dtype=mse_x0.dtype)

        # L4: kinematic self-consistency (diff/vel + yaw).
        if w_self_consistency > 0 and x0_raw is not None:
            self_cons = kinematic_self_consistency_loss(x0_raw, seq_mask)
        else:
            self_cons = torch.zeros((), device=device, dtype=mse_x0.dtype)

        # ─── V7-A: per-channel moment match (velocity + optional value) ─
        # Mirrors PB1's stable_local_speed_moment (train_anchordiff.py:1106).
        # Directly penalises std collapse on finite-diff magnitudes.
        if (w_moment_velocity > 0 or w_moment_value > 0) and x0_raw is not None:
            moment_match = channel_moment_match_loss(
                stage1_raw_pred=x0_raw,
                stage1_raw_gt=x0_gt_raw,
                seq_mask=seq_mask,
                velocity_match=(w_moment_velocity > 0),
                value_match=(w_moment_value > 0),
            )
        else:
            moment_match = torch.zeros((), device=device, dtype=mse_x0.dtype)
        # Use the larger of the two weights so a single combined weight
        # scales the helper (the helper sums value + velocity terms
        # internally). The variants we ship use only one of them at a time.
        w_moment_match = max(float(w_moment_velocity), float(w_moment_value))

        # ─── V7-B: yaw aggregate-statistic match (transition rate + range) ─
        # Mirrors PB1's gait transition_rate / duty_cycle pattern.
        if w_yaw_aggregate > 0 and x0_raw is not None:
            yaw_agg = yaw_aggregate_match_loss(
                stage1_raw_pred=x0_raw,
                stage1_raw_gt=x0_gt_raw,
                seq_mask=seq_mask,
            )
        else:
            yaw_agg = torch.zeros((), device=device, dtype=mse_x0.dtype)

        # ─── V7-C: cm-space SmoothL1 FK pos (PB1 L_pos scale weights) ───
        # Reuses the same predicted root_world derivation as L2/L3.
        if (w_fk_pos_cm > 0 or w_wrist_fk_pos > 0) and x0_raw is not None:
            # Compute root_world_pred once and reuse for V7-C and V8.
            root_local_pred = x0_raw[..., :3]
            root_world_t0 = motion[:, :1, 132:135].float()
            root_local_world_order = torch.stack(
                [
                    root_local_pred[..., 0],
                    root_local_pred[..., 2],
                    root_local_pred[..., 1],
                ], dim=-1,
            )
            root_world_pred = root_local_world_order + root_world_t0
        else:
            root_world_pred = None

        if w_fk_pos_cm > 0 and x0_raw is not None:
            fk_pos_cm = fk_pelvis_spine_pos_loss_cm(
                pelvis_rot6d_pred=x0_raw[..., CH_PELVIS_ROT6D],
                spine3_rot6d_pred=x0_raw[..., CH_SPINE3_ROT6D],
                root_world_pred=root_world_pred,
                gt_motion_135=motion,
                rest_offsets=rest_offsets,
                gt_joints=gt_joints,
                seq_mask=seq_mask,
                beta_cm=float(fk_pos_cm_beta),
            )
        else:
            fk_pos_cm = torch.zeros((), device=device, dtype=mse_x0.dtype)

        # ─── V8 — wrist FK supervision (extends V7-C target chain) ───
        if w_wrist_fk_pos > 0 and x0_raw is not None:
            contact_state = batch.get("contact_state", None)
            if contact_state is not None:
                contact_state = contact_state.to(device).float()
            wrist_kwargs: dict = dict(
                pelvis_rot6d_pred=x0_raw[..., CH_PELVIS_ROT6D],
                spine3_rot6d_pred=x0_raw[..., CH_SPINE3_ROT6D],
                root_world_pred=root_world_pred,
                gt_motion_135=motion,
                rest_offsets=rest_offsets,
                gt_joints=gt_joints,
                seq_mask=seq_mask,
                contact_state=contact_state,
                contact_mask_mode=str(wrist_fk_contact_mode),
                contact_active_weight=float(wrist_fk_contact_weight),
                add_velocity=bool(wrist_fk_add_velocity),
                velocity_weight=float(wrist_fk_velocity_weight),
                beta_cm=float(wrist_fk_beta_cm),
            )
            if wrist_fk_target_joints is not None:
                wrist_kwargs["target_joints"] = tuple(int(j) for j in wrist_fk_target_joints)
            if wrist_fk_joint_weights is not None:
                wrist_kwargs["joint_weights"] = tuple(float(w) for w in wrist_fk_joint_weights)
            wrist_fk = wrist_fk_supervision_loss(**wrist_kwargs)
        else:
            wrist_fk = torch.zeros((), device=device, dtype=mse_x0.dtype)

        # ─── V8 — frame-0 consistency on the 14 channels Stage-1 outputs ───
        if w_init_pose_consistency > 0 and init_pose_targets_z is not None:
            init_pose_cons = frame0_consistency_loss(
                stage1_pred_zscored=x0_pred,
                init_pose_targets_zscored=init_pose_targets_z,
            )
        else:
            init_pose_cons = torch.zeros((), device=device, dtype=mse_x0.dtype)

        if w_r36_raw_velocity > 0 and x0_raw is not None and x0_gt_raw is not None:
            r36_raw_vel = temporal_derivative_mse_loss(
                x0_raw,
                x0_gt_raw,
                seq_mask,
                order=1,
                channel_subset=r36_raw_dynamics_channel_subset,
                normalize_by_gt_std=bool(r36_raw_dynamics_normalize_by_gt_std),
            )
        else:
            r36_raw_vel = torch.zeros((), device=device, dtype=mse_x0.dtype)

        if w_r36_raw_acceleration > 0 and x0_raw is not None and x0_gt_raw is not None:
            r36_raw_acc = temporal_derivative_mse_loss(
                x0_raw,
                x0_gt_raw,
                seq_mask,
                order=2,
                channel_subset=r36_raw_dynamics_channel_subset,
                normalize_by_gt_std=bool(r36_raw_dynamics_normalize_by_gt_std),
            )
        else:
            r36_raw_acc = torch.zeros((), device=device, dtype=mse_x0.dtype)

        # ─── R40 — plan-invariant loss ─────────────────────────────────
        # Match (pred, gt) on plan-level invariants (speed/arc/turn/
        # root-object radial/heights/smoothness) rather than per-frame GT.
        # Lets the model land on any plausible mode of the multi-modal
        # plan distribution while keeping average/frozen plans expensive.
        r40_plan = torch.zeros((), device=device, dtype=mse_x0.dtype)
        r40_components: dict[str, Tensor] = {}
        if w_r40_plan_invariant > 0 and x0_raw is not None and x0_gt_raw is not None:
            root_world_t0_for_plan = motion[:, :1, 132:135].float()
            r40_plan, r40_components = stage1_plan_invariant_loss(
                stage1_raw_pred=x0_raw,
                stage1_raw_gt=x0_gt_raw,
                object_world_traj=object_traj,
                root_world_t0=root_world_t0_for_plan,
                seq_mask=seq_mask,
                component_weights=r40_plan_component_weights,
                beta=float(r40_plan_beta),
            )

        loss = (
            w_x0 * mse_x0
            + w_vel * vel_mse
            + w_yaw_smooth * yaw_sm
            + w_rot6d_ortho * ortho_loss
            + w_fk_pos * fk_pos
            + w_height_fk * height_fk
            + w_self_consistency * self_cons
            + w_moment_match * moment_match
            + w_yaw_aggregate * yaw_agg
            + w_fk_pos_cm * fk_pos_cm
            + w_wrist_fk_pos * wrist_fk
            + w_init_pose_consistency * init_pose_cons
            + w_r36_raw_velocity * r36_raw_vel
            + w_r36_raw_acceleration * r36_raw_acc
            + w_r40_plan_invariant * r40_plan
        )

        # R40 hotfix — NaN/Inf guard. If any per-component loss is
        # non-finite, identify the FIRST offender and replace the total
        # loss with a zero-grad sentinel for this step so the optimizer
        # skips it instead of poisoning every parameter with NaN.
        # (C3 step-50 NaN, 2026-06-01 — the original log showed every
        # component nan simultaneously, which is the cascade signature
        # of a single upstream NaN propagating. Without this guard the
        # next NaN is unfindable from training output.)
        _components_for_check = (
            ("mse_x0", mse_x0),
            ("vel_mse", vel_mse),
            ("yaw_smooth", yaw_sm),
            ("rot6d_ortho", ortho_loss),
            ("fk_pos", fk_pos),
            ("height_fk", height_fk),
            ("self_consistency", self_cons),
            ("moment_match", moment_match),
            ("yaw_aggregate", yaw_agg),
            ("fk_pos_cm", fk_pos_cm),
            ("wrist_fk", wrist_fk),
            ("init_pose_consistency", init_pose_cons),
            ("r36_raw_velocity", r36_raw_vel),
            ("r36_raw_acceleration", r36_raw_acc),
            ("r40_plan_invariant", r40_plan),
        )
        first_bad: str | None = None
        for _name, _val in _components_for_check:
            if not torch.isfinite(_val).all():
                first_bad = _name
                break
        if first_bad is not None:
            print(
                f"[stage1 step {global_step}] NaN/Inf detected — first bad "
                f"component: {first_bad}. Replacing loss with zero-grad "
                f"sentinel so this step is skipped by the optimizer."
            )
            # Zero-grad sentinel: detach the toxic loss and replace with
            # 0 * sum(x0_pred), which has a valid gradient path (= 0) so
            # backward() succeeds and the optimizer step is effectively
            # a no-op.
            loss = (x0_pred.sum() * 0.0)

        result: dict[str, Tensor] = {
            "loss": loss,
            "mse_x0": mse_x0.detach(),
            "mse_x0_unweighted": mse_x0_unweighted.detach(),
            "vel_mse": vel_mse.detach(),
            "vel_mse_unweighted": vel_mse_unweighted.detach(),
            "yaw_smooth": yaw_sm.detach(),
            "rot6d_ortho": ortho_loss.detach(),
            "fk_pos": fk_pos.detach(),
            "height_fk": height_fk.detach(),
            "self_consistency": self_cons.detach(),
            "moment_match": moment_match.detach(),
            "yaw_aggregate": yaw_agg.detach(),
            "fk_pos_cm": fk_pos_cm.detach(),
            "wrist_fk": wrist_fk.detach(),
            "init_pose_consistency": init_pose_cons.detach(),
            "r36_raw_velocity": r36_raw_vel.detach(),
            "r36_raw_acceleration": r36_raw_acc.detach(),
            "r40_plan_invariant": r40_plan.detach(),
            "r40_plan_invariant_weighted": (
                w_r40_plan_invariant * r40_plan
            ).detach(),
        }
        for comp_name, comp_val in r40_components.items():
            result[f"r40_plan_{comp_name}"] = comp_val.detach()
        return result

    return step_fn


# ----------------------------------------------------------------------------
# Main entry
# ----------------------------------------------------------------------------


def _make_dataloader(
    dataset, batch_size: int, num_workers: int, shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=True,
        drop_last=shuffle,
        collate_fn=collate_hoi,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run a single batch + backward to verify wiring; do not save.",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    # find_unused_parameters=True: defensive, matches Stage-2's trainer.
    # When use_text=False the text path is dead, and any future cond drop
    # variant could surface other unused branches; True avoids deadlock /
    # error on the per-iter DDP all-reduce.
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.get(
            "gradient_accumulation_steps", 1,
        ),
        mixed_precision=cfg.training.get("mixed_precision", "bf16"),
        kwargs_handlers=[ddp_kwargs],
    )
    set_seed(int(cfg.training.get("seed", 42)))
    device = accelerator.device

    accelerator.print("===== Stage-1 (Trajectory) training =====")
    accelerator.print(f"output_dir = {cfg.output_dir}")
    accelerator.print(f"smoke_test = {args.smoke_test}")

    # ─── Datasets ───
    train_dataset = _build_dataset(cfg, bucket="train", augment=True)
    val_dataset = None
    if int(cfg.training.get("val_every_epochs", 0)) > 0:
        val_dataset = _build_dataset(cfg, bucket="val", augment=False)
    accelerator.print(f"Train: {len(train_dataset)} clips")
    if val_dataset is not None:
        accelerator.print(f"Val:   {len(val_dataset)} clips")

    train_loader = _make_dataloader(
        train_dataset,
        batch_size=int(cfg.training.batch_size),
        num_workers=int(cfg.training.get("num_workers", 4)),
        shuffle=True,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = _make_dataloader(
            val_dataset,
            batch_size=int(cfg.training.batch_size),
            num_workers=int(cfg.training.get("num_workers", 4)),
            shuffle=False,
        )

    # ─── Model ───
    denoiser_cfg = Stage1DenoiserConfig(
        motion_dim=int(cfg.model.denoiser.motion_dim),
        object_traj_dim=int(cfg.model.denoiser.object_traj_dim),
        text_dim=int(cfg.model.denoiser.text_dim),
        object_token_dim=int(cfg.model.denoiser.object_token_dim),
        object_num_tokens=int(cfg.model.denoiser.object_num_tokens),
        d_model=int(cfg.model.denoiser.d_model),
        n_layers=int(cfg.model.denoiser.n_layers),
        n_heads=int(cfg.model.denoiser.n_heads),
        ff_mult=int(cfg.model.denoiser.ff_mult),
        dropout=float(cfg.model.denoiser.dropout),
        max_seq_length=int(cfg.data.max_seq_length),
        use_text=bool(cfg.model.denoiser.get("use_text", True)),
        init_pose_dim=int(cfg.model.denoiser.get("init_pose_dim", 0)),
    )
    if denoiser_cfg.motion_dim != STAGE1_COARSE_DIM:
        raise ValueError(
            f"Stage-1 motion_dim must be {STAGE1_COARSE_DIM}; got "
            f"{denoiser_cfg.motion_dim}."
        )
    model = Stage1Denoiser(denoiser_cfg)

    diff_cfg = DiffusionConfig(
        num_steps=int(cfg.model.diffusion.num_steps),
        schedule=str(cfg.model.diffusion.schedule),
        objective=str(cfg.model.diffusion.get("objective", "ddpm")),
        prediction_target=str(
            cfg.model.diffusion.get("prediction_target", "x0"),
        ),
    )
    diffusion = GaussianDiffusion(diff_cfg)

    object_encoder = ObjectEncoder(
        num_input_points=int(cfg.model.object_encoder.num_input_points),
        num_output_tokens=int(cfg.model.object_encoder.num_output_tokens),
        feature_dim=int(cfg.model.object_encoder.feature_dim),
    )

    if int(cfg.model.denoiser.text_dim) > 0:
        clip_model = load_clip_text_encoder(
            device=device,
            model_name=str(cfg.model.text_encoder.clip_version),
            download_root=str(
                cfg.model.text_encoder.get("download_root", "cache/clip"),
            ),
        )
    else:
        clip_model = None

    # ─── R41 prereq — optional warm-start from a prior Stage-1 ckpt ──
    # Default None → no-op, preserving V0/V7/V8 from-scratch behavior.
    # Must run before the optimizer is built so the optimizer captures
    # loaded params, and before ``accelerator.prepare(...)`` so the
    # load happens on plain (un-DDP-wrapped) modules.
    _init_ckpt_path = cfg.training.get("init_checkpoint", None)
    _init_ckpt_strict = bool(cfg.training.get("init_checkpoint_strict", True))
    if _init_ckpt_path:
        accelerator.print(
            f"[warm-start] loading init_checkpoint={_init_ckpt_path} "
            f"(strict={_init_ckpt_strict})"
        )
    _maybe_load_stage1_init_checkpoint(
        model=model,
        object_encoder=object_encoder,
        ckpt_path=str(_init_ckpt_path) if _init_ckpt_path else None,
        strict=_init_ckpt_strict,
    )

    # ─── Optimizer + scheduler ───
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(object_encoder.parameters()),
        lr=float(cfg.training.optimizer.lr),
        weight_decay=float(cfg.training.optimizer.weight_decay),
        betas=tuple(cfg.training.optimizer.get("betas", [0.9, 0.999])),
    )
    total_steps = int(
        cfg.training.num_epochs * len(train_loader)
        // int(cfg.training.get("gradient_accumulation_steps", 1))
    )
    scheduler = build_scheduler(
        optimizer,
        warmup_steps=int(cfg.training.scheduler.get("warmup_steps", 500)),
        total_steps=max(total_steps, 1),
    )

    # ─── Load Stage-1 norm stats so the GT target is z-scored ──────────
    # Stage-2 PB1 was trained against (raw - mean)/std using the same
    # cache; we match its target so Stage-1's output drops in directly.
    norm_mean_np, norm_std_np = load_stage1_coarse_norm(
        str(cfg.data.stage1_coarse_cache_root)
    )
    stage1_coarse_mean_t = (
        torch.from_numpy(norm_mean_np).to(device).float()
    )
    stage1_coarse_std_t = (
        torch.from_numpy(norm_std_np).to(device).float()
    )

    # ─── Prepare with accelerator ───
    (
        model, object_encoder, optimizer, train_loader, scheduler,
    ) = accelerator.prepare(
        model, object_encoder, optimizer, train_loader, scheduler,
    )
    if val_loader is not None:
        val_loader = accelerator.prepare(val_loader)
    diffusion = diffusion.to(device)

    step_fn = build_stage1_step_fn(
        model=model,
        diffusion=diffusion,
        object_encoder=object_encoder,
        clip_model=clip_model,
        device=device,
        stage1_coarse_mean_t=stage1_coarse_mean_t,
        stage1_coarse_std_t=stage1_coarse_std_t,
        cfg_drop_prob=float(cfg.model.get("cfg_drop_prob", 0.15)),
        w_x0=float(cfg.loss.w_x0),
        w_vel=float(cfg.loss.w_vel),
        w_yaw_smooth=float(cfg.loss.w_yaw_smooth),
        # R31 V2 ablation losses (default 0 if config doesn't specify).
        w_rot6d_ortho=float(cfg.loss.get("w_rot6d_ortho", 0.0)),
        w_fk_pos=float(cfg.loss.get("w_fk_pos", 0.0)),
        w_height_fk=float(cfg.loss.get("w_height_fk", 0.0)),
        w_self_consistency=float(cfg.loss.get("w_self_consistency", 0.0)),
        # R31 V7 anti-collapse losses (default 0 if config doesn't specify).
        w_moment_velocity=float(cfg.loss.get("w_moment_velocity", 0.0)),
        w_moment_value=float(cfg.loss.get("w_moment_value", 0.0)),
        w_yaw_aggregate=float(cfg.loss.get("w_yaw_aggregate", 0.0)),
        w_fk_pos_cm=float(cfg.loss.get("w_fk_pos_cm", 0.0)),
        fk_pos_cm_beta=float(cfg.loss.get("fk_pos_cm_beta", 1.0)),
        # R31 V8 — wrist FK supervision + init_pose F1/F2 + frame-0 consistency.
        w_wrist_fk_pos=float(cfg.loss.get("w_wrist_fk_pos", 0.0)),
        wrist_fk_target_joints=tuple(
            cfg.loss.get("wrist_fk_target_joints", [])
        ) or None,
        wrist_fk_joint_weights=tuple(
            cfg.loss.get("wrist_fk_joint_weights", [])
        ) or None,
        wrist_fk_contact_mode=str(cfg.loss.get("wrist_fk_contact_mode", "off")),
        wrist_fk_contact_weight=float(cfg.loss.get("wrist_fk_contact_weight", 4.0)),
        wrist_fk_add_velocity=bool(cfg.loss.get("wrist_fk_add_velocity", False)),
        wrist_fk_velocity_weight=float(cfg.loss.get("wrist_fk_velocity_weight", 0.5)),
        wrist_fk_beta_cm=float(cfg.loss.get("wrist_fk_beta_cm", 1.0)),
        w_init_pose_consistency=float(cfg.loss.get("w_init_pose_consistency", 0.0)),
        w_r36_raw_velocity=float(cfg.loss.get("w_r36_raw_velocity", 0.0)),
        w_r36_raw_acceleration=float(cfg.loss.get("w_r36_raw_acceleration", 0.0)),
        r36_raw_dynamics_channel_subset=tuple(
            cfg.loss.get("r36_raw_dynamics_channel_subset", [])
        ) or None,
        r36_raw_dynamics_normalize_by_gt_std=bool(
            cfg.loss.get("r36_raw_dynamics_normalize_by_gt_std", True),
        ),
        init_pose_dim=int(cfg.model.denoiser.get("init_pose_dim", 0)),
        vel_rot6d_weight=float(cfg.loss.get("vel_rot6d_weight", 1.0)),
        # R40 — per-channel x0 / vel weights + plan-invariant loss.
        # Empty list / missing key → all-ones (pre-R40 default behavior).
        x0_channel_weights=tuple(
            cfg.loss.get("x0_channel_weights", [])
        ) or None,
        vel_channel_weights=tuple(
            cfg.loss.get("vel_channel_weights", [])
        ) or None,
        w_r40_plan_invariant=float(cfg.loss.get("w_r40_plan_invariant", 0.0)),
        r40_plan_beta=float(cfg.loss.get("r40_plan_beta", 1.0)),
        r40_plan_component_weights=(
            dict(cfg.loss.get("r40_plan_component_weights", {}))
            or None
        ),
        use_min_snr_weighting=bool(
            cfg.loss.get("use_min_snr_weighting", True),
        ),
        min_snr_gamma=float(cfg.loss.get("min_snr_gamma", 5.0)),
    )

    if args.smoke_test:
        accelerator.print("Smoke test: running one batch.")
        batch = next(iter(train_loader))
        out = step_fn(model, batch, global_step=0)
        accelerator.print(
            f"loss = {out['loss'].item():.4f}  "
            f"mse_x0 = {out['mse_x0'].item():.4f}  "
            f"vel = {out['vel_mse'].item():.4f}  "
            f"yaw_sm = {out['yaw_smooth'].item():.4e}"
        )
        accelerator.print(
            f"  R31V2 — ortho={out['rot6d_ortho'].item():.4e}  "
            f"fk_pos={out['fk_pos'].item():.4e}  "
            f"height_fk={out['height_fk'].item():.4e}  "
            f"self_cons={out['self_consistency'].item():.4e}"
        )
        accelerator.print(
            f"  R31V7 — moment={out['moment_match'].item():.4e}  "
            f"yaw_agg={out['yaw_aggregate'].item():.4e}  "
            f"fk_pos_cm={out['fk_pos_cm'].item():.4e}"
        )
        accelerator.print(
            f"  R31V8 — wrist_fk={out['wrist_fk'].item():.4e}  "
            f"init_pose_cons={out['init_pose_consistency'].item():.4e}"
        )
        accelerator.print(
            f"  R36   raw_vel={out['r36_raw_velocity'].item():.4e}  "
            f"raw_acc={out['r36_raw_acceleration'].item():.4e}"
        )
        mse_x0_val = out["mse_x0"].item()
        r40_w_val = out["r40_plan_invariant_weighted"].item()
        ratio = r40_w_val / mse_x0_val if mse_x0_val > 0 else float("nan")
        accelerator.print(
            f"  R40 plan={out['r40_plan_invariant'].item():.4e}  "
            f"weighted={r40_w_val:.4e}  "
            f"weighted/mse={ratio:.3f}"
        )
        accelerator.backward(out["loss"])
        accelerator.print("Smoke test backward OK.")
        return

    run_training_loop(
        accelerator=accelerator,
        model=model,
        dataloader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        step_fn=step_fn,
        num_epochs=int(cfg.training.num_epochs),
        output_dir=cfg.output_dir,
        log_every=int(cfg.logging.get("log_every_n_steps", 50)),
        save_every_epochs=int(cfg.logging.get("save_every_n_epochs", 10)),
        max_grad_norm=float(cfg.training.get("max_grad_norm", 1.0)),
        extra_modules={"object_encoder": object_encoder},
        val_dataloader=val_loader,
        val_every_epochs=int(cfg.training.get("val_every_epochs", 0)),
        val_best_key=str(cfg.training.get("val_best_key", "mse_x0")),
    )


if __name__ == "__main__":
    main()
