"""Plan condition sensitivity test for the v10 plan-tokens Stage B.

Per analyses/piano_interaction_plan_pipeline_reframe_for_claude_code.md
§7.4 (mandatory): for a fixed clip, fixed noise seed, fixed text /
object / z_int, vary the plan tokens across:

    GT plan, zero plan, shuffled-time plan, wrong-clip plan,
    reversed-time plan, target-perturbed plan, part-swapped plan

and measure:

    far_unobserved_error_cm
    near_anchor_window_error_cm
    root_aligned_joint_error_cm
    motion-135 output delta
    contact realization distance

Pass criterion: GT plan must outperform zero / shuffled / wrong /
reversed plans on **unobserved** frames. If the only difference is at
anchor frames themselves, the model is still not routing plan
information.

Outputs:
- ``--output`` JSON with per-variant metrics
- ``--md`` Markdown summary table

Usage::

    python scripts/stage_b_generator/plan_condition_diagnostics.py \\
        --config configs/training/anchordiff_v10_plan_tokens_gt_overfit.yaml \\
        --ckpt   runs/training/stageB_anchordiff_v10_plan_tokens_gt_overfit/final.pt \\
        --output analyses/2026-05-10_v10_plan_tokens_gt_diagnostic.json \\
        --md     analyses/2026-05-10_v10_plan_tokens_gt_diagnostic_report.md
"""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from piano.data.dataset import (
    HOIDataset, collate_hoi, build_subject_split, extract_subject_id,
)
from piano.data.stage1_coarse_oracle import (
    extract_coarse_v1_batched,
    load_stage1_coarse_norm,
)
from piano.models.motion_anchordiff import (
    AnchorDenoiserConfig, AnchorDiffConfig, DiffusionConfig,
    MotionAnchorDiff, ZIntDims, pack_z_int,
)
from piano.models.object_encoder import ObjectEncoder
from piano.training.smpl_kinematics import (
    fk_from_global_rotations as _fk_from_global,
    rotation_6d_to_matrix as _rot6d_to_mat,
)
from piano.training.anchor_consistency_loss import (
    lift_object_local_to_world,
)
from piano.utils.clip_utils import encode_text_per_token, load_clip_text_encoder
from piano.utils.io_utils import load_json


# ---------------------------------------------------------------------------
# Plan variant constructors
# ---------------------------------------------------------------------------


def _gt_plan(plan: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in plan.items()}


def _zero_plan(plan: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """All anchors invalid; encoder + cross-attn see only padded slots."""
    out = {k: v.clone() for k, v in plan.items()}
    out["anchor_mask"][:] = False
    out["segment_mask"][:] = False
    out["anchor_part"][:] = 0.0
    out["anchor_target_local"][:] = 0.0
    out["anchor_target_world"][:] = 0.0
    out["anchor_conf"][:] = 0.0
    out["anchor_time"][:] = 0
    return out


def _shuffled_plan(plan: dict[str, torch.Tensor], seed: int) -> dict[str, torch.Tensor]:
    """Permute anchor_time within each clip while keeping anchor content."""
    out = {k: v.clone() for k, v in plan.items()}
    device = out["anchor_time"].device
    rng = torch.Generator(device="cpu").manual_seed(int(seed))
    B, K = out["anchor_time"].shape
    for b in range(B):
        valid = int(out["anchor_mask"][b].sum().item())
        if valid >= 2:
            perm = torch.randperm(valid, generator=rng).to(device)
            out["anchor_time"][b, :valid] = out["anchor_time"][b, perm]
    return out


def _reversed_plan(plan: dict[str, torch.Tensor], T: int) -> dict[str, torch.Tensor]:
    """Reverse anchor_time across the clip."""
    out = {k: v.clone() for k, v in plan.items()}
    out["anchor_time"] = (T - 1 - out["anchor_time"]).clamp(0, T - 1)
    out["segment_start"], out["segment_end"] = (
        (T - 1 - out["segment_end"]).clamp(0, T - 1),
        (T - 1 - out["segment_start"]).clamp(0, T - 1),
    )
    return out


def _wrong_clip_plan(
    plan_a: dict[str, torch.Tensor], plan_b: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Replace plan with a plan from a different clip."""
    return {k: v.clone() for k, v in plan_b.items()}


def _target_perturbed_plan(
    plan: dict[str, torch.Tensor], sigma_m: float = 0.10, seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Add Gaussian noise (in metres) to the anchor target_local + target_world."""
    out = {k: v.clone() for k, v in plan.items()}
    device = out["anchor_target_local"].device
    rng = torch.Generator(device="cpu").manual_seed(int(seed))
    n_local = torch.randn(out["anchor_target_local"].shape, generator=rng) * sigma_m
    n_world = torch.randn(out["anchor_target_world"].shape, generator=rng) * sigma_m
    out["anchor_target_local"] = (
        out["anchor_target_local"]
        + n_local.to(device=device, dtype=out["anchor_target_local"].dtype)
    )
    out["anchor_target_world"] = (
        out["anchor_target_world"]
        + n_world.to(device=device, dtype=out["anchor_target_world"].dtype)
    )
    return out


def _part_swapped_plan(plan: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rotate the per-anchor parts one slot to the right (L_hand → R_hand etc)."""
    out = {k: v.clone() for k, v in plan.items()}
    out["anchor_part"] = torch.roll(out["anchor_part"], shifts=1, dims=-1)
    out["anchor_target_local"] = torch.roll(
        out["anchor_target_local"], shifts=1, dims=-2,
    )
    out["anchor_target_world"] = torch.roll(
        out["anchor_target_world"], shifts=1, dims=-2,
    )
    return out


# ---------------------------------------------------------------------------
# Stage-1 route variant constructors (Round-22)
# ---------------------------------------------------------------------------


def _gt_route(stage1_coarse: torch.Tensor) -> torch.Tensor:
    return stage1_coarse.clone()


def _zero_route(stage1_coarse: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(stage1_coarse)


def _shuffled_route(stage1_coarse: torch.Tensor, seed: int) -> torch.Tensor:
    """Permute Stage-1 coarse frames within each clip."""
    out = stage1_coarse.clone()
    device = out.device
    rng = torch.Generator(device="cpu").manual_seed(int(seed))
    B, T, _ = out.shape
    for b in range(B):
        perm = torch.randperm(T, generator=rng).to(device)
        out[b] = out[b, perm]
    return out


def _wrong_clip_route(
    stage1_coarse: torch.Tensor,
    other_stage1_coarse: torch.Tensor,
) -> torch.Tensor:
    if other_stage1_coarse.shape == stage1_coarse.shape:
        return other_stage1_coarse.clone()
    # Defensive fallback for unusual diagnostic batches with different padded T.
    out = torch.zeros_like(stage1_coarse)
    T = min(stage1_coarse.shape[1], other_stage1_coarse.shape[1])
    out[:, :T] = other_stage1_coarse[:, :T]
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _compute_metrics(
    jpos_pred: torch.Tensor,         # (B, T, 22, 3)
    jpos_gt: torch.Tensor,           # (B, T, 22, 3)
    seq_mask: torch.Tensor,          # (B, T) bool
    anchor_time: torch.Tensor,       # (B, K) long
    anchor_mask: torch.Tensor,       # (B, K) bool
    anchor_part: torch.Tensor,       # (B, K, P) float
    anchor_target_world: torch.Tensor,  # (B, K, P, 3)
    part_to_joint: torch.Tensor,     # (P,) long
    window: int = 3,
) -> dict[str, float]:
    """Compute the plan-condition-sensitivity metrics for one rollout."""
    err = (jpos_pred - jpos_gt).pow(2).sum(-1).sqrt()                       # (B, T, 22)
    per_frame = err.mean(-1)                                                 # (B, T)
    # Root-aligned: subtract per-frame root translation from both before err.
    root_pred = jpos_pred[..., 0:1, :]
    root_gt = jpos_gt[..., 0:1, :]
    err_ra = (
        (jpos_pred - root_pred - (jpos_gt - root_gt))
        .pow(2).sum(-1).sqrt()
    ).mean(-1)                                                                # (B, T)

    B, T = per_frame.shape
    K = anchor_time.shape[1]
    device = per_frame.device

    # Window mask: ±window frames around any valid anchor
    t_grid = torch.arange(T, device=device).view(1, 1, T)
    a_t = anchor_time.view(B, K, 1)
    near = (
        (a_t - window <= t_grid)
        & (t_grid <= a_t + window)
        & anchor_mask.view(B, K, 1)
    )
    window_mask = near.any(dim=1) & seq_mask                                  # (B, T)
    valid_mask = seq_mask
    # Anchor-frame mask: t == anchor_time exactly
    at_anchor = (
        (a_t == t_grid)
        & anchor_mask.view(B, K, 1)
    ).any(dim=1) & seq_mask                                                   # (B, T)
    # Far-unobserved: not within window of any anchor
    far_mask = valid_mask & ~window_mask

    def _masked_mean(x: torch.Tensor, m: torch.Tensor) -> float:
        m_f = m.float()
        denom = m_f.sum().clamp_min(1.0)
        return float((x * m_f).sum() / denom)

    g_err = _masked_mean(per_frame, valid_mask) * 100.0          # cm
    g_err_ra = _masked_mean(err_ra, valid_mask) * 100.0
    obs_err = _masked_mean(per_frame, at_anchor) * 100.0
    near_err = _masked_mean(per_frame, window_mask) * 100.0
    far_err = _masked_mean(per_frame, far_mask) * 100.0
    far_err_ra = _masked_mean(err_ra, far_mask) * 100.0

    # Local joint velocity error (root-aligned vel)
    if T >= 2:
        vel_pred = jpos_pred[:, 1:] - jpos_pred[:, :-1]
        vel_gt = jpos_gt[:, 1:] - jpos_gt[:, :-1]
        vel_err = (vel_pred - vel_gt).pow(2).sum(-1).sqrt().mean(-1)          # (B, T-1)
        vw = window_mask[:, 1:] & window_mask[:, :-1]
        vfar = far_mask[:, 1:] & far_mask[:, :-1]
        local_vel_near = _masked_mean(vel_err, vw) * 100.0
        local_vel_far = _masked_mean(vel_err, vfar) * 100.0
    else:
        local_vel_near = 0.0
        local_vel_far = 0.0

    # Transition jump: velocity at anchor crossings
    trans_jump = local_vel_near    # the spec uses near-window vel

    # Plan anchor contact realization (per spec §E: report model, GT, and
    # GT-normalised difference + per-part breakdown so the structural
    # metric floor doesn't silently dominate the global mean).
    contact_realization = 0.0
    gt_contact_realization = 0.0
    contact_realization_minus_gt = 0.0
    per_part_model: dict[str, float] = {}
    per_part_gt: dict[str, float] = {}
    per_part_diff: dict[str, float] = {}
    part_names = ("L_hand", "R_hand", "L_foot", "R_foot", "pelvis")
    if anchor_mask.any():
        t_idx = (
            anchor_time.clamp(0, T - 1)
            .view(B, K, 1, 1)
            .expand(B, K, 22, 3)
        )
        fk_at_anchor_pred = torch.gather(jpos_pred, 1, t_idx)                # (B, K, 22, 3)
        fk_at_anchor_gt = torch.gather(jpos_gt, 1, t_idx)                    # (B, K, 22, 3)
        joint_at_part_pred = fk_at_anchor_pred[:, :, part_to_joint, :]       # (B, K, P, 3)
        joint_at_part_gt = fk_at_anchor_gt[:, :, part_to_joint, :]
        err_pred = (joint_at_part_pred - anchor_target_world).pow(2).sum(-1).sqrt()  # (B, K, P)
        err_gt = (joint_at_part_gt - anchor_target_world).pow(2).sum(-1).sqrt()
        act = anchor_mask.unsqueeze(-1).float() * anchor_part                # (B, K, P)
        denom = act.sum().clamp_min(1.0)
        contact_realization = float((err_pred * act).sum() / denom) * 100.0
        gt_contact_realization = float((err_gt * act).sum() / denom) * 100.0
        contact_realization_minus_gt = contact_realization - gt_contact_realization

        # Per-part breakdown (spec §E: report foot anchors separately).
        P = anchor_part.shape[-1]
        for p in range(min(P, len(part_names))):
            act_p = act[..., p]
            denom_p = act_p.sum().clamp_min(1.0)
            if act_p.sum() > 0.5:
                per_part_model[part_names[p]] = float(
                    (err_pred[..., p] * act_p).sum() / denom_p,
                ) * 100.0
                per_part_gt[part_names[p]] = float(
                    (err_gt[..., p] * act_p).sum() / denom_p,
                ) * 100.0
                per_part_diff[part_names[p]] = (
                    per_part_model[part_names[p]] - per_part_gt[part_names[p]]
                )
            else:
                per_part_model[part_names[p]] = 0.0
                per_part_gt[part_names[p]] = 0.0
                per_part_diff[part_names[p]] = 0.0

    return {
        "global_joint_error_cm": g_err,
        "root_aligned_joint_error_cm": g_err_ra,
        "observed_anchor_frame_error_cm": obs_err,
        "near_anchor_window_error_cm": near_err,
        "far_unobserved_error_cm": far_err,
        "far_unobserved_root_aligned_error_cm": far_err_ra,
        "local_vel_near_anchor_cm_per_frame": local_vel_near,
        "local_vel_far_unobs_cm_per_frame": local_vel_far,
        "transition_local_vel_jump_cm_per_frame": trans_jump,
        "plan_anchor_contact_realization_cm": contact_realization,
        "gt_anchor_realization_cm": gt_contact_realization,
        "anchor_realization_minus_gt_cm": contact_realization_minus_gt,
        "anchor_realization_per_part_model": per_part_model,
        "anchor_realization_per_part_gt": per_part_gt,
        "anchor_realization_per_part_diff": per_part_diff,
    }


def _append_route_consistency_metrics(
    metrics: dict[str, float],
    *,
    x0_pred: torch.Tensor,
    rest_offsets: torch.Tensor,
    seq_mask: torch.Tensor,
    cond_stage1_coarse: torch.Tensor | None,
    stage1_norm: tuple[torch.Tensor, torch.Tensor] | None,
) -> None:
    """Measure whether generated motion realizes the supplied route condition."""
    if cond_stage1_coarse is None or stage1_norm is None:
        return
    mean_t, std_t = stage1_norm
    pred_raw = extract_coarse_v1_batched(x0_pred, rest_offsets)
    pred_norm = (pred_raw - mean_t) / std_t
    err = (pred_norm - cond_stage1_coarse).pow(2).sum(-1).sqrt()
    mask_f = seq_mask.float()
    metrics["stage1_coarse_norm_l2"] = float(
        (err * mask_f).sum() / mask_f.sum().clamp_min(1.0)
    )


# ---------------------------------------------------------------------------
# Loader / model
# ---------------------------------------------------------------------------


def _build_dataset(cfg, bucket: str, augment: bool) -> Subset | torch.utils.data.ConcatDataset:
    from torch.utils.data import ConcatDataset
    subj_filter: set | None = None
    subj_cfg = cfg.data.get("subject_split")
    if subj_cfg is not None and subj_cfg.get("enabled", False):
        keys: set[tuple[str, str]] = set()
        for entry in cfg.data.datasets:
            meta = load_json(Path(entry.root) / "metadata_clean.json")
            for m in meta:
                sid = extract_subject_id(Path(entry.root).name, m.get("seq_id", ""))
                if sid is not None:
                    keys.add((Path(entry.root).name, sid))
        splits = build_subject_split(
            sorted(keys),
            train_pct=subj_cfg.train_pct,
            val_pct=subj_cfg.val_pct,
            seed=subj_cfg.seed,
        )
        subj_filter = splits[bucket]
    pseudo_label_subdir = cfg.data.get("pseudo_label_subdir", None)
    datasets = []
    for entry in cfg.data.datasets:
        sub_dir = (
            str(Path(entry.root) / pseudo_label_subdir) if pseudo_label_subdir else None
        )
        ds = HOIDataset(
            root=entry.root,
            pseudo_label_dir=sub_dir,
            max_seq_length=int(cfg.data.max_seq_length),
            subject_id_filter=subj_filter,
            subsample_n_per_object=cfg.data.get("subsample_n_per_object", None),
            subsample_seed=int(cfg.data.get("subsample_seed", 42)),
            support_collapse_hand_support=bool(
                cfg.data.get("support_collapse_hand_support", True)
            ),
            surface_obj_pose=True,
            force_world_frame=bool(cfg.data.get("force_world_frame", False)),
            motion_representation=str(cfg.data.get("motion_representation", "motion_263")),
        )
        datasets.append(ds)
    return ConcatDataset(datasets)


def _build_model(cfg, device: torch.device) -> tuple[MotionAnchorDiff, ObjectEncoder, ZIntDims]:
    z_dims = ZIntDims(
        num_parts=int(cfg.model.z_int.num_parts),
        phase_classes=int(cfg.model.z_int.phase_classes),
        support_classes=int(cfg.model.z_int.support_classes),
    )
    denoiser_cfg = AnchorDenoiserConfig(
        motion_dim=int(cfg.model.denoiser.motion_dim),
        z_int=z_dims,
        object_traj_dim=int(cfg.model.denoiser.object_traj_dim),
        init_pose_dim=int(cfg.model.denoiser.init_pose_dim),
        text_dim=int(cfg.model.denoiser.text_dim),
        object_token_dim=int(cfg.model.denoiser.object_token_dim),
        object_num_tokens=int(cfg.model.denoiser.object_num_tokens),
        cond_motion_dim=int(cfg.model.denoiser.get("cond_motion_dim", 0)),
        cond_motion_output_skip=bool(cfg.model.denoiser.get("cond_motion_output_skip", False)),
        cfg_drop_cond_motion=bool(cfg.model.denoiser.get("cfg_drop_cond_motion", False)),
        cond_motion_xt_inject=bool(cfg.model.denoiser.get("cond_motion_xt_inject", False)),
        use_interaction_plan=bool(cfg.model.denoiser.get("use_interaction_plan", False)),
        plan_k_max=int(cfg.model.denoiser.get("plan_k_max", 12)),
        plan_s_max=int(cfg.model.denoiser.get("plan_s_max", 12)),
        plan_num_anchor_types=int(cfg.model.denoiser.get("plan_num_anchor_types", 5)),
        plan_num_parts=int(cfg.model.denoiser.get("plan_num_parts", 5)),
        plan_use_segment_tokens=bool(cfg.model.denoiser.get("plan_use_segment_tokens", False)),
        plan_use_context_hint=bool(cfg.model.denoiser.get("plan_use_context_hint", True)),
        plan_d_hint=int(cfg.model.denoiser.get("plan_d_hint", 32)),
        plan_d_time_embed=int(cfg.model.denoiser.get("plan_d_time_embed", 64)),
        cfg_drop_plan=bool(cfg.model.denoiser.get("cfg_drop_plan", False)),
        plan_per_part_tokens=bool(cfg.model.denoiser.get("plan_per_part_tokens", False)),
        plan_context_hint_mode=str(cfg.model.denoiser.get("plan_context_hint_mode", "time_only")),
        use_dit_block=bool(cfg.model.denoiser.get("use_dit_block", False)),
        dit_block_use_plan_pool_in_cond=bool(
            cfg.model.denoiser.get("dit_block_use_plan_pool_in_cond", True)
        ),
        use_v13_dynhead=bool(cfg.model.denoiser.get("use_v13_dynhead", False)),
        v13_dynhead_gamma_init=float(cfg.model.denoiser.get("v13_dynhead_gamma_init", 0.1)),
        v13_dynhead_learnable_gamma=bool(
            cfg.model.denoiser.get("v13_dynhead_learnable_gamma", True)
        ),
        use_v13_temporal_conv=bool(cfg.model.denoiser.get("use_v13_temporal_conv", False)),
        v13_temporal_conv_kernel=int(cfg.model.denoiser.get("v13_temporal_conv_kernel", 5)),
        use_self_conditioning=bool(cfg.model.denoiser.get("use_self_conditioning", False)),
        self_conditioning_prob=float(cfg.model.denoiser.get("self_conditioning_prob", 0.0)),
        self_conditioning_mode=str(cfg.model.denoiser.get("self_conditioning_mode", "standard")),
        self_conditioning_t_max=int(cfg.model.denoiser.get("self_conditioning_t_max", 700)),
        self_conditioning_zero_init=bool(
            cfg.model.denoiser.get("self_conditioning_zero_init", True)
        ),
        stage1_coarse_dim=int(cfg.model.denoiser.get("stage1_coarse_dim", 0)),
        cfg_drop_stage1_coarse=bool(
            cfg.model.denoiser.get("cfg_drop_stage1_coarse", False)
        ),
        d_model=int(cfg.model.denoiser.d_model),
        n_layers=int(cfg.model.denoiser.n_layers),
        n_heads=int(cfg.model.denoiser.n_heads),
        ff_mult=int(cfg.model.denoiser.ff_mult),
        dropout=float(cfg.model.denoiser.dropout),
        max_seq_length=int(cfg.data.max_seq_length),
    )
    diff_cfg = DiffusionConfig(
        num_steps=int(cfg.model.diffusion.num_steps),
        schedule=str(cfg.model.diffusion.schedule),
        objective=str(cfg.model.diffusion.get("objective", "ddpm")),
        prediction_target=str(cfg.model.diffusion.get("prediction_target", "x0")),
        rf_eps_time=float(cfg.model.diffusion.get("rf_eps_time", 0.05)),
        rf_time_schedule=str(cfg.model.diffusion.get("rf_time_schedule", "uniform")),
        rf_denoiser_p_mean=float(cfg.model.diffusion.get("rf_denoiser_p_mean", -1.5)),
        rf_denoiser_p_std=float(cfg.model.diffusion.get("rf_denoiser_p_std", 0.8)),
        rf_denoiser_noise_scale=float(cfg.model.diffusion.get("rf_denoiser_noise_scale", 1.0)),
        rf_num_sampling_steps=int(cfg.model.diffusion.get("rf_num_sampling_steps", 100)),
        rf_sampler_type=str(cfg.model.diffusion.get("rf_sampler_type", "rectified_flow_ode")),
        rf_sde_gamma=float(cfg.model.diffusion.get("rf_sde_gamma", 0.0)),
    )
    model = MotionAnchorDiff(
        AnchorDiffConfig(
            diffusion=diff_cfg, denoiser=denoiser_cfg,
            cfg_drop_prob=float(cfg.model.cfg_drop_prob),
        )
    ).to(device).eval()
    object_encoder = ObjectEncoder(
        num_input_points=int(cfg.model.object_encoder.num_input_points),
        num_output_tokens=int(cfg.model.object_encoder.num_output_tokens),
        feature_dim=int(cfg.model.object_encoder.feature_dim),
    ).to(device).eval()
    return model, object_encoder, z_dims


def _stage1_norm_for_cfg(
    cfg,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    stage1_coarse_dim = int(cfg.model.denoiser.get("stage1_coarse_dim", 0))
    if stage1_coarse_dim <= 0:
        return None
    cache_root = cfg.data.get("stage1_coarse_cache_root", None)
    if cache_root is None:
        raise ValueError(
            "stage1_coarse_dim > 0 requires data.stage1_coarse_cache_root "
            "in the YAML config (directory containing normalization_train.json)."
        )
    mean, std = load_stage1_coarse_norm(str(cache_root))
    if mean.shape != (stage1_coarse_dim,) or std.shape != (stage1_coarse_dim,):
        raise ValueError(
            f"Stage-1 norm stats shape mismatch: mean={mean.shape}, "
            f"std={std.shape}, stage1_coarse_dim={stage1_coarse_dim}"
        )
    mean_t = torch.from_numpy(mean).to(device=device, dtype=torch.float32).view(1, 1, -1)
    std_t = torch.from_numpy(std).to(device=device, dtype=torch.float32).view(1, 1, -1)
    return mean_t, std_t


def _build_object_traj_for_cfg(
    *,
    cfg,
    obj_com: torch.Tensor,
    obj_rot6d: torch.Tensor,
    contact_target_xyz: torch.Tensor,
    obj_pos_world: torch.Tensor,
    obj_rot_world: torch.Tensor,
) -> torch.Tensor:
    """Mirror train_anchordiff._build_object_traj for diagnostic conds."""
    object_traj_dim = int(cfg.model.denoiser.object_traj_dim)
    components = [obj_com, obj_rot6d]
    if object_traj_dim >= 24:
        B, T = obj_com.shape[:2]
        target_world = lift_object_local_to_world(
            contact_target_xyz, obj_pos_world, obj_rot_world,
        ).reshape(B, T, -1)
        components.append(target_world)
    object_traj = torch.cat(components, dim=-1)
    if object_traj.shape[-1] != object_traj_dim:
        raise ValueError(
            f"object_traj_dim={object_traj_dim} but diagnostic built "
            f"{object_traj.shape[-1]} dims"
        )
    if (
        bool(cfg.model.get("zero_dense_contact_target_for_stageB", False))
        and object_traj.shape[-1] >= 24
    ):
        object_traj = object_traj.clone()
        object_traj[..., 9:] = 0.0
    return object_traj


def _build_cond(
    batch: dict, model: MotionAnchorDiff, object_encoder: ObjectEncoder,
    clip_model, z_dims: ZIntDims, cfg, device: torch.device,
    stage1_norm: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[dict, int]:
    """Mirror the trainer's cond construction (excluding plan variants)."""
    motion = batch["motion"].to(device)
    joints = batch["joints"].to(device)
    object_pc = batch["object_pc"].to(device)
    contact_state = batch["contact_state"].to(device)
    contact_target_xyz = batch["contact_target_xyz"].to(device)
    phase = batch["phase"].to(device)
    support = batch["support"].to(device)
    obj_com = batch["obj_com_canonical"].to(device)
    obj_rot6d = batch["obj_rot6d_canonical"].to(device)
    obj_pos_world = batch["object_positions"].to(device)
    obj_rot_world = batch["object_rotations"].to(device)
    seq_len = batch["seq_len"].to(device)
    B, T, _ = motion.shape

    import torch.nn.functional as F
    phase_soft = F.one_hot(phase.clamp_min(0).long(), num_classes=z_dims.phase_classes).float()
    support_soft = F.one_hot(support.clamp_min(0).long(), num_classes=z_dims.support_classes).float()

    # Mirror the trainer's fine-grained Stage-B zeroing (per
    # claude_code_v11_after_full_frozen_fix_handoff.md §C). Without this
    # the diagnostic feeds the WEAK_DENSE / PLAN_ONLY checkpoints OOD
    # signal in channels the model was trained to ignore.
    if bool(cfg.model.get("zero_contact_state_for_stageB", False)):
        contact_state = torch.zeros_like(contact_state)
    if bool(cfg.model.get("zero_contact_target_for_stageB", False)):
        contact_target_xyz_for_z = torch.zeros_like(contact_target_xyz)
    else:
        contact_target_xyz_for_z = contact_target_xyz
    if bool(cfg.model.get("zero_phase_for_stageB", False)):
        phase_soft = torch.zeros_like(phase_soft)
    if bool(cfg.model.get("zero_support_for_stageB", False)):
        support_soft = torch.zeros_like(support_soft)
    z_int = pack_z_int(contact_state, contact_target_xyz_for_z, phase_soft, support_soft, z_dims)
    if bool(cfg.model.get("zero_z_int_for_stageB", False)):
        z_int = torch.zeros_like(z_int)

    object_traj = _build_object_traj_for_cfg(
        cfg=cfg,
        obj_com=obj_com,
        obj_rot6d=obj_rot6d,
        contact_target_xyz=contact_target_xyz_for_z,
        obj_pos_world=obj_pos_world,
        obj_rot_world=obj_rot_world,
    )

    init_pose = joints[:, 0, :, :].reshape(B, -1)
    text_features, _ = encode_text_per_token(clip_model, batch["text"], device)
    obj_tokens = object_encoder(object_pc)
    cond = {
        "z_int": z_int,
        "object_world_traj": object_traj,
        "init_pose": init_pose,
        "text": text_features.float(),
        "object_tokens": obj_tokens,
    }
    stage1_coarse_dim = int(cfg.model.denoiser.get("stage1_coarse_dim", 0))
    if stage1_coarse_dim > 0:
        if str(cfg.data.get("motion_representation", "motion_263")) != "smpl_pose_135_plan":
            raise ValueError(
                "stage1_coarse_dim > 0 requires "
                "data.motion_representation='smpl_pose_135_plan'."
            )
        if "rest_offsets" not in batch:
            raise KeyError(
                "stage1_coarse_dim > 0 requires batch['rest_offsets'] for "
                "Coarse-v1 oracle extraction."
            )
        if stage1_norm is None:
            stage1_norm = _stage1_norm_for_cfg(cfg, device)
        assert stage1_norm is not None
        mean_t, std_t = stage1_norm
        coarse_raw = extract_coarse_v1_batched(
            motion=motion.float(),
            rest_offsets=batch["rest_offsets"].to(device).float(),
        )
        if coarse_raw.shape[-1] != stage1_coarse_dim:
            raise ValueError(
                f"Oracle Coarse-v1 dim {coarse_raw.shape[-1]} != "
                f"stage1_coarse_dim={stage1_coarse_dim}"
            )
        cond["stage1_coarse"] = (coarse_raw - mean_t) / std_t
    return cond, T


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--md", type=Path, default=None)
    parser.add_argument("--clip-idx", type=int, default=0)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bucket", default="train", choices=["train", "val"])
    parser.add_argument(
        "--render-dir", type=Path, default=None,
        help="Optional directory to write predicted-vs-GT MP4s for each "
             "rendered plan variant. If unset, no rendering is done.",
    )
    parser.add_argument(
        "--render-variants", type=str, default="gt,zero,part_swapped",
        help="Comma-separated list of plan variants to render (used only "
             "when --render-dir is set). Choices: gt, zero, shuffled_time, "
             "wrong_clip, reversed_time, target_perturbed, part_swapped, all.",
    )
    parser.add_argument("--fps", type=float, default=20.0)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build dataset
    train_dataset = _build_dataset(cfg, args.bucket, augment=False)
    overfit_n = int(cfg.data.get("overfit_n_clips", 0))
    if overfit_n > 0:
        train_dataset = Subset(
            train_dataset, list(range(min(overfit_n, len(train_dataset)))),
        )
    loader = DataLoader(
        train_dataset, batch_size=1, shuffle=False,
        collate_fn=collate_hoi, num_workers=0,
    )

    # Pick the target clip for sensitivity
    main_batch = None
    secondary_batch = None
    for i, batch in enumerate(loader):
        if i == args.clip_idx:
            main_batch = batch
        elif main_batch is not None and secondary_batch is None:
            secondary_batch = batch
            break
    if main_batch is None:
        raise RuntimeError("Could not find main clip for diagnostic")
    if secondary_batch is None:
        # Fall back to a synthesised "second clip" by reversing main_batch
        # plan times — at least the wrong-clip variant is non-trivial.
        secondary_batch = main_batch

    # Model + ckpt
    model, object_encoder, z_dims = _build_model(cfg, device)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model_state = state.get("model", state)
    model.load_state_dict(model_state)
    if "object_encoder" in state:
        object_encoder.load_state_dict(state["object_encoder"])
    elif "extra_modules" in state and "object_encoder" in state["extra_modules"]:
        object_encoder.load_state_dict(state["extra_modules"]["object_encoder"])

    clip_model = load_clip_text_encoder(
        device=device,
        model_name=str(cfg.model.text_encoder.clip_version),
        download_root=str(cfg.model.text_encoder.get("download_root", "cache/clip")),
    )

    stage1_norm = _stage1_norm_for_cfg(cfg, device)
    cond_main, T = _build_cond(
        main_batch, model, object_encoder, clip_model, z_dims, cfg, device,
        stage1_norm=stage1_norm,
    )
    cond_sec, _ = _build_cond(
        secondary_batch, model, object_encoder, clip_model, z_dims, cfg, device,
        stage1_norm=stage1_norm,
    )

    # Plan dicts (pull plan_* fields back into the schema the encoder wants)
    def _extract_plan(batch: dict) -> dict[str, torch.Tensor]:
        plan_keys = [
            "anchor_time", "anchor_part", "anchor_target_local",
            "anchor_target_world", "anchor_type", "anchor_phase",
            "anchor_support", "anchor_conf", "anchor_mask",
            "segment_start", "segment_end", "segment_part",
            "segment_target_summary_local", "segment_phase",
            "segment_support", "segment_conf", "segment_mask",
        ]
        return {k: batch[f"plan_{k}"].to(device) for k in plan_keys}

    plan_gt = _extract_plan(main_batch)
    plan_other = _extract_plan(secondary_batch)

    variants = {
        "gt": _gt_plan(plan_gt),
        "zero": _zero_plan(plan_gt),
        "shuffled_time": _shuffled_plan(plan_gt, seed=args.seed),
        "wrong_clip": _wrong_clip_plan(plan_gt, plan_other),
        "reversed_time": _reversed_plan(plan_gt, T=T),
        "target_perturbed": _target_perturbed_plan(plan_gt, sigma_m=0.10, seed=args.seed),
        "part_swapped": _part_swapped_plan(plan_gt),
    }

    # SMPL-22 part-to-joint map (must match dataset / trainer)
    part_to_joint = torch.tensor([20, 21, 10, 11, 0], dtype=torch.long, device=device)

    # GT FK
    motion_gt = main_batch["motion"].to(device)
    rest_offsets = main_batch["rest_offsets"].to(device).float()
    seq_len = main_batch["seq_len"].to(device)
    seq_idx = torch.arange(T, device=device).unsqueeze(0)
    seq_mask = (seq_idx < seq_len.unsqueeze(1))                              # (B, T)
    joints_gt = main_batch["joints"].to(device).float()

    results: dict[str, dict] = {}
    motion_outputs: dict[str, torch.Tensor] = {}

    def _run_one_variant(
        *,
        plan: dict[str, torch.Tensor],
        stage1_coarse: torch.Tensor | None = None,
    ) -> tuple[dict[str, float], torch.Tensor]:
        torch.manual_seed(args.seed)
        cond = {**cond_main, "interaction_plan": plan}
        if stage1_coarse is not None:
            cond["stage1_coarse"] = stage1_coarse
        with torch.no_grad():
            x0_pred = model.sample(
                cond=cond, seq_length=T, cfg_scale=args.cfg_scale,
                replacement="none", output_skip=False,
            )                                                                 # (B, T, 135)
        # FK
        rot_6d = x0_pred[..., :132].view(1, T, 22, 6).float()
        root_world = x0_pred[..., 132:135].float()
        rot_mat = _rot6d_to_mat(rot_6d)
        rest_per_frame = rest_offsets.unsqueeze(1).expand(1, T, 22, 3)
        jpos_pred = _fk_from_global(rot_mat, rest_per_frame, root_world)
        metrics = _compute_metrics(
            jpos_pred=jpos_pred, jpos_gt=joints_gt,
            seq_mask=seq_mask, anchor_time=plan["anchor_time"],
            anchor_mask=plan["anchor_mask"], anchor_part=plan["anchor_part"],
            anchor_target_world=plan["anchor_target_world"],
            part_to_joint=part_to_joint, window=3,
        )
        _append_route_consistency_metrics(
            metrics,
            x0_pred=x0_pred,
            rest_offsets=rest_offsets,
            seq_mask=seq_mask,
            cond_stage1_coarse=cond.get("stage1_coarse", None),
            stage1_norm=stage1_norm,
        )
        return metrics, x0_pred

    for name, plan in variants.items():
        metrics, x0_pred = _run_one_variant(plan=plan)
        results[name] = metrics
        motion_outputs[name] = x0_pred.cpu()

    # Cross-variant: motion-135 output delta vs GT
    base = motion_outputs["gt"]
    for name, x0 in motion_outputs.items():
        delta = (x0 - base).pow(2).sum(-1).sqrt().mean().item()
        results[name]["motion_135_delta_vs_gt"] = float(delta)

    # Round-22 route sensitivity: hold plan fixed to GT and vary
    # cond["stage1_coarse"]. Skipped automatically for pre-R22 configs.
    route_results: dict[str, dict] = {}
    route_motion_outputs: dict[str, torch.Tensor] = {}
    conflict_results: dict[str, dict] = {}
    conflict_motion_outputs: dict[str, torch.Tensor] = {}
    if "stage1_coarse" in cond_main:
        route_variants = {
            "gt_route": _gt_route(cond_main["stage1_coarse"]),
            "zero_route": _zero_route(cond_main["stage1_coarse"]),
            "shuffled_route": _shuffled_route(cond_main["stage1_coarse"], seed=args.seed),
            "wrong_clip_route": _wrong_clip_route(
                cond_main["stage1_coarse"], cond_sec["stage1_coarse"],
            ),
        }
        for name, route in route_variants.items():
            metrics, x0_pred = _run_one_variant(plan=_gt_plan(plan_gt), stage1_coarse=route)
            route_results[name] = metrics
            route_motion_outputs[name] = x0_pred.cpu()
        route_base = route_motion_outputs["gt_route"]
        for name, x0 in route_motion_outputs.items():
            delta = (x0 - route_base).pow(2).sum(-1).sqrt().mean().item()
            route_results[name]["motion_135_delta_vs_gt_route"] = float(delta)

        conflict_variants = {
            "gt_plan_wrong_route": (_gt_plan(plan_gt), route_variants["wrong_clip_route"]),
            "wrong_plan_gt_route": (_wrong_clip_plan(plan_gt, plan_other), route_variants["gt_route"]),
            "target_perturbed_plan_gt_route": (
                _target_perturbed_plan(plan_gt, sigma_m=0.10, seed=args.seed),
                route_variants["gt_route"],
            ),
        }
        for name, (plan, route) in conflict_variants.items():
            metrics, x0_pred = _run_one_variant(plan=plan, stage1_coarse=route)
            conflict_results[name] = metrics
            conflict_motion_outputs[name] = x0_pred.cpu()
        conflict_base = route_motion_outputs["gt_route"]
        for name, x0 in conflict_motion_outputs.items():
            delta = (x0 - conflict_base).pow(2).sum(-1).sqrt().mean().item()
            conflict_results[name]["motion_135_delta_vs_gt_plan_gt_route"] = float(delta)

    # ---------------------------------------------------------------------
    # Optional rendering: write per-variant MP4 next to the metrics
    # ---------------------------------------------------------------------
    if args.render_dir is not None:
        from piano.inference.visualize_motion import render_motion_video

        args.render_dir.mkdir(parents=True, exist_ok=True)
        valid_T = int(seq_len[0].item())
        seq_id = main_batch["seq_id"][0]
        subset = main_batch["subset"][0]
        text = main_batch["text"][0]

        # Decide which variants to render
        render_variants = args.render_variants.strip()
        if render_variants == "all":
            variant_names = list(motion_outputs.keys())
        else:
            variant_names = [v.strip() for v in render_variants.split(",") if v.strip()]
            unknown = set(variant_names) - set(motion_outputs.keys())
            if unknown:
                raise ValueError(
                    f"Unknown render variants: {unknown}. "
                    f"Available: {list(motion_outputs.keys())}"
                )

        # Object overlay (same across variants — only the motion changes)
        obj_pos_np = main_batch["object_positions"].squeeze(0).cpu().numpy()[:valid_T]
        obj_rot_np = main_batch["object_rotations"].squeeze(0).cpu().numpy()[:valid_T]
        obj_pc_np = main_batch["object_pc"].squeeze(0).cpu().numpy()
        joints_gt_np = joints_gt.squeeze(0).cpu().numpy()[:valid_T]

        # GT video — write once (overwrite if it exists; same content as
        # the v10 visualize step).
        gt_out = args.render_dir / f"{subset}_{seq_id}_gt.mp4"
        gt_title = f"{subset}/{seq_id}\n[GT]\ntext: {text[:80]}"
        print(f"  rendering GT → {gt_out.name}")
        render_motion_video(
            joints=joints_gt_np,
            output_path=gt_out,
            fps=args.fps,
            title=gt_title,
            object_positions=obj_pos_np,
            object_rotations=obj_rot_np,
            object_pc=obj_pc_np,
        )

        # Per-variant predicted video. FK from the cached x0 sample
        # (already computed under the fixed seed inside the metrics loop
        # above), so the rendered motion is bit-exact what the metrics
        # were measured on.
        rest_offsets_render = main_batch["rest_offsets"].to(device).float()
        for name in variant_names:
            x0_pred = motion_outputs[name].to(device)
            rot_6d = x0_pred[..., :132].view(1, T, 22, 6).float()
            root_world = x0_pred[..., 132:135].float()
            rot_mat = _rot6d_to_mat(rot_6d)
            rest_per_frame = rest_offsets_render.unsqueeze(1).expand(1, T, 22, 3)
            jpos_pred = _fk_from_global(rot_mat, rest_per_frame, root_world)
            jpos_pred_np = jpos_pred.squeeze(0).cpu().numpy()[:valid_T]

            far_err = results[name]["far_unobserved_error_cm"]
            delta = results[name]["motion_135_delta_vs_gt"]
            pred_out = args.render_dir / f"{subset}_{seq_id}_predicted_{name}.mp4"
            pred_title = (
                f"{subset}/{seq_id}\n[v10 plan_variant={name}]\n"
                f"far-unobs={far_err:.2f} cm  Δmotion-135={delta:.3f}"
            )
            print(f"  rendering {name:18s} → {pred_out.name}")
            render_motion_video(
                joints=jpos_pred_np,
                output_path=pred_out,
                fps=args.fps,
                title=pred_title,
                object_positions=obj_pos_np,
                object_rotations=obj_rot_np,
                object_pc=obj_pc_np,
            )
        print(f"\nRendered {len(variant_names)} variant(s) + GT to {args.render_dir}")

    # ---------------------------------------------------------------------
    # Pass / fail evaluation per spec §9.2
    # ---------------------------------------------------------------------
    far_gt = results["gt"]["far_unobserved_error_cm"]
    far_zero = results["zero"]["far_unobserved_error_cm"]
    far_wrong = results["wrong_clip"]["far_unobserved_error_cm"]
    pass_gate_unobs = (far_zero - far_gt) >= 5.0 and (far_wrong - far_gt) >= 5.0
    pass_gate_anchor = results["gt"]["plan_anchor_contact_realization_cm"] < 20.0
    pass_gate_transition = (
        results["gt"]["transition_local_vel_jump_cm_per_frame"] < 3.0
    )

    summary = {
        "config": str(args.config),
        "ckpt": str(args.ckpt),
        "cfg_scale": args.cfg_scale,
        "seed": args.seed,
        "T": T,
        "pass_gates": {
            "gt_better_than_zero_and_wrong_unobs_5cm": bool(pass_gate_unobs),
            "anchor_contact_realization_under_20cm": bool(pass_gate_anchor),
            "transition_vel_jump_under_3cm_per_frame": bool(pass_gate_transition),
        },
        "metrics": results,
        "route_metrics": route_results,
        "conflict_metrics": conflict_results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote JSON to {args.output}")

    # Markdown
    if args.md is not None:
        md_lines: list[str] = []
        md_lines.append("# Stage-B plan condition sensitivity")
        md_lines.append(f"**Date:** 2026-05-14")
        md_lines.append(f"**Config:** `{args.config}`")
        md_lines.append(f"**Checkpoint:** `{args.ckpt}`")
        md_lines.append(f"**cfg_scale:** {args.cfg_scale}    **seed:** {args.seed}    **T:** {T}\n")
        md_lines.append("## Pass gates (per spec §9.2)")
        md_lines.append(
            f"- GT plan ≥ 5 cm better than zero+wrong on far-unobs: "
            f"{'✓' if pass_gate_unobs else '✗'}  "
            f"(gt={far_gt:.2f}, zero={far_zero:.2f}, wrong={far_wrong:.2f})"
        )
        md_lines.append(
            f"- Anchor contact realisation < 20 cm: "
            f"{'✓' if pass_gate_anchor else '✗'}  "
            f"({results['gt']['plan_anchor_contact_realization_cm']:.2f})"
        )
        md_lines.append(
            f"- Transition vel jump < 3 cm/frame: "
            f"{'✓' if pass_gate_transition else '✗'}  "
            f"({results['gt']['transition_local_vel_jump_cm_per_frame']:.2f})"
        )
        md_lines.append("\n## Metrics by plan variant\n")
        cols = [
            "global_joint_error_cm",
            "root_aligned_joint_error_cm",
            "observed_anchor_frame_error_cm",
            "near_anchor_window_error_cm",
            "far_unobserved_error_cm",
            "far_unobserved_root_aligned_error_cm",
            "local_vel_near_anchor_cm_per_frame",
            "local_vel_far_unobs_cm_per_frame",
            "transition_local_vel_jump_cm_per_frame",
            "plan_anchor_contact_realization_cm",
            "motion_135_delta_vs_gt",
        ]
        if any("stage1_coarse_norm_l2" in row for row in results.values()):
            cols.append("stage1_coarse_norm_l2")
        md_lines.append("| variant | " + " | ".join(c.replace("_", " ") for c in cols) + " |")
        md_lines.append("|" + "|".join(["---"] * (len(cols) + 1)) + "|")
        for name in ["gt", "zero", "shuffled_time", "wrong_clip", "reversed_time",
                     "target_perturbed", "part_swapped"]:
            row = [name] + [f"{results[name].get(c, float('nan')):.3f}" for c in cols]
            md_lines.append("| " + " | ".join(row) + " |")
        if route_results:
            md_lines.append("\n## Metrics by Stage-1 Route Variant\n")
            route_cols = [
                "global_joint_error_cm",
                "near_anchor_window_error_cm",
                "far_unobserved_error_cm",
                "plan_anchor_contact_realization_cm",
                "stage1_coarse_norm_l2",
                "motion_135_delta_vs_gt_route",
            ]
            md_lines.append("| variant | " + " | ".join(c.replace("_", " ") for c in route_cols) + " |")
            md_lines.append("|" + "|".join(["---"] * (len(route_cols) + 1)) + "|")
            for name in ["gt_route", "zero_route", "shuffled_route", "wrong_clip_route"]:
                row = [name] + [f"{route_results[name].get(c, float('nan')):.3f}" for c in route_cols]
                md_lines.append("| " + " | ".join(row) + " |")
        if conflict_results:
            md_lines.append("\n## Plan / Route Conflict Cases\n")
            conflict_cols = [
                "global_joint_error_cm",
                "near_anchor_window_error_cm",
                "far_unobserved_error_cm",
                "plan_anchor_contact_realization_cm",
                "stage1_coarse_norm_l2",
                "motion_135_delta_vs_gt_plan_gt_route",
            ]
            md_lines.append("| variant | " + " | ".join(c.replace("_", " ") for c in conflict_cols) + " |")
            md_lines.append("|" + "|".join(["---"] * (len(conflict_cols) + 1)) + "|")
            for name in [
                "gt_plan_wrong_route",
                "wrong_plan_gt_route",
                "target_perturbed_plan_gt_route",
            ]:
                row = [name] + [f"{conflict_results[name].get(c, float('nan')):.3f}" for c in conflict_cols]
                md_lines.append("| " + " | ".join(row) + " |")
        md_lines.append("\n## Interpretation\n")
        md_lines.append(
            "If GT-plan unobs error is significantly lower than zero / wrong / "
            "shuffled / reversed unobs error, plan information is being routed "
            "to unobserved-frame predictions. If unobs error is flat across "
            "variants (≤ 1 cm spread), the model is ignoring the plan — proceed "
            "to architectural alternatives. For Round-22 configs, route variants "
            "test whether Stage-1 coarse is used; conflict cases test whether "
            "route overwhelms the interaction plan."
        )
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text("\n".join(md_lines), encoding="utf-8")
        print(f"Wrote Markdown report to {args.md}")

    # Console summary
    print("\nFar-unobs error (cm) by variant:")
    for name in ["gt", "zero", "shuffled_time", "wrong_clip", "reversed_time",
                 "target_perturbed", "part_swapped"]:
        print(f"  {name:18s}  {results[name]['far_unobserved_error_cm']:.3f}")
    if route_results:
        print("\nFar-unobs error (cm) by Stage-1 route variant:")
        for name in ["gt_route", "zero_route", "shuffled_route", "wrong_clip_route"]:
            print(f"  {name:18s}  {route_results[name]['far_unobserved_error_cm']:.3f}")
    if conflict_results:
        print("\nFar-unobs error (cm) by plan/route conflict case:")
        for name in [
            "gt_plan_wrong_route",
            "wrong_plan_gt_route",
            "target_perturbed_plan_gt_route",
        ]:
            print(f"  {name:30s}  {conflict_results[name]['far_unobserved_error_cm']:.3f}")
    print(f"\nPass: unobs={pass_gate_unobs}  anchor={pass_gate_anchor}  trans={pass_gate_transition}")


if __name__ == "__main__":
    main()
