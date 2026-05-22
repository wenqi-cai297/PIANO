"""P0 dynamics diagnostic for Stage B (per stageB_frozen_motion_diagnosis_and_fix_plan.md §7).

Compares GT vs DDPM-sampled vs one-step-reconstruction velocity / acceleration
distributions, temporal frequency spectrum, and HOI relative dynamics around
contact onsets — confirms the "frozen-motion via low-frequency x0 regression"
hypothesis with concrete statistics before any architectural change.

Outputs:
- JSON with per-clip + aggregate stats
- Markdown summary table with GT vs gen ratios and FFT band-energy ratios

Usage::

    python scripts/stage_b_generator/dynamics_diagnostic.py \\
        --config configs/training/anchordiff_v12_dit_block_no_planpool_FULL_N10.yaml \\
        --ckpt   runs/training/stageB_anchordiff_v12_dit_block_no_planpool_FULL_N10/final.pt \\
        --output analyses/2026-05-11_v12_a1_dynamics_diagnostic.json \\
        --md     analyses/2026-05-11_v12_a1_dynamics_diagnostic.md \\
        --num-clips 8 \\
        --recon-t 100
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import Tensor
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
# Dataset / model / cond builders (mirrored from plan_condition_diagnostics)
# ---------------------------------------------------------------------------


def _build_dataset(cfg, bucket: str) -> Subset | torch.utils.data.ConcatDataset:
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


def _balanced_subset_indices(dataset, num_clips: int) -> list[int]:
    if not hasattr(dataset, "datasets"):
        return list(range(min(num_clips, len(dataset))))
    subdatasets = list(dataset.datasets)
    if not subdatasets:
        return []
    base = int(num_clips) // len(subdatasets)
    rem = int(num_clips) % len(subdatasets)
    counts = [base + (1 if i < rem else 0) for i in range(len(subdatasets))]
    offsets: list[int] = []
    cur = 0
    for ds in subdatasets:
        offsets.append(cur)
        cur += len(ds)
    indices: list[int] = []
    for offset, ds, count in zip(offsets, subdatasets, counts):
        indices.extend(offset + i for i in range(min(count, len(ds))))
    return indices


def _build_model(cfg, device: torch.device) -> tuple[MotionAnchorDiff, ObjectEncoder, ZIntDims]:
    z_dims = ZIntDims(
        num_parts=int(cfg.model.z_int.num_parts),
        phase_classes=int(cfg.model.z_int.phase_classes),
        support_classes=int(cfg.model.z_int.support_classes),
    )
    d = cfg.model.denoiser
    denoiser_cfg = AnchorDenoiserConfig(
        motion_dim=int(d.motion_dim),
        z_int=z_dims,
        object_traj_dim=int(d.object_traj_dim),
        init_pose_dim=int(d.init_pose_dim),
        text_dim=int(d.text_dim),
        object_token_dim=int(d.object_token_dim),
        object_num_tokens=int(d.object_num_tokens),
        cond_motion_dim=int(d.get("cond_motion_dim", 0)),
        cond_motion_output_skip=bool(d.get("cond_motion_output_skip", False)),
        cfg_drop_cond_motion=bool(d.get("cfg_drop_cond_motion", False)),
        cond_motion_xt_inject=bool(d.get("cond_motion_xt_inject", False)),
        use_interaction_plan=bool(d.get("use_interaction_plan", False)),
        plan_k_max=int(d.get("plan_k_max", 12)),
        plan_s_max=int(d.get("plan_s_max", 12)),
        plan_num_anchor_types=int(d.get("plan_num_anchor_types", 5)),
        plan_num_parts=int(d.get("plan_num_parts", 5)),
        plan_use_segment_tokens=bool(d.get("plan_use_segment_tokens", False)),
        plan_use_context_hint=bool(d.get("plan_use_context_hint", True)),
        plan_d_hint=int(d.get("plan_d_hint", 32)),
        plan_d_time_embed=int(d.get("plan_d_time_embed", 64)),
        cfg_drop_plan=bool(d.get("cfg_drop_plan", False)),
        plan_per_part_tokens=bool(d.get("plan_per_part_tokens", False)),
        plan_context_hint_mode=str(d.get("plan_context_hint_mode", "time_only")),
        use_dit_block=bool(d.get("use_dit_block", False)),
        dit_block_use_plan_pool_in_cond=bool(
            d.get("dit_block_use_plan_pool_in_cond", True)
        ),
        use_v13_dynhead=bool(d.get("use_v13_dynhead", False)),
        v13_dynhead_gamma_init=float(d.get("v13_dynhead_gamma_init", 0.1)),
        v13_dynhead_learnable_gamma=bool(d.get("v13_dynhead_learnable_gamma", True)),
        use_v13_temporal_conv=bool(d.get("use_v13_temporal_conv", False)),
        v13_temporal_conv_kernel=int(d.get("v13_temporal_conv_kernel", 5)),
        use_self_conditioning=bool(d.get("use_self_conditioning", False)),
        self_conditioning_prob=float(d.get("self_conditioning_prob", 0.0)),
        self_conditioning_mode=str(d.get("self_conditioning_mode", "standard")),
        self_conditioning_t_max=int(d.get("self_conditioning_t_max", 700)),
        self_conditioning_zero_init=bool(d.get("self_conditioning_zero_init", True)),
        stage1_coarse_dim=int(d.get("stage1_coarse_dim", 0)),
        cfg_drop_stage1_coarse=bool(d.get("cfg_drop_stage1_coarse", False)),
        d_model=int(d.d_model),
        n_layers=int(d.n_layers),
        n_heads=int(d.n_heads),
        ff_mult=int(d.ff_mult),
        dropout=float(d.dropout),
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
) -> tuple[dict, int]:
    import torch.nn.functional as F
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
    B, T, _ = motion.shape

    phase_soft = F.one_hot(phase.clamp_min(0).long(), num_classes=z_dims.phase_classes).float()
    support_soft = F.one_hot(support.clamp_min(0).long(), num_classes=z_dims.support_classes).float()

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
# Velocity / acceleration distribution stats
# ---------------------------------------------------------------------------


# SMPL-22 part-to-joint map (must match dataset / trainer)
PART_JOINT = {
    "L_hand": 20, "R_hand": 21, "L_foot": 10, "R_foot": 11, "pelvis": 0,
}
LOCAL_JOINTS = list(range(1, 22))  # exclude root for root-aligned local stats


def _joint_velocity_acceleration(
    joints: torch.Tensor,                # (B, T, 22, 3) world joints
    seq_mask: torch.Tensor,              # (B, T) bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (vel_world, vel_local, acc_world, acc_local). vel/acc shape (B, T-1, 22, 3) / (B, T-2, 22, 3)."""
    vel_world = joints[:, 1:] - joints[:, :-1]                              # (B, T-1, 22, 3)
    root = joints[:, :, 0:1, :]                                              # (B, T, 1, 3)
    root_vel = root[:, 1:] - root[:, :-1]                                    # (B, T-1, 1, 3)
    vel_local = vel_world - root_vel                                         # (B, T-1, 22, 3), root-aligned
    acc_world = vel_world[:, 1:] - vel_world[:, :-1]                         # (B, T-2, 22, 3)
    acc_local = vel_local[:, 1:] - vel_local[:, :-1]                         # (B, T-2, 22, 3)
    return vel_world, vel_local, acc_world, acc_local


def _stats_from_magnitudes(
    mag: torch.Tensor,                   # (...,)
    mask: torch.Tensor,                  # broadcasts with mag, bool
) -> dict[str, float]:
    """Pooled stats — mean, median, p25/75/95, std — over flattened masked entries."""
    flat = mag[mask.bool()].cpu().double().numpy()
    if flat.size == 0:
        return {k: 0.0 for k in ("mean", "median", "p25", "p75", "p95", "std", "n")}
    return {
        "mean": float(flat.mean()),
        "median": float(np.median(flat)),
        "p25": float(np.percentile(flat, 25)),
        "p75": float(np.percentile(flat, 75)),
        "p95": float(np.percentile(flat, 95)),
        "std": float(flat.std()),
        "n": int(flat.size),
    }


def _per_joint_vel_stats(
    joints: torch.Tensor,                # (B, T, 22, 3)
    seq_mask: torch.Tensor,              # (B, T) bool
    scale_to_cm: float = 100.0,
) -> dict:
    """Per-joint root-aligned velocity stats in cm/frame, for the 5 interaction joints."""
    _, vel_local, _, acc_local = _joint_velocity_acceleration(joints, seq_mask)
    vel_mag_local = vel_local.pow(2).sum(-1).clamp_min(1e-12).sqrt() * scale_to_cm  # (B, T-1, 22)
    acc_mag_local = acc_local.pow(2).sum(-1).clamp_min(1e-12).sqrt() * scale_to_cm
    # Build masks aligned with T-1 / T-2
    vel_mask = seq_mask[:, 1:] & seq_mask[:, :-1]
    acc_mask = vel_mask[:, 1:] & vel_mask[:, :-1]
    out: dict = {"per_joint_vel_cm_per_frame": {}, "per_joint_acc_cm_per_frame": {}}
    for name, j in PART_JOINT.items():
        out["per_joint_vel_cm_per_frame"][name] = _stats_from_magnitudes(
            vel_mag_local[:, :, j], vel_mask,
        )
        out["per_joint_acc_cm_per_frame"][name] = _stats_from_magnitudes(
            acc_mag_local[:, :, j], acc_mask,
        )
    # Whole-body local-velocity pooled stats (21 non-root joints)
    body_vel = vel_mag_local[:, :, LOCAL_JOINTS]                             # (B, T-1, 21)
    body_acc = acc_mag_local[:, :, LOCAL_JOINTS]
    out["body_local_vel_cm_per_frame"] = _stats_from_magnitudes(
        body_vel, vel_mask.unsqueeze(-1).expand(-1, -1, len(LOCAL_JOINTS)),
    )
    out["body_local_acc_cm_per_frame"] = _stats_from_magnitudes(
        body_acc, acc_mask.unsqueeze(-1).expand(-1, -1, len(LOCAL_JOINTS)),
    )
    return out


# ---------------------------------------------------------------------------
# FFT temporal power spectrum
# ---------------------------------------------------------------------------


def _fft_band_energy(
    joints: torch.Tensor,                # (B, T, 22, 3)
    seq_mask: torch.Tensor,              # (B, T) bool
    fps: float = 20.0,
) -> dict:
    """Per-clip rfft along T axis on root-relative joint positions, banded into low/mid/high.

    Returns aggregate energy ratios across joints + xyz. To avoid masked-padding
    leakage, each clip is cropped to its valid length before FFT.

    Bands (cycles per second):
        low:  0 .. 1.0 Hz  (slow pose / posture)
        mid:  1.0 .. 4.0 Hz (locomotion, manipulation)
        high: 4.0 .. Nyquist (jitter, sharp transitions)

    At fps=20, Nyquist = 10 Hz.
    """
    B, T, J, _ = joints.shape
    # Root-relative position (so very-low DC dominated by world translation is removed)
    root = joints[:, :, 0:1, :]
    j_rel = joints - root                                                    # (B, T, 22, 3)
    total = {"low": 0.0, "mid": 0.0, "high": 0.0}
    n_clips = 0
    per_clip_energies: list[dict] = []
    for b in range(B):
        L = int(seq_mask[b].sum().item())
        if L < 32:
            continue
        x = j_rel[b, :L]                                                     # (L, 22, 3)
        # Detrend (subtract per-channel mean) so DC bin doesn't dominate.
        x = x - x.mean(dim=0, keepdim=True)
        # rfft along time
        X = torch.fft.rfft(x, dim=0)                                         # (L//2+1, 22, 3)
        # Power
        P = (X.real ** 2 + X.imag ** 2)                                      # (F, 22, 3)
        freqs = torch.fft.rfftfreq(L, d=1.0 / fps).to(P.device)              # (F,)
        # Sum over joints (skip root) + xyz
        P_body = P[:, LOCAL_JOINTS, :].sum(dim=(1, 2))                       # (F,)
        low_e = float(P_body[(freqs >= 0.0) & (freqs < 1.0)].sum())
        mid_e = float(P_body[(freqs >= 1.0) & (freqs < 4.0)].sum())
        high_e = float(P_body[freqs >= 4.0].sum())
        total["low"] += low_e
        total["mid"] += mid_e
        total["high"] += high_e
        per_clip_energies.append({
            "low": low_e, "mid": mid_e, "high": high_e,
        })
        n_clips += 1
    total_e = total["low"] + total["mid"] + total["high"] + 1e-12
    return {
        "energy_total": total_e,
        "energy_low": total["low"],
        "energy_mid": total["mid"],
        "energy_high": total["high"],
        "fraction_low": total["low"] / total_e,
        "fraction_mid": total["mid"] / total_e,
        "fraction_high": total["high"] / total_e,
        "n_clips": n_clips,
        "per_clip": per_clip_energies,
    }


# ---------------------------------------------------------------------------
# HOI relative dynamics around contact onsets
# ---------------------------------------------------------------------------


def _hoi_relative_velocity(
    joints: torch.Tensor,                # (B, T, 22, 3) predicted/GT joints
    obj_positions: torch.Tensor,         # (B, T, 3) object world position
    contact_state: torch.Tensor,         # (B, T, num_parts) float 0/1 — pseudo-labelled
    seq_mask: torch.Tensor,              # (B, T) bool
    window: int = 5,
    scale_to_cm: float = 100.0,
) -> dict:
    """Hand-vs-object relative velocity statistics.

    Computes ||hand_vel - obj_vel|| in cm/frame, conditioned on three windows:
      - inside-contact (contact_state==1)
      - onset window (window frames before contact start)
      - release window (window frames after contact end)

    Reported only for L_hand (j=20) and R_hand (j=21), since feet are typically
    in stable-support and pelvis is global locomotion.

    contact_state shape:
        (B, T, num_parts=5) where parts = [L_hand, R_hand, L_foot, R_foot, pelvis]
    """
    obj_vel = obj_positions[:, 1:] - obj_positions[:, :-1]                   # (B, T-1, 3)
    out: dict = {}
    for hand_name, j, part_idx in (("L_hand", 20, 0), ("R_hand", 21, 1)):
        hand_vel = joints[:, 1:, j, :] - joints[:, :-1, j, :]                # (B, T-1, 3)
        rel_vel = hand_vel - obj_vel                                         # (B, T-1, 3)
        rel_mag = rel_vel.pow(2).sum(-1).clamp_min(1e-12).sqrt() * scale_to_cm  # (B, T-1)
        # Per-frame contact label aligned with T-1 (use earlier frame's label)
        c = (contact_state[:, :-1, part_idx] > 0.5)                          # (B, T-1)
        vel_mask = seq_mask[:, 1:] & seq_mask[:, :-1]
        in_contact = c & vel_mask
        # Onset: contact_state transitions 0→1
        onset_at = (contact_state[:, 1:, part_idx] > 0.5) & (contact_state[:, :-1, part_idx] <= 0.5)  # (B, T-1)
        # Release: 1→0
        release_at = (contact_state[:, 1:, part_idx] <= 0.5) & (contact_state[:, :-1, part_idx] > 0.5)
        # Expand onset / release into ±window
        B, Tm1 = c.shape
        onset_window = torch.zeros_like(c)
        release_window = torch.zeros_like(c)
        for offset in range(-window, window + 1):
            shifted = torch.zeros_like(onset_at)
            if offset == 0:
                shifted = onset_at
            elif offset > 0:
                shifted[:, offset:] = onset_at[:, :-offset]
            else:
                shifted[:, :offset] = onset_at[:, -offset:]
            onset_window = onset_window | shifted
            shifted_r = torch.zeros_like(release_at)
            if offset == 0:
                shifted_r = release_at
            elif offset > 0:
                shifted_r[:, offset:] = release_at[:, :-offset]
            else:
                shifted_r[:, :offset] = release_at[:, -offset:]
            release_window = release_window | shifted_r
        onset_window = onset_window & vel_mask
        release_window = release_window & vel_mask
        out[hand_name] = {
            "rel_vel_in_contact_cm_per_frame": _stats_from_magnitudes(rel_mag, in_contact),
            "rel_vel_onset_window_cm_per_frame": _stats_from_magnitudes(rel_mag, onset_window),
            "rel_vel_release_window_cm_per_frame": _stats_from_magnitudes(rel_mag, release_window),
            "rel_vel_all_cm_per_frame": _stats_from_magnitudes(rel_mag, vel_mask),
            "n_onset_events": int(onset_at.sum().item()),
            "n_release_events": int(release_at.sum().item()),
            "n_contact_frames": int(in_contact.sum().item()),
        }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _fk_from_motion_135(motion: torch.Tensor, rest_offsets: torch.Tensor) -> torch.Tensor:
    """motion (B, T, 135) -> joints (B, T, 22, 3) via SMPL FK."""
    B, T, _ = motion.shape
    rot_6d = motion[..., :132].view(B, T, 22, 6).float()
    root_world = motion[..., 132:135].float()
    rot_mat = _rot6d_to_mat(rot_6d)
    rest_per_frame = rest_offsets.unsqueeze(1).expand(B, T, 22, 3)
    return _fk_from_global(rot_mat, rest_per_frame, root_world)


@torch.no_grad()
def _one_step_recon_motion(
    model: MotionAnchorDiff,
    motion_gt: Tensor,
    cond: dict[str, Any],
    recon_t: int,
) -> Tensor:
    """Return a clean-motion one-step prediction for the active objective.

    DDPM uses q(x_t | x0) and converts v-prediction back to x0. Rectified flow
    uses its own linear noise-to-data path. The integer ``recon_t`` remains a
    DDPM-style noisedness index, so RF maps it to rf_t = 1 - recon_t / 999.
    """
    diffusion = model.diffusion
    if getattr(diffusion, "objective", "ddpm") == "rectified_flow":
        noisedness = float(recon_t) / max(float(diffusion.num_steps - 1), 1.0)
        rf_t_value = float(np.clip(1.0 - noisedness, 0.0, 1.0))
        t_rf = torch.full(
            (motion_gt.shape[0],), rf_t_value, dtype=motion_gt.dtype,
            device=motion_gt.device,
        )
        noise = torch.randn_like(motion_gt) * float(diffusion.rf_denoiser_noise_scale)
        z_t = diffusion.rf_interpolate(motion_gt, t_rf, noise)
        t_idx = diffusion.rf_time_to_index(t_rf)
        return model.denoiser(z_t, t_idx, cond, cond_drop_mask=None)

    t_ddpm = torch.full(
        (motion_gt.shape[0],), int(recon_t), dtype=torch.long,
        device=motion_gt.device,
    )
    noise = torch.randn_like(motion_gt)
    x_t = diffusion.q_sample(motion_gt, t_ddpm, noise)
    raw = model.denoiser(x_t, t_ddpm, cond, cond_drop_mask=None)
    if getattr(diffusion, "prediction_target", "x0") == "v":
        return diffusion.predict_x0_from_v(x_t, t_ddpm, raw)
    return raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--md", type=Path, default=None)
    parser.add_argument(
        "--num-clips", type=int, default=8,
        help="Number of clips to process (from the start of the split).",
    )
    parser.add_argument(
        "--recon-t", type=int, default=100,
        help="Diffusion step at which to draw x_t for one-step reconstruction (range 0..num_steps-1).",
    )
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bucket", default="train", choices=["train", "val"])
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument(
        "--balanced-subsets",
        action="store_true",
        help="Pick clips evenly across configured subsets instead of taking the first N.",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # Dataset + loader
    dataset = _build_dataset(cfg, args.bucket)
    overfit_n = int(cfg.data.get("overfit_n_clips", 0))
    scale_subset_seed = cfg.data.get("scale_subset_seed", None)
    if overfit_n > 0:
        # Mirror trainer's shuffle-then-take-N (per
        # analyses/stageB_root_cause_analysis_v2_and_next_strategy.md §5).
        n_avail = len(dataset)
        indices = list(range(n_avail))
        if scale_subset_seed is not None:
            import random as _random
            _rng = _random.Random(int(scale_subset_seed))
            _rng.shuffle(indices)
        indices = indices[:min(overfit_n, n_avail)]
        dataset = Subset(dataset, indices)
    if args.balanced_subsets:
        clip_indices = _balanced_subset_indices(dataset, int(args.num_clips))
    else:
        clip_indices = list(range(min(args.num_clips, len(dataset))))
    dataset = Subset(dataset, clip_indices)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        collate_fn=collate_hoi, num_workers=0,
    )

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

    # Per-clip aggregation
    all_gt_joints: list[torch.Tensor] = []
    all_sample_joints: list[torch.Tensor] = []
    all_recon_joints: list[torch.Tensor] = []
    all_seq_masks: list[torch.Tensor] = []
    all_obj_positions: list[torch.Tensor] = []
    all_contact_state: list[torch.Tensor] = []
    per_clip_summary: list[dict] = []

    for i, batch in enumerate(loader):
        cond, T = _build_cond(batch, model, object_encoder, clip_model, z_dims, cfg, device)
        plan_keys = [
            "anchor_time", "anchor_part", "anchor_target_local",
            "anchor_target_world", "anchor_type", "anchor_phase",
            "anchor_support", "anchor_conf", "anchor_mask",
            "segment_start", "segment_end", "segment_part",
            "segment_target_summary_local", "segment_phase",
            "segment_support", "segment_conf", "segment_mask",
        ]
        plan = {k: batch[f"plan_{k}"].to(device) for k in plan_keys}
        cond_full = {**cond, "interaction_plan": plan}

        motion_gt = batch["motion"].to(device).float()                       # (1, T, 135)
        rest_offsets = batch["rest_offsets"].to(device).float()
        seq_len = batch["seq_len"].to(device)
        seq_idx = torch.arange(T, device=device).unsqueeze(0)
        seq_mask = (seq_idx < seq_len.unsqueeze(1))                          # (1, T)
        joints_gt = batch["joints"].to(device).float()                       # (1, T, 22, 3) — dataset-provided FK

        # --- DDPM sample (clean denoise from pure noise) ---
        torch.manual_seed(args.seed + i)
        with torch.no_grad():
            x0_sample = model.sample(
                cond=cond_full, seq_length=T, cfg_scale=args.cfg_scale,
                replacement="none", output_skip=False,
            )                                                                # (1, T, 135)
        joints_sample = _fk_from_motion_135(x0_sample, rest_offsets)

        # --- One-step reconstruction at t = args.recon-t (training-time forward) ---
        with torch.no_grad():
            x0_recon = _one_step_recon_motion(
                model, motion_gt, cond_full, recon_t=int(args.recon_t),
            )
        joints_recon = _fk_from_motion_135(x0_recon, rest_offsets)

        all_gt_joints.append(joints_gt)
        all_sample_joints.append(joints_sample)
        all_recon_joints.append(joints_recon)
        all_seq_masks.append(seq_mask)
        all_obj_positions.append(batch["object_positions"].to(device).float())
        all_contact_state.append(batch["contact_state"].to(device).float())

        per_clip_summary.append({
            "subset": batch["subset"][0],
            "seq_id": batch["seq_id"][0],
            "text": batch["text"][0][:120],
            "T": int(seq_len.item()),
        })
        print(f"  [{i+1}/{args.num_clips}] {batch['subset'][0]}/{batch['seq_id'][0]}  T={int(seq_len.item())}")

    # Concatenate over clips (pad to common T via mask)
    gt_joints = torch.cat(all_gt_joints, dim=0)
    sample_joints = torch.cat(all_sample_joints, dim=0)
    recon_joints = torch.cat(all_recon_joints, dim=0)
    seq_masks = torch.cat(all_seq_masks, dim=0)
    obj_positions = torch.cat(all_obj_positions, dim=0)
    contact_state = torch.cat(all_contact_state, dim=0)

    # --- Compute stats ---
    results: dict = {
        "config": str(args.config),
        "ckpt": str(args.ckpt),
        "num_clips": int(gt_joints.shape[0]),
        "recon_t": int(args.recon_t),
        "fps": float(args.fps),
        "per_clip": per_clip_summary,
        "gt": {
            "joint_vel_stats": _per_joint_vel_stats(gt_joints, seq_masks),
            "fft_spectrum": _fft_band_energy(gt_joints, seq_masks, fps=args.fps),
            "hoi": _hoi_relative_velocity(
                gt_joints, obj_positions, contact_state, seq_masks,
            ),
        },
        "sampled": {
            "joint_vel_stats": _per_joint_vel_stats(sample_joints, seq_masks),
            "fft_spectrum": _fft_band_energy(sample_joints, seq_masks, fps=args.fps),
            "hoi": _hoi_relative_velocity(
                sample_joints, obj_positions, contact_state, seq_masks,
            ),
        },
        "recon_one_step": {
            "joint_vel_stats": _per_joint_vel_stats(recon_joints, seq_masks),
            "fft_spectrum": _fft_band_energy(recon_joints, seq_masks, fps=args.fps),
            "hoi": _hoi_relative_velocity(
                recon_joints, obj_positions, contact_state, seq_masks,
            ),
        },
    }

    # --- Ratios (sampled / gt and recon / gt) ---
    def _ratio_block(num: dict, den: dict) -> dict:
        return {
            "body_local_vel_mean": num["body_local_vel_cm_per_frame"]["mean"]
            / max(den["body_local_vel_cm_per_frame"]["mean"], 1e-9),
            "body_local_vel_median": num["body_local_vel_cm_per_frame"]["median"]
            / max(den["body_local_vel_cm_per_frame"]["median"], 1e-9),
            "body_local_acc_mean": num["body_local_acc_cm_per_frame"]["mean"]
            / max(den["body_local_acc_cm_per_frame"]["mean"], 1e-9),
            "L_hand_vel_mean": num["per_joint_vel_cm_per_frame"]["L_hand"]["mean"]
            / max(den["per_joint_vel_cm_per_frame"]["L_hand"]["mean"], 1e-9),
            "R_hand_vel_mean": num["per_joint_vel_cm_per_frame"]["R_hand"]["mean"]
            / max(den["per_joint_vel_cm_per_frame"]["R_hand"]["mean"], 1e-9),
            "pelvis_vel_mean": num["per_joint_vel_cm_per_frame"]["pelvis"]["mean"]
            / max(den["per_joint_vel_cm_per_frame"]["pelvis"]["mean"], 1e-9),
        }
    results["ratios_sampled_over_gt"] = _ratio_block(
        results["sampled"]["joint_vel_stats"], results["gt"]["joint_vel_stats"],
    )
    results["ratios_recon_over_gt"] = _ratio_block(
        results["recon_one_step"]["joint_vel_stats"], results["gt"]["joint_vel_stats"],
    )
    results["subset_wise"] = {}
    for subset in sorted({c["subset"] for c in per_clip_summary}):
        row_idx = [i for i, c in enumerate(per_clip_summary) if c["subset"] == subset]
        if not row_idx:
            continue
        idx_t = torch.tensor(row_idx, device=gt_joints.device, dtype=torch.long)
        sub_gt = gt_joints.index_select(0, idx_t)
        sub_sample = sample_joints.index_select(0, idx_t)
        sub_recon = recon_joints.index_select(0, idx_t)
        sub_masks = seq_masks.index_select(0, idx_t)
        sub_obj = obj_positions.index_select(0, idx_t)
        sub_contact = contact_state.index_select(0, idx_t)
        sub_payload = {
            "num_clips": len(row_idx),
            "gt": {
                "joint_vel_stats": _per_joint_vel_stats(sub_gt, sub_masks),
                "fft_spectrum": _fft_band_energy(sub_gt, sub_masks, fps=args.fps),
                "hoi": _hoi_relative_velocity(sub_gt, sub_obj, sub_contact, sub_masks),
            },
            "sampled": {
                "joint_vel_stats": _per_joint_vel_stats(sub_sample, sub_masks),
                "fft_spectrum": _fft_band_energy(sub_sample, sub_masks, fps=args.fps),
                "hoi": _hoi_relative_velocity(sub_sample, sub_obj, sub_contact, sub_masks),
            },
            "recon_one_step": {
                "joint_vel_stats": _per_joint_vel_stats(sub_recon, sub_masks),
                "fft_spectrum": _fft_band_energy(sub_recon, sub_masks, fps=args.fps),
                "hoi": _hoi_relative_velocity(sub_recon, sub_obj, sub_contact, sub_masks),
            },
        }
        sub_payload["ratios_sampled_over_gt"] = _ratio_block(
            sub_payload["sampled"]["joint_vel_stats"],
            sub_payload["gt"]["joint_vel_stats"],
        )
        sub_payload["ratios_recon_over_gt"] = _ratio_block(
            sub_payload["recon_one_step"]["joint_vel_stats"],
            sub_payload["gt"]["joint_vel_stats"],
        )
        results["subset_wise"][subset] = sub_payload
    # FFT band-fraction shift
    results["fft_band_shift"] = {
        "gt": {
            "low": results["gt"]["fft_spectrum"]["fraction_low"],
            "mid": results["gt"]["fft_spectrum"]["fraction_mid"],
            "high": results["gt"]["fft_spectrum"]["fraction_high"],
        },
        "sampled": {
            "low": results["sampled"]["fft_spectrum"]["fraction_low"],
            "mid": results["sampled"]["fft_spectrum"]["fraction_mid"],
            "high": results["sampled"]["fft_spectrum"]["fraction_high"],
        },
        "recon_one_step": {
            "low": results["recon_one_step"]["fft_spectrum"]["fraction_low"],
            "mid": results["recon_one_step"]["fft_spectrum"]["fraction_mid"],
            "high": results["recon_one_step"]["fft_spectrum"]["fraction_high"],
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote JSON to {args.output}")

    # --- Markdown summary ---
    if args.md is not None:
        md: list[str] = []
        md.append("# v12 A1 — P0 dynamics diagnostic\n")
        md.append(f"**Config:** `{args.config}`")
        md.append(f"**Ckpt:**   `{args.ckpt}`")
        md.append(f"**Clips:**  {results['num_clips']}     **fps:** {args.fps}     **recon_t:** {args.recon_t}\n")

        md.append("## 1. Body-pooled local velocity (cm/frame, root-aligned)\n")
        md.append("| source | mean | median | p25 | p75 | p95 | std |")
        md.append("|---|---|---|---|---|---|---|")
        for label, key in (("GT", "gt"), ("DDPM sample", "sampled"), ("One-step recon", "recon_one_step")):
            s = results[key]["joint_vel_stats"]["body_local_vel_cm_per_frame"]
            md.append(
                f"| {label} | {s['mean']:.3f} | {s['median']:.3f} | "
                f"{s['p25']:.3f} | {s['p75']:.3f} | {s['p95']:.3f} | {s['std']:.3f} |"
            )

        md.append("\n## 2. Per-joint mean velocity (cm/frame, root-aligned)\n")
        md.append("| joint | GT | sampled | recon | sample/GT | recon/GT |")
        md.append("|---|---|---|---|---|---|")
        for name in ("L_hand", "R_hand", "L_foot", "R_foot", "pelvis"):
            g = results["gt"]["joint_vel_stats"]["per_joint_vel_cm_per_frame"][name]["mean"]
            s = results["sampled"]["joint_vel_stats"]["per_joint_vel_cm_per_frame"][name]["mean"]
            r = results["recon_one_step"]["joint_vel_stats"]["per_joint_vel_cm_per_frame"][name]["mean"]
            md.append(
                f"| {name} | {g:.3f} | {s:.3f} | {r:.3f} | "
                f"×{s/max(g, 1e-9):.3f} | ×{r/max(g, 1e-9):.3f} |"
            )

        md.append("\n## 3. Acceleration (body-pooled, cm/frame²)\n")
        md.append("| source | mean | median | p95 |")
        md.append("|---|---|---|---|")
        for label, key in (("GT", "gt"), ("DDPM sample", "sampled"), ("One-step recon", "recon_one_step")):
            s = results[key]["joint_vel_stats"]["body_local_acc_cm_per_frame"]
            md.append(f"| {label} | {s['mean']:.3f} | {s['median']:.3f} | {s['p95']:.3f} |")

        md.append("\n## 4. FFT temporal energy fraction (root-relative joints, body only)\n")
        md.append("| source | low (0–1 Hz) | mid (1–4 Hz) | high (4–10 Hz) |")
        md.append("|---|---|---|---|")
        for label, key in (("GT", "gt"), ("DDPM sample", "sampled"), ("One-step recon", "recon_one_step")):
            b = results["fft_band_shift"][key]
            md.append(f"| {label} | {b['low']:.3f} | {b['mid']:.3f} | {b['high']:.3f} |")

        if results.get("subset_wise"):
            md.append("\n## 4b. Subset-wise dynamics summary\n")
            md.append("| subset | clips | sample body vel xGT | recon body vel xGT | L hand xGT | R hand xGT | sampled FFT mid | acc p95 sample |")
            md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
            for subset, sub in results["subset_wise"].items():
                ratios_s = sub["ratios_sampled_over_gt"]
                ratios_r = sub["ratios_recon_over_gt"]
                fft_mid = sub["sampled"]["fft_spectrum"]["fraction_mid"]
                acc_p95 = sub["sampled"]["joint_vel_stats"]["body_local_acc_cm_per_frame"]["p95"]
                md.append(
                    f"| {subset} | {sub['num_clips']} | "
                    f"{ratios_s['body_local_vel_mean']:.3f} | "
                    f"{ratios_r['body_local_vel_mean']:.3f} | "
                    f"{ratios_s['L_hand_vel_mean']:.3f} | "
                    f"{ratios_s['R_hand_vel_mean']:.3f} | "
                    f"{fft_mid:.3f} | {acc_p95:.3f} |"
                )

        md.append("\n## 5. HOI hand-relative-to-object velocity (cm/frame)\n")
        for hand in ("L_hand", "R_hand"):
            md.append(f"\n### {hand}\n")
            md.append("| window | GT mean | sampled mean | recon mean | sample/GT | recon/GT | n |")
            md.append("|---|---|---|---|---|---|---|")
            for wkey, wlabel in (
                ("rel_vel_in_contact_cm_per_frame", "in-contact"),
                ("rel_vel_onset_window_cm_per_frame", "onset ±5"),
                ("rel_vel_release_window_cm_per_frame", "release ±5"),
                ("rel_vel_all_cm_per_frame", "all"),
            ):
                g = results["gt"]["hoi"][hand][wkey]["mean"]
                gn = results["gt"]["hoi"][hand][wkey]["n"]
                s = results["sampled"]["hoi"][hand][wkey]["mean"]
                r = results["recon_one_step"]["hoi"][hand][wkey]["mean"]
                md.append(
                    f"| {wlabel} | {g:.3f} | {s:.3f} | {r:.3f} | "
                    f"×{s/max(g, 1e-9):.3f} | ×{r/max(g, 1e-9):.3f} | {gn} |"
                )

        md.append("\n## 6. Interpretation\n")
        md.append(
            "**Expected failure mode (per stageB_frozen_motion_diagnosis_and_fix_plan.md §7.1–§7.2):**\n"
            "- Sampled body-local velocity << GT velocity (frozen body)\n"
            "- Sampled FFT high-band energy fraction << GT high-band fraction (loss of dynamics)\n"
            "- One-step recon velocity ≈ or > GT (training sees x_0_pred under noise; sampling is clean denoising)\n"
            "  → confirms train-vs-inference distribution shift on velocity\n\n"
            "**Pass for v13 architectural fix (P1 + P2):** sampled mean local velocity should rise to ≥ 0.8×GT\n"
            "without high-band energy ratio exceeding GT by more than 1.5× (jitter)."
        )

        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text("\n".join(md), encoding="utf-8")
        print(f"Wrote Markdown to {args.md}")

    # Console digest
    print("\n=== Summary ===")
    print(f"  Body local vel  GT={results['gt']['joint_vel_stats']['body_local_vel_cm_per_frame']['mean']:.3f}  "
          f"sample={results['sampled']['joint_vel_stats']['body_local_vel_cm_per_frame']['mean']:.3f}  "
          f"recon={results['recon_one_step']['joint_vel_stats']['body_local_vel_cm_per_frame']['mean']:.3f}  "
          f"(sample/gt=×{results['ratios_sampled_over_gt']['body_local_vel_mean']:.3f}, "
          f"recon/gt=×{results['ratios_recon_over_gt']['body_local_vel_mean']:.3f})")
    bs = results["fft_band_shift"]
    print(f"  FFT (low,mid,high) GT=({bs['gt']['low']:.3f},{bs['gt']['mid']:.3f},{bs['gt']['high']:.3f})  "
          f"sample=({bs['sampled']['low']:.3f},{bs['sampled']['mid']:.3f},{bs['sampled']['high']:.3f})  "
          f"recon=({bs['recon_one_step']['low']:.3f},{bs['recon_one_step']['mid']:.3f},{bs['recon_one_step']['high']:.3f})")


if __name__ == "__main__":
    main()
