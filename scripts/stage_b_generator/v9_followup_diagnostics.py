"""v9 follow-up diagnostics per claude_code_v9_condmdi_diagnostic_next_steps.md.

Implements the three required diagnostic protocols:

  §3 Required metrics — error decomposition (root vs root-aligned, observed
     vs unobserved, local vs global velocity, joint spread, bone-len std).
  §4 cfg=0/1 × K=8 random/linspace evaluation matrix on a given checkpoint.
  §5 Condition sensitivity test — fix everything else, vary cond_motion in
     {GT, zeros, shuffled-time, wrong-clip, reversed}.
  §6 Single-step teacher-forced denoising at t ∈ {0,100,250,500,750,999}.
  §11 JSON logging schema.

Usage:
    python scripts/stage_b_generator/v9_followup_diagnostics.py \\
        --config configs/training/anchordiff_v9_1_clean_x0_overfit.yaml \\
        --ckpt runs/training/stageB_anchordiff_v9_1_clean_x0_overfit/final.pt \\
        --version-name v9_1_clean_x0_overfit \\
        --output-dir runs/visualizations/anchordiff_v9_1_clean_x0_followup
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from piano.data.dataset import HOIDataset, collate_hoi
from piano.models.motion_anchordiff import (
    AnchorDenoiserConfig, AnchorDiffConfig, DiffusionConfig,
    MotionAnchorDiff, ZIntDims, pack_z_int,
)
from piano.models.object_encoder import ObjectEncoder
from piano.training.anchor_consistency_loss import lift_object_local_to_world
from piano.training.smpl_kinematics import (
    rotation_6d_to_matrix, fk_from_global_rotations, SMPL22_PARENTS,
)
from piano.utils.clip_utils import encode_text_per_token, load_clip_text_encoder


# ---------------------------------------------------------------------------
# Decoding helpers
# ---------------------------------------------------------------------------


def decode_135_to_joints(motion_135: torch.Tensor, rest_offsets: torch.Tensor) -> torch.Tensor:
    """135 = rot6d 132 + root 3.  Returns (B, T, 22, 3) joints (world)."""
    B, T, _ = motion_135.shape
    rot_6d = motion_135[..., :132].view(B, T, 22, 6)
    root = motion_135[..., 132:135]
    rot_mat = rotation_6d_to_matrix(rot_6d)
    if rest_offsets.dim() == 2:
        rest_offsets = rest_offsets.unsqueeze(0)
    rest_per_frame = rest_offsets.unsqueeze(1).expand(B, T, 22, 3)
    return fk_from_global_rotations(rot_mat, rest_per_frame, root)


def bone_len_std_cm(joints: torch.Tensor) -> float:
    """joints: (T, 22, 3). Max over bones of frame-to-frame std (cm)."""
    parents = torch.tensor(SMPL22_PARENTS)
    stds = []
    for j in range(1, 22):
        p = int(parents[j].item())
        bone = (joints[:, j, :] - joints[:, p, :]).norm(dim=-1)
        stds.append(float(bone.std().item()))
    return max(stds) * 100


# ---------------------------------------------------------------------------
# Required metric decomposition — §3.3
# ---------------------------------------------------------------------------


def decompose_metrics(
    joints_pred: torch.Tensor,    # (T, 22, 3)
    joints_gt: torch.Tensor,      # (T, 22, 3)
    obs_mask_T: torch.Tensor,     # (T,) bool
) -> dict:
    """Returns per-section error decomposition.

    Per spec §3.3, distinguishes:
      - global_joint_error_cm : ‖j_pred - j_gt‖
      - root_error_cm         : ‖root_pred - root_gt‖    (root := joint 0)
      - root_aligned_joint_error_cm : ‖(j_pred - root_pred) - (j_gt - root_gt)‖
      - observed_frame_joint_error_cm   : global err averaged over obs frames
      - unobserved_frame_joint_error_cm : global err averaged over un-obs frames
      - local_joint_velocity_error_cm   : ‖Δj_pred_local - Δj_gt_local‖
      - root_velocity_error_cm          : ‖Δroot_pred - Δroot_gt‖
      - joint_spread_cm                 : mean ‖j - center‖ across joints
      - bone_length_std_cm              : max-bone frame-to-frame std
    """
    T = joints_pred.shape[0]
    pred_root = joints_pred[:, 0:1, :]
    gt_root = joints_gt[:, 0:1, :]
    pred_local = joints_pred - pred_root  # (T, 22, 3)
    gt_local = joints_gt - gt_root

    err_global = (joints_pred - joints_gt).norm(dim=-1)         # (T, 22)
    err_root = (pred_root - gt_root).norm(dim=-1).squeeze(-1)   # (T,)
    err_local = (pred_local - gt_local).norm(dim=-1)            # (T, 22)

    # mean error per frame, averaged over joints
    err_global_per_frame = err_global.mean(dim=-1)  # (T,)
    err_local_per_frame = err_local.mean(dim=-1)    # (T,)

    obs_mask_T = obs_mask_T.bool()
    if obs_mask_T.any():
        obs_err_cm = float(err_global_per_frame[obs_mask_T].mean().item()) * 100
    else:
        obs_err_cm = float("nan")
    un_mask = ~obs_mask_T
    if un_mask.any():
        un_err_cm = float(err_global_per_frame[un_mask].mean().item()) * 100
    else:
        un_err_cm = float("nan")

    # velocity errors
    if T > 1:
        d_pred_local = pred_local[1:] - pred_local[:-1]
        d_gt_local = gt_local[1:] - gt_local[:-1]
        v_local_err = (d_pred_local - d_gt_local).norm(dim=-1).mean().item() * 100
        d_pred_root = pred_root.squeeze(1)[1:] - pred_root.squeeze(1)[:-1]
        d_gt_root = gt_root.squeeze(1)[1:] - gt_root.squeeze(1)[:-1]
        v_root_err = (d_pred_root - d_gt_root).norm(dim=-1).mean().item() * 100
    else:
        v_local_err = v_root_err = 0.0

    # joint spread (how 'spread out' is the body)
    pred_center = joints_pred.mean(dim=1, keepdim=True)
    spread_cm = (joints_pred - pred_center).norm(dim=-1).mean().item() * 100

    return {
        "global_joint_error_cm": float(err_global_per_frame.mean().item() * 100),
        "root_error_cm": float(err_root.mean().item() * 100),
        "root_aligned_joint_error_cm": float(err_local_per_frame.mean().item() * 100),
        "observed_frame_joint_error_cm": float(obs_err_cm),
        "unobserved_frame_joint_error_cm": float(un_err_cm),
        "local_joint_velocity_error_cm": float(v_local_err),
        "root_velocity_error_cm": float(v_root_err),
        "joint_spread_cm": float(spread_cm),
        "bone_length_std_cm": float(bone_len_std_cm(joints_pred)),
    }


# ---------------------------------------------------------------------------
# Cond / mask construction
# ---------------------------------------------------------------------------


def build_keyframe_mask(
    seq_len: int, T_full: int, K: int, mode: str, device, generator: torch.Generator,
) -> torch.Tensor:
    """Returns (1, T_full) float mask, 1 at K observed frames within [0, seq_len)."""
    mask = torch.zeros(1, T_full, device=device, dtype=torch.float32)
    K_eff = min(K, seq_len)
    if mode == "linspace":
        idx = torch.linspace(0, seq_len - 1, K_eff, device=device).round().long()
        idx = torch.unique(idx)
    elif mode == "random":
        idx = torch.randperm(seq_len, generator=generator, device=device)[:K_eff]
    else:
        raise ValueError(f"unknown mode {mode!r}")
    mask[0, idx] = 1.0
    return mask


def build_cond(
    sample, z_dims, clip_model, object_encoder, device,
    cond_motion_dim, cond_motion_full=None, obs_mask_full=None,
):
    batch = collate_hoi([sample])
    seq_len = int(batch["seq_len"].item())
    T_full = batch["motion"].shape[1]
    motion_gt = batch["motion"].to(device)
    joints_gt = batch["joints"].to(device)
    contact_state = batch["contact_state"].to(device)
    contact_target_xyz = batch["contact_target_xyz"].to(device)
    phase = batch["phase"].to(device)
    support = batch["support"].to(device)
    obj_com = batch["obj_com_canonical"].to(device)
    obj_rot6d = batch["obj_rot6d_canonical"].to(device)
    obj_pos_world = batch["object_positions"].to(device)
    obj_rot_world = batch["object_rotations"].to(device)
    object_pc = batch["object_pc"].to(device)

    phase_soft = F.one_hot(phase.clamp_min(0).long(), z_dims.phase_classes).float()
    support_soft = F.one_hot(support.clamp_min(0).long(), z_dims.support_classes).float()
    z_int = pack_z_int(contact_state, contact_target_xyz, phase_soft, support_soft, z_dims)

    target_world = lift_object_local_to_world(
        contact_target_xyz, obj_pos_world, obj_rot_world,
    ).reshape(1, T_full, -1)
    object_traj = torch.cat([obj_com, obj_rot6d, target_world], dim=-1)
    init_pose = joints_gt[:, 0, :, :].reshape(1, -1)
    text_features, _ = encode_text_per_token(clip_model, batch["text"], device)
    obj_tokens = object_encoder(object_pc)

    cond = {
        "z_int": z_int,
        "object_world_traj": object_traj,
        "init_pose": init_pose,
        "text": text_features.float(),
        "object_tokens": obj_tokens,
    }
    if cond_motion_dim > 0:
        cond["cond_motion_input"] = torch.cat(
            [cond_motion_full, obs_mask_full], dim=-1,
        )
    return cond, seq_len, T_full, motion_gt, joints_gt, batch["rest_offsets"].to(device)


# ---------------------------------------------------------------------------
# Model loader (matches v9_sanity_checks.py)
# ---------------------------------------------------------------------------


def load_v9_model(cfg, ckpt_path: Path, device):
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
        prediction_target=str(cfg.model.diffusion.get("prediction_target", "x0")),
    )
    model = MotionAnchorDiff(AnchorDiffConfig(
        diffusion=diff_cfg, denoiser=denoiser_cfg,
        cfg_drop_prob=float(cfg.model.cfg_drop_prob),
    ))
    object_encoder = ObjectEncoder(
        num_input_points=int(cfg.model.object_encoder.num_input_points),
        num_output_tokens=int(cfg.model.object_encoder.num_output_tokens),
        feature_dim=int(cfg.model.object_encoder.feature_dim),
    )
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state.get("model", state))
    if "object_encoder" in state:
        object_encoder.load_state_dict(state["object_encoder"])
    elif "extra_modules" in state and "object_encoder" in state["extra_modules"]:
        object_encoder.load_state_dict(state["extra_modules"]["object_encoder"])
    model.to(device).eval()
    object_encoder.to(device).eval()
    return model, object_encoder, z_dims


# ---------------------------------------------------------------------------
# Diagnostic 1 — sampling matrix (cfg × K-mode), per §3.1, §3.2, §10
# ---------------------------------------------------------------------------


def diag_sampling_matrix(
    cfg, model, object_encoder, clip_model, z_dims, sample, device,
    cfgs=(0.0, 1.0), K_modes=("random", "linspace"), K=8,
    seed: int = 123, replacement: str = "none",
    output_skip: bool | None = None,
):
    cmd_dim = int(cfg.model.denoiser.get("cond_motion_dim", 0))
    rows = []
    for K_mode in K_modes:
        gen = torch.Generator(device=device).manual_seed(seed)
        T_full = int(sample["motion"].shape[0])
        seq_len = int(sample["seq_len"].item())
        kf_mask = build_keyframe_mask(seq_len, T_full, K, K_mode, device, gen)
        cond_motion = sample["motion"].unsqueeze(0).to(device) * kf_mask.unsqueeze(-1)
        obs_mask_full = kf_mask.unsqueeze(-1)
        for cfg_scale in cfgs:
            cond, sl, T_full, motion_gt, joints_gt, rest = build_cond(
                sample, z_dims, clip_model, object_encoder, device,
                cmd_dim, cond_motion, obs_mask_full,
            )
            with torch.no_grad():
                torch.manual_seed(seed)
                x0 = model.sample(
                    cond=cond, seq_length=T_full,
                    cfg_scale=float(cfg_scale),
                    replacement=replacement,
                    output_skip=output_skip,
                )
            joints_pred = decode_135_to_joints(x0, rest)
            metrics = decompose_metrics(
                joints_pred[0, :seq_len], joints_gt[0, :seq_len], kf_mask[0, :seq_len],
            )
            metrics_tw = transition_window_metrics(
                joints_pred[0, :seq_len], joints_gt[0, :seq_len],
                kf_mask[0, :seq_len], window=3,
            )
            rows.append({
                "K": K, "K_mode": K_mode, "cfg": cfg_scale,
                "replacement": replacement,
                "output_skip": bool(output_skip) if output_skip is not None else None,
                **metrics, **metrics_tw,
            })
    return rows


# ---------------------------------------------------------------------------
# v9_4 §9.5 transition window metrics (popping detector)
# ---------------------------------------------------------------------------


def transition_window_metrics(
    joints_pred: torch.Tensor, joints_gt: torch.Tensor,
    obs_mask_T: torch.Tensor, window: int = 3,
) -> dict:
    """Compute pop-detection metrics around observed keyframes.

    Per spec §9.5: for each observed-keyframe index k, accumulate errors
    over [k-w, k+w] for joint position, local velocity, root velocity,
    acceleration. Returns aggregate stats (mean over all keyframe windows).
    """
    T = joints_pred.shape[0]
    obs_idx = torch.where(obs_mask_T.bool())[0]
    if obs_idx.numel() == 0:
        return {
            "transition_window_mean_err_cm": 0.0,
            "transition_local_vel_jump_cm": 0.0,
            "transition_root_vel_jump_cm": 0.0,
            "transition_accel_jump_cm": 0.0,
        }

    # Build a frame mask of all frames within ±window of any keyframe
    win_mask = torch.zeros(T, dtype=torch.bool, device=joints_pred.device)
    for k in obs_idx.tolist():
        lo = max(0, k - window)
        hi = min(T, k + window + 1)
        win_mask[lo:hi] = True
    if not win_mask.any():
        return {
            "transition_window_mean_err_cm": 0.0,
            "transition_local_vel_jump_cm": 0.0,
            "transition_root_vel_jump_cm": 0.0,
            "transition_accel_jump_cm": 0.0,
        }

    err = (joints_pred - joints_gt).norm(dim=-1).mean(dim=-1)  # (T,)
    win_err_cm = float(err[win_mask].mean().item()) * 100

    # Velocity jumps (frame-to-frame velocity error) on window frames
    if T > 1:
        pred_root = joints_pred[:, 0:1, :]
        gt_root = joints_gt[:, 0:1, :]
        pred_local = joints_pred - pred_root
        gt_local = joints_gt - gt_root
        d_pl = pred_local[1:] - pred_local[:-1]
        d_gl = gt_local[1:] - gt_local[:-1]
        d_pr = pred_root.squeeze(1)[1:] - pred_root.squeeze(1)[:-1]
        d_gr = gt_root.squeeze(1)[1:] - gt_root.squeeze(1)[:-1]
        vel_local_err = (d_pl - d_gl).norm(dim=-1).mean(dim=-1)  # (T-1,)
        vel_root_err = (d_pr - d_gr).norm(dim=-1)                # (T-1,)
        win_mask_vel = win_mask[:-1] | win_mask[1:]
        if win_mask_vel.any():
            v_local = float(vel_local_err[win_mask_vel].mean().item()) * 100
            v_root = float(vel_root_err[win_mask_vel].mean().item()) * 100
        else:
            v_local = v_root = 0.0
        if T > 2:
            a_pl = d_pl[1:] - d_pl[:-1]
            a_gl = d_gl[1:] - d_gl[:-1]
            accel_err = (a_pl - a_gl).norm(dim=-1).mean(dim=-1)  # (T-2,)
            win_mask_acc = win_mask[:-2] | win_mask[1:-1] | win_mask[2:]
            if win_mask_acc.any():
                a = float(accel_err[win_mask_acc].mean().item()) * 100
            else:
                a = 0.0
        else:
            a = 0.0
    else:
        v_local = v_root = a = 0.0

    return {
        "transition_window_mean_err_cm": win_err_cm,
        "transition_local_vel_jump_cm": v_local,
        "transition_root_vel_jump_cm": v_root,
        "transition_accel_jump_cm": a,
    }


# ---------------------------------------------------------------------------
# Diagnostic 2 — condition sensitivity, per §5
# ---------------------------------------------------------------------------


def diag_condition_sensitivity(
    cfg, model, object_encoder, clip_model, z_dims, ds, device,
    primary_idx: int = 0, alt_idx: int = 1,
    cfg_scale: float = 1.0, K: int = 8, K_mode: str = "linspace",
    seed: int = 123, output_skip: bool | None = None,
) -> list:
    cmd_dim = int(cfg.model.denoiser.get("cond_motion_dim", 0))
    if cmd_dim == 0:
        return [{"skipped": "cond_motion_dim=0"}]
    primary = ds[primary_idx]
    alt = ds[alt_idx % len(ds)]
    T_full = int(primary["motion"].shape[0])
    seq_len = int(primary["seq_len"].item())
    gen = torch.Generator(device=device).manual_seed(seed)
    kf_mask = build_keyframe_mask(seq_len, T_full, K, K_mode, device, gen)

    motion_primary = primary["motion"].unsqueeze(0).to(device)
    motion_alt = alt["motion"].unsqueeze(0).to(device)
    if motion_alt.shape[1] != T_full:
        # pad/crop alt motion to T_full
        if motion_alt.shape[1] < T_full:
            pad = torch.zeros(1, T_full - motion_alt.shape[1], motion_alt.shape[2], device=device, dtype=motion_alt.dtype)
            motion_alt = torch.cat([motion_alt, pad], dim=1)
        else:
            motion_alt = motion_alt[:, :T_full]

    variants = {
        "GT_self":   motion_primary * kf_mask.unsqueeze(-1),
        "zeros":     torch.zeros_like(motion_primary) * kf_mask.unsqueeze(-1),
        "shuffled":  motion_primary[:, torch.randperm(T_full, generator=gen, device=device), :] * kf_mask.unsqueeze(-1),
        "wrong_clip": motion_alt * kf_mask.unsqueeze(-1),
        "reversed":  motion_primary.flip(dims=[1]) * kf_mask.unsqueeze(-1),
    }

    rows = []
    samples_x0 = {}
    for label, cond_motion in variants.items():
        cond, sl, T_full2, motion_gt, joints_gt, rest = build_cond(
            primary, z_dims, clip_model, object_encoder, device,
            cmd_dim, cond_motion, kf_mask.unsqueeze(-1),
        )
        with torch.no_grad():
            torch.manual_seed(seed)
            x0 = model.sample(
                cond=cond, seq_length=T_full2,
                cfg_scale=float(cfg_scale),
                output_skip=output_skip,
            )
        samples_x0[label] = x0
        joints_pred = decode_135_to_joints(x0, rest)
        metrics = decompose_metrics(
            joints_pred[0, :seq_len], joints_gt[0, :seq_len], kf_mask[0, :seq_len],
        )
        rows.append({"variant": label, **metrics})

    # Pairwise output deltas (in motion-135 L2)
    deltas = {}
    base = samples_x0["GT_self"][0, :seq_len]
    for label, x0 in samples_x0.items():
        if label == "GT_self":
            continue
        deltas[f"||GT_self - {label}||_motion"] = float((x0[0, :seq_len] - base).norm(dim=-1).mean().item())
    return {"rows": rows, "output_deltas": deltas}


# ---------------------------------------------------------------------------
# Diagnostic 3 — single-step teacher-forced, per §6
# ---------------------------------------------------------------------------


@torch.no_grad()
def diag_teacher_forced(
    cfg, model, object_encoder, clip_model, z_dims, sample, device,
    t_values=(0, 100, 250, 500, 750, 999),
    K: int = 8, K_mode: str = "linspace", seed: int = 123,
):
    cmd_dim = int(cfg.model.denoiser.get("cond_motion_dim", 0))
    T_full = int(sample["motion"].shape[0])
    seq_len = int(sample["seq_len"].item())
    gen = torch.Generator(device=device).manual_seed(seed)
    kf_mask = build_keyframe_mask(seq_len, T_full, K, K_mode, device, gen)
    motion_primary = sample["motion"].unsqueeze(0).to(device)
    cond_motion = motion_primary * kf_mask.unsqueeze(-1)

    cond, sl, T_full2, motion_gt, joints_gt, rest = build_cond(
        sample, z_dims, clip_model, object_encoder, device,
        cmd_dim, cond_motion, kf_mask.unsqueeze(-1),
    )

    # v9_4 §9.4: report raw-vs-skipped for both x0_mse and joint err
    output_skip_attr = bool(getattr(model.cfg.denoiser, "cond_motion_output_skip", False))
    motion_dim = int(model.cfg.denoiser.motion_dim)
    cond_motion_full = cond.get("cond_motion_input", None)
    if cond_motion_full is not None:
        cm = cond_motion_full[..., :motion_dim]
        om = cond_motion_full[..., motion_dim:motion_dim + 1]
    else:
        cm = om = None

    rows = []
    diff = model.diffusion
    for t_int in t_values:
        torch.manual_seed(seed)
        noise = torch.randn_like(motion_gt)
        t = torch.full((1,), int(t_int), device=device, dtype=torch.long)
        x_t = diff.q_sample(motion_gt, t, noise)
        net_out = model.denoiser(x_t, t, cond, cond_drop_mask=None)
        if diff.prediction_target == "v":
            x0_raw = diff.predict_x0_from_v(x_t, t, net_out)
        else:
            x0_raw = net_out
        # Skipped output (only meaningful when ckpt was trained with skip)
        if output_skip_attr and cm is not None:
            x0_skipped = om * cm + (1.0 - om) * x0_raw
        else:
            x0_skipped = x0_raw

        joints_raw = decode_135_to_joints(x0_raw, rest)
        joints_skipped = decode_135_to_joints(x0_skipped, rest)

        x0_mse_raw_full = float(((x0_raw - motion_gt)[0, :seq_len]).pow(2).mean().item())
        x0_mse_skipped_full = float(((x0_skipped - motion_gt)[0, :seq_len]).pow(2).mean().item())

        m_raw = decompose_metrics(
            joints_raw[0, :seq_len], joints_gt[0, :seq_len], kf_mask[0, :seq_len],
        )
        m_skipped = decompose_metrics(
            joints_skipped[0, :seq_len], joints_gt[0, :seq_len], kf_mask[0, :seq_len],
        )

        rows.append({
            "t": int(t_int),
            "x0_mse_raw": x0_mse_raw_full,
            "x0_mse_skipped": x0_mse_skipped_full,
            "raw_global_joint_error_cm": m_raw["global_joint_error_cm"],
            "raw_observed_frame_joint_error_cm": m_raw["observed_frame_joint_error_cm"],
            "skipped_global_joint_error_cm": m_skipped["global_joint_error_cm"],
            "skipped_observed_frame_joint_error_cm": m_skipped["observed_frame_joint_error_cm"],
            "skipped_unobserved_frame_joint_error_cm": m_skipped["unobserved_frame_joint_error_cm"],
            "skipped_root_aligned_joint_error_cm": m_skipped["root_aligned_joint_error_cm"],
            # legacy fields kept for back-compat with prior reports
            "x0_mse": x0_mse_skipped_full,
            **m_skipped,
        })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--version-name", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--training-mode", type=str, default="one_clip_overfit",
                        choices=["one_clip_overfit", "production"])
    parser.add_argument("--clip-idx", type=int, default=0,
                        help="primary clip index in the chairs dataset for diagnostics")
    parser.add_argument("--alt-clip-idx", type=int, default=1,
                        help="alternate clip for wrong-clip cond_motion variant")
    parser.add_argument("--replacement", type=str, default="none",
                        choices=["none", "x0", "x_t"],
                        help="v9_3 replacement ablation (spec §8.2). Active in "
                             "diag1 sampling matrix only.")
    parser.add_argument("--output-skip", type=str, default="auto",
                        choices=["auto", "true", "false"],
                        help="v9_4 hard-observation output skip. 'auto' uses "
                             "the value the ckpt was trained with (denoiser_cfg). "
                             "'true'/'false' override for sampler ablations.")
    args = parser.parse_args()
    output_skip_arg: bool | None = (
        None if args.output_skip == "auto"
        else (args.output_skip == "true")
    )

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    motion_rep = str(cfg.data.get("motion_representation", "smpl_pose_135"))
    pseudo_label_subdir = cfg.data.get("pseudo_label_subdir", None)
    entry = cfg.data.datasets[0]
    sub_dir = (str(Path(entry.root) / pseudo_label_subdir)
               if pseudo_label_subdir is not None else None)
    ds = HOIDataset(
        root=entry.root,
        pseudo_label_dir=sub_dir,
        max_seq_length=cfg.data.max_seq_length,
        augment=None,
        support_collapse_hand_support=True,
        surface_obj_pose=True,
        force_world_frame=bool(cfg.data.get("force_world_frame", False)),
        motion_representation=motion_rep,
    )
    print(f"Loaded HOIDataset(root={entry.root}, n_clips={len(ds)})")
    sample = ds[args.clip_idx]
    print(f"Primary clip: idx={args.clip_idx}, seq_id={sample['seq_id']}")

    print("\nLoading model + object_encoder + CLIP …")
    model, object_encoder, z_dims = load_v9_model(cfg, Path(args.ckpt), device)
    clip_model = load_clip_text_encoder(
        device=device,
        model_name=str(cfg.model.text_encoder.clip_version),
        download_root=str(cfg.model.text_encoder.get("download_root", "cache/clip")),
    )

    # === Diagnostic 1: sampling matrix ===
    print("\n" + "=" * 70)
    print("DIAG 1  cfg × K-mode sampling matrix on primary clip")
    print("=" * 70)
    matrix_rows = diag_sampling_matrix(
        cfg, model, object_encoder, clip_model, z_dims, sample, device,
        cfgs=(0.0, 1.0), K_modes=("random", "linspace"), K=8,
        replacement=args.replacement,
        output_skip=output_skip_arg,
    )
    cfg_drop_cm = bool(cfg.model.denoiser.get("cfg_drop_cond_motion", False))
    print(f"  replacement={args.replacement} output_skip={args.output_skip} cfg_drop_cond_motion={cfg_drop_cm}")
    print(f"  CFG semantics: " + (
        "keyframe + semantic CFG (cfg_drop_cond_motion=True)"
        if cfg_drop_cm else
        "SEMANTIC/object/text CFG only — cond_motion is retained in both branches"
    ))
    print(f"  {'K_mode':10s}  {'cfg':4s}  {'global':>8s}  {'root':>7s}  {'r-aligned':>9s}  {'obs':>7s}  {'unobs':>7s}  {'twin_err':>9s}  {'twin_vel':>9s}")
    for r in matrix_rows:
        print(
            f"  {r['K_mode']:10s}  {r['cfg']:.1f}   "
            f"{r['global_joint_error_cm']:8.2f}  {r['root_error_cm']:7.2f}  "
            f"{r['root_aligned_joint_error_cm']:9.2f}  {r['observed_frame_joint_error_cm']:7.2f}  "
            f"{r['unobserved_frame_joint_error_cm']:7.2f}  "
            f"{r['transition_window_mean_err_cm']:9.2f}  {r['transition_local_vel_jump_cm']:9.2f}"
        )

    # === Diagnostic 2: condition sensitivity ===
    print("\n" + "=" * 70)
    print("DIAG 2  Condition sensitivity (cfg=1.0, K=8 linspace)")
    print("=" * 70)
    sens = diag_condition_sensitivity(
        cfg, model, object_encoder, clip_model, z_dims, ds, device,
        primary_idx=args.clip_idx, alt_idx=args.alt_clip_idx,
        cfg_scale=1.0, K=8, K_mode="linspace",
        output_skip=output_skip_arg,
    )
    if isinstance(sens, dict):
        print(f"  {'variant':12s}  {'global':>8s}  {'r-aligned':>9s}  {'obs':>7s}  {'unobs':>7s}")
        for r in sens["rows"]:
            print(
                f"  {r['variant']:12s}  {r['global_joint_error_cm']:8.2f}  "
                f"{r['root_aligned_joint_error_cm']:9.2f}  {r['observed_frame_joint_error_cm']:7.2f}  "
                f"{r['unobserved_frame_joint_error_cm']:7.2f}"
            )
        print("  Output deltas (motion-135 L2 per frame, vs GT_self):")
        for k, v in sens["output_deltas"].items():
            print(f"    {k:50s} = {v:.4f}")

    # === Diagnostic 3: teacher-forced single-step ===
    print("\n" + "=" * 70)
    print("DIAG 3  Single-step teacher-forced (K=8 linspace)")
    print("=" * 70)
    tf_rows = diag_teacher_forced(
        cfg, model, object_encoder, clip_model, z_dims, sample, device,
    )
    print(f"  {'t':>4s}  {'x0_mse':>8s}  {'global':>8s}  {'root':>7s}  {'r-aligned':>9s}  {'obs':>7s}  {'unobs':>7s}")
    for r in tf_rows:
        print(
            f"  {r['t']:4d}  {r['x0_mse']:8.4f}  {r['global_joint_error_cm']:8.2f}  "
            f"{r['root_error_cm']:7.2f}  {r['root_aligned_joint_error_cm']:9.2f}  "
            f"{r['observed_frame_joint_error_cm']:7.2f}  {r['unobserved_frame_joint_error_cm']:7.2f}"
        )

    # === Save JSON per §11 schema ===
    primary_seq_id = str(sample["seq_id"])
    cfg_drop_cm_flag = bool(cfg.model.denoiser.get("cfg_drop_cond_motion", False))
    output_skip_flag = bool(cfg.model.denoiser.get("cond_motion_output_skip", False))
    out = {
        "version": args.version_name,
        "checkpoint": str(args.ckpt),
        "clip_id": primary_seq_id,
        "training_mode": args.training_mode,
        "prediction_target": str(cfg.model.diffusion.get("prediction_target", "x0")),
        "cond_motion_output_skip": output_skip_flag,
        "cfg_drop_cond_motion": cfg_drop_cm_flag,
        "sampler_replacement": args.replacement,
        "sampler_output_skip_arg": args.output_skip,
        "cfg_semantics": (
            "keyframe + semantic CFG (cfg_drop_cond_motion=True)"
            if cfg_drop_cm_flag else
            "semantic/object/text CFG only — cond_motion is retained"
        ),
        "lambda_obs": float(cfg.loss.get("cond_motion_keyframe_weight", 0.0)),
        "diffusion_unobserved_only": bool(cfg.loss.get("diffusion_unobserved_only", False)),
        "diag1_sampling_matrix": matrix_rows,
        "diag2_condition_sensitivity": sens if isinstance(sens, dict) else {"skipped": True},
        "diag3_teacher_forced": tf_rows,
    }
    out_json = out_dir / f"{args.version_name}_diagnostics.json"
    out_json.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\nResults JSON written to: {out_json}")


if __name__ == "__main__":
    main()
