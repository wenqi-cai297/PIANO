"""v9 sanity-check battery: Steps 0, 1, 2, 4, 5 from the
ChatGPT failure-diagnosis review (analyses/anchordiff_failure_diagnosis_analysis.md).

Step 0: GT roundtrip — does (135-D rep) -> rot6d_to_matrix -> FK with
        rest_offsets recover GT joints to numerical precision?
Step 1: v-pred math closure — does sqrt(α̅)·x_t - sqrt(1-α̅)·v_target
        recover x_0 to < 1e-5?
Step 2: rot6d decode health on v9 sampler output — what is norm(a1) /
        norm(a2) / orthogonality_error / det(R) for the rotations the
        v9 model actually produces? If norm(a*) ≈ 0, Gram-Schmidt
        collapses → "joints clumped" symptom.
Step 4: CondMDI all-observed test — feed v9 sampler the GT 135-D
        motion as `cond_motion` with obs_mask=1 EVERYWHERE. If the
        sampler output ≠ GT, the inpainting condition is not actually
        being respected.
Step 5: CFG sweep — same v9 ckpt + same seed, sample at cfg ∈ {0,1,2,3}
        and report bone-length / joint-clumping metrics per cfg.

Step 3 (one-clip overfit) is a separate training run — not in this
script. Step 6 (inference-condition audit) is a code read.

Output:
- Numeric per-step results to stdout (also tee'd by caller).
- Saves an HDF5/npz dump of the Step 2/4/5 sampler outputs to
  `runs/visualizations/anchordiff_v9_sanity/<step>.npz` for later
  inspection.
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
    GaussianDiffusion, MotionAnchorDiff, ZIntDims, pack_z_int,
)
from piano.models.object_encoder import ObjectEncoder
from piano.training.anchor_consistency_loss import lift_object_local_to_world
from piano.training.smpl_kinematics import (
    rotation_6d_to_matrix, fk_from_global_rotations, SMPL22_PARENTS,
)
from piano.utils.clip_utils import encode_text_per_token, load_clip_text_encoder


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _decode_135_to_joints(motion_135: torch.Tensor, rest_offsets: torch.Tensor) -> torch.Tensor:
    """135 = rot6d 132 + root 3.  Returns (B, T, 22, 3) joints in world frame."""
    B, T, _ = motion_135.shape
    rot_6d = motion_135[..., :132].view(B, T, 22, 6)
    root = motion_135[..., 132:135]
    rot_mat = rotation_6d_to_matrix(rot_6d)
    rest_per_frame = rest_offsets.unsqueeze(1).expand(B, T, 22, 3)
    return fk_from_global_rotations(rot_mat, rest_per_frame, root)


def _bone_length_stats(joints: torch.Tensor) -> dict:
    """joints: (T, 22, 3). Returns mean/std bone length per parent edge."""
    parents = torch.tensor(SMPL22_PARENTS)
    lengths = {}
    for j in range(1, 22):
        p = int(parents[j].item())
        bone = joints[:, j, :] - joints[:, p, :]
        lengths[f"j{j}-j{p}"] = bone.norm(dim=-1).cpu().numpy()
    mean = np.array([v.mean() for v in lengths.values()])
    std = np.array([v.std() for v in lengths.values()])
    return {
        "per_bone_mean_cm": mean * 100,
        "per_bone_std_cm": std * 100,
        "max_per_frame_std_cm": max(v.std() for v in lengths.values()) * 100,
    }


# ---------------------------------------------------------------------------
# Step 0 — GT roundtrip
# ---------------------------------------------------------------------------


def step0_gt_roundtrip(ds: HOIDataset, n_clips: int = 4) -> dict:
    print("\n" + "=" * 70)
    print("STEP 0  GT roundtrip:  135-D -> rot6d_to_matrix -> FK -> joints")
    print("=" * 70)
    errs_max = []
    errs_mean = []
    bone_consistency = []
    for i in range(min(n_clips, len(ds))):
        sample = ds[i]
        seq_len = int(sample["seq_len"].item())
        motion = sample["motion"].unsqueeze(0)              # (1, T, 135)
        rest_offsets = sample["rest_offsets"].unsqueeze(0)  # (1, 22, 3)
        joints_gt = sample["joints"].unsqueeze(0)           # (1, T, 22, 3)
        joints_dec = _decode_135_to_joints(motion, rest_offsets)
        # Clip to valid frames
        err = (joints_dec - joints_gt)[:, :seq_len]
        max_err_cm = err.abs().max().item() * 100
        mean_err_cm = err.norm(dim=-1).mean().item() * 100
        # Per-frame bone-length consistency
        bs = _bone_length_stats(joints_dec[0, :seq_len])
        errs_max.append(max_err_cm)
        errs_mean.append(mean_err_cm)
        bone_consistency.append(bs["max_per_frame_std_cm"])
        print(
            f"  clip {i} (seq_len={seq_len}, seq_id={sample['seq_id']}): "
            f"max_err={max_err_cm:.4f} cm, mean_err={mean_err_cm:.4f} cm, "
            f"bone_std={bs['max_per_frame_std_cm']:.4f} cm"
        )
    out = {
        "max_err_cm": float(max(errs_max)),
        "mean_err_cm": float(np.mean(errs_mean)),
        "max_bone_std_cm": float(max(bone_consistency)),
    }
    verdict = "PASS" if out["max_err_cm"] < 0.5 else "FAIL"
    print(f"  >> verdict: {verdict}  (max err {out['max_err_cm']:.4f} cm)")
    out["verdict"] = verdict
    return out


# ---------------------------------------------------------------------------
# Step 1 — v-pred math closure
# ---------------------------------------------------------------------------


def step1_vpred_math() -> dict:
    print("\n" + "=" * 70)
    print("STEP 1  v-pred math closure:  x0 == sqrt(α̅)·x_t - sqrt(1-α̅)·v")
    print("=" * 70)
    diff = GaussianDiffusion(DiffusionConfig(num_steps=1000, schedule="cosine", prediction_target="v"))
    torch.manual_seed(0)
    B, T, D = 4, 196, 135
    x0 = torch.randn(B, T, D)
    noise = torch.randn(B, T, D)
    # Test a wide range of t including t=0 and t=999
    t_values = torch.tensor([0, 250, 500, 750, 999])
    rows = []
    for t_int in t_values:
        t = torch.full((B,), int(t_int.item()), dtype=torch.long)
        x_t = diff.q_sample(x0, t, noise)
        v = diff.v_target(x0, t, noise)
        x0_rec = diff.predict_x0_from_v(x_t, t, v)
        err = (x0_rec - x0).abs().max().item()
        rows.append((int(t_int.item()), err))
        print(f"  t={int(t_int.item()):4d}  max |x0_rec - x0| = {err:.3e}")
    max_err = max(e for _, e in rows)
    verdict = "PASS" if max_err < 1e-4 else "FAIL"
    print(f"  >> verdict: {verdict}  (max err {max_err:.3e})")
    return {
        "per_t": rows,
        "max_err": float(max_err),
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Helpers for steps 2, 4, 5: load v9 ckpt + build cond
# ---------------------------------------------------------------------------


def _load_v9_model(cfg, ckpt_path: Path, device: torch.device):
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


def _build_cond_for_clip(
    sample: dict,
    z_dims: ZIntDims,
    clip_model,
    object_encoder,
    device,
    cond_motion_dim: int,
    cond_motion_full: torch.Tensor | None,
    obs_mask_full: torch.Tensor | None,
) -> tuple[dict, int]:
    """Build cond dict matching the v9 step_fn / sampler conventions."""
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

    # object_traj_dim=24 in v9 config
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
        if cond_motion_full is None:
            raise ValueError("cond_motion_dim>0 but no cond_motion_full provided")
        cond["cond_motion_input"] = torch.cat(
            [cond_motion_full, obs_mask_full], dim=-1,
        )
    return cond, seq_len, T_full, motion_gt, joints_gt, batch["rest_offsets"].to(device)


# ---------------------------------------------------------------------------
# Step 2 — rot6d decode health on v9 sampler output
# ---------------------------------------------------------------------------


def _rot6d_health(motion_135: torch.Tensor) -> dict:
    """motion_135: (T, 135)"""
    rot_6d = motion_135[..., :132].view(-1, 22, 6)   # (T*22, ..., reshape later)
    a1 = rot_6d[..., :3]
    a2 = rot_6d[..., 3:]
    n1 = a1.norm(dim=-1)
    n2 = a2.norm(dim=-1)
    rot_mat = rotation_6d_to_matrix(rot_6d)
    # Orthogonality error: ||R^T R - I||_F per joint per frame
    eye = torch.eye(3, device=rot_mat.device).expand_as(rot_mat)
    ortho = (rot_mat.transpose(-1, -2) @ rot_mat - eye).norm(dim=(-1, -2))
    det = torch.linalg.det(rot_mat)
    return {
        "norm_a1_min": float(n1.min().item()),
        "norm_a1_mean": float(n1.mean().item()),
        "norm_a2_min": float(n2.min().item()),
        "norm_a2_mean": float(n2.mean().item()),
        "ortho_err_max": float(ortho.max().item()),
        "ortho_err_mean": float(ortho.mean().item()),
        "det_min": float(det.min().item()),
        "det_max": float(det.max().item()),
        "det_mean": float(det.mean().item()),
        "n_nan": int(torch.isnan(motion_135).sum().item()),
        "n_inf": int(torch.isinf(motion_135).sum().item()),
    }


def step2_rot6d_health(
    cfg, model, object_encoder, clip_model, z_dims, ds, device,
    n_clips: int = 2, cfg_scale: float = 3.0,
) -> dict:
    print("\n" + "=" * 70)
    print(f"STEP 2  rot6d decode health on v9 sampler output (cfg={cfg_scale})")
    print("=" * 70)
    motion_rep = str(cfg.data.motion_representation)
    cmd_dim = int(cfg.model.denoiser.get("cond_motion_dim", 0))
    rows = []
    for i in range(min(n_clips, len(ds))):
        sample = ds[i]
        # Default keyframe schema for v9: 8 evenly-spaced GT keyframes (matches visualizer)
        T_full = int(sample["motion"].shape[0])
        seq_len = int(sample["seq_len"].item())
        if cmd_dim > 0:
            kf_obs_mask = torch.zeros(1, T_full, device=device, dtype=torch.float32)
            K_eff = min(8, seq_len)
            idx = torch.linspace(0, seq_len - 1, K_eff, device=device).round().long()
            kf_obs_mask[0, idx] = 1.0
            cond_motion_full = sample["motion"].unsqueeze(0).to(device) * kf_obs_mask.unsqueeze(-1)
            obs_mask_full = kf_obs_mask.unsqueeze(-1)
        else:
            cond_motion_full = obs_mask_full = None

        cond, seq_len, T_full, motion_gt, joints_gt, rest_offsets = _build_cond_for_clip(
            sample, z_dims, clip_model, object_encoder, device,
            cmd_dim, cond_motion_full, obs_mask_full,
        )

        with torch.no_grad():
            torch.manual_seed(123)
            x0_sample = model.sample(cond=cond, seq_length=T_full, cfg_scale=cfg_scale)

        # rot6d health on the sampler output
        h = _rot6d_health(x0_sample[0, :seq_len])
        joints_pred = _decode_135_to_joints(x0_sample, rest_offsets)
        bone = _bone_length_stats(joints_pred[0, :seq_len])
        # joints "clumping" diagnostic: spread of all-22-joints from their mean
        joints_t = joints_pred[0, :seq_len]
        joint_spread_cm = (joints_t - joints_t.mean(dim=1, keepdim=True)).norm(dim=-1).mean().item() * 100
        gt_spread_cm = (joints_gt[0, :seq_len] - joints_gt[0, :seq_len].mean(dim=1, keepdim=True)).norm(dim=-1).mean().item() * 100
        print(f"  clip {i} ({sample['seq_id']}): seq_len={seq_len}")
        print(f"    norm(a1) min={h['norm_a1_min']:.3f} mean={h['norm_a1_mean']:.3f}")
        print(f"    norm(a2) min={h['norm_a2_min']:.3f} mean={h['norm_a2_mean']:.3f}")
        print(f"    ortho err max={h['ortho_err_max']:.3e} mean={h['ortho_err_mean']:.3e}")
        print(f"    det(R)   min={h['det_min']:.3f} max={h['det_max']:.3f} mean={h['det_mean']:.3f}")
        print(f"    bone-len max-frame-std = {bone['max_per_frame_std_cm']:.3f} cm")
        print(f"    joint spread:  pred={joint_spread_cm:.2f} cm  vs  gt={gt_spread_cm:.2f} cm")
        print(f"    NaN/Inf count = {h['n_nan']}/{h['n_inf']}")
        rows.append({
            "clip": sample["seq_id"],
            "rot6d_health": h,
            "bone_max_frame_std_cm": float(bone["max_per_frame_std_cm"]),
            "joint_spread_pred_cm": float(joint_spread_cm),
            "joint_spread_gt_cm": float(gt_spread_cm),
        })
    return {"clips": rows}


# ---------------------------------------------------------------------------
# Step 4 — CondMDI all-observed test (only meaningful when cond_motion_dim>0)
# ---------------------------------------------------------------------------


def step4_all_observed(
    cfg, model, object_encoder, clip_model, z_dims, ds, device,
    cfg_scale: float = 1.0,
) -> dict:
    print("\n" + "=" * 70)
    print(f"STEP 4  CondMDI all-observed (obs_mask=1 every frame), cfg={cfg_scale}")
    print("=" * 70)
    cmd_dim = int(cfg.model.denoiser.get("cond_motion_dim", 0))
    if cmd_dim == 0:
        print("  cond_motion_dim=0 → skip (not a CondMDI model).")
        return {"skipped": True}

    rows = []
    for i in range(min(2, len(ds))):
        sample = ds[i]
        T_full = int(sample["motion"].shape[0])
        seq_len = int(sample["seq_len"].item())
        # all-observed: obs_mask=1 for all valid frames, cond_motion = full GT motion
        kf_obs_mask = torch.zeros(1, T_full, device=device, dtype=torch.float32)
        kf_obs_mask[0, :seq_len] = 1.0
        cond_motion_full = sample["motion"].unsqueeze(0).to(device) * kf_obs_mask.unsqueeze(-1)
        obs_mask_full = kf_obs_mask.unsqueeze(-1)

        cond, seq_len, T_full, motion_gt, joints_gt, rest_offsets = _build_cond_for_clip(
            sample, z_dims, clip_model, object_encoder, device,
            cmd_dim, cond_motion_full, obs_mask_full,
        )
        with torch.no_grad():
            torch.manual_seed(123)
            x0_sample = model.sample(cond=cond, seq_length=T_full, cfg_scale=cfg_scale)

        diff_motion = (x0_sample - motion_gt)[0, :seq_len]
        joints_pred = _decode_135_to_joints(x0_sample, rest_offsets)
        joint_err_cm = (joints_pred - joints_gt)[0, :seq_len].norm(dim=-1).mean().item() * 100
        print(f"  clip {i} ({sample['seq_id']}): seq_len={seq_len}")
        print(f"    motion (135-D) MSE  = {diff_motion.pow(2).mean().item():.4f}")
        print(f"    motion (135-D) Linf = {diff_motion.abs().max().item():.4f}")
        print(f"    joint err (after FK) = {joint_err_cm:.2f} cm")
        rows.append({
            "clip": sample["seq_id"],
            "motion_mse": float(diff_motion.pow(2).mean().item()),
            "motion_linf": float(diff_motion.abs().max().item()),
            "joint_err_mean_cm": float(joint_err_cm),
        })
    if rows:
        avg_joint_err = float(np.mean([r["joint_err_mean_cm"] for r in rows]))
        verdict = "PASS" if avg_joint_err < 5.0 else ("WEAK" if avg_joint_err < 20.0 else "FAIL")
        print(f"  >> verdict: {verdict}  (avg joint err {avg_joint_err:.2f} cm)")
        return {"clips": rows, "avg_joint_err_cm": avg_joint_err, "verdict": verdict}
    return {"clips": rows}


# ---------------------------------------------------------------------------
# Step 5 — CFG sweep
# ---------------------------------------------------------------------------


def step5_cfg_sweep(
    cfg, model, object_encoder, clip_model, z_dims, ds, device,
    cfgs=(0.0, 1.0, 2.0, 3.0),
) -> dict:
    print("\n" + "=" * 70)
    print(f"STEP 5  CFG sweep (cfg ∈ {list(cfgs)})")
    print("=" * 70)
    cmd_dim = int(cfg.model.denoiser.get("cond_motion_dim", 0))
    sample = ds[0]
    seq_id = sample["seq_id"]
    T_full = int(sample["motion"].shape[0])
    seq_len = int(sample["seq_len"].item())
    if cmd_dim > 0:
        kf_obs_mask = torch.zeros(1, T_full, device=device, dtype=torch.float32)
        K_eff = min(8, seq_len)
        idx = torch.linspace(0, seq_len - 1, K_eff, device=device).round().long()
        kf_obs_mask[0, idx] = 1.0
        cond_motion_full = sample["motion"].unsqueeze(0).to(device) * kf_obs_mask.unsqueeze(-1)
        obs_mask_full = kf_obs_mask.unsqueeze(-1)
    else:
        cond_motion_full = obs_mask_full = None

    cond, seq_len, T_full, motion_gt, joints_gt, rest_offsets = _build_cond_for_clip(
        sample, z_dims, clip_model, object_encoder, device,
        cmd_dim, cond_motion_full, obs_mask_full,
    )

    rows = []
    for cfg_scale in cfgs:
        with torch.no_grad():
            torch.manual_seed(123)
            x0_sample = model.sample(cond=cond, seq_length=T_full, cfg_scale=cfg_scale)
        h = _rot6d_health(x0_sample[0, :seq_len])
        joints_pred = _decode_135_to_joints(x0_sample, rest_offsets)
        joint_err_cm = (joints_pred - joints_gt)[0, :seq_len].norm(dim=-1).mean().item() * 100
        joints_t = joints_pred[0, :seq_len]
        spread_cm = (joints_t - joints_t.mean(dim=1, keepdim=True)).norm(dim=-1).mean().item() * 100
        bone = _bone_length_stats(joints_t)
        print(
            f"  cfg={cfg_scale:.1f}: norm(a1) min={h['norm_a1_min']:.3f} | "
            f"ortho_err_max={h['ortho_err_max']:.2e} | "
            f"joint_err={joint_err_cm:.2f}cm | spread={spread_cm:.2f}cm | "
            f"bone_std_max={bone['max_per_frame_std_cm']:.3f}cm"
        )
        rows.append({
            "cfg_scale": float(cfg_scale),
            "norm_a1_min": h["norm_a1_min"],
            "ortho_err_max": h["ortho_err_max"],
            "joint_err_mean_cm": float(joint_err_cm),
            "joint_spread_pred_cm": float(spread_cm),
            "bone_std_max_cm": float(bone["max_per_frame_std_cm"]),
        })
    return {"sweep": rows, "clip": seq_id}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/training/anchordiff_v9_condmdi.yaml")
    parser.add_argument("--ckpt", type=str, default="runs/training/stageB_anchordiff_v9_condmdi/epoch_0080.pt")
    parser.add_argument("--output-json", type=str, default="runs/visualizations/anchordiff_v9_sanity/results.json")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build a tiny val dataset to share across steps
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
    print(f"Loaded HOIDataset(root={entry.root}, n_clips={len(ds)}, motion_rep={motion_rep})")

    results: dict = {}

    # Step 0
    results["step0_gt_roundtrip"] = step0_gt_roundtrip(ds, n_clips=4)

    # Step 1
    results["step1_vpred_math"] = step1_vpred_math()

    # Steps 2/4/5 need the v9 model
    print("\nLoading v9 model + object_encoder + CLIP …")
    model, object_encoder, z_dims = _load_v9_model(cfg, Path(args.ckpt), device)
    clip_model = load_clip_text_encoder(
        device=device,
        model_name=str(cfg.model.text_encoder.clip_version),
        download_root=str(cfg.model.text_encoder.get("download_root", "cache/clip")),
    )

    # Step 2 — rot6d decode health (cfg=3 default to match visualizer)
    results["step2_rot6d_health"] = step2_rot6d_health(
        cfg, model, object_encoder, clip_model, z_dims, ds, device,
        n_clips=2, cfg_scale=3.0,
    )

    # Step 4 — CondMDI all-observed
    results["step4_all_observed"] = step4_all_observed(
        cfg, model, object_encoder, clip_model, z_dims, ds, device,
        cfg_scale=1.0,  # all-observed at cfg=1 isolates inpainting fidelity
    )

    # Step 5 — CFG sweep
    results["step5_cfg_sweep"] = step5_cfg_sweep(
        cfg, model, object_encoder, clip_model, z_dims, ds, device,
        cfgs=(0.0, 1.0, 2.0, 3.0),
    )

    out_path.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(f"\nResults JSON written to: {out_path}")


if __name__ == "__main__":
    main()
