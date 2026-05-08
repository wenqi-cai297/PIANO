"""Compute geometry_prior_g for AnchorDiff M1.5 dynamic-weight updates.

Per PLAN.md M1.5 Step 3: estimate how much each motion_263 group
affects the task-space HOI geometry (world joint position of the
contact-relevant body parts after recover_from_ric).

Method (finite-difference probes through the actual task chain):

    For each calibration clip and each group g:
      delta_g = unit_normal_probe * sigma_g * eps   (placed at group g's slice, zero elsewhere)
      x_perturbed = x_gt + delta_g
      world_joints_gt        = lift(recover_from_ric(x_gt))
      world_joints_perturbed = lift(recover_from_ric(x_perturbed))
      diff_world             = world_joints_perturbed - world_joints_gt
      sensitivity_g = sqrt(mean(||diff_world||^2)) / sqrt(mean(delta_g^2) + eps)

      geometry_prior_g = sensitivity_g^2

    Average across calibration clips, normalize, clamp.

Notes:
- We measure the WORLD-FRAME body part position change (after lift via
  per-clip R_y/T_xz/T_y). Pure motion_263 perturbation propagates to
  world via recover_from_ric + lift, so root_rot_vel — which gets
  cumsum'd over T frames — produces a much larger world-space delta
  than joint_pos_local for the same input perturbation.

- We use SEVERAL fixed probes per group (3 by default) and average.
  Single probe direction can be unlucky.

- We use ONLY the 5 contact-relevant SMPL-22 joint indices
  (PART_TO_JOINT) since AnchorDiff cares about hand/foot/pelvis
  contact, not full 22-joint motion.

Usage:
    python scripts/stage_b_generator/anchordiff_compute_geometry_prior.py \\
        --config configs/training/anchordiff_v2_weighted.yaml \\
        --calibration analyses/2026-05-08_anchordiff_dynamic_metric/calibration_clips.json \\
        --output analyses/2026-05-08_anchordiff_dynamic_metric \\
        --num-probes 3 \\
        --eps 0.05
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from piano.data.dataset import HOIDataset
from piano.training.anchor_consistency_loss import (
    PART_TO_JOINT,
    AnchorConsistencyConfig,
    lift_canonical_joints_to_world,
    lift_motion263_to_joints,
)
from piano.training.feature_groups import FEATURE_GROUPS, MOTION_DIM
from piano.utils.canonical_frame import get_canonicalize_transform_from_clip
from piano.utils.io_utils import load_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--calibration", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--num-probes", type=int, default=3,
                        help="random probe directions per group (averaged)")
    parser.add_argument("--eps", type=float, default=0.05,
                        help="probe magnitude as fraction of GT std")
    parser.add_argument("--probe-seed", type=int, default=42)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_json(Path(args.calibration))
    clips_meta = manifest["clips"]
    print(f"Loaded {len(clips_meta)} calibration clips")

    # Build per-subset HOIDataset (val-filtered, matching the manifest).
    from piano.data.dataset import build_subject_split, extract_subject_id
    subj_cfg = cfg.data.subject_split
    keys = []
    pseudo_label_subdir = cfg.data.get("pseudo_label_subdir", None)
    for entry in cfg.data.datasets:
        # We rebuild val_filter the same way the manifest builder did.
        from piano.utils.io_utils import load_json as _load
        meta_path = Path(entry.root) / "metadata_clean.json"
        meta = _load(meta_path)
        for m in meta:
            sid = extract_subject_id(entry.name, m.get("seq_id", ""))
            if sid is not None:
                keys.append((entry.name, sid))
    keys = sorted(set(keys))
    splits = build_subject_split(
        keys,
        train_pct=subj_cfg.train_pct,
        val_pct=subj_cfg.val_pct,
        seed=subj_cfg.seed,
    )
    val_filter = splits["val"]

    datasets: dict[str, HOIDataset] = {}
    for entry in cfg.data.datasets:
        sub_dir = (str(Path(entry.root) / pseudo_label_subdir)
                   if pseudo_label_subdir is not None else None)
        ds = HOIDataset(
            root=entry.root,
            pseudo_label_dir=sub_dir,
            max_seq_length=cfg.data.max_seq_length,
            subject_id_filter=val_filter,
            augment=None,
            support_collapse_hand_support=True,
            surface_obj_pose=True,
        )
        datasets[entry.name] = ds

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.probe_seed)

    # Pre-compute fixed probe direction unit vectors per group.
    # Each probe: a unit vector in the group's slice, zero elsewhere.
    # Different probes use different random directions on the unit sphere.
    probes_per_group: dict[str, np.ndarray] = {}
    for g in FEATURE_GROUPS:
        d = g.n_dims
        # (num_probes, d) random Gaussian, normalized
        v = rng.standard_normal((args.num_probes, d)).astype(np.float32)
        v = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)
        probes_per_group[g.name] = v

    part_idx = torch.tensor(PART_TO_JOINT, device=device, dtype=torch.long)
    part_weights = np.asarray(
        AnchorConsistencyConfig().part_weights, dtype=np.float32,
    )

    # Estimate per-clip per-group GT std (per-feature std over time, then
    # Frobenius across features in the group).
    # Use this to scale the probe and to normalize the residual factor later.
    print("Estimating per-group GT std on calibration clips...")
    group_std: dict[str, list[float]] = {g.name: [] for g in FEATURE_GROUPS}

    sensitivities_per_clip: dict[str, list[float]] = {g.name: [] for g in FEATURE_GROUPS}

    for clip_meta in clips_meta:
        subset = clip_meta["subset"]
        clip_idx = clip_meta["clip_idx_in_filtered_dataset"]
        ds = datasets[subset]
        if clip_idx >= len(ds):
            print(f"  WARN clip_idx out of range: {subset}/{clip_meta['seq_id']}")
            continue
        sample = ds[clip_idx]
        if str(sample["seq_id"]) != clip_meta["seq_id"]:
            print(f"  WARN seq_id mismatch at clip_idx={clip_idx}: "
                  f"saved={clip_meta['seq_id']} got={sample['seq_id']}")
            # try to find by seq_id scan
            found = None
            for cand in range(len(ds)):
                cs = ds[cand]
                if str(cs["seq_id"]) == clip_meta["seq_id"]:
                    found = cs
                    break
            if found is None:
                continue
            sample = found

        seq_len = int(sample["seq_len"].item())
        motion = sample["motion"].numpy()[:seq_len]                # (T, 263)
        joints_world = sample["joints"].numpy()[:seq_len]          # (T, 22, 3)
        contact_state = sample["contact_state"].numpy()[:seq_len]  # (T, 5)

        if seq_len < 5:
            continue
        contact_mask = (contact_state >= 0.5).astype(np.float32)
        contact_mask = contact_mask * part_weights[None, :]
        if float(contact_mask.sum()) < 1.0:
            continue

        # GT std per group (over time × dims in group, scalar per group)
        for g in FEATURE_GROUPS:
            grp = motion[:, g.lo:g.hi]
            group_std[g.name].append(float(grp.std()))

        # Compute (R_y, T_xz, T_y) once
        canon_gt = lift_motion263_to_joints(
            torch.from_numpy(motion).float().unsqueeze(0)
        ).squeeze(0).numpy()
        R_y_n, T_xz_n, T_y_n = get_canonicalize_transform_from_clip(joints_world, canon_gt)

        motion_t = torch.from_numpy(motion).float().to(device).unsqueeze(0)  # (1, T, 263)
        R_y_t = torch.tensor([R_y_n], device=device, dtype=torch.float32)
        T_xz_t = torch.tensor([T_xz_n], device=device, dtype=torch.float32)
        T_y_t = torch.tensor([T_y_n], device=device, dtype=torch.float32)

        # World joints from GT motion (5 contact parts)
        canon_gt_t = lift_motion263_to_joints(motion_t)
        world_gt_t = lift_canonical_joints_to_world(canon_gt_t, R_y_t, T_xz_t, T_y_t)
        parts_gt = world_gt_t.index_select(2, part_idx).squeeze(0).cpu().numpy()  # (T, 5, 3)

        # Per-group probes
        for g in FEATURE_GROUPS:
            probes = probes_per_group[g.name]    # (P, n_dims)
            sigma_g = max(group_std[g.name][-1], 1e-6)
            probe_norms_sum_sq = 0.0
            world_diff_sum_sq = 0.0
            for p in probes:
                # delta: zero everywhere, fill probe at group slice, scale by sigma * eps
                delta = np.zeros_like(motion)
                delta[:, g.lo:g.hi] = p[None, :] * (sigma_g * args.eps)
                motion_pert_t = motion_t + torch.from_numpy(delta).float().to(device).unsqueeze(0)
                canon_p_t = lift_motion263_to_joints(motion_pert_t)
                world_p_t = lift_canonical_joints_to_world(canon_p_t, R_y_t, T_xz_t, T_y_t)
                parts_p = world_p_t.index_select(2, part_idx).squeeze(0).cpu().numpy()
                diff = parts_p - parts_gt
                part_l2_sq = np.sum(diff * diff, axis=-1)          # (T, 5)
                world_diff_sum_sq += float(
                    (part_l2_sq * contact_mask).sum()
                    / max(float(contact_mask.sum()), 1.0)
                )
                probe_norms_sum_sq += float(np.mean(delta * delta))
            avg_world_mse = world_diff_sum_sq / args.num_probes
            avg_probe_mse = probe_norms_sum_sq / args.num_probes
            sensitivity = np.sqrt(avg_world_mse) / max(np.sqrt(avg_probe_mse), 1e-9)
            sensitivities_per_clip[g.name].append(float(sensitivity))

    # Aggregate
    sensitivities = {g.name: float(np.mean(sensitivities_per_clip[g.name]))
                     for g in FEATURE_GROUPS}
    sensitivities_std = {g.name: float(np.std(sensitivities_per_clip[g.name]))
                         for g in FEATURE_GROUPS}
    gt_std_mean = {g.name: float(np.mean(group_std[g.name])) for g in FEATURE_GROUPS}

    # geometry_prior_g = sensitivity_g^2
    raw_prior = {n: max(sensitivities[n] ** 2, 1e-9) for n in sensitivities}
    # Normalize: keep mean of prior (weighted by n_dims) at 1.0
    total_dims = MOTION_DIM
    weighted_sum = sum(raw_prior[g.name] * g.n_dims for g in FEATURE_GROUPS)
    scale = total_dims / max(weighted_sum, 1e-9)
    norm_prior = {n: raw_prior[n] * scale for n in raw_prior}

    summary = {
        "config": str(args.config),
        "calibration": str(args.calibration),
        "num_calibration_clips_used": len(sensitivities_per_clip[FEATURE_GROUPS[0].name]),
        "num_probes_per_group": args.num_probes,
        "eps": args.eps,
        "probe_seed": args.probe_seed,
        "feature_groups": [
            {"name": g.name, "lo": g.lo, "hi": g.hi, "n_dims": g.n_dims}
            for g in FEATURE_GROUPS
        ],
        "gt_std_mean": gt_std_mean,
        "sensitivity_mean": sensitivities,
        "sensitivity_std": sensitivities_std,
        "raw_geometry_prior": raw_prior,
        "normalized_geometry_prior": norm_prior,
    }
    out_json = out_dir / "geometry_prior.json"
    out_json.write_text(json.dumps(summary, indent=2))

    print()
    print(f"{'group':>20} | {'gt_std':>10} | {'sensitivity':>12} | {'raw_prior':>14} | {'norm_prior':>12}")
    print("-" * 80)
    for g in FEATURE_GROUPS:
        n = g.name
        print(
            f"{n:>20} | {gt_std_mean[n]:>10.5f} | {sensitivities[n]:>12.4f} | "
            f"{raw_prior[n]:>14.4f} | {norm_prior[n]:>12.4f}"
        )
    print(f"\nResults written to {out_json}")


if __name__ == "__main__":
    main()
