"""Visual sanity check for AnchorDiff M1 output.

For each clip:
  1. Load batch (motion, joints, conditions) from HOIDataset.
  2. Use **GT z_int** + GT object_world_traj + GT init_pose + text + object_pc
     as conditioning (this isolates whether the diffusion model learned to
     generate plausible motion from the supervision it was trained on; we
     defer Stage A predicted-z_int evaluation to M2 ship gates).
  3. Run model.sample(cfg_scale=1.0) → predicted motion_263.
  4. recover_from_ric → uniform-skel canonical joints.
  5. Lift to world via per-clip (R_y, T_xz, T_y) computed from GT joints[0]
     and GT motion[0] (matches the training-time anchor loss path).
  6. Render generated MP4 + GT MP4 (object overlay) for side-by-side review.

Usage:
    python scripts/stage_b_generator/anchordiff_sample_visualize.py \\
        --config configs/training/anchordiff_v1.yaml \\
        --ckpt runs/training/stageB_anchordiff_v1/best_val.pt \\
        --output runs/visualizations/anchordiff_m1_visual_check \\
        --clips chairs:0 imhd:0 neuraldome:0 omomo_correct_v2:0 \\
                chairs:Sub0001_Obj116_Seg0_300 \\
        --cfg-scale 1.0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from piano.data.dataset import HOIDataset, collate_hoi
from piano.inference.anchor_postprocess import translate_world_joints_to_active_anchors
from piano.inference.visualize_motion import render_motion_video
from piano.models.motion_anchordiff import (
    AnchorDenoiserConfig, AnchorDiffConfig, DiffusionConfig,
    MotionAnchorDiff, ZIntDims, pack_z_int,
)
from piano.models.object_encoder import ObjectEncoder
from piano.training.anchor_consistency_loss import (
    lift_motion263_to_joints,
    lift_canonical_joints_to_world,
    lift_object_local_to_world,
)
from piano.utils.canonical_frame import (
    get_canonicalize_transform_from_clip, y_rotation_matrix,
)
from piano.utils.clip_utils import encode_text_per_token, load_clip_text_encoder


def _resolve_clip(ds: HOIDataset, spec: str) -> int:
    try:
        return int(spec)
    except ValueError:
        pass
    for i in range(len(ds)):
        sample = ds[i]
        if spec in str(sample["seq_id"]):
            return i
    raise ValueError(f"clip '{spec}' not found")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--clips", type=str, nargs="+", required=True,
                        help="subset:index_or_seq_id, e.g. chairs:0")
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--dpi", type=int, default=80)
    parser.add_argument("--post-anchor-translate-strength", type=float, default=0.0)
    parser.add_argument("--post-anchor-smooth-window", type=int, default=9)
    parser.add_argument("--post-anchor-max-offset", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    motion_representation = str(cfg.data.get("motion_representation", "motion_263"))
    object_traj_dim = int(cfg.model.denoiser.object_traj_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.seed is not None:
        np.random.seed(int(args.seed))
        torch.manual_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))

    # Build model + load ckpt
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

    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    if "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    if "object_encoder" in state:
        object_encoder.load_state_dict(state["object_encoder"])
    elif "extra_modules" in state and "object_encoder" in state["extra_modules"]:
        object_encoder.load_state_dict(state["extra_modules"]["object_encoder"])
    model.to(device).eval()
    object_encoder.to(device).eval()

    clip_model = load_clip_text_encoder(
        device=device,
        model_name=str(cfg.model.text_encoder.clip_version),
        download_root=str(cfg.model.text_encoder.get("download_root", "cache/clip")),
    )

    # Build per-subset HOIDataset (no augment)
    pseudo_label_subdir = cfg.data.get("pseudo_label_subdir", None)
    datasets = {}
    for entry in cfg.data.datasets:
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
            motion_representation=motion_representation,
        )
        datasets[entry.name] = ds

    print(f"Loaded ckpt: {args.ckpt}")
    print(f"cfg_scale = {args.cfg_scale}")
    print()

    for spec in args.clips:
        if ":" not in spec:
            print(f"Skip: malformed spec '{spec}'")
            continue
        subset, clip_spec = spec.split(":", 1)
        if subset not in datasets:
            print(f"Skip: unknown subset '{subset}'")
            continue
        ds = datasets[subset]
        try:
            idx = _resolve_clip(ds, clip_spec)
        except ValueError as e:
            print(f"Skip: {e}")
            continue

        sample = ds[idx]
        seq_id = sample["seq_id"]
        seq_len = int(sample["seq_len"].item())
        batch = collate_hoi([sample])

        motion_gt = batch["motion"].to(device)                       # (1, T, 263)
        joints_world_gt = batch["joints"].to(device)                 # (1, T, 22, 3)
        contact_state = batch["contact_state"].to(device)
        contact_target_xyz = batch["contact_target_xyz"].to(device)
        phase = batch["phase"].to(device)
        support = batch["support"].to(device)
        obj_com = batch["obj_com_canonical"].to(device)
        obj_rot6d = batch["obj_rot6d_canonical"].to(device)
        obj_pos_world = batch["object_positions"].to(device)
        obj_rot_world = batch["object_rotations"].to(device)
        object_pc = batch["object_pc"].to(device)

        T_full = motion_gt.shape[1]

        # Pack GT z_int
        import torch.nn.functional as F
        phase_soft = F.one_hot(phase.clamp_min(0).long(), num_classes=z_dims.phase_classes).float()
        support_soft = F.one_hot(support.clamp_min(0).long(), num_classes=z_dims.support_classes).float()
        z_int = pack_z_int(contact_state, contact_target_xyz, phase_soft, support_soft, z_dims)
        object_traj_parts = [obj_com, obj_rot6d]
        if object_traj_dim == 24:
            target_world = lift_object_local_to_world(
                contact_target_xyz,
                obj_pos_world,
                obj_rot_world,
            ).reshape(1, T_full, -1)
            object_traj_parts.append(target_world)
        object_traj = torch.cat(object_traj_parts, dim=-1)
        if object_traj.shape[-1] != object_traj_dim:
            raise ValueError(
                f"object_traj_dim={object_traj_dim} but built {object_traj.shape[-1]}"
            )
        init_pose = joints_world_gt[:, 0, :, :].reshape(1, -1)

        text_features, _ = encode_text_per_token(clip_model, batch["text"], device)
        obj_tokens = object_encoder(object_pc)

        cond = {
            "z_int": z_int,
            "object_world_traj": object_traj,
            "init_pose": init_pose,
            "text": text_features.float(),
            "object_tokens": obj_tokens,
        }

        # Sample
        print(f"[{subset}/{seq_id}] T={seq_len}  sampling (1000 DDPM steps, cfg={args.cfg_scale})...")
        with torch.no_grad():
            x0_sample = model.sample(cond=cond, seq_length=T_full, cfg_scale=args.cfg_scale)
        # Recover joints. v1-v3 output motion_263 and need recover/lift;
        # v4 outputs flattened world-frame joints directly;
        # v5 (joints22_world_with_rot6d) outputs 198-D = (jpos: 66, rot_6d: 132),
        # we just take the jpos sub-vector for visualization (rot_6d would need
        # FK + bone_offsets to render; jpos is already world XYZ).
        if motion_representation == "joints22_world":
            joints_pred_world_t = x0_sample.view(1, T_full, 22, 3)
        elif motion_representation == "joints22_world_with_rot6d":
            joints_pred_world_t = x0_sample[..., :66].view(1, T_full, 22, 3)
        else:
            canon_pred = lift_motion263_to_joints(x0_sample)          # (1, T, 22, 3)
            canon_gt = lift_motion263_to_joints(motion_gt)
            # Compute (R_y, T_xz, T_y) from GT joints[0] + GT motion[0]
            j0 = joints_world_gt[0].cpu().numpy()
            cn0 = canon_gt[0].cpu().numpy()
            R_y_n, T_xz_n, T_y_n = get_canonicalize_transform_from_clip(j0, cn0)
            R_y = torch.tensor([R_y_n], device=device)
            T_xz = torch.tensor([[T_xz_n[0], T_xz_n[1]]], device=device)
            T_y = torch.tensor([T_y_n], device=device)
            joints_pred_world_t = lift_canonical_joints_to_world(canon_pred, R_y, T_xz, T_y)

        if args.post_anchor_translate_strength > 0.0:
            joints_pred_world_t, guide_stats = translate_world_joints_to_active_anchors(
                joints_pred_world_t,
                contact_state,
                contact_target_xyz,
                obj_pos_world,
                obj_rot_world,
                strength=float(args.post_anchor_translate_strength),
                smooth_window=int(args.post_anchor_smooth_window),
                max_offset_m=float(args.post_anchor_max_offset),
            )
            print(f"  post-anchor translation: {guide_stats}")

        joints_pred_world = joints_pred_world_t.squeeze(0).cpu().numpy()
        # Truncate to seq_len for cleaner viz
        joints_pred_world = joints_pred_world[:seq_len]
        joints_world_gt_np = joints_world_gt.squeeze(0).cpu().numpy()[:seq_len]
        obj_pos_np = obj_pos_world.squeeze(0).cpu().numpy()[:seq_len]
        obj_rot_np = obj_rot_world.squeeze(0).cpu().numpy()[:seq_len]
        obj_pc_np = object_pc.squeeze(0).cpu().numpy()

        # Render
        title_pred = f"{subset}/{seq_id}\n[M1 PREDICTED]\ntext: {batch['text'][0][:80]}"
        title_gt = f"{subset}/{seq_id}\n[GT]\ntext: {batch['text'][0][:80]}"
        out_pred = out_dir / f"{subset}_{seq_id}_predicted.mp4"
        out_gt = out_dir / f"{subset}_{seq_id}_gt.mp4"
        print(f"  rendering predicted → {out_pred.name}")
        render_motion_video(
            joints=joints_pred_world,
            output_path=out_pred,
            fps=args.fps,
            title=title_pred,
            object_positions=obj_pos_np,
            object_rotations=obj_rot_np,
            object_pc=obj_pc_np,
            dpi=args.dpi,
        )
        print(f"  rendering gt        → {out_gt.name}")
        render_motion_video(
            joints=joints_world_gt_np,
            output_path=out_gt,
            fps=args.fps,
            title=title_gt,
            object_positions=obj_pos_np,
            object_rotations=obj_rot_np,
            object_pc=obj_pc_np,
            dpi=args.dpi,
        )

    print(f"\nDone. Videos in {out_dir}/")


if __name__ == "__main__":
    main()
