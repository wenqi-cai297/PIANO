"""Diagnose which motion_263 feature dimensions the model fails on.

motion_263 layout (HumanML3D, 22 joints):
    [0]      : root rotation velocity (Y-axis arcsin, rad/frame)
    [1:3]    : root linear velocity (XZ in body frame, m/frame)
    [3]      : root height Y (absolute, m)
    [4:67]   : body-relative joint positions (21 joints × 3)
    [67:193] : joint rotations (21 joints × 6D rep)
    [193:259]: joint velocities (22 joints × 3)
    [259:263]: foot contact labels (4)

The "person keeps spinning" symptom strongly suggests feature 0 is
noisy. Feature 0 is 1/263 of the MSE so a tiny per-feature error
becomes catastrophic after cumsum integration over 196 frames
(random walk std grows as √T).

This diagnostic runs sample() at multiple cfg_scales and compares
each feature group's RMSE against GT. Helps decide whether to
retrain with feature-weighted MSE or if inference-time CFG fixes it.
"""
from __future__ import annotations

import argparse
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
from piano.training.feature_groups import FEATURE_GROUPS as FEATURE_GROUP_DEFS
from piano.utils.clip_utils import encode_text_per_token, load_clip_text_encoder


FEATURE_GROUPS = [(g.name, g.lo, g.hi) for g in FEATURE_GROUP_DEFS]


def build_model(cfg, device):
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
    return model.to(device), object_encoder.to(device), z_dims


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--num-clips", type=int, default=8)
    parser.add_argument("--cfg-scales", type=float, nargs="+", default=[1.0, 3.0, 5.0])
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, object_encoder, z_dims = build_model(cfg, device)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    model_state = state.get("model", state)
    model.load_state_dict(model_state)
    if "object_encoder" in state:
        object_encoder.load_state_dict(state["object_encoder"])
    elif "extra_modules" in state and "object_encoder" in state["extra_modules"]:
        object_encoder.load_state_dict(state["extra_modules"]["object_encoder"])
    model.eval()
    object_encoder.eval()

    clip_model = load_clip_text_encoder(
        device=device,
        model_name=str(cfg.model.text_encoder.clip_version),
        download_root=str(cfg.model.text_encoder.get("download_root", "cache/clip")),
    )

    pseudo_label_subdir = cfg.data.get("pseudo_label_subdir", None)
    entry = cfg.data.datasets[0]   # use first subset for speed
    sub_dir = (str(Path(entry.root) / pseudo_label_subdir)
               if pseudo_label_subdir is not None else None)
    ds = HOIDataset(
        root=entry.root,
        pseudo_label_dir=sub_dir,
        max_seq_length=cfg.data.max_seq_length,
        augment=None,
        support_collapse_hand_support=True,
        surface_obj_pose=True,
    )

    samples = [collate_hoi([ds[i]]) for i in range(args.num_clips)]

    for scale in args.cfg_scales:
        print(f"\n========= CFG scale = {scale} =========")
        agg_rmse = {name: [] for name, *_ in FEATURE_GROUPS}
        for batch in samples:
            motion_gt = batch["motion"].to(device)
            joints_w = batch["joints"].to(device)
            cs = batch["contact_state"].to(device)
            ctx = batch["contact_target_xyz"].to(device)
            ph = batch["phase"].to(device)
            sp = batch["support"].to(device)
            obj_com = batch["obj_com_canonical"].to(device)
            obj_rot6d = batch["obj_rot6d_canonical"].to(device)
            object_pc = batch["object_pc"].to(device)
            seq_len = int(batch["seq_len"][0])

            phase_soft = F.one_hot(ph.clamp_min(0).long(), z_dims.phase_classes).float()
            support_soft = F.one_hot(sp.clamp_min(0).long(), z_dims.support_classes).float()
            z_int = pack_z_int(cs, ctx, phase_soft, support_soft, z_dims)
            object_traj = torch.cat([obj_com, obj_rot6d], dim=-1)
            init_pose = joints_w[:, 0, :, :].reshape(1, -1)
            text_features, _ = encode_text_per_token(clip_model, batch["text"], device)
            obj_tokens = object_encoder(object_pc)

            cond = {
                "z_int": z_int,
                "object_world_traj": object_traj,
                "init_pose": init_pose,
                "text": text_features.float(),
                "object_tokens": obj_tokens,
            }
            with torch.no_grad():
                x0 = model.sample(cond=cond, seq_length=motion_gt.shape[1], cfg_scale=scale)
            T_v = seq_len
            for name, lo, hi in FEATURE_GROUPS:
                diff = (x0[0, :T_v, lo:hi] - motion_gt[0, :T_v, lo:hi]).pow(2).mean().sqrt().item()
                gt_std = motion_gt[0, :T_v, lo:hi].std().item()
                agg_rmse[name].append((diff, gt_std))

        # Aggregate
        print(f"{'feature_group':>20} | {'pred RMSE':>12} | {'GT std':>10} | RMSE/std")
        print("-" * 65)
        for name, *_ in FEATURE_GROUPS:
            stats = agg_rmse[name]
            r = np.mean([s[0] for s in stats])
            g = np.mean([s[1] for s in stats])
            ratio = r / max(g, 1e-9)
            flag = "  ★ HIGH" if ratio > 1.0 else ""
            print(f"{name:>20} | {r:>12.5f} | {g:>10.5f} | {ratio:>6.3f}{flag}")


if __name__ == "__main__":
    main()
