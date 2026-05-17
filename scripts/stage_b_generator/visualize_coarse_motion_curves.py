"""Coarse motion curves visualization (Round 10, Task 5).

Renders per-clip plots of GT vs v18 coarse features over time. Generates
v18 samples at the requested seeds and plots:

- root XZ trajectory (top-down)
- root Y over time (height)
- facing yaw over time
- pelvis rot6d frame-to-frame change norm
- (v1) head height over time
- (v1) shoulder height over time
- (v1) torso lean (rad) over time

Outputs:
- analyses/visuals/2026-05-20_coarse_motion_curves/<subset>_<seq_id>_seed<S>.png
- analyses/2026-05-20_coarse_motion_visual_summary.md

This script is OPTIONAL and is only run on request. It re-uses the
extraction + rollout machinery from
`audit_coarse_motion_dynamics.py` but renders per-clip figures.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from condition_route_causal_sensitivity_diagnostic import _full_rollout
from diagnostic_common import extract_plan
from dynamics_diagnostic import _build_cond, _build_dataset, _build_model, _fk_from_motion_135
from extract_coarse_motion_representation import (
    COARSE_V0_DIM, extract_coarse_v0_v1,
)
from piano.data.dataset import collate_hoi
from piano.utils.clip_utils import load_clip_text_encoder


def _plot_clip(
    gt_feat: dict[str, Any], gen_feat: dict[str, Any], seq_len: int,
    title: str, out_path: Path,
) -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(12, 9))
    # (0,0): root XZ trajectory
    ax = axes[0, 0]
    ax.plot(gt_feat["root_world"][:seq_len, 0], gt_feat["root_world"][:seq_len, 2], label="GT", color="C0")
    ax.plot(gen_feat["root_world"][:seq_len, 0], gen_feat["root_world"][:seq_len, 2], label="v18", color="C1", linestyle="--")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)"); ax.set_title("Root XZ trajectory")
    ax.legend(fontsize=7); ax.set_aspect("equal", "datalim")
    # (0,1): root Y over time
    ax = axes[0, 1]
    ax.plot(gt_feat["root_world"][:seq_len, 1], label="GT", color="C0")
    ax.plot(gen_feat["root_world"][:seq_len, 1], label="v18", color="C1", linestyle="--")
    ax.set_xlabel("frame"); ax.set_ylabel("Y (m)"); ax.set_title("Root height")
    ax.legend(fontsize=7)
    # (1,0): facing yaw
    ax = axes[1, 0]
    ax.plot(gt_feat["yaw_unwrapped"][:seq_len], label="GT", color="C0")
    ax.plot(gen_feat["yaw_unwrapped"][:seq_len], label="v18", color="C1", linestyle="--")
    ax.set_xlabel("frame"); ax.set_ylabel("yaw (rad, unwrapped)"); ax.set_title("Facing yaw")
    ax.legend(fontsize=7)
    # (1,1): pelvis rot6d frame-to-frame change
    pr_gt = gt_feat["coarse_v0"][:seq_len, 9:15]
    pr_gn = gen_feat["coarse_v0"][:seq_len, 9:15]
    ax = axes[1, 1]
    if seq_len >= 2:
        ax.plot(np.linalg.norm(np.diff(pr_gt, axis=0), axis=-1), label="GT", color="C0")
        ax.plot(np.linalg.norm(np.diff(pr_gn, axis=0), axis=-1), label="v18", color="C1", linestyle="--")
    ax.set_xlabel("frame"); ax.set_ylabel("‖Δpelvis rot6d‖"); ax.set_title("Pelvis rotation velocity")
    ax.legend(fontsize=7)
    # (2,0): head and shoulder heights (v1)
    head_gt = gt_feat["coarse_v1"][:seq_len, COARSE_V0_DIM + 6]
    head_gn = gen_feat["coarse_v1"][:seq_len, COARSE_V0_DIM + 6]
    shoulder_gt = gt_feat["coarse_v1"][:seq_len, COARSE_V0_DIM + 7]
    shoulder_gn = gen_feat["coarse_v1"][:seq_len, COARSE_V0_DIM + 7]
    ax = axes[2, 0]
    ax.plot(head_gt, label="GT head", color="C0")
    ax.plot(head_gn, label="v18 head", color="C1", linestyle="--")
    ax.plot(shoulder_gt, label="GT shoulder", color="C2")
    ax.plot(shoulder_gn, label="v18 shoulder", color="C3", linestyle="--")
    ax.set_xlabel("frame"); ax.set_ylabel("height Y (m)"); ax.set_title("Head / shoulder height")
    ax.legend(fontsize=7)
    # (2,1): torso lean
    j_gt = gt_feat["joints_fk"]
    j_gn = gen_feat["joints_fk"]
    p2h_gt = j_gt[:, 15] - j_gt[:, 0]
    p2h_gn = j_gn[:, 15] - j_gn[:, 0]
    p2h_gt_norm = p2h_gt / (np.linalg.norm(p2h_gt, axis=-1, keepdims=True) + 1e-9)
    p2h_gn_norm = p2h_gn / (np.linalg.norm(p2h_gn, axis=-1, keepdims=True) + 1e-9)
    lean_gt = np.arccos(np.clip(p2h_gt_norm[:, 1], -1, 1))
    lean_gn = np.arccos(np.clip(p2h_gn_norm[:, 1], -1, 1))
    ax = axes[2, 1]
    ax.plot(lean_gt[:seq_len], label="GT", color="C0")
    ax.plot(lean_gn[:seq_len], label="v18", color="C1", linestyle="--")
    ax.set_xlabel("frame"); ax.set_ylabel("lean (rad)"); ax.set_title("Torso lean (pelvis→head vs +Y)")
    ax.legend(fontsize=7)

    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/training/anchordiff_v18_a1_FULL_DATA.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/training/stageB_anchordiff_v18_a1_FULL_DATA/final.pt"))
    parser.add_argument("--selection-json", type=Path, default=Path("analyses/2026-05-19_subset_balanced_failure_selection.json"))
    parser.add_argument("--visuals-dir", type=Path, default=Path("analyses/visuals/2026-05-20_coarse_motion_curves"))
    parser.add_argument("--md", type=Path, default=Path("analyses/2026-05-20_coarse_motion_visual_summary.md"))
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--bucket", choices=["train", "val"], default="train")
    parser.add_argument("--max-clips", type=int, default=24)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(args.config)
    sel_payload = json.loads(args.selection_json.read_text(encoding="utf-8"))
    entries = sel_payload.get("selected", [])[: int(args.max_clips)]
    indices = [int(e["dataset_global_index"]) for e in entries]
    full_ds = _build_dataset(cfg, args.bucket)
    sub_ds = Subset(full_ds, indices)
    loader = DataLoader(sub_ds, batch_size=1, shuffle=False, collate_fn=collate_hoi, num_workers=0)
    clip_batches = list(loader)

    model, object_encoder, z_dims = _build_model(cfg, device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("model", state))
    if "object_encoder" in state:
        object_encoder.load_state_dict(state["object_encoder"])
    elif "extra_modules" in state and "object_encoder" in state["extra_modules"]:
        object_encoder.load_state_dict(state["extra_modules"]["object_encoder"])
    model.eval(); object_encoder.eval()
    clip_model = load_clip_text_encoder(
        device=device, model_name=str(cfg.model.text_encoder.clip_version),
        download_root=str(cfg.model.text_encoder.download_root),
    )
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    lines = ["# Coarse Motion Curves — Visual Summary (Round 10, Task 5)", ""]
    lines.append(f"- Selection: `{args.selection_json}` ({len(clip_batches)} clips)")
    lines.append(f"- Seeds: {seeds}")
    lines.append("")
    lines.append("| subset | seq_id | seed | plot |")
    lines.append("|--------|--------|------|------|")
    for clip_idx, batch in enumerate(clip_batches):
        motion_gt = batch["motion"][0].numpy().astype(np.float32)
        rest_offsets = batch["rest_offsets"][0].numpy().astype(np.float32)
        seq_len = int(batch["seq_len"][0].item())
        sname = str(batch["subset"][0])
        sid = str(batch["seq_id"][0])
        text = str(batch["text"][0])
        gt_feat = extract_coarse_v0_v1(motion_gt, rest_offsets, seq_len)
        cond, T = _build_cond(batch, model, object_encoder, clip_model, z_dims, cfg, device)
        cond = {**cond, "interaction_plan": extract_plan(batch, device)}
        for seed in seeds:
            print(f"  clip={clip_idx} {sname}/{sid} seed={seed}", flush=True)
            motion_pred = _full_rollout(
                model, cond, seq_length=T,
                seed=int(seed) + clip_idx * 10000,
                cfg_scale=float(args.cfg_scale), alpha_hint=1.0, sampler="ddpm",
            )
            gen_motion_np = motion_pred[0].detach().cpu().numpy().astype(np.float32)
            gen_feat = extract_coarse_v0_v1(gen_motion_np, rest_offsets, seq_len)
            out_path = args.visuals_dir / f"{sname}_{sid}_seed{seed}.png"
            _plot_clip(
                gt_feat, gen_feat, seq_len,
                title=f"{sname}/{sid} seed={seed}\n{text[:100]}",
                out_path=out_path,
            )
            lines.append(f"| {sname} | {sid} | {seed} | `{out_path.as_posix()}` |")
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.md}")


if __name__ == "__main__":
    main()
