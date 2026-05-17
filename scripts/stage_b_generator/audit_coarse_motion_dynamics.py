"""Coarse motion dynamics audit GT vs v18 (Round 10, Task 4).

Generates v18 baseline samples (default DDPM) on the Round-9
subset-balanced 24-clip selection, extracts coarse-v0 / coarse-v1
features from BOTH GT motion and v18 generated motion, and compares
coarse dynamics per subset.

Key question: does v18 under-motion show up in root / facing /
pelvis / torso coarse features?

Outputs:
- analyses/2026-05-20_coarse_motion_dynamics_audit.{json,md}

Coarse dynamics features computed per clip:

- root displacement magnitude (max - min over clip, horizontal)
- root velocity magnitude mean (m/frame, both horizontal + vertical)
- root acceleration magnitude mean
- root jerk magnitude mean
- facing yaw range (rad, unwrapped max - min)
- yaw velocity magnitude mean
- pelvis rot6d frame-to-frame change norm mean
- (v1 only) spine3 rot6d frame-to-frame change norm mean
- (v1 only) head height range (m, max - min)
- (v1 only) shoulder center height range (m)
- (v1 only) head height velocity magnitude mean
- (v1 only) torso lean range (proxy via spine3 rot6d) and lean velocity

Aggregated by:
- subset
- seed
- clip

Reports:
- generated / GT ratios per coarse feature per subset
- per-clip dynamics table
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from condition_route_causal_sensitivity_diagnostic import _full_rollout
from diagnostic_common import extract_plan, stats_list
from dynamics_diagnostic import (
    _build_cond, _build_dataset, _build_model, _fk_from_motion_135,
)
from extract_coarse_motion_representation import (
    COARSE_V0_DIM, COARSE_V0_NAMES, COARSE_V1_DIM, COARSE_V1_EXTRA_NAMES,
    extract_coarse_v0_v1,
)
from piano.data.dataset import collate_hoi
from piano.utils.clip_utils import load_clip_text_encoder


def _coarse_dyn_metrics(features: dict[str, Any], T: int) -> dict[str, float]:
    """Compute scalar coarse-dynamics metrics from a clip's extracted features."""
    cv0 = features["coarse_v0"]  # (T, 15)
    cv1 = features["coarse_v1"]  # (T, 23)
    yaw_unwrapped = features["yaw_unwrapped"]
    root_world = features["root_world"]
    joints = features["joints_fk"]

    out: dict[str, float] = {}
    # Root displacement magnitude (range over clip, XZ separately + Y)
    out["root_xz_range_m"] = float(
        np.linalg.norm(root_world[:, [0, 2]].max(axis=0) - root_world[:, [0, 2]].min(axis=0))
    )
    out["root_y_range_m"] = float(root_world[:, 1].max() - root_world[:, 1].min())
    # Root velocity magnitude — combined XYZ, frame-to-frame (m/frame)
    if T >= 2:
        vel = np.diff(root_world, axis=0)
        vel_norm = np.linalg.norm(vel, axis=-1)
        out["root_vel_mean_m_per_frame"] = float(vel_norm.mean())
        out["root_vel_p95_m_per_frame"] = float(np.percentile(vel_norm, 95))
        if T >= 3:
            acc = np.diff(vel, axis=0)
            acc_norm = np.linalg.norm(acc, axis=-1)
            out["root_acc_mean_m_per_frame2"] = float(acc_norm.mean())
            out["root_acc_p95_m_per_frame2"] = float(np.percentile(acc_norm, 95))
        else:
            out["root_acc_mean_m_per_frame2"] = 0.0
            out["root_acc_p95_m_per_frame2"] = 0.0
        if T >= 4:
            jerk = np.diff(acc, axis=0)
            jerk_norm = np.linalg.norm(jerk, axis=-1)
            out["root_jerk_mean_m_per_frame3"] = float(jerk_norm.mean())
            out["root_jerk_p95_m_per_frame3"] = float(np.percentile(jerk_norm, 95))
        else:
            out["root_jerk_mean_m_per_frame3"] = 0.0
            out["root_jerk_p95_m_per_frame3"] = 0.0
    else:
        for k in ("root_vel_mean_m_per_frame", "root_vel_p95_m_per_frame",
                  "root_acc_mean_m_per_frame2", "root_acc_p95_m_per_frame2",
                  "root_jerk_mean_m_per_frame3", "root_jerk_p95_m_per_frame3"):
            out[k] = 0.0

    # Facing yaw range and velocity magnitude
    out["yaw_range_rad"] = float(yaw_unwrapped.max() - yaw_unwrapped.min())
    if T >= 2:
        yaw_vel = np.diff(yaw_unwrapped)
        out["yaw_vel_abs_mean_rad_per_frame"] = float(np.abs(yaw_vel).mean())
        out["yaw_vel_abs_p95_rad_per_frame"] = float(np.percentile(np.abs(yaw_vel), 95))
    else:
        out["yaw_vel_abs_mean_rad_per_frame"] = 0.0
        out["yaw_vel_abs_p95_rad_per_frame"] = 0.0

    # Pelvis rot6d frame-to-frame change norm (proxy for angular velocity)
    pelvis_rot6d = cv0[:, 9:15]
    if T >= 2:
        prv = np.diff(pelvis_rot6d, axis=0)
        prv_norm = np.linalg.norm(prv, axis=-1)
        out["pelvis_rot6d_vel_mean"] = float(prv_norm.mean())
        out["pelvis_rot6d_vel_p95"] = float(np.percentile(prv_norm, 95))
    else:
        out["pelvis_rot6d_vel_mean"] = 0.0
        out["pelvis_rot6d_vel_p95"] = 0.0

    # v1 extras
    spine3_rot6d = cv1[:, COARSE_V0_DIM:COARSE_V0_DIM + 6]
    head_h = cv1[:, COARSE_V0_DIM + 6]
    shoulder_h = cv1[:, COARSE_V0_DIM + 7]
    if T >= 2:
        srv = np.diff(spine3_rot6d, axis=0)
        srv_norm = np.linalg.norm(srv, axis=-1)
        out["spine3_rot6d_vel_mean"] = float(srv_norm.mean())
        out["spine3_rot6d_vel_p95"] = float(np.percentile(srv_norm, 95))
        out["head_height_vel_mean_m_per_frame"] = float(np.abs(np.diff(head_h)).mean())
        out["shoulder_height_vel_mean_m_per_frame"] = float(np.abs(np.diff(shoulder_h)).mean())
    else:
        for k in ("spine3_rot6d_vel_mean", "spine3_rot6d_vel_p95",
                  "head_height_vel_mean_m_per_frame", "shoulder_height_vel_mean_m_per_frame"):
            out[k] = 0.0
    out["head_height_range_m"] = float(head_h.max() - head_h.min())
    out["shoulder_height_range_m"] = float(shoulder_h.max() - shoulder_h.min())
    # Torso lean: angle between Y axis and pelvis-to-head vector (proxy).
    # head/shoulder positions in world frame from FK
    head_world = joints[:, 15, :]
    pelvis_world = joints[:, 0, :]
    p2h = head_world - pelvis_world
    p2h_norm = p2h / (np.linalg.norm(p2h, axis=-1, keepdims=True) + 1e-9)
    # cos(angle with +Y) = p2h_y
    lean_cos = np.clip(p2h_norm[:, 1], -1.0, 1.0)
    lean_rad = np.arccos(lean_cos)
    out["torso_lean_range_rad"] = float(lean_rad.max() - lean_rad.min())
    if T >= 2:
        out["torso_lean_vel_abs_mean_rad_per_frame"] = float(np.abs(np.diff(lean_rad)).mean())
    else:
        out["torso_lean_vel_abs_mean_rad_per_frame"] = 0.0
    return out


def _ratio(gen: dict[str, float], gt: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in gen.items():
        g = float(gt.get(k, 0.0))
        out[f"{k}_xGT"] = float(v / g) if abs(g) > 1e-6 else 0.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/training/anchordiff_v18_a1_FULL_DATA.yaml"),
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("runs/training/stageB_anchordiff_v18_a1_FULL_DATA/final.pt"),
    )
    parser.add_argument(
        "--selection-json", type=Path,
        default=Path("analyses/2026-05-19_subset_balanced_failure_selection.json"),
    )
    parser.add_argument(
        "--output-json", type=Path,
        default=Path("analyses/2026-05-20_coarse_motion_dynamics_audit.json"),
    )
    parser.add_argument(
        "--output-md", type=Path,
        default=Path("analyses/2026-05-20_coarse_motion_dynamics_audit.md"),
    )
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--bucket", choices=["train", "val"], default="train")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(args.config)

    sel_payload = json.loads(args.selection_json.read_text(encoding="utf-8"))
    entries = sel_payload.get("selected", [])
    indices = [int(e["dataset_global_index"]) for e in entries]
    full_ds = _build_dataset(cfg, args.bucket)
    sub_ds = Subset(full_ds, indices)
    loader = DataLoader(sub_ds, batch_size=1, shuffle=False, collate_fn=collate_hoi, num_workers=0)
    clip_batches = list(loader)
    if not clip_batches:
        raise SystemExit("No clips loaded.")

    # Build model + load v18
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
    print(f"Auditing {len(clip_batches)} clips × {len(seeds)} seeds…", flush=True)

    rows: list[dict[str, Any]] = []
    for clip_idx, batch in enumerate(clip_batches):
        motion_gt = batch["motion"][0].numpy().astype(np.float32)
        rest_offsets = batch["rest_offsets"][0].numpy().astype(np.float32)
        seq_len = int(batch["seq_len"][0].item())
        sname = str(batch["subset"][0])
        sid = str(batch["seq_id"][0])
        text = str(batch["text"][0])

        gt_feat = extract_coarse_v0_v1(motion_gt, rest_offsets, seq_len)
        gt_dyn = _coarse_dyn_metrics(gt_feat, seq_len)
        # build cond
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
            gen_dyn = _coarse_dyn_metrics(gen_feat, seq_len)
            ratio = _ratio(gen_dyn, gt_dyn)
            rows.append({
                "clip_idx": clip_idx,
                "subset": sname,
                "seq_id": sid,
                "text": text[:120],
                "seq_len": int(seq_len),
                "seed": int(seed),
                "gt": gt_dyn,
                "gen": gen_dyn,
                "ratio_xGT": ratio,
            })

    # Aggregate per subset
    by_subset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_subset[r["subset"]].append(r)

    METRIC_KEYS = [
        "root_xz_range_m",
        "root_y_range_m",
        "root_vel_mean_m_per_frame",
        "root_vel_p95_m_per_frame",
        "root_acc_mean_m_per_frame2",
        "root_acc_p95_m_per_frame2",
        "root_jerk_mean_m_per_frame3",
        "root_jerk_p95_m_per_frame3",
        "yaw_range_rad",
        "yaw_vel_abs_mean_rad_per_frame",
        "yaw_vel_abs_p95_rad_per_frame",
        "pelvis_rot6d_vel_mean",
        "pelvis_rot6d_vel_p95",
        "spine3_rot6d_vel_mean",
        "spine3_rot6d_vel_p95",
        "head_height_vel_mean_m_per_frame",
        "shoulder_height_vel_mean_m_per_frame",
        "head_height_range_m",
        "shoulder_height_range_m",
        "torso_lean_range_rad",
        "torso_lean_vel_abs_mean_rad_per_frame",
    ]

    subset_summary: dict[str, dict[str, Any]] = {}
    for sname, sub_rows in by_subset.items():
        summary: dict[str, Any] = {"n_rows": len(sub_rows)}
        for k in METRIC_KEYS:
            gt_vals = [r["gt"][k] for r in sub_rows]
            gen_vals = [r["gen"][k] for r in sub_rows]
            ratio_vals = [r["ratio_xGT"][f"{k}_xGT"] for r in sub_rows]
            summary[f"gt.{k}"] = stats_list(gt_vals)
            summary[f"gen.{k}"] = stats_list(gen_vals)
            summary[f"xGT.{k}"] = stats_list(ratio_vals)
        subset_summary[sname] = summary

    payload = {
        "config": str(args.config),
        "selection_json": str(args.selection_json),
        "seeds": seeds,
        "n_clips": len(clip_batches),
        "subset_summary": subset_summary,
        "rows": rows,
        "metric_keys": METRIC_KEYS,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")

    # MD: per-subset core table
    lines = [
        "# Coarse Motion Dynamics Audit GT vs v18 (Round 10, Task 4)",
        "",
        f"- Config: `{args.config}`",
        f"- Checkpoint: `{args.checkpoint}`",
        f"- Selection: `{args.selection_json}`",
        f"- Seeds: {seeds}",
        f"- Clips: {len(clip_batches)}",
        "",
        "## Per-subset coarse dynamics — generated / GT ratio (xGT)",
        "",
        "Closer to 1.0 = matches GT magnitude. << 1.0 = under-motion. >> 1.0 = over-motion.",
        "",
        "| subset | root vel xGT | root acc p95 xGT | root jerk p95 xGT | yaw range xGT | yaw vel xGT | pelvis rot vel xGT |",
        "|--------|---|---|---|---|---|---|",
    ]
    for sname, s in subset_summary.items():
        lines.append(
            f"| {sname} | "
            f"{s['xGT.root_vel_mean_m_per_frame']['mean']:.3f} | "
            f"{s['xGT.root_acc_p95_m_per_frame2']['mean']:.3f} | "
            f"{s['xGT.root_jerk_p95_m_per_frame3']['mean']:.3f} | "
            f"{s['xGT.yaw_range_rad']['mean']:.3f} | "
            f"{s['xGT.yaw_vel_abs_mean_rad_per_frame']['mean']:.3f} | "
            f"{s['xGT.pelvis_rot6d_vel_mean']['mean']:.3f} |"
        )
    lines += [
        "",
        "## Per-subset coarse-v1 extras (xGT)",
        "",
        "| subset | spine3 rot vel xGT | head height range xGT | head height vel xGT | shoulder height range xGT | torso lean range xGT | torso lean vel xGT |",
        "|--------|---|---|---|---|---|---|",
    ]
    for sname, s in subset_summary.items():
        lines.append(
            f"| {sname} | "
            f"{s['xGT.spine3_rot6d_vel_mean']['mean']:.3f} | "
            f"{s['xGT.head_height_range_m']['mean']:.3f} | "
            f"{s['xGT.head_height_vel_mean_m_per_frame']['mean']:.3f} | "
            f"{s['xGT.shoulder_height_range_m']['mean']:.3f} | "
            f"{s['xGT.torso_lean_range_rad']['mean']:.3f} | "
            f"{s['xGT.torso_lean_vel_abs_mean_rad_per_frame']['mean']:.3f} |"
        )
    lines += [
        "",
        "## Per-subset GT absolute values (sanity check — GT should be > 0)",
        "",
        "| subset | GT root vel | GT root acc p95 | GT yaw range | GT pelvis rot vel | GT head h range | GT torso lean range |",
        "|--------|---|---|---|---|---|---|",
    ]
    for sname, s in subset_summary.items():
        lines.append(
            f"| {sname} | "
            f"{s['gt.root_vel_mean_m_per_frame']['mean']:.4f} | "
            f"{s['gt.root_acc_p95_m_per_frame2']['mean']:.4f} | "
            f"{s['gt.yaw_range_rad']['mean']:.3f} | "
            f"{s['gt.pelvis_rot6d_vel_mean']['mean']:.4f} | "
            f"{s['gt.head_height_range_m']['mean']:.3f} | "
            f"{s['gt.torso_lean_range_rad']['mean']:.3f} |"
        )

    # Per-row table — first event-relevant subset rows
    lines += [
        "",
        "## Per-row detail (clip × seed)",
        "",
        "| subset | seq_id | seed | root vel xGT | root acc xGT | yaw range xGT | pelvis rot vel xGT | torso lean xGT |",
        "|--------|--------|------|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['subset']} | {r['seq_id']} | {r['seed']} | "
            f"{r['ratio_xGT']['root_vel_mean_m_per_frame_xGT']:.3f} | "
            f"{r['ratio_xGT']['root_acc_p95_m_per_frame2_xGT']:.3f} | "
            f"{r['ratio_xGT']['yaw_range_rad_xGT']:.3f} | "
            f"{r['ratio_xGT']['pelvis_rot6d_vel_mean_xGT']:.3f} | "
            f"{r['ratio_xGT']['torso_lean_range_rad_xGT']:.3f} |"
        )
    lines.append("")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
