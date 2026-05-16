"""Coarse motion representation extraction (Round 10, Task 3).

Defines and extracts two candidate coarse representations for the
Stage-B vNext Stage-1 prior:

- **Coarse-v0** (root / facing / pelvis only, ~15 dims/frame)
- **Coarse-v1** (v0 + torso/spine/head proxies, ~23 dims/frame)

This script is extraction + audit only. No model training. No model
loaded. Operates on GT motion_135 from the dataset (and optionally
v18-generated motion for comparison in the audit script).

Coarse-v0 per frame (15 dims):

    root_local_trans_xz  : 2  (X, Z relative to frame 0)
    root_local_trans_y   : 1  (height relative to frame 0)
    root_vel_xz          : 2  (horizontal frame-to-frame delta)
    root_vel_y           : 1  (vertical frame-to-frame delta)
    facing_yaw_sincos    : 2  (sin(yaw), cos(yaw))
    facing_yaw_velocity  : 1  (unwrapped yaw delta)
    pelvis_rot6d         : 6  (joint 0 global rot6d from motion_135)

Coarse-v1 = v0 + (Option A):

    spine3_rot6d_global  : 6  (joint 9 global rot6d from motion_135)
    head_height          : 1  (joint 15 world Y position via FK)
    shoulder_center_h    : 1  (mean Y of joints 16, 17 via FK)

For Option B (vector form), see contract doc. We default to Option A
because the rot6d slot already lives in motion_135 and is the
natural Stage-1 prediction target.

Excluded by spec (Stage 1 must be object/plan-free):
- hand/foot joint positions
- contact_state / contact_target_xyz
- object trajectory / plan tokens / object tokens

Outputs:
- analyses/2026-05-20_coarse_representation_extraction_audit.{json,md}

Per-clip extraction shape:
- coarse_v0: (T, 15)
- coarse_v1: (T, 23)
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from dynamics_diagnostic import _build_dataset, _fk_from_motion_135
from piano.data.dataset import collate_hoi
from piano.training.smpl_kinematics import (
    rotation_6d_to_matrix as _project_rotation_6d_to_matrix,
)


# Joint indices (verified against
# analyses/2026-05-20_coarse_representation_contract.md)
J_PELVIS = 0
J_SPINE3 = 9
J_HEAD = 15
J_L_SHOULDER = 16
J_R_SHOULDER = 17

# Coarse-v0 dim layout
COARSE_V0_DIM = 15  # 2 + 1 + 2 + 1 + 2 + 1 + 6
COARSE_V0_NAMES = [
    "root_local_trans_x", "root_local_trans_z", "root_local_trans_y",
    "root_vel_x", "root_vel_z", "root_vel_y",
    "facing_yaw_sin", "facing_yaw_cos", "facing_yaw_velocity",
    "pelvis_rot6d_0", "pelvis_rot6d_1", "pelvis_rot6d_2",
    "pelvis_rot6d_3", "pelvis_rot6d_4", "pelvis_rot6d_5",
]
# Coarse-v1 adds 8 dims (spine3 rot6d + head height + shoulder center height)
COARSE_V1_EXTRA_DIM = 8
COARSE_V1_DIM = COARSE_V0_DIM + COARSE_V1_EXTRA_DIM
COARSE_V1_EXTRA_NAMES = [
    "spine3_rot6d_0", "spine3_rot6d_1", "spine3_rot6d_2",
    "spine3_rot6d_3", "spine3_rot6d_4", "spine3_rot6d_5",
    "head_height_y", "shoulder_center_height_y",
]


def _facing_yaw_from_pelvis_rot6d(pelvis_rot6d: np.ndarray) -> np.ndarray:
    """Compute facing yaw (radians) per frame from pelvis global rot6d.

    Round-12 (post-Codex review): switched to the project-local upstream
    ``piano.training.smpl_kinematics.rotation_6d_to_matrix`` after the
    custom in-script helper was found to return ``R^T`` (row-vs-column
    stacking mismatch — see
    ``analyses/2026-05-22_stage1_coarse_prior_preflight_smoke_report.md``).

    Convention: the dataset stores ``global_rot_6d = matrix_to_rotation_6d(R)``
    via the project utility, so its inverse is ``rotation_6d_to_matrix``.
    For SMPL the body local forward is +Z_local; the body forward in world
    coords is ``R @ [0, 0, 1] = R[..., :, 2]`` (column 2). yaw_world is
    ``atan2(forward_x, forward_z)``.

    The previous (Round-10) custom helper extracted the same column index
    but on ``R^T``, which equals ``R[2, :]``; for pure yaw rotations that
    flips the X component, so the cached yaw was ``-theta`` instead of
    ``+theta``. Magnitude-based audit metrics (range, |velocity|) were
    unaffected, but signed yaw and yaw velocity were sign-flipped.
    """
    rot6d_t = torch.from_numpy(pelvis_rot6d).float()
    R = _project_rotation_6d_to_matrix(rot6d_t).numpy()  # (T, 3, 3)
    forward = R[..., :, 2]                               # body Z in world
    fx = forward[..., 0]
    fz = forward[..., 2]
    yaw = np.arctan2(fx, fz)
    return yaw.astype(np.float32)


def extract_coarse_v0_v1(
    motion: np.ndarray,         # (T, 135)
    rest_offsets: np.ndarray,   # (22, 3)
    seq_len: int,
    *, fps: float = 20.0,
) -> dict[str, Any]:
    """Compute coarse-v0 and coarse-v1 features for one clip.

    Returns a dict with keys: coarse_v0 (T, 15), coarse_v1 (T, 23),
    coarse_metadata (yaw raw + unwrapped, FK joints), shape info.
    """
    T_pad = motion.shape[0]
    T = min(int(seq_len), T_pad)
    rot6d = motion[:T, :132].reshape(T, 22, 6).astype(np.float32)
    root_world = motion[:T, 132:135].astype(np.float32)

    # Root local (relative to frame 0)
    root0 = root_world[0]
    root_local = root_world - root0[None, :]
    # Horizontal: X (idx 0) and Z (idx 2), vertical: Y (idx 1)
    root_local_x = root_local[:, 0]
    root_local_y = root_local[:, 1]
    root_local_z = root_local[:, 2]

    # Velocity (per-frame delta in world frame)
    if T >= 2:
        vel_world = np.diff(root_world, axis=0, prepend=root_world[:1])
    else:
        vel_world = np.zeros_like(root_world)
    vel_x = vel_world[:, 0]
    vel_y = vel_world[:, 1]
    vel_z = vel_world[:, 2]

    # Facing yaw from pelvis rot6d (joint 0)
    pelvis_rot6d = rot6d[:, J_PELVIS]              # (T, 6)
    yaw_raw = _facing_yaw_from_pelvis_rot6d(pelvis_rot6d)
    yaw_unwrapped = np.unwrap(yaw_raw)
    yaw_sin = np.sin(yaw_unwrapped).astype(np.float32)
    yaw_cos = np.cos(yaw_unwrapped).astype(np.float32)
    if T >= 2:
        yaw_vel = np.diff(yaw_unwrapped, prepend=yaw_unwrapped[:1]).astype(np.float32)
    else:
        yaw_vel = np.zeros(T, dtype=np.float32)

    # Pelvis rot6d already extracted

    # Stack coarse-v0
    coarse_v0 = np.stack([
        root_local_x, root_local_z, root_local_y,
        vel_x, vel_z, vel_y,
        yaw_sin, yaw_cos, yaw_vel,
    ], axis=-1).astype(np.float32)  # (T, 9)
    coarse_v0 = np.concatenate([coarse_v0, pelvis_rot6d], axis=-1)  # (T, 15)

    # FK for v1 head + shoulder center heights
    motion_t = torch.from_numpy(motion[:T, None, :]).float()           # (T, 1, 135)
    # _fk_from_motion_135 expects (B, T, 135); we use B=1, T=clip length
    motion_t = motion_t.transpose(0, 1)                                 # (1, T, 135)
    rest_offsets_t = torch.from_numpy(rest_offsets).float().unsqueeze(0)  # (1, 22, 3)
    joints = _fk_from_motion_135(motion_t, rest_offsets_t)              # (1, T, 22, 3)
    joints = joints.squeeze(0).numpy()                                  # (T, 22, 3)
    spine3_rot6d = rot6d[:, J_SPINE3]
    head_height = joints[:, J_HEAD, 1].astype(np.float32)
    shoulder_center_h = ((joints[:, J_L_SHOULDER, 1] + joints[:, J_R_SHOULDER, 1]) * 0.5).astype(np.float32)
    coarse_v1_extra = np.concatenate([
        spine3_rot6d,
        head_height[:, None],
        shoulder_center_h[:, None],
    ], axis=-1).astype(np.float32)
    coarse_v1 = np.concatenate([coarse_v0, coarse_v1_extra], axis=-1)  # (T, 23)

    assert coarse_v0.shape == (T, COARSE_V0_DIM), f"coarse_v0 shape {coarse_v0.shape} != ({T}, {COARSE_V0_DIM})"
    assert coarse_v1.shape == (T, COARSE_V1_DIM), f"coarse_v1 shape {coarse_v1.shape} != ({T}, {COARSE_V1_DIM})"

    return {
        "coarse_v0": coarse_v0,
        "coarse_v1": coarse_v1,
        "joints_fk": joints,
        "yaw_raw": yaw_raw,
        "yaw_unwrapped": yaw_unwrapped.astype(np.float32),
        "root_world": root_world,
    }


def _feature_stats(arr: np.ndarray) -> dict[str, float]:
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": int(arr.size),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/training/anchordiff_v18_a1_FULL_DATA.yaml"),
    )
    parser.add_argument(
        "--selection-json", type=Path,
        default=Path("analyses/2026-05-19_subset_balanced_failure_selection.json"),
    )
    parser.add_argument(
        "--output-json", type=Path,
        default=Path("analyses/2026-05-20_coarse_representation_extraction_audit.json"),
    )
    parser.add_argument(
        "--output-md", type=Path,
        default=Path("analyses/2026-05-20_coarse_representation_extraction_audit.md"),
    )
    parser.add_argument("--max-clips", type=int, default=24)
    parser.add_argument("--bucket", choices=["train", "val"], default="train")
    parser.add_argument("--subset-filter", type=str, default=None,
                        help="Comma-separated subset names to keep, e.g. chairs,imhd.")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    full_ds = _build_dataset(cfg, args.bucket)

    # Load subset-balanced selection by dataset_global_index
    sel_payload = json.loads(args.selection_json.read_text(encoding="utf-8"))
    entries = sel_payload.get("selected", [])
    if not entries:
        raise SystemExit(f"Empty selection in {args.selection_json}")
    if args.subset_filter:
        keep = {s.strip() for s in args.subset_filter.split(",") if s.strip()}
        entries = [e for e in entries if e.get("subset") in keep]
    entries = entries[: int(args.max_clips)]
    indices = [int(e["dataset_global_index"]) for e in entries]
    sub_ds = Subset(full_ds, indices)
    loader = DataLoader(sub_ds, batch_size=1, shuffle=False, collate_fn=collate_hoi, num_workers=0)

    per_clip: list[dict[str, Any]] = []
    for clip_idx, batch in enumerate(loader):
        motion = batch["motion"][0].numpy().astype(np.float32)
        rest_offsets = batch["rest_offsets"][0].numpy().astype(np.float32) if "rest_offsets" in batch else None
        if rest_offsets is None:
            raise SystemExit("rest_offsets not in batch; needed for FK on coarse-v1 head/shoulder.")
        seq_len = int(batch["seq_len"][0].item())
        sname = str(batch["subset"][0])
        sid = str(batch["seq_id"][0])
        text = str(batch["text"][0])
        out = extract_coarse_v0_v1(motion, rest_offsets, seq_len)
        cv0 = out["coarse_v0"]
        cv1 = out["coarse_v1"]
        # Per-dim stats — full clip
        stats_v0 = {n: _feature_stats(cv0[:, i]) for i, n in enumerate(COARSE_V0_NAMES)}
        stats_v1_extra = {n: _feature_stats(cv1[:, COARSE_V0_DIM + i]) for i, n in enumerate(COARSE_V1_EXTRA_NAMES)}
        # Sanity
        finite_v0 = bool(np.isfinite(cv0).all())
        finite_v1 = bool(np.isfinite(cv1).all())
        per_clip.append({
            "clip_idx": clip_idx,
            "subset": sname,
            "seq_id": sid,
            "text": text[:120],
            "seq_len": int(seq_len),
            "coarse_v0_shape": list(cv0.shape),
            "coarse_v1_shape": list(cv1.shape),
            "finite_v0": finite_v0,
            "finite_v1": finite_v1,
            "stats_v0": stats_v0,
            "stats_v1_extra": stats_v1_extra,
            "yaw_unwrapped_min_max": [
                float(out["yaw_unwrapped"].min()), float(out["yaw_unwrapped"].max())
            ],
            "root_local_max_abs": [
                float(np.abs(cv0[:, 0]).max()),
                float(np.abs(cv0[:, 1]).max()),
                float(np.abs(cv0[:, 2]).max()),
            ],
            "pelvis_rot6d_finite": bool(np.isfinite(cv0[:, 9:15]).all()),
        })

    # Aggregate per subset (means of stat means)
    subset_summary: dict[str, dict[str, Any]] = {}
    by_subset: dict[str, list[dict[str, Any]]] = {}
    for c in per_clip:
        by_subset.setdefault(c["subset"], []).append(c)
    for sname, clips in by_subset.items():
        if not clips:
            continue
        subset_summary[sname] = {
            "n_clips": len(clips),
            "mean_seq_len": float(np.mean([c["seq_len"] for c in clips])),
            "all_finite_v0": all(c["finite_v0"] for c in clips),
            "all_finite_v1": all(c["finite_v1"] for c in clips),
            "yaw_range_mean": float(np.mean([
                c["yaw_unwrapped_min_max"][1] - c["yaw_unwrapped_min_max"][0]
                for c in clips
            ])),
            "root_local_max_abs_xz_mean": float(np.mean([
                max(c["root_local_max_abs"][0], c["root_local_max_abs"][1])
                for c in clips
            ])),
            "root_local_max_abs_y_mean": float(np.mean([
                c["root_local_max_abs"][2] for c in clips
            ])),
        }

    payload = {
        "config": str(args.config),
        "selection_json": str(args.selection_json),
        "n_clips": len(per_clip),
        "coarse_v0_dim": COARSE_V0_DIM,
        "coarse_v0_names": COARSE_V0_NAMES,
        "coarse_v1_dim": COARSE_V1_DIM,
        "coarse_v1_extra_names": COARSE_V1_EXTRA_NAMES,
        "subset_summary": subset_summary,
        "per_clip": per_clip,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")

    lines = [
        "# Coarse Representation Extraction Audit (Round 10, Task 3)",
        "",
        f"- Config: `{args.config}`",
        f"- Selection: `{args.selection_json}`",
        f"- Clips processed: {len(per_clip)}",
        f"- Coarse-v0 dim: **{COARSE_V0_DIM}**",
        f"- Coarse-v1 dim: **{COARSE_V1_DIM}** (v0 + {COARSE_V1_EXTRA_DIM} extras)",
        "",
        "## Coarse-v0 channels",
        "",
    ]
    for i, n in enumerate(COARSE_V0_NAMES):
        lines.append(f"- `{i}`: {n}")
    lines += [
        "",
        "## Coarse-v1 extra channels",
        "",
    ]
    for i, n in enumerate(COARSE_V1_EXTRA_NAMES):
        lines.append(f"- `{COARSE_V0_DIM + i}`: {n}")
    lines += [
        "",
        "## Subset summary",
        "",
        "| subset | n_clips | mean T | yaw range (rad) | root XZ max (m) | root Y max (m) | all finite v0 | all finite v1 |",
        "|--------|---------|--------|------------------|------------------|----------------|---------------|---------------|",
    ]
    for sname, s in subset_summary.items():
        lines.append(
            f"| {sname} | {s['n_clips']} | {s['mean_seq_len']:.0f} | "
            f"{s['yaw_range_mean']:.2f} | "
            f"{s['root_local_max_abs_xz_mean']:.2f} | "
            f"{s['root_local_max_abs_y_mean']:.2f} | "
            f"{s['all_finite_v0']} | {s['all_finite_v1']} |"
        )

    lines += [
        "",
        "## Per-clip extraction stats",
        "",
        "| subset | seq_id | T | yaw range | root XZ max | root Y max | finite |",
        "|--------|--------|---|-----------|--------------|-------------|--------|",
    ]
    for c in per_clip:
        yaw_r = c["yaw_unwrapped_min_max"][1] - c["yaw_unwrapped_min_max"][0]
        lines.append(
            f"| {c['subset']} | {c['seq_id']} | {c['seq_len']} | "
            f"{yaw_r:.2f} | "
            f"{max(c['root_local_max_abs'][0], c['root_local_max_abs'][1]):.2f} | "
            f"{c['root_local_max_abs'][2]:.2f} | "
            f"{'OK' if c['finite_v0'] and c['finite_v1'] else 'NaN/inf!'} |"
        )
    lines.append("")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
