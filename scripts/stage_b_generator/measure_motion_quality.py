"""Motion plausibility / smoothness metric (N7 from 2026-05-02 metric review).

Per-step inference-time optimisation can produce motions with abnormally
high acceleration / jerk (third derivative of position). This metric
captures that — both as raw per-clip magnitudes, and as a distributional
comparison to GT_orig (KS distance) when run across multiple conditions.

For each generated motion clip:
  joints_world : (T, 22, 3) recovered via recover_from_ric + canonical→world
  jerk = third time-difference / dt^3        # (T-3, 22, 3)
  jerk_mag = ||jerk||                          # (T-3, 22)

Per-clip aggregations:
  - mean_jerk_m_per_s3        : mean over (t, j)
  - max_jerk_m_per_s3         : max over (t, j)
  - mean_jerk_hands_only      : mean over (t, j ∈ {20, 21}) — interaction-relevant
  - mean_jerk_pelvis          : mean for joint 0
  - per_joint_mean_jerk       : (22,) — for debugging which joint contributes

The script collects all per-(clip, joint, frame) jerk magnitudes per
condition in an output `jerk_samples_<label>.npz` so the unified
summarize script can compute KS distance between conditions later.

Usage::

    python scripts/stage_b_generator/measure_motion_quality.py \\
        --input-dir runs/eval/<...>_qual/full_guided \\
        --input-dir runs/eval/<...>_gt_roundtrip_80/gt_original \\
        --output-dir runs/eval/<...>_motion_quality \\
        --fps 20
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import piano.models.backbones.momask_adapter  # noqa: F401
from utils.motion_process import recover_from_ric

from piano.utils.canonical_frame import y_rotation_matrix
from piano.utils.io_utils import ensure_dir
from piano.utils.smpl_utils import BODY_PART_INDICES


def _lift_canonical_to_world(joints_canon, R_y_angle, T_xz):
    R = y_rotation_matrix(float(R_y_angle))
    rotated = joints_canon @ R.T
    rotated[..., 0] += float(T_xz[0])
    rotated[..., 2] += float(T_xz[1])
    return rotated.astype(np.float32)


def _jerk_magnitudes(joints: np.ndarray, fps: float) -> np.ndarray:
    """jerk[t, j] = ||(d/dt)^3 joints[t, j]|| in m/s^3. Returns (T-3, 22)."""
    if joints.shape[0] < 4:
        return np.zeros((0, joints.shape[1]), dtype=np.float32)
    dt = 1.0 / float(fps)
    # Third forward difference
    d1 = (joints[1:] - joints[:-1]) / dt           # velocity (T-1, 22, 3)
    d2 = (d1[1:] - d1[:-1]) / dt                   # acceleration
    d3 = (d2[1:] - d2[:-1]) / dt                   # jerk (T-3, 22, 3)
    return np.linalg.norm(d3, axis=-1).astype(np.float32)


def _measure_condition(input_dir: Path, *, fps: float, save_samples_to: Path | None) -> dict:
    npz_path = input_dir / "generated.npz"
    summ_path = input_dir / "summary.json"
    npz = np.load(npz_path)
    with summ_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    seq_ids = summary["seq_ids"]
    seq_lens = summary.get("seq_lens") or npz["seq_lens"].tolist()
    seq_lens = [int(L) for L in seq_lens]

    motion_263 = npz["motion_263"]
    world_R_y_angle = npz["world_R_y_angle"]
    world_T_xz = npz["world_T_xz"]

    per_clip = []
    all_samples: list[np.ndarray] = []  # for KS distance later
    for i, sid in enumerate(seq_ids):
        T = seq_lens[i]
        if T < 4:
            continue
        m_t = torch.from_numpy(motion_263[i, :T]).float().unsqueeze(0)
        canon = recover_from_ric(m_t, 22).squeeze(0).cpu().numpy().astype(np.float32)
        joints_world = _lift_canonical_to_world(
            canon, float(world_R_y_angle[i]), world_T_xz[i],
        )
        jerk_mag = _jerk_magnitudes(joints_world, fps)                       # (T-3, 22)
        if jerk_mag.size == 0:
            continue
        # Hands-only subset (left_hand=20, right_hand=21).
        hands_jerk = jerk_mag[:, [20, 21]]
        per_clip.append({
            "seq_id": sid,
            "T": T,
            "mean_jerk": round(float(jerk_mag.mean()), 3),
            "max_jerk": round(float(jerk_mag.max()), 3),
            "mean_jerk_hands": round(float(hands_jerk.mean()), 3),
            "mean_jerk_pelvis": round(float(jerk_mag[:, 0].mean()), 3),
            "per_joint_mean_jerk": [round(float(jerk_mag[:, j].mean()), 3) for j in range(22)],
        })
        all_samples.append(jerk_mag.flatten())

    if not per_clip:
        return {"per_clip": [], "agg": {}}

    samples_concat = np.concatenate(all_samples).astype(np.float32) if all_samples else np.array([], dtype=np.float32)
    if save_samples_to is not None and samples_concat.size > 0:
        ensure_dir(save_samples_to.parent)
        np.savez_compressed(save_samples_to, jerk=samples_concat)

    agg = {
        "n_clips": len(per_clip),
        "mean_jerk": round(float(np.mean([c["mean_jerk"] for c in per_clip])), 3),
        "max_jerk_avg": round(float(np.mean([c["max_jerk"] for c in per_clip])), 3),
        "max_jerk_overall": round(float(np.max([c["max_jerk"] for c in per_clip])), 3),
        "mean_jerk_hands": round(float(np.mean([c["mean_jerk_hands"] for c in per_clip])), 3),
        "mean_jerk_pelvis": round(float(np.mean([c["mean_jerk_pelvis"] for c in per_clip])), 3),
    }
    return {"per_clip": per_clip, "agg": agg, "samples_path": str(save_samples_to) if save_samples_to else None}


def _condition_label(input_dir: Path) -> str:
    return f"{input_dir.parent.name}/{input_dir.name}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument(
        "--save-samples", action="store_true",
        help="Save full per-(clip, joint, frame) jerk magnitudes for KS distance computation.",
    )
    parser.add_argument(
        "--detail", choices=["compact", "full"], default="compact",
    )
    args = parser.parse_args()
    ensure_dir(args.output_dir)

    summary = {
        "schema": "stage_b_motion_quality_v1",
        "fps": float(args.fps),
        "conditions": {},
    }

    for input_dir in args.input_dir:
        label = _condition_label(input_dir)
        print(f"  measuring {label} ...")
        save_samples_to = (
            args.output_dir / f"jerk_samples_{label.replace('/', '__')}.npz"
            if args.save_samples else None
        )
        info = _measure_condition(input_dir, fps=float(args.fps), save_samples_to=save_samples_to)
        if str(args.detail) == "full":
            summary["conditions"][label] = info
        else:
            summary["conditions"][label] = info.get("agg", {})
        agg = info.get("agg", {})
        if agg:
            print(
                f"    n={agg['n_clips']}  mean_jerk={agg['mean_jerk']} m/s^3  "
                f"max_jerk(avg)={agg['max_jerk_avg']} m/s^3  hands={agg['mean_jerk_hands']}  pelvis={agg['mean_jerk_pelvis']}",
            )

    out_path = args.output_dir / "summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
