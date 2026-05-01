"""Body-vs-object penetration metric (path b: 22-joint sphere vs object PC convex hull).

Implements N1 / N2 from analyses/2026-05-02_codec_floor_baselines.md and the
2026-05-03 metric review.

For each generated motion clip and each frame:
  1. Recover 22-joint world-frame skeleton via recover_from_ric + canonical→world.
  2. Build the object's per-frame world-frame convex hull (PC's convex hull in
     object-local frame, transformed to world by per-frame object pose).
     Convex hull is computed once per clip in object-local frame, joints are
     transformed back to object-local for SDF query (faster + same result).
  3. For each joint j with sphere radius r_j (anthropometric defaults below),
     compute SDF of joint position relative to hull. trimesh convention:
     positive = inside, negative = outside.
  4. penetration_depth_jt = max(sd_jt + r_j, 0)
     - sd > 0 (joint inside): sphere has fully pierced the surface; depth = sd + r
     - sd ∈ (-r, 0) (joint outside but within radius): sphere edge touches
       interior; depth = r + sd = r - |sd|
     - sd <= -r (sphere entirely outside): depth = 0

Aggregations:
  - mean_part_penetration_depth : mean over all (j, t) of penetration_depth
  - max_part_penetration_depth  : max over (j, t)
  - frac_frames_pen_gt_2cm      : fraction of frames where any joint has > 2 cm
                                   penetration (a "violation frame")

Caveats:
  - **Convex hull approximation.** Concave objects (chairs, tables) have
    "interior" space inside the hull that the body can validly occupy
    (e.g. legs under a table). This introduces false positives. Mitigation:
    we report GT_orig + GT_roundtrip on the same metric — those baselines
    represent the false-positive floor under this approximation. A model's
    penetration is meaningful only relative to those baselines.
  - **Finger-level out of scope.** PIANO uses 22-joint SMPL (no finger
    articulation). Hand region is approximated as a 5 cm sphere at the
    wrist joint (joints 20/21). Matches metric scope of InterDiff (ICCV 2023),
    CHOIS (CVPR 2024), HOI-Diff, and CG-HOI — all 22-joint methods.

Usage:
    python scripts/stage_b_generator/measure_penetration.py \\
        --input-dir runs/eval/<EVAL_PREFIX>_<ckpt>_qual/full_guided \\
        --input-dir runs/eval/<EVAL_PREFIX>_gt_roundtrip_<N>/gt_original \\
        --output-dir runs/eval/<EVAL_PREFIX>_<ckpt>_penetration
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import trimesh

import piano.models.backbones.momask_adapter  # noqa: F401 - sys.path side-effect
from utils.motion_process import recover_from_ric

from piano.utils.canonical_frame import axis_angle_to_matrix_np, y_rotation_matrix
from piano.utils.io_utils import ensure_dir


# ============================================================================
# 22-joint SMPL anthropometric body sphere radii (m).
# Per-joint sphere represents the soft-tissue volume centered at that joint.
# Values from typical adult SMPL body model dimensions; sized so that
# adjacent spheres (e.g. shoulder + upper arm) overlap reasonably.
# ============================================================================
JOINT_RADII: dict[int, float] = {
    0:  0.15,                       # pelvis (hip + buttock soft tissue)
    1:  0.12, 2:  0.12,             # l/r hip (upper thigh)
    3:  0.13,                       # spine1 (lower abdomen)
    4:  0.08, 5:  0.08,             # l/r knee (kneecap region)
    6:  0.13,                       # spine2 (mid torso)
    7:  0.06, 8:  0.06,             # l/r ankle
    9:  0.13,                       # spine3 (upper torso / chest)
    10: 0.06, 11: 0.06,             # l/r foot (mid-foot)
    12: 0.07,                       # neck
    13: 0.09, 14: 0.09,             # l/r collar
    15: 0.10,                       # head
    16: 0.10, 17: 0.10,             # l/r shoulder
    18: 0.06, 19: 0.06,             # l/r elbow
    20: 0.05, 21: 0.05,             # l/r wrist (hand region; finger out of scope)
}
JOINT_RADII_ARR = np.array([JOINT_RADII[j] for j in range(22)], dtype=np.float32)


# ============================================================================
# Geometric helpers (same conventions as measure_contact_distance.py)
# ============================================================================

def _lift_canonical_to_world(
    joints_canon: np.ndarray,        # (T, 22, 3)
    R_y_angle: float,
    T_xz: np.ndarray,                # (2,)
) -> np.ndarray:
    R = y_rotation_matrix(float(R_y_angle))
    rotated = joints_canon @ R.T
    rotated[..., 0] += float(T_xz[0])
    rotated[..., 2] += float(T_xz[1])
    return rotated.astype(np.float32)


def _world_joints_to_object_local(
    joints_world: np.ndarray,         # (T, 22, 3)
    object_positions: np.ndarray,     # (T, 3)
    object_rotations: np.ndarray,     # (T, 3) axis-angle
) -> np.ndarray:
    """Inverse of `_world_object_pc_per_frame`: world joints → object-local."""
    R_obj = axis_angle_to_matrix_np(object_rotations.astype(np.float32))   # (T, 3, 3)
    centered = joints_world - object_positions[:, None, :]                  # (T, 22, 3)
    # joints_local = R_obj^T @ (joints_world - obj_pos)
    return np.einsum("tij,tnj->tni", R_obj.transpose(0, 2, 1), centered)


# ============================================================================
# Per-clip penetration measurement
# ============================================================================

def _measure_clip_penetration(
    joints_world: np.ndarray,        # (T, 22, 3)
    object_pc_local: np.ndarray,     # (N_pc, 3)
    object_positions: np.ndarray,    # (T, 3)
    object_rotations: np.ndarray,    # (T, 3) axis-angle
) -> np.ndarray:
    """Return (T, 22) penetration depth in metres. Trimesh convention:
    signed_distance positive = inside, negative = outside. We compute
    the convex hull ONCE in object-local frame and transform query points
    back to object-local — strictly equivalent to per-frame world-frame
    hulls but ~T× faster.
    """
    # Convex hull in object-local frame (constant across t).
    try:
        hull_local = trimesh.PointCloud(object_pc_local).convex_hull
    except Exception as exc:
        # Fallback: degenerate PC (collinear / coplanar). Inflate slightly.
        jittered = object_pc_local + np.random.RandomState(0).randn(*object_pc_local.shape).astype(np.float32) * 1e-4
        hull_local = trimesh.PointCloud(jittered).convex_hull

    joints_local = _world_joints_to_object_local(
        joints_world, object_positions, object_rotations,
    )                                                                    # (T, 22, 3)
    T = int(joints_local.shape[0])
    flat = joints_local.reshape(-1, 3)                                   # (T*22, 3)
    sd_flat = trimesh.proximity.signed_distance(hull_local, flat)        # (+) inside
    sd = sd_flat.astype(np.float32).reshape(T, 22)
    pen = np.maximum(sd + JOINT_RADII_ARR[None, :], 0.0)                 # (T, 22)
    return pen


# ============================================================================
# Per-condition (one input dir = one condition)
# ============================================================================

def _measure_condition(input_dir: Path) -> dict:
    npz_path = input_dir / "generated.npz"
    summ_path = input_dir / "summary.json"
    if not npz_path.exists():
        raise FileNotFoundError(f"missing {npz_path}")
    if not summ_path.exists():
        raise FileNotFoundError(f"missing {summ_path}")

    npz = np.load(npz_path)
    with summ_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    seq_ids = summary["seq_ids"]
    seq_lens = summary.get("seq_lens", None)
    if seq_lens is None and "seq_lens" in npz.files:
        seq_lens = npz["seq_lens"].tolist()
    seq_lens = [int(L) for L in seq_lens]

    motion_263 = npz["motion_263"]
    object_pc = npz["object_pc"]
    object_positions = npz["object_positions"]
    object_rotations = npz["object_rotations"]
    world_R_y_angle = npz["world_R_y_angle"]
    world_T_xz = npz["world_T_xz"]

    per_clip = []
    for i, sid in enumerate(seq_ids):
        T = seq_lens[i]
        if T < 1:
            continue
        m_t = torch.from_numpy(motion_263[i, :T]).float().unsqueeze(0)
        canon = recover_from_ric(m_t, 22).squeeze(0).cpu().numpy().astype(np.float32)
        joints_world = _lift_canonical_to_world(
            canon, float(world_R_y_angle[i]), world_T_xz[i],
        )
        pen = _measure_clip_penetration(
            joints_world,
            object_pc[i].astype(np.float32),
            object_positions[i, :T].astype(np.float32),
            object_rotations[i, :T].astype(np.float32),
        )                                                                # (T, 22)
        # Per-frame max-over-joints penetration; "violation frame" = > 2 cm
        per_frame_max = pen.max(axis=1)                                  # (T,)
        per_clip.append({
            "seq_id": sid,
            "T": T,
            "mean_pen_m": round(float(pen.mean()), 5),
            "max_pen_m": round(float(pen.max()), 5),
            "frac_frames_pen_gt_2cm": round(float((per_frame_max > 0.02).mean()), 4),
            "frac_frames_pen_gt_5cm": round(float((per_frame_max > 0.05).mean()), 4),
            # Per-joint breakdown for debugging interesting clips
            "per_joint_mean_pen_m": [round(float(pen[:, j].mean()), 5) for j in range(22)],
        })

    if not per_clip:
        return {"per_clip": [], "agg": {}}

    agg = {
        "n_clips": len(per_clip),
        "mean_pen_m": round(float(np.mean([c["mean_pen_m"] for c in per_clip])), 5),
        "max_pen_m_avg": round(float(np.mean([c["max_pen_m"] for c in per_clip])), 5),
        "max_pen_m_overall": round(float(np.max([c["max_pen_m"] for c in per_clip])), 5),
        "frac_frames_pen_gt_2cm": round(float(np.mean([c["frac_frames_pen_gt_2cm"] for c in per_clip])), 4),
        "frac_frames_pen_gt_5cm": round(float(np.mean([c["frac_frames_pen_gt_5cm"] for c in per_clip])), 4),
        # Aggregate per-joint mean (which joints penetrate most often)
        "agg_per_joint_mean_pen_m": [
            round(float(np.mean([c["per_joint_mean_pen_m"][j] for c in per_clip])), 5)
            for j in range(22)
        ],
    }
    return {"per_clip": per_clip, "agg": agg}


# ============================================================================
# Main
# ============================================================================

def _condition_label(input_dir: Path) -> str:
    return f"{input_dir.parent.name}/{input_dir.name}"


def _compact_condition(info: dict) -> dict:
    agg = info.get("agg", {})
    if not agg:
        return {"n_clips": 0}
    return {
        "n_clips": int(agg["n_clips"]),
        "mean_pen_m": agg["mean_pen_m"],
        "max_pen_m_avg": agg["max_pen_m_avg"],
        "max_pen_m_overall": agg["max_pen_m_overall"],
        "frac_frames_pen_gt_2cm": agg["frac_frames_pen_gt_2cm"],
        "frac_frames_pen_gt_5cm": agg["frac_frames_pen_gt_5cm"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", action="append", required=True, type=Path,
        help="Repeatable. One per condition — directory with generated.npz + summary.json.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--detail", choices=["compact", "full"], default="compact",
        help="full keeps per-clip per-joint breakdown.",
    )
    args = parser.parse_args()
    ensure_dir(args.output_dir)

    summary = {
        "schema": "stage_b_penetration_v1",
        "joint_radii_m": {str(k): v for k, v in JOINT_RADII.items()},
        "approximation": "convex_hull_of_object_pc",
        "scope": "body_level_22_joint_sphere; finger_level_out_of_scope",
        "conditions": {},
    }

    for input_dir in args.input_dir:
        label = _condition_label(input_dir)
        print(f"  measuring {label} ...")
        info = _measure_condition(input_dir)
        if str(args.detail) == "full":
            summary["conditions"][label] = {
                "agg": info["agg"],
                "per_clip": info["per_clip"],
            }
        else:
            summary["conditions"][label] = _compact_condition(info)
        agg = info.get("agg", {})
        if agg:
            print(
                f"    n={agg['n_clips']}  mean_pen={agg['mean_pen_m']*100:.2f} cm  "
                f"max_pen_avg={agg['max_pen_m_avg']*100:.2f} cm  "
                f"frames>2cm={agg['frac_frames_pen_gt_2cm']*100:.1f}%  "
                f"frames>5cm={agg['frac_frames_pen_gt_5cm']*100:.1f}%",
            )

    out_path = args.output_dir / "summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
