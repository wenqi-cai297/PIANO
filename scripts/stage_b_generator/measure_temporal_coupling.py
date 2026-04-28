"""Measure whether generated body contact is temporally coupled to object motion.

The existing Stage B contact metric answers: "is some tracked body part close to
the object surface?"  It does not answer: "when the object moves, does a body
part move with it?"  This script adds that second check using the same
kinematic-coupling criterion used by the pseudo-label contact extractor.

Usage:

    python scripts/stage_b_generator/measure_temporal_coupling.py \
        --input-dir runs/eval/stageB_v0_12_w02_bv_k16_oracle/best \
        --output-dir runs/eval/stageB_v0_12_w02_bv_k16_oracle/temporal_coupling
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import piano.models.backbones.momask_adapter  # noqa: F401 - MoMask path side-effect
from utils.motion_process import recover_from_ric

from piano.data.pseudo_labels.extract_contact import (
    ContactConfig,
    _kinematic_contact_score,
)
from piano.training.contact_eval import (
    _lift_canonical_to_world,
    _per_frame_body_to_object_distance,
)
from piano.utils.io_utils import ensure_dir
from piano.utils.smpl_utils import BODY_PART_INDICES, BODY_PART_NAMES


def _round(x: float | None, ndigits: int = 4) -> float | None:
    if x is None or not np.isfinite(x):
        return None
    return round(float(x), ndigits)


def _mean_or_none(values: list[float | None]) -> float | None:
    xs = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not xs:
        return None
    return float(np.mean(xs))


def _object_motion_speed(
    object_positions: np.ndarray,
    object_rotations: np.ndarray | None,
    cfg: ContactConfig,
) -> np.ndarray:
    """Match the object-speed proxy used by `_kinematic_contact_score`."""
    T = len(object_positions)
    trans_vel = np.zeros(T, dtype=np.float32)
    if T > 1:
        trans_vel[1:] = (
            np.linalg.norm(np.diff(object_positions, axis=0), axis=-1) * cfg.fps
        )

    ang_vel = np.zeros(T, dtype=np.float32)
    if object_rotations is not None and T > 1:
        ang_vel[1:] = (
            np.linalg.norm(np.diff(object_rotations, axis=0), axis=-1) * cfg.fps
        )

    return trans_vel + float(cfg.kin_radius_proxy) * ang_vel


def _load_summary(input_dir: Path) -> dict[str, Any]:
    summary_path = input_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing {summary_path}")
    with summary_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_oracle_clip_meta(input_dir: Path) -> dict[str, dict[str, Any]]:
    """Load optional K-oracle metadata from the parent summary.

    `k_sample_oracle.py --save-best` writes best/generated.npz under
    `<oracle_dir>/best`, while the rich per-clip metadata lives in
    `<oracle_dir>/summary.json`.
    """
    parent_summary = input_dir.parent / "summary.json"
    if not parent_summary.exists():
        return {}
    try:
        with parent_summary.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return {}

    out: dict[str, dict[str, Any]] = {}
    for item in data.get("clip_selection", []):
        seq_id = str(item.get("seq_id", ""))
        if seq_id:
            out.setdefault(seq_id, {}).update({
                "subset": item.get("subset"),
                "object_id": item.get("object_id"),
                "dataset_index": item.get("index"),
            })
    for item in data.get("per_clip", []):
        seq_id = str(item.get("seq_id", ""))
        if seq_id:
            out.setdefault(seq_id, {}).update({
                "best_sample_index": item.get("best_sample_index"),
                "best_seed": item.get("best_seed"),
                "best_dist_cm": item.get("best_dist_cm"),
                "single_sample_dist_cm": item.get("single_sample_dist_cm"),
            })
    return out


def score_motion_temporal_coupling(
    *,
    motion_263_generated: np.ndarray,
    R_y_angle: float,
    T_xz: np.ndarray,
    object_pc_local: np.ndarray,
    object_positions: np.ndarray,
    object_rotations: np.ndarray | None,
    seq_len: int,
    fps: float = 20.0,
    coupling_threshold: float = 0.5,
    moving_speed_threshold: float | None = None,
) -> dict[str, Any]:
    """Score one generated motion for spatial contact and temporal coupling."""
    T = min(int(seq_len), int(motion_263_generated.shape[0]))
    if T < 1:
        return {}

    cfg = ContactConfig(fps=float(fps))
    speed_threshold = (
        float(moving_speed_threshold)
        if moving_speed_threshold is not None
        else float(cfg.kin_world_eps)
    )
    close_thresholds = np.array(
        [cfg.distance_thresholds[name] for name in BODY_PART_NAMES],
        dtype=np.float32,
    )

    motion_t = torch.from_numpy(motion_263_generated[:T]).float().unsqueeze(0)
    canon = recover_from_ric(motion_t, 22).squeeze(0).cpu().numpy().astype(np.float32)
    world_joints = _lift_canonical_to_world(canon, float(R_y_angle), T_xz)
    body_joints = world_joints[:, BODY_PART_INDICES, :]

    obj_pos = object_positions[:T]
    obj_rot = object_rotations[:T] if object_rotations is not None else None
    d = _per_frame_body_to_object_distance(
        body_joints,
        object_pc_local,
        obj_pos,
        obj_rot if obj_rot is not None else np.zeros((T, 3), dtype=np.float32),
    )
    min_per_frame = d.min(axis=1)
    close_any = (d <= close_thresholds[None, :]).any(axis=1)

    kin_scores = np.stack([
        _kinematic_contact_score(body_joints[:, p, :], obj_pos, obj_rot, cfg)
        for p in range(len(BODY_PART_NAMES))
    ], axis=1)
    best_kin = kin_scores.max(axis=1)
    coupled_any = best_kin >= float(coupling_threshold)

    speed = _object_motion_speed(obj_pos, obj_rot, cfg)
    moving = speed >= speed_threshold
    nonmoving = ~moving
    n_moving = int(moving.sum())
    n_nonmoving = int(nonmoving.sum())

    if n_moving > 0:
        moving_close = float(close_any[moving].mean())
        moving_coupled = float(coupled_any[moving].mean())
        moving_close_uncoupled = float((close_any[moving] & ~coupled_any[moving]).mean())
        moving_best_kin = float(best_kin[moving].mean())
    else:
        moving_close = None
        moving_coupled = None
        moving_close_uncoupled = None
        moving_best_kin = None

    nonmoving_coupled = (
        float(coupled_any[nonmoving].mean()) if n_nonmoving > 0 else None
    )

    return {
        "T": T,
        "n_moving_frames": n_moving,
        "moving_frame_frac": _round(n_moving / T),
        "mean_min_dist_m": _round(float(min_per_frame.mean())),
        "close_frame_frac": _round(float(close_any.mean())),
        "moving_close_frame_frac": _round(moving_close),
        "moving_coupled_frame_frac": _round(moving_coupled),
        "moving_close_but_uncoupled_frac": _round(moving_close_uncoupled),
        "moving_mean_best_kin_score": _round(moving_best_kin),
        "nonmoving_coupled_frame_frac": _round(nonmoving_coupled),
        "mean_object_speed_mps": _round(float(speed.mean())),
        "p95_object_speed_mps": _round(float(np.percentile(speed, 95))),
    }


def _aggregate(per_clip: list[dict[str, Any]]) -> dict[str, Any]:
    if not per_clip:
        return {}
    scalar_keys = [
        "mean_min_dist_m",
        "moving_frame_frac",
        "close_frame_frac",
        "moving_close_frame_frac",
        "moving_coupled_frame_frac",
        "moving_close_but_uncoupled_frac",
        "moving_mean_best_kin_score",
        "nonmoving_coupled_frame_frac",
    ]
    agg: dict[str, Any] = {"n_clips": len(per_clip)}
    for key in scalar_keys:
        agg[key] = _round(_mean_or_none([c.get(key) for c in per_clip]))
    agg["n_moving_clips"] = int(sum(int(c.get("n_moving_frames", 0)) > 0 for c in per_clip))
    return agg


def _measure_condition(
    input_dir: Path,
    *,
    fps: float,
    coupling_threshold: float,
    moving_speed_threshold: float | None,
) -> dict[str, Any]:
    npz_path = input_dir / "generated.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"missing {npz_path}")

    summary = _load_summary(input_dir)
    oracle_meta = _load_oracle_clip_meta(input_dir)

    npz = np.load(npz_path)
    seq_ids = [str(s) for s in summary["seq_ids"]]
    seq_lens = summary.get("seq_lens", None)
    if seq_lens is None and "seq_lens" in npz.files:
        seq_lens = npz["seq_lens"].tolist()
    if seq_lens is None:
        raise ValueError(f"{input_dir}/summary.json has no seq_lens")
    seq_lens = [int(x) for x in seq_lens]

    motion_263 = npz["motion_263"]
    object_pc = npz["object_pc"]
    object_positions = npz["object_positions"]
    object_rotations = npz["object_rotations"] if "object_rotations" in npz.files else None
    world_R_y_angle = npz["world_R_y_angle"]
    world_T_xz = npz["world_T_xz"]

    cfg = ContactConfig(fps=float(fps))
    speed_threshold = (
        float(moving_speed_threshold)
        if moving_speed_threshold is not None
        else float(cfg.kin_world_eps)
    )
    per_clip: list[dict[str, Any]] = []
    for i, seq_id in enumerate(seq_ids):
        T = min(int(seq_lens[i]), int(motion_263.shape[1]))
        if T < 1:
            continue

        item: dict[str, Any] = {
            "seq_id": seq_id,
        }
        item.update(score_motion_temporal_coupling(
            motion_263_generated=motion_263[i, :T],
            R_y_angle=float(world_R_y_angle[i]),
            T_xz=world_T_xz[i],
            object_pc_local=object_pc[i],
            object_positions=object_positions[i, :T],
            object_rotations=(object_rotations[i, :T] if object_rotations is not None else None),
            seq_len=T,
            fps=float(fps),
            coupling_threshold=float(coupling_threshold),
            moving_speed_threshold=speed_threshold,
        ))
        item.update(oracle_meta.get(seq_id, {}))
        per_clip.append(item)

    by_subset: dict[str, Any] = {}
    subsets = sorted({c.get("subset") for c in per_clip if c.get("subset")})
    for subset in subsets:
        subset_rows = [c for c in per_clip if c.get("subset") == subset]
        by_subset[str(subset)] = _aggregate(subset_rows)

    worst_temporal = sorted(
        [c for c in per_clip if int(c.get("n_moving_frames", 0)) >= 5],
        key=lambda c: (
            c.get("moving_coupled_frame_frac") is None,
            c.get("moving_coupled_frame_frac") if c.get("moving_coupled_frame_frac") is not None else 9.0,
            c.get("mean_min_dist_m", 9.0),
        ),
    )[:20]

    return {
        "input_dir": str(input_dir),
        "fps": float(fps),
        "coupling_threshold": float(coupling_threshold),
        "moving_speed_threshold": float(speed_threshold),
        "body_part_names": BODY_PART_NAMES,
        "body_part_indices": BODY_PART_INDICES,
        "aggregate": _aggregate(per_clip),
        "by_subset": by_subset,
        "worst_temporal_coupling": worst_temporal,
        "per_clip": per_clip,
    }


def _condition_label(input_dir: Path) -> str:
    return f"{input_dir.parent.name}/{input_dir.name}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input-dir",
        action="append",
        required=True,
        help="condition dir containing generated.npz + summary.json; can repeat",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        required=True,
        help="directory to write summary.json",
    )
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--coupling-threshold", type=float, default=0.5)
    parser.add_argument(
        "--moving-speed-threshold",
        type=float,
        default=None,
        help="object speed threshold in m/s; default uses ContactConfig.kin_world_eps",
    )
    args = parser.parse_args()

    out = {"conditions": {}}
    for raw in args.input_dir:
        input_dir = Path(raw)
        label = _condition_label(input_dir)
        print(f"  {label} ...", flush=True)
        out["conditions"][label] = _measure_condition(
            input_dir,
            fps=args.fps,
            coupling_threshold=args.coupling_threshold,
            moving_speed_threshold=args.moving_speed_threshold,
        )

    ensure_dir(args.output_dir)
    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"\nWrote {summary_path}")
    print()
    print(
        f"  {'condition':<55s} {'dist':>8s} {'move_close':>11s} "
        f"{'move_coupled':>13s} {'close_uncoup':>13s}"
    )
    print("  " + "-" * 106)
    for label, info in out["conditions"].items():
        agg = info.get("aggregate", {})
        print(
            f"  {label:<55s} "
            f"{agg.get('mean_min_dist_m', None)!s:>8s} "
            f"{agg.get('moving_close_frame_frac', None)!s:>11s} "
            f"{agg.get('moving_coupled_frame_frac', None)!s:>13s} "
            f"{agg.get('moving_close_but_uncoupled_frac', None)!s:>13s}"
        )

    print()
    print("  Interpretation:")
    print("    dist          = ordinary closest-body-part contact distance (m).")
    print("    move_close    = moving-object frames with any tracked body part near the object.")
    print("    move_coupled  = moving-object frames where a body part is stable in object-local frame.")
    print("    close_uncoup  = moving-object frames that are near but not rigidly coupled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
