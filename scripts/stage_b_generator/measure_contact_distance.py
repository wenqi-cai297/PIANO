"""Stage B contact-distance measurement (B0 from
`analyses/2026-04-27_v0_5_premise_review.md`).

Replaces the subjective mp4 ship gate with a number. For each clip ×
condition, lift the 263-d generated motion through ``recover_from_ric``
to canonical 22-joint pose, then to world frame via the saved
``world_R_y_angle`` / ``world_T_xz`` transform, and compute per-frame
distance from each of the 5 contact body parts (per
``piano.utils.smpl_utils.INTERACTION_BODY_PARTS``: left_hand 20,
right_hand 21, left_foot 10, right_foot 11, pelvis 0) to the object's
world-frame surface point cloud.

Aggregations per clip:
- ``per_body_part_mean_dist`` (5,): mean distance over all frames.
- ``per_body_part_min_dist`` (5,): closest approach (across the whole
  clip) for each body part — proxy for "did this body part actually
  reach the object?".
- ``mean_min_dist_per_frame`` (scalar): per frame take min across the 5
  body parts (= closest body part to object at that frame); average
  over time. Lower = body stays close to object on average.
- ``min_min_dist`` (scalar): minimum over time of the per-frame
  closest-body-part-to-object distance. The closest the body got to
  contacting the object at any moment in the clip.

Apply to v0.4 qual eval (full / text_only / swap) and the GT-VQ
roundtrip (gt_original / gt_roundtrip) to get three numbers that
resolve the v0.5 premise-review's question:

- ``d_codebook = mean(gt_roundtrip) − mean(gt_original)`` — irreducible
  loss from VQ encode/decode, with model held perfect.
- ``d_model = mean(v0.4_full) − mean(gt_roundtrip)`` — MaskTransformer
  prediction gap on top of codebook.
- ``d_total = mean(v0.4_full) − mean(gt_original)`` — overall gap to GT.

If ``d_codebook ≥ 0.5 × d_total``: codebook is the bottleneck → reopen
v0.3-γ codebook retrain. If ``d_model ≥ 0.5 × d_total``: model is the
bottleneck → push on v0.6 / v0.3-δ. If ``d_total < ~5 cm``: ship v0.4.

Pure post-processing on existing ``generated.npz`` files. No retrain,
no model forward, ~10s per condition.

Distance approximation: object surface is sampled at 1024 points (the
HOIDataset subsample); for a 30 cm object that's ~1 cm spacing, so
body-to-PC min distance is accurate to ≈0.5 cm. Plenty for
distinguishing 1 cm vs 5 cm vs 30 cm gaps. If sub-cm precision is
needed later, swap the PC distance for ``trimesh.proximity.closest_point``
on the actual object mesh.

Usage::

    python scripts/stage_b_generator/measure_contact_distance.py \\
        --input-dir runs/eval/stageB_v0_4_qual/full \\
        --input-dir runs/eval/stageB_v0_4_qual/text_only \\
        --input-dir runs/eval/stageB_v0_4_qual/swap \\
        --input-dir runs/eval/stageB_v0_4_gt_roundtrip/gt_original \\
        --input-dir runs/eval/stageB_v0_4_gt_roundtrip/gt_roundtrip \\
        --output-dir runs/eval/stageB_v0_4_contact_dist
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import piano.models.backbones.momask_adapter  # noqa: F401 — sys.path side-effect
from utils.motion_process import recover_from_ric

from piano.utils.canonical_frame import axis_angle_to_matrix_np, y_rotation_matrix
from piano.utils.io_utils import ensure_dir
from piano.utils.smpl_utils import BODY_PART_INDICES, BODY_PART_NAMES


# ============================================================================
# Geometric helpers
# ============================================================================

def _lift_canonical_to_world(
    joints_canon: np.ndarray,   # (T, 22, 3) in HumanML3D-canonical frame
    R_y_angle: float,           # rotation around Y to align canon → world
    T_xz: np.ndarray,           # (2,) — XZ pelvis translation at frame 0
) -> np.ndarray:
    """Apply ``world = R_y(angle) @ canonical + [T_xz[0], 0, T_xz[1]]``.

    Mirrors :func:`piano.utils.canonical_frame.get_canonicalize_transform_from_clip`'s
    forward direction (canonical → world) on a per-clip stored
    ``(R_y_angle, T_xz)`` pair.
    """
    R = y_rotation_matrix(float(R_y_angle))                # (3, 3)
    rotated = joints_canon @ R.T                           # (T, 22, 3); R for row vectors
    rotated[..., 0] += float(T_xz[0])
    rotated[..., 2] += float(T_xz[1])
    return rotated.astype(np.float32)


def _world_object_pc_per_frame(
    object_pc_local: np.ndarray,      # (N_pc, 3) in object-local frame
    object_positions: np.ndarray,     # (T, 3) world position
    object_rotations: np.ndarray,     # (T, 3) axis-angle, world frame
) -> np.ndarray:
    """Return ``(T, N_pc, 3)`` object PC in world frame at every frame.

    Vectorized over time via ``axis_angle_to_matrix_np`` (handles θ≈0
    safely; matches the project convention).
    """
    R_obj = axis_angle_to_matrix_np(object_rotations.astype(np.float32))   # (T, 3, 3)
    pc_world = np.einsum("tij,nj->tni", R_obj, object_pc_local.astype(np.float32))   # (T, N_pc, 3)
    pc_world += object_positions[:, None, :].astype(np.float32)
    return pc_world


def _per_frame_body_to_object_distance(
    body_joints_world: np.ndarray,    # (T, n_parts, 3)
    object_pc_local: np.ndarray,      # (N_pc, 3)
    object_positions: np.ndarray,     # (T, 3)
    object_rotations: np.ndarray,     # (T, 3) axis-angle
) -> np.ndarray:
    """Return ``(T, n_parts)`` min distance per body part per frame.

    Fully vectorized: builds ``(T, n_parts, N_pc, 3)`` diff tensor.
    For T≈150 / N_pc=1024 / n_parts=5 that's ≈9 MB intermediate — fine.
    """
    pc_world = _world_object_pc_per_frame(
        object_pc_local, object_positions, object_rotations,
    )                                                                 # (T, N_pc, 3)
    diff = body_joints_world[:, :, None, :] - pc_world[:, None, :, :] # (T, n_parts, N_pc, 3)
    d = np.linalg.norm(diff, axis=-1)                                  # (T, n_parts, N_pc)
    return d.min(axis=-1)                                              # (T, n_parts)


# ============================================================================
# Per-condition (one input dir = one condition) measurement
# ============================================================================

def _measure_condition(input_dir: Path) -> dict:
    """Measure body-to-object distances for every clip in ``input_dir``."""
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
    if seq_lens is None:
        raise ValueError(f"{summ_path} has no seq_lens, and {npz_path} doesn't either")
    seq_lens = [int(L) for L in seq_lens]

    motion_263 = npz["motion_263"]                # (N, T_max, 263)
    object_pc = npz["object_pc"]                  # (N, N_pc, 3)
    object_positions = npz["object_positions"]    # (N, T_max, 3)
    object_rotations = npz["object_rotations"]    # (N, T_max, 3) axis-angle world
    world_R_y_angle = npz["world_R_y_angle"]      # (N,)
    world_T_xz = npz["world_T_xz"]                # (N, 2)

    n_parts = len(BODY_PART_INDICES)
    per_clip = []
    for i, sid in enumerate(seq_ids):
        T = seq_lens[i]
        if T < 1:
            continue

        # 263 → canonical 22-joint pose (T, 22, 3).
        m_t = torch.from_numpy(motion_263[i, :T]).float().unsqueeze(0)      # (1, T, 263)
        canon = recover_from_ric(m_t, 22).squeeze(0).cpu().numpy().astype(np.float32)

        # Canonical → world via stored R_y_angle + T_xz (frame-0 anchored).
        world_joints = _lift_canonical_to_world(
            canon, float(world_R_y_angle[i]), world_T_xz[i],
        )                                                                    # (T, 22, 3)

        # Pick the 5 contact body parts (left_hand, right_hand, left_foot,
        # right_foot, pelvis).
        body_joints = world_joints[:, BODY_PART_INDICES, :]                  # (T, 5, 3)

        # Per-frame body-to-object distance, vectorized over PC.
        d = _per_frame_body_to_object_distance(
            body_joints,
            object_pc[i],
            object_positions[i, :T],
            object_rotations[i, :T],
        )                                                                    # (T, 5)

        # Aggregations.
        min_per_frame = d.min(axis=1)                                        # (T,) closest body-part-to-object at each frame
        per_clip.append({
            "seq_id": sid,
            "T": T,
            "per_body_part_mean_dist":   [round(float(d[:, p].mean()), 4) for p in range(n_parts)],
            "per_body_part_min_dist":    [round(float(d[:, p].min()),  4) for p in range(n_parts)],
            "mean_min_dist_per_frame":   round(float(min_per_frame.mean()), 4),
            "min_min_dist":              round(float(min_per_frame.min()),  4),
        })

    # Aggregate across clips (uniform-weighted, since clip count is small + balanced).
    if not per_clip:
        return {"per_clip": [], "agg": {}}
    agg = {
        "n_clips": len(per_clip),
        "agg_per_body_part_mean_dist": [
            round(float(np.mean([c["per_body_part_mean_dist"][p] for c in per_clip])), 4)
            for p in range(n_parts)
        ],
        "agg_per_body_part_min_dist": [
            round(float(np.mean([c["per_body_part_min_dist"][p] for c in per_clip])), 4)
            for p in range(n_parts)
        ],
        "agg_mean_min_dist_per_frame":
            round(float(np.mean([c["mean_min_dist_per_frame"] for c in per_clip])), 4),
        "agg_min_min_dist":
            round(float(np.mean([c["min_min_dist"]         for c in per_clip])), 4),
    }
    return {"per_clip": per_clip, "agg": agg}


# ============================================================================
# Main
# ============================================================================

def _condition_label(input_dir: Path) -> str:
    """Stable label for the condition: ``<parent_dir>/<dir_name>``."""
    return f"{input_dir.parent.name}/{input_dir.name}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input-dir", action="append", required=True,
        help="path to a condition dir containing generated.npz + summary.json. "
             "Pass multiple times for multiple conditions.",
    )
    parser.add_argument(
        "--output-dir", "-o", type=Path,
        default=Path("runs/eval/stageB_v0_4_contact_dist"),
        help="directory to write summary.json (created if absent).",
    )
    args = parser.parse_args()

    out: dict = {
        "body_part_names": BODY_PART_NAMES,
        "body_part_indices": BODY_PART_INDICES,
        "conditions": {},
    }
    for raw in args.input_dir:
        in_dir = Path(raw)
        label = _condition_label(in_dir)
        print(f"  {label} ...", flush=True)
        out["conditions"][label] = _measure_condition(in_dir)

    ensure_dir(args.output_dir)
    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {summary_path}")

    # Stdout summary table.
    header = f"  {'condition':<55s} {'mean_min_dist':>14s} {'min_min_dist':>13s}"
    print()
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label, info in out["conditions"].items():
        agg = info.get("agg", {})
        if not agg:
            print(f"  {label:<55s} {'(no clips)':>14s}")
            continue
        print(
            f"  {label:<55s} "
            f"{agg['agg_mean_min_dist_per_frame']:>13.3f} m "
            f"{agg['agg_min_min_dist']:>11.3f} m"
        )
    print()
    print("  Interpretation:")
    print("    mean_min_dist_per_frame = average over time of (closest body part to object).")
    print("                              Lower = body stays close throughout the clip.")
    print("    min_min_dist            = closest the body got to the object at any frame.")
    print("                              Lower = body actually reached the object at some point.")
    print()
    print("  Compare gt_original vs gt_roundtrip → codebook irreducible loss.")
    print("  Compare gt_roundtrip vs <model>/full → MaskTransformer prediction gap.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
