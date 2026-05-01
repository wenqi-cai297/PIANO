"""Compare v11 (existing) vs v12 (strict) contact definitions on local
GT motion data using point-cloud distance approximation.

The real pseudo-label extractors use object meshes (via
trimesh.proximity), but those datasets aren't synced locally. This
script approximates `points_to_mesh_distance` with the nearest-PC
distance — strictly an upper bound on the mesh distance, but accurate
to ~1-2 cm given dense PC sampling (256+ points).

Use this to:
  1. Verify the v12 strict definition produces noticeably fewer (but
     more meaningful) contact frames than v11.
  2. See per-subset breakdowns (chairs / IMHD / NeuralDome / OMOMO).
  3. Tune strict thresholds before kicking off the full server-side
     re-extraction.

Usage:
    python scripts/stage_b_generator/evaluate_contact_definitions_pc.py \\
        --input-dir runs/eval/<...>_gt_roundtrip_80/gt_original \\
        --output-dir runs/eval/_contact_def_compare
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import piano.models.backbones.momask_adapter  # noqa: F401
from utils.motion_process import recover_from_ric

from piano.data.pseudo_labels.extract_contact import (
    ContactConfig,
    DEFAULT_DISTANCE_THRESHOLDS,
    _kinematic_contact_score,
    _soft_sigmoid,
)
from piano.data.pseudo_labels.extract_strict_contact import (
    STRICT_DISTANCE_THRESHOLDS,
    StrictContactConfig,
    _filter_drifting_contacts,
    _filter_short_contacts,
    _static_engagement_score,
)
from piano.data.pseudo_labels._object_transform import world_to_object_local
from piano.utils.canonical_frame import y_rotation_matrix
from piano.utils.io_utils import ensure_dir
from piano.utils.smpl_utils import (
    BODY_PART_INDICES,
    BODY_PART_NAMES,
    NUM_BODY_PARTS,
)
from scipy.ndimage import median_filter, uniform_filter1d


def _lift_canonical_to_world(joints_canon, R_y_angle, T_xz):
    R = y_rotation_matrix(float(R_y_angle))
    rotated = joints_canon @ R.T
    rotated[..., 0] += float(T_xz[0])
    rotated[..., 2] += float(T_xz[1])
    return rotated.astype(np.float32)


def _nearest_pc_distance(points_local: np.ndarray, pc_local: np.ndarray) -> np.ndarray:
    """Approximate `points_to_mesh_distance` with nearest-PC distance.

    points_local : (T, 3)
    pc_local     : (N_pc, 3)
    Returns      : (T,) distance to nearest PC point.
    """
    # (T, N_pc, 3)
    diff = points_local[:, None, :] - pc_local[None, :, :]
    d = np.linalg.norm(diff, axis=-1)
    return d.min(axis=-1)


def _v11_contact(joints, object_pc, object_positions, object_rotations, *, fps=20.0):
    """V11-equivalent contact extraction with PC-distance approximation."""
    config = ContactConfig(fps=float(fps))
    T = len(joints)
    contact = np.zeros((T, NUM_BODY_PARTS), dtype=np.float32)

    for bp_idx, joint_idx in enumerate(BODY_PART_INDICES):
        bp_name = BODY_PART_NAMES[bp_idx]
        bp_world = joints[:, joint_idx, :]
        bp_local = world_to_object_local(bp_world, object_positions, object_rotations)
        distances = _nearest_pc_distance(bp_local, object_pc)
        threshold = DEFAULT_DISTANCE_THRESHOLDS[bp_name]
        dist_score = _soft_sigmoid(distances, threshold, config.distance_sigma)
        kin_score = _kinematic_contact_score(
            bp_world, object_positions, object_rotations, config,
        )
        contact[:, bp_idx] = np.maximum(dist_score, kin_score)

    for bp_idx in range(NUM_BODY_PARTS):
        contact[:, bp_idx] = median_filter(contact[:, bp_idx], size=config.median_filter_size)
    contact = _filter_short_contacts(contact, config.min_contact_duration)
    return contact


def _v12_strict_contact(joints, object_pc, object_positions, object_rotations, *, fps=20.0):
    """V12 strict contact extraction with PC-distance approximation."""
    sc = StrictContactConfig(fps=float(fps))
    base = ContactConfig(fps=float(fps))
    T = len(joints)
    contact = np.zeros((T, NUM_BODY_PARTS), dtype=np.float32)
    body_locals = np.zeros((T, NUM_BODY_PARTS, 3), dtype=np.float32)

    # Object speed proxy for static-engagement detection
    trans_vel = np.zeros(T, dtype=np.float32)
    trans_vel[1:] = np.linalg.norm(np.diff(object_positions, axis=0), axis=-1) * fps
    ang_vel = np.zeros(T, dtype=np.float32)
    if object_rotations is not None:
        ang_vel[1:] = np.linalg.norm(np.diff(object_rotations, axis=0), axis=-1) * fps
    obj_speed = trans_vel + base.kin_radius_proxy * ang_vel

    kin_window = max(3, int(round(base.kin_window_sec * fps)))

    for bp_idx, joint_idx in enumerate(BODY_PART_INDICES):
        bp_name = BODY_PART_NAMES[bp_idx]
        bp_world = joints[:, joint_idx, :]
        bp_local = world_to_object_local(bp_world, object_positions, object_rotations)
        body_locals[:, bp_idx, :] = bp_local

        distances = _nearest_pc_distance(bp_local, object_pc)
        thr = STRICT_DISTANCE_THRESHOLDS[bp_name]
        dist_score = _soft_sigmoid(distances, thr, sc.distance_sigma)

        kin_score = _kinematic_contact_score(
            bp_world, object_positions, object_rotations, base,
        )
        static_score = _static_engagement_score(
            bp_local, obj_speed,
            kin_window=kin_window,
            eps_mps=sc.static_engagement_eps_mps,
            local_std_thresh=sc.static_engagement_local_std_m,
        )
        engagement = np.maximum(kin_score, static_score)
        contact[:, bp_idx] = dist_score * engagement

    for bp_idx in range(NUM_BODY_PARTS):
        contact[:, bp_idx] = median_filter(contact[:, bp_idx], size=sc.median_filter_size)
    contact = _filter_short_contacts(contact, sc.min_contact_duration)
    contact = _filter_drifting_contacts(
        contact, body_locals, max_drift_m=sc.max_segment_drift_m,
    )
    return contact


def _segment_stats(binary: np.ndarray) -> dict:
    """Count and report contact segment statistics."""
    T, B = binary.shape
    n_seg = 0
    seg_durations = []
    for bp in range(B):
        b = binary[:, bp]
        changes = np.diff(b.astype(np.int8), prepend=0, append=0)
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        for s, e in zip(starts, ends):
            n_seg += 1
            seg_durations.append(e - s)
    return {
        "n_segments": n_seg,
        "mean_segment_duration_frames": float(np.mean(seg_durations)) if seg_durations else 0.0,
        "median_segment_duration_frames": float(np.median(seg_durations)) if seg_durations else 0.0,
    }


def _infer_subset(seq_id: str) -> str:
    """Pattern-match seq_id to dataset subset."""
    s = seq_id.lower()
    if seq_id.startswith("Sub") and "_Obj" in seq_id and "_Seg" in seq_id:
        return "chairs"
    if s.startswith("20230"):
        return "imhd"
    if s.startswith("subject"):
        return "neuraldome"
    if s.startswith("sub"):
        return "omomo_correct_v2"
    return "unknown"


def _measure_dir(input_dir: Path, fps: float = 20.0) -> dict:
    npz = np.load(input_dir / "generated.npz")
    with open(input_dir / "summary.json") as f:
        meta = json.load(f)
    seq_ids = meta["seq_ids"]
    seq_lens = meta.get("seq_lens") or npz["seq_lens"].tolist()
    seq_lens = [int(L) for L in seq_lens]

    motion_263 = npz["motion_263"]
    object_pc = npz["object_pc"]
    object_positions = npz["object_positions"]
    object_rotations = npz["object_rotations"]
    R_y = npz["world_R_y_angle"]
    T_xz = npz["world_T_xz"]

    rows = []
    for i, sid in enumerate(seq_ids):
        T = seq_lens[i]
        if T < 5:
            continue
        m = torch.from_numpy(motion_263[i, :T]).float().unsqueeze(0)
        canon = recover_from_ric(m, 22).squeeze(0).cpu().numpy().astype(np.float32)
        joints_world = _lift_canonical_to_world(canon, float(R_y[i]), T_xz[i])

        v11 = _v11_contact(
            joints_world, object_pc[i].astype(np.float32),
            object_positions[i, :T].astype(np.float32),
            object_rotations[i, :T].astype(np.float32),
            fps=fps,
        )
        v12 = _v12_strict_contact(
            joints_world, object_pc[i].astype(np.float32),
            object_positions[i, :T].astype(np.float32),
            object_rotations[i, :T].astype(np.float32),
            fps=fps,
        )

        v11_b = v11 > 0.5
        v12_b = v12 > 0.5
        rows.append({
            "seq_id": sid,
            "subset": _infer_subset(sid),
            "T": T,
            "v11_contact_frame_frac_any": float(v11_b.any(axis=1).mean()),
            "v12_contact_frame_frac_any": float(v12_b.any(axis=1).mean()),
            "v11_contact_frac_per_part": [float(v11_b[:, b].mean()) for b in range(NUM_BODY_PARTS)],
            "v12_contact_frac_per_part": [float(v12_b[:, b].mean()) for b in range(NUM_BODY_PARTS)],
            "v11_segments": _segment_stats(v11_b),
            "v12_segments": _segment_stats(v12_b),
            "v11_total_contact_frames": int(v11_b.any(axis=1).sum()),
            "v12_total_contact_frames": int(v12_b.any(axis=1).sum()),
        })

    return {"per_clip": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=20.0)
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    print(f"Measuring {args.input_dir} ...")
    result = _measure_dir(args.input_dir, fps=float(args.fps))
    rows = result["per_clip"]

    # Aggregate
    def _mean(rs, key):
        vs = [r[key] for r in rs]
        return float(np.mean(vs)) if vs else None

    agg = {
        "n_clips": len(rows),
        "v11_mean_frame_frac_any": _mean(rows, "v11_contact_frame_frac_any"),
        "v12_mean_frame_frac_any": _mean(rows, "v12_contact_frame_frac_any"),
        "v11_total_contact_frames": int(np.sum([r["v11_total_contact_frames"] for r in rows])),
        "v12_total_contact_frames": int(np.sum([r["v12_total_contact_frames"] for r in rows])),
        "v11_per_part_frac": [
            float(np.mean([r["v11_contact_frac_per_part"][b] for r in rows]))
            for b in range(NUM_BODY_PARTS)
        ],
        "v12_per_part_frac": [
            float(np.mean([r["v12_contact_frac_per_part"][b] for r in rows]))
            for b in range(NUM_BODY_PARTS)
        ],
        "v11_n_segments": int(np.sum([r["v11_segments"]["n_segments"] for r in rows])),
        "v12_n_segments": int(np.sum([r["v12_segments"]["n_segments"] for r in rows])),
        "v11_mean_seg_duration": float(np.mean([
            r["v11_segments"]["mean_segment_duration_frames"]
            for r in rows if r["v11_segments"]["n_segments"] > 0
        ])),
        "v12_mean_seg_duration": float(np.mean([
            r["v12_segments"]["mean_segment_duration_frames"]
            for r in rows if r["v12_segments"]["n_segments"] > 0
        ])),
    }

    # By subset
    by_subset = {}
    for sub in {r["subset"] for r in rows}:
        sr = [r for r in rows if r["subset"] == sub]
        by_subset[sub] = {
            "n_clips": len(sr),
            "v11_mean_frame_frac_any": _mean(sr, "v11_contact_frame_frac_any"),
            "v12_mean_frame_frac_any": _mean(sr, "v12_contact_frame_frac_any"),
            "v11_n_segments": int(np.sum([r["v11_segments"]["n_segments"] for r in sr])),
            "v12_n_segments": int(np.sum([r["v12_segments"]["n_segments"] for r in sr])),
        }

    summary = {
        "schema": "contact_definition_compare_v1",
        "approximation": "nearest_pc_distance (mesh approximation)",
        "v11_thresholds_m": dict(DEFAULT_DISTANCE_THRESHOLDS),
        "v12_thresholds_m": dict(STRICT_DISTANCE_THRESHOLDS),
        "aggregate": agg,
        "by_subset": by_subset,
        "per_clip": rows,
    }

    out = args.output_dir / "summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out}")
    print()
    print(f"=== Aggregate (n={agg['n_clips']} clips) ===")
    print(f"v11 mean contact frame frac (any part): {agg['v11_mean_frame_frac_any']*100:.2f}%")
    print(f"v12 mean contact frame frac (any part): {agg['v12_mean_frame_frac_any']*100:.2f}%")
    print(f"  reduction: {(1 - agg['v12_mean_frame_frac_any']/max(agg['v11_mean_frame_frac_any'],1e-6))*100:.1f}%")
    print(f"v11 total contact frames: {agg['v11_total_contact_frames']}")
    print(f"v12 total contact frames: {agg['v12_total_contact_frames']}")
    print(f"v11 #segments / mean dur: {agg['v11_n_segments']} / {agg['v11_mean_seg_duration']:.1f} frames")
    print(f"v12 #segments / mean dur: {agg['v12_n_segments']} / {agg['v12_mean_seg_duration']:.1f} frames")
    print()
    print("Per body part contact frac:")
    for b, name in enumerate(BODY_PART_NAMES):
        v11_f = agg["v11_per_part_frac"][b]
        v12_f = agg["v12_per_part_frac"][b]
        print(f"  {name:12} v11: {v11_f*100:6.2f}%   v12: {v12_f*100:6.2f}%   reduction: {(1-v12_f/max(v11_f,1e-6))*100:5.1f}%")
    print()
    print("By subset:")
    for sub, s in by_subset.items():
        print(f"  {sub:18} (n={s['n_clips']:2}): v11 {s['v11_mean_frame_frac_any']*100:5.2f}%  v12 {s['v12_mean_frame_frac_any']*100:5.2f}%   "
              f"#seg v11={s['v11_n_segments']:3} v12={s['v12_n_segments']:3}")


if __name__ == "__main__":
    main()
