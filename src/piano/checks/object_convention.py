"""Sanity-check object pose convention + contact extraction geometry.

Given one preprocessed sequence, this script:

1. Loads joints_22 (world frame) + object_positions + object_rotations
2. Loads the object mesh (object-local frame)
3. Transforms the mesh into world frame per frame:
       world_verts[t] = verts_local @ R(obj_rot[t]).T + obj_trans[t]
4. For a few test frames (first, middle, last), computes:
     - joint-to-world-mesh distance (correct, just for verification)
     - joint-to-local-mesh distance after inverse-transforming the joint
       (what our new extract_contact does)
5. Reports: the two distance series should be ~IDENTICAL. If they are,
   the convention ``world = R @ local + t`` is correct AND our
   inverse-transform in extract_contact is correct.
6. Also reports min hand-to-object distance across the sequence — gives
   an intuition for whether contact will fire.

Usage:
    piano-check-object-convention \\
        --data-dir /media/.../InterAct/piano/chairs \\
        --mesh-dir /media/.../InterAct/InterAct/chairs/objects \\
        [--seq-id Sub0001_Obj116_Seg0_0]
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from piano.data.pseudo_labels._object_transform import (
    axis_angle_to_rotmat,
    world_to_object_local,
)
from piano.data.pseudo_labels.run_all import _find_mesh
from piano.utils.geometry import load_mesh, points_to_mesh_distance
from piano.utils.io_utils import ensure_dir, load_json, save_json
from piano.utils.smpl_utils import BODY_PART_INDICES, BODY_PART_NAMES


def _transform_mesh_vertices(
    verts_local: np.ndarray,
    trans: np.ndarray,
    rot_aa: np.ndarray,
) -> np.ndarray:
    """world = R(aa) @ local + t  (single frame)."""
    R = axis_angle_to_rotmat(rot_aa.astype(np.float32))
    return verts_local @ R.T + trans


def run_check(
    data_dir: Path,
    mesh_dir: Path,
    seq_id: str | None,
    output_dir: Path,
) -> None:
    import trimesh

    output_dir = ensure_dir(output_dir)

    metadata = load_json(data_dir / "metadata.json")
    if seq_id is None:
        # Pick the first sequence that has all required fields
        chosen = None
        for m in metadata:
            sid = m["seq_id"]
            motion_path = data_dir / "motions" / f"{sid}.npz"
            if not motion_path.exists():
                continue
            data = np.load(motion_path)
            if "object_rotations" not in data.files:
                continue
            chosen = m
            break
        if chosen is None:
            raise RuntimeError("No sequence with object_rotations found. "
                               "Re-run preprocess_interact first.")
        seq_id = chosen["seq_id"]
        obj_id = chosen["object_id"]
    else:
        obj_id = next(m["object_id"] for m in metadata if m["seq_id"] == seq_id)

    print(f"Checking sequence: {seq_id} (object: {obj_id})")

    motion_data = np.load(data_dir / "motions" / f"{seq_id}.npz")
    joints = motion_data["joints_22"]                   # (T, 22, 3) world
    obj_pos = motion_data["object_positions"]           # (T, 3) world
    if "object_rotations" not in motion_data.files:
        raise RuntimeError(
            f"{seq_id} has no object_rotations. Re-run preprocess_interact."
        )
    obj_rot = motion_data["object_rotations"]           # (T, 3) axis-angle

    mesh_path = _find_mesh(mesh_dir, obj_id, ("_face1000", "_simplified", ""))
    if mesh_path is None:
        raise FileNotFoundError(f"Mesh for {obj_id} not found in {mesh_dir}")
    mesh = load_mesh(str(mesh_path))
    print(f"Loaded mesh: {mesh_path.name} ({len(mesh.vertices)} verts, "
          f"{len(mesh.faces)} faces)")

    T = len(joints)
    test_frames = [0, T // 2, T - 1]

    # For each test frame and each tracked body part, compute:
    #   d_world = distance from joint (world) to mesh transformed into world
    #   d_local = distance from inverse-transformed joint to the static local mesh
    # They should match (to floating-point precision).
    report: dict = {
        "seq_id": seq_id,
        "object_id": obj_id,
        "mesh_path": str(mesh_path),
        "num_frames": T,
        "frames_checked": test_frames,
        "per_frame": {},
    }

    max_discrepancy = 0.0
    min_hand_dist_local = np.inf

    verts_local = np.asarray(mesh.vertices, dtype=np.float32)

    # Also aggregate across ALL frames: compute min hand distance for each
    # body part using the local-frame method (the real one)
    hand_indices = [BODY_PART_INDICES[0], BODY_PART_INDICES[1]]  # left+right hand

    for t in range(T):
        hand_world = joints[t, hand_indices, :]     # (2, 3)
        hand_local = world_to_object_local(
            hand_world, obj_pos[t:t+1].repeat(2, axis=0), obj_rot[t:t+1].repeat(2, axis=0),
        )
        d_local, _ = points_to_mesh_distance(hand_local, mesh)
        min_hand_dist_local = min(min_hand_dist_local, float(d_local.min()))

    print(f"\nMin hand-to-object distance across all {T} frames: "
          f"{min_hand_dist_local*100:.1f} cm")

    for t in test_frames:
        # Joints at this frame (world)
        bp_world = joints[t, BODY_PART_INDICES, :]    # (5, 3)

        # Method 1: transform mesh to world, query in world
        mesh_verts_world = _transform_mesh_vertices(verts_local, obj_pos[t], obj_rot[t])
        mesh_world = trimesh.Trimesh(vertices=mesh_verts_world, faces=mesh.faces)
        d_world, _ = points_to_mesh_distance(bp_world, mesh_world)

        # Method 2: transform joints to local, query in local
        bp_local = world_to_object_local(
            bp_world,
            np.tile(obj_pos[t], (5, 1)),
            np.tile(obj_rot[t], (5, 1)),
        )
        d_local, _ = points_to_mesh_distance(bp_local, mesh)

        diff = np.abs(d_world - d_local).max()
        max_discrepancy = max(max_discrepancy, float(diff))

        print(f"\nFrame {t}:")
        print(f"  obj_trans = {obj_pos[t]}, obj_rot = {obj_rot[t]}")
        print(f"  {'body_part':15s} {'d_world (cm)':>14s} {'d_local (cm)':>14s} {'diff (mm)':>12s}")
        for bp_idx, name in enumerate(BODY_PART_NAMES):
            print(f"  {name:15s} {d_world[bp_idx]*100:>14.2f} {d_local[bp_idx]*100:>14.2f} "
                  f"{(d_world[bp_idx]-d_local[bp_idx])*1000:>12.3f}")

        report["per_frame"][int(t)] = {
            "obj_trans": obj_pos[t].tolist(),
            "obj_rot_aa": obj_rot[t].tolist(),
            "body_parts": {
                name: {
                    "d_world_m": float(d_world[bp_idx]),
                    "d_local_m": float(d_local[bp_idx]),
                }
                for bp_idx, name in enumerate(BODY_PART_NAMES)
            },
            "max_discrepancy_m": float(diff),
        }

    print(f"\nMax discrepancy across all frames checked: {max_discrepancy*1000:.3f} mm")
    if max_discrepancy < 1e-3:  # < 1mm
        print("CONVENTION CHECK: PASS ✓  (world=R@local+t matches our inverse-transform)")
        verdict = "pass"
    else:
        print("CONVENTION CHECK: FAIL ✗  (large discrepancy — convention may be different)")
        verdict = "fail"

    report["verdict"] = verdict
    report["max_discrepancy_m"] = max_discrepancy
    report["min_hand_distance_local_m"] = min_hand_dist_local
    report["timestamp"] = datetime.now().isoformat()
    save_json(output_dir / "summary.json", report)

    print(f"\nReport: {output_dir / 'summary.json'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--mesh-dir", type=Path, required=True)
    parser.add_argument("--seq-id", type=str, default=None,
                        help="Specific sequence to check (default: first with "
                             "object_rotations).")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.output_dir is None:
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        output_dir = Path("runs/checks/object_convention") / ts
    else:
        output_dir = args.output_dir
    run_check(args.data_dir, args.mesh_dir, args.seq_id, output_dir)


if __name__ == "__main__":
    main()
