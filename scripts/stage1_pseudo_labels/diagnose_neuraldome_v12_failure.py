"""Per-frame v12_strict pipeline trace on neuraldome/box vs omomo/plasticbox.

Replicates the inner loop of
``piano.data.pseudo_labels.extract_strict_contact.extract_strict_contact_state``
but logs every intermediate signal so we can see exactly which condition
is rejecting the neuraldome clips. Goal: identify which knob in v12_strict
is responsible for the 0% hand contact rate on neuraldome/box,
trolleycase, case (vs 40-45% on omomo's same-class plasticbox /
suitcase / largebox).

Per-frame logged signals (left_hand only, the most affected):
  - distance_m         — point-to-mesh distance, body in object-local
  - tight_dist_score   — soft sigmoid of distance vs strict 5 cm cap
  - loose_dist_score   — soft sigmoid vs loose 25 cm cap
  - kin_score          — kinematic engagement (rigid coupling) score
  - static_score       — static engagement (object stationary + body stable)
  - case_kinematic     — kin_score × loose_dist_score
  - case_static        — static_score × loose_dist_score
  - score              — max(case_kinematic, case_static)  (pre-smoothing)
  - obj_speed          — translation + radius × angular speed
  - body_local_std     — object-local frame std over kin_window (per-axis max)

Aggregated per-clip + per-subset summaries identify which gate is the
chokepoint:
  - If `loose_dist_score` is consistently 0 → distance threshold too tight
    or mesh / scale issue.
  - If `kin_score` is consistently 0 but distance is OK → kinematic
    engagement is failing (scale, fps, or kin_local_sigma).
  - If `static_score` is consistently 0 but object is stationary → static
    engagement filter mis-tuned.

Usage:
    python scripts/stage1_pseudo_labels/diagnose_neuraldome_v12_failure.py \\
        --output-dir analyses/2026-05-05_v12_neuraldome_diag

Reads:
    E:/Project/Datasets/InterAct/piano/{neuraldome,omomo_correct_v2}/...
    E:/Project/Datasets/InterAct/InterAct/{neuraldome,omomo_correct_v2}/objects/...
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from piano.data.pseudo_labels.extract_contact import (
    ContactConfig,
    _kinematic_contact_score,
)
from piano.data.pseudo_labels.extract_strict_contact import (
    BODY_PART_INDICES,
    BODY_PART_NAMES,
    LOOSE_DISTANCE_THRESHOLDS,
    NUM_BODY_PARTS,
    STRICT_DISTANCE_THRESHOLDS,
    StrictContactConfig,
    _soft_sigmoid,
    _static_engagement_score,
)
from piano.data.pseudo_labels._object_transform import world_to_object_local
from piano.data.pseudo_labels.run_all import _find_mesh
from piano.utils.geometry import load_mesh, points_to_mesh_distance
from piano.utils.io_utils import ensure_dir
from scipy.ndimage import uniform_filter1d


# Body parts to log: 0=l_hand. Could expand but l_hand is where we see
# the failure (v12 hand=0 in neuraldome/box).
BP_TO_LOG = 0


def _trace_one_clip(
    seq_id: str,
    motion_npz: Path,
    mesh,
    fps: float,
) -> dict:
    """Run the v12_strict signal stack and return per-frame arrays.

    Returns a dict with `T`, `seq_id`, plus per-frame arrays for the
    BP_TO_LOG body part. Returns None on failure.
    """
    if not motion_npz.exists():
        return {"seq_id": seq_id, "error": "motion_missing"}
    data = np.load(motion_npz, allow_pickle=False)
    if "joints_22" not in data.files:
        return {"seq_id": seq_id, "error": "no_joints_22"}
    joints = data["joints_22"].astype(np.float32)
    if "object_positions" not in data.files:
        return {"seq_id": seq_id, "error": "no_object_positions"}
    op = data["object_positions"].astype(np.float32)
    orot = data["object_rotations"].astype(np.float32) if "object_rotations" in data.files else np.zeros_like(op)
    T = min(len(joints), len(op))
    joints, op, orot = joints[:T], op[:T], orot[:T]

    # v12 config (matches extract_strict_contact_state internals)
    strict_cfg = StrictContactConfig()
    base_cfg = ContactConfig(
        kin_local_sigma=0.06,
        kin_local_transition=0.025,
        kin_world_eps=0.15,
        kin_world_sigma=0.04,
        kin_radius_proxy=0.3,
        kin_window_sec=0.5,
        fps=strict_cfg.fps,
    )

    # Object speed proxy
    trans_vel = np.zeros(T, dtype=np.float32)
    trans_vel[1:] = np.linalg.norm(np.diff(op, axis=0), axis=-1) * strict_cfg.fps
    ang_vel = np.zeros(T, dtype=np.float32)
    ang_vel[1:] = np.linalg.norm(np.diff(orot, axis=0), axis=-1) * strict_cfg.fps
    obj_speed = trans_vel + base_cfg.kin_radius_proxy * ang_vel

    kin_window = max(3, int(round(base_cfg.kin_window_sec * strict_cfg.fps)))

    # Single body part trace
    joint_idx = BODY_PART_INDICES[BP_TO_LOG]
    bp_name = BODY_PART_NAMES[BP_TO_LOG]
    bp_world = joints[:, joint_idx, :]
    bp_local = world_to_object_local(bp_world, op, orot)

    distances, _ = points_to_mesh_distance(bp_local, mesh)
    tight_thr = STRICT_DISTANCE_THRESHOLDS[bp_name]
    loose_thr = LOOSE_DISTANCE_THRESHOLDS[bp_name]
    tight_dist_score = _soft_sigmoid(distances, tight_thr, strict_cfg.distance_sigma)
    loose_dist_score = _soft_sigmoid(distances, loose_thr, strict_cfg.loose_distance_sigma)

    kin_score = _kinematic_contact_score(bp_world, op, orot, base_cfg)
    static_score = _static_engagement_score(
        bp_local, obj_speed,
        kin_window=kin_window,
        eps_mps=strict_cfg.static_engagement_eps_mps,
        local_std_thresh=strict_cfg.static_engagement_local_std_m,
    )

    case_kin = kin_score * loose_dist_score
    case_static = static_score * loose_dist_score
    score = np.maximum(case_kin, case_static)

    # Body-local std (the rigidity signal — input to kin_score)
    mean_x = uniform_filter1d(bp_local, size=kin_window, axis=0, mode="nearest")
    mean_x_sq = uniform_filter1d(bp_local ** 2, size=kin_window, axis=0, mode="nearest")
    rolling_var = np.maximum(mean_x_sq - mean_x ** 2, 0.0)
    body_local_std = np.sqrt(rolling_var + 1e-12).max(axis=-1)

    return {
        "seq_id": seq_id,
        "T": T,
        "bp_name": bp_name,
        "fps_assumed": strict_cfg.fps,
        "data_fps_likely": fps,
        "distance_m": distances,
        "tight_dist_score": tight_dist_score,
        "loose_dist_score": loose_dist_score,
        "kin_score": kin_score,
        "static_score": static_score,
        "case_kinematic": case_kin,
        "case_static": case_static,
        "score": score,
        "obj_speed": obj_speed,
        "body_local_std": body_local_std,
        "trans_vel": trans_vel,
        "ang_vel": ang_vel,
    }


def _summarize_subset(
    subset_label: str,
    traces: list[dict],
) -> dict:
    """Aggregate per-clip traces into a single summary table."""
    valid = [t for t in traces if "error" not in t]
    if not valid:
        return {"subset": subset_label, "n_valid_clips": 0}

    def _stack(field: str) -> np.ndarray:
        return np.concatenate([t[field] for t in valid])

    distances = _stack("distance_m")
    loose_score = _stack("loose_dist_score")
    tight_score = _stack("tight_dist_score")
    kin_score = _stack("kin_score")
    static_score = _stack("static_score")
    case_kin = _stack("case_kinematic")
    case_static = _stack("case_static")
    score = _stack("score")
    obj_speed = _stack("obj_speed")
    body_std = _stack("body_local_std")

    return {
        "subset": subset_label,
        "n_valid_clips": len(valid),
        "n_total_frames": int(len(distances)),
        "distance_m": {
            "mean": float(distances.mean()),
            "median": float(np.median(distances)),
            "p10": float(np.percentile(distances, 10)),
            "p90": float(np.percentile(distances, 90)),
        },
        "loose_dist_score (≥0.5 = within 25cm)": {
            "mean": float(loose_score.mean()),
            "frac_over_05": float((loose_score > 0.5).mean()),
        },
        "tight_dist_score (≥0.5 = within 5cm)": {
            "mean": float(tight_score.mean()),
            "frac_over_05": float((tight_score > 0.5).mean()),
        },
        "kin_score (≥0.5 = rigid coupling)": {
            "mean": float(kin_score.mean()),
            "frac_over_05": float((kin_score > 0.5).mean()),
        },
        "static_score (≥0.5 = obj stationary + body stable)": {
            "mean": float(static_score.mean()),
            "frac_over_05": float((static_score > 0.5).mean()),
        },
        "case_kinematic (gate × loose_dist)": {
            "mean": float(case_kin.mean()),
            "frac_over_05": float((case_kin > 0.5).mean()),
        },
        "case_static (gate × loose_dist)": {
            "mean": float(case_static.mean()),
            "frac_over_05": float((case_static > 0.5).mean()),
        },
        "score (pre-smoothing)": {
            "mean": float(score.mean()),
            "frac_over_05": float((score > 0.5).mean()),
        },
        "obj_speed_m_s": {
            "mean": float(obj_speed.mean()),
            "median": float(np.median(obj_speed)),
            "frac_below_0.05": float((obj_speed < 0.05).mean()),
        },
        "body_local_std_m": {
            "mean": float(body_std.mean()),
            "median": float(np.median(body_std)),
        },
    }


def _write_report(
    diag_neuraldome: dict,
    diag_omomo: dict,
    out_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# v12_strict diagnostic — neuraldome/box vs omomo/plasticbox")
    lines.append("")
    lines.append("Date: 2026-05-05. Body part traced: left_hand (joint 20).")
    lines.append("")
    lines.append(
        f"Neuraldome/box: {diag_neuraldome['n_valid_clips']} clips, "
        f"{diag_neuraldome.get('n_total_frames', 0)} frames"
    )
    lines.append(
        f"Omomo/plasticbox: {diag_omomo['n_valid_clips']} clips, "
        f"{diag_omomo.get('n_total_frames', 0)} frames"
    )
    lines.append("")

    keys_order = [
        "distance_m",
        "loose_dist_score (≥0.5 = within 25cm)",
        "tight_dist_score (≥0.5 = within 5cm)",
        "kin_score (≥0.5 = rigid coupling)",
        "static_score (≥0.5 = obj stationary + body stable)",
        "case_kinematic (gate × loose_dist)",
        "case_static (gate × loose_dist)",
        "score (pre-smoothing)",
        "obj_speed_m_s",
        "body_local_std_m",
    ]

    lines.append("## Side-by-side comparison")
    lines.append("")
    lines.append("| signal | metric | neuraldome/box | omomo/plasticbox | ratio (n/o) |")
    lines.append("|---|---|---:|---:|---:|")
    for k in keys_order:
        if k not in diag_neuraldome:
            continue
        for sub_metric, n_val in diag_neuraldome[k].items():
            o_val = diag_omomo[k].get(sub_metric)
            if isinstance(n_val, float) and isinstance(o_val, float):
                if abs(o_val) > 1e-9:
                    ratio = f"{n_val/o_val:.2f}"
                else:
                    ratio = "—" if abs(n_val) < 1e-9 else "∞"
                lines.append(
                    f"| {k} | {sub_metric} | {n_val:.4f} | {o_val:.4f} | {ratio} |"
                )
    lines.append("")
    lines.append("## How to read this")
    lines.append("")
    lines.append(
        "- `distance_m` = body-part distance to mesh (object-local frame). "
        "Comparable values would mean both subsets put hand at similar "
        "physical proximity to the object surface."
    )
    lines.append(
        "- `loose_dist_score` is the cap that gates *both* case_kinematic "
        "and case_static; if this is far lower for neuraldome, that means "
        "the wrist is consistently >25cm from the mesh — i.e., the mesh "
        "is too small or the body-to-object distance is genuinely larger."
    )
    lines.append(
        "- `kin_score` measures rigid coupling between body and object in "
        "object-local frame. If this is far lower for neuraldome, the "
        "kinematic engagement test is rejecting the clips. Combined with "
        "`body_local_std` will tell us if the issue is the data (wrist "
        "actually drifts more in neuraldome) or the threshold."
    )
    lines.append(
        "- `static_score` requires object stationary AND body stable. "
        "`obj_speed` near 0 is needed for the static path."
    )
    lines.append("")
    lines.append("## Files")
    lines.append("- `report.md` — this file")
    lines.append("- `traces.npz` — per-clip per-frame arrays (debugging)")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path,
                        default=Path("E:/Project/Datasets/InterAct/piano"))
    parser.add_argument("--mesh-root", type=Path,
                        default=Path("E:/Project/Datasets/InterAct/InterAct"))
    parser.add_argument("--n-clips", type=int, default=5)
    parser.add_argument("--data-fps", type=float, default=20.0,
                        help="actual fps of preprocessed data (used for context only — "
                             "v12_strict pipeline assumes fps=30 by default; this flag "
                             "lets us flag whether that mismatch matters).")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    out = ensure_dir(args.output_dir)

    targets = [
        ("neuraldome/box", "neuraldome", "box",
         args.data_root / "neuraldome", args.mesh_root / "neuraldome" / "objects"),
        ("omomo_correct_v2/plasticbox", "omomo_correct_v2", "plasticbox",
         args.data_root / "omomo_correct_v2",
         args.mesh_root / "omomo_correct_v2" / "objects"),
    ]
    mesh_suffixes = ("_face1000", "_simplified", "")

    summaries: dict[str, dict] = {}
    all_traces: dict[str, list] = {}

    for label, subset, kw, data_dir, mesh_dir in targets:
        meta_path = data_dir / "metadata.json"
        with open(meta_path, encoding="utf-8") as f:
            metadata = json.load(f)

        # Pick first n_clips that match the keyword
        sel = []
        for m in metadata:
            sid = m.get("seq_id", "")
            parts = sid.lower().split("_")
            if kw in parts:
                sel.append(m)
            if len(sel) >= args.n_clips:
                break

        print(f"\n=== {label} ({len(sel)} clips) ===")
        # Load mesh once (object_id is consistent across box clips here)
        mesh_cache: dict[str, object] = {}
        traces: list[dict] = []
        for m in sel:
            sid = m["seq_id"]
            obj_id = m.get("object_id", kw)
            if obj_id not in mesh_cache:
                mp = _find_mesh(mesh_dir, obj_id, mesh_suffixes)
                if mp is None:
                    print(f"  [skip] no mesh for {obj_id}")
                    continue
                mesh_cache[obj_id] = load_mesh(str(mp))
                v = mesh_cache[obj_id].vertices.shape[0]
                bb = mesh_cache[obj_id].bounding_box.extents
                print(f"  mesh {mp.name}: {v} verts, bbox {bb}")
            mesh = mesh_cache[obj_id]
            t = _trace_one_clip(
                sid, data_dir / "motions" / f"{sid}.npz", mesh, args.data_fps,
            )
            if "error" in t:
                print(f"  [skip] {sid}: {t['error']}")
                continue
            print(
                f"  {sid} (T={t['T']}): "
                f"dist mean={t['distance_m'].mean():.3f}m, "
                f"loose={t['loose_dist_score'].mean():.3f}, "
                f"kin={t['kin_score'].mean():.3f}, "
                f"static={t['static_score'].mean():.3f}, "
                f"score={t['score'].mean():.3f}, "
                f"frac>0.5={(t['score'] > 0.5).mean():.3f}"
            )
            traces.append(t)
        summaries[label] = _summarize_subset(label, traces)
        all_traces[label] = traces

    # Write summaries
    summary_path = out / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    print(f"\nWrote: {summary_path}")

    # Write report
    report_path = out / "report.md"
    keys = list(summaries.keys())
    if len(keys) >= 2:
        _write_report(summaries[keys[0]], summaries[keys[1]], report_path)
        print(f"Wrote: {report_path}")

    # Save raw traces for deeper inspection
    np_save_data: dict[str, np.ndarray] = {}
    for label, traces in all_traces.items():
        for t in traces:
            for k, v in t.items():
                if isinstance(v, np.ndarray):
                    np_save_data[f"{label.replace('/', '_')}__{t['seq_id']}__{k}"] = v
    np.savez(out / "traces.npz", **np_save_data)
    print(f"Wrote: {out / 'traces.npz'}")


if __name__ == "__main__":
    main()
