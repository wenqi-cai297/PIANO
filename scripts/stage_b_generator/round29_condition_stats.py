"""Round-29 condition statistics tool (prompt §7.4).

Walks a list of clip npz files and reports, for every condition family
variant, the per-family validity rates / amplitude distributions /
active-frame fractions. Used to catch degenerate conditions BEFORE they
hit the trainer (e.g. all-zero phase, always-on masks).

Usage:
    python scripts/stage_b_generator/round29_condition_stats.py \
        --dataset-root E:/Project/Datasets/InterAct/piano_official_process_4/omomo_correct_v2 \
        --num-clips 32 \
        --output analyses/round29_condition_stats.json

Reviewed prompt section 7.4 before implementing this group: yes.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from piano.data.stage2_oracle_conditions import (  # noqa: E402
    BODY_VARIANT_DIMS,
    COARSE_VARIANT_DIMS,
    INTERACTION_VARIANT_DIMS,
    SUPPORT_VARIANT_DIMS,
    build_body_refinement_condition,
    build_coarse_condition,
    build_interaction_condition,
    build_support_condition,
)


def _load_clip(npz_path: Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    data = np.load(npz_path, allow_pickle=False)
    joints = data["joints_22"].astype(np.float32)
    object_positions = (
        data["object_positions"].astype(np.float32)
        if "object_positions" in data.files else None
    )
    object_rotations = (
        data["object_rotations"].astype(np.float32)
        if "object_rotations" in data.files else None
    )
    return joints, object_positions, object_rotations


def _load_contact(label_path: Path) -> np.ndarray | None:
    if not label_path.exists():
        return None
    data = np.load(label_path, allow_pickle=False)
    if "contact_state" in data.files:
        return data["contact_state"].astype(np.float32)
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root", required=True,
        help="Dataset root containing motions/*.npz and pseudo_labels/<subdir>/*.npz",
    )
    parser.add_argument(
        "--pseudo-subdir",
        default="pseudo_labels/v18_h10_f05_pelvis20_official_semantic_marker",
    )
    parser.add_argument("--num-clips", type=int, default=32)
    parser.add_argument("--output", default="analyses/round29_condition_stats.json")
    parser.add_argument("--max-frames", type=int, default=196)
    args = parser.parse_args()

    root = Path(args.dataset_root)
    motions = sorted((root / "motions").glob("*.npz"))[: args.num_clips]
    if not motions:
        print(f"[R29-stats] no motion npz files under {root / 'motions'}", file=sys.stderr)
        return 2

    agg: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    clip_count = 0

    for motion_path in motions:
        seq_id = motion_path.stem
        try:
            joints, obj_pos, obj_rot = _load_clip(motion_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[R29-stats] skip {seq_id}: {exc}")
            continue
        T = int(min(joints.shape[0], args.max_frames))
        joints = joints[:T]
        if obj_pos is not None:
            obj_pos = obj_pos[:T]
        if obj_rot is not None:
            obj_rot = obj_rot[:T]
        contact = _load_contact(root / args.pseudo_subdir / f"{seq_id}.npz")
        if contact is not None:
            contact = contact[:T]
        clip_count += 1

        # Coarse
        for cv in COARSE_VARIANT_DIMS:
            arr, info = build_coarse_condition(joints, cv)
            agg["coarse"][f"{cv}/finite_frac"].append(info["finite_frac"])
            agg["coarse"][f"{cv}/max_abs"].append(info["max_abs"])
            agg["coarse"][f"{cv}/std"].append(info["std"])

        # Interaction (only when object + contact present)
        if obj_pos is not None and obj_rot is not None and contact is not None:
            for iv in INTERACTION_VARIANT_DIMS:
                try:
                    arr, info = build_interaction_condition(
                        joints, obj_pos, obj_rot, contact, variant=iv,
                    )
                except Exception as exc:  # noqa: BLE001
                    agg["interaction"][f"{iv}/ERROR"].append(str(exc))
                    continue
                agg["interaction"][f"{iv}/finite_frac"].append(info["finite_frac"])
                agg["interaction"][f"{iv}/contact_frame_frac"].append(
                    info.get("contact_frame_frac", 0.0),
                )

        # Support
        for sv in SUPPORT_VARIANT_DIMS:
            arr, info = build_support_condition(joints, variant=sv)
            agg["support"][f"{sv}/finite_frac"].append(info["finite_frac"])
            agg["support"][f"{sv}/phase_valid_frame_frac"].append(
                info.get("phase_valid_frame_frac", 1.0),
            )
            agg["support"][f"{sv}/footstep_target_valid_frame_frac"].append(
                info.get("footstep_target_valid_frame_frac", 1.0),
            )
            agg["support"][f"{sv}/walking_frame_frac"].append(
                info.get("walking_frame_frac", 0.0),
            )

        # Body
        for bv in BODY_VARIANT_DIMS:
            arr, info = build_body_refinement_condition(joints, variant=bv)
            agg["body_refine"][f"{bv}/finite_frac"].append(info["finite_frac"])
            agg["body_refine"][f"{bv}/active_joint_frac"].append(
                info.get("active_joint_frac", 0.0),
            )

    def _summarize(values: list[float]) -> dict[str, float] | dict[str, str]:
        if not values:
            return {"n": 0}
        if isinstance(values[0], str):
            return {"errors": values[:5]}
        arr = np.array(values, dtype=np.float64)
        return {
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "p50": float(np.median(arr)),
        }

    summary: dict[str, dict[str, dict[str, float]]] = {}
    for family, kvs in agg.items():
        summary[family] = {k: _summarize(v) for k, v in kvs.items()}

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {"num_clips": clip_count, "dataset_root": str(root), "stats": summary},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[R29-stats] wrote {out_path}  (clips={clip_count})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
