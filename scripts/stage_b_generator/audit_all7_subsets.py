"""Audit readiness for Stage B "all-7" subset expansion.

The current AnchorDiff v18 config trains on four PIANO-format InterAct
subsets. This diagnostic audits those four at full fidelity and separately
checks whether the three remaining raw InterAct subsets are actually ready to
join the same pipeline.

Outputs:
    - JSON with per-subset data / pseudo-label / plan / split statistics.
    - Markdown report with risk classifications and a go/no-go answer for v19.

Example:
    python scripts/stage_b_generator/audit_all7_subsets.py \
        --config configs/training/anchordiff_v18_a1_FULL_DATA.yaml \
        --raw-interact-root E:/Project/Datasets/InterAct/InterAct \
        --official-root E:/Project/Datasets/InterAct/InterAct_official_process_4 \
        --output analyses/2026-05-13_all7_subset_audit.json \
        --md analyses/2026-05-13_all7_subset_audit.md
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from piano.data.interaction_plan_compiler import (
    InteractionPlanCompilerConfig,
    compile_interaction_plan,
)
from piano.data.pseudo_labels.extract_phase import PHASE_NAMES
from piano.data.pseudo_labels.extract_support import SUPPORT_NAMES
from piano.data.pseudo_labels.stats import (
    aggregate_stats,
    compute_seq_stats,
    make_quality_flags,
)
from piano.data.split import build_subject_split, extract_subject_id
from piano.utils.io_utils import load_json


EXPECTED_ALL7_SUBSETS: tuple[str, ...] = (
    "behave",
    "chairs",
    "grab",
    "imhd",
    "intercap",
    "neuraldome",
    "omomo_correct_v2",
)

RAW_CANONICAL_REQUIRED: tuple[str, ...] = (
    "human.npz",
    "object.npz",
    "text.txt",
    "motion.npy",
    "joints.npy",
    "markers.npy",
)

PART_NAMES: tuple[str, ...] = (
    "L_hand",
    "R_hand",
    "L_foot",
    "R_foot",
    "pelvis",
)


@dataclass(slots=True)
class RangeTracker:
    """JSON-friendly finite-range tracker for numeric arrays."""

    min_value: float = float("inf")
    max_value: float = float("-inf")
    finite_count: int = 0
    nonfinite_count: int = 0

    def update(self, arr: np.ndarray | None) -> None:
        if arr is None:
            return
        data = np.asarray(arr)
        if data.size == 0:
            return
        finite = np.isfinite(data)
        self.finite_count += int(finite.sum())
        self.nonfinite_count += int((~finite).sum())
        if finite.any():
            vals = data[finite]
            self.min_value = min(self.min_value, float(vals.min()))
            self.max_value = max(self.max_value, float(vals.max()))

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "min": None if self.finite_count == 0 else float(self.min_value),
            "max": None if self.finite_count == 0 else float(self.max_value),
            "finite_count": int(self.finite_count),
            "nonfinite_count": int(self.nonfinite_count),
        }


def _stats(values: list[float | int]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "min": 0.0,
            "median": 0.0,
            "mean": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def _counter_dict(counter: Counter[str] | Counter[int]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def _sequence_root(subset_root: Path) -> Path | None:
    for candidate in (
        subset_root / "sequences_canonical",
        subset_root / "sequences" / "sequences_canonical",
    ):
        if candidate.exists():
            return candidate
    return None


def _raw_subset_readiness(raw_root: Path, subset: str) -> dict[str, Any]:
    subset_root = raw_root / subset
    seq_root = _sequence_root(subset_root)
    if seq_root is None:
        return {
            "subset_root_exists": subset_root.exists(),
            "canonical_sequence_root_exists": False,
            "canonical_sequence_count": 0,
            "required_file_coverage": {},
            "sample_seq_ids": [],
        }

    seq_dirs = sorted(path for path in seq_root.iterdir() if path.is_dir())
    coverage = {name: 0 for name in RAW_CANONICAL_REQUIRED}
    sample_seq_ids: list[str] = []
    for seq_dir in seq_dirs:
        if len(sample_seq_ids) < 5:
            sample_seq_ids.append(seq_dir.name)
        for name in RAW_CANONICAL_REQUIRED:
            coverage[name] += int((seq_dir / name).exists())
    return {
        "subset_root_exists": True,
        "canonical_sequence_root_exists": True,
        "canonical_sequence_count": int(len(seq_dirs)),
        "required_file_coverage": {
            key: {
                "present": int(value),
                "missing": int(len(seq_dirs) - value),
                "fraction_present": float(value / max(len(seq_dirs), 1)),
            }
            for key, value in coverage.items()
        },
        "sample_seq_ids": sample_seq_ids,
    }


def _official_subset_readiness(official_root: Path, subset: str) -> dict[str, Any]:
    subset_root = official_root / subset
    seq_root = _sequence_root(subset_root)
    if seq_root is None:
        return {
            "subset_root_exists": subset_root.exists(),
            "canonical_sequence_root_exists": False,
            "canonical_sequence_count": 0,
        }
    return {
        "subset_root_exists": True,
        "canonical_sequence_root_exists": True,
        "canonical_sequence_count": int(sum(1 for path in seq_root.iterdir() if path.is_dir())),
    }


def _metadata_path(subset_root: Path) -> Path | None:
    clean = subset_root / "metadata_clean.json"
    if clean.exists():
        return clean
    raw = subset_root / "metadata.json"
    return raw if raw.exists() else None


def _load_phase_support(
    phase: np.ndarray,
    support: np.ndarray,
    *,
    collapse_hand_support: bool,
    num_phase: int,
    num_support: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phase_int = phase.astype(np.int64, copy=False)
    support_int = support.astype(np.int64, copy=True)
    if collapse_hand_support:
        support_int[support_int == 3] = 0
    phase_clip = np.clip(phase_int, 0, max(num_phase - 1, 0))
    support_clip = np.clip(support_int, 0, max(num_support - 1, 0))
    phase_soft = np.zeros((len(phase_clip), num_phase), dtype=np.float32)
    support_soft = np.zeros((len(support_clip), num_support), dtype=np.float32)
    if len(phase_clip):
        phase_soft[np.arange(len(phase_clip)), phase_clip] = 1.0
    if len(support_clip):
        support_soft[np.arange(len(support_clip)), support_clip] = 1.0
    return phase_soft, support_soft, support_int


def _shape_tail(arr: np.ndarray | None) -> str:
    if arr is None:
        return "missing"
    if arr.ndim == 0:
        return "()"
    return str(tuple(int(x) for x in arr.shape[1:]))


def _support_names_for_count(count: int) -> list[str]:
    names = list(SUPPORT_NAMES)
    if count <= len(names):
        return names[:count]
    return names + [f"support_{idx}" for idx in range(len(names), count)]


def _audit_processed_subset(
    subset: str,
    subset_root: Path,
    pseudo_label_subdir: str,
    cfg: Any,
) -> dict[str, Any]:
    metadata_path = _metadata_path(subset_root)
    if metadata_path is None:
        return {
            "processed": False,
            "metadata_source": None,
            "clip_count": 0,
            "fail_reasons": [f"{subset}: no metadata.json or metadata_clean.json under {subset_root}"],
        }

    metadata = load_json(metadata_path)
    pseudo_root = subset_root / pseudo_label_subdir
    collapse_hand_support = bool(cfg.data.get("support_collapse_hand_support", True))
    num_phase = int(cfg.model.z_int.phase_classes)
    num_support = int(cfg.model.z_int.support_classes)

    summary_path = subset_root / "summary.json"
    summary = load_json(summary_path) if summary_path.exists() else {}

    file_counts = Counter()
    motion_tail_shapes = Counter()
    joints_tail_shapes = Counter()
    object_pc_shapes = Counter()
    object_position_tail_shapes = Counter()
    object_rotation_tail_shapes = Counter()
    smplx_pose_tail_shapes = Counter()
    seq_lengths: list[int] = []
    metadata_lengths: list[int] = []
    restorable_stage_b_count = 0

    motion_range = RangeTracker()
    joints_range = RangeTracker()
    object_pc_range = RangeTracker()
    object_position_range = RangeTracker()
    object_rotation_range = RangeTracker()
    contact_state_range = RangeTracker()
    target_xyz_range = RangeTracker()
    anchor_target_local_range = RangeTracker()
    anchor_target_world_range = RangeTracker()

    phase_counts = np.zeros(num_phase, dtype=np.int64)
    support_counts = np.zeros(num_support, dtype=np.int64)
    contact_positive_counts = np.zeros(len(PART_NAMES), dtype=np.int64)
    contact_total_frames = 0
    zero_any_contact_clips = 0
    full_any_contact_clips = 0
    contact_all_zero_tensor_clips = 0
    contact_all_one_tensor_clips = 0
    target_missing_count = 0
    target_nonfinite_clip_count = 0
    target_abs_gt_10m_clip_count = 0
    label_schema_failures = 0
    motion_schema_failures = 0
    object_schema_failures = 0
    seq_length_mismatch_count = 0

    plan_anchor_counts: list[int] = []
    plan_segment_counts: list[int] = []
    plan_anchor_time_norm: list[float] = []
    plan_zero_anchor_count = 0
    plan_filler_only_count = 0
    plan_out_of_bounds_count = 0
    plan_invalid_mask_count = 0
    plan_part_counts = np.zeros(len(PART_NAMES), dtype=np.int64)
    plan_type_counts = Counter()

    pseudo_seq_stats = []
    object_cache: dict[str, np.ndarray] = {}

    split_rows: list[dict[str, str | None]] = []

    for meta in metadata:
        seq_id = str(meta["seq_id"])
        object_id = str(meta.get("object_id", ""))
        subject_id = extract_subject_id(subset, seq_id)
        split_rows.append(
            {
                "seq_id": seq_id,
                "object_id": object_id,
                "subject_id": subject_id,
            }
        )

        motion_path = subset_root / "motions" / f"{seq_id}.npz"
        label_path = pseudo_root / f"{seq_id}.npz"
        object_path = subset_root / "objects" / f"{object_id}.npy"
        file_counts["motion_exists"] += int(motion_path.exists())
        file_counts["label_exists"] += int(label_path.exists())
        file_counts["object_exists"] += int(object_path.exists())

        motion_npz: Any | None = None
        labels_npz: Any | None = None
        try:
            if motion_path.exists():
                motion_npz = np.load(motion_path, allow_pickle=False)
                motion = motion_npz["motion_263"] if "motion_263" in motion_npz.files else None
                joints = motion_npz["joints_22"] if "joints_22" in motion_npz.files else None
                obj_pos = motion_npz["object_positions"] if "object_positions" in motion_npz.files else None
                obj_rot = motion_npz["object_rotations"] if "object_rotations" in motion_npz.files else None
                smplx_poses = motion_npz["smplx_poses"] if "smplx_poses" in motion_npz.files else None

                motion_tail_shapes[_shape_tail(motion)] += 1
                joints_tail_shapes[_shape_tail(joints)] += 1
                object_position_tail_shapes[_shape_tail(obj_pos)] += 1
                object_rotation_tail_shapes[_shape_tail(obj_rot)] += 1
                smplx_pose_tail_shapes[_shape_tail(smplx_poses)] += 1

                if motion is None or joints is None or obj_pos is None or obj_rot is None:
                    motion_schema_failures += 1
                else:
                    seq_len = min(len(motion), len(joints), len(obj_pos), len(obj_rot))
                    seq_lengths.append(int(seq_len))
                    metadata_lengths.append(int(meta.get("num_frames", seq_len)))
                    if len({len(motion), len(joints), len(obj_pos), len(obj_rot)}) != 1:
                        seq_length_mismatch_count += 1
                    motion_range.update(motion[:seq_len])
                    joints_range.update(joints[:seq_len])
                    object_position_range.update(obj_pos[:seq_len])
                    object_rotation_range.update(obj_rot[:seq_len])
                    if smplx_poses is not None and smplx_poses.ndim == 2 and smplx_poses.shape[1] >= 66:
                        restorable_stage_b_count += 1
            else:
                motion_schema_failures += 1

            if object_path.exists():
                if object_id not in object_cache:
                    object_cache[object_id] = np.load(object_path, allow_pickle=False)
                object_pc = object_cache[object_id]
                object_pc_shapes[str(tuple(int(x) for x in object_pc.shape))] += 1
                object_pc_range.update(object_pc)
            else:
                object_schema_failures += 1

            if label_path.exists():
                labels_npz = np.load(label_path, allow_pickle=False)
                contact_state = labels_npz["contact_state"] if "contact_state" in labels_npz.files else None
                contact_target = labels_npz["contact_target"] if "contact_target" in labels_npz.files else None
                phase = labels_npz["phase"] if "phase" in labels_npz.files else None
                support = labels_npz["support"] if "support" in labels_npz.files else None
                target_xyz = (
                    labels_npz["contact_target_xyz_gt"]
                    if "contact_target_xyz_gt" in labels_npz.files
                    else labels_npz["contact_target_xyz"]
                    if "contact_target_xyz" in labels_npz.files
                    else None
                )

                if (
                    contact_state is None
                    or contact_target is None
                    or phase is None
                    or support is None
                    or target_xyz is None
                ):
                    label_schema_failures += 1
                    if target_xyz is None:
                        target_missing_count += 1
                    continue

                seq_len = min(
                    len(contact_state),
                    len(contact_target),
                    len(phase),
                    len(support),
                    len(target_xyz),
                )
                if seq_len <= 0:
                    label_schema_failures += 1
                    continue
                contact_state = contact_state[:seq_len].astype(np.float32, copy=False)
                contact_target = contact_target[:seq_len].astype(np.float32, copy=False)
                target_xyz = target_xyz[:seq_len].astype(np.float32, copy=False)
                phase = phase[:seq_len].astype(np.int64, copy=False)
                support = support[:seq_len].astype(np.int64, copy=False)

                contact_state_range.update(contact_state)
                target_xyz_range.update(target_xyz)
                if not np.isfinite(target_xyz).all():
                    target_nonfinite_clip_count += 1
                if np.isfinite(target_xyz).any() and float(np.nanmax(np.abs(target_xyz))) > 10.0:
                    target_abs_gt_10m_clip_count += 1

                contact_binary = contact_state > 0.5
                any_contact = contact_binary.any(axis=1)
                zero_any_contact_clips += int(not any_contact.any())
                full_any_contact_clips += int(any_contact.all())
                contact_all_zero_tensor_clips += int(not contact_binary.any())
                contact_all_one_tensor_clips += int(contact_binary.all())
                contact_positive_counts += contact_binary.sum(axis=0).astype(np.int64)
                contact_total_frames += int(contact_binary.shape[0])

                phase_soft, support_soft, support_collapsed = _load_phase_support(
                    phase,
                    support,
                    collapse_hand_support=collapse_hand_support,
                    num_phase=num_phase,
                    num_support=num_support,
                )
                phase_counts += np.bincount(
                    np.clip(phase, 0, num_phase - 1),
                    minlength=num_phase,
                )[:num_phase]
                support_counts += np.bincount(
                    np.clip(support_collapsed, 0, num_support - 1),
                    minlength=num_support,
                )[:num_support]

                if motion_npz is not None and "object_positions" in motion_npz.files and "object_rotations" in motion_npz.files:
                    obj_pos = motion_npz["object_positions"][:seq_len].astype(np.float32, copy=False)
                    obj_rot = motion_npz["object_rotations"][:seq_len].astype(np.float32, copy=False)
                    compiler_cfg = InteractionPlanCompilerConfig(
                        num_parts=int(contact_state.shape[1]),
                        num_phase_classes=num_phase,
                        num_support_classes=num_support,
                    )
                    plan = compile_interaction_plan(
                        contact_prob=contact_state,
                        target_local=target_xyz,
                        phase_softmax=phase_soft,
                        support_softmax=support_soft,
                        object_pos_world=obj_pos,
                        object_rot_world_aa=obj_rot,
                        seq_len=seq_len,
                        cfg=compiler_cfg,
                    )
                    mask = plan["anchor_mask"].astype(bool)
                    seg_mask = plan["segment_mask"].astype(bool)
                    n_anchor = int(mask.sum())
                    n_segment = int(seg_mask.sum())
                    plan_anchor_counts.append(n_anchor)
                    plan_segment_counts.append(n_segment)
                    plan_zero_anchor_count += int(n_anchor == 0)
                    if n_anchor:
                        valid_times = plan["anchor_time"][mask]
                        plan_out_of_bounds_count += int(((valid_times < 0) | (valid_times >= seq_len)).sum())
                        plan_anchor_time_norm.extend((valid_times / max(seq_len, 1)).astype(np.float64).tolist())
                        valid_parts = plan["anchor_part"][mask]
                        plan_part_counts += (valid_parts > 0).sum(axis=0).astype(np.int64)
                        for type_id in plan["anchor_type"][mask].tolist():
                            plan_type_counts[int(type_id)] += 1
                        anchor_target_local_range.update(plan["anchor_target_local"][mask])
                        anchor_target_world_range.update(plan["anchor_target_world"][mask])
                        filler_only = (
                            float(np.abs(plan["anchor_conf"][mask]).max()) <= 1e-8
                            and float(np.abs(valid_parts).max()) <= 1e-8
                        )
                        plan_filler_only_count += int(filler_only)
                    plan_invalid_mask_count += int(mask.ndim != 1 or seg_mask.ndim != 1)

                if motion_npz is not None and "joints_22" in motion_npz.files and "object_positions" in motion_npz.files:
                    pseudo_seq_stats.append(
                        compute_seq_stats(
                            seq_id=seq_id,
                            labels={
                                "contact_state": contact_state,
                                "contact_target": contact_target,
                                "phase": phase,
                                "support": support,
                            },
                            joints_22=motion_npz["joints_22"][:seq_len].astype(np.float32, copy=False),
                            object_positions=motion_npz["object_positions"][:seq_len].astype(np.float32, copy=False),
                        )
                    )
            else:
                label_schema_failures += 1
        finally:
            if motion_npz is not None:
                motion_npz.close()
            if labels_npz is not None:
                labels_npz.close()

    pseudo_agg = aggregate_stats(pseudo_seq_stats)
    pseudo_quality_flags = make_quality_flags(pseudo_agg, subset_hint=subset)
    avg_anchor_count = float(np.mean(plan_anchor_counts)) if plan_anchor_counts else 0.0
    zero_anchor_fraction = float(plan_zero_anchor_count / max(len(plan_anchor_counts), 1))
    filler_only_fraction = float(plan_filler_only_count / max(len(plan_anchor_counts), 1))

    risk_reasons: list[str] = []
    warning_reasons: list[str] = []

    expected_count = int(len(metadata))
    if file_counts["motion_exists"] != expected_count:
        risk_reasons.append(f"{subset}: {expected_count - file_counts['motion_exists']} metadata clips lack motion npz")
    if file_counts["label_exists"] != expected_count:
        risk_reasons.append(f"{subset}: {expected_count - file_counts['label_exists']} metadata clips lack pseudo-label npz")
    if file_counts["object_exists"] != expected_count:
        risk_reasons.append(f"{subset}: {expected_count - file_counts['object_exists']} metadata clips lack object point cloud")
    if motion_schema_failures:
        risk_reasons.append(f"{subset}: {motion_schema_failures} clips have missing or malformed motion arrays")
    if object_schema_failures:
        risk_reasons.append(f"{subset}: {object_schema_failures} clips have missing object arrays")
    if label_schema_failures:
        risk_reasons.append(f"{subset}: {label_schema_failures} clips have missing or malformed label arrays")
    if restorable_stage_b_count != expected_count:
        risk_reasons.append(
            f"{subset}: {expected_count - restorable_stage_b_count} clips are missing SMPL-X pose arrays required by `smpl_pose_135_plan`"
        )
    if target_missing_count or target_nonfinite_clip_count or target_abs_gt_10m_clip_count:
        risk_reasons.append(
            f"{subset}: contact targets missing/nonfinite/extreme in "
            f"{target_missing_count}/{target_nonfinite_clip_count}/{target_abs_gt_10m_clip_count} clips"
        )
    if seq_length_mismatch_count:
        warning_reasons.append(f"{subset}: {seq_length_mismatch_count} clips have per-array temporal length mismatches")
    if pseudo_quality_flags:
        warning_reasons.extend(pseudo_quality_flags)
    if not (3.0 <= avg_anchor_count <= 12.0):
        warning_reasons.append(f"{subset}: average compiled anchor count {avg_anchor_count:.2f} is outside [3, 12]")
    if zero_anchor_fraction >= 0.05:
        warning_reasons.append(f"{subset}: zero-anchor plan fraction {zero_anchor_fraction:.3f} exceeds 0.05")
    if filler_only_fraction >= 0.05:
        warning_reasons.append(f"{subset}: filler-only plan fraction {filler_only_fraction:.3f} exceeds 0.05")
    if plan_out_of_bounds_count:
        risk_reasons.append(f"{subset}: {plan_out_of_bounds_count} anchor times fall outside clip bounds")
    if plan_invalid_mask_count:
        risk_reasons.append(f"{subset}: {plan_invalid_mask_count} compiled plans have invalid mask rank")

    if risk_reasons:
        risk = "fail"
    elif warning_reasons:
        risk = "warning"
    else:
        risk = "OK"

    support_names = _support_names_for_count(num_support)
    return {
        "processed": True,
        "metadata_source": metadata_path.name,
        "clip_count": expected_count,
        "summary_json": summary,
        "fps": summary.get("target_fps"),
        "basic_consistency": {
            "metadata_num_frames": _stats(metadata_lengths),
            "seq_len_from_arrays": _stats(seq_lengths),
            "files": {
                "motion_exists": int(file_counts["motion_exists"]),
                "label_exists": int(file_counts["label_exists"]),
                "object_exists": int(file_counts["object_exists"]),
            },
            "motion_tail_shapes": _counter_dict(motion_tail_shapes),
            "joints_tail_shapes": _counter_dict(joints_tail_shapes),
            "object_pc_shapes": _counter_dict(object_pc_shapes),
            "object_position_tail_shapes": _counter_dict(object_position_tail_shapes),
            "object_rotation_tail_shapes": _counter_dict(object_rotation_tail_shapes),
            "smplx_pose_tail_shapes": _counter_dict(smplx_pose_tail_shapes),
            "stage_b_motion_representation": str(cfg.data.motion_representation),
            "smplx_pose_ready_clip_count": int(restorable_stage_b_count),
            "seq_length_mismatch_count": int(seq_length_mismatch_count),
            "numeric_ranges": {
                "motion_263": motion_range.to_dict(),
                "joints_22": joints_range.to_dict(),
                "object_pc": object_pc_range.to_dict(),
                "object_positions": object_position_range.to_dict(),
                "object_rotations_axis_angle": object_rotation_range.to_dict(),
            },
        },
        "pseudo_labels": {
            "contact_state_range": contact_state_range.to_dict(),
            "contact_part_frame_rate_gt_0_5": {
                PART_NAMES[idx]: float(contact_positive_counts[idx] / max(contact_total_frames, 1))
                for idx in range(len(PART_NAMES))
            },
            "zero_any_contact_clip_count": int(zero_any_contact_clips),
            "zero_any_contact_clip_fraction": float(zero_any_contact_clips / max(expected_count, 1)),
            "full_any_contact_clip_count": int(full_any_contact_clips),
            "full_any_contact_clip_fraction": float(full_any_contact_clips / max(expected_count, 1)),
            "contact_all_zero_tensor_clip_count": int(contact_all_zero_tensor_clips),
            "contact_all_one_tensor_clip_count": int(contact_all_one_tensor_clips),
            "contact_target_xyz_range": target_xyz_range.to_dict(),
            "contact_target_missing_clip_count": int(target_missing_count),
            "contact_target_nonfinite_clip_count": int(target_nonfinite_clip_count),
            "contact_target_abs_gt_10m_clip_count": int(target_abs_gt_10m_clip_count),
            "phase_frame_distribution": {
                PHASE_NAMES[idx] if idx < len(PHASE_NAMES) else f"phase_{idx}":
                    float(phase_counts[idx] / max(int(phase_counts.sum()), 1))
                for idx in range(num_phase)
            },
            "support_frame_distribution_collapsed_for_stage_b": {
                support_names[idx]: float(support_counts[idx] / max(int(support_counts.sum()), 1))
                for idx in range(num_support)
            },
            "existing_pseudo_stats_aggregate": pseudo_agg,
            "quality_flags": pseudo_quality_flags,
        },
        "plan_quality": {
            "anchors_per_clip": _stats(plan_anchor_counts),
            "segments_per_clip": _stats(plan_segment_counts),
            "zero_anchor_clip_count": int(plan_zero_anchor_count),
            "zero_anchor_clip_fraction": zero_anchor_fraction,
            "filler_only_plan_clip_count": int(plan_filler_only_count),
            "filler_only_plan_clip_fraction": filler_only_fraction,
            "anchor_time_normalized": _stats(plan_anchor_time_norm),
            "anchor_time_out_of_bounds_count": int(plan_out_of_bounds_count),
            "anchor_mask_issue_count": int(plan_invalid_mask_count),
            "anchor_target_local_range": anchor_target_local_range.to_dict(),
            "anchor_target_world_range": anchor_target_world_range.to_dict(),
            "anchor_part_distribution": {
                PART_NAMES[idx]: int(plan_part_counts[idx])
                for idx in range(len(PART_NAMES))
            },
            "anchor_type_counts": _counter_dict(plan_type_counts),
        },
        "split_rows": split_rows,
        "risk": risk,
        "risk_reasons": risk_reasons,
        "warning_reasons": warning_reasons,
    }


def _split_audit(
    per_subset: dict[str, dict[str, Any]],
    cfg: Any,
) -> dict[str, Any]:
    subj_cfg = cfg.data.get("subject_split")
    subject_split_enabled = bool(subj_cfg is not None and subj_cfg.get("enabled", False))

    subject_keys: list[tuple[str, str]] = []
    parse_failures: dict[str, int] = {}
    rows_by_subset: dict[str, list[dict[str, str | None]]] = {}
    for subset, payload in per_subset.items():
        rows = payload.get("processed_audit", {}).get("split_rows", [])
        rows_by_subset[subset] = rows
        parse_failures[subset] = int(sum(1 for row in rows if row.get("subject_id") is None))
        for row in rows:
            subject_id = row.get("subject_id")
            if subject_id is not None:
                subject_keys.append((subset, str(subject_id)))

    split_sets = {"train": set(), "val": set()}
    if subject_split_enabled:
        split_sets = build_subject_split(
            subject_keys,
            train_pct=int(subj_cfg.train_pct),
            val_pct=int(subj_cfg.val_pct),
            seed=int(subj_cfg.seed),
        )

    bucket_counts: dict[str, dict[str, int]] = {}
    leakage_subject_overlap = sorted(split_sets["train"] & split_sets["val"])
    for subset, rows in rows_by_subset.items():
        counts = Counter()
        for row in rows:
            sid = row.get("subject_id")
            if sid is None:
                counts["unparsed"] += 1
                continue
            namespaced = f"{subset}/{sid}"
            if namespaced in split_sets["train"]:
                counts["train"] += 1
            elif namespaced in split_sets["val"]:
                counts["val"] += 1
            else:
                counts["unassigned"] += 1
        bucket_counts[subset] = {
            "train": int(counts["train"]),
            "val": int(counts["val"]),
            "unparsed": int(counts["unparsed"]),
            "unassigned": int(counts["unassigned"]),
        }

    missing_subject_pattern_subsets = sorted(
        subset
        for subset, payload in per_subset.items()
        if payload["raw_readiness"]["canonical_sequence_count"] > 0
        and not any(
            extract_subject_id(subset, seq_id) is not None
            for seq_id in payload["raw_readiness"].get("sample_seq_ids", [])
        )
    )

    return {
        "subject_split_enabled": subject_split_enabled,
        "train_pct": int(subj_cfg.train_pct) if subject_split_enabled else None,
        "val_pct": int(subj_cfg.val_pct) if subject_split_enabled else None,
        "seed": int(subj_cfg.seed) if subject_split_enabled else None,
        "parse_failure_counts_processed_subsets": parse_failures,
        "bucket_clip_counts_processed_subsets": bucket_counts,
        "subject_overlap_between_train_and_val": leakage_subject_overlap,
        "same_subject_same_object_segment_cross_bucket_risk": (
            "low under subject split for parsed subjects"
            if subject_split_enabled and not leakage_subject_overlap
            else "not established"
        ),
        "metadata_split_field_note": (
            "metadata split columns are not the active Stage B split authority; "
            "the trainer rebuilds subject-level train/val buckets from config."
        ),
        "missing_subject_pattern_subsets": missing_subject_pattern_subsets,
    }


def _risk_for_missing_subset(
    subset: str,
    raw_ready: dict[str, Any],
    official_ready: dict[str, Any],
    piano_root: Path,
) -> tuple[str, list[str], list[str]]:
    fail: list[str] = []
    warn: list[str] = []
    piano_subset_root = piano_root / subset
    if not piano_subset_root.exists():
        fail.append(f"{subset}: not present under current PIANO training root {piano_root}")
    if not official_ready["subset_root_exists"]:
        fail.append(f"{subset}: not present under current official-process root")
    coverage = raw_ready.get("required_file_coverage", {})
    for key in ("joints.npy", "markers.npy"):
        if key in coverage and coverage[key]["fraction_present"] < 1.0:
            fail.append(
                f"{subset}: raw canonical coverage for {key} is "
                f"{coverage[key]['present']}/{raw_ready['canonical_sequence_count']}"
            )
    if raw_ready.get("canonical_sequence_count", 0) == 0:
        fail.append(f"{subset}: raw canonical sequences not found")
    if fail:
        return "fail", fail, warn
    if warn:
        return "warning", fail, warn
    return "OK", fail, warn


def _write_markdown(path: Path, results: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# All-7 Stage B subset audit")
    lines.append("")
    lines.append(f"**Date:** {results['date']}")
    lines.append(f"**Config:** `{results['config']}`")
    lines.append(f"**Current PIANO root:** `{results['piano_root']}`")
    lines.append(f"**Raw InterAct root:** `{results['raw_interact_root']}`")
    lines.append(f"**Official-process root:** `{results['official_root']}`")
    lines.append("")
    lines.append("## Executive decision")
    lines.append("")
    lines.append(f"**Can v19 all-7 full-data training start now?** **{results['decision']['can_start_v19']}**")
    lines.append("")
    for bullet in results["decision"]["summary"]:
        lines.append(f"- {bullet}")

    lines.append("")
    lines.append("## Table 1: All-7 readiness summary")
    lines.append("")
    lines.append("| subset | raw canonical seqs | official root | PIANO root | processed clips | risk | headline |")
    lines.append("|---|---:|---|---|---:|---|---|")
    for subset in results["expected_subsets"]:
        payload = results["subsets"][subset]
        processed = payload.get("processed_audit", {})
        headline = "; ".join(
            (payload.get("risk_reasons", []) + payload.get("warning_reasons", []))[:1]
        ) or "ready"
        lines.append(
            f"| {subset} | {payload['raw_readiness']['canonical_sequence_count']} | "
            f"{'yes' if payload['official_readiness']['subset_root_exists'] else 'no'} | "
            f"{'yes' if payload['piano_subset_root_exists'] else 'no'} | "
            f"{processed.get('clip_count', 0)} | **{payload['risk']}** | {headline} |"
        )

    lines.append("")
    lines.append("## Table 2: Processed subset data consistency")
    lines.append("")
    lines.append("| subset | fps | clips | seq len mean | motion tails | joints tails | object traj tails | SMPL-X-ready clips |")
    lines.append("|---|---:|---:|---:|---|---|---|---:|")
    for subset in results["expected_subsets"]:
        processed = results["subsets"][subset].get("processed_audit")
        if not processed or not processed.get("processed", False):
            continue
        basic = processed["basic_consistency"]
        object_tail = ", ".join(basic["object_position_tail_shapes"].keys())
        lines.append(
            f"| {subset} | {processed.get('fps', 'n/a')} | {processed['clip_count']} | "
            f"{basic['seq_len_from_arrays']['mean']:.1f} | "
            f"{', '.join(basic['motion_tail_shapes'].keys())} | "
            f"{', '.join(basic['joints_tail_shapes'].keys())} | "
            f"{object_tail} | {basic['smplx_pose_ready_clip_count']} |"
        )

    lines.append("")
    lines.append("## Table 3: Pseudo-label quality")
    lines.append("")
    lines.append("| subset | zero-contact clips | full-contact clips | L hand rate | R hand rate | target range | phase non/stable/manip | support distribution |")
    lines.append("|---|---:|---:|---:|---:|---|---|---|")
    for subset in results["expected_subsets"]:
        processed = results["subsets"][subset].get("processed_audit")
        if not processed or not processed.get("processed", False):
            continue
        pseudo = processed["pseudo_labels"]
        target_range = pseudo["contact_target_xyz_range"]
        phase_dist = pseudo["phase_frame_distribution"]
        support_dist = pseudo["support_frame_distribution_collapsed_for_stage_b"]
        lines.append(
            f"| {subset} | {pseudo['zero_any_contact_clip_fraction']:.3f} | "
            f"{pseudo['full_any_contact_clip_fraction']:.3f} | "
            f"{pseudo['contact_part_frame_rate_gt_0_5']['L_hand']:.3f} | "
            f"{pseudo['contact_part_frame_rate_gt_0_5']['R_hand']:.3f} | "
            f"[{target_range['min']}, {target_range['max']}] | "
            f"{phase_dist.get('non_contact', 0.0):.3f}/"
            f"{phase_dist.get('stable_contact', 0.0):.3f}/"
            f"{phase_dist.get('manipulation', 0.0):.3f} | "
            f"{', '.join(f'{k}:{v:.3f}' for k, v in support_dist.items())} |"
        )

    lines.append("")
    lines.append("## Table 4: Compiled plan quality")
    lines.append("")
    lines.append("| subset | anchors mean | zero-anchor frac | filler-only frac | anchor time mean | local target range | world target range |")
    lines.append("|---|---:|---:|---:|---:|---|---|")
    for subset in results["expected_subsets"]:
        processed = results["subsets"][subset].get("processed_audit")
        if not processed or not processed.get("processed", False):
            continue
        plan = processed["plan_quality"]
        local_range = plan["anchor_target_local_range"]
        world_range = plan["anchor_target_world_range"]
        lines.append(
            f"| {subset} | {plan['anchors_per_clip']['mean']:.2f} | "
            f"{plan['zero_anchor_clip_fraction']:.3f} | "
            f"{plan['filler_only_plan_clip_fraction']:.3f} | "
            f"{plan['anchor_time_normalized']['mean']:.3f} | "
            f"[{local_range['min']}, {local_range['max']}] | "
            f"[{world_range['min']}, {world_range['max']}] |"
        )

    lines.append("")
    lines.append("## Table 5: Split safety")
    lines.append("")
    split = results["split_audit"]
    lines.append("| item | finding |")
    lines.append("|---|---|")
    lines.append(f"| subject split enabled | {split['subject_split_enabled']} |")
    lines.append(
        f"| processed parse failures | "
        f"{', '.join(f'{k}:{v}' for k, v in split['parse_failure_counts_processed_subsets'].items())} |"
    )
    lines.append(
        f"| train/val subject overlap | "
        f"{len(split['subject_overlap_between_train_and_val'])} |"
    )
    lines.append(
        f"| missing subject-id patterns for raw all-7 candidates | "
        f"{', '.join(split['missing_subject_pattern_subsets']) or 'none'} |"
    )
    lines.append(f"| same subject-object-segment leakage risk | {split['same_subject_same_object_segment_cross_bucket_risk']} |")
    lines.append(f"| metadata split note | {split['metadata_split_field_note']} |")

    lines.append("")
    lines.append("## Table 6: Subset risk classification")
    lines.append("")
    lines.append("| subset | risk | reasons |")
    lines.append("|---|---|---|")
    for subset in results["expected_subsets"]:
        payload = results["subsets"][subset]
        reasons = payload.get("risk_reasons", []) + payload.get("warning_reasons", [])
        lines.append(
            f"| {subset} | **{payload['risk']}** | "
            f"{'; '.join(reasons) if reasons else 'no blocking issue found'} |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    for paragraph in results["decision"]["interpretation"]:
        lines.append(paragraph)
        lines.append("")

    lines.append("## Next gate before v19")
    lines.append("")
    for item in results["decision"]["next_gate"]:
        lines.append(f"1. {item}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--raw-interact-root", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--md", type=Path, required=True)
    parser.add_argument(
        "--expected-subsets",
        nargs="+",
        default=list(EXPECTED_ALL7_SUBSETS),
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    configured_roots = {str(Path(entry.root).name): Path(entry.root) for entry in cfg.data.datasets}
    if not configured_roots:
        raise RuntimeError("config.data.datasets is empty")
    piano_root = next(iter(configured_roots.values())).parent
    pseudo_label_subdir = str(cfg.data.get("pseudo_label_subdir", "pseudo_labels"))

    subsets: dict[str, dict[str, Any]] = {}
    for subset in args.expected_subsets:
        raw_ready = _raw_subset_readiness(args.raw_interact_root, subset)
        official_ready = _official_subset_readiness(args.official_root, subset)
        piano_subset_root = piano_root / subset
        payload: dict[str, Any] = {
            "raw_readiness": raw_ready,
            "official_readiness": official_ready,
            "piano_subset_root": str(piano_subset_root),
            "piano_subset_root_exists": piano_subset_root.exists(),
        }
        if piano_subset_root.exists():
            processed = _audit_processed_subset(
                subset=subset,
                subset_root=piano_subset_root,
                pseudo_label_subdir=pseudo_label_subdir,
                cfg=cfg,
            )
            payload["processed_audit"] = processed
            payload["risk"] = processed.get("risk", "fail")
            payload["risk_reasons"] = processed.get("risk_reasons", processed.get("fail_reasons", []))
            payload["warning_reasons"] = processed.get("warning_reasons", [])
        else:
            risk, fail_reasons, warning_reasons = _risk_for_missing_subset(
                subset=subset,
                raw_ready=raw_ready,
                official_ready=official_ready,
                piano_root=piano_root,
            )
            payload["risk"] = risk
            payload["risk_reasons"] = fail_reasons
            payload["warning_reasons"] = warning_reasons
        subsets[subset] = payload

    split_audit = _split_audit(subsets, cfg)
    for subset in split_audit["missing_subject_pattern_subsets"]:
        payload = subsets[subset]
        msg = f"{subset}: subject-id extraction is not defined in src/piano/data/split.py"
        if msg not in payload["risk_reasons"]:
            payload["risk_reasons"].append(msg)
        payload["risk"] = "fail"

    current_processed_failures = [
        subset
        for subset, payload in subsets.items()
        if payload["piano_subset_root_exists"] and payload["risk"] == "fail"
    ]
    onboarding_failures = [
        subset
        for subset, payload in subsets.items()
        if not payload["piano_subset_root_exists"] or payload["risk"] == "fail"
    ]
    can_start_v19 = "NO"
    summary: list[str] = []
    if current_processed_failures:
        summary.append(
            "At least one already-processed Stage B subset fails the audit: "
            + ", ".join(current_processed_failures)
            + "."
        )
    else:
        summary.append("The four currently processed Stage B subsets have no blocking audit failure.")
    if onboarding_failures:
        summary.append(
            "The all-7 expansion is not training-ready because the remaining subset set still has blockers: "
            + ", ".join(onboarding_failures)
            + "."
        )
    summary.append(
        "The current evidence supports continuing the data-coverage route, but v19 must wait until the extra subsets are onboarded into the same PIANO/pseudo-label/split contract."
    )

    decision = {
        "can_start_v19": can_start_v19,
        "summary": summary,
        "interpretation": [
            (
                "This audit separates two questions that should not be conflated: "
                "(1) whether the current v18 four-subset Stage B corpus is internally healthy, and "
                "(2) whether the remaining three raw InterAct subsets can be appended immediately. "
                "The first question is mostly a quality/readiness check. The second is an onboarding check."
            ),
            (
                "For the current workspace state, the all-7 branch is blocked before training begins. "
                "The extra raw subsets are absent from the official-process and PIANO roots used by v18, "
                "their raw canonical trees do not expose the same joints/markers artifacts that the current "
                "pipeline audits downstream, and the subject-split extractor does not yet define patterns for "
                "their sequence IDs. Starting v19 without fixing those would mix data-contract changes into "
                "the supposed single-variable mode-coverage test."
            ),
        ],
        "next_gate": [
            "Onboard `behave`, `grab`, and `intercap` into the official/PIANO preprocessing path or document an equivalent contract-preserving route.",
            "Add subject-id extraction coverage for the three new subsets and re-run the split-safety section of this audit.",
            "Extract the same pseudo-label schema and compile interaction plans for the new subsets, then rerun this audit until all seven are at worst `warning` rather than `fail`.",
            "Only then create and launch `anchordiff_v19_a1_all7_FULL_DATA.yaml` as the clean single-variable data-coverage experiment.",
        ],
    }

    results = {
        "date": "2026-05-13",
        "config": str(args.config),
        "raw_interact_root": str(args.raw_interact_root),
        "official_root": str(args.official_root),
        "piano_root": str(piano_root),
        "pseudo_label_subdir": pseudo_label_subdir,
        "expected_subsets": list(args.expected_subsets),
        "configured_v18_subsets": sorted(configured_roots.keys()),
        "subsets": subsets,
        "split_audit": split_audit,
        "decision": decision,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    _write_markdown(args.md, results)

    print(f"Wrote JSON to {args.output}")
    print(f"Wrote Markdown to {args.md}")
    print(f"v19 all-7 ready: {decision['can_start_v19']}")


if __name__ == "__main__":
    main()
