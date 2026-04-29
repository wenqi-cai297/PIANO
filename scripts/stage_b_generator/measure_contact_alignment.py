"""Compare generated contact against GT contact in object-local coordinates.

This diagnostic is meant for the failure mode where aggregate body-object
distance is low but visual contact is still visibly misaligned. It compares a
generated run to a GT/roundtrip run on the same clips and reports:

- temporal contact overlap;
- whether the same tracked body part is in contact;
- whether the generated body reaches the GT object-local contact target;
- whether generated and GT nearest object-surface patches match.

The script is pure post-processing on ``generated.npz`` files. It does not load
generator checkpoints and does not require CUDA.

Usage::

    python scripts/stage_b_generator/measure_contact_alignment.py \
        --generated-dir runs/eval/stageB_v0_14_bc_k16_composite_oracle/best \
        --gt-dir runs/eval/stageB_v0_14_sampled_st_contact_gt_roundtrip_80/gt_roundtrip \
        --output-dir runs/eval/stageB_v0_14_bc_k16_composite_oracle/alignment_to_gt_roundtrip
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

import piano.models.backbones.momask_adapter  # noqa: F401 - MoMask path side-effect
from utils.motion_process import recover_from_ric

from piano.data.pseudo_labels.extract_contact import (
    ContactConfig,
    _kinematic_contact_score,
    _soft_sigmoid,
)
from piano.training.contact_eval import (
    _lift_canonical_to_world,
    _object_motion_speed,
    _world_object_pc_per_frame,
)
from piano.utils.canonical_frame import axis_angle_to_matrix_np
from piano.utils.io_utils import ensure_dir
from piano.utils.smpl_utils import BODY_PART_INDICES, BODY_PART_NAMES


@dataclass(frozen=True)
class RunData:
    input_dir: Path
    summary: dict[str, Any]
    npz: Any
    seq_ids: list[str]
    seq_lens: list[int]
    seq_to_index: dict[str, int]
    meta_by_seq: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class ContactFeatures:
    body_world: np.ndarray
    body_local: np.ndarray
    distances: np.ndarray
    nearest_pc_idx: np.ndarray
    nearest_pc_local: np.ndarray
    min_dist: np.ndarray
    min_part: np.ndarray
    contact_score: np.ndarray
    contact_any: np.ndarray
    contact_part: np.ndarray
    contact_point_local: np.ndarray
    moving: np.ndarray


def _round(x: float | None, ndigits: int = 4) -> float | None:
    if x is None or not np.isfinite(x):
        return None
    return round(float(x), ndigits)


def _safe_div(num: int | float, den: int | float) -> float | None:
    den_f = float(den)
    if den_f <= 0.0:
        return None
    return float(num) / den_f


def _mean_or_none(values: list[float | None]) -> float | None:
    xs = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not xs:
        return None
    return float(np.mean(xs))


def _weighted_mean_or_none(rows: list[dict[str, Any]], key: str, weight_key: str) -> float | None:
    num = 0.0
    den = 0
    for row in rows:
        value = row.get(key)
        weight = int(row.get(weight_key) or 0)
        if value is None or weight <= 0 or not np.isfinite(float(value)):
            continue
        num += float(value) * weight
        den += weight
    if den <= 0:
        return None
    return num / den


def _fraction_from_counts(rows: list[dict[str, Any]], num_key: str, den_key: str) -> float | None:
    num = sum(int(row.get(num_key) or 0) for row in rows)
    den = sum(int(row.get(den_key) or 0) for row in rows)
    return _safe_div(num, den)


def _load_run(input_dir: Path) -> RunData:
    npz_path = input_dir / "generated.npz"
    summary_path = input_dir / "summary.json"
    if not npz_path.exists():
        raise FileNotFoundError(f"missing {npz_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"missing {summary_path}")

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    npz = np.load(npz_path)

    seq_ids = [str(x) for x in summary["seq_ids"]]
    seq_lens = summary.get("seq_lens")
    if seq_lens is None and "seq_lens" in npz.files:
        seq_lens = npz["seq_lens"].tolist()
    if seq_lens is None:
        raise ValueError(f"{summary_path} has no seq_lens, and {npz_path} does not either")
    seq_lens = [int(x) for x in seq_lens]
    if len(seq_ids) != len(seq_lens):
        raise ValueError(f"{summary_path} has {len(seq_ids)} seq_ids but {len(seq_lens)} seq_lens")

    return RunData(
        input_dir=input_dir,
        summary=summary,
        npz=npz,
        seq_ids=seq_ids,
        seq_lens=seq_lens,
        seq_to_index={sid: i for i, sid in enumerate(seq_ids)},
        meta_by_seq=_load_meta_by_seq(input_dir, summary),
    )


def _load_meta_by_seq(input_dir: Path, summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Load clip metadata from this summary and optional oracle parent summary."""
    summaries = []
    parent_summary = input_dir.parent / "summary.json"
    if parent_summary.exists() and parent_summary != input_dir / "summary.json":
        try:
            with parent_summary.open("r", encoding="utf-8") as f:
                summaries.append(json.load(f))
        except json.JSONDecodeError:
            pass
    summaries.append(summary)

    meta: dict[str, dict[str, Any]] = {}
    for data in summaries:
        for item in data.get("clip_selection", []):
            seq_id = str(item.get("seq_id", ""))
            if not seq_id:
                continue
            meta.setdefault(seq_id, {}).update({
                "subset": item.get("subset"),
                "object_id": item.get("object_id"),
                "dataset_index": item.get("index"),
            })
        for item in data.get("per_clip", []):
            seq_id = str(item.get("seq_id", ""))
            if not seq_id:
                continue
            meta.setdefault(seq_id, {}).update({
                key: item.get(key)
                for key in (
                    "subset",
                    "object_id",
                    "best_sample_index",
                    "best_seed",
                    "best_dist_cm",
                )
                if key in item
            })
    return meta


def _recover_body_world(run: RunData, index: int, T: int) -> np.ndarray:
    motion = run.npz["motion_263"][index, :T]
    motion_t = torch.from_numpy(motion).float().unsqueeze(0)
    canon = recover_from_ric(motion_t, 22).squeeze(0).cpu().numpy().astype(np.float32)
    world = _lift_canonical_to_world(
        canon,
        float(run.npz["world_R_y_angle"][index]),
        run.npz["world_T_xz"][index],
    )
    return world[:, BODY_PART_INDICES, :].astype(np.float32)


def _world_points_to_object_local(
    points_world: np.ndarray,
    object_positions: np.ndarray,
    object_rotations: np.ndarray,
) -> np.ndarray:
    """Transform ``(T, P, 3)`` world points into object-local coordinates."""
    R_obj = axis_angle_to_matrix_np(object_rotations.astype(np.float32))
    centered = points_world - object_positions[:, None, :].astype(np.float32)
    return np.einsum("tji,tpj->tpi", R_obj, centered).astype(np.float32)


def _surface_nearest(
    body_world: np.ndarray,
    object_pc_local: np.ndarray,
    object_positions: np.ndarray,
    object_rotations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-part surface distance and nearest local PC point."""
    pc_world = _world_object_pc_per_frame(
        object_pc_local.astype(np.float32),
        object_positions.astype(np.float32),
        object_rotations.astype(np.float32),
    )
    diff = body_world[:, :, None, :] - pc_world[:, None, :, :]
    dist_all = np.linalg.norm(diff, axis=-1)
    nearest_idx = dist_all.argmin(axis=-1)
    distances = np.take_along_axis(dist_all, nearest_idx[:, :, None], axis=-1).squeeze(-1)
    nearest_local = object_pc_local.astype(np.float32)[nearest_idx]
    return distances.astype(np.float32), nearest_idx.astype(np.int32), nearest_local.astype(np.float32)


def _contact_features(
    *,
    run: RunData,
    index: int,
    T: int,
    cfg: ContactConfig,
    contact_threshold: float,
) -> ContactFeatures:
    body_world = _recover_body_world(run, index, T)
    object_pc_local = run.npz["object_pc"][index].astype(np.float32)
    object_positions = run.npz["object_positions"][index, :T].astype(np.float32)
    object_rotations = run.npz["object_rotations"][index, :T].astype(np.float32)

    body_local = _world_points_to_object_local(body_world, object_positions, object_rotations)
    distances, nearest_idx, nearest_local = _surface_nearest(
        body_world,
        object_pc_local,
        object_positions,
        object_rotations,
    )

    min_part = distances.argmin(axis=1).astype(np.int32)
    frame_idx = np.arange(T)
    min_dist = distances[frame_idx, min_part]

    thresholds = np.array(
        [cfg.distance_thresholds[name] for name in BODY_PART_NAMES],
        dtype=np.float32,
    )
    dist_score = _soft_sigmoid(distances, thresholds[None, :], float(cfg.distance_sigma))
    kin_score = np.stack(
        [
            _kinematic_contact_score(body_world[:, p, :], object_positions, object_rotations, cfg)
            for p in range(len(BODY_PART_NAMES))
        ],
        axis=1,
    )
    contact_score = np.maximum(dist_score, kin_score).astype(np.float32)
    contact_any = contact_score.max(axis=1) >= float(contact_threshold)
    contact_part = contact_score.argmax(axis=1).astype(np.int32)
    contact_point_local = nearest_local[frame_idx, contact_part]

    speed = _object_motion_speed(object_positions, object_rotations, cfg)
    moving = speed >= float(cfg.kin_world_eps)

    return ContactFeatures(
        body_world=body_world,
        body_local=body_local,
        distances=distances,
        nearest_pc_idx=nearest_idx,
        nearest_pc_local=nearest_local,
        min_dist=min_dist,
        min_part=min_part,
        contact_score=contact_score,
        contact_any=contact_any,
        contact_part=contact_part,
        contact_point_local=contact_point_local,
        moving=moving,
    )


def _mean_norm(values: np.ndarray, mask: np.ndarray) -> float | None:
    if int(mask.sum()) <= 0:
        return None
    return float(np.linalg.norm(values[mask], axis=-1).mean())


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float | None:
    if int(mask.sum()) <= 0:
        return None
    return float(values[mask].mean())


def _dominant_part(parts: np.ndarray, mask: np.ndarray) -> str | None:
    if int(mask.sum()) <= 0:
        return None
    counts = np.bincount(parts[mask].astype(np.int64), minlength=len(BODY_PART_NAMES))
    return BODY_PART_NAMES[int(counts.argmax())]


def _clip_metrics(
    *,
    seq_id: str,
    subset: str | None,
    object_id: str | None,
    T: int,
    gen: ContactFeatures,
    gt: ContactFeatures,
    contact_threshold: float,
) -> dict[str, Any]:
    gt_mask = gt.contact_any
    gen_mask = gen.contact_any
    both = gt_mask & gen_mask
    union = gt_mask | gen_mask
    moving = gt.moving
    moving_gt = moving & gt_mask
    moving_gen = moving & gen_mask
    moving_both = moving & both
    moving_union = moving & union

    frame_idx = np.arange(T)
    gt_part = gt.contact_part
    gen_part = gen.contact_part
    gen_same_gt_part_score = gen.contact_score[frame_idx, gt_part]
    gen_same_gt_part_contact = gen_same_gt_part_score >= float(contact_threshold)
    same_gt_part_surface_dist = gen.distances[frame_idx, gt_part]
    gen_same_gt_part_local = gen.body_local[frame_idx, gt_part]
    gt_same_part_local = gt.body_local[frame_idx, gt_part]
    same_gt_part_local_position_delta = gen_same_gt_part_local - gt_same_part_local
    target_part_local_delta = gen_same_gt_part_local - gt.contact_point_local
    contact_point_delta = gen.contact_point_local - gt.contact_point_local

    right_part_on_gt = gt_mask & gen_mask & (gen_part == gt_part)
    wrong_part_on_gt = gt_mask & gen_mask & (gen_part != gt_part)
    missed_on_gt = gt_mask & ~gen_mask
    same_part_on_gt = gt_mask & gen_same_gt_part_contact
    moving_right_part_on_gt = moving & right_part_on_gt
    moving_wrong_part_on_gt = moving & wrong_part_on_gt
    moving_missed_on_gt = moving & missed_on_gt
    moving_same_part_on_gt = moving & same_part_on_gt
    n_parts = len(BODY_PART_NAMES)
    part_confusion = np.zeros((n_parts, n_parts), dtype=np.int64)
    moving_part_confusion = np.zeros((n_parts, n_parts), dtype=np.int64)
    if int(both.sum()) > 0:
        np.add.at(part_confusion, (gt_part[both], gen_part[both]), 1)
    if int(moving_both.sum()) > 0:
        np.add.at(moving_part_confusion, (gt_part[moving_both], gen_part[moving_both]), 1)

    gt_contact_frames = int(gt_mask.sum())
    gen_contact_frames = int(gen_mask.sum())
    both_contact_frames = int(both.sum())
    union_contact_frames = int(union.sum())
    moving_gt_contact_frames = int(moving_gt.sum())
    moving_gen_contact_frames = int(moving_gen.sum())
    moving_both_contact_frames = int(moving_both.sum())
    moving_union_contact_frames = int(moving_union.sum())
    moving_frames = int(moving.sum())

    row = {
        "seq_id": seq_id,
        "subset": subset,
        "object_id": object_id,
        "T": int(T),
        "moving_frames": moving_frames,
        "gt_contact_frames": gt_contact_frames,
        "gen_contact_frames": gen_contact_frames,
        "both_contact_frames": both_contact_frames,
        "union_contact_frames": union_contact_frames,
        "moving_gt_contact_frames": moving_gt_contact_frames,
        "moving_gen_contact_frames": moving_gen_contact_frames,
        "moving_both_contact_frames": moving_both_contact_frames,
        "moving_union_contact_frames": moving_union_contact_frames,
        "gt_contact_frame_frac": _round(gt_contact_frames / T),
        "gen_contact_frame_frac": _round(gen_contact_frames / T),
        "contact_temporal_iou": _round(_safe_div(both_contact_frames, union_contact_frames)),
        "contact_recall_on_gt": _round(_safe_div(both_contact_frames, gt_contact_frames)),
        "contact_precision_vs_gt": _round(_safe_div(both_contact_frames, gen_contact_frames)),
        "moving_contact_temporal_iou": _round(
            _safe_div(moving_both_contact_frames, moving_union_contact_frames)
        ),
        "moving_contact_recall_on_gt": _round(
            _safe_div(moving_both_contact_frames, moving_gt_contact_frames)
        ),
        "moving_contact_precision_vs_gt": _round(
            _safe_div(moving_both_contact_frames, moving_gen_contact_frames)
        ),
        "part_match_on_both_contact": _round(
            float((gen_part[both] == gt_part[both]).mean()) if both_contact_frames else None
        ),
        "moving_part_match_on_both_contact": _round(
            float((gen_part[moving_both] == gt_part[moving_both]).mean())
            if moving_both_contact_frames else None
        ),
        "right_part_contact_recall_on_gt": _round(
            _safe_div(int(right_part_on_gt.sum()), gt_contact_frames)
        ),
        "wrong_part_contact_frac_on_gt": _round(
            _safe_div(int(wrong_part_on_gt.sum()), gt_contact_frames)
        ),
        "missed_contact_frac_on_gt": _round(
            _safe_div(int(missed_on_gt.sum()), gt_contact_frames)
        ),
        "same_gt_part_contact_recall_on_gt": _round(
            _safe_div(int(same_part_on_gt.sum()), gt_contact_frames)
        ),
        "moving_right_part_contact_recall_on_gt": _round(
            _safe_div(int(moving_right_part_on_gt.sum()), moving_gt_contact_frames)
        ),
        "moving_wrong_part_contact_frac_on_gt": _round(
            _safe_div(int(moving_wrong_part_on_gt.sum()), moving_gt_contact_frames)
        ),
        "moving_missed_contact_frac_on_gt": _round(
            _safe_div(int(moving_missed_on_gt.sum()), moving_gt_contact_frames)
        ),
        "moving_same_gt_part_contact_recall_on_gt": _round(
            _safe_div(int(moving_same_part_on_gt.sum()), moving_gt_contact_frames)
        ),
        "gt_mean_min_dist_m": _round(float(gt.min_dist.mean())),
        "gen_mean_min_dist_m": _round(float(gen.min_dist.mean())),
        "same_gt_part_surface_dist_m_on_gt_contact": _round(
            _masked_mean(same_gt_part_surface_dist, gt_mask)
        ),
        "same_gt_part_local_position_error_m_on_gt_contact": _round(
            _mean_norm(same_gt_part_local_position_delta, gt_mask)
        ),
        "target_part_local_error_m_on_gt_contact": _round(
            _mean_norm(target_part_local_delta, gt_mask)
        ),
        "nearest_contact_point_error_m_on_gt_contact": _round(
            _mean_norm(contact_point_delta, gt_mask)
        ),
        "nearest_contact_point_error_m_on_both_contact": _round(
            _mean_norm(contact_point_delta, both)
        ),
        "moving_same_gt_part_surface_dist_m_on_gt_contact": _round(
            _masked_mean(same_gt_part_surface_dist, moving_gt)
        ),
        "moving_same_gt_part_local_position_error_m_on_gt_contact": _round(
            _mean_norm(same_gt_part_local_position_delta, moving_gt)
        ),
        "moving_target_part_local_error_m_on_gt_contact": _round(
            _mean_norm(target_part_local_delta, moving_gt)
        ),
        "moving_nearest_contact_point_error_m_on_gt_contact": _round(
            _mean_norm(contact_point_delta, moving_gt)
        ),
        "moving_nearest_contact_point_error_m_on_both_contact": _round(
            _mean_norm(contact_point_delta, moving_both)
        ),
        "gt_dominant_contact_part": _dominant_part(gt_part, gt_mask),
        "gen_dominant_contact_part": _dominant_part(gen_part, gen_mask),
        "part_confusion_on_both_contact": part_confusion.tolist(),
        "moving_part_confusion_on_both_contact": moving_part_confusion.tolist(),
    }
    return row


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    frame_count = sum(int(row["T"]) for row in rows)
    moving_frame_count = sum(int(row["moving_frames"]) for row in rows)
    aggregate = {
        "n_clips": len(rows),
        "n_frames": frame_count,
        "n_moving_frames": moving_frame_count,
        "gt_contact_frame_frac": _round(
            _safe_div(sum(int(r["gt_contact_frames"]) for r in rows), frame_count)
        ),
        "gen_contact_frame_frac": _round(
            _safe_div(sum(int(r["gen_contact_frames"]) for r in rows), frame_count)
        ),
        "contact_temporal_iou": _round(
            _safe_div(
                sum(int(r["both_contact_frames"]) for r in rows),
                sum(int(r["union_contact_frames"]) for r in rows),
            )
        ),
        "contact_recall_on_gt": _round(
            _fraction_from_counts(rows, "both_contact_frames", "gt_contact_frames")
        ),
        "contact_precision_vs_gt": _round(
            _fraction_from_counts(rows, "both_contact_frames", "gen_contact_frames")
        ),
        "moving_contact_temporal_iou": _round(
            _safe_div(
                sum(int(r["moving_both_contact_frames"]) for r in rows),
                sum(int(r["moving_union_contact_frames"]) for r in rows),
            )
        ),
        "moving_contact_recall_on_gt": _round(
            _fraction_from_counts(rows, "moving_both_contact_frames", "moving_gt_contact_frames")
        ),
        "moving_contact_precision_vs_gt": _round(
            _fraction_from_counts(rows, "moving_both_contact_frames", "moving_gen_contact_frames")
        ),
        "part_match_on_both_contact": _round(
            _weighted_mean_or_none(rows, "part_match_on_both_contact", "both_contact_frames")
        ),
        "moving_part_match_on_both_contact": _round(
            _weighted_mean_or_none(
                rows,
                "moving_part_match_on_both_contact",
                "moving_both_contact_frames",
            )
        ),
        "right_part_contact_recall_on_gt": _round(
            _weighted_mean_or_none(rows, "right_part_contact_recall_on_gt", "gt_contact_frames")
        ),
        "wrong_part_contact_frac_on_gt": _round(
            _weighted_mean_or_none(rows, "wrong_part_contact_frac_on_gt", "gt_contact_frames")
        ),
        "missed_contact_frac_on_gt": _round(
            _weighted_mean_or_none(rows, "missed_contact_frac_on_gt", "gt_contact_frames")
        ),
        "same_gt_part_contact_recall_on_gt": _round(
            _weighted_mean_or_none(rows, "same_gt_part_contact_recall_on_gt", "gt_contact_frames")
        ),
        "moving_right_part_contact_recall_on_gt": _round(
            _weighted_mean_or_none(
                rows,
                "moving_right_part_contact_recall_on_gt",
                "moving_gt_contact_frames",
            )
        ),
        "moving_wrong_part_contact_frac_on_gt": _round(
            _weighted_mean_or_none(
                rows,
                "moving_wrong_part_contact_frac_on_gt",
                "moving_gt_contact_frames",
            )
        ),
        "moving_missed_contact_frac_on_gt": _round(
            _weighted_mean_or_none(
                rows,
                "moving_missed_contact_frac_on_gt",
                "moving_gt_contact_frames",
            )
        ),
        "moving_same_gt_part_contact_recall_on_gt": _round(
            _weighted_mean_or_none(
                rows,
                "moving_same_gt_part_contact_recall_on_gt",
                "moving_gt_contact_frames",
            )
        ),
        "gt_mean_min_dist_m": _round(_mean_or_none([r.get("gt_mean_min_dist_m") for r in rows])),
        "gen_mean_min_dist_m": _round(_mean_or_none([r.get("gen_mean_min_dist_m") for r in rows])),
        "same_gt_part_surface_dist_m_on_gt_contact": _round(
            _weighted_mean_or_none(
                rows,
                "same_gt_part_surface_dist_m_on_gt_contact",
                "gt_contact_frames",
            )
        ),
        "same_gt_part_local_position_error_m_on_gt_contact": _round(
            _weighted_mean_or_none(
                rows,
                "same_gt_part_local_position_error_m_on_gt_contact",
                "gt_contact_frames",
            )
        ),
        "target_part_local_error_m_on_gt_contact": _round(
            _weighted_mean_or_none(
                rows,
                "target_part_local_error_m_on_gt_contact",
                "gt_contact_frames",
            )
        ),
        "nearest_contact_point_error_m_on_gt_contact": _round(
            _weighted_mean_or_none(
                rows,
                "nearest_contact_point_error_m_on_gt_contact",
                "gt_contact_frames",
            )
        ),
        "nearest_contact_point_error_m_on_both_contact": _round(
            _weighted_mean_or_none(
                rows,
                "nearest_contact_point_error_m_on_both_contact",
                "both_contact_frames",
            )
        ),
        "moving_same_gt_part_surface_dist_m_on_gt_contact": _round(
            _weighted_mean_or_none(
                rows,
                "moving_same_gt_part_surface_dist_m_on_gt_contact",
                "moving_gt_contact_frames",
            )
        ),
        "moving_same_gt_part_local_position_error_m_on_gt_contact": _round(
            _weighted_mean_or_none(
                rows,
                "moving_same_gt_part_local_position_error_m_on_gt_contact",
                "moving_gt_contact_frames",
            )
        ),
        "moving_target_part_local_error_m_on_gt_contact": _round(
            _weighted_mean_or_none(
                rows,
                "moving_target_part_local_error_m_on_gt_contact",
                "moving_gt_contact_frames",
            )
        ),
        "moving_nearest_contact_point_error_m_on_gt_contact": _round(
            _weighted_mean_or_none(
                rows,
                "moving_nearest_contact_point_error_m_on_gt_contact",
                "moving_gt_contact_frames",
            )
        ),
        "moving_nearest_contact_point_error_m_on_both_contact": _round(
            _weighted_mean_or_none(
                rows,
                "moving_nearest_contact_point_error_m_on_both_contact",
                "moving_both_contact_frames",
            )
        ),
        "part_confusion_on_both_contact": _sum_confusion(
            rows,
            "part_confusion_on_both_contact",
        ),
        "moving_part_confusion_on_both_contact": _sum_confusion(
            rows,
            "moving_part_confusion_on_both_contact",
        ),
    }
    return aggregate


def _group_by_subset(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    subsets = sorted({str(r.get("subset")) for r in rows if r.get("subset") is not None})
    return {subset: _aggregate([r for r in rows if r.get("subset") == subset]) for subset in subsets}


def _sum_confusion(rows: list[dict[str, Any]], key: str) -> list[list[int]]:
    n = len(BODY_PART_NAMES)
    out = np.zeros((n, n), dtype=np.int64)
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        out += np.asarray(value, dtype=np.int64)
    return out.tolist()


def measure_alignment(
    *,
    generated_dir: Path,
    gt_dir: Path,
    output_dir: Path,
    detail: str,
    fps: float,
    contact_threshold: float,
) -> dict[str, Any]:
    gen_run = _load_run(generated_dir)
    gt_run = _load_run(gt_dir)
    cfg = ContactConfig(fps=float(fps))

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for seq_id in gen_run.seq_ids:
        if seq_id not in gt_run.seq_to_index:
            missing.append(seq_id)
            continue

        gen_i = gen_run.seq_to_index[seq_id]
        gt_i = gt_run.seq_to_index[seq_id]
        T = min(
            int(gen_run.seq_lens[gen_i]),
            int(gt_run.seq_lens[gt_i]),
            int(gen_run.npz["motion_263"].shape[1]),
            int(gt_run.npz["motion_263"].shape[1]),
        )
        if T < 1:
            continue

        gen_feat = _contact_features(
            run=gen_run,
            index=gen_i,
            T=T,
            cfg=cfg,
            contact_threshold=float(contact_threshold),
        )
        gt_feat = _contact_features(
            run=gt_run,
            index=gt_i,
            T=T,
            cfg=cfg,
            contact_threshold=float(contact_threshold),
        )

        row = _clip_metrics(
            seq_id=seq_id,
            subset=gen_run.meta_by_seq.get(seq_id, {}).get("subset"),
            object_id=gen_run.meta_by_seq.get(seq_id, {}).get("object_id"),
            T=T,
            gen=gen_feat,
            gt=gt_feat,
            contact_threshold=float(contact_threshold),
        )
        rows.append(row)

    worst_target = sorted(
        rows,
        key=lambda r: (
            -float(r.get("moving_target_part_local_error_m_on_gt_contact") or -1.0),
            float(r.get("moving_contact_temporal_iou") or 9.0),
        ),
    )[:10]
    worst_temporal = sorted(
        rows,
        key=lambda r: (
            float(r.get("moving_contact_temporal_iou") if r.get("moving_contact_temporal_iou") is not None else 9.0),
            -int(r.get("moving_gt_contact_frames") or 0),
        ),
    )[:10]

    summary: dict[str, Any] = {
        "schema": f"stage_b_contact_alignment_{detail}_v1",
        "generated_dir": str(generated_dir),
        "gt_dir": str(gt_dir),
        "fps": float(fps),
        "contact_threshold": float(contact_threshold),
        "body_part_names": BODY_PART_NAMES,
        "n_missing_in_gt": len(missing),
        "missing_in_gt": missing if detail == "full" else missing[:10],
        "aggregate": _aggregate(rows),
        "by_subset": _group_by_subset(rows),
        "worst_moving_target_alignment": _compact_rows(worst_target),
        "worst_moving_temporal_iou": _compact_rows(worst_temporal),
    }
    if detail == "full":
        summary["per_clip"] = rows

    ensure_dir(output_dir)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def _compact_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = [
        "seq_id",
        "subset",
        "object_id",
        "T",
        "gt_contact_frame_frac",
        "gen_contact_frame_frac",
        "moving_contact_temporal_iou",
        "moving_contact_recall_on_gt",
        "moving_right_part_contact_recall_on_gt",
        "moving_missed_contact_frac_on_gt",
        "moving_wrong_part_contact_frac_on_gt",
        "moving_target_part_local_error_m_on_gt_contact",
        "moving_same_gt_part_local_position_error_m_on_gt_contact",
        "moving_same_gt_part_surface_dist_m_on_gt_contact",
        "moving_nearest_contact_point_error_m_on_gt_contact",
        "gt_dominant_contact_part",
        "gen_dominant_contact_part",
    ]
    return [{k: row.get(k) for k in keys} for row in rows]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--detail", choices=["compact", "full"], default="compact")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument(
        "--contact-threshold",
        type=float,
        default=0.5,
        help="hard threshold on max(distance-score, kinematic-score)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = measure_alignment(
        generated_dir=args.generated_dir,
        gt_dir=args.gt_dir,
        output_dir=args.output_dir,
        detail=str(args.detail),
        fps=float(args.fps),
        contact_threshold=float(args.contact_threshold),
    )
    agg = summary["aggregate"]
    print(f"Saved: {args.output_dir / 'summary.json'}")
    print(
        "Aggregate: "
        f"temporal_iou={agg.get('contact_temporal_iou')} "
        f"moving_iou={agg.get('moving_contact_temporal_iou')} "
        f"moving_recall={agg.get('moving_contact_recall_on_gt')} "
        f"moving_right_part={agg.get('moving_right_part_contact_recall_on_gt')} "
        f"moving_target_err_m={agg.get('moving_target_part_local_error_m_on_gt_contact')}"
    )


if __name__ == "__main__":
    main()
