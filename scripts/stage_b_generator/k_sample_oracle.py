#!/usr/bin/env python
"""K-sample contact oracle for Stage B.

This no-retrain diagnostic asks whether the current generator distribution
already contains good-contact samples. For each fixed validation clip, it
generates K full-condition samples with different RNG seeds, scores each sample
with the same body-to-object contact metric used by Stage B evaluation, and
reports single-sample versus best-of-K contact.

If best-of-K approaches GT roundtrip while sample #0 remains poor, the next
strategy should be reranking/guidance. If best-of-K remains near the current
single-sample band, the learned distribution itself is wrong.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

import piano.models.backbones.momask_adapter  # noqa: F401
from piano.data.eval_sampling import (
    describe_eval_clip_selection,
    resolve_eval_clip_count,
    select_eval_clip_indices,
)
from piano.data.pseudo_labels.extract_contact import ContactConfig
from piano.data.humanml3d_repr import load_motion_stats
from piano.training.decoded_contact_loss import (
    _object_motion_speed_from_canonical,
    body_canonical_to_object_local_torch,
)
from piano.utils.io_utils import ensure_dir, save_json
from piano.utils.smpl_utils import BODY_PART_INDICES, BODY_PART_NAMES

# Reuse the load-bearing Stage B offline eval helpers. This file lives in the
# same directory as qual_eval.py, so direct script execution puts that directory
# on sys.path.
from qual_eval import (  # type: ignore
    _build_model,
    _build_val_dataset,
    _generate,
    _get_canon_to_world_transform,
    _save_condition_dir,
    _tokenize_z_int,
)
from measure_temporal_coupling import score_motion_temporal_coupling  # type: ignore
from utils.motion_process import recover_from_ric


def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cm(x_m: float) -> float:
    return round(float(x_m) * 100.0, 4)


def _summary_stats(values_m: list[float]) -> dict[str, float]:
    if not values_m:
        return {
            "mean_m": 0.0,
            "median_m": 0.0,
            "p25_m": 0.0,
            "p75_m": 0.0,
            "mean_cm": 0.0,
            "median_cm": 0.0,
            "p25_cm": 0.0,
            "p75_cm": 0.0,
        }
    arr = np.asarray(values_m, dtype=np.float64)
    return {
        "mean_m": round(float(arr.mean()), 6),
        "median_m": round(float(np.median(arr)), 6),
        "p25_m": round(float(np.percentile(arr, 25)), 6),
        "p75_m": round(float(np.percentile(arr, 75)), 6),
        "mean_cm": _cm(float(arr.mean())),
        "median_cm": _cm(float(np.median(arr))),
        "p25_cm": _cm(float(np.percentile(arr, 25))),
        "p75_cm": _cm(float(np.percentile(arr, 75))),
    }


def _mean_finite(values: list[float | None]) -> float | None:
    xs = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not xs:
        return None
    return float(np.mean(xs))


def _round_or_none(value: float | None, ndigits: int = 6) -> float | None:
    if value is None or not np.isfinite(float(value)):
        return None
    return round(float(value), ndigits)


def _cm_or_none(value_m: float | None) -> float | None:
    if value_m is None or not np.isfinite(float(value_m)):
        return None
    return _cm(float(value_m))


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    single = [float(r["single_sample_dist_m"]) for r in rows]
    best = [float(r["best_dist_m"]) for r in rows]
    sample_means = [float(r["sample_mean_dist_m"]) for r in rows]
    improvements = [s - b for s, b in zip(single, best)]

    out: dict[str, Any] = {
        "n_clips": len(rows),
        "single_sample": _summary_stats(single),
        "sample_mean": _summary_stats(sample_means),
        "best_of_k": _summary_stats(best),
        "improvement_single_minus_best": _summary_stats(improvements),
        "best_under_25cm_frac": (
            round(float(np.mean(np.asarray(best) <= 0.25)), 4) if best else 0.0
        ),
        "best_under_22cm_frac": (
            round(float(np.mean(np.asarray(best) <= 0.22)), 4) if best else 0.0
        ),
    }
    for key in (
        "best_alignment_primary_error_m",
        "best_alignment_target_error_m",
        "best_alignment_moving_target_error_m",
        "best_alignment_same_part_recall",
        "best_alignment_moving_same_part_recall",
        "best_alignment_contact_part_frame_frac",
        "best_alignment_moving_contact_part_frame_frac",
    ):
        value = _mean_finite([r.get(key) for r in rows])
        if value is not None:
            out[key] = round(float(value), 6)
            if key.endswith("_m"):
                out[key.replace("_m", "_cm")] = _cm(float(value))
    return out


def _aggregate_by_subset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_subset: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_subset.setdefault(str(row["subset"]), []).append(row)
    return {
        subset: _aggregate_rows(sub_rows)
        for subset, sub_rows in sorted(by_subset.items())
    }


def _selection_score(
    *,
    dist_m: float,
    temporal: dict[str, Any],
    metric: str,
    coupling_weight: float,
    uncoupled_penalty: float,
    min_moving_frame_frac: float,
) -> float:
    if metric == "distance":
        return float(dist_m)

    moving_frac = temporal.get("moving_frame_frac")
    coupled = temporal.get("moving_coupled_frame_frac")
    uncoupled = temporal.get("moving_close_but_uncoupled_frac")
    if moving_frac is None or float(moving_frac) < float(min_moving_frame_frac):
        return float(dist_m)
    if coupled is None:
        return float(dist_m)

    uncoupled_term = float(uncoupled) if uncoupled is not None else 0.0
    return (
        float(dist_m)
        + float(coupling_weight) * (1.0 - float(coupled))
        + float(uncoupled_penalty) * uncoupled_term
    )


def _weighted_mean_or_none(values: torch.Tensor, weights: torch.Tensor) -> float | None:
    weights = weights.to(device=values.device, dtype=values.dtype)
    denom = weights.sum()
    if float(denom.detach().cpu()) <= 1e-6:
        return None
    value = (values * weights).sum() / denom.clamp(min=1e-6)
    return float(value.detach().cpu())


def _fraction_or_none(mask: torch.Tensor, denom_mask: torch.Tensor) -> float | None:
    denom = int(denom_mask.sum().detach().cpu())
    if denom <= 0:
        return None
    num = int((mask & denom_mask).sum().detach().cpu())
    return float(num) / float(denom)


@torch.no_grad()
def _score_target_alignment(
    *,
    motion_263_generated: np.ndarray,
    sample: dict[str, Any],
    seq_len: int,
    device: torch.device,
    fps: float,
    moving_speed_threshold: float,
    kin_radius_proxy: float,
    contact_threshold: float,
) -> dict[str, Any]:
    """Score generated motion against part-specific GT contact targets.

    This is stricter than the legacy contact metric: it never minimizes over
    arbitrary body parts or arbitrary object points. It asks whether the body
    part indicated by ``contact_state`` reaches that part's
    ``contact_target_xyz`` in object-local coordinates.
    """
    if "contact_state" not in sample or "contact_target_xyz" not in sample:
        return {}
    if "obj_com_canonical" not in sample or "obj_rot6d_canonical" not in sample:
        return {}

    T = min(
        int(seq_len),
        int(motion_263_generated.shape[0]),
        int(sample["contact_state"].shape[0]),
        int(sample["contact_target_xyz"].shape[0]),
        int(sample["obj_com_canonical"].shape[0]),
        int(sample["obj_rot6d_canonical"].shape[0]),
    )
    if T < 1:
        return {}

    motion_t = torch.from_numpy(motion_263_generated[:T]).float().unsqueeze(0).to(device)
    body_idx = torch.as_tensor(BODY_PART_INDICES, device=device, dtype=torch.long)
    joints = recover_from_ric(motion_t, 22).float()
    body = joints.index_select(dim=2, index=body_idx)

    obj_com = sample["obj_com_canonical"][:T].unsqueeze(0).to(device=device, dtype=body.dtype)
    obj_rot6d = sample["obj_rot6d_canonical"][:T].unsqueeze(0).to(device=device, dtype=body.dtype)
    body_local = body_canonical_to_object_local_torch(body, obj_com, obj_rot6d)

    target = sample["contact_target_xyz"][:T].unsqueeze(0).to(device=device, dtype=body.dtype)
    contact = sample["contact_state"][:T].unsqueeze(0).to(device=device, dtype=body.dtype)
    contact_binary = contact >= float(contact_threshold)
    frame_mask = torch.arange(T, device=device).view(1, T, 1) < int(seq_len)
    valid_part = contact_binary & frame_mask
    valid_weights = contact.clamp(min=0.0, max=1.0) * valid_part.to(dtype=body.dtype)

    pos_dist = torch.linalg.vector_norm(body_local - target, dim=-1)

    cfg = ContactConfig(fps=float(fps), kin_radius_proxy=float(kin_radius_proxy))
    thresholds = torch.tensor(
        [cfg.distance_thresholds[name] for name in BODY_PART_NAMES],
        device=device,
        dtype=body.dtype,
    ).view(1, 1, -1)
    same_part_hit = pos_dist <= thresholds

    obj_speed = _object_motion_speed_from_canonical(
        obj_com,
        obj_rot6d,
        fps=float(fps),
        radius_proxy=float(kin_radius_proxy),
    )
    moving = obj_speed >= float(moving_speed_threshold)
    moving_valid = valid_part & moving[:, :, None]
    moving_weights = valid_weights * moving[:, :, None].to(dtype=body.dtype)

    target_error = _weighted_mean_or_none(pos_dist, valid_weights)
    moving_target_error = _weighted_mean_or_none(pos_dist, moving_weights)
    same_part_recall = _fraction_or_none(same_part_hit, valid_part)
    moving_same_part_recall = _fraction_or_none(same_part_hit, moving_valid)

    primary_error = (
        moving_target_error
        if moving_target_error is not None
        else target_error
    )
    primary_recall = (
        moving_same_part_recall
        if moving_same_part_recall is not None
        else same_part_recall
    )

    valid_frames = torch.any(valid_part, dim=-1)
    moving_valid_frames = torch.any(moving_valid, dim=-1)
    moving_frames = moving & (torch.arange(T, device=device).view(1, T) < int(seq_len))

    return {
        "alignment_primary_error_m": _round_or_none(primary_error),
        "alignment_primary_error_cm": _cm_or_none(primary_error),
        "alignment_target_error_m": _round_or_none(target_error),
        "alignment_target_error_cm": _cm_or_none(target_error),
        "alignment_moving_target_error_m": _round_or_none(moving_target_error),
        "alignment_moving_target_error_cm": _cm_or_none(moving_target_error),
        "alignment_same_part_recall": _round_or_none(same_part_recall, ndigits=4),
        "alignment_moving_same_part_recall": _round_or_none(
            moving_same_part_recall,
            ndigits=4,
        ),
        "alignment_contact_part_frames": int(valid_frames.sum().detach().cpu()),
        "alignment_moving_contact_part_frames": int(moving_valid_frames.sum().detach().cpu()),
        "alignment_moving_frames": int(moving_frames.sum().detach().cpu()),
        "alignment_contact_part_frame_frac": _round_or_none(
            float(valid_frames.float().mean().detach().cpu()),
            ndigits=4,
        ),
        "alignment_moving_contact_part_frame_frac": _round_or_none(
            (
                float(moving_valid_frames.sum().detach().cpu())
                / max(int(moving_frames.sum().detach().cpu()), 1)
            ),
            ndigits=4,
        ),
    }


def _alignment_selection_score(
    *,
    alignment: dict[str, Any],
    dist_m: float,
    temporal: dict[str, Any],
    recall_penalty: float,
    distance_weight: float,
    coupling_weight: float,
) -> float:
    primary_error = alignment.get("alignment_primary_error_m")
    if primary_error is None:
        return float(dist_m)
    recall = alignment.get("alignment_moving_same_part_recall")
    if recall is None:
        recall = alignment.get("alignment_same_part_recall")
    recall_term = 1.0 - float(recall if recall is not None else 0.0)
    coupled = temporal.get("moving_coupled_frame_frac")
    coupling_term = 1.0 - float(coupled) if coupled is not None else 0.0
    return (
        float(primary_error)
        + float(recall_penalty) * float(recall_term)
        + float(distance_weight) * float(dist_m)
        + float(coupling_weight) * float(coupling_term)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "runs/sweeps/stageB_v12_decoded_contact_weight_sweep/configs/"
            "generator_v12_decoded_contact_w02_diagnostics.yaml",
        ),
        help="Training config that matches the checkpoint.",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=Path("runs/training/generator_v12_decoded_contact_w02_diagnostics/best_val.pt"),
        help="Stage B checkpoint to diagnose.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/eval/stageB_v0_12_w02_bv_k_sample_oracle"))
    parser.add_argument("--num-clips", type=int, default=80)
    parser.add_argument(
        "--num-clips-per-subset",
        type=int,
        default=20,
        help="When metadata is available, overrides --num-clips as N per subset.",
    )
    parser.add_argument("--k", type=int, default=16, help="Number of samples per clip.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--w-text", type=float, default=4.0)
    parser.add_argument("--w-int", type=float, default=2.0)
    parser.add_argument("--timesteps", type=int, default=10)
    parser.add_argument("--res-cond-scale", type=float, default=2.0)
    parser.add_argument(
        "--selection-metric",
        choices=("distance", "composite", "alignment"),
        default="distance",
        help="How to choose the saved best sample. 'distance' preserves the original oracle. "
             "'composite' adds a penalty for weak moving-object kinematic coupling. "
             "'alignment' chooses the sample closest to the GT contact body part's "
             "object-local target trajectory.",
    )
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--coupling-threshold", type=float, default=0.5)
    parser.add_argument("--moving-speed-threshold", type=float, default=None)
    parser.add_argument(
        "--min-moving-frame-frac",
        type=float,
        default=0.05,
        help="Use distance-only scoring when fewer than this fraction of frames have moving objects.",
    )
    parser.add_argument(
        "--coupling-weight",
        type=float,
        default=0.12,
        help="Composite penalty in meters for no moving-object kinematic coupling.",
    )
    parser.add_argument(
        "--uncoupled-penalty",
        type=float,
        default=0.05,
        help="Composite penalty in meters for moving frames that are close but uncoupled.",
    )
    parser.add_argument(
        "--alignment-recall-penalty",
        type=float,
        default=0.25,
        help="Alignment score penalty in meters for missing the GT contact part.",
    )
    parser.add_argument(
        "--alignment-distance-weight",
        type=float,
        default=0.05,
        help="Small tie-break weight on legacy mean-min distance for alignment selection.",
    )
    parser.add_argument(
        "--alignment-coupling-weight",
        type=float,
        default=0.0,
        help="Optional tie-break penalty in meters for weak moving-object coupling.",
    )
    parser.add_argument(
        "--alignment-contact-threshold",
        type=float,
        default=0.5,
        help="Hard threshold on pseudo-label contact_state for alignment scoring.",
    )
    parser.add_argument(
        "--alignment-moving-speed-threshold",
        type=float,
        default=None,
        help="Object speed threshold for moving-contact alignment; default mirrors --moving-speed-threshold or 0.15.",
    )
    parser.add_argument(
        "--alignment-kin-radius-proxy",
        type=float,
        default=0.3,
        help="Radius proxy for rotational object speed in alignment scoring.",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--save-best",
        action="store_true",
        help="Save best-of-K motions as output-dir/best/generated.npz for visualization.",
    )
    args = parser.parse_args()

    if args.k <= 0:
        raise ValueError("--k must be positive")

    _set_all_seeds(args.seed)
    ensure_dir(args.output_dir)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = OmegaConf.load(args.config)
    alignment_moving_speed_threshold = (
        float(args.alignment_moving_speed_threshold)
        if args.alignment_moving_speed_threshold is not None
        else (
            float(args.moving_speed_threshold)
            if args.moving_speed_threshold is not None
            else 0.15
        )
    )

    print(f"Loading model from {args.ckpt} on {device} ...")
    transformer, vq_model, res_transformer, token_stride = _build_model(cfg, args.ckpt, device)
    motion_mean, motion_std = load_motion_stats(cfg.model.checkpoints.vq_vae)

    val_dataset = _build_val_dataset(cfg)
    num_clips = resolve_eval_clip_count(
        val_dataset,
        num_clips=int(args.num_clips),
        num_clips_per_subset=(
            int(args.num_clips_per_subset)
            if args.num_clips_per_subset is not None and int(args.num_clips_per_subset) > 0
            else None
        ),
    )
    sampled_idx = select_eval_clip_indices(val_dataset, num_clips, seed=args.seed)
    selected_rows = describe_eval_clip_selection(val_dataset, sampled_idx)
    samples = [val_dataset[i] for i in sampled_idx]

    print(f"Selected {len(samples)} stratified clips; K={args.k}")
    for row in selected_rows:
        print(
            "  "
            f"idx={row['index']} subset={row['subset']} "
            f"object={row['object_id']} seq={row['seq_id']}"
        )

    per_clip: list[dict[str, Any]] = []
    best_save_rows: list[dict[str, dict[str, Any]]] = []
    texts: list[str] = []
    seq_ids: list[str] = []
    seq_lens_frames: list[int] = []
    object_pcs: list[np.ndarray] = []
    object_positions: list[np.ndarray] = []
    object_rotations: list[np.ndarray] = []
    world_R_y: list[float] = []
    world_T_xz: list[np.ndarray] = []

    for clip_i, sample in enumerate(samples):
        seq_len = int(sample["seq_len"].item())
        if seq_len < token_stride:
            print(f"  [{clip_i + 1}/{len(samples)}] skip short clip seq_len={seq_len}")
            continue

        text = str(sample["text"])
        seq_id = str(sample["seq_id"])
        subset = str(selected_rows[clip_i]["subset"])
        object_id = str(selected_rows[clip_i]["object_id"])
        m_lens_tok = torch.tensor([max(1, seq_len // token_stride)], dtype=torch.long, device=device)
        int_kv, int_pad = _tokenize_z_int(transformer, sample, device)

        joints_world = sample["joints"].cpu().numpy().astype(np.float32)
        motion_src = sample["motion"].cpu().numpy().astype(np.float32)
        R_y, T_xz = _get_canon_to_world_transform(joints_world[:seq_len], motion_src[:seq_len])
        obj_pc = sample["object_pc"].cpu().numpy().astype(np.float32)
        obj_pos = (
            sample["object_positions"].cpu().numpy().astype(np.float32)
            if "object_positions" in sample
            else np.zeros((seq_len, 3), dtype=np.float32)
        )
        obj_rot = (
            sample["object_rotations"].cpu().numpy().astype(np.float32)
            if "object_rotations" in sample
            else np.zeros((seq_len, 3), dtype=np.float32)
        )

        sample_rows: list[dict[str, Any]] = []
        best_motion: np.ndarray | None = None
        best_base: np.ndarray | None = None
        best_score = float("inf")
        best_dist = float("inf")
        best_sample_idx = -1
        best_seed = -1

        print(f"  [{clip_i + 1}/{len(samples)}] {seq_id} subset={subset}")
        for k_i in range(args.k):
            sample_seed = int(args.seed + clip_i * 100_000 + k_i)
            _set_all_seeds(sample_seed)
            motion_gen, base_ids = _generate(
                transformer,
                vq_model,
                res_transformer,
                text=text,
                int_kv=int_kv,
                int_pad=int_pad,
                m_lens_tok=m_lens_tok,
                w_text=float(args.w_text),
                w_int=float(args.w_int),
                motion_mean=motion_mean,
                motion_std=motion_std,
                timesteps=int(args.timesteps),
                res_cond_scale=float(args.res_cond_scale),
                device=device,
            )
            temporal = score_motion_temporal_coupling(
                motion_263_generated=motion_gen,
                R_y_angle=R_y,
                T_xz=T_xz,
                object_pc_local=obj_pc,
                object_positions=obj_pos,
                object_rotations=obj_rot,
                seq_len=seq_len,
                fps=float(args.fps),
                coupling_threshold=float(args.coupling_threshold),
                moving_speed_threshold=args.moving_speed_threshold,
            )
            dist_m = float(temporal["mean_min_dist_m"])
            alignment: dict[str, Any] = {}
            if str(args.selection_metric) == "alignment":
                alignment = _score_target_alignment(
                    motion_263_generated=motion_gen,
                    sample=sample,
                    seq_len=seq_len,
                    device=device,
                    fps=float(args.fps),
                    moving_speed_threshold=float(alignment_moving_speed_threshold),
                    kin_radius_proxy=float(args.alignment_kin_radius_proxy),
                    contact_threshold=float(args.alignment_contact_threshold),
                )
                score = _alignment_selection_score(
                    alignment=alignment,
                    dist_m=dist_m,
                    temporal=temporal,
                    recall_penalty=float(args.alignment_recall_penalty),
                    distance_weight=float(args.alignment_distance_weight),
                    coupling_weight=float(args.alignment_coupling_weight),
                )
            else:
                score = _selection_score(
                    dist_m=dist_m,
                    temporal=temporal,
                    metric=str(args.selection_metric),
                    coupling_weight=float(args.coupling_weight),
                    uncoupled_penalty=float(args.uncoupled_penalty),
                    min_moving_frame_frac=float(args.min_moving_frame_frac),
                )
            sample_row = {
                "sample_index": k_i,
                "seed": sample_seed,
                "dist_m": round(float(dist_m), 6),
                "dist_cm": _cm(float(dist_m)),
                "selection_score": round(float(score), 6),
                "moving_frame_frac": temporal.get("moving_frame_frac"),
                "moving_close_frame_frac": temporal.get("moving_close_frame_frac"),
                "moving_coupled_frame_frac": temporal.get("moving_coupled_frame_frac"),
                "moving_close_but_uncoupled_frac": temporal.get("moving_close_but_uncoupled_frac"),
            }
            sample_row.update(alignment)
            sample_rows.append(sample_row)
            if score < best_score:
                best_score = float(score)
                best_dist = float(dist_m)
                best_sample_idx = k_i
                best_seed = sample_seed
                best_motion = motion_gen
                best_base = base_ids

        dists = [float(r["dist_m"]) for r in sample_rows]
        scores = [float(r["selection_score"]) for r in sample_rows]
        min_dist_idx = int(np.argmin(dists))
        best_sample_row = sample_rows[best_sample_idx]
        row = {
            "index": int(selected_rows[clip_i]["index"]),
            "subset": subset,
            "object_id": object_id,
            "seq_id": seq_id,
            "text": text,
            "seq_len": seq_len,
            "k": int(args.k),
            "samples": sample_rows,
            "single_sample_dist_m": round(float(dists[0]), 6),
            "single_sample_dist_cm": _cm(float(dists[0])),
            "sample_mean_dist_m": round(float(np.mean(dists)), 6),
            "sample_mean_dist_cm": _cm(float(np.mean(dists))),
            "sample_median_dist_m": round(float(np.median(dists)), 6),
            "sample_median_dist_cm": _cm(float(np.median(dists))),
            "selection_metric": str(args.selection_metric),
            "best_sample_index": int(best_sample_idx),
            "best_seed": int(best_seed),
            "best_selection_score": round(float(best_score), 6),
            "best_dist_m": round(float(best_dist), 6),
            "best_dist_cm": _cm(float(best_dist)),
            "best_moving_frame_frac": best_sample_row.get("moving_frame_frac"),
            "best_moving_close_frame_frac": best_sample_row.get("moving_close_frame_frac"),
            "best_moving_coupled_frame_frac": best_sample_row.get("moving_coupled_frame_frac"),
            "best_moving_close_but_uncoupled_frac": best_sample_row.get("moving_close_but_uncoupled_frac"),
            "best_alignment_primary_error_m": best_sample_row.get("alignment_primary_error_m"),
            "best_alignment_primary_error_cm": best_sample_row.get("alignment_primary_error_cm"),
            "best_alignment_target_error_m": best_sample_row.get("alignment_target_error_m"),
            "best_alignment_target_error_cm": best_sample_row.get("alignment_target_error_cm"),
            "best_alignment_moving_target_error_m": best_sample_row.get("alignment_moving_target_error_m"),
            "best_alignment_moving_target_error_cm": best_sample_row.get("alignment_moving_target_error_cm"),
            "best_alignment_same_part_recall": best_sample_row.get("alignment_same_part_recall"),
            "best_alignment_moving_same_part_recall": best_sample_row.get("alignment_moving_same_part_recall"),
            "best_alignment_contact_part_frame_frac": best_sample_row.get("alignment_contact_part_frame_frac"),
            "best_alignment_moving_contact_part_frame_frac": best_sample_row.get("alignment_moving_contact_part_frame_frac"),
            "min_dist_sample_index": int(min_dist_idx),
            "min_dist_m": round(float(dists[min_dist_idx]), 6),
            "min_dist_cm": _cm(float(dists[min_dist_idx])),
            "min_selection_score": round(float(np.min(scores)), 6),
            "improvement_single_minus_best_m": round(float(dists[0] - best_dist), 6),
            "improvement_single_minus_best_cm": _cm(float(dists[0] - best_dist)),
        }
        per_clip.append(row)
        print(
            f"    single={row['single_sample_dist_cm']:.2f}cm "
            f"mean={row['sample_mean_dist_cm']:.2f}cm "
            f"best={row['best_dist_cm']:.2f}cm "
            f"score={row['best_selection_score']:.3f} "
            f"(sample {best_sample_idx}, seed {best_seed})"
        )
        if str(args.selection_metric) == "alignment":
            print(
                "    alignment: "
                f"primary={row.get('best_alignment_primary_error_cm')}cm "
                f"moving_target={row.get('best_alignment_moving_target_error_cm')}cm "
                f"moving_recall={row.get('best_alignment_moving_same_part_recall')}"
            )

        if args.save_best:
            assert best_motion is not None
            assert best_base is not None
            best_save_rows.append({
                "full": {
                    "motion": best_motion,
                    "base": best_base,
                    "swap_from": None,
                },
            })
            texts.append(text)
            seq_ids.append(seq_id)
            seq_lens_frames.append(seq_len)
            object_pcs.append(obj_pc)
            object_positions.append(obj_pos)
            object_rotations.append(obj_rot)
            world_R_y.append(float(R_y))
            world_T_xz.append(np.asarray(T_xz, dtype=np.float32))

    summary = {
        "diagnostic": "stageB_k_sample_oracle",
        "config": str(args.config),
        "ckpt": str(args.ckpt),
        "seed": int(args.seed),
        "k": int(args.k),
        "num_clips_requested": int(args.num_clips),
        "num_clips_per_subset": int(args.num_clips_per_subset),
        "num_clips_evaluated": len(per_clip),
        "w_text": float(args.w_text),
        "w_int": float(args.w_int),
        "timesteps": int(args.timesteps),
        "res_cond_scale": float(args.res_cond_scale),
        "selection_metric": str(args.selection_metric),
        "fps": float(args.fps),
        "coupling_threshold": float(args.coupling_threshold),
        "moving_speed_threshold": args.moving_speed_threshold,
        "min_moving_frame_frac": float(args.min_moving_frame_frac),
        "coupling_weight": float(args.coupling_weight),
        "uncoupled_penalty": float(args.uncoupled_penalty),
        "alignment_recall_penalty": float(args.alignment_recall_penalty),
        "alignment_distance_weight": float(args.alignment_distance_weight),
        "alignment_coupling_weight": float(args.alignment_coupling_weight),
        "alignment_contact_threshold": float(args.alignment_contact_threshold),
        "alignment_moving_speed_threshold": float(alignment_moving_speed_threshold),
        "alignment_kin_radius_proxy": float(args.alignment_kin_radius_proxy),
        "clip_selection": selected_rows,
        "aggregate": _aggregate_rows(per_clip),
        "by_subset": _aggregate_by_subset(per_clip),
        "per_clip": per_clip,
    }
    save_json(args.output_dir / "summary.json", summary)
    print(f"Wrote {args.output_dir / 'summary.json'}")

    if args.save_best and best_save_rows:
        best_dir = args.output_dir / "best"
        _save_condition_dir(
            best_dir,
            best_save_rows,
            "full",
            texts,
            seq_lens_frames,
            seq_ids,
            object_pcs=object_pcs,
            object_positions=object_positions,
            object_rotations=object_rotations,
            world_R_y=world_R_y,
            world_T_xz=world_T_xz,
        )
        print(f"Wrote best-of-K visualization run to {best_dir}")

    agg = summary["aggregate"]
    print("\n=== K-sample oracle ===")
    print(f"clips: {agg['n_clips']}  K: {args.k}")
    print(f"single sample mean: {agg['single_sample']['mean_cm']:.2f} cm")
    print(f"sample mean:        {agg['sample_mean']['mean_cm']:.2f} cm")
    print(f"best-of-K mean:     {agg['best_of_k']['mean_cm']:.2f} cm")
    print(
        "single-best gain:  "
        f"{agg['improvement_single_minus_best']['mean_cm']:.2f} cm"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
