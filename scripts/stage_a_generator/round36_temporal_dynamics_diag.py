"""Round-36 temporal dynamics diagnostic for Stage-1 and Stage-1.5.

This CPU diagnostic compares sampled substitute-condition caches against the
oracle tensors used during training, with explicit velocity and acceleration
metrics. It is intentionally upstream-only: no PB1 inference is needed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from piano.data.dataset import collate_hoi
from piano.data.stage1_coarse_oracle import (
    extract_coarse_v1_batched,
    load_stage1_coarse_norm,
)
from piano.inference.sample_substitute_conds import _read_selection
from piano.training.train_anchordiff import _build_dataset


STAGE1_DIM = 23
STAGE1P5_C41_DIM = 18
STAGE1P5_S4_DIM = 13

STAGE1_GROUPS: tuple[tuple[str, slice], ...] = (
    ("root_local_xzy", slice(0, 3)),
    ("velocity_xzy", slice(3, 6)),
    ("yaw_sin_cos_vel", slice(6, 9)),
    ("pelvis_rot6d", slice(9, 15)),
    ("spine3_rot6d", slice(15, 21)),
    ("heights", slice(21, 23)),
    ("all", slice(0, 23)),
)

STAGE1_CHANNELS = (
    "root_local_x", "root_local_z", "root_local_y",
    "vel_x", "vel_z", "vel_y",
    "yaw_sin", "yaw_cos", "yaw_vel",
    "pelvis_r0", "pelvis_r1", "pelvis_r2", "pelvis_r3", "pelvis_r4", "pelvis_r5",
    "spine3_r0", "spine3_r1", "spine3_r2", "spine3_r3", "spine3_r4", "spine3_r5",
    "head_height", "shoulder_center_h",
)

STAGE1P5_GROUPS: tuple[tuple[str, slice], ...] = (
    ("c41_left_wrist", slice(0, 3)),
    ("c41_right_wrist", slice(3, 6)),
    ("c41_knees", slice(6, 12)),
    ("c41_neck", slice(12, 15)),
    ("c41_pelvis", slice(15, 18)),
    ("c41_all", slice(0, 18)),
)

STAGE1P5_CHANNELS = (
    "left_wrist_dx", "left_wrist_dy", "left_wrist_dz",
    "right_wrist_dx", "right_wrist_dy", "right_wrist_dz",
    "left_knee_dx", "left_knee_dy", "left_knee_dz",
    "right_knee_dx", "right_knee_dy", "right_knee_dz",
    "neck_dx", "neck_dy", "neck_dz",
    "pelvis_dx", "pelvis_dz", "pelvis_dy",
)


def _cache_bucket_root(root: Path, bucket: str) -> Path:
    return root / bucket if (root / bucket).is_dir() else root


def _safe_ratio(num: float, den: float) -> float:
    return float(num / den) if abs(den) > 1e-12 else 0.0


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0


def _cat_or_empty(chunks: list[np.ndarray], width: int) -> np.ndarray:
    if not chunks:
        return np.zeros((0, width), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def _diff_chunks(
    chunks: list[np.ndarray], order: int, sl: slice,
) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for arr in chunks:
        x = arr[:, sl]
        if x.shape[0] <= order:
            continue
        out.append(np.diff(x, n=order, axis=0).astype(np.float32))
    return out


def _summarize_group(
    gt_chunks: list[np.ndarray],
    pred_chunks: list[np.ndarray],
    sl: slice,
) -> dict[str, float]:
    width = len(range(*sl.indices(10_000)))
    gt_val = _cat_or_empty([x[:, sl] for x in gt_chunks], width)
    pr_val = _cat_or_empty([x[:, sl] for x in pred_chunks], width)
    gt_vel = _cat_or_empty(_diff_chunks(gt_chunks, 1, sl), width)
    pr_vel = _cat_or_empty(_diff_chunks(pred_chunks, 1, sl), width)
    gt_acc = _cat_or_empty(_diff_chunks(gt_chunks, 2, sl), width)
    pr_acc = _cat_or_empty(_diff_chunks(pred_chunks, 2, sl), width)

    return {
        "value_residual_rms": _rms(pr_val - gt_val),
        "velocity_residual_rms": _rms(pr_vel - gt_vel),
        "acceleration_residual_rms": _rms(pr_acc - gt_acc),
        "gt_value_std": float(gt_val.std()) if gt_val.size else 0.0,
        "pred_value_std": float(pr_val.std()) if pr_val.size else 0.0,
        "value_std_ratio": _safe_ratio(
            float(pr_val.std()) if pr_val.size else 0.0,
            float(gt_val.std()) if gt_val.size else 0.0,
        ),
        "gt_velocity_rms": _rms(gt_vel),
        "pred_velocity_rms": _rms(pr_vel),
        "velocity_rms_ratio": _safe_ratio(_rms(pr_vel), _rms(gt_vel)),
        "gt_acceleration_rms": _rms(gt_acc),
        "pred_acceleration_rms": _rms(pr_acc),
        "acceleration_rms_ratio": _safe_ratio(_rms(pr_acc), _rms(gt_acc)),
    }


def _top_channels_by_acc_residual(
    gt_chunks: list[np.ndarray],
    pred_chunks: list[np.ndarray],
    channel_names: tuple[str, ...],
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for i, name in enumerate(channel_names):
        sl = slice(i, i + 1)
        gt_acc = _cat_or_empty(_diff_chunks(gt_chunks, 2, sl), 1)
        pr_acc = _cat_or_empty(_diff_chunks(pred_chunks, 2, sl), 1)
        gt_vel = _cat_or_empty(_diff_chunks(gt_chunks, 1, sl), 1)
        pr_vel = _cat_or_empty(_diff_chunks(pred_chunks, 1, sl), 1)
        rows.append({
            "channel": name,
            "acceleration_residual_rms": _rms(pr_acc - gt_acc),
            "acceleration_rms_ratio": _safe_ratio(_rms(pr_acc), _rms(gt_acc)),
            "velocity_rms_ratio": _safe_ratio(_rms(pr_vel), _rms(gt_vel)),
        })
    rows.sort(key=lambda r: r["acceleration_residual_rms"], reverse=True)
    return rows[:12]


def _load_stage1_cache(
    root: Path, bucket: str, subset: str, seq_id: str,
) -> tuple[np.ndarray, int]:
    p = _cache_bucket_root(root, bucket) / subset / f"{seq_id}.npz"
    if not p.exists():
        raise FileNotFoundError(p)
    with np.load(p) as data:
        if "stage1_coarse" not in data.files:
            raise KeyError(f"{p}: missing stage1_coarse")
        arr = data["stage1_coarse"].astype(np.float32)
        valid_t = int(data["valid_T"]) if "valid_T" in data.files else arr.shape[0]
    if arr.ndim != 2 or arr.shape[-1] != STAGE1_DIM:
        raise ValueError(f"{p}: expected (T, {STAGE1_DIM}), got {arr.shape}")
    return arr, valid_t


def _load_stage1p5_cache(
    root: Path, bucket: str, subset: str, seq_id: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    p = _cache_bucket_root(root, bucket) / subset / f"{seq_id}.npz"
    if not p.exists():
        raise FileNotFoundError(p)
    with np.load(p) as data:
        for key in ("stage2_coarse_extra", "stage2_support"):
            if key not in data.files:
                raise KeyError(f"{p}: missing {key}")
        c41 = data["stage2_coarse_extra"].astype(np.float32)
        s4 = data["stage2_support"].astype(np.float32)
        valid_t = int(data["valid_T"]) if "valid_T" in data.files else c41.shape[0]
    if c41.ndim != 2 or c41.shape[-1] != STAGE1P5_C41_DIM:
        raise ValueError(f"{p}: expected C41 (T, {STAGE1P5_C41_DIM}), got {c41.shape}")
    if s4.ndim != 2 or s4.shape[-1] != STAGE1P5_S4_DIM:
        raise ValueError(f"{p}: expected S4 (T, {STAGE1P5_S4_DIM}), got {s4.shape}")
    return c41, s4, valid_t


def _collect_stage1(
    cfg, generated_dir: Path, selection_json: Path, bucket: str,
) -> tuple[list[np.ndarray], list[np.ndarray], int]:
    selection = _read_selection(selection_json)
    mean_np, std_np = load_stage1_coarse_norm(str(cfg.data.stage1_coarse_cache_root))
    mean = mean_np.astype(np.float32).reshape(1, STAGE1_DIM)
    std = std_np.astype(np.float32).reshape(1, STAGE1_DIM)

    dataset = _build_dataset(cfg, bucket, augment=False)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=collate_hoi, num_workers=0,
    )
    gt_chunks: list[np.ndarray] = []
    pred_chunks: list[np.ndarray] = []
    for batch in loader:
        subset = str(batch["subset"][0])
        seq_id = str(batch["seq_id"][0])
        if (subset, seq_id) not in selection:
            continue
        pred_z, valid_pred = _load_stage1_cache(generated_dir, bucket, subset, seq_id)
        motion = batch["motion"].float()
        rest_offsets = batch["rest_offsets"].float()
        oracle_raw = extract_coarse_v1_batched(
            motion=motion, rest_offsets=rest_offsets,
        )[0].cpu().numpy().astype(np.float32)
        pred_raw = pred_z * std + mean
        seq_len = int(batch["seq_len"][0].item())
        valid_t = min(seq_len, valid_pred, pred_raw.shape[0], oracle_raw.shape[0])
        gt_chunks.append(oracle_raw[:valid_t])
        pred_chunks.append(pred_raw[:valid_t])
    return gt_chunks, pred_chunks, len(gt_chunks)


def _collect_stage1p5(
    cfg, generated_dir: Path, selection_json: Path, bucket: str,
) -> tuple[list[np.ndarray], list[np.ndarray], int]:
    selection = _read_selection(selection_json)
    dataset = _build_dataset(cfg, bucket, augment=False)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=collate_hoi, num_workers=0,
    )
    gt_chunks: list[np.ndarray] = []
    pred_chunks: list[np.ndarray] = []
    for batch in loader:
        subset = str(batch["subset"][0])
        seq_id = str(batch["seq_id"][0])
        if (subset, seq_id) not in selection:
            continue
        if "stage2_coarse_extra" not in batch:
            raise KeyError("config must surface stage2_coarse_extra=C41-current")
        pred_c41, _pred_s4, valid_pred = _load_stage1p5_cache(
            generated_dir, bucket, subset, seq_id,
        )
        gt_c41 = batch["stage2_coarse_extra"][0].numpy().astype(np.float32)
        seq_len = int(batch["seq_len"][0].item())
        valid_t = min(seq_len, valid_pred, pred_c41.shape[0], gt_c41.shape[0])
        gt_chunks.append(gt_c41[:valid_t])
        pred_chunks.append(pred_c41[:valid_t])
    return gt_chunks, pred_chunks, len(gt_chunks)


def _write_report(out_md: Path, stats: dict) -> None:
    lines = [
        "# Round-36 Temporal Dynamics Diagnostic",
        "",
        f"- stage: {stats['stage']}",
        f"- space: {stats['space']}",
        f"- config: `{stats['config']}`",
        f"- generated_dir: `{stats['generated_dir']}`",
        f"- selection_json: `{stats['selection_json']}`",
        f"- bucket: {stats['bucket']}",
        f"- n_clips: {stats['n_clips']}",
        f"- n_frames: {stats['n_frames']}",
        "",
        "## Group Summary",
        "",
        "| group | value_rms | vel_rms | acc_rms | value_std_ratio | vel_ratio | acc_ratio | gt_acc_rms | pred_acc_rms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in stats["groups"].items():
        lines.append(
            f"| {name} | {row['value_residual_rms']:.5f} | "
            f"{row['velocity_residual_rms']:.5f} | "
            f"{row['acceleration_residual_rms']:.5f} | "
            f"{row['value_std_ratio']:.3f} | "
            f"{row['velocity_rms_ratio']:.3f} | "
            f"{row['acceleration_rms_ratio']:.3f} | "
            f"{row['gt_acceleration_rms']:.5f} | "
            f"{row['pred_acceleration_rms']:.5f} |"
        )
    lines += [
        "",
        "## Top Channels By Acceleration Residual",
        "",
        "| rank | channel | acc_residual_rms | acc_ratio | vel_ratio |",
        "|---:|---|---:|---:|---:|",
    ]
    for i, row in enumerate(stats["top_channels_by_acc_residual"], start=1):
        lines.append(
            f"| {i} | {row['channel']} | "
            f"{row['acceleration_residual_rms']:.5f} | "
            f"{row['acceleration_rms_ratio']:.3f} | "
            f"{row['velocity_rms_ratio']:.3f} |"
        )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["stage1", "stage1p5"], required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--generated-dir", type=Path, required=True)
    ap.add_argument("--selection-json", type=Path, required=True)
    ap.add_argument("--bucket", choices=["train", "val"], default="val")
    ap.add_argument("--out-md", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    cfg = OmegaConf.load(str(args.config))
    if args.stage == "stage1":
        gt_chunks, pred_chunks, n_clips = _collect_stage1(
            cfg, args.generated_dir, args.selection_json, args.bucket,
        )
        groups = STAGE1_GROUPS
        channels = STAGE1_CHANNELS
        space = "Stage-1 raw 23-D coarse (un-z-scored)"
    else:
        gt_chunks, pred_chunks, n_clips = _collect_stage1p5(
            cfg, args.generated_dir, args.selection_json, args.bucket,
        )
        groups = STAGE1P5_GROUPS
        channels = STAGE1P5_CHANNELS
        space = "Stage-1.5 raw C41"

    if n_clips == 0:
        raise SystemExit("no selected clips processed")

    n_frames = int(sum(x.shape[0] for x in gt_chunks))
    stats = {
        "stage": args.stage,
        "space": space,
        "config": str(args.config),
        "generated_dir": str(args.generated_dir),
        "selection_json": str(args.selection_json),
        "bucket": args.bucket,
        "n_clips": n_clips,
        "n_frames": n_frames,
        "groups": {
            name: _summarize_group(gt_chunks, pred_chunks, sl)
            for name, sl in groups
        },
        "top_channels_by_acc_residual": _top_channels_by_acc_residual(
            gt_chunks, pred_chunks, channels,
        ),
    }

    out_json = args.out_json or args.out_md.with_suffix(".json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    _write_report(args.out_md, stats)
    print(f"wrote {args.out_md}")
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
