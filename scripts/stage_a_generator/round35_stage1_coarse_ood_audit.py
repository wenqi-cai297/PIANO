"""Round-35 Stage-1 coarse OOD audit.

Compares oracle ``stage1_coarse`` (computed from GT motion, z-scored) with
generated Stage-1 substitute caches. This is the cheap diagnostic to run before
training a cascade-aware Stage-1.5 variant.

Input generated cache schema::

    <generated-dir>/<bucket>/<subset>/<seq_id>.npz
        stage1_coarse: (T, 23), z-scored
        valid_T: int

The script emits a Markdown report plus JSON stats.
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


STAGE1_COARSE_DIM = 23
FPS_DEFAULT = 20.0

GROUPS: tuple[tuple[str, slice], ...] = (
    ("root_local_xzy", slice(0, 3)),
    ("velocity_xzy", slice(3, 6)),
    ("yaw_sin_cos_vel", slice(6, 9)),
    ("pelvis_rot6d", slice(9, 15)),
    ("spine3_rot6d", slice(15, 21)),
    ("heights", slice(21, 23)),
    ("all", slice(0, 23)),
)

CHANNEL_NAMES = (
    "root_local_x", "root_local_z", "root_local_y",
    "vel_x", "vel_z", "vel_y",
    "yaw_sin", "yaw_cos", "yaw_vel",
    "pelvis_r0", "pelvis_r1", "pelvis_r2", "pelvis_r3", "pelvis_r4", "pelvis_r5",
    "spine3_r0", "spine3_r1", "spine3_r2", "spine3_r3", "spine3_r4", "spine3_r5",
    "head_height", "shoulder_center_h",
)


def _cache_bucket_root(root: Path, bucket: str) -> Path:
    """Accept either cache root or already-bucketed dir."""
    return root / bucket if (root / bucket).is_dir() else root


def _load_stage1_cache(root: Path, bucket: str, subset: str, seq_id: str) -> tuple[np.ndarray, int]:
    p = _cache_bucket_root(root, bucket) / subset / f"{seq_id}.npz"
    if not p.exists():
        raise FileNotFoundError(p)
    data = np.load(p)
    if "stage1_coarse" not in data.files:
        raise KeyError(f"{p}: missing stage1_coarse (keys={list(data.files)})")
    arr = data["stage1_coarse"].astype(np.float32)
    if arr.ndim != 2 or arr.shape[-1] != STAGE1_COARSE_DIM:
        raise ValueError(f"{p}: expected (T, 23), got {arr.shape}")
    valid_T = int(data["valid_T"]) if "valid_T" in data.files else arr.shape[0]
    return arr, valid_T


def _safe_ratio(num: float, den: float) -> float:
    return float(num / den) if abs(den) > 1e-12 else 0.0


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0


def _band_energy(x: np.ndarray, fps: float) -> dict[str, float]:
    """FFT band energy over a valid (T, C) prefix."""
    if x.shape[0] <= 1:
        return {"low_0_1": 0.0, "mid_1_4": 0.0, "high_4_10": 0.0, "total": 0.0}
    freqs = np.fft.rfftfreq(x.shape[0], d=1.0 / fps)
    X = np.fft.rfft(x.astype(np.float32), axis=0)
    e = np.abs(X) ** 2
    bands = {
        "low_0_1": freqs <= 1.0,
        "mid_1_4": (freqs > 1.0) & (freqs <= 4.0),
        "high_4_10": freqs > 4.0,
        "total": np.ones_like(freqs, dtype=bool),
    }
    return {k: float(e[m].sum()) for k, m in bands.items()}


def _collect_clip_pairs(cfg, generated_dir: Path, selection_json: Path, bucket: str):
    sel_pairs = _read_selection(selection_json)
    dataset = _build_dataset(cfg, bucket, augment=False)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=collate_hoi, num_workers=0,
    )
    for batch in loader:
        subset = str(batch["subset"][0])
        seq_id = str(batch["seq_id"][0])
        if (subset, seq_id) not in sel_pairs:
            continue
        pred, vt_pred = _load_stage1_cache(generated_dir, bucket, subset, seq_id)
        motion = batch["motion"].float()
        rest_offsets = batch["rest_offsets"].float()
        oracle_raw = extract_coarse_v1_batched(
            motion=motion, rest_offsets=rest_offsets,
        )[0].cpu().numpy().astype(np.float32)
        seq_len = int(batch["seq_len"][0].item())
        valid_T = min(seq_len, vt_pred, pred.shape[0], oracle_raw.shape[0])
        yield subset, seq_id, oracle_raw, pred, valid_T


def _summarize_group(oracle: np.ndarray, pred: np.ndarray, sl: slice, fps: float) -> dict:
    gt = oracle[:, sl]
    pr = pred[:, sl]
    res = pr - gt
    gt_std = float(gt.std())
    pr_std = float(pr.std())
    gt_vel = np.diff(gt, axis=0) if gt.shape[0] > 1 else gt[:0]
    pr_vel = np.diff(pr, axis=0) if pr.shape[0] > 1 else pr[:0]
    gt_e = _band_energy(gt, fps=fps)
    pr_e = _band_energy(pr, fps=fps)
    res_e = _band_energy(res, fps=fps)
    return {
        "mae": float(np.mean(np.abs(res))),
        "rms": _rms(res),
        "gt_std": gt_std,
        "pred_std": pr_std,
        "std_ratio": _safe_ratio(pr_std, gt_std),
        "gt_vel_rms": _rms(gt_vel),
        "pred_vel_rms": _rms(pr_vel),
        "vel_rms_ratio": _safe_ratio(_rms(pr_vel), _rms(gt_vel)),
        "psd_pred_gt_ratio": {
            k: _safe_ratio(pr_e[k], gt_e[k]) for k in ("low_0_1", "mid_1_4", "high_4_10", "total")
        },
        "psd_residual_gt_ratio": {
            k: _safe_ratio(res_e[k], gt_e[k]) for k in ("low_0_1", "mid_1_4", "high_4_10", "total")
        },
    }


def _write_report(out_md: Path, stats: dict) -> None:
    lines: list[str] = [
        "# Round-35 Stage-1 Coarse OOD Audit",
        "",
        f"- config: `{stats['config']}`",
        f"- generated_dir: `{stats['generated_dir']}`",
        f"- selection_json: `{stats['selection_json']}`",
        f"- bucket: {stats['bucket']}",
        f"- n_clips: {stats['n_clips']}",
        f"- fps: {stats['fps']}",
        "",
        "## Group Summary",
        "",
        "| group | rms | mae | std_ratio | vel_ratio | pred/gt PSD low | pred/gt PSD mid | pred/gt PSD high | residual/gt PSD low |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in stats["groups"].items():
        pr = row["psd_pred_gt_ratio"]
        rr = row["psd_residual_gt_ratio"]
        lines.append(
            f"| {name} | {row['rms']:.4f} | {row['mae']:.4f} | "
            f"{row['std_ratio']:.3f} | {row['vel_rms_ratio']:.3f} | "
            f"{pr['low_0_1']:.3f} | {pr['mid_1_4']:.3f} | {pr['high_4_10']:.3f} | "
            f"{rr['low_0_1']:.3f} |"
        )
    lines.extend(["", "## Top Channels By Residual RMS", ""])
    lines.append("| rank | channel | residual_rms | pred_std / gt_std |")
    lines.append("|---:|---|---:|---:|")
    for i, row in enumerate(stats["top_channels_by_rms"], start=1):
        lines.append(
            f"| {i} | {row['channel']} | {row['residual_rms']:.4f} | {row['std_ratio']:.3f} |"
        )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--generated-dir", type=Path, required=True)
    ap.add_argument("--selection-json", type=Path, required=True)
    ap.add_argument("--bucket", choices=["train", "val"], default="val")
    ap.add_argument("--out-md", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--fps", type=float, default=FPS_DEFAULT)
    args = ap.parse_args()

    cfg = OmegaConf.load(str(args.config))
    mean_np, std_np = load_stage1_coarse_norm(str(cfg.data.stage1_coarse_cache_root))
    mean = mean_np.astype(np.float32).reshape(1, STAGE1_COARSE_DIM)
    std = std_np.astype(np.float32).reshape(1, STAGE1_COARSE_DIM)

    oracle_all: list[np.ndarray] = []
    pred_all: list[np.ndarray] = []
    n_clips = 0
    for _subset, _seq_id, oracle_raw, pred, valid_T in _collect_clip_pairs(
        cfg, args.generated_dir, args.selection_json, args.bucket,
    ):
        oracle_z = (oracle_raw - mean) / std
        oracle_all.append(oracle_z[:valid_T])
        pred_all.append(pred[:valid_T])
        n_clips += 1
    if n_clips == 0:
        raise SystemExit("no selected clips processed")

    oracle_cat = np.concatenate(oracle_all, axis=0)
    pred_cat = np.concatenate(pred_all, axis=0)
    if oracle_cat.shape != pred_cat.shape:
        raise RuntimeError(f"shape mismatch after concat: {oracle_cat.shape} vs {pred_cat.shape}")

    groups = {
        name: _summarize_group(oracle_cat, pred_cat, sl, fps=args.fps)
        for name, sl in GROUPS
    }
    per_ch = []
    for i, name in enumerate(CHANNEL_NAMES):
        gt = oracle_cat[:, i]
        pr = pred_cat[:, i]
        per_ch.append({
            "channel": name,
            "residual_rms": _rms(pr - gt),
            "std_ratio": _safe_ratio(float(pr.std()), float(gt.std())),
        })
    per_ch.sort(key=lambda x: x["residual_rms"], reverse=True)

    stats = {
        "config": str(args.config),
        "generated_dir": str(args.generated_dir),
        "selection_json": str(args.selection_json),
        "bucket": args.bucket,
        "fps": float(args.fps),
        "n_clips": n_clips,
        "n_frames": int(oracle_cat.shape[0]),
        "groups": groups,
        "top_channels_by_rms": per_ch[:10],
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
