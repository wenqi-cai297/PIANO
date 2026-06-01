"""Round-40 Stage-1 plan-quality diagnostic.

Companion to the Round-31 downstream diag launcher. While the downstream
diag tells us how a Stage-1 ckpt's generated cond degrades PB1, this
script measures Stage-1's plan-level quality directly:

  - R35 OOD audit headline metrics (std_ratio, vel_ratio, PSD band ratios
    per channel group). Reproduced here so the R40 summary can compare
    apples-to-apples against R35.
  - Plan metrics: root speed mean/std ratio, arc length ratio, displacement
    ratio, root-object radial profile errors, yaw turn-rate ratios, yaw
    range ratio, pelvis/spine3 rot6d activity ratios, head/shoulder height
    mean/min/max errors.

The plan metrics mirror the components of ``stage1_plan_invariant_loss``
so the R40 summary can tell whether a variant moved the *quantities the
loss supervised*.

Cache contract (matches ``sample_substitute_conds``):
    <pred-dir>/<bucket>/<subset>/<seq_id>.npz
        stage1_coarse : (T, 23) z-scored
        valid_T       : int

Outputs:
    <out-dir>/plan_stats.json       — machine-readable
    <out-dir>/plan_summary.md       — human-readable

Run:
    python -u scripts/stage_a_generator/round40_stage1_plan_diag.py \\
        --config configs/training/stage1_r40_c2_plan_energy.yaml \\
        --pred-dir analyses/round40_stage1_substitute_conds_c2_plan_energy \\
        --selection-json analyses/round29_val_diag_indices_48_balanced.json \\
        --bucket val \\
        --out-dir analyses/round40_stage1_plan_diag_c2_plan_energy
"""
from __future__ import annotations

import argparse
import json
import math
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


# Same group layout as R35 so summary lines line up.
R35_GROUPS: tuple[tuple[str, slice], ...] = (
    ("root_local_xzy", slice(0, 3)),
    ("velocity_xzy", slice(3, 6)),
    ("yaw_sin_cos_vel", slice(6, 9)),
    ("pelvis_rot6d", slice(9, 15)),
    ("spine3_rot6d", slice(15, 21)),
    ("heights", slice(21, 23)),
    ("all", slice(0, 23)),
)


def _cache_bucket_root(root: Path, bucket: str) -> Path:
    return root / bucket if (root / bucket).is_dir() else root


def _load_stage1_cache(
    root: Path, bucket: str, subset: str, seq_id: str,
) -> tuple[np.ndarray, int]:
    p = _cache_bucket_root(root, bucket) / subset / f"{seq_id}.npz"
    if not p.exists():
        raise FileNotFoundError(p)
    data = np.load(p)
    if "stage1_coarse" not in data.files:
        raise KeyError(f"{p}: missing stage1_coarse (keys={list(data.files)})")
    arr = data["stage1_coarse"].astype(np.float32)
    if arr.ndim != 2 or arr.shape[-1] != STAGE1_COARSE_DIM:
        raise ValueError(f"{p}: expected (T, 23), got {arr.shape}")
    valid_T = (
        int(data["valid_T"]) if "valid_T" in data.files else arr.shape[0]
    )
    return arr, valid_T


def _safe_ratio(num: float, den: float) -> float:
    return float(num / den) if abs(den) > 1e-12 else 0.0


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0


def _band_energy(x: np.ndarray, fps: float) -> dict[str, float]:
    if x.shape[0] <= 1:
        return {
            "low_0_1": 0.0, "mid_1_4": 0.0, "high_4_10": 0.0, "total": 0.0,
        }
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


def _summarize_group(
    oracle: np.ndarray, pred: np.ndarray, sl: slice, fps: float,
) -> dict:
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
            k: _safe_ratio(pr_e[k], gt_e[k])
            for k in ("low_0_1", "mid_1_4", "high_4_10", "total")
        },
        "psd_residual_gt_ratio": {
            k: _safe_ratio(res_e[k], gt_e[k])
            for k in ("low_0_1", "mid_1_4", "high_4_10", "total")
        },
    }


# ────────────────────────────────────────────────────────────────────────────
# Plan metrics on raw (un-z-scored) stage1_coarse + world-frame reconstruction
# ────────────────────────────────────────────────────────────────────────────


def _plan_metrics_per_clip(
    pred_raw: np.ndarray,                 # (T, 23) un-z-scored
    gt_raw: np.ndarray,                   # (T, 23) un-z-scored
    obj_world_xz: np.ndarray,             # (T, 2) world (x, z)
    root_world_t0_xz: np.ndarray,         # (2,)
    valid_T: int,
) -> dict[str, float]:
    """Per-clip plan metrics. Channel order matches stage1_coarse_oracle."""
    T = min(pred_raw.shape[0], gt_raw.shape[0], obj_world_xz.shape[0], valid_T)
    if T < 2:
        return {}
    p = pred_raw[:T]
    g = gt_raw[:T]
    obj = obj_world_xz[:T]

    # Root world XZ = root_local (x, z) + t0 (x, z). Oracle stores local as
    # (x, z, y) at indices 0/1/2; we use 0 and 1 for XZ.
    rwx_p = p[:, 0] + root_world_t0_xz[0]
    rwz_p = p[:, 1] + root_world_t0_xz[1]
    rwx_g = g[:, 0] + root_world_t0_xz[0]
    rwz_g = g[:, 1] + root_world_t0_xz[1]

    # Frame-to-frame speed.
    dx_p = np.diff(rwx_p); dz_p = np.diff(rwz_p)
    dx_g = np.diff(rwx_g); dz_g = np.diff(rwz_g)
    sp_p = np.sqrt(dx_p ** 2 + dz_p ** 2)
    sp_g = np.sqrt(dx_g ** 2 + dz_g ** 2)
    root_speed_mean_p, root_speed_std_p = float(sp_p.mean()), float(sp_p.std())
    root_speed_mean_g, root_speed_std_g = float(sp_g.mean()), float(sp_g.std())
    root_arc_p = float(sp_p.sum())
    root_arc_g = float(sp_g.sum())

    # Displacement (final − initial).
    disp_p = math.hypot(rwx_p[-1] - rwx_p[0], rwz_p[-1] - rwz_p[0])
    disp_g = math.hypot(rwx_g[-1] - rwx_g[0], rwz_g[-1] - rwz_g[0])

    # Root-object radial distance profile.
    dist_p = np.sqrt((rwx_p - obj[:, 0]) ** 2 + (rwz_p - obj[:, 1]) ** 2)
    dist_g = np.sqrt((rwx_g - obj[:, 0]) ** 2 + (rwz_g - obj[:, 1]) ** 2)
    ror_mean_err = float(abs(dist_p.mean() - dist_g.mean()))
    ror_std_err = float(abs(dist_p.std() - dist_g.std()))
    ror_min_err = float(abs(dist_p.min() - dist_g.min()))
    ror_final_err = float(abs(dist_p[-1] - dist_g[-1]))

    # Yaw turn-rate + cumulative range.
    yaw_p = np.arctan2(p[:, 6], p[:, 7])
    yaw_g = np.arctan2(g[:, 6], g[:, 7])
    twopi = 2.0 * math.pi
    d_yp = np.diff(yaw_p)
    d_yg = np.diff(yaw_g)
    d_yp = (d_yp + math.pi) % twopi - math.pi
    d_yg = (d_yg + math.pi) % twopi - math.pi
    abs_dyp = np.abs(d_yp)
    abs_dyg = np.abs(d_yg)
    yaw_rate_mean_p = float(abs_dyp.mean())
    yaw_rate_mean_g = float(abs_dyg.mean())
    yaw_rate_std_p = float(abs_dyp.std())
    yaw_rate_std_g = float(abs_dyg.std())
    cs_p = np.cumsum(d_yp)
    cs_g = np.cumsum(d_yg)
    yaw_range_p = float(cs_p.max() - cs_p.min()) if cs_p.size else 0.0
    yaw_range_g = float(cs_g.max() - cs_g.min()) if cs_g.size else 0.0

    # Rot6d activity (per-frame 6-D L2 of the diff).
    d_pel_p = np.linalg.norm(p[1:, 9:15] - p[:-1, 9:15], axis=-1)
    d_pel_g = np.linalg.norm(g[1:, 9:15] - g[:-1, 9:15], axis=-1)
    d_sp_p = np.linalg.norm(p[1:, 15:21] - p[:-1, 15:21], axis=-1)
    d_sp_g = np.linalg.norm(g[1:, 15:21] - g[:-1, 15:21], axis=-1)

    # Heights.
    head_p, head_g = p[:, 21], g[:, 21]
    sh_p, sh_g = p[:, 22], g[:, 22]

    return {
        "root_speed_mean_pred": root_speed_mean_p,
        "root_speed_mean_gt": root_speed_mean_g,
        "root_speed_std_pred": root_speed_std_p,
        "root_speed_std_gt": root_speed_std_g,
        "root_arc_pred": root_arc_p,
        "root_arc_gt": root_arc_g,
        "root_disp_pred": float(disp_p),
        "root_disp_gt": float(disp_g),
        "root_object_radial_mean_err": ror_mean_err,
        "root_object_radial_std_err": ror_std_err,
        "root_object_radial_min_err": ror_min_err,
        "root_object_radial_final_err": ror_final_err,
        "yaw_rate_mean_pred": yaw_rate_mean_p,
        "yaw_rate_mean_gt": yaw_rate_mean_g,
        "yaw_rate_std_pred": yaw_rate_std_p,
        "yaw_rate_std_gt": yaw_rate_std_g,
        "yaw_range_pred": yaw_range_p,
        "yaw_range_gt": yaw_range_g,
        "pelvis_rot_act_mean_pred": float(d_pel_p.mean()),
        "pelvis_rot_act_mean_gt": float(d_pel_g.mean()),
        "pelvis_rot_act_std_pred": float(d_pel_p.std()),
        "pelvis_rot_act_std_gt": float(d_pel_g.std()),
        "spine3_rot_act_mean_pred": float(d_sp_p.mean()),
        "spine3_rot_act_mean_gt": float(d_sp_g.mean()),
        "spine3_rot_act_std_pred": float(d_sp_p.std()),
        "spine3_rot_act_std_gt": float(d_sp_g.std()),
        "head_height_mean_pred": float(head_p.mean()),
        "head_height_mean_gt": float(head_g.mean()),
        "head_height_min_pred": float(head_p.min()),
        "head_height_min_gt": float(head_g.min()),
        "head_height_max_pred": float(head_p.max()),
        "head_height_max_gt": float(head_g.max()),
        "shoulder_height_mean_pred": float(sh_p.mean()),
        "shoulder_height_mean_gt": float(sh_g.mean()),
    }


def _aggregate_plan_stats(per_clip: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate ratios + absolute errors across clips.

    For pred/gt pairs we report:
        <key>_ratio = mean(pred) / mean(gt)   (zero-safe)
        <key>_err   = mean(|pred - gt|)
    For _err keys we just average.
    """
    if not per_clip:
        return {}
    out: dict[str, float] = {}
    keys = per_clip[0].keys()
    pred_keys = {k[:-5] for k in keys if k.endswith("_pred")}
    err_keys = {k for k in keys if k.endswith("_err")}
    for base in sorted(pred_keys):
        pred_vals = np.array(
            [c[f"{base}_pred"] for c in per_clip], dtype=np.float64,
        )
        gt_vals = np.array(
            [c[f"{base}_gt"] for c in per_clip], dtype=np.float64,
        )
        out[f"{base}_pred_mean"] = float(pred_vals.mean())
        out[f"{base}_gt_mean"] = float(gt_vals.mean())
        out[f"{base}_ratio"] = _safe_ratio(
            float(pred_vals.mean()), float(gt_vals.mean()),
        )
        out[f"{base}_abs_err_mean"] = float(np.mean(np.abs(pred_vals - gt_vals)))
    for k in sorted(err_keys):
        vals = np.array([c[k] for c in per_clip], dtype=np.float64)
        out[k + "_mean"] = float(vals.mean())
        out[k + "_max"] = float(vals.max())
    return out


def _collect_clip_pairs(cfg, pred_dir: Path, selection_json: Path, bucket: str):
    """Yield (subset, seq_id, oracle_raw, oracle_z, pred_z, valid_T,
    obj_world_xz, root_world_t0_xz)."""
    mean_np, std_np = load_stage1_coarse_norm(
        str(cfg.data.stage1_coarse_cache_root),
    )
    mean = mean_np.astype(np.float32).reshape(1, STAGE1_COARSE_DIM)
    std = std_np.astype(np.float32).reshape(1, STAGE1_COARSE_DIM)

    sel_pairs = _read_selection(selection_json)
    dataset = _build_dataset(cfg, bucket, augment=False)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        collate_fn=collate_hoi, num_workers=0,
    )
    for batch in loader:
        subset = str(batch["subset"][0])
        seq_id = str(batch["seq_id"][0])
        if (subset, seq_id) not in sel_pairs:
            continue
        pred, vt_pred = _load_stage1_cache(pred_dir, bucket, subset, seq_id)
        motion = batch["motion"].float()
        rest_offsets = batch["rest_offsets"].float()
        oracle_raw = extract_coarse_v1_batched(
            motion=motion, rest_offsets=rest_offsets,
        )[0].cpu().numpy().astype(np.float32)
        oracle_z = (oracle_raw - mean) / std
        seq_len = int(batch["seq_len"][0].item())
        valid_T = min(seq_len, vt_pred, pred.shape[0], oracle_raw.shape[0])
        # Object world XZ from obj_com_canonical (B, T, 3).
        obj_com = batch["obj_com_canonical"][0].float().cpu().numpy()
        obj_world_xz = obj_com[:, [0, 2]].astype(np.float32)
        # Root world t0 from motion[..., 132:135] (world x, y, z).
        root_world_t0 = motion[0, 0, 132:135].cpu().numpy()
        root_world_t0_xz = np.array(
            [root_world_t0[0], root_world_t0[2]], dtype=np.float32,
        )
        # Un-z-score pred for the plan metric pass.
        pred_raw = pred * std + mean
        yield (
            subset, seq_id, oracle_raw, oracle_z, pred, pred_raw,
            valid_T, obj_world_xz, root_world_t0_xz,
        )


def _write_report(out_md: Path, stats: dict, plan_summary: dict) -> None:
    lines: list[str] = [
        "# Round-40 Stage-1 Plan-Quality Diagnostic",
        "",
        f"- config: `{stats['config']}`",
        f"- pred_dir: `{stats['pred_dir']}`",
        f"- selection_json: `{stats['selection_json']}`",
        f"- bucket: {stats['bucket']}",
        f"- n_clips: {stats['n_clips']}",
        f"- fps: {stats['fps']}",
        "",
        "## R35 OOD audit headline (drop-in compare against round35 reports)",
        "",
        "| group | std_ratio | vel_ratio | PSD pred/gt low | PSD pred/gt mid | PSD pred/gt high |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in stats["groups"].items():
        pr = row["psd_pred_gt_ratio"]
        lines.append(
            f"| {name} | {row['std_ratio']:.3f} | "
            f"{row['vel_rms_ratio']:.3f} | "
            f"{pr['low_0_1']:.3f} | {pr['mid_1_4']:.3f} | {pr['high_4_10']:.3f} |"
        )

    lines.extend([
        "",
        "## Plan metrics (mirror stage1_plan_invariant_loss components)",
        "",
        "| metric | pred mean | gt mean | ratio | mean |pred-gt| |",
        "|---|---:|---:|---:|---:|",
    ])
    # Stable display order.
    display_order = [
        "root_speed_mean",
        "root_speed_std",
        "root_arc",
        "root_disp",
        "yaw_rate_mean",
        "yaw_rate_std",
        "yaw_range",
        "pelvis_rot_act_mean",
        "pelvis_rot_act_std",
        "spine3_rot_act_mean",
        "spine3_rot_act_std",
        "head_height_mean",
        "head_height_min",
        "head_height_max",
        "shoulder_height_mean",
    ]
    for base in display_order:
        if f"{base}_pred_mean" not in plan_summary:
            continue
        pm = plan_summary[f"{base}_pred_mean"]
        gm = plan_summary[f"{base}_gt_mean"]
        rt = plan_summary[f"{base}_ratio"]
        ae = plan_summary[f"{base}_abs_err_mean"]
        lines.append(f"| {base} | {pm:.4f} | {gm:.4f} | {rt:.3f} | {ae:.4f} |")

    lines.extend([
        "",
        "## Root-object radial errors (independent of side / direction)",
        "",
        "| metric | mean | max |",
        "|---|---:|---:|",
    ])
    for k in (
        "root_object_radial_mean_err",
        "root_object_radial_std_err",
        "root_object_radial_min_err",
        "root_object_radial_final_err",
    ):
        if f"{k}_mean" not in plan_summary:
            continue
        lines.append(
            f"| {k} | {plan_summary[k + '_mean']:.4f} | "
            f"{plan_summary[k + '_max']:.4f} |"
        )

    lines.extend([
        "",
        "## Top channels by residual RMS (z-scored)",
        "",
        "| rank | channel | residual_rms | pred_std / gt_std |",
        "|---:|---|---:|---:|",
    ])
    for i, row in enumerate(stats["top_channels_by_rms"], start=1):
        lines.append(
            f"| {i} | {row['channel']} | {row['residual_rms']:.4f} | "
            f"{row['std_ratio']:.3f} |"
        )

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


CHANNEL_NAMES = (
    "root_local_x", "root_local_z", "root_local_y",
    "vel_x", "vel_z", "vel_y",
    "yaw_sin", "yaw_cos", "yaw_vel",
    "pelvis_r0", "pelvis_r1", "pelvis_r2", "pelvis_r3", "pelvis_r4", "pelvis_r5",
    "spine3_r0", "spine3_r1", "spine3_r2", "spine3_r3", "spine3_r4", "spine3_r5",
    "head_height", "shoulder_center_h",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument(
        "--pred-dir", type=Path, required=True,
        help="Substitute-conds dir containing <bucket>/<subset>/<seq>.npz "
             "with z-scored stage1_coarse.",
    )
    ap.add_argument("--selection-json", type=Path, required=True)
    ap.add_argument("--bucket", choices=["train", "val"], default="val")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=FPS_DEFAULT)
    args = ap.parse_args()

    cfg = OmegaConf.load(str(args.config))

    oracle_z_all: list[np.ndarray] = []
    pred_z_all: list[np.ndarray] = []
    per_clip_plan: list[dict[str, float]] = []
    n_clips = 0
    for (
        _subset, _seq_id, oracle_raw, oracle_z, pred_z, pred_raw,
        valid_T, obj_world_xz, root_world_t0_xz,
    ) in _collect_clip_pairs(
        cfg, args.pred_dir, args.selection_json, args.bucket,
    ):
        oracle_z_all.append(oracle_z[:valid_T])
        pred_z_all.append(pred_z[:valid_T])
        per_clip_plan.append(
            _plan_metrics_per_clip(
                pred_raw=pred_raw,
                gt_raw=oracle_raw,
                obj_world_xz=obj_world_xz,
                root_world_t0_xz=root_world_t0_xz,
                valid_T=valid_T,
            )
        )
        n_clips += 1
    if n_clips == 0:
        raise SystemExit("no selected clips processed")

    oracle_cat = np.concatenate(oracle_z_all, axis=0)
    pred_cat = np.concatenate(pred_z_all, axis=0)
    if oracle_cat.shape != pred_cat.shape:
        raise RuntimeError(
            f"shape mismatch after concat: {oracle_cat.shape} vs {pred_cat.shape}"
        )

    groups = {
        name: _summarize_group(oracle_cat, pred_cat, sl, fps=args.fps)
        for name, sl in R35_GROUPS
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

    plan_summary = _aggregate_plan_stats(
        [c for c in per_clip_plan if c]
    )

    stats = {
        "config": str(args.config),
        "pred_dir": str(args.pred_dir),
        "selection_json": str(args.selection_json),
        "bucket": args.bucket,
        "fps": float(args.fps),
        "n_clips": n_clips,
        "n_frames": int(oracle_cat.shape[0]),
        "groups": groups,
        "top_channels_by_rms": per_ch[:10],
        "plan_summary": plan_summary,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "plan_stats.json"
    out_md = args.out_dir / "plan_summary.md"
    out_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    _write_report(out_md, stats, plan_summary)
    print(f"wrote {out_md}")
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
