"""Round-31 Phase 1 — Stage-1 vs GT oracle stage1_coarse dynamic-info audit.

Per analyses/2026-05-30_round31_v2_chatgpt_review_response.md §4 ("Phase
1") + the user's 2026-05-30 follow-up ("测一测 stage1 的输出和 GT 直接
提取的 oracle 输出差了多远；不只是绝对的 xyz 的差值，还有各种速度，相
对速度，相对位置等可以体现动态信息的差别").

Reads:
- Generated stage1_coarse cached by Phase 0.5 cfg=1.0 ddim_eta0 (the same
  combo that fed every R31 V2 downstream-diag and Phase 0.5 ablation; this
  is the cond the 18 cm drift_max sweep saw).
- GT oracle stage1_coarse computed on-the-fly via
  ``piano.data.stage1_coarse_oracle.extract_coarse_v1_batched`` from each
  clip's motion_135 + rest_offsets.

Both are compared in **raw** (un-z-scored) space. The generated cache is
z-scored on disk — we un-z-score before comparison.

Reports (per-clip, per-channel, per-channel-group), all on the same 48
val-balanced clips:

1. 1st-4th moments per channel    : mean, std, skew, kurt; abs diff
   between GT and generated.
2. Per-frame velocity & accel     : finite-diff 1st and 2nd derivatives
   per channel; report vel_RMS_pred / vel_RMS_gt + accel_RMS ratio.
3. PSD per channel group          : Welch PSD on 5 channel groups (root,
   vel, yaw, pelvis_rot6d, spine3_rot6d, heights). Report per-band
   energy + cross-correlation between GT and generated PSD curves.
4. Spatial / chain consistency    : pelvis world position drift (from
   channels [0:3]) over time; head_height (ch 21) vs FK-derived head
   Y; rot6d orthogonality (||a1||, ||a2||, <a1,a2>) over time.

Outputs:
    analyses/round31_phase1_dyn_audit_<stamp>/
        audit_report.md        (summary tables + ASCII bar charts)
        audit_stats.json       (numbers, per-clip + per-channel)
        per_clip_dump.npz      (raw GT vs generated arrays for 48 clips)

Run:
    python -u scripts/stage_a_generator/round31_phase1_dyn_audit.py \\
        --upstream-dir analyses/round31_stage1_substitute_conds_phase0p5_cfg1p0_ddim_eta0/val \\
        --stage1-cfg configs/training/stage1_v2_v0_baseline.yaml \\
        --selection-json analyses/round29_val_diag_indices_48_balanced.json \\
        --out-dir analyses/round31_phase1_dyn_audit_$(date +%Y%m%d_%H%M%S)

CPU only — no GPU needed. ~3-5 min on the 48-clip selection.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy import signal as sp_signal
from scipy.stats import kurtosis, skew
from torch.utils.data import DataLoader

from piano.data.stage1_coarse_oracle import (
    extract_coarse_v1_batched,
    load_stage1_coarse_norm,
)
from piano.data.dataset import collate_hoi
from piano.training.train_anchordiff import _build_dataset


# Channel layout — must match stage1_coarse_oracle.py:191-214.
CHANNEL_NAMES: tuple[str, ...] = (
    "root_local_x", "root_local_z", "root_local_y",
    "vel_x", "vel_z", "vel_y",
    "yaw_sin", "yaw_cos", "yaw_vel",
    "pelvis_rot6d_0", "pelvis_rot6d_1", "pelvis_rot6d_2",
    "pelvis_rot6d_3", "pelvis_rot6d_4", "pelvis_rot6d_5",
    "spine3_rot6d_0", "spine3_rot6d_1", "spine3_rot6d_2",
    "spine3_rot6d_3", "spine3_rot6d_4", "spine3_rot6d_5",
    "head_height", "shoulder_center_h",
)
CHANNEL_GROUPS: dict[str, tuple[int, int]] = {
    "root": (0, 3),
    "vel": (3, 6),
    "yaw": (6, 9),
    "pelvis_rot6d": (9, 15),
    "spine3_rot6d": (15, 21),
    "heights": (21, 23),
}


def _read_selection(path: Path) -> set[tuple[str, str]]:
    sel = json.loads(path.read_text("utf-8"))
    items = sel.get("selected") or sel.get("candidates") or sel.get("clips") or []
    if not items:
        raise SystemExit(f"empty selection: {path}")
    return {(e["subset"], e["seq_id"]) for e in items}


def _read_generated(path: Path) -> tuple[np.ndarray, int]:
    """Load one Stage-1 generated stage1_coarse cache. Returns
    (z-scored array (T, 23), valid_T)."""
    data = np.load(path)
    if "stage1_coarse" not in data.files:
        raise KeyError(f"{path} has no 'stage1_coarse' key (got {data.files})")
    arr = data["stage1_coarse"].astype(np.float32)              # (T, 23) z-scored
    valid_T = int(data["valid_T"]) if "valid_T" in data.files else arr.shape[0]
    return arr, valid_T


def _finite_diff(x: np.ndarray, n: int = 1) -> np.ndarray:
    """n-th order finite-difference along axis 0. Output has same first-axis
    length as ``x`` with leading n rows zero-padded (matches the velocity
    convention in extract_coarse_v1_batched)."""
    out = x.astype(np.float32, copy=True)
    for _ in range(n):
        d = np.diff(out, axis=0)
        out = np.concatenate([np.zeros_like(out[:1]), d], axis=0)
    return out


def _per_clip_stats(arr: np.ndarray) -> dict[str, np.ndarray]:
    """Per-channel moments + velocity + accel statistics for one clip.

    ``arr``: (T_valid, 23) raw-space.
    Returns dict of (23,) arrays."""
    vel = _finite_diff(arr, n=1)
    accel = _finite_diff(arr, n=2)
    return {
        "mean": arr.mean(axis=0),
        "std": arr.std(axis=0),
        "skew": skew(arr, axis=0, bias=False, nan_policy="omit"),
        "kurt": kurtosis(arr, axis=0, bias=False, nan_policy="omit"),
        "vel_rms": np.sqrt((vel ** 2).mean(axis=0)),
        "vel_max": np.abs(vel).max(axis=0),
        "accel_rms": np.sqrt((accel ** 2).mean(axis=0)),
        "accel_max": np.abs(accel).max(axis=0),
    }


PSD_NPERSEG: int = 64
"""Fixed Welch nperseg so freqs vector length (= nperseg//2 + 1 = 33) is
identical across clips of different T. Clips shorter than nperseg are
zero-padded by scipy (and emit a UserWarning we filter out)."""


def _psd_per_group(arr: np.ndarray, fps: float = 20.0) -> dict[str, dict[str, np.ndarray]]:
    """Welch PSD per channel group. Returns dict keyed by group name.

    Each value is {"freqs": (33,), "psd_sum": (33,) — summed over channels
    in the group, "total_energy": float}. nperseg fixed at PSD_NPERSEG to
    guarantee freqs length is identical across clips."""
    out: dict[str, dict[str, np.ndarray]] = {}
    T = arr.shape[0]
    if T < PSD_NPERSEG:
        # Zero-pad up to nperseg so scipy doesn't shrink the freq grid.
        pad = np.zeros((PSD_NPERSEG - T, arr.shape[1]), dtype=arr.dtype)
        block_arr = np.concatenate([arr, pad], axis=0)
    else:
        block_arr = arr
    for name, (lo, hi) in CHANNEL_GROUPS.items():
        block = block_arr[:, lo:hi]                                  # (T, n_ch)
        freqs, psd = sp_signal.welch(
            block, fs=fps, nperseg=PSD_NPERSEG, axis=0,
        )                                                       # psd (F, n_ch)
        psd_sum = psd.sum(axis=-1)                              # (F,)
        out[name] = {
            "freqs": freqs.astype(np.float32),
            "psd_sum": psd_sum.astype(np.float32),
            "total_energy": float(psd_sum.sum()),
        }
    return out


def _rot6d_ortho_violations(arr_pelvis: np.ndarray, arr_spine: np.ndarray) -> dict[str, float]:
    """Mean ||a1||−1, ||a2||−1, |<a1, a2>| over T frames for each rot6d
    block. Inputs ``arr_*`` are (T, 6) raw."""
    out: dict[str, float] = {}
    for name, a in (("pelvis", arr_pelvis), ("spine3", arr_spine)):
        a1 = a[:, :3]; a2 = a[:, 3:]
        n1 = np.linalg.norm(a1, axis=-1)
        n2 = np.linalg.norm(a2, axis=-1)
        dot = (a1 * a2).sum(axis=-1)
        out[f"{name}_a1_norm_dev"] = float(np.mean(np.abs(n1 - 1.0)))
        out[f"{name}_a2_norm_dev"] = float(np.mean(np.abs(n2 - 1.0)))
        out[f"{name}_dot_abs"] = float(np.mean(np.abs(dot)))
    return out


# ───────────────────────── reporting helpers ─────────────────────────


def _agg_per_channel(per_clip: list[dict[str, np.ndarray]], key: str) -> np.ndarray:
    """Mean over clips of a per-channel stat. Returns (23,)."""
    stack = np.stack([c[key] for c in per_clip], axis=0)
    return stack.mean(axis=0)


def _ratio_safe(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return a / np.maximum(np.abs(b), eps)


def _mk_per_channel_table(
    gt_agg: dict[str, np.ndarray],
    pred_agg: dict[str, np.ndarray],
) -> list[str]:
    """Produce a markdown table of per-channel (gt, pred, gt-pred, ratio)
    for the 5 most-informative columns."""
    cols = ["mean", "std", "vel_rms", "accel_rms"]
    lines = [
        "| ch | name | "
        + " | ".join(f"gt_{c} | pred_{c} | Δ | ratio" for c in cols)
        + " |",
        "|---:|---|"
        + "|".join([":---:|" * 4 for _ in cols])
        + "",
    ]
    for c in range(23):
        cells = [f"{c}", CHANNEL_NAMES[c]]
        for col in cols:
            g = float(gt_agg[col][c])
            p = float(pred_agg[col][c])
            d = p - g
            r = p / (abs(g) + 1e-6)
            cells.extend([f"{g:+.3f}", f"{p:+.3f}", f"{d:+.3f}", f"{r:.2f}"])
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _mk_group_summary(
    gt_agg: dict[str, np.ndarray],
    pred_agg: dict[str, np.ndarray],
) -> list[str]:
    """Per-group rollup of mean/std/vel/accel RMS pred/gt ratios."""
    lines = [
        "| group | channels | gt mean |Δ| | gt std ratio | gt vel_rms ratio | gt accel_rms ratio |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, (lo, hi) in CHANNEL_GROUPS.items():
        mean_abs_d = float(np.abs(pred_agg["mean"][lo:hi] - gt_agg["mean"][lo:hi]).mean())
        std_ratio = float(np.mean(pred_agg["std"][lo:hi] / np.maximum(gt_agg["std"][lo:hi], 1e-6)))
        vel_ratio = float(np.mean(pred_agg["vel_rms"][lo:hi] / np.maximum(gt_agg["vel_rms"][lo:hi], 1e-6)))
        acc_ratio = float(np.mean(pred_agg["accel_rms"][lo:hi] / np.maximum(gt_agg["accel_rms"][lo:hi], 1e-6)))
        lines.append(
            f"| {name} | [{lo}:{hi}] | {mean_abs_d:.3f} | "
            f"{std_ratio:.2f} | {vel_ratio:.2f} | {acc_ratio:.2f} |"
        )
    return lines


def _mk_psd_summary(
    gt_psd: list[dict[str, dict[str, np.ndarray]]],
    pred_psd: list[dict[str, dict[str, np.ndarray]]],
) -> list[str]:
    """Per-group PSD energy split into 3 bands (low/mid/high). Reports
    pred/gt energy ratio per band.

    Bands at fps=20: low [0, 1] Hz, mid [1, 4] Hz, high [4, 10] Hz."""
    bands = [("low (0-1 Hz)", 0.0, 1.0),
             ("mid (1-4 Hz)", 1.0, 4.0),
             ("high (4-10 Hz)", 4.0, 10.0)]
    lines = [
        "| group | " + " | ".join(b[0] for b in bands) + " | total ratio |",
        "|---|" + "|".join([":---:" for _ in bands]) + "|:---:|",
    ]
    for group in CHANNEL_GROUPS:
        gt_total = 0.0
        pred_total = 0.0
        band_ratios: list[str] = []
        for _, lo, hi in bands:
            g_band, p_band = 0.0, 0.0
            for gt_clip, pred_clip in zip(gt_psd, pred_psd):
                gt_d = gt_clip[group]
                pr_d = pred_clip[group]
                fmask = (gt_d["freqs"] >= lo) & (gt_d["freqs"] < hi)
                g_band += float(gt_d["psd_sum"][fmask].sum())
                p_band += float(pr_d["psd_sum"][fmask].sum())
            ratio = p_band / max(g_band, 1e-12)
            band_ratios.append(f"{ratio:.2f}")
            gt_total += g_band
            pred_total += p_band
        total_ratio = pred_total / max(gt_total, 1e-12)
        lines.append(
            f"| {group} | " + " | ".join(band_ratios) + f" | {total_ratio:.2f} |"
        )
    return lines


# ───────────────────────── main ─────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-dir", type=Path, required=True,
                        help="Phase 0.5 generated stage1_coarse cache dir, "
                             "e.g. analyses/round31_stage1_substitute_conds_phase0p5_cfg1p0_ddim_eta0/val")
    parser.add_argument("--stage1-cfg", type=Path, required=True,
                        help="Stage-1 training config (used to build the val "
                             "dataset; only data section is consulted).")
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bucket", default="val", choices=["train", "val"])
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(str(args.stage1_cfg))

    # z-score stats (so we can un-zscore the cached generated cond).
    mean_np, std_np = load_stage1_coarse_norm(str(cfg.data.stage1_coarse_cache_root))
    if mean_np.shape != (23,) or std_np.shape != (23,):
        raise SystemExit(
            f"unexpected stage1_coarse norm shapes mean={mean_np.shape} std={std_np.shape}"
        )

    sel_pairs = _read_selection(args.selection_json)
    print(f"[audit] selection size = {len(sel_pairs)}")

    dataset = _build_dataset(cfg, args.bucket, augment=False)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        collate_fn=collate_hoi, num_workers=0,
    )

    # Per-clip accumulators.
    per_clip_gt_stats: list[dict[str, np.ndarray]] = []
    per_clip_pred_stats: list[dict[str, np.ndarray]] = []
    per_clip_gt_psd: list[dict[str, dict[str, np.ndarray]]] = []
    per_clip_pred_psd: list[dict[str, dict[str, np.ndarray]]] = []
    per_clip_ortho_gt: list[dict[str, float]] = []
    per_clip_ortho_pred: list[dict[str, float]] = []
    per_clip_meta: list[dict[str, Any]] = []

    # For optional npz dump.
    gt_arrays: list[np.ndarray] = []
    pred_arrays: list[np.ndarray] = []

    n_found = 0
    for batch in loader:
        subset = str(batch["subset"][0])
        seq_id = str(batch["seq_id"][0])
        if (subset, seq_id) not in sel_pairs:
            continue

        # Generated cache path.
        cache_path = args.upstream_dir / subset / f"{seq_id}.npz"
        if not cache_path.exists():
            print(f"[audit] WARN: missing generated cache for ({subset}, {seq_id}) at {cache_path}")
            continue

        # GT extraction (raw).
        motion = batch["motion"][0]                                # (T, 135)
        rest_offsets = batch["rest_offsets"][0]                    # (22, 3)
        seq_len = int(batch["seq_len"][0].item())
        T = int(motion.shape[0])
        valid_T = min(T, seq_len)

        gt_coarse_raw = extract_coarse_v1_batched(
            motion.unsqueeze(0), rest_offsets.unsqueeze(0),
        )[0].numpy().astype(np.float32)                            # (T, 23) raw

        # Generated cache: z-scored on disk, un-zscore here.
        gen_z, gen_valid_T = _read_generated(cache_path)           # (T_cache, 23)
        if gen_z.shape[0] < valid_T:
            print(f"[audit] WARN: ({subset}, {seq_id}) gen cache T={gen_z.shape[0]} < valid_T={valid_T}")
            valid_T = min(valid_T, gen_z.shape[0])
        gen_raw = gen_z[:valid_T] * std_np[None, :] + mean_np[None, :]
        gt_raw = gt_coarse_raw[:valid_T]

        # Per-clip stats.
        per_clip_gt_stats.append(_per_clip_stats(gt_raw))
        per_clip_pred_stats.append(_per_clip_stats(gen_raw))
        per_clip_gt_psd.append(_psd_per_group(gt_raw))
        per_clip_pred_psd.append(_psd_per_group(gen_raw))
        per_clip_ortho_gt.append(_rot6d_ortho_violations(gt_raw[:, 9:15], gt_raw[:, 15:21]))
        per_clip_ortho_pred.append(_rot6d_ortho_violations(gen_raw[:, 9:15], gen_raw[:, 15:21]))
        per_clip_meta.append({
            "subset": subset, "seq_id": seq_id, "valid_T": valid_T,
        })
        gt_arrays.append(gt_raw)
        pred_arrays.append(gen_raw)

        n_found += 1
        if n_found % 8 == 0:
            print(f"[audit] processed {n_found}/{len(sel_pairs)}")

    if n_found == 0:
        raise SystemExit("[audit] no clips matched the selection AND had a generated cache.")

    print(f"[audit] processed {n_found} clips total.")

    # Aggregate across clips.
    gt_agg = {k: _agg_per_channel(per_clip_gt_stats, k)
              for k in per_clip_gt_stats[0].keys()}
    pred_agg = {k: _agg_per_channel(per_clip_pred_stats, k)
                for k in per_clip_pred_stats[0].keys()}

    # rot6d ortho aggregate.
    ortho_keys = list(per_clip_ortho_gt[0].keys())
    ortho_gt_agg = {k: float(np.mean([c[k] for c in per_clip_ortho_gt])) for k in ortho_keys}
    ortho_pred_agg = {k: float(np.mean([c[k] for c in per_clip_ortho_pred])) for k in ortho_keys}

    # ── Markdown report ─────────────────────────────────────────────
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md: list[str] = [
        "# R31 Phase 1 — Stage-1 generated vs GT oracle stage1_coarse dynamic-info audit",
        "",
        f"Generated: {stamp}",
        f"Cache    : `{args.upstream_dir}`",
        f"Selection: `{args.selection_json}` ({n_found} clips)",
        "",
        "## 1. Per-channel-group rollup",
        "",
        "Reading the columns:",
        "- **gt mean |Δ|** : mean abs difference between pred and GT per-channel-mean across clips.",
        "  Large → pred's central tendency shifted away from GT.",
        "- **std ratio** : pred std / GT std (mean over channels in group).",
        "  < 1.0 → pred under-disperses (predicted distribution is tighter than GT).",
        "  > 1.0 → over-disperses.",
        "- **vel_rms / accel_rms ratio** : pred finite-difference RMS over GT.",
        "  < 1.0 → pred is smoother / under-articulated; > 1.0 → jitterier.",
        "",
    ]
    md += _mk_group_summary(gt_agg, pred_agg)

    md += [
        "",
        "## 2. Per-channel detail (mean / std / vel_rms / accel_rms)",
        "",
        "All numbers are raw-space (post-un-z-score). Δ = pred − GT. ratio = pred / |GT|.",
        "",
    ]
    md += _mk_per_channel_table(gt_agg, pred_agg)

    md += [
        "",
        "## 3. PSD band-energy ratio (pred / GT, per channel group)",
        "",
        "Welch PSD computed per clip then summed within frequency bands.",
        "fps=20, so high band 4-10 Hz is the Nyquist top half.",
        "Below 1.0 means pred has less energy in that band; above 1.0 means more.",
        "",
    ]
    md += _mk_psd_summary(per_clip_gt_psd, per_clip_pred_psd)

    md += [
        "",
        "## 4. rot6d orthogonality (raw-space)",
        "",
        "Per-frame violation of the SO(3) Gram-Schmidt invariants.",
        "GT should be ~0 by construction (it comes from valid SMPL rotations).",
        "Generated should approach 0 if Stage-1 learned the manifold.",
        "",
        "| block | metric | GT | pred | pred − GT |",
        "|---|---|---:|---:|---:|",
    ]
    for k in ortho_keys:
        g = ortho_gt_agg[k]; p = ortho_pred_agg[k]
        block, metric = k.split("_", 1)
        md.append(f"| {block} | {metric} | {g:.4f} | {p:.4f} | {p - g:+.4f} |")

    md += [
        "",
        "## 5. Per-clip drift summary (channels [0:3] — root_local)",
        "",
        "RMS of `pred_root_local[t] − gt_root_local[t]` over t, in raw-space (metres).",
        "If pred preserves the frame-0 anchor convention this should be small at t=0",
        "and grow with t. We report mean and max over t per clip, then median + IQR across clips.",
        "",
    ]
    drift_per_clip = []
    for gt_arr, pred_arr in zip(gt_arrays, pred_arrays):
        diff = pred_arr[:, 0:3] - gt_arr[:, 0:3]                  # (T_valid, 3)
        rms_per_frame = np.sqrt((diff ** 2).sum(-1))               # (T_valid,)
        drift_per_clip.append({
            "rms_at_t0": float(rms_per_frame[0]),
            "rms_mean": float(rms_per_frame.mean()),
            "rms_max": float(rms_per_frame.max()),
        })
    rms_t0 = np.array([d["rms_at_t0"] for d in drift_per_clip])
    rms_mean = np.array([d["rms_mean"] for d in drift_per_clip])
    rms_max = np.array([d["rms_max"] for d in drift_per_clip])
    md += [
        "| stat | t=0 (m) | mean over t (m) | max over t (m) |",
        "|---|---:|---:|---:|",
        f"| median | {np.median(rms_t0):.4f} | {np.median(rms_mean):.4f} | {np.median(rms_max):.4f} |",
        f"| q25    | {np.quantile(rms_t0, 0.25):.4f} | {np.quantile(rms_mean, 0.25):.4f} | {np.quantile(rms_max, 0.25):.4f} |",
        f"| q75    | {np.quantile(rms_t0, 0.75):.4f} | {np.quantile(rms_mean, 0.75):.4f} | {np.quantile(rms_max, 0.75):.4f} |",
        f"| max    | {rms_t0.max():.4f} | {rms_mean.max():.4f} | {rms_max.max():.4f} |",
        "",
        "**Frame-0 invariant check**: GT root_local channel [0:3] is exactly 0 at t=0 by",
        "construction (`stage1_coarse_oracle.py:158` does `root_world − root_world[:, :1]`).",
        "If `rms_at_t0` is non-negligible, Stage-1 is producing a root_local that does NOT",
        "anchor at zero at frame 0 — that is a violation of the cond contract PB1 was",
        "trained on.",
        "",
        "## 6. Top-5 channels by std mismatch (pred std / gt std deviation from 1)",
        "",
        "| ch | name | gt std | pred std | ratio | log(ratio) |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    ratio = pred_agg["std"] / np.maximum(gt_agg["std"], 1e-6)
    log_ratio_abs = np.abs(np.log(np.maximum(ratio, 1e-6)))
    order = np.argsort(-log_ratio_abs)
    for i in order[:5]:
        md.append(
            f"| {i} | {CHANNEL_NAMES[i]} | {gt_agg['std'][i]:.3f} | "
            f"{pred_agg['std'][i]:.3f} | {ratio[i]:.3f} | {np.log(ratio[i]):+.3f} |"
        )

    md += [
        "",
        "## 7. Top-5 channels by velocity mismatch (pred vel_rms / gt vel_rms)",
        "",
        "| ch | name | gt vel_rms | pred vel_rms | ratio | log(ratio) |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    vel_ratio = pred_agg["vel_rms"] / np.maximum(gt_agg["vel_rms"], 1e-6)
    log_vel_ratio_abs = np.abs(np.log(np.maximum(vel_ratio, 1e-6)))
    order = np.argsort(-log_vel_ratio_abs)
    for i in order[:5]:
        md.append(
            f"| {i} | {CHANNEL_NAMES[i]} | {gt_agg['vel_rms'][i]:.3f} | "
            f"{pred_agg['vel_rms'][i]:.3f} | {vel_ratio[i]:.3f} | {np.log(vel_ratio[i]):+.3f} |"
        )

    md += [
        "",
        "## 8. Interpretation cheat-sheet",
        "",
        "- **mean |Δ| > 0.1 m or > 0.1 rad on a channel** → distribution-mean shift.",
        "  H1 (cond OOD) supported on that channel.",
        "- **std ratio < 0.7 on a channel group** → pred under-disperses.",
        "  Canonical signature of MSE training producing posterior-mean.",
        "  H7 (posterior mean under MSE) supported.",
        "- **vel_rms ratio < 0.7 on a channel group** → pred is too smooth on that group.",
        "  Same as above but at the velocity level.",
        "- **PSD high-band ratio < 0.5** → pred kills high-frequency content present in GT.",
        "  Common in diffusion-x0 + L2 training; under-articulation lives here.",
        "- **rot6d ortho dev > 0.05 in pred but ~0 in GT** → Stage-1 emits 6-D vectors that",
        "  don't lie on the orthogonal-pair manifold. Gram-Schmidt projection saves PB1's",
        "  rot6d_to_matrix from crashing but the projection's discontinuity is OOD.",
        "- **rms_at_t0 > 0.01 m for channels [0:3]** → frame-0 invariant violated.",
        "  H5 (channel-order / frame-0 plumbing) supported.",
        "",
    ]

    report_path = args.out_dir / "audit_report.md"
    report_path.write_text("\n".join(md), encoding="utf-8")
    print(f"[audit] wrote {report_path}")

    # ── JSON dump ────────────────────────────────────────────────────
    out_stats: dict[str, Any] = {
        "n_clips": n_found,
        "channel_names": list(CHANNEL_NAMES),
        "channel_groups": {k: list(v) for k, v in CHANNEL_GROUPS.items()},
        "gt_agg": {k: v.tolist() for k, v in gt_agg.items()},
        "pred_agg": {k: v.tolist() for k, v in pred_agg.items()},
        "rot6d_ortho_gt": ortho_gt_agg,
        "rot6d_ortho_pred": ortho_pred_agg,
        "per_clip_meta": per_clip_meta,
        "per_clip_root_drift": drift_per_clip,
        "psd": {
            group: {
                "freqs": per_clip_gt_psd[0][group]["freqs"].tolist(),
                "gt_psd_sum_mean": (
                    np.stack([c[group]["psd_sum"] for c in per_clip_gt_psd], axis=0).mean(0).tolist()
                ),
                "pred_psd_sum_mean": (
                    np.stack([c[group]["psd_sum"] for c in per_clip_pred_psd], axis=0).mean(0).tolist()
                ),
            }
            for group in CHANNEL_GROUPS
        },
    }
    json_path = args.out_dir / "audit_stats.json"
    json_path.write_text(json.dumps(out_stats, indent=2), encoding="utf-8")
    print(f"[audit] wrote {json_path}")

    # ── Per-clip raw dump (optional, ~few MB) ───────────────────────
    dump_path = args.out_dir / "per_clip_dump.npz"
    max_T = max(g.shape[0] for g in gt_arrays)
    pad_gt = np.zeros((len(gt_arrays), max_T, 23), dtype=np.float32)
    pad_pred = np.zeros((len(pred_arrays), max_T, 23), dtype=np.float32)
    valid_Ts = np.zeros((len(gt_arrays),), dtype=np.int32)
    subsets = []
    seq_ids = []
    for i, (g, p, meta) in enumerate(zip(gt_arrays, pred_arrays, per_clip_meta)):
        Tv = g.shape[0]
        pad_gt[i, :Tv] = g
        pad_pred[i, :Tv] = p
        valid_Ts[i] = Tv
        subsets.append(meta["subset"])
        seq_ids.append(meta["seq_id"])
    np.savez(
        dump_path,
        gt=pad_gt, pred=pad_pred, valid_T=valid_Ts,
        subsets=np.array(subsets), seq_ids=np.array(seq_ids),
    )
    print(f"[audit] wrote {dump_path}")


if __name__ == "__main__":
    main()
