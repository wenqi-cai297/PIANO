"""Round-32 Phase 1 — Stage-1.5 vs GT oracle (C41, S4) dynamic-info audit.

Mirrors ``round31_phase1_dyn_audit.py`` for Stage-1.5 (the upstream of
PB1's stage2_coarse_extra (C41, 18-D) + stage2_support (S4, 13-D)).
The motivation is identical: Stage-1.5 V0 was trained with a
trivial-loss design (MSE on C41 + MSE on S4 + a few BCEs); we suspect
the same mode-collapse failure mode R31 V0 hit. This audit makes the
failure mode explicit so the next-round V7-style anti-collapse design
can target it precisely.

Reads:
- Generated (C41, S4) cached by the Stage-1.5 downstream-coupling
  diagnostic, i.e.::

      analyses/round32_stage1p5_substitute_conds_v8_audit/val/<subset>/<seq_id>.npz

  Each npz has keys:
      stage2_coarse_extra  (T, 18) raw
      stage2_support       (T, 13) raw
- GT C41 + S4 from on-the-fly ``build_coarse_condition`` +
  ``build_support_condition`` (the same code path the trainer + the
  PB1 inference both use, so this is the apples-to-apples GT).

Both compared in RAW space (Stage-1.5 trains and emits raw). No
z-scoring step.

Channel layout (31-D = C41 18 + S4 13):

    C41 (18) — pelvis-local, current-yaw frame, frame-0-relative
      [ 0: 3]  left_wrist  Δxyz
      [ 3: 6]  right_wrist Δxyz
      [ 6: 9]  left_knee   Δxyz
      [ 9:12]  right_knee  Δxyz
      [12:15]  neck        Δxyz
      [15:18]  pelvis      Δxzy  (root0-yaw frame; see oracle docs)

    S4 (13) — gait + footstep
      [18:20]  foot_stance L, R
      [20:22]  ankle_height_norm L, R
      [22:23]  walking_mask
      [23:27]  phase_sin/cos L, phase_sin/cos R
      [27:31]  footstep_x/z L, footstep_x/z R

Outputs:
    analyses/round32_phase1_dyn_audit_<stamp>/
        audit_report.md
        audit_stats.json
        per_clip_dump.npz

Run:
    python -u scripts/stage_a_generator/round32_phase1_dyn_audit.py \
        --upstream-dir analyses/round32_stage1p5_substitute_conds_v8_audit/val \
        --stage1p5-cfg configs/training/stage1p5_interaction_v0.yaml \
        --selection-json analyses/round29_val_diag_indices_48_balanced.json \
        --out-dir analyses/round32_phase1_dyn_audit_$(date +%Y%m%d_%H%M%S)

CPU only. ~3-5 min on 48 clips.
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

from piano.data.dataset import collate_hoi
from piano.data.stage2_oracle_conditions import (
    build_coarse_condition,
    build_support_condition,
)
from piano.training.train_anchordiff import _build_dataset


# ─── Channel layout (31-D = C41 18 + S4 13) ────────────────────────────
CHANNEL_NAMES: tuple[str, ...] = (
    # C41
    "lw_dx", "lw_dy", "lw_dz",
    "rw_dx", "rw_dy", "rw_dz",
    "lk_dx", "lk_dy", "lk_dz",
    "rk_dx", "rk_dy", "rk_dz",
    "neck_dx", "neck_dy", "neck_dz",
    "pelvis_dx", "pelvis_dz", "pelvis_dy",
    # S4
    "foot_stance_L", "foot_stance_R",
    "ankle_h_L", "ankle_h_R",
    "walking_mask",
    "phase_sin_L", "phase_cos_L",
    "phase_sin_R", "phase_cos_R",
    "footstep_x_L", "footstep_z_L",
    "footstep_x_R", "footstep_z_R",
)
assert len(CHANNEL_NAMES) == 31

CHANNEL_GROUPS: dict[str, tuple[int, int]] = {
    # C41 — per body part (the failure axis we most care about).
    "left_wrist":  (0, 3),
    "right_wrist": (3, 6),
    "left_knee":   (6, 9),
    "right_knee":  (9, 12),
    "neck":        (12, 15),
    "pelvis_delta": (15, 18),
    # S4 — per semantic family.
    "foot_stance":   (18, 20),
    "ankle_height":  (20, 22),
    "walking_mask":  (22, 23),
    "phase":         (23, 27),
    "footstep":      (27, 31),
}

# Indices used by the special-case checks.
IDX_PHASE_SIN_COS = (23, 24, 25, 26)
IDX_STANCE = (18, 19)
IDX_WALKING = 22

PSD_NPERSEG: int = 64


def _read_selection(path: Path) -> set[tuple[str, str]]:
    sel = json.loads(path.read_text("utf-8"))
    items = sel.get("selected") or sel.get("candidates") or sel.get("clips") or []
    if not items:
        raise SystemExit(f"empty selection: {path}")
    return {(e["subset"], e["seq_id"]) for e in items}


def _read_generated(path: Path) -> tuple[np.ndarray, int]:
    """Load one Stage-1.5 generated cache. Returns (raw (T, 31), valid_T)."""
    data = np.load(path)
    if "stage2_coarse_extra" not in data.files or "stage2_support" not in data.files:
        raise KeyError(
            f"{path} missing stage2_coarse_extra/stage2_support (got {data.files})"
        )
    c41 = data["stage2_coarse_extra"].astype(np.float32)        # (T, 18)
    s4 = data["stage2_support"].astype(np.float32)              # (T, 13)
    if c41.shape[-1] != 18 or s4.shape[-1] != 13:
        raise RuntimeError(
            f"{path} unexpected shapes: c41 {c41.shape} s4 {s4.shape}"
        )
    arr = np.concatenate([c41, s4], axis=-1).astype(np.float32)  # (T, 31)
    valid_T = int(data["valid_T"]) if "valid_T" in data.files else arr.shape[0]
    return arr, valid_T


def _build_gt(joints_22: np.ndarray, fps: float = 20.0) -> np.ndarray:
    """Compute GT (C41, S4) for one clip via the same oracle builders the
    trainer + PB1 inference use. Returns (T, 31)."""
    coarse, _ = build_coarse_condition(joints_22, "C41-current")    # (T, 18)
    support, _ = build_support_condition(joints_22, "S4-S1-phase-footstep", fps=fps)  # (T, 13)
    return np.concatenate([coarse, support], axis=-1).astype(np.float32)


def _finite_diff(x: np.ndarray, n: int = 1) -> np.ndarray:
    out = x.astype(np.float32, copy=True)
    for _ in range(n):
        d = np.diff(out, axis=0)
        out = np.concatenate([np.zeros_like(out[:1]), d], axis=0)
    return out


def _per_clip_stats(arr: np.ndarray) -> dict[str, np.ndarray]:
    """Per-channel moments + velocity + accel statistics. arr (T, 31)."""
    vel = _finite_diff(arr, n=1)
    accel = _finite_diff(arr, n=2)
    return {
        "mean":      arr.mean(axis=0),
        "std":       arr.std(axis=0),
        "skew":      skew(arr, axis=0, bias=False, nan_policy="omit"),
        "kurt":      kurtosis(arr, axis=0, bias=False, nan_policy="omit"),
        "vel_rms":   np.sqrt((vel ** 2).mean(axis=0)),
        "vel_max":   np.abs(vel).max(axis=0),
        "accel_rms": np.sqrt((accel ** 2).mean(axis=0)),
        "accel_max": np.abs(accel).max(axis=0),
    }


def _psd_per_group(arr: np.ndarray, fps: float = 20.0) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    T = arr.shape[0]
    if T < PSD_NPERSEG:
        pad = np.zeros((PSD_NPERSEG - T, arr.shape[1]), dtype=arr.dtype)
        block_arr = np.concatenate([arr, pad], axis=0)
    else:
        block_arr = arr
    for name, (lo, hi) in CHANNEL_GROUPS.items():
        block = block_arr[:, lo:hi]
        freqs, psd = sp_signal.welch(
            block, fs=fps, nperseg=PSD_NPERSEG, axis=0,
        )
        psd_sum = psd.sum(axis=-1)
        out[name] = {
            "freqs": freqs.astype(np.float32),
            "psd_sum": psd_sum.astype(np.float32),
            "total_energy": float(psd_sum.sum()),
        }
    return out


def _phase_unit_circle_violation(arr: np.ndarray) -> dict[str, float]:
    """Phase channels (sin, cos for L+R) should satisfy sin² + cos² = 1.
    Returns mean |sin² + cos² − 1| separately for L and R."""
    sin_L = arr[:, IDX_PHASE_SIN_COS[0]]
    cos_L = arr[:, IDX_PHASE_SIN_COS[1]]
    sin_R = arr[:, IDX_PHASE_SIN_COS[2]]
    cos_R = arr[:, IDX_PHASE_SIN_COS[3]]
    return {
        "L_unit_circle_dev": float(np.mean(np.abs(sin_L ** 2 + cos_L ** 2 - 1.0))),
        "R_unit_circle_dev": float(np.mean(np.abs(sin_R ** 2 + cos_R ** 2 - 1.0))),
    }


def _binary_channel_stats(arr: np.ndarray, idx: int) -> dict[str, float]:
    """For a logits-style channel that should encode 0/1 (stance, walking),
    report mean (sigmoid would push toward 0/1) and saturation rates."""
    ch = arr[:, idx]
    # Stage-1.5 emits logits via Linear (no sigmoid) per train_stage1p5 docs.
    # GT clips to [0, 1]. So if pred has collapsed to a single logit, we
    # see it as a degenerate raw value.
    return {
        "mean":           float(ch.mean()),
        "std":            float(ch.std()),
        "frac_below_-2":  float((ch < -2.0).mean()),   # saturated "off"
        "frac_above_+2":  float((ch > 2.0).mean()),    # saturated "on"
        "frac_in_pm1":    float((np.abs(ch) <= 1.0).mean()),
    }


# ───────────────────────── reporting helpers ─────────────────────────


def _agg_per_channel(per_clip: list[dict[str, np.ndarray]], key: str) -> np.ndarray:
    stack = np.stack([c[key] for c in per_clip], axis=0)
    return stack.mean(axis=0)


def _mk_group_summary(
    gt_agg: dict[str, np.ndarray],
    pred_agg: dict[str, np.ndarray],
) -> list[str]:
    lines = [
        "| group | channels | gt mean |Δ| | std ratio | vel_rms ratio | accel_rms ratio |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, (lo, hi) in CHANNEL_GROUPS.items():
        mean_abs_d = float(np.abs(pred_agg["mean"][lo:hi] - gt_agg["mean"][lo:hi]).mean())
        std_ratio = float(np.mean(
            pred_agg["std"][lo:hi] / np.maximum(gt_agg["std"][lo:hi], 1e-6)
        ))
        vel_ratio = float(np.mean(
            pred_agg["vel_rms"][lo:hi] / np.maximum(gt_agg["vel_rms"][lo:hi], 1e-6)
        ))
        acc_ratio = float(np.mean(
            pred_agg["accel_rms"][lo:hi] / np.maximum(gt_agg["accel_rms"][lo:hi], 1e-6)
        ))
        lines.append(
            f"| {name} | [{lo}:{hi}] | {mean_abs_d:.3f} | "
            f"{std_ratio:.2f} | {vel_ratio:.2f} | {acc_ratio:.2f} |"
        )
    return lines


def _mk_per_channel_table(
    gt_agg: dict[str, np.ndarray],
    pred_agg: dict[str, np.ndarray],
) -> list[str]:
    cols = ["mean", "std", "vel_rms", "accel_rms"]
    headers = ["ch", "name"]
    for c in cols:
        headers.extend([f"gt_{c}", f"pred_{c}", "Δ", "ratio"])
    sep = ["---:" if (i not in (0, 1)) else (":---" if i == 1 else "---:")
           for i in range(len(headers))]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(sep) + "|",
    ]
    for c in range(31):
        row = [f"{c}", CHANNEL_NAMES[c]]
        for col in cols:
            g = float(gt_agg[col][c])
            p = float(pred_agg[col][c])
            d = p - g
            r = p / (abs(g) + 1e-6)
            row.extend([f"{g:+.3f}", f"{p:+.3f}", f"{d:+.3f}", f"{r:.2f}"])
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _mk_psd_summary(
    gt_psd: list[dict[str, dict[str, np.ndarray]]],
    pred_psd: list[dict[str, dict[str, np.ndarray]]],
) -> list[str]:
    bands = [("low (0-1 Hz)", 0.0, 1.0),
             ("mid (1-4 Hz)", 1.0, 4.0),
             ("high (4-10 Hz)", 4.0, 10.0)]
    lines = [
        "| group | " + " | ".join(b[0] for b in bands) + " | total ratio |",
        "|---|" + "|".join([":---:" for _ in bands]) + "|:---:|",
    ]
    for group in CHANNEL_GROUPS:
        band_ratios: list[str] = []
        gt_total = 0.0
        pred_total = 0.0
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
                        help="Stage-1.5 generated cache dir, e.g. "
                             "analyses/round32_stage1p5_substitute_conds_v8_audit/val")
    parser.add_argument("--stage1p5-cfg", type=Path, required=True,
                        help="Stage-1.5 training config (used to build the "
                             "val dataset; only data section is consulted).")
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bucket", default="val", choices=["train", "val"])
    parser.add_argument("--fps", type=float, default=20.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(str(args.stage1p5_cfg))

    sel_pairs = _read_selection(args.selection_json)
    print(f"[audit] selection size = {len(sel_pairs)}")

    dataset = _build_dataset(cfg, args.bucket, augment=False)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        collate_fn=collate_hoi, num_workers=0,
    )

    per_clip_gt_stats: list[dict[str, np.ndarray]] = []
    per_clip_pred_stats: list[dict[str, np.ndarray]] = []
    per_clip_gt_psd: list[dict[str, dict[str, np.ndarray]]] = []
    per_clip_pred_psd: list[dict[str, dict[str, np.ndarray]]] = []
    per_clip_phase_gt: list[dict[str, float]] = []
    per_clip_phase_pred: list[dict[str, float]] = []
    per_clip_stance_L_pred: list[dict[str, float]] = []
    per_clip_stance_R_pred: list[dict[str, float]] = []
    per_clip_walking_pred: list[dict[str, float]] = []
    per_clip_meta: list[dict[str, Any]] = []

    gt_arrays: list[np.ndarray] = []
    pred_arrays: list[np.ndarray] = []

    n_found = 0
    for batch in loader:
        subset = str(batch["subset"][0])
        seq_id = str(batch["seq_id"][0])
        if (subset, seq_id) not in sel_pairs:
            continue

        cache_path = args.upstream_dir / subset / f"{seq_id}.npz"
        if not cache_path.exists():
            print(f"[audit] WARN: missing cache for ({subset}, {seq_id}) at {cache_path}")
            continue

        seq_len = int(batch["seq_len"][0].item())
        joints_np = batch["joints"][0].cpu().numpy().astype(np.float32)  # (T, 22, 3)
        T = int(joints_np.shape[0])
        valid_T = min(T, seq_len)
        gt = _build_gt(joints_np[:valid_T], fps=args.fps)                # (valid_T, 31)

        gen, gen_valid = _read_generated(cache_path)                      # (T_gen, 31)
        if gen.shape[0] < valid_T:
            print(f"[audit] WARN: ({subset}, {seq_id}) gen T={gen.shape[0]} < valid_T={valid_T}")
            valid_T = min(valid_T, gen.shape[0])
        pred = gen[:valid_T]
        gt = gt[:valid_T]

        per_clip_gt_stats.append(_per_clip_stats(gt))
        per_clip_pred_stats.append(_per_clip_stats(pred))
        per_clip_gt_psd.append(_psd_per_group(gt))
        per_clip_pred_psd.append(_psd_per_group(pred))
        per_clip_phase_gt.append(_phase_unit_circle_violation(gt))
        per_clip_phase_pred.append(_phase_unit_circle_violation(pred))
        per_clip_stance_L_pred.append(_binary_channel_stats(pred, IDX_STANCE[0]))
        per_clip_stance_R_pred.append(_binary_channel_stats(pred, IDX_STANCE[1]))
        per_clip_walking_pred.append(_binary_channel_stats(pred, IDX_WALKING))
        per_clip_meta.append({
            "subset": subset, "seq_id": seq_id, "valid_T": valid_T,
        })
        gt_arrays.append(gt)
        pred_arrays.append(pred)

        n_found += 1
        if n_found % 8 == 0:
            print(f"[audit] processed {n_found}/{len(sel_pairs)}")

    if n_found == 0:
        raise SystemExit("[audit] no clips matched the selection AND had a cache.")

    print(f"[audit] processed {n_found} clips total.")

    gt_agg = {k: _agg_per_channel(per_clip_gt_stats, k)
              for k in per_clip_gt_stats[0].keys()}
    pred_agg = {k: _agg_per_channel(per_clip_pred_stats, k)
                for k in per_clip_pred_stats[0].keys()}

    # ── Markdown report ─────────────────────────────────────────────
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md: list[str] = [
        "# R32 Phase 1 — Stage-1.5 generated vs GT oracle (C41, S4) dynamic-info audit",
        "",
        f"Generated: {stamp}",
        f"Cache    : `{args.upstream_dir}`",
        f"Selection: `{args.selection_json}` ({n_found} clips)",
        "",
        "Mirrors `round31_phase1_dyn_audit.py`. Output is 31-D (C41 18 + S4 13),",
        "raw space (Stage-1.5 trains in raw — no z-scoring on either side).",
        "",
        "## 1. Per-channel-group rollup",
        "",
        "Reading the columns:",
        "- **gt mean |Δ|** : mean abs difference between pred and GT per-channel-mean across clips.",
        "- **std ratio** : pred std / GT std (mean over channels in group).",
        "  < 1.0 → pred under-disperses; signature of mode collapse.",
        "- **vel_rms / accel_rms ratio** : pred finite-diff RMS over GT.",
        "  < 1.0 → pred is smoother / under-articulated.",
        "",
    ]
    md += _mk_group_summary(gt_agg, pred_agg)

    md += [
        "",
        "## 2. Per-channel detail (mean / std / vel_rms / accel_rms)",
        "",
        "All numbers raw-space. Δ = pred − GT. ratio = pred / |GT|.",
        "",
    ]
    md += _mk_per_channel_table(gt_agg, pred_agg)

    md += [
        "",
        "## 3. PSD band-energy ratio (pred / GT, per channel group)",
        "",
        "Welch PSD per clip then summed within frequency bands.",
        "fps=20, so high band 4-10 Hz is the Nyquist top half.",
        "Below 1.0 means pred has less energy in that band; above 1.0 means more.",
        "",
    ]
    md += _mk_psd_summary(per_clip_gt_psd, per_clip_pred_psd)

    # ── §4: Phase unit-circle violation ─────────────────────────────
    phase_L_gt = float(np.mean([c["L_unit_circle_dev"] for c in per_clip_phase_gt]))
    phase_R_gt = float(np.mean([c["R_unit_circle_dev"] for c in per_clip_phase_gt]))
    phase_L_pred = float(np.mean([c["L_unit_circle_dev"] for c in per_clip_phase_pred]))
    phase_R_pred = float(np.mean([c["R_unit_circle_dev"] for c in per_clip_phase_pred]))
    md += [
        "",
        "## 4. Phase unit-circle violation (raw S4 phase channels)",
        "",
        "Stage-1.5 outputs phase as (sin, cos) for L and R legs. For a valid",
        "phase angle these satisfy sin² + cos² = 1. Pred deviation = how far",
        "the learned 2-channel pair sits off the unit circle.",
        "",
        "| leg | GT |Δ| | pred |Δ| |",
        "|---|---:|---:|",
        f"| L | {phase_L_gt:.4f} | {phase_L_pred:.4f} |",
        f"| R | {phase_R_gt:.4f} | {phase_R_pred:.4f} |",
    ]

    # ── §5: Binary-channel dead-channel check (stance + walking) ────
    def _avg_bin(rows: list[dict[str, float]]) -> dict[str, float]:
        keys = list(rows[0].keys())
        return {k: float(np.mean([r[k] for r in rows])) for k in keys}

    stance_L_pred_avg = _avg_bin(per_clip_stance_L_pred)
    stance_R_pred_avg = _avg_bin(per_clip_stance_R_pred)
    walking_pred_avg = _avg_bin(per_clip_walking_pred)
    md += [
        "",
        "## 5. Binary-channel saturation check",
        "",
        "S4 stance + walking_mask channels are 0/1 in the GT (BCE training",
        "target). Stage-1.5 emits raw logits; a healthy logit distribution",
        "should straddle 0. Dead channels saturate either to large negative",
        "(always-off) or large positive (always-on) values.",
        "",
        "| channel | mean | std | %< -2 | %> +2 | %in [-1, +1] |",
        "|---|---:|---:|---:|---:|---:|",
        f"| foot_stance_L | {stance_L_pred_avg['mean']:+.3f} | "
        f"{stance_L_pred_avg['std']:.3f} | {stance_L_pred_avg['frac_below_-2']:.2f} | "
        f"{stance_L_pred_avg['frac_above_+2']:.2f} | {stance_L_pred_avg['frac_in_pm1']:.2f} |",
        f"| foot_stance_R | {stance_R_pred_avg['mean']:+.3f} | "
        f"{stance_R_pred_avg['std']:.3f} | {stance_R_pred_avg['frac_below_-2']:.2f} | "
        f"{stance_R_pred_avg['frac_above_+2']:.2f} | {stance_R_pred_avg['frac_in_pm1']:.2f} |",
        f"| walking_mask | {walking_pred_avg['mean']:+.3f} | "
        f"{walking_pred_avg['std']:.3f} | {walking_pred_avg['frac_below_-2']:.2f} | "
        f"{walking_pred_avg['frac_above_+2']:.2f} | {walking_pred_avg['frac_in_pm1']:.2f} |",
    ]

    # ── §6: Per-clip wrist drift (C41 left_wrist [0:3] + right_wrist [3:6]) ──
    md += [
        "",
        "## 6. Per-clip C41 wrist drift (RMS pred − GT, in raw m)",
        "",
        "RMS of `pred_C41[t] − gt_C41[t]` over t, per clip. Wrist channels",
        "[0:6] are pelvis-local current-yaw Δxyz; t=0 row is exactly 0 by",
        "construction (delta against frame 0). If `rms_at_t0` is non-zero,",
        "Stage-1.5 is violating the frame-0 invariant — analog of the R31",
        "Phase 1 root_local rms_at_t0 = 9.8 cm finding.",
        "",
    ]
    lw_drift = []
    rw_drift = []
    for gt_arr, pred_arr in zip(gt_arrays, pred_arrays):
        diff_lw = pred_arr[:, 0:3] - gt_arr[:, 0:3]
        diff_rw = pred_arr[:, 3:6] - gt_arr[:, 3:6]
        rms_lw = np.sqrt((diff_lw ** 2).sum(-1))    # (T,)
        rms_rw = np.sqrt((diff_rw ** 2).sum(-1))
        lw_drift.append({
            "rms_at_t0": float(rms_lw[0]),
            "rms_mean": float(rms_lw.mean()),
            "rms_max": float(rms_lw.max()),
        })
        rw_drift.append({
            "rms_at_t0": float(rms_rw[0]),
            "rms_mean": float(rms_rw.mean()),
            "rms_max": float(rms_rw.max()),
        })

    def _summary_drift(drift_list: list[dict[str, float]], label: str) -> list[str]:
        rms_t0 = np.array([d["rms_at_t0"] for d in drift_list])
        rms_mean = np.array([d["rms_mean"] for d in drift_list])
        rms_max = np.array([d["rms_max"] for d in drift_list])
        return [
            f"### {label}",
            "| stat | t=0 (m) | mean over t (m) | max over t (m) |",
            "|---|---:|---:|---:|",
            f"| median | {np.median(rms_t0):.4f} | {np.median(rms_mean):.4f} | {np.median(rms_max):.4f} |",
            f"| q25    | {np.quantile(rms_t0, 0.25):.4f} | {np.quantile(rms_mean, 0.25):.4f} | {np.quantile(rms_max, 0.25):.4f} |",
            f"| q75    | {np.quantile(rms_t0, 0.75):.4f} | {np.quantile(rms_mean, 0.75):.4f} | {np.quantile(rms_max, 0.75):.4f} |",
            f"| max    | {rms_t0.max():.4f} | {rms_mean.max():.4f} | {rms_max.max():.4f} |",
            "",
        ]

    md += _summary_drift(lw_drift, "left_wrist drift")
    md += _summary_drift(rw_drift, "right_wrist drift")

    # ── §7: Top mismatched channels ─────────────────────────────────
    ratio = pred_agg["std"] / np.maximum(gt_agg["std"], 1e-6)
    log_ratio_abs = np.abs(np.log(np.maximum(ratio, 1e-6)))
    order = np.argsort(-log_ratio_abs)
    md += [
        "",
        "## 7. Top-5 channels by std mismatch (pred std / gt std deviation from 1)",
        "",
        "| ch | name | gt std | pred std | ratio | log(ratio) |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for i in order[:5]:
        md.append(
            f"| {i} | {CHANNEL_NAMES[i]} | {gt_agg['std'][i]:.3f} | "
            f"{pred_agg['std'][i]:.3f} | {ratio[i]:.3f} | {np.log(ratio[i]):+.3f} |"
        )

    vel_ratio = pred_agg["vel_rms"] / np.maximum(gt_agg["vel_rms"], 1e-6)
    log_vel_ratio_abs = np.abs(np.log(np.maximum(vel_ratio, 1e-6)))
    order_v = np.argsort(-log_vel_ratio_abs)
    md += [
        "",
        "## 8. Top-5 channels by velocity mismatch (pred vel_rms / gt vel_rms)",
        "",
        "| ch | name | gt vel_rms | pred vel_rms | ratio | log(ratio) |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for i in order_v[:5]:
        md.append(
            f"| {i} | {CHANNEL_NAMES[i]} | {gt_agg['vel_rms'][i]:.3f} | "
            f"{pred_agg['vel_rms'][i]:.3f} | {vel_ratio[i]:.3f} | {np.log(vel_ratio[i]):+.3f} |"
        )

    md += [
        "",
        "## 9. Interpretation cheat-sheet",
        "",
        "- **std ratio < 0.7 on a C41 wrist group** → V0's MSE collapsed the",
        "  wrist channels to mean; same failure mode as R31 V0.",
        "- **PSD high-band ratio < 0.5 on wrist** → under-articulation; same",
        "  diagnosis path as R31 → V7 anti-mode-collapse stack applies.",
        "- **stance saturation (%< -2 or %> +2 > 0.5)** → BCE collapsed the",
        "  logit to constant; the BCE target probably wasn't matching reality.",
        "- **phase unit-circle violation > 0.1** → phase pair lost geometric",
        "  consistency; needs an explicit phase unit-norm constraint stronger",
        "  than the current `(sin² + cos² − 1)²` w=0.05 term.",
        "- **C41 wrist rms_at_t0 > 0.01 m** → frame-0 invariant violated;",
        "  analog of R31 root_local rms_at_t0 = 9.8 cm.",
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
        "phase_unit_circle_gt": {
            "L": phase_L_gt, "R": phase_R_gt,
        },
        "phase_unit_circle_pred": {
            "L": phase_L_pred, "R": phase_R_pred,
        },
        "stance_L_pred": stance_L_pred_avg,
        "stance_R_pred": stance_R_pred_avg,
        "walking_pred": walking_pred_avg,
        "per_clip_meta": per_clip_meta,
        "per_clip_lw_drift": lw_drift,
        "per_clip_rw_drift": rw_drift,
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

    # ── Per-clip raw dump ───────────────────────────────────────────
    dump_path = args.out_dir / "per_clip_dump.npz"
    max_T = max(g.shape[0] for g in gt_arrays)
    pad_gt = np.zeros((len(gt_arrays), max_T, 31), dtype=np.float32)
    pad_pred = np.zeros((len(pred_arrays), max_T, 31), dtype=np.float32)
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
