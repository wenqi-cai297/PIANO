"""Round-34 C41/S4 quality metrics (per ChatGPT followup §7.4).

For a given Stage-1.5 variant's substitute_conds dir (the pred), compare
against the oracle GT dump (from ``dump_gt_c41_s4_oracle.py``) and report:

  - C41 wrist LOW-band PSD ratio  (pred / GT)  [ratio < 1.0 = pred under-energizes]
  - C41 wrist LOW-band MSE  (rFFT-MSE, the loss R34-C2 optimizes)
  - C41 wrist HIGH-band MSE
  - C41 wrist FULL-band MSE
  - C41 non-wrist (knee+neck+pelvis) MSE
  - S4 stance BCE-equivalent + phase unit-circle violation

Operates on the same 48 balanced val clips D1/D2 used. No PB1 inference;
no training. CPU only. ~30 s on 48 clips.

This is the "did the audit-level signal move?" diagnostic. Run for each
R34 variant after training to separate "PSD ratio moved but drift didn't"
from "PSD ratio moved AND drift moved" decisions (per ChatGPT §9.4 / §9.3).

Run:
    python -u scripts/stage_a_generator/round34_c41_s4_quality_metrics.py \\
        --gt-dir   analyses/2026-05-31_stage1p5_wrist_external_review_work/oracle_dump \\
        --pred-dir analyses/round32_stage1p5_substitute_conds_r34_<variant> \\
        --bucket   val \\
        --out      analyses/round34_rfft_followup_20260531/c41_s4_quality_<variant>.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Re-use FFT split + IO from D1.
import sys
sys.path.insert(0, str(Path(__file__).parent))
from build_d1_spectral_swap_variants import (   # type: ignore[import]
    split_low_high, _load_npz,
    C41_DIM, S4_DIM,
    C41_WRIST_SLICE, C41_NON_WRIST_SLICE,
    FPS_DEFAULT, CUTOFF_HZ_DEFAULT,
)


# S4 channel indices (S4-LOCAL; binding from src/piano/data/stage2_oracle_conditions.py)
S4_STANCE_SLICE = slice(0, 2)         # foot_stance L, R (BCE target {0,1})
S4_ANKLE_H_SLICE = slice(2, 4)        # ankle_height_norm L, R
S4_WALKING_SLICE = slice(4, 5)        # walking_mask (BCE target)
S4_PHASE_SLICE = slice(5, 9)          # phase sin/cos L, R (unit circle)
S4_FOOTSTEP_SLICE = slice(9, 13)      # footstep x/z L, R


def psd_lowband_ratio(
    pred: np.ndarray, gt: np.ndarray, valid_T: int,
    fps: float = FPS_DEFAULT, cutoff_hz: float = CUTOFF_HZ_DEFAULT,
) -> float:
    """Mean-over-channels ratio of LOW-band PSD energy (pred / GT).

    ratio < 1.0 = pred under-energizes the low band (the Phase 1 audit's
    signature of the wrist failure mode).
    """
    pred_v = pred[:valid_T]
    gt_v = gt[:valid_T]
    pred_low, _ = split_low_high(pred_v, fps=fps, cutoff_hz=cutoff_hz)
    gt_low, _ = split_low_high(gt_v, fps=fps, cutoff_hz=cutoff_hz)
    pred_energy = float((pred_low ** 2).mean())
    gt_energy = float((gt_low ** 2).mean())
    if gt_energy < 1e-12:
        return float("nan")
    return pred_energy / gt_energy


def band_mse(
    pred: np.ndarray, gt: np.ndarray, valid_T: int, band: str,
    fps: float = FPS_DEFAULT, cutoff_hz: float = CUTOFF_HZ_DEFAULT,
) -> float:
    """Mean-square error in time domain after FFT-band filtering.

    band: 'low' | 'high' | 'full'.
    """
    pred_v = pred[:valid_T]
    gt_v = gt[:valid_T]
    if band == "full":
        return float(((pred_v - gt_v) ** 2).mean())
    pred_low, pred_high = split_low_high(pred_v, fps=fps, cutoff_hz=cutoff_hz)
    gt_low, gt_high = split_low_high(gt_v, fps=fps, cutoff_hz=cutoff_hz)
    if band == "low":
        return float(((pred_low - gt_low) ** 2).mean())
    if band == "high":
        return float(((pred_high - gt_high) ** 2).mean())
    raise ValueError(f"band must be 'low' | 'high' | 'full'; got {band!r}")


def stance_bce(pred: np.ndarray, gt: np.ndarray, valid_T: int) -> float:
    """Sigmoid-then-BCE on S4 stance channels [0:2].

    Pred from Stage-1.5 is RAW logits (no sigmoid in model output for BCE
    channels). GT is in {0, 1}. We compute BCE with logits in numpy.
    """
    pred_v = pred[:valid_T, S4_STANCE_SLICE]
    gt_v = gt[:valid_T, S4_STANCE_SLICE]
    # Numerically stable BCE-with-logits: max(z, 0) - z*y + log(1 + exp(-|z|))
    z = pred_v.astype(np.float64)
    y = np.clip(gt_v.astype(np.float64), 0.0, 1.0)
    bce = np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))
    return float(bce.mean())


def phase_unit_violation(pred: np.ndarray, valid_T: int) -> float:
    """Mean abs (sin² + cos² − 1) per leg on S4 phase channels [5:9]."""
    pred_v = pred[:valid_T, S4_PHASE_SLICE]
    sin_l = pred_v[..., 0]; cos_l = pred_v[..., 1]
    sin_r = pred_v[..., 2]; cos_r = pred_v[..., 3]
    viol_l = np.abs(sin_l ** 2 + cos_l ** 2 - 1.0)
    viol_r = np.abs(sin_r ** 2 + cos_r ** 2 - 1.0)
    return float(0.5 * (viol_l.mean() + viol_r.mean()))


def aggregate_clip(
    gt_path: Path, pred_path: Path,
    fps: float = FPS_DEFAULT, cutoff_hz: float = CUTOFF_HZ_DEFAULT,
) -> dict[str, float]:
    c41_gt, s4_gt, vt = _load_npz(gt_path)
    c41_pred, s4_pred, _ = _load_npz(pred_path)
    if c41_gt.shape != c41_pred.shape or s4_gt.shape != s4_pred.shape:
        raise RuntimeError(
            f"shape mismatch for {gt_path.name}: "
            f"c41 {c41_gt.shape}/{c41_pred.shape} s4 {s4_gt.shape}/{s4_pred.shape}"
        )

    # C41 wrist
    c41_wrist_gt = c41_gt[..., C41_WRIST_SLICE]
    c41_wrist_pred = c41_pred[..., C41_WRIST_SLICE]
    psd_ratio = psd_lowband_ratio(c41_wrist_pred, c41_wrist_gt, vt, fps, cutoff_hz)
    wrist_mse_low = band_mse(c41_wrist_pred, c41_wrist_gt, vt, "low", fps, cutoff_hz)
    wrist_mse_high = band_mse(c41_wrist_pred, c41_wrist_gt, vt, "high", fps, cutoff_hz)
    wrist_mse_full = band_mse(c41_wrist_pred, c41_wrist_gt, vt, "full", fps, cutoff_hz)

    # C41 non-wrist
    c41_nw_gt = c41_gt[..., C41_NON_WRIST_SLICE]
    c41_nw_pred = c41_pred[..., C41_NON_WRIST_SLICE]
    nw_mse_full = band_mse(c41_nw_pred, c41_nw_gt, vt, "full", fps, cutoff_hz)

    # S4
    stance_bce_val = stance_bce(s4_pred, s4_gt, vt)
    phase_viol = phase_unit_violation(s4_pred, vt)
    s4_full_mse = float(((s4_pred[:vt] - s4_gt[:vt]) ** 2).mean())

    return {
        "c41_wrist_lowband_psd_ratio": psd_ratio,
        "c41_wrist_lowband_mse": wrist_mse_low,
        "c41_wrist_highband_mse": wrist_mse_high,
        "c41_wrist_full_mse": wrist_mse_full,
        "c41_nonwrist_full_mse": nw_mse_full,
        "s4_full_mse": s4_full_mse,
        "s4_stance_bce": stance_bce_val,
        "s4_phase_unit_violation": phase_viol,
        "valid_T": float(vt),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", type=Path, required=True,
                    help="root containing <bucket>/<subset>/<seq_id>.npz with GT C41/S4")
    ap.add_argument("--pred-dir", type=Path, required=True,
                    help="root containing <bucket>/<subset>/<seq_id>.npz with Stage-1.5 pred")
    ap.add_argument("--bucket", choices=["train", "val"], default="val")
    ap.add_argument("--out", type=Path, required=True,
                    help="output markdown report")
    ap.add_argument("--variant-label", type=str, default="(unspecified)",
                    help="label printed in the report header")
    ap.add_argument("--fps", type=float, default=FPS_DEFAULT)
    ap.add_argument("--cutoff-hz", type=float, default=CUTOFF_HZ_DEFAULT)
    args = ap.parse_args()

    gt_root = args.gt_dir / args.bucket
    pred_root = args.pred_dir / args.bucket
    if not gt_root.is_dir():
        raise SystemExit(f"--gt-dir missing bucket subdir: {gt_root}")
    if not pred_root.is_dir():
        raise SystemExit(f"--pred-dir missing bucket subdir: {pred_root}")

    pairs: list[tuple[str, str]] = []
    for subset_d in sorted(p for p in gt_root.iterdir() if p.is_dir()):
        for npz in sorted(subset_d.glob("*.npz")):
            pairs.append((subset_d.name, npz.stem))
    if not pairs:
        raise SystemExit(f"no .npz under {gt_root}")

    per_clip = []
    n_missing = 0
    for subset, seq_id in pairs:
        gt_p = gt_root / subset / f"{seq_id}.npz"
        pred_p = pred_root / subset / f"{seq_id}.npz"
        if not pred_p.exists():
            n_missing += 1
            continue
        m = aggregate_clip(gt_p, pred_p, fps=args.fps, cutoff_hz=args.cutoff_hz)
        m["subset"] = subset
        m["seq_id"] = seq_id
        per_clip.append(m)

    if not per_clip:
        raise SystemExit(f"no pred files matched gt under {pred_root}")

    keys = [
        "c41_wrist_lowband_psd_ratio",
        "c41_wrist_lowband_mse",
        "c41_wrist_highband_mse",
        "c41_wrist_full_mse",
        "c41_nonwrist_full_mse",
        "s4_full_mse",
        "s4_stance_bce",
        "s4_phase_unit_violation",
    ]
    agg = {k: float(np.mean([c[k] for c in per_clip if np.isfinite(c[k])])) for k in keys}

    md = [
        f"# R34 C41/S4 quality metrics — {args.variant_label}",
        "",
        f"- gt_dir   : `{args.gt_dir}`",
        f"- pred_dir : `{args.pred_dir}`",
        f"- bucket   : {args.bucket}",
        f"- n_clips  : {len(per_clip)} (missing pred: {n_missing})",
        f"- fps      : {args.fps}",
        f"- cutoff_hz: {args.cutoff_hz}",
        "",
        "## Aggregate (mean over clips)",
        "",
        "| metric | value | reference (Stage-1.5 V0 audit) |",
        "|---|---:|---|",
        f"| C41 wrist LOW-band PSD ratio | {agg['c41_wrist_lowband_psd_ratio']:.3f} | left_wrist 0.50, right_wrist 0.45 (R32 Phase 1 audit) |",
        f"| C41 wrist LOW-band MSE       | {agg['c41_wrist_lowband_mse']:.4f}      | — |",
        f"| C41 wrist HIGH-band MSE      | {agg['c41_wrist_highband_mse']:.4f}     | — |",
        f"| C41 wrist FULL MSE           | {agg['c41_wrist_full_mse']:.4f}         | — |",
        f"| C41 non-wrist FULL MSE       | {agg['c41_nonwrist_full_mse']:.4f}      | — |",
        f"| S4 FULL MSE                  | {agg['s4_full_mse']:.4f}                | — |",
        f"| S4 stance BCE                | {agg['s4_stance_bce']:.4f}              | — |",
        f"| S4 phase unit-circle violation | {agg['s4_phase_unit_violation']:.4f} | 0.027–0.030 (R32 Phase 1 audit) |",
        "",
        "## Interpretation (per ChatGPT followup §9.3/§9.4 decision rules)",
        "",
        "If this variant is R34-C2 (lowband-loss-only) and downstream `drift_max` did NOT improve:",
        "",
        "- C41 wrist LOW-band PSD ratio close to 1.0 + drift_max unchanged → §9.4 branch:",
        "  audit signal moved but downstream metric decoupled. Pivot to contact-window",
        "  weighted wrist MSE / DCT loss / inference-time spatial guidance (§10).",
        "- C41 wrist LOW-band MSE NOT decreased vs C0 control → §9.3 branch: implementation",
        "  or loss-scale problem; check `r34_wrist_lowband_weighted` magnitude in train log.",
        "",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {args.out}")

    # Also dump per-clip JSON for later inspection
    json_path = args.out.with_suffix(".json")
    json_path.write_text(
        json.dumps({"aggregate": agg, "per_clip": per_clip}, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
