"""DCT-vs-FFT sanity check for the R34 low-band loss basis choice.

Per ChatGPT followup §8: rFFT assumes periodic boundary which doesn't
fit non-periodic motion clips. DCT-II uses an implicit reflection
boundary, which is more appropriate for finite clips.

This script builds 3 variants in the substitute_conds-format directory
schema (drop-in replacement for ``run_round34_d1_d2_diag.sh`` Phase 3):

    DCT0  base generated (= D1-V0; pred unchanged)
    DCT1  wrist DCT-low oracle: c41[:, 0:6] = gt_DCT_low + pred_DCT_high
    DCT2  wrist DCT-high oracle: c41[:, 0:6] = pred_DCT_low + gt_DCT_high

Decision (ChatGPT §8):

    If DCT1 reproduces D1-V2's drift_max improvement → rFFT basis choice
    is fine; the LOW-band causality survives the basis change.

    If DCT1 substantially differs from D1-V2 → rFFT periodic-boundary
    artifact is masking the true causal locus; switch R34-C2 to use a
    DCT-based loss instead of rFFT.

CPU only. ~30 s on 48 clips.

Cutoff: brief §5.3 says cutoff_hz=1.0 for the FFT. For DCT we need to
pick an "equivalent" cutoff. DCT-II of length T has frequencies at
k/(2T) cycles per sample (k=0..T-1), so freq_hz = k * fps / (2T).
For T=196, fps=20: cutoff_hz=1.0 → k_max = floor(2 * T * cutoff_hz / fps)
                                          = floor(2 * 196 * 1.0 / 20) = 19.

Equivalently: rFFT cutoff_hz=1.0 covers freq bins {0, 1/T, ..., k/T}
with k = floor(T * cutoff_hz / fps) = floor(196 * 1.0 / 20) = 9.

So DCT has roughly 2× the frequency resolution as rFFT in the LOW band.
We pick DCT cutoff k=19 (10 cycles per sequence) to span the same
physical frequency range.

Run:
    python -u scripts/stage_a_generator/build_dct_sanity_check_variants.py \\
        --gt-dir   analyses/2026-05-31_stage1p5_wrist_external_review_work/oracle_dump \\
        --pred-dir analyses/round32_stage1p5_substitute_conds_r33_stage1p5_r33_v1_xattn \\
        --bucket   val \\
        --out-root analyses/2026-05-31_stage1p5_dct_sanity_variants
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.fft import dct, idct

import sys
sys.path.insert(0, str(Path(__file__).parent))
from build_d1_spectral_swap_variants import (   # type: ignore[import]
    C41_DIM, S4_DIM, C41_WRIST_SLICE,
    FPS_DEFAULT, CUTOFF_HZ_DEFAULT,
    _load_npz, _save_npz,
)


DCT_VARIANTS = ("dct0", "dct1", "dct2")


def split_low_high_dct(
    x: np.ndarray,
    fps: float = FPS_DEFAULT, cutoff_hz: float = CUTOFF_HZ_DEFAULT,
) -> tuple[np.ndarray, np.ndarray]:
    """DCT-II based low/high split along time axis.

    Per the discussion above, the DCT cutoff index is
    ``k_max = floor(2 * T * cutoff_hz / fps)``, which matches the
    physical frequency range of the rFFT cutoff.

    Reconstructs via inverse DCT (DCT-III with normalization='ortho').
    Sum of (low, high) == original up to numerical precision.

    Args:
        x: (T, C) float32; time axis is axis=0.

    Returns:
        (x_low, x_high) both (T, C); their sum = x within ~1e-6.
    """
    if x.ndim != 2:
        raise ValueError(f"expects (T, C); got {x.shape}")
    T = x.shape[0]
    X = dct(x.astype(np.float32), axis=0, norm="ortho")   # (T, C)
    k_max = int(np.floor(2 * T * cutoff_hz / fps))
    k_max = min(max(k_max, 0), T - 1)
    X_low = np.zeros_like(X)
    X_high = np.zeros_like(X)
    X_low[:k_max + 1, :] = X[:k_max + 1, :]
    X_high[k_max + 1:, :] = X[k_max + 1:, :]
    x_low = idct(X_low, axis=0, norm="ortho").astype(np.float32)
    x_high = idct(X_high, axis=0, norm="ortho").astype(np.float32)
    return x_low, x_high


def _check_dct_split(x: np.ndarray) -> None:
    xl, xh = split_low_high_dct(x)
    if xl.shape != x.shape or xh.shape != x.shape:
        raise AssertionError(f"DCT split shape mismatch: {x.shape}")
    err = float(np.max(np.abs((xl + xh) - x)))
    if err > 1e-4:
        raise AssertionError(f"DCT split reconstruction err {err:.3e} > 1e-4")


def _build_variant(
    variant: str,
    c41_pred: np.ndarray, s4_pred: np.ndarray,
    c41_gt: np.ndarray, s4_gt: np.ndarray,
    valid_T: int,
    fps: float, cutoff_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    c41 = c41_pred.copy()
    s4 = s4_pred.copy()

    if variant == "dct0":
        return c41, s4

    c41_pred_v = c41_pred[:valid_T, C41_WRIST_SLICE]
    c41_gt_v = c41_gt[:valid_T, C41_WRIST_SLICE]
    pred_low, pred_high = split_low_high_dct(c41_pred_v, fps=fps, cutoff_hz=cutoff_hz)
    gt_low, gt_high = split_low_high_dct(c41_gt_v, fps=fps, cutoff_hz=cutoff_hz)

    if variant == "dct1":
        mixed = gt_low + pred_high
        c41[:valid_T, C41_WRIST_SLICE] = mixed
    elif variant == "dct2":
        mixed = pred_low + gt_high
        c41[:valid_T, C41_WRIST_SLICE] = mixed
    else:
        raise ValueError(f"unknown DCT variant: {variant}")
    return c41, s4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--pred-dir", type=Path, required=True)
    ap.add_argument("--bucket", choices=["train", "val"], default="val")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=FPS_DEFAULT)
    ap.add_argument("--cutoff-hz", type=float, default=CUTOFF_HZ_DEFAULT)
    args = ap.parse_args()

    # Smoke
    rng = np.random.default_rng(0)
    _check_dct_split(rng.standard_normal((196, 6)).astype(np.float32))
    _check_dct_split(rng.standard_normal((100, 6)).astype(np.float32))
    print("[dct] split_low_high_dct round-trip OK")

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
    print(f"[dct] {len(pairs)} clips found under {gt_root}")

    n_per_variant = {v: 0 for v in DCT_VARIANTS}
    for subset, seq_id in pairs:
        gt_p = gt_root / subset / f"{seq_id}.npz"
        pred_p = pred_root / subset / f"{seq_id}.npz"
        if not pred_p.exists():
            print(f"[dct] WARN: pred missing for ({subset}, {seq_id})")
            continue
        c41_gt, s4_gt, vt_gt = _load_npz(gt_p)
        c41_pred, s4_pred, vt_pred = _load_npz(pred_p)
        valid_T = max(vt_gt, vt_pred)

        for variant in DCT_VARIANTS:
            c41_out, s4_out = _build_variant(
                variant,
                c41_pred=c41_pred, s4_pred=s4_pred,
                c41_gt=c41_gt, s4_gt=s4_gt,
                valid_T=valid_T,
                fps=args.fps, cutoff_hz=args.cutoff_hz,
            )
            out_path = args.out_root / variant / args.bucket / subset / f"{seq_id}.npz"
            _save_npz(out_path, c41_out, s4_out, valid_T)
            n_per_variant[variant] += 1

    print("[dct] DONE")
    for v in DCT_VARIANTS:
        print(f"  {v}: {n_per_variant[v]} clips → {args.out_root / v / args.bucket}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
