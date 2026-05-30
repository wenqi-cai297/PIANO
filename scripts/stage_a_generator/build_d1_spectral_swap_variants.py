"""D1 — C41 wrist spectral-swap causal-test variant builder.

Reads:
    --gt-dir   : oracle dump from ``dump_gt_c41_s4_oracle.py``
                  (<bucket>/<subset>/<seq_id>.npz with C41 18-D + S4 13-D)
    --pred-dir : Stage-1.5 R33 V1 generated substitute_conds
                  (same schema, from the R33 sweep)

Writes 7 variant directories under ``--out-root/{v0..v7}/<bucket>/...``
each populated by mixing (c41_pred, s4_pred) with (c41_gt, s4_gt) per the
brief's D1 rules. Each variant is a drop-in substitute_conds dir for
``--substitute-conds-dir`` in the downstream-coupling diag scripts.

Frequency split is FFT-based along the time axis at cutoff_hz=1.0,
fps=20.0. The split helper is bit-identical to the snippet in the
brief (§5.3) and is unit-tested below (irfft round-trip on random
tensors).

Variants (from brief §5.4):

    D1-V0  base generated                       c41 = c41_pred ;             s4 = s4_pred
    D1-V1  full wrist oracle                    c41[:,:,0:6] = c41_gt[0:6];  s4 = s4_pred
    D1-V2  wrist LOW-band oracle                c41[:,:,0:6] = gt_low + pred_high  (wrist only)
    D1-V3  wrist HIGH-band oracle               c41[:,:,0:6] = pred_low + gt_high
    D1-V4  non-wrist C41 oracle                 c41[:,:,6:18] = c41_gt[6:18]; wrist pred
    D1-V5  S4 oracle                            c41 = c41_pred             ; s4 = s4_gt
    D1-V6  full oracle C41, generated S4        c41 = c41_gt               ; s4 = s4_pred
    D1-V7  generated C41, full oracle S4        (alias of V5; included for clarity)

For the per-clip operation we operate on the VALID-LENGTH prefix only
(c41[:valid_T], s4[:valid_T]) so the FFT split is over the meaningful
window. Padding zeros outside valid_T are restored from the gt/pred
copy unchanged.

CPU only. ~30-60 s on 48 clips.

Run (after dump_gt_c41_s4_oracle.py has populated --gt-dir):

    python -u scripts/stage_a_generator/build_d1_spectral_swap_variants.py \\
        --gt-dir   analyses/2026-05-31_stage1p5_wrist_d1_oracle_dump \\
        --pred-dir analyses/round32_stage1p5_substitute_conds_r33_stage1p5_r33_v1_xattn \\
        --bucket   val \\
        --out-root analyses/2026-05-31_stage1p5_wrist_d1_variants
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# ─── Channel constants (binding, from src/piano/data/stage2_oracle_conditions.py) ─
C41_DIM = 18
S4_DIM = 13
C41_WRIST_SLICE = slice(0, 6)        # left_wrist (3) + right_wrist (3)
C41_NON_WRIST_SLICE = slice(6, 18)   # knee L+R + neck + pelvis_delta

# D1 variant codes
D1_VARIANTS = ("v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7")

# Frequency split
FPS_DEFAULT = 20.0
CUTOFF_HZ_DEFAULT = 1.0
IRFFT_TOL = 1e-4   # generous tol for float32 round-trip


def split_low_high(
    x: np.ndarray, fps: float = FPS_DEFAULT, cutoff_hz: float = CUTOFF_HZ_DEFAULT,
) -> tuple[np.ndarray, np.ndarray]:
    """Bit-identical to brief §5.3 snippet, NumPy version.

    Args:
        x: (T, C) float32; time axis is axis=0.

    Returns:
        (x_low, x_high) both (T, C); their sum = x within IRFFT_TOL.
    """
    if x.ndim != 2:
        raise ValueError(f"split_low_high expects (T, C); got {x.shape}")
    T = x.shape[0]
    freqs = np.fft.rfftfreq(T, d=1.0 / fps)
    low_mask = freqs <= cutoff_hz                  # (n_freqs,)
    X = np.fft.rfft(x.astype(np.float32), axis=0)  # (n_freqs, C) complex
    X_low = np.zeros_like(X)
    X_high = np.zeros_like(X)
    X_low[low_mask, :] = X[low_mask, :]
    X_high[~low_mask, :] = X[~low_mask, :]
    x_low = np.fft.irfft(X_low, n=T, axis=0).astype(np.float32)
    x_high = np.fft.irfft(X_high, n=T, axis=0).astype(np.float32)
    return x_low, x_high


def _check_split(x: np.ndarray) -> None:
    """Round-trip sanity check on one tensor."""
    xl, xh = split_low_high(x)
    if xl.shape != x.shape or xh.shape != x.shape:
        raise AssertionError(f"split shape mismatch: {x.shape} vs {xl.shape} / {xh.shape}")
    rec = xl + xh
    err = float(np.max(np.abs(rec - x)))
    if err > IRFFT_TOL:
        raise AssertionError(
            f"low+high reconstruction error {err:.3e} > tol {IRFFT_TOL}"
        )


def _load_npz(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Returns (c41 (T_pad, 18), s4 (T_pad, 13), valid_T)."""
    data = np.load(path)
    for k in ("stage2_coarse_extra", "stage2_support"):
        if k not in data.files:
            raise KeyError(f"{path}: missing {k} (got {list(data.files)})")
    c41 = data["stage2_coarse_extra"].astype(np.float32)
    s4 = data["stage2_support"].astype(np.float32)
    if c41.shape[-1] != C41_DIM or s4.shape[-1] != S4_DIM:
        raise RuntimeError(f"{path}: bad dims c41 {c41.shape} s4 {s4.shape}")
    if c41.shape[0] != s4.shape[0]:
        raise RuntimeError(f"{path}: T mismatch c41 {c41.shape} vs s4 {s4.shape}")
    valid_T = int(data["valid_T"]) if "valid_T" in data.files else c41.shape[0]
    return c41, s4, valid_T


def _save_npz(path: Path, c41: np.ndarray, s4: np.ndarray, valid_T: int) -> None:
    if c41.shape[-1] != C41_DIM or s4.shape[-1] != S4_DIM:
        raise AssertionError(f"_save_npz bad shapes: c41 {c41.shape} s4 {s4.shape}")
    if c41.shape[0] != s4.shape[0]:
        raise AssertionError(f"_save_npz T mismatch: c41 {c41.shape} s4 {s4.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        stage2_coarse_extra=c41.astype(np.float32),
        stage2_support=s4.astype(np.float32),
        valid_T=np.int32(valid_T),
        seed=np.int32(0),
    )


def _build_one_variant(
    variant: str,
    c41_pred: np.ndarray, s4_pred: np.ndarray,
    c41_gt: np.ndarray, s4_gt: np.ndarray,
    valid_T: int,
    fps: float, cutoff_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per the brief §5.4 variant rules.

    Operates on the valid_T prefix for the FFT split; pad region stays
    as the corresponding pred copy (D1 variants are about the meaningful
    window only).
    """
    # Defensive copies — never mutate the inputs.
    c41 = c41_pred.copy()
    s4 = s4_pred.copy()

    if variant == "v0":
        pass  # base generated; identity
    elif variant == "v1":
        c41[:, C41_WRIST_SLICE] = c41_gt[:, C41_WRIST_SLICE]
    elif variant == "v2":
        # wrist LOW-band oracle.
        c41_pred_v = c41_pred[:valid_T, C41_WRIST_SLICE]
        c41_gt_v = c41_gt[:valid_T, C41_WRIST_SLICE]
        pred_low, pred_high = split_low_high(c41_pred_v, fps=fps, cutoff_hz=cutoff_hz)
        gt_low, gt_high = split_low_high(c41_gt_v, fps=fps, cutoff_hz=cutoff_hz)
        mixed = gt_low + pred_high
        c41[:valid_T, C41_WRIST_SLICE] = mixed
    elif variant == "v3":
        # wrist HIGH-band oracle.
        c41_pred_v = c41_pred[:valid_T, C41_WRIST_SLICE]
        c41_gt_v = c41_gt[:valid_T, C41_WRIST_SLICE]
        pred_low, pred_high = split_low_high(c41_pred_v, fps=fps, cutoff_hz=cutoff_hz)
        gt_low, gt_high = split_low_high(c41_gt_v, fps=fps, cutoff_hz=cutoff_hz)
        mixed = pred_low + gt_high
        c41[:valid_T, C41_WRIST_SLICE] = mixed
    elif variant == "v4":
        c41[:, C41_NON_WRIST_SLICE] = c41_gt[:, C41_NON_WRIST_SLICE]
    elif variant == "v5":
        s4 = s4_gt.copy()
    elif variant == "v6":
        c41 = c41_gt.copy()
    elif variant == "v7":
        # alias of v5 — generated C41, oracle S4
        s4 = s4_gt.copy()
    else:
        raise ValueError(f"unknown D1 variant: {variant}")
    return c41, s4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", type=Path, required=True,
                    help="root containing <bucket>/<subset>/<seq_id>.npz with GT C41/S4.")
    ap.add_argument("--pred-dir", type=Path, required=True,
                    help="root containing <bucket>/<subset>/<seq_id>.npz with Stage-1.5 R33 V1 pred.")
    ap.add_argument("--bucket", choices=["train", "val"], default="val")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=FPS_DEFAULT)
    ap.add_argument("--cutoff-hz", type=float, default=CUTOFF_HZ_DEFAULT)
    ap.add_argument(
        "--variants", default=",".join(D1_VARIANTS),
        help="comma-separated subset of {v0..v7}",
    )
    args = ap.parse_args()

    selected = tuple(v.strip() for v in args.variants.split(",") if v.strip())
    for v in selected:
        if v not in D1_VARIANTS:
            raise SystemExit(f"--variants contains unknown {v!r}; allowed {D1_VARIANTS}")
    print(f"[d1] variants = {selected}; cutoff_hz = {args.cutoff_hz}; fps = {args.fps}")

    # Smoke test the FFT helper before iterating.
    rng = np.random.default_rng(0)
    smoke = rng.standard_normal((196, 6)).astype(np.float32)
    _check_split(smoke)
    _check_split(rng.standard_normal((100, 6)).astype(np.float32))
    print("[d1] split_low_high round-trip sanity OK")

    gt_root = args.gt_dir / args.bucket
    pred_root = args.pred_dir / args.bucket
    if not gt_root.is_dir():
        raise SystemExit(f"--gt-dir missing bucket subdir: {gt_root}")
    if not pred_root.is_dir():
        raise SystemExit(f"--pred-dir missing bucket subdir: {pred_root}")

    # Enumerate (subset, seq_id) from the GT dir.
    pairs: list[tuple[str, str]] = []
    for subset_d in sorted(p for p in gt_root.iterdir() if p.is_dir()):
        for npz in sorted(subset_d.glob("*.npz")):
            pairs.append((subset_d.name, npz.stem))
    if not pairs:
        raise SystemExit(f"[d1] no .npz under {gt_root}")
    print(f"[d1] {len(pairs)} clips found under {gt_root}")

    n_per_variant: dict[str, int] = {v: 0 for v in selected}
    n_missing = 0
    for subset, seq_id in pairs:
        gt_p = gt_root / subset / f"{seq_id}.npz"
        pred_p = pred_root / subset / f"{seq_id}.npz"
        if not pred_p.exists():
            n_missing += 1
            print(f"[d1] WARN: pred missing for ({subset}, {seq_id}); skipping")
            continue

        c41_gt, s4_gt, vt_gt = _load_npz(gt_p)
        c41_pred, s4_pred, vt_pred = _load_npz(pred_p)
        if c41_gt.shape != c41_pred.shape:
            raise RuntimeError(
                f"[d1] ({subset}, {seq_id}) C41 shape mismatch: "
                f"gt {c41_gt.shape} vs pred {c41_pred.shape}"
            )
        if s4_gt.shape != s4_pred.shape:
            raise RuntimeError(
                f"[d1] ({subset}, {seq_id}) S4 shape mismatch: "
                f"gt {s4_gt.shape} vs pred {s4_pred.shape}"
            )
        valid_T = max(vt_gt, vt_pred)  # both should be equal; max is defensive

        for variant in selected:
            c41_out, s4_out = _build_one_variant(
                variant,
                c41_pred=c41_pred, s4_pred=s4_pred,
                c41_gt=c41_gt, s4_gt=s4_gt,
                valid_T=valid_T,
                fps=args.fps, cutoff_hz=args.cutoff_hz,
            )
            out_path = args.out_root / variant / args.bucket / subset / f"{seq_id}.npz"
            _save_npz(out_path, c41_out, s4_out, valid_T)
            n_per_variant[variant] += 1

    print("[d1] DONE")
    for v in selected:
        print(f"  {v}: {n_per_variant[v]} clips → {args.out_root / v / args.bucket}")
    if n_missing:
        print(f"[d1] WARN: {n_missing} clips missing in --pred-dir")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
