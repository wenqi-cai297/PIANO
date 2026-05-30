"""D2 — PB1 C41 residual-sensitivity-curve variant builder.

Reads:
    --gt-dir   : oracle dump from ``dump_gt_c41_s4_oracle.py``
    --pred-dir : Stage-1.5 R33 V1 generated substitute_conds

Writes 6 groups × 5 alpha levels = 30 variant dirs under
``--out-root/{g0..g5}_alpha{010..100}/<bucket>/...``. Each variant is a
substitute_conds dir consumable by the downstream-coupling diag.

Residual definition (binding, from brief §6.2):

    res_c41 = c41_pred - c41_gt
    res_s4  = s4_pred  - s4_gt

Perturbation (in oracle baseline space):

    c41_aug = c41_gt + alpha * residual_component
    s4_aug  = s4_gt  + alpha * residual_component

Groups (brief §6.3):

    G0 full C41 residual:          c41_aug = c41_gt + α * res_c41;  s4 = s4_gt
    G1 wrist C41 residual only:    c41_aug[:,:,0:6]  += α * res_c41[:,:,0:6];  rest of c41 = gt; s4 = s4_gt
    G2 non-wrist C41 residual:     c41_aug[:,:,6:18] += α * res_c41[:,:,6:18]; wrist = gt; s4 = s4_gt
    G3 wrist LOW-band residual:    c41_aug[:,:,0:6]  += α * res_wrist_low (FFT-split res_c41 wrist); s4 = s4_gt
    G4 wrist HIGH-band residual:   c41_aug[:,:,0:6]  += α * res_wrist_high;                          s4 = s4_gt
    G5 S4 residual only:           c41 = c41_gt;  s4_aug = s4_gt + α * res_s4

α ∈ {0.10, 0.25, 0.50, 0.75, 1.00}.

The α=0.0 baseline is *not* emitted here — the existing oracle dump
(``dump_gt_c41_s4_oracle.py`` output) IS the α=0.0 baseline; point the
diag at it directly. The α=1.0 row of G0 (full residual at full
strength) recovers Stage-1.5 R33 V1's pred baseline modulo round-off.

CPU only. ~1-2 min on 48 clips × 30 variants = 1440 npz files.

Run (after dump_gt_c41_s4_oracle.py + the same pred dir as D1):

    python -u scripts/stage_a_generator/build_d2_residual_sensitivity_variants.py \\
        --gt-dir   analyses/2026-05-31_stage1p5_wrist_d1_oracle_dump \\
        --pred-dir analyses/round32_stage1p5_substitute_conds_r33_stage1p5_r33_v1_xattn \\
        --bucket   val \\
        --out-root analyses/2026-05-31_stage1p5_wrist_d2_variants
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Re-use the FFT split + IO helpers from D1.
from build_d1_spectral_swap_variants import (    # type: ignore[import]
    C41_DIM, S4_DIM, C41_WRIST_SLICE, C41_NON_WRIST_SLICE,
    FPS_DEFAULT, CUTOFF_HZ_DEFAULT,
    split_low_high, _check_split, _load_npz, _save_npz,
)

D2_GROUPS = ("g0", "g1", "g2", "g3", "g4", "g5")
D2_ALPHAS = (0.10, 0.25, 0.50, 0.75, 1.00)


def _alpha_tag(a: float) -> str:
    """Stable, sortable file-system tag like 'alpha010' for α=0.10."""
    return f"alpha{int(round(a * 100)):03d}"


def _build_one_group_alpha(
    group: str, alpha: float,
    c41_pred: np.ndarray, s4_pred: np.ndarray,
    c41_gt: np.ndarray, s4_gt: np.ndarray,
    valid_T: int,
    fps: float, cutoff_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per brief §6.3."""
    res_c41 = c41_pred - c41_gt
    res_s4 = s4_pred - s4_gt

    # Defensive: start from oracle baseline.
    c41 = c41_gt.copy()
    s4 = s4_gt.copy()

    if group == "g0":
        c41 = c41_gt + alpha * res_c41
    elif group == "g1":
        c41[:, C41_WRIST_SLICE] = (
            c41_gt[:, C41_WRIST_SLICE]
            + alpha * res_c41[:, C41_WRIST_SLICE]
        )
    elif group == "g2":
        c41[:, C41_NON_WRIST_SLICE] = (
            c41_gt[:, C41_NON_WRIST_SLICE]
            + alpha * res_c41[:, C41_NON_WRIST_SLICE]
        )
    elif group == "g3":
        # wrist LOW-band residual only.
        res_w = res_c41[:valid_T, C41_WRIST_SLICE]
        res_w_low, _res_w_high = split_low_high(res_w, fps=fps, cutoff_hz=cutoff_hz)
        c41[:valid_T, C41_WRIST_SLICE] = (
            c41_gt[:valid_T, C41_WRIST_SLICE] + alpha * res_w_low
        )
    elif group == "g4":
        # wrist HIGH-band residual only.
        res_w = res_c41[:valid_T, C41_WRIST_SLICE]
        _res_w_low, res_w_high = split_low_high(res_w, fps=fps, cutoff_hz=cutoff_hz)
        c41[:valid_T, C41_WRIST_SLICE] = (
            c41_gt[:valid_T, C41_WRIST_SLICE] + alpha * res_w_high
        )
    elif group == "g5":
        # S4 residual only — note the brief uses res_s4 in this group.
        s4 = s4_gt + alpha * res_s4
    else:
        raise ValueError(f"unknown D2 group: {group}")
    return c41.astype(np.float32), s4.astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--pred-dir", type=Path, required=True)
    ap.add_argument("--bucket", choices=["train", "val"], default="val")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=FPS_DEFAULT)
    ap.add_argument("--cutoff-hz", type=float, default=CUTOFF_HZ_DEFAULT)
    ap.add_argument(
        "--groups", default=",".join(D2_GROUPS),
        help="comma-separated subset of {g0..g5}",
    )
    ap.add_argument(
        "--alphas", default=",".join(f"{a:.2f}" for a in D2_ALPHAS),
        help="comma-separated floats; default 0.10,0.25,0.50,0.75,1.00",
    )
    args = ap.parse_args()

    groups_selected = tuple(g.strip() for g in args.groups.split(",") if g.strip())
    for g in groups_selected:
        if g not in D2_GROUPS:
            raise SystemExit(f"--groups contains unknown {g!r}; allowed {D2_GROUPS}")
    alphas_selected = tuple(float(s.strip()) for s in args.alphas.split(",") if s.strip())
    for a in alphas_selected:
        if not (0.0 < a <= 1.0):
            raise SystemExit(f"--alphas value {a} outside (0, 1]")
    print(f"[d2] groups = {groups_selected}; alphas = {alphas_selected}")
    print(f"[d2] cutoff_hz = {args.cutoff_hz}; fps = {args.fps}")

    # Smoke test the FFT helper (re-used) before iterating.
    rng = np.random.default_rng(0)
    _check_split(rng.standard_normal((196, 6)).astype(np.float32))
    _check_split(rng.standard_normal((100, 6)).astype(np.float32))
    print("[d2] split_low_high round-trip sanity OK")

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
        raise SystemExit(f"[d2] no .npz under {gt_root}")
    print(f"[d2] {len(pairs)} clips found")

    n_per_variant: dict[str, int] = {}
    n_missing = 0
    for subset, seq_id in pairs:
        pred_p = pred_root / subset / f"{seq_id}.npz"
        if not pred_p.exists():
            n_missing += 1
            print(f"[d2] WARN: pred missing for ({subset}, {seq_id}); skipping")
            continue
        c41_gt, s4_gt, vt_gt = _load_npz(gt_root / subset / f"{seq_id}.npz")
        c41_pred, s4_pred, vt_pred = _load_npz(pred_p)
        if c41_gt.shape != c41_pred.shape or s4_gt.shape != s4_pred.shape:
            raise RuntimeError(
                f"[d2] ({subset}, {seq_id}) shape mismatch between gt and pred"
            )
        valid_T = max(vt_gt, vt_pred)

        for group in groups_selected:
            for alpha in alphas_selected:
                c41_out, s4_out = _build_one_group_alpha(
                    group, alpha,
                    c41_pred=c41_pred, s4_pred=s4_pred,
                    c41_gt=c41_gt, s4_gt=s4_gt,
                    valid_T=valid_T,
                    fps=args.fps, cutoff_hz=args.cutoff_hz,
                )
                tag = f"{group}_{_alpha_tag(alpha)}"
                out_path = args.out_root / tag / args.bucket / subset / f"{seq_id}.npz"
                _save_npz(out_path, c41_out, s4_out, valid_T)
                n_per_variant[tag] = n_per_variant.get(tag, 0) + 1

    print("[d2] DONE")
    for tag in sorted(n_per_variant):
        print(f"  {tag}: {n_per_variant[tag]} clips → {args.out_root / tag / args.bucket}")
    if n_missing:
        print(f"[d2] WARN: {n_missing} clips missing in --pred-dir")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
