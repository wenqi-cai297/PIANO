"""Round-34 V2 Stage-1.5 wrist low-band loss — λ-sweep config generator.

Per the R34 V1 failure-mode analysis (R34 V1 used λ=1.0 producing
weighted_lowband / mse_c41 ratio = 50-110× across training; gradient
dominated, S4 collapsed, drift_max 67-69 cm). This sweep brackets the
"loss-scale-balanced" regime around an order-of-magnitude calibration:

  Architectural ratio explanation (reduction-denominator analysis):
    mse_c41        denom = B * T_valid * 18  ≈ 130k elements
    lowband_loss   denom = B * n_low * 6     ≈   2880 elements (45× smaller)
  → even with equal per-element MSE, raw lowband is ~45× larger than mse_c41.

Calibration on R32 V0 audit dump (48 val clips, B=48 single-batch):
  base mse_c41 at pred=0       : 1.44 (random init regime)
  raw lowband at pred=0        : 50.6
  base mse_c41 at pred=R32 V0  : 0.20 (a proxy for well-trained R33 V1 substrate)
  raw lowband at pred=R32 V0   : 46.1 (NOT 13 — R32 V0 pred is closer to gt
                                       than R34 V1 C2's lambda-dominated pred)

R34 V1 C2 actual late-epoch numbers:
  step 5000: mse_c41 = 0.124, raw lowband = 13.15, ratio = 106× at λ=1.0
  → R34 V1 C2 lowband was self-suppressed (13 vs 46) because λ=1.0 dominated
    gradient and pushed pred toward small-amplitude wrist signal. So the
    "true" final-epoch ratio when λ is calibrated will fall between
    the R32 V0 calibration (raw_lb=46) and R34 V1 C2 calibration
    (raw_lb=13). Use both as bounds.

Late-epoch predicted (weighted / mse_c41) ratio bounds per λ
(lower bound: R34 V1 C2 self-suppression; upper bound: R32 V0 calibration):

| variant | λ     | early step ratio | late ratio (lower–upper) |
|---------|------:|-----------------:|--------------------------:|
| V2-A    | 0.005 | ~0.2×            | 0.5× – 1.2×               |
| V2-B    | 0.02  | ~0.7×            | 2×   – 5×                 |
| V2-C    | 0.05  | ~1.8×            | 5×   – 12×                |
| V2-D    | 0.1   | ~3.5×            | 10×  – 23×                |

Target ratio (late, weighted ≈ single base-loss term magnitude): 1× – 5×
  → V2-B (λ=0.02) is the lowest-risk candidate
  → V2-C (λ=0.05) is high-target if lowband self-suppresses
  → V2-A (λ=0.005) is the "too-weak" boundary (under-fits if doesn't self-suppress)
  → V2-D (λ=0.1) is the "too-strong" boundary (risks soft-dominate)

The four-variant log-spaced sweep brackets a 20× range over 1.3 decades
to localize the sweet spot AND map the failure boundaries.

Each variant inherits R33 V1 substrate (per-block obj_xattn).
σ_cond_aug = 0 across V2 — the cond-aug axis is disentangled here.
cutoff_hz = 1.0 (same as V1; ChatGPT §9.1 sweep over cutoff is Phase 3,
conditional on positive Phase 1 result).

Output:
    configs/training/stage1p5_r34v2_a_lambda0p005.yaml
    configs/training/stage1p5_r34v2_b_lambda0p02.yaml
    configs/training/stage1p5_r34v2_c_lambda0p05.yaml
    configs/training/stage1p5_r34v2_d_lambda0p1.yaml

Run:
    python -u scripts/stage_a_generator/round34v2_make_stage1p5_configs.py \\
        --base-cfg configs/training/stage1p5_r33_v1_xattn.yaml \\
        --out-dir  configs/training/
"""
from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf


# (vid, λ_lowband)
# σ_cond_aug = 0 for all; cutoff_hz = 1.0 for all.
# vid encodes λ to 3-sig-fig precision in filename-safe form.
VARIANTS = (
    {"vid": "a_lambda0p005", "wrist_lowband_weight": 0.005},
    {"vid": "b_lambda0p02",  "wrist_lowband_weight": 0.02},
    {"vid": "c_lambda0p05",  "wrist_lowband_weight": 0.05},
    {"vid": "d_lambda0p1",   "wrist_lowband_weight": 0.1},
)

SHARED_KNOBS = {
    "wrist_lowband_cutoff_hz": 1.0,
    "wrist_lowband_fps": 20.0,
    "cond_aug_sigma_max": 0.0,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-cfg", type=Path,
        default=Path("configs/training/stage1p5_r33_v1_xattn.yaml"),
    )
    ap.add_argument(
        "--out-dir", type=Path,
        default=Path("configs/training/"),
    )
    args = ap.parse_args()
    if not args.base_cfg.exists():
        raise SystemExit(f"base cfg missing: {args.base_cfg}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    base = OmegaConf.load(args.base_cfg)
    # Sanity: enable_per_block_obj_xattn must be true (R33 V1 substrate).
    if not bool(base.model.denoiser.get("enable_per_block_obj_xattn", False)):
        raise SystemExit(
            "base cfg does not have enable_per_block_obj_xattn=true; "
            "R34 V2 requires the R33 V1 substrate."
        )

    for v in VARIANTS:
        cfg = OmegaConf.create(OmegaConf.to_container(base, resolve=True))
        run_name = f"stage1p5_r34v2_{v['vid']}"
        cfg.output_dir = f"runs/training/{run_name}"
        cfg.logging.run_name = run_name

        # R34 loss knobs (default to 0 in trainer; explicit here for audit trail).
        # Matches the cfg key names the trainer reads at train_stage1p5.py:507-525.
        cfg.loss.w_r34_wrist_lowband = float(v["wrist_lowband_weight"])
        cfg.loss.r34_wrist_lowband_cutoff_hz = float(SHARED_KNOBS["wrist_lowband_cutoff_hz"])
        cfg.loss.r34_wrist_lowband_fps = float(SHARED_KNOBS["wrist_lowband_fps"])
        cfg.loss.r34_cond_aug_sigma_max = float(SHARED_KNOBS["cond_aug_sigma_max"])

        out_path = args.out_dir / f"{run_name}.yaml"
        OmegaConf.save(cfg, out_path)
        print(f"  wrote {out_path}  (λ={v['wrist_lowband_weight']})")

    print(f"DONE: {len(VARIANTS)} R34 V2 configs under {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
