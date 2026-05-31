"""Round-37 Stage-1.5 C41 dynamics-loss config generator.

Generates 4 ablation configs on the R34 V2-A substrate
(`stage1p5_r34v2_a_lambda0p005`):

    A0  Full proposal              — 6 dynamics terms + jerk + R34 FFT
    A1  FFT ablation               — A0 with r34_wrist_lowband = 0
    A2  Jerk ablation              — A0 with c41_jerk = 0
    A3  Mask ablation              — A0 with per-group mask disabled

Design provenance:
- Stage-2 (PB1) anchordiff trainer uses cm-scale SmoothL1 + per-bodypart
  stable/contact masks + acc weight ≪ vel weight (ship metric 7.55 cm).
- Paper arXiv:2605.26879 (Wei et al. CVPR 2026) confirms λ_A/λ_V = 0.1
  ratio and adds an un-normalized jerk smoothness regularizer.
- R36 was a single-cell single-design failure (statistical
  `normalize_by_gt_std` + uniform mask + acc weight > vel ratio); R37
  re-derives the loss in Stage-2's cm-physical-unit space and uses 4
  cells to isolate FFT vs jerk vs mask design choices.

Layout decisions per cell:
  A0 (main candidate):
    wrist_vel_cm     λ = 0.5
    knee_vel_cm      λ = 0.3
    pelvis_vel_cm    λ = 0.2
    neck_vel_cm      λ = 0.1
    pelvis_acc_cm    λ = 0.02    (10× smaller than pelvis_vel; paper inform)
    c41_jerk_cm      λ = 0.0005  (paper E_jerk inform; small at cm-scale)
    c41_speed_moment_cm λ = 0.02
    r34_wrist_lowband λ = 0.005  (keep R34 V2-A best value)

Expected weighted-vs-mse_c41 ratios (predicted):
    wrist_vel weighted     ~ 0.5  × 0.5  = 0.25
    knee_vel               ~ 0.3  × 0.3  = 0.09
    pelvis_vel             ~ 0.3  × 0.2  = 0.06
    neck_vel               ~ 0.5  × 0.1  = 0.05
    pelvis_acc             ~ 1.5  × 0.02 = 0.03
    jerk                   ~ 0.5  × 0.0005 = 0.00025
    speed_moment           ~ 0.05 × 0.02 = 0.001
    r34_lowband weighted   ~ 1.5  × 1    = 1.5  (already on substrate)
    SUM R37 dynamics       ~ 0.5
    mse_c41 baseline (R34 V2-A late epoch)  ~ 2.0
    → R37 dynamics total ~ 25% of mse_c41 — Stage-2 philosophy holds.

If the smoke-test calibration shows a per-term weighted ratio more than
1× mse_c41, lower that term's weight before launch. The launcher prints
the smoke-test summary so this audit is cheap.

Output:
    configs/training/stage1p5_r37_a0_full.yaml
    configs/training/stage1p5_r37_a1_no_fft.yaml
    configs/training/stage1p5_r37_a2_no_jerk.yaml
    configs/training/stage1p5_r37_a3_no_mask.yaml

Run:
    python -u scripts/stage_a_generator/round37_make_stage1p5_configs.py \\
        --base-cfg configs/training/stage1p5_r34v2_a_lambda0p005.yaml \\
        --out-dir  configs/training/
"""
from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf


# All four cells share these R37 weights unless explicitly overridden.
R37_FULL_WEIGHTS: dict[str, float | int | bool] = {
    "w_r37_wrist_vel_cm": 0.5,
    "w_r37_knee_vel_cm": 0.3,
    "w_r37_pelvis_vel_cm": 0.2,
    "w_r37_neck_vel_cm": 0.1,
    "w_r37_pelvis_acc_cm": 0.02,
    "w_r37_c41_jerk_cm": 0.0005,
    "w_r37_c41_speed_moment_cm": 0.02,
    "r37_smoothl1_beta": 1.0,
    "r37_erode_half_window": 1,
    "r37_use_contact_state": True,
}


# (vid_slug, knob overrides applied on top of R37_FULL_WEIGHTS).
# vid_slug uniquely names the run, the config file, and (via the launcher
# ROUND37_*_TAG env vars) the downstream-diag output directories.
VARIANTS = (
    ("a0_full",     {}),
    ("a1_no_fft",   {"w_r34_wrist_lowband": 0.0}),
    ("a2_no_jerk",  {"w_r37_c41_jerk_cm": 0.0}),
    ("a3_no_mask",  {"r37_use_contact_state": False,
                     "r37_erode_half_window": 0}),
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-cfg", type=Path,
        default=Path("configs/training/stage1p5_r34v2_a_lambda0p005.yaml"),
        help="Base R34 V2-A config to inherit from.",
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
    # Sanity: base substrate must be R33 V1 architecture (per-block obj
    # cross-attn). R34 V2-A inherits this; double-check the inheritance
    # didn't drop it.
    denoiser = base.model.denoiser
    if not bool(denoiser.get("enable_per_block_obj_xattn", False)):
        raise SystemExit(
            "base cfg does not have enable_per_block_obj_xattn=true; "
            "R37 requires the R33 V1 / R34 V2-A substrate."
        )
    # Sanity: R34 wrist low-band knob must already be wired with the
    # R34 V2-A best λ = 0.005 (A1 explicitly turns it off).
    if float(base.loss.get("w_r34_wrist_lowband", 0.0)) != 0.005:
        raise SystemExit(
            "base cfg w_r34_wrist_lowband is not 0.005 — expected the "
            "R34 V2-A substrate. Generate it first with "
            "round34v2_make_stage1p5_configs.py."
        )

    wrote: list[Path] = []
    for vid, overrides in VARIANTS:
        cfg = OmegaConf.create(OmegaConf.to_container(base, resolve=True))
        run_name = f"stage1p5_r37_{vid}"
        cfg.output_dir = f"runs/training/{run_name}"
        cfg.logging.run_name = run_name

        # Apply the full R37 weights first, then per-cell overrides.
        for key, val in R37_FULL_WEIGHTS.items():
            cfg.loss[key] = val
        for key, val in overrides.items():
            cfg.loss[key] = val

        # Explicitly turn off R36 in case the base inherits any non-zero
        # R36 weights — R37 supersedes R36 entirely.
        cfg.loss.w_r36_c41_velocity = 0.0
        cfg.loss.w_r36_c41_acceleration = 0.0

        out_path = args.out_dir / f"{run_name}.yaml"
        OmegaConf.save(cfg, out_path)
        wrote.append(out_path)

        # Friendly summary of what's enabled.
        active = (
            f"wrist={float(cfg.loss.w_r37_wrist_vel_cm):g} "
            f"knee={float(cfg.loss.w_r37_knee_vel_cm):g} "
            f"pelvis_v={float(cfg.loss.w_r37_pelvis_vel_cm):g} "
            f"neck_v={float(cfg.loss.w_r37_neck_vel_cm):g} "
            f"pelvis_a={float(cfg.loss.w_r37_pelvis_acc_cm):g} "
            f"jerk={float(cfg.loss.w_r37_c41_jerk_cm):g} "
            f"moment={float(cfg.loss.w_r37_c41_speed_moment_cm):g} "
            f"r34_lowband={float(cfg.loss.w_r34_wrist_lowband):g} "
            f"use_contact_state={bool(cfg.loss.r37_use_contact_state)} "
            f"erode_half={int(cfg.loss.r37_erode_half_window)}"
        )
        print(f"  wrote {out_path}")
        print(f"    {active}")

    print(f"DONE: {len(wrote)} R37 configs under {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
