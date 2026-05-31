"""Round-38 Stage-1.5 condition + contact-supervision ablation configs.

Background — R37 failure post-mortem:
- R37's 4-cell C41 dynamics ablation all came in at 86-96 cm drift_max
  (vs R34 V2-A's 13.86 cm), despite R37 driving val_mse_c41 LOWER than
  R34 V2-A.
- Root cause: C41 is pelvis-local current-yaw delta xyz. Frame-level
  dynamics supervision on C41 entangles wrist motion with pelvis-frame
  rotation, pushing c41_pred into a sub-space PB1 cannot consume.

R38 changes the supervision *direction*: instead of adding more loss
terms on the existing cond set, we add the only feasible new cond
(init_pose, available at inference) and a value-domain contact-aware
loss that mirrors PB1's anchor pattern.

Feasibility audit of candidate condition additions:
  - init_pose (frame-0 motion[:, 0, :], 135-D): ✓
        Stage-1 V8 V6 + PB1 both use it; sample_substitute_conds
        and diagnostic_helpers already populate it from batch motion.
  - contact_target_xyz (object-local surface contact point): ✗
        Derived from GT motion via FK + closest-mesh-point. Not
        accessible at end-to-end inference (no GT motion). PB1 ship
        cfg uses object_traj_dim=9 → zeros this out anyway.
  - contact_state (T, 5 per-frame label): ✗ as cond.
        Pseudo-label from GT motion. PB1 uses it as LOSS SUPERVISION
        only, never as a model cond. R38 reuses this pattern.
  - Explicit pelvis_rot6d (already in stage1_coarse [9:15]): ⓘ
        Already provided. Independent exposure would be re-arrangement,
        not new information.

R38 4-cell matrix on the R34 V2-A substrate (per-block obj_xattn,
r34_wrist_lowband λ=0.005, σ_cond_aug=0):

    B0 baseline       — R34 V2-A configuration, sanity check.
                        Expected drift_max ≈ 13.86 cm.
    B1 + init_pose    — Adds init_pose (135-D F1) via zero-init Linear.
                        Tests: frame-0 wrist offset contribution.
    B2 + contact_wrist — Adds contact-window weighted wrist value MSE
                        (PB1 anchor pattern, value-domain not dynamics).
                        Independent of B1 — same substrate as B0.
    B3 = B1 + B2      — Both. Tests additivity.

Weight choice for w_r38_contact_wrist: 0.5.
  - C41 wrist MSE inside the contact window should be comparable in
    magnitude to the existing mse_c41 (which is sum-over-channels then
    mean-over-frames). Contact-window MSE has a smaller denominator
    (only hand-contact frames) so raw value is higher; 0.5 weight
    targets weighted/mse_c41 ratio ≈ 0.3-0.8× in early training.
  - This is conservative vs PB1's anchor_joint_vel_weight=2.0; that
    weight is on world-frame joint velocity which has different
    scale. We err small to avoid R37-style scale-dominate disasters
    and let the smoke-test calibration audit guide a future sweep.

Output:
    configs/training/stage1p5_r38_b0_baseline.yaml
    configs/training/stage1p5_r38_b1_init_pose.yaml
    configs/training/stage1p5_r38_b2_contact_wrist.yaml
    configs/training/stage1p5_r38_b3_full.yaml

Run:
    python -u scripts/stage_a_generator/round38_make_stage1p5_configs.py \\
        --base-cfg configs/training/stage1p5_r34v2_a_lambda0p005.yaml \\
        --out-dir  configs/training/
"""
from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf


# Shared knobs across all 4 cells (default values that map back to R34 V2-A).
R38_DEFAULTS: dict[str, float | int | bool] = {
    # init_pose disabled by default; B1/B3 turn it on.
    "init_pose_dim": 0,
    # contact-wrist loss disabled by default; B2/B3 turn it on.
    "w_r38_contact_wrist": 0.0,
    "r38_contact_threshold": 0.5,
    "r38_contact_erode_half": 1,
}


# (vid, knob overrides applied on top of R38_DEFAULTS).
# "init_pose_dim" lives under cfg.model.denoiser; "w_r38_contact_wrist"
# and friends live under cfg.loss. The applier routes by key.
VARIANTS = (
    ("b0_baseline",      {}),
    ("b1_init_pose",     {"init_pose_dim": 135}),
    ("b2_contact_wrist", {"w_r38_contact_wrist": 0.5}),
    ("b3_full",          {"init_pose_dim": 135,
                          "w_r38_contact_wrist": 0.5}),
)


# Routing table: which sub-section in the cfg each knob lives in.
_KEYS_UNDER_DENOISER = {"init_pose_dim"}
# Everything else lives under cfg.loss.


def _apply_override(cfg, key: str, value) -> None:
    if key in _KEYS_UNDER_DENOISER:
        cfg.model.denoiser[key] = value
    else:
        cfg.loss[key] = value


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
            "R38 requires the R33 V1 / R34 V2-A substrate."
        )
    if float(base.loss.get("w_r34_wrist_lowband", 0.0)) != 0.005:
        raise SystemExit(
            "base cfg w_r34_wrist_lowband is not 0.005 — expected the "
            "R34 V2-A substrate. Generate it first with "
            "round34v2_make_stage1p5_configs.py."
        )

    wrote: list[Path] = []
    for vid, overrides in VARIANTS:
        cfg = OmegaConf.create(OmegaConf.to_container(base, resolve=True))
        run_name = f"stage1p5_r38_{vid}"
        cfg.output_dir = f"runs/training/{run_name}"
        cfg.logging.run_name = run_name

        # Apply R38 defaults first, then per-cell overrides.
        for key, val in R38_DEFAULTS.items():
            _apply_override(cfg, key, val)
        for key, val in overrides.items():
            _apply_override(cfg, key, val)

        # Explicitly turn off R36 / R37 in case the base inherits any
        # non-zero weights. R38 supersedes both prior rounds.
        cfg.loss.w_r36_c41_velocity = 0.0
        cfg.loss.w_r36_c41_acceleration = 0.0
        cfg.loss.w_r37_wrist_vel_cm = 0.0
        cfg.loss.w_r37_knee_vel_cm = 0.0
        cfg.loss.w_r37_pelvis_vel_cm = 0.0
        cfg.loss.w_r37_neck_vel_cm = 0.0
        cfg.loss.w_r37_pelvis_acc_cm = 0.0
        cfg.loss.w_r37_c41_jerk_cm = 0.0
        cfg.loss.w_r37_c41_speed_moment_cm = 0.0

        out_path = args.out_dir / f"{run_name}.yaml"
        OmegaConf.save(cfg, out_path)
        wrote.append(out_path)

        # Friendly summary of what's enabled.
        active = (
            f"init_pose_dim={int(cfg.model.denoiser.init_pose_dim):d} "
            f"w_r38_contact_wrist={float(cfg.loss.w_r38_contact_wrist):g} "
            f"contact_thr={float(cfg.loss.r38_contact_threshold):g} "
            f"erode_half={int(cfg.loss.r38_contact_erode_half):d} "
            f"r34_lowband={float(cfg.loss.w_r34_wrist_lowband):g}"
        )
        print(f"  wrote {out_path}")
        print(f"    {active}")

    print(f"DONE: {len(wrote)} R38 configs under {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
