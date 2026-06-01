"""Round-40 Stage-1 plan-sampler ablation configs.

R40 turns Stage-1 from a "23-D GT regression target" into a "coarse-plan
sampler" by (i) downweighting exact-GT MSE on under-determined channels
(root, vel, yaw, pelvis_rot6d) via per-channel weight lists, and
(ii) adding a plan-invariant loss that supervises plan-level invariants
(speed envelope, arc length, turn activity, root-object radial profile,
height envelope, smoothness).

Background — see ``analyses/2026-06-01_stage1_underdetermination_for_codex.md``
and ``analyses/2026-06-01_round40_stage1_plan_sampler_handoff_for_claude.md``.

R40 4-cell matrix on the V8 V6 substrate (anti-collapse stack inherited):

  C0 baseline           — exact V8 V6 config (run-name/output rewritten).
                          Reproduces current generated stage1_coarse.
  C1 weak GT            — keeps x0/vel MSE, downweights ambiguous channels.
                          No plan loss (sanity baseline for the channel
                          weighting alone).
  C2 plan energy        — C1 weights + plan-invariant loss at 0.20.
                          Ship candidate.
  C3 strong plan energy — stronger channel downweighting + plan loss at 0.50.
                          Upper-end probe of plan-energy direction.

Output:
    configs/training/stage1_r40_c0_v8v6_baseline.yaml
    configs/training/stage1_r40_c1_weak_gt.yaml
    configs/training/stage1_r40_c2_plan_energy.yaml
    configs/training/stage1_r40_c3_plan_energy_strong.yaml

Run:
    python -u scripts/stage_a_generator/round40_make_stage1_plan_configs.py \\
        --base-cfg configs/training/stage1_v8_v6_full_f1.yaml \\
        --out-dir  configs/training/
"""
from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf


# Channel order reminder (stage1_coarse_oracle.py):
# 0   root_x          (root_local x — moves)
# 1   root_z          (root_local z — moves)
# 2   root_y          (root_local y — height)
# 3   vel_x           (vel x)
# 4   vel_z           (vel z)
# 5   vel_y           (vel y)
# 6   yaw_sin
# 7   yaw_cos
# 8   yaw_vel
# 9-14  pelvis_rot6d (6)
# 15-20 spine3_rot6d (6)
# 21  head_height
# 22  shoulder_height


# C1 — weak GT: downweight the under-determined channels (root x/z, vel x/z,
# yaw block, pelvis rot6d) so plan-level invariants drive the path.
# Spine3 + heights stay close to GT (less mode-multiplicity).
C1_X0_WEIGHTS: tuple[float, ...] = (
    0.20, 0.20, 1.00,
    0.05, 0.05, 0.50,
    0.10, 0.10, 0.10,
    0.35, 0.35, 0.35, 0.35, 0.35, 0.35,
    0.50, 0.50, 0.50, 0.50, 0.50, 0.50,
    1.00, 1.00,
)
C1_VEL_WEIGHTS: tuple[float, ...] = (
    0.10, 0.10, 0.50,
    0.05, 0.05, 0.25,
    0.10, 0.10, 0.10,
    0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
    0.40, 0.40, 0.40, 0.40, 0.40, 0.40,
    0.50, 0.50,
)

# C3 — strong plan energy: further downweight the ambiguous channels.
C3_X0_WEIGHTS: tuple[float, ...] = (
    0.05, 0.05, 1.00,
    0.02, 0.02, 0.50,
    0.05, 0.05, 0.05,
    0.20, 0.20, 0.20, 0.20, 0.20, 0.20,
    0.35, 0.35, 0.35, 0.35, 0.35, 0.35,
    1.00, 1.00,
)
C3_VEL_WEIGHTS: tuple[float, ...] = (
    0.05, 0.05, 0.50,
    0.02, 0.02, 0.25,
    0.05, 0.05, 0.05,
    0.15, 0.15, 0.15, 0.15, 0.15, 0.15,
    0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
    0.50, 0.50,
)


# Per-cell loss-section overrides. Anything missing means "inherit from base".
VARIANTS = (
    ("c0_v8v6_baseline", {
        # No new R40 knobs — exact V8 V6 behavior.
    }),
    ("c1_weak_gt", {
        "x0_channel_weights": list(C1_X0_WEIGHTS),
        "vel_channel_weights": list(C1_VEL_WEIGHTS),
        "w_r40_plan_invariant": 0.0,
    }),
    ("c2_plan_energy", {
        "x0_channel_weights": list(C1_X0_WEIGHTS),
        "vel_channel_weights": list(C1_VEL_WEIGHTS),
        "w_r40_plan_invariant": 0.20,
        # default component weights — leave unset to use library defaults.
    }),
    ("c3_plan_energy_strong", {
        "x0_channel_weights": list(C3_X0_WEIGHTS),
        "vel_channel_weights": list(C3_VEL_WEIGHTS),
        "w_r40_plan_invariant": 0.50,
    }),
)


def _validate_lengths():
    for arr, name in (
        (C1_X0_WEIGHTS, "C1_X0_WEIGHTS"),
        (C1_VEL_WEIGHTS, "C1_VEL_WEIGHTS"),
        (C3_X0_WEIGHTS, "C3_X0_WEIGHTS"),
        (C3_VEL_WEIGHTS, "C3_VEL_WEIGHTS"),
    ):
        if len(arr) != 23:
            raise SystemExit(
                f"{name} must have exactly 23 entries; got {len(arr)}"
            )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-cfg", type=Path,
        default=Path("configs/training/stage1_v8_v6_full_f1.yaml"),
        help="Base Stage-1 config to inherit from (default V8 V6).",
    )
    ap.add_argument(
        "--out-dir", type=Path,
        default=Path("configs/training/"),
    )
    args = ap.parse_args()
    if not args.base_cfg.exists():
        raise SystemExit(
            f"base cfg missing: {args.base_cfg}\n"
            "Regenerate Round 31 V8 configs first via:\n"
            "  python scripts/stage_a_generator/"
            "round31_make_stage1_v8_configs.py"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    _validate_lengths()

    base = OmegaConf.load(args.base_cfg)

    # Sanity: base must have the V8 anti-collapse + init_pose stack so R40
    # tests the *additional* plan-energy axis, not "anti-collapse alone".
    loss_cfg = base.loss
    if float(loss_cfg.get("w_moment_velocity", 0.0)) <= 0:
        raise SystemExit(
            "base cfg w_moment_velocity must be > 0 (R31 V7-A anti-collapse "
            "stack); R40 is the next axis on top of V8 V6."
        )
    if float(loss_cfg.get("w_yaw_aggregate", 0.0)) <= 0:
        raise SystemExit(
            "base cfg w_yaw_aggregate must be > 0 (R31 V7-B); R40 needs the "
            "V8 V6 substrate."
        )
    if int(base.model.denoiser.get("init_pose_dim", 0)) != 135:
        raise SystemExit(
            "base cfg model.denoiser.init_pose_dim must be 135 (R31 V8 F1 "
            "frame-0 anchor); R40 inherits this and does not re-tune it."
        )

    print(f"[R40 cfg gen] base: {args.base_cfg}")
    print("[R40 cfg gen] variant table:")
    print(f"  {'variant':<28s}  {'x0w':>4s}  {'velw':>4s}  {'plan':>5s}")
    for vid, overrides in VARIANTS:
        has_x0w = "yes" if "x0_channel_weights" in overrides else "no"
        has_velw = "yes" if "vel_channel_weights" in overrides else "no"
        plan_w = overrides.get("w_r40_plan_invariant", 0.0)
        print(f"  {vid:<28s}  {has_x0w:>4s}  {has_velw:>4s}  {plan_w:>5.2f}")

    wrote: list[Path] = []
    for vid, overrides in VARIANTS:
        cfg = OmegaConf.create(OmegaConf.to_container(base, resolve=True))
        run_name = f"stage1_r40_{vid}"
        cfg.output_dir = f"runs/training/{run_name}"
        cfg.logging.run_name = run_name

        # Apply overrides under cfg.loss. R40 has no model-side knobs.
        for key, val in overrides.items():
            cfg.loss[key] = val

        # Sanity-check the materialised weights when present.
        for key in ("x0_channel_weights", "vel_channel_weights"):
            v = cfg.loss.get(key, None)
            if v is not None and len(v) not in (0, 23):
                raise SystemExit(
                    f"[{vid}] cfg.loss.{key} length {len(v)} must be 23 "
                    f"or empty."
                )

        out_path = args.out_dir / f"{run_name}.yaml"
        OmegaConf.save(cfg, out_path)
        wrote.append(out_path)

        active = (
            f"plan_invariant={float(cfg.loss.get('w_r40_plan_invariant', 0.0)):g} "
            f"x0w={'set' if cfg.loss.get('x0_channel_weights') else 'off'} "
            f"velw={'set' if cfg.loss.get('vel_channel_weights') else 'off'}"
        )
        print(f"  wrote {out_path}")
        print(f"    {active}")

    print(f"DONE: {len(wrote)} R40 configs under {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
