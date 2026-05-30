"""Round-34 Stage-1.5 wrist low-band + cond-aug ablation YAML generator.

Per ``analyses/2026-05-31_stage1p5_wrist_drift_claude_code_ablation_plan.md``
§7.4. The 4 variants share R33 V1 substrate (per-block obj_xattn) and only
differ in (cond_aug_sigma_max, wrist_lowband_weight).

CRITICAL: only run AFTER D1 + D2 reports exist (analyses/2026-05-31_…/02
and 03) AND at least one of the brief's §7.1 gate conditions holds. The
trainer reads zero R34 knobs unless YAMLs set them, so V0/V7/R33 ckpts
remain bit-identical.

Output:
    configs/training/stage1p5_r34_c0_control.yaml      # σ=0,    λ=0
    configs/training/stage1p5_r34_c1_cond_aug.yaml     # σ=0.15, λ=0
    configs/training/stage1p5_r34_c2_lowband.yaml      # σ=0,    λ=1.0
    configs/training/stage1p5_r34_c3_combined.yaml     # σ=0.15, λ=1.0

CPU only. ~1 s.

Run:
    python -u scripts/stage_a_generator/round34_make_stage1p5_configs.py \\
        --base-cfg configs/training/stage1p5_r33_v1_xattn.yaml \\
        --out-dir  configs/training/
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

from omegaconf import OmegaConf


VARIANTS = (
    {
        "vid": "c0_control",
        "cond_aug_sigma_max": 0.00,
        "wrist_lowband_weight": 0.00,
        "wrist_lowband_cutoff_hz": 1.00,
    },
    {
        "vid": "c1_cond_aug",
        "cond_aug_sigma_max": 0.15,
        "wrist_lowband_weight": 0.00,
        "wrist_lowband_cutoff_hz": 1.00,
    },
    {
        "vid": "c2_lowband",
        "cond_aug_sigma_max": 0.00,
        "wrist_lowband_weight": 1.00,
        "wrist_lowband_cutoff_hz": 1.00,
    },
    {
        "vid": "c3_combined",
        "cond_aug_sigma_max": 0.15,
        "wrist_lowband_weight": 1.00,
        "wrist_lowband_cutoff_hz": 1.00,
    },
)


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
            "R34 requires the R33 V1 substrate."
        )

    for v in VARIANTS:
        cfg = OmegaConf.create(OmegaConf.to_container(base, resolve=True))
        run_name = f"stage1p5_r34_{v['vid']}"
        cfg.output_dir = f"runs/training/{run_name}"
        cfg.logging.run_name = run_name

        # R34 loss knobs (default to 0 in trainer; explicit here for audit trail).
        cfg.loss.w_r34_wrist_lowband = float(v["wrist_lowband_weight"])
        cfg.loss.r34_wrist_lowband_cutoff_hz = float(v["wrist_lowband_cutoff_hz"])
        cfg.loss.r34_wrist_lowband_fps = 20.0
        cfg.loss.r34_cond_aug_sigma_max = float(v["cond_aug_sigma_max"])

        out_path = args.out_dir / f"{run_name}.yaml"
        OmegaConf.save(cfg, out_path)
        print(f"  wrote {out_path}")

    print(f"DONE: {len(VARIANTS)} R34 configs under {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
