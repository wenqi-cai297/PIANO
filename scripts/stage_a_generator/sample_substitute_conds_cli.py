"""CLI wrapper around ``piano.inference.sample_substitute_conds``.

Examples:

    # R31 step 1: sample Stage-1 outputs for the val 48-clip selection.
    python scripts/stage_a_generator/sample_substitute_conds_cli.py \\
        --stage stage1 \\
        --config configs/training/stage1_traj_v0.yaml \\
        --ckpt   runs/training/stage1_traj_v0/final.pt \\
        --selection-json analyses/round29_val_diag_indices_48_balanced.json \\
        --bucket val \\
        --out-dir analyses/round31_stage1_substitute_conds/val

    # R32 step 1: sample Stage-1.5 outputs against oracle Stage-1.
    python scripts/stage_a_generator/sample_substitute_conds_cli.py \\
        --stage stage1p5 \\
        --config configs/training/stage1p5_interaction_v0.yaml \\
        --ckpt   runs/training/stage1p5_interaction_v0/final.pt \\
        --selection-json analyses/round29_val_diag_indices_48_balanced.json \\
        --bucket val \\
        --out-dir analyses/round32_stage1p5_substitute_conds/val

    # End-to-end step 1b: sample Stage-1.5 against Stage-1's cache.
    python scripts/stage_a_generator/sample_substitute_conds_cli.py \\
        --stage stage1p5 \\
        --config configs/training/stage1p5_interaction_v0.yaml \\
        --ckpt   runs/training/stage1p5_interaction_v0/final.pt \\
        --selection-json analyses/round29_val_diag_indices_48_balanced.json \\
        --bucket val \\
        --out-dir analyses/round31_32_end_to_end_substitute_conds/val \\
        --upstream-dir analyses/round31_stage1_substitute_conds/val
"""
from __future__ import annotations

import argparse
from pathlib import Path

from piano.inference.sample_substitute_conds import sample_substitute_conds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", required=True, choices=["stage1", "stage1p5"],
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--bucket", choices=["train", "val"], default="val")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--upstream-dir", type=Path, default=None,
        help=(
            "Optional dir of Stage-1 sampled outputs (npz per clip). "
            "When given, Stage-1.5 sampling reads stage1_coarse from "
            "this cache instead of the GT oracle. Required for the "
            "end-to-end (D) comparison."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument(
        "--sampler", choices=["ddpm", "ddim_eta0", "ddpm_det"],
        default="ddim_eta0",
    )
    args = parser.parse_args()

    n = sample_substitute_conds(
        config_path=args.config,
        ckpt_path=args.ckpt,
        selection_json=args.selection_json,
        out_dir=args.out_dir,
        bucket=args.bucket,
        stage=args.stage,
        upstream_dir=args.upstream_dir,
        seed=args.seed,
        cfg_scale=args.cfg_scale,
        sampler=args.sampler,
    )
    print(f"DONE: {n} clips → {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
