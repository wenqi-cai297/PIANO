"""Round-41 Stage-1 cascade fine-tune config generator.

Generates 5 R41 configs from the V8 V6 base:

  A0 cascade_off         — control, w_cascade=0; sanity check that
                           cascade trainer + warm-start preserve V8 V6
                           behavior bit-for-bit.
  A1 motion_mse_only     — cascade = min-SNR weighted motion MSE only.
  A2 + world_joint_vel   — A1 + world-frame joint velocity MSE.
  A3 + l_pos_full        — A2 + FK-derived dense L_pos (hand/foot 2x).
  A4 + anchor_joint_pos  — A3 + contact-active wrist anchor.

All 4 non-control cells include the σ=0.05 noise injection on Stage-1's
stage1_coarse output before feeding PB1 (matches PB1 training-time
``stage1_coarse_noise_std``).

The ``w_cascade_total`` field is left at 1.0 in the generated configs.
The launcher's calibration phase reads each cell's enabled cascade
terms, runs a single P0-style 1-batch forward, measures
``grad_norm(cascade) / grad_norm(stage1_self)``, and patches
``cascade.w_total`` to bring the ratio into [0.5, 1.5] before training.

All cells inherit V8 V6 self loss stack verbatim (V7+V8 anti-collapse,
wrist_fk, init_pose F1). User direction: "loss 先不要改" — cascade is
added on top, V8 V6 self stack unchanged.

Run:

    python scripts/stage_a_generator/round41_make_stage1_cascade_configs.py \\
        --base-cfg configs/training/stage1_v8_v6_full_f1.yaml \\
        --out-dir  configs/training/
"""
from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf


# Default PB1 cfg + ckpt paths the cascade trainer loads. Override per-
# cell if needed.
DEFAULT_PB1_CFG = "configs/training/anchordiff_r29_pb_a1_adaln_s4.yaml"
DEFAULT_PB1_CKPT = (
    "runs/training/stageB_anchordiff_r29_pb_a1_adaln_s4/final.pt"
)


# Per-cell cascade flag overrides. Each cell turns on a strict superset
# of the previous (A1 ⊂ A2 ⊂ A3 ⊂ A4).
#
# Initial w_* are the PB1 ship weights from anchordiff_r29_pb_a1_adaln_s4.yaml
# (motion_mse=1.0 implicit, world_joint_velocity=1.0, pos_loss=5.0,
# anchor_joint_pos=10.0). The launcher rescales them via cascade.w_total
# after calibration.
VARIANTS = (
    ("a0_cascade_off", {
        "enabled": False,
        "w_motion_mse": 0.0,
        "w_world_joint_vel": 0.0,
        "w_l_pos_full": 0.0,
        "w_anchor_joint_pos": 0.0,
    }),
    ("a1_motion_mse", {
        "enabled": True,
        "w_motion_mse": 1.0,
        "w_world_joint_vel": 0.0,
        "w_l_pos_full": 0.0,
        "w_anchor_joint_pos": 0.0,
    }),
    ("a2_world_vel", {
        "enabled": True,
        "w_motion_mse": 1.0,
        "w_world_joint_vel": 1.0,
        "w_l_pos_full": 0.0,
        "w_anchor_joint_pos": 0.0,
    }),
    ("a3_l_pos_full", {
        "enabled": True,
        "w_motion_mse": 1.0,
        "w_world_joint_vel": 1.0,
        "w_l_pos_full": 5.0,
        "w_anchor_joint_pos": 0.0,
    }),
    ("a4_anchor_pos", {
        "enabled": True,
        "w_motion_mse": 1.0,
        "w_world_joint_vel": 1.0,
        "w_l_pos_full": 5.0,
        "w_anchor_joint_pos": 10.0,
    }),
)


# Shared cascade settings applied to every non-control cell.
SHARED_CASCADE = {
    "pb1_config": DEFAULT_PB1_CFG,
    "pb1_checkpoint": DEFAULT_PB1_CKPT,
    "w_total": 1.0,                      # calibrated by launcher
    "use_min_snr": True,
    "min_snr_gamma": 5.0,
    "stage1_coarse_noise_std": 0.05,     # matches PB1 training-time
    "l_pos_hand_endpoint_weight": 2.0,
    "l_pos_foot_endpoint_weight": 2.0,
    "anchor_part_weights": [2.0, 2.0, 0.0, 0.0, 0.5],
    "anchor_contact_threshold": 0.5,
}


# R41 trainer settings — fine-tune (not from-scratch) so lr is lower,
# epochs fewer, warm start required.
R41_TRAINING = {
    "init_checkpoint": "runs/training/stage1_v8_v6_full_f1/final.pt",
    "init_checkpoint_strict": True,
    "num_epochs": 40,                # fine-tune (V8 V6 trained 80)
    "lr_override": 2.0e-5,           # 1/5 of V8 V6 from-scratch (1e-4)
    "warmup_steps": 200,
}


# R29 variants the dataloader needs for PB1 cond surface. Same as in the
# P0 hybrid cfg (confirmed by P0 check 1 batch_contract pass).
R41_DATA_R29_OVERRIDES = {
    "r29_coarse_variant": "C41-current",
    "r29_support_variant": "S4-S1-phase-footstep",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-cfg", type=Path,
        default=Path("configs/training/stage1_v8_v6_full_f1.yaml"),
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

    # Sanity — base must be V8 V6 (init_pose_dim=135 F1, full anti-
    # collapse + wrist_fk stack).
    if int(base.model.denoiser.get("init_pose_dim", 0)) != 135:
        raise SystemExit(
            "base cfg model.denoiser.init_pose_dim must be 135 (V8 V6 F1). "
            "R41 inherits this from V8 V6 unchanged."
        )
    if float(base.loss.get("w_moment_velocity", 0.0)) <= 0:
        raise SystemExit(
            "base cfg w_moment_velocity must be > 0 (V7 anti-collapse). "
            "R41 keeps the V8 V6 self stack."
        )

    print(f"[R41 cfg gen] base: {args.base_cfg}")
    print(f"[R41 cfg gen] variants (5 = 1 control + 4 cascade cells):")

    wrote: list[Path] = []
    for vid, cascade_overrides in VARIANTS:
        cfg = OmegaConf.create(OmegaConf.to_container(base, resolve=True))
        run_name = f"stage1_r41_{vid}"
        cfg.output_dir = f"runs/training/{run_name}"
        cfg.logging.run_name = run_name

        # R41 training overrides.
        cfg.training.init_checkpoint = R41_TRAINING["init_checkpoint"]
        cfg.training.init_checkpoint_strict = R41_TRAINING[
            "init_checkpoint_strict"
        ]
        cfg.training.num_epochs = int(R41_TRAINING["num_epochs"])
        cfg.training.optimizer.lr = float(R41_TRAINING["lr_override"])
        cfg.training.scheduler.warmup_steps = int(
            R41_TRAINING["warmup_steps"]
        )

        # R29 variant surface (so dataloader produces stage2_coarse_extra
        # + stage2_support that the cascade PB1 forward needs).
        for k, v in R41_DATA_R29_OVERRIDES.items():
            cfg.data[k] = v

        # Cascade section.
        cascade_block = dict(SHARED_CASCADE)
        cascade_block.update(cascade_overrides)
        cfg.cascade = cascade_block

        out_path = args.out_dir / f"{run_name}.yaml"
        OmegaConf.save(cfg, out_path)
        wrote.append(out_path)

        if cascade_block["enabled"]:
            active_terms = [
                k.replace("w_", "") for k in (
                    "w_motion_mse", "w_world_joint_vel",
                    "w_l_pos_full", "w_anchor_joint_pos",
                )
                if cascade_block[k] > 0
            ]
            summary = "+".join(active_terms)
        else:
            summary = "off (control)"
        print(f"  wrote {out_path}   cascade={summary}")

    print(f"DONE: {len(wrote)} R41 configs under {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
