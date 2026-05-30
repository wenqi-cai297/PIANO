"""Round-32 V7 — Stage-1.5 anti-bug ablation matrix generator.

Per the R32 Phase 1 audit
(analyses/round32_phase1_dyn_audit_20260530_121621/audit_report.md):
Stage-1.5 V0 is NOT mode-collapsed (std ratios mostly in [0.83, 1.29]);
it has 5 distinct localised failure modes:

  B1 + B2 (wrist + footstep velocity under-articulation)
  B3 (phase unit-circle violation)
  B4 (stance / walking BCE not driving saturation)
  B5 (C41 wrist frame-0 invariant violated 5 cm)

V7 matrix:

    V0 control           = re-train stage1p5_interaction_v0 with seed 42
                           (noise floor measurement).
    V1 moment            = V0 + V7-A channel_moment_match_loss on all 31
                           channels (w_v7_moment_velocity=0.5). Targets
                           B1 + B2.
    V2 phase             = V0 with V0's weak w_s4_phase=0.05 turned OFF
                           and V7-B's stronger unit-norm + angle
                           consistency turned ON (1.0 + 0.5). Targets B3.
    V3 stance            = V0 + w_s4_stance 0.5 -> 2.0 + w_s4_walking
                           0.5 -> 2.0. Targets B4. Pure config-weight
                           change, no new loss.
    V4 frame0            = V0 + V7-D c41_wrist_frame0_consistency_loss
                           (w_v7_c41_frame0_wrist=1.0). Targets B5.
    V5 full              = V0 + V7-A + V7-B + V7-C + V7-D (everything on).
                           Primary ship candidate.

Outputs:
    configs/training/stage1p5_v7_v{0..5}_*.yaml
    analyses/round32_stage1p5_v7_manifest.json
    analyses/round32_stage1p5_v7_manifest.md

Usage:
    python scripts/stage_a_generator/round32_make_stage1p5_v7_configs.py
    python scripts/stage_a_generator/round32_make_stage1p5_v7_configs.py --dry-run
    python scripts/stage_a_generator/round32_make_stage1p5_v7_configs.py \
        --data-root /media/8TB_data/Cai/datasets/InterAct/piano_official_process_4
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = ROOT / "configs" / "training"
DEFAULT_ANALYSES_DIR = ROOT / "analyses"
DEFAULT_DATA_ROOT = "E:/Project/Datasets/InterAct/piano_official_process_4"
DATASET_SUBSET_NAMES: tuple[str, ...] = (
    "chairs", "imhd", "neuraldome", "omomo_correct_v2",
)

SCHEDULE_BATCH_SIZE = 48
SCHEDULE_ACCUM_STEPS = 1
SCHEDULE_NUM_EPOCHS = 80

ARCH_D_MODEL = 384
ARCH_N_LAYERS = 6
ARCH_N_HEADS = 4
ARCH_FF_MULT = 4
ARCH_DROPOUT = 0.1


@dataclass(slots=True)
class Stage1p5V7Variant:
    variant_id: str
    purpose: str

    d_model: int = ARCH_D_MODEL
    n_layers: int = ARCH_N_LAYERS
    n_heads: int = ARCH_N_HEADS
    ff_mult: int = ARCH_FF_MULT
    dropout: float = ARCH_DROPOUT
    use_text: bool = True

    # V0 baseline loss weights.
    w_x0_c41: float = 1.0
    w_x0_s4: float = 1.0
    w_c41_jl: float = 0.1
    c41_joint_limit_m: float = 1.5
    w_s4_stance: float = 0.5
    w_s4_phase: float = 0.05
    w_s4_walking: float = 0.5

    # R32 V7 anti-bug loss weights (all OFF in baseline).
    w_v7_moment_velocity: float = 0.0
    w_v7_moment_value: float = 0.0
    w_v7_phase_unit_norm: float = 0.0
    w_v7_phase_angle: float = 0.0
    w_v7_c41_frame0_wrist: float = 0.0

    use_min_snr_weighting: bool = True
    min_snr_gamma: float = 5.0


# ─── R32 V7 ablation matrix — 6 variants ──────────────────────────────────

VARIANTS: list[Stage1p5V7Variant] = [
    Stage1p5V7Variant(
        variant_id="stage1p5_v7_v0_control",
        purpose=(
            "V0 control. Same loss design as stage1p5_interaction_v0 "
            "(MSE on C41+S4, weak phase unit-norm, stance+walking BCE). "
            "Re-trained from scratch with seed 42 so V7 noise floor can "
            "be compared against this re-baseline rather than the "
            "historical V0 ckpt."
        ),
    ),
    Stage1p5V7Variant(
        variant_id="stage1p5_v7_v1_moment",
        purpose=(
            "V1 = V0 + V7-A channel moment match (all 31 channels, "
            "w_v7_moment_velocity=0.5). Targets B1+B2 (wrist + footstep "
            "vel under-articulation). normalize_by_gt_std=True so "
            "heterogeneous channel scales don't dominate."
        ),
        w_v7_moment_velocity=0.5,
    ),
    Stage1p5V7Variant(
        variant_id="stage1p5_v7_v2_phase",
        purpose=(
            "V2 = V0 + V7-B phase unit-norm + angle consistency. "
            "Turn V0's weak w_s4_phase OFF (0.05 -> 0.0) and add stronger "
            "unit-norm penalty (w=1.0) plus angle consistency "
            "(1 - cos(angle_p - angle_g), w=0.5). Targets B3 "
            "(audit phase unit-circle dev 0.027/0.030)."
        ),
        w_s4_phase=0.0,
        w_v7_phase_unit_norm=1.0,
        w_v7_phase_angle=0.5,
    ),
    Stage1p5V7Variant(
        variant_id="stage1p5_v7_v3_stance",
        purpose=(
            "V3 = V0 + V7-C stance/walking BCE weight 0.5 -> 2.0. Pure "
            "config-weight change. Targets B4 (stance mean 0.51 std "
            "0.38, no logit ever |x| > 2; BCE pressure too small)."
        ),
        w_s4_stance=2.0,
        w_s4_walking=2.0,
    ),
    Stage1p5V7Variant(
        variant_id="stage1p5_v7_v4_frame0",
        purpose=(
            "V4 = V0 + V7-D C41 wrist frame-0 consistency "
            "(w_v7_c41_frame0_wrist=1.0). Forces "
            "pred_c41[t=0, 0:6] -> 0 (by construction GT = 0). Targets "
            "B5 (audit wrist rms_at_t0 median 5.3 cm)."
        ),
        w_v7_c41_frame0_wrist=1.0,
    ),
    Stage1p5V7Variant(
        variant_id="stage1p5_v7_v5_full",
        purpose=(
            "V5 = V0 + V7-A + V7-B + V7-C + V7-D, all combined. Primary "
            "ship candidate. If V5 closes the wrist/footstep gap and "
            "phase unit-circle violation drops below 0.005, V5 becomes "
            "the new Stage-1.5 mainline; otherwise the per-bug ablations "
            "V1..V4 tell us which mechanism is load-bearing."
        ),
        # V7-A
        w_v7_moment_velocity=0.5,
        # V7-B (V0 phase weight off)
        w_s4_phase=0.0,
        w_v7_phase_unit_norm=1.0,
        w_v7_phase_angle=0.5,
        # V7-C
        w_s4_stance=2.0,
        w_s4_walking=2.0,
        # V7-D
        w_v7_c41_frame0_wrist=1.0,
    ),
]


def _render_datasets_block(data_root: str) -> str:
    stripped = data_root.rstrip("/").rstrip("\\")
    lines: list[str] = []
    for sub in DATASET_SUBSET_NAMES:
        lines.append(f'    - name: "{sub}"')
        lines.append(f'      root: "{stripped}/{sub}"')
    return "\n".join(lines)


def _render_yaml(v: Stage1p5V7Variant, *, data_root: str) -> str:
    run_name = v.variant_id
    use_text = "true" if v.use_text else "false"
    use_min_snr = "true" if v.use_min_snr_weighting else "false"
    return f"""# Stage-1.5 V7 anti-bug ablation — {v.variant_id}
#
# Generated by scripts/stage_a_generator/round32_make_stage1p5_v7_configs.py.
# Phase 1 audit:
#   analyses/round32_phase1_dyn_audit_20260530_121621/audit_report.md
#
# Output: 31-D (C41 18 + S4 13). PB1 cond contract.
# FULL DATA, {SCHEDULE_NUM_EPOCHS} ep / val_every=5 / save_every=10 / warmup=500.
# Schedule: bs={SCHEDULE_BATCH_SIZE} accum={SCHEDULE_ACCUM_STEPS} (2x 5080).

model:
  cfg_drop_prob: 0.15
  diffusion:
    num_steps: 1000
    schedule: "cosine"
    prediction_target: "x0"
  denoiser:
    motion_dim: 31
    stage1_coarse_dim: 23
    object_traj_dim: 9
    text_dim: 512
    object_token_dim: 256
    object_num_tokens: 128
    d_model: {v.d_model}
    n_layers: {v.n_layers}
    n_heads: {v.n_heads}
    ff_mult: {v.ff_mult}
    dropout: {v.dropout}
    use_text: {use_text}
  object_encoder:
    num_input_points: 1024
    num_output_tokens: 128
    feature_dim: 256
  text_encoder:
    clip_version: "ViT-B/32"
    download_root: "cache/clip"

data:
  datasets:
{_render_datasets_block(data_root)}
  pseudo_label_dir: null
  pseudo_label_subdir: "pseudo_labels/v18_h10_f05_pelvis20_official_semantic_marker"
  max_seq_length: 196
  support_collapse_hand_support: true
  motion_representation: "smpl_pose_135_plan"
  force_world_frame: true
  subsample_n_per_object: null
  overfit_n_clips: 0
  stage1_coarse_cache_root: "cache/stage1_coarse_v1_full"
  surface_temporal_aux_fields: false
  r29_coarse_variant: "C41-current"
  r29_interaction_variant: "I0"
  r29_support_variant: "S4-S1-phase-footstep"
  r29_body_variant: "B0"
  r29_body_coord_frame: null
  r29_body_energy_threshold: 0.05
  r29_body_lowpass_window: 9
  r29_hand_offset_clamp_m: 2.0
  subject_split:
    enabled: true
    train_pct: 85
    val_pct: 15
    seed: 42
  augmentation:
    enabled: false
    mirror_prob: 0.0
    rotate_around_y_prob: 0.0
    pc_jitter_std: 0.0

training:
  batch_size: {SCHEDULE_BATCH_SIZE}
  num_epochs: {SCHEDULE_NUM_EPOCHS}
  num_workers: 4
  seed: 42
  optimizer:
    name: "adamw"
    lr: 8.0e-5
    weight_decay: 0.01
    betas: [0.9, 0.999]
  scheduler:
    name: "cosine"
    warmup_steps: 500
  gradient_accumulation_steps: {SCHEDULE_ACCUM_STEPS}
  max_grad_norm: 1.0
  mixed_precision: "bf16"
  val_every_epochs: 5
  val_best_key: "loss"

loss:
  w_x0_c41: {v.w_x0_c41}
  w_x0_s4: {v.w_x0_s4}
  w_c41_jl: {v.w_c41_jl}
  c41_joint_limit_m: {v.c41_joint_limit_m}
  w_s4_stance: {v.w_s4_stance}
  w_s4_phase: {v.w_s4_phase}
  w_s4_walking: {v.w_s4_walking}
  # R32 V7 anti-bug losses (default 0 = OFF):
  w_v7_moment_velocity: {v.w_v7_moment_velocity}
  w_v7_moment_value: {v.w_v7_moment_value}
  w_v7_phase_unit_norm: {v.w_v7_phase_unit_norm}
  w_v7_phase_angle: {v.w_v7_phase_angle}
  w_v7_c41_frame0_wrist: {v.w_v7_c41_frame0_wrist}
  use_min_snr_weighting: {use_min_snr}
  min_snr_gamma: {v.min_snr_gamma}

logging:
  project: "piano"
  run_name: "{run_name}"
  log_every_n_steps: 50
  save_every_n_epochs: 10

output_dir: "runs/training/{run_name}"
"""


def _manifest_row(v: Stage1p5V7Variant) -> dict:
    return {
        "variant_id": v.variant_id,
        "group": "Stage1p5InteractionV7",
        "purpose": v.purpose,
        "train": True,
        "from_scratch": True,
        "architecture": {
            "d_model": v.d_model, "n_layers": v.n_layers,
            "n_heads": v.n_heads, "ff_mult": v.ff_mult,
            "dropout": v.dropout, "use_text": v.use_text,
        },
        "schedule": {
            "batch_size": SCHEDULE_BATCH_SIZE,
            "gradient_accumulation_steps": SCHEDULE_ACCUM_STEPS,
            "num_epochs": SCHEDULE_NUM_EPOCHS,
            "lr": 8.0e-5, "warmup_steps": 500,
            "mixed_precision": "bf16", "seed": 42,
            "val_every_epochs": 5, "save_every_n_epochs": 10,
        },
        "loss": {
            "w_x0_c41": v.w_x0_c41,
            "w_x0_s4": v.w_x0_s4,
            "w_c41_jl": v.w_c41_jl,
            "w_s4_stance": v.w_s4_stance,
            "w_s4_phase": v.w_s4_phase,
            "w_s4_walking": v.w_s4_walking,
            "w_v7_moment_velocity": v.w_v7_moment_velocity,
            "w_v7_moment_value": v.w_v7_moment_value,
            "w_v7_phase_unit_norm": v.w_v7_phase_unit_norm,
            "w_v7_phase_angle": v.w_v7_phase_angle,
            "w_v7_c41_frame0_wrist": v.w_v7_c41_frame0_wrist,
        },
        "config_path": f"configs/training/{v.variant_id}.yaml",
        "output_dir": f"runs/training/{v.variant_id}",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--data-root",
        default=os.environ.get("DATASETS_ROOT", DEFAULT_DATA_ROOT),
    )
    parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--analyses-dir", default=str(DEFAULT_ANALYSES_DIR))
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    analyses_dir = Path(args.analyses_dir)

    rows: list[dict] = []
    for v in VARIANTS:
        out_path = config_dir / f"{v.variant_id}.yaml"
        content = _render_yaml(v, data_root=args.data_root)
        rows.append(_manifest_row(v))
        if args.dry_run:
            print(f"DRY-RUN would write: {out_path}  ({len(content)} bytes)")
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(content, encoding="utf-8")
            print(f"wrote {out_path}")

    manifest_json = analyses_dir / "round32_stage1p5_v7_manifest.json"
    manifest_md = analyses_dir / "round32_stage1p5_v7_manifest.md"
    md_lines: list[str] = [
        "# Round-32 V7 Stage-1.5 anti-bug manifest",
        "",
        "Per Phase 1 dynamic-info audit "
        "(analyses/round32_phase1_dyn_audit_20260530_121621/audit_report.md). "
        "FULL InterAct train set, from scratch, seed 42, "
        f"{SCHEDULE_NUM_EPOCHS} ep, bs={SCHEDULE_BATCH_SIZE} / "
        f"accum={SCHEDULE_ACCUM_STEPS} (2x 5080).",
        "",
        "## Variants",
        "",
        "| variant | V7-A moment_vel | V7-B phase unit | V7-B phase angle | V7-C stance/walk x4 | V7-D frame0 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for v in VARIANTS:
        x4 = "yes" if v.w_s4_stance == 2.0 else "no"
        md_lines.append(
            f"| `{v.variant_id}` | "
            f"{v.w_v7_moment_velocity} | {v.w_v7_phase_unit_norm} | "
            f"{v.w_v7_phase_angle} | {x4} | {v.w_v7_c41_frame0_wrist} |"
        )
    md_lines.append("")
    md_lines.append("## Decision tree after results")
    md_lines.append("")
    md_lines.append("| outcome | next step |")
    md_lines.append("|---|---|")
    md_lines.append(
        "| V5 (full) closes >= 6 cm wrist drift over V0 | "
        "Ship V5; queue cascade training (V8) on top |"
    )
    md_lines.append(
        "| V1 (moment) alone closes >= 4 cm | "
        "V7-A is dominant; ship V1; consider cascade only if V1 insufficient |"
    )
    md_lines.append(
        "| V4 (frame0) alone closes >= 3 cm | "
        "B5 was the dominant bug; ship V4 + queue rest as V9 |"
    )
    md_lines.append(
        "| All V7 variants close < 2 cm wrist drift | "
        "Per-channel loss design not enough; pivot to cascade training (Stage-1.5 "
        "learns to consume Stage-1 generated cond) |"
    )
    md_lines.append("")

    if args.dry_run:
        print(f"DRY-RUN would write: {manifest_json}")
        print(f"DRY-RUN would write: {manifest_md}")
    else:
        analyses_dir.mkdir(parents=True, exist_ok=True)
        manifest_json.write_text(
            json.dumps(
                {"variants": rows, "data_root": args.data_root}, indent=2,
            ),
            encoding="utf-8",
        )
        manifest_md.write_text("\n".join(md_lines), encoding="utf-8")
        print(f"wrote {manifest_json}")
        print(f"wrote {manifest_md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
