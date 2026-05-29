"""Round-31 V7 — Stage-1 anti-mode-collapse ablation matrix generator.

Per ``analyses/2026-05-30_round31_v7_anti_collapse_design.md`` (to be
written) and the Phase 1 dynamic-info audit
(``analyses/round31_phase1_dyn_audit_20260530_043948/audit_report.md``).

The Phase 1 audit established that R31 V0's failure is **mode collapse**
under L2 + multimodal conditioning:

  - All 6 channel groups (root, vel, yaw, pelvis_rot6d, spine3_rot6d,
    heights) show std ratio 0.24–0.46 vs GT.
  - PSD high-band energy ratio is 0.00–0.16 except heights — pred output
    has almost no high-frequency content.
  - yaw_vel finite-diff RMS is 2 % of GT — model emits a near-stationary
    yaw.
  - yaw_cos pred mean = 0.91 vs GT 0.30 — model defaults to a single
    facing direction across all clips.

V2's design (rot6d ortho + FK pos + height-FK + kinematic self-consistency)
attacked rot6d-on-SO(3) and channel-consistency, but std collapse was
never targeted; result: 18.5 cm drift across all 6 V2 variants.

This V7 matrix borrows the mechanisms that pulled Stage-2 PB1 out of the
same mode-collapse trap (train_anchordiff.py:485–1166):

  V7-A  channel_moment_match_loss   (velocity moments)
        → mirror of stable_local_speed_moment_weight (0.02).

  V7-B  yaw_aggregate_match_loss
        → mirror of r29_gait_transition_rate_weight (0.2) +
          r29_gait_duty_cycle_weight (0.1).

  V7-C  fk_pelvis_spine_pos_loss_cm  (cm-space SmoothL1)
        → mirror of pos_loss_weight=5.0 + stable_local_vel_cm_weight=0.05.

Outputs:
    configs/training/stage1_v7_v{0..5}_*.yaml
    analyses/round31_stage1_v7_manifest.json
    analyses/round31_stage1_v7_manifest.md

Usage:
    python scripts/stage_a_generator/round31_make_stage1_v7_configs.py
    python scripts/stage_a_generator/round31_make_stage1_v7_configs.py --dry-run
    python scripts/stage_a_generator/round31_make_stage1_v7_configs.py \\
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

SCHEDULE_BATCH_SIZE = 64
SCHEDULE_ACCUM_STEPS = 1
SCHEDULE_NUM_EPOCHS = 80

ARCH_D_MODEL = 256
ARCH_N_LAYERS = 4
ARCH_N_HEADS = 4
ARCH_FF_MULT = 4
ARCH_DROPOUT = 0.1


@dataclass(slots=True)
class Stage1V7Variant:
    variant_id: str
    purpose: str

    # Architecture.
    d_model: int = ARCH_D_MODEL
    n_layers: int = ARCH_N_LAYERS
    n_heads: int = ARCH_N_HEADS
    ff_mult: int = ARCH_FF_MULT
    dropout: float = ARCH_DROPOUT
    use_text: bool = True

    # V0 loss baseline.
    w_x0: float = 1.0
    w_vel: float = 1.0
    w_yaw_smooth: float = 0.0

    # R31 V2 ablation losses (default OFF — V7 doesn't use them).
    w_rot6d_ortho: float = 0.0
    w_fk_pos: float = 0.0          # L2 (m² L2 form; superseded by V7-C in V7)
    w_height_fk: float = 0.0
    w_self_consistency: float = 0.0
    vel_rot6d_weight: float = 1.0

    # R31 V7 anti-collapse losses.
    # V7-A: per-channel (mean, std) match of finite-diff magnitudes
    # (raw-space). Directly penalises std collapse. Magnitude ~ (rad/frame)²
    # ≈ 1e-2; weight 0.5 brings contribution to the same order as MSE-x0.
    w_moment_velocity: float = 0.0
    # V7-A' (rarely used): same on raw values, not derivatives. Catches
    # mean shift directly. Magnitude is per-channel value² so much larger;
    # weight 0.01–0.05 typical.
    w_moment_value: float = 0.0
    # V7-B: yaw aggregate (transition rate + cumulative range). rad/frame
    # ≈ 0.05 × per-clip SmoothL1; weight 2.0 to give it real bite.
    w_yaw_aggregate: float = 0.0
    # V7-C: cm-space SmoothL1 FK pos. ~ cm magnitude, SmoothL1 reduction.
    # Weight 0.1–0.5 (PB1 uses 0.05 on stable_local_vel_cm; we're applying
    # to a larger set of joints — head/neck/shoulders — so similar scale).
    w_fk_pos_cm: float = 0.0
    fk_pos_cm_beta: float = 1.0

    use_min_snr_weighting: bool = True
    min_snr_gamma: float = 5.0

    val_best_key: str = "mse_x0"


# ─── R31 V7 ablation matrix — 6 variants ─────────────────────────────────
# Rationale: isolate each anti-collapse mechanism, then combine.

VARIANTS: list[Stage1V7Variant] = [
    Stage1V7Variant(
        variant_id="stage1_v7_v0_baseline",
        purpose=(
            "V0 baseline. Same as stage1_v2_v0_baseline. Re-train to "
            "establish noise floor against which V7-A/B/C are scored."
        ),
    ),
    Stage1V7Variant(
        variant_id="stage1_v7_v1_moment",
        purpose=(
            "V7-A only — per-channel velocity (mean, std) match "
            "(w=0.5). Phase-1-audit-driven attack on std collapse. "
            "Hypothesis: directly penalising pred velocity std collapse "
            "should restore high-frequency content in yaw_vel and rot6d "
            "channels, closing >=3 cm of the 10 cm drift gap."
        ),
        w_moment_velocity=0.5,
    ),
    Stage1V7Variant(
        variant_id="stage1_v7_v2_yaw_agg",
        purpose=(
            "V7-B only — yaw aggregate transition-rate + cumulative-range "
            "matching (w=2.0). Targets the dataset-mean yaw collapse "
            "(audit: yaw_cos pred 0.91 vs GT 0.30; yaw_vel RMS 2% of GT). "
            "Mode-invariant: doesn't penalise CW vs CCW choice."
        ),
        w_yaw_aggregate=2.0,
    ),
    Stage1V7Variant(
        variant_id="stage1_v7_v3_fk_pos_cm",
        purpose=(
            "V7-C only — cm-space SmoothL1 FK pos on head/neck/shoulders "
            "(w=0.2). Mirrors PB1's stable_local_vel_cm pattern (cm scale "
            "+ SmoothL1). V2's L2 fk_pos was at weight 0.10 m² and "
            "contributed <1% of total loss; this is 100× larger in "
            "magnitude per unit weight."
        ),
        w_fk_pos_cm=0.2,
    ),
    Stage1V7Variant(
        variant_id="stage1_v7_v4_moment_yaw",
        purpose=(
            "V7-A + V7-B: velocity moment match + yaw aggregate. "
            "Tests whether the two latent-space anti-collapse mechanisms "
            "compound. Both directly attack the std-collapse axis."
        ),
        w_moment_velocity=0.5,
        w_yaw_aggregate=2.0,
    ),
    Stage1V7Variant(
        variant_id="stage1_v7_v5_full",
        purpose=(
            "V7-A + V7-B + V7-C: full anti-collapse stack. Mirror of "
            "PB1's combined L_pos (cm) + stable_local_speed_moment + "
            "G1 transition_rate mechanism. Hypothesis: this is the "
            "minimum loss design needed to escape the V0 mode-collapse "
            "regime; expected drift gap close 5–8 cm if H7 is dominant."
        ),
        w_moment_velocity=0.5,
        w_yaw_aggregate=2.0,
        w_fk_pos_cm=0.2,
    ),
]


def _render_datasets_block(data_root: str) -> str:
    stripped = data_root.rstrip("/").rstrip("\\")
    lines: list[str] = []
    for sub in DATASET_SUBSET_NAMES:
        lines.append(f'    - name: "{sub}"')
        lines.append(f'      root: "{stripped}/{sub}"')
    return "\n".join(lines)


def _render_yaml(v: Stage1V7Variant, *, data_root: str) -> str:
    run_name = v.variant_id
    use_text = "true" if v.use_text else "false"
    use_min_snr = "true" if v.use_min_snr_weighting else "false"
    return f"""# Stage-1 (Trajectory) V7 anti-mode-collapse — {v.variant_id}
#
# Generated by scripts/stage_a_generator/round31_make_stage1_v7_configs.py.
# Per analyses/2026-05-30_round31_v7_anti_collapse_design.md (to write).
# Phase 1 audit:
#   analyses/round31_phase1_dyn_audit_20260530_043948/audit_report.md
#
# Output: 23-D stage1_coarse (Stage-2 PB1 cond contract). FULL DATA.
# {SCHEDULE_NUM_EPOCHS} ep / val_every=5 / save_every=10 / warmup=500.
# Schedule: bs={SCHEDULE_BATCH_SIZE} accum={SCHEDULE_ACCUM_STEPS} (2× 5080).

model:
  cfg_drop_prob: 0.15
  diffusion:
    num_steps: 1000
    schedule: "cosine"
    prediction_target: "x0"
  denoiser:
    motion_dim: 23
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
  r29_coarse_variant: "C23"
  r29_interaction_variant: "I0"
  r29_support_variant: "S0"
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
    lr: 1.0e-4
    weight_decay: 0.01
    betas: [0.9, 0.999]
  scheduler:
    name: "cosine"
    warmup_steps: 500
  gradient_accumulation_steps: {SCHEDULE_ACCUM_STEPS}
  max_grad_norm: 1.0
  mixed_precision: "bf16"
  val_every_epochs: 5
  val_best_key: "{v.val_best_key}"

loss:
  w_x0: {v.w_x0}
  w_vel: {v.w_vel}
  w_yaw_smooth: {v.w_yaw_smooth}
  # R31 V2 ablation losses (default 0 = OFF in V7):
  w_rot6d_ortho: {v.w_rot6d_ortho}
  w_fk_pos: {v.w_fk_pos}
  w_height_fk: {v.w_height_fk}
  w_self_consistency: {v.w_self_consistency}
  vel_rot6d_weight: {v.vel_rot6d_weight}
  # R31 V7 anti-mode-collapse losses:
  w_moment_velocity: {v.w_moment_velocity}
  w_moment_value: {v.w_moment_value}
  w_yaw_aggregate: {v.w_yaw_aggregate}
  w_fk_pos_cm: {v.w_fk_pos_cm}
  fk_pos_cm_beta: {v.fk_pos_cm_beta}
  use_min_snr_weighting: {use_min_snr}
  min_snr_gamma: {v.min_snr_gamma}

logging:
  project: "piano"
  run_name: "{run_name}"
  log_every_n_steps: 50
  save_every_n_epochs: 10

output_dir: "runs/training/{run_name}"
"""


def _manifest_row(v: Stage1V7Variant) -> dict:
    return {
        "variant_id": v.variant_id,
        "group": "Stage1TrajectoryV7",
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
            "lr": 1.0e-4, "warmup_steps": 500,
            "mixed_precision": "bf16", "seed": 42,
            "val_every_epochs": 5, "save_every_n_epochs": 10,
            "val_best_key": v.val_best_key,
        },
        "loss": {
            "w_x0": v.w_x0, "w_vel": v.w_vel,
            "w_yaw_smooth": v.w_yaw_smooth,
            "w_rot6d_ortho": v.w_rot6d_ortho,
            "w_fk_pos": v.w_fk_pos,
            "w_height_fk": v.w_height_fk,
            "w_self_consistency": v.w_self_consistency,
            "vel_rot6d_weight": v.vel_rot6d_weight,
            "w_moment_velocity": v.w_moment_velocity,
            "w_moment_value": v.w_moment_value,
            "w_yaw_aggregate": v.w_yaw_aggregate,
            "w_fk_pos_cm": v.w_fk_pos_cm,
            "fk_pos_cm_beta": v.fk_pos_cm_beta,
            "use_min_snr_weighting": v.use_min_snr_weighting,
            "min_snr_gamma": v.min_snr_gamma,
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

    manifest_json = analyses_dir / "round31_stage1_v7_manifest.json"
    manifest_md = analyses_dir / "round31_stage1_v7_manifest.md"
    md_lines: list[str] = [
        "# Round-31 V7 Stage-1 anti-mode-collapse manifest",
        "",
        "Per Phase 1 dynamic-info audit "
        "(analyses/round31_phase1_dyn_audit_20260530_043948/audit_report.md). "
        "FULL InterAct train set, from scratch, seed 42, "
        f"{SCHEDULE_NUM_EPOCHS} ep, bs={SCHEDULE_BATCH_SIZE} / "
        f"accum={SCHEDULE_ACCUM_STEPS} (2× 5080).",
        "",
        "## Variants (R31 V7 ablation matrix)",
        "",
        "| variant | arch | V7-A moment_vel | V7-B yaw_agg | V7-C fk_pos_cm |",
        "|---|---|---:|---:|---:|",
    ]
    for v in VARIANTS:
        md_lines.append(
            f"| `{v.variant_id}` | "
            f"d_model={v.d_model} L={v.n_layers} | "
            f"{v.w_moment_velocity} | {v.w_yaw_aggregate} | {v.w_fk_pos_cm} |"
        )
    md_lines.append("")
    md_lines.append("## Decision tree after results")
    md_lines.append("")
    md_lines.append("| outcome | next step |")
    md_lines.append("|---|---|")
    md_lines.append(
        "| V7-A (moment) alone closes >=3 cm | Ship V1; weight tune for V6 |"
    )
    md_lines.append(
        "| V7-B (yaw) alone closes >=3 cm | Yaw collapse was the dominant axis; combine with V7-A |"
    )
    md_lines.append(
        "| V7-C (fk_pos_cm) alone closes >=3 cm | PB1 anti-collapse was physics-driven; ship V3 |"
    )
    md_lines.append(
        "| V5 (all three) closes >=8 cm | Anti-collapse worked; downstream gap mostly from H7 |"
    )
    md_lines.append(
        "| V5 closes <3 cm | H7 wasn't dominant; reconsider H1/H4/H5 next |"
    )
    md_lines.append("")

    if args.dry_run:
        print(f"DRY-RUN would write: {manifest_json}")
        print(f"DRY-RUN would write: {manifest_md}")
    else:
        analyses_dir.mkdir(parents=True, exist_ok=True)
        manifest_json.write_text(
            json.dumps(
                {"variants": rows, "data_root": args.data_root},
                indent=2,
            ),
            encoding="utf-8",
        )
        manifest_md.write_text("\n".join(md_lines), encoding="utf-8")
        print(f"wrote {manifest_json}")
        print(f"wrote {manifest_md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
