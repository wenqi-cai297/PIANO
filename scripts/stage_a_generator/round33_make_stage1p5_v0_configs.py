"""Round-33 — Stage-1.5 per-block obj_xattn ablation matrix generator.

Per the R32 V7 verdict
(analyses/round32_v7_matrix_summary_20260530_130411.md):

  All 6 V7 anti-bug loss variants closed < 1 cm wrist drift over V0.
  The pre-block obj_xattn audit signals from R32 V0 (per-channel
  std/vel ratios mostly OK, audit "bugs" largely repaired in V7 but
  drift unchanged) point to a STRUCTURAL issue: the per-frame stream
  inside Stage-1.5's DiT stack only sees obj_traj (COM + rot6d),
  while obj_tokens (the 128 PointNet++ surface tokens — the actual
  spatial signal for "where on the object should the wrist contact")
  only enters at the end-of-encoder cross-attention. That's a weak
  supervision pathway for a task whose primary output (C41 wrist Δxyz)
  is a per-frame spatial pointer into the object surface.

R33 tests the DiT-XL pattern: insert an AdaLN-Zero cross-attn sub-block
inside each DiT layer (between self-attn and MLP), giving every per-
frame token query access to obj_tokens at every depth.

Variants:

    V0 control             = R32 V7 V0 control re-trained, no per-block xattn.
                             Noise-floor measurement on identical schedule.
    V1 per_block_xattn     = + enable_per_block_obj_xattn=True. Pure
                             architecture change; loss stays V0.
    V2 xattn + moment      = V1 + V7-A channel_moment_match_loss
                             (w_v7_moment_velocity=0.5). Pairs the new
                             structure with the most general V7 loss.
    V3 xattn + v7_full     = V1 + V7 V5 full anti-bug stack (moment,
                             phase, stance, frame0). Upper-bound test.

Outputs:
    configs/training/stage1p5_v0_v{0..3}_*.yaml
    analyses/round33_stage1p5_v0_manifest.json
    analyses/round33_stage1p5_v0_manifest.md

(``_v0_`` in the variant_id is the R33 v0 iteration of obj-xattn
ablations; if a follow-up iteration is needed, name it v1.)

Usage:
    python scripts/stage_a_generator/round33_make_stage1p5_v0_configs.py
    python scripts/stage_a_generator/round33_make_stage1p5_v0_configs.py --dry-run
    python scripts/stage_a_generator/round33_make_stage1p5_v0_configs.py \
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
class Stage1p5R33Variant:
    variant_id: str
    purpose: str

    d_model: int = ARCH_D_MODEL
    n_layers: int = ARCH_N_LAYERS
    n_heads: int = ARCH_N_HEADS
    ff_mult: int = ARCH_FF_MULT
    dropout: float = ARCH_DROPOUT
    use_text: bool = True

    # R33 architecture switch.
    enable_per_block_obj_xattn: bool = False

    # V0 baseline loss weights.
    w_x0_c41: float = 1.0
    w_x0_s4: float = 1.0
    w_c41_jl: float = 0.1
    c41_joint_limit_m: float = 1.5
    w_s4_stance: float = 0.5
    w_s4_phase: float = 0.05
    w_s4_walking: float = 0.5

    # R32 V7 anti-bug losses (OFF by default).
    w_v7_moment_velocity: float = 0.0
    w_v7_moment_value: float = 0.0
    w_v7_phase_unit_norm: float = 0.0
    w_v7_phase_angle: float = 0.0
    w_v7_c41_frame0_wrist: float = 0.0

    use_min_snr_weighting: bool = True
    min_snr_gamma: float = 5.0


VARIANTS: list[Stage1p5R33Variant] = [
    Stage1p5R33Variant(
        variant_id="stage1p5_r33_v0_control",
        purpose=(
            "V0 control. Architecturally identical to V7 V0 control "
            "(no per-block obj_xattn). Re-trained with seed 42 to "
            "establish R33 noise floor for variant comparisons."
        ),
    ),
    Stage1p5R33Variant(
        variant_id="stage1p5_r33_v1_xattn",
        purpose=(
            "V1 = V0 + enable_per_block_obj_xattn=True. Inserts an "
            "AdaLN-Zero cross-attn sub-block over object_tokens inside "
            "EACH DiT layer. Pure architecture change; loss stays V0 "
            "(matches stage1p5_interaction_v0)."
        ),
        enable_per_block_obj_xattn=True,
    ),
    Stage1p5R33Variant(
        variant_id="stage1p5_r33_v2_xattn_moment",
        purpose=(
            "V2 = V1 + V7-A channel moment match (w=0.5) on the full "
            "31-D output. Tests whether the architecture change + the "
            "most general V7 loss compound."
        ),
        enable_per_block_obj_xattn=True,
        w_v7_moment_velocity=0.5,
    ),
    Stage1p5R33Variant(
        variant_id="stage1p5_r33_v3_xattn_v7full",
        purpose=(
            "V3 = V1 + full V7 V5 anti-bug stack (V7-A moment, V7-B "
            "phase unit-norm + angle, V7-C stance/walking BCE x4, V7-D "
            "frame-0 wrist consistency). Upper-bound test. If V3 closes "
            "wrist drift and V1 alone doesn't, V7 losses ONLY work in "
            "the presence of the new architecture; otherwise V1 alone "
            "is enough."
        ),
        enable_per_block_obj_xattn=True,
        # V7-A
        w_v7_moment_velocity=0.5,
        # V7-B
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


def _render_yaml(v: Stage1p5R33Variant, *, data_root: str) -> str:
    run_name = v.variant_id
    use_text = "true" if v.use_text else "false"
    use_min_snr = "true" if v.use_min_snr_weighting else "false"
    enable_xattn = "true" if v.enable_per_block_obj_xattn else "false"
    return f"""# Stage-1.5 R33 per-block obj_xattn ablation — {v.variant_id}
#
# Generated by scripts/stage_a_generator/round33_make_stage1p5_v0_configs.py.
# Per R32 V7 verdict + R32 Phase 1 audit + R32 verdict notes.
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
    # R33 — per-block AdaLN-Zero cross-attn over object_tokens
    # inside each DiT layer (DiT-XL pattern).
    enable_per_block_obj_xattn: {enable_xattn}
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


def _manifest_row(v: Stage1p5R33Variant) -> dict:
    return {
        "variant_id": v.variant_id,
        "group": "Stage1p5InteractionR33",
        "purpose": v.purpose,
        "train": True,
        "from_scratch": True,
        "architecture": {
            "d_model": v.d_model, "n_layers": v.n_layers,
            "n_heads": v.n_heads, "ff_mult": v.ff_mult,
            "dropout": v.dropout, "use_text": v.use_text,
            "enable_per_block_obj_xattn": v.enable_per_block_obj_xattn,
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

    manifest_json = analyses_dir / "round33_stage1p5_v0_manifest.json"
    manifest_md = analyses_dir / "round33_stage1p5_v0_manifest.md"
    md_lines: list[str] = [
        "# Round-33 V0 Stage-1.5 per-block obj_xattn manifest",
        "",
        "Per R32 V7 negative verdict "
        "(analyses/round32_v7_matrix_summary_20260530_130411.md) + R32 audit "
        "+ structural hypothesis (obj_tokens too weakly injected at the "
        "end-of-encoder xattn only).",
        "",
        f"FULL InterAct train set, from scratch, seed 42, "
        f"{SCHEDULE_NUM_EPOCHS} ep, bs={SCHEDULE_BATCH_SIZE} / "
        f"accum={SCHEDULE_ACCUM_STEPS} (2x 5080).",
        "",
        "## Variants",
        "",
        "| variant | per-block obj_xattn | V7-A moment | V7-B phase | V7-C stance | V7-D frame0 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for v in VARIANTS:
        xattn = "ON" if v.enable_per_block_obj_xattn else "off"
        v7c = "yes" if v.w_s4_stance == 2.0 else "no"
        md_lines.append(
            f"| `{v.variant_id}` | {xattn} | {v.w_v7_moment_velocity} | "
            f"{v.w_v7_phase_unit_norm} | {v7c} | {v.w_v7_c41_frame0_wrist} |"
        )
    md_lines.append("")
    md_lines.append("## Decision tree after results")
    md_lines.append("")
    md_lines.append("| outcome | next step |")
    md_lines.append("|---|---|")
    md_lines.append(
        "| V1 alone closes >= 5 cm wrist drift over V0 | "
        "Architecture is dominant; ship V1 + queue cascade training. |"
    )
    md_lines.append(
        "| V2 - V1 >= 2 cm | "
        "V7-A moment-match has real ROI ONLY in the presence of the new "
        "structure; ship V2. |"
    )
    md_lines.append(
        "| V3 - V1 >= 4 cm | "
        "Multiple V7 losses synergise with the structure change; ship V3. |"
    )
    md_lines.append(
        "| All R33 variants close < 2 cm | "
        "Architecture isn't the dominant issue either; pivot fully to "
        "cascade training (Stage-1.5 sees Stage-1 generated cond). |"
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
