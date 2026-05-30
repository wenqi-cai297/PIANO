"""Round-31 V8 — Stage-1 wrist FK + init_pose ablation matrix generator.

Baseline: V7 V5 (full anti-collapse stack), which closed 43% of the
oracle gap mostly on gait/pelvis but did not move wrist drift (still
~35 cm). V8 adds two new mechanisms targeted at wrist:

  V8-W : extend V7-C's FK target chain from {neck/head/shoulder} to
         include {elbow, wrist} — gives gradient at the chain end.
         Parameterised via:
           - contact_mask_mode ∈ {off, reweight, hard}
           - joint_weights     (PB1 hand_endpoint x2 style)
           - add_velocity      (PB1 anchor_joint_vel style)

  V8-F : frame-0 anchor injection. Phase-1 audit showed Stage-1
         violates the frame-0 invariant (rms_at_t0 = 9.8 cm). PB1 has
         the equivalent (init_pose=GT joints_22 frame 0); Stage-1 has
         none. Two modes:
           F1 : motion_135[:, 0, :] raw (135-D)
           F2 : 12 rot6d + 2 heights z-scored (14-D), with optional
                frame-0 consistency loss forcing Stage-1's t=0 output
                on those channels to match.

Seven variants in this matrix:

    V8.0 = V5 control            (W0 + F0; reproduces V7 V5 = 17.52 cm)
    V8.1 = V5 + W1               (wrist FK extended, plain)
    V8.2 = V5 + W2               (wrist FK extended + reweight + heavy + vel)
    V8.3 = V5 + F2               (frame-0 14-D init_pose + consistency loss)
    V8.4 = V5 + W1 + F2          (wrist W1 + frame-0 F2)
    V8.5 = V5 + W2 + F2          (full stack: PB1-style wrist + frame-0 F2)
    V8.6 = V5 + W2 + F1          (full stack but F1 = 135-D init_pose)

Outputs:
    configs/training/stage1_v8_v{0..6}_*.yaml
    analyses/round31_stage1_v8_manifest.json
    analyses/round31_stage1_v8_manifest.md

Usage:
    python scripts/stage_a_generator/round31_make_stage1_v8_configs.py
    python scripts/stage_a_generator/round31_make_stage1_v8_configs.py --dry-run
    python scripts/stage_a_generator/round31_make_stage1_v8_configs.py \\
        --data-root /media/8TB_data/Cai/datasets/InterAct/piano_official_process_4
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
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

# V8-W default extended target set (V7-C 4 joints + arm chain to wrist).
TARGET_8 = (12, 15, 16, 17, 18, 19, 20, 21)  # neck head L_sh R_sh L_el R_el L_wr R_wr
# V8-W2 joint weights: 1 everywhere, 2x on wrist (PB1 hand_endpoint_weight).
JOINT_WEIGHTS_W2 = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0)


@dataclass(slots=True)
class Stage1V8Variant:
    variant_id: str
    purpose: str

    # Architecture (fixed across V8).
    d_model: int = ARCH_D_MODEL
    n_layers: int = ARCH_N_LAYERS
    n_heads: int = ARCH_N_HEADS
    ff_mult: int = ARCH_FF_MULT
    dropout: float = ARCH_DROPOUT
    use_text: bool = True

    # V8 architectural switch: init_pose injection mode.
    # 0 = OFF, 14 = F2 (z-scored 14-D), 135 = F1 (raw 135-D).
    init_pose_dim: int = 0

    # V0 baseline loss (always on).
    w_x0: float = 1.0
    w_vel: float = 1.0
    w_yaw_smooth: float = 0.0

    # V2 ablation losses — OFF in V8.
    w_rot6d_ortho: float = 0.0
    w_fk_pos: float = 0.0
    w_height_fk: float = 0.0
    w_self_consistency: float = 0.0
    vel_rot6d_weight: float = 1.0

    # V7 anti-collapse stack — INHERITED from V5 (LOCKED across V8).
    w_moment_velocity: float = 0.5
    w_moment_value: float = 0.0
    w_yaw_aggregate: float = 2.0
    w_fk_pos_cm: float = 0.2
    fk_pos_cm_beta: float = 1.0

    # V8 — wrist FK supervision.
    w_wrist_fk_pos: float = 0.0
    wrist_fk_target_joints: tuple[int, ...] | None = None
    wrist_fk_joint_weights: tuple[float, ...] | None = None
    wrist_fk_contact_mode: str = "off"
    wrist_fk_contact_weight: float = 4.0
    wrist_fk_add_velocity: bool = False
    wrist_fk_velocity_weight: float = 0.5
    wrist_fk_beta_cm: float = 1.0

    # V8 — frame-0 consistency.
    w_init_pose_consistency: float = 0.0

    use_min_snr_weighting: bool = True
    min_snr_gamma: float = 5.0

    val_best_key: str = "mse_x0"


# ─── R31 V8 ablation matrix — 7 variants ─────────────────────────────────

VARIANTS: list[Stage1V8Variant] = [
    Stage1V8Variant(
        variant_id="stage1_v8_v0_v5_control",
        purpose=(
            "V8.0 = V7 V5 control (W0 + F0). Reproduces stage1_v7_v5_full "
            "with seed 42; lets us measure stochastic noise floor for V8."
        ),
    ),
    Stage1V8Variant(
        variant_id="stage1_v8_v1_wrist_plain",
        purpose=(
            "V8.1 = V5 + W1. Extends V7-C's FK target chain to "
            "(L/R wrist + L/R elbow). w_wrist_fk_pos=0.2 same scale as "
            "V7-C. No contact masking, no wrist reweighting, no velocity. "
            "Tests whether adding wrist supervision alone is enough."
        ),
        w_wrist_fk_pos=0.2,
        wrist_fk_target_joints=TARGET_8,
    ),
    Stage1V8Variant(
        variant_id="stage1_v8_v2_wrist_pb1style",
        purpose=(
            "V8.2 = V5 + W2. Full PB1 anchor style: target=wrist chain, "
            "wrist x2 weighting (hand_endpoint), contact-reweight 4x on "
            "hand-contact frames, plus velocity matching. Tests max-out "
            "wrist supervision under V5 anti-collapse."
        ),
        w_wrist_fk_pos=0.2,
        wrist_fk_target_joints=TARGET_8,
        wrist_fk_joint_weights=JOINT_WEIGHTS_W2,
        wrist_fk_contact_mode="reweight",
        wrist_fk_contact_weight=4.0,
        wrist_fk_add_velocity=True,
        wrist_fk_velocity_weight=0.5,
    ),
    Stage1V8Variant(
        variant_id="stage1_v8_v3_initpose_f2",
        purpose=(
            "V8.3 = V5 + F2. Inject 14-D frame-0 (12 rot6d + 2 heights) "
            "z-scored via init_pose_proj zero-init Linear + frame-0 "
            "consistency loss forcing pred[t=0] on those channels to "
            "match. Tests whether frame-0 anchor alone moves the needle."
        ),
        init_pose_dim=14,
        w_init_pose_consistency=1.0,
    ),
    Stage1V8Variant(
        variant_id="stage1_v8_v4_wrist_plain_initpose_f2",
        purpose=(
            "V8.4 = V5 + W1 + F2. Combine plain wrist FK with F2 frame-0. "
            "Tests additivity of the two mechanisms when each is at its "
            "simplest configuration."
        ),
        w_wrist_fk_pos=0.2,
        wrist_fk_target_joints=TARGET_8,
        init_pose_dim=14,
        w_init_pose_consistency=1.0,
    ),
    Stage1V8Variant(
        variant_id="stage1_v8_v5_full_f2",
        purpose=(
            "V8.5 = V5 + W2 + F2. Full PB1-style wrist + F2 frame-0. "
            "Primary ship candidate: matches PB1's L_pos + endpoint + "
            "init_pose triad while inheriting V5's anti-collapse stack."
        ),
        w_wrist_fk_pos=0.2,
        wrist_fk_target_joints=TARGET_8,
        wrist_fk_joint_weights=JOINT_WEIGHTS_W2,
        wrist_fk_contact_mode="reweight",
        wrist_fk_contact_weight=4.0,
        wrist_fk_add_velocity=True,
        wrist_fk_velocity_weight=0.5,
        init_pose_dim=14,
        w_init_pose_consistency=1.0,
    ),
    Stage1V8Variant(
        variant_id="stage1_v8_v6_full_f1",
        purpose=(
            "V8.6 = V5 + W2 + F1. Same as V8.5 but F1 (full 135-D "
            "motion_135 frame-0 slice) instead of F2 (14-D). Tests "
            "whether the extra dims help or hurt. F1 has no frame-0 "
            "consistency loss (no canonical target slice to match)."
        ),
        w_wrist_fk_pos=0.2,
        wrist_fk_target_joints=TARGET_8,
        wrist_fk_joint_weights=JOINT_WEIGHTS_W2,
        wrist_fk_contact_mode="reweight",
        wrist_fk_contact_weight=4.0,
        wrist_fk_add_velocity=True,
        wrist_fk_velocity_weight=0.5,
        init_pose_dim=135,
        # F1 has no canonical 14-D target → leave consistency loss off.
    ),
]


def _render_datasets_block(data_root: str) -> str:
    stripped = data_root.rstrip("/").rstrip("\\")
    lines: list[str] = []
    for sub in DATASET_SUBSET_NAMES:
        lines.append(f'    - name: "{sub}"')
        lines.append(f'      root: "{stripped}/{sub}"')
    return "\n".join(lines)


def _yaml_list(values: tuple | None, fmt: str) -> str:
    if values is None or len(values) == 0:
        return "[]"
    return "[" + ", ".join(fmt.format(v) for v in values) + "]"


def _render_yaml(v: Stage1V8Variant, *, data_root: str) -> str:
    run_name = v.variant_id
    use_text = "true" if v.use_text else "false"
    use_min_snr = "true" if v.use_min_snr_weighting else "false"
    add_vel = "true" if v.wrist_fk_add_velocity else "false"
    targets_yaml = _yaml_list(v.wrist_fk_target_joints, "{}")
    weights_yaml = _yaml_list(v.wrist_fk_joint_weights, "{:.3f}")
    return f"""# Stage-1 (Trajectory) V8 wrist+frame-0 ablation — {v.variant_id}
#
# Generated by scripts/stage_a_generator/round31_make_stage1_v8_configs.py.
# Baseline: V7 V5 (anti-collapse stack inherited).
# Phase 1 audit:
#   analyses/round31_phase1_dyn_audit_20260530_043948/audit_report.md
# V7 result:
#   analyses/round31_v7_matrix_summary_20260530_060934.md
#
# Output: 23-D stage1_coarse (Stage-2 PB1 cond contract). FULL DATA.
# {SCHEDULE_NUM_EPOCHS} ep / val_every=5 / save_every=10 / warmup=500.
# Schedule: bs={SCHEDULE_BATCH_SIZE} accum={SCHEDULE_ACCUM_STEPS} (2x 5080).

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
    # R31 V8 — init_pose injection (0 = OFF, 14 = F2, 135 = F1).
    init_pose_dim: {v.init_pose_dim}
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
  # R31 V2 ablation losses (OFF in V8):
  w_rot6d_ortho: {v.w_rot6d_ortho}
  w_fk_pos: {v.w_fk_pos}
  w_height_fk: {v.w_height_fk}
  w_self_consistency: {v.w_self_consistency}
  vel_rot6d_weight: {v.vel_rot6d_weight}
  # R31 V7 anti-collapse stack (LOCKED to V5 values across V8):
  w_moment_velocity: {v.w_moment_velocity}
  w_moment_value: {v.w_moment_value}
  w_yaw_aggregate: {v.w_yaw_aggregate}
  w_fk_pos_cm: {v.w_fk_pos_cm}
  fk_pos_cm_beta: {v.fk_pos_cm_beta}
  # R31 V8 — wrist FK supervision (extends V7-C chain).
  w_wrist_fk_pos: {v.w_wrist_fk_pos}
  wrist_fk_target_joints: {targets_yaml}
  wrist_fk_joint_weights: {weights_yaml}
  wrist_fk_contact_mode: "{v.wrist_fk_contact_mode}"
  wrist_fk_contact_weight: {v.wrist_fk_contact_weight}
  wrist_fk_add_velocity: {add_vel}
  wrist_fk_velocity_weight: {v.wrist_fk_velocity_weight}
  wrist_fk_beta_cm: {v.wrist_fk_beta_cm}
  # R31 V8 — frame-0 consistency loss (active under F2 mode).
  w_init_pose_consistency: {v.w_init_pose_consistency}
  use_min_snr_weighting: {use_min_snr}
  min_snr_gamma: {v.min_snr_gamma}

logging:
  project: "piano"
  run_name: "{run_name}"
  log_every_n_steps: 50
  save_every_n_epochs: 10

output_dir: "runs/training/{run_name}"
"""


def _manifest_row(v: Stage1V8Variant) -> dict:
    return {
        "variant_id": v.variant_id,
        "group": "Stage1TrajectoryV8",
        "purpose": v.purpose,
        "train": True,
        "from_scratch": True,
        "architecture": {
            "d_model": v.d_model, "n_layers": v.n_layers,
            "n_heads": v.n_heads, "ff_mult": v.ff_mult,
            "dropout": v.dropout, "use_text": v.use_text,
            "init_pose_dim": v.init_pose_dim,
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
            "w_moment_velocity": v.w_moment_velocity,
            "w_yaw_aggregate": v.w_yaw_aggregate,
            "w_fk_pos_cm": v.w_fk_pos_cm,
            "w_wrist_fk_pos": v.w_wrist_fk_pos,
            "wrist_fk_target_joints": list(v.wrist_fk_target_joints or []),
            "wrist_fk_joint_weights": list(v.wrist_fk_joint_weights or []),
            "wrist_fk_contact_mode": v.wrist_fk_contact_mode,
            "wrist_fk_contact_weight": v.wrist_fk_contact_weight,
            "wrist_fk_add_velocity": v.wrist_fk_add_velocity,
            "wrist_fk_velocity_weight": v.wrist_fk_velocity_weight,
            "w_init_pose_consistency": v.w_init_pose_consistency,
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

    manifest_json = analyses_dir / "round31_stage1_v8_manifest.json"
    manifest_md = analyses_dir / "round31_stage1_v8_manifest.md"
    md_lines: list[str] = [
        "# Round-31 V8 Stage-1 wrist FK + frame-0 anchor manifest",
        "",
        "Baseline = V7 V5 (anti-collapse stack inherited). Phase 1 audit "
        "+ V7 result drove the two new mechanisms tested here.",
        "",
        "## Variants (R31 V8 ablation matrix)",
        "",
        "| variant | wrist_fk | targets | contact | wrist_w | vel | init_pose | f0_cons |",
        "|---|---:|---|---|---|---|---|---:|",
    ]
    for v in VARIANTS:
        contact = v.wrist_fk_contact_mode
        wrist_w = "yes" if v.wrist_fk_joint_weights else "no"
        vel = "yes" if v.wrist_fk_add_velocity else "no"
        ip = {0: "off", 14: "F2", 135: "F1"}.get(v.init_pose_dim, "?")
        n_targets = len(v.wrist_fk_target_joints or ())
        md_lines.append(
            f"| `{v.variant_id}` | {v.w_wrist_fk_pos} | {n_targets}-joint | "
            f"{contact} | {wrist_w} | {vel} | {ip} | {v.w_init_pose_consistency} |"
        )
    md_lines.append("")
    md_lines.append("## Decision tree")
    md_lines.append("")
    md_lines.append("| outcome | next step |")
    md_lines.append("|---|---|")
    md_lines.append(
        "| V8.1 closes >= 5 cm wrist drift over V5 | wrist FK extension alone is dominant; "
        "tune weights in V9 |"
    )
    md_lines.append(
        "| V8.3 closes >= 3 cm but V8.1 < 2 cm | frame-0 anchor is the missing piece; "
        "ship V8.3 + queue wrist for V9 |"
    )
    md_lines.append(
        "| V8.5 - V8.4 > 2 cm | contact mask + wrist heavy weighting + vel are load-bearing; "
        "ship V8.5 |"
    )
    md_lines.append(
        "| V8.6 ~ V8.5 | F1 not better than F2; prefer F2 for simplicity |"
    )
    md_lines.append(
        "| V8.6 significantly better than V8.5 | 135-D init_pose carries useful extra info; "
        "ship V8.6 |"
    )
    md_lines.append(
        "| All V8.* close < 2 cm wrist drift over V5 | wrist failure is NOT a "
        "Stage-1 supervision problem; pivot to PB1 conditioning augmentation |"
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
