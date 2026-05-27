"""Generate Round-28 configs (Group A/A1b/A2b, B0..B4, C1..C3) from a single base.

Each variant differs from `anchordiff_t0a3_full_oracle_hint_48clip.yaml`
only in a small block of: oracle hint enable/disable, body-action hint
mode, oracle injection mode, and the temporal-loss weight subset.

Usage:
    python scripts/stage_b_generator/round28_make_configs.py [--dry-run]
    python scripts/stage_b_generator/round28_make_configs.py \
        --best-injection-mode gated_input_m1

Writes:
    configs/training/anchordiff_r28_<variant>_48clip.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs" / "training"
BASE_CONFIG = CONFIG_DIR / "anchordiff_t0a3_full_oracle_hint_48clip.yaml"

V27_CKPT = "runs/training/stageB_anchordiff_v27_stage2_anchoraware_FULL_DATA/final.pt"
BALANCED_SUBSET_FILE = "analyses/round27_tier0_train_indices_48_balanced.json"
BODY_ACTION_SUBSET_FILE = "analyses/round28_body_action_train_indices_48.json"


# (variant_id, description, overrides)
# overrides is a dict applied to specific YAML lines via simple string patching.
# Format: section.field: value-literal-as-yaml-string.
VARIANTS: list[tuple[str, str, dict]] = [
    # ---------------- Group A: injection-mechanism ablation (interaction hint only)
    (
        "r28_a0_input_add",
        "Reproduce T0-A3: interaction hint only, input_add injection.",
        {
            "use_oracle_interaction_hint": "true",
            "oracle_hint_variant": "full",
            "use_oracle_interaction_hint_model": "true",
            "oracle_hint_dim": "13",
            "use_body_action_hint": "false",
            "use_body_action_hint_model": "false",
            "body_action_hint_dim": "0",
            "oracle_hint_injection_mode": "input_add",
            "best_label": "input_add",
            "temporal_weights": "all_zero",
        },
    ),
    (
        "r28_a1_gated_input",
        "Interaction hint only, gated_input injection, conservative gate bias -3.",
        {
            "use_oracle_interaction_hint": "true",
            "oracle_hint_variant": "full",
            "use_oracle_interaction_hint_model": "true",
            "oracle_hint_dim": "13",
            "use_body_action_hint": "false",
            "use_body_action_hint_model": "false",
            "body_action_hint_dim": "0",
            "oracle_hint_injection_mode": "gated_input",
            "oracle_hint_gate_bias_init": "-3.0",
            "best_label": "gated_input_m3",
            "temporal_weights": "all_zero",
        },
    ),
    (
        "r28_a1b_gated_input_open",
        "Interaction hint only, gated_input injection, fairer gate bias -1.",
        {
            "use_oracle_interaction_hint": "true",
            "oracle_hint_variant": "full",
            "use_oracle_interaction_hint_model": "true",
            "oracle_hint_dim": "13",
            "use_body_action_hint": "false",
            "use_body_action_hint_model": "false",
            "body_action_hint_dim": "0",
            "oracle_hint_injection_mode": "gated_input",
            "oracle_hint_gate_bias_init": "-1.0",
            "best_label": "gated_input_m1",
            "temporal_weights": "all_zero",
        },
    ),
    (
        "r28_a2_per_layer_adapter",
        "Interaction hint only, input_add + per_layer_adapter injection.",
        {
            "use_oracle_interaction_hint": "true",
            "oracle_hint_variant": "full",
            "use_oracle_interaction_hint_model": "true",
            "oracle_hint_dim": "13",
            "use_body_action_hint": "false",
            "use_body_action_hint_model": "false",
            "body_action_hint_dim": "0",
            "oracle_hint_injection_mode": "per_layer_adapter",
            "best_label": "per_layer_adapter",
            "temporal_weights": "all_zero",
        },
    ),
    (
        "r28_a2b_adapter_only",
        "Interaction hint only, PURE per-layer adapter injection (no input "
        "add). Isolates the adapter contribution from the input-add baseline "
        "so A2 vs A2b separates 'adapter helps on top of input_add' from "
        "'adapter alone is sufficient'.",
        {
            "use_oracle_interaction_hint": "true",
            "oracle_hint_variant": "full",
            "use_oracle_interaction_hint_model": "true",
            "oracle_hint_dim": "13",
            "use_body_action_hint": "false",
            "use_body_action_hint_model": "false",
            "body_action_hint_dim": "0",
            "oracle_hint_injection_mode": "adapter_only",
            "best_label": "adapter_only",
            "temporal_weights": "all_zero",
        },
    ),
    (
        "r28_a3_best_long",
        "Best Group A injection (set by --best-injection-mode): 1000 "
        "epoch oracle upper bound.",
        {
            "use_oracle_interaction_hint": "true",
            "oracle_hint_variant": "full",
            "use_oracle_interaction_hint_model": "true",
            "oracle_hint_dim": "13",
            "use_body_action_hint": "false",
            "use_body_action_hint_model": "false",
            "body_action_hint_dim": "0",
            "oracle_hint_injection_mode": "best",
            "temporal_weights": "all_zero",
            "num_epochs": "1000",
        },
    ),
    # ---------------- Group B: body-action hint family
    (
        "r28_b0_baseline",
        "v27 baseline reproduction on body-action subset: no hints.",
        {
            "use_oracle_interaction_hint": "false",
            "oracle_hint_variant": "full",
            "use_oracle_interaction_hint_model": "false",
            "oracle_hint_dim": "0",
            "use_body_action_hint": "false",
            "use_body_action_hint_model": "false",
            "body_action_hint_dim": "0",
            "oracle_hint_injection_mode": "input_add",
            "temporal_weights": "all_zero",
            "subset_kind": "body_action",
        },
    ),
    (
        "r28_b1_interaction_only",
        "Best Group A interaction injection only: does interaction hint "
        "alone help body-only actions?",
        {
            "use_oracle_interaction_hint": "true",
            "oracle_hint_variant": "full",
            "use_oracle_interaction_hint_model": "true",
            "oracle_hint_dim": "13",
            "use_body_action_hint": "false",
            "use_body_action_hint_model": "false",
            "body_action_hint_dim": "0",
            "oracle_hint_injection_mode": "best",
            "temporal_weights": "all_zero",
            "subset_kind": "body_action",
        },
    ),
    (
        "r28_b2_body_only_all_on",
        "Body-action hint only (24D, all_on mask). Upper bound on body-only "
        "actions.",
        {
            "use_oracle_interaction_hint": "false",
            "oracle_hint_variant": "full",
            "use_oracle_interaction_hint_model": "false",
            "oracle_hint_dim": "0",
            "use_body_action_hint": "true",
            "use_body_action_hint_model": "true",
            "body_action_hint_dim": "24",
            "body_action_hint_mask_mode": "all_on",
            "oracle_hint_injection_mode": "best",
            "temporal_weights": "all_zero",
            "subset_kind": "body_action",
        },
    ),
    (
        "r28_b3_body_only_energy",
        "Body-action hint only (24D, energy mask). Realistic sparse signal.",
        {
            "use_oracle_interaction_hint": "false",
            "oracle_hint_variant": "full",
            "use_oracle_interaction_hint_model": "false",
            "oracle_hint_dim": "0",
            "use_body_action_hint": "true",
            "use_body_action_hint_model": "true",
            "body_action_hint_dim": "24",
            "body_action_hint_mask_mode": "energy",
            "body_action_energy_threshold": "0.05",
            "oracle_hint_injection_mode": "best",
            "temporal_weights": "all_zero",
            "subset_kind": "body_action",
        },
    ),
    (
        "r28_b4_interaction_plus_body",
        "Interaction hint + body-action hint together: complementarity test.",
        {
            "use_oracle_interaction_hint": "true",
            "oracle_hint_variant": "full",
            "use_oracle_interaction_hint_model": "true",
            "oracle_hint_dim": "13",
            "use_body_action_hint": "true",
            "use_body_action_hint_model": "true",
            "body_action_hint_dim": "24",
            "body_action_hint_mask_mode": "all_on",
            "oracle_hint_injection_mode": "best",
            "temporal_weights": "all_zero",
            "subset_kind": "body_action",
        },
    ),
    # ---------------- Group C: gait losses + small consistency loss
    (
        "r28_c1_hints_plus_gait",
        "Best hints + gait losses only (NO contact temporal losses).",
        {
            "use_oracle_interaction_hint": "true",
            "oracle_hint_variant": "full",
            "use_oracle_interaction_hint_model": "true",
            "oracle_hint_dim": "13",
            "use_body_action_hint": "true",
            "use_body_action_hint_model": "true",
            "body_action_hint_dim": "24",
            "body_action_hint_mask_mode": "all_on",
            "oracle_hint_injection_mode": "best",
            "temporal_weights": "gait_only",
        },
    ),
    (
        "r28_c2_hints_plus_hint_consistency",
        "Best hints + small hint-contact consistency loss (weight 0.25).",
        {
            "use_oracle_interaction_hint": "true",
            "oracle_hint_variant": "full",
            "use_oracle_interaction_hint_model": "true",
            "oracle_hint_dim": "13",
            "use_body_action_hint": "true",
            "use_body_action_hint_model": "true",
            "body_action_hint_dim": "24",
            "body_action_hint_mask_mode": "all_on",
            "oracle_hint_injection_mode": "best",
            "temporal_weights": "hint_consistency_only",
        },
    ),
    (
        "r28_c3_hints_gait_consistency",
        "Best hints + gait losses + small hint-contact consistency (R28 candidate).",
        {
            "use_oracle_interaction_hint": "true",
            "oracle_hint_variant": "full",
            "use_oracle_interaction_hint_model": "true",
            "oracle_hint_dim": "13",
            "use_body_action_hint": "true",
            "use_body_action_hint_model": "true",
            "body_action_hint_dim": "24",
            "body_action_hint_mask_mode": "all_on",
            "oracle_hint_injection_mode": "best",
            "temporal_weights": "gait_plus_hint_consistency",
        },
    ),
]


TEMPORAL_WEIGHT_SETS = {
    "all_zero": {
        "contact_rel_offset_weight": 0.0,
        "contact_drift_weight": 0.0,
        "contact_tracking_weight": 0.0,
        "gait_both_airborne_weight": 0.0,
        "gait_stance_velocity_weight": 0.0,
        "hint_contact_consistency_weight": 0.0,
        "body_action_consistency_weight": 0.0,
    },
    "gait_only": {
        "contact_rel_offset_weight": 0.0,
        "contact_drift_weight": 0.0,
        "contact_tracking_weight": 0.0,
        "gait_both_airborne_weight": 1.0,
        "gait_stance_velocity_weight": 1.0,
        "hint_contact_consistency_weight": 0.0,
        "body_action_consistency_weight": 0.0,
    },
    "hint_consistency_only": {
        "contact_rel_offset_weight": 0.0,
        "contact_drift_weight": 0.0,
        "contact_tracking_weight": 0.0,
        "gait_both_airborne_weight": 0.0,
        "gait_stance_velocity_weight": 0.0,
        "hint_contact_consistency_weight": 0.25,
        "body_action_consistency_weight": 0.0,
    },
    "gait_plus_hint_consistency": {
        "contact_rel_offset_weight": 0.0,
        "contact_drift_weight": 0.0,
        "contact_tracking_weight": 0.0,
        "gait_both_airborne_weight": 1.0,
        "gait_stance_velocity_weight": 1.0,
        "hint_contact_consistency_weight": 0.25,
        "body_action_consistency_weight": 0.0,
    },
}


def _render_temporal_block(weights: dict[str, float]) -> str:
    L = ["  temporal_interaction:"]
    for k, v in weights.items():
        L.append(f"    {k}: {v}")
    L.append("    contact_threshold: 0.5")
    L.append("    contact_rel_clamp_m: 2.0")
    L.append("    tracking_margin_m: 0.03")
    L.append("    tracking_min_obj_disp_m: 0.05")
    L.append("    floor_quantile: 0.05")
    L.append("    grounded_threshold_above_floor_m: 0.10")
    L.append("    grounded_softness_m: 0.03")
    return "\n".join(L)


def _bool(b: str) -> str:
    return "true" if b.lower() == "true" else "false"


def _resolve_best_injection(best: str) -> tuple[str, str]:
    """Return (model injection mode, gate bias) for an A-group winner label."""
    if best == "input_add":
        return "input_add", "-3.0"
    if best in {"gated_input", "gated_input_m3"}:
        return "gated_input", "-3.0"
    if best == "gated_input_m1":
        return "gated_input", "-1.0"
    if best == "per_layer_adapter":
        return "per_layer_adapter", "-3.0"
    if best == "adapter_only":
        return "adapter_only", "-3.0"
    raise ValueError(f"unknown best injection label: {best}")


def _render_config(
    variant_id: str,
    description: str,
    ov: dict,
    *,
    best_injection_mode: str,
    balanced_subset_file: str,
    body_action_subset_file: str,
) -> str:
    """Build a complete YAML config for one R28 variant from scratch.

    Mirrors the t0a3 layout so it loads via the same trainer."""
    weights = TEMPORAL_WEIGHT_SETS[ov["temporal_weights"]]
    temporal_block = _render_temporal_block(weights)
    num_epochs = ov.get("num_epochs", "300")
    surface_temporal_aux = (
        "true" if (
            weights["gait_both_airborne_weight"] > 0
            or weights["gait_stance_velocity_weight"] > 0
        ) else "false"
    )
    run_name = f"stageB_anchordiff_{variant_id}_48clip"
    body_mask_mode = ov.get("body_action_hint_mask_mode", "all_on")
    body_energy_thr = ov.get("body_action_energy_threshold", "0.05")
    injection_mode = str(ov["oracle_hint_injection_mode"])
    gate_bias = str(ov.get("oracle_hint_gate_bias_init", "-3.0"))
    if injection_mode == "best":
        injection_mode, gate_bias = _resolve_best_injection(str(best_injection_mode))
    subset_file = (
        str(body_action_subset_file)
        if ov.get("subset_kind", "balanced") == "body_action"
        else str(balanced_subset_file)
    )

    return f"""# Round-28 {variant_id}: {description}
#
# Generated by scripts/stage_b_generator/round28_make_configs.py.
# Per analyses/round28_claude_code_stage2_oracle_interface_prompt.md.
#
# Warm-starts from v27 final.pt. New parameter tensors
# (oracle_hint_proj.*, body_action_hint_proj.*, *_gate.*, *_adapters.*)
# are zero-init so step-0 forward is bit-exact equal to v27 baseline;
# `partial_init_allow_shape_mismatch: true` lets the ckpt load skip the
# new keys.

model:
  cfg_drop_prob: 0.15

  diffusion:
    num_steps: 1000
    schedule: "cosine"
    prediction_target: "x0"

  denoiser:
    motion_dim: 135
    object_traj_dim: 9
    init_pose_dim: 66
    text_dim: 512
    object_token_dim: 256
    object_num_tokens: 128
    d_model: 512
    n_layers: 8
    n_heads: 4
    ff_mult: 4
    dropout: 0.1
    stage1_coarse_dim: 23
    # Round-28 oracle hint branches.
    use_oracle_interaction_hint: {_bool(ov["use_oracle_interaction_hint_model"])}
    oracle_hint_dim: {ov["oracle_hint_dim"]}
    use_body_action_hint: {_bool(ov["use_body_action_hint_model"])}
    body_action_hint_dim: {ov["body_action_hint_dim"]}
    oracle_hint_injection_mode: "{injection_mode}"
    oracle_hint_gate_bias_init: {gate_bias}
    separate_hint_branches: true
    zero_init_hint_adapters: true

  object_encoder:
    num_input_points: 1024
    num_output_tokens: 128
    feature_dim: 256

  text_encoder:
    clip_version: "ViT-B/32"
    download_root: "cache/clip"

data:
  datasets:
    - name: "chairs"
      root: "E:/Project/Datasets/InterAct/piano_official_process_4/chairs"
    - name: "imhd"
      root: "E:/Project/Datasets/InterAct/piano_official_process_4/imhd"
    - name: "neuraldome"
      root: "E:/Project/Datasets/InterAct/piano_official_process_4/neuraldome"
    - name: "omomo_correct_v2"
      root: "E:/Project/Datasets/InterAct/piano_official_process_4/omomo_correct_v2"
  pseudo_label_dir: null
  pseudo_label_subdir: "pseudo_labels/v18_h10_f05_pelvis20_official_semantic_marker"
  max_seq_length: 196
  support_collapse_hand_support: true
  motion_representation: "smpl_pose_135_plan"
  force_world_frame: true
  subsample_n_per_object: null
  overfit_n_clips: 0
  stage1_coarse_cache_root: "cache/stage1_coarse_v1_full"
  subset_indices_file: "{subset_file}"
  # Round-28 oracle hint switches (dataset side).
  use_oracle_interaction_hint: {_bool(ov["use_oracle_interaction_hint"])}
  oracle_hint_variant: "{ov["oracle_hint_variant"]}"
  oracle_hint_fps: 20.0
  use_body_action_hint: {_bool(ov["use_body_action_hint"])}
  body_action_hint_mask_mode: "{body_mask_mode}"
  body_action_energy_threshold: {body_energy_thr}
  surface_temporal_aux_fields: {surface_temporal_aux}

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
  init_checkpoint: "{V27_CKPT}"
  partial_init_allow_shape_mismatch: true
  batch_size: 8
  num_epochs: {num_epochs}
  num_workers: 4
  seed: 42
  stage1_coarse_noise_std: 0.0
  optimizer:
    name: "adamw"
    lr: 5.0e-5
    weight_decay: 0.01
    betas: [0.9, 0.999]
  scheduler:
    name: "cosine"
    warmup_steps: 50
  gradient_accumulation_steps: 4
  max_grad_norm: 1.0
  mixed_precision: "bf16"
  val_on_train_subset: true
  val_every_epochs: 50
  val_best_key: "loss_anchor_joint_pos"

loss:
  anchor_weight: 0.0
  contact_threshold: 0.5

  stable_root_vel_weight: 0.5
  stable_root_acc_weight: 0.25
  stable_support_erode: 4
  stable_local_vel_cm_weight: 0.05
  stable_local_speed_moment_weight: 0.02

  hand_endpoint_weight: 2.0
  foot_endpoint_weight: 2.0
  pos_loss_weight: 5.0

  anchor_joint_pos_weight: 10.0
  anchor_joint_vel_weight: 2.0
  anchor_joint_part_weights: [2.0, 2.0, 0.0, 0.0, 0.5]

  use_min_snr_weighting: true
  min_snr_gamma: 5.0
  motion_feature_weights:
    root_rot_vel: 1.0
    root_lin_vel: 1.0
    root_height_y: 1.0
    joint_pos_local: 1.0
    joint_rot_6d: 1.0
    joint_velocity: 1.0
    foot_contact: 1.0
  dynamic_metric:
    enabled: false
  motion_geometric:
    enabled: false
  world_joint_velocity_weight: 1.0
  fk_consistency_weight: 0.0

{temporal_block}

logging:
  project: "piano"
  run_name: "{run_name}"
  log_every_n_steps: 10
  save_every_n_epochs: 50

output_dir: "runs/training/{run_name}"
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print which files would be written without touching disk.")
    parser.add_argument(
        "--best-injection-mode",
        choices=[
            "input_add",
            "gated_input",
            "gated_input_m3",
            "gated_input_m1",
            "per_layer_adapter",
            "adapter_only",
        ],
        default="per_layer_adapter",
        help=(
            "Winner label to use for A3/B/C configs after A-group decides "
            "the best branch. `gated_input` is an alias for gated_input_m3; "
            "use gated_input_m1 if A1b wins."
        ),
    )
    parser.add_argument(
        "--balanced-subset-file",
        default=BALANCED_SUBSET_FILE,
        help="Subset JSON for A/C contact+gait-oriented variants.",
    )
    parser.add_argument(
        "--body-action-subset-file",
        default=BODY_ACTION_SUBSET_FILE,
        help="Subset JSON for B body-action variants.",
    )
    args = parser.parse_args()

    if not BASE_CONFIG.exists():
        raise FileNotFoundError(f"base config missing: {BASE_CONFIG}")

    n = 0
    for variant_id, desc, ov in VARIANTS:
        out_path = CONFIG_DIR / f"anchordiff_{variant_id}_48clip.yaml"
        content = _render_config(
            variant_id,
            desc,
            ov,
            best_injection_mode=str(args.best_injection_mode),
            balanced_subset_file=str(args.balanced_subset_file),
            body_action_subset_file=str(args.body_action_subset_file),
        )
        if args.dry_run:
            print(f"DRY-RUN would write: {out_path}  ({len(content)} bytes)")
        else:
            out_path.write_text(content, encoding="utf-8")
            print(f"wrote {out_path}  ({len(content)} bytes)")
        n += 1
    print(f"\n{n} configs {'identified' if args.dry_run else 'written'}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
