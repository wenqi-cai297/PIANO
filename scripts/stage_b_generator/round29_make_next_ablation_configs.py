"""Generate Round-29 next-baseline ablation configs (B0/B1/G1/G2/H1).

Per ``analyses/2026-05-27_round29_next_ablation_execution_prompt_for_claude_code.md``:
five new train variants on top of the empirical baseline R0
(``r29_ft_r0_clean_a3_baseline``) to answer three questions:

  1. Does R0's win depend on all R29 condition families, or mostly C41?
  2. Can phase-free gait losses fix R2's degeneracy without GT phase locking?
  3. Is current I3/I5 interaction too weak, or is the contact problem deeper
     than condition content?

Five variants:

  B0  r29_nb_b0_no_r29_cond                  No R29 C/I/S/B injection. Stage-1
                                              Coarse-v1 23D only. Tests whether
                                              R29 condition families matter.
  B1  r29_nb_b1_c41_only                     Only C41 extra (dim=18). I/S/B
                                              dims=0. Tests if C41 alone
                                              recovers R0.
  G1  r29_nb_g1_phasefree_gait_fixed         R0 cond + new phase-free gait
                                              losses (soft-stance velocity,
                                              transition rate, duty cycle,
                                              both-state match). Avoids R2's
                                              one-foot-airborne degeneracy.
  G2  r29_nb_g2_strong_s4_oracle             R0 cond + strong S4 execution
                                              losses (BCE=0.30, footstep=0.40,
                                              swing/airborne/stance_vel=0.10).
  H1  r29_nb_h1_r0_plus_oracle_full_hint     R0 cond + full oracle interaction
                                              hint (dim=13, input_add).

R0 itself is referenced in the manifest with ``"train": false`` so the
summarizer can read its existing R29-FT diagnostic stats as the comparison
baseline. We do not retrain R0 unless the user explicitly asks for it.

All five new variants: FULL InterAct train set (no subset_indices_file),
80 ep, heldout val (val_every=5), save_every=10, warmup=250,
stage1_coarse_noise_std=0.05. Schedule: bs=32 / accum=1 (2× 5080, intended).

Outputs:
    configs/training/anchordiff_r29_nb_b0_no_r29_cond.yaml
    configs/training/anchordiff_r29_nb_b1_c41_only.yaml
    configs/training/anchordiff_r29_nb_g1_phasefree_gait_fixed.yaml
    configs/training/anchordiff_r29_nb_g2_strong_s4_oracle.yaml
    configs/training/anchordiff_r29_nb_h1_r0_plus_oracle_full_hint.yaml
    analyses/round29_next_ablation_manifest.json
    analyses/round29_next_ablation_manifest.md

Usage:
    python scripts/stage_b_generator/round29_make_next_ablation_configs.py
    python scripts/stage_b_generator/round29_make_next_ablation_configs.py --dry-run
    python scripts/stage_b_generator/round29_make_next_ablation_configs.py \\
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


# --------------------------------------------------------------------------- #
# Schedule (per prompt §"Non-Negotiable Experiment Rules"): 2× 5080 intended,
# batch_size=32, gradient_accumulation_steps=1, num_epochs=80.
# --------------------------------------------------------------------------- #

SCHEDULE_BATCH_SIZE = 32
SCHEDULE_ACCUM_STEPS = 1
SCHEDULE_NUM_EPOCHS = 80

# R0 reference (existing R29-FT) — used by the summarizer as the comparison
# baseline; NOT retrained by this matrix.
R0_REFERENCE_VARIANT_ID = "r29_ft_r0_clean_a3_baseline"
R0_REFERENCE_CONFIG_PATH = (
    "configs/training/anchordiff_r29_ft_r0_clean_a3_baseline.yaml"
)
R0_REFERENCE_OUTPUT_DIR = (
    "runs/training/stageB_anchordiff_r29_ft_r0_clean_a3_baseline"
)


@dataclass(slots=True)
class NextAblationVariant:
    variant_id: str
    purpose: str
    decision_question: str

    # Round-29 condition injection axis.
    use_round29_cond_injection: bool = True
    r29_injection_mode: str = "input_add_adapter"

    # Round-29 condition families (R0 default: C41 + I3 + S4 + B4).
    r29_coarse_variant: str = "C41-current"
    r29_coarse_extra_dim: int = 18
    r29_interaction_variant: str = "I3-contact-offset-masked"
    r29_interaction_dim: int = 8
    r29_support_variant: str = "S4-S1-phase-footstep"
    r29_support_dim: int = 13
    r29_body_variant: str = "B4-lowpass-residual-mask"
    r29_body_refine_dim: int = 20

    # Round-28 oracle interaction hint axis (H1 only).
    use_oracle_interaction_hint_data: bool = False
    use_oracle_interaction_hint_model: bool = False
    oracle_hint_variant: str = ""
    oracle_hint_dim: int = 0
    oracle_hint_injection_mode: str = "input_add"

    # Absolute-GT auxiliary loss weights — all five inherit R0's baseline.
    pos_loss_weight: float = 5.0
    hand_endpoint_weight: float = 2.0
    foot_endpoint_weight: float = 2.0
    anchor_joint_pos_weight: float = 10.0
    anchor_joint_vel_weight: float = 2.0
    world_joint_velocity_weight: float = 1.0

    # R29 existing condition-consistency / support / swing weights — only
    # non-zero on G2 (per prompt §G2 strong-S4).
    r29_support_both_airborne_weight: float = 0.0
    r29_support_stance_velocity_weight: float = 0.0
    r29_swing_clearance_weight: float = 0.0
    r29_swing_clearance_m: float = 0.05

    # R3 exact-S4 execution (only non-zero on G2 per prompt §G2).
    r29_s4_stance_bce_weight: float = 0.0
    r29_s4_footstep_target_weight: float = 0.0

    # G1 phase-free gait losses (only non-zero on G1).
    r29_gait_soft_stance_velocity_weight: float = 0.0
    r29_gait_transition_rate_weight: float = 0.0
    r29_gait_duty_cycle_weight: float = 0.0
    r29_gait_both_state_match_weight: float = 0.0
    r29_gait_ankle_smooth_weight: float = 0.0
    r29_gait_antiphase_corr_weight: float = 0.0

    val_best_key: str = "loss_anchor_joint_pos"
    diagnostics: tuple[str, ...] = field(default_factory=lambda: (
        "sustained_contact", "gait", "body_action",
    ))


# --------------------------------------------------------------------------- #
# Five variants per prompt §"Required Variant Matrix".
# --------------------------------------------------------------------------- #

VARIANTS: list[NextAblationVariant] = [
    NextAblationVariant(
        variant_id="r29_nb_b0_no_r29_cond",
        purpose=(
            "B0: no R29 C/I/S/B injection. Stage-1 Coarse-v1 23D only. "
            "Tests whether R29 condition families are the source of R0's "
            "8.19 cm result or whether the schedule alone explains it."
        ),
        decision_question=(
            "If B0 ≈ R0, R29 C/I/S/B content is not load-bearing. "
            "If B0 << R0, at least one R29 condition family is load-bearing."
        ),
        use_round29_cond_injection=False,
        r29_coarse_variant="C23",
        r29_coarse_extra_dim=0,
        r29_interaction_variant="I0",
        r29_interaction_dim=0,
        r29_support_variant="S0",
        r29_support_dim=0,
        r29_body_variant="B0",
        r29_body_refine_dim=0,
    ),
    NextAblationVariant(
        variant_id="r29_nb_b1_c41_only",
        purpose=(
            "B1: only C41 extra is active (dim=18). I/S/B dims=0. Tests "
            "whether C41 alone recovers most of R0 — gates Stage-1.5 "
            "C41-prediction priority."
        ),
        decision_question=(
            "If B1 ≈ R0 and B0 << R0, Stage-1.5 should primarily predict "
            "C41-like key-joint deltas. If B1 << R0, I/S/B also carry "
            "useful information."
        ),
        r29_coarse_variant="C41-current",
        r29_coarse_extra_dim=18,
        r29_interaction_variant="I0",
        r29_interaction_dim=0,
        r29_support_variant="S0",
        r29_support_dim=0,
        r29_body_variant="B0",
        r29_body_refine_dim=0,
    ),
    NextAblationVariant(
        variant_id="r29_nb_g1_phasefree_gait_fixed",
        purpose=(
            "G1: R0 cond + phase-free gait losses (soft-stance velocity, "
            "transition rate, duty cycle, both-state match). Avoids R2's "
            "height-only loophole by combining ankle height with horizontal "
            "speed; matches aggregate stats without per-frame L/R alignment."
        ),
        decision_question=(
            "If G1 improves gait (L_R_corr, step_period_rate) without "
            "degeneration (frac_both_swing < 0.70) and without contact/body "
            "regression, phase-free behavior gait is the right next mainline. "
            "Compare to G2 to decide vs explicit S4 execution."
        ),
        # G1 recommended weights (prompt §G1).
        r29_gait_soft_stance_velocity_weight=0.05,
        r29_gait_transition_rate_weight=0.20,
        r29_gait_duty_cycle_weight=0.10,
        r29_gait_both_state_match_weight=0.10,
        r29_gait_ankle_smooth_weight=0.02,
        r29_gait_antiphase_corr_weight=0.02,
    ),
    NextAblationVariant(
        variant_id="r29_nb_g2_strong_s4_oracle",
        purpose=(
            "G2: R0 cond + strong S4 execution losses (existing R3 losses at "
            "stronger weights). Stance BCE=0.30, footstep target=0.40, "
            "both_airborne + stance_velocity + swing_clearance=0.10 each."
        ),
        decision_question=(
            "If G2 beats G1 on gait without contact/body regression, "
            "Stage-1.5 should output an explicit gait/phase/footstep "
            "schedule. If G1 matches or beats G2, prefer phase-free."
        ),
        # G2 strong S4 weights (prompt §G2).
        r29_support_both_airborne_weight=0.10,
        r29_support_stance_velocity_weight=0.10,
        r29_swing_clearance_weight=0.10,
        r29_swing_clearance_m=0.05,
        r29_s4_stance_bce_weight=0.30,
        r29_s4_footstep_target_weight=0.40,
    ),
    NextAblationVariant(
        variant_id="r29_nb_h1_r0_plus_oracle_full_hint",
        purpose=(
            "H1: R0 cond + Round-28 full oracle interaction hint (dim=13, "
            "variant=full, input_add). Tests whether richer condition "
            "content can still improve sustained contact under R0's schedule."
        ),
        decision_question=(
            "If H1 strongly improves hand drift/p95, current I3/I5 is too "
            "weak; next architecture needs stronger Stage-1.5 contact "
            "planner/hint. If H1 does not improve, the bottleneck is "
            "deeper than condition content (motion repr / decoder / objective)."
        ),
        use_oracle_interaction_hint_data=True,
        use_oracle_interaction_hint_model=True,
        oracle_hint_variant="full",
        oracle_hint_dim=13,
        oracle_hint_injection_mode="input_add",
    ),
]


def _render_datasets_block(data_root: str) -> str:
    stripped = data_root.rstrip("/").rstrip("\\")
    lines: list[str] = []
    for sub in DATASET_SUBSET_NAMES:
        lines.append(f'    - name: "{sub}"')
        lines.append(f'      root: "{stripped}/{sub}"')
    return "\n".join(lines)


def _render_yaml(v: NextAblationVariant, *, data_root: str) -> str:
    run_name = f"stageB_anchordiff_{v.variant_id}"
    use_r29 = "true" if v.use_round29_cond_injection else "false"
    use_oh_data = "true" if v.use_oracle_interaction_hint_data else "false"
    use_oh_model = "true" if v.use_oracle_interaction_hint_model else "false"
    oracle_hint_block_data = (
        f'\n  use_oracle_interaction_hint: {use_oh_data}'
        f'\n  oracle_hint_variant: "{v.oracle_hint_variant}"'
        f'\n  oracle_hint_fps: 20.0'
        if v.use_oracle_interaction_hint_data else ""
    )
    oracle_hint_block_model = (
        f'\n    use_oracle_interaction_hint: {use_oh_model}'
        f'\n    oracle_hint_dim: {v.oracle_hint_dim}'
        f'\n    oracle_hint_injection_mode: "{v.oracle_hint_injection_mode}"'
        if v.use_oracle_interaction_hint_model else ""
    )
    return f"""# Round-29 next-baseline ablation {v.variant_id}: {v.purpose}
#
# Generated by scripts/stage_b_generator/round29_make_next_ablation_configs.py.
# Per analyses/2026-05-27_round29_next_ablation_execution_prompt_for_claude_code.md.
#
# FULL DATA. No subset_indices_file. From scratch (no init_checkpoint).
# Heldout val. {SCHEDULE_NUM_EPOCHS} ep / val_every=5 / save_every=10 / warmup=250.
# Schedule: bs={SCHEDULE_BATCH_SIZE} accum={SCHEDULE_ACCUM_STEPS} (2× 5080).

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
    use_round29_cond_injection: {use_r29}
    r29_coarse_extra_dim: {v.r29_coarse_extra_dim}
    r29_interaction_dim: {v.r29_interaction_dim}
    r29_support_dim: {v.r29_support_dim}
    r29_body_refine_dim: {v.r29_body_refine_dim}
    r29_injection_mode: "{v.r29_injection_mode}"
    r29_gate_bias_init: -1.0
    r29_per_family_modes: null
    r29_zero_init_adapters: true{oracle_hint_block_model}
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
  # Full data → NO subset_indices_file.
  r29_coarse_variant: "{v.r29_coarse_variant}"
  r29_interaction_variant: "{v.r29_interaction_variant}"
  r29_support_variant: "{v.r29_support_variant}"
  r29_body_variant: "{v.r29_body_variant}"
  r29_body_coord_frame: null
  r29_body_energy_threshold: 0.05
  r29_body_lowpass_window: 9
  r29_hand_offset_clamp_m: 2.0{oracle_hint_block_data}
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
  stage1_coarse_noise_std: 0.05
  optimizer:
    name: "adamw"
    lr: 5.0e-5
    weight_decay: 0.01
    betas: [0.9, 0.999]
  scheduler:
    name: "cosine"
    warmup_steps: 250
  gradient_accumulation_steps: {SCHEDULE_ACCUM_STEPS}
  max_grad_norm: 1.0
  mixed_precision: "bf16"
  val_on_train_subset: false
  val_every_epochs: 5
  val_best_key: "{v.val_best_key}"

loss:
  anchor_weight: 0.0
  contact_threshold: 0.5
  stable_root_vel_weight: 0.5
  stable_root_acc_weight: 0.25
  stable_support_erode: 4
  stable_local_vel_cm_weight: 0.05
  stable_local_speed_moment_weight: 0.02
  hand_endpoint_weight: {v.hand_endpoint_weight}
  foot_endpoint_weight: {v.foot_endpoint_weight}
  pos_loss_weight: {v.pos_loss_weight}
  anchor_joint_pos_weight: {v.anchor_joint_pos_weight}
  anchor_joint_vel_weight: {v.anchor_joint_vel_weight}
  anchor_joint_part_weights: [2.0, 2.0, 0.0, 0.0, 0.5]
  use_min_snr_weighting: true
  min_snr_gamma: 5.0
  world_joint_velocity_weight: {v.world_joint_velocity_weight}
  temporal_interaction:
    contact_rel_offset_weight: 0.0
    contact_drift_weight: 0.0
    contact_tracking_weight: 0.0
    gait_both_airborne_weight: 0.0
    gait_stance_velocity_weight: 0.0
    hint_contact_consistency_weight: 0.0
    body_action_consistency_weight: 0.0
    r29_interaction_consistency_weight: 0.0
    r29_support_both_airborne_weight: {v.r29_support_both_airborne_weight}
    r29_support_stance_velocity_weight: {v.r29_support_stance_velocity_weight}
    r29_swing_clearance_weight: {v.r29_swing_clearance_weight}
    r29_swing_clearance_m: {v.r29_swing_clearance_m}
    # G2 — exact S4 execution.
    r29_s4_stance_bce_weight: {v.r29_s4_stance_bce_weight}
    r29_s4_footstep_target_weight: {v.r29_s4_footstep_target_weight}
    # G1 — phase-free gait losses.
    r29_gait_soft_stance_velocity_weight: {v.r29_gait_soft_stance_velocity_weight}
    r29_gait_transition_rate_weight: {v.r29_gait_transition_rate_weight}
    r29_gait_duty_cycle_weight: {v.r29_gait_duty_cycle_weight}
    r29_gait_both_state_match_weight: {v.r29_gait_both_state_match_weight}
    r29_gait_ankle_smooth_weight: {v.r29_gait_ankle_smooth_weight}
    r29_gait_antiphase_corr_weight: {v.r29_gait_antiphase_corr_weight}
    # Old R2 one-foot-support intentionally OFF for G1 (it was the
    # degeneracy source). Kept zero across all five variants here.
    r29_gait_one_foot_support_weight: 0.0
    r29_gait_pred_stance_velocity_weight: 0.0
    # R4 / R5 contact-lock — OFF on this matrix (covered by H1 oracle hint).
    r29_contact_lock_offset_weight: 0.0
    r29_contact_lock_segment_drift_weight: 0.0
    r29_contact_lock_tracking_weight: 0.0
    contact_threshold: 0.5
    contact_rel_clamp_m: 2.0
    tracking_margin_m: 0.03
    tracking_min_obj_disp_m: 0.05
    floor_quantile: 0.05
    grounded_threshold_above_floor_m: 0.10
    grounded_softness_m: 0.03

logging:
  project: "piano"
  run_name: "{run_name}"
  log_every_n_steps: 50
  save_every_n_epochs: 10

output_dir: "runs/training/{run_name}"
"""


def _manifest_row(v: NextAblationVariant) -> dict:
    canonical_cfg = f"configs/training/anchordiff_{v.variant_id}.yaml"
    return {
        "variant_id": v.variant_id,
        "group": "NextBaselineAblation",
        "purpose": v.purpose,
        "decision_question": v.decision_question,
        "train": True,
        "injection_mode": v.r29_injection_mode if v.use_round29_cond_injection else "off",
        "use_round29_cond_injection": v.use_round29_cond_injection,
        "condition": {
            "r29_coarse_variant": v.r29_coarse_variant,
            "r29_coarse_extra_dim": v.r29_coarse_extra_dim,
            "r29_interaction_variant": v.r29_interaction_variant,
            "r29_interaction_dim": v.r29_interaction_dim,
            "r29_support_variant": v.r29_support_variant,
            "r29_support_dim": v.r29_support_dim,
            "r29_body_variant": v.r29_body_variant,
            "r29_body_refine_dim": v.r29_body_refine_dim,
        },
        "oracle_hint": {
            "data_enabled": v.use_oracle_interaction_hint_data,
            "model_enabled": v.use_oracle_interaction_hint_model,
            "variant": v.oracle_hint_variant,
            "dim": v.oracle_hint_dim,
            "injection_mode": v.oracle_hint_injection_mode,
        },
        "loss_knobs": {
            "pos_loss_weight": v.pos_loss_weight,
            "hand_endpoint_weight": v.hand_endpoint_weight,
            "foot_endpoint_weight": v.foot_endpoint_weight,
            "anchor_joint_pos_weight": v.anchor_joint_pos_weight,
            "anchor_joint_vel_weight": v.anchor_joint_vel_weight,
            "world_joint_velocity_weight": v.world_joint_velocity_weight,
            "r29_support_both_airborne_weight": v.r29_support_both_airborne_weight,
            "r29_support_stance_velocity_weight": v.r29_support_stance_velocity_weight,
            "r29_swing_clearance_weight": v.r29_swing_clearance_weight,
            "r29_s4_stance_bce_weight": v.r29_s4_stance_bce_weight,
            "r29_s4_footstep_target_weight": v.r29_s4_footstep_target_weight,
            "r29_gait_soft_stance_velocity_weight": v.r29_gait_soft_stance_velocity_weight,
            "r29_gait_transition_rate_weight": v.r29_gait_transition_rate_weight,
            "r29_gait_duty_cycle_weight": v.r29_gait_duty_cycle_weight,
            "r29_gait_both_state_match_weight": v.r29_gait_both_state_match_weight,
            "r29_gait_ankle_smooth_weight": v.r29_gait_ankle_smooth_weight,
            "r29_gait_antiphase_corr_weight": v.r29_gait_antiphase_corr_weight,
            "r29_gait_one_foot_support_weight": 0.0,
        },
        "training_schedule": {
            "batch_size": SCHEDULE_BATCH_SIZE,
            "gradient_accumulation_steps": SCHEDULE_ACCUM_STEPS,
            "num_epochs": SCHEDULE_NUM_EPOCHS,
            "lr": 5.0e-5,
            "warmup_steps": 250,
            "mixed_precision": "bf16",
            "seed": 42,
            "val_every_epochs": 5,
            "save_every_n_epochs": 10,
            "val_best_key": v.val_best_key,
        },
        "training_data": "full",
        "config_path": canonical_cfg,
        "output_dir": f"runs/training/stageB_anchordiff_{v.variant_id}",
        "diagnostics": list(v.diagnostics),
    }


def _r0_reference_row() -> dict:
    """R0 reference entry — not retrained by this matrix. Read by the
    summarizer to compare B0/B1/G1/G2/H1 against the existing R29-FT R0
    diagnostic stats.
    """
    return {
        "variant_id": R0_REFERENCE_VARIANT_ID,
        "group": "NextBaselineAblation",
        "purpose": (
            "R0 (reference only — not retrained by this matrix): existing "
            "Round-29 failure-targeted clean baseline. Comparison baseline "
            "for B0/B1/G1/G2/H1."
        ),
        "decision_question": "Reference baseline (not retrained).",
        "train": False,
        "injection_mode": "input_add_adapter",
        "use_round29_cond_injection": True,
        "condition": {
            "r29_coarse_variant": "C41-current",
            "r29_coarse_extra_dim": 18,
            "r29_interaction_variant": "I3-contact-offset-masked",
            "r29_interaction_dim": 8,
            "r29_support_variant": "S4-S1-phase-footstep",
            "r29_support_dim": 13,
            "r29_body_variant": "B4-lowpass-residual-mask",
            "r29_body_refine_dim": 20,
        },
        "oracle_hint": {
            "data_enabled": False,
            "model_enabled": False,
            "variant": "",
            "dim": 0,
            "injection_mode": "",
        },
        "loss_knobs": {
            "pos_loss_weight": 5.0,
            "hand_endpoint_weight": 2.0,
            "foot_endpoint_weight": 2.0,
            "anchor_joint_pos_weight": 10.0,
            "anchor_joint_vel_weight": 2.0,
            "world_joint_velocity_weight": 1.0,
        },
        "training_schedule": {
            "batch_size": SCHEDULE_BATCH_SIZE,
            "gradient_accumulation_steps": SCHEDULE_ACCUM_STEPS,
            "num_epochs": SCHEDULE_NUM_EPOCHS,
            "lr": 5.0e-5,
            "warmup_steps": 250,
            "mixed_precision": "bf16",
            "seed": 42,
            "val_every_epochs": 5,
            "save_every_n_epochs": 10,
            "val_best_key": "loss_anchor_joint_pos",
        },
        "training_data": "full",
        "config_path": R0_REFERENCE_CONFIG_PATH,
        "output_dir": R0_REFERENCE_OUTPUT_DIR,
        "diagnostics": ["sustained_contact", "gait", "body_action"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Round-29 next-baseline ablation configs (B0/B1/G1/G2/H1) "
            "+ manifest. R0 is referenced (not retrained)."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--data-root",
        default=os.environ.get("DATASETS_ROOT", DEFAULT_DATA_ROOT),
        help=(
            "Root directory containing the four InterAct subsets. "
            "On the Linux server pass --data-root /media/8TB_data/Cai/datasets/InterAct/piano_official_process_4 "
            "or export DATASETS_ROOT=..."
        ),
    )
    parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--analyses-dir", default=str(DEFAULT_ANALYSES_DIR))
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    analyses_dir = Path(args.analyses_dir)

    train_rows: list[dict] = []
    for v in VARIANTS:
        out_path = config_dir / f"anchordiff_{v.variant_id}.yaml"
        content = _render_yaml(v, data_root=args.data_root)
        train_rows.append(_manifest_row(v))
        if args.dry_run:
            print(f"DRY-RUN would write: {out_path}  ({len(content)} bytes)")
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(content, encoding="utf-8")
            print(f"wrote {out_path}")

    ref_row = _r0_reference_row()
    all_rows = train_rows + [ref_row]

    manifest_json = analyses_dir / "round29_next_ablation_manifest.json"
    manifest_md = analyses_dir / "round29_next_ablation_manifest.md"

    md_lines: list[str] = [
        "# Round-29 next-baseline ablation manifest",
        "",
        "Per `analyses/2026-05-27_round29_next_ablation_execution_prompt_for_claude_code.md`:",
        "5-train-variant matrix (B0/B1/G1/G2/H1) + R0 reference (not retrained)",
        "to answer three questions: (1) is R0 from C/I/S/B content or schedule?",
        "(2) can phase-free gait losses fix R2? (3) is I3/I5 the contact bottleneck?",
        "",
        f"All 5 new variants: FULL InterAct train set (no subset_indices_file), "
        f"from scratch (no init_checkpoint), {SCHEDULE_NUM_EPOCHS} ep, heldout val, "
        f"save_every=10, warmup=250, stage1_coarse_noise_std=0.05, "
        f"bs={SCHEDULE_BATCH_SIZE} / accum={SCHEDULE_ACCUM_STEPS} (2× 5080).",
        "",
        "| variant | retrain | C | I | S | B | oracle hint | targeted intervention |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in all_rows:
        c = r["condition"]
        oh = r["oracle_hint"]
        oh_str = (
            f"{oh['variant']}/dim={oh['dim']}/{oh['injection_mode']}"
            if oh["model_enabled"] else "—"
        )
        retrain = "yes" if r["train"] else "no (ref)"
        md_lines.append(
            f"| `{r['variant_id']}` | {retrain} | "
            f"{c['r29_coarse_variant']} (dim={c['r29_coarse_extra_dim']}) | "
            f"{c['r29_interaction_variant']} (dim={c['r29_interaction_dim']}) | "
            f"{c['r29_support_variant']} (dim={c['r29_support_dim']}) | "
            f"{c['r29_body_variant']} (dim={c['r29_body_refine_dim']}) | "
            f"{oh_str} | "
            f"{r['purpose'].split(': ', 1)[-1]} |"
        )
    md_lines.append("")
    md_lines.append("## Decision questions")
    md_lines.append("")
    for r in train_rows:
        md_lines.append(f"- **`{r['variant_id']}`**: {r['decision_question']}")
    md_lines.append("")
    md_lines.append("## Loss-knob activations (only non-zero weights per variant)")
    md_lines.append("")
    for r in train_rows:
        active = {k: v for k, v in r["loss_knobs"].items() if float(v) != 0.0}
        md_lines.append(f"### `{r['variant_id']}`")
        md_lines.append("")
        if not active:
            md_lines.append("- (baseline absolute losses only)")
        for k, v in active.items():
            md_lines.append(f"- `{k}` = {v}")
        md_lines.append("")

    if args.dry_run:
        print(f"DRY-RUN would write: {manifest_json}")
        print(f"DRY-RUN would write: {manifest_md}")
    else:
        analyses_dir.mkdir(parents=True, exist_ok=True)
        manifest_json.write_text(
            json.dumps(
                {"variants": all_rows, "data_root": args.data_root},
                indent=2,
            ),
            encoding="utf-8",
        )
        manifest_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
        print(f"wrote {manifest_json}")
        print(f"wrote {manifest_md}")
    print(
        f"\n{len(train_rows)} new train variants + 1 R0 reference "
        f"({len(all_rows)} manifest rows total)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
