"""Generate Round-29 prior-bias (PB) ablation configs.

Per Codex review §4 + §9 of
``analyses/2026-05-29_round29_cond_injection_prior_codex_review_for_claude_code.md``
and the Phase 0 verdict in
``analyses/2026-05-29_round29_cond_usage_verdict.md``.

PB1 — AdaLN-S4. Single variant this round. Decided by Phase 0:

  - C41 + S4 actively_used by A1 (zero S4 → ankle 17-19 cm,
    gait_delta_rel 45.8%); direct corroboration of "A1 vs A0 gait win
    was S4 consumption".
  - Nothing temporally_used (max time_shuffle/zero = 0.83 << 1.20) →
    pooled AdaLN summary loses nothing the model was using.
  - scale_linearity 0.70 (sub-linear) → AdaLN gating is the textbook
    fix for "model responds to cond magnitude but sub-linearly".
  - I3 + B4 task-arm ignored → no point adding them under any
    injection path this round (cancelled per PLAN §2).

Difference from A1 (single variable):
  - r29_use_cond_adaln: true
  - r29_adaln_families: ["support"]                    # Codex §4.4: S4 only
  - r29_adaln_pool: "support_walking_mean"             # walking_mask-weighted mean

Everything else is identical to A1 — same data (C41 + S4), same loss
config (G1 phase-free gait), same schedule (bs=32 / accum=1 / 80 ep / seed 42),
same architecture (d_model=512, n_layers=8, n_heads=4, ff_mult=4).

The new branch is bit-identical to A1 at step 0 (cond_summary_mlp final
Linear is zero-init; see GlobalCondSummary.__init__). PB1 must be trained
**from scratch** to keep the comparison fair (per Codex §4.4 + user
instruction); init_checkpoint is intentionally absent.

Outputs:
    configs/training/anchordiff_r29_pb_a1_adaln_s4.yaml
    analyses/round29_prior_bias_ablation_manifest.json
    analyses/round29_prior_bias_ablation_manifest.md

Usage:
    python scripts/stage_b_generator/round29_make_prior_bias_ablation_configs.py
    python scripts/stage_b_generator/round29_make_prior_bias_ablation_configs.py --dry-run
    python scripts/stage_b_generator/round29_make_prior_bias_ablation_configs.py \\
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

SCHEDULE_BATCH_SIZE = 32
SCHEDULE_ACCUM_STEPS = 1
SCHEDULE_NUM_EPOCHS = 80

ARCH_D_MODEL = 512
ARCH_N_LAYERS = 8
ARCH_N_HEADS = 4
ARCH_FF_MULT = 4
ARCH_DROPOUT = 0.1

_VALID_POOL_MODES: tuple[str, ...] = ("mean", "support_walking_mean")
_VALID_ADALN_FAMILIES: tuple[str, ...] = (
    "coarse_extra", "interaction", "support", "body_refine",
)


@dataclass(slots=True)
class PriorBiasVariant:
    variant_id: str
    purpose: str
    decision_question: str

    # --- DATA SIDE (same as A1). ---
    r29_coarse_variant: str = "C41-current"
    r29_interaction_variant: str = "I0"
    r29_support_variant: str = "S4-S1-phase-footstep"
    r29_body_variant: str = "B0"
    r29_body_coord_frame: str | None = None
    r29_body_energy_threshold: float = 0.05
    r29_body_lowpass_window: int = 9
    r29_hand_offset_clamp_m: float = 2.0

    # --- MODEL SIDE (same as A1 except for PB1 AdaLN). ---
    use_round29_cond_injection: bool = True
    r29_coarse_extra_dim: int = 18
    r29_interaction_dim: int = 0
    r29_support_dim: int = 13
    r29_body_refine_dim: int = 0
    r29_injection_mode: str = "input_add_adapter"
    r29_zero_init_adapters: bool = True

    # --- PB1 NEW FIELDS. ---
    r29_use_cond_adaln: bool = True
    r29_adaln_families: tuple[str, ...] = ("support",)
    r29_adaln_pool: str = "support_walking_mean"

    # --- ARCHITECTURE. ---
    d_model: int = ARCH_D_MODEL
    n_layers: int = ARCH_N_LAYERS
    n_heads: int = ARCH_N_HEADS
    ff_mult: int = ARCH_FF_MULT
    dropout: float = ARCH_DROPOUT

    # --- LOSS KNOBS — A1 settings verbatim. ---
    pos_loss_weight: float = 5.0
    hand_endpoint_weight: float = 2.0
    foot_endpoint_weight: float = 2.0
    anchor_joint_pos_weight: float = 10.0
    anchor_joint_vel_weight: float = 2.0
    world_joint_velocity_weight: float = 1.0
    r29_support_both_airborne_weight: float = 0.0
    r29_support_stance_velocity_weight: float = 0.0
    r29_swing_clearance_weight: float = 0.0
    r29_swing_clearance_m: float = 0.05
    r29_s4_stance_bce_weight: float = 0.0
    r29_s4_footstep_target_weight: float = 0.0
    # G1 phase-free gait losses ON (4 keepers from R29-NB G1).
    r29_gait_soft_stance_velocity_weight: float = 0.05
    r29_gait_transition_rate_weight: float = 0.20
    r29_gait_duty_cycle_weight: float = 0.10
    r29_gait_both_state_match_weight: float = 0.10
    r29_gait_soft_stance_speed_threshold_mps: float = 0.30
    r29_gait_soft_stance_speed_softness_mps: float = 0.10
    r29_gait_ankle_smooth_weight: float = 0.02
    r29_gait_antiphase_corr_weight: float = 0.02
    # Old R2 one-foot-support: NEVER enable (degeneracy source).
    r29_gait_one_foot_support_weight: float = 0.0
    r29_gait_pred_stance_velocity_weight: float = 0.0
    # R4/R5 contact-lock — OFF (same as A1).
    r29_contact_lock_offset_weight: float = 0.0
    r29_contact_lock_segment_drift_weight: float = 0.0
    r29_contact_lock_tracking_weight: float = 0.0

    val_best_key: str = "loss_anchor_joint_pos"
    diagnostics: tuple[str, ...] = field(default_factory=lambda: (
        "sustained_contact", "gait", "body_action", "g1_soft_stance",
    ))


# --------------------------------------------------------------------------- #
# Single PB1 variant per Phase 0 verdict + PLAN.md §2.
# --------------------------------------------------------------------------- #

VARIANTS: list[PriorBiasVariant] = [
    PriorBiasVariant(
        variant_id="r29_pb_a1_adaln_s4",
        purpose=(
            "PB1: AdaLN-cond branch added to A1. Pools the support family "
            "(S4) embedding into a per-sample (B, D) summary via "
            "walking_mask-weighted mean, then adds via a zero-init Linear "
            "to t_emb before the DiT block AdaLN. Bit-identical to A1 at "
            "step 0 (cond_summary_mlp final Linear zero-init). Trained "
            "from scratch."
        ),
        decision_question=(
            "Compare PB1 vs A1: if soft-stance gates improve (low_alt_amp "
            "↓, low_trans ↓, soft_alt_std ↑, constant_mid_rate ↓) without "
            "drift_max regression, the sample-level AdaLN-S4 gate is the "
            "right next architecture and PB1 ships as the new mainline. "
            "Post-PB1 cond-usage re-probe must show S4 scale_linearity ↑ "
            "vs A1's 0.70 — that is how we tell mechanism shift from "
            "stochastic minimum."
        ),
    ),
]


# --------------------------------------------------------------------------- #
# Validation.
# --------------------------------------------------------------------------- #

_COARSE_DIMS: dict[str, int] = {
    "C23": 0,
    "C38-current": 15, "C41-current": 18,
    "C38-root0": 15,   "C41-root0": 18,
}
_INTERACTION_DIMS: dict[str, int] = {
    "I0": 0,
    "I1-contact": 2,
    "I2-offset-masked": 6,
    "I3-contact-offset-masked": 8,
    "I4-contact-offset-unmasked": 8,
    "I5-allpart-contact-offset-masked": 20,
}
_SUPPORT_DIMS: dict[str, int] = {
    "S0": 0,
    "S1-stance-height-walking": 5,
    "S2-S1-phase": 9,
    "S3-S1-footstep-target": 9,
    "S4-S1-phase-footstep": 13,
}
_BODY_DIMS: dict[str, int] = {
    "B0": 0,
    "B1-mask-only": 5,
    "B2-absolute-delta": 15,
    "B3-lowpass-residual": 15,
    "B4-lowpass-residual-mask": 20,
}


def _validate_variant(v: PriorBiasVariant) -> None:
    """Catches misconfigurations the trainer would only flag at first
    forward pass: bad pool mode, AdaLN family not active, mismatched
    data/model dims.
    """
    # 1. PB1 fields must be self-consistent.
    if v.r29_use_cond_adaln:
        if v.r29_adaln_pool not in _VALID_POOL_MODES:
            raise ValueError(
                f"{v.variant_id}: r29_adaln_pool={v.r29_adaln_pool!r} "
                f"not in {_VALID_POOL_MODES}"
            )
        if not v.r29_adaln_families:
            raise ValueError(
                f"{v.variant_id}: r29_use_cond_adaln=True but "
                "r29_adaln_families is empty"
            )
        for fam in v.r29_adaln_families:
            if fam not in _VALID_ADALN_FAMILIES:
                raise ValueError(
                    f"{v.variant_id}: r29_adaln_families contains "
                    f"unknown family {fam!r}; valid: {_VALID_ADALN_FAMILIES}"
                )
        # 1a. Every AdaLN family must actually be ACTIVE in the model
        # (r29_<family>_dim > 0). A non-active family produces an empty
        # cache entry and the pool would fall back to zero.
        active_map = {
            "coarse_extra": v.r29_coarse_extra_dim,
            "interaction":  v.r29_interaction_dim,
            "support":      v.r29_support_dim,
            "body_refine":  v.r29_body_refine_dim,
        }
        for fam in v.r29_adaln_families:
            if active_map[fam] <= 0:
                raise ValueError(
                    f"{v.variant_id}: AdaLN family {fam!r} is requested "
                    f"but model dim for that family is "
                    f"{active_map[fam]} (0 = not consumed). Either "
                    "increase the dim or remove the family from "
                    "r29_adaln_families."
                )
        # 1b. support_walking_mean requires support family with dim >= 5
        # (walking_mask is S4 dim 4). Otherwise the pool falls back to mean.
        if (
            v.r29_adaln_pool == "support_walking_mean"
            and "support" not in v.r29_adaln_families
        ):
            raise ValueError(
                f"{v.variant_id}: r29_adaln_pool=support_walking_mean "
                "requires 'support' in r29_adaln_families."
            )
        if (
            v.r29_adaln_pool == "support_walking_mean"
            and v.r29_support_dim < 5
        ):
            raise ValueError(
                f"{v.variant_id}: r29_adaln_pool=support_walking_mean "
                f"requires r29_support_dim >= 5 (walking_mask is dim 4); "
                f"got {v.r29_support_dim}."
            )

    # 2. Data/model dim consistency (same as next-step generator).
    data_inter_dim = _INTERACTION_DIMS.get(v.r29_interaction_variant)
    if data_inter_dim is None:
        raise ValueError(
            f"{v.variant_id}: unknown r29_interaction_variant "
            f"{v.r29_interaction_variant!r}"
        )
    if v.r29_interaction_dim not in (0, data_inter_dim):
        raise ValueError(
            f"{v.variant_id}: model r29_interaction_dim={v.r29_interaction_dim} "
            f"incompatible with data r29_interaction_variant={v.r29_interaction_variant!r} "
            f"(dim={data_inter_dim})."
        )
    data_coarse_dim = _COARSE_DIMS.get(v.r29_coarse_variant)
    if data_coarse_dim is None:
        raise ValueError(
            f"{v.variant_id}: unknown r29_coarse_variant {v.r29_coarse_variant!r}"
        )
    if v.r29_coarse_extra_dim not in (0, data_coarse_dim):
        raise ValueError(
            f"{v.variant_id}: model r29_coarse_extra_dim={v.r29_coarse_extra_dim} "
            f"mismatches data r29_coarse_variant={v.r29_coarse_variant!r} "
            f"(dim={data_coarse_dim})."
        )
    data_support_dim = _SUPPORT_DIMS.get(v.r29_support_variant)
    if data_support_dim is None:
        raise ValueError(
            f"{v.variant_id}: unknown r29_support_variant "
            f"{v.r29_support_variant!r}"
        )
    if v.r29_support_dim not in (0, data_support_dim):
        raise ValueError(
            f"{v.variant_id}: model r29_support_dim={v.r29_support_dim} "
            f"mismatches data r29_support_variant={v.r29_support_variant!r} "
            f"(dim={data_support_dim})."
        )
    data_body_dim = _BODY_DIMS.get(v.r29_body_variant)
    if data_body_dim is None:
        raise ValueError(
            f"{v.variant_id}: unknown r29_body_variant {v.r29_body_variant!r}"
        )
    if v.r29_body_refine_dim not in (0, data_body_dim):
        raise ValueError(
            f"{v.variant_id}: model r29_body_refine_dim={v.r29_body_refine_dim} "
            f"mismatches data r29_body_variant={v.r29_body_variant!r} "
            f"(dim={data_body_dim})."
        )
    # 3. R2 one_foot_support degeneracy source must stay off.
    if v.r29_gait_one_foot_support_weight > 0.0:
        raise ValueError(
            f"{v.variant_id}: r29_gait_one_foot_support_weight must be 0; "
            "documented R2 degeneracy source."
        )


def _yaml_must_not_contain_oracle_hint_fields(yaml_text: str) -> None:
    forbidden = (
        "use_oracle_interaction_hint",
        "oracle_hint_variant",
        "oracle_hint_dim",
        "oracle_hint_injection_mode",
        "oracle_hint_gate_bias_init",
    )
    bad = [k for k in forbidden if k in yaml_text]
    if bad:
        raise ValueError(
            f"generated YAML must not contain dead oracle-hint fields; "
            f"found {bad}"
        )


# --------------------------------------------------------------------------- #
# YAML rendering.
# --------------------------------------------------------------------------- #

def _render_datasets_block(data_root: str) -> str:
    stripped = data_root.rstrip("/").rstrip("\\")
    lines: list[str] = []
    for sub in DATASET_SUBSET_NAMES:
        lines.append(f'    - name: "{sub}"')
        lines.append(f'      root: "{stripped}/{sub}"')
    return "\n".join(lines)


def _render_adaln_families(families: tuple[str, ...]) -> str:
    items = ", ".join(f'"{f}"' for f in families)
    return f"[{items}]"


def _render_yaml(v: PriorBiasVariant, *, data_root: str) -> str:
    run_name = f"stageB_anchordiff_{v.variant_id}"
    use_r29 = "true" if v.use_round29_cond_injection else "false"
    use_pb1 = "true" if v.r29_use_cond_adaln else "false"
    yaml = f"""# Round-29 prior-bias ablation {v.variant_id}: {v.purpose}
#
# Generated by scripts/stage_b_generator/round29_make_prior_bias_ablation_configs.py.
# Per Codex review §4 + §9 of
# analyses/2026-05-29_round29_cond_injection_prior_codex_review_for_claude_code.md
# and Phase 0 verdict in
# analyses/2026-05-29_round29_cond_usage_verdict.md.
#
# FULL DATA. No subset_indices_file. From scratch (no init_checkpoint) —
# fairness requirement per user direction.
# Heldout val. {SCHEDULE_NUM_EPOCHS} ep / val_every=5 / save_every=10 / warmup=250.
# Schedule: bs={SCHEDULE_BATCH_SIZE} accum={SCHEDULE_ACCUM_STEPS} (2× 5080).
#
# PB1 single-variable change vs A1:
#   r29_use_cond_adaln: true            (new)
#   r29_adaln_families: ["support"]     (S4 only — C41 stays input_add per Codex §4.4)
#   r29_adaln_pool: "support_walking_mean"
# All other fields are identical to A1 (anchordiff_r29_ns_a1_c41_s4_g1.yaml).

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
    d_model: {v.d_model}
    n_layers: {v.n_layers}
    n_heads: {v.n_heads}
    ff_mult: {v.ff_mult}
    dropout: {v.dropout}
    stage1_coarse_dim: 23
    use_round29_cond_injection: {use_r29}
    r29_coarse_extra_dim: {v.r29_coarse_extra_dim}
    r29_interaction_dim: {v.r29_interaction_dim}
    r29_support_dim: {v.r29_support_dim}
    r29_body_refine_dim: {v.r29_body_refine_dim}
    r29_injection_mode: "{v.r29_injection_mode}"
    r29_gate_bias_init: -1.0
    r29_per_family_modes: null
    r29_zero_init_adapters: {str(v.r29_zero_init_adapters).lower()}
    # PB1 — AdaLN-cond branch.
    r29_use_cond_adaln: {use_pb1}
    r29_adaln_families: {_render_adaln_families(v.r29_adaln_families)}
    r29_adaln_pool: "{v.r29_adaln_pool}"
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
  r29_body_coord_frame: {("null" if v.r29_body_coord_frame is None else repr(v.r29_body_coord_frame))}
  r29_body_energy_threshold: {v.r29_body_energy_threshold}
  r29_body_lowpass_window: {v.r29_body_lowpass_window}
  r29_hand_offset_clamp_m: {v.r29_hand_offset_clamp_m}
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
    # R3 exact S4 execution — OFF.
    r29_s4_stance_bce_weight: {v.r29_s4_stance_bce_weight}
    r29_s4_footstep_target_weight: {v.r29_s4_footstep_target_weight}
    # G1 phase-free gait losses ON (4 keepers).
    r29_gait_soft_stance_velocity_weight: {v.r29_gait_soft_stance_velocity_weight}
    r29_gait_transition_rate_weight: {v.r29_gait_transition_rate_weight}
    r29_gait_duty_cycle_weight: {v.r29_gait_duty_cycle_weight}
    r29_gait_both_state_match_weight: {v.r29_gait_both_state_match_weight}
    r29_gait_soft_stance_speed_threshold_mps: {v.r29_gait_soft_stance_speed_threshold_mps}
    r29_gait_soft_stance_speed_softness_mps: {v.r29_gait_soft_stance_speed_softness_mps}
    r29_gait_ankle_smooth_weight: {v.r29_gait_ankle_smooth_weight}
    r29_gait_antiphase_corr_weight: {v.r29_gait_antiphase_corr_weight}
    # Old R2 one-foot-support: DISABLED.
    r29_gait_one_foot_support_weight: 0.0
    r29_gait_pred_stance_velocity_weight: 0.0
    # R4/R5 contact-lock — OFF.
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
    _yaml_must_not_contain_oracle_hint_fields(yaml)
    return yaml


def _manifest_row(v: PriorBiasVariant) -> dict:
    return {
        "variant_id": v.variant_id,
        "group": "PriorBiasAblation",
        "purpose": v.purpose,
        "decision_question": v.decision_question,
        "train": True,
        "valid_for_decision": True,
        "condition": {
            "data_variants": {
                "r29_coarse_variant": v.r29_coarse_variant,
                "r29_interaction_variant": v.r29_interaction_variant,
                "r29_support_variant": v.r29_support_variant,
                "r29_body_variant": v.r29_body_variant,
            },
            "model_dims": {
                "use_round29_cond_injection": v.use_round29_cond_injection,
                "r29_coarse_extra_dim": v.r29_coarse_extra_dim,
                "r29_interaction_dim": v.r29_interaction_dim,
                "r29_support_dim": v.r29_support_dim,
                "r29_body_refine_dim": v.r29_body_refine_dim,
                "r29_injection_mode": v.r29_injection_mode,
                "r29_zero_init_adapters": v.r29_zero_init_adapters,
            },
            "pb1": {
                "r29_use_cond_adaln": v.r29_use_cond_adaln,
                "r29_adaln_families": list(v.r29_adaln_families),
                "r29_adaln_pool": v.r29_adaln_pool,
            },
        },
        "architecture": {
            "d_model": v.d_model, "n_layers": v.n_layers,
            "n_heads": v.n_heads, "ff_mult": v.ff_mult,
            "dropout": v.dropout,
        },
        "schedule": {
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
        "from_scratch": True,
        "config_path": f"configs/training/anchordiff_{v.variant_id}.yaml",
        "output_dir": f"runs/training/stageB_anchordiff_{v.variant_id}",
        "diagnostics": list(v.diagnostics),
        "references": {
            "primary": "r29_ns_a1_c41_s4_g1",
            "secondary": "r29_ft_r0_clean_a3_baseline",
        },
    }


# --------------------------------------------------------------------------- #
# Main.
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Round-29 prior-bias (PB1) ablation config. Single "
            "variant this round per Phase 0 verdict + PLAN.md §2 — PB2 "
            "and B4 / I-family variants are deferred or cancelled."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--data-root",
        default=os.environ.get("DATASETS_ROOT", DEFAULT_DATA_ROOT),
        help=(
            "Root directory containing the four InterAct subsets. On the "
            "Linux server pass --data-root /media/8TB_data/Cai/datasets/InterAct/piano_official_process_4 "
            "or export DATASETS_ROOT=..."
        ),
    )
    parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--analyses-dir", default=str(DEFAULT_ANALYSES_DIR))
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    analyses_dir = Path(args.analyses_dir)

    rows: list[dict] = []
    for v in VARIANTS:
        _validate_variant(v)
        out_path = config_dir / f"anchordiff_{v.variant_id}.yaml"
        content = _render_yaml(v, data_root=args.data_root)
        rows.append(_manifest_row(v))
        if args.dry_run:
            print(f"DRY-RUN would write: {out_path}  ({len(content)} bytes)")
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(content, encoding="utf-8")
            print(f"wrote {out_path}")

    manifest_json = analyses_dir / "round29_prior_bias_ablation_manifest.json"
    manifest_md = analyses_dir / "round29_prior_bias_ablation_manifest.md"

    md_lines: list[str] = [
        "# Round-29 prior-bias ablation manifest",
        "",
        "Per Codex review §4 + §9 of "
        "[2026-05-29_round29_cond_injection_prior_codex_review_for_claude_code.md]"
        "(2026-05-29_round29_cond_injection_prior_codex_review_for_claude_code.md) "
        "and Phase 0 verdict in "
        "[2026-05-29_round29_cond_usage_verdict.md](2026-05-29_round29_cond_usage_verdict.md).",
        "",
        f"FULL InterAct train set, from scratch, seed 42, "
        f"{SCHEDULE_NUM_EPOCHS} ep, heldout val, save_every=10, warmup=250, "
        f"stage1_coarse_noise_std=0.05, bs={SCHEDULE_BATCH_SIZE} / "
        f"accum={SCHEDULE_ACCUM_STEPS} (2× 5080). All variants trained "
        "from scratch for fairness — no init_checkpoint.",
        "",
        "## PB1 variant",
        "",
        "| variant | base | PB1 fields | active cond | reference |",
        "|---|---|---|---|---|",
    ]
    for v in VARIANTS:
        fam = "+".join(v.r29_adaln_families)
        md_lines.append(
            f"| `{v.variant_id}` | A1 | adaln={v.r29_use_cond_adaln}, "
            f"families=[{fam}], pool={v.r29_adaln_pool} | "
            f"C41+S4 | r29_ns_a1_c41_s4_g1 |"
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
            ) + "\n",
            encoding="utf-8",
        )
        manifest_md.write_text("\n".join(md_lines), encoding="utf-8")
        print(f"wrote {manifest_json}")
        print(f"wrote {manifest_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
