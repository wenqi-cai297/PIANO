"""Generate Round-29 failure-targeted ablation configs (R0-R5).

Per ``analyses/2026-05-27_round29_failure_targeted_ablation_prompt_for_claude_code.md``:
take the current full-data winner (`r29_lsf_a3_baseline_from_scratch`) and
test five orthogonal interventions against R0 (clean rerun) to identify the
load-bearing fix for the user's three failure modes:

  - gait flicker / unnatural ankle twist     (R2 behavior gait, R3 oracle S4)
  - sustained hand-object drift              (R4 contact-lock on I3,
                                               R5 all-part interaction + lock)
  - is C41 extra coarse even useful?         (R1 no-coarse-extra)
  - clean patched reference                  (R0)

Six-variant matrix:

  R0  r29_ft_r0_clean_a3_baseline      Clean rerun of the current full-data
                                       winner. Baseline for all others.
  R1  r29_ft_r1_no_coarse_extra        C23 coarse, no C41 extra → ablate
                                       Stage-2 oracle coarse-extra channel.
  R2  r29_ft_r2_behavior_gait_loss     Behavior-level gait losses (no GT
                                       phase target; uses pred_grounded_prob).
  R3  r29_ft_r3_oracle_s4_gait_loss    Exact S4 stance BCE + footstep target
                                       (on top of existing R29 swing/airborne).
  R4  r29_ft_r4_i3_contact_lock       Contact-lock losses on I3 (offset +
                                       segment-drift + tracking).
  R5  r29_ft_r5_allpart_interaction_lock  I5 (all 5 contact parts) + contact-
                                          lock generalized to 5 parts.

All six: FULL InterAct train set (no subset_indices_file), 80 ep, heldout val
(val_every=5), save_every=10, warmup=250, stage1_coarse_noise_std=0.05.
Schedule: bs=32 / accum=1 (preferred for 3× 5080) per prompt §3.

Outputs:
    configs/training/anchordiff_r29_ft_r0_clean_a3_baseline.yaml
    configs/training/anchordiff_r29_ft_r1_no_coarse_extra.yaml
    configs/training/anchordiff_r29_ft_r2_behavior_gait_loss.yaml
    configs/training/anchordiff_r29_ft_r3_oracle_s4_gait_loss.yaml
    configs/training/anchordiff_r29_ft_r4_i3_contact_lock.yaml
    configs/training/anchordiff_r29_ft_r5_allpart_interaction_lock.yaml
    analyses/round29_failure_targeted_ablation_manifest.json
    analyses/round29_failure_targeted_ablation_manifest.md

Usage:
    python scripts/stage_b_generator/round29_make_failure_targeted_ablation_configs.py
    python scripts/stage_b_generator/round29_make_failure_targeted_ablation_configs.py --dry-run
    python scripts/stage_b_generator/round29_make_failure_targeted_ablation_configs.py \\
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
# Schedule.
#
# Prompt §3: preferred bs=32, accum=1, 80 ep on 3× 5080 (preserves eff-batch
# 96 from the closed R29 matrix, restores opt-update budget). Fallback bs=24/
# accum=1/90 ep — pick consistently for all six. Generator default is bs=32.
# --------------------------------------------------------------------------- #

SCHEDULE_BATCH_SIZE = 32
SCHEDULE_ACCUM_STEPS = 1
SCHEDULE_NUM_EPOCHS = 80


@dataclass(slots=True)
class FailureTargetedVariant:
    variant_id: str
    purpose: str
    decision_question: str

    # Condition family selectors (R0-R4 keep R0's C41+I3+S4+B4; R1 drops C41
    # to C23; R5 swaps I3 → I5).
    r29_coarse_variant: str = "C41-current"
    r29_coarse_extra_dim: int = 18
    r29_interaction_variant: str = "I3-contact-offset-masked"
    r29_interaction_dim: int = 8
    r29_support_variant: str = "S4-S1-phase-footstep"
    r29_support_dim: int = 13
    r29_body_variant: str = "B4-lowpass-residual-mask"
    r29_body_refine_dim: int = 20

    # Absolute-GT auxiliary loss weights — all six keep R0 baseline_from_scratch
    # (pos_loss=5, anchor_pos=10, anchor_vel=2, world_vel=1, hand/foot=2).
    pos_loss_weight: float = 5.0
    hand_endpoint_weight: float = 2.0
    foot_endpoint_weight: float = 2.0
    anchor_joint_pos_weight: float = 10.0
    anchor_joint_vel_weight: float = 2.0
    world_joint_velocity_weight: float = 1.0

    # Existing R29 condition-consistency weights (kept at 0 except where a
    # variant explicitly activates one). NOT the anchor2_mixed strategy.
    contact_rel_offset_weight: float = 0.0
    contact_drift_weight: float = 0.0
    contact_tracking_weight: float = 0.0
    r29_interaction_consistency_weight: float = 0.0
    r29_support_both_airborne_weight: float = 0.0
    r29_support_stance_velocity_weight: float = 0.0
    r29_swing_clearance_weight: float = 0.0
    r29_swing_clearance_m: float = 0.05

    # R2 behavior-level gait losses (new keys; default 0).
    r29_gait_one_foot_support_weight: float = 0.0
    r29_gait_pred_stance_velocity_weight: float = 0.0
    r29_gait_ankle_smooth_weight: float = 0.0
    r29_gait_antiphase_corr_weight: float = 0.0

    # R3 oracle-S4 execution losses (new keys; default 0).
    r29_s4_stance_bce_weight: float = 0.0
    r29_s4_footstep_target_weight: float = 0.0

    # R4 / R5 contact-lock losses (new keys; default 0).
    r29_contact_lock_offset_weight: float = 0.0
    r29_contact_lock_segment_drift_weight: float = 0.0
    r29_contact_lock_tracking_weight: float = 0.0

    val_best_key: str = "loss_anchor_joint_pos"
    diagnostics: tuple[str, ...] = field(default_factory=lambda: (
        "sustained_contact", "gait", "body_action",
    ))


# --------------------------------------------------------------------------- #
# Six variants.
# --------------------------------------------------------------------------- #

VARIANTS: list[FailureTargetedVariant] = [
    FailureTargetedVariant(
        variant_id="r29_ft_r0_clean_a3_baseline",
        purpose=(
            "R0: clean rerun of the current full-data winner "
            "(r29_lsf_a3_baseline_from_scratch) under the patched trainer / "
            "schedule. Reference baseline for R1-R5."
        ),
        decision_question=(
            "Does R0 differ materially from the closed R29 matrix verdict? "
            "If yes, the old verdict mixed in trainer/schedule effects."
        ),
    ),
    FailureTargetedVariant(
        variant_id="r29_ft_r1_no_coarse_extra",
        purpose=(
            "R1: drop oracle C41-current 18D coarse-extra channel "
            "(r29_coarse_variant=C23, r29_coarse_extra_dim=0). Tests whether "
            "C41 extra is actually load-bearing — gates Stage-1 Coarse-v2."
        ),
        decision_question=(
            "If R1 is within ~5% of R0 on contact/gait/body, C41 extra is not "
            "load-bearing; defer Stage-1 Coarse-v2."
        ),
        r29_coarse_variant="C23",
        r29_coarse_extra_dim=0,
    ),
    FailureTargetedVariant(
        variant_id="r29_ft_r2_behavior_gait_loss",
        purpose=(
            "R2: behavior-level gait losses (no GT left/right phase target). "
            "Uses pred_grounded_prob + walking_mask only; respects multimodal "
            "left/right equivalence. Targets gait flicker + ankle twist."
        ),
        decision_question=(
            "If R2 improves L_R_corr, step_period_rate, and visual ankle "
            "flicker without hurting contact/body, behavior-level gait "
            "supervision is the right next mainline."
        ),
        r29_gait_one_foot_support_weight=0.20,
        r29_gait_pred_stance_velocity_weight=0.10,
        r29_gait_ankle_smooth_weight=0.02,
        r29_gait_antiphase_corr_weight=0.05,
    ),
    FailureTargetedVariant(
        variant_id="r29_ft_r3_oracle_s4_gait_loss",
        purpose=(
            "R3: exact S4 gait/footstep execution losses. Stance BCE against "
            "stage2_support[..., 0:2] + footstep target SmoothL1 against "
            "stage2_support[..., 9:13]. Plus existing R29 swing/airborne "
            "terms. Tests whether explicit GT-phase locking is needed."
        ),
        decision_question=(
            "If R3 >> R2, next architecture direction is explicit Stage-1.5 "
            "gait/phase/footstep condition. If R2 ~= R3, prefer R2 (multimodal)."
        ),
        # Existing R29 support terms re-activated alongside the new S4 terms.
        r29_support_both_airborne_weight=0.10,
        r29_support_stance_velocity_weight=0.10,
        r29_swing_clearance_weight=0.10,
        r29_swing_clearance_m=0.05,
        # New exact S4 terms.
        r29_s4_stance_bce_weight=0.10,
        r29_s4_footstep_target_weight=0.20,
    ),
    FailureTargetedVariant(
        variant_id="r29_ft_r4_i3_contact_lock",
        purpose=(
            "R4: hand contact-lock losses on the current I3 condition. "
            "Offset + segment-drift + tracking; all relative behavior, not "
            "absolute imitation. Keeps strong baseline absolute stabilizers."
        ),
        decision_question=(
            "If R4 improves hand drift without hurting gait/body, contact-"
            "lock is a mainline component. If R4 does not improve, I3 content "
            "is insufficient or the model cannot use it."
        ),
        r29_contact_lock_offset_weight=0.50,
        r29_contact_lock_segment_drift_weight=0.50,
        r29_contact_lock_tracking_weight=0.25,
    ),
    FailureTargetedVariant(
        variant_id="r29_ft_r5_allpart_interaction_lock",
        purpose=(
            "R5: all-part interaction condition (I5-allpart-contact-offset-"
            "masked, dim=20: 5 contact + 5×3 object-local offsets for "
            "L_hand, R_hand, L_foot, R_foot, pelvis) + contact-lock losses "
            "generalized to 5 parts."
        ),
        decision_question=(
            "If R5 >> R4 (especially on feet/pelvis or hand-foot mixed), I3 "
            "is too narrow; upgrade interaction condition to all-part. "
            "If R4 ~= R5, hands-only I3 was enough and the lock loss is key."
        ),
        r29_interaction_variant="I5-allpart-contact-offset-masked",
        r29_interaction_dim=20,
        r29_contact_lock_offset_weight=0.50,
        r29_contact_lock_segment_drift_weight=0.50,
        r29_contact_lock_tracking_weight=0.25,
    ),
]


def _render_datasets_block(data_root: str) -> str:
    stripped = data_root.rstrip("/").rstrip("\\")
    lines: list[str] = []
    for sub in DATASET_SUBSET_NAMES:
        lines.append(f'    - name: "{sub}"')
        lines.append(f'      root: "{stripped}/{sub}"')
    return "\n".join(lines)


def _render_yaml(v: FailureTargetedVariant, *, data_root: str) -> str:
    run_name = f"stageB_anchordiff_{v.variant_id}"
    return f"""# Round-29 failure-targeted ablation {v.variant_id}: {v.purpose}
#
# Generated by scripts/stage_b_generator/round29_make_failure_targeted_ablation_configs.py.
# Per analyses/2026-05-27_round29_failure_targeted_ablation_prompt_for_claude_code.md.
#
# FULL DATA. No subset_indices_file. Heldout val. 80 ep / val_every=5 /
# save_every=10 / warmup=250. Schedule: bs={SCHEDULE_BATCH_SIZE} accum={SCHEDULE_ACCUM_STEPS} (3× 5080).

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
    use_round29_cond_injection: true
    r29_coarse_extra_dim: {v.r29_coarse_extra_dim}
    r29_interaction_dim: {v.r29_interaction_dim}
    r29_support_dim: {v.r29_support_dim}
    r29_body_refine_dim: {v.r29_body_refine_dim}
    r29_injection_mode: "input_add_adapter"
    r29_gate_bias_init: -1.0
    r29_per_family_modes: null
    r29_zero_init_adapters: true
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
    contact_rel_offset_weight: {v.contact_rel_offset_weight}
    contact_drift_weight: {v.contact_drift_weight}
    contact_tracking_weight: {v.contact_tracking_weight}
    gait_both_airborne_weight: 0.0
    gait_stance_velocity_weight: 0.0
    hint_contact_consistency_weight: 0.0
    body_action_consistency_weight: 0.0
    r29_interaction_consistency_weight: {v.r29_interaction_consistency_weight}
    r29_support_both_airborne_weight: {v.r29_support_both_airborne_weight}
    r29_support_stance_velocity_weight: {v.r29_support_stance_velocity_weight}
    r29_swing_clearance_weight: {v.r29_swing_clearance_weight}
    r29_swing_clearance_m: {v.r29_swing_clearance_m}
    # R2 behavior-level gait losses.
    r29_gait_one_foot_support_weight: {v.r29_gait_one_foot_support_weight}
    r29_gait_pred_stance_velocity_weight: {v.r29_gait_pred_stance_velocity_weight}
    r29_gait_ankle_smooth_weight: {v.r29_gait_ankle_smooth_weight}
    r29_gait_antiphase_corr_weight: {v.r29_gait_antiphase_corr_weight}
    # R3 exact S4 execution losses.
    r29_s4_stance_bce_weight: {v.r29_s4_stance_bce_weight}
    r29_s4_footstep_target_weight: {v.r29_s4_footstep_target_weight}
    # R4 / R5 contact-lock losses.
    r29_contact_lock_offset_weight: {v.r29_contact_lock_offset_weight}
    r29_contact_lock_segment_drift_weight: {v.r29_contact_lock_segment_drift_weight}
    r29_contact_lock_tracking_weight: {v.r29_contact_lock_tracking_weight}
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


def _manifest_row(v: FailureTargetedVariant) -> dict:
    canonical_cfg = f"configs/training/anchordiff_{v.variant_id}.yaml"
    return {
        "variant_id": v.variant_id,
        "group": "FailureTargetedAblation",
        "purpose": v.purpose,
        "decision_question": v.decision_question,
        "injection_mode": "input_add_adapter",
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
        "loss_knobs": {
            "pos_loss_weight": v.pos_loss_weight,
            "hand_endpoint_weight": v.hand_endpoint_weight,
            "foot_endpoint_weight": v.foot_endpoint_weight,
            "anchor_joint_pos_weight": v.anchor_joint_pos_weight,
            "anchor_joint_vel_weight": v.anchor_joint_vel_weight,
            "world_joint_velocity_weight": v.world_joint_velocity_weight,
            "contact_rel_offset_weight": v.contact_rel_offset_weight,
            "contact_drift_weight": v.contact_drift_weight,
            "contact_tracking_weight": v.contact_tracking_weight,
            "r29_interaction_consistency_weight": v.r29_interaction_consistency_weight,
            "r29_support_both_airborne_weight": v.r29_support_both_airborne_weight,
            "r29_support_stance_velocity_weight": v.r29_support_stance_velocity_weight,
            "r29_swing_clearance_weight": v.r29_swing_clearance_weight,
            "r29_gait_one_foot_support_weight": v.r29_gait_one_foot_support_weight,
            "r29_gait_pred_stance_velocity_weight": v.r29_gait_pred_stance_velocity_weight,
            "r29_gait_ankle_smooth_weight": v.r29_gait_ankle_smooth_weight,
            "r29_gait_antiphase_corr_weight": v.r29_gait_antiphase_corr_weight,
            "r29_s4_stance_bce_weight": v.r29_s4_stance_bce_weight,
            "r29_s4_footstep_target_weight": v.r29_s4_footstep_target_weight,
            "r29_contact_lock_offset_weight": v.r29_contact_lock_offset_weight,
            "r29_contact_lock_segment_drift_weight": v.r29_contact_lock_segment_drift_weight,
            "r29_contact_lock_tracking_weight": v.r29_contact_lock_tracking_weight,
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Round-29 failure-targeted ablation configs (6 variants: "
            "R0 clean baseline + R1 no-coarse-extra + R2 behavior gait + "
            "R3 oracle S4 + R4 I3 contact-lock + R5 I5 all-part)."
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

    rows: list[dict] = []
    for v in VARIANTS:
        out_path = config_dir / f"anchordiff_{v.variant_id}.yaml"
        content = _render_yaml(v, data_root=args.data_root)
        rows.append(_manifest_row(v))
        if args.dry_run:
            print(f"DRY-RUN would write: {out_path}  ({len(content)} bytes)")
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(content, encoding="utf-8")
            print(f"wrote {out_path}")

    manifest_json = analyses_dir / "round29_failure_targeted_ablation_manifest.json"
    manifest_md = analyses_dir / "round29_failure_targeted_ablation_manifest.md"

    md_lines: list[str] = [
        "# Round-29 failure-targeted ablation manifest",
        "",
        "Per `analyses/2026-05-27_round29_failure_targeted_ablation_prompt_for_claude_code.md`:",
        "6-variant matrix targeting the three failure modes the visual review",
        "exposed in the closed R29 winner (gait flicker, sustained hand-object",
        "drift, fine limb residual action).",
        "",
        f"All six: FULL InterAct train set (no subset_indices_file), {SCHEDULE_NUM_EPOCHS} ep,",
        "heldout val, save_every=10, warmup=250, stage1_coarse_noise_std=0.05,",
        f"bs={SCHEDULE_BATCH_SIZE} / accum={SCHEDULE_ACCUM_STEPS} (3× 5080), input_add_adapter injection.",
        "",
        "Decision questions (one per variant — see prompt §7):",
        "",
        "| variant | C | I | S | B | targeted intervention |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        c = r["condition"]
        md_lines.append(
            f"| `{r['variant_id']}` | {c['r29_coarse_variant']} (dim={c['r29_coarse_extra_dim']}) | "
            f"{c['r29_interaction_variant']} (dim={c['r29_interaction_dim']}) | "
            f"{c['r29_support_variant']} (dim={c['r29_support_dim']}) | "
            f"{c['r29_body_variant']} (dim={c['r29_body_refine_dim']}) | "
            f"{r['purpose'].split(': ', 1)[-1]} |"
        )
    md_lines.append("")
    md_lines.append("## Decision questions")
    md_lines.append("")
    for r in rows:
        md_lines.append(f"- **`{r['variant_id']}`**: {r['decision_question']}")
    md_lines.append("")
    md_lines.append("## Loss-knob activations (only non-zero weights per variant)")
    md_lines.append("")
    for r in rows:
        active = {k: v for k, v in r["loss_knobs"].items() if float(v) != 0.0}
        md_lines.append(f"### `{r['variant_id']}`")
        md_lines.append("")
        for k, v in active.items():
            md_lines.append(f"- `{k}` = {v}")
        md_lines.append("")

    if args.dry_run:
        print(f"DRY-RUN would write: {manifest_json}")
        print(f"DRY-RUN would write: {manifest_md}")
    else:
        analyses_dir.mkdir(parents=True, exist_ok=True)
        manifest_json.write_text(
            json.dumps({"variants": rows, "data_root": args.data_root}, indent=2),
            encoding="utf-8",
        )
        manifest_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
        print(f"wrote {manifest_json}")
        print(f"wrote {manifest_md}")
    print(f"\n{len(rows)} failure-targeted variant(s) emitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
