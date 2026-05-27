"""Generate Round-29 next-step ablation configs (A0/A1/H1/A2).

Per ``analyses/2026-05-28_round29_next_step_ablation_execution_prompt_for_claude_code.md``.

Codex's corrections to the previous next-baseline matrix:

  1. The old ``r29_nb_h1_r0_plus_oracle_full_hint`` is INVALID — the YAML
     fields ``use_oracle_interaction_hint`` / ``oracle_hint_dim`` /
     ``oracle_hint_injection_mode`` are not consumed by the current
     dataset/trainer/model. The old H1 is silently equivalent to R0.
     This matrix replaces it with a live R29-native I5 upper bound.
  2. C41 is load-bearing, but B1 vs R0 CI included zero — do not
     overclaim that I/S/B contribute exactly zero. The new matrix
     isolates whether C41+G1-loss can match R0 without I/S/B as model
     condition, AND whether S4 needs to be CONSUMED (A1) vs only used
     as LOSS TARGET (A0).
  3. G1 is the best current gait direction but may still satisfy its
     aggregate-statistic losses via a constant-mid soft-stance
     degeneracy. The G1 soft-stance diagnostic checks this separately.

Four train variants:

  A0  r29_ns_a0_c41_g1_loss_s4         B1's cond (only C41) + G1 losses.
                                       Dataset emits S4 so G1 losses can
                                       read it, but model does NOT consume
                                       S4 (model support_dim=0). Tests if
                                       Stage-2 needs to consume S4 as cond,
                                       or only as loss target.
  A1  r29_ns_a1_c41_s4_g1              Same as A0 but model also CONSUMES
                                       S4 (model support_dim=13). Isolates
                                       S4-as-condition vs S4-as-loss-target.
  H1  r29_ns_h1_i5_upper_bound         R0 cond with I3 swapped for I5
                                       (all-part contact + offsets, dim=20).
                                       Live R29-native condition upper bound
                                       replacing the dead oracle-hint H1.
  A2  r29_ns_a2_c41_i5_g1               C41 + I5 + G1 (no S4, no B). Plausible
                                       future mainline if A0/A1 say S4 is
                                       loss-only and H1 says I5 is useful.

Four reference rows (NOT retrained):

  R0      r29_ft_r0_clean_a3_baseline           existing R29-FT baseline
  B1      r29_nb_b1_c41_only                    existing next-baseline B1
  G1      r29_nb_g1_phasefree_gait_fixed        existing next-baseline G1
  H1_old  r29_nb_h1_r0_plus_oracle_full_hint    INVALID — historical only,
                                                marked valid_for_decision=false

All train variants: FULL InterAct train set (no subset_indices_file),
from scratch (no init_checkpoint), seed 42, 80 ep, heldout val
(val_every=5), save_every=10, warmup=250, stage1_coarse_noise_std=0.05,
bs=32 / accum=1 (2× 5080), input_add_adapter injection.

A0's decoupled support: dataset surfaces ``stage2_support`` (variant=S4,
dim=13) so G1 losses can read it from ``cond["stage2_support"]``, but
``model.denoiser.r29_support_dim=0`` keeps it out of
``Round29CondInjectionModule.active_families()``. Verified in code at:

  - src/piano/data/dataset.py:710-721 emits stage2_support when
    r29_support_variant != "S0"
  - src/piano/training/train_anchordiff.py:414-421 forwards
    batch["stage2_support"] to cond unconditionally
  - src/piano/models/round29_cond_injection.py:269 iterates only over
    cfg.active_families() — dim=0 family is NOT active, key not read

Outputs:
    configs/training/anchordiff_r29_ns_a0_c41_g1_loss_s4.yaml
    configs/training/anchordiff_r29_ns_a1_c41_s4_g1.yaml
    configs/training/anchordiff_r29_ns_h1_i5_upper_bound.yaml
    configs/training/anchordiff_r29_ns_a2_c41_i5_g1.yaml
    analyses/round29_next_step_ablation_manifest.json
    analyses/round29_next_step_ablation_manifest.md

Usage:
    python scripts/stage_b_generator/round29_make_next_step_ablation_configs.py
    python scripts/stage_b_generator/round29_make_next_step_ablation_configs.py --dry-run
    python scripts/stage_b_generator/round29_make_next_step_ablation_configs.py \\
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
# --------------------------------------------------------------------------- #

SCHEDULE_BATCH_SIZE = 32
SCHEDULE_ACCUM_STEPS = 1
SCHEDULE_NUM_EPOCHS = 80

# Architecture defaults (kept exposed so future capacity variants can
# override without hand-editing YAML — see prompt §9).
ARCH_D_MODEL = 512
ARCH_N_LAYERS = 8
ARCH_N_HEADS = 4
ARCH_FF_MULT = 4
ARCH_DROPOUT = 0.1


# --------------------------------------------------------------------------- #
# Family dim lookup — for validation only. Mirrors
# src/piano/data/stage2_oracle_conditions.py COARSE/INTERACTION/SUPPORT/BODY
# _VARIANT_DIMS so the generator can sanity-check data variant vs model dim
# without importing the heavy piano package.
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


@dataclass(slots=True)
class NextStepVariant:
    variant_id: str
    purpose: str
    decision_question: str
    train: bool = True
    valid_for_decision: bool = True
    invalid_reason: str = ""

    # --- DATA SIDE — what the dataset emits to batch. ---
    r29_coarse_variant: str = "C41-current"
    r29_interaction_variant: str = "I3-contact-offset-masked"
    r29_support_variant: str = "S4-S1-phase-footstep"
    r29_body_variant: str = "B4-lowpass-residual-mask"
    r29_body_coord_frame: str | None = None
    r29_body_energy_threshold: float = 0.05
    r29_body_lowpass_window: int = 9
    r29_hand_offset_clamp_m: float = 2.0

    # --- MODEL SIDE — what Round29CondInjectionModule consumes. ---
    # Decoupled from data side per prompt §2.2: A0 wants S4 from dataset
    # (G1 loss reads it) but model_support_dim=0.
    use_round29_cond_injection: bool = True
    r29_coarse_extra_dim: int = 18
    r29_interaction_dim: int = 8
    r29_support_dim: int = 13
    r29_body_refine_dim: int = 20
    r29_injection_mode: str = "input_add_adapter"
    r29_zero_init_adapters: bool = True

    # --- ARCHITECTURE (default-shared; placeholder for capacity variants). ---
    d_model: int = ARCH_D_MODEL
    n_layers: int = ARCH_N_LAYERS
    n_heads: int = ARCH_N_HEADS
    ff_mult: int = ARCH_FF_MULT
    dropout: float = ARCH_DROPOUT

    # --- LOSS KNOBS — same default zero pattern as round29_make_next_ablation_configs. ---
    pos_loss_weight: float = 5.0
    hand_endpoint_weight: float = 2.0
    foot_endpoint_weight: float = 2.0
    anchor_joint_pos_weight: float = 10.0
    anchor_joint_vel_weight: float = 2.0
    world_joint_velocity_weight: float = 1.0

    # R29 existing condition-consistency / support / swing — all OFF on this matrix.
    r29_support_both_airborne_weight: float = 0.0
    r29_support_stance_velocity_weight: float = 0.0
    r29_swing_clearance_weight: float = 0.0
    r29_swing_clearance_m: float = 0.05

    # R3 exact-S4 execution — all OFF on this matrix.
    r29_s4_stance_bce_weight: float = 0.0
    r29_s4_footstep_target_weight: float = 0.0

    # G1 phase-free gait losses (the 4 keepers per prompt §G1 recommendation).
    r29_gait_soft_stance_velocity_weight: float = 0.0
    r29_gait_transition_rate_weight: float = 0.0
    r29_gait_duty_cycle_weight: float = 0.0
    r29_gait_both_state_match_weight: float = 0.0
    r29_gait_soft_stance_speed_threshold_mps: float = 0.30
    r29_gait_soft_stance_speed_softness_mps: float = 0.10
    r29_gait_ankle_smooth_weight: float = 0.0
    r29_gait_antiphase_corr_weight: float = 0.0

    # Old R2 one-foot-support — must remain 0 (degeneracy source).
    r29_gait_one_foot_support_weight: float = 0.0
    r29_gait_pred_stance_velocity_weight: float = 0.0

    # R4/R5 contact-lock — OFF on this matrix.
    r29_contact_lock_offset_weight: float = 0.0
    r29_contact_lock_segment_drift_weight: float = 0.0
    r29_contact_lock_tracking_weight: float = 0.0

    val_best_key: str = "loss_anchor_joint_pos"
    diagnostics: tuple[str, ...] = field(default_factory=lambda: (
        "sustained_contact", "gait", "body_action",
    ))


# --------------------------------------------------------------------------- #
# Four train variants per prompt §2.2-§2.5.
# --------------------------------------------------------------------------- #

VARIANTS: list[NextStepVariant] = [
    # ---------------- A0 — C41 + G1 losses, S4 loss-only ----------------
    NextStepVariant(
        variant_id="r29_ns_a0_c41_g1_loss_s4",
        purpose=(
            "A0: only C41 model condition + G1 phase-free gait losses. "
            "Dataset emits S4 so G1 losses can read it, but model does "
            "NOT consume S4 (r29_support_dim=0). Tests whether Stage-2 "
            "needs S4 as a model condition or only as a training target."
        ),
        decision_question=(
            "Compare A0 vs B1: contact/body should match B1, gait should "
            "approach G1 → A0 is the new mainline. Compare A0 vs G1: "
            "if A0 matches G1 on gait, Stage-2 does NOT need S4 as a "
            "consumed condition; S4 stays training-only."
        ),
        # DATA: C41 + I0 + S4 (loss-only) + B0
        r29_coarse_variant="C41-current",
        r29_interaction_variant="I0",
        r29_support_variant="S4-S1-phase-footstep",
        r29_body_variant="B0",
        # MODEL: only C41 consumed
        r29_coarse_extra_dim=18,
        r29_interaction_dim=0,
        r29_support_dim=0,  # ← decouple: S4 in data, NOT in model
        r29_body_refine_dim=0,
        # G1 losses ON
        r29_gait_soft_stance_velocity_weight=0.05,
        r29_gait_transition_rate_weight=0.20,
        r29_gait_duty_cycle_weight=0.10,
        r29_gait_both_state_match_weight=0.10,
        r29_gait_ankle_smooth_weight=0.02,
        r29_gait_antiphase_corr_weight=0.02,
    ),
    # ---------------- A1 — C41 + S4 consumed + G1 losses ----------------
    NextStepVariant(
        variant_id="r29_ns_a1_c41_s4_g1",
        purpose=(
            "A1: C41 + S4 consumed by model + G1 phase-free gait losses. "
            "Isolates whether Stage-2 benefits from S4 as a consumed "
            "condition (vs A0 which only uses S4 as G1 loss target)."
        ),
        decision_question=(
            "Compare A1 vs A0: if A1 >> A0 on gait without contact "
            "regression, Stage-1.5 needs to output an explicit support "
            "schedule. If A1 ≈ A0, S4 can stay loss-only. If A1 improves "
            "gait but regresses contact/body, S4 cond competes with "
            "contact representation — do not promote without further work."
        ),
        # DATA: C41 + I0 + S4 + B0
        r29_coarse_variant="C41-current",
        r29_interaction_variant="I0",
        r29_support_variant="S4-S1-phase-footstep",
        r29_body_variant="B0",
        # MODEL: C41 + S4 consumed
        r29_coarse_extra_dim=18,
        r29_interaction_dim=0,
        r29_support_dim=13,  # ← S4 consumed
        r29_body_refine_dim=0,
        # G1 losses ON (same as A0)
        r29_gait_soft_stance_velocity_weight=0.05,
        r29_gait_transition_rate_weight=0.20,
        r29_gait_duty_cycle_weight=0.10,
        r29_gait_both_state_match_weight=0.10,
        r29_gait_ankle_smooth_weight=0.02,
        r29_gait_antiphase_corr_weight=0.02,
    ),
    # ---------------- H1 — Live I5 upper bound ----------------
    NextStepVariant(
        variant_id="r29_ns_h1_i5_upper_bound",
        purpose=(
            "H1: replaces the invalid old oracle-hint H1 with a live "
            "R29-native I5 (all-part contact + offsets, dim=20) upper "
            "bound. R0 cond with I3 swapped for I5. No contact-lock "
            "losses, no G1 losses — pure condition-content test."
        ),
        decision_question=(
            "Compare H1 vs R0: if H1 improves sustained contact / hand-"
            "foot-pelvis drift without body/gait regression, I3 was too "
            "weak and Stage-1.5 needs richer all-part contact planning. "
            "If H1 does not improve, contact-content upper bound is weak "
            "and representation floor / capacity become more likely."
        ),
        # DATA: C41 + I5 + S4 + B4 (R0 cond with I3 → I5)
        r29_coarse_variant="C41-current",
        r29_interaction_variant="I5-allpart-contact-offset-masked",
        r29_support_variant="S4-S1-phase-footstep",
        r29_body_variant="B4-lowpass-residual-mask",
        # MODEL: matches data — full R0 + I5
        r29_coarse_extra_dim=18,
        r29_interaction_dim=20,  # ← I5
        r29_support_dim=13,
        r29_body_refine_dim=20,
        # No losses beyond baseline absolute.
    ),
    # ---------------- A2 — C41 + I5 + G1 ----------------
    NextStepVariant(
        variant_id="r29_ns_a2_c41_i5_g1",
        purpose=(
            "A2: plausible combined mainline — C41 + I5 (all-part contact) "
            "+ G1 phase-free gait losses, NO S4 consumed, NO B4. Tests "
            "whether the contact half of the upper bound (I5) and the "
            "gait half of G1 combine cleanly."
        ),
        decision_question=(
            "Compare A2 vs H1, A2 vs A0: if A2 ≥ H1 on contact AND ≥ A0 "
            "on gait without regression, A2 is the strongest single-"
            "variant mainline candidate."
        ),
        # DATA: C41 + I5 + S4 (loss-only) + B0
        r29_coarse_variant="C41-current",
        r29_interaction_variant="I5-allpart-contact-offset-masked",
        r29_support_variant="S4-S1-phase-footstep",
        r29_body_variant="B0",
        # MODEL: C41 + I5 consumed; S4 loss-only; no B
        r29_coarse_extra_dim=18,
        r29_interaction_dim=20,  # ← I5
        r29_support_dim=0,        # ← loss-only
        r29_body_refine_dim=0,
        # G1 losses ON
        r29_gait_soft_stance_velocity_weight=0.05,
        r29_gait_transition_rate_weight=0.20,
        r29_gait_duty_cycle_weight=0.10,
        r29_gait_both_state_match_weight=0.10,
        r29_gait_ankle_smooth_weight=0.02,
        r29_gait_antiphase_corr_weight=0.02,
    ),
]


# --------------------------------------------------------------------------- #
# Reference rows (NOT retrained).
# --------------------------------------------------------------------------- #

@dataclass(slots=True)
class ReferenceRow:
    variant_id: str
    purpose: str
    train: bool = False
    valid_for_decision: bool = True
    invalid_reason: str = ""
    # The diag-dir prefix differs between R29-FT (``r29_ft``) and R29-NB
    # (``r29_nb``); the summarizer looks them up by variant_id directly.
    diag_dir_prefix: str = "r29_ft"


REFERENCES: list[ReferenceRow] = [
    ReferenceRow(
        variant_id="r29_ft_r0_clean_a3_baseline",
        purpose=(
            "R0 reference (R29-FT clean baseline): C41 + I3 + S4 + B4 "
            "consumed by model, baseline absolute losses only. Comparison "
            "anchor for all train variants."
        ),
        diag_dir_prefix="r29_ft",
    ),
    ReferenceRow(
        variant_id="r29_nb_b1_c41_only",
        purpose=(
            "B1 reference (previous next-baseline): only C41 consumed by "
            "model, no I/S/B, no G1 losses. Direct A0 contact/body anchor."
        ),
        diag_dir_prefix="r29_nb",
    ),
    ReferenceRow(
        variant_id="r29_nb_g1_phasefree_gait_fixed",
        purpose=(
            "G1 reference (previous next-baseline): R0 cond + G1 phase-free "
            "gait losses. A0/A1/A2 gait anchor."
        ),
        diag_dir_prefix="r29_nb",
    ),
    ReferenceRow(
        variant_id="r29_nb_h1_r0_plus_oracle_full_hint",
        purpose=(
            "INVALID old H1: oracle-hint YAML keys were emitted but NOT "
            "consumed by the current dataset/trainer/model. Effectively "
            "equivalent to R0 with dead YAML. Listed for historical context "
            "ONLY; do not use as a valid contact-content upper bound."
        ),
        valid_for_decision=False,
        invalid_reason=(
            "oracle hint YAML keys (use_oracle_interaction_hint, "
            "oracle_hint_dim, oracle_hint_injection_mode) were not consumed "
            "by dataset/trainer/model; do not use for contact-content verdict"
        ),
        diag_dir_prefix="r29_nb",
    ),
]


# --------------------------------------------------------------------------- #
# Validation — guard against config errors that the trainer cannot detect.
# --------------------------------------------------------------------------- #

def _g1_losses_active(v: NextStepVariant) -> bool:
    return (
        v.r29_gait_soft_stance_velocity_weight > 0.0
        or v.r29_gait_transition_rate_weight > 0.0
        or v.r29_gait_duty_cycle_weight > 0.0
        or v.r29_gait_both_state_match_weight > 0.0
    )


def _validate_variant(v: NextStepVariant) -> None:
    """Per prompt §5 validation block.

    Catches misconfigurations the trainer would only flag at first
    forward pass on the server (e.g. G1 loss enabled but data emits no
    stage2_support, or model interaction_dim=20 but data emits I3).
    """
    # 1. If G1 losses are active, dataset must emit stage2_support (i.e.
    #    data-side support variant must have dim >= 5), even if the model
    #    support dim is 0.
    if _g1_losses_active(v):
        data_support_dim = _SUPPORT_DIMS.get(v.r29_support_variant)
        if data_support_dim is None:
            raise ValueError(
                f"{v.variant_id}: unknown r29_support_variant "
                f"{v.r29_support_variant!r}"
            )
        if data_support_dim < 5:
            raise ValueError(
                f"{v.variant_id}: G1 losses are active but data-side "
                f"r29_support_variant={v.r29_support_variant!r} (dim="
                f"{data_support_dim}) does not surface stage2_support. "
                f"Use an S-family variant with dim>=5 (S1/S2/S3/S4)."
            )
    # 2. Model interaction_dim must match data interaction variant dim.
    data_inter_dim = _INTERACTION_DIMS.get(v.r29_interaction_variant)
    if data_inter_dim is None:
        raise ValueError(
            f"{v.variant_id}: unknown r29_interaction_variant "
            f"{v.r29_interaction_variant!r}"
        )
    if v.r29_interaction_dim not in (0, data_inter_dim):
        raise ValueError(
            f"{v.variant_id}: model r29_interaction_dim={v.r29_interaction_dim} "
            f"is incompatible with data r29_interaction_variant={v.r29_interaction_variant!r} "
            f"(dim={data_inter_dim}). Must be either 0 (model does not consume) "
            f"or {data_inter_dim} (model consumes)."
        )
    # 3. Model coarse_extra_dim must match data coarse variant dim.
    data_coarse_dim = _COARSE_DIMS.get(v.r29_coarse_variant)
    if data_coarse_dim is None:
        raise ValueError(
            f"{v.variant_id}: unknown r29_coarse_variant "
            f"{v.r29_coarse_variant!r}"
        )
    if v.r29_coarse_extra_dim not in (0, data_coarse_dim):
        raise ValueError(
            f"{v.variant_id}: model r29_coarse_extra_dim="
            f"{v.r29_coarse_extra_dim} mismatches data "
            f"r29_coarse_variant={v.r29_coarse_variant!r} (dim={data_coarse_dim})."
        )
    # 4. Model body_refine_dim must match data body variant dim.
    data_body_dim = _BODY_DIMS.get(v.r29_body_variant)
    if data_body_dim is None:
        raise ValueError(
            f"{v.variant_id}: unknown r29_body_variant "
            f"{v.r29_body_variant!r}"
        )
    if v.r29_body_refine_dim not in (0, data_body_dim):
        raise ValueError(
            f"{v.variant_id}: model r29_body_refine_dim="
            f"{v.r29_body_refine_dim} mismatches data "
            f"r29_body_variant={v.r29_body_variant!r} (dim={data_body_dim})."
        )
    # 5. Old R2 one_foot_support must NEVER be enabled (degeneracy source).
    if v.r29_gait_one_foot_support_weight > 0.0:
        raise ValueError(
            f"{v.variant_id}: r29_gait_one_foot_support_weight must be 0; "
            "it is the documented R2 degeneracy source."
        )


def _loss_only_families(v: NextStepVariant) -> list[str]:
    """Families that the dataset emits but the model does NOT consume.
    A0's S4 is the canonical example.
    """
    out: list[str] = []
    if (
        _COARSE_DIMS.get(v.r29_coarse_variant, 0) > 0
        and v.r29_coarse_extra_dim == 0
    ):
        out.append("coarse")
    if (
        _INTERACTION_DIMS.get(v.r29_interaction_variant, 0) > 0
        and v.r29_interaction_dim == 0
    ):
        out.append("interaction")
    if (
        _SUPPORT_DIMS.get(v.r29_support_variant, 0) > 0
        and v.r29_support_dim == 0
    ):
        out.append("support")
    if (
        _BODY_DIMS.get(v.r29_body_variant, 0) > 0
        and v.r29_body_refine_dim == 0
    ):
        out.append("body")
    return out


def _yaml_must_not_contain_oracle_hint_fields(yaml_text: str) -> None:
    """Per prompt §2.4: H1 (and the whole next-step matrix) must NOT
    emit the dead oracle-hint YAML fields anywhere."""
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


def _render_yaml(v: NextStepVariant, *, data_root: str) -> str:
    run_name = f"stageB_anchordiff_{v.variant_id}"
    use_r29 = "true" if v.use_round29_cond_injection else "false"
    yaml = f"""# Round-29 next-step ablation {v.variant_id}: {v.purpose}
#
# Generated by scripts/stage_b_generator/round29_make_next_step_ablation_configs.py.
# Per analyses/2026-05-28_round29_next_step_ablation_execution_prompt_for_claude_code.md.
#
# FULL DATA. No subset_indices_file. From scratch (no init_checkpoint).
# Heldout val. {SCHEDULE_NUM_EPOCHS} ep / val_every=5 / save_every=10 / warmup=250.
# Schedule: bs={SCHEDULE_BATCH_SIZE} accum={SCHEDULE_ACCUM_STEPS} (2× 5080).
#
# Decoupled data vs model: dataset surfaces stage2_<family> based on
# r29_<family>_variant; model consumes only families with r29_<family>_dim > 0.

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
    # R3 exact S4 execution — OFF on this matrix.
    r29_s4_stance_bce_weight: {v.r29_s4_stance_bce_weight}
    r29_s4_footstep_target_weight: {v.r29_s4_footstep_target_weight}
    # G1 phase-free gait losses (A0/A1/A2 only).
    r29_gait_soft_stance_velocity_weight: {v.r29_gait_soft_stance_velocity_weight}
    r29_gait_transition_rate_weight: {v.r29_gait_transition_rate_weight}
    r29_gait_duty_cycle_weight: {v.r29_gait_duty_cycle_weight}
    r29_gait_both_state_match_weight: {v.r29_gait_both_state_match_weight}
    r29_gait_soft_stance_speed_threshold_mps: {v.r29_gait_soft_stance_speed_threshold_mps}
    r29_gait_soft_stance_speed_softness_mps: {v.r29_gait_soft_stance_speed_softness_mps}
    r29_gait_ankle_smooth_weight: {v.r29_gait_ankle_smooth_weight}
    r29_gait_antiphase_corr_weight: {v.r29_gait_antiphase_corr_weight}
    # Old R2 one-foot-support: DISABLED (degeneracy source).
    r29_gait_one_foot_support_weight: 0.0
    r29_gait_pred_stance_velocity_weight: 0.0
    # R4/R5 contact-lock — OFF on this matrix.
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
    # Guard: no dead oracle-hint fields in any generated YAML.
    _yaml_must_not_contain_oracle_hint_fields(yaml)
    return yaml


def _manifest_row_train(v: NextStepVariant) -> dict:
    canonical_cfg = f"configs/training/anchordiff_{v.variant_id}.yaml"
    loss_only = _loss_only_families(v)
    return {
        "variant_id": v.variant_id,
        "group": "NextStepAblation",
        "purpose": v.purpose,
        "decision_question": v.decision_question,
        "train": v.train,
        "valid_for_decision": v.valid_for_decision,
        "invalid_reason": v.invalid_reason,
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
            "loss_only_families": loss_only,
        },
        "architecture": {
            "d_model": v.d_model, "n_layers": v.n_layers,
            "n_heads": v.n_heads, "ff_mult": v.ff_mult,
            "dropout": v.dropout,
        },
        "loss_knobs": {
            "pos_loss_weight": v.pos_loss_weight,
            "hand_endpoint_weight": v.hand_endpoint_weight,
            "foot_endpoint_weight": v.foot_endpoint_weight,
            "anchor_joint_pos_weight": v.anchor_joint_pos_weight,
            "anchor_joint_vel_weight": v.anchor_joint_vel_weight,
            "world_joint_velocity_weight": v.world_joint_velocity_weight,
            # R29 support / swing / S4 — all 0 here unless explicitly turned on.
            "r29_support_both_airborne_weight": v.r29_support_both_airborne_weight,
            "r29_support_stance_velocity_weight": v.r29_support_stance_velocity_weight,
            "r29_swing_clearance_weight": v.r29_swing_clearance_weight,
            "r29_s4_stance_bce_weight": v.r29_s4_stance_bce_weight,
            "r29_s4_footstep_target_weight": v.r29_s4_footstep_target_weight,
            # G1.
            "r29_gait_soft_stance_velocity_weight": v.r29_gait_soft_stance_velocity_weight,
            "r29_gait_transition_rate_weight": v.r29_gait_transition_rate_weight,
            "r29_gait_duty_cycle_weight": v.r29_gait_duty_cycle_weight,
            "r29_gait_both_state_match_weight": v.r29_gait_both_state_match_weight,
            "r29_gait_soft_stance_speed_threshold_mps": v.r29_gait_soft_stance_speed_threshold_mps,
            "r29_gait_soft_stance_speed_softness_mps": v.r29_gait_soft_stance_speed_softness_mps,
            "r29_gait_ankle_smooth_weight": v.r29_gait_ankle_smooth_weight,
            "r29_gait_antiphase_corr_weight": v.r29_gait_antiphase_corr_weight,
            # Old R2 — always 0 on this matrix.
            "r29_gait_one_foot_support_weight": v.r29_gait_one_foot_support_weight,
            # R4/R5 — always 0 on this matrix.
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
        "references": {
            "primary": "r29_ft_r0_clean_a3_baseline",
            "secondary": _secondary_reference(v.variant_id),
        },
    }


def _secondary_reference(variant_id: str) -> str | None:
    """Per prompt §7.2/§7.3 decision logic, certain variants compare
    against a second reference in addition to R0."""
    return {
        "r29_ns_a0_c41_g1_loss_s4": "r29_nb_b1_c41_only",
        "r29_ns_a1_c41_s4_g1": "r29_ns_a0_c41_g1_loss_s4",
        "r29_ns_a2_c41_i5_g1": "r29_ns_a0_c41_g1_loss_s4",
    }.get(variant_id)


def _manifest_row_reference(r: ReferenceRow) -> dict:
    return {
        "variant_id": r.variant_id,
        "group": "NextStepAblation",
        "purpose": r.purpose,
        "decision_question": "Reference baseline (not retrained).",
        "train": r.train,
        "valid_for_decision": r.valid_for_decision,
        "invalid_reason": r.invalid_reason,
        "diag_dir_prefix": r.diag_dir_prefix,
        "config_path": f"configs/training/anchordiff_{r.variant_id}.yaml",
        "output_dir": f"runs/training/stageB_anchordiff_{r.variant_id}",
        "diagnostics": ["sustained_contact", "gait", "body_action"],
    }


# --------------------------------------------------------------------------- #
# Main.
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Round-29 next-step ablation configs (A0/A1/H1/A2) + "
            "manifest. R0/B1/G1 and the invalid old H1 are referenced "
            "(not retrained)."
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
        _validate_variant(v)
        out_path = config_dir / f"anchordiff_{v.variant_id}.yaml"
        content = _render_yaml(v, data_root=args.data_root)
        train_rows.append(_manifest_row_train(v))
        if args.dry_run:
            print(f"DRY-RUN would write: {out_path}  ({len(content)} bytes)")
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(content, encoding="utf-8")
            print(f"wrote {out_path}")

    ref_rows = [_manifest_row_reference(r) for r in REFERENCES]
    all_rows = train_rows + ref_rows

    manifest_json = analyses_dir / "round29_next_step_ablation_manifest.json"
    manifest_md = analyses_dir / "round29_next_step_ablation_manifest.md"

    md_lines: list[str] = [
        "# Round-29 next-step ablation manifest",
        "",
        "Per `analyses/2026-05-28_round29_next_step_ablation_execution_prompt_for_claude_code.md`:",
        "4 train variants (A0/A1/H1/A2) on top of R0/B1/G1 references. The old "
        "`r29_nb_h1_r0_plus_oracle_full_hint` is listed for historical context "
        "only and marked **valid_for_decision=false** because its oracle-hint "
        "YAML keys were not consumed by the trainer.",
        "",
        f"All 4 new variants: FULL InterAct train set (no subset_indices_file), "
        f"from scratch (no init_checkpoint), {SCHEDULE_NUM_EPOCHS} ep, heldout val, "
        f"save_every=10, warmup=250, stage1_coarse_noise_std=0.05, "
        f"bs={SCHEDULE_BATCH_SIZE} / accum={SCHEDULE_ACCUM_STEPS} (2× 5080).",
        "",
        "Decoupled data vs model: dataset surfaces `stage2_<family>` whenever "
        "`r29_<family>_variant` has dim > 0; model consumes only families with "
        "`r29_<family>_dim > 0`. A0 uses S4 as loss-only (data emits, model "
        "ignores) — verified at:",
        "",
        "- `src/piano/data/dataset.py:710-721` (data emits)",
        "- `src/piano/training/train_anchordiff.py:414-421` (trainer forwards)",
        "- `src/piano/models/round29_cond_injection.py:269` (model iterates "
        "  only active_families — dim=0 is not active)",
        "",
        "## Train variants",
        "",
        "| variant | C (data/model) | I (data/model) | S (data/model) | B (data/model) | G1 losses | primary ref | secondary ref |",
        "| --- | --- | --- | --- | --- | :---: | --- | --- |",
    ]
    for r in train_rows:
        d = r["condition"]["data_variants"]; m = r["condition"]["model_dims"]
        g1_on = any(
            float(r["loss_knobs"].get(k, 0.0)) > 0.0
            for k in (
                "r29_gait_soft_stance_velocity_weight",
                "r29_gait_transition_rate_weight",
                "r29_gait_duty_cycle_weight",
                "r29_gait_both_state_match_weight",
            )
        )
        refs = r["references"]
        primary_ref = refs["primary"]
        sec_ref = refs.get("secondary")
        sec_cell = f"`{sec_ref}`" if sec_ref else "—"
        g1_cell = "✓" if g1_on else "—"
        md_lines.append(
            f"| `{r['variant_id']}` | "
            f"{d['r29_coarse_variant']}/{m['r29_coarse_extra_dim']} | "
            f"{d['r29_interaction_variant']}/{m['r29_interaction_dim']} | "
            f"{d['r29_support_variant']}/{m['r29_support_dim']} | "
            f"{d['r29_body_variant']}/{m['r29_body_refine_dim']} | "
            f"{g1_cell} | "
            f"`{primary_ref}` | "
            f"{sec_cell} |"
        )
    md_lines += [
        "",
        "## Reference rows (not retrained)",
        "",
        "| variant | source matrix | valid for decision? |",
        "| --- | --- | :---: |",
    ]
    for r in ref_rows:
        valid_str = "✓" if r["valid_for_decision"] else "❌ INVALID"
        md_lines.append(
            f"| `{r['variant_id']}` | {r['diag_dir_prefix']} | {valid_str} |"
        )
        if not r["valid_for_decision"]:
            md_lines.append(f"|     | {r['invalid_reason']} |  |")
    md_lines += ["", "## Decision questions", ""]
    for r in train_rows:
        md_lines.append(f"- **`{r['variant_id']}`**: {r['decision_question']}")
    md_lines += ["", "## Loss-knob activations (only non-zero weights per variant)", ""]
    for r in train_rows:
        active = {k: v for k, v in r["loss_knobs"].items() if float(v) != 0.0}
        md_lines.append(f"### `{r['variant_id']}`")
        md_lines.append("")
        if not active:
            md_lines.append("- (baseline absolute losses only)")
        for k, v in active.items():
            md_lines.append(f"- `{k}` = {v}")
        md_lines.append("")
    md_lines += [
        "## Loss-only families per variant",
        "",
        "Families where the dataset emits `stage2_<family>` (data variant has "
        "dim > 0) but the model does NOT consume them (model dim = 0). The "
        "trainer's R29 cond injection module ignores these keys; G1 losses "
        "or future loss-target consumers can still read them from `cond`.",
        "",
    ]
    for r in train_rows:
        families = r["condition"]["loss_only_families"]
        if families:
            md_lines.append(f"- `{r['variant_id']}`: {', '.join(families)}")
        else:
            md_lines.append(f"- `{r['variant_id']}`: (none — model consumes "
                            f"every family the dataset emits)")
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
    n_train = sum(1 for r in all_rows if r.get("train", True))
    n_ref = sum(1 for r in all_rows if not r.get("train", True))
    n_invalid = sum(
        1 for r in all_rows if not r.get("valid_for_decision", True)
    )
    print(
        f"\n{n_train} new train variant(s) + {n_ref} reference(s) "
        f"({n_invalid} marked INVALID)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
