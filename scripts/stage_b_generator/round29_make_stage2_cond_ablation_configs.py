"""Generate Round-29 Stage-2 condition + injection ablation configs.

Per analyses/2026-05-26_stage2_cond_injection_ablation_claude_code_prompt.md
§5.4 + Codex post-review fixes from
analyses/2026-05-26_round29_claude_code_fix_after_codex_review_prompt.md
(portable paths, configurable init-checkpoint, real F3/F4 overrides,
emit repo-relative paths only).

Writes:

    <config-dir>/anchordiff_r29_<variant_id>.yaml          (one per variant)
    <analyses-dir>/round29_stage2_cond_ablation_manifest.json
    <analyses-dir>/round29_stage2_cond_ablation_manifest.md

The manifest is the source of truth consumed by:

    scripts/stage_b_generator/round29_summarize_stage2_cond_ablation.py
    scripts/stage_b_generator/round29_stage2_cond_smoke_test.py
    scripts/stage_b_generator/run_round29_stage2_cond_ablation.{sh,py}

Reviewed prompt section 5.4 before implementing this group: yes.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# We import dim tables from the source-of-truth library so the manifest's
# expected_dense_dims values cannot drift from the dataset's actual output.
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from piano.data.stage2_oracle_conditions import (  # noqa: E402
    BODY_VARIANT_DIMS,
    COARSE_VARIANT_DIMS,
    INTERACTION_VARIANT_DIMS,
    SUPPORT_VARIANT_DIMS,
)
from piano.models.round29_cond_injection import (  # noqa: E402
    VALID_INJECTION_MODES,
)

DEFAULT_CONFIG_DIR = ROOT / "configs" / "training"
DEFAULT_ANALYSES_DIR = ROOT / "analyses"

# Default checkpoint path — relative to repo root. The CLI override
# (--init-checkpoint) and ROUND29_INIT_CKPT env var let users point at a
# real on-disk checkpoint without editing this file. The launcher's
# preflight verifies the path actually exists before training.
DEFAULT_INIT_CKPT = (
    "runs/training/stageB_anchordiff_v27_stage2_anchoraware_FULL_DATA/final.pt"
)
# Selection JSON used by BOTH the trainer (via `data.subset_indices_file`,
# which reads the `indices: [int]` field) AND the diag scripts (via
# `--selection-json`, which after 2026-05-26 falls back to the `clips:
# [{subset, seq_id}]` field that the train_indices builder also emits).
# So one file per subset is enough; the dual-path split that lived here
# briefly was unnecessary.
DEFAULT_SUBSET_FILE = "analyses/round27_tier0_train_indices_48_balanced.json"
BODY_ACTION_SUBSET_FILE = "analyses/round28_body_action_train_indices_48.json"

# Default dataset root for THIS dev machine (Windows). The Linux server
# overrides via --data-root or DATASETS_ROOT env so the generated YAML
# carries the correct on-disk paths and we don't need a separate
# _local.yaml second-copy. Subset names below are appended to this root.
DEFAULT_DATA_ROOT = "E:/Project/Datasets/InterAct/piano_official_process_4"
DATASET_SUBSET_NAMES: tuple[str, ...] = (
    "chairs", "imhd", "neuraldome", "omomo_correct_v2",
)

# Default condition for "FULL-DENSE" (prompt §3.5).
FULL_DENSE = dict(
    coarse_variant="C41-current",
    interaction_variant="I3-contact-offset-masked",
    support_variant="S4-S1-phase-footstep",
    body_variant="B4-lowpass-residual-mask",
)


@dataclass(slots=True)
class Variant:
    variant_id: str
    group: str
    purpose: str
    coarse_variant: str = "C23"
    interaction_variant: str = "I0"
    support_variant: str = "S0"
    body_variant: str = "B0"
    injection_mode: str = "input_add"
    gate_bias_init: float = -1.0
    per_family_modes: dict[str, str] | None = None
    subset_kind: str = "balanced"           # "balanced" or "body_action"
    num_epochs: int = 300
    # Codex review §P1 — F3/F4 robustness variants must actually differ.
    seed: int = 42
    val_on_train_subset: bool = True
    diagnostics: tuple[str, ...] = field(default_factory=lambda: (
        "sustained_contact", "gait", "body_action",
    ))


# ---------------------------------------------------------------------------
# Group A — injection ablation with FULL-DENSE content (§6.1)
# ---------------------------------------------------------------------------

GROUP_A_VARIANTS: list[Variant] = [
    Variant(
        variant_id="r29_a0_input_add",
        group="A_injection",
        purpose=(
            "FULL-DENSE oracle content, J0 input-add injection. Baseline "
            "for the injection comparison."
        ),
        **FULL_DENSE,
        injection_mode="input_add",
    ),
    Variant(
        variant_id="r29_a1_gated_input_open",
        group="A_injection",
        purpose=(
            "FULL-DENSE content, J1 gated_input with gate_bias=-1.0 "
            "(open-ish init per prompt §4.2). Tests whether a learned "
            "gate routes the dense condition selectively."
        ),
        **FULL_DENSE,
        injection_mode="gated_input",
        gate_bias_init=-1.0,
    ),
    Variant(
        variant_id="r29_a2_adapter_only",
        group="A_injection",
        purpose=(
            "FULL-DENSE content, J2 adapter-only: pure per-DiT-block "
            "adapters, NO input-token add."
        ),
        **FULL_DENSE,
        injection_mode="adapter_only",
    ),
    Variant(
        variant_id="r29_a3_input_add_adapter",
        group="A_injection",
        purpose=(
            "FULL-DENSE content, J3 input-add + per-DiT-block adapter "
            "(input-add baseline plus deep refinement)."
        ),
        **FULL_DENSE,
        injection_mode="input_add_adapter",
    ),
    Variant(
        variant_id="r29_a4_typed",
        group="A_injection",
        purpose=(
            "FULL-DENSE content, J4 typed injection (separate methods "
            "per family per prompt §4.5): coarse=input_add, "
            "interaction=adapter_only, support=adapter_only, "
            "body_refine=input_add_adapter. J4-dense variant — token "
            "cross-attention NOT implemented (would require event-token "
            "encoder; see prompt §4.5 final paragraph)."
        ),
        **FULL_DENSE,
        injection_mode="typed",
        per_family_modes={
            "coarse_extra": "input_add",
            "interaction": "adapter_only",
            "support": "adapter_only",
            "body_refine": "input_add_adapter",
        },
    ),
]


# ---------------------------------------------------------------------------
# Group B — coarse scaffold content + coordinate frame (§6.2)
# ---------------------------------------------------------------------------

GROUP_B_VARIANTS: list[Variant] = [
    Variant(
        variant_id="r29_b0_c23_only",
        group="B_coarse",
        purpose="C23 only (Stage-1 23-D Coarse-v1 baseline, no extra channels).",
        coarse_variant="C23",
        injection_mode="input_add",
    ),
    Variant(
        variant_id="r29_b1_c38_current_only",
        group="B_coarse",
        purpose="C38-current only (5 key joints × 3D, per-frame pelvis-local).",
        coarse_variant="C38-current",
        injection_mode="input_add",
    ),
    Variant(
        variant_id="r29_b2_c41_current_only",
        group="B_coarse",
        purpose="C41-current only (C38-current + pelvis_delta in root0 frame).",
        coarse_variant="C41-current",
        injection_mode="input_add",
    ),
    Variant(
        variant_id="r29_b3_c38_root0_only",
        group="B_coarse",
        purpose="C38-root0 only (5 key joints × 3D, root0-yaw canonical).",
        coarse_variant="C38-root0",
        injection_mode="input_add",
    ),
    Variant(
        variant_id="r29_b4_c41_root0_only",
        group="B_coarse",
        purpose="C41-root0 only (C38-root0 + pelvis_delta in root0 frame).",
        coarse_variant="C41-root0",
        injection_mode="input_add",
    ),
    Variant(
        variant_id="r29_b5_bestC_full_ISB",
        group="B_coarse",
        purpose=(
            "Best coarse (set by --best-coarse) with FULL I/S/B + best "
            "injection (set by --best-injection). Tests whether coarse "
            "improvement persists when the other three families fire."
        ),
        coarse_variant="best",
        interaction_variant="I3-contact-offset-masked",
        support_variant="S4-S1-phase-footstep",
        body_variant="B4-lowpass-residual-mask",
        injection_mode="best",
    ),
    Variant(
        variant_id="r29_b6_secondBestC_full_ISB",
        group="B_coarse",
        purpose=(
            "Second-best coarse (--second-best-coarse) with FULL I/S/B "
            "+ best injection. Robustness check on the §6.2 winner."
        ),
        coarse_variant="second_best",
        interaction_variant="I3-contact-offset-masked",
        support_variant="S4-S1-phase-footstep",
        body_variant="B4-lowpass-residual-mask",
        injection_mode="best",
    ),
]


# ---------------------------------------------------------------------------
# Group C — interaction content (§6.3)
# ---------------------------------------------------------------------------

GROUP_C_VARIANTS: list[Variant] = [
    Variant(
        variant_id="r29_c0_i0_none",
        group="C_interaction",
        purpose="Best C, no interaction hint. Anchor for the I-axis.",
        coarse_variant="best",
        interaction_variant="I0",
        injection_mode="best",
    ),
    Variant(
        variant_id="r29_c1_i1_contact",
        group="C_interaction",
        purpose="Best C + I1 (hand_contact_prob only, 2D).",
        coarse_variant="best",
        interaction_variant="I1-contact",
        injection_mode="best",
    ),
    Variant(
        variant_id="r29_c2_i2_offset_masked",
        group="C_interaction",
        purpose="Best C + I2 (offset masked, 6D).",
        coarse_variant="best",
        interaction_variant="I2-offset-masked",
        injection_mode="best",
    ),
    Variant(
        variant_id="r29_c3_i3_contact_offset_masked",
        group="C_interaction",
        purpose="Best C + I3 (contact + masked offset, 8D — R28-like).",
        coarse_variant="best",
        interaction_variant="I3-contact-offset-masked",
        injection_mode="best",
    ),
    Variant(
        variant_id="r29_c4_i4_contact_offset_unmasked",
        group="C_interaction",
        purpose="Best C + I4 (contact + unmasked offset, 8D — pre-contact approach test).",
        coarse_variant="best",
        interaction_variant="I4-contact-offset-unmasked",
        injection_mode="best",
    ),
]


# ---------------------------------------------------------------------------
# Group D — support/gait content (§6.4)
# ---------------------------------------------------------------------------

GROUP_D_VARIANTS: list[Variant] = [
    Variant(
        variant_id="r29_d0_s0_none",
        group="D_support",
        purpose="Best C, no support hint.",
        coarse_variant="best",
        support_variant="S0",
        injection_mode="best",
    ),
    Variant(
        variant_id="r29_d1_s1_stance_height_walking",
        group="D_support",
        purpose="Best C + S1 (stance + height + walking, 5D).",
        coarse_variant="best",
        support_variant="S1-stance-height-walking",
        injection_mode="best",
    ),
    Variant(
        variant_id="r29_d2_s2_s1_phase",
        group="D_support",
        purpose="Best C + S2 (S1 + foot phase sincos, 9D).",
        coarse_variant="best",
        support_variant="S2-S1-phase",
        injection_mode="best",
    ),
    Variant(
        variant_id="r29_d3_s3_s1_footstep_target",
        group="D_support",
        purpose="Best C + S3 (S1 + footstep target XZ, 9D).",
        coarse_variant="best",
        support_variant="S3-S1-footstep-target",
        injection_mode="best",
    ),
    Variant(
        variant_id="r29_d4_s4_s1_phase_footstep",
        group="D_support",
        purpose="Best C + S4 (S1 + phase + footstep, 13D).",
        coarse_variant="best",
        support_variant="S4-S1-phase-footstep",
        injection_mode="best",
    ),
    Variant(
        variant_id="r29_d5_bestI_bestS",
        group="D_support",
        purpose=(
            "Best C + best I (from Group C) + best S (the winner of D0..D4). "
            "Tests gait/interaction interaction."
        ),
        coarse_variant="best",
        interaction_variant="best",
        support_variant="best",
        injection_mode="best",
    ),
]


# ---------------------------------------------------------------------------
# Group E — body refinement content (§6.5)
# ---------------------------------------------------------------------------

GROUP_E_VARIANTS: list[Variant] = [
    Variant(
        variant_id="r29_e0_b0_none",
        group="E_body",
        purpose="Best C, no body refinement.",
        coarse_variant="best",
        body_variant="B0",
        injection_mode="best",
        subset_kind="body_action",
    ),
    Variant(
        variant_id="r29_e1_b1_mask_only",
        group="E_body",
        purpose="Best C + B1 (active key-joint mask, 5D).",
        coarse_variant="best",
        body_variant="B1-mask-only",
        injection_mode="best",
        subset_kind="body_action",
    ),
    Variant(
        variant_id="r29_e2_b2_absolute_delta",
        group="E_body",
        purpose="Best C + B2 (absolute key-joint delta, 15D — may duplicate coarse).",
        coarse_variant="best",
        body_variant="B2-absolute-delta",
        injection_mode="best",
        subset_kind="body_action",
    ),
    Variant(
        variant_id="r29_e3_b3_lowpass_residual",
        group="E_body",
        purpose=(
            "Best C + B3 (low-pass residual, 15D). NOTE per Codex review: "
            "this variant runs B3 as a residual SIDE-CHANNEL on top of the "
            "selected C variant; it does NOT replace C with lowpass(delta). "
            "A pure Stage-1-lowpass / Stage-1.5-residual factorization is "
            "deferred (see oracle_conditions.py B3/B4 docstring + report)."
        ),
        coarse_variant="best",
        body_variant="B3-lowpass-residual",
        injection_mode="best",
        subset_kind="body_action",
    ),
    Variant(
        variant_id="r29_e4_b4_lowpass_residual_mask",
        group="E_body",
        purpose=(
            "Best C + B4 (mask + low-pass residual, 20D). Same residual-"
            "side-channel caveat as B3 (see oracle_conditions.py docstring)."
        ),
        coarse_variant="best",
        body_variant="B4-lowpass-residual-mask",
        injection_mode="best",
        subset_kind="body_action",
    ),
    Variant(
        variant_id="r29_e5_bestI_bestB",
        group="E_body",
        purpose="Best C + best I + best B (cross-family).",
        coarse_variant="best",
        interaction_variant="best",
        body_variant="best",
        injection_mode="best",
        subset_kind="body_action",
    ),
    Variant(
        variant_id="r29_e6_bestS_bestB",
        group="E_body",
        purpose="Best C + best S + best B.",
        coarse_variant="best",
        support_variant="best",
        body_variant="best",
        injection_mode="best",
        subset_kind="body_action",
    ),
    Variant(
        variant_id="r29_e7_bestI_bestS_bestB",
        group="E_body",
        purpose="Best C + best I + best S + best B (FULL combined, body-action subset).",
        coarse_variant="best",
        interaction_variant="best",
        support_variant="best",
        body_variant="best",
        injection_mode="best",
        subset_kind="body_action",
    ),
]


# ---------------------------------------------------------------------------
# Group F — final combination + robustness (§6.6)
# ---------------------------------------------------------------------------

GROUP_F_VARIANTS: list[Variant] = [
    Variant(
        variant_id="r29_f0_baseline",
        group="F_final",
        purpose="C23-only baseline reproduction (the §6.6 anchor).",
        coarse_variant="C23",
        injection_mode="input_add",
    ),
    Variant(
        variant_id="r29_f1_best_full",
        group="F_final",
        purpose="Best C + best I + best S + best B + best injection.",
        coarse_variant="best",
        interaction_variant="best",
        support_variant="best",
        body_variant="best",
        injection_mode="best",
    ),
    Variant(
        variant_id="r29_f2_minimal_near_best",
        group="F_final",
        purpose=(
            "Minimal-near-best — set by --minimal-near-best as a comma-"
            "separated family list, e.g. coarse=best,interaction=best."
        ),
        coarse_variant="best",
        interaction_variant="best",
        support_variant="S0",
        body_variant="B0",
        injection_mode="best",
    ),
    Variant(
        variant_id="r29_f3_best_full_heldout",
        group="F_final",
        purpose=(
            "F1 re-run with TRUE held-out validation "
            "(val_on_train_subset=false). Uses the 15% subject-split "
            "validation bucket, separate from the 48-clip train indices."
        ),
        coarse_variant="best",
        interaction_variant="best",
        support_variant="best",
        body_variant="best",
        injection_mode="best",
        val_on_train_subset=False,
    ),
    Variant(
        variant_id="r29_f4_best_full_seed2",
        group="F_final",
        purpose=(
            "F1 re-run with training seed=43 (variance check; "
            "subject_split.seed stays at 42 so the indexed population "
            "is identical to F1)."
        ),
        coarse_variant="best",
        interaction_variant="best",
        support_variant="best",
        body_variant="best",
        injection_mode="best",
        seed=43,
    ),
]


ALL_VARIANTS: list[Variant] = (
    GROUP_A_VARIANTS
    + GROUP_B_VARIANTS
    + GROUP_C_VARIANTS
    + GROUP_D_VARIANTS
    + GROUP_E_VARIANTS
    + GROUP_F_VARIANTS
)


# ---------------------------------------------------------------------------
# Resolution + rendering
# ---------------------------------------------------------------------------

def _resolve_variant(
    v: Variant,
    *,
    best_coarse: str,
    second_best_coarse: str,
    best_interaction: str,
    best_support: str,
    best_body: str,
    best_injection: str,
) -> dict[str, Any]:
    cv = v.coarse_variant
    if cv == "best":
        cv = best_coarse
    elif cv == "second_best":
        cv = second_best_coarse
    iv = v.interaction_variant if v.interaction_variant != "best" else best_interaction
    sv = v.support_variant if v.support_variant != "best" else best_support
    bv = v.body_variant if v.body_variant != "best" else best_body
    inj = v.injection_mode if v.injection_mode != "best" else best_injection

    return {
        "coarse_variant": cv,
        "interaction_variant": iv,
        "support_variant": sv,
        "body_variant": bv,
        "injection_mode": inj,
    }


def _render_per_family_modes_yaml(pfm: dict[str, str] | None) -> str:
    if pfm is None:
        return "null"
    lines = []
    for k, val in pfm.items():
        lines.append(f"      {k}: \"{val}\"")
    return "\n" + "\n".join(lines)


def _render_datasets_block(data_root: str) -> str:
    """Render the data.datasets YAML list pointing at <data_root>/<subset>."""
    rstripped = data_root.rstrip("/").rstrip("\\")
    lines: list[str] = []
    for sub in DATASET_SUBSET_NAMES:
        lines.append(f'    - name: "{sub}"')
        lines.append(f'      root: "{rstripped}/{sub}"')
    return "\n".join(lines)


def _render_yaml(
    v: Variant,
    resolved: dict[str, Any],
    *,
    base_subset_file: str,
    body_action_subset_file: str,
    init_checkpoint: str,
    data_root: str,
) -> str:
    subset_file = (
        body_action_subset_file if v.subset_kind == "body_action"
        else base_subset_file
    )
    coarse_extra_dim = COARSE_VARIANT_DIMS[resolved["coarse_variant"]]
    interaction_dim = INTERACTION_VARIANT_DIMS[resolved["interaction_variant"]]
    support_dim = SUPPORT_VARIANT_DIMS[resolved["support_variant"]]
    body_refine_dim = BODY_VARIANT_DIMS[resolved["body_variant"]]
    any_active = any(
        d > 0 for d in (
            coarse_extra_dim, interaction_dim, support_dim, body_refine_dim,
        )
    )
    run_name = f"stageB_anchordiff_{v.variant_id}"
    per_family_yaml = _render_per_family_modes_yaml(v.per_family_modes)
    val_on_train_subset_yaml = "true" if v.val_on_train_subset else "false"
    return f"""# Round-29 {v.variant_id}: {v.purpose}
#
# Generated by scripts/stage_b_generator/round29_make_stage2_cond_ablation_configs.py.
# Per analyses/2026-05-26_stage2_cond_injection_ablation_claude_code_prompt.md
# + analyses/2026-05-26_round29_claude_code_fix_after_codex_review_prompt.md.

model:
  cfg_drop_prob: 0.15
  diffusion:
    num_steps: 1000
    schedule: "cosine"
    prediction_target: "x0"
  z_int:
    num_parts: 5
    phase_classes: 3
    support_classes: 3
  denoiser:
    motion_dim: 135
    object_traj_dim: 9
    init_pose_dim: 66
    text_dim: 512
    object_token_dim: 256
    object_num_tokens: 128
    use_interaction_plan: true
    plan_k_max: 12
    plan_s_max: 12
    plan_num_anchor_types: 5
    plan_num_parts: 5
    plan_use_segment_tokens: false
    plan_use_context_hint: true
    plan_d_hint: 32
    plan_d_time_embed: 64
    cfg_drop_plan: false
    plan_per_part_tokens: true
    plan_context_hint_mode: "target_aware"
    use_dit_block: true
    dit_block_use_plan_pool_in_cond: false
    d_model: 512
    n_layers: 8
    n_heads: 4
    ff_mult: 4
    dropout: 0.1
    stage1_coarse_dim: 23
    cfg_drop_stage1_coarse: false
    plan_xattn_relative_time_bias: true
    plan_xattn_time_bias_init: 0.5
    plan_tokens_force_null: true
    # Round-28 hint paths are DISABLED — R29 uses the typed bundle path.
    use_oracle_interaction_hint: false
    oracle_hint_dim: 0
    use_body_action_hint: false
    body_action_hint_dim: 0
    oracle_hint_injection_mode: "input_add"
    # Round-29 typed condition injection.
    use_round29_cond_injection: {"true" if any_active else "false"}
    r29_coarse_extra_dim: {coarse_extra_dim}
    r29_interaction_dim: {interaction_dim}
    r29_support_dim: {support_dim}
    r29_body_refine_dim: {body_refine_dim}
    r29_injection_mode: "{resolved['injection_mode']}"
    r29_gate_bias_init: {v.gate_bias_init}
    r29_per_family_modes: {per_family_yaml}
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
  subset_indices_file: "{subset_file}"
  use_oracle_interaction_hint: false
  use_body_action_hint: false
  surface_temporal_aux_fields: true
  # Round-29 typed bundle controls.
  r29_coarse_variant: "{resolved['coarse_variant']}"
  r29_interaction_variant: "{resolved['interaction_variant']}"
  r29_support_variant: "{resolved['support_variant']}"
  r29_body_variant: "{resolved['body_variant']}"
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
  init_checkpoint: "{init_checkpoint}"
  partial_init_allow_shape_mismatch: true
  batch_size: 8
  num_epochs: {v.num_epochs}
  num_workers: 4
  seed: {v.seed}
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
  val_on_train_subset: {val_on_train_subset_yaml}
  val_every_epochs: 50
  val_best_key: "loss_anchor_joint_pos"

loss:
  anchor_weight: 0.0
  contact_threshold: 0.5
  plan_anchor_weight: 0.0
  plan_segment_weight: 0.0
  plan_transition_vel_weight: 0.0
  plan_transition_acc_weight: 0.0
  plan_transition_window: 3
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
  temporal_interaction:
    contact_rel_offset_weight: 0.0
    contact_drift_weight: 0.0
    contact_tracking_weight: 0.0
    gait_both_airborne_weight: 0.0
    gait_stance_velocity_weight: 0.0
    hint_contact_consistency_weight: 0.0
    body_action_consistency_weight: 0.0
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
  log_every_n_steps: 10
  save_every_n_epochs: 50

output_dir: "runs/training/{run_name}"
"""


# ---------------------------------------------------------------------------
# Manifest writers
# ---------------------------------------------------------------------------

def _manifest_row(
    v: Variant,
    resolved: dict[str, Any],
    *,
    config_path: str,
    base_subset_file: str,
    body_action_subset_file: str,
    init_checkpoint: str,
) -> dict[str, Any]:
    cv = resolved["coarse_variant"]
    iv = resolved["interaction_variant"]
    sv = resolved["support_variant"]
    bv = resolved["body_variant"]
    inj = resolved["injection_mode"]
    is_body = v.subset_kind == "body_action"
    subset_file = (
        body_action_subset_file if is_body else base_subset_file
    )
    return {
        "variant_id": v.variant_id,
        "group": v.group,
        "purpose": v.purpose,
        "coarse_variant": cv,
        "interaction_variant": iv,
        "support_variant": sv,
        "body_variant": bv,
        "injection_mode": inj,
        "gate_bias_init": v.gate_bias_init,
        "per_family_modes": v.per_family_modes,
        "expected_dense_dims": {
            "coarse_extra": COARSE_VARIANT_DIMS[cv],
            "interaction": INTERACTION_VARIANT_DIMS[iv],
            "support": SUPPORT_VARIANT_DIMS[sv],
            "body_refine": BODY_VARIANT_DIMS[bv],
        },
        "subset_kind": v.subset_kind,
        # Single subset file shared by trainer (reads `indices`) AND
        # diag scripts (read `clips`/`selected`/`candidates`).
        "subset_file": subset_file,
        "num_epochs": v.num_epochs,
        # Codex review: per-variant training overrides exposed in the
        # manifest so summarizer/launcher can show what was actually run.
        "seed": v.seed,
        "val_on_train_subset": v.val_on_train_subset,
        "init_checkpoint": init_checkpoint,
        "config_path": config_path,
        "output_dir": f"runs/training/stageB_anchordiff_{v.variant_id}",
        "diagnostics": list(v.diagnostics),
    }


def _render_manifest_md(rows: list[dict[str, Any]]) -> str:
    L: list[str] = [
        "# Round-29 Stage-2 condition + injection ablation manifest",
        "",
        "Auto-generated by "
        "`scripts/stage_b_generator/round29_make_stage2_cond_ablation_configs.py`.",
        "Source-of-truth = `round29_stage2_cond_ablation_manifest.json`.",
        "",
        "| group | variant | C | I | S | B | inject | dims (C+I+S+B) | subset | seed | heldout | epochs |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        dims = r["expected_dense_dims"]
        d_str = (
            f"{dims['coarse_extra']}+{dims['interaction']}+"
            f"{dims['support']}+{dims['body_refine']}"
        )
        heldout = "✓" if not r["val_on_train_subset"] else " "
        L.append(
            f"| {r['group']} | {r['variant_id']} | "
            f"{r['coarse_variant']} | {r['interaction_variant']} | "
            f"{r['support_variant']} | {r['body_variant']} | "
            f"{r['injection_mode']} | {d_str} | {r['subset_kind']} | "
            f"{r['seed']} | {heldout} | {r['num_epochs']} |"
        )
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _to_posix_relpath(p: Path | str, root: Path) -> str:
    """Return a portable repo-relative POSIX path. Absolute or out-of-root
    paths are returned unchanged (e.g. external data roots on Linux)."""
    s = str(p)
    if not s:
        return s
    p_obj = Path(s)
    try:
        rel = p_obj.resolve().relative_to(root.resolve())
        return rel.as_posix()
    except (ValueError, OSError):
        # Already relative or not under root.
        return p_obj.as_posix() if not p_obj.is_absolute() else s


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Round-29 Stage-2 condition + injection ablation configs.",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be written without touching disk.")
    parser.add_argument(
        "--best-coarse",
        default="C41-current",
        choices=sorted(COARSE_VARIANT_DIMS),
        help="Best coarse variant (from Group B). Substituted into 'best'.",
    )
    parser.add_argument(
        "--second-best-coarse",
        default="C38-current",
        choices=sorted(COARSE_VARIANT_DIMS),
        help="Second-best coarse for r29_b6_secondBestC_full_ISB.",
    )
    parser.add_argument(
        "--best-interaction",
        default="I3-contact-offset-masked",
        choices=sorted(INTERACTION_VARIANT_DIMS),
        help="Best I variant (from Group C). Substituted into 'best'.",
    )
    parser.add_argument(
        "--best-support",
        default="S4-S1-phase-footstep",
        choices=sorted(SUPPORT_VARIANT_DIMS),
        help="Best S variant (from Group D). Substituted into 'best'.",
    )
    parser.add_argument(
        "--best-body",
        default="B4-lowpass-residual-mask",
        choices=sorted(BODY_VARIANT_DIMS),
        help="Best B variant (from Group E). Substituted into 'best'.",
    )
    parser.add_argument(
        "--best-injection",
        default="input_add_adapter",
        choices=sorted(set(VALID_INJECTION_MODES) - {"typed"}),
        help="Best injection mode (from Group A). Substituted into 'best'.",
    )
    parser.add_argument(
        "--balanced-subset-file", default=DEFAULT_SUBSET_FILE,
        help=(
            "Selection JSON for contact+gait variants. Consumed by "
            "the trainer (reads `indices: [int]`) AND the diag scripts "
            "(read `clips: [{subset, seq_id}]`). Generated by "
            "round27_build_tier0_train_indices.py."
        ),
    )
    parser.add_argument(
        "--body-action-subset-file", default=BODY_ACTION_SUBSET_FILE,
        help=(
            "Selection JSON for body-action variants. Same dual-purpose "
            "as --balanced-subset-file. Generated by "
            "round28_build_body_action_subset.py."
        ),
    )
    parser.add_argument(
        "--init-checkpoint",
        default=os.environ.get("ROUND29_INIT_CKPT", DEFAULT_INIT_CKPT),
        help=(
            "Init checkpoint path written into every config. Defaults to "
            "ROUND29_INIT_CKPT env var or the v27 final.pt path. The "
            "launcher's preflight verifies the file actually exists."
        ),
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("DATASETS_ROOT", DEFAULT_DATA_ROOT),
        help=(
            "Root directory containing the four InterAct subsets "
            "(chairs/imhd/neuraldome/omomo_correct_v2). Defaults to "
            "DATASETS_ROOT env var or the dev-machine Windows path. "
            "On the Linux server pass --data-root "
            "/media/.../InterAct/piano_official_process_4 (or export "
            "DATASETS_ROOT=...) so the generated YAML carries the "
            "correct on-disk paths."
        ),
    )
    parser.add_argument(
        "--config-dir",
        default=str(DEFAULT_CONFIG_DIR),
        help="Directory where generated YAML configs are written.",
    )
    parser.add_argument(
        "--analyses-dir",
        default=str(DEFAULT_ANALYSES_DIR),
        help="Directory where the manifest JSON+MD are written.",
    )
    parser.add_argument(
        "--only-groups", default="",
        help="Comma-separated group names to emit (e.g. 'A_injection,B_coarse'). Empty = all.",
    )
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    analyses_dir = Path(args.analyses_dir)

    only = (
        set(g.strip() for g in args.only_groups.split(",") if g.strip())
        if args.only_groups else None
    )

    init_ckpt = _to_posix_relpath(args.init_checkpoint, ROOT)
    base_subset = _to_posix_relpath(args.balanced_subset_file, ROOT)
    body_subset = _to_posix_relpath(args.body_action_subset_file, ROOT)

    rows: list[dict[str, Any]] = []
    written = 0
    for v in ALL_VARIANTS:
        if only is not None and v.group not in only:
            continue
        resolved = _resolve_variant(
            v,
            best_coarse=args.best_coarse,
            second_best_coarse=args.second_best_coarse,
            best_interaction=args.best_interaction,
            best_support=args.best_support,
            best_body=args.best_body,
            best_injection=args.best_injection,
        )
        out_path = config_dir / f"anchordiff_{v.variant_id}.yaml"
        content = _render_yaml(
            v, resolved,
            base_subset_file=base_subset,
            body_action_subset_file=body_subset,
            init_checkpoint=init_ckpt,
            data_root=args.data_root,
        )
        # The manifest always records the canonical repo-relative
        # config path (configs/training/<name>.yaml), even when
        # --config-dir writes elsewhere (e.g. tmp_path during tests).
        # Tools resolve it via `ROOT / row["config_path"]`.
        canonical_cfg = f"configs/training/anchordiff_{v.variant_id}.yaml"
        row = _manifest_row(
            v, resolved,
            config_path=canonical_cfg,
            base_subset_file=base_subset,
            body_action_subset_file=body_subset,
            init_checkpoint=init_ckpt,
        )
        rows.append(row)
        if args.dry_run:
            print(f"DRY-RUN would write: {out_path}  ({len(content)} bytes)")
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(content, encoding="utf-8")
            print(f"wrote {out_path}")
            written += 1

    manifest_json = analyses_dir / "round29_stage2_cond_ablation_manifest.json"
    manifest_md = analyses_dir / "round29_stage2_cond_ablation_manifest.md"
    if args.dry_run:
        print(f"DRY-RUN would write: {manifest_json}")
        print(f"DRY-RUN would write: {manifest_md}")
    else:
        analyses_dir.mkdir(parents=True, exist_ok=True)
        manifest_json.write_text(
            json.dumps(
                {
                    "variants": rows,
                    "best_resolved": {
                        "coarse": args.best_coarse,
                        "interaction": args.best_interaction,
                        "support": args.best_support,
                        "body": args.best_body,
                        "injection": args.best_injection,
                    },
                    "defaults": {
                        "init_checkpoint": init_ckpt,
                        "balanced_subset_file": base_subset,
                        "body_action_subset_file": body_subset,
                        "data_root": args.data_root,
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        manifest_md.write_text(_render_manifest_md(rows), encoding="utf-8")
        print(f"wrote {manifest_json}")
        print(f"wrote {manifest_md}")
    print(
        f"\n{len(rows)} variants {'identified' if args.dry_run else f'written ({written})'}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
