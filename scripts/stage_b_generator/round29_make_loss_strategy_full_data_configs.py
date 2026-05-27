"""Generate Round-29 loss-strategy FULL-DATA configs (v2 winner promotion).

Per Codex review of v2 48-clip result
(`analyses/2026-05-27_round29_loss_strategy_v2_codex_review.md`):
take the v2 winner (`anchor2_mixed`) to full-data on both A2 and A3,
**paired with a fair full-data baseline_from_scratch** so the
comparison is single-axis (loss strategy only; same data, same epochs,
same init regime).

Four-variant matrix:

  r29_lsf_a2_baseline_from_scratch    A2 (adapter_only), original a-group losses
                                       (pos_loss=5, anchor_pos=10, anchor_vel=2,
                                        world_vel=1). Fair full-data reference.
  r29_lsf_a3_baseline_from_scratch    A3 (input_add_adapter), same.
  r29_lsf_a2_anchor2_mixed            A2, v2 winner loss weights (anchor_pos=2,
                                       anchor_vel=0.5, world_vel=0.5, R29 weights
                                       0.10 each, swing_clearance ON).
  r29_lsf_a3_anchor2_mixed            A3, same v2 winner loss weights.

All four: FULL-DENSE C/I/S/B oracle content, full InterAct train set
(no subset_indices_file), 80 ep, heldout val (val_every=5),
save_every=10, warmup=250, stage1_coarse_noise_std=0.05 — schedule
matches the existing anchordiff_a{2,3}_full_data.yaml / v27 FULL_DATA.

Outputs:
    configs/training/anchordiff_r29_lsf_a2_baseline_from_scratch.yaml
    configs/training/anchordiff_r29_lsf_a3_baseline_from_scratch.yaml
    configs/training/anchordiff_r29_lsf_a2_anchor2_mixed.yaml
    configs/training/anchordiff_r29_lsf_a3_anchor2_mixed.yaml
    analyses/round29_loss_strategy_full_data_manifest.json
    analyses/round29_loss_strategy_full_data_manifest.md

Usage:
    python scripts/stage_b_generator/round29_make_loss_strategy_full_data_configs.py
    python scripts/stage_b_generator/round29_make_loss_strategy_full_data_configs.py \\
        --data-root /media/.../piano_official_process_4
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


@dataclass(slots=True)
class FullDataLossStrategyVariant:
    variant_id: str
    purpose: str
    injection_mode: str           # "adapter_only" (A2) or "input_add_adapter" (A3)
    loss_strategy: str            # "baseline_from_scratch" or "anchor2_mixed"
    # Absolute-GT auxiliary loss weights.
    pos_loss_weight: float
    hand_endpoint_weight: float
    foot_endpoint_weight: float
    anchor_joint_pos_weight: float
    anchor_joint_vel_weight: float
    world_joint_velocity_weight: float
    # Relative + R29 condition-consistency weights.
    contact_rel_offset_weight: float = 0.0
    contact_drift_weight: float = 0.0
    contact_tracking_weight: float = 0.0
    r29_interaction_consistency_weight: float = 0.0
    r29_support_both_airborne_weight: float = 0.0
    r29_support_stance_velocity_weight: float = 0.0
    r29_swing_clearance_weight: float = 0.0
    r29_swing_clearance_m: float = 0.05
    # Best-ckpt selector — must match an active loss component.
    val_best_key: str = "loss_anchor_joint_pos"
    diagnostics: tuple[str, ...] = field(default_factory=lambda: (
        "sustained_contact", "gait", "body_action",
    ))


# Two loss-strategy presets per Codex v2 review §"Final recommendation".
# `baseline_from_scratch`: matches the original a-group losses used by the
#   existing anchordiff_a{2,3}_full_data.yaml. Provides the fair Rule-1
#   reference at full-data scale.
# `anchor2_mixed`: v2 winner from the 48-clip mechanism screen
#   (analyses/2026-05-27_round29_loss_strategy_v2_verdict.md). Same weights
#   that passed 4/4 hard criteria on both A2 and A3 at 48-clip.

_BASELINE_FROM_SCRATCH = dict(
    loss_strategy="baseline_from_scratch",
    pos_loss_weight=5.0,
    hand_endpoint_weight=2.0,
    foot_endpoint_weight=2.0,
    anchor_joint_pos_weight=10.0,
    anchor_joint_vel_weight=2.0,
    world_joint_velocity_weight=1.0,
    # All relative / R29 weights at 0.
    val_best_key="loss_anchor_joint_pos",
)

_ANCHOR2_MIXED = dict(
    loss_strategy="anchor2_mixed",
    pos_loss_weight=0.0,
    hand_endpoint_weight=1.0,
    foot_endpoint_weight=1.0,
    anchor_joint_pos_weight=2.0,
    anchor_joint_vel_weight=0.5,
    world_joint_velocity_weight=0.5,
    contact_rel_offset_weight=0.25,
    contact_drift_weight=0.25,
    contact_tracking_weight=0.25,
    r29_interaction_consistency_weight=0.10,
    r29_support_both_airborne_weight=0.10,
    r29_support_stance_velocity_weight=0.10,
    r29_swing_clearance_weight=0.10,
    r29_swing_clearance_m=0.05,
    val_best_key="loss_anchor_joint_pos",   # anchor is non-zero (=2.0)
)

_FAMILY_PRESETS: dict[str, dict] = {
    "baseline_from_scratch": _BASELINE_FROM_SCRATCH,
    "anchor2_mixed": _ANCHOR2_MIXED,
}

_FAMILY_PURPOSE_PHRASE: dict[str, str] = {
    "baseline_from_scratch": (
        "fair full-data baseline using the original a-group losses "
        "(pos_loss=5, anchor_pos=10, anchor_vel=2, world_vel=1, no relative "
        "or R29 consistency terms). No init_checkpoint — pure from-scratch. "
        "Provides the Rule-1 reference for the loss-strategy comparison."
    ),
    "anchor2_mixed": (
        "v2 winner from the 48-clip mechanism screen "
        "(analyses/2026-05-27_round29_loss_strategy_v2_verdict.md). Weak "
        "absolute stabilizer (anchor_pos=2, anchor_vel=0.5) + low R29 "
        "condition-consistency (0.10 each) + swing_clearance. Passed 4/4 "
        "hard criteria on both A2 and A3 at 48-clip. Full-data verdict "
        "decides whether this is the loss-strategy mainline."
    ),
}


def _make_variants() -> list[FullDataLossStrategyVariant]:
    out: list[FullDataLossStrategyVariant] = []
    for inj_label, inj_mode in (("a2", "adapter_only"), ("a3", "input_add_adapter")):
        for family_id, preset in _FAMILY_PRESETS.items():
            out.append(FullDataLossStrategyVariant(
                variant_id=f"r29_lsf_{inj_label}_{family_id}",
                purpose=(
                    f"{inj_label.upper()} injection ({inj_mode}), full-data, "
                    f"{family_id}: {_FAMILY_PURPOSE_PHRASE[family_id]}"
                ),
                injection_mode=inj_mode,
                **preset,
            ))
    return out


VARIANTS: list[FullDataLossStrategyVariant] = _make_variants()


def _render_datasets_block(data_root: str) -> str:
    stripped = data_root.rstrip("/").rstrip("\\")
    lines: list[str] = []
    for sub in DATASET_SUBSET_NAMES:
        lines.append(f'    - name: "{sub}"')
        lines.append(f'      root: "{stripped}/{sub}"')
    return "\n".join(lines)


def _render_yaml(v: FullDataLossStrategyVariant, *, data_root: str) -> str:
    run_name = f"stageB_anchordiff_{v.variant_id}"
    return f"""# Round-29 loss-strategy {v.variant_id}: {v.purpose}
#
# Generated by scripts/stage_b_generator/round29_make_loss_strategy_full_data_configs.py.
# Per analyses/2026-05-27_round29_loss_strategy_v2_codex_review.md §"Final recommendation":
# 4-run full-data matrix (A2/A3 × {{baseline_from_scratch, anchor2_mixed}}).
#
# FULL DATA. No subset_indices_file. Heldout val. 80 ep / val_every=5 /
# save_every=10 / warmup=250 — matches anchordiff_a{{2,3}}_full_data.yaml
# and v27 FULL_DATA schedule.

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
    # Round-29 typed condition injection — FULL-DENSE C/I/S/B content.
    use_round29_cond_injection: true
    r29_coarse_extra_dim: 18
    r29_interaction_dim: 8
    r29_support_dim: 13
    r29_body_refine_dim: 20
    r29_injection_mode: "{v.injection_mode}"
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
  # R29 4-family content: FULL-DENSE.
  r29_coarse_variant: "C41-current"
  r29_interaction_variant: "I3-contact-offset-masked"
  r29_support_variant: "S4-S1-phase-footstep"
  r29_body_variant: "B4-lowpass-residual-mask"
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
  batch_size: 8
  num_epochs: 80
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
  gradient_accumulation_steps: 4
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Round-29 loss-strategy FULL-DATA configs (4-variant "
            "matrix: A2/A3 × baseline_from_scratch / anchor2_mixed) "
            "for the v2 winner promotion to full-data."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--data-root",
        default=os.environ.get("DATASETS_ROOT", DEFAULT_DATA_ROOT),
        help=(
            "Root directory containing the four InterAct subsets. "
            "On the Linux server pass --data-root /media/.../piano_official_process_4 "
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
        canonical_cfg = f"configs/training/anchordiff_{v.variant_id}.yaml"
        row = {
            "variant_id": v.variant_id,
            "group": "LossStrategyFullData",
            "purpose": v.purpose,
            "injection_mode": v.injection_mode,
            "loss_strategy": v.loss_strategy,
            "knobs": {
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
                "r29_swing_clearance_m": v.r29_swing_clearance_m,
            },
            "training_data": "full",
            "num_epochs": 80,
            "seed": 42,
            "val_on_train_subset": False,
            "val_every_epochs": 5,
            "val_best_key": v.val_best_key,
            "config_path": canonical_cfg,
            "output_dir": f"runs/training/stageB_anchordiff_{v.variant_id}",
            "diagnostics": list(v.diagnostics),
        }
        rows.append(row)
        if args.dry_run:
            print(f"DRY-RUN would write: {out_path}  ({len(content)} bytes)")
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(content, encoding="utf-8")
            print(f"wrote {out_path}")

    manifest_json = analyses_dir / "round29_loss_strategy_full_data_manifest.json"
    manifest_md = analyses_dir / "round29_loss_strategy_full_data_manifest.md"

    md_lines: list[str] = [
        "# Round-29 loss-strategy FULL-DATA manifest (v2 winner promotion)",
        "",
        "Per analyses/2026-05-27_round29_loss_strategy_v2_codex_review.md §Final recommendation:",
        "4-run full-data matrix to decide whether the v2 mixed strategy",
        "improves the full-data tradeoff vs the original loss strategy.",
        "",
        "Two families × two injections:",
        "  - `baseline_from_scratch`: original a-group losses",
        "    (pos_loss=5, anchor_pos=10, anchor_vel=2, world_vel=1).",
        "    Fair Rule-1 reference at full-data scale.",
        "  - `anchor2_mixed`: v2 48-clip winner (anchor_pos=2, anchor_vel=0.5,",
        "    R29 weights 0.10 each, swing_clearance ON at 5 cm threshold).",
        "",
        "All four: FULL InterAct train set (no subset_indices_file), 80 ep,",
        "heldout val, save_every=10, warmup=250, stage1_coarse_noise_std=0.05.",
        "",
        "| variant | injection | family | val_best_key | pos_loss | anchor_pos | anchor_vel | r29_int | r29_air | r29_st_v | swing_clear |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        k = r["knobs"]
        md_lines.append(
            f"| `{r['variant_id']}` | {r['injection_mode']} | {r['loss_strategy']} | "
            f"`{r['val_best_key']}` | "
            f"{k['pos_loss_weight']} | {k['anchor_joint_pos_weight']} | "
            f"{k['anchor_joint_vel_weight']} | "
            f"{k['r29_interaction_consistency_weight']} | "
            f"{k['r29_support_both_airborne_weight']} | "
            f"{k['r29_support_stance_velocity_weight']} | "
            f"{k['r29_swing_clearance_weight']} |"
        )
    md_lines.append("")
    md_lines.append(
        "Fair comparisons (single-axis):"
    )
    md_lines.append(
        "  - `a{2,3}_baseline_from_scratch` vs `a{2,3}_anchor2_mixed`: "
        "tests loss strategy at full-data, injection held constant."
    )
    md_lines.append(
        "  - Within `anchor2_mixed`: A2 vs A3 picks the injection mainline."
    )
    md_lines.append(
        "  - Within `baseline_from_scratch`: A2 vs A3 also picks the injection "
        "mainline under the original loss recipe."
    )

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
    print(f"\n{len(rows)} full-data loss-strategy variant(s) emitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
