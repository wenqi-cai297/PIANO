"""Generate Round-29 loss-strategy ablation configs.

Per ``analyses/2026-05-27_round29_loss_strategy_ablation_prompt_for_claude_code.md``.

This ablation group tests a different axis from Tier-2: not "remove
legacy inputs" but "absolute-GT supervision vs condition/behavior
consistency". The motivation is that the current Stage-2 auxiliary
losses (``pos_loss_full``, ``anchor_joint_pos``, ``anchor_joint_vel``,
strong velocity matching) pull every prediction toward the ONE GT
realization in the dataset. When the condition is underspecified (text
+ similar objects, no explicit side), there can be multiple equally-
valid modes (e.g. left-hand-first vs right-hand-first); absolute-GT
losses penalise the non-GT mode and push the model toward averaged
motion.

Important: when the R29 condition DOES specify side (I3 left/right
hand, S4 left/right foot stance, S4 left/right footstep target), the
loss must still enforce it. The R29 condition-consistency losses
(``loss_r29_interaction_consistency``, ``loss_r29_support_both_airborne``,
``loss_r29_support_stance_velocity``) read sides directly from the
condition channels — they are NOT permutation-invariant.

This generator produces 4 P0 variants:

  r29_ls_a2_no_dense_pos           A2 injection (adapter_only),
                                    drop dense FK pos loss only.
                                    Keeps anchor_joint_pos/vel weights.
  r29_ls_a3_no_dense_pos           A3 injection (input_add_adapter),
                                    drop dense FK pos loss only.
  r29_ls_a2_relative_behavior      A2 injection, drop all big absolute-GT
                                    auxiliary losses + add R29 consistency.
  r29_ls_a3_relative_behavior      A3 injection, drop all big absolute-GT
                                    auxiliary losses + add R29 consistency.

Compare against the regular R29 a2 / a3 baselines on the same 48-clip
subset and same 300-epoch schedule.

Outputs:
    configs/training/anchordiff_r29_ls_a2_no_dense_pos.yaml
    configs/training/anchordiff_r29_ls_a3_no_dense_pos.yaml
    configs/training/anchordiff_r29_ls_a2_relative_behavior.yaml
    configs/training/anchordiff_r29_ls_a3_relative_behavior.yaml
    analyses/round29_loss_strategy_ablation_manifest.json
    analyses/round29_loss_strategy_ablation_manifest.md

Usage:
    python scripts/stage_b_generator/round29_make_loss_strategy_ablation_configs.py
    python scripts/stage_b_generator/round29_make_loss_strategy_ablation_configs.py \\
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
# 48-clip subset used for both training and diag selection (matches
# A-group + Tier-1/Tier-2-of-original schema). Per prompt §8, this is
# a strategy-screening ablation; we deliberately stay on 48-clip overfit
# until the strategy direction is validated.
DEFAULT_SUBSET_FILE = "analyses/round27_tier0_train_indices_48_balanced.json"
DATASET_SUBSET_NAMES: tuple[str, ...] = (
    "chairs", "imhd", "neuraldome", "omomo_correct_v2",
)


# ----------------------------------------------------------------------
# Variant definitions
# ----------------------------------------------------------------------


@dataclass(slots=True)
class LossStrategyVariant:
    variant_id: str
    purpose: str
    # Injection (A-axis): "adapter_only" (A2) or "input_add_adapter" (A3).
    injection_mode: str
    # Loss strategy (L-axis): "no_dense_pos" or "relative_behavior".
    loss_strategy: str
    # Absolute-GT auxiliary loss weights.
    pos_loss_weight: float = 5.0
    hand_endpoint_weight: float = 2.0
    foot_endpoint_weight: float = 2.0
    anchor_joint_pos_weight: float = 10.0
    anchor_joint_vel_weight: float = 2.0
    world_joint_velocity_weight: float = 1.0
    # Existing relative losses (temporal_interaction.*).
    contact_rel_offset_weight: float = 0.0
    contact_drift_weight: float = 0.0
    contact_tracking_weight: float = 0.0
    # New R29 condition-consistency losses.
    r29_interaction_consistency_weight: float = 0.0
    r29_support_both_airborne_weight: float = 0.0
    r29_support_stance_velocity_weight: float = 0.0
    # Validation checkpoint selector. ``relative_behavior`` disables
    # anchor_joint_pos, so it must not select best_val.pt on that metric.
    val_best_key: str = "loss_anchor_joint_pos"
    diagnostics: tuple[str, ...] = field(default_factory=lambda: (
        "sustained_contact", "gait", "body_action",
    ))


# Build the 4-variant matrix: A-axis × L-axis.

_NO_DENSE_POS = dict(
    loss_strategy="no_dense_pos",
    pos_loss_weight=0.0,
    hand_endpoint_weight=1.0,
    foot_endpoint_weight=1.0,
    # Anchor (absolute-GT active-part pull) is KEPT in no_dense_pos.
    anchor_joint_pos_weight=10.0,
    anchor_joint_vel_weight=2.0,
    world_joint_velocity_weight=1.0,
    # All relative + R29 consistency weights stay at 0.
)

# Per prompt §7.2 — modest initial weights for R29 consistency losses,
# weak global velocity prior (0.2 not 1.0), anchor weights OFF.
_RELATIVE_BEHAVIOR = dict(
    loss_strategy="relative_behavior",
    pos_loss_weight=0.0,
    hand_endpoint_weight=1.0,
    foot_endpoint_weight=1.0,
    anchor_joint_pos_weight=0.0,
    anchor_joint_vel_weight=0.0,
    world_joint_velocity_weight=0.2,
    contact_rel_offset_weight=0.25,
    contact_drift_weight=0.25,
    contact_tracking_weight=0.25,
    r29_interaction_consistency_weight=0.75,
    r29_support_both_airborne_weight=0.5,
    r29_support_stance_velocity_weight=0.25,
    val_best_key="loss",
)


def _make_variants() -> list[LossStrategyVariant]:
    out: list[LossStrategyVariant] = []
    for inj_label, inj_mode in (("a2", "adapter_only"), ("a3", "input_add_adapter")):
        # no_dense_pos
        out.append(LossStrategyVariant(
            variant_id=f"r29_ls_{inj_label}_no_dense_pos",
            purpose=(
                f"{inj_label.upper()} injection ({inj_mode}), no_dense_pos: "
                "drop the dense FK position MSE (pos_loss_weight=0). Keep "
                "active-part anchor_joint_pos/vel. Isolates whether dense "
                "FK is causing over-absolute supervision while leaving "
                "active-part anchor in place."
            ),
            injection_mode=inj_mode,
            **_NO_DENSE_POS,
        ))
        # relative_behavior
        out.append(LossStrategyVariant(
            variant_id=f"r29_ls_{inj_label}_relative_behavior",
            purpose=(
                f"{inj_label.upper()} injection ({inj_mode}), relative_behavior: "
                "turn off the largest absolute-GT auxiliary pulls "
                "(pos_loss=0, anchor_joint_pos/vel=0, weak vel=0.2) and "
                "switch to condition-consistency losses (R29 interaction + "
                "support both-airborne + stance velocity) and existing "
                "relative contact losses (rel_offset, drift, tracking). "
                "Tests whether condition-realization supervision is enough."
            ),
            injection_mode=inj_mode,
            **_RELATIVE_BEHAVIOR,
        ))
    return out


VARIANTS: list[LossStrategyVariant] = _make_variants()


# ----------------------------------------------------------------------
# YAML rendering
# ----------------------------------------------------------------------


def _render_datasets_block(data_root: str) -> str:
    stripped = data_root.rstrip("/").rstrip("\\")
    lines: list[str] = []
    for sub in DATASET_SUBSET_NAMES:
        lines.append(f'    - name: "{sub}"')
        lines.append(f'      root: "{stripped}/{sub}"')
    return "\n".join(lines)


def _render_yaml(v: LossStrategyVariant, *, subset_file: str, data_root: str) -> str:
    run_name = f"stageB_anchordiff_{v.variant_id}"
    return f"""# Round-29 loss-strategy {v.variant_id}: {v.purpose}
#
# Generated by scripts/stage_b_generator/round29_make_loss_strategy_ablation_configs.py.
# Per analyses/2026-05-27_round29_loss_strategy_ablation_prompt_for_claude_code.md.
# Strategy axis: absolute-GT supervision vs condition/behavior consistency.
# Injection ({v.injection_mode}) matches the corresponding a-group variant.

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
  subset_indices_file: "{subset_file}"
  surface_temporal_aux_fields: false
  # R29 4-family content: FULL-DENSE (same as a2/a3 baselines).
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
  num_epochs: 300
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


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def _to_posix_relpath(p: str | Path, root: Path) -> str:
    s = str(p)
    if not s:
        return s
    p_obj = Path(s)
    try:
        rel = p_obj.resolve().relative_to(root.resolve())
        return rel.as_posix()
    except (ValueError, OSError):
        return p_obj.as_posix() if not p_obj.is_absolute() else s


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Round-29 loss-strategy ablation configs "
            "(absolute-GT vs condition-consistency, A2/A3 × no_dense_pos/relative_behavior)."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--data-root",
        default=os.environ.get("DATASETS_ROOT", DEFAULT_DATA_ROOT),
    )
    parser.add_argument(
        "--subset-file", default=DEFAULT_SUBSET_FILE,
        help=(
            "Subset JSON consumed by trainer + diagnostics. Default = the "
            "Round-27 48-clip balanced selection (same as the a-group)."
        ),
    )
    parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--analyses-dir", default=str(DEFAULT_ANALYSES_DIR))
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    analyses_dir = Path(args.analyses_dir)
    subset_file = _to_posix_relpath(args.subset_file, ROOT)

    rows: list[dict] = []
    for v in VARIANTS:
        out_path = config_dir / f"anchordiff_{v.variant_id}.yaml"
        content = _render_yaml(v, subset_file=subset_file, data_root=args.data_root)
        canonical_cfg = f"configs/training/anchordiff_{v.variant_id}.yaml"
        row = {
            "variant_id": v.variant_id,
            "group": "LossStrategy",
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
            },
            "subset_kind": "balanced",
            "subset_file": subset_file,
            "num_epochs": 300,
            "seed": 42,
            "val_on_train_subset": True,
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

    manifest_json = analyses_dir / "round29_loss_strategy_ablation_manifest.json"
    manifest_md = analyses_dir / "round29_loss_strategy_ablation_manifest.md"

    md_lines: list[str] = [
        "# Round-29 loss-strategy ablation manifest",
        "",
        "Per analyses/2026-05-27_round29_loss_strategy_ablation_prompt_for_claude_code.md.",
        "Each variant fixes the R29 FULL-DENSE C/I/S/B content and the A-axis",
        "injection mode (A2=adapter_only or A3=input_add_adapter), then varies",
        "the loss strategy: keep absolute-GT pulls (baseline a2/a3 — NOT in",
        "this group), drop only dense FK (`no_dense_pos`), or fully switch to",
        "condition-consistency (`relative_behavior`).",
        "",
        "All variants use the same 48-clip balanced subset and 300 ep schedule",
        "as the regular A-group baselines for direct comparison.",
        "",
        "| variant | injection | strategy | val_best_key | pos_loss | anchor_pos | r29_int_cons | r29_supp_air | r29_stance_vel |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        k = r["knobs"]
        md_lines.append(
            f"| `{r['variant_id']}` | {r['injection_mode']} | {r['loss_strategy']} | "
            f"`{r['val_best_key']}` | "
            f"{k['pos_loss_weight']} | {k['anchor_joint_pos_weight']} | "
            f"{k['r29_interaction_consistency_weight']} | "
            f"{k['r29_support_both_airborne_weight']} | "
            f"{k['r29_support_stance_velocity_weight']} |"
        )
    md_lines.append("")
    md_lines.append(
        "Compare each variant's three diag outputs against the corresponding "
        "regular A-group ckpt (a2_adapter_only / a3_input_add_adapter on the "
        "same 48-clip subset, 300 ep)."
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
    print(f"\n{len(rows)} loss-strategy variant(s) emitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
