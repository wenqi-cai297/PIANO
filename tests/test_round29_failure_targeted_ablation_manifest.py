"""Round-29 failure-targeted ablation generator + manifest tests.

Per analyses/2026-05-27_round29_failure_targeted_ablation_prompt_for_claude_code.md §4.4
and §3 (six exact variants R0-R5 with the listed condition + loss-weight
specs, all on full data at the bs=32/accum=1/80ep schedule).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = (
    ROOT / "scripts" / "stage_b_generator"
    / "round29_make_failure_targeted_ablation_configs.py"
)

EXPECTED_VARIANTS = (
    "r29_ft_r0_clean_a3_baseline",
    "r29_ft_r1_no_coarse_extra",
    "r29_ft_r2_behavior_gait_loss",
    "r29_ft_r3_oracle_s4_gait_loss",
    "r29_ft_r4_i3_contact_lock",
    "r29_ft_r5_allpart_interaction_lock",
)


def _run_generator(tmp_path: Path) -> tuple[Path, dict]:
    cfg_dir = tmp_path / "configs" / "training"
    ana_dir = tmp_path / "analyses"
    cfg_dir.mkdir(parents=True)
    ana_dir.mkdir(parents=True)
    res = subprocess.run(
        [
            sys.executable, str(GENERATOR),
            "--config-dir", str(cfg_dir),
            "--analyses-dir", str(ana_dir),
        ],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert res.returncode == 0, res.stderr
    manifest_path = ana_dir / "round29_failure_targeted_ablation_manifest.json"
    assert manifest_path.exists(), f"manifest missing at {manifest_path}"
    return cfg_dir, json.loads(manifest_path.read_text(encoding="utf-8"))


def _yaml_for(cfg_dir: Path, variant_id: str) -> dict:
    p = cfg_dir / f"anchordiff_{variant_id}.yaml"
    assert p.exists(), f"missing yaml {p}"
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def test_generator_dry_run_runs_clean() -> None:
    res = subprocess.run(
        [sys.executable, str(GENERATOR), "--dry-run"],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert res.returncode == 0, res.stderr


def test_emits_six_variants_in_order(tmp_path: Path) -> None:
    _, manifest = _run_generator(tmp_path)
    ids = tuple(v["variant_id"] for v in manifest["variants"])
    assert ids == EXPECTED_VARIANTS


def test_all_full_data_no_subset_indices_file(tmp_path: Path) -> None:
    cfg_dir, manifest = _run_generator(tmp_path)
    for v in manifest["variants"]:
        cfg = _yaml_for(cfg_dir, v["variant_id"])
        assert "subset_indices_file" not in cfg["data"], (
            f"{v['variant_id']} must NOT have subset_indices_file (full-data)"
        )
        assert int(cfg["training"]["num_epochs"]) == 80
        assert int(cfg["training"]["batch_size"]) == 32
        assert int(cfg["training"]["gradient_accumulation_steps"]) == 1
        assert bool(cfg["training"]["val_on_train_subset"]) is False
        assert int(cfg["training"]["val_every_epochs"]) == 5
        assert int(cfg["logging"]["save_every_n_epochs"]) == 10
        assert int(cfg["training"]["scheduler"]["warmup_steps"]) == 250
        assert float(cfg["training"]["stage1_coarse_noise_std"]) == 0.05


def test_all_use_input_add_adapter(tmp_path: Path) -> None:
    cfg_dir, manifest = _run_generator(tmp_path)
    for v in manifest["variants"]:
        cfg = _yaml_for(cfg_dir, v["variant_id"])
        assert cfg["model"]["denoiser"]["r29_injection_mode"] == "input_add_adapter"


def test_r0_r2_r3_r4_r5_use_c41_i3_s4_b4(tmp_path: Path) -> None:
    """R0/R2/R3/R4 keep C41-current + I3 + S4 + B4. R5 swaps I3 -> I5."""
    cfg_dir, _ = _run_generator(tmp_path)
    for vid in ("r29_ft_r0_clean_a3_baseline",
                "r29_ft_r2_behavior_gait_loss",
                "r29_ft_r3_oracle_s4_gait_loss",
                "r29_ft_r4_i3_contact_lock"):
        cfg = _yaml_for(cfg_dir, vid)
        assert cfg["data"]["r29_coarse_variant"] == "C41-current"
        assert int(cfg["model"]["denoiser"]["r29_coarse_extra_dim"]) == 18
        assert cfg["data"]["r29_interaction_variant"] == "I3-contact-offset-masked"
        assert int(cfg["model"]["denoiser"]["r29_interaction_dim"]) == 8
        assert cfg["data"]["r29_support_variant"] == "S4-S1-phase-footstep"
        assert int(cfg["model"]["denoiser"]["r29_support_dim"]) == 13
        assert cfg["data"]["r29_body_variant"] == "B4-lowpass-residual-mask"
        assert int(cfg["model"]["denoiser"]["r29_body_refine_dim"]) == 20


def test_r1_drops_c41_to_c23_and_zero_extra(tmp_path: Path) -> None:
    cfg_dir, _ = _run_generator(tmp_path)
    cfg = _yaml_for(cfg_dir, "r29_ft_r1_no_coarse_extra")
    assert cfg["data"]["r29_coarse_variant"] == "C23"
    assert int(cfg["model"]["denoiser"]["r29_coarse_extra_dim"]) == 0
    # Keep I3 + S4 + B4 from R0.
    assert cfg["data"]["r29_interaction_variant"] == "I3-contact-offset-masked"
    assert cfg["data"]["r29_support_variant"] == "S4-S1-phase-footstep"
    assert cfg["data"]["r29_body_variant"] == "B4-lowpass-residual-mask"
    # Stage1_coarse stays 23 (Stage-1 itself unchanged).
    assert int(cfg["model"]["denoiser"]["stage1_coarse_dim"]) == 23


def test_r5_uses_i5_dim_20(tmp_path: Path) -> None:
    cfg_dir, _ = _run_generator(tmp_path)
    cfg = _yaml_for(cfg_dir, "r29_ft_r5_allpart_interaction_lock")
    assert cfg["data"]["r29_interaction_variant"] == "I5-allpart-contact-offset-masked"
    assert int(cfg["model"]["denoiser"]["r29_interaction_dim"]) == 20


def test_baseline_loss_weights_match_r0(tmp_path: Path) -> None:
    """All six variants keep the A3 baseline absolute stabilizers."""
    cfg_dir, _ = _run_generator(tmp_path)
    for vid in EXPECTED_VARIANTS:
        cfg = _yaml_for(cfg_dir, vid)
        L = cfg["loss"]
        assert float(L["pos_loss_weight"]) == 5.0
        assert float(L["hand_endpoint_weight"]) == 2.0
        assert float(L["foot_endpoint_weight"]) == 2.0
        assert float(L["anchor_joint_pos_weight"]) == 10.0
        assert float(L["anchor_joint_vel_weight"]) == 2.0
        assert float(L["world_joint_velocity_weight"]) == 1.0


def test_r2_activates_behavior_gait_only(tmp_path: Path) -> None:
    cfg_dir, _ = _run_generator(tmp_path)
    ti = _yaml_for(cfg_dir, "r29_ft_r2_behavior_gait_loss")["loss"]["temporal_interaction"]
    assert float(ti["r29_gait_one_foot_support_weight"]) == 0.20
    assert float(ti["r29_gait_pred_stance_velocity_weight"]) == 0.10
    assert float(ti["r29_gait_ankle_smooth_weight"]) == 0.02
    assert float(ti["r29_gait_antiphase_corr_weight"]) == 0.05
    # R2 must NOT activate exact S4 nor contact-lock terms.
    assert float(ti["r29_s4_stance_bce_weight"]) == 0.0
    assert float(ti["r29_s4_footstep_target_weight"]) == 0.0
    assert float(ti["r29_contact_lock_offset_weight"]) == 0.0
    assert float(ti["r29_contact_lock_segment_drift_weight"]) == 0.0
    assert float(ti["r29_contact_lock_tracking_weight"]) == 0.0


def test_r3_activates_exact_s4_plus_existing_r29_support(tmp_path: Path) -> None:
    cfg_dir, _ = _run_generator(tmp_path)
    ti = _yaml_for(cfg_dir, "r29_ft_r3_oracle_s4_gait_loss")["loss"]["temporal_interaction"]
    # New exact S4 terms.
    assert float(ti["r29_s4_stance_bce_weight"]) == 0.10
    assert float(ti["r29_s4_footstep_target_weight"]) == 0.20
    # Plus the existing R29 support-family terms.
    assert float(ti["r29_support_both_airborne_weight"]) == 0.10
    assert float(ti["r29_support_stance_velocity_weight"]) == 0.10
    assert float(ti["r29_swing_clearance_weight"]) == 0.10
    # R3 must NOT activate behavior-gait or contact-lock.
    assert float(ti["r29_gait_one_foot_support_weight"]) == 0.0
    assert float(ti["r29_contact_lock_offset_weight"]) == 0.0


def test_r4_activates_contact_lock_only(tmp_path: Path) -> None:
    cfg_dir, _ = _run_generator(tmp_path)
    ti = _yaml_for(cfg_dir, "r29_ft_r4_i3_contact_lock")["loss"]["temporal_interaction"]
    assert float(ti["r29_contact_lock_offset_weight"]) == 0.50
    assert float(ti["r29_contact_lock_segment_drift_weight"]) == 0.50
    assert float(ti["r29_contact_lock_tracking_weight"]) == 0.25
    # R4 keeps baseline absolute stabilizers; must NOT activate behavior-gait
    # or exact S4 or anchor2_mixed-style relative losses.
    assert float(ti["r29_gait_one_foot_support_weight"]) == 0.0
    assert float(ti["r29_s4_stance_bce_weight"]) == 0.0
    assert float(ti["contact_rel_offset_weight"]) == 0.0
    assert float(ti["contact_drift_weight"]) == 0.0
    assert float(ti["contact_tracking_weight"]) == 0.0


def test_r5_activates_same_contact_lock_as_r4(tmp_path: Path) -> None:
    """R5 = R4's lock losses + I5 interaction. Lock weights must match."""
    cfg_dir, _ = _run_generator(tmp_path)
    ti4 = _yaml_for(cfg_dir, "r29_ft_r4_i3_contact_lock")["loss"]["temporal_interaction"]
    ti5 = _yaml_for(cfg_dir, "r29_ft_r5_allpart_interaction_lock")["loss"]["temporal_interaction"]
    for key in (
        "r29_contact_lock_offset_weight",
        "r29_contact_lock_segment_drift_weight",
        "r29_contact_lock_tracking_weight",
    ):
        assert float(ti5[key]) == float(ti4[key])


def test_r0_activates_no_new_terms(tmp_path: Path) -> None:
    cfg_dir, _ = _run_generator(tmp_path)
    ti = _yaml_for(cfg_dir, "r29_ft_r0_clean_a3_baseline")["loss"]["temporal_interaction"]
    for k in (
        "r29_gait_one_foot_support_weight",
        "r29_gait_pred_stance_velocity_weight",
        "r29_gait_ankle_smooth_weight",
        "r29_gait_antiphase_corr_weight",
        "r29_s4_stance_bce_weight",
        "r29_s4_footstep_target_weight",
        "r29_contact_lock_offset_weight",
        "r29_contact_lock_segment_drift_weight",
        "r29_contact_lock_tracking_weight",
        "r29_interaction_consistency_weight",
        "r29_support_both_airborne_weight",
        "r29_support_stance_velocity_weight",
        "r29_swing_clearance_weight",
        "contact_rel_offset_weight",
        "contact_drift_weight",
        "contact_tracking_weight",
    ):
        assert float(ti[k]) == 0.0, f"R0 leaked non-zero on {k!r}"


def test_no_init_checkpoint_in_any_variant(tmp_path: Path) -> None:
    """All six train from scratch (no warm-start), per prompt §3."""
    cfg_dir, _ = _run_generator(tmp_path)
    for vid in EXPECTED_VARIANTS:
        cfg = _yaml_for(cfg_dir, vid)
        training = cfg.get("training", {})
        assert "init_checkpoint" not in training, (
            f"{vid} must NOT have init_checkpoint (from-scratch per prompt §3)"
        )


def test_manifest_includes_decision_question(tmp_path: Path) -> None:
    _, manifest = _run_generator(tmp_path)
    for v in manifest["variants"]:
        q = v.get("decision_question", "")
        assert isinstance(q, str) and len(q) > 0
