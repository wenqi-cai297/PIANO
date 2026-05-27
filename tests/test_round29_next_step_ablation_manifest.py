"""Round-29 next-step ablation generator + manifest tests.

Per analyses/2026-05-28_round29_next_step_ablation_execution_prompt_for_claude_code.md §8.1.

The matrix contains 4 train variants (A0/A1/H1/A2) + 4 reference rows
(R0, B1, G1, INVALID old H1). Required assertions:

  - Train variants are full-data, from scratch, bs=32/accum=1/80ep.
  - A0: data S4 + model support_dim=0 (S4 loss-only), G1 losses ON.
  - A1: data S4 + model support_dim=13, G1 losses ON.
  - H1: data I5 + model interaction_dim=20, no oracle hint YAML fields.
  - A2: I5 + G1, S4 loss-only.
  - No old oracle-hint fields anywhere.
  - Generator honors DATASETS_ROOT.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = (
    ROOT / "scripts" / "stage_b_generator"
    / "round29_make_next_step_ablation_configs.py"
)

EXPECTED_TRAIN = (
    "r29_ns_a0_c41_g1_loss_s4",
    "r29_ns_a1_c41_s4_g1",
    "r29_ns_h1_i5_upper_bound",
    "r29_ns_a2_c41_i5_g1",
)
REFERENCES = (
    "r29_ft_r0_clean_a3_baseline",
    "r29_nb_b1_c41_only",
    "r29_nb_g1_phasefree_gait_fixed",
    "r29_nb_h1_r0_plus_oracle_full_hint",  # INVALID
)
INVALID_OLD = "r29_nb_h1_r0_plus_oracle_full_hint"


def _run_generator(tmp_path: Path) -> tuple[Path, dict]:
    cfg_dir = tmp_path / "configs" / "training"
    ana_dir = tmp_path / "analyses"
    cfg_dir.mkdir(parents=True); ana_dir.mkdir(parents=True)
    res = subprocess.run(
        [sys.executable, str(GENERATOR),
         "--config-dir", str(cfg_dir), "--analyses-dir", str(ana_dir)],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert res.returncode == 0, res.stderr
    manifest_path = ana_dir / "round29_next_step_ablation_manifest.json"
    assert manifest_path.exists()
    return cfg_dir, json.loads(manifest_path.read_text(encoding="utf-8"))


def _yaml_for(cfg_dir: Path, variant_id: str) -> dict:
    p = cfg_dir / f"anchordiff_{variant_id}.yaml"
    assert p.exists(), f"missing yaml {p}"
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def _yaml_text(cfg_dir: Path, variant_id: str) -> str:
    p = cfg_dir / f"anchordiff_{variant_id}.yaml"
    return p.read_text(encoding="utf-8")


def test_generator_dry_run_clean() -> None:
    res = subprocess.run(
        [sys.executable, str(GENERATOR), "--dry-run"],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert res.returncode == 0, res.stderr
    # 4 new train + 4 references (1 marked INVALID).
    assert "4 new train variant(s)" in res.stdout
    assert "4 reference(s)" in res.stdout
    assert "1 marked INVALID" in res.stdout


def test_manifest_has_4_train_plus_4_references(tmp_path) -> None:
    _, manifest = _run_generator(tmp_path)
    variants = manifest["variants"]
    train_rows = [v for v in variants if v.get("train", True)]
    ref_rows = [v for v in variants if not v.get("train", True)]
    assert len(train_rows) == 4
    assert len(ref_rows) == 4
    assert {v["variant_id"] for v in train_rows} == set(EXPECTED_TRAIN)
    assert {v["variant_id"] for v in ref_rows} == set(REFERENCES)


def test_old_h1_reference_marked_invalid(tmp_path) -> None:
    _, manifest = _run_generator(tmp_path)
    old_h1 = next(
        v for v in manifest["variants"] if v["variant_id"] == INVALID_OLD
    )
    assert old_h1["train"] is False
    assert old_h1["valid_for_decision"] is False
    assert "oracle hint" in old_h1["invalid_reason"]


def test_all_train_variants_are_full_data_from_scratch(tmp_path) -> None:
    cfg_dir, _ = _run_generator(tmp_path)
    for vid in EXPECTED_TRAIN:
        cfg = _yaml_for(cfg_dir, vid)
        assert "subset_indices_file" not in (cfg.get("data") or {}), vid
        assert "init_checkpoint" not in (cfg.get("training") or {}), vid
        assert cfg["training"]["batch_size"] == 32, vid
        assert cfg["training"]["gradient_accumulation_steps"] == 1, vid
        assert cfg["training"]["num_epochs"] == 80, vid


def test_a0_data_s4_but_model_support_dim_zero(tmp_path) -> None:
    """The critical A0 wiring: dataset emits S4, model doesn't consume it."""
    cfg_dir, manifest = _run_generator(tmp_path)
    cfg = _yaml_for(cfg_dir, "r29_ns_a0_c41_g1_loss_s4")
    assert cfg["data"]["r29_support_variant"] == "S4-S1-phase-footstep"
    assert cfg["model"]["denoiser"]["r29_support_dim"] == 0
    # Other dims also reflect "C41 only" model consumption.
    assert cfg["model"]["denoiser"]["r29_coarse_extra_dim"] == 18
    assert cfg["model"]["denoiser"]["r29_interaction_dim"] == 0
    assert cfg["model"]["denoiser"]["r29_body_refine_dim"] == 0
    # Manifest must mark "support" as a loss-only family.
    a0 = next(v for v in manifest["variants"]
              if v["variant_id"] == "r29_ns_a0_c41_g1_loss_s4")
    assert "support" in a0["condition"]["loss_only_families"]


def test_a0_g1_losses_active_and_one_foot_off(tmp_path) -> None:
    cfg_dir, _ = _run_generator(tmp_path)
    cfg = _yaml_for(cfg_dir, "r29_ns_a0_c41_g1_loss_s4")
    ti = cfg["loss"]["temporal_interaction"]
    assert float(ti["r29_gait_soft_stance_velocity_weight"]) > 0.0
    assert float(ti["r29_gait_transition_rate_weight"]) > 0.0
    assert float(ti["r29_gait_duty_cycle_weight"]) > 0.0
    assert float(ti["r29_gait_both_state_match_weight"]) > 0.0
    # Old R2 one-foot-support and explicit S4 BCE/footstep must be OFF.
    assert float(ti["r29_gait_one_foot_support_weight"]) == 0.0
    assert float(ti["r29_s4_stance_bce_weight"]) == 0.0
    assert float(ti["r29_s4_footstep_target_weight"]) == 0.0
    assert float(ti["r29_support_both_airborne_weight"]) == 0.0
    assert float(ti["r29_support_stance_velocity_weight"]) == 0.0
    assert float(ti["r29_swing_clearance_weight"]) == 0.0
    # And no contact-lock losses.
    assert float(ti["r29_contact_lock_offset_weight"]) == 0.0
    assert float(ti["r29_contact_lock_segment_drift_weight"]) == 0.0
    assert float(ti["r29_contact_lock_tracking_weight"]) == 0.0


def test_a1_consumes_s4_dim_13(tmp_path) -> None:
    cfg_dir, _ = _run_generator(tmp_path)
    cfg = _yaml_for(cfg_dir, "r29_ns_a1_c41_s4_g1")
    assert cfg["data"]["r29_support_variant"] == "S4-S1-phase-footstep"
    assert cfg["model"]["denoiser"]["r29_support_dim"] == 13
    # G1 losses still active.
    ti = cfg["loss"]["temporal_interaction"]
    assert float(ti["r29_gait_transition_rate_weight"]) > 0.0


def test_h1_i5_no_oracle_hint_fields(tmp_path) -> None:
    """H1: I5 cond live + NO old oracle-hint YAML keys anywhere."""
    cfg_dir, _ = _run_generator(tmp_path)
    cfg = _yaml_for(cfg_dir, "r29_ns_h1_i5_upper_bound")
    # I5 ON.
    assert cfg["data"]["r29_interaction_variant"] == "I5-allpart-contact-offset-masked"
    assert cfg["model"]["denoiser"]["r29_interaction_dim"] == 20
    # Full R0 cond (C41 + S4 + B4 + I5).
    assert cfg["data"]["r29_coarse_variant"] == "C41-current"
    assert cfg["data"]["r29_support_variant"] == "S4-S1-phase-footstep"
    assert cfg["data"]["r29_body_variant"] == "B4-lowpass-residual-mask"
    # No oracle-hint fields ANYWHERE in the YAML text.
    text = _yaml_text(cfg_dir, "r29_ns_h1_i5_upper_bound")
    for forbidden in (
        "use_oracle_interaction_hint",
        "oracle_hint_variant",
        "oracle_hint_dim",
        "oracle_hint_injection_mode",
        "oracle_hint_gate_bias_init",
    ):
        assert forbidden not in text, f"H1 yaml contains forbidden {forbidden!r}"
    # No contact-lock losses on H1 (per prompt §H1: condition upper bound, not loss test).
    ti = cfg["loss"]["temporal_interaction"]
    assert float(ti["r29_contact_lock_offset_weight"]) == 0.0
    assert float(ti["r29_contact_lock_segment_drift_weight"]) == 0.0
    assert float(ti["r29_contact_lock_tracking_weight"]) == 0.0
    assert float(ti["hint_contact_consistency_weight"]) == 0.0
    # No G1 losses on H1.
    assert float(ti["r29_gait_transition_rate_weight"]) == 0.0


def test_a2_c41_i5_g1_loss_only_s4(tmp_path) -> None:
    cfg_dir, _ = _run_generator(tmp_path)
    cfg = _yaml_for(cfg_dir, "r29_ns_a2_c41_i5_g1")
    den = cfg["model"]["denoiser"]
    # C41 + I5 consumed, NO S4 / NO B.
    assert den["r29_coarse_extra_dim"] == 18
    assert den["r29_interaction_dim"] == 20
    assert den["r29_support_dim"] == 0
    assert den["r29_body_refine_dim"] == 0
    # G1 losses ON.
    ti = cfg["loss"]["temporal_interaction"]
    assert float(ti["r29_gait_transition_rate_weight"]) > 0.0
    # S4 in data so G1 losses can read it.
    assert cfg["data"]["r29_support_variant"] == "S4-S1-phase-footstep"


def test_no_oracle_hint_fields_in_any_train_yaml(tmp_path) -> None:
    """Per prompt §2.4: no train variant in this matrix may emit the dead
    oracle-hint YAML keys."""
    cfg_dir, _ = _run_generator(tmp_path)
    for vid in EXPECTED_TRAIN:
        text = _yaml_text(cfg_dir, vid)
        for forbidden in (
            "use_oracle_interaction_hint",
            "oracle_hint_variant",
            "oracle_hint_dim",
            "oracle_hint_injection_mode",
            "oracle_hint_gate_bias_init",
        ):
            assert forbidden not in text, (
                f"{vid} yaml contains forbidden {forbidden!r}"
            )


def test_generator_honors_datasets_root(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATASETS_ROOT", "/tmp/fake_data_root_for_test")
    cfg_dir, manifest = _run_generator(tmp_path)
    assert manifest["data_root"].rstrip("/").endswith("fake_data_root_for_test")
    cfg = _yaml_for(cfg_dir, "r29_ns_a0_c41_g1_loss_s4")
    roots = [d["root"] for d in cfg["data"]["datasets"]]
    assert all("/tmp/fake_data_root_for_test/" in r for r in roots), roots


def test_manifest_rows_have_required_structure(tmp_path) -> None:
    _, manifest = _run_generator(tmp_path)
    for v in manifest["variants"]:
        if not v.get("train", True):
            continue
        # Per prompt §5: required manifest fields.
        assert "condition" in v
        assert "data_variants" in v["condition"]
        assert "model_dims" in v["condition"]
        assert "loss_only_families" in v["condition"]
        assert "loss_knobs" in v
        assert "architecture" in v
        assert "training_schedule" in v
        assert "references" in v
        assert "decision_question" in v


def test_validation_rejects_g1_loss_without_support_variant(tmp_path) -> None:
    """The generator's _validate_variant must reject configurations where
    G1 losses are active but data side has S0 (no support emitted)."""
    # Patch the VARIANTS list to add a broken one and check the generator
    # raises. Easiest path: import the module directly and call _validate_variant.
    sys.path.insert(0, str(ROOT / "scripts" / "stage_b_generator"))
    try:
        import round29_make_next_step_ablation_configs as mod
        v = mod.NextStepVariant(
            variant_id="broken_g1_no_s",
            purpose="test",
            decision_question="test",
            r29_support_variant="S0",            # data has no support
            r29_support_dim=0,
            r29_gait_transition_rate_weight=0.20,  # G1 loss active
        )
        with pytest.raises(ValueError, match="stage2_support"):
            mod._validate_variant(v)
    finally:
        sys.path.pop(0)


def test_validation_rejects_one_foot_support_weight_nonzero(tmp_path) -> None:
    """one_foot_support_weight > 0 must be rejected — it was R2's degeneracy
    source and is in the §8 'do not bring back' list."""
    sys.path.insert(0, str(ROOT / "scripts" / "stage_b_generator"))
    try:
        import round29_make_next_step_ablation_configs as mod
        v = mod.NextStepVariant(
            variant_id="broken_one_foot",
            purpose="test", decision_question="test",
            r29_gait_one_foot_support_weight=0.5,
        )
        with pytest.raises(ValueError, match="one_foot_support"):
            mod._validate_variant(v)
    finally:
        sys.path.pop(0)


def test_validation_rejects_i5_dim_with_i3_variant(tmp_path) -> None:
    """Model interaction_dim=20 (I5) but data variant=I3 must be rejected."""
    sys.path.insert(0, str(ROOT / "scripts" / "stage_b_generator"))
    try:
        import round29_make_next_step_ablation_configs as mod
        v = mod.NextStepVariant(
            variant_id="broken_i3_with_i5_dim",
            purpose="test", decision_question="test",
            r29_interaction_variant="I3-contact-offset-masked",
            r29_interaction_dim=20,  # model says I5
        )
        with pytest.raises(ValueError, match="interaction_dim"):
            mod._validate_variant(v)
    finally:
        sys.path.pop(0)


def test_all_use_input_add_adapter_injection(tmp_path) -> None:
    cfg_dir, _ = _run_generator(tmp_path)
    for vid in EXPECTED_TRAIN:
        cfg = _yaml_for(cfg_dir, vid)
        assert (
            cfg["model"]["denoiser"]["r29_injection_mode"] == "input_add_adapter"
        ), vid
