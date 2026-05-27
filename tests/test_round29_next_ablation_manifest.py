"""Round-29 next-baseline ablation generator + manifest tests.

Per analyses/2026-05-27_round29_next_ablation_execution_prompt_for_claude_code.md
§"Tests Required" / §"Required Variant Matrix".

The matrix contains 5 train variants (B0/B1/G1/G2/H1) + 1 R0 reference
entry (train=false). All 5 train variants must be full-data, from
scratch, bs=32/accum=1/80ep.
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
    / "round29_make_next_ablation_configs.py"
)

EXPECTED_TRAIN_VARIANTS = (
    "r29_nb_b0_no_r29_cond",
    "r29_nb_b1_c41_only",
    "r29_nb_g1_phasefree_gait_fixed",
    "r29_nb_g2_strong_s4_oracle",
    "r29_nb_h1_r0_plus_oracle_full_hint",
)
R0_REF_VARIANT = "r29_ft_r0_clean_a3_baseline"


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
    manifest_path = ana_dir / "round29_next_ablation_manifest.json"
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
    # 5 new train variants + manifest + 1 R0 ref summary line.
    assert "5 new train variants + 1 R0 reference" in res.stdout


def test_manifest_has_five_train_variants_plus_r0_reference(tmp_path) -> None:
    _, manifest = _run_generator(tmp_path)
    variants = manifest["variants"]
    train_rows = [v for v in variants if v.get("train", True)]
    ref_rows = [v for v in variants if not v.get("train", True)]
    assert len(train_rows) == 5, [v["variant_id"] for v in train_rows]
    assert len(ref_rows) == 1, [v["variant_id"] for v in ref_rows]
    train_ids = {v["variant_id"] for v in train_rows}
    assert train_ids == set(EXPECTED_TRAIN_VARIANTS)
    assert ref_rows[0]["variant_id"] == R0_REF_VARIANT


def test_all_train_variants_are_full_data_from_scratch(tmp_path) -> None:
    cfg_dir, _ = _run_generator(tmp_path)
    for vid in EXPECTED_TRAIN_VARIANTS:
        cfg = _yaml_for(cfg_dir, vid)
        # No subset_indices_file → full data.
        assert "subset_indices_file" not in (cfg.get("data") or {}), vid
        # No init_checkpoint → from scratch.
        assert "init_checkpoint" not in (cfg.get("training") or {}), vid
        # Schedule.
        assert cfg["training"]["batch_size"] == 32, vid
        assert cfg["training"]["gradient_accumulation_steps"] == 1, vid
        assert cfg["training"]["num_epochs"] == 80, vid


def test_b0_has_all_r29_dims_zero_and_injection_off(tmp_path) -> None:
    cfg_dir, manifest = _run_generator(tmp_path)
    cfg = _yaml_for(cfg_dir, "r29_nb_b0_no_r29_cond")
    den = cfg["model"]["denoiser"]
    assert den["use_round29_cond_injection"] is False
    assert den["r29_coarse_extra_dim"] == 0
    assert den["r29_interaction_dim"] == 0
    assert den["r29_support_dim"] == 0
    assert den["r29_body_refine_dim"] == 0
    data = cfg["data"]
    assert data["r29_coarse_variant"] == "C23"
    assert data["r29_interaction_variant"] == "I0"
    assert data["r29_support_variant"] == "S0"
    assert data["r29_body_variant"] == "B0"


def test_b1_has_c41_only(tmp_path) -> None:
    cfg_dir, _ = _run_generator(tmp_path)
    cfg = _yaml_for(cfg_dir, "r29_nb_b1_c41_only")
    den = cfg["model"]["denoiser"]
    assert den["use_round29_cond_injection"] is True
    assert den["r29_coarse_extra_dim"] == 18, "C41 extra dim must be 18"
    assert den["r29_interaction_dim"] == 0
    assert den["r29_support_dim"] == 0
    assert den["r29_body_refine_dim"] == 0
    data = cfg["data"]
    assert data["r29_coarse_variant"] == "C41-current"
    assert data["r29_interaction_variant"] == "I0"
    assert data["r29_support_variant"] == "S0"
    assert data["r29_body_variant"] == "B0"


def test_g1_activates_new_phasefree_gait_weights(tmp_path) -> None:
    cfg_dir, _ = _run_generator(tmp_path)
    cfg = _yaml_for(cfg_dir, "r29_nb_g1_phasefree_gait_fixed")
    ti = cfg["loss"]["temporal_interaction"]
    # New G1 weights non-zero.
    assert float(ti["r29_gait_soft_stance_velocity_weight"]) > 0.0
    assert float(ti["r29_gait_transition_rate_weight"]) > 0.0
    assert float(ti["r29_gait_duty_cycle_weight"]) > 0.0
    assert float(ti["r29_gait_both_state_match_weight"]) > 0.0
    # Old R2 one_foot_support MUST be 0 (it was the degeneracy source).
    assert float(ti["r29_gait_one_foot_support_weight"]) == 0.0


def test_g2_uses_strong_s4_weights(tmp_path) -> None:
    cfg_dir, _ = _run_generator(tmp_path)
    cfg = _yaml_for(cfg_dir, "r29_nb_g2_strong_s4_oracle")
    ti = cfg["loss"]["temporal_interaction"]
    assert float(ti["r29_support_both_airborne_weight"]) == 0.10
    assert float(ti["r29_support_stance_velocity_weight"]) == 0.10
    assert float(ti["r29_swing_clearance_weight"]) == 0.10
    assert float(ti["r29_swing_clearance_m"]) == 0.05
    assert float(ti["r29_s4_stance_bce_weight"]) == 0.30
    assert float(ti["r29_s4_footstep_target_weight"]) == 0.40


def test_h1_enables_oracle_interaction_hint_full(tmp_path) -> None:
    cfg_dir, _ = _run_generator(tmp_path)
    cfg = _yaml_for(cfg_dir, "r29_nb_h1_r0_plus_oracle_full_hint")
    # Data side.
    data = cfg["data"]
    assert data.get("use_oracle_interaction_hint") is True
    assert data.get("oracle_hint_variant") == "full"
    assert float(data.get("oracle_hint_fps", 0.0)) == 20.0
    # Model side.
    den = cfg["model"]["denoiser"]
    assert den.get("use_oracle_interaction_hint") is True
    assert int(den.get("oracle_hint_dim", 0)) == 13
    assert den.get("oracle_hint_injection_mode") == "input_add"
    # Body action hint OFF.
    assert data.get("use_body_action_hint") in (None, False)
    assert den.get("body_action_hint_dim", 0) in (0, None)
    # H1 still uses R0 cond (C41 + I3 + S4 + B4).
    assert data["r29_coarse_variant"] == "C41-current"
    assert data["r29_interaction_variant"] == "I3-contact-offset-masked"


def test_manifest_uses_data_root_from_env(tmp_path, monkeypatch) -> None:
    """Generator must honour DATASETS_ROOT (so the launcher can regenerate
    configs with the server path)."""
    monkeypatch.setenv("DATASETS_ROOT", "/tmp/fake_data_root_for_test")
    cfg_dir, manifest = _run_generator(tmp_path)
    assert manifest["data_root"].rstrip("/").endswith("fake_data_root_for_test")
    cfg = _yaml_for(cfg_dir, "r29_nb_b0_no_r29_cond")
    roots = [d["root"] for d in cfg["data"]["datasets"]]
    assert all("/tmp/fake_data_root_for_test/" in r for r in roots), roots


def test_all_variants_use_input_add_adapter_injection(tmp_path) -> None:
    """All five new variants use input_add_adapter injection (R0 default)."""
    cfg_dir, _ = _run_generator(tmp_path)
    for vid in EXPECTED_TRAIN_VARIANTS:
        cfg = _yaml_for(cfg_dir, vid)
        assert (
            cfg["model"]["denoiser"]["r29_injection_mode"] == "input_add_adapter"
        ), vid


def test_h1_does_not_enable_hint_contact_consistency(tmp_path) -> None:
    """Per prompt §H1: H1 is a condition upper-bound test, not a
    loss-consistency test. hint_contact_consistency_weight must be 0."""
    cfg_dir, _ = _run_generator(tmp_path)
    cfg = _yaml_for(cfg_dir, "r29_nb_h1_r0_plus_oracle_full_hint")
    ti = cfg["loss"]["temporal_interaction"]
    assert float(ti.get("hint_contact_consistency_weight", 0.0)) == 0.0


def test_r0_reference_entry_marks_train_false(tmp_path) -> None:
    _, manifest = _run_generator(tmp_path)
    ref = next(
        v for v in manifest["variants"] if v["variant_id"] == R0_REF_VARIANT
    )
    assert ref["train"] is False
    # The reference should point at the existing R29-FT artifacts.
    assert "r29_ft" in ref["config_path"]
    assert "r29_ft" in ref["output_dir"]
