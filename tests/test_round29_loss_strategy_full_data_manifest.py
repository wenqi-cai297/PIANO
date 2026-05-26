"""Round-29 loss-strategy FULL-DATA generator tests.

Per analyses/2026-05-27_round29_loss_strategy_v2_codex_review.md §Final recommendation:
the 4-variant full-data matrix must emit (A2, A3) × (baseline_from_scratch,
anchor2_mixed) with the correct loss weights, full-data schedule, and
val_best_key, and have NO subset_indices_file.
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
    / "round29_make_loss_strategy_full_data_configs.py"
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
    manifest_path = ana_dir / "round29_loss_strategy_full_data_manifest.json"
    assert manifest_path.exists(), f"manifest missing at {manifest_path}"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return cfg_dir, manifest


def test_generator_dry_run_runs_clean() -> None:
    res = subprocess.run(
        [sys.executable, str(GENERATOR), "--dry-run"],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert res.returncode == 0, res.stderr


def test_four_variants_emitted(tmp_path: Path) -> None:
    _, manifest = _run_generator(tmp_path)
    ids = {v["variant_id"] for v in manifest["variants"]}
    assert ids == {
        "r29_lsf_a2_baseline_from_scratch",
        "r29_lsf_a3_baseline_from_scratch",
        "r29_lsf_a2_anchor2_mixed",
        "r29_lsf_a3_anchor2_mixed",
    }


def test_injection_modes_per_variant(tmp_path: Path) -> None:
    _, manifest = _run_generator(tmp_path)
    by_id = {v["variant_id"]: v for v in manifest["variants"]}
    for family in ("baseline_from_scratch", "anchor2_mixed"):
        assert by_id[f"r29_lsf_a2_{family}"]["injection_mode"] == "adapter_only"
        assert by_id[f"r29_lsf_a3_{family}"]["injection_mode"] == "input_add_adapter"


def test_baseline_uses_original_a_group_weights(tmp_path: Path) -> None:
    _, manifest = _run_generator(tmp_path)
    for v in manifest["variants"]:
        if v["loss_strategy"] != "baseline_from_scratch":
            continue
        k = v["knobs"]
        assert k["pos_loss_weight"] == 5.0
        assert k["anchor_joint_pos_weight"] == 10.0
        assert k["anchor_joint_vel_weight"] == 2.0
        assert k["world_joint_velocity_weight"] == 1.0
        # No relative / R29 / swing_clearance.
        assert k["r29_interaction_consistency_weight"] == 0.0
        assert k["r29_support_both_airborne_weight"] == 0.0
        assert k["r29_support_stance_velocity_weight"] == 0.0
        assert k["r29_swing_clearance_weight"] == 0.0
        assert k["contact_rel_offset_weight"] == 0.0


def test_anchor2_mixed_uses_v2_winner_weights(tmp_path: Path) -> None:
    """Must match the v2 48-clip winner weights exactly."""
    _, manifest = _run_generator(tmp_path)
    for v in manifest["variants"]:
        if v["loss_strategy"] != "anchor2_mixed":
            continue
        k = v["knobs"]
        assert k["pos_loss_weight"] == 0.0
        assert k["anchor_joint_pos_weight"] == 2.0
        assert k["anchor_joint_vel_weight"] == 0.5
        assert k["world_joint_velocity_weight"] == 0.5
        # Existing relative contact losses (Codex §"swing_clearance kept" + §7.2).
        assert k["contact_rel_offset_weight"] == 0.25
        assert k["contact_drift_weight"] == 0.25
        assert k["contact_tracking_weight"] == 0.25
        # R29 weights all at 0.10 per Codex retune.
        assert k["r29_interaction_consistency_weight"] == 0.10
        assert k["r29_support_both_airborne_weight"] == 0.10
        assert k["r29_support_stance_velocity_weight"] == 0.10
        assert k["r29_swing_clearance_weight"] == 0.10
        assert k["r29_swing_clearance_m"] == 0.05


def test_full_data_schedule(tmp_path: Path) -> None:
    """No subset_indices_file; 80 ep; heldout val; warmup=250;
    save_every=10; stage1_coarse_noise_std=0.05."""
    cfg_dir, manifest = _run_generator(tmp_path)
    for v in manifest["variants"]:
        yaml_path = cfg_dir / Path(v["config_path"]).name
        cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        assert "subset_indices_file" not in cfg["data"], (
            "full-data YAML should NOT have a subset_indices_file"
        )
        assert int(cfg["training"]["num_epochs"]) == 80
        assert bool(cfg["training"]["val_on_train_subset"]) is False
        assert int(cfg["training"]["val_every_epochs"]) == 5
        assert int(cfg["training"]["scheduler"]["warmup_steps"]) == 250
        assert float(cfg["training"]["stage1_coarse_noise_std"]) == 0.05
        assert int(cfg["logging"]["save_every_n_epochs"]) == 10


def test_yaml_injection_mode_matches_manifest(tmp_path: Path) -> None:
    cfg_dir, manifest = _run_generator(tmp_path)
    for v in manifest["variants"]:
        yaml_path = cfg_dir / Path(v["config_path"]).name
        cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        assert cfg["model"]["denoiser"]["r29_injection_mode"] == v["injection_mode"]


def test_val_best_key_is_loss_anchor_joint_pos(tmp_path: Path) -> None:
    """All 4 variants have anchor_joint_pos_weight > 0 (baseline=10, mixed=2),
    so loss_anchor_joint_pos is a valid live metric for best-ckpt selection."""
    cfg_dir, manifest = _run_generator(tmp_path)
    for v in manifest["variants"]:
        assert v["val_best_key"] == "loss_anchor_joint_pos"
        yaml_path = cfg_dir / Path(v["config_path"]).name
        cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        assert cfg["training"]["val_best_key"] == "loss_anchor_joint_pos"
        assert cfg["loss"]["anchor_joint_pos_weight"] > 0.0


def test_r29_consistency_weights_in_yaml_for_anchor2_mixed(tmp_path: Path) -> None:
    cfg_dir, manifest = _run_generator(tmp_path)
    for v in manifest["variants"]:
        if v["loss_strategy"] != "anchor2_mixed":
            continue
        yaml_path = cfg_dir / Path(v["config_path"]).name
        cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        ti = cfg["loss"]["temporal_interaction"]
        assert ti["r29_interaction_consistency_weight"] == 0.10
        assert ti["r29_support_both_airborne_weight"] == 0.10
        assert ti["r29_support_stance_velocity_weight"] == 0.10
        assert ti["r29_swing_clearance_weight"] == 0.10
        assert ti["r29_swing_clearance_m"] == 0.05


def test_swing_clearance_off_for_baseline(tmp_path: Path) -> None:
    cfg_dir, manifest = _run_generator(tmp_path)
    for v in manifest["variants"]:
        if v["loss_strategy"] != "baseline_from_scratch":
            continue
        yaml_path = cfg_dir / Path(v["config_path"]).name
        cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        ti = cfg["loss"]["temporal_interaction"]
        assert ti["r29_swing_clearance_weight"] == 0.0


def test_full_dense_content_in_all_variants(tmp_path: Path) -> None:
    """All 4 variants must use FULL-DENSE C/I/S/B content (so any differences
    in diag results are due to loss strategy / injection, not condition content)."""
    cfg_dir, manifest = _run_generator(tmp_path)
    for v in manifest["variants"]:
        yaml_path = cfg_dir / Path(v["config_path"]).name
        cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        assert cfg["data"]["r29_coarse_variant"] == "C41-current"
        assert cfg["data"]["r29_interaction_variant"] == "I3-contact-offset-masked"
        assert cfg["data"]["r29_support_variant"] == "S4-S1-phase-footstep"
        assert cfg["data"]["r29_body_variant"] == "B4-lowpass-residual-mask"
