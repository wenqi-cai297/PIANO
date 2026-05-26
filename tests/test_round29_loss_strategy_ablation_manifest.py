"""Round-29 loss-strategy ablation generator tests.

Per analyses/2026-05-27_round29_loss_strategy_ablation_prompt_for_claude_code.md
§10 (config-generator tests).

Runs the generator into a tmp_path (NOT the real repo) and verifies:

* 4 variants are emitted.
* Variant IDs and injection modes line up (A2=adapter_only, A3=input_add_adapter).
* `relative_behavior` variants set pos_loss / anchor_pos / anchor_vel = 0
  and have positive R29 consistency weights.
* `no_dense_pos` variants set pos_loss=0 but KEEP anchor weights > 0.
* Generated YAMLs use the 48-clip balanced subset by default.
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
    / "round29_make_loss_strategy_ablation_configs.py"
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
    manifest_path = ana_dir / "round29_loss_strategy_ablation_manifest.json"
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
        "r29_ls_a2_no_dense_pos",
        "r29_ls_a3_no_dense_pos",
        "r29_ls_a2_relative_behavior",
        "r29_ls_a3_relative_behavior",
    }


def test_injection_modes_per_variant(tmp_path: Path) -> None:
    _, manifest = _run_generator(tmp_path)
    by_id = {v["variant_id"]: v for v in manifest["variants"]}
    # A2 ⇒ adapter_only.
    assert by_id["r29_ls_a2_no_dense_pos"]["injection_mode"] == "adapter_only"
    assert by_id["r29_ls_a2_relative_behavior"]["injection_mode"] == "adapter_only"
    # A3 ⇒ input_add_adapter.
    assert by_id["r29_ls_a3_no_dense_pos"]["injection_mode"] == "input_add_adapter"
    assert by_id["r29_ls_a3_relative_behavior"]["injection_mode"] == "input_add_adapter"


def test_no_dense_pos_keeps_anchor_drops_pos_loss(tmp_path: Path) -> None:
    _, manifest = _run_generator(tmp_path)
    for v in manifest["variants"]:
        if v["loss_strategy"] != "no_dense_pos":
            continue
        k = v["knobs"]
        assert k["pos_loss_weight"] == 0.0
        assert k["anchor_joint_pos_weight"] > 0.0
        assert k["anchor_joint_vel_weight"] > 0.0
        # No R29 consistency weights in the no_dense_pos variant.
        assert k["r29_interaction_consistency_weight"] == 0.0
        assert k["r29_support_both_airborne_weight"] == 0.0
        assert k["r29_support_stance_velocity_weight"] == 0.0


def test_relative_behavior_drops_all_absolute_gt_and_enables_r29_consistency(
    tmp_path: Path,
) -> None:
    _, manifest = _run_generator(tmp_path)
    for v in manifest["variants"]:
        if v["loss_strategy"] != "relative_behavior":
            continue
        k = v["knobs"]
        # Absolute-GT pulls off.
        assert k["pos_loss_weight"] == 0.0
        assert k["anchor_joint_pos_weight"] == 0.0
        assert k["anchor_joint_vel_weight"] == 0.0
        # Weak global velocity prior (per prompt §7.2).
        assert 0.0 < k["world_joint_velocity_weight"] < 1.0
        # R29 consistency weights enabled.
        assert k["r29_interaction_consistency_weight"] > 0.0
        assert k["r29_support_both_airborne_weight"] > 0.0
        assert k["r29_support_stance_velocity_weight"] > 0.0
        # Existing relative contact losses also on (rel_offset / drift / tracking).
        assert k["contact_rel_offset_weight"] > 0.0
        assert k["contact_drift_weight"] > 0.0
        assert k["contact_tracking_weight"] > 0.0


def test_yamls_use_48_clip_balanced_subset(tmp_path: Path) -> None:
    cfg_dir, manifest = _run_generator(tmp_path)
    for v in manifest["variants"]:
        yaml_path = cfg_dir / Path(v["config_path"]).name
        cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        # subset_indices_file is present and points at the balanced JSON.
        sif = cfg["data"]["subset_indices_file"]
        assert sif.endswith("round27_tier0_train_indices_48_balanced.json")
        # 48-clip overfit defaults (per prompt §8).
        assert int(cfg["training"]["num_epochs"]) == 300
        assert bool(cfg["training"]["val_on_train_subset"]) is True


def test_yaml_injection_mode_matches_manifest(tmp_path: Path) -> None:
    """Sanity-check that the rendered YAML's r29_injection_mode matches
    the manifest's injection_mode field."""
    cfg_dir, manifest = _run_generator(tmp_path)
    for v in manifest["variants"]:
        yaml_path = cfg_dir / Path(v["config_path"]).name
        cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        assert cfg["model"]["denoiser"]["r29_injection_mode"] == v["injection_mode"]


def test_relative_behavior_yaml_has_r29_consistency_weights(tmp_path: Path) -> None:
    """The relative_behavior YAML must declare every r29_*_consistency
    weight under loss.temporal_interaction with the expected sign."""
    cfg_dir, manifest = _run_generator(tmp_path)
    for v in manifest["variants"]:
        if v["loss_strategy"] != "relative_behavior":
            continue
        yaml_path = cfg_dir / Path(v["config_path"]).name
        cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        ti = cfg["loss"]["temporal_interaction"]
        assert ti["r29_interaction_consistency_weight"] > 0.0
        assert ti["r29_support_both_airborne_weight"] > 0.0
        assert ti["r29_support_stance_velocity_weight"] > 0.0


def test_no_dense_pos_yaml_keeps_anchor(tmp_path: Path) -> None:
    cfg_dir, manifest = _run_generator(tmp_path)
    for v in manifest["variants"]:
        if v["loss_strategy"] != "no_dense_pos":
            continue
        yaml_path = cfg_dir / Path(v["config_path"]).name
        cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        assert cfg["loss"]["pos_loss_weight"] == 0.0
        assert cfg["loss"]["anchor_joint_pos_weight"] > 0.0
        assert cfg["loss"]["anchor_joint_vel_weight"] > 0.0
