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


def test_six_variants_emitted(tmp_path: Path) -> None:
    _, manifest = _run_generator(tmp_path)
    ids = {v["variant_id"] for v in manifest["variants"]}
    expected = {
        "r29_ls_a2_baseline_from_scratch",
        "r29_ls_a3_baseline_from_scratch",
        "r29_ls_a2_relbeh_v2_anchor0_low",
        "r29_ls_a3_relbeh_v2_anchor0_low",
        "r29_ls_a2_relbeh_v2_anchor2_mixed",
        "r29_ls_a3_relbeh_v2_anchor2_mixed",
    }
    assert ids == expected


def test_injection_modes_per_variant(tmp_path: Path) -> None:
    _, manifest = _run_generator(tmp_path)
    by_id = {v["variant_id"]: v for v in manifest["variants"]}
    for family in ("baseline_from_scratch", "relbeh_v2_anchor0_low", "relbeh_v2_anchor2_mixed"):
        assert by_id[f"r29_ls_a2_{family}"]["injection_mode"] == "adapter_only"
        assert by_id[f"r29_ls_a3_{family}"]["injection_mode"] == "input_add_adapter"


def test_baseline_from_scratch_uses_original_a_group_weights(tmp_path: Path) -> None:
    """baseline_from_scratch must mirror the original a-group losses
    (pos_loss=5, anchor_pos=10, anchor_vel=2, world_vel=1) so the
    only varied axis is the init regime (no init_checkpoint)."""
    _, manifest = _run_generator(tmp_path)
    for v in manifest["variants"]:
        if v["loss_strategy"] != "baseline_from_scratch":
            continue
        k = v["knobs"]
        assert k["pos_loss_weight"] == 5.0
        assert k["anchor_joint_pos_weight"] == 10.0
        assert k["anchor_joint_vel_weight"] == 2.0
        assert k["world_joint_velocity_weight"] == 1.0
        # No relative / R29 consistency weights in the baseline.
        assert k["r29_interaction_consistency_weight"] == 0.0
        assert k["r29_support_both_airborne_weight"] == 0.0
        assert k["r29_support_stance_velocity_weight"] == 0.0
        assert k["r29_swing_clearance_weight"] == 0.0
        assert k["contact_rel_offset_weight"] == 0.0


def test_anchor0_low_pure_condition_supervision(tmp_path: Path) -> None:
    """anchor0_low must be a pure condition-consistency test:
    pos_loss=0, anchor=0, weak world_vel=0.5, low R29 weights at 0.10,
    swing_clearance ON at 0.10."""
    _, manifest = _run_generator(tmp_path)
    for v in manifest["variants"]:
        if v["loss_strategy"] != "relbeh_v2_anchor0_low":
            continue
        k = v["knobs"]
        assert k["pos_loss_weight"] == 0.0
        assert k["anchor_joint_pos_weight"] == 0.0
        assert k["anchor_joint_vel_weight"] == 0.0
        assert 0.0 < k["world_joint_velocity_weight"] < 1.0
        # R29 weights low and uniform per Codex review.
        assert k["r29_interaction_consistency_weight"] == 0.10
        assert k["r29_support_both_airborne_weight"] == 0.10
        assert k["r29_support_stance_velocity_weight"] == 0.10
        assert k["r29_swing_clearance_weight"] == 0.10
        # Existing relative contact losses on.
        assert k["contact_rel_offset_weight"] == 0.25
        assert k["contact_drift_weight"] == 0.25
        assert k["contact_tracking_weight"] == 0.25


def test_anchor2_mixed_weak_absolute_stabilizer(tmp_path: Path) -> None:
    """anchor2_mixed must have anchor_joint_pos=2 (weak stabilizer),
    anchor_vel=0.5, same low R29 weights as anchor0_low, AND
    swing_clearance ON. This is the 'is a weak absolute pull needed'
    counterfactual to anchor0_low."""
    _, manifest = _run_generator(tmp_path)
    for v in manifest["variants"]:
        if v["loss_strategy"] != "relbeh_v2_anchor2_mixed":
            continue
        k = v["knobs"]
        assert k["pos_loss_weight"] == 0.0
        assert k["anchor_joint_pos_weight"] == 2.0   # weak stabilizer
        assert k["anchor_joint_vel_weight"] == 0.5
        assert k["world_joint_velocity_weight"] == 0.5
        # Same low R29 weights as anchor0_low.
        assert k["r29_interaction_consistency_weight"] == 0.10
        assert k["r29_support_both_airborne_weight"] == 0.10
        assert k["r29_support_stance_velocity_weight"] == 0.10
        assert k["r29_swing_clearance_weight"] == 0.10


def test_yamls_use_48_clip_balanced_subset(tmp_path: Path) -> None:
    cfg_dir, manifest = _run_generator(tmp_path)
    for v in manifest["variants"]:
        yaml_path = cfg_dir / Path(v["config_path"]).name
        cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        sif = cfg["data"]["subset_indices_file"]
        assert sif.endswith("round27_tier0_train_indices_48_balanced.json")
        assert int(cfg["training"]["num_epochs"]) == 300
        assert bool(cfg["training"]["val_on_train_subset"]) is True


def test_yaml_injection_mode_matches_manifest(tmp_path: Path) -> None:
    cfg_dir, manifest = _run_generator(tmp_path)
    for v in manifest["variants"]:
        yaml_path = cfg_dir / Path(v["config_path"]).name
        cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        assert cfg["model"]["denoiser"]["r29_injection_mode"] == v["injection_mode"]


def test_r29_consistency_weights_in_yaml_for_v2_families(tmp_path: Path) -> None:
    """Both v2 families must declare every r29_*_weight (including
    swing_clearance) under loss.temporal_interaction."""
    cfg_dir, manifest = _run_generator(tmp_path)
    for v in manifest["variants"]:
        if not v["loss_strategy"].startswith("relbeh_v2"):
            continue
        yaml_path = cfg_dir / Path(v["config_path"]).name
        cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        ti = cfg["loss"]["temporal_interaction"]
        assert ti["r29_interaction_consistency_weight"] == 0.10
        assert ti["r29_support_both_airborne_weight"] == 0.10
        assert ti["r29_support_stance_velocity_weight"] == 0.10
        assert ti["r29_swing_clearance_weight"] == 0.10
        # Threshold default 5 cm.
        assert ti["r29_swing_clearance_m"] == 0.05


def test_swing_clearance_off_for_baseline(tmp_path: Path) -> None:
    """baseline_from_scratch must NOT enable swing_clearance — it tests
    the original a-group loss configuration."""
    cfg_dir, manifest = _run_generator(tmp_path)
    for v in manifest["variants"]:
        if v["loss_strategy"] != "baseline_from_scratch":
            continue
        yaml_path = cfg_dir / Path(v["config_path"]).name
        cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        ti = cfg["loss"]["temporal_interaction"]
        assert ti["r29_swing_clearance_weight"] == 0.0


def test_val_best_key_matches_enabled_loss_strategy(tmp_path: Path) -> None:
    """Do not select best_val.pt on a disabled loss component.
    - baseline_from_scratch: anchor active -> loss_anchor_joint_pos
    - relbeh_v2_anchor0_low: anchor disabled -> loss (total)
    - relbeh_v2_anchor2_mixed: anchor active (weight 2) -> loss_anchor_joint_pos
    """
    cfg_dir, manifest = _run_generator(tmp_path)
    for v in manifest["variants"]:
        yaml_path = cfg_dir / Path(v["config_path"]).name
        cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        if v["loss_strategy"] == "relbeh_v2_anchor0_low":
            assert v["val_best_key"] == "loss"
            assert cfg["training"]["val_best_key"] == "loss"
            assert cfg["loss"]["anchor_joint_pos_weight"] == 0.0
        else:
            assert v["val_best_key"] == "loss_anchor_joint_pos"
            assert cfg["training"]["val_best_key"] == "loss_anchor_joint_pos"
            assert cfg["loss"]["anchor_joint_pos_weight"] > 0.0


def test_step_fn_closure_does_not_reference_bare_cfg() -> None:
    """Regression test for the 2026-05-27 bug where step_fn referenced
    ``cfg.data.get(...)`` instead of pulling the value from its closure
    scope (build_anchordiff_step_fn). step_fn is a closure built outside
    main()'s scope, so the only valid config-like names it may reference
    are its own kwargs (anchor_cfg, temporal_loss_cfg, etc.) — NOT the
    bare ``cfg`` from main().

    Failure mode caught: at first batch, ``NameError: name 'cfg' is not
    defined`` would crash training. Pre-existing tests did not exercise
    the step_fn path, so a YAML smoke run would always have caught it
    earlier than a server training launch.
    """
    import ast

    src_path = (
        Path(__file__).resolve().parents[1]
        / "src" / "piano" / "training" / "train_anchordiff.py"
    )
    tree = ast.parse(src_path.read_text(encoding="utf-8"))

    builder_fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "build_anchordiff_step_fn":
            builder_fn = node
            break
    assert builder_fn is not None, "build_anchordiff_step_fn not found in train_anchordiff.py"

    step_fn_node = None
    for node in ast.walk(builder_fn):
        if isinstance(node, ast.FunctionDef) and node.name == "step_fn":
            step_fn_node = node
            break
    assert step_fn_node is not None, "step_fn closure not found inside build_anchordiff_step_fn"

    # Walk step_fn's body for any bare `cfg` name read.
    illegal_lines: list[int] = []
    for node in ast.walk(step_fn_node):
        if isinstance(node, ast.Name) and node.id == "cfg":
            illegal_lines.append(node.lineno)
        # Catch ``cfg.data.X`` / ``cfg.training.X`` attribute chains too.
        if isinstance(node, ast.Attribute):
            base = node
            while isinstance(base, ast.Attribute):
                base = base.value
            if isinstance(base, ast.Name) and base.id == "cfg":
                illegal_lines.append(node.lineno)

    assert not illegal_lines, (
        "step_fn closure references bare ``cfg`` at line(s) "
        f"{sorted(set(illegal_lines))} — this is a scope leak (cfg only "
        "exists in main()). Pass the value as a kwarg to "
        "build_anchordiff_step_fn or via temporal_loss_cfg instead."
    )


def test_temporal_interaction_loss_config_has_r29_hand_offset_clamp() -> None:
    """Regression test: the R29 interaction-consistency loss needs the
    hand_offset_clamp_m value that matches the dataset's I3 condition
    builder. Both must read from the same source (data.r29_hand_offset_clamp_m).
    """
    from piano.training.temporal_interaction_losses import TemporalInteractionLossConfig

    cfg = TemporalInteractionLossConfig()
    assert hasattr(cfg, "r29_hand_offset_clamp_m")
    # Default must match the dataset builder default (see
    # piano.data.stage2_oracle_conditions.build_interaction_condition).
    assert float(cfg.r29_hand_offset_clamp_m) == 2.0
