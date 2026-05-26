"""Manifest-generator integration test (Round-29 prompt §9.7-9.8).

Runs the config generator into a `tmp_path` (NOT the real repo) and
verifies that every variant declared in the source-of-truth list is
rendered with dims that match the condition-builder dim tables, and
that F3/F4 actually differ from F1 in the documented fields.

Per Codex post-review prompt §P2, tests must NOT mutate the real
workspace; the generator now exposes ``--config-dir`` /
``--analyses-dir`` and we drive both into tmp_path here.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts" / "stage_b_generator" / "round29_make_stage2_cond_ablation_configs.py"


def _run_generator(
    tmp_path: Path,
    *,
    extra_args: list[str] | None = None,
) -> tuple[Path, Path, dict]:
    cfg_dir = tmp_path / "configs" / "training"
    ana_dir = tmp_path / "analyses"
    cfg_dir.mkdir(parents=True)
    ana_dir.mkdir(parents=True)
    cmd = [
        sys.executable, str(GENERATOR),
        "--config-dir", str(cfg_dir),
        "--analyses-dir", str(ana_dir),
    ]
    if extra_args:
        cmd.extend(extra_args)
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    assert res.returncode == 0, res.stderr
    manifest_path = ana_dir / "round29_stage2_cond_ablation_manifest.json"
    assert manifest_path.exists(), f"manifest not written at {manifest_path}"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return cfg_dir, ana_dir, manifest


def test_generator_dry_run_runs_clean() -> None:
    """The generator must succeed in dry-run mode without disk side effects."""
    res = subprocess.run(
        [sys.executable, str(GENERATOR), "--dry-run"],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert res.returncode == 0, res.stderr


def test_generator_emits_all_groups_into_tmp(tmp_path: Path) -> None:
    """All six groups should be present in the rendered manifest."""
    _, _, manifest = _run_generator(tmp_path)
    groups = {v["group"] for v in manifest["variants"]}
    assert {
        "A_injection", "B_coarse", "C_interaction",
        "D_support", "E_body", "F_final",
    }.issubset(groups), f"Missing groups: {groups}"


def test_manifest_dims_match_builder_dim_tables(tmp_path: Path) -> None:
    """Every manifest row's expected_dense_dims must match the builder
    dim tables. Source-of-truth divergence here means a config will
    mis-build the model dataset side."""
    from piano.data.stage2_oracle_conditions import (
        BODY_VARIANT_DIMS,
        COARSE_VARIANT_DIMS,
        INTERACTION_VARIANT_DIMS,
        SUPPORT_VARIANT_DIMS,
    )

    _, _, manifest = _run_generator(tmp_path)
    for v in manifest["variants"]:
        ed = v["expected_dense_dims"]
        assert ed["coarse_extra"] == COARSE_VARIANT_DIMS[v["coarse_variant"]]
        assert ed["interaction"] == INTERACTION_VARIANT_DIMS[v["interaction_variant"]]
        assert ed["support"] == SUPPORT_VARIANT_DIMS[v["support_variant"]]
        assert ed["body_refine"] == BODY_VARIANT_DIMS[v["body_variant"]]


def test_manifest_paths_are_repo_relative(tmp_path: Path) -> None:
    """Codex P2: no absolute paths in manifest — must be portable repo-relative."""
    _, _, manifest = _run_generator(tmp_path)
    for v in manifest["variants"]:
        for key in ("config_path", "output_dir", "subset_file"):
            val = v[key]
            assert val, f"{v['variant_id']} {key} is empty"
            # Reject Windows-style absolutes (E:\ ...) and POSIX absolutes (/...).
            assert ":" not in val, f"{v['variant_id']} {key} contains drive letter: {val!r}"
            assert not val.startswith("/"), f"{v['variant_id']} {key} is POSIX-absolute: {val!r}"
    defaults = manifest["defaults"]
    for key in ("balanced_subset_file", "body_action_subset_file"):
        val = defaults[key]
        assert ":" not in val, f"defaults.{key} contains drive letter: {val!r}"
        assert not val.startswith("/"), f"defaults.{key} is POSIX-absolute: {val!r}"


def test_f3_heldout_differs_from_f1_in_val_subset(tmp_path: Path) -> None:
    """Codex P1: r29_f3_best_full_heldout must set val_on_train_subset=false."""
    _, _, manifest = _run_generator(tmp_path)
    by_id = {v["variant_id"]: v for v in manifest["variants"]}
    assert by_id["r29_f1_best_full"]["val_on_train_subset"] is True
    assert by_id["r29_f3_best_full_heldout"]["val_on_train_subset"] is False


def test_f4_seed_differs_from_f1(tmp_path: Path) -> None:
    """Codex P1: r29_f4_best_full_seed2 must set seed=43."""
    _, _, manifest = _run_generator(tmp_path)
    by_id = {v["variant_id"]: v for v in manifest["variants"]}
    assert by_id["r29_f1_best_full"]["seed"] == 42
    assert by_id["r29_f4_best_full_seed2"]["seed"] == 43


def test_generated_configs_reflect_f3_f4_overrides(tmp_path: Path) -> None:
    """The YAML files for F3/F4 must actually contain the overrides."""
    cfg_dir, _, _ = _run_generator(tmp_path)
    f1 = (cfg_dir / "anchordiff_r29_f1_best_full.yaml").read_text("utf-8")
    f3 = (cfg_dir / "anchordiff_r29_f3_best_full_heldout.yaml").read_text("utf-8")
    f4 = (cfg_dir / "anchordiff_r29_f4_best_full_seed2.yaml").read_text("utf-8")
    assert "val_on_train_subset: true" in f1
    assert "val_on_train_subset: false" in f3
    assert "val_on_train_subset: true" in f4
    assert "seed: 42" in f1
    assert "seed: 42" in f3
    assert "seed: 43" in f4


def test_only_groups_filter(tmp_path: Path) -> None:
    cfg_dir, _, manifest = _run_generator(
        tmp_path, extra_args=["--only-groups", "A_injection"],
    )
    groups = {v["group"] for v in manifest["variants"]}
    assert groups == {"A_injection"}
    # Files for non-A groups must NOT have been written.
    for vid in ("r29_b0_c23_only", "r29_f0_baseline"):
        assert not (cfg_dir / f"anchordiff_{vid}.yaml").exists()


def test_data_root_override_propagates_into_configs(tmp_path: Path) -> None:
    """--data-root must replace the default data root in every YAML's
    data.datasets block. Caught the bug from the first server run where
    Windows paths leaked into Linux configs."""
    fake_root = "/srv/some/path/InterAct/piano_official_process_4"
    cfg_dir, _, manifest = _run_generator(
        tmp_path, extra_args=["--data-root", fake_root],
    )
    sample = (cfg_dir / "anchordiff_r29_a0_input_add.yaml").read_text("utf-8")
    for sub in ("chairs", "imhd", "neuraldome", "omomo_correct_v2"):
        assert f'root: "{fake_root}/{sub}"' in sample
    # The default Windows path must NOT appear.
    assert "E:/Project/Datasets" not in sample
    # Manifest defaults block records the override.
    assert manifest["defaults"]["data_root"] == fake_root
