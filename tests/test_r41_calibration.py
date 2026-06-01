"""Tests for round41_cascade_calibration.py — recommendation logic.

Covers Codex r41_calibration_next_steps 2026-06-02 spec:
- target_center=0.3 conservative nudge probe (default)
- --max-w-total cap (default 5.0)
- capped flag + recommended_w_total_uncapped reporting
- control-cell skip
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_omegaconf_available = importlib.util.find_spec("omegaconf") is not None
needs_omegaconf = pytest.mark.skipif(
    not _omegaconf_available, reason="omegaconf not installed"
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/stage_a_generator/round41_cascade_calibration.py"


def _load_calib_module():
    spec = importlib.util.spec_from_file_location(
        "round41_cascade_calibration_under_test", SCRIPT,
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_defaults_match_codex_spec():
    m = _load_calib_module()
    assert m.DEFAULT_TARGET_MIN == 0.2
    assert m.DEFAULT_TARGET_MAX == 0.5
    assert m.DEFAULT_TARGET_CENTER == 0.3
    assert m.DEFAULT_MAX_W_TOTAL == 5.0


def test_recommend_a3_capped_at_target_center_03():
    """A3 server measurement (ratio 0.069) — at center=0.3 the linear
    rescale would be ~4.35; under the 5.0 cap it stays uncapped."""
    m = _load_calib_module()
    rec = m._recommend_w_total(
        measured_ratio=0.069,
        target_center=0.3, target_min=0.2, target_max=0.5,
        max_w_total=5.0, current_w_total=1.0,
    )
    # ~4.35 ± rounding
    assert 4.3 < rec["recommended_w_total"] < 4.4
    assert rec["capped"] is False
    assert not rec["in_band"]


def test_recommend_a3_capped_at_center_10():
    """A3 ratio 0.069 at center=1.0 would recommend ~14.5x;
    cap to 5.0."""
    m = _load_calib_module()
    rec = m._recommend_w_total(
        measured_ratio=0.069,
        target_center=1.0, target_min=0.5, target_max=1.5,
        max_w_total=5.0, current_w_total=1.0,
    )
    assert rec["recommended_w_total"] == 5.0
    assert rec["capped"] is True
    assert 14.0 < rec["recommended_w_total_uncapped"] < 15.0


def test_recommend_in_band_keeps_current():
    """Ratio inside [0.2, 0.5] — recommendation = current."""
    m = _load_calib_module()
    rec = m._recommend_w_total(
        measured_ratio=0.3,
        target_center=0.3, target_min=0.2, target_max=0.5,
        max_w_total=5.0, current_w_total=2.5,
    )
    assert rec["in_band"] is True
    assert rec["recommended_w_total"] == 2.5
    assert rec["capped"] is False


def test_recommend_disabled_cap():
    """max_w_total=None → no cap."""
    m = _load_calib_module()
    rec = m._recommend_w_total(
        measured_ratio=0.069,
        target_center=1.0, target_min=0.5, target_max=1.5,
        max_w_total=None, current_w_total=1.0,
    )
    assert rec["capped"] is False
    assert 14.0 < rec["recommended_w_total"] < 15.0


def test_recommend_zero_ratio_returns_current():
    m = _load_calib_module()
    rec = m._recommend_w_total(
        measured_ratio=0.0,
        target_center=0.3, target_min=0.2, target_max=0.5,
        max_w_total=5.0, current_w_total=1.0,
    )
    assert rec["recommended_w_total"] == 1.0
    assert rec["capped"] is False


@needs_omegaconf
def test_read_cascade_info_control_cell(tmp_path):
    """A0-style cfg (cascade.enabled=False) detected as control."""
    m = _load_calib_module()
    cfg = tmp_path / "a0_control.yaml"
    cfg.write_text(
        "cascade:\n"
        "  enabled: false\n"
        "  w_total: 1.0\n"
        "  w_motion_mse: 0.0\n"
        "  w_world_joint_vel: 0.0\n"
        "  w_l_pos_full: 0.0\n"
        "  w_anchor_joint_pos: 0.0\n"
    )
    info = m._read_cascade_info(cfg)
    assert info["control_cell"] is True
    assert info["enabled"] is False


@needs_omegaconf
def test_read_cascade_info_enabled_cell(tmp_path):
    """A1-style cfg (cascade.enabled=True, motion_mse=1.0) NOT control."""
    m = _load_calib_module()
    cfg = tmp_path / "a1_cascade.yaml"
    cfg.write_text(
        "cascade:\n"
        "  enabled: true\n"
        "  w_total: 1.0\n"
        "  w_motion_mse: 1.0\n"
        "  w_world_joint_vel: 0.0\n"
        "  w_l_pos_full: 0.0\n"
        "  w_anchor_joint_pos: 0.0\n"
    )
    info = m._read_cascade_info(cfg)
    assert info["control_cell"] is False
    assert info["enabled"] is True
    assert info["weights"]["w_motion_mse"] == 1.0


@needs_omegaconf
def test_read_cascade_info_enabled_but_all_zero(tmp_path):
    """Edge case: enabled=true but all w_* = 0 → still control."""
    m = _load_calib_module()
    cfg = tmp_path / "edge.yaml"
    cfg.write_text(
        "cascade:\n"
        "  enabled: true\n"
        "  w_total: 1.0\n"
        "  w_motion_mse: 0.0\n"
        "  w_world_joint_vel: 0.0\n"
        "  w_l_pos_full: 0.0\n"
        "  w_anchor_joint_pos: 0.0\n"
    )
    info = m._read_cascade_info(cfg)
    # all-zero weights → no cascade gradient → control behavior
    assert info["control_cell"] is True
