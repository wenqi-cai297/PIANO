"""Unit tests for motion-representation round-trip floor diagnostic helpers.

Per analyses/2026-05-28_round29_next_step_ablation_execution_prompt_for_claude_code.md §8.4.

Isolates the pure metric helpers (FK-vs-raw error, aggregation,
interpretation thresholds). The real diag script needs torch + dataset +
NPZ files; these tests synthesise arrays directly to validate the math.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "stage_b_generator"
sys.path.insert(0, str(SCRIPTS))

from round29_motion_repr_roundtrip_diag import (  # noqa: E402
    PerClipRow, SMPL_JOINT_NAMES, PART_TO_JOINT_IDX,
    _per_joint_error_cm, _per_part_contact_floor,
    _percentile, _aggregate, _interpretation, _valid_length_from_batch,
)


def test_per_joint_error_zero_when_arrays_match():
    fk = np.random.RandomState(0).randn(20, 22, 3).astype(np.float32)
    raw = fk.copy()
    err_cm = _per_joint_error_cm(fk, raw)
    assert err_cm.shape == (20, 22)
    assert np.allclose(err_cm, 0.0, atol=1e-6)


def test_per_joint_error_uses_cm_conversion():
    """Error in metres must be converted to cm (×100)."""
    fk = np.zeros((10, 22, 3), dtype=np.float32)
    raw = np.zeros((10, 22, 3), dtype=np.float32)
    raw[..., 0] = 0.10   # 10 cm shift on X for all joints/frames
    err_cm = _per_joint_error_cm(fk, raw)
    assert np.allclose(err_cm, 10.0, atol=1e-6)


def test_per_joint_error_handles_length_mismatch():
    """Length truncation to min(fk.T, raw.T)."""
    fk = np.zeros((15, 22, 3), dtype=np.float32)
    raw = np.zeros((20, 22, 3), dtype=np.float32)
    err_cm = _per_joint_error_cm(fk, raw)
    assert err_cm.shape == (15, 22)


def test_per_part_contact_floor_skips_inactive_parts():
    """If contact_state[..., col] is all zero, that part returns 0 frames
    and NaN stats (not crashing)."""
    T = 30
    fk = np.zeros((T, 22, 3), dtype=np.float32)
    raw = np.zeros((T, 22, 3), dtype=np.float32)
    contact = np.zeros((T, 5), dtype=np.float32)  # nobody touching
    floor = _per_part_contact_floor(fk, raw, contact)
    for part in PART_TO_JOINT_IDX:
        assert floor[part]["n_frames"] == 0
        assert np.isnan(floor[part]["mean_cm"])


def test_per_part_contact_floor_computes_active_part():
    """On frames where contact > 0.5 for a part, compute mean/p95/max
    on that part's joint Euclidean error."""
    T = 30
    fk = np.zeros((T, 22, 3), dtype=np.float32)
    raw = np.zeros((T, 22, 3), dtype=np.float32)
    # left_hand = SMPL idx 20 (wrist). Make raw differ from fk by 5 cm on
    # the wrist for the active frames.
    raw[:, 20, 0] = 0.05
    contact = np.zeros((T, 5), dtype=np.float32)
    contact[5:25, 0] = 1.0  # left_hand active 20 frames
    floor = _per_part_contact_floor(fk, raw, contact)
    assert floor["left_hand"]["n_frames"] == 20
    assert pytest.approx(floor["left_hand"]["mean_cm"], abs=1e-4) == 5.0
    # Right hand has no active frames.
    assert floor["right_hand"]["n_frames"] == 0


def test_percentile_handles_empty():
    assert np.isnan(_percentile(np.array([]), 50))
    assert _percentile(np.array([10.0]), 50) == pytest.approx(10.0)


def test_valid_length_from_seq_len_fallback():
    batch = {
        "seq_len": np.array([37]),
        "motion": np.zeros((1, 196, 135), dtype=np.float32),
    }
    assert _valid_length_from_batch(batch) == 37


def test_valid_length_prefers_legacy_seq_mask():
    batch = {
        "seq_mask": np.array([[1] * 12 + [0] * 8], dtype=bool),
        "seq_len": np.array([99]),
        "motion": np.zeros((1, 20, 135), dtype=np.float32),
    }
    assert _valid_length_from_batch(batch) == 12


def test_valid_length_clamps_to_motion_length():
    batch = {
        "seq_len": np.array([99]),
        "motion": np.zeros((1, 20, 135), dtype=np.float32),
    }
    assert _valid_length_from_batch(batch) == 20


def test_aggregate_empty_returns_placeholder():
    agg = _aggregate([])
    assert agg["n_clips"] == 0
    for name in SMPL_JOINT_NAMES:
        assert agg["per_joint"][name]["mean_cm"] is None
    for p in PART_TO_JOINT_IDX:
        assert agg["per_part_contact_floor"][p]["mean_cm"] is None


def test_aggregate_accumulates_per_joint_stats():
    """Two clips: per-joint mean should average across clips."""
    row_a = PerClipRow(
        subset="chairs", seq_id="a", n_frames=50,
        per_joint_mean_cm=[1.0] * 22,
        per_joint_p95_cm=[3.0] * 22,
        per_part_floor_cm={
            p: {"n_frames": 10, "mean_cm": 2.0, "p95_cm": 4.0, "max_cm": 6.0}
            for p in PART_TO_JOINT_IDX
        },
    )
    row_b = PerClipRow(
        subset="chairs", seq_id="b", n_frames=50,
        per_joint_mean_cm=[3.0] * 22,
        per_joint_p95_cm=[5.0] * 22,
        per_part_floor_cm={
            p: {"n_frames": 10, "mean_cm": 6.0, "p95_cm": 8.0, "max_cm": 10.0}
            for p in PART_TO_JOINT_IDX
        },
    )
    agg = _aggregate([row_a, row_b])
    assert agg["n_clips"] == 2
    # Mean across the two clips on left_wrist should be 2.0.
    assert agg["per_joint"]["left_wrist"]["mean_cm"] == pytest.approx(2.0)
    # Per-part mean weighted by n_frames: (10*2 + 10*6) / 20 = 4.0.
    assert agg["per_part_contact_floor"]["left_hand"]["mean_cm"] == pytest.approx(4.0)


def test_interpretation_low_when_hand_floor_small():
    agg = {
        "per_joint": {
            "left_wrist": {"mean_cm": 0.5, "p95_cm": 2.0, "max_cm": 3.0},
            "right_wrist": {"mean_cm": 0.7, "p95_cm": 2.5, "max_cm": 3.5},
        }
    }
    interp = _interpretation(agg)
    assert interp["verdict"] == "representation_floor_low"
    assert "not the contact bottleneck" in interp["reason"].lower()


def test_interpretation_critical_when_hand_floor_large():
    agg = {
        "per_joint": {
            "left_wrist": {"mean_cm": 8.0, "p95_cm": 14.0, "max_cm": 20.0},
            "right_wrist": {"mean_cm": 9.0, "p95_cm": 15.0, "max_cm": 22.0},
        }
    }
    interp = _interpretation(agg)
    assert interp["verdict"] == "representation_floor_critical"
    assert "critical path" in interp["reason"]


def test_interpretation_borderline_in_between():
    agg = {
        "per_joint": {
            "left_wrist": {"mean_cm": 3.0, "p95_cm": 7.0, "max_cm": 10.0},
            "right_wrist": {"mean_cm": 3.5, "p95_cm": 7.5, "max_cm": 11.0},
        }
    }
    interp = _interpretation(agg)
    assert interp["verdict"] == "representation_floor_borderline"


def test_interpretation_unknown_when_missing_stats():
    interp = _interpretation({"per_joint": {}})
    assert interp["verdict"] == "unknown"


def test_argparse_smoke():
    """The script's argparse should accept the required flags. Build the
    parser by importing the module and constructing it from the source
    (no subprocess needed for this smoke)."""
    import inspect
    from round29_motion_repr_roundtrip_diag import main
    src = inspect.getsource(main)
    # Check required CLI flags are mentioned in main's argparse setup.
    for flag in ("--config", "--selection-json", "--bucket", "--output-dir"):
        assert flag in src, f"main() must accept {flag}"


def test_smpl_joint_names_length():
    assert len(SMPL_JOINT_NAMES) == 22
    assert SMPL_JOINT_NAMES[0] == "pelvis"
    assert SMPL_JOINT_NAMES[20] == "left_wrist"
    assert SMPL_JOINT_NAMES[21] == "right_wrist"


def test_part_to_joint_idx_mapping():
    """The 5-part mapping must use ankle indices for foot (consistent with
    R29 dataset conventions: foot stance == ankle stance)."""
    assert PART_TO_JOINT_IDX["left_hand"] == 20    # left_wrist
    assert PART_TO_JOINT_IDX["right_hand"] == 21
    assert PART_TO_JOINT_IDX["left_foot"] == 7     # left_ankle
    assert PART_TO_JOINT_IDX["right_foot"] == 8
    assert PART_TO_JOINT_IDX["pelvis"] == 0
