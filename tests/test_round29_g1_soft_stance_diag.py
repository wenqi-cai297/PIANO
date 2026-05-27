"""Unit tests for the G1 soft-stance diagnostic helpers.

Per analyses/2026-05-28_round29_next_step_ablation_execution_prompt_for_claude_code.md §8.3.

Isolates the pure metric helpers (`_soft_metrics_for_segment` and
`_hard_alt_from_joints`) and checks they correctly flag:

  - Alternating stance sequence: low constant_mid_rate, high soft_alt_std,
    nonzero transition density, no flags.
  - Constant-mid soft stance: high constant_mid_rate, low soft_alt_std,
    flagged degenerate.
  - Both-swing sequence: high soft_both_swing, low soft_both_stance.

These checks are valuable independent of the full diagnostic because the
full diag pipeline requires a trained model + dataset; the metric helpers
do not.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "stage_b_generator"
sys.path.insert(0, str(SCRIPTS))

# These helpers are deliberately importable from the diag script.
from round29_g1_soft_stance_diag import (  # noqa: E402
    _hard_alt_from_joints, _soft_metrics_for_segment,
    LOW_ALT_AMPLITUDE_STD, LOW_TRANSITION_RATIO,
    CONSTANT_MID_LO, CONSTANT_MID_HI,
)
# SMPL-22 indices used by the soft-stance helper.
from piano.training.temporal_interaction_losses import (  # noqa: E402
    LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX,
)


def _alternating_soft_seq(T: int = 80, period: int = 8) -> np.ndarray:
    """Build a (T, 2) soft-stance trajectory with clean alternation:
    one foot ~1.0, the other ~0.0, switching every `period` frames."""
    out = np.zeros((T, 2), dtype=np.float32)
    for t in range(T):
        if (t // period) % 2 == 0:
            out[t, 0] = 0.95
            out[t, 1] = 0.05
        else:
            out[t, 0] = 0.05
            out[t, 1] = 0.95
    return out


def _constant_mid_seq(T: int = 80) -> np.ndarray:
    """Both feet always near 0.5 — the canonical degeneracy."""
    out = np.full((T, 2), 0.5, dtype=np.float32)
    # Add tiny noise so std isn't literally zero.
    out += 0.01 * np.sin(np.arange(T))[:, None]
    return out


def _both_swing_seq(T: int = 80) -> np.ndarray:
    """Both feet always near 0.0 (both airborne)."""
    return np.full((T, 2), 0.02, dtype=np.float32)


def _gt_alternating_hard_alt(T: int = 80, period: int = 8) -> np.ndarray:
    """GT hard alt = L_stance - R_stance ∈ {-1, 0, +1}."""
    out = np.zeros(T, dtype=np.float32)
    for t in range(T):
        out[t] = +1.0 if (t // period) % 2 == 0 else -1.0
    return out


def test_alternating_soft_is_healthy():
    soft = _alternating_soft_seq(T=80, period=8)
    gt = _gt_alternating_hard_alt(T=80, period=8)
    m = _soft_metrics_for_segment(soft, gt)
    assert m["n_frames"] == 80
    assert m["constant_mid_rate"] < 0.05, (
        f"alternating should have low constant_mid_rate, got "
        f"{m['constant_mid_rate']}"
    )
    assert m["soft_alt_std"] > 0.5, (
        f"alternating should have high soft_alt_std, got {m['soft_alt_std']}"
    )
    assert m["soft_transition_density"] > 0.0
    # When predicted soft alternation matches GT density, low_transition
    # flag must be False.
    assert m["low_transition"] is False
    assert m["low_alt_amplitude"] is False


def test_constant_mid_soft_is_flagged_degenerate():
    soft = _constant_mid_seq(T=80)
    gt = _gt_alternating_hard_alt(T=80, period=8)
    m = _soft_metrics_for_segment(soft, gt)
    # All frames have both pL and pR in [0.4, 0.6] → constant_mid_rate ≈ 1.0.
    assert m["constant_mid_rate"] > 0.9, (
        f"constant_mid_rate should be ≈1, got {m['constant_mid_rate']}"
    )
    # std(pL - pR) ≈ 0 → low_alt_amplitude flagged.
    assert m["soft_alt_std"] < LOW_ALT_AMPLITUDE_STD
    assert m["low_alt_amplitude"] is True
    # Soft transition density is near zero while GT has real density →
    # low_transition flagged.
    assert m["low_transition"] is True


def test_both_swing_soft_flags_aggregate():
    soft = _both_swing_seq(T=80)
    gt = _gt_alternating_hard_alt(T=80, period=8)
    m = _soft_metrics_for_segment(soft, gt)
    # both_swing aggregate is (1-pL)(1-pR) ≈ (0.98)² ≈ 0.96.
    assert m["soft_both_swing"] > 0.9
    # both_stance is pL*pR ≈ 0.02² ≈ 0.0004.
    assert m["soft_both_stance"] < 0.01


def test_hard_alt_from_joints_alternates():
    """Synthetic joints where L_ankle is grounded for first 8 frames, R for
    next 8, etc. _hard_alt_from_joints should return values in {-1, 0, +1}
    with the right pattern."""
    T = 32
    joints = np.zeros((T, 22, 3), dtype=np.float32)
    # Place both ankles at non-overlapping XZ positions and slow them down
    # (foot horizontal speed near zero in the synthetic).
    joints[:, LEFT_ANKLE_IDX, 0] = +0.1
    joints[:, RIGHT_ANKLE_IDX, 0] = -0.1
    # L_ankle low for first half, high for second; R_ankle opposite.
    for t in range(T):
        if (t // 8) % 2 == 0:
            joints[t, LEFT_ANKLE_IDX, 1] = 0.02       # low
            joints[t, RIGHT_ANKLE_IDX, 1] = 0.20       # high
        else:
            joints[t, LEFT_ANKLE_IDX, 1] = 0.20
            joints[t, RIGHT_ANKLE_IDX, 1] = 0.02
    alt = _hard_alt_from_joints(joints, stance_height=0.12, stance_speed=0.1)
    # First 8 frames: L low + slow, R high → L_stance=1, R_stance=0 → +1.
    # Skipping t=0 because the speed convention is "delta from t-1" → first
    # frame has horiz_vel=0 by convention, both slow OK.
    assert alt[5] == +1.0
    assert alt[10] == -1.0


def test_short_segment_returns_safe_placeholders():
    """A 1-frame segment cannot compute transition density; helper must
    return placeholders without crashing."""
    soft = np.array([[0.5, 0.5]], dtype=np.float32)
    gt = np.array([0.0], dtype=np.float32)
    m = _soft_metrics_for_segment(soft, gt)
    assert m["n_frames"] == 1
    assert m["soft_transition_density"] is None
    assert m["constant_mid_rate"] is None


def test_low_transition_only_fires_when_ratio_below_threshold():
    """If soft_trans / gt_trans is just above the threshold, low_transition
    is False. Just below → True."""
    # Build a soft trajectory that has high gt_trans (alternating GT) but
    # very small soft_trans (near constant).
    soft = np.full((80, 2), 0.50, dtype=np.float32)
    # Add slight variation so soft_trans is non-zero but very small.
    soft[1:, 0] += 0.01 * np.sign(np.diff(np.arange(80)))
    gt = _gt_alternating_hard_alt(T=80, period=8)
    m = _soft_metrics_for_segment(soft, gt)
    assert m["gt_transition_density"] > 0.0
    if m["gt_transition_density"] > 0:
        ratio = m["soft_transition_density"] / m["gt_transition_density"]
        if ratio < LOW_TRANSITION_RATIO:
            assert m["low_transition"] is True
        else:
            assert m["low_transition"] is False
