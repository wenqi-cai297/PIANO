"""Unit tests for the Phase-0 cond-usage probe helpers.

Per Codex review §3 / §10. Isolates the pure perturbation primitives,
aggregation, and label thresholds. The full diag script needs a model
+ dataset + sampler; these tests validate the math layer with synthetic
arrays only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "stage_b_generator"
sys.path.insert(0, str(SCRIPTS))

from round29_cond_usage_probe import (  # noqa: E402
    KEY_JOINT_INDICES,
    PERTURBATIONS_DEFAULT,
    PerClipPerturbationResult,
    THRESH_IGNORED_KEY_CM,
    THRESH_TEMPORALLY_USED_FRACTION,
    THRESH_WEAK_KEY_CM,
    aggregate_per_family,
    compute_clip_delta,
    label_family_usage,
    perturbation_batch_shuffle,
    perturbation_scale,
    perturbation_time_shuffle,
    perturbation_zero,
)


# --------------------------------------------------------------------------- #
# perturbation_zero
# --------------------------------------------------------------------------- #


def test_zero_produces_all_zero_same_shape():
    x = np.random.RandomState(0).randn(2, 50, 13).astype(np.float32)
    z = perturbation_zero(x)
    assert z.shape == x.shape
    assert np.array_equal(z, np.zeros_like(x))


# --------------------------------------------------------------------------- #
# perturbation_scale
# --------------------------------------------------------------------------- #


def test_scale_multiplies_uniformly():
    x = np.ones((1, 10, 4), dtype=np.float32) * 3.0
    assert np.array_equal(perturbation_scale(x, 0.5), np.full_like(x, 1.5))
    assert np.array_equal(perturbation_scale(x, 2.0), np.full_like(x, 6.0))


# --------------------------------------------------------------------------- #
# perturbation_time_shuffle
# --------------------------------------------------------------------------- #


def test_time_shuffle_keeps_padded_frames_untouched():
    """valid_T frames are permuted; frames >= valid_T are bit-identical."""
    T = 30
    valid_T = 20
    x = np.zeros((1, T, 3), dtype=np.float32)
    # Distinct per-frame signature so we can verify which frames moved.
    for t in range(T):
        x[0, t, :] = float(t)
    # Use a seed that will not produce identity permutation on 20 frames.
    rng = np.random.RandomState(0)
    out = perturbation_time_shuffle(x, valid_T, rng)
    # Padded frames untouched.
    assert np.array_equal(out[0, valid_T:], x[0, valid_T:])
    # Valid frames: same SET of values (a permutation), but order changed.
    valid_values_in = sorted(x[0, :valid_T, 0].tolist())
    valid_values_out = sorted(out[0, :valid_T, 0].tolist())
    assert valid_values_in == valid_values_out, (
        "time_shuffle should permute valid frames, not invent new values"
    )
    # With seed 0 we expect a non-identity permutation.
    assert not np.array_equal(out[0, :valid_T], x[0, :valid_T]), (
        "seed 0 gave identity permutation on 20 frames; this is statistically "
        "unlikely — check the seed or permutation logic"
    )


def test_time_shuffle_handles_short_clip_safely():
    """valid_T < 2 → no shuffle possible; should return as-is, not crash."""
    x = np.random.RandomState(0).randn(1, 10, 3).astype(np.float32)
    rng = np.random.RandomState(0)
    out = perturbation_time_shuffle(x, valid_T=1, rng=rng)
    assert np.array_equal(out, x)
    out2 = perturbation_time_shuffle(x, valid_T=0, rng=rng)
    assert np.array_equal(out2, x)


def test_time_shuffle_rejects_wrong_shape():
    x = np.zeros((10, 3), dtype=np.float32)
    rng = np.random.RandomState(0)
    with pytest.raises(ValueError, match=r"\(B, T, D\)"):
        perturbation_time_shuffle(x, 5, rng)


# --------------------------------------------------------------------------- #
# perturbation_batch_shuffle
# --------------------------------------------------------------------------- #


def test_batch_shuffle_rejects_batch_size_1():
    x = np.ones((1, 10, 3), dtype=np.float32)
    rng = np.random.RandomState(0)
    with pytest.raises(ValueError, match="batch size"):
        perturbation_batch_shuffle(x, rng)


def test_batch_shuffle_produces_derangement_when_possible():
    """No row should map to itself when B >= 2."""
    B, T, D = 5, 4, 2
    x = np.zeros((B, T, D), dtype=np.float32)
    for b in range(B):
        x[b] = float(b)
    rng = np.random.RandomState(0)
    out = perturbation_batch_shuffle(x, rng)
    assert out.shape == x.shape
    # Each row's value is one of the original rows (no new values invented).
    in_vals = {float(b) for b in range(B)}
    out_vals = {float(out[b, 0, 0]) for b in range(B)}
    assert in_vals == out_vals
    # Derangement: no row maps to itself.
    for b in range(B):
        assert out[b, 0, 0] != x[b, 0, 0], (
            f"row {b} got its own data — batch_shuffle should be a derangement"
        )


def test_batch_shuffle_b2_uses_roll_fallback_if_needed():
    """B=2 only has one valid derangement (swap). The function must produce it."""
    x = np.zeros((2, 1, 1), dtype=np.float32)
    x[0, 0, 0] = 10.0
    x[1, 0, 0] = 20.0
    rng = np.random.RandomState(0)
    out = perturbation_batch_shuffle(x, rng)
    assert out[0, 0, 0] == 20.0
    assert out[1, 0, 0] == 10.0


# --------------------------------------------------------------------------- #
# compute_clip_delta
# --------------------------------------------------------------------------- #


def test_compute_clip_delta_zero_when_identical():
    j = np.random.RandomState(0).randn(20, 22, 3).astype(np.float32)
    mean_cm, p95_cm, kj = compute_clip_delta(j, j)
    assert mean_cm == pytest.approx(0.0, abs=1e-6)
    assert p95_cm == pytest.approx(0.0, abs=1e-6)
    for name in KEY_JOINT_INDICES:
        assert kj[name] == pytest.approx(0.0, abs=1e-6)


def test_compute_clip_delta_uses_cm_conversion():
    """5 cm shift on X dimension of all joints → mean_cm = 5 cm."""
    a = np.zeros((10, 22, 3), dtype=np.float32)
    b = np.zeros((10, 22, 3), dtype=np.float32)
    b[..., 0] = 0.05   # 5 cm in metres
    mean_cm, p95_cm, kj = compute_clip_delta(a, b)
    assert mean_cm == pytest.approx(5.0, abs=1e-4)
    assert p95_cm == pytest.approx(5.0, abs=1e-4)
    for name in KEY_JOINT_INDICES:
        assert kj[name] == pytest.approx(5.0, abs=1e-4)


def test_compute_clip_delta_handles_empty():
    a = np.zeros((0, 22, 3), dtype=np.float32)
    b = np.zeros((0, 22, 3), dtype=np.float32)
    mean_cm, p95_cm, kj = compute_clip_delta(a, b)
    assert np.isnan(mean_cm)
    assert np.isnan(p95_cm)
    for name in KEY_JOINT_INDICES:
        assert np.isnan(kj[name])


# --------------------------------------------------------------------------- #
# aggregate_per_family
# --------------------------------------------------------------------------- #


def _row(family: str, pert: str, mean_cm: float, **key_overrides) -> PerClipPerturbationResult:
    kj = {n: mean_cm for n in KEY_JOINT_INDICES}
    for k, v in key_overrides.items():
        kj[k] = v
    return PerClipPerturbationResult(
        variant_id="test", bucket="val",
        family=family, perturbation=pert,
        subset="chairs", seq_id=f"seq_{mean_cm}",
        pred_delta_joints_cm_mean=mean_cm,
        pred_delta_joints_cm_p95=mean_cm * 1.5,
        key_joint_delta_cm=kj,
    )


def test_aggregate_groups_by_family_and_perturbation():
    rows = [
        _row("support", "zero", 5.0),
        _row("support", "zero", 7.0),
        _row("support", "time_shuffle", 3.0),
        _row("coarse_extra", "zero", 10.0),
    ]
    agg = aggregate_per_family(rows)
    assert "support" in agg
    assert "coarse_extra" in agg
    assert agg["support"]["zero"]["n_clips"] == 2
    assert agg["support"]["zero"]["pred_delta_joints_cm_mean"] == pytest.approx(6.0)
    assert agg["support"]["time_shuffle"]["pred_delta_joints_cm_mean"] == pytest.approx(3.0)
    assert agg["coarse_extra"]["zero"]["pred_delta_joints_cm_mean"] == pytest.approx(10.0)


# --------------------------------------------------------------------------- #
# label_family_usage — boundary cases of the Codex §3.3 thresholds
# --------------------------------------------------------------------------- #


def _make_family_agg(zero_key_max: float, zero_mean: float | None = None,
                     time_shuffle_mean: float | None = None) -> dict:
    """Build a synthetic per-family aggregate matching what
    ``aggregate_per_family`` returns."""
    if zero_mean is None:
        zero_mean = zero_key_max
    agg = {
        "zero": {
            "n_clips": 10,
            "pred_delta_joints_cm_mean": float(zero_mean),
            "pred_delta_joints_cm_p95": float(zero_mean) * 1.5,
            "key_joint_delta_cm_mean": {
                n: float(zero_key_max) for n in KEY_JOINT_INDICES
            },
        },
    }
    if time_shuffle_mean is not None:
        agg["time_shuffle"] = {
            "n_clips": 10,
            "pred_delta_joints_cm_mean": float(time_shuffle_mean),
            "pred_delta_joints_cm_p95": float(time_shuffle_mean) * 1.5,
            "key_joint_delta_cm_mean": {
                n: float(time_shuffle_mean) for n in KEY_JOINT_INDICES
            },
        }
    return agg


def test_label_ignored_when_zero_key_below_threshold():
    agg = _make_family_agg(zero_key_max=THRESH_IGNORED_KEY_CM * 0.5)
    lab = label_family_usage(agg)
    assert lab["label"] == "ignored"


def test_label_weakly_used_in_band():
    agg = _make_family_agg(
        zero_key_max=(THRESH_IGNORED_KEY_CM + THRESH_WEAK_KEY_CM) / 2.0,
    )
    lab = label_family_usage(agg)
    assert lab["label"] == "weakly_used"


def test_label_actively_used_above_weak_threshold():
    agg = _make_family_agg(zero_key_max=THRESH_WEAK_KEY_CM * 2.0)
    lab = label_family_usage(agg)
    assert lab["label"] == "actively_used"


def test_temporally_used_flag_triggers_when_time_shuffle_hurts_more():
    """time_shuffle / zero > THRESH_TEMPORALLY_USED_FRACTION → flag set."""
    agg = _make_family_agg(
        zero_key_max=2.0, zero_mean=2.0,
        time_shuffle_mean=2.0 * (THRESH_TEMPORALLY_USED_FRACTION + 0.5),
    )
    lab = label_family_usage(agg)
    assert lab["temporally_used"] is True
    # And the standard label is whatever bucket zero falls into.
    assert lab["label"] == "weakly_used"


def test_temporally_used_flag_off_when_time_shuffle_similar_to_zero():
    agg = _make_family_agg(
        zero_key_max=2.0, zero_mean=2.0, time_shuffle_mean=2.0 * 1.05,
    )
    lab = label_family_usage(agg)
    assert lab["temporally_used"] is False


def test_label_unknown_when_zero_perturbation_missing():
    """If zero perturbation wasn't run, can't label."""
    agg = {
        "scale_0.5": {
            "n_clips": 5,
            "pred_delta_joints_cm_mean": 1.0,
            "pred_delta_joints_cm_p95": 2.0,
            "key_joint_delta_cm_mean": {n: 1.0 for n in KEY_JOINT_INDICES},
        }
    }
    lab = label_family_usage(agg)
    assert lab["label"] == "unknown"


# --------------------------------------------------------------------------- #
# Sanity on module-level constants (used by the launcher + docs).
# --------------------------------------------------------------------------- #


def test_perturbations_default_includes_all_six():
    assert set(PERTURBATIONS_DEFAULT) == {
        "baseline", "zero", "time_shuffle", "batch_shuffle",
        "scale_0.5", "scale_2.0",
    }


def test_key_joint_indices_match_smpl_22():
    assert KEY_JOINT_INDICES["left_wrist"] == 20
    assert KEY_JOINT_INDICES["right_wrist"] == 21
    assert KEY_JOINT_INDICES["left_ankle"] == 7
    assert KEY_JOINT_INDICES["right_ankle"] == 8
    assert KEY_JOINT_INDICES["neck"] == 12
    assert KEY_JOINT_INDICES["pelvis"] == 0


def test_thresholds_are_ordered():
    assert THRESH_IGNORED_KEY_CM < THRESH_WEAK_KEY_CM
    assert THRESH_TEMPORALLY_USED_FRACTION > 1.0
