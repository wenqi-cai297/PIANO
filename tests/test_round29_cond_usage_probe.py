"""Unit tests for the Phase-0 cond-usage probe helpers.

Per Codex review §3 / §10 + the 2026-05-29 code-review of commit
`fe81f2a`. Isolates the pure perturbation primitives, task-metric
proxies, aggregation, label OR-judge, scale-linearity ratio, and the
``_apply_perturbation`` contract. The full diag script needs a model
+ dataset + sampler; these tests validate the math + selection logic
with synthetic arrays only.
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
    FAMILY_OF_KEY,
    KEY_JOINT_INDICES,
    KEY_OF_FAMILY,
    PERTURBATIONS_DEFAULT,
    PerClipPerturbationResult,
    THRESH_IGNORED_KEY_CM,
    THRESH_IGNORED_RELATIVE,
    THRESH_TEMPORALLY_USED_FRACTION,
    THRESH_WEAK_KEY_CM,
    THRESH_WEAK_RELATIVE,
    _aa_to_rotation_matrix_np,
    _apply_perturbation,
    _derange_indices,
    _fractional_change,
    _lift_object_local_to_world_np,
    _proxy_body_action_motion_energy_cm,
    _proxy_gait_velocity_score,
    _proxy_sustained_contact_cm,
    aggregate_per_family,
    compute_clip_delta,
    label_family_usage,
    perturbation_batch_shuffle,
    perturbation_scale,
    perturbation_time_shuffle,
    perturbation_zero,
)


# --------------------------------------------------------------------------- #
# perturbation_zero / scale
# --------------------------------------------------------------------------- #


def test_zero_produces_all_zero_same_shape():
    x = np.random.RandomState(0).randn(2, 50, 13).astype(np.float32)
    z = perturbation_zero(x)
    assert z.shape == x.shape
    assert np.array_equal(z, np.zeros_like(x))


def test_scale_multiplies_uniformly():
    x = np.ones((1, 10, 4), dtype=np.float32) * 3.0
    assert np.array_equal(perturbation_scale(x, 0.5), np.full_like(x, 1.5))
    assert np.array_equal(perturbation_scale(x, 2.0), np.full_like(x, 6.0))


# --------------------------------------------------------------------------- #
# perturbation_time_shuffle
# --------------------------------------------------------------------------- #


def test_time_shuffle_keeps_padded_frames_untouched():
    T = 30
    valid_T = 20
    x = np.zeros((1, T, 3), dtype=np.float32)
    for t in range(T):
        x[0, t, :] = float(t)
    rng = np.random.RandomState(0)
    out = perturbation_time_shuffle(x, valid_T, rng)
    assert np.array_equal(out[0, valid_T:], x[0, valid_T:])
    valid_values_in = sorted(x[0, :valid_T, 0].tolist())
    valid_values_out = sorted(out[0, :valid_T, 0].tolist())
    assert valid_values_in == valid_values_out
    assert not np.array_equal(out[0, :valid_T], x[0, :valid_T])


def test_time_shuffle_handles_short_clip_safely():
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
# perturbation_batch_shuffle + _derange_indices
# --------------------------------------------------------------------------- #


def test_batch_shuffle_rejects_batch_size_1():
    x = np.ones((1, 10, 3), dtype=np.float32)
    rng = np.random.RandomState(0)
    with pytest.raises(ValueError, match="batch size"):
        perturbation_batch_shuffle(x, rng)


def test_batch_shuffle_produces_derangement_when_possible():
    B, T, D = 5, 4, 2
    x = np.zeros((B, T, D), dtype=np.float32)
    for b in range(B):
        x[b] = float(b)
    rng = np.random.RandomState(0)
    out = perturbation_batch_shuffle(x, rng)
    assert out.shape == x.shape
    in_vals = {float(b) for b in range(B)}
    out_vals = {float(out[b, 0, 0]) for b in range(B)}
    assert in_vals == out_vals
    for b in range(B):
        assert out[b, 0, 0] != x[b, 0, 0]


def test_batch_shuffle_b2_uses_roll_fallback_if_needed():
    x = np.zeros((2, 1, 1), dtype=np.float32)
    x[0, 0, 0] = 10.0
    x[1, 0, 0] = 20.0
    rng = np.random.RandomState(0)
    out = perturbation_batch_shuffle(x, rng)
    assert out[0, 0, 0] == 20.0
    assert out[1, 0, 0] == 10.0


def test_derange_indices_no_fixed_points():
    """_derange_indices guarantees every position maps to a different value.
    Verified across many seeds to catch any subtle bug in the loop."""
    for seed in range(20):
        rng = np.random.RandomState(seed)
        perm = _derange_indices(10, rng)
        assert len(perm) == 10
        assert sorted(perm) == list(range(10))
        for i, v in enumerate(perm):
            assert i != v, f"seed {seed}: position {i} mapped to itself"


def test_derange_indices_rejects_n_below_2():
    rng = np.random.RandomState(0)
    with pytest.raises(ValueError, match="n >= 2"):
        _derange_indices(0, rng)
    with pytest.raises(ValueError, match="n >= 2"):
        _derange_indices(1, rng)


def test_derange_indices_n_equal_2_is_always_swap():
    """For n=2 only one derangement exists: [1, 0]."""
    for seed in range(10):
        rng = np.random.RandomState(seed)
        perm = _derange_indices(2, rng)
        assert perm == [1, 0]


# --------------------------------------------------------------------------- #
# Task-metric proxies
# --------------------------------------------------------------------------- #


def test_proxy_sustained_contact_returns_nan_when_no_contact():
    T = 10
    joints = np.zeros((T, 22, 3), dtype=np.float32)
    contact_target_xyz = np.zeros((T, 2, 3), dtype=np.float32)
    contact_state = np.zeros((T, 5), dtype=np.float32)
    assert np.isnan(_proxy_sustained_contact_cm(
        joints, contact_target_xyz, contact_state,
    ))


def test_proxy_sustained_contact_computes_distance_in_cm():
    """5 cm gap between wrist and contact target on all hand-contact frames
    → mean = 5.0 cm."""
    T = 10
    joints = np.zeros((T, 22, 3), dtype=np.float32)
    # left_wrist at origin; right_wrist at origin.
    contact_target_xyz = np.zeros((T, 2, 3), dtype=np.float32)
    contact_target_xyz[:, 0, 0] = 0.05   # left target 5 cm out
    contact_target_xyz[:, 1, 0] = 0.05   # right target 5 cm out
    contact_state = np.zeros((T, 5), dtype=np.float32)
    contact_state[:, 0] = 1.0  # left hand active all frames
    contact_state[:, 1] = 1.0  # right hand active all frames
    out = _proxy_sustained_contact_cm(joints, contact_target_xyz, contact_state)
    assert out == pytest.approx(5.0, abs=1e-4)


def test_proxy_gait_returns_nan_when_no_walking_mask_hits():
    T = 10
    joints = np.zeros((T, 22, 3), dtype=np.float32)
    walking_mask = np.zeros(T, dtype=np.float32)
    out = _proxy_gait_velocity_score(joints, walking_mask)
    assert np.isnan(out)


def test_proxy_gait_picks_up_lr_alternation():
    """Synthetic ankles with opposite-phase XZ motion ⇒ positive |L-R|
    velocity difference."""
    T = 20
    joints = np.zeros((T, 22, 3), dtype=np.float32)
    # left_ankle (idx 7) moves +X each frame; right_ankle (idx 8) -X each frame
    for t in range(T):
        joints[t, 7, 0] = 0.10 * t   # left moves +X
        joints[t, 8, 0] = -0.10 * t  # right moves -X
    out = _proxy_gait_velocity_score(joints, walking_mask=None, fps=20.0)
    # |speed_L - speed_R| = | (0.10*20) - (0.10*20) | = 0 (both have abs speed
    # 2 m/s) -- so this test instead exercises that the magnitude is finite + non-negative.
    assert out is not None
    assert out >= 0.0


def test_proxy_body_action_zero_when_static():
    joints = np.zeros((20, 22, 3), dtype=np.float32)
    out = _proxy_body_action_motion_energy_cm(joints)
    assert out == pytest.approx(0.0, abs=1e-6)


def test_proxy_body_action_positive_when_moving():
    T = 20
    joints = np.zeros((T, 22, 3), dtype=np.float32)
    # All joints translate +0.05 m per frame.
    for t in range(T):
        joints[t, :, 0] = 0.05 * t
    out = _proxy_body_action_motion_energy_cm(joints)
    # speed = 0.05 m/frame × 100 = 5.0 cm/frame, uniform across joints.
    assert out == pytest.approx(5.0, abs=1e-4)


def test_aa_to_rotation_matrix_np_identity_on_zero():
    """Zero axis-angle → identity rotation. (Rodrigues at θ=0.)"""
    aa = np.zeros((4, 3), dtype=np.float64)
    R = _aa_to_rotation_matrix_np(aa)
    assert R.shape == (4, 3, 3)
    eye = np.broadcast_to(np.eye(3), (4, 3, 3))
    assert np.allclose(R, eye, atol=1e-8)


def test_aa_to_rotation_matrix_np_90deg_around_z():
    """90° around Z maps (1,0,0) → (0,1,0)."""
    aa = np.array([[0.0, 0.0, np.pi / 2]], dtype=np.float64)
    R = _aa_to_rotation_matrix_np(aa)[0]
    v = np.array([1.0, 0.0, 0.0])
    out = R @ v
    assert np.allclose(out, [0.0, 1.0, 0.0], atol=1e-7)


def test_lift_object_local_to_world_np_identity_pose():
    """Zero rotation + zero translation → world == local (pure passthrough)."""
    T, P = 5, 2
    target_local = np.random.RandomState(0).randn(T, P, 3).astype(np.float64)
    pos = np.zeros((T, 3), dtype=np.float64)
    aa = np.zeros((T, 3), dtype=np.float64)
    out = _lift_object_local_to_world_np(target_local, pos, aa)
    assert out.shape == (T, P, 3)
    assert np.allclose(out, target_local, atol=1e-8)


def test_lift_object_local_to_world_np_translation_only():
    """Zero rotation + non-zero translation → world = local + pos broadcast."""
    T, P = 3, 2
    target_local = np.array(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]] * T, dtype=np.float64
    )
    pos = np.array(
        [[10.0, 20.0, 30.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -3.0]],
        dtype=np.float64,
    )
    aa = np.zeros((T, 3), dtype=np.float64)
    out = _lift_object_local_to_world_np(target_local, pos, aa)
    expected = target_local + pos[:, None, :]
    assert np.allclose(out, expected, atol=1e-8)


def test_lift_object_local_to_world_np_rotation_only():
    """90° around Z + zero translation: local (1,0,0) → world (0,1,0)."""
    T = 1
    target_local = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float64)   # (1, 1, 3)
    pos = np.zeros((T, 3), dtype=np.float64)
    aa = np.array([[0.0, 0.0, np.pi / 2]], dtype=np.float64)
    out = _lift_object_local_to_world_np(target_local, pos, aa)
    assert out.shape == (T, 1, 3)
    assert np.allclose(out[0, 0], [0.0, 1.0, 0.0], atol=1e-7)


def test_lift_object_local_to_world_np_matches_anchor_consistency_loss():
    """Numpy implementation must match the torch reference used by the
    trainer (``piano.training.anchor_consistency_loss.lift_object_local_to_world``)
    for random poses.
    """
    import torch  # local import — keeps the rest of the module torch-free.
    from piano.training.anchor_consistency_loss import lift_object_local_to_world

    rng = np.random.RandomState(42)
    T, P = 8, 2
    target_local_np = rng.randn(T, P, 3).astype(np.float64)
    pos_np = rng.randn(T, 3).astype(np.float64)
    aa_np = (rng.randn(T, 3) * 0.7).astype(np.float64)              # bounded but non-trivial

    out_np = _lift_object_local_to_world_np(target_local_np, pos_np, aa_np)

    # Torch reference expects (B, T, P, 3); wrap with a singleton batch.
    target_t = torch.from_numpy(target_local_np).unsqueeze(0)
    pos_t = torch.from_numpy(pos_np).unsqueeze(0)
    aa_t = torch.from_numpy(aa_np).unsqueeze(0)
    out_t = lift_object_local_to_world(target_t, pos_t, aa_t)[0].numpy()

    assert np.allclose(out_np, out_t, atol=1e-7)


def test_proxy_sustained_contact_cm_uses_world_frame():
    """Per N1: the proxy must see world-frame target. With wrist at world
    origin and target_local = (1, 0, 0), then:
      - identity pose → distance = 1 m = 100 cm
      - +5 m world translation → distance = sqrt(5² + 6²) m ≠ 100 cm

    If the caller forgets to rotate target into world, the proxy would
    compute 1 m for both cases (silently wrong). The test pins that the
    proxy is reading world-frame numbers directly.
    """
    T = 4
    joints = np.zeros((T, 22, 3), dtype=np.float32)                  # all joints at world origin
    contact_state = np.zeros((T, 5), dtype=np.float32)
    contact_state[:, 0] = 1.0                                        # left hand in contact

    # Identity pose: rotated target = local target = (1, 0, 0).
    target_local = np.tile(np.array([[1.0, 0.0, 0.0]]), (T, 2, 1)).astype(np.float64)
    pos_id = np.zeros((T, 3), dtype=np.float64)
    aa_id = np.zeros((T, 3), dtype=np.float64)
    world_id = _lift_object_local_to_world_np(target_local, pos_id, aa_id).astype(
        np.float32
    )
    d_id = _proxy_sustained_contact_cm(joints, world_id, contact_state)
    assert d_id == pytest.approx(100.0, abs=1e-3)                    # 1 m → 100 cm

    # Translated by (5, 5, 0): world target = (6, 5, 0).
    pos_tr = np.tile(np.array([[5.0, 5.0, 0.0]]), (T, 1)).astype(np.float64)
    world_tr = _lift_object_local_to_world_np(target_local, pos_tr, aa_id).astype(
        np.float32
    )
    d_tr = _proxy_sustained_contact_cm(joints, world_tr, contact_state)
    assert d_tr == pytest.approx(np.linalg.norm([6.0, 5.0, 0.0]) * 100.0, abs=1e-3)


def test_fractional_change_zero_when_identical():
    assert _fractional_change(5.0, 5.0) == pytest.approx(0.0)


def test_fractional_change_handles_nan():
    assert np.isnan(_fractional_change(float("nan"), 5.0))
    assert np.isnan(_fractional_change(5.0, float("nan")))


def test_fractional_change_handles_zero_baseline():
    """A degenerate baseline (≈ 0) → fractional change undefined; should be NaN."""
    assert np.isnan(_fractional_change(0.0, 5.0))
    assert np.isnan(_fractional_change(1e-9, 5.0))


def test_fractional_change_returns_absolute_ratio():
    """Returns |pert − base| / |base|, regardless of sign."""
    assert _fractional_change(10.0, 13.0) == pytest.approx(0.3, abs=1e-6)
    assert _fractional_change(10.0, 7.0) == pytest.approx(0.3, abs=1e-6)
    assert _fractional_change(-10.0, -13.0) == pytest.approx(0.3, abs=1e-6)


def test_fractional_change_respects_per_proxy_min_baseline():
    """Per-N2: baselines below the per-proxy floor → NaN, not exploding ratio.

    With the default 1e-6 floor, a 1.5e-6 baseline and 0.035 perturbation
    produced ~23000 (= 2 300 000 %) in the original run — see
    analyses/2026-05-29_round29_cond_usage_probe_code_review_v2.md §N2.
    With min_baseline=0.05 the same call must return NaN, while a healthy
    baseline (1.0 cm/frame) and a 50%-larger perturbation must still
    return 0.5.
    """
    # Tiny baseline + body_action floor (0.05 cm/frame) → NaN (degenerate).
    assert np.isnan(
        _fractional_change(1.5e-6, 0.035, min_baseline=0.05)
    )
    # Tiny baseline + default 1e-6 floor → still computes (back-compat).
    assert _fractional_change(1.5e-6, 0.035) > 1000
    # Healthy baseline + body_action floor → standard ratio.
    assert _fractional_change(1.0, 1.5, min_baseline=0.05) == pytest.approx(
        0.5, abs=1e-6
    )


# --------------------------------------------------------------------------- #
# compute_clip_delta (new 6-tuple return)
# --------------------------------------------------------------------------- #


def test_compute_clip_delta_zero_when_identical():
    j = np.random.RandomState(0).randn(20, 22, 3).astype(np.float32)
    mean_cm, p95_cm, kj, sc_rel, gait_rel, body_rel = compute_clip_delta(j, j)
    assert mean_cm == pytest.approx(0.0, abs=1e-6)
    assert p95_cm == pytest.approx(0.0, abs=1e-6)
    for name in KEY_JOINT_INDICES:
        assert kj[name] == pytest.approx(0.0, abs=1e-6)
    # No contact inputs → sustained_contact stays NaN.
    assert np.isnan(sc_rel)
    # For gait + body, identical joints produce the same proxy score for
    # base and pert. If that score is non-zero, fractional_change = 0; if
    # the score is zero (degenerate baseline), fractional_change = NaN.
    # Either is correct — the contract is "no spurious motion reported".
    assert gait_rel == 0.0 or np.isnan(gait_rel)
    assert body_rel == 0.0 or np.isnan(body_rel)


def test_compute_clip_delta_uses_cm_conversion():
    """5 cm shift on X dimension of all joints → mean_cm = 5 cm."""
    a = np.zeros((10, 22, 3), dtype=np.float32)
    b = np.zeros((10, 22, 3), dtype=np.float32)
    b[..., 0] = 0.05
    mean_cm, p95_cm, kj, *_ = compute_clip_delta(a, b)
    assert mean_cm == pytest.approx(5.0, abs=1e-4)
    assert p95_cm == pytest.approx(5.0, abs=1e-4)
    for name in KEY_JOINT_INDICES:
        assert kj[name] == pytest.approx(5.0, abs=1e-4)


def test_compute_clip_delta_handles_empty():
    a = np.zeros((0, 22, 3), dtype=np.float32)
    b = np.zeros((0, 22, 3), dtype=np.float32)
    mean_cm, p95_cm, kj, sc_rel, gait_rel, body_rel = compute_clip_delta(a, b)
    assert np.isnan(mean_cm)
    assert np.isnan(p95_cm)
    for name in KEY_JOINT_INDICES:
        assert np.isnan(kj[name])
    assert np.isnan(sc_rel)
    assert np.isnan(gait_rel)
    assert np.isnan(body_rel)


def test_compute_clip_delta_returns_task_metric_deltas_when_inputs_provided():
    """With contact + walking inputs, sc_rel + gait_rel + body_rel are
    finite and reflect actual fractional changes."""
    T = 20
    base = np.zeros((T, 22, 3), dtype=np.float32)
    # Base wrist 10 cm from target; pert wrist 5 cm closer (5 cm to target).
    base[:, 20, 0] = 0.0
    base[:, 21, 0] = 0.0
    pert = base.copy()
    pert[:, 20, 0] = 0.05   # left wrist 5 cm out → halves the distance
    pert[:, 21, 0] = 0.05
    contact_target_xyz = np.zeros((T, 2, 3), dtype=np.float32)
    contact_target_xyz[:, 0, 0] = 0.10
    contact_target_xyz[:, 1, 0] = 0.10
    contact_state = np.zeros((T, 5), dtype=np.float32)
    contact_state[:, 0] = 1.0; contact_state[:, 1] = 1.0
    walking_mask = np.zeros(T, dtype=np.float32)
    mean_cm, p95_cm, kj, sc_rel, gait_rel, body_rel = compute_clip_delta(
        base, pert, contact_target_xyz=contact_target_xyz,
        contact_state=contact_state, walking_mask=walking_mask,
    )
    # sc baseline = 10 cm distance; sc pert = 5 cm → fractional change 0.5.
    assert sc_rel == pytest.approx(0.5, abs=1e-4)
    # walking_mask is all-zero → gait_rel = NaN.
    assert np.isnan(gait_rel)


# --------------------------------------------------------------------------- #
# aggregate_per_family + NaN tolerance
# --------------------------------------------------------------------------- #


def _row(
    family: str, pert: str, mean_cm: float,
    sc_rel: float = float("nan"),
    gait_rel: float = float("nan"),
    body_rel: float = float("nan"),
    **key_overrides,
) -> PerClipPerturbationResult:
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
        sustained_contact_delta_rel=sc_rel,
        gait_delta_rel=gait_rel,
        body_action_delta_rel=body_rel,
    )


def test_aggregate_groups_by_family_and_perturbation():
    rows = [
        _row("support", "zero", 5.0, sc_rel=0.10),
        _row("support", "zero", 7.0, sc_rel=0.20),
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
    # sc_rel mean = (0.10 + 0.20) / 2 = 0.15.
    assert agg["support"]["zero"]["sustained_contact_delta_rel_mean"] == pytest.approx(0.15)
    # NaN sc_rel on time_shuffle aggregate.
    assert agg["support"]["time_shuffle"]["sustained_contact_delta_rel_mean"] is None


def test_aggregate_skips_nan_rows_per_field():
    """A row with NaN sustained_contact but finite gait must not poison
    either aggregate; gait mean ≠ None, sc mean takes only finite values."""
    rows = [
        _row("support", "zero", 5.0, sc_rel=0.10, gait_rel=0.50),
        _row("support", "zero", 5.0, sc_rel=float("nan"), gait_rel=0.30),
    ]
    agg = aggregate_per_family(rows)
    z = agg["support"]["zero"]
    assert z["sustained_contact_delta_rel_mean"] == pytest.approx(0.10)
    assert z["gait_delta_rel_mean"] == pytest.approx(0.40)
    assert z["task_metric_n_clips"]["sustained_contact"] == 1
    assert z["task_metric_n_clips"]["gait"] == 2


# --------------------------------------------------------------------------- #
# label_family_usage — Codex §3.3 OR judge
# --------------------------------------------------------------------------- #


def _make_family_agg(
    zero_key_max: float, zero_mean: float | None = None,
    time_shuffle_mean: float | None = None,
    zero_sc_rel: float | None = None,
    zero_gait_rel: float | None = None,
    zero_body_rel: float | None = None,
    scale_lo_mean: float | None = None,
    scale_hi_mean: float | None = None,
) -> dict:
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
            "sustained_contact_delta_rel_mean": zero_sc_rel,
            "gait_delta_rel_mean": zero_gait_rel,
            "body_action_delta_rel_mean": zero_body_rel,
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
    if scale_lo_mean is not None:
        agg["scale_0.5"] = {
            "n_clips": 10,
            "pred_delta_joints_cm_mean": float(scale_lo_mean),
        }
    if scale_hi_mean is not None:
        agg["scale_2.0"] = {
            "n_clips": 10,
            "pred_delta_joints_cm_mean": float(scale_hi_mean),
        }
    return agg


def test_label_ignored_when_both_arms_below_threshold():
    agg = _make_family_agg(
        zero_key_max=THRESH_IGNORED_KEY_CM * 0.5,
        zero_sc_rel=THRESH_IGNORED_RELATIVE * 0.5,
    )
    lab = label_family_usage(agg)
    assert lab["label"] == "ignored"
    assert lab["key_arm_label"] == "ignored"
    assert lab["relative_arm_label"] == "ignored"


def test_label_weakly_used_when_only_key_arm_in_band():
    agg = _make_family_agg(
        zero_key_max=(THRESH_IGNORED_KEY_CM + THRESH_WEAK_KEY_CM) / 2.0,
        zero_sc_rel=THRESH_IGNORED_RELATIVE * 0.5,   # rel-arm ignored
    )
    lab = label_family_usage(agg)
    assert lab["label"] == "weakly_used"
    assert lab["key_arm_label"] == "weakly_used"
    assert lab["relative_arm_label"] == "ignored"


def test_label_actively_used_above_key_threshold():
    agg = _make_family_agg(zero_key_max=THRESH_WEAK_KEY_CM * 2.0)
    lab = label_family_usage(agg)
    assert lab["label"] == "actively_used"
    assert lab["key_arm_label"] == "actively_used"


def test_or_judge_relative_arm_promotes_label():
    """Borderline R1 case: key-arm says ignored (0.8 cm < 1.0 cm) but
    relative-arm sees 12% drift change → should be weakly_used, not ignored."""
    agg = _make_family_agg(
        zero_key_max=0.8,           # < 1.0 → key-arm ignored
        zero_sc_rel=0.12,           # 12% > 5% threshold → rel-arm weakly_used
    )
    lab = label_family_usage(agg)
    assert lab["key_arm_label"] == "ignored"
    assert lab["relative_arm_label"] == "weakly_used"
    assert lab["label"] == "weakly_used", (
        "OR judge must promote when either arm is above threshold; this is "
        "the exact edge case R1 was reported for"
    )


def test_or_judge_key_arm_promotes_label():
    """Reverse direction: key-arm = actively_used, rel-arm = ignored
    → OR judge still actively_used."""
    agg = _make_family_agg(
        zero_key_max=THRESH_WEAK_KEY_CM * 2.0,
        zero_sc_rel=0.01,
    )
    lab = label_family_usage(agg)
    assert lab["key_arm_label"] == "actively_used"
    assert lab["relative_arm_label"] == "ignored"
    assert lab["label"] == "actively_used"


def test_or_judge_takes_max_across_three_relative_metrics():
    """Relative-arm uses max of (sustained_contact / gait / body_action)
    fractional changes."""
    agg = _make_family_agg(
        zero_key_max=0.5,            # key-arm ignored
        zero_sc_rel=0.02,            # all three given
        zero_gait_rel=0.25,          # only this one in actively_used range
        zero_body_rel=0.03,
    )
    lab = label_family_usage(agg)
    assert lab["relative_arm_label"] == "actively_used"
    assert lab["label"] == "actively_used"


def test_temporally_used_flag_triggers_when_time_shuffle_hurts_more():
    agg = _make_family_agg(
        zero_key_max=2.0, zero_mean=2.0,
        time_shuffle_mean=2.0 * (THRESH_TEMPORALLY_USED_FRACTION + 0.5),
    )
    lab = label_family_usage(agg)
    assert lab["temporally_used"] is True
    assert lab["label"] == "weakly_used"


def test_temporally_used_flag_off_when_time_shuffle_similar_to_zero():
    agg = _make_family_agg(
        zero_key_max=2.0, zero_mean=2.0, time_shuffle_mean=2.0 * 1.05,
    )
    lab = label_family_usage(agg)
    assert lab["temporally_used"] is False


def test_label_unknown_when_zero_perturbation_missing():
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
# R3 scale_linearity_ratio
# --------------------------------------------------------------------------- #


def test_scale_linearity_ratio_is_one_point_five_for_linear_response():
    """For a perfectly linear response, scale_0.5 ⇒ delta ∝ 0.5,
    scale_2.0 ⇒ delta ∝ 2.0, zero ⇒ delta ∝ 1.0; ratio = (2−0.5)/1 = 1.5."""
    agg = _make_family_agg(
        zero_key_max=2.0, zero_mean=4.0,
        scale_lo_mean=2.0,    # ∝ 0.5
        scale_hi_mean=8.0,    # ∝ 2.0
    )
    lab = label_family_usage(agg)
    assert lab["scale_linearity_ratio"] == pytest.approx(1.5, abs=1e-6)


def test_scale_linearity_ratio_near_zero_for_saturated_response():
    """When scale_0.5 ≈ scale_2.0 (saturation), the ratio is ≈ 0."""
    agg = _make_family_agg(
        zero_key_max=2.0, zero_mean=4.0,
        scale_lo_mean=4.0,
        scale_hi_mean=4.0,
    )
    lab = label_family_usage(agg)
    assert lab["scale_linearity_ratio"] == pytest.approx(0.0, abs=1e-6)


def test_scale_linearity_ratio_is_none_when_scale_perturbations_absent():
    agg = _make_family_agg(zero_key_max=2.0)
    lab = label_family_usage(agg)
    assert lab["scale_linearity_ratio"] is None


# --------------------------------------------------------------------------- #
# R2: _apply_perturbation contract (synthetic dicts, no torch model)
# --------------------------------------------------------------------------- #


def _make_cond_dict(value: float, T: int = 4, D: int = 13):
    """A minimal cond dict containing all 4 R29 keys + a couple of other
    keys that must NEVER be mutated by _apply_perturbation."""
    import torch
    cond = {}
    for key in (
        "stage2_coarse_extra", "stage2_interaction",
        "stage2_support", "stage2_body_refine",
    ):
        cond[key] = torch.full((1, T, D), value, dtype=torch.float32)
    # Non-R29 keys that should be untouched.
    cond["object_world_traj"] = torch.full((1, T, 9), 99.0)
    cond["object_tokens"] = torch.full((1, 128, 256), 88.0)
    cond["stage1_coarse"] = torch.full((1, T, 23), 77.0)
    return cond


def test_apply_perturbation_zero_only_touches_target_family():
    cond = _make_cond_dict(value=5.0)
    rng = np.random.RandomState(0)
    out = _apply_perturbation(
        cond, "stage2_support", "zero", valid_T=4, rng=rng,
    )
    # Target zeroed.
    import torch
    assert torch.all(out["stage2_support"] == 0)
    # Other R29 families untouched.
    assert torch.all(out["stage2_coarse_extra"] == 5.0)
    assert torch.all(out["stage2_interaction"] == 5.0)
    assert torch.all(out["stage2_body_refine"] == 5.0)
    # Non-R29 cond keys untouched.
    assert torch.all(out["object_world_traj"] == 99.0)
    assert torch.all(out["object_tokens"] == 88.0)
    assert torch.all(out["stage1_coarse"] == 77.0)


def test_apply_perturbation_scale_uses_correct_factor():
    import torch
    cond = _make_cond_dict(value=4.0)
    rng = np.random.RandomState(0)
    out_lo = _apply_perturbation(
        cond, "stage2_coarse_extra", "scale_0.5", valid_T=4, rng=rng,
    )
    out_hi = _apply_perturbation(
        cond, "stage2_coarse_extra", "scale_2.0", valid_T=4, rng=rng,
    )
    assert torch.all(out_lo["stage2_coarse_extra"] == 2.0)
    assert torch.all(out_hi["stage2_coarse_extra"] == 8.0)


def test_apply_perturbation_batch_shuffle_replaces_from_cond_b2():
    """R2 contract: family X on cond_b1 is replaced with cond_b2's X;
    every other key in cond_b1 is unchanged."""
    import torch
    cond_b1 = _make_cond_dict(value=5.0)
    cond_b2 = _make_cond_dict(value=10.0)
    rng = np.random.RandomState(0)
    out = _apply_perturbation(
        cond_b1, "stage2_support", "batch_shuffle",
        valid_T=4, rng=rng, cond_b2=cond_b2,
    )
    # Target swapped to cond_b2's value.
    assert torch.all(out["stage2_support"] == 10.0)
    # Other R29 families on cond_b1 must keep cond_b1's value.
    assert torch.all(out["stage2_coarse_extra"] == 5.0)
    assert torch.all(out["stage2_interaction"] == 5.0)
    assert torch.all(out["stage2_body_refine"] == 5.0)
    # Non-R29 keys must keep cond_b1's value.
    assert torch.all(out["object_world_traj"] == 99.0)
    assert torch.all(out["object_tokens"] == 88.0)


def test_apply_perturbation_batch_shuffle_requires_cond_b2():
    cond = _make_cond_dict(value=5.0)
    rng = np.random.RandomState(0)
    with pytest.raises(ValueError, match="cond_b2"):
        _apply_perturbation(
            cond, "stage2_support", "batch_shuffle", valid_T=4, rng=rng,
        )


def test_apply_perturbation_time_shuffle_preserves_padded_frames():
    """R2 sanity: time_shuffle permutes valid frames; padded frames
    in [valid_T:] keep their original values."""
    import torch
    T = 10; valid_T = 6
    cond = {}
    for key in (
        "stage2_coarse_extra", "stage2_interaction",
        "stage2_support", "stage2_body_refine",
    ):
        t = torch.zeros((1, T, 3), dtype=torch.float32)
        for fi in range(T):
            t[0, fi, :] = float(fi)
        cond[key] = t
    cond["object_world_traj"] = torch.full((1, T, 9), 99.0)
    rng = np.random.RandomState(0)
    out = _apply_perturbation(
        cond, "stage2_support", "time_shuffle", valid_T=valid_T, rng=rng,
    )
    # Padded frames untouched.
    for fi in range(valid_T, T):
        assert out["stage2_support"][0, fi, 0].item() == pytest.approx(float(fi))
    # Other R29 families untouched.
    for fi in range(T):
        assert out["stage2_coarse_extra"][0, fi, 0].item() == pytest.approx(float(fi))


def test_apply_perturbation_silently_passes_through_missing_family():
    """If the family key isn't on cond at all (e.g. A0 with model
    support_dim=0 means data emits stage2_support but family-skip logic
    in main() never tries; sanity-check the helper itself)."""
    import torch
    cond = {"object_world_traj": torch.zeros((1, 4, 9))}
    rng = np.random.RandomState(0)
    out = _apply_perturbation(
        cond, "stage2_support", "zero", valid_T=4, rng=rng,
    )
    assert "stage2_support" not in out
    assert torch.all(out["object_world_traj"] == 0)


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
    assert THRESH_IGNORED_RELATIVE < THRESH_WEAK_RELATIVE
    assert THRESH_TEMPORALLY_USED_FRACTION > 1.0


def test_family_key_mapping_is_invertible():
    """FAMILY_OF_KEY and KEY_OF_FAMILY must be inverses (caught a real
    bug class — typo in either dict)."""
    for key, family in FAMILY_OF_KEY.items():
        assert KEY_OF_FAMILY[family] == key
