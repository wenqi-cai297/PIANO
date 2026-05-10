"""Unit tests for the Interaction Plan Compiler.

Covers:
- output schema (shapes, dtypes, padding semantics)
- hysteresis segmentation (asymmetric thresholds, gap merge, min duration)
- contact anchor onset / stable / release generation
- phase / support change anchors
- temporal NMS + budget
- world-frame target lifting matches the trainer's torch convention
- behavior on degenerate inputs (zero contact, very short clip)
- determinism (same input → same output)

The compiler is part of the method (see
analyses/piano_interaction_plan_pipeline_reframe_for_claude_code.md), so
a pinned set of expected shapes / counts here protects downstream Stage B
training from silent regressions in the compiler internals.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from piano.data.interaction_plan_compiler import (
    ANCHOR_TYPE_ONSET,
    ANCHOR_TYPE_PHASE_CHANGE,
    ANCHOR_TYPE_RELEASE,
    ANCHOR_TYPE_STABLE,
    ANCHOR_TYPE_SUPPORT_CHANGE,
    InteractionPlanCompilerConfig,
    collate_interaction_plans,
    compile_interaction_plan,
    hysteresis_segments,
    lift_target_local_to_world_np,
    smooth_contact,
)


# ---------------------------------------------------------------------------
# Helpers — synthetic clip builders
# ---------------------------------------------------------------------------


def _make_pulse(T: int, intervals: list[tuple[int, int]]) -> np.ndarray:
    """Step pulse: 1.0 inside intervals, 0.0 outside."""
    arr = np.zeros(T, dtype=np.float32)
    for s, e in intervals:
        arr[s : e + 1] = 1.0
    return arr


def _make_clip(
    T: int = 80,
    P: int = 5,
    contact_intervals_per_part: dict[int, list[tuple[int, int]]] | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a synthetic clip + per-part contact intervals.

    Returns (contact_prob, target_local, phase, support, obj_pos, obj_rot).
    """
    rng = np.random.default_rng(seed)
    contact = np.zeros((T, P), dtype=np.float32)
    intervals = contact_intervals_per_part or {}
    for p, ints in intervals.items():
        contact[:, p] = _make_pulse(T, ints)
    target_local = rng.normal(0, 0.01, size=(T, P, 3)).astype(np.float32)
    # Make contact targets sit at part-specific xyz so anchors get
    # distinguishable target_local values per part.
    for p, ints in intervals.items():
        for s, e in ints:
            target_local[s : e + 1, p, :] = np.array([float(p), 0.5, 0.0], dtype=np.float32)
    # Phase / support — simple two-region softmax with a transition.
    phase = np.zeros((T, 3), dtype=np.float32)
    phase[: T // 2, 0] = 1.0
    phase[T // 2 :, 1] = 1.0
    support = np.zeros((T, 3), dtype=np.float32)
    support[:, 0] = 1.0
    obj_pos = np.tile(np.array([1.0, 0.0, 2.0], dtype=np.float32), (T, 1))
    obj_rot = np.zeros((T, 3), dtype=np.float32)  # identity rotation
    return contact, target_local, phase, support, obj_pos, obj_rot


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


def test_output_schema_shapes_and_dtypes() -> None:
    cfg = InteractionPlanCompilerConfig(num_parts=5, k_max=12, s_max=12)
    contact, tgt, phase, sup, obj_p, obj_r = _make_clip(
        T=80, contact_intervals_per_part={0: [(10, 25)], 1: [(40, 55)]},
    )
    plan = compile_interaction_plan(
        contact_prob=contact,
        target_local=tgt,
        phase_softmax=phase,
        support_softmax=sup,
        object_pos_world=obj_p,
        object_rot_world_aa=obj_r,
        seq_len=80,
        cfg=cfg,
    )
    expected_shapes = {
        "anchor_time": (12,),
        "anchor_part": (12, 5),
        "anchor_target_local": (12, 5, 3),
        "anchor_target_world": (12, 5, 3),
        "anchor_type": (12,),
        "anchor_phase": (12,),
        "anchor_support": (12,),
        "anchor_conf": (12,),
        "anchor_mask": (12,),
        "segment_start": (12,),
        "segment_end": (12,),
        "segment_part": (12, 5),
        "segment_target_summary_local": (12, 5, 3),
        "segment_phase": (12,),
        "segment_support": (12,),
        "segment_conf": (12,),
        "segment_mask": (12,),
    }
    for key, shape in expected_shapes.items():
        assert plan[key].shape == shape, f"{key}: got {plan[key].shape} want {shape}"

    assert plan["anchor_mask"].dtype == np.bool_
    assert plan["segment_mask"].dtype == np.bool_
    assert plan["anchor_time"].dtype == np.int64
    assert plan["anchor_target_local"].dtype == np.float32


def test_invalid_slots_are_zero() -> None:
    cfg = InteractionPlanCompilerConfig()
    contact, tgt, phase, sup, obj_p, obj_r = _make_clip(
        T=80, contact_intervals_per_part={0: [(10, 25)]},
    )
    plan = compile_interaction_plan(
        contact, tgt, phase, sup, obj_p, obj_r, seq_len=80, cfg=cfg,
    )
    invalid = ~plan["anchor_mask"]
    if invalid.any():
        assert np.allclose(plan["anchor_target_local"][invalid], 0.0)
        assert np.allclose(plan["anchor_target_world"][invalid], 0.0)
        assert plan["anchor_time"][invalid].sum() == 0


# ---------------------------------------------------------------------------
# Hysteresis tests
# ---------------------------------------------------------------------------


def test_hysteresis_basic_pulse() -> None:
    prob = np.zeros(60, dtype=np.float32)
    prob[10:30] = 0.8
    segs = hysteresis_segments(prob, enter=0.55, exit=0.35, min_duration=4, gap_merge=3)
    assert segs == [(10, 29)]


def test_hysteresis_filters_short() -> None:
    prob = np.zeros(60, dtype=np.float32)
    prob[10:13] = 0.8     # 3-frame, below min_duration=4
    prob[20:35] = 0.8     # 15-frame, kept
    segs = hysteresis_segments(prob, enter=0.55, exit=0.35, min_duration=4, gap_merge=3)
    assert segs == [(20, 34)]


def test_hysteresis_merges_close() -> None:
    prob = np.zeros(60, dtype=np.float32)
    prob[10:20] = 0.8
    prob[22:35] = 0.8     # gap of 2 frames, merge_gap=3 → merge
    segs = hysteresis_segments(prob, enter=0.55, exit=0.35, min_duration=4, gap_merge=3)
    assert segs == [(10, 34)]


def test_hysteresis_does_not_merge_far() -> None:
    prob = np.zeros(60, dtype=np.float32)
    prob[10:20] = 0.8
    prob[30:40] = 0.8     # gap of 10 → don't merge
    segs = hysteresis_segments(prob, enter=0.55, exit=0.35, min_duration=4, gap_merge=3)
    assert segs == [(10, 19), (30, 39)]


# ---------------------------------------------------------------------------
# Anchor generation tests
# ---------------------------------------------------------------------------


def test_contact_segment_yields_onset_stable_release() -> None:
    cfg = InteractionPlanCompilerConfig(merge_window=1, temporal_nms_window=1, k_max=20)
    contact, tgt, phase, sup, obj_p, obj_r = _make_clip(
        T=80, contact_intervals_per_part={0: [(10, 40)]},
    )
    plan = compile_interaction_plan(
        contact, tgt, phase, sup, obj_p, obj_r, seq_len=80, cfg=cfg,
    )
    types = plan["anchor_type"][plan["anchor_mask"]]
    assert ANCHOR_TYPE_ONSET in types
    assert ANCHOR_TYPE_STABLE in types
    assert ANCHOR_TYPE_RELEASE in types


def test_phase_change_anchor_emitted() -> None:
    cfg = InteractionPlanCompilerConfig(merge_window=0, temporal_nms_window=0, k_max=20)
    T = 80
    P = 5
    contact = np.zeros((T, P), dtype=np.float32)
    tgt = np.zeros((T, P, 3), dtype=np.float32)
    phase = np.zeros((T, 3), dtype=np.float32)
    phase[: T // 2, 0] = 1.0
    phase[T // 2 :, 1] = 1.0
    sup = np.zeros((T, 3), dtype=np.float32)
    sup[:, 0] = 1.0
    obj_p = np.zeros((T, 3), dtype=np.float32)
    obj_r = np.zeros((T, 3), dtype=np.float32)
    plan = compile_interaction_plan(
        contact, tgt, phase, sup, obj_p, obj_r, seq_len=T, cfg=cfg,
    )
    types = plan["anchor_type"][plan["anchor_mask"]]
    assert ANCHOR_TYPE_PHASE_CHANGE in types


def test_support_change_anchor_emitted() -> None:
    cfg = InteractionPlanCompilerConfig(merge_window=0, temporal_nms_window=0, k_max=20)
    T = 80
    P = 5
    contact = np.zeros((T, P), dtype=np.float32)
    tgt = np.zeros((T, P, 3), dtype=np.float32)
    phase = np.zeros((T, 3), dtype=np.float32)
    phase[:, 0] = 1.0
    sup = np.zeros((T, 3), dtype=np.float32)
    sup[: T // 2, 0] = 1.0
    sup[T // 2 :, 1] = 1.0
    obj_p = np.zeros((T, 3), dtype=np.float32)
    obj_r = np.zeros((T, 3), dtype=np.float32)
    plan = compile_interaction_plan(
        contact, tgt, phase, sup, obj_p, obj_r, seq_len=T, cfg=cfg,
    )
    types = plan["anchor_type"][plan["anchor_mask"]]
    assert ANCHOR_TYPE_SUPPORT_CHANGE in types


def test_temporal_nms_caps_anchor_count() -> None:
    cfg = InteractionPlanCompilerConfig(k_max=4, temporal_nms_window=2)
    T = 200
    P = 5
    # Many contact intervals on different parts to force > k_max anchors
    contact = np.zeros((T, P), dtype=np.float32)
    for p in range(P):
        for k in range(5):
            contact[10 + 30 * k : 25 + 30 * k, p] = 1.0
    tgt = np.zeros((T, P, 3), dtype=np.float32)
    phase = np.zeros((T, 3), dtype=np.float32)
    phase[:, 0] = 1.0
    sup = np.zeros((T, 3), dtype=np.float32)
    sup[:, 0] = 1.0
    obj_p = np.zeros((T, 3), dtype=np.float32)
    obj_r = np.zeros((T, 3), dtype=np.float32)
    plan = compile_interaction_plan(
        contact, tgt, phase, sup, obj_p, obj_r, seq_len=T, cfg=cfg,
    )
    assert int(plan["anchor_mask"].sum()) <= 4


def test_zero_evidence_returns_k_min_fillers() -> None:
    cfg = InteractionPlanCompilerConfig(k_min=3)
    T = 80
    P = 5
    contact = np.zeros((T, P), dtype=np.float32)
    tgt = np.zeros((T, P, 3), dtype=np.float32)
    phase = np.zeros((T, 3), dtype=np.float32)
    phase[:, 0] = 1.0
    sup = np.zeros((T, 3), dtype=np.float32)
    sup[:, 0] = 1.0
    obj_p = np.zeros((T, 3), dtype=np.float32)
    obj_r = np.zeros((T, 3), dtype=np.float32)
    plan = compile_interaction_plan(
        contact, tgt, phase, sup, obj_p, obj_r, seq_len=T, cfg=cfg,
    )
    n_anch = int(plan["anchor_mask"].sum())
    assert n_anch >= 3, f"expected ≥ k_min=3 anchors with no evidence, got {n_anch}"


def test_short_clip_returns_empty_plan() -> None:
    cfg = InteractionPlanCompilerConfig()
    T = 1
    contact = np.zeros((T, 5), dtype=np.float32)
    tgt = np.zeros((T, 5, 3), dtype=np.float32)
    phase = np.zeros((T, 3), dtype=np.float32)
    sup = np.zeros((T, 3), dtype=np.float32)
    obj_p = np.zeros((T, 3), dtype=np.float32)
    obj_r = np.zeros((T, 3), dtype=np.float32)
    plan = compile_interaction_plan(
        contact, tgt, phase, sup, obj_p, obj_r, seq_len=T, cfg=cfg,
    )
    assert int(plan["anchor_mask"].sum()) == 0
    assert int(plan["segment_mask"].sum()) == 0


# ---------------------------------------------------------------------------
# World-lifting consistency tests
# ---------------------------------------------------------------------------


def test_world_lifting_matches_torch_trainer_convention() -> None:
    """The numpy lift must match the torch lift in anchor_consistency_loss
    to numerical precision; otherwise plan target_world drifts away from
    where the loss / anchor terms compute world coords."""
    from piano.training.anchor_consistency_loss import lift_object_local_to_world

    rng = np.random.default_rng(0)
    T, P = 50, 5
    target_local = rng.normal(0, 0.5, size=(T, P, 3)).astype(np.float32)
    object_pos = rng.normal(0, 1.0, size=(T, 3)).astype(np.float32)
    # Random axis-angle rotations with realistic magnitudes
    object_rot = rng.normal(0, 0.7, size=(T, 3)).astype(np.float32)

    np_world = lift_target_local_to_world_np(target_local, object_pos, object_rot)
    torch_world = lift_object_local_to_world(
        torch.from_numpy(target_local).unsqueeze(0),
        torch.from_numpy(object_pos).unsqueeze(0),
        torch.from_numpy(object_rot).unsqueeze(0),
    ).squeeze(0).numpy()
    np.testing.assert_allclose(np_world, torch_world, atol=1e-5)


def test_anchor_target_world_uses_object_pose() -> None:
    cfg = InteractionPlanCompilerConfig()
    T = 80
    contact, tgt, phase, sup, obj_p, obj_r = _make_clip(
        T=T, contact_intervals_per_part={0: [(10, 30)]},
    )
    obj_p = np.tile(np.array([5.0, 0.0, -2.0], dtype=np.float32), (T, 1))
    plan = compile_interaction_plan(
        contact, tgt, phase, sup, obj_p, obj_r, seq_len=T, cfg=cfg,
    )
    # First anchor of part 0: target_local[*, 0] = (0, 0.5, 0); world = local + obj_pos
    mask = plan["anchor_mask"]
    parts = plan["anchor_part"][mask, 0]
    assert (parts > 0).any()
    idx = np.where((parts > 0))[0][0]
    world = plan["anchor_target_world"][mask][idx, 0]
    assert abs(world[0] - 5.0) < 0.5    # x ≈ obj.x + 0
    assert abs(world[2] - (-2.0)) < 0.5  # z ≈ obj.z + 0


# ---------------------------------------------------------------------------
# Smoothing tests
# ---------------------------------------------------------------------------


def test_smooth_contact_preserves_steady_state() -> None:
    arr = np.tile(np.array([0.8, 0.2, 0.0, 0.0, 0.0]), (50, 1)).astype(np.float32)
    out = smooth_contact(arr, window=5)
    np.testing.assert_allclose(out, arr, atol=1e-5)


def test_smooth_contact_window_one_is_identity() -> None:
    arr = np.random.RandomState(0).rand(20, 5).astype(np.float32)
    out = smooth_contact(arr, window=1)
    np.testing.assert_allclose(out, arr)


# ---------------------------------------------------------------------------
# Determinism + collation
# ---------------------------------------------------------------------------


def test_determinism() -> None:
    cfg = InteractionPlanCompilerConfig()
    contact, tgt, phase, sup, obj_p, obj_r = _make_clip(
        T=80, contact_intervals_per_part={0: [(10, 30)], 1: [(50, 70)]},
    )
    p1 = compile_interaction_plan(
        contact, tgt, phase, sup, obj_p, obj_r, seq_len=80, cfg=cfg,
    )
    p2 = compile_interaction_plan(
        contact, tgt, phase, sup, obj_p, obj_r, seq_len=80, cfg=cfg,
    )
    for k in p1:
        np.testing.assert_array_equal(p1[k], p2[k])


def test_collation_stacks_correctly() -> None:
    cfg = InteractionPlanCompilerConfig()
    contact, tgt, phase, sup, obj_p, obj_r = _make_clip(
        T=80, contact_intervals_per_part={0: [(10, 30)]},
    )
    p1 = compile_interaction_plan(
        contact, tgt, phase, sup, obj_p, obj_r, seq_len=80, cfg=cfg,
    )
    p2 = compile_interaction_plan(
        contact, tgt, phase, sup, obj_p, obj_r, seq_len=80, cfg=cfg,
    )
    batched = collate_interaction_plans([p1, p2])
    for k, v in batched.items():
        assert v.shape[0] == 2
        assert v.shape[1:] == p1[k].shape


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_config_rejects_bad_thresholds() -> None:
    with pytest.raises(ValueError):
        InteractionPlanCompilerConfig(
            contact_enter_threshold=0.3,
            contact_exit_threshold=0.5,
        )


def test_config_rejects_part_priority_mismatch() -> None:
    with pytest.raises(ValueError):
        InteractionPlanCompilerConfig(num_parts=5, part_priority=(1.0, 1.0))
