"""Unit tests for Round-30 ILD diagnosis helpers.

Per analyses/2026-05-29_round30_idle_local_detail_diagnosis_plan.md.
Covers:
  - round30_build_ild_subset.py: feature extractors + classifier rules
  - round30_text_condition_probe.py: pure helpers (no model)

Heavy dependencies (omegaconf, torch dataset) are not exercised here;
they would require fixtures of dataset npzs and a CLIP encoder. Smoke
testing of the full pipeline happens on the server.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "stage_b_generator"
sys.path.insert(0, str(SCRIPTS))

# Imports under test — split to keep failure messages local.
from round30_build_ild_subset import (  # noqa: E402
    IDLE_LOCAL_DETAIL_KEYWORDS,
    PELVIS_IDX,
    UPPER_BODY_JOINT_INDICES,
    ClipFeatures,
    _contact_event_count_and_frac,
    _keyword_hit,
    _root_xz_p95_m,
    _stratified_size_match_control,
    _upper_body_vel_rms_mps,
)
from round30_text_condition_probe import (  # noqa: E402
    PERTURBATIONS_DEFAULT as TEXT_PERTURBATIONS_DEFAULT,
    UPPER_BODY_JOINT_INDICES as TEXT_UPPER_BODY_JOINT_INDICES,
    UPPER_BODY_JOINT_NAMES as TEXT_UPPER_BODY_JOINT_NAMES,
    _aggregate,
    _full_body_delta_cm,
    _gate_verdict,
    _upper_body_delta_cm,
)


# --------------------------------------------------------------------------- #
# Shared constants — sanity guards so a typo in either file is caught.
# --------------------------------------------------------------------------- #


def test_upper_body_joint_indices_match_between_modules():
    """The filter and the eval must agree on which 8 joints are 'upper body'."""
    assert UPPER_BODY_JOINT_INDICES == TEXT_UPPER_BODY_JOINT_INDICES
    assert TEXT_UPPER_BODY_JOINT_NAMES == (
        "neck", "L_shoulder", "R_shoulder", "L_elbow", "R_elbow",
        "spine1", "spine2", "spine3",
    )


def test_upper_body_indices_are_smpl_22_legal():
    for idx in UPPER_BODY_JOINT_INDICES:
        assert 0 <= idx < 22


def test_pelvis_idx_is_smpl_pelvis():
    assert PELVIS_IDX == 0


# --------------------------------------------------------------------------- #
# _root_xz_p95_m
# --------------------------------------------------------------------------- #


def test_root_xz_p95_zero_when_stationary():
    T = 30
    joints = np.zeros((T, 22, 3), dtype=np.float32)
    assert _root_xz_p95_m(joints, T) == 0.0


def test_root_xz_p95_picks_up_walk():
    """1 m diagonal walk over 30 frames → p95 ≈ 1.4 m (sqrt(2))."""
    T = 30
    joints = np.zeros((T, 22, 3), dtype=np.float32)
    for t in range(T):
        joints[t, PELVIS_IDX, 0] = (t / (T - 1)) * 1.0   # +X 1 m
        joints[t, PELVIS_IDX, 2] = (t / (T - 1)) * 1.0   # +Z 1 m
    out = _root_xz_p95_m(joints, T)
    assert out > 1.3 and out < 1.45


def test_root_xz_p95_handles_short_clip():
    assert _root_xz_p95_m(np.zeros((1, 22, 3)), 1) == 0.0
    assert _root_xz_p95_m(np.zeros((0, 22, 3)), 0) == 0.0


# --------------------------------------------------------------------------- #
# _contact_event_count_and_frac
# --------------------------------------------------------------------------- #


def test_contact_event_zero_when_no_contact():
    cs = np.zeros((20, 5), dtype=np.float32)
    n, frac = _contact_event_count_and_frac(cs, 20)
    assert n == 0
    assert frac == 0.0


def test_contact_event_counts_zero_to_one_transitions():
    """Left hand goes 0→1 at t=5, stays on. Right hand stays off.
    Expect 1 event, frac = 15/20 = 0.75."""
    cs = np.zeros((20, 5), dtype=np.float32)
    cs[5:, 0] = 1.0
    n, frac = _contact_event_count_and_frac(cs, 20)
    assert n == 1
    assert frac == pytest.approx(15.0 / 20.0)


def test_contact_event_counts_both_hands():
    """L 0→1 at t=3, R 0→1 at t=10 → 2 events."""
    cs = np.zeros((20, 5), dtype=np.float32)
    cs[3:, 0] = 1.0
    cs[10:, 1] = 1.0
    n, _ = _contact_event_count_and_frac(cs, 20)
    assert n == 2


def test_contact_event_handles_none():
    n, frac = _contact_event_count_and_frac(None, 20)
    assert n == 0
    assert frac == 0.0


def test_contact_event_stable_contact_zero_events():
    """Hand starts already in contact and stays — that's stable contact,
    NOT a contact event."""
    cs = np.ones((20, 5), dtype=np.float32)   # left hand on the whole time
    cs[:, 1:] = 0.0
    n, _ = _contact_event_count_and_frac(cs, 20)
    assert n == 0


# --------------------------------------------------------------------------- #
# _upper_body_vel_rms_mps
# --------------------------------------------------------------------------- #


def test_upper_body_vel_rms_zero_when_static():
    joints = np.zeros((30, 22, 3), dtype=np.float32)
    assert _upper_body_vel_rms_mps(joints, 30) == 0.0


def test_upper_body_vel_rms_picks_up_motion():
    """Neck joint moves +0.05 m/frame on X. fps=20. Speed = 1.0 m/s
    on neck only. RMS over 8 upper joints = sqrt(1/8) = 0.353 m/s."""
    T = 30
    joints = np.zeros((T, 22, 3), dtype=np.float32)
    for t in range(T):
        joints[t, 12, 0] = 0.05 * t   # neck moves
    out = _upper_body_vel_rms_mps(joints, T, fps=20.0, walking_mask=None)
    # joint-mean per frame = 1.0/8; RMS = 1/8.
    assert out == pytest.approx(1.0 / 8.0, abs=1e-4)


def test_upper_body_vel_rms_excludes_walking_frames():
    """If walking_mask is True everywhere, the (T-1) frames are all
    excluded, function falls through to all-frame fallback (because
    we mask away ALL the diff frames, returning 0)."""
    T = 30
    joints = np.zeros((T, 22, 3), dtype=np.float32)
    for t in range(T):
        joints[t, 12, 0] = 0.05 * t
    wm = np.ones(T - 1, dtype=np.float32)
    out = _upper_body_vel_rms_mps(joints, T, fps=20.0, walking_mask=wm)
    assert out == 0.0


# --------------------------------------------------------------------------- #
# Keyword regex
# --------------------------------------------------------------------------- #


@pytest.fixture
def compiled_kw():
    return [re.compile(p, re.IGNORECASE) for p in IDLE_LOCAL_DETAIL_KEYWORDS]


def test_keyword_hit_positives(compiled_kw):
    for text in [
        "A person sits and stretches their arms.",
        "She rests her chin on her hand.",
        "He puts his hand on his head and scratches.",
        "They cross their arms and lean back.",
        "She covers her face with her palms.",
        "He raises a hand and waves.",
        "She nods slowly.",
    ]:
        assert _keyword_hit(text, compiled_kw), f"missed: {text!r}"


def test_keyword_hit_negatives(compiled_kw):
    for text in [
        "A person walks to the chair and sits down.",
        "She picks up the box and carries it.",
        "He kicks the ball.",
        "They run forward.",
    ]:
        assert not _keyword_hit(text, compiled_kw), f"false positive: {text!r}"


# --------------------------------------------------------------------------- #
# ClipFeatures classification rules
# --------------------------------------------------------------------------- #


def _feat(
    *,
    root_xz=0.02, walking=0.01, contact_events=0,
    contact_any=0.0, ub_vel=0.05, keyword=False,
    subset="chairs", seq_id="seq000", split="train",
):
    return ClipFeatures(
        subset=subset, seq_id=seq_id, split=split,
        text="x", num_frames=100,
        root_xz_p95_m=root_xz, walking_frac=walking,
        contact_event_count=contact_events,
        contact_any_frac=contact_any,
        upper_body_vel_rms_mps=ub_vel,
        keyword_hit=keyword,
    )


def test_is_ild_canonical_positive():
    f = _feat(root_xz=0.02, walking=0.01, contact_events=0, ub_vel=0.05)
    assert f.is_stationary(0.05, 0.05)
    assert not f.has_significant_contact()
    assert f.has_upper_body_motion(0.03)


def test_is_not_ild_when_walking():
    f = _feat(root_xz=0.30, walking=0.40, ub_vel=0.05)
    assert not f.is_stationary(0.05, 0.05)


def test_is_not_ild_with_contact_event():
    f = _feat(contact_events=1)
    assert f.has_significant_contact()


def test_is_not_ild_with_persistent_contact_no_event():
    """Regression on the first server run: neuraldome 'lifts the box with
    both hands' had contact_event_count=0 but contact_any_frac=1.0 (hand
    is in contact for the entire clip). That should be EXCLUDED from ILD
    — it's continuous manipulation, not idle local detail.
    """
    f = _feat(contact_events=0, contact_any=1.0)
    assert f.has_significant_contact()


def test_persistent_contact_threshold_is_30pct():
    """Just under threshold → not significant; just over → significant."""
    assert not _feat(contact_events=0, contact_any=0.25).has_significant_contact()
    assert _feat(contact_events=0, contact_any=0.35).has_significant_contact()


def test_keyword_alone_qualifies_as_upper_body_motion():
    """If text says 'sits and stretches' but the actor barely moves the
    joints (low velocity), the keyword still triggers ILD."""
    f = _feat(ub_vel=0.001, keyword=True)
    assert f.has_upper_body_motion(0.03)


def test_neither_keyword_nor_velocity_fails():
    f = _feat(ub_vel=0.001, keyword=False)
    assert not f.has_upper_body_motion(0.03)


# --------------------------------------------------------------------------- #
# Stratified control sampling
# --------------------------------------------------------------------------- #


def test_stratified_match_respects_subset_quotas():
    # Build a feature pool: 10 chairs (3 are ILD), 10 imhd (1 ILD).
    # That leaves 7 non-ILD chairs + 9 non-ILD imhd, well above the
    # quotas (3, 1).
    feats = []
    for i in range(10):
        feats.append(_feat(subset="chairs", seq_id=f"c{i}"))
    for i in range(10):
        feats.append(_feat(subset="imhd", seq_id=f"i{i}"))
    ild_keys = {("chairs", "c0"), ("chairs", "c1"), ("chairs", "c2"),
                ("imhd", "i0")}
    target = {"chairs": 3, "imhd": 1}
    rng = np.random.default_rng(0)
    control = _stratified_size_match_control(feats, ild_keys, target, rng)
    # Got exactly 3 chairs + 1 imhd, none of which is in ild_keys.
    assert len(control) == 4
    chairs = [f for f in control if f.subset == "chairs"]
    imhd = [f for f in control if f.subset == "imhd"]
    assert len(chairs) == 3
    assert len(imhd) == 1
    for f in control:
        assert (f.subset, f.seq_id) not in ild_keys


def test_stratified_match_caps_at_pool_size():
    """Asking for more control clips than non-ILD pool has → returns
    whatever is available, no crash."""
    feats = [_feat(subset="chairs", seq_id=f"c{i}") for i in range(3)]
    ild_keys = {("chairs", "c0")}
    target = {"chairs": 10}
    rng = np.random.default_rng(0)
    control = _stratified_size_match_control(feats, ild_keys, target, rng)
    assert len(control) == 2   # only c1, c2 left


# --------------------------------------------------------------------------- #
# build_subject_split return format — pin against round30 split lookup bug
# --------------------------------------------------------------------------- #


def test_build_subject_split_returns_namespaced_string_keys():
    """The first server run had subj_to_bucket misuse: keys were inserted
    as (subset, sid) tuples but build_subject_split actually returns
    namespaced strings "{subset}/{sid}". The lookup silently failed and
    val ended up with 0 clips. This test pins the actual return format so
    a future regression on either side is caught locally.
    """
    from piano.data.split import build_subject_split

    keys = [
        ("chairs", "Sub0001"), ("chairs", "Sub0002"),
        ("imhd", "songzn"), ("omomo_correct_v2", "sub10"),
    ]
    splits = build_subject_split(keys, train_pct=85, val_pct=15, seed=42)
    assert "train" in splits and "val" in splits
    # Every element is a "{subset}/{sid}" string, NOT a tuple.
    all_ids = splits["train"] | splits["val"]
    for k in all_ids:
        assert isinstance(k, str), f"split id {k!r} is not str"
        assert "/" in k, f"split id {k!r} is not namespaced as 'subset/sid'"


# --------------------------------------------------------------------------- #
# E1: _upper_body_delta_cm / _full_body_delta_cm
# --------------------------------------------------------------------------- #


def test_upper_body_delta_zero_when_identical():
    j = np.zeros((10, 22, 3), dtype=np.float32)
    m, p = _upper_body_delta_cm(j, j)
    assert m == 0.0
    assert p == 0.0


def test_upper_body_delta_uses_cm_conversion():
    """5 cm shift on neck only — delta on upper-body should be ~5 cm
    averaged over 8 joints × T frames = 5/8 cm = 0.625 cm mean."""
    T = 10
    a = np.zeros((T, 22, 3), dtype=np.float32)
    b = np.zeros((T, 22, 3), dtype=np.float32)
    b[:, 12, 0] = 0.05   # neck shifted 5 cm
    m, p = _upper_body_delta_cm(a, b)
    # 1 of 8 joints moves 5 cm → mean = 5/8 cm. p95 = 5 cm (the moved joint).
    assert m == pytest.approx(5.0 / 8.0, abs=1e-3)
    assert p == pytest.approx(5.0, abs=1e-3)


def test_upper_body_delta_ignores_lower_body():
    """Shift only ankle (idx 7) — upper-body should not move."""
    T = 10
    a = np.zeros((T, 22, 3), dtype=np.float32)
    b = np.zeros((T, 22, 3), dtype=np.float32)
    b[:, 7, 0] = 0.10
    m, p = _upper_body_delta_cm(a, b)
    assert m == pytest.approx(0.0, abs=1e-6)
    assert p == pytest.approx(0.0, abs=1e-6)


def test_full_body_delta_includes_all_joints():
    T = 10
    a = np.zeros((T, 22, 3), dtype=np.float32)
    b = np.zeros((T, 22, 3), dtype=np.float32)
    b[..., 0] = 0.05
    m, p = _full_body_delta_cm(a, b)
    assert m == pytest.approx(5.0, abs=1e-3)


# --------------------------------------------------------------------------- #
# E1: _aggregate
# --------------------------------------------------------------------------- #


def _trow(pert, ub_m, fb_m=None, subset="chairs", seq_id="s0"):
    from round30_text_condition_probe import TextPerturbationResult
    return TextPerturbationResult(
        variant_id="test", subset=subset, seq_id=seq_id,
        perturbation=pert,
        upper_body_mean_cm=ub_m,
        upper_body_p95_cm=ub_m * 1.5,
        full_body_mean_cm=ub_m if fb_m is None else fb_m,
        full_body_p95_cm=(ub_m if fb_m is None else fb_m) * 1.5,
    )


def test_aggregate_averages_per_perturbation():
    rows = [
        _trow("text_zero", 5.0, seq_id="s0"),
        _trow("text_zero", 7.0, seq_id="s1"),
        _trow("text_swap_neutral", 1.0, seq_id="s0"),
    ]
    agg = _aggregate(rows)
    assert agg["text_zero"]["n_clips"] == 2
    assert agg["text_zero"]["upper_body_mean_cm"] == pytest.approx(6.0)
    assert agg["text_swap_neutral"]["upper_body_mean_cm"] == pytest.approx(1.0)


def test_aggregate_skips_nan():
    rows = [
        _trow("text_zero", 5.0),
        _trow("text_zero", float("nan"), seq_id="s1"),
    ]
    agg = _aggregate(rows)
    assert agg["text_zero"]["n_clips"] == 2
    assert agg["text_zero"]["upper_body_mean_cm"] == pytest.approx(5.0)


# --------------------------------------------------------------------------- #
# E1: _gate_verdict
# --------------------------------------------------------------------------- #


def _gate_input(*, z, sn=None, sa=None):
    """Build a minimal ild_aggregate dict for _gate_verdict."""
    agg = {"text_zero": {"upper_body_mean_cm": z}}
    if sn is not None:
        agg["text_swap_neutral"] = {"upper_body_mean_cm": sn}
    if sa is not None:
        agg["text_swap_antonym"] = {"upper_body_mean_cm": sa}
    return agg


def test_gate_text_dead_when_zero_and_swap_low():
    """The H5 trigger: zero AND swap both < 2 cm → text effectively dead."""
    g = _gate_verdict(_gate_input(z=1.0, sn=1.5, sa=1.8))
    assert g["label"] == "text_dead"


def test_gate_text_semantically_alive():
    """zero ≥ 5 cm AND swap_antonym > 1.3 × swap_neutral → semantically alive."""
    g = _gate_verdict(_gate_input(z=8.0, sn=3.0, sa=5.0))
    assert g["label"] == "text_semantically_alive"


def test_gate_text_responds_but_aspecific():
    """zero ≥ 5 cm but swap_antonym ≈ swap_neutral → text reacts to zero
    but lacks semantic resolution."""
    g = _gate_verdict(_gate_input(z=8.0, sn=4.0, sa=4.1))
    assert g["label"] == "text_responds_but_aspecific"


def test_gate_text_partial_in_between():
    """zero in [2, 5) → ambiguous partial response."""
    g = _gate_verdict(_gate_input(z=3.0, sn=2.5, sa=2.5))
    assert g["label"] == "text_partial"


def test_gate_unknown_when_zero_missing():
    g = _gate_verdict({})
    assert g["label"] == "unknown"


# --------------------------------------------------------------------------- #
# Default perturbation set sanity
# --------------------------------------------------------------------------- #


def test_text_perturbations_default_includes_all_five():
    assert set(TEXT_PERTURBATIONS_DEFAULT) == {
        "baseline",
        "text_zero",
        "text_swap_neutral",
        "text_swap_antonym",
        "text_shuffle_token",
    }
