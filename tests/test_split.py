"""Unit tests for piano.data.split — subject/object split logic.

Pure-Python (no torch). Imports directly from `piano.data.split` so
the test runs in any env that has the `piano` package installed,
without requiring torch.

Coverage:
- extract_subject_id: per-subset regex correctness on real seq_id formats
- build_subject_split: determinism, stratification, ratio, small-N clamp
- build_object_split: ratio sanity (back-compat regression)
"""
from __future__ import annotations

from piano.data.split import (
    _SUBJECT_PATTERNS,
    build_object_split,
    build_subject_split,
    extract_subject_id,
)


# ---------------------------------------------------------------------------
# extract_subject_id
# ---------------------------------------------------------------------------

def test_extract_subject_id_chairs() -> None:
    """chairs: Sub<digits>_Obj<digits>_Seg<X>_<frame>"""
    assert extract_subject_id("chairs", "Sub0001_Obj116_Seg0_0") == "Sub0001"
    assert extract_subject_id("chairs", "Sub0612_Obj118_Seg0_0_1") == "Sub0612"
    assert extract_subject_id("chairs", "Sub1407_Obj158_Seg0_0") == "Sub1407"


def test_extract_subject_id_imhd() -> None:
    """imhd: <YYYYMMDD>_<subjectname>_<rest>"""
    assert extract_subject_id("imhd", "20230825_songzn_bat_bat_holdhandle_hit_0_0") == "songzn"
    assert extract_subject_id("imhd", "20230901_wangwzh_suitcase_suitcase_twohands_push_3_0") == "wangwzh"
    assert extract_subject_id("imhd", "20231014_dujsh_kettlebell_kettlebell_swing3_1_0") == "dujsh"


def test_extract_subject_id_neuraldome() -> None:
    """neuraldome: subject<digits>_<rest>"""
    assert extract_subject_id("neuraldome", "subject01_baseball_0") == "subject01"
    assert extract_subject_id("neuraldome", "subject06_table_745") == "subject06"
    assert extract_subject_id("neuraldome", "subject01_keyboard_917_1") == "subject01"


def test_extract_subject_id_omomo() -> None:
    """omomo_correct_v2: sub<digits>_<rest>"""
    assert extract_subject_id("omomo_correct_v2", "sub10_clothesstand_000") == "sub10"
    assert extract_subject_id("omomo_correct_v2", "sub2_monitor_009") == "sub2"
    assert extract_subject_id("omomo_correct_v2", "sub16_largebox_048") == "sub16"


def test_extract_subject_id_unknown_subset_returns_none() -> None:
    """Unknown subset → None (caller drops the entry)."""
    assert extract_subject_id("nonsense_subset", "Sub0001_Obj1_Seg0_0") is None


def test_extract_subject_id_malformed_seq_id_returns_none() -> None:
    """seq_id that doesn't match the per-subset pattern → None."""
    assert extract_subject_id("chairs", "no_subject_prefix_here") is None
    assert extract_subject_id("imhd", "bare_seq_no_date") is None
    assert extract_subject_id("omomo_correct_v2", "subject01_wrong_pattern") is None


def test_subject_patterns_dict_covers_4_interact_subsets() -> None:
    """Locks in the 4-subset coverage so a refactor doesn't silently
    drop one — pseudo-label re-extraction would still work but the
    split would silently put all of that subset's clips into 'unknown'
    and drop them."""
    assert set(_SUBJECT_PATTERNS) == {"chairs", "imhd", "neuraldome", "omomo_correct_v2"}


# ---------------------------------------------------------------------------
# build_subject_split
# ---------------------------------------------------------------------------

def _mock_subject_keys() -> list[tuple[str, str]]:
    """Mirror v11 InterAct sizes: 403 / 9 / 10 / 17 subjects."""
    keys: list[tuple[str, str]] = []
    keys.extend(("chairs", f"Sub{i:04d}") for i in range(403))
    keys.extend(("imhd", n) for n in ["songzn", "wangwzh", "zhaochf", "dujsh", "lijh", "wangmj", "zhaoxq", "liuts", "yangcz"])
    keys.extend(("neuraldome", f"subject{i:02d}") for i in range(1, 11))
    keys.extend(("omomo_correct_v2", f"sub{i}") for i in range(1, 18))
    return keys


def test_subject_split_is_deterministic() -> None:
    """Same input + same seed → identical buckets across runs."""
    keys = _mock_subject_keys()
    s1 = build_subject_split(keys, train_pct=85, val_pct=15, seed=42)
    s2 = build_subject_split(keys, train_pct=85, val_pct=15, seed=42)
    assert s1 == s2


def test_subject_split_seed_changes_assignment() -> None:
    """Different seed → at least some subjects swap buckets."""
    keys = _mock_subject_keys()
    s1 = build_subject_split(keys, seed=42)
    s2 = build_subject_split(keys, seed=43)
    assert s1["train"] != s2["train"], "different seeds should produce different splits"


def test_subject_split_disjoint_buckets() -> None:
    """train and val partitions are disjoint and cover every subject."""
    keys = _mock_subject_keys()
    splits = build_subject_split(keys, seed=42)
    train, val = splits["train"], splits["val"]
    assert train.isdisjoint(val), "train and val overlap — split is broken"
    expected_total = sum(len(set(rid for s, rid in keys if s == subset)) for subset in {"chairs", "imhd", "neuraldome", "omomo_correct_v2"})
    assert len(train) + len(val) == expected_total


def test_subject_split_per_subset_stratified() -> None:
    """Every subset must appear in BOTH train and val (when n>=2 in
    that subset). The 9-subject imhd is the worst case — must get 8/1."""
    keys = _mock_subject_keys()
    splits = build_subject_split(keys, train_pct=85, val_pct=15, seed=42)
    by_subset_train: dict[str, int] = {}
    by_subset_val: dict[str, int] = {}
    for k in splits["train"]:
        sub = k.split("/", 1)[0]
        by_subset_train[sub] = by_subset_train.get(sub, 0) + 1
    for k in splits["val"]:
        sub = k.split("/", 1)[0]
        by_subset_val[sub] = by_subset_val.get(sub, 0) + 1
    for sub in ["chairs", "imhd", "neuraldome", "omomo_correct_v2"]:
        assert by_subset_train.get(sub, 0) >= 1, f"{sub} missing from train"
        assert by_subset_val.get(sub, 0) >= 1, f"{sub} missing from val (small-N stratification failed)"


def test_subject_split_ratio_close_to_target() -> None:
    """Per-subset ratio should be close to the target percentages."""
    keys = _mock_subject_keys()
    splits = build_subject_split(keys, train_pct=85, val_pct=15, seed=42)
    # Aggregate ratio across all subsets
    n_train = len(splits["train"])
    n_val = len(splits["val"])
    train_pct = n_train / (n_train + n_val) * 100
    # Within ~5pp of 85% for 439 total subjects (rounding + per-subset
    # clamping for small-N subsets means we won't hit exactly 85)
    assert 80 <= train_pct <= 90, f"train ratio {train_pct:.1f}% off target 85%"


def test_subject_split_keys_are_namespaced() -> None:
    """Output keys must be 'subset/raw_id' so cross-subset collisions
    are impossible (chairs/Sub10 vs hypothetical omomo/Sub10)."""
    keys = _mock_subject_keys()
    splits = build_subject_split(keys, seed=42)
    for k in splits["train"] | splits["val"]:
        assert "/" in k, f"split key {k!r} not namespaced as 'subset/raw_id'"
        subset, raw = k.split("/", 1)
        assert subset in {"chairs", "imhd", "neuraldome", "omomo_correct_v2"}
        assert raw, f"empty raw subject id in {k!r}"


def test_subject_split_dedup_within_subset() -> None:
    """Duplicate (subset, raw_id) entries are deduped — caller doesn't
    need to pre-dedup. Two passes of the same key → still one assignment."""
    keys = [("chairs", "Sub0001"), ("chairs", "Sub0001"), ("chairs", "Sub0002")]
    splits = build_subject_split(keys, train_pct=85, val_pct=15, seed=42)
    total = len(splits["train"]) + len(splits["val"])
    assert total == 2, f"expected 2 unique subjects after dedup, got {total}"


def test_subject_split_pct_validation() -> None:
    """train_pct + val_pct != 100 → raise."""
    import pytest
    with pytest.raises(ValueError):
        build_subject_split([("chairs", "Sub0001")], train_pct=80, val_pct=15)


def test_subject_split_single_subject_per_subset() -> None:
    """When a subset has exactly 1 subject, it goes into train (no val
    assignment is forced — we'd rather lose that subject from val than
    have train be empty)."""
    keys = [("imhd", "songzn")]
    splits = build_subject_split(keys, seed=42)
    assert "imhd/songzn" in splits["train"]
    assert "imhd/songzn" not in splits["val"]


# ---------------------------------------------------------------------------
# build_object_split (legacy / regression)
# ---------------------------------------------------------------------------

def test_object_split_pct_validation() -> None:
    """train_pct + val_pct + test_pct != 100 → raise."""
    import pytest
    with pytest.raises(ValueError):
        build_object_split(["A", "B"], train_pct=80, val_pct=15, test_pct=10)


def test_object_split_disjoint() -> None:
    """train / val / test are disjoint."""
    obj_ids = [f"Obj{i:03d}" for i in range(100)]
    splits = build_object_split(obj_ids, seed=42)
    train, val, test = splits["train"], splits["val"], splits["test"]
    assert train.isdisjoint(val) and train.isdisjoint(test) and val.isdisjoint(test)
    assert len(train) + len(val) + len(test) == 100


def test_object_split_deterministic() -> None:
    obj_ids = [f"Obj{i:03d}" for i in range(50)]
    s1 = build_object_split(obj_ids, seed=42)
    s2 = build_object_split(obj_ids, seed=42)
    assert s1 == s2
