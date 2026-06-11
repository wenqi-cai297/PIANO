"""Unit tests for Stage-1.5 cond source selection (R43 Stage A).

Covers Codex r43_p0_finalized_review_for_claude_code §5 checklist:
- generated cache is loaded as z-scored, not re-normalized
- mixed mode selects in z-space
- eval mixed mode uses generated only
- missing cache entry fails with (subset, seq_id, path)
- wrong shape and non-finite values fail clearly
- loader validation paths
- oracle bit-equivalence (passthrough)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from piano.training.stage1p5_cond_sources import (
    GeneratedCoarseCacheError,
    VALID_COND_SOURCES,
    load_generated_coarse_z_for_batch,
    select_stage1_coarse,
)


_T = 60
_C = 23
_B = 4


def _make_batch(
    *, subsets: list[str], seq_ids: list[str], T: int = _T
) -> dict:
    """Minimal batch mimicking what collate_hoi produces (dataset.py:1332)."""
    B = len(subsets)
    return {
        "subset": subsets,
        "seq_id": seq_ids,
        "motion": torch.zeros(B, T, 135, dtype=torch.float32),
    }


def _write_cache_entry(
    root: Path, subset: str, seq_id: str, arr: np.ndarray
) -> Path:
    out_dir = root / subset
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{seq_id}.npz"
    np.savez(path, stage1_coarse=arr.astype(np.float32),
             valid_T=np.int32(arr.shape[0]), seed=np.int32(0))
    return path


@pytest.fixture
def populated_cache(tmp_path: Path) -> tuple[Path, list[str], list[str], np.ndarray]:
    """Build a cache dir with B = 4 entries of shape (T, 23)."""
    root = tmp_path / "cache"
    subsets = ["chairs", "chairs", "tables", "tables"]
    seq_ids = ["Sub0001_Obj01_Seg0_0", "Sub0002_Obj02_Seg0_0",
               "Sub0003_Obj03_Seg0_0", "Sub0004_Obj04_Seg0_0"]
    rng = np.random.RandomState(123)
    # z-scored: roughly zero mean, unit std
    truth = rng.randn(len(subsets), _T, _C).astype(np.float32)
    for i, (s, q) in enumerate(zip(subsets, seq_ids)):
        _write_cache_entry(root, s, q, truth[i])
    return root, subsets, seq_ids, truth


def test_loader_returns_z_scored_values(populated_cache):
    """The loader must return the exact values stored on disk, NOT re-normalized.

    This is Codex's load-bearing claim (§1): the cache is already z-scored,
    so any transform inside the loader is wrong.
    """
    root, subsets, seq_ids, truth = populated_cache
    batch = _make_batch(subsets=subsets, seq_ids=seq_ids)
    out = load_generated_coarse_z_for_batch(
        batch=batch, cache_root=root, expected_T=_T,
    )
    assert out.shape == (len(subsets), _T, _C)
    np.testing.assert_allclose(
        out.cpu().numpy(), truth, rtol=0.0, atol=1e-6,
    )


def test_loader_respects_device_dtype(populated_cache):
    root, subsets, seq_ids, _ = populated_cache
    batch = _make_batch(subsets=subsets, seq_ids=seq_ids)
    out = load_generated_coarse_z_for_batch(
        batch=batch, cache_root=root, expected_T=_T,
        device=torch.device("cpu"), dtype=torch.float64,
    )
    assert out.dtype == torch.float64
    assert out.device.type == "cpu"


def test_loader_trims_longer_cache(tmp_path):
    """T_cached > expected_T: trim. Matches inference helper convention."""
    root = tmp_path / "cache"
    arr = np.zeros((80, _C), dtype=np.float32)
    arr[:_T, 0] = 1.0  # mark first valid frames
    _write_cache_entry(root, "chairs", "X", arr)
    batch = _make_batch(subsets=["chairs"], seq_ids=["X"])
    out = load_generated_coarse_z_for_batch(
        batch=batch, cache_root=root, expected_T=_T,
    )
    assert out.shape == (1, _T, _C)
    assert torch.allclose(out[0, :, 0], torch.ones(_T))


def test_loader_fails_on_missing_entry(tmp_path):
    root = tmp_path / "cache"
    root.mkdir()
    batch = _make_batch(subsets=["chairs"], seq_ids=["missing"])
    with pytest.raises(GeneratedCoarseCacheError) as exc:
        load_generated_coarse_z_for_batch(
            batch=batch, cache_root=root, expected_T=_T,
        )
    # Message must mention subset + seq_id + path so the operator can
    # locate the offending entry.
    msg = str(exc.value)
    assert "chairs" in msg
    assert "missing" in msg
    assert ".npz" in msg


def test_loader_fails_on_short_cache(tmp_path):
    root = tmp_path / "cache"
    arr = np.zeros((_T - 5, _C), dtype=np.float32)
    _write_cache_entry(root, "chairs", "Y", arr)
    batch = _make_batch(subsets=["chairs"], seq_ids=["Y"])
    with pytest.raises(GeneratedCoarseCacheError) as exc:
        load_generated_coarse_z_for_batch(
            batch=batch, cache_root=root, expected_T=_T,
        )
    msg = str(exc.value)
    assert str(_T - 5) in msg and str(_T) in msg


def test_loader_fails_on_wrong_shape(tmp_path):
    root = tmp_path / "cache"
    arr = np.zeros((_T, _C - 1), dtype=np.float32)  # wrong channel
    _write_cache_entry(root, "chairs", "Z", arr)
    batch = _make_batch(subsets=["chairs"], seq_ids=["Z"])
    with pytest.raises(GeneratedCoarseCacheError) as exc:
        load_generated_coarse_z_for_batch(
            batch=batch, cache_root=root, expected_T=_T,
        )
    assert "shape" in str(exc.value).lower()


def test_loader_fails_on_non_finite(tmp_path):
    root = tmp_path / "cache"
    arr = np.zeros((_T, _C), dtype=np.float32)
    arr[3, 5] = float("nan")
    _write_cache_entry(root, "chairs", "Q", arr)
    batch = _make_batch(subsets=["chairs"], seq_ids=["Q"])
    with pytest.raises(GeneratedCoarseCacheError) as exc:
        load_generated_coarse_z_for_batch(
            batch=batch, cache_root=root, expected_T=_T,
        )
    assert "finite" in str(exc.value).lower()


def test_loader_fails_on_missing_stage1_coarse_key(tmp_path):
    root = tmp_path / "cache"
    (root / "chairs").mkdir(parents=True)
    np.savez(
        root / "chairs" / "K.npz",
        stage2_coarse_extra=np.zeros((_T, 18), dtype=np.float32),
    )
    batch = _make_batch(subsets=["chairs"], seq_ids=["K"])
    with pytest.raises(GeneratedCoarseCacheError) as exc:
        load_generated_coarse_z_for_batch(
            batch=batch, cache_root=root, expected_T=_T,
        )
    assert "stage1_coarse" in str(exc.value)


def test_select_oracle_passthrough(populated_cache):
    """Oracle mode: returns oracle_z bit-identically (no cache touched)."""
    root, subsets, seq_ids, _ = populated_cache
    batch = _make_batch(subsets=subsets, seq_ids=seq_ids)
    oracle_z = torch.randn(len(subsets), _T, _C)
    out = select_stage1_coarse(
        cond_source="oracle",
        oracle_z=oracle_z,
        batch=batch,
        cache_root=root,        # ignored
        generated_prob=0.5,     # ignored
        training=True,
    )
    assert out is oracle_z  # exact same tensor identity, no copy


def test_select_oracle_does_not_need_cache_root(populated_cache):
    """Oracle mode must work with cache_root=None (R38-B1 cfg has none)."""
    _, subsets, seq_ids, _ = populated_cache
    batch = _make_batch(subsets=subsets, seq_ids=seq_ids)
    oracle_z = torch.randn(len(subsets), _T, _C)
    out = select_stage1_coarse(
        cond_source="oracle",
        oracle_z=oracle_z,
        batch=batch,
        cache_root=None,
        generated_prob=0.0,
        training=True,
    )
    assert out is oracle_z


def test_select_generated_cache_uses_cached_values(populated_cache):
    root, subsets, seq_ids, truth = populated_cache
    batch = _make_batch(subsets=subsets, seq_ids=seq_ids)
    oracle_z = torch.full((len(subsets), _T, _C), 999.0)
    out = select_stage1_coarse(
        cond_source="generated_cache",
        oracle_z=oracle_z,
        batch=batch,
        cache_root=root,
        generated_prob=0.0,
        training=True,
    )
    np.testing.assert_allclose(out.cpu().numpy(), truth, atol=1e-6)


def test_select_mixed_eval_uses_pure_generated(populated_cache):
    """eval-mode mixed must use generated only (Codex §3.2)."""
    root, subsets, seq_ids, truth = populated_cache
    batch = _make_batch(subsets=subsets, seq_ids=seq_ids)
    oracle_z = torch.full((len(subsets), _T, _C), -7.0)
    out = select_stage1_coarse(
        cond_source="mixed",
        oracle_z=oracle_z,
        batch=batch,
        cache_root=root,
        generated_prob=0.5,
        training=False,         # eval
    )
    np.testing.assert_allclose(out.cpu().numpy(), truth, atol=1e-6)


def test_select_mixed_training_p0_is_all_oracle(populated_cache):
    """generated_prob=0 in training mode = always oracle (per-item)."""
    root, subsets, seq_ids, _ = populated_cache
    batch = _make_batch(subsets=subsets, seq_ids=seq_ids)
    oracle_z = torch.full((len(subsets), _T, _C), 3.14)
    torch.manual_seed(0)
    out = select_stage1_coarse(
        cond_source="mixed",
        oracle_z=oracle_z,
        batch=batch,
        cache_root=root,
        generated_prob=0.0,
        training=True,
    )
    assert torch.allclose(out, oracle_z)


def test_select_mixed_training_p1_is_all_generated(populated_cache):
    """generated_prob=1 in training mode = always generated."""
    root, subsets, seq_ids, truth = populated_cache
    batch = _make_batch(subsets=subsets, seq_ids=seq_ids)
    oracle_z = torch.full((len(subsets), _T, _C), -1.5)
    torch.manual_seed(0)
    out = select_stage1_coarse(
        cond_source="mixed",
        oracle_z=oracle_z,
        batch=batch,
        cache_root=root,
        generated_prob=1.0,
        training=True,
    )
    np.testing.assert_allclose(out.cpu().numpy(), truth, atol=1e-6)


def test_select_mixed_training_p05_picks_per_item(populated_cache):
    """generated_prob=0.5 must produce per-item mask, not whole-batch.

    Check that each item is either fully oracle or fully generated
    (no within-clip mixing) and that over many trials roughly half are
    generated.
    """
    root, subsets, seq_ids, truth = populated_cache
    batch = _make_batch(subsets=subsets, seq_ids=seq_ids)
    oracle_z = torch.full((len(subsets), _T, _C), -42.0)

    gen_count = 0
    n_trials = 200
    for trial in range(n_trials):
        torch.manual_seed(trial)
        out = select_stage1_coarse(
            cond_source="mixed",
            oracle_z=oracle_z,
            batch=batch,
            cache_root=root,
            generated_prob=0.5,
            training=True,
        )
        # Each item is either == oracle_z slice or == truth slice
        for i in range(len(subsets)):
            is_oracle = torch.allclose(out[i], oracle_z[i])
            is_gen = np.allclose(out[i].cpu().numpy(), truth[i], atol=1e-6)
            assert is_oracle ^ is_gen, (
                f"item {i} in trial {trial} is neither pure oracle nor "
                "pure generated — mixing within a clip is a bug"
            )
            if is_gen:
                gen_count += 1
    # Statistical sanity: expect ~50% generated picks
    total = n_trials * len(subsets)
    assert 0.4 * total < gen_count < 0.6 * total, (
        f"gen_count={gen_count}/{total} far from p=0.5 expectation"
    )


def test_select_unknown_source_raises(populated_cache):
    root, subsets, seq_ids, _ = populated_cache
    batch = _make_batch(subsets=subsets, seq_ids=seq_ids)
    oracle_z = torch.randn(len(subsets), _T, _C)
    with pytest.raises(ValueError, match="unknown cond_source"):
        select_stage1_coarse(
            cond_source="bogus",
            oracle_z=oracle_z,
            batch=batch,
            cache_root=root,
            generated_prob=0.5,
            training=True,
        )


def test_select_non_oracle_requires_cache_root(populated_cache):
    _, subsets, seq_ids, _ = populated_cache
    batch = _make_batch(subsets=subsets, seq_ids=seq_ids)
    oracle_z = torch.randn(len(subsets), _T, _C)
    with pytest.raises(ValueError, match="cache_root"):
        select_stage1_coarse(
            cond_source="generated_cache",
            oracle_z=oracle_z,
            batch=batch,
            cache_root=None,
            generated_prob=0.5,
            training=True,
        )


def test_valid_sources_contract():
    assert "oracle" in VALID_COND_SOURCES
    assert "generated_cache" in VALID_COND_SOURCES
    assert "mixed" in VALID_COND_SOURCES
    assert len(VALID_COND_SOURCES) == 3
