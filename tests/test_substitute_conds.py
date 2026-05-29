"""Tests for the substitute_conds plumbing used by R31/R32 diag.

Covers:
  - load_substitute_conds_for_clip: shape + truncation + missing-file
    error semantics.
  - sample_substitute_conds .npz cache layout consistency (Stage-1 +
    Stage-1.5 keys present, shapes match the model output dims).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from piano.inference.diagnostic_helpers import load_substitute_conds_for_clip
from piano.models.stage1_trajectory import STAGE1_COARSE_DIM
from piano.models.stage1p5_interaction import (
    STAGE1P5_C41_DIM,
    STAGE1P5_S4_DIM,
)


def _write_clip(out_root: Path, subset: str, seq_id: str, **arrays) -> Path:
    sub = out_root / subset
    sub.mkdir(parents=True, exist_ok=True)
    p = sub / f"{seq_id}.npz"
    np.savez(p, **arrays)
    return p


def test_load_returns_none_when_substitute_dir_is_none():
    out = load_substitute_conds_for_clip(
        None, "anysub", "anyseq", T=196, device=torch.device("cpu"),
    )
    assert out is None


def test_load_reads_stage1_only(tmp_path: Path):
    T = 32
    arr = np.random.RandomState(0).randn(T, STAGE1_COARSE_DIM).astype(np.float32)
    _write_clip(
        tmp_path, "chairs", "clip0",
        stage1_coarse=arr, valid_T=np.int32(T), seed=np.int32(42),
    )
    out = load_substitute_conds_for_clip(
        tmp_path, "chairs", "clip0", T=T, device=torch.device("cpu"),
    )
    assert set(out.keys()) == {"stage1_coarse"}
    assert out["stage1_coarse"].shape == (1, T, STAGE1_COARSE_DIM)
    assert torch.allclose(
        out["stage1_coarse"][0].float(), torch.from_numpy(arr).float(),
    )


def test_load_reads_stage1p5_both_keys(tmp_path: Path):
    T = 24
    c41 = np.random.RandomState(1).randn(T, STAGE1P5_C41_DIM).astype(np.float32)
    s4 = np.random.RandomState(2).randn(T, STAGE1P5_S4_DIM).astype(np.float32)
    _write_clip(
        tmp_path, "imhd", "clip1",
        stage2_coarse_extra=c41,
        stage2_support=s4,
        valid_T=np.int32(T), seed=np.int32(42),
    )
    out = load_substitute_conds_for_clip(
        tmp_path, "imhd", "clip1", T=T, device=torch.device("cpu"),
    )
    assert set(out.keys()) == {"stage2_coarse_extra", "stage2_support"}
    assert out["stage2_coarse_extra"].shape == (1, T, STAGE1P5_C41_DIM)
    assert out["stage2_support"].shape == (1, T, STAGE1P5_S4_DIM)


def test_load_reads_end_to_end_merged_clip(tmp_path: Path):
    """End-to-end (D) clips carry stage1_coarse + C41 + S4 in one .npz."""
    T = 16
    sc = np.random.RandomState(3).randn(T, STAGE1_COARSE_DIM).astype(np.float32)
    c41 = np.random.RandomState(4).randn(T, STAGE1P5_C41_DIM).astype(np.float32)
    s4 = np.random.RandomState(5).randn(T, STAGE1P5_S4_DIM).astype(np.float32)
    _write_clip(
        tmp_path, "neuraldome", "clip2",
        stage1_coarse=sc,
        stage2_coarse_extra=c41,
        stage2_support=s4,
        valid_T=np.int32(T), seed=np.int32(42),
    )
    out = load_substitute_conds_for_clip(
        tmp_path, "neuraldome", "clip2", T=T, device=torch.device("cpu"),
    )
    assert set(out.keys()) == {
        "stage1_coarse", "stage2_coarse_extra", "stage2_support",
    }


def test_load_truncates_to_diag_T(tmp_path: Path):
    T_cached = 64
    T_diag = 32
    sc = np.random.RandomState(0).randn(T_cached, STAGE1_COARSE_DIM).astype(np.float32)
    _write_clip(
        tmp_path, "omomo", "clipL",
        stage1_coarse=sc, valid_T=np.int32(T_cached), seed=np.int32(42),
    )
    out = load_substitute_conds_for_clip(
        tmp_path, "omomo", "clipL", T=T_diag, device=torch.device("cpu"),
    )
    assert out["stage1_coarse"].shape == (1, T_diag, STAGE1_COARSE_DIM)
    assert torch.allclose(
        out["stage1_coarse"][0].float(),
        torch.from_numpy(sc[:T_diag]).float(),
    )


def test_load_raises_when_clip_missing(tmp_path: Path):
    """We want a FileNotFoundError, not silent fall-through to oracle."""
    # Note: pass tmp_path as substitute_dir without writing any clip.
    with pytest.raises(FileNotFoundError, match="clipX"):
        load_substitute_conds_for_clip(
            tmp_path, "chairs", "clipX", T=16,
            device=torch.device("cpu"),
        )


def test_load_raises_when_cached_T_too_short(tmp_path: Path):
    T_cached = 8
    T_diag = 32
    sc = np.random.RandomState(0).randn(T_cached, STAGE1_COARSE_DIM).astype(np.float32)
    _write_clip(
        tmp_path, "chairs", "clipS",
        stage1_coarse=sc, valid_T=np.int32(T_cached), seed=np.int32(42),
    )
    with pytest.raises(ValueError, match="T="):
        load_substitute_conds_for_clip(
            tmp_path, "chairs", "clipS", T=T_diag,
            device=torch.device("cpu"),
        )
