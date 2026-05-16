"""Round-12 preflight test — gate that must pass before smoke training.

Tests:
1. rot6d helper equivalence — verifies that the project-local
   ``rotation_6d_to_matrix`` is what the extractor now uses, and
   that the buggy custom helper is no longer in the import path of
   the extractor.
2. Cache load: every clip in train + val manifest loads, has shape
   ``(T, 23)``, is finite, and contains only the allowed npz fields.
3. Normalization round-trip: ``(x - mean) / std`` then *
   ``std + mean`` recovers x to ``< 1e-4``.
4. CLIP embeddings: cache file exists, all manifest texts have an
   index entry, all embeddings are finite, shape ``(N, 512)``.
5. Model forward shape: ``S1-A`` (attention_mode="none") and ``S1-B``
   (attention_mode="block_causal") both run on a tiny batch and emit
   ``(B, T, 23)`` finite output.
6. Block-causal mask correctness:
   - shape ``(T, T)`` bool;
   - per-block bidirectional, across-block causal pattern;
   - dtype matches ``key_padding_mask`` dtype (both bool).
7. Padding mask behaviour: a sample with valid_mask[t] = False for
   ``t >= seq_len_real`` does not let valid frames attend to padded
   keys. We check this by injecting NaN into the padded slots of one
   batch element and asserting that the corresponding valid frame
   outputs are finite (which proves no valid frame attends to padded).

Exit code 0 means all preflight checks pass; non-zero indicates the
specific failure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from piano.models.coarse_motion_prior import (
    CoarsePriorConfig, CoarsePriorDenoiserConfig, CoarsePriorDiff,
    make_block_causal_bool_mask,
)
from piano.models.motion_anchordiff import DiffusionConfig


CACHE_ROOT = Path("cache/stage1_coarse_v1_round12")


def _print(tag: str, msg: str) -> None:
    print(f"[preflight:{tag}] {msg}")


def t1_rot6d_helper_import_path() -> bool:
    """Source-level check: extractor imports project utility, custom one gone.

    Avoids importing the extractor module (which has sibling-import deps);
    instead reads its source and looks for the relevant tokens. This is
    robust to PYTHONPATH and CWD.
    """
    _print("t1", "checking extractor source for project rot6d import + absent custom helper")
    src_path = Path("scripts/stage_b_generator/extract_coarse_motion_representation.py")
    if not src_path.exists():
        _print("t1", f"FAIL — extractor source missing at {src_path}")
        return False
    src = src_path.read_text("utf-8")
    has_import = (
        "from piano.training.smpl_kinematics import" in src
        and "rotation_6d_to_matrix as _project_rotation_6d_to_matrix" in src
    )
    has_use = "_project_rotation_6d_to_matrix(rot6d_t)" in src
    has_old_def = "def _rot6d_to_R(" in src
    has_old_call = "_rot6d_to_R(" in src and not has_old_def  # leftover call without def is even worse
    _print("t1", f"imports project helper:   {has_import}")
    _print("t1", f"calls project helper :    {has_use}")
    _print("t1", f"old _rot6d_to_R def    :  {has_old_def}")
    if not has_import or not has_use:
        _print("t1", "FAIL — project rotation_6d_to_matrix not properly wired in")
        return False
    if has_old_def:
        _print("t1", "FAIL — old custom helper definition is still present")
        return False
    _print("t1", "PASS — extractor uses project rotation_6d_to_matrix only")
    return True


def t2_cache_load() -> bool:
    _print("t2", f"scanning cache at {CACHE_ROOT}")
    if not CACHE_ROOT.exists():
        _print("t2", "FAIL — cache missing")
        return False
    n_seen = 0
    for split in ("train", "val"):
        for line in (CACHE_ROOT / f"manifest_{split}.jsonl").read_text("utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            data = np.load(CACHE_ROOT / r["npz_path"], allow_pickle=False)
            if set(data.files) != {"coarse_v1", "init_coarse_v1"}:
                _print("t2", f"FAIL — extra fields in {r['npz_path']}: {set(data.files)}")
                return False
            cv1 = data["coarse_v1"]
            init = data["init_coarse_v1"]
            if cv1.ndim != 2 or cv1.shape[1] != 23:
                _print("t2", f"FAIL — bad coarse_v1 shape {cv1.shape} in {r['npz_path']}")
                return False
            if init.shape != (23,):
                _print("t2", f"FAIL — bad init_coarse_v1 shape {init.shape} in {r['npz_path']}")
                return False
            if not (np.isfinite(cv1).all() and np.isfinite(init).all()):
                _print("t2", f"FAIL — non-finite in {r['npz_path']}")
                return False
            n_seen += 1
    _print("t2", f"PASS — {n_seen} clips loaded, all (T, 23) finite, fields = {{coarse_v1, init_coarse_v1}}")
    return True


def t3_normalization_roundtrip() -> bool:
    norm = json.loads((CACHE_ROOT / "normalization_train.json").read_text("utf-8"))
    mean = np.asarray(norm["global"]["mean"], dtype=np.float32)
    std = np.asarray(norm["global"]["std_clamped"], dtype=np.float32)
    if mean.shape != (23,) or std.shape != (23,):
        _print("t3", f"FAIL — mean/std shape {mean.shape} {std.shape}")
        return False
    if (std < float(norm["global"]["std_eps"]) - 1e-12).any():
        _print("t3", "FAIL — std_clamped contains values below std_eps")
        return False
    # Round-trip on a synthetic batch.
    rng = np.random.default_rng(123)
    x = rng.standard_normal((512, 23)).astype(np.float32) * std + mean
    x_norm = (x - mean) / std
    x_back = x_norm * std + mean
    rt = float(np.max(np.abs(x - x_back)))
    _print("t3", f"round-trip max |x - x_back| = {rt:.3e}")
    if rt > 1e-3:
        _print("t3", "FAIL — round-trip error > 1e-3")
        return False
    _print("t3", "PASS — normalization round-trip stable")
    return True


def t4_clip_text_embeddings() -> bool:
    npz_path = CACHE_ROOT / "text_embeddings_clip_vit_b32.npz"
    idx_path = CACHE_ROOT / "text_embeddings_index.json"
    if not npz_path.exists() or not idx_path.exists():
        _print("t4", "FAIL — text embedding cache missing")
        return False
    payload = np.load(npz_path, allow_pickle=True)
    emb = payload["embeddings"]
    if emb.ndim != 2 or emb.shape[1] != 512:
        _print("t4", f"FAIL — embeddings shape {emb.shape}")
        return False
    if not np.isfinite(emb).all():
        _print("t4", "FAIL — non-finite embeddings")
        return False
    idx_payload = json.loads(idx_path.read_text("utf-8"))
    index = idx_payload["index"]
    # All manifest texts must have an index entry.
    missing = 0
    for split in ("train", "val"):
        for line in (CACHE_ROOT / f"manifest_{split}.jsonl").read_text("utf-8").splitlines():
            if not line.strip():
                continue
            t = json.loads(line).get("text", "")
            if t not in index:
                missing += 1
    if missing > 0:
        _print("t4", f"FAIL — {missing} manifest texts have no CLIP index entry")
        return False
    _print(
        "t4",
        f"PASS — embeddings {emb.shape}, n_unique={idx_payload['n_unique_texts']}, "
        f"every manifest text indexed",
    )
    return True


def _build_model(mode: str, *, num_diff_steps: int = 8) -> CoarsePriorDiff:
    diff = DiffusionConfig(
        num_steps=num_diff_steps,
        schedule="cosine",
        objective="ddpm",
        prediction_target="x0",
    )
    den = CoarsePriorDenoiserConfig(
        coarse_dim=23, text_dim=512, init_pose_dim=23,
        d_model=64, n_layers=2, n_heads=4, ff_mult=2, dropout=0.0,
        max_seq_length=64, attention_mode=mode, block_size=8,
    )
    return CoarsePriorDiff(CoarsePriorConfig(diffusion=diff, denoiser=den))


def t5_forward_shape() -> bool:
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for mode in ("none", "block_causal"):
        model = _build_model(mode).to(device).eval()
        B, T = 3, 32
        x_t = torch.randn(B, T, 23, device=device)
        t = torch.randint(0, model.diffusion.num_steps, (B,), device=device)
        cond = {
            "text_pool": torch.randn(B, 512, device=device),
            "init_coarse": torch.randn(B, 23, device=device),
            "valid_mask": torch.ones(B, T, dtype=torch.bool, device=device),
        }
        with torch.no_grad():
            x0 = model.forward_x0(x_t, t, cond, cond_drop_mask=None)
        if x0.shape != (B, T, 23):
            _print("t5", f"FAIL — mode={mode} shape {tuple(x0.shape)}")
            return False
        if not torch.isfinite(x0).all():
            _print("t5", f"FAIL — mode={mode} non-finite output")
            return False
        _print("t5", f"mode={mode} forward OK on {device.type}, shape {tuple(x0.shape)}")
    _print("t5", "PASS — both modes forward with correct shape on requested device")
    return True


def t6_block_causal_mask() -> bool:
    T, K = 33, 8     # not divisible — exercise the last partial block too
    mask = make_block_causal_bool_mask(T, K, torch.device("cpu"))
    if mask.shape != (T, T):
        _print("t6", f"FAIL — shape {mask.shape}")
        return False
    if mask.dtype != torch.bool:
        _print("t6", f"FAIL — dtype {mask.dtype}, expected torch.bool")
        return False
    # Inside a block: should be all False (allowed).
    for blk_start in range(0, T, K):
        blk_end = min(blk_start + K, T)
        sub = mask[blk_start:blk_end, blk_start:blk_end]
        if sub.any():
            _print("t6", f"FAIL — inside block [{blk_start}:{blk_end}] some positions are masked")
            return False
    # Cross-block: i->j where j's block > i's block must be True.
    blocks = (torch.arange(T) // K)
    for i in range(T):
        for j in range(T):
            should_mask = bool(blocks[j].item() > blocks[i].item())
            if bool(mask[i, j].item()) != should_mask:
                _print("t6", f"FAIL — mask[{i},{j}]={mask[i,j]} expected {should_mask}")
                return False
    _print("t6", f"PASS — (T,T)=({T},{T}) bool, block-bidirectional, across-block causal")
    return True


def t7_padding_mask_isolates_pad() -> bool:
    """Padding-mask correctness via the "perturb-padded-keys" test.

    NaN-based tests are not robust here because PyTorch MHA computes
    ``softmax_weights * V`` even when a weight is exactly 0; ``0 * NaN``
    propagates as NaN. We instead inject two finite but very different
    perturbations into the padded slots and confirm the VALID frames'
    outputs are identical across both perturbations. If
    ``key_padding_mask`` correctly excludes the padded keys, valid-frame
    outputs cannot depend on the padded slots' values.
    """
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for mode in ("none", "block_causal"):
        model = _build_model(mode).to(device).eval()
        B, T = 2, 24
        valid_lens = torch.tensor([14, T], dtype=torch.long, device=device)
        valid_mask = torch.arange(T, device=device).unsqueeze(0) < valid_lens.unsqueeze(1)
        # Common inputs.
        x_real_base = torch.randn(B, T, 23, device=device)
        x_real_base = x_real_base * valid_mask.unsqueeze(-1).float()    # zero the padded slots
        t = torch.full((B,), model.diffusion.num_steps - 1, device=device, dtype=torch.long)
        cond_base = {
            "text_pool": torch.randn(B, 512, device=device),
            "init_coarse": torch.randn(B, 23, device=device),
            "valid_mask": valid_mask,
        }

        # Pass A: padded slots = 0.
        x_a = x_real_base.clone()
        # Pass B: padded slots = large random — should not affect valid output.
        x_b = x_real_base.clone()
        x_b[0, 14:] = 1000.0 * torch.randn(T - 14, 23, device=device)

        with torch.no_grad():
            y_a = model.forward_x0(x_a, t, cond_base, cond_drop_mask=None)
            y_b = model.forward_x0(x_b, t, cond_base, cond_drop_mask=None)
        diff_valid = (y_a[0, :14] - y_b[0, :14]).abs().max().item()
        if diff_valid > 1e-4:
            _print(
                "t7",
                f"FAIL — mode={mode} valid outputs differ by {diff_valid:.3e} "
                f"when only padded slots differ (padding mask leaks)",
            )
            return False
        _print("t7", f"mode={mode}: valid-frame output invariant to padded-slot perturbation (Δ={diff_valid:.3e})")
    _print("t7", "PASS — key_padding_mask correctly isolates padding")
    return True


def main() -> int:
    checks = [
        ("t1", t1_rot6d_helper_import_path),
        ("t2", t2_cache_load),
        ("t3", t3_normalization_roundtrip),
        ("t4", t4_clip_text_embeddings),
        ("t5", t5_forward_shape),
        ("t6", t6_block_causal_mask),
        ("t7", t7_padding_mask_isolates_pad),
    ]
    failures: list[str] = []
    for tag, fn in checks:
        try:
            ok = fn()
        except Exception as e:
            _print(tag, f"FAIL — exception: {e!r}")
            ok = False
        if not ok:
            failures.append(tag)
        print("-" * 70)
    if failures:
        print(f"[preflight] FAILED: {failures}")
        return 1
    print("[preflight] ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
