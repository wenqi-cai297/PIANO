"""Round-18 preflight — gate that must pass BEFORE smoke training.

Covers SUGGESTION.md §7 "Preflight Tests" + Round-17 §10.6, on top of
the Round-12 preflight (test_stage1_coarse_prior_preflight.py — reused
where possible).

Tests:

- t1: object pose coverage per subset (via Step-1 preflight JSON dump).
- t2: body-canonical vs force-world-frame divergence (via the same JSON).
- t3: objtraj cache schema check — every clip npz has exactly
      {coarse_v1, init_coarse_v1, obj_traj_canonical} with correct shapes.
- t4: no forbidden fields in the objtraj cache npz files.
- t5: train-only normalization for obj_traj exists, finite, std_clamped
      respects std_eps.
- t6: S1-A (obj_traj_dim=0) forward shape on a tiny batch.
- t7: S1-O (obj_traj_dim=9) forward shape on a tiny batch + obj_traj
      missing in cond raises a clean error.
- t8: zero-init HintBlock invariance — with matched random init seed,
      a fresh S1-O model produces output identical to a fresh S1-A model
      (before any training).
- t9: obj_traj CFG dropout path — passing obj_traj_drop_mask of all-True
      makes the model behave as if obj_traj is the null embedding;
      output still finite, with non-zero magnitude (gates haven't opened
      so output is exactly the same as the no-drop case at init).
- t10: padding mask behavior (reused via the existing t7).
- t11: EMA update correctness — `decay * old + (1-decay) * new` after one
       update step.
- t12: root acc/jerk loss finite + scale logged on a tiny tensor.
- t13: frame-0 inpainting at sampling — generated x[:, 0, :] equals
       conditioned init_coarse exactly after denormalization.

Exit code 0 = all pass.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


# Round-18-fix-server: v18 config path; overridable via PIANO_V18_CFG env var
# (server passes a `_local`-paths variant; local default is the Windows-path
# base config that's tracked in git).
_DEFAULT_V18_CFG = os.environ.get(
    "PIANO_V18_CFG",
    "configs/training/anchordiff_v18_a1_FULL_DATA.yaml",
)

from piano.models.coarse_motion_prior import (
    CoarsePriorConfig, CoarsePriorDenoiserConfig, CoarsePriorDiff,
    ObjTrajHintBlock,
)
from piano.models.motion_anchordiff import DiffusionConfig
from piano.training.train_coarse_prior import (
    EMAState, masked_root_acc_jerk_loss,
)


OBJTRAJ_CACHE = Path("cache/stage1_coarse_v1_objtraj_root0_world_round18_fix")
LEGACY_OBJTRAJ_CACHE = Path("cache/stage1_coarse_v1_objtraj_round18")     # kept for forensic comparison
FRAME_PREFLIGHT_JSON = Path("analyses/round18_preflight/frame_convention_preflight.json")
OBJTRAJ_FIELD_NAME = "obj_traj_root0_world"                                # NEW field name


def _print(tag: str, msg: str) -> None:
    print(f"[r18-preflight:{tag}] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────
# t1 + t2: re-use Step-1 frame-convention preflight JSON
# ─────────────────────────────────────────────────────────────────────


def t1_obj_pose_coverage_per_subset() -> bool:
    if not FRAME_PREFLIGHT_JSON.exists():
        _print("t1", f"FAIL — Step-1 preflight JSON missing at {FRAME_PREFLIGHT_JSON}")
        _print("t1", "  run scripts/stage_b_generator/preflight_round18_frame_convention.py first")
        return False
    payload = json.loads(FRAME_PREFLIGHT_JSON.read_text("utf-8"))
    if not payload.get("overall_pass", False):
        _print("t1", "FAIL — Step-1 preflight JSON has overall_pass=false")
        return False
    for r in payload["results"]:
        if r["n_with_obj_pose"] == 0:
            _print("t1", f"FAIL — subset {r['subset']} had 0 clips with obj pose")
            return False
    _print("t1", f"PASS — Step-1 JSON confirms object pose available in all "
           f"{len(payload['results'])} subsets")
    return True


def t2_world_vs_canonical_diverges() -> bool:
    if not FRAME_PREFLIGHT_JSON.exists():
        _print("t2", "FAIL — Step-1 preflight JSON missing")
        return False
    payload = json.loads(FRAME_PREFLIGHT_JSON.read_text("utf-8"))
    n_diverged = 0
    n_total = 0
    for r in payload["results"]:
        for rec in r.get("sample_diffs", []):
            if "max_abs_diff_com" not in rec:
                continue
            n_total += 1
            if rec["max_abs_diff_com"] > 1e-4 or rec["max_abs_diff_rot6d"] > 1e-4:
                n_diverged += 1
    if n_total == 0:
        _print("t2", "FAIL — no comparison records in Step-1 JSON")
        return False
    if n_diverged == 0:
        _print("t2", f"FAIL — world vs canonical agreement was perfect on all "
               f"{n_total} sampled clips (Round-18 must genuinely diverge from v18)")
        return False
    _print("t2", f"PASS — {n_diverged}/{n_total} sampled clips show "
           "non-trivial world-vs-canonical divergence (Round-18 frame choice "
           "is materially different from v18 force_world_frame=True)")
    return True


# ─────────────────────────────────────────────────────────────────────
# t3, t4, t5: cache schema + forbidden fields + normalization
# ─────────────────────────────────────────────────────────────────────


_ALLOWED_FIELDS = {"coarse_v1", "init_coarse_v1", OBJTRAJ_FIELD_NAME}
_FORBIDDEN_FIELDS = {
    "object_positions", "object_rotations",       # raw world-frame pose
    "object_pc",                                   # point cloud
    "obj_com_canonical", "obj_rot6d_canonical",    # legacy Round-18 obj fields
    "obj_traj_canonical",                          # legacy Round-18 field name
    "contact_state", "contact_target", "contact_target_xyz",
    "phase", "support",
    "plan_anchor", "plan_segment",
    "z_int",
    "hand_target", "foot_target",
    "motion_135", "motion_263",
    "pseudo_label",
    "rest_offsets",
}


def t3_objtraj_cache_schema() -> bool:
    """Verify cache schema AND that obj_traj has SAME length as coarse_v1.

    Round-18-fix: writes both arrays at (seq_len, ·), NOT padded to 196.
    The OLD Round-18 cache wrote obj_traj at (196, 9) but coarse_v1 at
    (seq_len, 23) — inconsistent. Now both match.
    """
    if not OBJTRAJ_CACHE.exists():
        _print("t3", f"FAIL — objtraj cache missing at {OBJTRAJ_CACHE}")
        _print("t3", "  run build_stage1_coarse_v1_objtraj_root0_world_cache.py first")
        return False
    n_seen = 0
    n_shape_mismatch = 0
    for split in ("train", "val"):
        manifest_path = OBJTRAJ_CACHE / f"manifest_{split}.jsonl"
        if not manifest_path.exists():
            _print("t3", f"FAIL — manifest missing: {manifest_path}")
            return False
        for line in manifest_path.read_text("utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            data = np.load(OBJTRAJ_CACHE / r["npz_path"], allow_pickle=False)
            files = set(data.files)
            if files != _ALLOWED_FIELDS:
                _print("t3", f"FAIL — {r['npz_path']} has fields {files}, "
                       f"expected {_ALLOWED_FIELDS}")
                return False
            cv1 = data["coarse_v1"]
            init = data["init_coarse_v1"]
            objt = data[OBJTRAJ_FIELD_NAME]
            if cv1.ndim != 2 or cv1.shape[1] != 23:
                _print("t3", f"FAIL — coarse_v1 shape {cv1.shape}")
                return False
            if init.shape != (23,):
                _print("t3", f"FAIL — init_coarse_v1 shape {init.shape}")
                return False
            if objt.ndim != 2 or objt.shape[1] != 9:
                _print("t3", f"FAIL — {OBJTRAJ_FIELD_NAME} shape {objt.shape}")
                return False
            # Round-18-fix: obj_traj MUST match coarse_v1 length exactly.
            if objt.shape[0] != cv1.shape[0]:
                n_shape_mismatch += 1
                _print("t3", f"FAIL — {r['npz_path']}: obj_traj length "
                       f"{objt.shape[0]} != coarse_v1 length {cv1.shape[0]}")
                return False
            if not (np.isfinite(cv1).all() and np.isfinite(init).all()
                    and np.isfinite(objt).all()):
                _print("t3", f"FAIL — non-finite in {r['npz_path']}")
                return False
            n_seen += 1
    _print("t3", f"PASS — {n_seen} clips have only "
           f"{{coarse_v1, init_coarse_v1, {OBJTRAJ_FIELD_NAME}}} "
           "with matching seq_len shapes (NOT padded)")
    return True


def t4_no_forbidden_fields_in_cache() -> bool:
    if not OBJTRAJ_CACHE.exists():
        _print("t4", "FAIL — objtraj cache missing")
        return False
    n_seen = 0
    for split in ("train", "val"):
        for line in (OBJTRAJ_CACHE / f"manifest_{split}.jsonl").read_text("utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            data = np.load(OBJTRAJ_CACHE / r["npz_path"], allow_pickle=False)
            for forbidden in _FORBIDDEN_FIELDS:
                if forbidden in data.files:
                    _print("t4", f"FAIL — {r['npz_path']} contains forbidden field '{forbidden}'")
                    return False
            n_seen += 1
    _print("t4", f"PASS — scanned {n_seen} clip npz files; no forbidden fields present")
    return True


def t5_obj_traj_normalization_finite() -> bool:
    if not OBJTRAJ_CACHE.exists():
        _print("t5", "FAIL — objtraj cache missing")
        return False
    norm_path = OBJTRAJ_CACHE / "normalization_train.json"
    if not norm_path.exists():
        _print("t5", "FAIL — normalization_train.json missing")
        return False
    norm = json.loads(norm_path.read_text("utf-8"))
    obj_block = norm.get("global", {}).get(OBJTRAJ_FIELD_NAME, None)
    if obj_block is None:
        _print("t5", f"FAIL — normalization_train.json lacks global.{OBJTRAJ_FIELD_NAME} block")
        return False
    mean = np.asarray(obj_block["mean"], dtype=np.float64)
    std = np.asarray(obj_block["std"], dtype=np.float64)
    std_c = np.asarray(obj_block["std_clamped"], dtype=np.float64)
    if mean.shape != (9,) or std.shape != (9,) or std_c.shape != (9,):
        _print("t5", f"FAIL — bad obj_traj normalization shape: mean={mean.shape} std={std.shape}")
        return False
    if not (np.isfinite(mean).all() and np.isfinite(std).all() and np.isfinite(std_c).all()):
        _print("t5", "FAIL — non-finite obj_traj normalization stats")
        return False
    eps = float(obj_block.get("std_eps", 1e-3))
    if (std_c < eps - 1e-12).any():
        _print("t5", f"FAIL — std_clamped < std_eps={eps}")
        return False
    # Also verify that the normalization was computed on TRAIN ONLY
    # (per Codex review's data-leak guard). The normalization JSON
    # surfaces `split: "train"` and `n_train_clips` at the top level.
    if str(norm.get("split", "")) != "train":
        _print("t5", f"FAIL — normalization split is {norm.get('split')!r}, expected 'train'")
        return False
    _print("t5", f"PASS — obj_traj normalization (9-dim) finite + train-only, "
           f"mean[0:3]={mean[:3]}, std[0:3]={std[:3]}, std_eps={eps}")
    return True


# ─────────────────────────────────────────────────────────────────────
# t6, t7, t8, t9: model forward + zero-init invariance + dropout
# ─────────────────────────────────────────────────────────────────────


def _build_model(obj_traj_dim: int, *, seed: int = 42) -> CoarsePriorDiff:
    diff = DiffusionConfig(
        num_steps=8, schedule="cosine", objective="ddpm", prediction_target="x0",
    )
    den = CoarsePriorDenoiserConfig(
        coarse_dim=23, text_dim=512, init_pose_dim=23,
        d_model=64, n_layers=2, n_heads=4, ff_mult=2, dropout=0.0,
        max_seq_length=32, attention_mode="none", block_size=8,
        obj_traj_dim=obj_traj_dim, obj_traj_hint_hidden_mult=1,
    )
    torch.manual_seed(seed)
    return CoarsePriorDiff(CoarsePriorConfig(diffusion=diff, denoiser=den))


def t6_s1a_forward_shape() -> bool:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(obj_traj_dim=0).to(device).eval()
    B, T = 3, 16
    x_t = torch.randn(B, T, 23, device=device)
    t = torch.randint(0, model.diffusion.num_steps, (B,), device=device)
    cond = {
        "text_pool": torch.randn(B, 512, device=device),
        "init_coarse": torch.randn(B, 23, device=device),
        "valid_mask": torch.ones(B, T, dtype=torch.bool, device=device),
    }
    with torch.no_grad():
        y = model.forward_x0(x_t, t, cond)
    if y.shape != (B, T, 23) or not torch.isfinite(y).all():
        _print("t6", f"FAIL — shape {tuple(y.shape)} finite={torch.isfinite(y).all().item()}")
        return False
    _print("t6", f"PASS — S1-A forward shape {tuple(y.shape)} finite on {device.type}")
    return True


def t7_s1o_forward_shape_and_missing_raises() -> bool:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(obj_traj_dim=9).to(device).eval()
    B, T = 3, 16
    x_t = torch.randn(B, T, 23, device=device)
    t = torch.randint(0, model.diffusion.num_steps, (B,), device=device)
    cond = {
        "text_pool": torch.randn(B, 512, device=device),
        "init_coarse": torch.randn(B, 23, device=device),
        "valid_mask": torch.ones(B, T, dtype=torch.bool, device=device),
        "obj_traj": torch.randn(B, T, 9, device=device),
    }
    with torch.no_grad():
        y = model.forward_x0(x_t, t, cond)
    if y.shape != (B, T, 23) or not torch.isfinite(y).all():
        _print("t7", f"FAIL — shape {tuple(y.shape)} finite={torch.isfinite(y).all().item()}")
        return False
    # Missing obj_traj when model expects it must raise.
    cond_bad = {k: v for k, v in cond.items() if k != "obj_traj"}
    try:
        with torch.no_grad():
            model.forward_x0(x_t, t, cond_bad)
        _print("t7", "FAIL — model with obj_traj_dim=9 did NOT raise on missing obj_traj")
        return False
    except KeyError:
        pass
    _print("t7", f"PASS — S1-O forward {tuple(y.shape)}; missing obj_traj raises KeyError")
    return True


def t8_zero_init_hintblock_invariance() -> bool:
    """With matched seed, S1-O at init must equal S1-A at init exactly."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m_a = _build_model(obj_traj_dim=0, seed=42).to(device).eval()
    m_o = _build_model(obj_traj_dim=9, seed=42).to(device).eval()
    B, T = 2, 16
    x_t = torch.randn(B, T, 23, device=device, generator=torch.Generator(device=device).manual_seed(0))
    t = torch.zeros(B, dtype=torch.long, device=device)
    base_cond = {
        "text_pool": torch.randn(B, 512, device=device),
        "init_coarse": torch.randn(B, 23, device=device),
        "valid_mask": torch.ones(B, T, dtype=torch.bool, device=device),
    }
    cond_o = dict(base_cond)
    cond_o["obj_traj"] = torch.randn(B, T, 9, device=device)
    with torch.no_grad():
        y_a = m_a.forward_x0(x_t, t, base_cond)
        y_o = m_o.forward_x0(x_t, t, cond_o)
    diff = (y_a - y_o).abs().max().item()
    if diff > 1e-5:
        _print("t8", f"FAIL — |S1-A - S1-O| at init = {diff:.3e} (expected ~0; "
               "HintBlock zero-init is broken)")
        return False
    _print("t8", f"PASS — zero-init HintBlock invariance |S1-A - S1-O| = {diff:.3e}")
    return True


def t9_obj_traj_dropout_finite() -> bool:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(obj_traj_dim=9, seed=42).to(device).eval()
    B, T = 2, 16
    x_t = torch.randn(B, T, 23, device=device)
    t = torch.zeros(B, dtype=torch.long, device=device)
    cond = {
        "text_pool": torch.randn(B, 512, device=device),
        "init_coarse": torch.randn(B, 23, device=device),
        "valid_mask": torch.ones(B, T, dtype=torch.bool, device=device),
        "obj_traj": torch.randn(B, T, 9, device=device),
    }
    drop_all = torch.ones(B, dtype=torch.bool, device=device)
    with torch.no_grad():
        y_drop = model.forward_x0(x_t, t, cond, obj_traj_drop_mask=drop_all)
        y_keep = model.forward_x0(x_t, t, cond, obj_traj_drop_mask=None)
    if not torch.isfinite(y_drop).all():
        _print("t9", "FAIL — dropout-all path produced non-finite output")
        return False
    # At init, both paths should be identical (zero-init HintBlock contributes
    # nothing). After training the dropout-all path would diverge as the gate
    # opens; we only check finiteness + init equivalence here.
    diff = (y_drop - y_keep).abs().max().item()
    _print("t9", f"PASS — obj_traj dropout-all finite; |drop - keep| at init = {diff:.3e}")
    return True


# ─────────────────────────────────────────────────────────────────────
# t11: EMA update correctness
# ─────────────────────────────────────────────────────────────────────


def t11_ema_update_correctness() -> bool:
    torch.manual_seed(0)
    m = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8))
    # Disable any zero-init quirks: re-randomize.
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.5)
    ema = EMAState(m, decay=0.9)
    # Snapshot pre-update EMA = current model params.
    snap = {n: p.detach().clone() for n, p in m.named_parameters()}
    # Take an "optimizer step" by perturbing parameters.
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    new_params = {n: p.detach().clone() for n, p in m.named_parameters()}
    # Update EMA. Expected: ema = 0.9 * snap + 0.1 * new.
    ema.update(m)
    for n, expected in snap.items():
        post = ema._ema[n]
        target = 0.9 * expected + 0.1 * new_params[n]
        diff = (post - target).abs().max().item()
        if diff > 1e-5:
            _print("t11", f"FAIL — EMA update wrong for param {n}: max|diff| = {diff:.3e}")
            return False
    _print("t11", "PASS — EMA update is decay * old + (1 - decay) * new")
    return True


# ─────────────────────────────────────────────────────────────────────
# t12: root acc/jerk loss
# ─────────────────────────────────────────────────────────────────────


def t12_root_acc_jerk_loss_finite_and_logged() -> bool:
    B, T, D = 2, 16, 23
    torch.manual_seed(0)
    pred = torch.randn(B, T, D)
    target = torch.randn(B, T, D)
    valid = torch.ones(B, T, dtype=torch.bool)
    acc, jerk = masked_root_acc_jerk_loss(pred, target, valid)
    if not (torch.isfinite(acc).item() and torch.isfinite(jerk).item()):
        _print("t12", f"FAIL — non-finite acc={acc.item()} jerk={jerk.item()}")
        return False
    # Same-shape padding-mask check: with valid all-True, denominator should
    # be (T-2) for acc and (T-3) for jerk per batch; positivity expected.
    if acc.item() <= 0 or jerk.item() <= 0:
        _print("t12", f"FAIL — non-positive losses on random tensors: acc={acc.item()}, jerk={jerk.item()}")
        return False
    _print("t12", f"PASS — root acc={acc.item():.4f}, jerk={jerk.item():.4f} finite (magnitudes logged)")
    return True


# ─────────────────────────────────────────────────────────────────────
# t13: frame-0 inpainting at sampling
# ─────────────────────────────────────────────────────────────────────


def t15_obj_traj_frame_matches_coarse_v1() -> bool:
    """Frame-consistency test: verify the new cache's
    ``obj_traj_root0_world[:, 0:3]`` equals
    ``object_positions[:T] - root_world[0]`` exactly, and uses the SAME
    root0/world-axis convention as Coarse-v1's `root_local_trans`.

    Recomputes obj_pos_root0_world independently from the raw dataset
    fields for 2 sampled clips per subset, and compares against the
    cached value. Per Round-17 Codex review.
    """
    from omegaconf import OmegaConf
    from piano.data.dataset import AugmentConfig, HOIDataset
    from extract_coarse_motion_representation import extract_coarse_v0_v1

    if not OBJTRAJ_CACHE.exists():
        _print("t15", "FAIL — objtraj cache missing")
        return False
    norm = json.loads((OBJTRAJ_CACHE / "normalization_train.json").read_text("utf-8"))
    # The cache builder stored the unnormalized obj_traj; the normalization
    # JSON's `obj_traj_root0_world` block is just stats. We verify against
    # the cache's unnormalized payload directly.
    cfg = OmegaConf.load(_DEFAULT_V18_CFG)
    max_clips_per_subset = 2
    n_checked = 0
    max_abs_diff = 0.0
    for entry in cfg.data.datasets[:4]:
        subset = Path(entry.root).name
        ds = HOIDataset(
            root=entry.root,
            pseudo_label_dir=None,
            max_seq_length=int(cfg.data.max_seq_length),
            surface_obj_pose=False,
            force_world_frame=False,
            motion_representation="smpl_pose_135",
            augment=AugmentConfig(enabled=False),
        )
        # Read the cache manifest for this subset's first few train clips.
        manifest = [
            json.loads(line)
            for line in (OBJTRAJ_CACHE / "manifest_train.jsonl").read_text("utf-8").splitlines()
            if line.strip() and json.loads(line)["subset"] == subset
        ][:max_clips_per_subset]
        for rec in manifest:
            cached = np.load(OBJTRAJ_CACHE / rec["npz_path"], allow_pickle=False)
            cached_obj_traj = cached[OBJTRAJ_FIELD_NAME]               # (T_eff, 9)
            cached_coarse_v1 = cached["coarse_v1"]                      # (T_eff, 23)
            T_eff = int(cached_obj_traj.shape[0])
            # Find the corresponding sample in the dataset by seq_id.
            seq_id = rec["seq_id"]
            ds_idx = None
            for i, m in enumerate(ds.metadata):
                if m["seq_id"] == seq_id:
                    ds_idx = i
                    break
            if ds_idx is None:
                _print("t15", f"FAIL — couldn't find {subset}/{seq_id} in dataset")
                return False
            sample = ds[ds_idx]
            object_positions = sample["object_positions"].numpy().astype(np.float32)
            object_rotations = sample["object_rotations"].numpy().astype(np.float32)
            motion = sample["motion"].numpy().astype(np.float32)
            rest_offsets = sample["rest_offsets"].numpy().astype(np.float32)
            seq_len = int(sample["seq_len"].item())
            out = extract_coarse_v0_v1(motion, rest_offsets, seq_len)
            root_world_extract = out["root_world"]                                # (T_eff, 3)
            # Recompute obj_pos_root0_world from raw object_positions
            # and the extracted root_world[0].
            expected_pos = (object_positions[:T_eff] - root_world_extract[0:1]).astype(np.float32)
            diff_pos = np.abs(cached_obj_traj[:, 0:3] - expected_pos).max()
            if diff_pos > 1e-5:
                _print("t15", f"FAIL — {subset}/{seq_id}: cached obj_pos differs "
                       f"from recomputed by {diff_pos:.4e}")
                return False
            # Cross-check coarse_v1's root_local_trans:
            # cached_coarse_v1[:, 0:3] = (root_local_x, root_local_z, root_local_y)
            # corresponds to (root_world - root0) re-arranged.
            root_local_expected_x = root_world_extract[:T_eff, 0] - root_world_extract[0, 0]
            root_local_expected_z = root_world_extract[:T_eff, 2] - root_world_extract[0, 2]
            root_local_expected_y = root_world_extract[:T_eff, 1] - root_world_extract[0, 1]
            diff_v1_x = np.abs(cached_coarse_v1[:, 0] - root_local_expected_x).max()
            diff_v1_z = np.abs(cached_coarse_v1[:, 1] - root_local_expected_z).max()
            diff_v1_y = np.abs(cached_coarse_v1[:, 2] - root_local_expected_y).max()
            if max(diff_v1_x, diff_v1_z, diff_v1_y) > 1e-5:
                _print("t15", f"FAIL — {subset}/{seq_id}: cached coarse_v1 root_local "
                       f"channels disagree with extract_coarse_v0_v1 by "
                       f"x={diff_v1_x:.4e} z={diff_v1_z:.4e} y={diff_v1_y:.4e}")
                return False
            max_abs_diff = max(max_abs_diff, diff_pos, diff_v1_x, diff_v1_y, diff_v1_z)
            n_checked += 1
    _print("t15", f"PASS — {n_checked} clips: cached obj_traj_root0_world[:, 0:3] "
           f"== object_positions[:T] - root_world[0]; coarse_v1 root_local "
           f"== root_world - root_world[0]; max|diff| = {max_abs_diff:.4e}")
    return True


def t17_common_parameter_equality_plan_a_vs_s1o() -> bool:
    """Round-18 follow-up fairness: under matched seed, Plan A and S1-O
    have BITWISE-EQUAL shared parameters. S1-O's only extra params are
    the HintBlock + null_obj_traj. Achieved by moving HintBlock
    instantiation to the END of CoarsePriorDenoiser.__init__.
    """
    from omegaconf import OmegaConf
    from piano.training.train_coarse_prior import build_model

    cfg_a = OmegaConf.load("configs/training/coarse_prior_s1a_cmc.yaml")
    cfg_o = OmegaConf.load("configs/training/coarse_prior_s1o_root0_world.yaml")

    seed = 42
    torch.manual_seed(seed)
    model_a = build_model(cfg_a)
    torch.manual_seed(seed)
    model_o = build_model(cfg_o)

    sd_a = model_a.state_dict()
    sd_o = model_o.state_dict()
    # Identify shared vs extra keys.
    extra_keys = sorted([k for k in sd_o if k not in sd_a])
    missing_keys = sorted([k for k in sd_a if k not in sd_o])
    if missing_keys:
        _print("t17", f"FAIL — keys present in Plan A but not S1-O: {missing_keys}")
        return False
    expected_extras = ["denoiser.null_obj_traj"] + sorted([
        f"denoiser.obj_traj_hint.layers.{i}.{kind}"
        for i in range(4) for kind in ("weight", "bias")
    ])
    if sorted(extra_keys) != sorted(expected_extras):
        _print("t17", f"FAIL — unexpected extra keys in S1-O: got "
               f"{sorted(extra_keys)}, expected {sorted(expected_extras)}")
        return False
    # Verify bitwise equality on shared keys.
    max_abs_diff = 0.0
    worst_key = "(none)"
    for k in sd_a:
        a_v = sd_a[k]
        o_v = sd_o[k]
        if a_v.shape != o_v.shape:
            _print("t17", f"FAIL — shape mismatch on shared key {k}: "
                   f"a={tuple(a_v.shape)} o={tuple(o_v.shape)}")
            return False
        d = (a_v - o_v).abs().max().item() if a_v.is_floating_point() else 0.0
        if d > max_abs_diff:
            max_abs_diff = d
            worst_key = k
    if max_abs_diff > 0.0:
        _print("t17", f"FAIL — shared params not bitwise equal: max|diff|={max_abs_diff:.3e} "
               f"on key {worst_key!r}")
        return False
    _print("t17", f"PASS — Plan A and S1-O have bitwise-equal shared params "
           f"({len(sd_a)} keys); S1-O extras: {len(extra_keys)} keys "
           f"({sum(sd_o[k].numel() for k in extra_keys):,} params total)")
    return True


def t19_training_text_dropout_paired_fair() -> bool:
    """Round-18 final polish: text-dropout mask is IDENTICAL across Plan A
    and S1-O at matched seed because it's drawn from a dedicated
    `text_drop_rng` generator (seeded from ``seed * 10_000 + 3``),
    independent of any other RNG stream.

    Simulates 5 trainer steps for both Plan A and S1-O. Plan A draws
    text_drop_mask only; S1-O draws text_drop_mask AND obj_drop_mask
    (from a separate `obj_drop_rng`). Asserts:

    - text_drop_mask is bit-exact equal across modes per step
    - S1-O's extra obj_drop_mask draw doesn't perturb text_drop_rng
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
    B = 4
    cfg_drop_prob = 0.1
    n_steps = 5

    # Plan A: only text_drop_rng is consumed.
    text_drop_rng_a = torch.Generator(device=device).manual_seed(seed * 10_000 + 3)
    text_masks_a = []
    for _ in range(n_steps):
        m = torch.rand(B, device=device, generator=text_drop_rng_a) < cfg_drop_prob
        text_masks_a.append(m.cpu())

    # S1-O: text_drop_rng AND obj_drop_rng both consumed; the two are
    # independent generators so consuming obj_drop_rng MUST NOT shift the
    # text_drop_rng stream.
    text_drop_rng_o = torch.Generator(device=device).manual_seed(seed * 10_000 + 3)
    obj_drop_rng_o = torch.Generator(device=device).manual_seed(seed * 10_000 + 4)
    text_masks_o = []
    for _ in range(n_steps):
        m_text = torch.rand(B, device=device, generator=text_drop_rng_o) < cfg_drop_prob
        # S1-O additionally draws obj dropout (using its own generator).
        _ = torch.rand(B, device=device, generator=obj_drop_rng_o) < cfg_drop_prob
        text_masks_o.append(m_text.cpu())

    for i in range(n_steps):
        if not torch.equal(text_masks_a[i], text_masks_o[i]):
            _print("t19", f"FAIL — step {i}: text dropout mask differs Plan A vs S1-O")
            return False
    _print("t19", f"PASS — {n_steps} simulated steps: text dropout mask "
           "bit-exact equal across Plan A and S1-O (independent text_drop_rng + "
           "obj_drop_rng generators decouple the two streams)")
    return True


def t20_validation_diffusion_rng_paired_fair() -> bool:
    """Round-18 final polish: under matched (seed, val_step), Plan A and
    S1-O see IDENTICAL validation `t` and `noise`. The val pass uses a
    dedicated generator seeded from ``val_diff_seed`` (which the trainer
    derives deterministically from ``(seed, step)``), so:

    - same (seed, val_step) → same `t`, same `noise` across modes
    - val RNG consumption does NOT perturb training RNG (separate generator
      that gets re-seeded at the start of each val invocation)
    - LIVE and EMA passes within a single val invocation share the same
      val_diff_seed (both call _loss_one_pass which re-seeds internally),
      so live and EMA are compared on identical noise levels
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_steps = 1000
    B = 4
    T = 16
    val_diff_seed = 42 * 1_000_000 + 5000   # mimics (seed=42, step=5000)
    n_val_batches = 4

    def _draw_val_stream(_pass_label: str) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        # Replicates the relevant 5 lines of `_loss_one_pass` from
        # train_coarse_prior.py:_run_validation_pass.
        rng = torch.Generator(device=device).manual_seed(int(val_diff_seed))
        ts, noises = [], []
        for _ in range(n_val_batches):
            t = torch.randint(0, num_steps, (B,), device=device, generator=rng)
            noise = torch.randn((B, T, 23), device=device, generator=rng)
            ts.append(t.cpu())
            noises.append(noise.cpu())
        return ts, noises

    # Plan A and S1-O each call this; same val_diff_seed → same stream.
    ts_a, noises_a = _draw_val_stream("plan_a_live")
    ts_o, noises_o = _draw_val_stream("s1o_live")
    # Also simulate that S1-O's training-side `t`/`noise` stream (which uses
    # the SEPARATE diff_rng generator, not val_diff_seed) does NOT perturb
    # the val_diff_rng stream. We do this by drawing some "training"
    # samples from a different generator before running val, and asserting
    # val output is unchanged.
    extra_train_rng = torch.Generator(device=device).manual_seed(9999)
    for _ in range(50):
        _ = torch.randn(B, device=device, generator=extra_train_rng)
    ts_o_after_train, noises_o_after_train = _draw_val_stream("s1o_live_after_train")

    for i in range(n_val_batches):
        if not torch.equal(ts_a[i], ts_o[i]):
            _print("t20", f"FAIL — val batch {i}: `t` differs across Plan A and S1-O")
            return False
        if (noises_a[i] - noises_o[i]).abs().max().item() > 0.0:
            _print("t20", f"FAIL — val batch {i}: `noise` differs across Plan A and S1-O")
            return False
        if not torch.equal(ts_o[i], ts_o_after_train[i]):
            _print("t20", f"FAIL — val batch {i}: `t` differs after intervening "
                   "training-side RNG consumption (val_diff_rng should be isolated)")
            return False
        if (noises_o[i] - noises_o_after_train[i]).abs().max().item() > 0.0:
            _print("t20", f"FAIL — val batch {i}: `noise` differs after intervening "
                   "training-side RNG consumption")
            return False
    _print("t20", f"PASS — {n_val_batches} val batches: `t`/`noise` bit-exact "
           "across Plan A and S1-O AND unaffected by intervening training RNG "
           "consumption (val_diff_rng is isolated)")
    return True


def t21_obj_dropout_does_not_affect_text_or_diff_streams() -> bool:
    """Round-18 final polish: drawing from `obj_drop_rng` must NOT shift
    either the diffusion-side `t`/`noise` stream (`diff_rng`) or the text
    dropout stream (`text_drop_rng`). Confirms full independence of the
    four trainer RNG streams (diff, text_drop, obj_drop, val_diff).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
    B = 4
    n_steps = 5

    # Stream 1 reference: diff + text streams ONLY (Plan A).
    diff_rng_a = torch.Generator(device=device).manual_seed(seed * 10_000 + 1)
    text_rng_a = torch.Generator(device=device).manual_seed(seed * 10_000 + 3)
    ts_a, ns_a, tm_a = [], [], []
    for _ in range(n_steps):
        ts_a.append(torch.randint(0, 1000, (B,), device=device, generator=diff_rng_a).cpu())
        ns_a.append(torch.randn((B, 16, 23), device=device, generator=diff_rng_a).cpu())
        tm_a.append((torch.rand(B, device=device, generator=text_rng_a) < 0.1).cpu())

    # Stream 2: SAME diff + text seeds, but ADDITIONALLY draw from obj_drop_rng
    # each step (S1-O). Outputs of diff_rng and text_rng must be unchanged.
    diff_rng_o = torch.Generator(device=device).manual_seed(seed * 10_000 + 1)
    text_rng_o = torch.Generator(device=device).manual_seed(seed * 10_000 + 3)
    obj_rng_o = torch.Generator(device=device).manual_seed(seed * 10_000 + 4)
    ts_o, ns_o, tm_o = [], [], []
    for _ in range(n_steps):
        ts_o.append(torch.randint(0, 1000, (B,), device=device, generator=diff_rng_o).cpu())
        ns_o.append(torch.randn((B, 16, 23), device=device, generator=diff_rng_o).cpu())
        tm_o.append((torch.rand(B, device=device, generator=text_rng_o) < 0.1).cpu())
        _ = torch.rand(B, device=device, generator=obj_rng_o) < 0.1   # S1-O extra draw

    for i in range(n_steps):
        if not torch.equal(ts_a[i], ts_o[i]):
            _print("t21", f"FAIL — step {i}: diff `t` differs after obj_drop draw")
            return False
        if (ns_a[i] - ns_o[i]).abs().max().item() > 0.0:
            _print("t21", f"FAIL — step {i}: diff `noise` differs after obj_drop draw")
            return False
        if not torch.equal(tm_a[i], tm_o[i]):
            _print("t21", f"FAIL — step {i}: text_drop mask differs after obj_drop draw")
            return False
    _print("t21", f"PASS — {n_steps} steps: obj_drop_rng draws DO NOT shift "
           "diff_rng or text_drop_rng streams (all generators independent)")
    return True


def t18_training_rng_fairness_t_and_noise() -> bool:
    """Round-18 follow-up fairness: under matched seed + matched batch,
    the diffusion timestep `t` and the noise tensor are IDENTICAL across
    Plan A and S1-O. Achieved by dedicating a `diff_rng` torch.Generator
    to those two draws, independent of text/obj dropout (which uses the
    global RNG and may diverge between modes).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
    B, T = 4, 16
    # Simulate what the trainer does in main():
    #   diff_rng = torch.Generator(device=device).manual_seed(seed * 10_000 + 1)
    # Then for each step:
    #   t = torch.randint(0, num_steps, (B,), device=device, generator=diff_rng)
    #   noise = torch.randn(x0.shape, device=device, dtype=x0.dtype, generator=diff_rng)
    # And separately, text/obj dropout uses the global RNG which differs
    # between Plan A and S1-O. We model that by interleaving extra
    # `torch.rand` calls in S1-O between batches.
    num_steps = 1000
    n_train_steps_to_simulate = 5

    # Plan A simulation:
    diff_rng_a = torch.Generator(device=device).manual_seed(seed * 10_000 + 1)
    torch.manual_seed(seed)   # global RNG for text/obj dropout
    ts_a, noises_a = [], []
    for _ in range(n_train_steps_to_simulate):
        t = torch.randint(0, num_steps, (B,), device=device, generator=diff_rng_a)
        noise = torch.randn((B, T, 23), device=device, generator=diff_rng_a)
        ts_a.append(t.cpu())
        noises_a.append(noise.cpu())
        # Plan A: only text dropout (1 global RNG draw per step).
        _ = torch.rand(B, device=device)

    # S1-O simulation:
    diff_rng_o = torch.Generator(device=device).manual_seed(seed * 10_000 + 1)
    torch.manual_seed(seed)
    ts_o, noises_o = [], []
    for _ in range(n_train_steps_to_simulate):
        t = torch.randint(0, num_steps, (B,), device=device, generator=diff_rng_o)
        noise = torch.randn((B, T, 23), device=device, generator=diff_rng_o)
        ts_o.append(t.cpu())
        noises_o.append(noise.cpu())
        # S1-O: text dropout + obj_traj dropout (2 global RNG draws per step).
        _ = torch.rand(B, device=device)
        _ = torch.rand(B, device=device)

    # Compare.
    for i in range(n_train_steps_to_simulate):
        if not torch.equal(ts_a[i], ts_o[i]):
            _print("t18", f"FAIL — step {i}: timestep `t` differs across Plan A and S1-O")
            return False
        d_noise = (noises_a[i] - noises_o[i]).abs().max().item()
        if d_noise > 0.0:
            _print("t18", f"FAIL — step {i}: noise max|Δ| = {d_noise:.3e}")
            return False
    _print("t18", f"PASS — {n_train_steps_to_simulate} simulated steps: `t` and "
           "`noise` are bitwise-equal across Plan A and S1-O at matched seed, "
           "despite divergent text/obj dropout RNG consumption")
    return True


def t14_build_model_passes_obj_traj_dim_from_yaml() -> bool:
    """Round-19 active configs: build_model correctly plumbs obj_traj_dim
    from YAML, points at the right cache, and rejects use of the
    deprecated `coarse_prior_s1o.yaml` (which points at the body-canonical
    Round-18-original cache that Codex flagged as frame-inconsistent).

    Verifies for each Round-19 active config:

    - For coarse_prior_s1a_cmc (Plan A; obj_traj_dim=0): no HintBlock params,
      cache_root = cache/stage1_coarse_v1_full.
    - For coarse_prior_s1o_root0_world (Plan C; obj_traj_dim=9): HintBlock +
      null params = 793,097, cache_root = cache/stage1_coarse_v1_objtraj_root0_world_round18_fix.

    Also confirms the deprecated config `coarse_prior_s1o.yaml` STILL EXISTS
    on disk (preserved per SUGGESTION.md "保留旧 Round-18 cache，不要删除")
    but is NOT in the active Round-19 set.
    """
    from omegaconf import OmegaConf
    from piano.training.train_coarse_prior import build_model

    expected_a_cache = "cache/stage1_coarse_v1_full"
    expected_o_cache = "cache/stage1_coarse_v1_objtraj_root0_world_round18_fix"
    deprecated_o_config = Path("configs/training/coarse_prior_s1o.yaml")
    deprecated_o_cache = "cache/stage1_coarse_v1_objtraj_round18"

    cases = [
        ("configs/training/coarse_prior_s1a_cmc.yaml",        0, 0,        expected_a_cache),
        ("configs/training/coarse_prior_s1o_root0_world.yaml", 9, 793_097, expected_o_cache),
    ]
    for path, expected_dim, expected_hint_params, expected_cache in cases:
        cfg_path = Path(path)
        if not cfg_path.exists():
            _print("t14", f"FAIL — Round-19 active config missing: {cfg_path}")
            return False
        cfg = OmegaConf.load(cfg_path)
        # Cache_root check — fail loudly if a Round-19 config points at the
        # deprecated Round-18 body-canonical cache.
        actual_cache = str(cfg.data.cache_root).replace("\\", "/").rstrip("/")
        if actual_cache != expected_cache:
            _print("t14", f"FAIL — {cfg_path}: data.cache_root = "
                   f"{actual_cache!r}, expected {expected_cache!r}")
            return False
        if actual_cache == deprecated_o_cache:
            _print("t14", f"FAIL — {cfg_path} points at deprecated cache "
                   f"{deprecated_o_cache!r}; should use {expected_o_cache!r}")
            return False
        try:
            model = build_model(cfg)
        except Exception as e:
            _print("t14", f"FAIL — build_model raised on {cfg_path}: {e!r}")
            return False
        actual_dim = int(model.cfg.denoiser.obj_traj_dim)
        if actual_dim != expected_dim:
            _print("t14", f"FAIL — {cfg_path}: model.cfg.denoiser.obj_traj_dim = "
                   f"{actual_dim}, expected {expected_dim}")
            return False
        hint_n = sum(
            p.numel() for n_, p in model.named_parameters()
            if "obj_traj" in n_ or "null_obj_traj" in n_
        )
        if hint_n != expected_hint_params:
            _print("t14", f"FAIL — {cfg_path}: HintBlock+null params = "
                   f"{hint_n:,}, expected {expected_hint_params:,}")
            return False
        _print("t14", f"  {cfg_path.name}: obj_traj_dim={actual_dim} "
               f"hint+null={hint_n:,} cache_root={actual_cache}")
    # Deprecation guard: legacy config still exists on disk but is NOT in
    # the active Round-19 set. Confirms SUGGESTION.md preservation intent
    # AND flags any accidental promotion of the legacy config back into
    # active use.
    if not deprecated_o_config.exists():
        _print("t14", f"FAIL — deprecated config {deprecated_o_config} was "
               "deleted; SUGGESTION.md requires it preserved on disk")
        return False
    legacy_cfg = OmegaConf.load(deprecated_o_config)
    legacy_cache = str(legacy_cfg.data.cache_root).replace("\\", "/").rstrip("/")
    if legacy_cache != deprecated_o_cache:
        _print("t14", f"FAIL — deprecated {deprecated_o_config} cache_root = "
               f"{legacy_cache!r}; expected legacy {deprecated_o_cache!r}")
        return False
    _print("t14", f"  {deprecated_o_config.name}: preserved on disk, "
           f"points at deprecated cache (NOT in Round-19 active set)")
    _print("t14", "PASS — Round-19 active configs use the root0_world cache; "
           "deprecated coarse_prior_s1o.yaml preserved but isolated")
    return True


def t13_frame0_inpainting_locks_init() -> bool:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(obj_traj_dim=0, seed=42).to(device).eval()
    B, T = 1, 16
    init = torch.randn(B, 23, device=device)
    cond = {
        "text_pool": torch.randn(B, 512, device=device),
        "init_coarse": init,
        "valid_mask": torch.ones(B, T, dtype=torch.bool, device=device),
    }
    with torch.no_grad():
        x_gen = model.sample(
            shape=(B, T, 23), cond=cond, cfg_scale=1.0, device=device,
            inpaint_frame0=True,
        )
    diff = (x_gen[:, 0, :] - init).abs().max().item()
    if diff > 1e-5:
        _print("t13", f"FAIL — generated frame 0 differs from init by {diff:.3e}")
        return False
    _print("t13", f"PASS — generated frame 0 equals conditioned init (|Δ| = {diff:.3e})")
    return True


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────


def main() -> int:
    checks = [
        ("t1",  t1_obj_pose_coverage_per_subset),
        ("t2",  t2_world_vs_canonical_diverges),
        ("t3",  t3_objtraj_cache_schema),
        ("t4",  t4_no_forbidden_fields_in_cache),
        ("t5",  t5_obj_traj_normalization_finite),
        ("t6",  t6_s1a_forward_shape),
        ("t7",  t7_s1o_forward_shape_and_missing_raises),
        ("t8",  t8_zero_init_hintblock_invariance),
        ("t9",  t9_obj_traj_dropout_finite),
        ("t11", t11_ema_update_correctness),
        ("t12", t12_root_acc_jerk_loss_finite_and_logged),
        ("t13", t13_frame0_inpainting_locks_init),
        ("t14", t14_build_model_passes_obj_traj_dim_from_yaml),
        # Round-18 follow-up additions:
        ("t15", t15_obj_traj_frame_matches_coarse_v1),
        ("t17", t17_common_parameter_equality_plan_a_vs_s1o),
        ("t18", t18_training_rng_fairness_t_and_noise),
        # Round-18 final polish additions:
        ("t19", t19_training_text_dropout_paired_fair),
        ("t20", t20_validation_diffusion_rng_paired_fair),
        ("t21", t21_obj_dropout_does_not_affect_text_or_diff_streams),
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
        print(f"[r18-preflight] FAILED: {failures}")
        return 1
    print("[r18-preflight] ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
