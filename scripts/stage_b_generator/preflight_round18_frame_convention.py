"""Round-18 Step-1 preflight: object-pose coverage + frame-convention comparison.

Verifies, BEFORE any cache build or model code change:

1. ``object_positions`` and ``object_rotations`` exist in clip npz files
   across every PIANO subset:
   chairs, imhd, neuraldome, omomo_correct_v2.
2. With ``surface_obj_pose=True, force_world_frame=False`` (the Round-18
   choice), every loaded clip exposes finite ``obj_com_canonical``
   ``(T, 3)`` and ``obj_rot6d_canonical`` ``(T, 6)`` aligned with
   ``seq_len``.
3. ``force_world_frame=True`` (v18 active) and ``force_world_frame=False``
   (Round-18) produce MATERIALLY DIFFERENT obj_com / obj_rot6d on a few
   sampled clips — proof that v18's "canonical" object pose is effectively
   world-frame (R_y=0, T_xz=[0,0]) while Round-18 uses true body-canonical.

This script is read-only: no cache is written, no checkpoints touched.
It prints a pass/fail summary and writes a small JSON record under
``analyses/round18_preflight/`` so the Round-18 implementation report
can cite specific numbers.

Usage::

    $env:PYTHONIOENCODING="utf-8"
    conda run -n piano python scripts/stage_b_generator/preflight_round18_frame_convention.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from omegaconf import OmegaConf

from piano.data.dataset import AugmentConfig, HOIDataset


SUBSETS = ("chairs", "imhd", "neuraldome", "omomo_correct_v2")
# Round-18-fix-server: dataset roots are read from the v18 config below,
# not from this hard-coded pattern (which is the LOCAL-Windows path used
# for documentation only). On the server the env var PIANO_V18_CFG
# overrides V18_CONFIG to point at the server-paths variant
# (`anchordiff_v18_a1_FULL_DATA_local.yaml`).
PIANO_ROOT_PATTERN = "E:/Project/Datasets/InterAct/piano_official_process_4/{subset}"
V18_CONFIG = Path(
    os.environ.get(
        "PIANO_V18_CFG",
        "configs/training/anchordiff_v18_a1_FULL_DATA.yaml",
    )
)
OUT_DIR = Path("analyses/round18_preflight")
N_CLIPS_PER_SUBSET = 3        # small sample; preflight only
N_FRAMES_PER_CLIP = 5         # how many frame indices to compare in the world-vs-canonical diff


def _log(msg: str) -> None:
    print(f"[preflight-r18] {msg}", flush=True)


def _build_subset_dataset(
    cfg, subset_root: Path, *, force_world_frame: bool, surface_obj_pose: bool,
) -> HOIDataset:
    """Identical knob set to build_stage1_coarse_v1_cache.py but with the
    `force_world_frame` flag exposed so we can flip it.
    """
    return HOIDataset(
        root=subset_root,
        pseudo_label_dir=None,
        max_seq_length=int(cfg.data.max_seq_length),
        subject_id_filter=None,
        subsample_n_per_object=None,
        subsample_seed=int(cfg.data.get("subsample_seed", 42)),
        support_collapse_hand_support=bool(
            cfg.data.get("support_collapse_hand_support", True),
        ),
        surface_obj_pose=surface_obj_pose,
        force_world_frame=force_world_frame,
        motion_representation="smpl_pose_135",
        augment=AugmentConfig(enabled=False),
    )


def _subset_root_from_cfg(cfg, subset: str) -> Path:
    """Resolve subset root from the loaded v18 config's `data.datasets`
    list (NOT from the hard-coded Windows pattern). Falls back to the
    pattern only if the subset isn't found in the cfg (which would
    indicate a misconfigured server v18 config)."""
    for entry in cfg.data.datasets:
        if Path(entry.root).name == subset:
            return Path(entry.root)
    # Fallback — log a warning so the operator notices.
    fallback = Path(PIANO_ROOT_PATTERN.format(subset=subset))
    _log(f"WARNING — subset {subset!r} not found in cfg.data.datasets; "
         f"falling back to hard-coded pattern {fallback}")
    return fallback


def check_subset(
    cfg, subset: str,
) -> dict[str, Any]:
    """Sub-test for one subset. Returns a dict of {pass, n_total, n_with_obj,
    canonical_shapes_ok, finite_ok, sample_diffs}."""
    subset_root = _subset_root_from_cfg(cfg, subset)
    _log(f"--- {subset} (root={subset_root}) ---")

    result: dict[str, Any] = {
        "subset": subset,
        "subset_root": str(subset_root),
        "n_total": 0,
        "n_with_obj_pose": 0,
        "canonical_shape_ok": True,
        "finite_ok": True,
        "sample_diffs": [],          # per-clip comparison records (world vs canonical)
        "errors": [],
        "pass": True,
    }

    if not subset_root.exists():
        msg = f"FAIL — subset root missing: {subset_root}"
        _log(msg)
        result["errors"].append(msg)
        result["pass"] = False
        return result

    # Step A: surface_obj_pose=True, force_world_frame=False (Round-18 choice).
    try:
        ds_canonical = _build_subset_dataset(
            cfg, subset_root, force_world_frame=False, surface_obj_pose=True,
        )
    except Exception as e:
        msg = f"FAIL — could not build HOIDataset force_world_frame=False: {e!r}"
        _log(msg)
        result["errors"].append(msg)
        result["pass"] = False
        return result

    n_total = len(ds_canonical)
    result["n_total"] = int(n_total)
    if n_total == 0:
        msg = "FAIL — dataset has 0 clips"
        _log(msg)
        result["errors"].append(msg)
        result["pass"] = False
        return result

    # Step B: same root with force_world_frame=True (v18 active path) for the
    # comparison samples.
    try:
        ds_world = _build_subset_dataset(
            cfg, subset_root, force_world_frame=True, surface_obj_pose=True,
        )
    except Exception as e:
        msg = f"FAIL — could not build HOIDataset force_world_frame=True: {e!r}"
        _log(msg)
        result["errors"].append(msg)
        result["pass"] = False
        return result

    # Sample N clips. Iterate up to a few extra in case some have no obj pose.
    sample_attempts = min(N_CLIPS_PER_SUBSET * 4, n_total)
    samples_taken = 0
    for idx in range(sample_attempts):
        if samples_taken >= N_CLIPS_PER_SUBSET:
            break
        try:
            s_canon = ds_canonical[idx]
            s_world = ds_world[idx]
        except Exception as e:
            result["errors"].append(f"clip {idx}: __getitem__ raised {e!r}")
            continue

        seq_id = str(s_canon.get("seq_id", f"idx_{idx}"))
        seq_len = int(s_canon["seq_len"].item())

        has_obj_canon = ("obj_com_canonical" in s_canon) and ("obj_rot6d_canonical" in s_canon)
        has_obj_world = ("obj_com_canonical" in s_world) and ("obj_rot6d_canonical" in s_world)
        if not has_obj_canon:
            # Clip lacks object pose fields entirely — record and skip.
            result["sample_diffs"].append({
                "idx": int(idx), "seq_id": seq_id, "seq_len": seq_len,
                "skipped_reason": "no obj_com_canonical / obj_rot6d_canonical returned",
            })
            continue
        result["n_with_obj_pose"] += 1

        obj_com_canon = s_canon["obj_com_canonical"].numpy()
        obj_rot6d_canon = s_canon["obj_rot6d_canonical"].numpy()
        obj_com_world = s_world["obj_com_canonical"].numpy()
        obj_rot6d_world = s_world["obj_rot6d_canonical"].numpy()

        # Shape + dtype + finiteness check on the "canonical" (body-frame) one.
        if obj_com_canon.shape[1] != 3 or obj_rot6d_canon.shape[1] != 6:
            result["canonical_shape_ok"] = False
            result["errors"].append(
                f"clip {idx} {seq_id}: bad canonical shapes "
                f"com={obj_com_canon.shape} rot6d={obj_rot6d_canon.shape}"
            )
        if not (np.isfinite(obj_com_canon[:seq_len]).all() and np.isfinite(obj_rot6d_canon[:seq_len]).all()):
            result["finite_ok"] = False
            result["errors"].append(f"clip {idx} {seq_id}: non-finite canonical values in valid range")

        # Compute valid-range max-abs difference between world-frame and body-canonical
        # obj_com to confirm they're genuinely different.
        valid = min(seq_len, obj_com_canon.shape[0], obj_com_world.shape[0])
        if valid > 0:
            diff_com = np.abs(obj_com_canon[:valid] - obj_com_world[:valid])
            diff_rot6d = np.abs(obj_rot6d_canon[:valid] - obj_rot6d_world[:valid])
            sample_record = {
                "idx": int(idx),
                "seq_id": seq_id,
                "seq_len": seq_len,
                "world_first5_com": obj_com_world[: min(5, valid)].tolist(),
                "canonical_first5_com": obj_com_canon[: min(5, valid)].tolist(),
                "max_abs_diff_com": float(diff_com.max()),
                "mean_abs_diff_com": float(diff_com.mean()),
                "max_abs_diff_rot6d": float(diff_rot6d.max()),
                "mean_abs_diff_rot6d": float(diff_rot6d.mean()),
            }
            result["sample_diffs"].append(sample_record)
            samples_taken += 1

    # Aggregate verdict per subset.
    if not result["canonical_shape_ok"]:
        result["pass"] = False
    if not result["finite_ok"]:
        result["pass"] = False
    if result["n_with_obj_pose"] == 0:
        result["pass"] = False
        result["errors"].append("no clip in the sampled range exposed obj_com_canonical / obj_rot6d_canonical")
    # We REQUIRE that world vs canonical actually differ. If they're identical
    # for every sampled clip, either force_world_frame is broken or
    # surface_obj_pose isn't producing canonical-frame values.
    if any(
        rec.get("max_abs_diff_com", 0.0) > 1e-4 for rec in result["sample_diffs"]
    ):
        pass  # Good — at least one clip has a non-trivial difference.
    elif result["sample_diffs"]:
        result["pass"] = False
        result["errors"].append(
            "world vs canonical agreement was perfect on all sampled clips; "
            "expected materially different values when force_world_frame is False"
        )

    _log(
        f"{subset}: n_total={result['n_total']}, "
        f"n_with_obj_pose={result['n_with_obj_pose']}/{samples_taken}, "
        f"canonical_shape_ok={result['canonical_shape_ok']}, "
        f"finite_ok={result['finite_ok']}, "
        f"pass={result['pass']}"
    )
    if result["sample_diffs"]:
        for rec in result["sample_diffs"][:2]:
            if "max_abs_diff_com" in rec:
                _log(
                    f"  sample {rec['idx']} {rec['seq_id']}: "
                    f"max|world-canon|: com={rec['max_abs_diff_com']:.4f}, "
                    f"rot6d={rec['max_abs_diff_rot6d']:.4f}"
                )
    if result["errors"]:
        for err in result["errors"]:
            _log(f"  ERR: {err}")
    return result


def main() -> int:
    if not V18_CONFIG.exists():
        print(f"FAIL — config missing: {V18_CONFIG}")
        return 1
    cfg = OmegaConf.load(V18_CONFIG)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    per_subset = []
    overall_pass = True
    for subset in SUBSETS:
        rec = check_subset(cfg, subset)
        per_subset.append(rec)
        if not rec["pass"]:
            overall_pass = False

    payload = {
        "round": "round18",
        "step": 1,
        "config_source": str(V18_CONFIG),
        "subsets": list(SUBSETS),
        "n_clips_per_subset_sampled": N_CLIPS_PER_SUBSET,
        "results": per_subset,
        "overall_pass": overall_pass,
        "elapsed_seconds": float(time.time() - t0),
    }
    out_path = OUT_DIR / "frame_convention_preflight.json"
    out_path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    _log(f"wrote {out_path}")

    print()
    print("=" * 70)
    if overall_pass:
        print("[preflight-r18-step1] PASS — object pose coverage + body-canonical "
              "vs force-world-frame divergence confirmed across all subsets")
        return 0
    else:
        print("[preflight-r18-step1] FAIL — see errors above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
