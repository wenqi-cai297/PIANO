"""Round-18 Step-2: build the Stage-1 Coarse-v1 + obj_traj cache.

Forked from ``build_stage1_coarse_v1_cache.py`` with three changes:

1. ``surface_obj_pose=True`` (matches old builder) AND
   ``force_world_frame=False`` (**different from the v18 active config**)
   so the dataset returns true body-canonical obj pose, not world-frame.
2. Adds ``obj_traj_canonical = concat(obj_com_canonical,
   obj_rot6d_canonical)`` (T, 9) per clip into the npz payload.
3. Computes train-only z-score stats for ``obj_traj_canonical`` alongside
   the existing Coarse-v1 stats; saves both under ``normalization_train.json``.

Cache contract (per Round-17 §8):

    cache/stage1_coarse_v1_objtraj_round18/
      manifest_train.jsonl
      manifest_val.jsonl
      clips/<subset>/<safe_seq_id>.npz       # only: coarse_v1, init_coarse_v1, obj_traj_canonical
      normalization_train.json               # adds obj_traj_canonical stats
      text_embeddings_clip_vit_b32.npz       # SHARED with the Round-14 cache, not rebuilt here
      text_embeddings_index.json
      README_cache_contract.md
      build_summary.json

NOTE on CLIP text cache: the Round-14 cache at
``cache/stage1_coarse_v1_full/text_embeddings_*`` already covers the
full text manifest. To avoid duplicate work this builder COPIES (or
hard-links) those two files into the new cache root if they exist.
Otherwise it builds them in-place.

Hard constraints (Round-18):

- v18 untouched. No modification to v18 configs or ckpts.
- Stage-2 untouched.
- Official dataset directories not written into.
- No object_pc, no z_int / contact / phase / support, no plan_*, no
  pseudo-labels, no hand/foot fields stored in the new cache.
- Raw ``object_positions`` / ``object_rotations`` NOT stored (only the
  body-canonical 9-dim derived field).

Usage::

    $env:PYTHONIOENCODING="utf-8"
    conda run -n piano python scripts/stage_b_generator/build_stage1_coarse_v1_objtraj_cache.py \
        --output-root cache/stage1_coarse_v1_objtraj_round18 \
        --max-per-subset -1 \
        --cache-version round18_2026-05-23_objtraj
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf


# Round-18-fix-server: v18 config path; overridable via PIANO_V18_CFG env var.
_DEFAULT_V18_CFG = os.environ.get(
    "PIANO_V18_CFG",
    "configs/training/anchordiff_v18_a1_FULL_DATA.yaml",
)

from piano.data.dataset import (
    AugmentConfig, HOIDataset, build_subject_split, extract_subject_id,
)
from piano.utils.io_utils import load_json

from extract_coarse_motion_representation import (
    COARSE_V1_DIM, COARSE_V0_NAMES, COARSE_V1_EXTRA_NAMES,
    extract_coarse_v0_v1,
)


SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
DEFAULT_FPS = 20.0
OBJ_TRAJ_DIM = 9     # obj_com (3) + obj_rot6d (6) — body-canonical


def _safe_filename(name: str) -> str:
    return SAFE_NAME_RE.sub("_", name)[:96]


def _resolve_subject_splits(cfg) -> dict[str, set] | None:
    subj_cfg = cfg.data.get("subject_split", None)
    if subj_cfg is None or not subj_cfg.get("enabled", False):
        return None
    keys: set[tuple[str, str]] = set()
    for entry in cfg.data.datasets:
        meta_path = Path(entry.root) / "metadata_clean.json"
        if not meta_path.exists():
            meta_path = Path(entry.root) / "metadata.json"
        for m in load_json(meta_path):
            sid = extract_subject_id(Path(entry.root).name, m.get("seq_id", ""))
            if sid is not None:
                keys.add((Path(entry.root).name, sid))
    return build_subject_split(
        sorted(keys),
        train_pct=int(subj_cfg.train_pct),
        val_pct=int(subj_cfg.val_pct),
        seed=int(subj_cfg.seed),
    )


def _build_subset_dataset(cfg, subset_entry) -> HOIDataset:
    """Build a single-subset HOIDataset with:

    - ``surface_obj_pose=True``   (returns obj_com_canonical / obj_rot6d_canonical)
    - ``force_world_frame=False`` (**Round-18 choice**: true body-canonical;
      DIFFERS from v18 active config which sets this True).
    - ``motion_representation="smpl_pose_135"`` (plan-free)
    - ``pseudo_label_dir=None`` (no z_int / contact / phase / support).
    - ``augment=AugmentConfig(enabled=False)``.
    """
    return HOIDataset(
        root=subset_entry.root,
        pseudo_label_dir=None,
        max_seq_length=int(cfg.data.max_seq_length),
        subject_id_filter=None,
        subsample_n_per_object=None,
        subsample_seed=int(cfg.data.get("subsample_seed", 42)),
        support_collapse_hand_support=bool(
            cfg.data.get("support_collapse_hand_support", True),
        ),
        surface_obj_pose=True,            # <- need obj_com_canonical / obj_rot6d_canonical
        force_world_frame=False,          # <- Round-18 chooses true body-canonical
        motion_representation="smpl_pose_135",
        augment=AugmentConfig(enabled=False),
    )


def _split_for_clip(splits: dict[str, set] | None, subset: str, seq_id: str) -> str:
    if splits is None:
        return "train"
    sid = extract_subject_id(subset, seq_id)
    if sid is None:
        return "train"
    key = f"{subset}/{sid}"
    if key in splits.get("train", set()):
        return "train"
    if key in splits.get("val", set()):
        return "val"
    return "skip"


def _z_score_stats(arr: np.ndarray, eps: float) -> dict[str, list[float] | float | int]:
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    std_safe = np.where(std < eps, eps, std)
    return {
        "mean": mean.astype(np.float64).tolist(),
        "std": std.astype(np.float64).tolist(),
        "std_eps": float(eps),
        "std_clamped": std_safe.astype(np.float64).tolist(),
        "n_frames": int(arr.shape[0]),
    }


def _try_copy_text_cache(src_root: Path, dst_root: Path) -> bool:
    """Copy the existing text-embedding cache from src_root to dst_root if
    both files are present in src. Returns True on success."""
    src_npz = src_root / "text_embeddings_clip_vit_b32.npz"
    src_idx = src_root / "text_embeddings_index.json"
    if not (src_npz.exists() and src_idx.exists()):
        return False
    dst_npz = dst_root / "text_embeddings_clip_vit_b32.npz"
    dst_idx = dst_root / "text_embeddings_index.json"
    shutil.copy2(src_npz, dst_npz)
    shutil.copy2(src_idx, dst_idx)
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=Path(_DEFAULT_V18_CFG),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("cache/stage1_coarse_v1_objtraj_round18"),
    )
    parser.add_argument(
        "--max-per-subset", type=int, default=120,
        help="Cap per-subset clip count for smoke. Pass -1 for full dataset.",
    )
    parser.add_argument(
        "--cache-version", type=str, default="round18_2026-05-23_objtraj",
    )
    parser.add_argument("--std-eps", type=float, default=1e-3)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--copy-text-cache-from", type=Path,
        default=Path("cache/stage1_coarse_v1_full"),
        help="If text-embeddings cache exists at this root, copy it into the "
             "new cache so we don't rebuild. Pass empty string to skip and "
             "force in-place CLIP build.",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    out_root: Path = args.output_root
    clips_root = out_root / "clips"
    clips_root.mkdir(parents=True, exist_ok=True)

    print(f"[objtraj-cache] motion_representation = smpl_pose_135 (plan-free)")
    print(f"[objtraj-cache] surface_obj_pose      = True")
    print(f"[objtraj-cache] force_world_frame     = False  (Round-18 intentional)")
    print(f"[objtraj-cache] output_root           = {out_root}")
    print(f"[objtraj-cache] cache_version         = {args.cache_version}")
    print(f"[objtraj-cache] max_per_subset        = {args.max_per_subset}  (-1 = all)")
    print()

    splits = _resolve_subject_splits(cfg)
    if splits is None:
        print("[objtraj-cache] subject_split not enabled — all clips go to 'train'")
    else:
        for k, v in splits.items():
            print(f"[objtraj-cache] subject_split[{k}] = {len(v)} subjects")

    manifest_train: list[dict[str, Any]] = []
    manifest_val: list[dict[str, Any]] = []
    per_subset_train_counts: dict[str, int] = defaultdict(int)
    per_subset_val_counts: dict[str, int] = defaultdict(int)
    per_subset_skip_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"short": 0, "split_skipped": 0, "no_obj": 0, "non_finite": 0},
    )
    per_subset_train_concat_coarse: dict[str, list[np.ndarray]] = defaultdict(list)
    per_subset_train_concat_obj: dict[str, list[np.ndarray]] = defaultdict(list)
    train_global_concat_coarse: list[np.ndarray] = []
    train_global_concat_obj: list[np.ndarray] = []

    t_start = time.time()

    for entry in cfg.data.datasets:
        subset = Path(entry.root).name
        subset_dir = clips_root / subset
        subset_dir.mkdir(parents=True, exist_ok=True)
        ds = _build_subset_dataset(cfg, entry)
        n_total = len(ds)
        print(f"[objtraj-cache] [{subset}] dataset size = {n_total} clips (pre-cap)")
        indices = np.arange(n_total)
        if args.max_per_subset >= 0:
            indices = indices[: int(args.max_per_subset)]

        kept_train = 0
        kept_val = 0
        for idx in indices:
            sample = ds[int(idx)]
            seq_id = str(sample["seq_id"])
            text = str(sample.get("text", ""))
            motion = sample["motion"].numpy().astype(np.float32)
            seq_len = int(sample["seq_len"].item())
            if seq_len < 4:
                per_subset_skip_counts[subset]["short"] += 1
                continue
            split = _split_for_clip(splits, subset, seq_id)
            if split == "skip":
                per_subset_skip_counts[subset]["split_skipped"] += 1
                continue
            if "rest_offsets" not in sample:
                raise SystemExit(
                    f"rest_offsets missing for {subset}/{seq_id}; can't FK head/shoulder height."
                )
            rest_offsets = sample["rest_offsets"].numpy().astype(np.float32)

            if (
                "obj_com_canonical" not in sample
                or "obj_rot6d_canonical" not in sample
            ):
                per_subset_skip_counts[subset]["no_obj"] += 1
                continue
            obj_com = sample["obj_com_canonical"].numpy().astype(np.float32)         # (T, 3)
            obj_rot6d = sample["obj_rot6d_canonical"].numpy().astype(np.float32)     # (T, 6)
            obj_traj_canonical = np.concatenate(
                [obj_com, obj_rot6d], axis=-1,
            ).astype(np.float32)                                                     # (T, 9)
            if obj_traj_canonical.shape[1] != OBJ_TRAJ_DIM:
                raise SystemExit(
                    f"obj_traj_canonical wrong dim for {subset}/{seq_id}: "
                    f"got {obj_traj_canonical.shape}, expected (T, {OBJ_TRAJ_DIM})"
                )

            out = extract_coarse_v0_v1(motion, rest_offsets, seq_len, fps=args.fps)
            coarse_v1: np.ndarray = out["coarse_v1"]
            assert coarse_v1.shape[1] == COARSE_V1_DIM
            if not np.isfinite(coarse_v1).all():
                per_subset_skip_counts[subset]["non_finite"] += 1
                print(f"  [skip non-finite coarse_v1] {subset}/{seq_id}")
                continue
            if not np.isfinite(obj_traj_canonical[:seq_len]).all():
                per_subset_skip_counts[subset]["non_finite"] += 1
                print(f"  [skip non-finite obj_traj] {subset}/{seq_id}")
                continue
            init_coarse_v1 = coarse_v1[0].astype(np.float32)

            safe_id = _safe_filename(seq_id)
            npz_path = subset_dir / f"{safe_id}.npz"
            np.savez_compressed(
                npz_path,
                coarse_v1=coarse_v1,
                init_coarse_v1=init_coarse_v1,
                obj_traj_canonical=obj_traj_canonical,
                # NO raw object_positions/object_rotations stored.
                # NO object_pc stored.
                # NO z_int / contact / phase / support / plan / pseudo-labels.
            )
            record = {
                "subset": subset,
                "seq_id": seq_id,
                "safe_seq_id": safe_id,
                "npz_path": str(npz_path.relative_to(out_root).as_posix()),
                "seq_len": int(seq_len),
                "text": text,
                "split": split,
                "cache_version": args.cache_version,
                "fps": float(args.fps),
                "coarse_v1_dim": int(COARSE_V1_DIM),
                "obj_traj_dim": int(OBJ_TRAJ_DIM),
                "channel_names_coarse": COARSE_V0_NAMES + COARSE_V1_EXTRA_NAMES,
                "channel_names_obj_traj": [
                    "obj_com_x", "obj_com_y", "obj_com_z",
                    "obj_rot6d_0", "obj_rot6d_1", "obj_rot6d_2",
                    "obj_rot6d_3", "obj_rot6d_4", "obj_rot6d_5",
                ],
            }
            if split == "train":
                manifest_train.append(record)
                per_subset_train_counts[subset] += 1
                kept_train += 1
                train_global_concat_coarse.append(coarse_v1)
                # Only concat valid frames of obj_traj for stats — padded tail
                # is zero by construction (np.zeros pad).
                train_global_concat_obj.append(obj_traj_canonical[:seq_len])
                per_subset_train_concat_coarse[subset].append(coarse_v1)
                per_subset_train_concat_obj[subset].append(obj_traj_canonical[:seq_len])
            else:
                manifest_val.append(record)
                per_subset_val_counts[subset] += 1
                kept_val += 1

        skip = per_subset_skip_counts[subset]
        print(
            f"[objtraj-cache] [{subset}] kept train={kept_train} val={kept_val}  "
            f"(short={skip['short']} split_skipped={skip['split_skipped']} "
            f"no_obj={skip['no_obj']} non_finite={skip['non_finite']} "
            f"pre-cap={len(indices)})"
        )

    # Manifests
    def _write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
        with path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    _write_jsonl(manifest_train, out_root / "manifest_train.jsonl")
    _write_jsonl(manifest_val, out_root / "manifest_val.jsonl")

    # Normalization stats (train only)
    if not train_global_concat_coarse:
        raise SystemExit("[objtraj-cache] no train clips produced — cannot compute normalization")
    coarse_train = np.concatenate(train_global_concat_coarse, axis=0)
    obj_train = np.concatenate(train_global_concat_obj, axis=0)
    coarse_global = _z_score_stats(coarse_train, eps=args.std_eps)
    obj_global = _z_score_stats(obj_train, eps=args.std_eps)
    per_subset_stats: dict[str, dict[str, Any]] = {}
    for subset in per_subset_train_concat_coarse:
        per_subset_stats[subset] = {
            "coarse_v1": _z_score_stats(
                np.concatenate(per_subset_train_concat_coarse[subset], axis=0),
                eps=args.std_eps,
            ),
            "obj_traj_canonical": _z_score_stats(
                np.concatenate(per_subset_train_concat_obj[subset], axis=0),
                eps=args.std_eps,
            ),
        }
    norm = {
        "split": "train",
        "cache_version": args.cache_version,
        "n_train_clips": len(manifest_train),
        "global": {
            "coarse_v1": coarse_global,
            "obj_traj_canonical": obj_global,
        },
        "per_subset": per_subset_stats,
        "fps": float(args.fps),
        "config_source": str(args.config),
        "motion_representation": "smpl_pose_135",
        "surface_obj_pose": True,
        "force_world_frame": False,
        "subject_split_used": splits is not None,
        # BACKWARDS-COMPAT shim: trainer code that expects the Round-14 schema
        # (i.e. `norm["global"]["mean"]` for the 23-dim Coarse-v1) should
        # still work because we copy the coarse_v1 stats up one level. New
        # trainer code uses the nested form above.
        "n_dims": int(COARSE_V1_DIM),
        "channel_names": COARSE_V0_NAMES + COARSE_V1_EXTRA_NAMES,
    }
    # Promote coarse_v1 stats to top-level for trainer back-compat.
    norm["global"].update({
        "mean": coarse_global["mean"],
        "std": coarse_global["std"],
        "std_eps": coarse_global["std_eps"],
        "std_clamped": coarse_global["std_clamped"],
        "n_frames": coarse_global["n_frames"],
    })
    (out_root / "normalization_train.json").write_text(
        json.dumps(norm, indent=2), encoding="utf-8",
    )

    # Try to inherit text-embeddings cache from the Round-14 cache.
    text_cache_copied = False
    if args.copy_text_cache_from and str(args.copy_text_cache_from):
        text_cache_copied = _try_copy_text_cache(args.copy_text_cache_from, out_root)
        if text_cache_copied:
            print(f"[objtraj-cache] copied text-embeddings cache from {args.copy_text_cache_from}")
        else:
            print(f"[objtraj-cache] text-embeddings cache not found at {args.copy_text_cache_from}; "
                  "build separately if needed.")

    # Sanity round-trip
    sample_batch = coarse_train[: min(2048, coarse_train.shape[0])]
    mean = np.asarray(coarse_global["mean"], dtype=np.float32)
    std = np.asarray(coarse_global["std_clamped"], dtype=np.float32)
    rt_err = float(np.max(np.abs(sample_batch - (((sample_batch - mean) / std) * std + mean))))
    print(f"[objtraj-cache] coarse_v1 normalization round-trip max|x - x_back| = {rt_err:.3e}")

    obj_sample = obj_train[: min(2048, obj_train.shape[0])]
    obj_mean = np.asarray(obj_global["mean"], dtype=np.float32)
    obj_std = np.asarray(obj_global["std_clamped"], dtype=np.float32)
    rt_err_obj = float(np.max(np.abs(obj_sample - (((obj_sample - obj_mean) / obj_std) * obj_std + obj_mean))))
    print(f"[objtraj-cache] obj_traj    normalization round-trip max|x - x_back| = {rt_err_obj:.3e}")

    # README contract
    is_smoke = args.max_per_subset >= 0
    scope_block = (
        (
            "> **SCOPE: Smoke / preflight only — NOT a variance-protocol cache.**\n"
            f"> Built with `--max-per-subset {args.max_per_subset}`; some subsets\n"
            "> may have zero val coverage. Run with `--max-per-subset -1` for full.\n"
        )
        if is_smoke
        else (
            "> **SCOPE: Full cache (no per-subset cap).** Eligible for variance-\n"
            "> protocol training. Inherits Round-14 strict-mode checks via the\n"
            "> existing verify_stage1_cache_for_variance_protocol.py (extended\n"
            "> for obj_traj_canonical field in Round-18 preflight tests).\n"
        )
    )
    (out_root / "README_cache_contract.md").write_text(
        f"""# Stage-1 Coarse-v1 + obj_traj cache contract (Round 18)

{scope_block}
> Round-18 design rationale:
> `analyses/2026-05-23_stage1_round17_objtraj_conditioned_design_review.md`.
> Implementation report:
> `analyses/2026-05-23_stage1_round18_s1o_implementation_preflight_report.md`.

Cache version: `{args.cache_version}`
Built: {time.strftime("%Y-%m-%dT%H:%M:%S")}
Source config: `{args.config}`
Motion representation: `smpl_pose_135` (plan-free; NO plan compilation).
Surface object pose: True (returns obj_com_canonical, obj_rot6d_canonical).
Force world frame: **False** (Round-18 chooses true body-canonical;
DIFFERS from v18 active config which sets force_world_frame: true).
FPS: {args.fps}
Coarse-v1 dim: {COARSE_V1_DIM}
obj_traj dim: {OBJ_TRAJ_DIM} (obj_com 3 + obj_rot6d 6, body-canonical)
Subject split enabled: {splits is not None}
Per-subset cap: {args.max_per_subset}

## Frame convention note

Under v18's active config, `force_world_frame=True` short-circuits
`_compute_canonical_object_pose` to identity transform — so v18's
"obj_com_canonical" is effectively world-frame.  Round-18 explicitly
disables this short-circuit (`force_world_frame=False`) to get TRUE
body-canonical object pose, aligned to the per-clip frame-0 facing
yaw + root position. This is intentional and documented; it diverges
from v18's storage convention.

## Coarse-v1 channels

{chr(10).join(f"- `[{i}]` {n}" for i, n in enumerate(COARSE_V0_NAMES + COARSE_V1_EXTRA_NAMES))}

## obj_traj_canonical channels

- `[0]` obj_com_x  (body-canonical)
- `[1]` obj_com_y  (body-canonical)
- `[2]` obj_com_z  (body-canonical)
- `[3]` obj_rot6d_0
- `[4]` obj_rot6d_1
- `[5]` obj_rot6d_2
- `[6]` obj_rot6d_3
- `[7]` obj_rot6d_4
- `[8]` obj_rot6d_5

## What each clip .npz contains

- `coarse_v1: (T, 23) float32`
- `init_coarse_v1: (23,) float32`
- `obj_traj_canonical: (T, 9) float32`

That is the entire payload. The cache excludes:

- raw `object_positions` / `object_rotations` (world frame)
- `object_pc` / object tokens / object mesh
- z_int (`contact_state, contact_target_xyz, phase, support`)
- interaction plan (`plan_*`)
- pseudo labels, hand/foot targets
- raw motion_135 / motion_263

## normalization_train.json

- `global.coarse_v1.{{mean,std,std_clamped}}` — 23-dim Coarse-v1 stats
- `global.obj_traj_canonical.{{mean,std,std_clamped}}` — 9-dim obj_traj stats
- `per_subset.<subset>.{{coarse_v1,obj_traj_canonical}}.{{...}}` — diagnostic only
- `global.{{mean,std,std_clamped,n_frames}}` — back-compat shim equal to
  `global.coarse_v1.*` for trainer code that expects the Round-14 schema.

All stats computed on the TRAIN split only. Val never touched.

## Round-9 selection compatibility

Same as Round-14 cache: every clip carries `(subset, seq_id)` so the
existing `--selection-json analyses/2026-05-19_subset_balanced_failure_selection.json`
flag on eval_stage1_coarse_prior.py works unchanged.
""",
        encoding="utf-8",
    )

    elapsed = time.time() - t_start
    summary = {
        "cache_version": args.cache_version,
        "n_train_clips": len(manifest_train),
        "n_val_clips": len(manifest_val),
        "per_subset_train": dict(per_subset_train_counts),
        "per_subset_val": dict(per_subset_val_counts),
        "per_subset_skips": {k: dict(v) for k, v in per_subset_skip_counts.items()},
        "n_train_frames_coarse": int(coarse_train.shape[0]),
        "n_train_frames_obj": int(obj_train.shape[0]),
        "coarse_v1_mean_first5": [float(x) for x in coarse_global["mean"][:5]],
        "obj_traj_mean": [float(x) for x in obj_global["mean"]],
        "obj_traj_std": [float(x) for x in obj_global["std"]],
        "coarse_rt_err": rt_err,
        "obj_rt_err": rt_err_obj,
        "elapsed_seconds": float(elapsed),
        "max_per_subset": args.max_per_subset,
        "motion_representation": "smpl_pose_135",
        "surface_obj_pose": True,
        "force_world_frame": False,
        "text_cache_copied": bool(text_cache_copied),
    }
    (out_root / "build_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )

    print()
    print(f"[objtraj-cache] wrote manifest_train.jsonl  n={len(manifest_train)}")
    print(f"[objtraj-cache] wrote manifest_val.jsonl    n={len(manifest_val)}")
    print(f"[objtraj-cache] wrote normalization_train.json")
    print(f"[objtraj-cache] wrote README_cache_contract.md")
    print(f"[objtraj-cache] wrote build_summary.json")
    print(f"[objtraj-cache] elapsed {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
