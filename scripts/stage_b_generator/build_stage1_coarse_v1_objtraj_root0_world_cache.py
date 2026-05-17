"""Round-18 follow-up: build the Stage-1 obj_traj cache in the SAME frame
as Coarse-v1's targets.

Why this exists (vs ``build_stage1_coarse_v1_objtraj_cache.py``):

Round-18's first attempt used ``surface_obj_pose=True,
force_world_frame=False`` to get "body-canonical" obj pose. But the
Codex review pointed out that **Coarse-v1's root_local_trans is
``root_world - root0`` (world-axis, root0-relative, NO yaw rotation
+ Y offset by ``root_world[0,Y]``)**, while body-canonical applies an
inverse Y-rotation AND a MoMask-style floor-height Y offset
(``world.min(axis=(0,1))[1]``). The two frames differ by both a
rotation and a Y shift → the obj_traj condition and Coarse-v1 target
were in inconsistent frames.

The fix is to store obj_traj in **the exact same frame as Coarse-v1**:

    obj_pos_root0_world  = object_positions[:T] - root_world[0]    # (T, 3)
    obj_rot6d_world      = matrix_to_rotation_6d(R_world(aa))      # (T, 6)
    obj_traj_root0_world = concat(obj_pos_root0_world, obj_rot6d_world)
                                                                    # (T, 9)

This matches Coarse-v1's `root_local_trans = root_world - root0` and
keeps world-axis orientation throughout. No yaw canonicalization
applied to either side.

Hard constraints (unchanged from earlier Round-18 builder):

- v18 untouched.
- Stage-2 untouched.
- Official dataset directories not written.
- No object_pc, no z_int / contact / phase / support, no plan_*, no
  pseudo-labels, no hand/foot stored.
- Old `cache/stage1_coarse_v1_objtraj_round18/` is NOT modified or
  deleted — kept on disk for forensic comparison; only the new
  Round-18-fix cache at `cache/stage1_coarse_v1_objtraj_root0_world_round18_fix/`
  is used by Round-19+ training.

Usage::

    $env:PYTHONIOENCODING="utf-8"
    conda run -n piano python scripts/stage_b_generator/build_stage1_coarse_v1_objtraj_root0_world_cache.py \
        --output-root cache/stage1_coarse_v1_objtraj_root0_world_round18_fix \
        --max-per-subset -1 \
        --cache-version round18_fix_2026-05-23_root0_world
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
from omegaconf import OmegaConf


# Round-18-fix-server: v18 config path; overridable via PIANO_V18_CFG env var.
_DEFAULT_V18_CFG = os.environ.get(
    "PIANO_V18_CFG",
    "configs/training/anchordiff_v18_a1_FULL_DATA.yaml",
)

from piano.data.dataset import (
    AugmentConfig, HOIDataset, build_subject_split, extract_subject_id,
)
from piano.utils.canonical_frame import (
    axis_angle_to_matrix_np, matrix_to_rotation_6d_np,
)
from piano.utils.io_utils import load_json

from extract_coarse_motion_representation import (
    COARSE_V1_DIM, COARSE_V0_NAMES, COARSE_V1_EXTRA_NAMES,
    extract_coarse_v0_v1,
)


SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
DEFAULT_FPS = 20.0
OBJ_TRAJ_DIM = 9     # obj_pos_root0_world (3) + obj_rot6d_world (6)


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
    """surface_obj_pose=False so the dataset does NOT compute body-canonical
    obj pose (we compute root0-relative world ourselves from
    object_positions + object_rotations, which the dataset always returns
    when present). force_world_frame doesn't matter when surface_obj_pose
    is False, but we pass False for clarity.
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
        surface_obj_pose=False,           # <- explicitly skip canonical-pose path
        force_world_frame=False,          # irrelevant when surface_obj_pose=False
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
    src_npz = src_root / "text_embeddings_clip_vit_b32.npz"
    src_idx = src_root / "text_embeddings_index.json"
    if not (src_npz.exists() and src_idx.exists()):
        return False
    shutil.copy2(src_npz, dst_root / "text_embeddings_clip_vit_b32.npz")
    shutil.copy2(src_idx, dst_root / "text_embeddings_index.json")
    return True


def _compute_obj_traj_root0_world(
    object_positions: np.ndarray,        # (T_pad, 3) world-frame, padded
    object_rotations: np.ndarray,        # (T_pad, 3) axis-angle, world-frame, padded
    root_world: np.ndarray,              # (T_eff, 3) frame-0..(T_eff-1) root world from extract_coarse_v0_v1
    seq_len_eff: int,
) -> np.ndarray:
    """Compute the new Round-18-fix obj_traj field in the SAME frame as
    Coarse-v1's root_local_trans.

    Stage-1 needs the (per-frame) object position expressed relative to
    the human's frame-0 root world position, AND the (per-frame) object
    orientation in the world frame (Coarse-v1 has no yaw canonicalization
    applied to its targets, so the conditioning should also use world-
    axis rotations).

    Both inputs are sliced to ``[:seq_len_eff]`` so the returned tensor
    has shape ``(seq_len_eff, 9)``, matching the (un-padded) shape of
    Coarse-v1.
    """
    if root_world.shape[0] != seq_len_eff:
        raise ValueError(
            f"root_world len ({root_world.shape[0]}) != seq_len_eff ({seq_len_eff})"
        )
    obj_pos_world = object_positions[:seq_len_eff].astype(np.float32)            # (T, 3)
    obj_rot_world_aa = object_rotations[:seq_len_eff].astype(np.float32)         # (T, 3)
    if obj_pos_world.shape[0] != seq_len_eff or obj_rot_world_aa.shape[0] != seq_len_eff:
        raise ValueError(
            f"object_positions/rotations shorter than seq_len_eff: "
            f"pos={obj_pos_world.shape[0]} rot={obj_rot_world_aa.shape[0]} "
            f"need={seq_len_eff}"
        )
    obj_pos_root0_world = (obj_pos_world - root_world[0:1]).astype(np.float32)   # (T, 3)
    R_world = axis_angle_to_matrix_np(obj_rot_world_aa)                          # (T, 3, 3)
    obj_rot6d_world = matrix_to_rotation_6d_np(R_world).astype(np.float32)       # (T, 6)
    out = np.concatenate([obj_pos_root0_world, obj_rot6d_world], axis=-1)        # (T, 9)
    if out.shape != (seq_len_eff, OBJ_TRAJ_DIM):
        raise ValueError(f"obj_traj_root0_world shape {out.shape} != "
                         f"({seq_len_eff}, {OBJ_TRAJ_DIM})")
    return out.astype(np.float32, copy=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=Path(_DEFAULT_V18_CFG),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("cache/stage1_coarse_v1_objtraj_root0_world_round18_fix"),
    )
    parser.add_argument(
        "--max-per-subset", type=int, default=120,
        help="Cap per-subset clip count for smoke. Pass -1 for full dataset.",
    )
    parser.add_argument(
        "--cache-version", type=str, default="round18_fix_2026-05-23_root0_world",
    )
    parser.add_argument("--std-eps", type=float, default=1e-3)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--copy-text-cache-from", type=Path,
        default=Path("cache/stage1_coarse_v1_full"),
        help="Inherit the text-embeddings cache from this root.",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    out_root: Path = args.output_root
    clips_root = out_root / "clips"
    clips_root.mkdir(parents=True, exist_ok=True)

    print(f"[r18fix-cache] motion_representation = smpl_pose_135 (plan-free)")
    print(f"[r18fix-cache] surface_obj_pose      = False (compute root0_world ourselves)")
    print(f"[r18fix-cache] force_world_frame     = False (irrelevant when surface_obj_pose=False)")
    print(f"[r18fix-cache] obj_traj convention   = obj_pos - root_world[0] (world axis), obj_rot6d in world frame")
    print(f"[r18fix-cache] obj_traj shape        = (seq_len, {OBJ_TRAJ_DIM})  (matches Coarse-v1, NOT padded)")
    print(f"[r18fix-cache] output_root           = {out_root}")
    print(f"[r18fix-cache] cache_version         = {args.cache_version}")
    print(f"[r18fix-cache] max_per_subset        = {args.max_per_subset}  (-1 = all)")
    print()

    splits = _resolve_subject_splits(cfg)
    if splits is None:
        print("[r18fix-cache] subject_split not enabled — all clips go to 'train'")
    else:
        for k, v in splits.items():
            print(f"[r18fix-cache] subject_split[{k}] = {len(v)} subjects")

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
        print(f"[r18fix-cache] [{subset}] dataset size = {n_total} clips (pre-cap)")
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
                "object_positions" not in sample
                or "object_rotations" not in sample
            ):
                per_subset_skip_counts[subset]["no_obj"] += 1
                continue
            object_positions = sample["object_positions"].numpy().astype(np.float32)
            object_rotations = sample["object_rotations"].numpy().astype(np.float32)

            # Compute Coarse-v1 first; we need its root_world to reference frame 0.
            out = extract_coarse_v0_v1(motion, rest_offsets, seq_len, fps=args.fps)
            coarse_v1: np.ndarray = out["coarse_v1"]                              # (T_eff, 23)
            root_world: np.ndarray = out["root_world"]                            # (T_eff, 3)
            T_eff = int(coarse_v1.shape[0])
            assert coarse_v1.shape[1] == COARSE_V1_DIM
            if not np.isfinite(coarse_v1).all():
                per_subset_skip_counts[subset]["non_finite"] += 1
                print(f"  [skip non-finite coarse_v1] {subset}/{seq_id}")
                continue

            # Now compute obj_traj in the SAME frame as Coarse-v1's
            # root_local_trans (which is `root_world - root0`).
            try:
                obj_traj_root0_world = _compute_obj_traj_root0_world(
                    object_positions, object_rotations, root_world, T_eff,
                )
            except ValueError as e:
                per_subset_skip_counts[subset]["non_finite"] += 1
                print(f"  [skip {subset}/{seq_id}: {e}]")
                continue
            if not np.isfinite(obj_traj_root0_world).all():
                per_subset_skip_counts[subset]["non_finite"] += 1
                print(f"  [skip non-finite obj_traj_root0_world] {subset}/{seq_id}")
                continue

            init_coarse_v1 = coarse_v1[0].astype(np.float32)

            safe_id = _safe_filename(seq_id)
            npz_path = subset_dir / f"{safe_id}.npz"
            np.savez_compressed(
                npz_path,
                coarse_v1=coarse_v1,                          # (T_eff, 23)
                init_coarse_v1=init_coarse_v1,                # (23,)
                obj_traj_root0_world=obj_traj_root0_world,    # (T_eff, 9)  ← same length as coarse_v1
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
                    "obj_pos_root0_world_x", "obj_pos_root0_world_y", "obj_pos_root0_world_z",
                    "obj_rot6d_world_0", "obj_rot6d_world_1", "obj_rot6d_world_2",
                    "obj_rot6d_world_3", "obj_rot6d_world_4", "obj_rot6d_world_5",
                ],
            }
            if split == "train":
                manifest_train.append(record)
                per_subset_train_counts[subset] += 1
                kept_train += 1
                train_global_concat_coarse.append(coarse_v1)
                train_global_concat_obj.append(obj_traj_root0_world)
                per_subset_train_concat_coarse[subset].append(coarse_v1)
                per_subset_train_concat_obj[subset].append(obj_traj_root0_world)
            else:
                manifest_val.append(record)
                per_subset_val_counts[subset] += 1
                kept_val += 1

        skip = per_subset_skip_counts[subset]
        print(
            f"[r18fix-cache] [{subset}] kept train={kept_train} val={kept_val}  "
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
        raise SystemExit("[r18fix-cache] no train clips produced — cannot compute normalization")
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
            "obj_traj_root0_world": _z_score_stats(
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
            "obj_traj_root0_world": obj_global,
        },
        "per_subset": per_subset_stats,
        "fps": float(args.fps),
        "config_source": str(args.config),
        "motion_representation": "smpl_pose_135",
        "surface_obj_pose": False,
        "force_world_frame": False,
        "obj_traj_frame": "obj_pos_root0_world + obj_rot6d_world",
        "subject_split_used": splits is not None,
        # BACKWARDS-COMPAT shim for trainer code that expects the old
        # Round-14 schema (`norm["global"]["mean"]` for the 23-dim
        # Coarse-v1). Copy coarse_v1 stats up one level.
        "n_dims": int(COARSE_V1_DIM),
        "channel_names": COARSE_V0_NAMES + COARSE_V1_EXTRA_NAMES,
    }
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

    text_cache_copied = False
    if args.copy_text_cache_from and str(args.copy_text_cache_from):
        text_cache_copied = _try_copy_text_cache(args.copy_text_cache_from, out_root)
        if text_cache_copied:
            print(f"[r18fix-cache] copied text-embeddings cache from {args.copy_text_cache_from}")
        else:
            print(f"[r18fix-cache] text-embeddings cache not found at {args.copy_text_cache_from}; "
                  "build separately if needed.")

    sample_batch = coarse_train[: min(2048, coarse_train.shape[0])]
    mean = np.asarray(coarse_global["mean"], dtype=np.float32)
    std = np.asarray(coarse_global["std_clamped"], dtype=np.float32)
    rt_err = float(np.max(np.abs(sample_batch - (((sample_batch - mean) / std) * std + mean))))
    print(f"[r18fix-cache] coarse_v1 normalization round-trip max|x - x_back| = {rt_err:.3e}")

    obj_sample = obj_train[: min(2048, obj_train.shape[0])]
    obj_mean = np.asarray(obj_global["mean"], dtype=np.float32)
    obj_std = np.asarray(obj_global["std_clamped"], dtype=np.float32)
    rt_err_obj = float(np.max(np.abs(obj_sample - (((obj_sample - obj_mean) / obj_std) * obj_std + obj_mean))))
    print(f"[r18fix-cache] obj_traj   normalization round-trip max|x - x_back| = {rt_err_obj:.3e}")

    is_smoke = args.max_per_subset >= 0
    scope_block = (
        (
            "> **SCOPE: Smoke / preflight only — NOT a variance-protocol cache.**\n"
            f"> Built with `--max-per-subset {args.max_per_subset}`.\n"
        )
        if is_smoke
        else (
            "> **SCOPE: Full cache (no per-subset cap).** Eligible for variance-\n"
            "> protocol training. Round-19+ Plan C / S1-O uses this cache.\n"
        )
    )
    (out_root / "README_cache_contract.md").write_text(
        f"""# Stage-1 Coarse-v1 + obj_traj cache contract (Round 18-fix)

{scope_block}
> Design rationale + frame-consistency proof:
> `analyses/2026-05-23_stage1_round18_s1o_frame_rng_fix_report.md`.
> Supersedes the Round-18 cache at
> `cache/stage1_coarse_v1_objtraj_round18/` (kept on disk for forensic
> comparison; trainer/eval no longer point at the old cache by default).

Cache version: `{args.cache_version}`
Built: {time.strftime("%Y-%m-%dT%H:%M:%S")}
Source config: `{args.config}`
Motion representation: `smpl_pose_135` (plan-free; NO plan compilation).
Surface object pose: **False** (skip dataset's canonical-pose path).
Force world frame: False (irrelevant when surface_obj_pose=False).

## Frame convention — matches Coarse-v1 target exactly

Coarse-v1's `root_local_trans` is `root_world - root_world[0]` —
world-axis, root0-relative, no yaw canonicalization, Y offset by
`root_world[0,Y]`. This cache stores obj_traj in the SAME frame:

    obj_pos_root0_world  = object_positions[:T] - root_world[0]   # (T, 3) world axis
    obj_rot6d_world      = matrix_to_rotation_6d(R_world(aa))     # (T, 6) world frame
    obj_traj_root0_world = concat(obj_pos_root0_world, obj_rot6d_world)
                                                                   # (T, 9)

`R_world(aa) = axis_angle_to_matrix(object_rotations)`. The Coarse-v1
target and the obj_traj condition are now both in `(root_world − root0,
world-axis)` coordinates, so the model can learn a consistent
human-relative-to-object relationship without resolving an implicit
frame mismatch.

This DIFFERS from Round-18's original cache
(`cache/stage1_coarse_v1_objtraj_round18/`), which stored
`obj_com_canonical / obj_rot6d_canonical` produced by the dataset's
`world_to_canonical_object_pose` — that path applies an inverse Y
rotation (frame-0 facing alignment) AND a MoMask-style floor-Y offset.
Neither of those operations is applied to Coarse-v1 target signals;
they were a frame mismatch.

## Shape contract

- `coarse_v1: (seq_len, 23) float32`         — NOT padded.
- `init_coarse_v1: (23,) float32`            — `coarse_v1[0]`.
- `obj_traj_root0_world: (seq_len, 9) float32` — NOT padded; SAME
  length as `coarse_v1`. The trainer's collate function pads BOTH to
  `(B, T_max, ·)` consistently.

The OLD Round-18 cache wrote `obj_traj_canonical` at `(196, 9)`
(padded to max_seq_length). This cache writes at the unpadded length.

## Coarse-v1 channels

{chr(10).join(f"- `[{i}]` {n}" for i, n in enumerate(COARSE_V0_NAMES + COARSE_V1_EXTRA_NAMES))}

## obj_traj_root0_world channels

- `[0]` obj_pos_root0_world_x  = (object_positions[:, 0] − root_world[0, 0])
- `[1]` obj_pos_root0_world_y  = (object_positions[:, 1] − root_world[0, 1])
- `[2]` obj_pos_root0_world_z  = (object_positions[:, 2] − root_world[0, 2])
- `[3]` obj_rot6d_world_0
- `[4]` obj_rot6d_world_1
- `[5]` obj_rot6d_world_2
- `[6]` obj_rot6d_world_3
- `[7]` obj_rot6d_world_4
- `[8]` obj_rot6d_world_5

## What each clip .npz contains

- `coarse_v1` (seq_len, 23)
- `init_coarse_v1` (23,)
- `obj_traj_root0_world` (seq_len, 9)

Forbidden (verified by preflight t4):

- raw `object_positions` / `object_rotations`
- `obj_com_canonical` / `obj_rot6d_canonical` (the OLD field names — explicitly NOT stored)
- `object_pc` / object tokens / object mesh
- z_int (`contact_state, contact_target_xyz, phase, support`)
- interaction plan (`plan_*`)
- pseudo labels, hand/foot targets
- raw motion_135 / motion_263

## normalization_train.json

- `global.coarse_v1.{{mean,std,std_clamped}}` — 23-dim Coarse-v1 stats
- `global.obj_traj_root0_world.{{mean,std,std_clamped}}` — 9-dim obj_traj stats
- `per_subset.<subset>.{{coarse_v1,obj_traj_root0_world}}.{{...}}` — diagnostic only
- `global.{{mean,std,std_clamped,n_frames}}` — back-compat shim equal to
  `global.coarse_v1.*` for trainer code that expects the Round-14 schema.

All stats computed on the TRAIN split only. Val never touched.
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
        "surface_obj_pose": False,
        "obj_traj_frame": "obj_pos_root0_world + obj_rot6d_world",
        "text_cache_copied": bool(text_cache_copied),
    }
    (out_root / "build_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )

    print()
    print(f"[r18fix-cache] wrote manifest_train.jsonl  n={len(manifest_train)}")
    print(f"[r18fix-cache] wrote manifest_val.jsonl    n={len(manifest_val)}")
    print(f"[r18fix-cache] wrote normalization_train.json")
    print(f"[r18fix-cache] wrote README_cache_contract.md")
    print(f"[r18fix-cache] wrote build_summary.json")
    print(f"[r18fix-cache] elapsed {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
