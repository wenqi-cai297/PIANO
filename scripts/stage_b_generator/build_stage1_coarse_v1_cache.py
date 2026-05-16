"""Round-12 preflight (Step 2): build the Stage-1 Coarse-v1 cache.

Builds an object-free / plan-free / contact-free / hand-foot-free cache
of Coarse-v1 (23-d/frame) Stage-1 prior training inputs, per clip per
subset, with text + init Coarse-v1 metadata.

Design constraints
------------------

- Use ``motion_representation="smpl_pose_135"`` to bypass the plan
  compiler inside ``HOIDataset.__getitem__`` (the plan compiler runs
  only under ``smpl_pose_135_plan``; see ``src/piano/data/dataset.py``
  around L581-688 and L632 specifically — the ``if`` gate is conditioned
  on the representation string).
- Pass ``pseudo_label_dir=None`` so the dataset returns ``labels={}``
  with no contact/phase/support arrays.
- Store ONLY Coarse-v1-safe fields per clip:
  ``coarse_v1, init_coarse_v1, seq_len, text, seq_id, subset, split,
  cache_version, source_seq_npz_size_bytes, fps``.
  No object trajectory, no object point cloud, no z_int, no contact
  labels, no plan, no hand/foot targets.
- Build manifests for train and val splits per the project's
  subject-split convention (train 85 / val 15, seed 42, same as v18
  ``configs/training/anchordiff_v18_a1_FULL_DATA.yaml``).
- Compute global per-dim z-score stats on the TRAIN manifest ONLY.
  Save per-subset stats for diagnostics only — never used for
  normalization.

After this cache is built, the trainer never invokes ``HOIDataset``
again — it loads clips directly from the cache.

Usage
-----

    $env:PYTHONIOENCODING="utf-8"
    conda run -n piano python scripts/stage_b_generator/build_stage1_coarse_v1_cache.py \
        --config configs/training/anchordiff_v18_a1_FULL_DATA.yaml \
        --output-root cache/stage1_coarse_v1_round12 \
        --max-per-subset 120 \
        --cache-version round12_2026-05-22_v1

The ``--max-per-subset`` cap controls smoke-test cache size; pass
``--max-per-subset -1`` for the full dataset.

Output layout
-------------

    cache/stage1_coarse_v1_round12/
        manifest_train.jsonl     # one JSON line per clip
        manifest_val.jsonl
        clips/<subset>/<safe_seq_id>.npz
        normalization_train.json
        README_cache_contract.md
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

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


def _build_subset_dataset(cfg, subset_entry, *, motion_rep: str) -> HOIDataset:
    """Build a single-subset HOIDataset with no augment, no pseudo labels,
    and the requested motion_representation. Plan compilation is bypassed
    when motion_rep == "smpl_pose_135"."""
    return HOIDataset(
        root=subset_entry.root,
        pseudo_label_dir=None,  # no pseudo-labels → no contact / phase / support
        max_seq_length=int(cfg.data.max_seq_length),
        subject_id_filter=None,
        subsample_n_per_object=None,
        subsample_seed=int(cfg.data.get("subsample_seed", 42)),
        support_collapse_hand_support=bool(
            cfg.data.get("support_collapse_hand_support", True)
        ),
        surface_obj_pose=True,
        force_world_frame=bool(cfg.data.get("force_world_frame", False)),
        motion_representation=motion_rep,
        augment=AugmentConfig(enabled=False),
    )


def _split_for_clip(splits: dict[str, set] | None, subset: str, seq_id: str) -> str:
    if splits is None:
        return "train"
    sid = extract_subject_id(subset, seq_id)
    if sid is None:
        return "train"
    # build_subject_split returns namespaced keys f"{subset}/{raw_id}", not tuples.
    key = f"{subset}/{sid}"
    if key in splits.get("train", set()):
        return "train"
    if key in splits.get("val", set()):
        return "val"
    return "skip"


def _z_score_stats(arr: np.ndarray, eps: float) -> dict[str, list[float] | float | int]:
    """Per-dim mean/std on a (N_frames_concat, D) flat array."""
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/training/anchordiff_v18_a1_FULL_DATA.yaml"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("cache/stage1_coarse_v1_round12"),
    )
    parser.add_argument(
        "--motion-representation", choices=["smpl_pose_135"], default="smpl_pose_135",
        help="Force smpl_pose_135 (no _plan) so plan compilation is bypassed.",
    )
    parser.add_argument(
        "--max-per-subset", type=int, default=120,
        help="Cap per-subset clip count for smoke. Pass -1 for full dataset.",
    )
    parser.add_argument(
        "--cache-version", type=str, default="round12_2026-05-22_v1",
    )
    parser.add_argument("--std-eps", type=float, default=1e-3)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    rng = np.random.default_rng(args.seed)

    out_root: Path = args.output_root
    clips_root = out_root / "clips"
    clips_root.mkdir(parents=True, exist_ok=True)

    print(f"[cache] motion_representation = {args.motion_representation} (plan-free)")
    print(f"[cache] output_root = {out_root}")
    print(f"[cache] cache_version = {args.cache_version}")
    print(f"[cache] max_per_subset = {args.max_per_subset}  (-1 = all)")
    print()

    splits = _resolve_subject_splits(cfg)
    if splits is None:
        print("[cache] subject_split not enabled — all clips go to 'train' manifest.")
    else:
        for k, v in splits.items():
            print(f"[cache] subject_split[{k}] = {len(v)} subjects")

    manifest_train: list[dict[str, Any]] = []
    manifest_val: list[dict[str, Any]] = []
    per_subset_train_counts: dict[str, int] = defaultdict(int)
    per_subset_val_counts: dict[str, int] = defaultdict(int)
    per_subset_train_concat: dict[str, list[np.ndarray]] = defaultdict(list)
    train_global_concat: list[np.ndarray] = []

    t_start = time.time()

    for entry in cfg.data.datasets:
        subset = Path(entry.root).name
        subset_dir = clips_root / subset
        subset_dir.mkdir(parents=True, exist_ok=True)
        ds = _build_subset_dataset(cfg, entry, motion_rep=args.motion_representation)
        n_total = len(ds)
        print(f"[cache] [{subset}] dataset size = {n_total} clips (pre-cap)")
        indices = np.arange(n_total)
        # Optional cap per subset (for smoke-scale cache). We do NOT subsample
        # randomly across subjects; we just take the first N to keep things
        # deterministic and reproducible. The subject split still applies.
        if args.max_per_subset >= 0:
            indices = indices[: int(args.max_per_subset)]

        kept_train = 0
        kept_val = 0
        skipped_split = 0
        skipped_short = 0
        text_seen: dict[str, str] = {}

        for idx in indices:
            sample = ds[int(idx)]
            seq_id = str(sample["seq_id"])
            text = str(sample.get("text", ""))
            motion = sample["motion"].numpy().astype(np.float32)  # (max_T, 135)
            seq_len = int(sample["seq_len"].item())
            if seq_len < 4:
                skipped_short += 1
                continue
            split = _split_for_clip(splits, subset, seq_id)
            if split == "skip":
                skipped_split += 1
                continue
            if "rest_offsets" not in sample:
                raise SystemExit(
                    f"rest_offsets missing for {subset}/{seq_id}; can't FK head/shoulder height."
                )
            rest_offsets = sample["rest_offsets"].numpy().astype(np.float32)

            out = extract_coarse_v0_v1(motion, rest_offsets, seq_len, fps=args.fps)
            coarse_v1: np.ndarray = out["coarse_v1"]                    # (T, 23)
            assert coarse_v1.shape[1] == COARSE_V1_DIM
            if not np.isfinite(coarse_v1).all():
                # Skip clips with NaNs/infs; record for the manifest log.
                print(f"  [skip non-finite] {subset}/{seq_id}")
                continue
            init_coarse_v1 = coarse_v1[0].astype(np.float32)            # (23,)

            safe_id = _safe_filename(seq_id)
            npz_path = subset_dir / f"{safe_id}.npz"
            np.savez_compressed(
                npz_path,
                coarse_v1=coarse_v1,
                init_coarse_v1=init_coarse_v1,
                # NOTE: only Stage-1-safe scalars / arrays. NO object_*,
                # NO plan_*, NO z_int / contact / phase / support / hand
                # / foot fields.
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
                "channel_names": COARSE_V0_NAMES + COARSE_V1_EXTRA_NAMES,
            }
            if split == "train":
                manifest_train.append(record)
                per_subset_train_counts[subset] += 1
                kept_train += 1
                train_global_concat.append(coarse_v1)
                per_subset_train_concat[subset].append(coarse_v1)
            else:
                manifest_val.append(record)
                per_subset_val_counts[subset] += 1
                kept_val += 1
            text_seen[seq_id] = text

        print(
            f"[cache] [{subset}] kept train={kept_train} val={kept_val}  "
            f"(short={skipped_short} split_skipped={skipped_split} pre-cap={len(indices)})"
        )

    # -----------------------------------------------------------------------
    # Manifests
    # -----------------------------------------------------------------------
    def _write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
        with path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    train_path = out_root / "manifest_train.jsonl"
    val_path = out_root / "manifest_val.jsonl"
    _write_jsonl(manifest_train, train_path)
    _write_jsonl(manifest_val, val_path)

    # -----------------------------------------------------------------------
    # Train-only normalization stats (and diagnostic per-subset stats).
    # -----------------------------------------------------------------------
    train_concat = (
        np.concatenate(train_global_concat, axis=0)
        if train_global_concat else np.zeros((0, COARSE_V1_DIM), dtype=np.float32)
    )
    if train_concat.shape[0] == 0:
        raise SystemExit("[cache] No train clips produced — cannot compute normalization.")

    global_stats = _z_score_stats(train_concat, eps=args.std_eps)
    per_subset_stats: dict[str, dict[str, Any]] = {}
    for subset, blocks in per_subset_train_concat.items():
        if not blocks:
            continue
        per_subset_stats[subset] = _z_score_stats(
            np.concatenate(blocks, axis=0), eps=args.std_eps
        )

    norm = {
        "split": "train",
        "cache_version": args.cache_version,
        "channel_names": COARSE_V0_NAMES + COARSE_V1_EXTRA_NAMES,
        "n_dims": int(COARSE_V1_DIM),
        "n_train_clips": len(manifest_train),
        "n_train_frames": int(train_concat.shape[0]),
        "global": global_stats,
        "per_subset": per_subset_stats,
        "fps": float(args.fps),
        "config_source": str(args.config),
        "motion_representation": args.motion_representation,
        "subject_split_used": splits is not None,
    }
    norm_path = out_root / "normalization_train.json"
    norm_path.write_text(json.dumps(norm, indent=2), encoding="utf-8")

    # Sanity: round-trip normalize / denormalize on a small batch.
    sample_batch = train_concat[: min(2048, train_concat.shape[0])]
    mean = np.asarray(global_stats["mean"], dtype=np.float32)
    std = np.asarray(global_stats["std_clamped"], dtype=np.float32)
    norm_batch = (sample_batch - mean) / std
    denorm_batch = norm_batch * std + mean
    rt_err = float(np.max(np.abs(sample_batch - denorm_batch)))
    print(f"[cache] normalization round-trip max |x - x_back| = {rt_err:.3e}")

    # -----------------------------------------------------------------------
    # README contract
    # -----------------------------------------------------------------------
    is_smoke = args.max_per_subset >= 0
    scope_block = (
        (
            "> **SCOPE: Smoke / preflight only — NOT a variance-protocol cache.**\n"
            f"> Built with `--max-per-subset {args.max_per_subset}`; some subsets\n"
            "> may have zero val coverage under the subject-split policy. Run\n"
            "> the builder with `--max-per-subset -1` to produce a full /\n"
            "> variance-protocol cache.\n"
        )
        if is_smoke
        else (
            "> **SCOPE: Full cache (no per-subset cap).** Eligible for variance-\n"
            "> protocol training after passing\n"
            "> `scripts/stage_b_generator/verify_stage1_cache_for_variance_protocol.py`.\n"
        )
    )
    readme = out_root / "README_cache_contract.md"
    readme.write_text(
        f"""# Stage-1 Coarse-v1 cache contract

{scope_block}
> See `analyses/2026-05-23_stage1_cache_and_config_contract.md` for the
> Stage-1 cache + config contract this README is one half of.

Cache version: `{args.cache_version}`
Built: {time.strftime("%Y-%m-%dT%H:%M:%S")}
Source config: `{args.config}`
Motion representation: `{args.motion_representation}` (plan-free; NO plan
compilation invoked).
FPS: {args.fps}
Coarse-v1 dim: {COARSE_V1_DIM}
Subject split enabled: {splits is not None}
Per-subset cap: {args.max_per_subset}

## Channels

{chr(10).join(f"- `[{i}]` {n}" for i, n in enumerate(COARSE_V0_NAMES + COARSE_V1_EXTRA_NAMES))}

## What each clip .npz contains

- `coarse_v1: (T, 23) float32` — unnormalized Coarse-v1.
- `init_coarse_v1: (23,) float32` — `coarse_v1[0]`.

That is the entire payload. The cache deliberately excludes:

- object trajectory, object point cloud, object_id, object tokens
- z_int (contact_state, contact_target_xyz, phase, support)
- interaction plan (plan_anchor_*, plan_segment_*)
- pseudo labels, hand/foot targets, future-anchor coords
- raw motion_135 / motion_263 (Stage-1 trains on Coarse-v1 only)

### Metadata source of truth = manifest (Option A)

Per the Round-13 cache contract decision (see
`analyses/2026-05-23_stage1_cache_and_config_contract.md`):

- the per-clip `.npz` carries ONLY numeric float32 arrays
  (`coarse_v1`, `init_coarse_v1`);
- every other field — `subset`, `seq_id`, `safe_seq_id`, `seq_len`,
  `text`, `split`, `cache_version`, `fps`, `coarse_v1_dim`,
  `channel_names` — lives in the JSONL manifest entry;
- `Stage1CacheDataset` reads metadata from the manifest and arrays
  from the `.npz`. Treat the manifest as canonical.

The text and metadata live in the manifest, not in the .npz.

## Manifests

- `manifest_train.jsonl`: one JSON-line per train clip with
  `subset, seq_id, safe_seq_id, npz_path, seq_len, text, split,
  cache_version, fps, coarse_v1_dim, channel_names`.
- `manifest_val.jsonl`: identical schema for the val split.

The `split` field is derived from the project subject-split policy
in the source config's `data.subject_split` section.

## Normalization

`normalization_train.json` holds:

- `global.mean: (23,)`, `global.std: (23,)`, `global.std_clamped: (23,)`
  computed from the TRAIN manifest concatenated frames only.
- `per_subset.{{chairs,imhd,neuraldome,omomo_correct_v2}}.{{mean,std,...}}`
  are diagnostic only; the trainer must use `global.{{mean,std_clamped}}`.
- The val split was NEVER touched while computing these stats.
- The Round-9 24-clip audit selection was NEVER touched while
  computing these stats (the cache builder iterates the FULL HOIDataset,
  unrelated to `analyses/2026-05-19_subset_balanced_failure_selection.json`).

## Provenance

- The 6D-rotation-to-matrix helper used during extraction was changed
  in Round 12 to the project-local
  `piano.training.smpl_kinematics.rotation_6d_to_matrix`. See
  `analyses/2026-05-22_stage1_coarse_prior_preflight_smoke_report.md`
  for the equivalence test and the impact of the fix.
- The plan-compile branch in `src/piano/data/dataset.py` is gated by
  `motion_representation == "smpl_pose_135_plan"`. By choosing
  `smpl_pose_135` for this cache, plan compilation is bypassed
  by static control flow (no runtime dependency on plan-related
  configuration or files).
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
        "n_train_frames": int(train_concat.shape[0]),
        "global_mean_first5": [float(x) for x in mean[:5]],
        "global_std_first5": [float(x) for x in std[:5]],
        "rt_err": rt_err,
        "elapsed_seconds": float(elapsed),
        "max_per_subset": args.max_per_subset,
        "motion_representation": args.motion_representation,
    }
    (out_root / "build_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print()
    print(f"[cache] wrote manifest_train.jsonl  n={len(manifest_train)}")
    print(f"[cache] wrote manifest_val.jsonl    n={len(manifest_val)}")
    print(f"[cache] wrote normalization_train.json")
    print(f"[cache] wrote README_cache_contract.md")
    print(f"[cache] wrote build_summary.json")
    print(f"[cache] elapsed {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
