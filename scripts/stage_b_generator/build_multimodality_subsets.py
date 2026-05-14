"""Build controlled-multimodality subsets for Stage B P1 diagnostic.

Per analyses/stageB_root_cause_analysis_v2_and_next_strategy.md §6 (P1).
Four 2×2 designs at fixed N≈100:

    S1: same object   + same action     (minimal multimodality)
    S2: same object   + different action
    S3: different obj + same action
    S4: different obj + different action (maximum multimodality)

If S1 produces dynamic motion at sampling while S4 collapses, multimodality
averaging is THE root cause and we move to mode-collapse-resistant fixes.

Construction heuristics (necessarily coarse — InterAct doesn't ship action
labels):
- "object" axis  = `object_id` field
- "action" axis  = first verb of the `text` description after "a person ".

This is reproducible and grounded in the metadata; absolute action separation
is not perfect but separates the major motion classes (lift, sit, push, hold,
swing, …).

Anchor choices:
- "Anchor object" = the object with the most train-bucket clips in InterAct
  after subsample_n_per_object=10 + subject_split. Per analysis below this is
  `omomo whitechair` (10 clips, the maximum N per object under subsample_n=10).
- "Anchor action" = the dominant action verb across the train bucket. Per
  analysis: `lift` (counts dominate omomo / chairs).

Since subsample_n_per_object=10 caps clips per object at 10, no single object
has 100 clips in the train bucket. Therefore N=100 strict per-object subsets
are infeasible at this subsample level. Strategy:
- S1 (same obj + same act): widen "object" to a small cluster of similar
  objects (whitechair + woodchair, both omomo, both chair-shaped objects)
  + filter to "lift" action.
- S2 (same obj cluster, diff actions): chair cluster, all actions.
- S3 (diff obj, same act): all objects, "lift" only.
- S4 (diff obj, diff act): all objects, all actions = random 100 from
  the existing scale-curve seed-42 shuffle.

All subsets are N=100 (or as close as the data allows). Output: one JSON per
subset under `data/subsets/stageB_S{1,2,3,4}_N100.json` with:
- `indices`: list of train-bucket dataset indices (0..618)
- `clips`: per-clip (subset, seq_id, object_id, action_verb, text)
- summary stats

Usage::

    python scripts/stage_b_generator/build_multimodality_subsets.py \\
        --config configs/training/anchordiff_v12_dit_block_no_planpool_FULL_N10.yaml \\
        --target-n 100 \\
        --output data/subsets/
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from omegaconf import OmegaConf

from piano.data.dataset import HOIDataset, build_subject_split, extract_subject_id
from piano.utils.io_utils import load_json


# Anchor object cluster (chair-shaped, omomo)
ANCHOR_OBJECT_CLUSTER = {"whitechair", "woodchair"}
ANCHOR_ACTION = "lift"


def _build_dataset(cfg, bucket: str, override_subsample_n: int | None = None):
    from torch.utils.data import ConcatDataset
    subj_filter: set | None = None
    subj_cfg = cfg.data.get("subject_split")
    if subj_cfg is not None and subj_cfg.get("enabled", False):
        keys: set[tuple[str, str]] = set()
        for entry in cfg.data.datasets:
            meta = load_json(Path(entry.root) / "metadata_clean.json")
            for m in meta:
                sid = extract_subject_id(Path(entry.root).name, m.get("seq_id", ""))
                if sid is not None:
                    keys.add((Path(entry.root).name, sid))
        splits = build_subject_split(
            sorted(keys),
            train_pct=subj_cfg.train_pct,
            val_pct=subj_cfg.val_pct,
            seed=subj_cfg.seed,
        )
        subj_filter = splits[bucket]
    pseudo_label_subdir = cfg.data.get("pseudo_label_subdir", None)
    datasets = []
    for entry in cfg.data.datasets:
        sub_dir = (
            str(Path(entry.root) / pseudo_label_subdir) if pseudo_label_subdir else None
        )
        ds = HOIDataset(
            root=entry.root,
            pseudo_label_dir=sub_dir,
            max_seq_length=int(cfg.data.max_seq_length),
            subject_id_filter=subj_filter,
            subsample_n_per_object=(
                override_subsample_n
                if override_subsample_n is not None
                else cfg.data.get("subsample_n_per_object", None)
            ),
            subsample_seed=int(cfg.data.get("subsample_seed", 42)),
            support_collapse_hand_support=bool(
                cfg.data.get("support_collapse_hand_support", True)
            ),
            surface_obj_pose=True,
            force_world_frame=bool(cfg.data.get("force_world_frame", False)),
            motion_representation=str(cfg.data.get("motion_representation", "motion_263")),
        )
        datasets.append(ds)
    return ConcatDataset(datasets)


def extract_action_verb(text: str) -> str:
    """Return the first verb-like word from the text, lowercased."""
    if not text:
        return ""
    t = text.lower().strip()
    t = re.sub(r"^(a |the )?person ", "", t)
    words = t.split()
    if not words:
        return ""
    w = words[0].strip(",.")
    # Normalize plural verbs (sits → sit, lifts → lift, picks → pick)
    if w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    return w


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--target-n", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("data/subsets"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-subsample", action="store_true",
        help="Disable subsample_n_per_object (use all ~7000 train-bucket clips). "
             "Needed because S1 (same object+action) needs >100 clips per cluster, "
             "and subsample_n_per_object=10 caps at 10/object.",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    override_subsample = 1_000_000 if args.no_subsample else None
    train_dataset = _build_dataset(cfg, bucket="train", override_subsample_n=override_subsample)
    n_total = len(train_dataset)
    print(f"Total train-bucket clips: {n_total}")

    # Index → metadata
    metas: list[dict] = []
    for global_idx in range(n_total):
        ds_idx = 0
        local_idx = global_idx
        for cum in train_dataset.cumulative_sizes:
            if global_idx < cum:
                break
            ds_idx += 1
        if ds_idx > 0:
            local_idx = global_idx - train_dataset.cumulative_sizes[ds_idx - 1]
        ds = train_dataset.datasets[ds_idx]
        m = ds.metadata[local_idx]
        subset_name = Path(ds.root).name
        seq_id = str(m.get("seq_id", ""))
        text = str(m.get("text", ""))
        object_id = str(m.get("object_id", "_unknown"))
        action = extract_action_verb(text)
        metas.append({
            "global_idx": global_idx,
            "subset": subset_name,
            "seq_id": seq_id,
            "object_id": object_id,
            "action": action,
            "text": text[:120],
        })

    # Anchor object cluster: in the omomo subset, whitechair/woodchair are
    # chair-shaped objects. After subsample_n_per_object=10 each has at most
    # 10 clips → cluster has ≤20 clips total. Diagnose available counts:
    anchor_obj_counts = Counter()
    anchor_act_counts = Counter()
    cluster_counts = Counter()
    action_counts = Counter()
    for m in metas:
        if m["subset"] == "omomo_correct_v2":
            anchor_obj_counts[m["object_id"]] += 1
        anchor_act_counts[m["action"]] += 1
        action_counts[m["action"]] += 1
        if m["object_id"] in ANCHOR_OBJECT_CLUSTER:
            cluster_counts[(m["object_id"], m["action"])] += 1
    print(f"\nTop omomo objects in train bucket: "
          f"{anchor_obj_counts.most_common(8)}")
    print(f"Top action verbs in train bucket: "
          f"{anchor_act_counts.most_common(10)}")
    print(f"\n(anchor_obj_cluster, action) breakdown: "
          f"{dict(cluster_counts)}")

    # Build the 4 subsets
    rng = random.Random(int(args.seed))
    target_n = int(args.target_n)

    def _sample(pool: list[dict], k: int, label: str) -> list[dict]:
        if len(pool) <= k:
            print(f"  [{label}] only {len(pool)} clips match; using all (target {k})")
            return list(pool)
        return rng.sample(pool, k)

    # S1: anchor object cluster + anchor action (most uniform)
    pool_s1 = [m for m in metas if m["object_id"] in ANCHOR_OBJECT_CLUSTER
               and m["action"] == ANCHOR_ACTION]
    s1 = _sample(pool_s1, target_n, "S1")

    # S2: anchor object cluster + ANY action (same-object varied actions)
    pool_s2 = [m for m in metas if m["object_id"] in ANCHOR_OBJECT_CLUSTER]
    s2 = _sample(pool_s2, target_n, "S2")

    # S3: ANY object + anchor action (varied objects, same action)
    pool_s3 = [m for m in metas if m["action"] == ANCHOR_ACTION]
    s3 = _sample(pool_s3, target_n, "S3")

    # S4: ANY object + ANY action (max diversity)
    s4 = _sample(list(metas), target_n, "S4")

    print()
    print(f"{'subset':>6}  {'n':>4}  {'#objs':>6}  {'#actions':>9}  description")
    print("-" * 70)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, clips in (("S1", s1), ("S2", s2), ("S3", s3), ("S4", s4)):
        n_objs = len({m["object_id"] for m in clips})
        n_acts = len({m["action"] for m in clips})
        # The TRAINER picks clips via overfit_n_clips + scale_subset_seed.
        # To pin a specific subset we'll need to save the indices and have
        # the trainer load them. For now we write the JSON and add a separate
        # `subset_indices_file` mechanism to the trainer.
        out_path = args.output / f"stageB_{name}_N{target_n}.json"
        out_path.write_text(
            json.dumps({
                "subset_name": name,
                "target_n": target_n,
                "actual_n": len(clips),
                "anchor_object_cluster": sorted(ANCHOR_OBJECT_CLUSTER),
                "anchor_action": ANCHOR_ACTION,
                "seed": int(args.seed),
                "indices": [m["global_idx"] for m in clips],
                "clips": clips,
                "unique_objects": sorted({m["object_id"] for m in clips}),
                "unique_actions": sorted({m["action"] for m in clips}),
            }, indent=2),
            encoding="utf-8",
        )
        print(f"{name:>6}  {len(clips):>4}  {n_objs:>6}  {n_acts:>9}")

    print(f"\nWrote 4 subset files to {args.output}")


if __name__ == "__main__":
    main()
