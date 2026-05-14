"""Build deterministic scale-curve subsets for Stage B scale-curve diagnostic.

Per analyses/stageB_root_cause_analysis_v2_and_next_strategy.md §5 + §5.1: writes
JSON files documenting which clips belong to each scale-curve subset
(1 / 5 / 10 / 50 / 100 / 700 clips). The TRAINER applies the same shuffle at
runtime via `data.scale_subset_seed` so the actual subsets match these files
exactly without needing to read them from disk.

Subsets are NESTED: scale_001 ⊂ scale_010 ⊂ scale_100 ⊂ scale_700 (same seed
truncated at different points), so the scale curve is a clean monotone sequence.

Usage::

    python scripts/stage_b_generator/build_scale_subsets.py \\
        --config configs/training/anchordiff_v12_dit_block_no_planpool_FULL_N10.yaml \\
        --seed 42 \\
        --output data/subsets/

Prints a summary table; writes one JSON per scale.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from omegaconf import OmegaConf
from torch.utils.data import Subset

from piano.data.dataset import (
    HOIDataset, build_subject_split, extract_subject_id,
)
from piano.utils.io_utils import load_json


def _build_dataset(cfg, bucket: str):
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
            subsample_n_per_object=cfg.data.get("subsample_n_per_object", None),
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


SCALES = [1, 5, 10, 50, 100, 700]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, required=True,
        help="A trainer-style yaml (e.g. anchordiff_v12_dit_block_no_planpool_FULL_N10.yaml). "
             "Used to load the same data pipeline (subsample_n_per_object, subject_split, etc.).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Shuffle seed. Same value used in trainer's data.scale_subset_seed for consistency.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/subsets"),
        help="Output directory for the scale_XXX.json files.",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    train_dataset = _build_dataset(cfg, bucket="train")
    n_total = len(train_dataset)
    print(f"Total train-bucket clips: {n_total}")

    # Index → (subset, seq_id, text)
    metas: list[dict] = []
    # ConcatDataset stores .datasets and .cumulative_sizes; index into the concat
    # peels off the right child dataset's clip metadata.
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
        text = str(m.get("text_description", m.get("caption", "")))[:80]
        object_id = str(m.get("object_id", "_unknown"))
        metas.append({
            "global_idx": global_idx,
            "subset": subset_name,
            "seq_id": seq_id,
            "object_id": object_id,
            "text": text,
        })

    # Shuffle with the seed (matches trainer's Random(seed).shuffle).
    indices = list(range(n_total))
    rng = random.Random(int(args.seed))
    rng.shuffle(indices)

    args.output.mkdir(parents=True, exist_ok=True)

    # Subset diversity stats (per scale: number of unique objects + datasets)
    print()
    print(f"{'scale':>6}  {'n':>4}  {'unique_objects':>15}  {'unique_subsets':>15}")
    print("-" * 50)
    for N in SCALES:
        N_eff = min(N, n_total)
        picked = [metas[indices[i]] for i in range(N_eff)]
        unique_objects = {m["object_id"] for m in picked}
        unique_subsets = {m["subset"] for m in picked}
        out_path = args.output / f"stageB_scale_{N:03d}.json"
        out_path.write_text(
            json.dumps({
                "seed": int(args.seed),
                "scale": N_eff,
                "config_basis": str(args.config),
                "indices": indices[:N_eff],
                "clips": picked,
                "unique_objects": sorted(unique_objects),
                "unique_subsets": sorted(unique_subsets),
            }, indent=2),
            encoding="utf-8",
        )
        print(f"{N:>6}  {N_eff:>4}  {len(unique_objects):>15}  {len(unique_subsets):>15}")
    print(f"\nWrote {len(SCALES)} subset files to {args.output}")
    print(f"To use in training: add `data.scale_subset_seed: {args.seed}` to the yaml")
    print(f"and `data.overfit_n_clips: N` to pick the first N shuffled clips.")


if __name__ == "__main__":
    main()
