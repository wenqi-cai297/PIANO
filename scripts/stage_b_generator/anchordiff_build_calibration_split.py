"""Build a fixed calibration split for AnchorDiff M1.5 dynamic-weight updates.

Per PLAN.md M1.5 Step 1: dynamic weights must track stable residuals,
not noisy random train batches. So we freeze a 80-160-clip manifest
once and use the same clips at every weight update during training.

Selection policy:
- Prefer val-bucket clips (subject_split bucket=val) so we don't
  contaminate the train signal.
- Balance counts across the 4 active subsets.
- Include drift-sensitive / tool-like clips: bat / racket / suitcase
  / chair sit, which are the worst-case root-rotation generators.
- Save manifest with seq_id + subset + sample idx for reproducibility.

Usage:
    python scripts/stage_b_generator/anchordiff_build_calibration_split.py \\
        --config configs/training/anchordiff_v2_weighted.yaml \\
        --num-clips 120 \\
        --output analyses/2026-05-08_anchordiff_dynamic_metric
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from omegaconf import OmegaConf

from piano.data.dataset import HOIDataset, build_subject_split, extract_subject_id
from piano.utils.io_utils import load_json


# Heuristic patterns for "tool-like" / "drift-sensitive" clips.
TOOL_PATTERNS = (
    re.compile(r"(?i)bat"),
    re.compile(r"(?i)racket|tennis"),
    re.compile(r"(?i)suitcase|trolley"),
    re.compile(r"(?i)baseball"),
    re.compile(r"(?i)chair|sit"),
)


def _read_metadata(roots: list) -> list[tuple[str, dict]]:
    out = []
    for entry in roots:
        root = Path(entry.root)
        meta_path = root / "metadata_clean.json"
        if not meta_path.exists():
            meta_path = root / "metadata.json"
        for m in load_json(meta_path):
            out.append((root.name, m))
    return out


def _is_tool_like(seq_id: str, text: str) -> bool:
    s = f"{seq_id} {text}".lower()
    return any(p.search(s) for p in TOOL_PATTERNS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num-clips", type=int, default=120)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    rng = random.Random(args.seed)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reproduce the val subject split exactly.
    subj_cfg = cfg.data.subject_split
    keys = sorted({
        (subset, extract_subject_id(subset, m.get("seq_id", "")))
        for subset, m in _read_metadata(cfg.data.datasets)
        if extract_subject_id(subset, m.get("seq_id", "")) is not None
    })
    splits = build_subject_split(
        keys,
        train_pct=subj_cfg.train_pct,
        val_pct=subj_cfg.val_pct,
        seed=subj_cfg.seed,
    )
    val_filter = splits["val"]
    print(f"val subject_split: {len(val_filter)} subjects")

    # Build per-subset val-only metadata pool, with sample index → seq_id mapping.
    pseudo_label_subdir = cfg.data.get("pseudo_label_subdir", None)
    pool_per_subset: dict[str, list[dict]] = {}
    for entry in cfg.data.datasets:
        sub_dir = (str(Path(entry.root) / pseudo_label_subdir)
                   if pseudo_label_subdir is not None else None)
        ds = HOIDataset(
            root=entry.root,
            pseudo_label_dir=sub_dir,
            max_seq_length=cfg.data.max_seq_length,
            subject_id_filter=val_filter,
            augment=None,
            support_collapse_hand_support=True,
            surface_obj_pose=True,
        )
        # We DON'T iterate the full HOIDataset (would require __getitem__ load
        # for each clip). Instead pull from its metadata directly.
        rows = []
        for clip_idx in range(len(ds)):
            meta = ds.metadata[clip_idx] if hasattr(ds, "metadata") else None
            if meta is None:
                # Fall back to one __getitem__ to peek at seq_id.
                sample = ds[clip_idx]
                seq_id = str(sample["seq_id"])
                text = str(sample["text"])
            else:
                seq_id = str(meta.get("seq_id", ""))
                text = str(meta.get("text", ""))
            rows.append({
                "subset": entry.name,
                "seq_id": seq_id,
                "text": text,
                "clip_idx_in_filtered_dataset": clip_idx,
                "is_tool_like": _is_tool_like(seq_id, text),
            })
        pool_per_subset[entry.name] = rows
        print(f"  {entry.name}: {len(rows)} val clips")

    # Allocate per-subset quotas (balanced).
    subsets = list(pool_per_subset.keys())
    per_subset_quota = max(args.num_clips // len(subsets), 4)
    n_tool_target = max(per_subset_quota // 3, 2)   # at least ~33% tool-like

    selected: list[dict] = []
    for subset, rows in pool_per_subset.items():
        tool_rows = [r for r in rows if r["is_tool_like"]]
        nontool_rows = [r for r in rows if not r["is_tool_like"]]
        rng.shuffle(tool_rows)
        rng.shuffle(nontool_rows)
        n_tool = min(n_tool_target, len(tool_rows))
        n_remain = per_subset_quota - n_tool
        chosen = tool_rows[:n_tool] + nontool_rows[:n_remain]
        rng.shuffle(chosen)
        selected.extend(chosen)
        print(f"  {subset}: picked {n_tool} tool-like + {n_remain} regular = {len(chosen)}")

    print(f"Total calibration manifest: {len(selected)} clips")

    manifest = {
        "generated_at": "2026-05-08",
        "config": str(args.config),
        "seed": args.seed,
        "num_clips": len(selected),
        "selection_policy": {
            "bucket": "val",
            "balanced_across_subsets": True,
            "tool_like_quota_per_subset": n_tool_target,
            "tool_patterns": [p.pattern for p in TOOL_PATTERNS],
        },
        "clips": selected,
    }
    out_path = out_dir / "calibration_clips.json"
    out_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
