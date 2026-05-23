"""Round-25 D4 helper: convert D1 multimodal eval subset JSON →
``subset_indices_file`` JSON for the trainer's overfit path.

Per Codex audit
(analyses/2026-05-23_codex_round25_p0_implementation_review.md §5 E1):
``overfit_n_clips: 8`` without ``scale_subset_seed`` takes the FIRST
N clips of the train bucket alphabetically — NOT the multimodal
clips D4 is meant to overfit. The trainer DOES support
``data.subset_indices_file`` which takes precedence over
``overfit_n_clips`` and reads explicit indices from a JSON file
(train_anchordiff.py:1707-1718).

This script:
  1. Loads the D1 selection JSON.
  2. Walks the v26 train-bucket dataset to find matching
     (subset, seq_id) entries.
  3. Emits a JSON with ``{"indices": [...]}`` that D4 configs
     reference via ``data.subset_indices_file``.

Two output files: 8-clip and 16-clip variants.

Usage:
    conda run -n piano python scripts/stage_b_generator/round25_d4_build_subset_indices.py \
        --config configs/training/anchordiff_v26_FULL_DATA_local.yaml \
        --selection-json analyses/round25_multimodal_eval_subset.json \
        --n-clips 8 \
        --output analyses/round25_d4_indices_8.json

    (and again with --n-clips 16 --output analyses/round25_d4_indices_16.json)

    NOTE: use --n-clips (not bare --n). Argparse abbreviation matching
    inside `conda run` greedily matches `--n` to `--name`, so the
    canonical name is --n-clips.

Note: D1 selection is the val-bucket clip list. D4 overfit needs
TRAIN-bucket indices. So we re-curate from D1 val candidates by
finding any (subset, seq_id) match in the TRAIN bucket. If the same
seq_id doesn't exist in train (it usually won't — subject splits are
disjoint), we pick the closest semantic match: clips with the same
mode_category from the train bucket. The user can also pass
``--train-selection-json`` directly with a hand-curated 8/16-clip
train-bucket list.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from plan_condition_diagnostics import _build_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path,
                        help="D1 val-bucket selection JSON.")
    parser.add_argument("--train-selection-json", type=Path,
                        help="Optional: pre-curated TRAIN-bucket selection JSON. "
                             "If provided, --selection-json is ignored and the train "
                             "list is used directly.")
    parser.add_argument("--n-clips", "--n", type=int, default=8,
                        dest="n",
                        help="Number of train clips to include (8 or 16). "
                             "Use --n-clips (canonical) — `--n` works as an "
                             "alias inside `python` but is ambiguous with "
                             "`conda run --name`, so the launcher uses "
                             "--n-clips.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--match-by-category", action="store_true",
                        help="If set and --selection-json is given (but not "
                             "--train-selection-json), pick train clips whose "
                             "mode_category matches the val list (one per "
                             "category up to --n).")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    train_ds = _build_dataset(cfg, "train", augment=False)
    print(f"[d4-build] train bucket size = {len(train_ds)}")

    # Build train-bucket (subset, seq_id) → index map.
    train_map: dict[tuple[str, str], int] = {}
    for i in range(len(train_ds)):
        sample = train_ds[i]
        train_map[(str(sample["subset"]), str(sample["seq_id"]))] = i

    if args.train_selection_json is not None:
        sel = json.loads(args.train_selection_json.read_text("utf-8"))
        clips = sel.get("selected", sel.get("candidates", []))
        indices: list[int] = []
        missing: list[dict] = []
        for c in clips[: args.n]:
            key = (c["subset"], c["seq_id"])
            if key in train_map:
                indices.append(train_map[key])
            else:
                missing.append(c)
        if missing:
            print(f"[d4-build] WARNING: {len(missing)} clips missing from train bucket:")
            for c in missing:
                print(f"  - {c['subset']}/{c['seq_id']}")
        clip_records = [{"subset": c["subset"], "seq_id": c["seq_id"],
                         "mode_category": c.get("mode_category",
                                                 c.get("mode_category_guess", "unknown"))}
                        for c in clips[: args.n] if (c["subset"], c["seq_id"]) in train_map]
    elif args.selection_json is not None:
        # Fallback heuristic: pick train clips with matching mode_category.
        # The exact val clips will NOT be in train (subject split is
        # disjoint), so we pick from train by category match.
        if not args.match_by_category:
            raise SystemExit(
                "[d4-build] --selection-json is a val list; train clips will not "
                "match by (subset, seq_id). Pass --match-by-category to pick by "
                "mode_category instead, or supply a hand-curated --train-selection-json."
            )
        val_sel = json.loads(args.selection_json.read_text("utf-8"))
        val_clips = val_sel.get("selected", val_sel.get("candidates", []))
        # Bucket val clips by category.
        val_by_cat: dict[str, list[dict]] = {}
        for c in val_clips:
            cat = c.get("mode_category", c.get("mode_category_guess", "unknown"))
            val_by_cat.setdefault(cat, []).append(c)
        # For each category present in val, pick the first train clip whose
        # text contains a category keyword. This is approximate — D1's
        # text-keyword scoring is the canonical proposer.
        # Reuse the same keyword table from D1.
        from round25_d1_propose_multimodal_candidates import _CATEGORY_KEYWORDS

        chosen: list[tuple[int, str, str, str]] = []  # (train_idx, subset, seq_id, category)
        seen_train: set[int] = set()
        # Iterate categories in a fixed order so output is deterministic.
        ordered_cats = sorted(val_by_cat.keys())
        # Round-robin across categories until we hit args.n.
        cat_iters: dict[str, int] = {c: 0 for c in ordered_cats}
        while len(chosen) < args.n and any(
            cat_iters[c] < len(val_by_cat[c]) for c in ordered_cats
        ):
            for cat in ordered_cats:
                if len(chosen) >= args.n:
                    break
                # Find a train clip with text matching this category's keywords.
                kws = [kw.lower() for kw in _CATEGORY_KEYWORDS.get(cat, [])]
                if not kws:
                    cat_iters[cat] += 1
                    continue
                # Linear scan train; pick first not-yet-chosen match.
                found = False
                for i in range(len(train_ds)):
                    if i in seen_train:
                        continue
                    sample = train_ds[i]
                    text = str(sample["text"]).lower()
                    if any(kw in text for kw in kws):
                        chosen.append((i, str(sample["subset"]),
                                       str(sample["seq_id"]), cat))
                        seen_train.add(i)
                        found = True
                        break
                cat_iters[cat] += 1
                if not found:
                    # Category exhausted in train.
                    pass

        indices = [c[0] for c in chosen]
        clip_records = [
            {"subset": c[1], "seq_id": c[2], "mode_category": c[3]} for c in chosen
        ]
    else:
        raise SystemExit("[d4-build] need --selection-json or --train-selection-json")

    if len(indices) < args.n:
        print(f"[d4-build] WARNING: only found {len(indices)} train clips "
              f"(requested {args.n}). Consider hand-curating a train-bucket "
              f"selection JSON with diverse mode_category coverage.")

    out = {
        "description": (
            "Round-25 D4 overfit subset indices. Selected by "
            f"{'train-selection-json' if args.train_selection_json else 'category-match heuristic'}."
        ),
        "source_config": str(args.config),
        "source_selection_json": str(args.train_selection_json or args.selection_json),
        "n_requested": args.n,
        "n_found": len(indices),
        "indices": indices,
        "clips": clip_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    print(f"[d4-build] wrote {len(indices)} indices to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
