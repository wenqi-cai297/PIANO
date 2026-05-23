"""Round-25 curation helper — pick the final multimodal eval subsets
from the D1 candidate JSON. Run after round25_d1_propose_multimodal_candidates.py.

Picks ~50 val clips for the multimodal eval subset, weighted by
category quality and discriminating power. Also picks ~16 train
clips for the D4 overfit (mirror_swing-heavy because train has lots
of omomo whitechair repeats and val crossed_legs entries don't appear
in train due to subject split).

Selection logic (val):
    crossed_legs: 12   (gold standard — explicit limb-side text)
    mirror_swing: 8    (cw/ccw swing — dedupe omomo by subject)
    circular_direction: 8  (cw vs ccw walking around object)
    hand_choice: 12    (left/right hand in pick/lift/push — text usually fixes side but model may average)
    sit_pose: 10       (under-specified seated poses — generic "shifts around" etc.)

Train selection follows a similar logic but from a train-bucket run
of D1.

Usage:
    conda run -n piano python scripts/stage_b_generator/round25_curate_subsets.py \
        --val-candidates analyses/round25_multimodal_candidates_val.json \
        --val-output     analyses/round25_multimodal_eval_subset.json \
        --train-candidates analyses/round25_multimodal_candidates_train.json \
        --train-output     analyses/round25_d4_train_selection.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def _dedup_by_key(items: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for c in items:
        k = (c["subset"], c["seq_id"])
        if k in seen:
            continue
        seen.add(k)
        out.append(c)
    return out


def _base_seq_id(subset: str, seq_id: str) -> str:
    """Collapse sub-segment cuts of the same physical clip without
    conflating DIFFERENT frame-windows of the same recording.

    Patterns we DO collapse (sub-segment suffix only):
        ``Sub0087_Obj24_Seg0_0_0`` → ``Sub0087_Obj24_Seg0_0``
        ``Sub0972_Obj81_Seg0_210_1`` → ``Sub0972_Obj81_Seg0_210``
        ``..._bat_bat_holdhead_hit_1_0_2`` → ``..._bat_bat_holdhead_hit_1_0``

    Patterns we DO NOT collapse (different frame windows):
        ``Sub0598_Obj118_Seg0_180`` ≠ ``Sub0598_Obj118_Seg0_600``
        (different starts of the SAME segment but possibly different motion).

    Rule: strip only if there are TWO numeric tails. ``_\d+_\d+`` →
    strip last ``_\d+``. Otherwise leave alone.
    """
    m = re.match(r"(.*_\d+)_\d+$", seq_id)
    if m:
        return f"{subset}::{m.group(1)}"
    return f"{subset}::{seq_id}"


def _dedup_by_base(items: list[dict], max_per_base: int = 1) -> list[dict]:
    """Keep at most max_per_base entries per (subset, base_seq_id)."""
    seen: dict[str, int] = {}
    out = []
    for c in items:
        base = _base_seq_id(c["subset"], c["seq_id"])
        if seen.get(base, 0) >= max_per_base:
            continue
        seen[base] = seen.get(base, 0) + 1
        out.append(c)
    return out


def _curate(candidates: list[dict], targets: dict[str, int],
            extra_filters: dict | None = None) -> list[dict]:
    """Pick `targets[cat]` clips per category from candidates.

    Filtering policy:
        - For mirror_swing on omomo: dedup by 'sub<NN>_' prefix so we
          don't have 5 copies of the same subject's whitechair clips.
        - For all categories: prefer high-confidence (0.8) hits first,
          then fall back to 0.5 hits.
        - Skip clips whose text is the literal placeholder noise.
    """
    # First dedup exact-duplicate (subset, seq_id) entries, then collapse
    # near-duplicate segment cuts of the same underlying clip.
    deduped = _dedup_by_base(_dedup_by_key(candidates), max_per_base=1)
    by_cat = defaultdict(list)
    for c in deduped:
        by_cat[c["mode_category_guess"]].append(c)

    out: list[dict] = []
    for cat, n_target in targets.items():
        items = by_cat.get(cat, [])
        # Sort: high confidence first, then by seq_id alpha.
        items.sort(key=lambda c: (-c["confidence"], c["subset"], c["seq_id"]))

        # Extra mirror_swing collapse: omomo whitechair clips all have
        # identical text "Grab the top of the whitechair...". Pick at
        # most 1 per (subject, object) so we don't take 5 clips with
        # identical text.
        if cat == "mirror_swing":
            seen_subj = set()
            collapsed = []
            for c in items:
                if c["subset"] == "omomo_correct_v2":
                    m = re.match(r"(sub\d+_\w+?)_\d+", c["seq_id"])
                    key = m.group(1) if m else c["seq_id"]
                    if key in seen_subj:
                        continue
                    seen_subj.add(key)
                collapsed.append(c)
            items = collapsed

        picked = items[:n_target]
        for c in picked:
            out.append({
                "subset": c["subset"],
                "seq_id": c["seq_id"],
                "text": c["text"],
                "mode_category": cat,
                "confidence": c["confidence"],
                "n_known_valid_modes": 2,   # default — user may refine
            })
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-candidates", type=Path,
                        default=Path("analyses/round25_multimodal_candidates_val.json"))
    parser.add_argument("--val-output", type=Path,
                        default=Path("analyses/round25_multimodal_eval_subset.json"))
    parser.add_argument("--train-candidates", type=Path,
                        default=Path("analyses/round25_multimodal_candidates_train.json"),
                        help="Optional D1 train-bucket candidates file.")
    parser.add_argument("--train-output", type=Path,
                        default=Path("analyses/round25_d4_train_selection.json"))
    args = parser.parse_args()

    # ---- VAL curation ----
    val_cands = json.loads(args.val_candidates.read_text("utf-8"))["candidates"]
    val_targets = {
        "crossed_legs": 12,
        "mirror_swing": 8,
        "circular_direction": 8,
        "hand_choice": 12,
        "sit_pose": 10,
    }
    val_selected = _curate(val_cands, val_targets)
    out_val = {
        "description": (
            "Round-25 multimodal eval subset (D1 output, human-curated). "
            "Targets clips where text is mode-ambiguous w.r.t. limb pose. "
            "Used by D2 diversity + D3 oracle-vs-sampled diagnostics."
        ),
        "selection_source": str(args.val_candidates),
        "bucket": "val",
        "n_clips": len(val_selected),
        "per_category_count": {
            cat: sum(1 for s in val_selected if s["mode_category"] == cat)
            for cat in val_targets
        },
        "selected": val_selected,
    }
    args.val_output.parent.mkdir(parents=True, exist_ok=True)
    args.val_output.write_text(json.dumps(out_val, indent=2, ensure_ascii=False),
                                encoding="utf-8")
    print(f"[curate] val: wrote {len(val_selected)} clips to {args.val_output}")
    print(f"[curate] val per-category: {out_val['per_category_count']}")

    # ---- TRAIN curation (if available) ----
    if args.train_candidates.exists():
        train_cands = json.loads(args.train_candidates.read_text("utf-8"))["candidates"]
        train_targets = {
            "mirror_swing": 4,
            # Train bucket has very few crossed_legs (e.g. 3 unique on
            # subject-split=85/15). Take whatever's available.
            "crossed_legs": 3,
            "circular_direction": 4,
            "hand_choice": 5,
        }
        train_selected_raw = _curate(train_cands, train_targets)
        # Round-robin across categories so any prefix of the list
        # (e.g. first 8 for D4-8) spans diverse mode_categories.
        by_cat_t: dict[str, list[dict]] = defaultdict(list)
        for c in train_selected_raw:
            by_cat_t[c["mode_category"]].append(c)
        train_selected: list[dict] = []
        cat_order = list(train_targets.keys())
        i = 0
        while sum(len(v) for v in by_cat_t.values()) > 0:
            cat = cat_order[i % len(cat_order)]
            i += 1
            if by_cat_t[cat]:
                train_selected.append(by_cat_t[cat].pop(0))
        out_train = {
            "description": (
                "Round-25 D4 train-bucket overfit selection — 16 clips "
                "spanning the same mode_category coverage as the val "
                "subset but from the train bucket (subject split is "
                "disjoint so val seq_ids do not appear here)."
            ),
            "selection_source": str(args.train_candidates),
            "bucket": "train",
            "n_clips": len(train_selected),
            "per_category_count": {
                cat: sum(1 for s in train_selected if s["mode_category"] == cat)
                for cat in train_targets
            },
            "selected": train_selected,
        }
        args.train_output.write_text(
            json.dumps(out_train, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[curate] train: wrote {len(train_selected)} clips to {args.train_output}")
        print(f"[curate] train per-category: {out_train['per_category_count']}")
    else:
        print(f"[curate] train candidates not found ({args.train_candidates}); "
              "run round25_d1_propose_multimodal_candidates.py --bucket train first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
