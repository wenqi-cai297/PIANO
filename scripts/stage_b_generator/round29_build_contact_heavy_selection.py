"""Build a contact-heavy non-chairs subset for visual review.

The default ``round27_tier0_train_indices_48_balanced.json`` and
``round29_val_diag_indices_48_balanced.json`` are 12/12/12/12 balanced
across {chairs, imhd, neuraldome, omomo_correct_v2}. They list chairs
first, so a renderer that takes the first N clips only sees chair
sit-down / stand-up samples — visually informative for gait, but useless
for judging hand-object contact quality because chair contact is too
easy.

This script reads the source selection JSON and keeps only the clips
where the hand-object interaction is non-trivial:

    subset != "chairs"
    mode_category == "manipulation"
    hand_contact_frac >= --min-hand-contact (default 0.8)

The resulting 18 clips per default (6 per non-chairs subset) cover:
  - imhd:             baseball-bat swing / hit with hand-held object
  - neuraldome:       case-waving, pan-lifting, chair-grabbing
  - omomo_correct_v2: wood-chair lift+move, tripod pull, trashcan lift

These are exactly the clips where the A1 ckpt's contact quality (12-15 cm
hand drift, %drift>10cm = 26.6% on val) shows up most strongly.

Output JSON schema mirrors the source so the renderer + 26_sustained_contact_diag
loaders accept it as-is.

Usage:
    python scripts/stage_b_generator/round29_build_contact_heavy_selection.py
    python scripts/stage_b_generator/round29_build_contact_heavy_selection.py \\
        --source analyses/round29_val_diag_indices_48_balanced.json \\
        --output analyses/round29_contact_heavy_nonchairs_val_selection.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "analyses" / "round27_tier0_train_indices_48_balanced.json"
DEFAULT_OUTPUT = ROOT / "analyses" / "round29_contact_heavy_nonchairs_selection.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=DEFAULT_SOURCE,
        help="Source selection JSON with rich metadata "
             "(mode_category, hand_contact_frac, ...).",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="Output selection JSON path.",
    )
    parser.add_argument(
        "--min-hand-contact", type=float, default=0.8,
        help="Minimum hand_contact_frac to keep a clip (default 0.8).",
    )
    parser.add_argument(
        "--allow-subset", action="append",
        default=None,
        help="Subsets to keep (default: every subset except chairs). "
             "Pass multiple times to whitelist multiple subsets.",
    )
    parser.add_argument(
        "--require-mode-category", default="manipulation",
        help="Required mode_category. Set to '' to disable the filter "
             "(default 'manipulation').",
    )
    args = parser.parse_args()

    src = json.loads(args.source.read_text(encoding="utf-8"))
    clips = (src.get("clips") or src.get("selected") or src.get("candidates")
             or [])
    if not clips:
        raise SystemExit(f"empty source selection: {args.source}")

    if args.allow_subset:
        allow_subsets = set(args.allow_subset)
    else:
        # default: anything except chairs.
        allow_subsets = {c["subset"] for c in clips} - {"chairs"}

    kept: list[dict] = []
    for c in clips:
        if c["subset"] not in allow_subsets:
            continue
        if args.require_mode_category and (
            c.get("mode_category") != args.require_mode_category
        ):
            continue
        try:
            if float(c.get("hand_contact_frac", 0.0)) < args.min_hand_contact:
                continue
        except (TypeError, ValueError):
            continue
        kept.append(c)

    if not kept:
        raise SystemExit(
            "no clips matched the filter "
            f"(allow_subsets={sorted(allow_subsets)}, "
            f"mode={args.require_mode_category!r}, "
            f"min_hand_contact={args.min_hand_contact})"
        )

    out = {
        "description": (
            f"Contact-heavy non-chairs subset built from {args.source.name}. "
            f"Filter: subset in {sorted(allow_subsets)}, "
            f"mode_category={args.require_mode_category!r}, "
            f"hand_contact_frac >= {args.min_hand_contact}."
        ),
        "source_selection": str(args.source.relative_to(ROOT) if
                                 args.source.is_relative_to(ROOT)
                                 else args.source),
        "filter": {
            "allow_subsets": sorted(allow_subsets),
            "mode_category": args.require_mode_category,
            "min_hand_contact_frac": args.min_hand_contact,
        },
        "n_clips": len(kept),
        "n_found": len(kept),
        "bucket": src.get("bucket", "train"),
        "clips": kept,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    counts = Counter(c["subset"] for c in kept)
    print(f"wrote {args.output}")
    print(f"  kept {len(kept)} clips, per subset: {dict(counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
