"""Round-25 D1: propose candidate clips for the multimodal eval subset.

Scans val-bucket clips and ranks them by likelihood that text label is
AMBIGUOUS w.r.t. limb-pose mode (left vs right arm/leg on top, hand
side interchange, mirrored / clockwise actions). The user then makes
the final 50-clip cut manually from the proposed candidates.

Design source:
    analyses/2026-05-23_round25_diagnostic_bundle_design.md §D1.

Why a proposer, not a fully automated selector:
    Mode-ambiguity is a semantic property of the text label, not a
    syntactic property. A human review pass is needed on the
    candidates. This script narrows the search from ~1000 val clips
    to ~150 plausible candidates so the human pass is feasible.

Usage:
    conda run -n piano python scripts/stage_b_generator/round25_d1_propose_multimodal_candidates.py \
        --config configs/training/anchordiff_v26_FULL_DATA_local.yaml \
        --output analyses/round25_multimodal_candidates.json \
        --top-k 200

Output: a JSON list of candidates, each with subset, seq_id, text,
        mode_category (best guess), and a confidence score [0, 1].
        The user reviews this list and produces the final
        analyses/round25_multimodal_eval_subset.json with ~50 clips.
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


# Per-category keyword patterns. Each clip can hit ≥1 category.
# All matching is case-insensitive substring.
_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "crossed_arms": [
        "cross arm", "crossed arm", "fold arm", "folded arm",
        "arms cross", "arms folded", "arm across", "抱胸",
    ],
    "crossed_legs": [
        "cross leg", "crossed leg", "leg cross", "legs cross",
        "leg over", "knee over", "二郎腿", "翘二郎腿",
    ],
    "hand_choice": [
        # Ambiguous hand-side actions where left vs right is not
        # constrained by the object pose.
        "pick up", "grab", "lift", "hold", "carry", "reach",
        "touch", "tap", "push", "pull",
    ],
    "mirror_swing": [
        # Swing/hit/throw actions — sides often interchangeable.
        "swing", "hit", "throw", "strike", "wave", "swipe",
    ],
    "circular_direction": [
        # Clockwise vs counter-clockwise ambiguity.
        "rotate", "spin", "turn", "circle", "twist",
    ],
    "sit_pose": [
        # Sitting pose where leg arrangement varies (legs apart,
        # together, crossed, one-up).
        "sit", "seated",
    ],
}


def _score_text(text: str) -> dict[str, float]:
    """Return per-category match score in [0, 1] for a text label."""
    text_l = text.lower()
    out: dict[str, float] = {}
    for cat, kws in _CATEGORY_KEYWORDS.items():
        hits = sum(1 for kw in kws if kw.lower() in text_l)
        if hits == 0:
            out[cat] = 0.0
        else:
            # Saturating: 1 hit = 0.5, 2 hits = 0.8, 3+ = 1.0.
            out[cat] = min(1.0, 0.5 + 0.3 * (hits - 1))
    return out


def _primary_category(scores: dict[str, float]) -> str | None:
    nonzero = {k: v for k, v in scores.items() if v > 0}
    if not nonzero:
        return None
    return max(nonzero, key=lambda k: nonzero[k])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True,
                        help="v26 (or compatible) training config; we use it only for dataset construction.")
    parser.add_argument("--output", type=Path,
                        default=Path("analyses/round25_multimodal_candidates.json"))
    parser.add_argument("--bucket", default="val", choices=["train", "val"])
    parser.add_argument("--top-k", type=int, default=200,
                        help="Number of candidates to emit. Pad with low-confidence "
                             "hits if too few high-confidence matches.")
    parser.add_argument("--min-confidence", type=float, default=0.5,
                        help="Below this confidence, a clip is only kept if top-k "
                             "quota is still unmet.")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    dataset = _build_dataset(cfg, args.bucket, augment=False)

    print(f"[d1] scanning {len(dataset)} {args.bucket} clips...")

    candidates: list[dict] = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        subset = str(sample["subset"])
        seq_id = str(sample["seq_id"])
        text = str(sample["text"])
        scores = _score_text(text)
        primary = _primary_category(scores)
        if primary is None:
            continue
        confidence = scores[primary]
        candidates.append({
            "subset": subset,
            "seq_id": seq_id,
            "text": text,
            "mode_category_guess": primary,
            "confidence": float(confidence),
            "per_category_scores": {k: float(v) for k, v in scores.items() if v > 0},
        })

    candidates.sort(key=lambda c: (-c["confidence"], c["subset"], c["seq_id"]))

    # Apply min-confidence + top-k.
    high = [c for c in candidates if c["confidence"] >= args.min_confidence]
    if len(high) >= args.top_k:
        selected = high[: args.top_k]
    else:
        # Pad with the next-best below threshold.
        rest = [c for c in candidates if c["confidence"] < args.min_confidence]
        selected = high + rest[: args.top_k - len(high)]

    # Per-category histogram.
    hist: dict[str, int] = {}
    for c in selected:
        cat = c["mode_category_guess"]
        hist[cat] = hist.get(cat, 0) + 1

    out = {
        "description": (
            "Round-25 D1 candidate proposer output. The user reviews this list "
            "and produces the final analyses/round25_multimodal_eval_subset.json "
            "with ~50 clips after manual disambiguation."
        ),
        "source_config": str(args.config),
        "bucket": args.bucket,
        "total_scanned": len(dataset),
        "n_candidates": len(selected),
        "per_category_histogram": hist,
        "candidates": selected,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    print(f"[d1] wrote {len(selected)} candidates to {args.output}")
    print(f"[d1] per-category histogram: {hist}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
