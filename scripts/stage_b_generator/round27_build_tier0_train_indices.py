"""Build a balanced Round-27 Tier-0 train-index JSON.

The first Tier-0 helper reused the Round-25 multimodality selection and
accidentally produced 48 chair-only clips. That is too narrow for the
current question: oracle interaction hints and temporal losses must be
tested on hand-object manipulation, walking/gait, and chair-contact cases.

This script scans the train bucket, scores candidate clips from each
subset, and emits the trainer's ``data.subset_indices_file`` schema:

    {"indices": [...], "clips": [...]}

It is intentionally heuristic; the goal is a stronger Tier-0 overfit set,
not a final benchmark split.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from plan_condition_diagnostics import _build_dataset  # noqa: E402


MANIPULATION_KEYWORDS = (
    "grab", "hold", "lift", "carry", "pick", "place", "put", "push",
    "pull", "open", "close", "move", "throw", "catch", "swing", "hit",
    "strike", "touch",
)
WALKING_KEYWORDS = (
    "walk", "walking", "step", "steps", "pace", "circle", "around",
    "approach", "leave", "turn",
)
CHAIR_KEYWORDS = (
    "sit", "sitting", "chair", "stool", "bench", "sofa", "recline",
    "lean",
)


def _scan_indices(n: int, cap: int) -> list[int]:
    if cap <= 0 or cap >= n:
        return list(range(n))
    return np.linspace(0, n - 1, cap, dtype=int).tolist()


def _has_any(text: str, keywords: tuple[str, ...]) -> bool:
    text_l = text.lower()
    return any(k in text_l for k in keywords)


def _categorise(
    *,
    subset: str,
    text: str,
    hand_contact_frac: float,
    walking_frac: float,
    root_disp_m: float,
) -> str:
    manip = _has_any(text, MANIPULATION_KEYWORDS)
    walk = _has_any(text, WALKING_KEYWORDS)
    chair = subset == "chairs" or _has_any(text, CHAIR_KEYWORDS)
    if manip or hand_contact_frac >= 0.08:
        return "manipulation"
    if walk or walking_frac >= 0.25 or root_disp_m >= 0.50:
        return "walking"
    if chair:
        return "chair_contact"
    return "general"


def _score_row(row: dict[str, Any]) -> float:
    text = str(row["text"]).lower()
    keyword_hits = sum(
        1
        for kw in MANIPULATION_KEYWORDS + WALKING_KEYWORDS + CHAIR_KEYWORDS
        if kw in text
    )
    category_boost = {
        "manipulation": 25.0,
        "walking": 20.0,
        "chair_contact": 10.0,
        "general": 0.0,
    }[row["mode_category"]]
    return (
        category_boost
        + 80.0 * float(row["hand_contact_frac"])
        + 40.0 * float(row["walking_frac"])
        + 10.0 * float(row["foot_contact_frac"])
        + 3.0 * float(keyword_hits)
        + min(float(row["seq_len"]) / 196.0, 1.0)
    )


def _row_from_sample(sample: dict[str, Any], global_idx: int) -> dict[str, Any]:
    seq_len = int(sample["seq_len"].item() if hasattr(sample["seq_len"], "item") else sample["seq_len"])
    contact = sample["contact_state"][:seq_len].detach().cpu().numpy().astype(np.float32)
    joints = sample["joints"][:seq_len].detach().cpu().numpy().astype(np.float32)
    text = str(sample.get("text", ""))
    subset = str(sample["subset"])
    seq_id = str(sample["seq_id"])

    hand_contact = contact[:, :2].max(axis=1) > 0.5
    foot_contact = contact[:, 2:4].max(axis=1) > 0.5
    root_xz = joints[:, 0, [0, 2]]
    speed = np.zeros(seq_len, dtype=np.float32)
    if seq_len > 1:
        speed[1:] = np.linalg.norm(root_xz[1:] - root_xz[:-1], axis=-1)
    walking = speed > 0.005
    root_disp_m = float(np.linalg.norm(root_xz[-1] - root_xz[0])) if seq_len > 1 else 0.0

    hand_contact_frac = float(hand_contact.mean()) if seq_len > 0 else 0.0
    foot_contact_frac = float(foot_contact.mean()) if seq_len > 0 else 0.0
    walking_frac = float(walking.mean()) if seq_len > 0 else 0.0
    mode_category = _categorise(
        subset=subset,
        text=text,
        hand_contact_frac=hand_contact_frac,
        walking_frac=walking_frac,
        root_disp_m=root_disp_m,
    )
    row = {
        "dataset_global_index": int(global_idx),
        "subset": subset,
        "seq_id": seq_id,
        "text": text[:220],
        "seq_len": int(seq_len),
        "mode_category": mode_category,
        "hand_contact_frac": hand_contact_frac,
        "foot_contact_frac": foot_contact_frac,
        "walking_frac": walking_frac,
        "root_disp_m": root_disp_m,
    }
    row["score"] = float(_score_row(row))
    return row


def _pick_balanced(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    subsets = sorted({r["subset"] for r in rows})
    if not subsets:
        return []
    base = n // len(subsets)
    rem = n % len(subsets)
    subset_quota = {s: base + (1 if i < rem else 0) for i, s in enumerate(subsets)}
    categories = ["manipulation", "walking", "chair_contact", "general"]
    by_pair: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for subset in subsets:
        for cat in categories:
            bucket = [
                r for r in rows
                if r["subset"] == subset and r["mode_category"] == cat
            ]
            bucket.sort(key=lambda r: (-float(r["score"]), r["seq_id"]))
            by_pair[(subset, cat)] = bucket

    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    subset_counts: Counter[str] = Counter()
    made_progress = True
    while len(selected) < n and made_progress:
        made_progress = False
        for cat in categories:
            for subset in subsets:
                if len(selected) >= n:
                    break
                if subset_counts[subset] >= subset_quota[subset]:
                    continue
                bucket = by_pair[(subset, cat)]
                while bucket and int(bucket[0]["dataset_global_index"]) in seen:
                    bucket.pop(0)
                if not bucket:
                    continue
                row = bucket.pop(0)
                selected.append(row)
                seen.add(int(row["dataset_global_index"]))
                subset_counts[subset] += 1
                made_progress = True

    if len(selected) < n:
        fallback = sorted(rows, key=lambda r: (-float(r["score"]), r["seq_id"]))
        for row in fallback:
            idx = int(row["dataset_global_index"])
            if idx in seen:
                continue
            selected.append(row)
            seen.add(idx)
            if len(selected) >= n:
                break
    return selected[:n]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-clips", type=int, default=48)
    parser.add_argument("--bucket", choices=["train", "val"], default="train")
    parser.add_argument("--max-candidates-per-subset", type=int, default=600)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    dataset = _build_dataset(cfg, args.bucket, augment=False)
    if not hasattr(dataset, "datasets"):
        raise SystemExit("Expected ConcatDataset with one child per InterAct subset.")

    rows: list[dict[str, Any]] = []
    offset = 0
    for sub_ds in dataset.datasets:
        subset_name = Path(sub_ds.root).name
        scan = _scan_indices(len(sub_ds), int(args.max_candidates_per_subset))
        print(f"[round27-build] scanning {subset_name}: {len(scan)}/{len(sub_ds)}")
        for local_idx in scan:
            sample = sub_ds[int(local_idx)]
            rows.append(_row_from_sample(sample, offset + int(local_idx)))
        offset += len(sub_ds)

    selected = _pick_balanced(rows, int(args.n_clips))
    if len(selected) < int(args.n_clips):
        raise SystemExit(
            f"Only selected {len(selected)} clips; requested {args.n_clips}."
        )

    selected.sort(key=lambda r: int(r["dataset_global_index"]))
    out = {
        "description": (
            "Round-27 Tier-0 balanced train-bucket overfit subset. "
            "Selected across InterAct subsets and manipulation/walking/chair "
            "failure categories."
        ),
        "source_config": str(args.config),
        "bucket": args.bucket,
        "n_requested": int(args.n_clips),
        "n_found": len(selected),
        "max_candidates_per_subset": int(args.max_candidates_per_subset),
        "indices": [int(r["dataset_global_index"]) for r in selected],
        "clips": [
            {
                "subset": r["subset"],
                "seq_id": r["seq_id"],
                "mode_category": r["mode_category"],
                "text": r["text"],
                "hand_contact_frac": round(float(r["hand_contact_frac"]), 4),
                "foot_contact_frac": round(float(r["foot_contact_frac"]), 4),
                "walking_frac": round(float(r["walking_frac"]), 4),
                "root_disp_m": round(float(r["root_disp_m"]), 4),
                "score": round(float(r["score"]), 4),
            }
            for r in selected
        ],
        "subset_counts": dict(Counter(r["subset"] for r in selected)),
        "mode_category_counts": dict(Counter(r["mode_category"] for r in selected)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[round27-build] wrote {len(selected)} clips -> {args.output}")
    print(f"[round27-build] subset_counts={out['subset_counts']}")
    print(f"[round27-build] mode_category_counts={out['mode_category_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
