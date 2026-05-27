"""Build a Round-28 body-action train-index JSON.

The Round-27 Tier-0 subset is balanced for interaction/walking/chair
failures, but Round-28 Group B needs clips that actually exercise
body-only semantics such as neck stretch, leg stretch/cross/kick, arm
motion, and bend/lean/turn actions. This script scans the selected
bucket and emits the trainer's ``data.subset_indices_file`` schema:

    {"indices": [...], "clips": [...]}

It is intentionally heuristic; the goal is a compact diagnostic subset,
not a final benchmark split.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from piano.inference.diagnostic_helpers import _build_dataset

from piano.data.interaction_hint import (  # noqa: E402
    BODY_ACTION_KEY_JOINT_NAMES,
    build_body_action_oracle_hint,
)


NECK_KEYWORDS = ("neck", "head", "look")
LEG_KEYWORDS = (
    "left leg", "right leg", "leg", "legs", "knee", "cross", "kick",
    "stand on", "foot", "feet",
)
ARM_KEYWORDS = (
    "arm", "arms", "wrist", "wave", "raise", "stretch their arms",
    "stretches their arms",
)
BODY_KEYWORDS = (
    "bend", "bends", "lean", "leans", "turn", "turns", "twist",
    "stretches their back", "stretch their back", "stretch their body",
    "backward", "forward",
)
STRETCH_KEYWORDS = ("stretch", "stretches", "extend", "extends")


def _scan_indices(n: int, cap: int) -> list[int]:
    if cap <= 0 or cap >= n:
        return list(range(n))
    return np.linspace(0, n - 1, cap, dtype=int).tolist()


def _has_any(text: str, keywords: tuple[str, ...]) -> bool:
    text_l = text.lower()
    return any(k in text_l for k in keywords)


def _category(text: str) -> str:
    text_l = text.lower()
    if _has_any(text_l, STRETCH_KEYWORDS) and _has_any(text_l, NECK_KEYWORDS):
        return "stretch_neck"
    if _has_any(text_l, STRETCH_KEYWORDS) and _has_any(text_l, LEG_KEYWORDS):
        return "stretch_leg"
    if _has_any(text_l, LEG_KEYWORDS):
        return "leg_action"
    if _has_any(text_l, ARM_KEYWORDS):
        return "arm_action"
    if _has_any(text_l, BODY_KEYWORDS):
        return "bend_lean_turn"
    return "other_body"


def _keyword_score(text: str) -> float:
    text_l = text.lower()
    score = 0.0
    for weight, kws in (
        (8.0, STRETCH_KEYWORDS),
        (8.0, NECK_KEYWORDS),
        (6.0, LEG_KEYWORDS),
        (5.0, ARM_KEYWORDS),
        (4.0, BODY_KEYWORDS),
    ):
        score += weight * sum(1 for k in kws if k in text_l)
    return score


def _load_visual_failure_seqs(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    out: set[str] = set()
    text = path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        if "|" not in line:
            continue
        if not _has_any(line, STRETCH_KEYWORDS + NECK_KEYWORDS + LEG_KEYWORDS + BODY_KEYWORDS):
            continue
        for seq_id in re.findall(r"`([^`]+)`", line):
            out.add(seq_id)
    return out


def _row_from_sample(
    sample: dict[str, Any],
    global_idx: int,
    visual_failure_seqs: set[str],
    energy_threshold: float,
) -> dict[str, Any]:
    seq_len = int(sample["seq_len"].item() if hasattr(sample["seq_len"], "item") else sample["seq_len"])
    joints = sample["joints"][:seq_len].detach().cpu().numpy().astype(np.float32)
    contact = sample["contact_state"][:seq_len].detach().cpu().numpy().astype(np.float32)
    text = str(sample.get("text", ""))
    subset = str(sample["subset"])
    seq_id = str(sample["seq_id"])

    hint = build_body_action_oracle_hint(
        joints,
        mask_mode="energy",
        energy_threshold=float(energy_threshold),
    )
    J = len(BODY_ACTION_KEY_JOINT_NAMES)
    joint_mask = hint[0, :J].astype(np.float32)
    delta = hint[:, J:].reshape(seq_len, J, 3)
    amp = np.linalg.norm(delta, axis=-1).mean(axis=0)

    hand_contact = contact[:, :2].max(axis=1) > 0.5 if seq_len > 0 else np.zeros(0)
    hand_contact_frac = float(hand_contact.mean()) if seq_len > 0 else 0.0
    cat = _category(text)
    visual_bonus = 25.0 if seq_id in visual_failure_seqs else 0.0
    body_energy = float(amp.mean())
    active_joints = int(joint_mask.sum())
    score = (
        visual_bonus
        + _keyword_score(text)
        + 80.0 * body_energy
        + 4.0 * active_joints
        + 4.0 * (1.0 - min(hand_contact_frac, 1.0))
    )
    return {
        "dataset_global_index": int(global_idx),
        "subset": subset,
        "seq_id": seq_id,
        "text": text[:260],
        "seq_len": int(seq_len),
        "body_action_category": cat,
        "body_energy_m": body_energy,
        "active_joints": active_joints,
        "active_joint_names": [
            name for name, active in zip(BODY_ACTION_KEY_JOINT_NAMES, joint_mask)
            if float(active) > 0.5
        ],
        "hand_contact_frac": hand_contact_frac,
        "visual_failure_bonus": visual_bonus,
        "score": float(score),
    }


def _pick_balanced(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    categories = [
        "stretch_neck",
        "stretch_leg",
        "leg_action",
        "arm_action",
        "bend_lean_turn",
        "other_body",
    ]
    subsets = sorted({r["subset"] for r in rows})
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for cat in categories:
        for subset in subsets:
            bucket = [
                r for r in rows
                if r["body_action_category"] == cat and r["subset"] == subset
            ]
            bucket.sort(key=lambda r: (-float(r["score"]), r["seq_id"]))
            buckets[(cat, subset)] = bucket

    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    made_progress = True
    while len(selected) < n and made_progress:
        made_progress = False
        for cat in categories:
            for subset in subsets:
                bucket = buckets[(cat, subset)]
                while bucket and int(bucket[0]["dataset_global_index"]) in seen:
                    bucket.pop(0)
                if not bucket:
                    continue
                row = bucket.pop(0)
                selected.append(row)
                seen.add(int(row["dataset_global_index"]))
                made_progress = True
                if len(selected) >= n:
                    break
            if len(selected) >= n:
                break

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
    parser.add_argument("--max-candidates-per-subset", type=int, default=900)
    parser.add_argument("--energy-threshold", type=float, default=0.05)
    parser.add_argument(
        "--visual-summary",
        type=Path,
        default=Path("analyses/round26_visual_review/v27_final/summary.md"),
        help="Optional visual-review summary whose body-action rows get a score bonus.",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    dataset = _build_dataset(cfg, args.bucket, augment=False)
    if not hasattr(dataset, "datasets"):
        raise SystemExit("Expected ConcatDataset with one child per InterAct subset.")
    visual_failure_seqs = _load_visual_failure_seqs(args.visual_summary)
    print(f"[round28-body-subset] visual failure seqs: {len(visual_failure_seqs)}")

    rows: list[dict[str, Any]] = []
    offset = 0
    for sub_ds in dataset.datasets:
        subset_name = Path(sub_ds.root).name
        scan = _scan_indices(len(sub_ds), int(args.max_candidates_per_subset))
        print(f"[round28-body-subset] scanning {subset_name}: {len(scan)}/{len(sub_ds)}")
        for local_idx in scan:
            sample = sub_ds[int(local_idx)]
            row = _row_from_sample(
                sample,
                offset + int(local_idx),
                visual_failure_seqs,
                float(args.energy_threshold),
            )
            if row["score"] > 0.0:
                rows.append(row)
        offset += len(sub_ds)

    selected = _pick_balanced(rows, int(args.n_clips))
    if len(selected) < int(args.n_clips):
        raise SystemExit(
            f"Only selected {len(selected)} clips; requested {args.n_clips}."
        )
    selected.sort(key=lambda r: int(r["dataset_global_index"]))

    out = {
        "description": (
            "Round-28 body-action train-bucket overfit subset. Selected for "
            "stretch neck/leg, leg/arm actions, and bend/lean/turn captions, "
            "with optional visual-failure bonus."
        ),
        "source_config": str(args.config),
        "bucket": args.bucket,
        "n_requested": int(args.n_clips),
        "n_found": len(selected),
        "max_candidates_per_subset": int(args.max_candidates_per_subset),
        "energy_threshold": float(args.energy_threshold),
        "visual_summary": str(args.visual_summary) if args.visual_summary else None,
        "indices": [int(r["dataset_global_index"]) for r in selected],
        "clips": [
            {
                "subset": r["subset"],
                "seq_id": r["seq_id"],
                "body_action_category": r["body_action_category"],
                "text": r["text"],
                "body_energy_m": round(float(r["body_energy_m"]), 4),
                "active_joints": int(r["active_joints"]),
                "active_joint_names": r["active_joint_names"],
                "hand_contact_frac": round(float(r["hand_contact_frac"]), 4),
                "visual_failure_bonus": round(float(r["visual_failure_bonus"]), 1),
                "score": round(float(r["score"]), 4),
            }
            for r in selected
        ],
        "subset_counts": dict(Counter(r["subset"] for r in selected)),
        "body_action_category_counts": dict(
            Counter(r["body_action_category"] for r in selected)
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"subset_counts={out['subset_counts']}")
    print(f"body_action_category_counts={out['body_action_category_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
