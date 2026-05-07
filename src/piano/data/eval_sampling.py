"""Deterministic clip selection for small offline validation sets.

The Stage B contact metric is expensive enough that we often evaluate on
small subsets. A naive "first N" or purely random N can overfit the
checkpoint choice to one dataset subset or object type, so this helper
balances by dataset subset first and object id second.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch.utils.data import ConcatDataset


@dataclass(frozen=True)
class EvalClipCandidate:
    """Metadata for one validation clip without loading its arrays."""

    index: int
    subset: str
    object_id: str
    seq_id: str


def select_eval_clip_indices(
    dataset: Any,
    num_clips: int,
    *,
    seed: int = 42,
) -> list[int]:
    """Pick a deterministic, type-diverse validation subset.

    Selection is balanced by ``subset`` first (chairs / imhd /
    neuraldome / omomo...), then round-robin by ``object_id`` within each
    subset. This keeps a 20-clip contact eval from being dominated by
    whichever subset happens to appear first in a ``ConcatDataset``.
    """
    if num_clips <= 0:
        return []

    candidates = _collect_candidates(dataset)
    if not candidates:
        return _fallback_random_indices(len(dataset), num_clips, seed)
    if num_clips >= len(candidates):
        return [c.index for c in candidates]

    rng = random.Random(seed)
    by_subset: dict[str, list[EvalClipCandidate]] = {}
    for cand in candidates:
        by_subset.setdefault(cand.subset, []).append(cand)

    subsets = sorted(by_subset)
    rng.shuffle(subsets)
    quotas = _balanced_quotas(
        {subset: len(by_subset[subset]) for subset in subsets},
        total=num_clips,
        subset_order=subsets,
    )

    selected_by_subset: dict[str, list[int]] = {}
    for subset in subsets:
        selected_by_subset[subset] = _select_within_subset(
            by_subset[subset],
            quotas.get(subset, 0),
            rng,
        )

    # Interleave subsets so the qual_eval "swap with next clip" condition
    # also tends to cross dataset/object types instead of cycling inside one
    # contiguous block.
    selected: list[int] = []
    while len(selected) < num_clips:
        advanced = False
        for subset in subsets:
            bucket = selected_by_subset[subset]
            if bucket:
                selected.append(bucket.pop(0))
                advanced = True
                if len(selected) >= num_clips:
                    break
        if not advanced:
            break

    if len(selected) < num_clips:
        already = set(selected)
        fill = [c.index for c in candidates if c.index not in already]
        rng.shuffle(fill)
        selected.extend(fill[: num_clips - len(selected)])

    return selected[:num_clips]


def resolve_eval_clip_count(
    dataset: Any,
    *,
    num_clips: int | None = None,
    num_clips_per_subset: int | None = None,
) -> int:
    """Resolve an absolute clip count for fixed-set evaluation.

    ``num_clips`` is the legacy total-count knob used by offline eval.
    ``num_clips_per_subset`` is better for checkpoint selection because it
    keeps the per-subset sample size stable when the number of dataset
    subsets changes. If metadata is unavailable, fall back to ``num_clips``.
    """
    if num_clips_per_subset is not None and num_clips_per_subset > 0:
        candidates = _collect_candidates(dataset)
        if candidates:
            subsets = {cand.subset for cand in candidates}
            return min(len(candidates), int(num_clips_per_subset) * len(subsets))

    if num_clips is None:
        return 0
    return max(0, int(num_clips))


def describe_eval_clip_selection(dataset: Any, indices: list[int]) -> list[dict[str, str]]:
    """Return lightweight provenance rows for logging selected clips."""
    by_index = {cand.index: cand for cand in _collect_candidates(dataset)}
    rows: list[dict[str, str]] = []
    for idx in indices:
        cand = by_index.get(idx)
        if cand is None:
            rows.append({
                "index": str(idx),
                "subset": "<unknown_subset>",
                "object_id": "<unknown_object>",
                "seq_id": "<unknown_seq>",
            })
        else:
            rows.append({
                "index": str(cand.index),
                "subset": cand.subset,
                "object_id": cand.object_id,
                "seq_id": cand.seq_id,
            })
    return rows


def select_eval_clip_indices_by_seq_id(dataset: Any, seq_ids: list[str]) -> list[int]:
    """Resolve explicit seq_ids to dataset indices, preserving request order."""
    candidates = _collect_candidates(dataset)
    by_seq: dict[str, EvalClipCandidate] = {}
    for cand in candidates:
        by_seq.setdefault(cand.seq_id, cand)

    missing = [seq_id for seq_id in seq_ids if seq_id not in by_seq]
    if missing:
        preview = ", ".join(missing[:5])
        raise KeyError(f"seq_id(s) not found in eval dataset: {preview}")

    return [by_seq[seq_id].index for seq_id in seq_ids]


def _collect_candidates(dataset: Any) -> list[EvalClipCandidate]:
    if isinstance(dataset, ConcatDataset):
        out: list[EvalClipCandidate] = []
        offset = 0
        for child in dataset.datasets:
            out.extend(_collect_leaf_candidates(child, offset))
            offset += len(child)
        return out
    return _collect_leaf_candidates(dataset, offset=0)


def _collect_leaf_candidates(dataset: Any, offset: int) -> list[EvalClipCandidate]:
    metadata = getattr(dataset, "metadata", None)
    if metadata is None:
        return []

    root = getattr(dataset, "root", "")
    subset = Path(root).name if root else getattr(dataset, "name", "")
    if not subset:
        subset = "<unknown_subset>"

    candidates: list[EvalClipCandidate] = []
    for local_idx, meta in enumerate(metadata):
        candidates.append(
            EvalClipCandidate(
                index=offset + local_idx,
                subset=str(subset),
                object_id=str(meta.get("object_id", "<unknown_object>")),
                seq_id=str(meta.get("seq_id", f"idx_{local_idx}")),
            ),
        )
    return candidates


def _balanced_quotas(
    counts: dict[str, int],
    *,
    total: int,
    subset_order: list[str],
) -> dict[str, int]:
    active = [subset for subset in subset_order if counts.get(subset, 0) > 0]
    quotas = {subset: 0 for subset in subset_order}
    remaining = total

    while remaining > 0 and active:
        base = max(1, remaining // len(active))
        next_active: list[str] = []
        for subset in active:
            room = counts[subset] - quotas[subset]
            add = min(base, room, remaining)
            quotas[subset] += add
            remaining -= add
            if quotas[subset] < counts[subset]:
                next_active.append(subset)
            if remaining <= 0:
                break
        active = next_active

    return quotas


def _select_within_subset(
    candidates: list[EvalClipCandidate],
    quota: int,
    rng: random.Random,
) -> list[int]:
    if quota <= 0:
        return []

    by_object: dict[str, list[EvalClipCandidate]] = {}
    for cand in candidates:
        by_object.setdefault(cand.object_id, []).append(cand)

    object_ids = sorted(by_object)
    rng.shuffle(object_ids)
    for object_id in object_ids:
        by_object[object_id].sort(key=lambda c: (c.seq_id, c.index))
        rng.shuffle(by_object[object_id])

    selected: list[int] = []
    while len(selected) < quota:
        advanced = False
        for object_id in object_ids:
            bucket = by_object[object_id]
            if bucket:
                selected.append(bucket.pop(0).index)
                advanced = True
                if len(selected) >= quota:
                    break
        if not advanced:
            break
    return selected


def _fallback_random_indices(length: int, num_clips: int, seed: int) -> list[int]:
    pool = list(range(length))
    random.Random(seed).shuffle(pool)
    return pool[:num_clips]
