from __future__ import annotations

from collections import Counter
from pathlib import Path

from torch.utils.data import ConcatDataset, Dataset

from piano.data.eval_sampling import (
    describe_eval_clip_selection,
    resolve_eval_clip_count,
    select_eval_clip_indices,
)


class _FakeMetaDataset(Dataset):
    def __init__(self, root: str, rows: list[dict[str, str]]) -> None:
        self.root = Path(root)
        self.metadata = rows

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict[str, str]:
        return self.metadata[idx]


def _rows(prefix: str, n: int, objects: tuple[str, ...]) -> list[dict[str, str]]:
    return [
        {
            "seq_id": f"{prefix}_{i:03d}",
            "object_id": objects[i % len(objects)],
        }
        for i in range(n)
    ]


def test_select_eval_clip_indices_balances_concat_subsets():
    dataset = ConcatDataset([
        _FakeMetaDataset("chairs", _rows("chair", 10, ("chair_a", "chair_b"))),
        _FakeMetaDataset("imhd", _rows("imhd", 10, ("box", "table"))),
        _FakeMetaDataset("neuraldome", _rows("neural", 10, ("basket", "stool"))),
        _FakeMetaDataset("omomo_correct_v2", _rows("omomo", 10, ("plasticbox", "whitechair"))),
    ])

    indices = select_eval_clip_indices(dataset, 20, seed=42)
    rows = describe_eval_clip_selection(dataset, indices)

    assert len(indices) == 20
    assert len(set(indices)) == 20
    assert Counter(row["subset"] for row in rows) == {
        "chairs": 5,
        "imhd": 5,
        "neuraldome": 5,
        "omomo_correct_v2": 5,
    }


def test_select_eval_clip_indices_round_robins_objects_within_subset():
    dataset = _FakeMetaDataset(
        "omomo_correct_v2",
        _rows("omomo", 12, ("plasticbox", "whitechair", "largebox")),
    )

    indices = select_eval_clip_indices(dataset, 6, seed=7)
    rows = describe_eval_clip_selection(dataset, indices)

    assert len(indices) == 6
    counts = Counter(row["object_id"] for row in rows)
    assert set(counts) == {"plasticbox", "whitechair", "largebox"}
    assert all(count == 2 for count in counts.values())


def test_resolve_eval_clip_count_from_per_subset():
    dataset = ConcatDataset([
        _FakeMetaDataset("chairs", _rows("chair", 10, ("chair_a", "chair_b"))),
        _FakeMetaDataset("imhd", _rows("imhd", 10, ("box", "table"))),
        _FakeMetaDataset("neuraldome", _rows("neural", 10, ("basket", "stool"))),
        _FakeMetaDataset("omomo_correct_v2", _rows("omomo", 10, ("plasticbox", "whitechair"))),
    ])

    assert resolve_eval_clip_count(dataset, num_clips_per_subset=6) == 24


def test_resolve_eval_clip_count_caps_at_dataset_size():
    dataset = ConcatDataset([
        _FakeMetaDataset("chairs", _rows("chair", 3, ("chair_a",))),
        _FakeMetaDataset("imhd", _rows("imhd", 4, ("box",))),
    ])

    assert resolve_eval_clip_count(dataset, num_clips_per_subset=10) == 7
