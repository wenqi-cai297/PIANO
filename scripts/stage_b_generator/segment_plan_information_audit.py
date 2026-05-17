"""No-training segment-token feasibility audit for Stage B interaction plans."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from diagnostic_common import format_md_table, selected_balanced_batches, stats_list, write_json
from dynamics_diagnostic import _build_dataset, _balanced_subset_indices
from piano.data.dataset import collate_hoi


KEYWORDS = (
    "lie", "lying", "lay", "laying", "recline", "reclining", "lean", "leaning",
    "rest", "sofa", "bed", "chair", "sit", "sitting",
)
PART_NAMES = ["L_hand", "R_hand", "L_foot", "R_foot", "pelvis"]


def _keyword_hit(text: str) -> bool:
    low = text.lower()
    return any(k in low for k in KEYWORDS)


def _row_from_batch(batch: dict[str, Any], idx: int) -> dict[str, Any]:
    seq_len = int(batch["seq_len"][0].item())
    seg_mask = batch["plan_segment_mask"][0].bool()
    starts = batch["plan_segment_start"][0][seg_mask].cpu().numpy().astype(np.int64)
    ends = batch["plan_segment_end"][0][seg_mask].cpu().numpy().astype(np.int64)
    parts = batch["plan_segment_part"][0][seg_mask].cpu().numpy()
    durations = (ends - starts + 1).clip(min=0)
    long_mask = durations >= 20
    anchor_mask = batch["plan_anchor_mask"][0].bool()
    anchor_types = batch["plan_anchor_type"][0][anchor_mask].cpu().numpy().astype(np.int64)
    anchor_parts = batch["plan_anchor_part"][0][anchor_mask].cpu().numpy()
    part_counts = {name: int((parts[:, i] > 0.0).sum()) if parts.size else 0 for i, name in enumerate(PART_NAMES)}
    anchor_part_counts = {name: int((anchor_parts[:, i] > 0.0).sum()) if anchor_parts.size else 0 for i, name in enumerate(PART_NAMES)}
    return {
        "index": int(idx),
        "subset": str(batch["subset"][0]),
        "seq_id": str(batch["seq_id"][0]),
        "object_id": str(batch["object_id"][0]),
        "text": str(batch["text"][0]),
        "keyword_group": "keyword" if _keyword_hit(str(batch["text"][0])) else "control",
        "seq_len": seq_len,
        "n_segments": int(seg_mask.sum().item()),
        "n_anchors": int(anchor_mask.sum().item()),
        "segment_durations": [int(v) for v in durations.tolist()],
        "long_segment_count_ge20": int(long_mask.sum()),
        "max_segment_duration": int(durations.max()) if durations.size else 0,
        "segment_part_counts": part_counts,
        "anchor_part_counts": anchor_part_counts,
        "anchor_type_counts": {str(i): int((anchor_types == i).sum()) for i in range(5)},
    }


def _write_report(payload: dict[str, Any], path: Path) -> None:
    rows = [["group", "clips", "segments/clip", "long seg/clip", "max dur p95", "anchors/clip"]]
    for group, item in payload["group_summary"].items():
        rows.append([
            group,
            item["clips"],
            f"{item['segments_per_clip_mean']:.2f}",
            f"{item['long_segments_per_clip_mean']:.2f}",
            f"{item['max_segment_duration_p95']:.1f}",
            f"{item['anchors_per_clip_mean']:.2f}",
        ])
    clip_rows = [["subset", "seq_id", "group", "segments", "long", "max dur", "anchors", "text"]]
    for row in payload["clips"][:30]:
        clip_rows.append([
            row["subset"],
            row["seq_id"],
            row["keyword_group"],
            row["n_segments"],
            row["long_segment_count_ge20"],
            row["max_segment_duration"],
            row["n_anchors"],
            row["text"][:70],
        ])
    lines = [
        "# Segment Plan Information Audit",
        "",
        f"- Config: `{payload['config']}`",
        f"- Clips: {len(payload['clips'])}",
        "",
        "## Verdict",
        "",
        payload["verdict"],
        "",
        "## Summary",
        "",
        format_md_table(rows),
        "",
        "## Clip Inventory",
        "",
        format_md_table(clip_rows),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/training/anchordiff_v18_a1_FULL_DATA.yaml"))
    parser.add_argument("--output", type=Path, default=Path("analyses/2026-05-14_segment_plan_information_audit.json"))
    parser.add_argument("--md", type=Path, default=Path("analyses/2026-05-14_segment_plan_information_audit.md"))
    parser.add_argument("--bucket", choices=["train", "val"], default="train")
    parser.add_argument("--num-clips", type=int, default=64)
    parser.add_argument("--num-candidates", type=int, default=512)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    dataset = _build_dataset(cfg, args.bucket)
    indices = _balanced_subset_indices(dataset, int(args.num_candidates))
    subset = Subset(dataset, indices[: int(args.num_candidates)])
    loader = DataLoader(subset, batch_size=1, shuffle=False, collate_fn=collate_hoi, num_workers=0)
    rows: list[dict[str, Any]] = []
    for i, batch in enumerate(loader):
        if len(rows) >= int(args.num_clips):
            break
        rows.append(_row_from_batch(batch, indices[i]))

    group_summary: dict[str, Any] = {}
    for group in ("keyword", "control"):
        group_rows = [r for r in rows if r["keyword_group"] == group]
        if not group_rows:
            continue
        group_summary[group] = {
            "clips": len(group_rows),
            "segments_per_clip_mean": float(np.mean([r["n_segments"] for r in group_rows])),
            "long_segments_per_clip_mean": float(np.mean([r["long_segment_count_ge20"] for r in group_rows])),
            "max_segment_duration_p95": float(np.percentile([r["max_segment_duration"] for r in group_rows], 95)),
            "anchors_per_clip_mean": float(np.mean([r["n_anchors"] for r in group_rows])),
        }

    long_rate = float(np.mean([r["long_segment_count_ge20"] > 0 for r in rows])) if rows else 0.0
    if long_rate >= 0.30:
        verdict = (
            "Long contact/support segments are common in this audit sample; segment tokens may carry useful sustained-contact information later, but no segment-token training is recommended until P0/P1 and plan-sensitivity evidence point that way."
        )
    else:
        verdict = (
            "Long contact/support segments are not common enough in this sample to prioritize segment-token training now."
        )

    payload = {
        "config": str(args.config),
        "clips": rows,
        "group_summary": group_summary,
        "duration_stats": stats_list([d for row in rows for d in row["segment_durations"]]),
        "long_segment_clip_rate_ge20": long_rate,
        "verdict": verdict,
    }
    write_json(args.output, payload)
    _write_report(payload, args.md)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.md}")


if __name__ == "__main__":
    main()

