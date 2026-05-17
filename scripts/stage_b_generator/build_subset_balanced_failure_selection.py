"""Subset-balanced failure-atlas selection builder (Round 9, Task 1).

Iterates each subset dataset (chairs / imhd / neuraldome / omomo_correct_v2)
separately and picks N clips per subset that:

1. Have at least one valid hand onset/release event under metric v2:
   - event frame >= edge_margin AND event frame + window_k <= seq_len - 1 - edge_margin
   - segment duration > flicker_max_frames (default 2)
2. Are NOT dominated by flicker-only contact (mean segment duration > 2 frames).
3. Have at least one of: hand contact, pelvis contact (object interaction).
4. Sequence length sufficient for event windows (seq_len > 2 * window_k).
5. Have text/action diversity (preference, scoring tie-breaker).

Per-subset selection: top-K by (n_valid_v2_events) ordering, with text-
diversity preference among ties.

Falls back gracefully if a subset has fewer than the minimum required
clips — picks all available with explicit min_per_subset enforcement.

Outputs:
  analyses/2026-05-19_subset_balanced_failure_selection.json
  analyses/2026-05-19_subset_balanced_failure_selection.md
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from dynamics_diagnostic import _build_dataset
from piano.data.dataset import collate_hoi


def _segment_pairs(c_bool: np.ndarray, seq_len: int) -> list[tuple[int, int]]:
    if c_bool.size < 2:
        return []
    onset = (c_bool[1:] & ~c_bool[:-1])
    release = (~c_bool[1:] & c_bool[:-1])
    onset_idx = (np.where(onset)[0] + 1).tolist()
    release_idx = (np.where(release)[0] + 1).tolist()
    segs: list[tuple[int, int]] = []
    ri = 0
    for o in onset_idx:
        while ri < len(release_idx) and release_idx[ri] <= o:
            ri += 1
        if ri < len(release_idx):
            segs.append((o, release_idx[ri]))
            ri += 1
        else:
            segs.append((o, seq_len))
    return segs


def _score_clip(
    contact_state: np.ndarray, seq_len: int,
    *, threshold: float, window_k: int, edge_margin: int, flicker_max_frames: int,
) -> dict[str, int]:
    """Return per-clip event-validity statistics (hand parts 0, 1 only)."""
    out = {
        "n_contact_events": 0,
        "n_valid_v2_slope": 0,
        "n_valid_v2_signed": 0,
        "n_flicker": 0,
        "n_boundary": 0,
        "n_hand_contact_frames": 0,
        "first_valid_event_frame": -1,
        "first_valid_event_part_idx": -1,
        "first_valid_event_kind": "",
    }
    for p_idx, part_name in ((0, "L_hand"), (1, "R_hand")):
        c_bool = contact_state[:seq_len, p_idx] > float(threshold)
        out["n_hand_contact_frames"] += int(c_bool.sum())
        segs = _segment_pairs(c_bool, seq_len)
        for s, e in segs:
            duration = max(1, e - s)
            for kind, t_ev in (("onset", s), ("release", e)):
                t_ev = int(min(max(0, t_ev), seq_len - 1))
                out["n_contact_events"] += 1
                in_pre_range = (kind == "onset" and t_ev - window_k >= 0) or kind == "release"
                in_post_range = (kind == "release" and t_ev + window_k <= seq_len - 1) or kind == "onset"
                away_from_edge = (
                    t_ev >= int(edge_margin) and t_ev <= seq_len - 1 - int(edge_margin)
                )
                not_flicker = duration > int(flicker_max_frames)
                if duration <= int(flicker_max_frames):
                    out["n_flicker"] += 1
                if not away_from_edge:
                    out["n_boundary"] += 1
                if in_pre_range and in_post_range and away_from_edge and not_flicker:
                    out["n_valid_v2_slope"] += 1
                    out["n_valid_v2_signed"] += 1
                    if out["first_valid_event_frame"] < 0:
                        out["first_valid_event_frame"] = int(t_ev)
                        out["first_valid_event_part_idx"] = int(p_idx)
                        out["first_valid_event_kind"] = kind
    return out


def _action_keywords(text: str) -> list[str]:
    """Coarse keyword tags for diversity tie-breaking."""
    t = text.lower()
    out = []
    for kw in (
        "sit", "lie", "lying", "recline", "stand", "turn", "lean",
        "hold", "swing", "hit", "strike", "throw", "pick", "place", "put", "lift",
        "carry", "open", "close", "pull", "push", "grab", "kick", "use",
    ):
        if kw in t:
            out.append(kw)
    return out


def _diversity_rank(rows: list[dict[str, Any]]) -> list[int]:
    """Return order to maximise text-action diversity given equal n_valid scores.

    Greedy: pick the row with the highest n_valid; among ties, pick the
    row whose keywords are most disjoint from already-picked rows.
    """
    remaining = list(range(len(rows)))
    picked: list[int] = []
    picked_kw: set[str] = set()
    while remaining:
        best = remaining[0]
        best_score = (rows[best]["n_valid_v2_slope"], -len(set(rows[best]["keywords"]) & picked_kw))
        for r in remaining[1:]:
            kw = set(rows[r]["keywords"])
            score = (rows[r]["n_valid_v2_slope"], -len(kw & picked_kw))
            if score > best_score:
                best, best_score = r, score
        picked.append(best)
        picked_kw |= set(rows[best]["keywords"])
        remaining.remove(best)
    return picked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/training/anchordiff_v18_a1_FULL_DATA.yaml"),
    )
    parser.add_argument(
        "--output-json", type=Path,
        default=Path("analyses/2026-05-19_subset_balanced_failure_selection.json"),
    )
    parser.add_argument(
        "--output-md", type=Path,
        default=Path("analyses/2026-05-19_subset_balanced_failure_selection.md"),
    )
    parser.add_argument("--bucket", choices=["train", "val"], default="train")
    parser.add_argument("--per-subset-target", type=int, default=6,
                        help="Preferred clips per subset (24 total).")
    parser.add_argument("--per-subset-minimum", type=int, default=4,
                        help="Hard minimum per subset (16 total).")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--window-k", type=int, default=10)
    parser.add_argument("--edge-margin", type=int, default=5)
    parser.add_argument("--flicker-max-frames", type=int, default=2)
    parser.add_argument("--max-candidates-per-subset", type=int, default=400,
                        help="Per-subset dataloader scan cap (for speed).")
    parser.add_argument("--include-easy-controls", action="store_true", default=True,
                        help="Among the picks, prefer at least 1 high-event clip (hard) + 1 modest-event clip (control) per subset.")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    full_ds = _build_dataset(cfg, args.bucket)
    # Build a per-subset map: subset_name -> list[(global_idx, base_ds)]
    if not hasattr(full_ds, "datasets"):
        raise SystemExit("Expected ConcatDataset across subsets; got a single dataset")
    subdatasets = list(full_ds.datasets)
    subset_names = [Path(ds.root).name for ds in subdatasets]
    print(f"Subsets discovered: {subset_names}", flush=True)

    # Scan each subset separately with a DataLoader of batch=1
    per_subset_rows: dict[str, list[dict[str, Any]]] = {n: [] for n in subset_names}
    per_subset_global_offsets: dict[str, int] = {}
    cur = 0
    for ds, sname in zip(subdatasets, subset_names):
        per_subset_global_offsets[sname] = cur
        cur += len(ds)

    for ds, sname in zip(subdatasets, subset_names):
        if len(ds) == 0:
            continue
        sample_cap = min(int(args.max_candidates_per_subset), len(ds))
        # Take an evenly spaced subset to scan
        if sample_cap < len(ds):
            indices = np.linspace(0, len(ds) - 1, sample_cap, dtype=int).tolist()
        else:
            indices = list(range(len(ds)))
        sub = Subset(ds, indices)
        loader = DataLoader(sub, batch_size=1, shuffle=False, collate_fn=collate_hoi, num_workers=0)
        for local_i, batch in zip(indices, loader):
            seq_len = int(batch["seq_len"][0].item())
            if seq_len < 2 * int(args.window_k):
                continue
            cs = batch["contact_state"][0].detach().cpu().numpy().astype(np.float32)
            stats = _score_clip(
                cs, seq_len,
                threshold=float(args.threshold),
                window_k=int(args.window_k),
                edge_margin=int(args.edge_margin),
                flicker_max_frames=int(args.flicker_max_frames),
            )
            if stats["n_valid_v2_slope"] < 1 and stats["n_hand_contact_frames"] < int(args.window_k):
                continue
            text = str(batch["text"][0])
            seq_id = str(batch["seq_id"][0])
            obj_id = str(batch.get("object_id", [""])[0])
            row = {
                "subset": sname,
                "dataset_index_within_subset": int(local_i),
                "dataset_global_index": int(per_subset_global_offsets[sname] + local_i),
                "seq_id": seq_id,
                "object_id": obj_id,
                "text": text[:200],
                "T": int(seq_len),
                "keywords": _action_keywords(text),
                **stats,
            }
            per_subset_rows[sname].append(row)

    # Per-subset top-K with diversity preference
    rejected_summary: dict[str, dict[str, Any]] = {}
    selected: list[dict[str, Any]] = []
    per_subset_target = int(args.per_subset_target)
    per_subset_min = int(args.per_subset_minimum)

    for sname in subset_names:
        rows = per_subset_rows[sname]
        # Sort by validity-event count desc, then by seq_len asc (prefer shorter to fit GPU)
        rows.sort(
            key=lambda r: (-r["n_valid_v2_slope"], -r["n_contact_events"], r["T"]),
        )
        # Diversity ranking among top candidates: use top 3x target as candidate pool
        pool_size = max(per_subset_target * 3, per_subset_target)
        pool = rows[:pool_size]
        order = _diversity_rank(pool)
        picks: list[dict[str, Any]] = []
        for idx in order:
            picks.append(pool[idx])
            if len(picks) >= per_subset_target:
                break
        # Fall back if we don't have enough
        if len(picks) < per_subset_min:
            # Pad from all rows (in original sort order)
            seen = {(p["dataset_global_index"]) for p in picks}
            for r in rows:
                if r["dataset_global_index"] in seen:
                    continue
                picks.append(r)
                seen.add(r["dataset_global_index"])
                if len(picks) >= per_subset_min:
                    break
        for p in picks:
            p["selection_reason"] = (
                f"top by n_valid_v2_slope={p['n_valid_v2_slope']} with text-diversity"
            )
        selected.extend(picks)
        rejected_summary[sname] = {
            "n_candidates_scanned": int(len(rows)),
            "n_after_filters": int(len(rows)),  # already filtered to valid candidates
            "n_selected": int(len(picks)),
            "min_n_valid_v2_slope_in_pool": int(min((r["n_valid_v2_slope"] for r in rows), default=0)),
            "max_n_valid_v2_slope_in_pool": int(max((r["n_valid_v2_slope"] for r in rows), default=0)),
        }

    composition = {sname: sum(1 for r in selected if r["subset"] == sname) for sname in subset_names}

    payload = {
        "config": str(args.config),
        "bucket": args.bucket,
        "per_subset_target": per_subset_target,
        "per_subset_minimum": per_subset_min,
        "filters": {
            "threshold": float(args.threshold),
            "window_k": int(args.window_k),
            "edge_margin": int(args.edge_margin),
            "flicker_max_frames": int(args.flicker_max_frames),
        },
        "subset_names": subset_names,
        "realized_composition": composition,
        "per_subset_summary": rejected_summary,
        # Match recon_ladder selection format so _load_selection can read it.
        "selected": [
            {
                "seq_id": r["seq_id"],
                "subset": r["subset"],
                "object_id": r["object_id"],
                "text": r["text"],
                "T": r["T"],
                "dataset_global_index": r["dataset_global_index"],
                "n_valid_v2_slope": r["n_valid_v2_slope"],
                "n_contact_events": r["n_contact_events"],
                "n_flicker": r["n_flicker"],
                "n_boundary": r["n_boundary"],
                "first_valid_event_frame": r["first_valid_event_frame"],
                "first_valid_event_part_idx": r["first_valid_event_part_idx"],
                "first_valid_event_kind": r["first_valid_event_kind"],
                "selection_reason": r["selection_reason"],
                # Keys expected by _event_from_metadata in
                # recon_ladder_truncated_rollout_diagnostic — wrap in 'event' so
                # the existing helpers can parse it cleanly.
                "event": {
                    "kind": r["first_valid_event_kind"] or "onset",
                    "frame": max(0, r["first_valid_event_frame"]),
                    "part": "L_hand" if r["first_valid_event_part_idx"] == 0 else "R_hand",
                },
            }
            for r in selected
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")

    lines = [
        "# Subset-Balanced Failure-Atlas Selection (Round 9, Task 1)",
        "",
        f"- Config: `{args.config}`",
        f"- Bucket: {args.bucket}",
        f"- Per-subset target: {per_subset_target}; minimum: {per_subset_min}",
        f"- Realized composition: {composition}",
        "",
        "## Filters",
        "",
        f"- threshold = {args.threshold}",
        f"- window_k = {args.window_k}",
        f"- edge_margin = {args.edge_margin}",
        f"- flicker_max_frames = {args.flicker_max_frames}",
        f"- max_candidates_scanned_per_subset = {args.max_candidates_per_subset}",
        "",
        "## Per-subset candidate-pool stats",
        "",
        "| subset | scanned | n_selected | min n_valid_v2 | max n_valid_v2 |",
        "|--------|---------|-----------:|---------------:|---------------:|",
    ]
    for sname in subset_names:
        s = rejected_summary[sname]
        lines.append(
            f"| {sname} | {s['n_candidates_scanned']} | {s['n_selected']} | "
            f"{s['min_n_valid_v2_slope_in_pool']} | {s['max_n_valid_v2_slope_in_pool']} |"
        )

    lines += [
        "",
        "## Selected clips",
        "",
        "| subset | seq_id | object_id | T | n_valid_v2 | n_contact | n_flicker | text |",
        "|--------|--------|-----------|---|------------|-----------|-----------|------|",
    ]
    for r in selected:
        lines.append(
            f"| {r['subset']} | {r['seq_id']} | {r['object_id']} | {r['T']} | "
            f"{r['n_valid_v2_slope']} | {r['n_contact_events']} | {r['n_flicker']} | "
            f"{r['text'][:90].replace('|', '/')} |"
        )
    lines.append("")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    print(f"Realized composition: {composition}")


if __name__ == "__main__":
    main()
