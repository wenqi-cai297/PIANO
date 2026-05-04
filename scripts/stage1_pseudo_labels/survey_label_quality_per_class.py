"""Per-action-class pseudo-label quality survey.

For each clip in (imhd, neuraldome, omomo_correct_v2, chairs):
  - Parse object/action keyword from seq_id (and text where naming
    is opaque, e.g., chairs has numeric object IDs).
  - Load v11 (looser) and v12_strict pseudo labels.
  - Compute contact frame fractions per body part.
  - Aggregate by class keyword.

Output: a markdown table sorted by v12_strict_hand_contact_frac
ascending (most under-labeled classes first), plus per-subset
breakdowns. Goal: identify which interaction classes have
v12_strict << v11 (label sparsity in v12_strict harming generation)
and which classes have stable, dense labels in both.

Usage:
    python scripts/stage1_pseudo_labels/survey_label_quality_per_class.py \\
        --output-dir analyses/2026-05-05_label_quality_per_class

The --output-dir gets:
  - by_class.csv     — full table per (subset, keyword) combo
  - report.md        — narrative summary with sparse-label outliers
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


# ============================================================================
# Class keyword extraction
# ============================================================================

# Action / object verbs to look for in clip text or seq_id middle segment.
# Order matters: more specific first (e.g., "tabletall" before "table").
ACTION_KEYWORDS = (
    # Sport / dynamic tool use — the v12_strict failure mode
    "tennis", "baseball", "bat",
    "racket", "racquet", "pingpong",
    "kick",
    "swing", "hit", "throw",
    # Held object (continuous grip)
    "suitcase", "trolleycase", "case",
    "bucket", "basket",
    "trashcan", "trashbin",
    "box", "smallbox", "largebox", "plasticbox",
    "bag", "backpack",
    "umbrella",
    "cup", "bottle",
    # Carry / lift
    "tripod", "lamp", "floorlamp",
    "monitor", "keyboard",
    "vacuum",
    "book", "bookshelf",
    "pillow", "cushion",
    "flower", "vase",
    "pan", "pot",
    "clothesstand",
    # Push / pull (heavy stationary)
    "table", "tabletall", "desk", "smalltable", "largetable",
    "chair", "smallchair", "whitechair", "woodchair", "smallsofa",
    # Sit / lean (chair-class, slow contact)
    "stool", "bench", "sofa",
)


def _extract_keyword_from_seq_id_or_text(
    seq_id: str, text: str | None,
) -> str:
    """Best-effort: pull a human-readable interaction class.

    For imhd: middle segments encode object — e.g.
    `20230825_wangwzh_bat_bat_lefthand_swing_5_0` -> bat.
    For omomo / neuraldome: second segment is object —
    `subject04_tabletall_1100` -> tabletall, `sub5_plasticbox_041` ->
    plasticbox.
    For chairs: seq_id is numeric (`Sub1475_Obj48_...`). Fall back to
    parsing the first noun-y token in `text`.
    """
    s = seq_id.lower()
    for kw in ACTION_KEYWORDS:
        if f"_{kw}_" in s or s.startswith(f"{kw}_") or s.endswith(f"_{kw}"):
            return kw
    # chairs: object name not in seq_id, try text
    if text:
        t = text.lower()
        for kw in ACTION_KEYWORDS:
            if re.search(rf"\b{re.escape(kw)}\b", t):
                return kw
    return "unknown"


# ============================================================================
# Per-clip stats
# ============================================================================

def _load_contact_state(npz_path: Path) -> np.ndarray | None:
    """Load contact_state (T, 5) from a pseudo_label npz, or None if missing."""
    if not npz_path.exists():
        return None
    try:
        d = np.load(npz_path, allow_pickle=False)
        if "contact_state" not in d.files:
            return None
        return d["contact_state"].astype(np.float32)
    except Exception:
        return None


def _contact_fracs(cs: np.ndarray | None) -> dict[str, float]:
    """Per-body-part contact fraction. cs is (T, 5) {0,1}."""
    if cs is None or cs.size == 0:
        return {p: float("nan") for p in (
            "l_hand", "r_hand", "l_foot", "r_foot", "pelvis", "any_hand",
            "any_foot", "any",
        )}
    return {
        "l_hand": float(cs[:, 0].mean()),
        "r_hand": float(cs[:, 1].mean()),
        "l_foot": float(cs[:, 2].mean()),
        "r_foot": float(cs[:, 3].mean()),
        "pelvis": float(cs[:, 4].mean()),
        "any_hand": float(np.maximum(cs[:, 0], cs[:, 1]).mean()),
        "any_foot": float(np.maximum(cs[:, 2], cs[:, 3]).mean()),
        "any": float(cs.max(axis=-1).mean()),
    }


# ============================================================================
# Main survey
# ============================================================================

def _iter_clip_records(
    data_root: Path,
    subsets: Iterable[str],
    a_label: str = "v12_strict",
    b_label: str = "v11",
) -> Iterable[dict]:
    """Yield per-clip dicts comparing label set ``a`` vs label set ``b``.

    Both labels are loaded from ``<subset>/pseudo_labels/<...>``. The
    "v11" label set lives at ``pseudo_labels/`` (root, the original
    layout) — `b_label="v11"` resolves there. Any other label name
    resolves to ``pseudo_labels/<label>/``.
    """
    def _label_dir(subset: str, label: str) -> Path:
        if label == "v11":
            return data_root / subset / "pseudo_labels"
        return data_root / subset / "pseudo_labels" / label

    for subset in subsets:
        meta_path = data_root / subset / "metadata.json"
        if not meta_path.exists():
            print(f"[skip] {subset}: no metadata.json")
            continue
        with open(meta_path, encoding="utf-8") as f:
            metadata = json.load(f)
        a_dir = _label_dir(subset, a_label)
        b_dir = _label_dir(subset, b_label)
        for m in metadata:
            sid = m["seq_id"]
            text = m.get("text", "")
            kw = _extract_keyword_from_seq_id_or_text(sid, text)
            cs_a = _load_contact_state(a_dir / f"{sid}.npz")
            cs_b = _load_contact_state(b_dir / f"{sid}.npz")
            if cs_a is None and cs_b is None:
                continue
            f_a = _contact_fracs(cs_a)
            f_b = _contact_fracs(cs_b)
            yield {
                "subset": subset,
                "seq_id": sid,
                "keyword": kw,
                "num_frames": int(m.get("num_frames", 0)) or (
                    cs_a.shape[0] if cs_a is not None
                    else (cs_b.shape[0] if cs_b is not None else 0)
                ),
                # Keys keep the legacy v12_/v11_ prefix to avoid touching
                # downstream aggregation. ``v12_*`` = label set A,
                # ``v11_*`` = label set B.
                **{f"v12_{k}": v for k, v in f_a.items()},
                **{f"v11_{k}": v for k, v in f_b.items()},
            }


def _aggregate_by_keyword(records: list[dict]) -> list[dict]:
    by_class: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        by_class[(r["subset"], r["keyword"])].append(r)
    rows: list[dict] = []
    for (subset, kw), recs in by_class.items():
        n_clips = len(recs)
        # Frame-weighted means.
        total_frames = sum(r["num_frames"] for r in recs) or 1
        def _w_mean(field: str) -> float:
            num = sum(
                (r[field] if not (np.isnan(r[field])) else 0.0) * r["num_frames"]
                for r in recs
            )
            return float(num / total_frames)
        rows.append({
            "subset": subset,
            "keyword": kw,
            "n_clips": n_clips,
            "total_frames": total_frames,
            "v12_any_hand_frac": _w_mean("v12_any_hand"),
            "v12_any_foot_frac": _w_mean("v12_any_foot"),
            "v12_any_frac":      _w_mean("v12_any"),
            "v11_any_hand_frac": _w_mean("v11_any_hand"),
            "v11_any_foot_frac": _w_mean("v11_any_foot"),
            "v11_any_frac":      _w_mean("v11_any"),
            # v12 / v11 ratio for hand contact: large drop = strict killed
            # too much. ~1.0 = labels stable. Add small epsilon for div.
            "v12_v11_hand_ratio": (
                _w_mean("v12_any_hand") / (_w_mean("v11_any_hand") + 1e-6)
            ),
            "v12_v11_foot_ratio": (
                _w_mean("v12_any_foot") / (_w_mean("v11_any_foot") + 1e-6)
            ),
        })
    rows.sort(key=lambda r: (r["v12_any_hand_frac"], -r["n_clips"]))
    return rows


def _write_csv(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "subset", "keyword", "n_clips", "total_frames",
        "v12_any_hand_frac", "v12_any_foot_frac", "v12_any_frac",
        "v11_any_hand_frac", "v11_any_foot_frac", "v11_any_frac",
        "v12_v11_hand_ratio", "v12_v11_foot_ratio",
    ]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in r.items() if k in fieldnames})


def _write_report(
    rows: list[dict], records: list[dict], out: Path,
    a_label: str = "v12_strict", b_label: str = "v11",
) -> None:
    """Markdown summary highlighting sparse-label outliers."""
    out.parent.mkdir(parents=True, exist_ok=True)

    n_total = len(records)
    n_classes = len(rows)
    n_subsets = len({r["subset"] for r in records})

    # Outliers by hand sparsity (interesting for swing / tool-use):
    # min n_clips >= 5 to skip noise, v12_v11_hand_ratio < 0.4 = strict
    # killed > 60% of v11 hand contact.
    sparse_hand = [
        r for r in rows
        if r["n_clips"] >= 5 and r["v11_any_hand_frac"] >= 0.20
        and r["v12_v11_hand_ratio"] < 0.4
    ]

    # Stable hand classes (low ratio drop):
    stable_hand = [
        r for r in rows
        if r["n_clips"] >= 5 and r["v11_any_hand_frac"] >= 0.20
        and r["v12_v11_hand_ratio"] >= 0.7
    ]

    # Foot-related diagnostics:
    sparse_foot = [
        r for r in rows
        if r["n_clips"] >= 5 and r["v11_any_foot_frac"] >= 0.05
        and r["v12_v11_foot_ratio"] < 0.4
    ]

    lines: list[str] = []
    lines.append(f"# Pseudo-label quality survey — {a_label} vs {b_label} by class")
    lines.append("")
    lines.append(
        f"Scope: {n_total} clips across {n_subsets} subsets, grouped into "
        f"{n_classes} (subset, keyword) classes. "
        f"Columns prefixed `v12_*` are label set A = `{a_label}`, "
        f"prefixed `v11_*` are label set B = `{b_label}`."
    )
    lines.append("")

    lines.append("## Method")
    lines.append("")
    lines.append(
        "- Group clips by `(subset, keyword)` where keyword is parsed from "
        "seq_id middle segments or, for chairs (numeric object IDs), from "
        "the text caption."
    )
    lines.append(
        "- Compute frame-weighted mean of v12_strict and v11 contact frame "
        "fraction per body part (any_hand = max(l_hand, r_hand) per frame, "
        "averaged over frames; same for foot and any-part)."
    )
    lines.append(
        "- Sparsity flag: `v12_v11_hand_ratio = v12_any_hand / v11_any_hand`. "
        "Values < 0.4 (60%+ drop) flagged as suspect."
    )
    lines.append("")

    lines.append("## Sparse-hand classes (likely affected by swing failure)")
    lines.append("")
    if not sparse_hand:
        lines.append("_(none flagged)_")
    else:
        lines.append(
            "Filter: ≥5 clips, v11 any-hand ≥ 20%, v12 strict drops > 60%."
        )
        lines.append("")
        lines.append(
            "| subset | keyword | n_clips | v12_hand | v11_hand | ratio | "
            "v12_foot | v11_foot |"
        )
        lines.append(
            "|---|---|---:|---:|---:|---:|---:|---:|"
        )
        for r in sparse_hand:
            lines.append(
                f"| {r['subset']} | {r['keyword']} | {r['n_clips']} | "
                f"{r['v12_any_hand_frac']:.3f} | {r['v11_any_hand_frac']:.3f} | "
                f"**{r['v12_v11_hand_ratio']:.2f}** | "
                f"{r['v12_any_foot_frac']:.3f} | {r['v11_any_foot_frac']:.3f} |"
            )
    lines.append("")

    lines.append("## Stable-hand classes (v12_strict labels look fine)")
    lines.append("")
    if not stable_hand:
        lines.append("_(none flagged)_")
    else:
        lines.append("Filter: ≥5 clips, v11 any-hand ≥ 20%, v12/v11 ratio ≥ 0.7.")
        lines.append("")
        lines.append(
            "| subset | keyword | n_clips | v12_hand | v11_hand | ratio |"
        )
        lines.append("|---|---|---:|---:|---:|---:|")
        for r in stable_hand:
            lines.append(
                f"| {r['subset']} | {r['keyword']} | {r['n_clips']} | "
                f"{r['v12_any_hand_frac']:.3f} | {r['v11_any_hand_frac']:.3f} | "
                f"{r['v12_v11_hand_ratio']:.2f} |"
            )
    lines.append("")

    lines.append("## Sparse-foot classes (kicking, stepping, foot-on-furniture)")
    lines.append("")
    if not sparse_foot:
        lines.append("_(none flagged at v11 ≥ 0.05)_")
    else:
        lines.append("Filter: ≥5 clips, v11 any-foot ≥ 5%, v12/v11 ratio < 0.4.")
        lines.append("")
        lines.append(
            "| subset | keyword | n_clips | v12_foot | v11_foot | ratio |"
        )
        lines.append("|---|---|---:|---:|---:|---:|")
        for r in sparse_foot:
            lines.append(
                f"| {r['subset']} | {r['keyword']} | {r['n_clips']} | "
                f"{r['v12_any_foot_frac']:.3f} | {r['v11_any_foot_frac']:.3f} | "
                f"**{r['v12_v11_foot_ratio']:.2f}** |"
            )
    lines.append("")

    lines.append("## Full table (top 30 by hand-sparsity, n_clips ≥ 5)")
    lines.append("")
    rows_main = [r for r in rows if r["n_clips"] >= 5][:30]
    lines.append(
        "| subset | keyword | n_clips | v12_hand | v11_hand | ratio | "
        "v12_foot | v11_foot |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in rows_main:
        lines.append(
            f"| {r['subset']} | {r['keyword']} | {r['n_clips']} | "
            f"{r['v12_any_hand_frac']:.3f} | {r['v11_any_hand_frac']:.3f} | "
            f"{r['v12_v11_hand_ratio']:.2f} | "
            f"{r['v12_any_foot_frac']:.3f} | {r['v11_any_foot_frac']:.3f} |"
        )
    lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-root", type=Path,
        default=Path("E:/Project/Datasets/InterAct/piano"),
    )
    parser.add_argument(
        "--subsets", nargs="*",
        default=["imhd", "neuraldome", "omomo_correct_v2", "chairs"],
    )
    parser.add_argument(
        "--label-a", default="v12_strict",
        help="Label set A. Resolves to <subset>/pseudo_labels/<label>/. "
             "Default v12_strict. Use 'v13_centered' to compare a re-extracted "
             "label set against v11.",
    )
    parser.add_argument(
        "--label-b", default="v11",
        help="Label set B. The literal value 'v11' resolves to "
             "<subset>/pseudo_labels/ (the original layout); any other value "
             "resolves to <subset>/pseudo_labels/<label>/.",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory to write by_class.csv + report.md.",
    )
    args = parser.parse_args()

    print(f"Surveying {args.subsets} under {args.data_root} ...")
    print(f"  label A = {args.label_a}, label B = {args.label_b}")
    records = list(_iter_clip_records(
        args.data_root, args.subsets,
        a_label=args.label_a, b_label=args.label_b,
    ))
    print(f"  loaded {len(records)} clips")
    rows = _aggregate_by_keyword(records)
    print(f"  aggregated into {len(rows)} (subset, keyword) classes")

    csv_out = args.output_dir / "by_class.csv"
    md_out = args.output_dir / "report.md"
    _write_csv(rows, csv_out)
    _write_report(rows, records, md_out, a_label=args.label_a, b_label=args.label_b)
    print(f"Wrote: {csv_out}")
    print(f"Wrote: {md_out}")


if __name__ == "__main__":
    main()
