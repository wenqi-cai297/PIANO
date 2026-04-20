"""Dump InterAct text / annotation file formats for all 4 subsets.

The `piano-action-segment-sweep` first pass failed with 0 / N parseable
text.txt because the HumanML3D-style tail (#postag#start#end) either
isn't present or has placeholder 0.0 timestamps. The real action
windows live in CSV files under `<InterAct>/annotation/<kind>/` (kinds:
action, natural, raw, change, shorten).

This one-shot probe dumps enough raw content to decide the right
parsing path for each of our 4 subsets:

    For each subset:
      - Full content of 3 sample text.txt files (not truncated).
      - List of every CSV file under `annotation/<kind>/` with its size,
        row count, and first 5 rows verbatim.

Output: runs/checks/text_annotations/<ts>/summary.json + preview.md

Usage:
    piano-probe-text-annotations \\
        --interact-dir /media/.../InterAct/InterAct \\
        [--subsets chairs imhd neuraldome omomo_correct_v2]
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

from piano.utils.io_utils import ensure_dir, save_json


DEFAULT_SUBSETS = ("chairs", "imhd", "neuraldome", "omomo_correct_v2")
ANNOTATION_KINDS = ("action", "natural", "raw", "change", "shorten")


def _dump_text_samples(subset_dir: Path, n: int = 3, max_chars: int = 2000) -> list[dict]:
    """Read n text.txt files from subset_dir/sequences_canonical/*/ verbatim."""
    seqs_dir = subset_dir / "sequences_canonical"
    if not seqs_dir.exists():
        return [{"error": f"missing {seqs_dir}"}]
    samples: list[dict] = []
    seq_dirs = sorted([p for p in seqs_dir.iterdir() if p.is_dir()])[:n]
    for sd in seq_dirs:
        text_path = sd / "text.txt"
        entry: dict = {"seq_id": sd.name}
        if not text_path.exists():
            entry["error"] = "text.txt missing"
            samples.append(entry)
            continue
        raw = text_path.read_text(encoding="utf-8", errors="replace")
        entry["length"] = len(raw)
        entry["line_count"] = raw.count("\n") + (0 if raw.endswith("\n") else 1)
        entry["full_content"] = raw[:max_chars]
        entry["truncated"] = len(raw) > max_chars
        samples.append(entry)
    return samples


def _dump_csv(csv_path: Path, n_rows: int = 5, max_cell_chars: int = 500) -> dict:
    """Return headers + first n rows of a CSV file."""
    if not csv_path.exists():
        return {"exists": False, "path": str(csv_path)}
    size = csv_path.stat().st_size
    try:
        with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
    except Exception as e:
        return {"exists": True, "size_bytes": size, "error": str(e)}

    headers = rows[0] if rows else []
    body = rows[1:]
    truncated_rows: list[list[str]] = []
    for row in body[:n_rows]:
        truncated_rows.append([
            (cell if len(cell) <= max_cell_chars else cell[:max_cell_chars] + "...[truncated]")
            for cell in row
        ])
    return {
        "exists": True,
        "size_bytes": size,
        "num_rows": len(body),
        "headers": headers,
        "sample_rows": truncated_rows,
    }


def _list_annotation_kind(ann_dir: Path) -> list[dict]:
    """List every file under annotation/<kind>/ with size."""
    if not ann_dir.exists():
        return []
    entries = []
    for p in sorted(ann_dir.iterdir()):
        if p.is_file():
            entries.append({
                "name": p.name,
                "size_bytes": p.stat().st_size,
            })
    return entries


def run_probe(interact_dir: Path, subsets: tuple[str, ...], output_dir: Path) -> None:
    output_dir = ensure_dir(output_dir)
    summary: dict = {
        "timestamp": datetime.now().isoformat(),
        "interact_dir": str(interact_dir),
        "subsets": subsets,
        "per_subset": {},
        "annotation_root": {},
    }

    md_lines: list[str] = ["# InterAct text / annotation probe", ""]
    md_lines.append(f"- interact_dir: `{interact_dir}`")
    md_lines.append(f"- subsets: {list(subsets)}")
    md_lines.append("")

    # Per-subset: text.txt samples (verbatim)
    for subset in subsets:
        subset_dir = interact_dir / subset
        text_samples = _dump_text_samples(subset_dir)
        summary["per_subset"][subset] = {"text_samples": text_samples}
        md_lines.append(f"## {subset} — sample text.txt files")
        md_lines.append("")
        for entry in text_samples:
            sid = entry.get("seq_id", "?")
            if "error" in entry:
                md_lines.append(f"- **{sid}**: ERROR {entry['error']}")
                continue
            md_lines.append(
                f"- **{sid}** — length={entry['length']}, "
                f"lines={entry['line_count']}, "
                f"{'TRUNCATED' if entry['truncated'] else 'COMPLETE'}"
            )
            md_lines.append("")
            md_lines.append("```")
            md_lines.append(entry["full_content"])
            md_lines.append("```")
            md_lines.append("")

    # Annotation root: list every file under each kind
    ann_root = interact_dir / "annotation"
    md_lines.append("## annotation/ directory inventory")
    md_lines.append("")
    for kind in ANNOTATION_KINDS:
        kind_dir = ann_root / kind
        files = _list_annotation_kind(kind_dir)
        summary["annotation_root"][kind] = {
            "path": str(kind_dir),
            "exists": kind_dir.exists(),
            "files": files,
        }
        md_lines.append(f"### annotation/{kind}/")
        md_lines.append("")
        if not kind_dir.exists():
            md_lines.append("_directory missing_")
            md_lines.append("")
            continue
        if not files:
            md_lines.append("_empty directory_")
            md_lines.append("")
            continue
        md_lines.append("| filename | size (bytes) |")
        md_lines.append("|---|---:|")
        for f in files:
            md_lines.append(f"| `{f['name']}` | {f['size_bytes']:,} |")
        md_lines.append("")

    # For the 4 subsets of interest, dump CSV contents if a matching
    # file exists under each annotation kind. Matching is by basename
    # (e.g. "chairs.csv" under annotation/shorten/).
    md_lines.append("## annotation CSV samples for our 4 subsets")
    md_lines.append("")
    for subset in subsets:
        md_lines.append(f"### {subset}")
        md_lines.append("")
        for kind in ANNOTATION_KINDS:
            csv_path = ann_root / kind / f"{subset}.csv"
            info = _dump_csv(csv_path)
            summary["per_subset"].setdefault(subset, {}).setdefault("annotation_csvs", {})[kind] = info
            if not info.get("exists"):
                md_lines.append(f"- `{kind}/{subset}.csv` — **MISSING**")
                continue
            if "error" in info:
                md_lines.append(f"- `{kind}/{subset}.csv` — ERROR {info['error']}")
                continue
            md_lines.append(
                f"- `{kind}/{subset}.csv` — {info['size_bytes']:,} bytes, "
                f"{info['num_rows']} rows"
            )
            md_lines.append("")
            md_lines.append(f"  headers: `{info['headers']}`")
            md_lines.append("")
            md_lines.append(f"  first {len(info['sample_rows'])} data rows:")
            md_lines.append("")
            for i, row in enumerate(info["sample_rows"]):
                md_lines.append(f"  **row {i}:**")
                md_lines.append("")
                md_lines.append("  ```")
                md_lines.append("  " + " || ".join(repr(c) for c in row))
                md_lines.append("  ```")
                md_lines.append("")
        md_lines.append("")

    save_json(output_dir / "summary.json", summary)
    (output_dir / "preview.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\nWrote {output_dir / 'summary.json'}")
    print(f"Wrote {output_dir / 'preview.md'}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--interact-dir", type=Path,
        default=Path("/media/gpu-server-1/4TB_for_data/Cai/datasets/InterAct/InterAct"),
    )
    p.add_argument("--subsets", nargs="+", default=list(DEFAULT_SUBSETS))
    p.add_argument("--output-dir", type=Path, default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir or (
        Path("runs/checks/text_annotations") /
        datetime.now().strftime("%Y-%m-%d_%H%M%S")
    )
    run_probe(args.interact_dir, tuple(args.subsets), output_dir)


if __name__ == "__main__":
    main()
