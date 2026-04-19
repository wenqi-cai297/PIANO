"""Pick a few sequence ids by keyword match in text description.

Used to grab "representative" sequences (containing words like sit/lift/
move) for visualization — rather than just the first N entries in
metadata, which may all describe the same action.

Usage:
    piano-sample-by-keyword \\
        --data-dir /path/to/piano/<subset> \\
        [--keywords sit lift move pick carry] \\
        [--per-keyword 1] \\
        [--output-dir runs/checks/sample_seq_by_keyword/<ts>/]
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from piano.utils.io_utils import ensure_dir, load_json, save_json


DEFAULT_KEYWORDS: tuple[str, ...] = (
    "sit", "lift", "move", "pick", "carry", "put", "push", "pull",
    "hold", "swing", "hit", "place",
)


def find_samples_by_keyword(
    metadata: list[dict],
    keywords: list[str],
    per_keyword: int,
) -> list[dict]:
    """Return at most *per_keyword* sequences for each keyword matched
    (case-insensitive) in the text field. Returns a flat de-duplicated list.
    """
    seen: set[str] = set()
    picked: list[dict] = []
    for kw in keywords:
        kw_lower = kw.lower()
        count = 0
        for m in metadata:
            if count >= per_keyword:
                break
            sid = m.get("seq_id", "")
            text = m.get("text", "").lower()
            if sid in seen:
                continue
            if kw_lower in text:
                picked.append({
                    "seq_id": sid,
                    "text": m.get("text", ""),
                    "matched_keyword": kw,
                })
                seen.add(sid)
                count += 1
    return picked


def run(data_dir: Path, keywords: list[str], per_keyword: int, output_dir: Path) -> None:
    output_dir = ensure_dir(output_dir)
    metadata = load_json(data_dir / "metadata.json")

    print(f"Sampling from {data_dir} ({len(metadata)} sequences)")
    print(f"Keywords: {keywords}  (up to {per_keyword} per keyword)")
    print()

    samples = find_samples_by_keyword(metadata, keywords, per_keyword)
    for s in samples:
        print(f"  [{s['matched_keyword']:8s}] {s['seq_id']}")
        print(f"             {s['text'][:100]}")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "data_dir": str(data_dir),
        "keywords": keywords,
        "per_keyword": per_keyword,
        "num_samples": len(samples),
        "samples": samples,
    }
    save_json(output_dir / "summary.json", summary)

    # Also emit a plain list of seq_ids for easy piping
    (output_dir / "seq_ids.txt").write_text(
        "\n".join(s["seq_id"] for s in samples) + "\n",
    )
    print(f"\nSummary: {output_dir / 'summary.json'}")
    print(f"Seq ids: {output_dir / 'seq_ids.txt'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--keywords", nargs="+", default=list(DEFAULT_KEYWORDS),
        help=f"Keywords to match (default: {list(DEFAULT_KEYWORDS)})",
    )
    parser.add_argument("--per-keyword", type=int, default=1)
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Default: runs/checks/sample_seq_by_keyword/<timestamp>/",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.output_dir is None:
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        output_dir = Path("runs/checks/sample_seq_by_keyword") / ts
    else:
        output_dir = args.output_dir
    run(args.data_dir, args.keywords, args.per_keyword, output_dir)


if __name__ == "__main__":
    main()
