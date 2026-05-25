"""Convert a Round-27/28 subset-indices JSON into diagnostic selection JSON.

The trainer consumes ``{"indices": [...], "clips": [...]}``; the diagnostic
scripts consume ``{"selected": [{"subset": ..., "seq_id": ...}, ...]}``.
Keeping this as a small entry script avoids fragile inline shell/Python blocks
and preserves the source bucket (train vs val) in the output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _selected_row(clip: dict[str, Any]) -> dict[str, Any]:
    return {
        "subset": clip["subset"],
        "seq_id": clip["seq_id"],
        "mode_category": clip.get(
            "mode_category",
            clip.get("body_action_category", "unknown"),
        ),
        "text": clip.get("text", ""),
        "confidence": 1.0,
        "n_known_valid_modes": 1,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument(
        "--bucket",
        choices=["train", "val"],
        default=None,
        help="Override bucket; defaults to the source JSON's bucket field.",
    )
    args = parser.parse_args()

    src = json.loads(args.indices_json.read_text(encoding="utf-8"))
    clips = src.get("clips", [])
    if not clips:
        raise SystemExit(f"No clips found in {args.indices_json}")
    bucket = str(args.bucket or src.get("bucket", "train"))
    out = {
        "description": str(args.description),
        "selection_source": str(args.indices_json),
        "bucket": bucket,
        "n_clips": len(clips),
        "selected": [_selected_row(c) for c in clips],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {len(clips)} clips to {args.output} (bucket={bucket})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
