"""Apply caption-eval major exclusions to InterAct metadata_clean.json.

This is intentionally narrower than the older generic pseudo-label cleaner:
it only removes sequences that the final caption-alignment scan marked as
major mismatches. The raw metadata.json stays intact; training code already
prefers metadata_clean.json when present.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from piano.utils.io_utils import load_json, save_json


DEFAULT_ROOT = Path("E:/Project/Datasets/InterAct/piano_official_process_4")
DEFAULT_EXCLUDE = Path(
    "analyses/2026-05-06_v18_h10_f05_pelvis20_official_semantic_marker_caption_eval_strict/"
    "exclude_candidates.json"
)
DEFAULT_SUBSETS = ("chairs", "imhd", "neuraldome", "omomo_correct_v2")


def _entry_key(entry: dict[str, Any]) -> tuple[str, str]:
    return str(entry["subset"]), str(entry["seq_id"])


def load_exclusions(path: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    candidates = load_json(path)
    by_subset: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for item in candidates:
        subset, seq_id = _entry_key(item)
        by_subset[subset][seq_id].append(item)
    return {subset: dict(seq_map) for subset, seq_map in by_subset.items()}


def apply_subset(
    subset_dir: Path,
    exclusions: dict[str, list[dict[str, Any]]],
    dry_run: bool = False,
) -> dict[str, Any]:
    metadata_path = subset_dir / "metadata.json"
    if not metadata_path.exists():
        return {"subset": subset_dir.name, "error": f"missing {metadata_path}"}

    metadata = load_json(metadata_path)
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    rule_counts: Counter[str] = Counter()

    for entry in metadata:
        seq_id = str(entry["seq_id"])
        reasons = exclusions.get(seq_id)
        if not reasons:
            kept.append(entry)
            continue

        rules = [str(reason.get("rule", "unknown")) for reason in reasons]
        rule_counts.update(rules)
        dropped.append(
            {
                "seq_id": seq_id,
                "text": (entry.get("text") or "")[:240],
                "rules": rules,
                "observed": [reason.get("observed") for reason in reasons],
            }
        )

    report = {
        "timestamp": datetime.now().isoformat(),
        "subset": subset_dir.name,
        "source_metadata": str(metadata_path),
        "num_in_metadata": len(metadata),
        "num_kept": len(kept),
        "num_dropped_unique": len(dropped),
        "num_major_findings": sum(rule_counts.values()),
        "rule_counts": dict(rule_counts.most_common()),
        "dropped": dropped,
    }

    if not dry_run:
        save_json(subset_dir / "metadata_clean.json", kept)
        save_json(subset_dir / "caption_exclusion_report.json", report)

    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--exclude-candidates", type=Path, default=DEFAULT_EXCLUDE)
    parser.add_argument("--subsets", nargs="+", default=DEFAULT_SUBSETS)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    exclusions_by_subset = load_exclusions(args.exclude_candidates)

    root_report: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "data_dir": str(args.data_dir),
        "exclude_candidates": str(args.exclude_candidates),
        "dry_run": bool(args.dry_run),
        "subsets": [],
        "totals": {
            "num_in_metadata": 0,
            "num_kept": 0,
            "num_dropped_unique": 0,
            "num_major_findings": 0,
        },
        "rule_counts": {},
    }
    total_rules: Counter[str] = Counter()

    for subset in args.subsets:
        subset_dir = args.data_dir / subset
        report = apply_subset(
            subset_dir,
            exclusions_by_subset.get(subset, {}),
            dry_run=args.dry_run,
        )
        if "error" in report:
            print(f"[{subset}] error: {report['error']}")
            root_report["subsets"].append(report)
            continue

        print(
            f"[{subset}] kept {report['num_kept']} / {report['num_in_metadata']} "
            f"(dropped {report['num_dropped_unique']} unique, "
            f"{report['num_major_findings']} findings)"
        )
        for rule, count in report["rule_counts"].items():
            print(f"    {rule}: {count}")

        root_report["subsets"].append(report)
        for key in root_report["totals"]:
            root_report["totals"][key] += int(report[key])
        total_rules.update(report["rule_counts"])

    root_report["rule_counts"] = dict(total_rules.most_common())
    totals = root_report["totals"]
    print("\n=== TOTALS ===")
    print(
        f"kept {totals['num_kept']} / {totals['num_in_metadata']} "
        f"(dropped {totals['num_dropped_unique']} unique, "
        f"{totals['num_major_findings']} findings)"
    )
    for rule, count in root_report["rule_counts"].items():
        print(f"  {rule}: {count}")

    if not args.dry_run:
        save_json(args.data_dir / "caption_exclusion_report.json", root_report)


if __name__ == "__main__":
    main()
