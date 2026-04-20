"""Post-hoc pseudo-label quality stats.

Reads an already-extracted ``pseudo_labels/`` directory (plus the
corresponding preprocessed ``motions/`` directory for geometric sanity)
and writes a rich stats report to the same directory. Use this to get
evaluation-grade numbers out of extraction runs that predate the
inline-stats change in ``run_all.py``.

The result is written as ``stats.json`` next to the existing
``summary.json`` (so the original file stays untouched), and a short
markdown summary at ``stats.md``.

Usage::

    piano-pseudo-label-stats --data-dir /media/.../InterAct/piano/chairs
    piano-pseudo-label-stats --all-subsets --piano-root /media/.../InterAct/piano
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

from piano.data.pseudo_labels.stats import (
    aggregate_stats,
    compute_seq_stats,
    make_quality_flags,
)
from piano.utils.io_utils import ensure_dir, load_json, save_json


def _iter_seq_ids(label_dir: Path) -> list[str]:
    """All sequence ids present as ``.npz`` under the label dir."""
    return sorted(p.stem for p in label_dir.glob("*.npz"))


def _format_markdown(
    subset: str | None,
    agg: dict,
    flags: list[str],
    label_dir: Path,
) -> str:
    """Render a short human-readable summary."""
    lines: list[str] = []
    title = subset or label_dir.parent.name or "pseudo-labels"
    lines.append(f"# Pseudo-label stats — {title}")
    lines.append("")
    lines.append(f"- Path: `{label_dir}`")
    lines.append(f"- Sequences evaluated: **{agg.get('num_sequences', 0)}**")
    lines.append(f"- Total frames: {agg.get('total_frames', 0)}")
    lines.append("")

    if agg.get("num_sequences", 0) == 0:
        lines.append("_No sequences found._")
        return "\n".join(lines)

    # Contact
    lines.append("## Contact")
    lines.append("| body part | frame_rate (mean ± std) | no-contact-seq % |")
    lines.append("|---|---|---|")
    for name, s in agg["contact_stats"]["per_body_part"].items():
        lines.append(
            f"| {name} | {s['frame_rate_mean']:.3f} ± {s['frame_rate_std']:.3f} "
            f"| {s['seq_without_contact_fraction'] * 100:.1f}% |"
        )
    lines.append(
        f"- any-part frame rate (mean over seqs): "
        f"{agg['contact_stats']['any_part_frame_rate_mean']:.3f}"
    )
    lines.append(
        f"- zero-contact sequences: "
        f"{agg['contact_stats']['zero_contact_seq_count']} "
        f"({agg['contact_stats']['zero_contact_seq_fraction'] * 100:.1f}%)"
    )
    lines.append("")

    # Phase
    lines.append("## Phase")
    lines.append("| phase | frame fraction | seq-reached fraction |")
    lines.append("|---|---|---|")
    fd = agg["phase_stats"]["frame_distribution"]
    sr = agg["phase_stats"]["seq_reached_phase_fraction"]
    for name in fd:
        lines.append(
            f"| {name} | {fd[name] * 100:.1f}% | {sr[name] * 100:.1f}% |"
        )
    lines.append(
        f"- transitions / seq: "
        f"mean={agg['phase_stats']['mean_transitions_per_seq']:.1f}, "
        f"median={agg['phase_stats']['median_transitions_per_seq']:.1f}"
    )
    lines.append(
        f"- seqs with zero transitions: "
        f"{agg['phase_stats']['seq_with_zero_transitions_count']} "
        f"({agg['phase_stats']['seq_with_zero_transitions_fraction'] * 100:.1f}%)"
    )
    lines.append("")

    # Support
    lines.append("## Support")
    lines.append("| support | frame fraction | seq-entered fraction |")
    lines.append("|---|---|---|")
    sfd = agg["support_stats"]["frame_distribution"]
    ssr = agg["support_stats"]["seq_entered_support_fraction"]
    for name in sfd:
        lines.append(
            f"| {name} | {sfd[name] * 100:.1f}% | {ssr[name] * 100:.1f}% |"
        )
    lines.append(
        f"- seqs stuck in `both_feet` only: "
        f"{agg['support_stats']['seq_stuck_in_both_feet_count']} "
        f"({agg['support_stats']['seq_stuck_in_both_feet_fraction'] * 100:.1f}%)"
    )
    lines.append("")

    # Target
    lines.append("## Contact target (soft patch assignment)")
    ts = agg["target_stats"]
    lines.append(
        f"- entropy: mean={ts['entropy_mean']:.3f}, median={ts['entropy_median']:.3f}, "
        f"p10={ts['entropy_p10']:.3f}, p90={ts['entropy_p90']:.3f} "
        f"(max possible = {ts['entropy_max_possible']:.3f})"
    )
    lines.append(
        f"- degenerate (~hard) seqs: "
        f"{ts['degenerate_seq_count']} / {ts['num_sequences_with_contact_frames']} "
        f"({ts['degenerate_seq_fraction'] * 100:.1f}%)"
    )
    lines.append(f"- patch coverage: {ts['patch_coverage_fraction'] * 100:.1f}% of patches used")
    lines.append("")

    # Geometric
    geo = agg["geometric_sanity"].get("min_hand_to_obj_center_dist_m")
    lines.append("## Geometric sanity")
    if geo:
        lines.append(
            f"- min hand-to-object-center distance (per seq): "
            f"median={geo['median']:.3f} m, p90={geo['p90']:.3f} m, max={geo['max']:.3f} m"
        )
        lines.append(
            f"- outlier seqs (> 2 m): {geo['outlier_seq_count_gt_2m']}"
        )
    else:
        lines.append("_No object data available in motion files._")
    lines.append("")

    # Flags
    lines.append("## Quality flags")
    if flags:
        for f in flags:
            lines.append(f"- {f}")
    else:
        lines.append("_None fired._")
    lines.append("")

    return "\n".join(lines)


def run_for_subset(
    data_dir: Path,
    label_dir: Path | None = None,
    output_stats_path: Path | None = None,
    subset_hint: str | None = None,
    num_patches: int = 16,
) -> dict:
    """Compute stats for one preprocessed subset + its pseudo-label dir.

    Parameters
    ----------
    data_dir : e.g. ``/.../piano/chairs``; contains ``motions/`` + ``metadata.json``.
    label_dir : defaults to ``<data_dir>/pseudo_labels``.
    output_stats_path : defaults to ``<label_dir>/stats.json``.
    subset_hint : used to specialise quality flags (pass the subset name).
    """
    data_dir = Path(data_dir)
    label_dir = Path(label_dir) if label_dir else data_dir / "pseudo_labels"
    if not label_dir.exists():
        raise FileNotFoundError(f"Pseudo-label dir not found: {label_dir}")

    metadata = load_json(data_dir / "metadata.json")
    obj_by_seq = {m["seq_id"]: m.get("object_id") for m in metadata}

    seq_ids = _iter_seq_ids(label_dir)
    print(f"Scanning {len(seq_ids)} sequences in {label_dir}")

    per_seq = []
    n_missing_motion = 0
    for sid in tqdm(seq_ids, desc=subset_hint or label_dir.parent.name):
        labels_path = label_dir / f"{sid}.npz"
        try:
            labels = np.load(labels_path, allow_pickle=False)
            labels_dict = {k: labels[k] for k in labels.files}
        except Exception as e:
            print(f"  [warn] failed to load {labels_path.name}: {e}")
            continue

        motion_path = data_dir / "motions" / f"{sid}.npz"
        joints = None
        obj_pos = None
        if motion_path.exists():
            try:
                mdata = np.load(motion_path, allow_pickle=False)
                if "joints_22" in mdata.files:
                    joints = mdata["joints_22"]
                if "object_positions" in mdata.files:
                    obj_pos = mdata["object_positions"]
            except Exception as e:
                print(f"  [warn] failed to load motion for {sid}: {e}")
        else:
            n_missing_motion += 1

        try:
            per_seq.append(
                compute_seq_stats(
                    seq_id=sid,
                    labels=labels_dict,
                    joints_22=joints,
                    object_positions=obj_pos,
                )
            )
        except Exception as e:
            print(f"  [warn] stats failed for {sid}: {e}")

    agg = aggregate_stats(per_seq, num_patches=num_patches)
    flags = make_quality_flags(agg, subset_hint=subset_hint)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "data_dir": str(data_dir),
        "label_dir": str(label_dir),
        "subset": subset_hint,
        "num_label_files_scanned": len(seq_ids),
        "num_sequences_with_missing_motion": n_missing_motion,
        "num_patches": num_patches,
        "quality_flags": flags,
        "stats": agg,
    }

    if output_stats_path is None:
        output_stats_path = label_dir / "stats.json"
    save_json(output_stats_path, summary)

    md_path = output_stats_path.with_suffix(".md")
    md_path.write_text(
        _format_markdown(subset_hint, agg, flags, label_dir),
        encoding="utf-8",
    )

    print(f"\nStats written: {output_stats_path}")
    print(f"Markdown:      {md_path}")
    if flags:
        print(f"Quality flags ({len(flags)}):")
        for f in flags:
            print(f"  - {f}")
    else:
        print("Quality flags: none fired")

    return summary


def _is_subset_root(p: Path) -> bool:
    """Heuristic: directory with metadata.json + pseudo_labels/ looks like a subset."""
    return (p / "metadata.json").exists() and (p / "pseudo_labels").is_dir()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--data-dir", type=Path, default=None,
        help="Preprocessed subset root (contains motions/ and pseudo_labels/).",
    )
    parser.add_argument(
        "--label-dir", type=Path, default=None,
        help="Pseudo-label dir (default: <data-dir>/pseudo_labels).",
    )
    parser.add_argument(
        "--subset", type=str, default=None,
        help="Subset name used for quality flag specialisation "
             "(default: inferred from --data-dir).",
    )
    parser.add_argument(
        "--all-subsets", action="store_true",
        help="Scan every subset root under --piano-root that has a "
             "pseudo_labels/ dir.",
    )
    parser.add_argument(
        "--piano-root", type=Path, default=None,
        help="Parent dir for --all-subsets (e.g. /.../InterAct/piano).",
    )
    parser.add_argument("--num-patches", type=int, default=16)
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Optional aggregate output dir; when set, writes "
             "<output-dir>/<subset>_stats.json/.md in addition to the "
             "in-place stats.json/.md.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.all_subsets:
        if args.piano_root is None:
            raise SystemExit("--all-subsets requires --piano-root")
        root = args.piano_root
        subset_dirs = [p for p in sorted(root.iterdir()) if p.is_dir() and _is_subset_root(p)]
        if not subset_dirs:
            raise SystemExit(f"No subset roots (metadata.json + pseudo_labels/) under {root}")
        if args.output_dir:
            ensure_dir(args.output_dir)
        for d in subset_dirs:
            print(f"\n=== {d.name} ===")
            out = run_for_subset(
                d,
                subset_hint=d.name,
                num_patches=args.num_patches,
            )
            if args.output_dir:
                save_json(args.output_dir / f"{d.name}_stats.json", out)
        return

    if args.data_dir is None:
        raise SystemExit("Provide --data-dir (or --all-subsets with --piano-root)")
    subset_hint = args.subset or args.data_dir.name
    run_for_subset(
        args.data_dir,
        label_dir=args.label_dir,
        subset_hint=subset_hint,
        num_patches=args.num_patches,
    )


if __name__ == "__main__":
    main()
