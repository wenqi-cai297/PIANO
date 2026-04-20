"""Action-segment contact-rate analysis.

Each InterAct ``text.txt`` stores a caption followed by POS tags and a
start/end time window that tells us *when* in the sequence that caption
actually happens. The plain threshold sweep only answers "what fraction
of frames get labelled contact"; this tool answers the sharper
question: **does the contact label fire inside the described action
window, and stay quiet outside it?**

Per (threshold, body_part) we report:

- ``inside_frame_rate``  : contact rate averaged over frames in [start, end).
- ``outside_frame_rate`` : contact rate averaged over frames outside [start, end).
- ``delta_pp``           : inside − outside (in percentage points).

For a well-calibrated threshold on a well-labelled dataset, ``inside``
should be noticeably higher than ``outside`` for the body part the
action involves (pelvis for chairs, hands for imhd / omomo). ``delta_pp``
is easier to reason about than a ratio because the denominator can be
near zero.

The tool reuses the cached ``distances.npz`` written by
``piano-threshold-sweep``, so it does not re-query meshes — it only
adds text.txt parsing and per-sequence inside/outside aggregation.

The text.txt parse assumes the HumanML3D convention
``natural#postag#start_sec#end_sec`` on the first line. If the assumption
fails for any sequence, it is recorded in ``num_unparseable`` and
excluded from the analysis. The tool also prints a small sample of
successfully-parsed captions + windows so callers can sanity-check
the assumption before trusting the numbers.

Usage::

    piano-action-segment-sweep \\
        --distances-npz runs/threshold_sweep/<ts>/chairs/distances.npz \\
        --interact-dir /media/.../InterAct/chairs \\
        --output-dir  runs/threshold_sweep/<ts>/chairs \\
        --subset chairs
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from scipy.special import expit

from piano.data.pseudo_labels.extract_contact import _filter_short_contacts
from piano.utils.io_utils import ensure_dir, save_json
from piano.utils.smpl_utils import BODY_PART_NAMES


# Threshold grid used for the sweep. Denser near the interesting range
# (0.04-0.30 m) where contact labels actually move. Outer tails of the
# plain sweep were flat so we drop them here to keep the table short.
DEFAULT_THRESHOLD_GRID_M: tuple[float, ...] = (
    0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18,
    0.20, 0.22, 0.25, 0.28, 0.30,
)


def _parse_text_file(path: Path) -> tuple[str, float, float] | None:
    """Parse HumanML3D-style ``text#postag#start#end`` on the first line.

    Returns ``(caption, start_sec, end_sec)`` or ``None`` if the file is
    missing, empty, or not in the expected format.
    """
    if not path.exists():
        return None
    raw = path.read_text(encoding="utf-8", errors="replace").strip()
    if not raw:
        return None
    first_line = raw.splitlines()[0]
    parts = first_line.split("#")
    if len(parts) < 4:
        return None
    try:
        start = float(parts[2])
        end = float(parts[3])
    except ValueError:
        return None
    if end <= start:
        return None
    return parts[0].strip(), start, end


def _apply_pipeline(
    distances_slice: np.ndarray,
    threshold: float,
    sigma: float = 0.03,
    median_size: int = 5,
    min_duration: int = 3,
) -> np.ndarray:
    """Replicate the extract_contact filtering chain, returning binary mask."""
    score = expit(-(distances_slice - threshold) / sigma)
    for bp in range(score.shape[1]):
        score[:, bp] = median_filter(score[:, bp], size=median_size)
    score = _filter_short_contacts(score, min_duration)
    return score > 0.5


def _collect_segments(
    interact_subset_dir: Path,
    seq_ids: list[str],
    fps: float,
) -> tuple[list[tuple[int, int] | None], list[dict]]:
    """Parse text.txt for every seq. Returns (segments, samples) where
    segments[i] is (start_frame, end_frame) at target fps or None if the
    text.txt is missing/unparseable. ``samples`` is a few examples of
    successfully-parsed files (for sanity-checking the format)."""
    segments: list[tuple[int, int] | None] = []
    samples: list[dict] = []
    n_sampled = 0
    for sid in seq_ids:
        text_path = interact_subset_dir / "sequences_canonical" / sid / "text.txt"
        parsed = _parse_text_file(text_path)
        if parsed is None:
            segments.append(None)
            continue
        caption, start_sec, end_sec = parsed
        start_frame = int(round(start_sec * fps))
        end_frame = int(round(end_sec * fps))
        segments.append((start_frame, end_frame))
        if n_sampled < 5:
            samples.append({
                "seq_id": sid,
                "caption": caption,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "start_frame": start_frame,
                "end_frame": end_frame,
            })
            n_sampled += 1
    return segments, samples


def analyze(
    distances_npz: Path,
    interact_subset_dir: Path,
    threshold_grid: tuple[float, ...] = DEFAULT_THRESHOLD_GRID_M,
    fps: float = 20.0,
) -> dict:
    data = np.load(distances_npz, allow_pickle=True)
    distances = data["distances"]
    offsets = data["frame_offsets"]
    seq_ids = [str(s) for s in data["seq_ids"]]
    N = len(seq_ids)
    B = distances.shape[1]

    segments, samples = _collect_segments(interact_subset_dir, seq_ids, fps=fps)

    # action window ratio distribution (sanity on whether start/end cover
    # only a fragment or essentially the whole clip)
    ratios: list[float] = []
    clamped_too_short = 0
    for i, seg in enumerate(segments):
        if seg is None:
            continue
        T = int(offsets[i + 1] - offsets[i])
        seg_start = max(0, min(seg[0], T))
        seg_end = max(0, min(seg[1], T))
        if seg_end <= seg_start:
            clamped_too_short += 1
            continue
        ratios.append((seg_end - seg_start) / max(T, 1))
    ratios_np = np.asarray(ratios, dtype=np.float64)

    # Per (threshold, body_part) inside / outside contact rates
    sweep: list[dict] = []
    for thr in threshold_grid:
        inside_contact = np.zeros(B, dtype=np.int64)
        inside_total = np.zeros(B, dtype=np.int64)
        outside_contact = np.zeros(B, dtype=np.int64)
        outside_total = np.zeros(B, dtype=np.int64)
        seq_contact_inside = np.zeros(B, dtype=np.int64)
        seq_with_segment = 0

        for i, seg in enumerate(segments):
            if seg is None:
                continue
            s, e = int(offsets[i]), int(offsets[i + 1])
            seq_slice = distances[s:e]
            T = seq_slice.shape[0]
            seg_start = max(0, min(seg[0], T))
            seg_end = max(0, min(seg[1], T))
            if seg_end <= seg_start:
                continue

            mask = _apply_pipeline(seq_slice, thr)
            in_mask = np.zeros(T, dtype=bool)
            in_mask[seg_start:seg_end] = True
            out_mask = ~in_mask

            in_n = int(in_mask.sum())
            out_n = int(out_mask.sum())

            contact_in = mask[in_mask].sum(axis=0).astype(np.int64)
            contact_out = mask[out_mask].sum(axis=0).astype(np.int64)
            inside_contact += contact_in
            outside_contact += contact_out
            inside_total += in_n
            outside_total += out_n
            seq_contact_inside += (contact_in > 0).astype(np.int64)
            seq_with_segment += 1

        per_bp = {}
        for bp in range(B):
            in_rate = (
                float(inside_contact[bp] / inside_total[bp])
                if inside_total[bp] > 0 else 0.0
            )
            out_rate = (
                float(outside_contact[bp] / outside_total[bp])
                if outside_total[bp] > 0 else 0.0
            )
            per_bp[BODY_PART_NAMES[bp]] = {
                "inside_frame_rate": in_rate,
                "outside_frame_rate": out_rate,
                "delta_pp": (in_rate - out_rate) * 100.0,
                "seq_inside_reached_fraction": float(
                    seq_contact_inside[bp] / max(seq_with_segment, 1)
                ),
            }
        sweep.append({"threshold_m": float(thr), "per_body_part": per_bp})

    return {
        "num_sequences": N,
        "num_with_parseable_segment": int(sum(1 for s in segments if s is not None)),
        "num_unparseable": int(sum(1 for s in segments if s is None)),
        "num_segment_clamped_to_empty": clamped_too_short,
        "parse_samples": samples,
        "action_window_ratio": {
            "median": float(np.median(ratios_np)) if ratios_np.size else None,
            "p10": float(np.percentile(ratios_np, 10)) if ratios_np.size else None,
            "p25": float(np.percentile(ratios_np, 25)) if ratios_np.size else None,
            "p75": float(np.percentile(ratios_np, 75)) if ratios_np.size else None,
            "p90": float(np.percentile(ratios_np, 90)) if ratios_np.size else None,
        },
        "fps": fps,
        "sigma_m": 0.03,
        "median_filter_size": 5,
        "min_contact_duration": 3,
        "sweep": sweep,
    }


def format_markdown(result: dict, subset: str | None = None) -> str:
    lines: list[str] = []
    title = subset or "subset"
    lines.append(f"# Action-segment contact sweep — {title}")
    lines.append("")
    N = result["num_sequences"]
    ok = result["num_with_parseable_segment"]
    lines.append(f"- Sequences: {N}")
    lines.append(
        f"- Parseable text.txt (action segment): {ok} / {N} "
        f"({ok / max(N, 1) * 100:.1f}%)"
    )
    lines.append(f"- Unparseable: {result['num_unparseable']}")
    lines.append(f"- Segment empty after clamping: {result['num_segment_clamped_to_empty']}")

    awr = result["action_window_ratio"]
    if awr["median"] is not None:
        lines.append(
            f"- Action window ratio (window_len / total_len): "
            f"median={awr['median'] * 100:.1f}%, "
            f"p25={awr['p25'] * 100:.1f}%, p75={awr['p75'] * 100:.1f}% "
            f"(p10={awr['p10'] * 100:.1f}%, p90={awr['p90'] * 100:.1f}%)"
        )
    lines.append(
        f"- Fixed: sigma={result['sigma_m']} m, "
        f"median_filter={result['median_filter_size']}, "
        f"min_contact_duration={result['min_contact_duration']}, "
        f"fps={result['fps']}"
    )
    lines.append("")

    # Sanity samples of parsed text
    if result.get("parse_samples"):
        lines.append("## Parse samples (first 5 valid text.txt)")
        lines.append("")
        lines.append("| seq_id | caption | start_sec | end_sec | start_frame | end_frame |")
        lines.append("|---|---|---|---|---|---|")
        for s in result["parse_samples"]:
            caption = s["caption"].replace("|", "\\|")
            if len(caption) > 60:
                caption = caption[:57] + "..."
            lines.append(
                f"| {s['seq_id']} | {caption} | {s['start_sec']:.2f} | "
                f"{s['end_sec']:.2f} | {s['start_frame']} | {s['end_frame']} |"
            )
        lines.append("")

    # Sweep tables per body part
    for bp_name in BODY_PART_NAMES:
        lines.append(f"## {bp_name}")
        lines.append("")
        lines.append("| threshold (m) | inside_action | outside_action | delta (pp) | seq_with_inside_contact |")
        lines.append("|---|---|---|---|---|")
        for entry in result["sweep"]:
            s = entry["per_body_part"][bp_name]
            lines.append(
                f"| {entry['threshold_m']:.3f} | "
                f"{s['inside_frame_rate'] * 100:.1f}% | "
                f"{s['outside_frame_rate'] * 100:.1f}% | "
                f"{s['delta_pp']:+.1f} | "
                f"{s['seq_inside_reached_fraction'] * 100:.1f}% |"
            )
        lines.append("")

    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--distances-npz", type=Path, required=True,
        help="Cached distances from piano-threshold-sweep (phase 1 output).",
    )
    p.add_argument(
        "--interact-dir", type=Path, required=True,
        help="InterAct subset root (contains sequences_canonical/<seq_id>/text.txt).",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--subset", type=str, default=None,
                   help="Subset label for the markdown header.")
    p.add_argument("--threshold-grid", type=float, nargs="+",
                   default=list(DEFAULT_THRESHOLD_GRID_M))
    p.add_argument("--fps", type=float, default=20.0,
                   help="Target fps (match preprocess target_fps; default 20).")
    return p


def main() -> None:
    args = build_parser().parse_args()
    output_dir = ensure_dir(args.output_dir)
    subset = args.subset or args.distances_npz.parent.name

    result = analyze(
        distances_npz=args.distances_npz,
        interact_subset_dir=args.interact_dir,
        threshold_grid=tuple(args.threshold_grid),
        fps=args.fps,
    )

    json_path = output_dir / "action_segment_analysis.json"
    md_path = output_dir / "action_segment_analysis.md"
    save_json(json_path, result)
    md_path.write_text(format_markdown(result, subset=subset), encoding="utf-8")

    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")
    print(
        f"\nParsed segments: {result['num_with_parseable_segment']} / "
        f"{result['num_sequences']}"
    )
    awr = result["action_window_ratio"]
    if awr["median"] is not None:
        print(
            f"Action window median: {awr['median'] * 100:.1f}% of total duration"
        )


if __name__ == "__main__":
    main()
