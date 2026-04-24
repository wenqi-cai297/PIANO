"""Distance-threshold sweep for ``extract_contact``.

Splits the expensive and the cheap parts of threshold tuning:

    Phase 1 (collect) — for every sequence in a PIANO data root, compute
    the per-frame per-body-part distance from the (inverse-transformed)
    joint to the object mesh surface. This is the only CPU-heavy step;
    it runs once and writes a single ``distances.npz`` per subset.

    Phase 2 (analyze) — load the cached distances, apply a grid of
    thresholds, and re-run the exact downstream filters
    (``median_filter`` + ``min_contact_duration``) per sequence. Reports
    ``frame_rate`` and ``seq_reached`` per body part per threshold as
    JSON + markdown.

The two phases are a single CLI so callers only need one command; pass
``--skip-collect`` to analyze-only on a cached ``distances.npz``.

Typical usage on the server::

    piano-threshold-sweep \
        --data-dir /media/.../InterAct/piano/chairs \
        --mesh-dir /media/.../InterAct/InterAct/chairs/objects \
        --output-dir runs/threshold_sweep/$(date +%Y-%m-%d_%H%M%S)_chairs

The shell wrapper ``scripts/stage1_pseudo_labels/threshold_sweep.sh`` runs this for
all 4 subsets back-to-back.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from tqdm import tqdm

from piano.data.pseudo_labels._object_transform import world_to_object_local
from piano.data.pseudo_labels.extract_contact import _filter_short_contacts
from piano.data.pseudo_labels.run_all import _find_mesh
from piano.utils.geometry import load_mesh, points_to_mesh_distance
from piano.utils.io_utils import ensure_dir, load_json, save_json
from piano.utils.smpl_utils import BODY_PART_INDICES, BODY_PART_NAMES


# Default sweep grid, in meters. Covers the anatomically-plausible
# joint-to-skin range for all 5 body parts with enough density near the
# expected ranges (hand ~0.05-0.10, foot ~0.08-0.15, pelvis ~0.15-0.25).
DEFAULT_THRESHOLD_GRID_M: tuple[float, ...] = (
    0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16,
    0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35, 0.40,
)


# ---------------------------------------------------------------------------
# Phase 1: collect raw joint-to-mesh distances
# ---------------------------------------------------------------------------

def collect_distances(
    data_dir: Path,
    mesh_dir: Path,
    output_npz: Path,
    mesh_suffixes: tuple[str, ...] = ("_face1000", "_simplified", ""),
) -> dict:
    """Compute per-frame joint-to-mesh distances for every sequence.

    Output ``.npz`` contains:
        distances :     (total_frames, 5) float32, in object-local-frame units
        frame_offsets : (N+1,) int64, prefix-sum indexing into `distances`
        seq_ids :       (N,) str
        object_ids :    (N,) str
    """
    data_dir = Path(data_dir)
    mesh_dir = Path(mesh_dir)
    ensure_dir(output_npz.parent)

    metadata = load_json(data_dir / "metadata.json")

    import trimesh
    mesh_cache: dict[str, trimesh.Trimesh | None] = {}

    seq_distances: list[np.ndarray] = []
    seq_ids: list[str] = []
    obj_ids: list[str] = []
    frame_counts: list[int] = []
    n_skip = 0

    for entry in tqdm(metadata, desc=f"collect({data_dir.name})"):
        seq_id = entry["seq_id"]
        obj_id = entry["object_id"]

        motion_path = data_dir / "motions" / f"{seq_id}.npz"
        if not motion_path.exists():
            n_skip += 1
            continue

        motion = np.load(motion_path, allow_pickle=False)
        joints = motion["joints_22"]
        obj_pos = motion["object_positions"] if "object_positions" in motion.files else None
        obj_rot = motion["object_rotations"] if "object_rotations" in motion.files else None
        if obj_pos is None or obj_rot is None:
            n_skip += 1
            continue

        if obj_id not in mesh_cache:
            mesh_path = _find_mesh(mesh_dir, obj_id, mesh_suffixes)
            if mesh_path is None:
                mesh_cache[obj_id] = None
            else:
                try:
                    mesh_cache[obj_id] = load_mesh(str(mesh_path))
                except Exception as e:
                    print(f"  [warn] mesh load failed {mesh_path}: {e}")
                    mesh_cache[obj_id] = None
        mesh = mesh_cache[obj_id]
        if mesh is None:
            n_skip += 1
            continue

        T = len(joints)
        dist_per_bp = np.empty((T, len(BODY_PART_INDICES)), dtype=np.float32)
        for bp_idx, joint_idx in enumerate(BODY_PART_INDICES):
            bp_world = joints[:, joint_idx, :]          # (T, 3)
            bp_local = world_to_object_local(bp_world, obj_pos, obj_rot)
            d, _ = points_to_mesh_distance(bp_local, mesh)
            dist_per_bp[:, bp_idx] = d.astype(np.float32)

        seq_distances.append(dist_per_bp)
        seq_ids.append(seq_id)
        obj_ids.append(obj_id)
        frame_counts.append(T)

    if not seq_distances:
        raise RuntimeError(f"No sequences produced distances under {data_dir}")

    distances = np.concatenate(seq_distances, axis=0).astype(np.float32)
    offsets = np.concatenate([[0], np.cumsum(frame_counts)]).astype(np.int64)

    np.savez_compressed(
        output_npz,
        distances=distances,
        frame_offsets=offsets,
        seq_ids=np.array(seq_ids, dtype=object),
        object_ids=np.array(obj_ids, dtype=object),
    )

    return {
        "num_sequences": len(seq_ids),
        "num_skipped": n_skip,
        "total_frames": int(distances.shape[0]),
        "output": str(output_npz),
        "unique_objects": len(set(obj_ids)),
    }


# ---------------------------------------------------------------------------
# Phase 2: threshold sweep
# ---------------------------------------------------------------------------

def _apply_pipeline(
    distances_slice: np.ndarray,
    threshold: float,
    median_filter_size: int = 5,
    min_contact_duration: int = 3,
) -> np.ndarray:
    """Replicate the soft-sigmoid + median + min-duration filter chain for
    one sequence, returning a binary (T, B) contact mask.

    Using the same sigma-width assumption as ``extract_contact`` (sigma =
    0.03 m). The downstream binarization threshold (0.5) is preserved, so
    the returned mask matches what extract_contact would write for the
    same threshold.
    """
    from scipy.special import expit

    SIGMA = 0.03
    score = expit(-(distances_slice - threshold) / SIGMA)   # (T, B)
    for bp in range(score.shape[1]):
        score[:, bp] = median_filter(score[:, bp], size=median_filter_size)
    score = _filter_short_contacts(score, min_contact_duration)
    return score > 0.5


def analyze(
    distances_npz: Path,
    threshold_grid: tuple[float, ...] = DEFAULT_THRESHOLD_GRID_M,
    output_json: Path | None = None,
) -> dict:
    data = np.load(distances_npz, allow_pickle=True)
    distances = data["distances"]             # (total_T, B)
    offsets = data["frame_offsets"]           # (N+1,)
    seq_ids = [str(s) for s in data["seq_ids"]]
    N = len(seq_ids)
    total_T = distances.shape[0]

    # Raw (no threshold) distance distribution per body part
    raw_dist = {
        BODY_PART_NAMES[bp]: {
            "min": float(distances[:, bp].min()),
            "p10": float(np.percentile(distances[:, bp], 10)),
            "p25": float(np.percentile(distances[:, bp], 25)),
            "p50": float(np.percentile(distances[:, bp], 50)),
            "p75": float(np.percentile(distances[:, bp], 75)),
            "p90": float(np.percentile(distances[:, bp], 90)),
            "max": float(distances[:, bp].max()),
        }
        for bp in range(distances.shape[1])
    }

    # Sweep
    sweep: list[dict] = []
    for thr in threshold_grid:
        # Per-body-part aggregators
        B = distances.shape[1]
        frame_contact_counts = np.zeros(B, dtype=np.int64)
        seq_reached_counts = np.zeros(B, dtype=np.int64)

        for i in range(N):
            s, e = offsets[i], offsets[i + 1]
            seq_slice = distances[s:e]
            mask = _apply_pipeline(seq_slice, thr)      # (T, B) binary
            frame_contact_counts += mask.sum(axis=0).astype(np.int64)
            seq_reached_counts += mask.any(axis=0).astype(np.int64)

        per_body_part = {
            BODY_PART_NAMES[bp]: {
                "frame_rate": float(frame_contact_counts[bp] / max(total_T, 1)),
                "seq_reached": float(seq_reached_counts[bp] / max(N, 1)),
            }
            for bp in range(B)
        }
        sweep.append({"threshold_m": float(thr), "per_body_part": per_body_part})

    result = {
        "num_sequences": N,
        "total_frames": int(total_T),
        "raw_distance_distribution": raw_dist,
        "sigma_m": 0.03,
        "median_filter_size": 5,
        "min_contact_duration": 3,
        "sweep": sweep,
    }

    if output_json is not None:
        save_json(output_json, result)
    return result


def format_markdown(result: dict, subset: str | None = None) -> str:
    """Render the sweep result as a human-readable markdown report."""
    lines: list[str] = []
    title = subset or "subset"
    lines.append(f"# Threshold sweep — {title}")
    lines.append("")
    lines.append(f"- Sequences: {result['num_sequences']}")
    lines.append(f"- Frames: {result['total_frames']}")
    lines.append(f"- Fixed parameters: sigma = {result['sigma_m']} m, "
                 f"median_filter_size = {result['median_filter_size']}, "
                 f"min_contact_duration = {result['min_contact_duration']}")
    lines.append("")

    # Raw distance distribution
    lines.append("## Raw joint-to-mesh distance distribution (meters)")
    lines.append("")
    lines.append("| body part | min | p10 | p25 | p50 | p75 | p90 | max |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for name, stats in result["raw_distance_distribution"].items():
        lines.append(
            f"| {name} | {stats['min']:.3f} | {stats['p10']:.3f} | "
            f"{stats['p25']:.3f} | {stats['p50']:.3f} | {stats['p75']:.3f} | "
            f"{stats['p90']:.3f} | {stats['max']:.3f} |"
        )
    lines.append("")

    # Sweep table, one per body part (easier to read than a cross-product)
    for bp in result["sweep"][0]["per_body_part"]:
        lines.append(f"## {bp}")
        lines.append("")
        lines.append("| threshold (m) | frame_rate | seq_reached |")
        lines.append("|---|---|---|")
        for entry in result["sweep"]:
            stats = entry["per_body_part"][bp]
            lines.append(
                f"| {entry['threshold_m']:.3f} | "
                f"{stats['frame_rate'] * 100:.1f}% | "
                f"{stats['seq_reached'] * 100:.1f}% |"
            )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="PIANO subset root (contains motions/, metadata.json).")
    parser.add_argument("--mesh-dir", type=Path, default=None,
                        help="Object mesh dir (required for collect; ignored for --skip-collect).")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Where to write distances.npz and analysis.{json,md}")
    parser.add_argument("--mesh-suffixes", nargs="+",
                        default=["_face1000", "_simplified", ""])
    parser.add_argument("--skip-collect", action="store_true",
                        help="Reuse existing distances.npz from --output-dir.")
    parser.add_argument("--threshold-grid", type=float, nargs="+",
                        default=list(DEFAULT_THRESHOLD_GRID_M),
                        help="Thresholds (m) to evaluate.")
    parser.add_argument("--subset", type=str, default=None,
                        help="Subset label for the markdown header (default: inferred).")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = ensure_dir(args.output_dir)
    subset = args.subset or args.data_dir.name
    distances_path = output_dir / "distances.npz"

    if not args.skip_collect:
        if args.mesh_dir is None:
            raise SystemExit("--mesh-dir is required unless --skip-collect is set")
        meta = collect_distances(
            data_dir=args.data_dir,
            mesh_dir=args.mesh_dir,
            output_npz=distances_path,
            mesh_suffixes=tuple(args.mesh_suffixes),
        )
        save_json(output_dir / "collect_meta.json", meta)
        print(f"Collected distances for {meta['num_sequences']} seqs "
              f"({meta['num_skipped']} skipped) → {distances_path}")
    else:
        if not distances_path.exists():
            raise SystemExit(f"--skip-collect set but {distances_path} missing")

    result = analyze(
        distances_npz=distances_path,
        threshold_grid=tuple(args.threshold_grid),
        output_json=output_dir / "analysis.json",
    )
    md = format_markdown(result, subset=subset)
    md_path = output_dir / "analysis.md"
    md_path.write_text(md, encoding="utf-8")

    print(f"\nWrote {output_dir / 'analysis.json'}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
