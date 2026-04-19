"""Visualize pseudo-labels overlaid on the skeleton.

Renders an MP4 showing the raw skeleton (from joints_22) with the
pseudo-label signals overlaid as visual cues:
    - Tracked body parts (hands, feet, pelvis) are colored by contact_state:
      blue when not in contact, interpolating to red as soft-contact approaches 1
    - The object center is shown as a red triangle (same as visualize_motion)
    - Text overlay per-frame shows: phase name + support name

This lets us eyeball whether the pseudo-labels fire at sensible moments:
    * Does contact turn on when a hand reaches the object?
    * Does phase transition from approach → pre-contact → manipulation?
    * Does support flip from both_feet → sitting when the person sits?

Usage:
    piano-visualize-pseudo-labels \\
        --data-dir /path/to/piano/<subset> \\
        --pseudo-label-dir /path/to/piano/<subset>/pseudo_labels \\
        --seq-ids sub10_clothesstand_000 sub11_largebox_001 \\
        [--output-dir runs/visualizations/<timestamp>_pseudo_labels/]
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from piano.data.pseudo_labels.extract_phase import PHASE_NAMES
from piano.data.pseudo_labels.extract_support import SUPPORT_NAMES
from piano.inference.visualize_motion import SKELETON_CONNECTIONS
from piano.utils.io_utils import ensure_dir, load_json, save_json
from piano.utils.smpl_utils import BODY_PART_INDICES, BODY_PART_NAMES


def load_sample_with_labels(
    data_dir: Path,
    pseudo_label_dir: Path,
    seq_id: str,
    metadata_by_seq: dict[str, dict] | None = None,
) -> dict | None:
    """Load one sequence's joints/object + pseudo-labels.

    Also loads the per-object point cloud from ``<data_dir>/objects/<obj_id>.npy``
    (via the metadata's ``object_id``) so the object can be rendered as a
    point cloud, not just a single-point marker.

    Returns None if any required file is missing.
    """
    motion_path = data_dir / "motions" / f"{seq_id}.npz"
    labels_path = pseudo_label_dir / f"{seq_id}.npz"

    if not motion_path.exists():
        print(f"  [skip] motions file missing: {motion_path}")
        return None
    if not labels_path.exists():
        print(f"  [skip] pseudo-labels file missing: {labels_path}")
        return None

    motion_data = np.load(motion_path)
    labels = np.load(labels_path)

    # Load object point cloud if we can locate the object_id via metadata
    object_pc: np.ndarray | None = None
    object_rotations: np.ndarray | None = None
    if metadata_by_seq is not None and seq_id in metadata_by_seq:
        obj_id = metadata_by_seq[seq_id].get("object_id")
        if obj_id:
            obj_path = data_dir / "objects" / f"{obj_id}.npy"
            if obj_path.exists():
                object_pc = np.load(obj_path).astype(np.float32)  # (N, 3) object-local

    # object_rotations may be present if preprocessing saved it (new preprocess runs)
    if "object_rotations" in motion_data.files:
        object_rotations = motion_data["object_rotations"].astype(np.float32)

    return {
        "seq_id": seq_id,
        "joints_22": motion_data["joints_22"].astype(np.float32),
        "object_positions": motion_data.get("object_positions", None),
        "object_rotations": object_rotations,
        "object_pc": object_pc,
        "contact_state": labels["contact_state"].astype(np.float32),
        "phase": labels["phase"].astype(np.int64),
        "support": labels["support"].astype(np.int64),
    }


def _axis_angle_to_rotmat(aa: np.ndarray) -> np.ndarray:
    """Rodrigues: (3,) axis-angle → (3, 3) rotation matrix."""
    theta = np.linalg.norm(aa)
    if theta < 1e-8:
        return np.eye(3, dtype=np.float32)
    axis = aa / theta
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ], dtype=np.float32)
    return (np.eye(3, dtype=np.float32)
            + np.sin(theta) * K
            + (1 - np.cos(theta)) * K @ K)


def render_with_labels(
    joints: np.ndarray,
    contact_state: np.ndarray,
    phase: np.ndarray,
    support: np.ndarray,
    output_path: Path,
    object_positions: np.ndarray | None = None,
    object_rotations: np.ndarray | None = None,
    object_pc: np.ndarray | None = None,
    fps: float = 20.0,
    title: str = "",
    dpi: int = 80,
    elev: float = 15.0,
    azim: float = -60.0,
) -> None:
    """Render skeleton + contact coloring + phase/support text to MP4.

    Parameters
    ----------
    joints : (T, 22, 3) — absolute joint positions
    contact_state : (T, 5) — soft contact per body part
    phase : (T,) — integer phase label
    support : (T,) — integer support label
    object_positions : (T, 3) — per-frame object translation
    object_rotations : (T, 3) — per-frame object axis-angle rotation (optional)
    object_pc : (N, 3) — static object point cloud in object-local frame.
        When provided together with object_positions, the cloud is transformed
        (and optionally rotated) per frame and rendered as red dots,
        giving a real shape instead of a single marker.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    T = len(joints)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Pre-compute per-frame transformed object point cloud (if available)
    obj_cloud_world: np.ndarray | None = None
    if object_pc is not None and object_positions is not None:
        T_obj = len(object_positions)
        N = object_pc.shape[0]
        obj_cloud_world = np.empty((T_obj, N, 3), dtype=np.float32)
        for t in range(T_obj):
            if object_rotations is not None:
                R = _axis_angle_to_rotmat(object_rotations[t])
                obj_cloud_world[t] = object_pc @ R.T + object_positions[t]
            else:
                obj_cloud_world[t] = object_pc + object_positions[t]

    # Axis limits (include transformed object cloud)
    all_pos = [joints.reshape(-1, 3)]
    if obj_cloud_world is not None:
        all_pos.append(obj_cloud_world.reshape(-1, 3))
    elif object_positions is not None:
        all_pos.append(object_positions)
    all_pos = np.concatenate(all_pos, axis=0)
    center = all_pos.mean(axis=0)
    max_range = max((all_pos.max(axis=0) - all_pos.min(axis=0)).max() / 2 * 1.2, 0.5)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Non-tracked joints (all 22 minus the 5 tracked parts) in gray
    non_tracked_idx = [i for i in range(22) if i not in BODY_PART_INDICES]
    non_tracked_scatter = ax.scatter([], [], [], c="lightblue", s=15)

    # Tracked body parts: color will be updated per frame based on contact
    tracked_scatter = ax.scatter([], [], [], c="blue", s=80)

    # Skeleton lines
    lines = [ax.plot([], [], [], c="gray", linewidth=1.5)[0] for _ in SKELETON_CONNECTIONS]

    # Object marker: full point cloud if we have it, else single triangle
    object_scatter = None
    if obj_cloud_world is not None:
        object_scatter = ax.scatter([], [], [], c="red", s=2, marker="o", alpha=0.5)
    elif object_positions is not None:
        object_scatter = ax.scatter([], [], [], c="red", s=50, marker="^")

    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[2] - max_range, center[2] + max_range)
    ax.set_zlim(center[1] - max_range, center[1] + max_range)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")
    ax.view_init(elev=elev, azim=azim)
    title_artist = ax.set_title("")

    def contact_to_color(soft: float) -> tuple[float, float, float, float]:
        """Interpolate blue → red as soft contact goes 0 → 1."""
        s = float(max(0.0, min(1.0, soft)))
        return (s, 0.0, 1.0 - s, 1.0)

    def update(t: int) -> list:
        artists: list = []

        # Non-tracked joints
        non_tracked_scatter._offsets3d = (
            joints[t, non_tracked_idx, 0],
            joints[t, non_tracked_idx, 2],
            joints[t, non_tracked_idx, 1],
        )
        artists.append(non_tracked_scatter)

        # Tracked body parts with contact-color
        tracked_pts = joints[t, BODY_PART_INDICES, :]        # (5, 3)
        tracked_colors = [contact_to_color(contact_state[t, bp]) for bp in range(len(BODY_PART_INDICES))]
        tracked_scatter._offsets3d = (
            tracked_pts[:, 0],
            tracked_pts[:, 2],
            tracked_pts[:, 1],
        )
        tracked_scatter.set_color(tracked_colors)
        artists.append(tracked_scatter)

        # Skeleton connections
        for (i, j), line in zip(SKELETON_CONNECTIONS, lines):
            line.set_data(
                [joints[t, i, 0], joints[t, j, 0]],
                [joints[t, i, 2], joints[t, j, 2]],
            )
            line.set_3d_properties([joints[t, i, 1], joints[t, j, 1]])
            artists.append(line)

        # Object rendering: full transformed cloud if available, else single point
        if object_scatter is not None:
            if obj_cloud_world is not None:
                pts = obj_cloud_world[t]
                object_scatter._offsets3d = (pts[:, 0], pts[:, 2], pts[:, 1])
            elif object_positions is not None:
                p = object_positions[t]
                object_scatter._offsets3d = ([p[0]], [p[2]], [p[1]])
            artists.append(object_scatter)

        # Text: title + current labels
        phase_name = PHASE_NAMES[int(phase[t])] if 0 <= phase[t] < len(PHASE_NAMES) else "?"
        support_name = SUPPORT_NAMES[int(support[t])] if 0 <= support[t] < len(SUPPORT_NAMES) else "?"
        contacts_on = [
            name for name, bp in zip(BODY_PART_NAMES, range(5))
            if contact_state[t, bp] > 0.5
        ]
        contacts_label = ",".join(contacts_on) if contacts_on else "—"

        title_text = (
            f"{title}\n"
            f"Frame {t+1}/{T} ({t/fps:.2f}s)  |  "
            f"phase: {phase_name}  |  support: {support_name}\n"
            f"contact: {contacts_label}"
        )
        title_artist.set_text(title_text)
        artists.append(title_artist)
        return artists

    anim = FuncAnimation(
        fig, update, frames=T, interval=1000 / fps, blit=False, repeat=False,
    )

    suffix = output_path.suffix.lower()
    try:
        if suffix == ".mp4":
            anim.save(str(output_path), writer="ffmpeg", fps=fps, dpi=dpi)
        else:
            anim.save(str(output_path), writer="pillow", fps=fps, dpi=dpi)
    except Exception as e:
        gif_path = output_path.with_suffix(".gif")
        print(f"  [warn] {e}; falling back to {gif_path}")
        anim.save(str(gif_path), writer="pillow", fps=fps, dpi=dpi)
        output_path = gif_path
    finally:
        plt.close(fig)

    print(f"  Saved: {output_path}")


def run_visualization(
    samples: list[dict],
    output_dir: Path,
    fps: float = 20.0,
) -> None:
    output_dir = ensure_dir(output_dir)
    index: list[dict] = []

    for sample in samples:
        seq_id = sample["seq_id"]
        joints = sample["joints_22"]
        T = len(joints)

        # Truncate pseudo-labels + object arrays to motion length (should match)
        contact_state = sample["contact_state"][:T]
        phase = sample["phase"][:T]
        support = sample["support"][:T]
        obj_pos = sample["object_positions"]
        obj_rot = sample.get("object_rotations")
        obj_pc = sample.get("object_pc")
        if obj_pos is not None:
            obj_pos = obj_pos[:T]
        if obj_rot is not None:
            obj_rot = obj_rot[:T]

        out_path = output_dir / f"{seq_id}.mp4"
        print(f"\nRendering {seq_id} ({T} frames) → {out_path}")
        render_with_labels(
            joints=joints,
            contact_state=contact_state,
            phase=phase,
            support=support,
            object_positions=obj_pos,
            object_rotations=obj_rot,
            object_pc=obj_pc,
            output_path=out_path,
            fps=fps,
            title=seq_id,
        )

        # Collect per-frame label stats for the summary
        index.append({
            "seq_id": seq_id,
            "num_frames": T,
            "file": out_path.name,
            "contact_rates": {
                name: float((contact_state[:, bp] > 0.5).mean())
                for bp, name in enumerate(BODY_PART_NAMES)
            },
            "phase_distribution": {
                PHASE_NAMES[p]: int((phase == p).sum())
                for p in range(len(PHASE_NAMES))
            },
            "support_distribution": {
                SUPPORT_NAMES[s]: int((support == s).sum())
                for s in range(len(SUPPORT_NAMES))
            },
        })

    summary = {
        "timestamp": datetime.now().isoformat(),
        "num_videos": len(index),
        "fps": fps,
        "videos": index,
    }
    save_json(output_dir / "summary.json", summary)
    print(f"\nRendered {len(index)} videos. Summary: {output_dir / 'summary.json'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--data-dir", type=Path, required=True,
        help="PIANO data root (contains metadata.json, motions/, objects/)",
    )
    parser.add_argument(
        "--pseudo-label-dir", type=Path, default=None,
        help="Pseudo-label directory (default: <data-dir>/pseudo_labels)",
    )
    parser.add_argument(
        "--seq-ids", nargs="*", default=None,
        help="Specific sequence ids (default: first --num-samples)",
    )
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Default: runs/visualizations/<timestamp>_pseudo_labels/",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    pseudo_dir = args.pseudo_label_dir or (args.data_dir / "pseudo_labels")

    # Resolve seq ids + build metadata lookup for object_id resolution
    metadata = load_json(args.data_dir / "metadata.json")
    metadata_by_seq = {m["seq_id"]: m for m in metadata}
    if args.seq_ids:
        wanted = set(args.seq_ids)
        seq_ids = [m["seq_id"] for m in metadata if m["seq_id"] in wanted]
    else:
        seq_ids = [m["seq_id"] for m in metadata[: args.num_samples]]

    samples: list[dict] = []
    for sid in seq_ids:
        sample = load_sample_with_labels(args.data_dir, pseudo_dir, sid, metadata_by_seq)
        if sample is not None:
            samples.append(sample)

    if not samples:
        raise RuntimeError(f"No valid samples with pseudo-labels found at {pseudo_dir}")

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        output_dir = Path("runs/visualizations") / f"{timestamp}_pseudo_labels"
    else:
        output_dir = args.output_dir

    run_visualization(samples, output_dir, fps=args.fps)


if __name__ == "__main__":
    main()
