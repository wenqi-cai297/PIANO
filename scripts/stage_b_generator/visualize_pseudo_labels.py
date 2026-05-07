"""Visualize v18 pseudo labels overlaid on skeleton + object animation.

Renders one MP4 per clip with all pseudo-label info baked in:
  - body skeleton (raw joints_world); body parts in contact at each
    frame are color-coded per-part (left_hand cyan, right_hand magenta,
    left_foot green, right_foot yellow, pelvis red).
  - object world point cloud (red).
  - contact target points: for parts in contact at frame t, the object-local
    contact_target_xyz_gt[t, p] is lifted to world via object_pose and
    rendered as a star at the predicted contact location, color-matched
    to the body part. So you can visually verify "where the body part
    should be contacting" matches "where the body part actually is".
  - frame caption with phase / support / per-part contact bitmask /
    text caption.

Usage:
    python scripts/stage_b_generator/visualize_pseudo_labels.py \\
        --config configs/training/anchordiff_v2_weighted.yaml \\
        --output runs/visualizations/pseudo_label_visual_check \\
        --clips chairs:0 chairs:Sub0001_Obj116_Seg0_0 \\
                imhd:0 neuraldome:0 omomo_correct_v2:0 \\
                imhd:bat omomo_correct_v2:suitcase
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from piano.data.dataset import HOIDataset


PART_NAMES = ("left_hand", "right_hand", "left_foot", "right_foot", "pelvis")
# matplotlib colors
PART_COLORS = ("#00CCCC", "#CC00CC", "#33CC33", "#CCCC00", "#FF3333")
# SMPL-22 joint indices for each body part (matches PART_TO_JOINT)
PART_TO_JOINT_LOCAL: tuple[int, ...] = (20, 21, 10, 11, 0)

PHASE_NAMES = ("non_contact", "stable_contact", "manipulation")
SUPPORT_NAMES = ("both_feet", "single_foot", "sitting")  # 3-way collapsed

SKELETON_CONNECTIONS: list[tuple[int, int]] = [
    (0, 1), (0, 2), (0, 3),
    (1, 4), (2, 5), (3, 6),
    (4, 7), (5, 8), (6, 9),
    (7, 10), (8, 11),
    (9, 12), (9, 13), (9, 14),
    (12, 15),
    (13, 16), (14, 17),
    (16, 18), (17, 19),
    (18, 20), (19, 21),
]


def _axis_angle_to_rotmat(aa: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(aa)
    if theta < 1e-8:
        return np.eye(3, dtype=np.float32)
    axis = aa / theta
    K = np.array(
        [[0, -axis[2], axis[1]],
         [axis[2], 0, -axis[0]],
         [-axis[1], axis[0], 0]],
        dtype=np.float32,
    )
    return (np.eye(3, dtype=np.float32) + np.sin(theta) * K
            + (1 - np.cos(theta)) * K @ K)


def _resolve_clip(ds: HOIDataset, spec: str) -> int:
    try:
        return int(spec)
    except ValueError:
        pass
    for i in range(len(ds)):
        sample = ds[i]
        if spec in str(sample["seq_id"]):
            return i
    raise ValueError(f"clip '{spec}' not found")


def render_clip(
    output_path: Path,
    seq_id: str,
    subset: str,
    text: str,
    joints: np.ndarray,                 # (T, 22, 3) world
    object_pc: np.ndarray,              # (N, 3) object-local
    object_positions: np.ndarray,       # (T, 3) world
    object_rotations: np.ndarray,       # (T, 3) axis-angle world
    contact_state: np.ndarray,          # (T, 5)
    contact_target_xyz: np.ndarray,     # (T, 5, 3) object-local
    phase: np.ndarray,                  # (T,)
    support: np.ndarray,                # (T,)
    fps: float = 20.0,
    dpi: int = 80,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    T = len(joints)

    # Pre-rotate object PC + lift contact targets to world per frame.
    obj_pc_world = np.empty((T, len(object_pc), 3), dtype=np.float32)
    target_world = np.empty((T, 5, 3), dtype=np.float32)
    for t in range(T):
        R = _axis_angle_to_rotmat(object_rotations[t])
        obj_pc_world[t] = object_pc @ R.T + object_positions[t]
        for p in range(5):
            target_world[t, p] = R @ contact_target_xyz[t, p] + object_positions[t]

    # Axis limits over the whole clip.
    all_pos = np.concatenate([
        joints.reshape(-1, 3),
        obj_pc_world.reshape(-1, 3),
        target_world.reshape(-1, 3),
    ], axis=0)
    center = all_pos.mean(axis=0)
    max_range = max((all_pos.max(axis=0) - all_pos.min(axis=0)).max() / 2 * 1.2, 0.5)

    fig = plt.figure(figsize=(8.5, 7.5))
    ax = fig.add_subplot(111, projection="3d")

    joint_scatter = ax.scatter([], [], [], c="#7777FF", s=18)   # default body joints (no contact)
    contact_joint_scatters = []
    for col in PART_COLORS:
        contact_joint_scatters.append(
            ax.scatter([], [], [], c=col, s=80, marker="o", edgecolors="black", linewidths=0.6)
        )
    bone_lines = [ax.plot([], [], [], c="gray", linewidth=1.2)[0] for _ in SKELETON_CONNECTIONS]
    obj_pc_scat = ax.scatter([], [], [], c="red", s=2, alpha=0.4)
    target_scats = [
        ax.scatter([], [], [], c=col, s=140, marker="*", edgecolors="black", linewidths=0.7, alpha=0.95)
        for col in PART_COLORS
    ]

    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[2] - max_range, center[2] + max_range)
    ax.set_zlim(center[1] - max_range, center[1] + max_range)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")
    ax.view_init(elev=18, azim=-65)

    title_artist = ax.set_title("")

    # Legend (one-time, in axes coords)
    handles = []
    for name, col in zip(PART_NAMES, PART_COLORS):
        handles.append(plt.Line2D([0], [0], marker="o", color=col, label=name,
                                  markeredgecolor="black", markersize=8, linestyle=""))
    handles.append(plt.Line2D([0], [0], marker="*", color="gray",
                              markeredgecolor="black", label="contact_target_xyz",
                              markersize=12, linestyle=""))
    ax.legend(handles=handles, loc="upper right", fontsize=7)

    def update(t: int) -> list:
        cs_t = contact_state[t]      # (5,)
        ph_t = int(phase[t])
        sp_t = int(support[t])

        # body joints split into contacted vs not
        non_contact_mask = np.ones(22, dtype=bool)
        for p_idx in range(5):
            joint_idx = PART_TO_JOINT_LOCAL[p_idx]
            non_contact_mask[joint_idx] = (cs_t[p_idx] < 0.5)
        nc = joints[t][non_contact_mask]
        joint_scatter._offsets3d = (nc[:, 0], nc[:, 2], nc[:, 1])

        for p_idx, scat in enumerate(contact_joint_scatters):
            joint_idx = PART_TO_JOINT_LOCAL[p_idx]
            jp = joints[t, joint_idx]
            if cs_t[p_idx] >= 0.5:
                scat._offsets3d = ([jp[0]], [jp[2]], [jp[1]])
            else:
                scat._offsets3d = ([], [], [])

        # bones
        for (i, j), line in zip(SKELETON_CONNECTIONS, bone_lines):
            line.set_data(
                [joints[t, i, 0], joints[t, j, 0]],
                [joints[t, i, 2], joints[t, j, 2]],
            )
            line.set_3d_properties([joints[t, i, 1], joints[t, j, 1]])

        # object PC
        op = obj_pc_world[t]
        obj_pc_scat._offsets3d = (op[:, 0], op[:, 2], op[:, 1])

        # contact target stars (only for parts in contact)
        for p_idx, scat in enumerate(target_scats):
            if cs_t[p_idx] >= 0.5:
                tw = target_world[t, p_idx]
                scat._offsets3d = ([tw[0]], [tw[2]], [tw[1]])
            else:
                scat._offsets3d = ([], [], [])

        contact_bits = "".join(["1" if cs_t[p] >= 0.5 else "0" for p in range(5)])
        contact_label = " ".join([
            f"{n[0].upper()}{n[-1].upper()}={'Y' if cs_t[i]>=0.5 else '.'}"
            for i, n in enumerate(["LH", "RH", "LF", "RF", "PE"])
        ])
        ph_name = PHASE_NAMES[ph_t] if 0 <= ph_t < len(PHASE_NAMES) else f"?{ph_t}"
        sp_name = SUPPORT_NAMES[sp_t] if 0 <= sp_t < len(SUPPORT_NAMES) else f"?{sp_t}"
        title_artist.set_text(
            f"{subset}/{seq_id}\n"
            f"{text[:80]}\n"
            f"frame {t+1}/{T}  phase={ph_name}  support={sp_name}\n"
            f"contact: {contact_label}"
        )

        artists = (
            [joint_scatter] + contact_joint_scatters + bone_lines
            + [obj_pc_scat] + target_scats + [title_artist]
        )
        return artists

    anim = FuncAnimation(fig, update, frames=T, interval=1000 / fps,
                         blit=False, repeat=False)
    suffix = output_path.suffix.lower()
    try:
        if suffix == ".mp4":
            anim.save(str(output_path), writer="ffmpeg", fps=fps, dpi=dpi)
        else:
            anim.save(str(output_path), writer="pillow", fps=fps, dpi=dpi)
    except Exception as e:
        plt.close(fig)
        raise RuntimeError(f"Animation save failed: {e}")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--clips", type=str, nargs="+", required=True)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--dpi", type=int, default=80)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    pseudo_label_subdir = cfg.data.get("pseudo_label_subdir", None)
    datasets = {}
    for entry in cfg.data.datasets:
        sub_dir = (str(Path(entry.root) / pseudo_label_subdir)
                   if pseudo_label_subdir is not None else None)
        ds = HOIDataset(
            root=entry.root,
            pseudo_label_dir=sub_dir,
            max_seq_length=cfg.data.max_seq_length,
            augment=None,
            support_collapse_hand_support=True,
            surface_obj_pose=True,
        )
        datasets[entry.name] = ds

    for spec in args.clips:
        if ":" not in spec:
            print(f"Skip malformed: {spec}")
            continue
        subset, clip_spec = spec.split(":", 1)
        if subset not in datasets:
            print(f"Skip unknown subset: {subset}")
            continue
        ds = datasets[subset]
        try:
            idx = _resolve_clip(ds, clip_spec)
        except ValueError as e:
            print(f"Skip: {e}")
            continue

        sample = ds[idx]
        seq_id = str(sample["seq_id"])
        text = str(sample["text"])
        seq_len = int(sample["seq_len"].item())
        joints = sample["joints"].numpy()[:seq_len]
        object_pc = sample["object_pc"].numpy()
        object_positions = sample["object_positions"].numpy()[:seq_len]
        object_rotations = sample["object_rotations"].numpy()[:seq_len]
        contact_state = sample["contact_state"].numpy()[:seq_len]
        ctxyz = sample["contact_target_xyz"].numpy()[:seq_len]
        phase = sample["phase"].numpy()[:seq_len]
        support = sample["support"].numpy()[:seq_len]

        out_path = out_dir / f"{subset}_{seq_id}_pseudo_labels.mp4"
        # Phase / support distribution preview
        ph_dist = np.bincount(phase.clip(0, 2), minlength=3) / max(seq_len, 1)
        sp_dist = np.bincount(support.clip(0, 2), minlength=3) / max(seq_len, 1)
        cs_rate = contact_state.mean(axis=0)
        print(f"[{subset}/{seq_id}]  T={seq_len}  text={text[:60]}")
        print(f"  phase dist: nc={ph_dist[0]:.2f} stable={ph_dist[1]:.2f} manip={ph_dist[2]:.2f}")
        print(f"  supp  dist: both={sp_dist[0]:.2f} single={sp_dist[1]:.2f} sit={sp_dist[2]:.2f}")
        print(f"  contact rate: LH={cs_rate[0]:.2f} RH={cs_rate[1]:.2f} "
              f"LF={cs_rate[2]:.2f} RF={cs_rate[3]:.2f} PE={cs_rate[4]:.2f}")

        render_clip(
            output_path=out_path,
            seq_id=seq_id,
            subset=subset,
            text=text,
            joints=joints,
            object_pc=object_pc,
            object_positions=object_positions,
            object_rotations=object_rotations,
            contact_state=contact_state,
            contact_target_xyz=ctxyz,
            phase=phase,
            support=support,
            fps=args.fps,
            dpi=args.dpi,
        )

    print(f"\nDone. Videos in {out_dir}/")


if __name__ == "__main__":
    main()
