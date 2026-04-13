"""Motion visualization utilities.

Converts generated HumanML3D 263-dim features back to SMPL joint positions
and provides simple skeleton plotting for quick qualitative evaluation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


# SMPL 22-joint skeleton connectivity for visualization
SKELETON_CONNECTIONS: list[tuple[int, int]] = [
    (0, 1), (0, 2), (0, 3),        # pelvis → left_hip, right_hip, spine1
    (1, 4), (2, 5), (3, 6),        # hips → knees, spine1 → spine2
    (4, 7), (5, 8), (6, 9),        # knees → ankles, spine2 → spine3
    (7, 10), (8, 11),              # ankles → feet
    (9, 12), (9, 13), (9, 14),    # spine3 → neck, collars
    (12, 15),                       # neck → head
    (13, 16), (14, 17),           # collars → shoulders
    (16, 18), (17, 19),           # shoulders → elbows
    (18, 20), (19, 21),           # elbows → wrists
]


def motion_263_to_joints(
    motion: np.ndarray,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> np.ndarray:
    """Extract approximate joint positions from HumanML3D 263-dim features.

    The 263-dim representation encodes root velocity, relative joint positions,
    velocities, rotations, and foot contact. We extract the relative joint
    positions (dims 4:67) and reconstruct absolute positions via root integration.

    Parameters
    ----------
    motion : (T, 263) — HumanML3D features (normalized or raw)
    mean, std : (263,) — normalization statistics for denormalization

    Returns
    -------
    joints : (T, 22, 3) — approximate absolute joint positions
    """
    from piano.data.humanml3d_repr import denormalize_motion

    if mean is not None and std is not None:
        motion = denormalize_motion(motion, mean, std)

    T = len(motion)

    # Root features
    root_ang_vel = motion[:, 0]      # (T,)
    root_vel_xz = motion[:, 1:3]    # (T, 2) — [vx, vz]
    root_height = motion[:, 3]      # (T,)

    # Relative joint positions (21 joints × 3 = 63 dims)
    rel_positions = motion[:, 4:67].reshape(T, 21, 3)

    # Integrate root trajectory
    root_pos = np.zeros((T, 3))
    root_pos[:, 1] = root_height  # y = height
    for t in range(1, T):
        root_pos[t, 0] = root_pos[t - 1, 0] + root_vel_xz[t, 0] / 30.0  # x
        root_pos[t, 2] = root_pos[t - 1, 2] + root_vel_xz[t, 1] / 30.0  # z

    # Absolute positions = root + relative
    joints = np.zeros((T, 22, 3))
    joints[:, 0, :] = root_pos
    joints[:, 1:, :] = rel_positions + root_pos[:, None, :]

    return joints


def save_animation_frames(
    joints: np.ndarray,
    output_dir: str | Path,
    fps: float = 30.0,
    elev: float = 15.0,
    azim: float = -60.0,
) -> None:
    """Save skeleton animation as PNG frames.

    Parameters
    ----------
    joints : (T, 22, 3) — joint positions
    output_dir : directory to save frames
    fps : frame rate (for filename labeling)
    elev, azim : camera angles for 3D plot
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    T = len(joints)

    # Compute axis limits from all frames
    all_pos = joints.reshape(-1, 3)
    center = all_pos.mean(axis=0)
    max_range = (all_pos.max(axis=0) - all_pos.min(axis=0)).max() / 2 * 1.2

    for t in range(T):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Plot joints
        ax.scatter(
            joints[t, :, 0], joints[t, :, 2], joints[t, :, 1],
            c="blue", s=20,
        )

        # Plot skeleton connections
        for i, j in SKELETON_CONNECTIONS:
            ax.plot(
                [joints[t, i, 0], joints[t, j, 0]],
                [joints[t, i, 2], joints[t, j, 2]],
                [joints[t, i, 1], joints[t, j, 1]],
                c="gray", linewidth=1.5,
            )

        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[2] - max_range, center[2] + max_range)
        ax.set_zlim(center[1] - max_range, center[1] + max_range)
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_zlabel("Y")
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"Frame {t} / {T} ({t/fps:.2f}s)")

        plt.tight_layout()
        plt.savefig(output_dir / f"frame_{t:04d}.png", dpi=100)
        plt.close(fig)

    print(f"Saved {T} frames to {output_dir}")
