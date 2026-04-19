"""Extract contact target pseudo-labels from HOI motion data.

For each frame where a body part is in contact, identifies *which region*
of the object surface is being contacted.  The object surface is divided
into K patches via farthest point sampling, and each contact is assigned
a soft distribution over these patches.

Output: soft target array of shape ``(T, B, K)`` where B=5 body parts
and K=num_patches.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from piano.utils.geometry import (
    build_kdtree,
    cluster_surface_patches,
    query_nearest,
    soft_patch_assignment,
)
from piano.utils.smpl_utils import BODY_PART_INDICES, NUM_BODY_PARTS


@dataclass(slots=True)
class TargetConfig:
    """Configuration for contact target extraction."""

    num_patches: int = 16           # number of surface patches (K)
    num_surface_samples: int = 4096  # points sampled for clustering
    soft_sigma: float = 0.01        # temperature for soft assignment
    contact_threshold: float = 0.5   # minimum contact score to assign target


def extract_contact_target(
    joints: np.ndarray,
    object_mesh: "trimesh.Trimesh",
    contact_state: np.ndarray,
    object_positions: np.ndarray | None = None,
    object_rotations: np.ndarray | None = None,
    config: TargetConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract per-frame contact target region on the object surface.

    Patch centers are computed once in the object-local frame. For each
    contact frame we inverse-transform the body-part world position into
    the object-local frame before assigning to the nearest patch — exactly
    the same correction as ``extract_contact_state``.

    Parameters
    ----------
    joints : (T, 22, 3) — world-frame SMPL 22-joint positions
    object_mesh : trimesh.Trimesh — object mesh in object-local frame
    contact_state : (T, 5) — soft contact state from ``extract_contact``
    object_positions : (T, 3) — per-frame object translation in world frame
    object_rotations : (T, 3) — per-frame object axis-angle rotation
    config : extraction parameters

    Returns
    -------
    target : (T, 5, K) — soft assignment over K patches per body part
    patch_centers : (K, 3) — patch center positions in object-local frame
    """
    from piano.data.pseudo_labels._object_transform import world_to_object_local

    if config is None:
        config = TargetConfig()

    T = len(joints)
    K = config.num_patches
    target = np.zeros((T, NUM_BODY_PARTS, K), dtype=np.float32)

    # Compute patch centers via FPS on object surface (object-local frame)
    patch_centers = cluster_surface_patches(
        object_mesh,
        num_patches=K,
        num_surface_samples=config.num_surface_samples,
    )  # (K, 3)

    for bp_idx, joint_idx in enumerate(BODY_PART_INDICES):
        bp_positions_world = joints[:, joint_idx, :]  # (T, 3)

        # Inverse-transform each frame's joint to object-local frame
        if object_positions is not None:
            bp_positions_local = world_to_object_local(
                bp_positions_world, object_positions, object_rotations,
            )
        else:
            bp_positions_local = bp_positions_world

        for t in range(T):
            if contact_state[t, bp_idx] < config.contact_threshold:
                continue
            target[t, bp_idx] = soft_patch_assignment(
                bp_positions_local[t],
                patch_centers,
                sigma=config.soft_sigma,
            )

    return target, patch_centers
