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

    num_patches: int = 16            # number of surface patches (K)
    num_surface_samples: int = 4096  # points sampled for clustering
    soft_sigma: float = 0.05         # Gaussian bandwidth in meters
    contact_threshold: float = 0.5   # minimum contact score to assign target


def extract_contact_target(
    joints: np.ndarray,
    object_mesh: "trimesh.Trimesh",
    contact_state: np.ndarray,
    object_positions: np.ndarray | None = None,
    object_rotations: np.ndarray | None = None,
    config: TargetConfig | None = None,
    patch_centers: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract per-frame contact target region on the object surface.

    Patch centers are in the object-local frame. For each contact frame we
    inverse-transform the body-part world position into the object-local
    frame before assigning to the nearest patch — exactly the same
    correction as ``extract_contact_state``.

    Parameters
    ----------
    joints : (T, 22, 3) — world-frame SMPL 22-joint positions
    object_mesh : trimesh.Trimesh — object mesh in object-local frame
    contact_state : (T, 5) — soft contact state from ``extract_contact``
    object_positions : (T, 3) — per-frame object translation in world frame
    object_rotations : (T, 3) — per-frame object axis-angle rotation
    config : extraction parameters
    patch_centers : (K, 3) or None — precomputed per-object patch atlas in
        object-local frame. Pass a cached atlas to keep patch ids stable
        across sequences of the same object (required for downstream
        classification). If None, patches are recomputed from the mesh with
        a non-deterministic FPS start — only use for one-off debugging.

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

    if patch_centers is None:
        # Fallback: recompute. Non-deterministic without a seed, so this
        # path should only be used for single-sequence debugging.
        patch_centers = cluster_surface_patches(
            object_mesh,
            num_patches=K,
            num_surface_samples=config.num_surface_samples,
        )  # (K, 3)
    else:
        if patch_centers.shape != (K, 3):
            raise ValueError(
                f"patch_centers shape {patch_centers.shape} does not match "
                f"config.num_patches={K}; re-run patch atlas precomputation."
            )

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
