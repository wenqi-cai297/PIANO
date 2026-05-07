"""Extract contact target pseudo-labels from HOI motion data.

For each frame where a body part is in contact, identifies *where on
the object surface* the body part touches. Two outputs:

- ``contact_target_xyz_gt (T, B, 3)`` — closest point on the mesh
  surface in object-local coordinates per body part per frame. This
  is the ground truth for the predictor's xyz regression head
  (HOI-Diff / CG-HOI / ContactGen convention). Computed via
  ``trimesh.proximity.closest_point`` in a batched mesh query. New as of
  the v10 pseudo-label pass.
- ``contact_target (T, B, K)`` — legacy soft distribution over K FPS
  patches. Kept for backward compatibility, downstream visualisation,
  and entropy diagnostics. Not used by the predictor any more.

The xyz GT replaces the previous "softmax-weighted patch centroid"
approximation that HOIDataset computed at load time. That approximation
introduced an estimated 5-10 cm bias against the true closest-surface-
point — directly visible in the v2 Stage A train-target plateau at 18
cm vs the model's actual capacity. Re-extracting with this exact GT
removes that floor; train target should drop to ≤5 cm and val target
should drop with it (Kendall et al. CVPR'18-style multi-task weighting
+ this xyz fix together attack the two main v2 failure modes).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from piano.utils.geometry import (
    cluster_surface_patches,
    points_to_mesh_distance,
)
from piano.utils.smpl_utils import BODY_PART_INDICES, NUM_BODY_PARTS


@dataclass(slots=True)
class TargetConfig:
    """Configuration for contact target extraction."""

    num_patches: int = 16            # number of surface patches (K)
    num_surface_samples: int = 4096  # points sampled for clustering
    # Gaussian bandwidth in meters. v2 extraction used sigma=0.05 and
    # produced entropy_mean=0.26 / 2.77max on chairs (60% of sequences
    # flagged as near-hard target). Raised to 0.12 so the exp(-d²/2σ²)
    # kernel is actually "soft" at typical patch spacings (InterAct mesh
    # BB diag ~0.5-1.0 m, K=16 patches → neighbour spacing ~0.15-0.3 m).
    soft_sigma: float = 0.12
    contact_threshold: float = 0.5   # minimum contact score to assign target
    # Preserve historical dense labels by default. For current gated
    # Stage A/B losses, pseudo-label extraction can set this False and
    # skip closest-point queries on non-contact frame/body-part cells.
    query_all_frames: bool = True


def extract_contact_target(
    joints: np.ndarray,
    object_mesh: "trimesh.Trimesh",
    contact_state: np.ndarray,
    object_positions: np.ndarray | None = None,
    object_rotations: np.ndarray | None = None,
    config: TargetConfig | None = None,
    patch_centers: np.ndarray | None = None,
    anchor_points_world: np.ndarray | None = None,
    target_points_local_override: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract per-frame contact target region on the object surface.

    Returns three arrays in the object-local frame:

    1. ``target_xyz_gt (T, 5, 3)`` — closest point on the mesh surface
       per body part per frame. Defined for every frame (the loss
       contact-gates at training time, so non-contact rows still being
       a "valid" closest-surface-point doesn't hurt). This is the
       v2 → v3 fix that replaces the softmax-weighted patch-centroid
       approximation HOIDataset used to compute at load time, which
       carried an estimated 5-10 cm bias.
    2. ``target (T, 5, K)`` — legacy soft K-way distribution over the
       FPS patch atlas, contact-gated (zero rows where not in contact).
       Kept for backward compat / visualisation / entropy diagnostics.
    3. ``patch_centers (K, 3)`` — the per-object patch atlas itself.

    Both spatial outputs use the same ``world_to_object_local`` correction
    that ``extract_contact_state`` applies, so distances and patch IDs
    agree across the pipeline.

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
    anchor_points_world : (T, 5, 3) or None
        Optional per-body-part surface anchor positions in world frame. When
        present, closest-surface targets are computed from these anchors
        instead of the coarse 22-joint body-part positions. This is useful
        for hand contact labels derived from official InterAct surface
        markers: the marker sits on the hand surface, while the 22-joint
        wrist can be 5-20 cm away from the object handle.

    Returns
    -------
    target_xyz_gt : (T, 5, 3) — closest-surface-point xyz per body part
    target : (T, 5, K) — soft K-way assignment per body part (legacy)
    patch_centers : (K, 3) — patch center positions in object-local frame
    """
    if config is None:
        config = TargetConfig()

    T = len(joints)
    K = config.num_patches
    target = np.zeros((T, NUM_BODY_PARTS, K), dtype=np.float32)
    target_xyz_gt = np.zeros((T, NUM_BODY_PARTS, 3), dtype=np.float32)

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

    if anchor_points_world is not None and anchor_points_world.shape != (T, NUM_BODY_PARTS, 3):
        raise ValueError(
            f"anchor_points_world shape {anchor_points_world.shape} does not "
            f"match expected {(T, NUM_BODY_PARTS, 3)}"
        )
    if (
        target_points_local_override is not None
        and target_points_local_override.shape != (T, NUM_BODY_PARTS, 3)
    ):
        raise ValueError(
            f"target_points_local_override shape {target_points_local_override.shape} "
            f"does not match expected {(T, NUM_BODY_PARTS, 3)}"
        )

    if anchor_points_world is None:
        bp_positions_world = joints[:, BODY_PART_INDICES, :]
    else:
        bp_positions_world = anchor_points_world

    if object_positions is not None:
        from piano.data.pseudo_labels._object_transform import world_points_batch_to_local

        bp_positions_local = world_points_batch_to_local(
            bp_positions_world, object_positions, object_rotations,
        )
    else:
        bp_positions_local = bp_positions_world.astype(np.float32)

    contact_mask = contact_state >= config.contact_threshold
    override_mask = np.zeros((T, NUM_BODY_PARTS), dtype=bool)
    if target_points_local_override is not None:
        override_mask = np.isfinite(target_points_local_override).all(axis=-1)
        target_xyz_gt[override_mask] = target_points_local_override[override_mask].astype(np.float32)

    query_mask = (
        np.ones((T, NUM_BODY_PARTS), dtype=bool)
        if config.query_all_frames
        else contact_mask
    )
    query_mask = query_mask & ~override_mask
    if np.any(query_mask):
        # Batch all selected body parts into one trimesh proximity query to
        # reuse the mesh spatial index instead of querying it once per part.
        _, closest_pts = points_to_mesh_distance(bp_positions_local[query_mask], object_mesh)
        target_xyz_gt[query_mask] = closest_pts.astype(np.float32)

    if np.any(contact_mask):
        query = target_xyz_gt[contact_mask]
        dists = np.linalg.norm(query[:, None, :] - patch_centers[None, :, :], axis=-1)
        logits = -(dists ** 2) / (2.0 * config.soft_sigma ** 2)
        logits -= logits.max(axis=1, keepdims=True)
        weights = np.exp(logits)
        target[contact_mask] = (weights / weights.sum(axis=1, keepdims=True)).astype(np.float32)

    return target_xyz_gt, target, patch_centers
