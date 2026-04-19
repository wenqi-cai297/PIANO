"""Geometry utilities for point cloud and mesh operations.

Provides lightweight wrappers around trimesh for distance queries,
surface sampling, and patch clustering used in pseudo-label extraction.
"""
from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Mesh loading & surface sampling
# ---------------------------------------------------------------------------

def load_mesh(path: str) -> trimesh.Trimesh:
    """Load a triangle mesh from file (obj, ply, stl, etc.)."""
    mesh = trimesh.load(path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Expected a single Trimesh, got {type(mesh)} from {path}")
    return mesh


def sample_surface_fps(
    mesh: trimesh.Trimesh,
    num_points: int,
    seed: int | None = None,
) -> np.ndarray:
    """Sample *num_points* on the mesh surface using farthest point sampling.

    Returns an array of shape ``(num_points, 3)``.
    """
    rng = np.random.default_rng(seed)
    # Oversample, then FPS down. trimesh's sampler takes a seed directly.
    oversampled, _ = trimesh.sample.sample_surface(mesh, num_points * 10, seed=seed)
    indices = _farthest_point_sample(oversampled, num_points, rng=rng)
    return oversampled[indices]


# ---------------------------------------------------------------------------
# Distance queries
# ---------------------------------------------------------------------------

def points_to_mesh_distance(
    points: np.ndarray,
    mesh: trimesh.Trimesh,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute signed distance from each point to the mesh surface.

    Parameters
    ----------
    points : (N, 3) array
    mesh : trimesh.Trimesh

    Returns
    -------
    distances : (N,) array — unsigned distance to closest surface point
    closest_points : (N, 3) array — closest point on the mesh surface
    """
    closest_points, distances, _ = trimesh.proximity.closest_point(mesh, points)
    return distances, closest_points


def points_to_points_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise L2 distance between two point sets.

    Parameters
    ----------
    a : (N, 3) array
    b : (M, 3) array

    Returns
    -------
    dists : (N, M) array
    """
    # Use broadcasting: (N,1,3) - (1,M,3) -> (N,M,3) -> norm -> (N,M)
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)


# ---------------------------------------------------------------------------
# Patch clustering
# ---------------------------------------------------------------------------

def cluster_surface_patches(
    mesh: trimesh.Trimesh,
    num_patches: int,
    num_surface_samples: int = 4096,
    seed: int | None = None,
) -> np.ndarray:
    """Cluster mesh surface into *num_patches* regions via FPS.

    Returns patch center positions of shape ``(num_patches, 3)``.
    These are used as contact target anchors for pseudo-label extraction.

    Pass a ``seed`` to make patch ids deterministic — otherwise the FPS
    random starting point produces a different ordering on every call,
    breaking any downstream model that consumes ``contact_target`` as a
    fixed categorical vector.
    """
    rng = np.random.default_rng(seed)
    surface_points, _ = trimesh.sample.sample_surface(
        mesh, num_surface_samples, seed=seed,
    )
    indices = _farthest_point_sample(surface_points, num_patches, rng=rng)
    return surface_points[indices]


def soft_patch_assignment(
    query_point: np.ndarray,
    patch_centers: np.ndarray,
    sigma: float = 0.01,
) -> np.ndarray:
    """Compute soft assignment of a point to patch centers.

    Parameters
    ----------
    query_point : (3,) array
    patch_centers : (K, 3) array
    sigma : temperature for softmax

    Returns
    -------
    weights : (K,) array summing to 1
    """
    dists = np.linalg.norm(patch_centers - query_point[None, :], axis=-1)
    logits = -dists / (2.0 * sigma ** 2)
    logits -= logits.max()  # numerical stability
    weights = np.exp(logits)
    return weights / weights.sum()


# ---------------------------------------------------------------------------
# Farthest Point Sampling (numpy, CPU)
# ---------------------------------------------------------------------------

def _farthest_point_sample(
    points: np.ndarray,
    num_samples: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Greedy farthest point sampling on a (N, 3) point cloud.

    Returns indices of shape ``(num_samples,)``. When ``rng`` is provided,
    the starting index is drawn from that generator — use a seeded generator
    (``np.random.default_rng(seed)``) for reproducible patch atlases.
    """
    n = len(points)
    if num_samples >= n:
        return np.arange(n)

    if rng is None:
        rng = np.random.default_rng()

    indices = np.zeros(num_samples, dtype=np.int64)
    indices[0] = int(rng.integers(n))
    min_dists = np.full(n, np.inf)

    for i in range(1, num_samples):
        current = points[indices[i - 1]]
        dists = np.linalg.norm(points - current[None, :], axis=-1)
        min_dists = np.minimum(min_dists, dists)
        indices[i] = np.argmax(min_dists)

    return indices


# ---------------------------------------------------------------------------
# KD-tree nearest neighbour (for fast contact target lookup)
# ---------------------------------------------------------------------------

def build_kdtree(points: np.ndarray) -> cKDTree:
    """Build a KD-tree from a (N, 3) point array."""
    return cKDTree(points)


def query_nearest(tree: cKDTree, query: np.ndarray, k: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Query nearest neighbour(s) in the KD-tree.

    Returns
    -------
    distances : (N,) or (N, k) array
    indices : (N,) or (N, k) array
    """
    return tree.query(query, k=k)
