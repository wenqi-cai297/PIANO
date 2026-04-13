"""Standard motion generation metrics following HumanML3D evaluation protocol.

Computes FID, R-Precision (top-1/2/3), MM-Dist, Diversity, and MultiModality
using the pretrained motion/text feature extractors from the HumanML3D
evaluation toolkit.

These metrics measure generation quality against the motion distribution
and text-motion alignment, independent of interaction quality.

Usage:
    piano-eval --generated runs/eval/generated.npy --gt data/humanml3d/test.npy
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


@dataclass(slots=True)
class MotionMetrics:
    """Container for standard motion generation metrics."""

    fid: float
    r_precision_top1: float
    r_precision_top2: float
    r_precision_top3: float
    mm_dist: float
    diversity: float
    multimodality: float


def compute_fid(
    gen_features: np.ndarray,
    gt_features: np.ndarray,
) -> float:
    """Compute Frechet Inception Distance between two feature distributions.

    Parameters
    ----------
    gen_features : (N, D) — features from generated motions
    gt_features : (M, D) — features from ground truth motions
    """
    mu_gen = gen_features.mean(axis=0)
    mu_gt = gt_features.mean(axis=0)
    sigma_gen = np.cov(gen_features, rowvar=False)
    sigma_gt = np.cov(gt_features, rowvar=False)

    diff = mu_gen - mu_gt
    # Product of covariance matrices
    from scipy.linalg import sqrtm

    covmean = sqrtm(sigma_gen @ sigma_gt)
    # Numerical stability: discard imaginary component
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_gen + sigma_gt - 2 * covmean)
    return float(fid)


def compute_r_precision(
    motion_features: np.ndarray,
    text_features: np.ndarray,
    top_k: int = 3,
    batch_size: int = 32,
) -> tuple[float, float, float]:
    """Compute R-Precision (top-1, top-2, top-3).

    For each motion, rank all text descriptions by cosine similarity
    and check if the correct text is in the top-k.

    Parameters
    ----------
    motion_features : (N, D) — motion features
    text_features : (N, D) — corresponding text features (same order)
    """
    N = len(motion_features)
    correct = {1: 0, 2: 0, 3: 0}

    for i in range(0, N, batch_size):
        end = min(i + batch_size, N)
        m_batch = motion_features[i:end]  # (B, D)
        t_batch = text_features[i:end]    # (B, D)
        B = len(m_batch)

        # Cosine similarity: (B, B)
        m_norm = m_batch / (np.linalg.norm(m_batch, axis=1, keepdims=True) + 1e-8)
        t_norm = t_batch / (np.linalg.norm(t_batch, axis=1, keepdims=True) + 1e-8)
        sim = m_norm @ t_norm.T  # (B, B)

        # For each motion (row), check if the diagonal (correct text) is in top-k
        rankings = np.argsort(-sim, axis=1)  # (B, B) — descending
        for b in range(B):
            rank = np.where(rankings[b] == b)[0][0]
            for k in [1, 2, 3]:
                if rank < k:
                    correct[k] += 1

    return correct[1] / N, correct[2] / N, correct[3] / N


def compute_diversity(
    features: np.ndarray,
    num_pairs: int = 300,
) -> float:
    """Compute diversity as average pairwise distance between random motion pairs."""
    N = len(features)
    if N < 2:
        return 0.0

    rng = np.random.default_rng(42)
    idx_a = rng.choice(N, size=num_pairs, replace=True)
    idx_b = rng.choice(N, size=num_pairs, replace=True)

    dists = np.linalg.norm(features[idx_a] - features[idx_b], axis=1)
    return float(dists.mean())


def compute_mm_dist(
    motion_features: np.ndarray,
    text_features: np.ndarray,
) -> float:
    """Compute multimodal distance: average distance between matched motion-text pairs."""
    dists = np.linalg.norm(motion_features - text_features, axis=1)
    return float(dists.mean())


def compute_multimodality(
    features_per_text: list[np.ndarray],
    num_pairs: int = 100,
) -> float:
    """Compute multimodality: average diversity of motions generated from the same text.

    Parameters
    ----------
    features_per_text : list of (K, D) arrays — K generations per text prompt
    """
    if not features_per_text:
        return 0.0

    rng = np.random.default_rng(42)
    all_dists = []
    for feats in features_per_text:
        K = len(feats)
        if K < 2:
            continue
        idx_a = rng.choice(K, size=min(num_pairs, K * (K - 1) // 2), replace=True)
        idx_b = rng.choice(K, size=min(num_pairs, K * (K - 1) // 2), replace=True)
        dists = np.linalg.norm(feats[idx_a] - feats[idx_b], axis=1)
        all_dists.append(dists.mean())

    return float(np.mean(all_dists)) if all_dists else 0.0


def evaluate_all(
    gen_motion_features: np.ndarray,
    gen_text_features: np.ndarray,
    gt_motion_features: np.ndarray,
    gt_text_features: np.ndarray,
    gen_features_per_text: list[np.ndarray] | None = None,
) -> MotionMetrics:
    """Compute all standard motion metrics.

    Parameters
    ----------
    gen_motion_features : (N, D) — generated motion features
    gen_text_features : (N, D) — text features for generated motions
    gt_motion_features : (M, D) — ground truth motion features
    gt_text_features : (M, D) — ground truth text features
    gen_features_per_text : optional, for multimodality computation
    """
    fid = compute_fid(gen_motion_features, gt_motion_features)
    top1, top2, top3 = compute_r_precision(gen_motion_features, gen_text_features)
    mm_dist = compute_mm_dist(gen_motion_features, gen_text_features)
    diversity = compute_diversity(gen_motion_features)
    multimodality = compute_multimodality(gen_features_per_text or [])

    return MotionMetrics(
        fid=fid,
        r_precision_top1=top1,
        r_precision_top2=top2,
        r_precision_top3=top3,
        mm_dist=mm_dist,
        diversity=diversity,
        multimodality=multimodality,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entrypoint for ``piano-eval``."""
    parser = argparse.ArgumentParser(description="Evaluate generated motions")
    parser.add_argument("--gen-features", type=Path, required=True, help="Generated motion features npz")
    parser.add_argument("--gt-features", type=Path, required=True, help="Ground truth features npz")
    args = parser.parse_args()

    gen = np.load(args.gen_features)
    gt = np.load(args.gt_features)

    metrics = evaluate_all(
        gen["motion_features"], gen["text_features"],
        gt["motion_features"], gt["text_features"],
    )

    print(f"FID:              {metrics.fid:.4f}")
    print(f"R-Precision@1:    {metrics.r_precision_top1:.4f}")
    print(f"R-Precision@2:    {metrics.r_precision_top2:.4f}")
    print(f"R-Precision@3:    {metrics.r_precision_top3:.4f}")
    print(f"MM-Dist:          {metrics.mm_dist:.4f}")
    print(f"Diversity:        {metrics.diversity:.4f}")
    print(f"MultiModality:    {metrics.multimodality:.4f}")


if __name__ == "__main__":
    main()
