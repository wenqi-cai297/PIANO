"""Object encoder: PointNet++ with PointNeXt-style refinement.

Encodes an object point cloud into a set of feature tokens that represent
local surface regions. These tokens are the K/V for the Interaction
Predictor's object cross-attention.

The 2026-04-24 rewrite:

    * Output **128** tokens (up from 16). Cross-attention KV count is
      decoupled from the K=16 classification label space. 16 tokens was
      too coarse to discriminate patches on articulated objects (chair
      seat vs. armrest vs. backrest collapsed into one token) — the
      contact-target head had no geometric context to work with.

    * Two SA stages (1024 → 512 → 128) instead of three (1024 → 256 →
      64 → 16). Shallower but wider; better information preservation
      at modest cost.

    * PointNeXt-style refinements:
        - GELU activations (replaces ReLU)
        - Inverted-residual MLP refinement block on centroid features
          after the final SA stage
      Full PointNeXt's training-recipe improvements (label smoothing,
      mixup, etc.) belong in the training loop; this file implements
      only the architectural changes that survive independently.

Reference: Qian et al., PointNeXt (NeurIPS 2022), arXiv 2206.04670.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class SetAbstractionLayer(nn.Module):
    """PointNet++ Set Abstraction (SA) layer with PointNeXt activations.

    Downsamples a point cloud by selecting a subset of centroids via
    farthest point sampling, grouping neighbours within a ball radius,
    and applying a shared MLP to each group.
    """

    def __init__(
        self,
        num_points: int,
        radius: float,
        num_samples: int,
        in_channels: int,
        mlp_channels: list[int],
    ) -> None:
        super().__init__()
        self.num_points = num_points
        self.radius = radius
        self.num_samples = num_samples

        layers: list[nn.Module] = []
        prev_ch = in_channels + 3  # +3 for relative xyz
        for out_ch in mlp_channels:
            layers.extend([
                nn.Conv1d(prev_ch, out_ch, 1),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
            ])
            prev_ch = out_ch
        self.mlp = nn.Sequential(*layers)
        self.out_channels = mlp_channels[-1]

    def forward(self, xyz: Tensor, features: Tensor | None) -> tuple[Tensor, Tensor]:
        """Forward pass.

        Parameters
        ----------
        xyz : (B, N, 3) — input point positions
        features : (B, N, C) or None — per-point features

        Returns
        -------
        new_xyz : (B, num_points, 3) — centroid positions
        new_features : (B, num_points, out_channels) — aggregated features
        """
        B, N, _ = xyz.shape

        # Farthest point sampling (batched)
        centroids = self._fps(xyz, self.num_points)  # (B, num_points)
        new_xyz = self._gather(xyz, centroids)       # (B, num_points, 3)

        # Ball query and grouping
        grouped_xyz, grouped_feat = self._ball_query_and_group(
            xyz, new_xyz, features,
        )

        # Shared MLP + max pool over the neighbourhood
        grouped = grouped_xyz if grouped_feat is None else torch.cat(
            [grouped_xyz, grouped_feat], dim=-1
        )
        B, M, S, C = grouped.shape
        grouped = grouped.reshape(B * M, S, C).permute(0, 2, 1)  # (B*M, C, S)
        grouped = self.mlp(grouped)                              # (B*M, out, S)
        grouped = grouped.max(dim=-1).values                     # (B*M, out)
        new_features = grouped.reshape(B, M, -1)                 # (B, M, out)

        return new_xyz, new_features

    def _fps(self, xyz: Tensor, num_samples: int) -> Tensor:
        """Batched farthest point sampling. Returns indices (B, num_samples)."""
        B, N, _ = xyz.shape
        device = xyz.device
        centroids = torch.zeros(B, num_samples, dtype=torch.long, device=device)
        distance = torch.full((B, N), 1e10, device=device)
        farthest = torch.randint(0, N, (B,), device=device)

        for i in range(num_samples):
            centroids[:, i] = farthest
            centroid_xyz = xyz[torch.arange(B, device=device), farthest].unsqueeze(1)
            dist = torch.sum((xyz - centroid_xyz) ** 2, dim=-1)
            distance = torch.minimum(distance, dist)
            farthest = distance.argmax(dim=-1)

        return centroids

    def _gather(self, xyz: Tensor, indices: Tensor) -> Tensor:
        """Gather points by indices: (B, N, 3) + (B, M) -> (B, M, 3)."""
        idx_expanded = indices.unsqueeze(-1).expand(-1, -1, 3)
        return torch.gather(xyz, 1, idx_expanded)

    def _ball_query_and_group(
        self,
        xyz: Tensor,
        new_xyz: Tensor,
        features: Tensor | None,
    ) -> tuple[Tensor, Tensor | None]:
        """Ball query around each centroid, group relative coords + features.

        Returns grouped_xyz (B, M, S, 3) and grouped_feat (B, M, S, C) or None.
        """
        B, N, _ = xyz.shape
        M = new_xyz.shape[1]
        S = min(self.num_samples, N)

        # Pairwise distances: (B, M, N)
        dists = torch.cdist(new_xyz, xyz)

        dists_sorted, idx_sorted = dists.sort(dim=-1)
        idx_sorted = idx_sorted[:, :, :S]  # (B, M, S)

        # Clamp points outside radius to the nearest neighbour
        mask = dists_sorted[:, :, :S] > self.radius
        first_idx = idx_sorted[:, :, 0:1].expand_as(idx_sorted)
        idx_sorted = torch.where(mask, first_idx, idx_sorted)

        idx_flat = idx_sorted.reshape(B, -1)
        grouped_xyz = self._gather_features(xyz, idx_flat).reshape(B, M, S, 3)
        grouped_xyz = grouped_xyz - new_xyz.unsqueeze(2)

        grouped_feat = None
        if features is not None:
            C = features.shape[-1]
            grouped_feat = self._gather_features(features, idx_flat).reshape(B, M, S, C)

        return grouped_xyz, grouped_feat

    @staticmethod
    def _gather_features(src: Tensor, idx: Tensor) -> Tensor:
        """Gather from (B, N, C) using indices (B, K) -> (B, K, C)."""
        C = src.shape[-1]
        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, C)
        return torch.gather(src, 1, idx_expanded)


class InvResMLPBlock(nn.Module):
    """Inverted-residual MLP block (PointNeXt), applied per-centroid.

    Expands to ``expansion × dim``, GELUs, compresses back, and adds a
    residual. BatchNorm is over the (B*M) dimension — safe here because
    we apply this after SA stages, where M is fixed per batch.
    """

    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = dim * expansion
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, M, dim) — per-centroid features."""
        return x + self.fc2(self.drop(self.act(self.fc1(self.norm(x)))))


class ObjectEncoder(nn.Module):
    """PointNet++ (+ PointNeXt refinement) encoder for object point clouds.

    Takes a raw point cloud (B, N, 3) and outputs M feature tokens
    (B, M, feature_dim) representing local object surface regions.

    Default: 1024 input points → 512 centroids → 128 centroids, feature
    dimension matching the predictor's ``d_model``.
    """

    def __init__(
        self,
        num_input_points: int = 1024,
        num_output_tokens: int = 128,
        feature_dim: int = 384,
    ) -> None:
        super().__init__()
        self.num_output_tokens = num_output_tokens

        # Two SA stages: 1024 → 512 → 128
        self.sa1 = SetAbstractionLayer(
            num_points=512, radius=0.15, num_samples=32,
            in_channels=0, mlp_channels=[64, 128],
        )
        self.sa2 = SetAbstractionLayer(
            num_points=num_output_tokens, radius=0.3, num_samples=64,
            in_channels=128, mlp_channels=[128, 256, feature_dim],
        )

        # PointNeXt-style refinement on the final centroid features.
        # Cheap (one block) and lets the encoder mix information across
        # the 128 tokens after max-pool already summarised each group.
        self.refine = InvResMLPBlock(feature_dim, expansion=4, dropout=0.0)

    def forward(self, point_cloud: Tensor) -> Tensor:
        """Encode object point cloud into feature tokens.

        Parameters
        ----------
        point_cloud : (B, N, 3)

        Returns
        -------
        tokens : (B, M, feature_dim) where M = num_output_tokens
        """
        xyz = point_cloud
        feat: Tensor | None = None

        xyz, feat = self.sa1(xyz, feat)
        xyz, feat = self.sa2(xyz, feat)
        feat = self.refine(feat)
        return feat
