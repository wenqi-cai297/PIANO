"""Tests for stage1_losses helpers (R31 V2 ablation matrix).

Each loss is tested for: (a) zero on a hand-crafted "correct" input,
(b) positive and finite on a perturbed input, (c) mask behaviour.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from piano.training.smpl_kinematics import (
    fk_from_global_rotations,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)
from piano.training.stage1_losses import (
    CH_HEAD_HEIGHT,
    CH_PELVIS_ROT6D,
    CH_SHOULDER_H,
    CH_SPINE3_ROT6D,
    J_HEAD,
    J_L_SHOULDER,
    J_PELVIS,
    J_R_SHOULDER,
    J_SPINE3,
    fk_height_consistency_loss,
    fk_pelvis_spine_pos_loss,
    kinematic_self_consistency_loss,
    rot6d_ortho_loss,
)


# ──────────────────────────────────────────────────────────────────────────
# L1: rot6d orthogonality
# ──────────────────────────────────────────────────────────────────────────


def test_rot6d_ortho_zero_on_identity_rot6d():
    """A perfectly orthonormal (a1, a2) gives zero loss.

    a1 = (1, 0, 0), a2 = (0, 1, 0) → ||a1|| = ||a2|| = 1, <a1, a2> = 0.
    """
    rot6d = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
    loss = rot6d_ortho_loss(rot6d)
    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_rot6d_ortho_positive_on_non_unit_a1():
    """||a1|| = 2 → norm violation."""
    rot6d = torch.tensor([[2.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
    loss = rot6d_ortho_loss(rot6d)
    # (2 - 1)² = 1 from the norm_a1 term.
    assert loss.item() == pytest.approx(1.0, abs=1e-6)


def test_rot6d_ortho_positive_on_non_orthogonal():
    """a1 = a2 → dot = 1 → loss > 0."""
    rot6d = torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
    loss = rot6d_ortho_loss(rot6d)
    # norm_a1 = 1, norm_a2 = 1, dot = 1 → 0 + 0 + 1 = 1.
    assert loss.item() == pytest.approx(1.0, abs=1e-6)


def test_rot6d_ortho_respects_mask():
    """Masked-out element should not contribute."""
    bad = torch.tensor([5.0, 0.0, 0.0, 0.0, 1.0, 0.0])           # bad: ||a1||=5
    good = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    rot6d = torch.stack([bad, good])                              # (2, 6)
    mask = torch.tensor([0.0, 1.0])                               # ignore bad
    loss = rot6d_ortho_loss(rot6d, mask=mask)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_rot6d_ortho_real_pred_from_random():
    """Random 6D output (post-init) should yield a positive but bounded loss."""
    torch.manual_seed(0)
    rot6d = torch.randn(2, 16, 6)
    loss = rot6d_ortho_loss(rot6d)
    assert torch.isfinite(loss)
    assert loss.item() > 0


# ──────────────────────────────────────────────────────────────────────────
# L2: FK pelvis+spine position loss
# ──────────────────────────────────────────────────────────────────────────


def _make_synthetic_motion(B: int, T: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a GT motion_135 with identity rot for every joint, root at zero.

    With identity rot everywhere, FK output equals the rest_offsets cumulative
    sum along the kintree. This gives a stable reference for tests.
    """
    rng = torch.Generator().manual_seed(42)
    rot_mat = (
        torch.eye(3)
        .reshape(1, 1, 1, 3, 3)
        .expand(B, T, 22, 3, 3)
        .contiguous()
    )
    rot6d = matrix_to_rotation_6d(rot_mat)                        # (B, T, 22, 6)
    rot6d_flat = rot6d.reshape(B, T, 22 * 6)                      # 132
    root_world = torch.zeros(B, T, 3)
    motion_135 = torch.cat([rot6d_flat, root_world], dim=-1)
    rest_offsets = torch.randn(B, 22, 3, generator=rng) * 0.1
    return motion_135, rest_offsets, root_world


def test_fk_pelvis_spine_zero_when_pred_equals_gt():
    """Stage-1 predicts the GT pelvis + spine3 rotation → loss = 0."""
    B, T = 2, 16
    motion_135, rest_offsets, root_world = _make_synthetic_motion(B, T)
    # Pelvis + spine3 rot6d = identity (a1=(1,0,0), a2=(0,1,0)).
    identity_rot6d = (
        torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        .reshape(1, 1, 6).expand(B, T, 6).contiguous()
    )
    seq_mask = torch.ones(B, T)
    # GT joints under identity rotation = cumulative rest_offsets along kintree.
    gt_rot_mat = rotation_6d_to_matrix(motion_135[..., :132].reshape(B, T, 22, 6))
    gt_joints = fk_from_global_rotations(
        gt_rot_mat,
        rest_offsets.unsqueeze(1).expand(B, T, 22, 3),
        root_world,
    )
    loss = fk_pelvis_spine_pos_loss(
        pelvis_rot6d_pred=identity_rot6d,
        spine3_rot6d_pred=identity_rot6d,
        root_world_pred=root_world,
        gt_motion_135=motion_135,
        rest_offsets=rest_offsets,
        gt_joints=gt_joints,
        seq_mask=seq_mask,
    )
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_fk_pelvis_spine_positive_when_pred_wrong():
    """Perturb pelvis rot6d → non-zero loss."""
    B, T = 2, 16
    motion_135, rest_offsets, root_world = _make_synthetic_motion(B, T)
    identity_rot6d = (
        torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        .reshape(1, 1, 6).expand(B, T, 6).contiguous()
    )
    # Perturbed: rotate by some non-trivial axis (the network has not learned).
    perturbed_rot6d = (
        torch.tensor([0.0, 1.0, 0.0, -1.0, 0.0, 0.0])
        .reshape(1, 1, 6).expand(B, T, 6).contiguous()
    )
    seq_mask = torch.ones(B, T)
    gt_rot_mat = rotation_6d_to_matrix(motion_135[..., :132].reshape(B, T, 22, 6))
    gt_joints = fk_from_global_rotations(
        gt_rot_mat,
        rest_offsets.unsqueeze(1).expand(B, T, 22, 3),
        root_world,
    )
    loss = fk_pelvis_spine_pos_loss(
        pelvis_rot6d_pred=perturbed_rot6d,
        spine3_rot6d_pred=identity_rot6d,
        root_world_pred=root_world,
        gt_motion_135=motion_135,
        rest_offsets=rest_offsets,
        gt_joints=gt_joints,
        seq_mask=seq_mask,
    )
    assert loss.item() > 1e-6
    assert torch.isfinite(loss)


# ──────────────────────────────────────────────────────────────────────────
# L3: FK-height consistency
# ──────────────────────────────────────────────────────────────────────────


def test_fk_height_zero_when_pred_matches_fk():
    """If the scalar height channels match the FK-derived y, loss = 0."""
    B, T = 2, 8
    motion_135, rest_offsets, root_world = _make_synthetic_motion(B, T)
    identity_rot6d = (
        torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        .reshape(1, 1, 6).expand(B, T, 6).contiguous()
    )
    # Compute FK outputs with the same identity rotations.
    gt_rot_mat = rotation_6d_to_matrix(motion_135[..., :132].reshape(B, T, 22, 6))
    fk_joints = fk_from_global_rotations(
        gt_rot_mat,
        rest_offsets.unsqueeze(1).expand(B, T, 22, 3),
        root_world,
    )
    fk_head_y = fk_joints[..., J_HEAD, 1]                          # (B, T)
    fk_shoulder_y = (
        fk_joints[..., J_L_SHOULDER, 1]
        + fk_joints[..., J_R_SHOULDER, 1]
    ) * 0.5

    seq_mask = torch.ones(B, T)
    loss = fk_height_consistency_loss(
        head_height_pred=fk_head_y,
        shoulder_h_pred=fk_shoulder_y,
        pelvis_rot6d_pred=identity_rot6d,
        spine3_rot6d_pred=identity_rot6d,
        root_world_pred=root_world,
        gt_motion_135=motion_135,
        rest_offsets=rest_offsets,
        seq_mask=seq_mask,
    )
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_fk_height_positive_when_scalar_disagrees():
    """Predicted head height = 999 m → big loss."""
    B, T = 2, 8
    motion_135, rest_offsets, root_world = _make_synthetic_motion(B, T)
    identity_rot6d = (
        torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        .reshape(1, 1, 6).expand(B, T, 6).contiguous()
    )
    seq_mask = torch.ones(B, T)
    loss = fk_height_consistency_loss(
        head_height_pred=torch.full((B, T), 999.0),
        shoulder_h_pred=torch.zeros(B, T),
        pelvis_rot6d_pred=identity_rot6d,
        spine3_rot6d_pred=identity_rot6d,
        root_world_pred=root_world,
        gt_motion_135=motion_135,
        rest_offsets=rest_offsets,
        seq_mask=seq_mask,
    )
    assert loss.item() > 100.0
    assert torch.isfinite(loss)


# ──────────────────────────────────────────────────────────────────────────
# L4: kinematic self-consistency
# ──────────────────────────────────────────────────────────────────────────


def test_kinematic_consistency_zero_on_well_formed():
    """Construct a 23-D trajectory where vel = diff(root_local) and
    yaw_vel = diff(atan2(yaw_sin, yaw_cos)). Loss should be 0.
    """
    B, T = 1, 5
    stage1 = torch.zeros(B, T, 23)
    # Make root_local move by (0.1, 0.2, 0.3) per frame.
    for t in range(T):
        stage1[0, t, 0] = 0.1 * t
        stage1[0, t, 1] = 0.2 * t
        stage1[0, t, 2] = 0.3 * t
    # vel = (0.1, 0.2, 0.3) at every t.
    stage1[0, :, 3:6] = torch.tensor([0.1, 0.2, 0.3])
    # yaw flat at 0 → sin=0, cos=1, yaw_vel=0.
    stage1[0, :, 6] = 0.0
    stage1[0, :, 7] = 1.0
    stage1[0, :, 8] = 0.0

    seq_mask = torch.ones(B, T)
    loss = kinematic_self_consistency_loss(stage1, seq_mask)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_kinematic_consistency_positive_on_inconsistent():
    """vel disagrees with diff(root_local) → loss > 0."""
    B, T = 1, 4
    stage1 = torch.zeros(B, T, 23)
    # root_local moves but vel = 0 everywhere → trans-error nonzero.
    for t in range(T):
        stage1[0, t, 0] = float(t)        # moves by 1 m/frame
    stage1[0, :, 7] = 1.0                  # cos = 1 so atan2 is finite
    seq_mask = torch.ones(B, T)
    loss = kinematic_self_consistency_loss(stage1, seq_mask)
    # vel = 0, diff = 1 in x → (1-0)² per frame = 1.
    assert loss.item() > 0.5
    assert torch.isfinite(loss)


def test_kinematic_consistency_handles_T1():
    """T=1 has no diffs → 0 loss with safe grad sentinel."""
    stage1 = torch.zeros(1, 1, 23, requires_grad=True)
    seq_mask = torch.ones(1, 1)
    loss = kinematic_self_consistency_loss(stage1, seq_mask)
    assert loss.item() == 0.0
    loss.backward()
    assert stage1.grad is not None
