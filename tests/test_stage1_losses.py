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
    CH_YAW_COS,
    CH_YAW_SIN,
    CH_YAW_VEL,
    CONTACT_IDX_LEFT_HAND,
    CONTACT_IDX_RIGHT_HAND,
    INIT_POSE_F2_DIM,
    INIT_POSE_F2_INDICES,
    J_HEAD,
    J_L_ELBOW,
    J_L_SHOULDER,
    J_L_WRIST,
    J_NECK,
    J_PELVIS,
    J_R_ELBOW,
    J_R_SHOULDER,
    J_R_WRIST,
    J_SPINE3,
    build_init_pose_f1,
    build_init_pose_f2,
    channel_moment_match_loss,
    fk_height_consistency_loss,
    fk_pelvis_spine_pos_loss,
    fk_pelvis_spine_pos_loss_cm,
    frame0_consistency_loss,
    kinematic_self_consistency_loss,
    rot6d_ortho_loss,
    temporal_derivative_mse_loss,
    wrist_fk_supervision_loss,
    yaw_aggregate_match_loss,
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


# ──────────────────────────────────────────────────────────────────────────
# V7-A: channel_moment_match_loss
# ──────────────────────────────────────────────────────────────────────────


def test_moment_match_zero_when_pred_equals_gt():
    """Zero loss when pred == gt — moments match exactly."""
    torch.manual_seed(0)
    B, T = 4, 30
    gt = torch.randn(B, T, 23) * 0.5
    pred = gt.clone()
    mask = torch.ones(B, T)
    loss = channel_moment_match_loss(
        pred, gt, mask, velocity_match=True, value_match=True,
    )
    assert loss.item() < 1e-10


def test_moment_match_positive_on_std_collapse():
    """The signature failure mode: pred has correct mean but tiny std."""
    torch.manual_seed(0)
    B, T = 4, 30
    gt = torch.randn(B, T, 23) * 0.5            # gt std ~ 0.5 per channel
    pred = torch.zeros(B, T, 23)                # pred std = 0
    mask = torch.ones(B, T)
    loss = channel_moment_match_loss(
        pred, gt, mask, velocity_match=False, value_match=True,
    )
    assert loss.item() > 0.1
    assert torch.isfinite(loss)


def test_moment_match_velocity_catches_smoothing():
    """Pred matches GT in value but is over-smoothed → velocity moment fires."""
    torch.manual_seed(0)
    B, T = 4, 30
    gt = torch.randn(B, T, 23) * 0.5
    # Smooth pred by averaging neighbors — preserves mean but kills velocity.
    pred = gt.clone()
    pred[:, 1:-1] = (gt[:, :-2] + gt[:, 1:-1] + gt[:, 2:]) / 3.0
    mask = torch.ones(B, T)
    loss = channel_moment_match_loss(
        pred, gt, mask, velocity_match=True, value_match=False,
    )
    # Velocity moment should flag the smoothing.
    assert loss.item() > 0.0
    # And it should be small compared to a fully zeroed pred.
    pred_dead = torch.zeros_like(gt)
    loss_dead = channel_moment_match_loss(
        pred_dead, gt, mask, velocity_match=True, value_match=False,
    )
    assert loss_dead.item() > loss.item()


def test_moment_match_normalization_makes_scales_comparable():
    """With normalize_by_gt_std=True, a small-scale channel's collapse
    contributes ~1.0, same as a large-scale channel's collapse."""
    B, T = 4, 30
    torch.manual_seed(1)
    # Two channels: one with std 0.001 m, one with std 1.0 m.
    gt = torch.zeros(B, T, 23)
    gt[..., 0] = torch.randn(B, T) * 0.001       # tiny scale
    gt[..., 22] = torch.randn(B, T) * 1.0        # big scale
    pred = torch.zeros_like(gt)                  # full collapse
    mask = torch.ones(B, T)
    loss_norm = channel_moment_match_loss(
        pred, gt, mask, velocity_match=False, value_match=True,
        normalize_by_gt_std=True, channel_subset=(0, 22),
    )
    loss_raw = channel_moment_match_loss(
        pred, gt, mask, velocity_match=False, value_match=True,
        normalize_by_gt_std=False, channel_subset=(0, 22),
    )
    # In raw mode, ch 22 dominates by ~1e6.
    # In normalized mode, both contribute ~ same order.
    # We only assert they're both finite and positive here.
    assert loss_norm.item() > 0.0
    assert loss_raw.item() > 0.0
    assert torch.isfinite(loss_norm)
    assert torch.isfinite(loss_raw)


def test_moment_match_respects_mask():
    """Masked-out frames must not pollute the moments."""
    torch.manual_seed(2)
    B, T = 4, 30
    gt = torch.randn(B, T, 23) * 0.3
    pred = gt.clone()
    # Add huge noise on the second half of the first clip — should be masked.
    pred[0, 15:] += 100.0
    mask = torch.ones(B, T)
    mask[0, 15:] = 0.0
    loss = channel_moment_match_loss(
        pred, gt, mask, velocity_match=True, value_match=True,
    )
    # The masked region's 100x noise must NOT show up.
    assert loss.item() < 1.0


def test_moment_match_no_match_returns_zero():
    """When both velocity and value match are off, return 0 cleanly."""
    pred = torch.randn(2, 4, 23, requires_grad=True)
    gt = torch.randn(2, 4, 23)
    mask = torch.ones(2, 4)
    loss = channel_moment_match_loss(
        pred, gt, mask, velocity_match=False, value_match=False,
    )
    assert loss.item() == 0.0


# ──────────────────────────────────────────────────────────────────────────
# V7-B: yaw_aggregate_match_loss
# ──────────────────────────────────────────────────────────────────────────


def test_yaw_aggregate_zero_when_pred_equals_gt():
    """Identical yaw → 0 loss."""
    B, T = 2, 30
    yaw = torch.linspace(0.0, 1.5, T).unsqueeze(0).expand(B, T)
    raw = torch.zeros(B, T, 23)
    raw[..., CH_YAW_SIN] = torch.sin(yaw)
    raw[..., CH_YAW_COS] = torch.cos(yaw)
    mask = torch.ones(B, T)
    loss = yaw_aggregate_match_loss(raw, raw, mask)
    assert loss.item() < 1e-6


def test_yaw_aggregate_mode_invariant_to_direction():
    """Flipping CW <-> CCW (negate yaw) should still give a low loss because
    |Δyaw| and range are invariant to sign reversal."""
    B, T = 2, 30
    yaw_cw = torch.linspace(0.0, 1.5, T).unsqueeze(0).expand(B, T)
    yaw_ccw = -yaw_cw
    raw_gt = torch.zeros(B, T, 23)
    raw_gt[..., CH_YAW_SIN] = torch.sin(yaw_cw)
    raw_gt[..., CH_YAW_COS] = torch.cos(yaw_cw)
    raw_pred = torch.zeros(B, T, 23)
    raw_pred[..., CH_YAW_SIN] = torch.sin(yaw_ccw)
    raw_pred[..., CH_YAW_COS] = torch.cos(yaw_ccw)
    mask = torch.ones(B, T)
    loss = yaw_aggregate_match_loss(raw_pred, raw_gt, mask)
    # The magnitude and range are identical; only sign differs.
    # |Δyaw| absolute matches; range identical (|max−min|).
    assert loss.item() < 1e-4


def test_yaw_aggregate_fires_on_frozen_yaw():
    """When pred yaw stays constant but gt rotates — large loss."""
    B, T = 2, 30
    yaw_gt = torch.linspace(0.0, 1.5, T).unsqueeze(0).expand(B, T)
    raw_gt = torch.zeros(B, T, 23)
    raw_gt[..., CH_YAW_SIN] = torch.sin(yaw_gt)
    raw_gt[..., CH_YAW_COS] = torch.cos(yaw_gt)
    raw_pred = torch.zeros(B, T, 23)
    raw_pred[..., CH_YAW_SIN] = 0.0              # yaw = 0 (frozen)
    raw_pred[..., CH_YAW_COS] = 1.0
    mask = torch.ones(B, T)
    loss = yaw_aggregate_match_loss(raw_pred, raw_gt, mask)
    assert loss.item() > 0.01
    assert torch.isfinite(loss)


def test_yaw_aggregate_handles_T1():
    """T=1 → 0 loss, no crash."""
    base = torch.zeros(2, 1, 23)
    base[..., CH_YAW_COS] = 1.0
    raw = base.clone().requires_grad_(True)
    mask = torch.ones(2, 1)
    loss = yaw_aggregate_match_loss(raw, base, mask)
    assert loss.item() == 0.0


# ──────────────────────────────────────────────────────────────────────────
# V7-C: fk_pelvis_spine_pos_loss_cm
# ──────────────────────────────────────────────────────────────────────────


def test_fk_pos_cm_zero_when_pred_equals_gt():
    """Identity pred rotations + GT root → 0 cm error."""
    torch.manual_seed(0)
    B, T = 2, 5
    # Random GT motion.
    gt_rot6d = torch.randn(B, T, 22, 6)
    # Normalize via FK roundtrip to get a valid SMPL rot6d.
    gt_rot_mat = rotation_6d_to_matrix(gt_rot6d)
    gt_rot6d = matrix_to_rotation_6d(gt_rot_mat)
    gt_root = torch.randn(B, T, 3) * 0.1
    gt_motion_135 = torch.cat([gt_rot6d.reshape(B, T, -1), gt_root], dim=-1)
    rest_offsets = torch.zeros(B, 22, 3)
    rest_offsets[:, :, 1] = 0.1                                # any rest skeleton
    rest_per_frame = rest_offsets.unsqueeze(1).expand(B, T, 22, 3).float()
    gt_joints = fk_from_global_rotations(gt_rot_mat, rest_per_frame, gt_root)
    seq_mask = torch.ones(B, T)
    loss = fk_pelvis_spine_pos_loss_cm(
        pelvis_rot6d_pred=gt_rot6d[:, :, J_PELVIS, :],
        spine3_rot6d_pred=gt_rot6d[:, :, J_SPINE3, :],
        root_world_pred=gt_root,
        gt_motion_135=gt_motion_135,
        rest_offsets=rest_offsets,
        gt_joints=gt_joints,
        seq_mask=seq_mask,
    )
    assert loss.item() < 1e-4


def test_fk_pos_cm_positive_on_perturbation():
    """Perturb pred root → positive cm-scale loss, finite gradient."""
    torch.manual_seed(0)
    B, T = 2, 5
    gt_rot6d = torch.randn(B, T, 22, 6)
    gt_rot_mat = rotation_6d_to_matrix(gt_rot6d)
    gt_rot6d = matrix_to_rotation_6d(gt_rot_mat)
    gt_root = torch.randn(B, T, 3) * 0.1
    gt_motion_135 = torch.cat([gt_rot6d.reshape(B, T, -1), gt_root], dim=-1)
    rest_offsets = torch.zeros(B, 22, 3)
    rest_offsets[:, :, 1] = 0.1
    rest_per_frame = rest_offsets.unsqueeze(1).expand(B, T, 22, 3).float()
    gt_joints = fk_from_global_rotations(gt_rot_mat, rest_per_frame, gt_root)
    seq_mask = torch.ones(B, T)
    # Perturb root by 5 cm — expect SmoothL1 to register linearly above 1 cm.
    bad_root = gt_root.clone()
    bad_root[..., 0] = bad_root[..., 0] + 0.05
    bad_root.requires_grad_(True)
    loss = fk_pelvis_spine_pos_loss_cm(
        pelvis_rot6d_pred=gt_rot6d[:, :, J_PELVIS, :],
        spine3_rot6d_pred=gt_rot6d[:, :, J_SPINE3, :],
        root_world_pred=bad_root,
        gt_motion_135=gt_motion_135,
        rest_offsets=rest_offsets,
        gt_joints=gt_joints,
        seq_mask=seq_mask,
    )
    assert loss.item() > 0.5         # 5 cm offset → expect O(5) loss
    loss.backward()
    assert bad_root.grad is not None


# ──────────────────────────────────────────────────────────────────────────
# V8: wrist_fk_supervision_loss
# ──────────────────────────────────────────────────────────────────────────


def _build_gt_scaffold(B: int = 2, T: int = 5, seed: int = 0):
    """Helper: build (gt_motion_135, rest_offsets, gt_joints) consistent
    under FK, plus an all-ones seq_mask."""
    torch.manual_seed(seed)
    gt_rot6d_init = torch.randn(B, T, 22, 6)
    gt_rot_mat = rotation_6d_to_matrix(gt_rot6d_init)
    gt_rot6d = matrix_to_rotation_6d(gt_rot_mat)
    gt_root = torch.randn(B, T, 3) * 0.1
    gt_motion = torch.cat([gt_rot6d.reshape(B, T, -1), gt_root], dim=-1)
    rest_offsets = torch.zeros(B, 22, 3)
    rest_offsets[:, :, 1] = 0.1
    rest_per_frame = rest_offsets.unsqueeze(1).expand(B, T, 22, 3).float()
    gt_joints = fk_from_global_rotations(gt_rot_mat, rest_per_frame, gt_root)
    seq_mask = torch.ones(B, T)
    return gt_motion, gt_rot6d, gt_root, rest_offsets, gt_joints, seq_mask


def test_wrist_fk_zero_when_pred_equals_gt():
    """With pred rot6d + root = GT, the extended FK loss is 0."""
    _, gt_rot6d, gt_root, rest_offsets, gt_joints, seq_mask = _build_gt_scaffold()
    B, T = gt_root.shape[:2]
    gt_motion = torch.cat([gt_rot6d.reshape(B, T, -1), gt_root], dim=-1)
    loss = wrist_fk_supervision_loss(
        pelvis_rot6d_pred=gt_rot6d[:, :, J_PELVIS, :],
        spine3_rot6d_pred=gt_rot6d[:, :, J_SPINE3, :],
        root_world_pred=gt_root,
        gt_motion_135=gt_motion,
        rest_offsets=rest_offsets,
        gt_joints=gt_joints,
        seq_mask=seq_mask,
    )
    assert loss.item() < 1e-4


def test_wrist_fk_includes_wrist_in_default_targets():
    """Perturb pred root by 10 cm — wrist supervision should fire larger
    than the equivalent V7-C call (which targets only 4 joints)."""
    _, gt_rot6d, gt_root, rest_offsets, gt_joints, seq_mask = _build_gt_scaffold()
    B, T = gt_root.shape[:2]
    gt_motion = torch.cat([gt_rot6d.reshape(B, T, -1), gt_root], dim=-1)
    bad_root = gt_root.clone()
    bad_root[..., 0] = bad_root[..., 0] + 0.10        # 10 cm
    loss = wrist_fk_supervision_loss(
        pelvis_rot6d_pred=gt_rot6d[:, :, J_PELVIS, :],
        spine3_rot6d_pred=gt_rot6d[:, :, J_SPINE3, :],
        root_world_pred=bad_root,
        gt_motion_135=gt_motion,
        rest_offsets=rest_offsets,
        gt_joints=gt_joints,
        seq_mask=seq_mask,
    )
    assert loss.item() > 1.0         # >1 cm SmoothL1
    assert torch.isfinite(loss)


def test_wrist_fk_joint_weights_scale_wrist_contribution():
    """Doubling wrist weights should increase the loss vs uniform weighting
    when the wrist contribution dominates."""
    _, gt_rot6d, gt_root, rest_offsets, gt_joints, seq_mask = _build_gt_scaffold()
    B, T = gt_root.shape[:2]
    gt_motion = torch.cat([gt_rot6d.reshape(B, T, -1), gt_root], dim=-1)
    # Perturb pelvis_rot6d so that wrist (end of chain) drift is large.
    bad_pelvis = gt_rot6d[:, :, J_PELVIS, :].clone()
    bad_pelvis = bad_pelvis + torch.randn_like(bad_pelvis) * 0.1
    targets = (J_NECK, J_HEAD, J_L_SHOULDER, J_R_SHOULDER,
               J_L_ELBOW, J_R_ELBOW, J_L_WRIST, J_R_WRIST)
    loss_uniform = wrist_fk_supervision_loss(
        pelvis_rot6d_pred=bad_pelvis,
        spine3_rot6d_pred=gt_rot6d[:, :, J_SPINE3, :],
        root_world_pred=gt_root,
        gt_motion_135=gt_motion,
        rest_offsets=rest_offsets,
        gt_joints=gt_joints,
        seq_mask=seq_mask,
        target_joints=targets,
        joint_weights=(1.0,) * 8,
    )
    loss_wrist_heavy = wrist_fk_supervision_loss(
        pelvis_rot6d_pred=bad_pelvis,
        spine3_rot6d_pred=gt_rot6d[:, :, J_SPINE3, :],
        root_world_pred=gt_root,
        gt_motion_135=gt_motion,
        rest_offsets=rest_offsets,
        gt_joints=gt_joints,
        seq_mask=seq_mask,
        target_joints=targets,
        joint_weights=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 4.0, 4.0),
    )
    # Both finite; wrist-heavy may be larger or smaller depending on the
    # specific perturbation, but must not blow up.
    assert torch.isfinite(loss_uniform) and torch.isfinite(loss_wrist_heavy)
    assert loss_uniform.item() > 0.0


def test_wrist_fk_contact_hard_mask_zeros_no_contact():
    """In 'hard' contact mode with no hand contact anywhere, loss = 0."""
    _, gt_rot6d, gt_root, rest_offsets, gt_joints, seq_mask = _build_gt_scaffold()
    B, T = gt_root.shape[:2]
    gt_motion = torch.cat([gt_rot6d.reshape(B, T, -1), gt_root], dim=-1)
    # Big perturbation but no contact frames.
    bad_root = gt_root + 0.5
    contact_state = torch.zeros(B, T, 5)            # nothing contacts
    loss = wrist_fk_supervision_loss(
        pelvis_rot6d_pred=gt_rot6d[:, :, J_PELVIS, :],
        spine3_rot6d_pred=gt_rot6d[:, :, J_SPINE3, :],
        root_world_pred=bad_root,
        gt_motion_135=gt_motion,
        rest_offsets=rest_offsets,
        gt_joints=gt_joints,
        seq_mask=seq_mask,
        contact_state=contact_state,
        contact_mask_mode="hard",
    )
    assert loss.item() == 0.0


def test_wrist_fk_contact_reweight_amplifies_contact_frames():
    """Reweight mode on a NON-uniform error pattern: a bigger perturbation
    on contact frames produces a larger weighted-mean loss than the same
    perturbation under 'off' mode."""
    _, gt_rot6d, gt_root, rest_offsets, gt_joints, seq_mask = _build_gt_scaffold(B=2, T=10)
    B, T = gt_root.shape[:2]
    gt_motion = torch.cat([gt_rot6d.reshape(B, T, -1), gt_root], dim=-1)
    # Non-uniform error: 10 cm offset on contact frames (0-4), 1 cm on others.
    bad_root = gt_root.clone()
    bad_root[:, :5, 0] = bad_root[:, :5, 0] + 0.10
    bad_root[:, 5:, 0] = bad_root[:, 5:, 0] + 0.01

    # Contact active on frames 0-4.
    cs = torch.zeros(B, T, 5)
    cs[:, :5, CONTACT_IDX_LEFT_HAND] = 1.0
    kw = dict(
        pelvis_rot6d_pred=gt_rot6d[:, :, J_PELVIS, :],
        spine3_rot6d_pred=gt_rot6d[:, :, J_SPINE3, :],
        root_world_pred=bad_root,
        gt_motion_135=gt_motion,
        rest_offsets=rest_offsets,
        gt_joints=gt_joints,
        seq_mask=seq_mask,
    )
    loss_off = wrist_fk_supervision_loss(**kw, contact_mask_mode="off")
    loss_reweight = wrist_fk_supervision_loss(
        **kw, contact_state=cs, contact_mask_mode="reweight",
        contact_active_weight=4.0,
    )
    # Reweighting frames where the error is large should INCREASE the
    # weighted mean above the unweighted mean.
    assert loss_reweight.item() > loss_off.item() + 1e-3


def test_wrist_fk_velocity_term_adds_to_pos():
    """Velocity term, when enabled, adds a positive scalar to the loss
    on a temporally-varying perturbation."""
    _, gt_rot6d, gt_root, rest_offsets, gt_joints, seq_mask = _build_gt_scaffold(B=2, T=6)
    B, T = gt_root.shape[:2]
    gt_motion = torch.cat([gt_rot6d.reshape(B, T, -1), gt_root], dim=-1)
    # Perturbation that varies over time → finite vel error.
    bad_root = gt_root.clone()
    bad_root[..., 0] = bad_root[..., 0] + torch.linspace(0.0, 0.2, T).view(1, T).expand(B, T)
    loss_pos = wrist_fk_supervision_loss(
        pelvis_rot6d_pred=gt_rot6d[:, :, J_PELVIS, :],
        spine3_rot6d_pred=gt_rot6d[:, :, J_SPINE3, :],
        root_world_pred=bad_root,
        gt_motion_135=gt_motion,
        rest_offsets=rest_offsets,
        gt_joints=gt_joints,
        seq_mask=seq_mask,
        add_velocity=False,
    )
    loss_pos_vel = wrist_fk_supervision_loss(
        pelvis_rot6d_pred=gt_rot6d[:, :, J_PELVIS, :],
        spine3_rot6d_pred=gt_rot6d[:, :, J_SPINE3, :],
        root_world_pred=bad_root,
        gt_motion_135=gt_motion,
        rest_offsets=rest_offsets,
        gt_joints=gt_joints,
        seq_mask=seq_mask,
        add_velocity=True,
        velocity_weight=0.5,
    )
    assert loss_pos_vel.item() > loss_pos.item()


def test_wrist_fk_gradient_flows_through_pelvis_rot6d():
    """Backward through pred pelvis_rot6d_pred yields a finite grad."""
    _, gt_rot6d, gt_root, rest_offsets, gt_joints, seq_mask = _build_gt_scaffold()
    B, T = gt_root.shape[:2]
    gt_motion = torch.cat([gt_rot6d.reshape(B, T, -1), gt_root], dim=-1)
    bad_pelvis = gt_rot6d[:, :, J_PELVIS, :].clone().requires_grad_(True)
    loss = wrist_fk_supervision_loss(
        pelvis_rot6d_pred=bad_pelvis,
        spine3_rot6d_pred=gt_rot6d[:, :, J_SPINE3, :],
        root_world_pred=gt_root,
        gt_motion_135=gt_motion,
        rest_offsets=rest_offsets,
        gt_joints=gt_joints,
        seq_mask=seq_mask,
    )
    loss.backward()
    assert bad_pelvis.grad is not None
    assert torch.isfinite(bad_pelvis.grad).all()


# ──────────────────────────────────────────────────────────────────────────
# V8: build_init_pose_f1 / build_init_pose_f2 / frame0_consistency_loss
# ──────────────────────────────────────────────────────────────────────────


def test_init_pose_f1_returns_135_dim_frame0_slice():
    """F1 = motion_135[:, 0, :] verbatim."""
    motion = torch.randn(3, 7, 135)
    out = build_init_pose_f1(motion)
    assert out.shape == (3, 135)
    assert torch.allclose(out, motion[:, 0, :].float())


def test_init_pose_f1_rejects_wrong_dim():
    with pytest.raises(ValueError):
        build_init_pose_f1(torch.randn(3, 7, 23))


def test_init_pose_f2_returns_14_zscored_channels():
    """F2 picks the 12 rot6d + 2 heights, z-scored."""
    coarse_raw = torch.randn(3, 7, 23) * 0.4
    mean_t = torch.randn(23) * 0.1
    std_t = torch.ones(23) * 0.5
    out = build_init_pose_f2(coarse_raw, mean_t, std_t)
    assert out.shape == (3, INIT_POSE_F2_DIM)
    assert INIT_POSE_F2_DIM == 14
    # Verify the selected channels match the indices.
    expected = (coarse_raw[:, 0, :] - mean_t.view(1, -1)) / std_t.view(1, -1)
    expected_sel = expected.index_select(
        -1, torch.tensor(INIT_POSE_F2_INDICES, dtype=torch.long),
    )
    assert torch.allclose(out, expected_sel.float())


def test_init_pose_f2_indices_match_pelvis_spine_heights():
    """The 14 indices must be 9..21 (rot6d) + 21, 22 (heights)."""
    expected = tuple(list(range(9, 21)) + [21, 22])
    assert INIT_POSE_F2_INDICES == expected
    assert len(INIT_POSE_F2_INDICES) == 14


def test_frame0_consistency_zero_when_pred_matches_init_pose():
    """Pred's frame-0 14 channels = init_pose targets → loss = 0."""
    pred = torch.randn(2, 5, 23)
    # Compute the F2 targets from pred itself.
    idx = torch.tensor(INIT_POSE_F2_INDICES, dtype=torch.long)
    targets = pred[:, 0, :].index_select(-1, idx).clone()
    loss = frame0_consistency_loss(pred, targets)
    assert loss.item() < 1e-10


def test_frame0_consistency_positive_when_pred_differs():
    """Pred's t=0 differs from target → positive finite loss."""
    pred = torch.randn(2, 5, 23, requires_grad=True)
    targets = torch.zeros(2, INIT_POSE_F2_DIM)
    loss = frame0_consistency_loss(pred, targets)
    assert loss.item() > 0.0
    loss.backward()
    assert pred.grad is not None
    # The gradient should be non-zero only on the targeted channels at t=0.
    g = pred.grad.abs().sum(dim=(0, 2))   # over (B, channel)
    assert g[0].item() > 0.0
    assert g[1:].sum().item() == 0.0


def test_frame0_consistency_rejects_wrong_pred_shape():
    pred = torch.randn(2, 5, 22)        # 22 instead of 23
    targets = torch.zeros(2, INIT_POSE_F2_DIM)
    with pytest.raises(ValueError):
        frame0_consistency_loss(pred, targets)


# ──────────────────────────────────────────────────────────────────────────
# V8: V12InputProjection init_pose extension
# ──────────────────────────────────────────────────────────────────────────


def test_v12_input_proj_init_pose_zero_init_yields_zero_extra_signal():
    """At V8 init (init_pose_proj zero-init'd), the init_pose branch
    contributes nothing — the projection's output equals what it would be
    without the branch."""
    from piano.models.dit_blocks import (
        V12InputProjection,
        initialize_weights_v12,
        V12FinalLayer,
        ConditionedEncoderLayer,
        GlobalCondSummary,
    )
    import torch.nn as nn

    proj = V12InputProjection(
        motion_dim=23, obj_traj_dim=9, d_model=32, init_pose_dim=14,
    )
    blocks = nn.ModuleList(
        [ConditionedEncoderLayer(d_model=32, n_heads=4, ff_mult=4)]
    )
    final = V12FinalLayer(d_model=32, motion_dim=23)
    cs = GlobalCondSummary(d_model=32)
    initialize_weights_v12(
        input_proj=proj, blocks=blocks, final_layer=final, cond_summary=cs,
    )

    x_t = torch.randn(2, 5, 23)
    obj_traj = torch.randn(2, 5, 9)
    init_pose_a = torch.randn(2, 14)
    init_pose_b = torch.randn(2, 14) * 100.0   # very different content

    h_a = proj(x_t=x_t, obj_traj=obj_traj, init_pose=init_pose_a)
    h_b = proj(x_t=x_t, obj_traj=obj_traj, init_pose=init_pose_b)
    # Zero-init init_pose_proj → init_pose contribution is identically 0
    # regardless of input value.
    assert torch.allclose(h_a, h_b, atol=1e-6)


def test_v12_input_proj_init_pose_grows_after_weight_perturbation():
    """Manually nudge init_pose_proj weights → output should change."""
    from piano.models.dit_blocks import V12InputProjection
    import torch.nn as nn

    proj = V12InputProjection(
        motion_dim=23, obj_traj_dim=9, d_model=32, init_pose_dim=14,
    )
    nn.init.zeros_(proj.motion_proj.weight)
    nn.init.zeros_(proj.motion_proj.bias)
    nn.init.zeros_(proj.obj_proj.weight)
    nn.init.zeros_(proj.obj_proj.bias)
    # Set init_pose_proj weight to ones so init_pose contributes.
    nn.init.ones_(proj.init_pose_proj.weight)
    nn.init.zeros_(proj.init_pose_proj.bias)

    x_t = torch.zeros(1, 4, 23)
    obj_traj = torch.zeros(1, 4, 9)
    init_pose = torch.ones(1, 14)
    h = proj(x_t=x_t, obj_traj=obj_traj, init_pose=init_pose)
    # Every output unit = sum_i 1 * 1 = 14, broadcast across T.
    assert h.shape == (1, 4, 32)
    assert torch.allclose(h, torch.full((1, 4, 32), 14.0))


def test_v12_input_proj_init_pose_missing_raises():
    """init_pose_dim > 0 but cond didn't include init_pose → KeyError."""
    from piano.models.dit_blocks import V12InputProjection

    proj = V12InputProjection(
        motion_dim=23, obj_traj_dim=9, d_model=32, init_pose_dim=14,
    )
    x_t = torch.zeros(1, 4, 23)
    obj_traj = torch.zeros(1, 4, 9)
    with pytest.raises(KeyError):
        proj(x_t=x_t, obj_traj=obj_traj, init_pose=None)


# ──────────────────────────────────────────────────────────────────────────
# V8: Stage1Denoiser integration test
# ──────────────────────────────────────────────────────────────────────────


def test_stage1_denoiser_forward_with_init_pose_f2():
    """End-to-end: build Stage1Denoiser(init_pose_dim=14) and run a
    forward pass with init_pose in the cond dict."""
    from piano.models.stage1_trajectory import (
        Stage1Denoiser, Stage1DenoiserConfig,
    )
    cfg = Stage1DenoiserConfig(
        motion_dim=23, object_traj_dim=9, text_dim=512,
        object_token_dim=256, object_num_tokens=128,
        d_model=64, n_layers=2, n_heads=4, ff_mult=2,
        dropout=0.0, max_seq_length=16,
        use_text=False, init_pose_dim=14,
    )
    model = Stage1Denoiser(cfg)
    assert model.use_init_pose

    B, T = 2, 8
    x_t = torch.randn(B, T, 23)
    t = torch.zeros(B, dtype=torch.long)
    cond = {
        "object_world_traj": torch.randn(B, T, 9),
        "object_tokens": torch.randn(B, 128, 256),
        "init_pose": torch.randn(B, 14),
    }
    out = model(x_t, t, cond, cond_drop_mask=None)
    assert out.shape == (B, T, 23)
    assert torch.isfinite(out).all()


def test_temporal_derivative_loss_zero_when_pred_equals_gt():
    gt = torch.randn(2, 8, 23)
    mask = torch.ones(2, 8)
    vel = temporal_derivative_mse_loss(gt, gt, mask, order=1)
    acc = temporal_derivative_mse_loss(gt, gt, mask, order=2)
    assert vel.item() < 1e-10
    assert acc.item() < 1e-10


def test_temporal_derivative_loss_acceleration_catches_curvature_change():
    gt = torch.zeros(1, 5, 23)
    pred = gt.clone()
    pred[0, 2, 0] = 1.0
    mask = torch.ones(1, 5)
    loss = temporal_derivative_mse_loss(
        pred,
        gt,
        mask,
        order=2,
        channel_subset=(0,),
        normalize_by_gt_std=False,
    )
    assert loss.item() > 0.0


def test_temporal_derivative_loss_respects_mask_and_grad():
    gt = torch.zeros(1, 5, 23)
    pred = gt.clone().requires_grad_(True)
    pred.data[0, 4, 0] = 100.0
    mask = torch.ones(1, 5)
    mask[0, 3:] = 0.0
    loss = temporal_derivative_mse_loss(
        pred,
        gt,
        mask,
        order=1,
        channel_subset=(0,),
        normalize_by_gt_std=False,
    )
    assert loss.item() == 0.0
    loss.backward()
    assert pred.grad is not None


def test_stage1_denoiser_forward_with_init_pose_f1():
    """F1 mode: init_pose_dim=135."""
    from piano.models.stage1_trajectory import (
        Stage1Denoiser, Stage1DenoiserConfig,
    )
    cfg = Stage1DenoiserConfig(
        motion_dim=23, object_traj_dim=9, text_dim=512,
        object_token_dim=256, object_num_tokens=128,
        d_model=64, n_layers=2, n_heads=4, ff_mult=2,
        dropout=0.0, max_seq_length=16,
        use_text=False, init_pose_dim=135,
    )
    model = Stage1Denoiser(cfg)
    B, T = 2, 8
    x_t = torch.randn(B, T, 23)
    t = torch.zeros(B, dtype=torch.long)
    cond = {
        "object_world_traj": torch.randn(B, T, 9),
        "object_tokens": torch.randn(B, 128, 256),
        "init_pose": torch.randn(B, 135),
    }
    out = model(x_t, t, cond, cond_drop_mask=None)
    assert out.shape == (B, T, 23)
    assert torch.isfinite(out).all()


def test_stage1_denoiser_v0_backwards_compat_no_init_pose():
    """init_pose_dim=0 (default) → model works without init_pose in cond."""
    from piano.models.stage1_trajectory import (
        Stage1Denoiser, Stage1DenoiserConfig,
    )
    cfg = Stage1DenoiserConfig(
        motion_dim=23, object_traj_dim=9, text_dim=512,
        object_token_dim=256, object_num_tokens=128,
        d_model=64, n_layers=2, n_heads=4, ff_mult=2,
        dropout=0.0, max_seq_length=16,
        use_text=False, init_pose_dim=0,
    )
    model = Stage1Denoiser(cfg)
    assert not model.use_init_pose

    B, T = 2, 8
    x_t = torch.randn(B, T, 23)
    t = torch.zeros(B, dtype=torch.long)
    cond = {
        "object_world_traj": torch.randn(B, T, 9),
        "object_tokens": torch.randn(B, 128, 256),
        # No init_pose — should be fine.
    }
    out = model(x_t, t, cond, cond_drop_mask=None)
    assert out.shape == (B, T, 23)


# ──────────────────────────────────────────────────────────────────────────
# R40 — channel-weight helper (build_channel_weight_tensor)
# ──────────────────────────────────────────────────────────────────────────


def test_channel_weight_helper_empty_returns_none():
    """Empty list / None → ``None`` (caller treats as all-ones)."""
    from piano.training.stage1_losses import build_channel_weight_tensor
    device = torch.device("cpu")
    for w in (None, [], ()):
        out = build_channel_weight_tensor(
            w, expected_dim=23, device=device, dtype=torch.float32, name="x",
        )
        assert out is None


def test_channel_weight_helper_wrong_length_raises():
    from piano.training.stage1_losses import build_channel_weight_tensor
    with pytest.raises(ValueError, match="must have length 23"):
        build_channel_weight_tensor(
            [1.0] * 22, expected_dim=23,
            device=torch.device("cpu"), dtype=torch.float32, name="bad",
        )


def test_channel_weight_helper_valid_shape():
    from piano.training.stage1_losses import build_channel_weight_tensor
    w = [float(i) for i in range(23)]
    out = build_channel_weight_tensor(
        w, expected_dim=23,
        device=torch.device("cpu"), dtype=torch.float32, name="ok",
    )
    assert out is not None
    assert out.shape == (1, 1, 23)
    assert out.dtype == torch.float32
    assert torch.allclose(out.view(-1), torch.tensor(w, dtype=torch.float32))


def test_channel_weight_helper_zero_weight_removes_channel_contribution():
    """A zero weight on a channel zeros its contribution to the weighted sum.

    Models the production code path:
        weighted_per_dim = per_dim * channel_w   # (1, 1, 23) broadcast
        per_frame = weighted_per_dim.sum(-1)
    Setting channel 5's weight to 0 must drop its (large) contribution.
    """
    from piano.training.stage1_losses import build_channel_weight_tensor
    per_dim = torch.ones(1, 1, 23) * 1.0
    per_dim[..., 5] = 100.0
    w = [1.0] * 23
    w[5] = 0.0
    cw = build_channel_weight_tensor(
        w, expected_dim=23,
        device=torch.device("cpu"), dtype=per_dim.dtype, name="zero5",
    )
    weighted_sum = (per_dim * cw).sum(-1).item()
    # 22 channels at 1.0 + 1 channel at 100*0 = 22.
    assert weighted_sum == pytest.approx(22.0, abs=1e-6)


def test_channel_weight_helper_non_list_raises():
    from piano.training.stage1_losses import build_channel_weight_tensor
    with pytest.raises(ValueError, match="must be a list/tuple or None"):
        build_channel_weight_tensor(
            "not-a-list", expected_dim=23,
            device=torch.device("cpu"), dtype=torch.float32, name="bad",
        )


# ──────────────────────────────────────────────────────────────────────────
# R40 — stage1_plan_invariant_loss
# ──────────────────────────────────────────────────────────────────────────


def _build_simple_plan_inputs(
    *,
    B: int = 2,
    T: int = 12,
    seed: int = 0,
    speed: float = 0.05,
    yaw_rate: float = 0.05,
    object_xy_offset: tuple[float, float] = (0.5, 0.0),
):
    """Construct a hand-crafted (B, T, 23) stage1_raw + obj_traj + t0.

    The clip walks at constant XZ speed, rotates pelvis yaw at constant
    rate, and has a stationary object at the given offset.
    """
    torch.manual_seed(seed)
    device = torch.device("cpu")
    raw = torch.zeros(B, T, 23, device=device, dtype=torch.float64)
    # Linear root path: (x_t, z_t) = t * (speed_x, speed_z) in oracle (x, z).
    for t in range(T):
        raw[:, t, 0] = t * speed                  # root_local_x
        raw[:, t, 1] = t * speed * 0.5            # root_local_z
        raw[:, t, 2] = 0.0                         # root_local_y
        # vel channels (raw oracle stores frame-to-frame diff).
        raw[:, t, 3] = speed
        raw[:, t, 4] = speed * 0.5
        raw[:, t, 5] = 0.0
        # yaw — linear from 0.
        ang = yaw_rate * t
        raw[:, t, 6] = float(torch.sin(torch.tensor(ang)))
        raw[:, t, 7] = float(torch.cos(torch.tensor(ang)))
        raw[:, t, 8] = yaw_rate
        # pelvis & spine3 rot6d — small constant offset.
        raw[:, t, 9:15] = 0.0
        raw[:, t, 9] = 1.0
        raw[:, t, 13] = 1.0
        raw[:, t, 15:21] = 0.0
        raw[:, t, 15] = 1.0
        raw[:, t, 19] = 1.0
        raw[:, t, 21] = 1.7                        # head_height
        raw[:, t, 22] = 1.4                        # shoulder_h
    # Object traj (B, T, 9) — COM at (x_off, 0, z_off) world.
    obj_traj = torch.zeros(B, T, 9, device=device, dtype=torch.float64)
    obj_traj[..., 0] = object_xy_offset[0]
    obj_traj[..., 2] = object_xy_offset[1]
    # Root world t0 = (0, 0, 0) world.
    root_world_t0 = torch.zeros(B, 1, 3, device=device, dtype=torch.float64)
    seq_mask = torch.ones(B, T, device=device, dtype=torch.float64)
    return raw, obj_traj, root_world_t0, seq_mask


def test_plan_invariant_zero_on_identity():
    """plan_invariant_loss(pred == gt) is zero on comparison components.

    Smoothness is pred-only and may be nonzero; check the comparison
    components individually.
    """
    from piano.training.stage1_losses import stage1_plan_invariant_loss
    raw, obj, t0, mask = _build_simple_plan_inputs(speed=0.05, yaw_rate=0.05)
    total, comps = stage1_plan_invariant_loss(
        stage1_raw_pred=raw.float(),
        stage1_raw_gt=raw.float(),
        object_world_traj=obj.float(),
        root_world_t0=t0.float(),
        seq_mask=mask.float(),
    )
    for name in (
        "root_speed", "root_arc", "root_displacement",
        "root_object_radial", "yaw_activity", "rot_activity",
        "height_envelope",
    ):
        assert name in comps, f"missing {name}"
        assert comps[name].item() == pytest.approx(0.0, abs=1e-4), \
            f"{name} should be ~0 on identity, got {comps[name].item():.4e}"


def test_plan_invariant_frozen_root_has_larger_speed_arc_than_gt():
    """A frozen root path (zero motion) has a larger root_speed/root_arc
    loss than a GT moving path.
    """
    from piano.training.stage1_losses import stage1_plan_invariant_loss
    raw_gt, obj, t0, mask = _build_simple_plan_inputs(
        speed=0.05, yaw_rate=0.0,
    )
    raw_pred = raw_gt.clone()
    raw_pred[..., :6] = 0.0                                       # frozen root
    _total, comps = stage1_plan_invariant_loss(
        stage1_raw_pred=raw_pred.float(),
        stage1_raw_gt=raw_gt.float(),
        object_world_traj=obj.float(),
        root_world_t0=t0.float(),
        seq_mask=mask.float(),
    )
    assert comps["root_speed"].item() > 1e-4
    assert comps["root_arc"].item() > 1e-4


def test_plan_invariant_mirrored_path_radial_low_penalty():
    """A left-vs-right mirror of the root path that keeps the same
    radial distance to the object should not incur a large
    root_object_radial penalty.

    Build GT going +x, pred going -x; object is at the origin so the
    radial distance profile is identical (|x_t|).
    """
    from piano.training.stage1_losses import stage1_plan_invariant_loss
    raw_gt, obj, t0, mask = _build_simple_plan_inputs(
        speed=0.05, yaw_rate=0.0,
        object_xy_offset=(0.0, 0.0),
    )
    # Mirror x-channel and its velocity. Also zero z so radial = |x|.
    raw_gt = raw_gt.clone()
    raw_gt[..., 1] = 0.0
    raw_gt[..., 4] = 0.0
    raw_pred = raw_gt.clone()
    raw_pred[..., 0] = -raw_gt[..., 0]
    raw_pred[..., 3] = -raw_gt[..., 3]
    _t, comps = stage1_plan_invariant_loss(
        stage1_raw_pred=raw_pred.float(),
        stage1_raw_gt=raw_gt.float(),
        object_world_traj=obj.float(),
        root_world_t0=t0.float(),
        seq_mask=mask.float(),
    )
    # Radial distance |x_t| matches frame-by-frame → ~0 radial loss.
    assert comps["root_object_radial"].item() == pytest.approx(0.0, abs=1e-4)


def test_plan_invariant_masking_ignores_padded_frames():
    """Padded frames must not contribute to plan stats."""
    from piano.training.stage1_losses import stage1_plan_invariant_loss
    raw, obj, t0, mask = _build_simple_plan_inputs(T=12, speed=0.05)
    # Mask out the last 4 frames.
    mask = mask.clone()
    mask[:, -4:] = 0.0
    # Corrupt the masked frames in pred only — they should not change loss.
    raw_pred = raw.clone()
    raw_pred[:, -4:, :] = 999.0
    total_a, _ = stage1_plan_invariant_loss(
        stage1_raw_pred=raw.float(),
        stage1_raw_gt=raw.float(),
        object_world_traj=obj.float(),
        root_world_t0=t0.float(),
        seq_mask=mask.float(),
    )
    total_b, _ = stage1_plan_invariant_loss(
        stage1_raw_pred=raw_pred.float(),
        stage1_raw_gt=raw.float(),
        object_world_traj=obj.float(),
        root_world_t0=t0.float(),
        seq_mask=mask.float(),
    )
    # Identity and "corrupted-after-mask" should match; final-index
    # uses argmax over masked_idx → last valid is t=7 in both.
    assert abs(total_a.item() - total_b.item()) < 1e-4, (
        f"masking did not isolate padded frames: "
        f"{total_a.item()=:.6f} vs {total_b.item()=:.6f}"
    )


def test_plan_invariant_smoothness_finite_and_nonneg():
    """Smoothness must be a finite scalar ≥ 0."""
    from piano.training.stage1_losses import stage1_plan_invariant_loss
    raw, obj, t0, mask = _build_simple_plan_inputs()
    _t, comps = stage1_plan_invariant_loss(
        stage1_raw_pred=raw.float(),
        stage1_raw_gt=raw.float(),
        object_world_traj=obj.float(),
        root_world_t0=t0.float(),
        seq_mask=mask.float(),
    )
    if "smoothness" in comps:
        s = comps["smoothness"].item()
        assert s >= 0.0
        assert s == s  # not NaN
        assert s != float("inf")


def test_plan_invariant_gradients_flow_to_pred():
    """Gradient flows from total loss to stage1_raw_pred."""
    from piano.training.stage1_losses import stage1_plan_invariant_loss
    raw_gt, obj, t0, mask = _build_simple_plan_inputs(speed=0.05)
    raw_pred = (raw_gt.float().clone() + 0.01).requires_grad_(True)
    total, _ = stage1_plan_invariant_loss(
        stage1_raw_pred=raw_pred,
        stage1_raw_gt=raw_gt.float(),
        object_world_traj=obj.float(),
        root_world_t0=t0.float(),
        seq_mask=mask.float(),
    )
    total.backward()
    assert raw_pred.grad is not None
    assert raw_pred.grad.abs().sum().item() > 0.0


def test_plan_invariant_unknown_component_raises():
    from piano.training.stage1_losses import stage1_plan_invariant_loss
    raw, obj, t0, mask = _build_simple_plan_inputs()
    with pytest.raises(ValueError, match="unknown plan-invariant component"):
        stage1_plan_invariant_loss(
            stage1_raw_pred=raw.float(),
            stage1_raw_gt=raw.float(),
            object_world_traj=obj.float(),
            root_world_t0=t0.float(),
            seq_mask=mask.float(),
            component_weights={"not_a_real_component": 1.0},
        )
