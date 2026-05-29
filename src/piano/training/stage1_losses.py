"""Stage-1 ablation loss helpers.

The R31 V0 baseline (MSE + vel + min-SNR-γ) trained but its 23-D output
caused catastrophic Stage-2 PB1 downstream regression: drift_max
+11 cm, body_action direction_cos -0.5 on every non-pelvis joint.
The R31 diagnosis traced the regression to:

  - rot6d channels [9:21] not on the SO(3) manifold (Gram-Schmidt
    only gives a valid R when the input 6 numbers are well-behaved).
  - FK-derived height channels [21:22] inconsistent with the
    rot6d channels — the model treats them as independent
    regression targets.

This module collects the four candidate fix losses described in
``analyses/2026-05-30_round31_v2_stage1_loss_ablation.md``:

  L1 : rot6d_ortho_loss          — penalize ||a1||≠1, ||a2||≠1, <a1,a2>≠0
  L2 : fk_pelvis_spine_pos_loss  — FK 22 joints with Stage-1 rot6d for
                                    pelvis (joint 0) + spine3 (joint 9),
                                    GT rot6d for the rest; MSE on
                                    target joints (head, neck, shoulders).
  L3 : fk_height_consistency_loss — Stage-1's channel [21] head_height
                                    and [22] shoulder_center_h must
                                    match the FK output Y of those joints.
  L4 : kinematic_self_consistency_loss
                                  — diff(root_local) ≈ vel; diff(yaw) ≈ yaw_vel.

All helpers operate on RAW (un-z-scored) Stage-1 outputs. The trainer
is responsible for un-normalizing before calling these.
"""
from __future__ import annotations

import torch
from torch import Tensor

from piano.training.smpl_kinematics import (
    fk_from_global_rotations,
    rotation_6d_to_matrix,
)


# ─── Channel layout (23-D) — must match stage1_coarse_oracle.py:191-214 ────
# [ 0: 3]  root_local_x, root_local_z, root_local_y
# [ 3: 6]  vel_x, vel_z, vel_y
# [ 6: 9]  yaw_sin, yaw_cos, yaw_vel
# [ 9:15]  pelvis_rot6d
# [15:21]  spine3_rot6d
# [21:22]  head_height
# [22:23]  shoulder_center_h
CH_ROOT_LOCAL = slice(0, 3)
CH_VEL = slice(3, 6)
CH_YAW_SIN = 6
CH_YAW_COS = 7
CH_YAW_VEL = 8
CH_PELVIS_ROT6D = slice(9, 15)
CH_SPINE3_ROT6D = slice(15, 21)
CH_HEAD_HEIGHT = 21
CH_SHOULDER_H = 22

# SMPL-22 joint indices (mirrors stage1_coarse_oracle.py:37-42).
J_PELVIS = 0
J_SPINE3 = 9
J_HEAD = 15
J_L_SHOULDER = 16
J_R_SHOULDER = 17
J_NECK = 12


# ──────────────────────────────────────────────────────────────────────────
# L1: rot6d orthogonality
# ──────────────────────────────────────────────────────────────────────────


def rot6d_ortho_loss(rot6d: Tensor, mask: Tensor | None = None) -> Tensor:
    """Penalize the rot6d 6-vector for not being a valid (a1, a2) on
    the SO(3) Stiefel manifold:

        ||a1|| = 1, ||a2|| = 1, <a1, a2> = 0

    ``rot6d`` shape: (..., 6).
    ``mask``: optional (...,) tensor of 0/1 weights (e.g. ``seq_mask``).
    Returns scalar mean violation.

    Note: ``rotation_6d_to_matrix`` already runs Gram-Schmidt internally,
    so the output IS a valid rotation. But the *gradient* into the
    network's 6 raw output channels only carries the projection direction,
    not the manifold violation magnitude. This explicit loss adds a
    surface penalty that drives the raw 6 numbers toward the manifold.
    """
    a1 = rot6d[..., :3]
    a2 = rot6d[..., 3:]
    norm_a1 = a1.norm(dim=-1)                 # (...,)
    norm_a2 = a2.norm(dim=-1)                 # (...,)
    dot = (a1 * a2).sum(-1)                   # (...,)
    per_elem = (
        (norm_a1 - 1.0).pow(2)
        + (norm_a2 - 1.0).pow(2)
        + dot.pow(2)
    )                                          # (...,)
    if mask is not None:
        return (per_elem * mask).sum() / mask.sum().clamp_min(1.0)
    return per_elem.mean()


# ──────────────────────────────────────────────────────────────────────────
# L2: FK position loss on pelvis + spine3 rotations
# ──────────────────────────────────────────────────────────────────────────


def fk_pelvis_spine_pos_loss(
    *,
    pelvis_rot6d_pred: Tensor,    # (B, T, 6) — predicted (raw, un-z-scored)
    spine3_rot6d_pred: Tensor,    # (B, T, 6) — predicted
    root_world_pred: Tensor,      # (B, T, 3) — predicted (raw)
    gt_motion_135: Tensor,        # (B, T, 135) — ground truth motion
    rest_offsets: Tensor,         # (B, 22, 3)
    gt_joints: Tensor,            # (B, T, 22, 3) world frame
    seq_mask: Tensor,             # (B, T)
    target_joint_indices: tuple[int, ...] = (J_NECK, J_HEAD, J_L_SHOULDER, J_R_SHOULDER),
) -> Tensor:
    """Run FK with Stage-1's predicted pelvis + spine3 rotations and
    GT rotations for the other 20 joints. MSE on the target joints'
    world-frame positions vs GT.

    Why partial: Stage-1 only predicts rot6d for pelvis + spine3.
    Replacing only those two in the otherwise GT rotation chain
    isolates the effect of Stage-1's predictions. Target joints are
    chosen to be downstream of pelvis/spine3 on the kinematic tree
    (neck/head/shoulders) so the loss is sensitive to those two
    rotations.

    Returns scalar in m².
    """
    B, T, _ = gt_motion_135.shape

    # Extract 22 rot6d from GT motion_135.
    gt_rot6d = gt_motion_135[..., :132].reshape(B, T, 22, 6).float()
    gt_rot_mat = rotation_6d_to_matrix(gt_rot6d)                  # (B, T, 22, 3, 3)

    # Substitute pelvis + spine3 with Stage-1 predictions.
    pred_pelvis_mat = rotation_6d_to_matrix(pelvis_rot6d_pred)    # (B, T, 3, 3)
    pred_spine3_mat = rotation_6d_to_matrix(spine3_rot6d_pred)    # (B, T, 3, 3)
    rot_mat = gt_rot_mat.clone()
    rot_mat[:, :, J_PELVIS] = pred_pelvis_mat
    rot_mat[:, :, J_SPINE3] = pred_spine3_mat

    # FK chain.
    rest_per_frame = rest_offsets.unsqueeze(1).expand(B, T, 22, 3).float()
    joints_pred = fk_from_global_rotations(
        rot_mat, rest_per_frame, root_world_pred.float(),
    )                                                              # (B, T, 22, 3)

    # MSE on target joints, masked to valid frames.
    idx = torch.tensor(target_joint_indices, device=joints_pred.device)
    pred_sel = joints_pred.index_select(2, idx)                    # (B, T, K, 3)
    gt_sel = gt_joints.float().index_select(2, idx)                # (B, T, K, 3)
    err = (pred_sel - gt_sel).pow(2).sum(-1)                       # (B, T, K)
    err = err.mean(-1)                                              # (B, T)
    return (err * seq_mask).sum() / seq_mask.sum().clamp_min(1.0)


# ──────────────────────────────────────────────────────────────────────────
# L3: FK-derived height consistency
# ──────────────────────────────────────────────────────────────────────────


def fk_height_consistency_loss(
    *,
    head_height_pred: Tensor,     # (B, T) — predicted [21] channel (raw m)
    shoulder_h_pred: Tensor,      # (B, T) — predicted [22] channel (raw m)
    pelvis_rot6d_pred: Tensor,    # (B, T, 6)
    spine3_rot6d_pred: Tensor,    # (B, T, 6)
    root_world_pred: Tensor,      # (B, T, 3)
    gt_motion_135: Tensor,        # (B, T, 135)
    rest_offsets: Tensor,         # (B, 22, 3)
    seq_mask: Tensor,             # (B, T)
) -> Tensor:
    """The [21] head_height and [22] shoulder_center_h channels are
    derived in oracle from FK. If Stage-1 predicts these independently
    of its rot6d channels, the model can satisfy MSE on each in
    isolation but the result is geometrically inconsistent (this is
    exactly the R31 failure mode).

    Force them to agree: run FK with the predicted rotations and
    take the Y-coordinate of head and shoulder joints; MSE against
    the predicted scalar channels.

    Returns scalar in m².
    """
    B, T, _ = gt_motion_135.shape

    gt_rot6d = gt_motion_135[..., :132].reshape(B, T, 22, 6).float()
    gt_rot_mat = rotation_6d_to_matrix(gt_rot6d)
    pred_pelvis_mat = rotation_6d_to_matrix(pelvis_rot6d_pred)
    pred_spine3_mat = rotation_6d_to_matrix(spine3_rot6d_pred)
    rot_mat = gt_rot_mat.clone()
    rot_mat[:, :, J_PELVIS] = pred_pelvis_mat
    rot_mat[:, :, J_SPINE3] = pred_spine3_mat

    rest_per_frame = rest_offsets.unsqueeze(1).expand(B, T, 22, 3).float()
    joints_pred = fk_from_global_rotations(
        rot_mat, rest_per_frame, root_world_pred.float(),
    )                                                              # (B, T, 22, 3)

    fk_head_y = joints_pred[..., J_HEAD, 1]                        # (B, T)
    fk_shoulder_y = (
        joints_pred[..., J_L_SHOULDER, 1]
        + joints_pred[..., J_R_SHOULDER, 1]
    ) * 0.5                                                         # (B, T)

    err_head = (head_height_pred - fk_head_y).pow(2)               # (B, T)
    err_sh = (shoulder_h_pred - fk_shoulder_y).pow(2)              # (B, T)
    return (
        ((err_head + err_sh) * seq_mask).sum()
        / seq_mask.sum().clamp_min(1.0)
    )


# ──────────────────────────────────────────────────────────────────────────
# L4: kinematic self-consistency (intra-23-D)
# ──────────────────────────────────────────────────────────────────────────


def kinematic_self_consistency_loss(
    stage1_raw: Tensor,           # (B, T, 23) — raw (un-z-scored) prediction
    seq_mask: Tensor,             # (B, T)
) -> Tensor:
    """The 23-D layout has redundant channels that should agree by
    construction:

      - root_local[t] - root_local[t-1] ≈ vel[t]    (channels [0:3] and [3:6])
      - yaw_unwrap(yaw_sin, yaw_cos)[t] - yaw_unwrap[t-1] ≈ yaw_vel[t]
        (channels [6:8] and [8])

    Stage-1's MSE on raw channels does not enforce these. Predicting
    inconsistent values is the canonical failure for diffusion models
    on multi-channel targets: high noise steps mix the channels and
    the model trivially achieves MSE without learning the consistency.

    Returns: scalar (m² + rad²) — small magnitude, expected weight 0.05.
    """
    B, T, _ = stage1_raw.shape
    if T < 2:
        return stage1_raw.sum() * 0.0

    valid = seq_mask[:, 1:] * seq_mask[:, :-1]                     # (B, T-1)

    # Translation: diff(root_local) ≈ vel
    rl = stage1_raw[..., CH_ROOT_LOCAL]                            # (B, T, 3)
    vel = stage1_raw[..., CH_VEL]                                  # (B, T, 3)
    diff_rl = rl[:, 1:] - rl[:, :-1]                               # (B, T-1, 3)
    # vel[t] is the velocity from t-1 to t, so compare diff_rl with vel[:, 1:].
    err_trans = (diff_rl - vel[:, 1:]).pow(2).sum(-1)              # (B, T-1)

    # Yaw: diff(yaw_unwrapped) ≈ yaw_vel
    # We don't unwrap here (avoids modulo branch); instead use atan2 on
    # (sin, cos) → angle ∈ [-π, π], wrap the diff to [-π, π].
    yaw_sin = stage1_raw[..., CH_YAW_SIN]                          # (B, T)
    yaw_cos = stage1_raw[..., CH_YAW_COS]                          # (B, T)
    yaw_vel = stage1_raw[..., CH_YAW_VEL]                          # (B, T)
    yaw_ang = torch.atan2(yaw_sin, yaw_cos)                        # (B, T)
    diff_y = yaw_ang[:, 1:] - yaw_ang[:, :-1]
    diff_y = (diff_y + 3.14159265) % (2 * 3.14159265) - 3.14159265
    err_yaw = (diff_y - yaw_vel[:, 1:]).pow(2)                     # (B, T-1)

    total = err_trans + err_yaw                                     # (B, T-1)
    return (total * valid).sum() / valid.sum().clamp_min(1.0)
