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
# R31 V8 — wrist/elbow indices for extended FK supervision.
J_L_ELBOW = 18
J_R_ELBOW = 19
J_L_WRIST = 20
J_R_WRIST = 21
# Contact-state layout (5-part) — index of wrist contact channels.
# (left_hand=0, right_hand=1, left_foot=2, right_foot=3, pelvis=4)
CONTACT_IDX_LEFT_HAND = 0
CONTACT_IDX_RIGHT_HAND = 1


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


# ──────────────────────────────────────────────────────────────────────────
# R31 V7 anti-mode-collapse losses
# ──────────────────────────────────────────────────────────────────────────
#
# Phase 1 dynamic-info audit
# (analyses/round31_phase1_dyn_audit_20260530_043948/audit_report.md)
# showed Stage-1 V0 collapses to the conditional mean: std ratio
# 0.24–0.46 across every channel group, PSD high-band 0.00–0.16, yaw_vel
# RMS = 2 % of GT. This is the canonical signature of MSE training under
# multimodal-conditional generation: outputting the mean minimises L2 in
# expectation when modes exist. V2's loss-design ablation didn't move
# the needle because none of V1–V5 attacked the std-collapse axis.
#
# Inspired by what got Stage-2 PB1 out of the same problem
# (analyses/round29_*; train_anchordiff.py:485–1166):
#
#   PB1's anti-collapse stack is:
#     (1) FK-space L_pos with weight 5.0 — heavy physical-space MSE,
#     (2) `stable_local_speed_moment` (weight 0.02) — direct (mean,std)
#         matching at velocity magnitudes, the only loss that *literally*
#         penalises std collapse.
#     (3) G1 aggregate-statistic gait losses (transition_rate /
#         duty_cycle / both_state_match) — match cross-segment statistics
#         that are mode-invariant.
#
# Two of these mechanisms are searchable in Stage-1's 23-D output:
#
#   V7-A `channel_moment_match_loss`: per-channel-group (mean, std)
#       matching of finite-difference magnitudes. Directly penalises
#       under-dispersion at the velocity level.
#
#   V7-B `yaw_aggregate_match_loss`: cross-clip aggregate statistics of
#       yaw dynamics. Match (yaw transition rate, total yaw range) without
#       fixing per-frame yaw values, so mode multiplicity (CW vs CCW
#       rotations) doesn't get averaged into a frozen-yaw mode.
#
# Both operate on RAW (un-z-scored) Stage-1 outputs.


def channel_moment_match_loss(
    stage1_raw_pred: Tensor,        # (B, T, 23) — raw (un-z-scored) prediction
    stage1_raw_gt: Tensor,          # (B, T, 23) — raw GT
    seq_mask: Tensor,               # (B, T)
    *,
    velocity_match: bool = True,
    value_match: bool = False,
    channel_subset: tuple[int, ...] | None = None,
    normalize_by_gt_std: bool = True,
) -> Tensor:
    """V7-A — per-channel (mean, std) matching, modelled on PB1's
    ``stable_local_speed_moment`` (train_anchordiff.py:1106–1121).

    For each selected channel (default: all 23), compute the per-batch
    mean and std of either the value or its 1-frame finite difference;
    penalise the gap to GT's matching moment. This directly attacks
    distribution-collapse: if pred std drops to 0, the loss explodes.

    Parameters
    ----------
    stage1_raw_pred, stage1_raw_gt : (B, T, 23) raw-space tensors.
    seq_mask : (B, T) — float mask of valid frames.
    velocity_match : add a (mean, std) match on 1-frame finite diff.
        Catches under-articulation directly.
    value_match : add a (mean, std) match on the raw value.
        Catches mean-shift / std-collapse on absolute values.
    channel_subset : tuple of channel indices to score. None = all 23.
    normalize_by_gt_std : when True (default), divide each channel's
        contribution by (GT std)² + ε so channels with different physical
        scales (m vs rad vs dimensionless) contribute comparably. Without
        this, big-magnitude channels (e.g. heights ~ 1.4 m) dominate
        zero-near-zero channels (e.g. vel_y ~ 0.005 m/frame).

    Returns
    -------
    Scalar loss. Magnitude is dimensionless when normalized (channels
    enter on equal footing), else raw² (m² / rad² / dimensionless rot6d²).
    Suggested weight: ~0.5 for velocity-only, ~0.05 for the value variant
    if used. Compatible with the bf16 + accelerate trainer.
    """
    if not velocity_match and not value_match:
        return stage1_raw_pred.sum() * 0.0

    B, T, D = stage1_raw_pred.shape
    if channel_subset is None:
        idx = torch.arange(D, device=stage1_raw_pred.device)
    else:
        idx = torch.tensor(channel_subset, device=stage1_raw_pred.device,
                           dtype=torch.long)

    pred = stage1_raw_pred.index_select(-1, idx)        # (B, T, K)
    gt = stage1_raw_gt.index_select(-1, idx)            # (B, T, K)
    mask = seq_mask                                      # (B, T)

    loss = stage1_raw_pred.new_zeros(())

    def _moment_pair_per_channel(
        p_btk: Tensor, g_btk: Tensor, w_bt: Tensor,
    ) -> Tensor:
        """Compute weighted mean and std of (p_btk, g_btk) per channel
        under mask w_bt (broadcast to the (B, T, K) shape), and return
        the scalar mean over channels of
            (mean_pred − mean_gt)² + (std_pred − std_gt)²
        optionally divided by (gt_std² + eps) per channel for scale
        invariance across heterogeneous channels.

        p_btk, g_btk : (B, T, K) raw tensors.
        w_bt         : (B, T)    float mask.
        """
        # Flatten the (B, T) dims; reduce over them per channel.
        BT = p_btk.shape[0] * p_btk.shape[1]
        p_flat = p_btk.reshape(BT, -1)                     # (BT, K)
        g_flat = g_btk.reshape(BT, -1)                     # (BT, K)
        w_flat = w_bt.reshape(BT).float()                  # (BT,)
        w_sum = w_flat.sum().clamp_min(1.0)
        w_col = w_flat.unsqueeze(-1)                       # (BT, 1)
        mean_p = (p_flat * w_col).sum(0) / w_sum           # (K,)
        mean_g = (g_flat * w_col).sum(0) / w_sum
        var_p = ((p_flat - mean_p).pow(2) * w_col).sum(0) / w_sum
        var_g = ((g_flat - mean_g).pow(2) * w_col).sum(0) / w_sum
        std_p = var_p.clamp_min(1e-12).sqrt()
        std_g = var_g.clamp_min(1e-12).sqrt()
        per_ch = (mean_p - mean_g).pow(2) + (std_p - std_g).pow(2)  # (K,)
        if normalize_by_gt_std:
            # Scale-invariant: divide by gt std² so each channel contributes
            # ~1.0 when pred is fully collapsed (std_p ≈ 0, mean off by gt_std).
            scale_sq = var_g.clamp_min(1e-6)
            per_ch = per_ch / scale_sq
        return per_ch.mean()

    if value_match:
        # (mean, std) matching on raw values, vectorized across channels.
        loss = loss + _moment_pair_per_channel(pred, gt, mask)

    if velocity_match and T >= 2:
        # (mean, std) matching on 1-frame finite-difference magnitudes,
        # vectorized across channels. We use |Δx| (not Δx²) so the moment
        # matches a speed-like quantity, mirroring PB1's
        # stable_local_speed_moment which works on ||vel||.
        pair_mask = mask[:, 1:] * mask[:, :-1]           # (B, T-1)
        d_pred = (pred[:, 1:] - pred[:, :-1]).abs()      # (B, T-1, K)
        d_gt = (gt[:, 1:] - gt[:, :-1]).abs()            # (B, T-1, K)
        loss = loss + _moment_pair_per_channel(d_pred, d_gt, pair_mask)

    return loss


def yaw_aggregate_match_loss(
    stage1_raw_pred: Tensor,        # (B, T, 23) — raw (un-z-scored) prediction
    stage1_raw_gt: Tensor,          # (B, T, 23) — raw GT
    seq_mask: Tensor,               # (B, T)
) -> Tensor:
    """V7-B — yaw cross-segment statistics, mirroring PB1's
    ``loss_r29_gait_transition_rate`` + ``loss_r29_gait_duty_cycle``
    (temporal_interaction_losses.py:1303–1410).

    Two aggregate stats per clip:

      transition_rate = mean_t |Δyaw_unwrapped| over valid frame pairs
                       — how much yaw rotates per frame on average.
      cumulative_range = max_t yaw_unwrapped − min_t yaw_unwrapped
                       — total yaw envelope.

    Matched in raw radians. Mode-invariant: a CW vs CCW rotation of the
    same magnitude gives the same |Δyaw| and the same range, so the
    loss does not collapse multimodal yaw into the dataset mean.

    Returns
    -------
    Scalar in rad². Suggested weight: ~1.0–5.0 (yaw mean shift was 0.6
    rad in audit, so this needs real magnitude to compete with x0-MSE).
    """
    B, T, _ = stage1_raw_pred.shape
    if T < 2:
        return stage1_raw_pred.sum() * 0.0

    yaw_p = torch.atan2(stage1_raw_pred[..., CH_YAW_SIN],
                        stage1_raw_pred[..., CH_YAW_COS])           # (B, T)
    yaw_g = torch.atan2(stage1_raw_gt[..., CH_YAW_SIN],
                        stage1_raw_gt[..., CH_YAW_COS])             # (B, T)

    # 1-frame diffs, wrapped to [-π, π].
    twopi = 2.0 * 3.141592653589793
    d_p = yaw_p[:, 1:] - yaw_p[:, :-1]
    d_g = yaw_g[:, 1:] - yaw_g[:, :-1]
    d_p = (d_p + 3.141592653589793) % twopi - 3.141592653589793
    d_g = (d_g + 3.141592653589793) % twopi - 3.141592653589793
    pair_mask = (seq_mask[:, 1:] * seq_mask[:, :-1])                # (B, T-1)

    # Per-clip transition rate = mean |Δyaw|.
    abs_dp = d_p.abs() * pair_mask
    abs_dg = d_g.abs() * pair_mask
    denom = pair_mask.sum(dim=-1).clamp_min(1.0)                    # (B,)
    rate_p = abs_dp.sum(dim=-1) / denom                             # (B,)
    rate_g = abs_dg.sum(dim=-1) / denom                             # (B,)

    # Per-clip cumulative range = max_t yaw_unwrapped − min_t.
    # Use cumulative sum of wrapped diffs to recover unwrapped yaw locally
    # within a clip (avoid the atan2 ±π discontinuity).
    cumsum_p = torch.cumsum(d_p, dim=1)                              # (B, T-1)
    cumsum_g = torch.cumsum(d_g, dim=1)                              # (B, T-1)
    # Min/max over valid frames only — replace invalid with ±large
    # sentinels that the (max, min) reductions will reject.
    valid_b = (pair_mask > 0.5)                                       # (B, T-1) bool
    big = float(1e6)
    cp_max = torch.where(valid_b, cumsum_p, cumsum_p.new_full((), -big)).max(dim=1).values
    cp_min = torch.where(valid_b, cumsum_p, cumsum_p.new_full((), big)).min(dim=1).values
    cg_max = torch.where(valid_b, cumsum_g, cumsum_g.new_full((), -big)).max(dim=1).values
    cg_min = torch.where(valid_b, cumsum_g, cumsum_g.new_full((), big)).min(dim=1).values
    range_p = (cp_max - cp_min).clamp_min(0.0)                      # (B,)
    range_g = (cg_max - cg_min).clamp_min(0.0)                      # (B,)

    # SmoothL1 on the two statistics, averaged across the batch. SmoothL1
    # matches PB1's gait family (transition_rate uses smooth_l1_loss).
    loss_rate = torch.nn.functional.smooth_l1_loss(rate_p, rate_g.detach(),
                                                    reduction="mean")
    loss_range = torch.nn.functional.smooth_l1_loss(range_p, range_g.detach(),
                                                     reduction="mean")
    return loss_rate + loss_range


def fk_pelvis_spine_pos_loss_cm(
    *,
    pelvis_rot6d_pred: Tensor,    # (B, T, 6) — predicted (raw, un-z-scored)
    spine3_rot6d_pred: Tensor,    # (B, T, 6) — predicted
    root_world_pred: Tensor,      # (B, T, 3) — predicted (raw)
    gt_motion_135: Tensor,        # (B, T, 135) — ground truth motion
    rest_offsets: Tensor,         # (B, 22, 3)
    gt_joints: Tensor,            # (B, T, 22, 3) world frame
    seq_mask: Tensor,             # (B, T)
    target_joint_indices: tuple[int, ...] = (J_NECK, J_HEAD, J_L_SHOULDER, J_R_SHOULDER),
    beta_cm: float = 1.0,
) -> Tensor:
    """V7-style — cm-space SmoothL1 variant of :func:`fk_pelvis_spine_pos_loss`.

    Mirrors PB1's ``stable_local_vel_cm`` pattern (train_anchordiff.py:1090–
    1104): scale residuals by 100 (m → cm) and use SmoothL1 instead of L2.
    SmoothL1 is half-quadratic below ±1 cm (clean gradient at sub-cm) and
    linear above (tolerant of large outliers, no quadratic blow-up).

    Returns scalar in cm² (SmoothL1 reduction). Much larger numerical
    magnitude than the L2 version → use a smaller weight (~0.05–0.2 instead
    of 5.0–10.0).
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

    idx = torch.tensor(target_joint_indices, device=joints_pred.device)
    pred_sel = joints_pred.index_select(2, idx)                    # (B, T, K, 3)
    gt_sel = gt_joints.float().index_select(2, idx)

    # m → cm.
    diff_cm = (pred_sel - gt_sel) * 100.0                          # (B, T, K, 3)
    # SmoothL1 against zero. Mask invalid frames.
    mask_xyz = seq_mask.unsqueeze(-1).unsqueeze(-1).expand_as(diff_cm)  # (B, T, K, 3)
    diff_valid = diff_cm[mask_xyz > 0.5]
    if diff_valid.numel() == 0:
        return diff_cm.sum() * 0.0
    return torch.nn.functional.smooth_l1_loss(
        diff_valid, torch.zeros_like(diff_valid),
        reduction="mean", beta=beta_cm,
    )


# ──────────────────────────────────────────────────────────────────────────
# R31 V8 — wrist FK supervision (extends V7-C target set down the arm chain)
# ──────────────────────────────────────────────────────────────────────────
#
# V7-C (fk_pelvis_spine_pos_loss_cm) supervised the FK output at (neck,
# head, L_shoulder, R_shoulder) — the chain stops at shoulder. Wrist drift
# in the downstream PB1 diagnostic stayed ~35 cm on V7 V5; we never told
# Stage-1 (via gradient) that its pelvis_rot6d + spine3_rot6d errors
# cascade through GT L/R_collar + L/R_shoulder + L/R_elbow rotations all
# the way to wrist position.
#
# PB1 anti-wrist-drift uses:
#   anchor_joint_pos_weight=10.0  on (wrist, wrist, foot, foot, pelvis)
#                                  at contact-active frames only,
#   anchor_joint_vel_weight=2.0   same mask but velocity matching,
#   hand_endpoint_weight=2.0      x2 reweighting of wrist channels in
#                                  the dense L_pos loss.
#
# We can replicate the *signal* on Stage-1 by:
#   - extending the FK target list to include L/R wrist (and elbow as
#     intermediate; gives more leverage for the chain),
#   - optionally weighting the wrist channels higher (PB1 hand_endpoint
#     style),
#   - optionally masking to contact-active frames (PB1 anchor style),
#   - optionally adding a velocity term.
#
# Stage-1 does not predict shoulder/elbow rot6d, so the FK chain uses
# GT rotations there (same partial-FK substitution as V7-C). Only pred
# root_world + pred pelvis_rot6d + pred spine3_rot6d propagate gradients.


def wrist_fk_supervision_loss(
    *,
    pelvis_rot6d_pred: Tensor,    # (B, T, 6) — predicted (raw, un-z-scored)
    spine3_rot6d_pred: Tensor,    # (B, T, 6) — predicted
    root_world_pred: Tensor,      # (B, T, 3) — predicted (raw)
    gt_motion_135: Tensor,        # (B, T, 135) — ground truth motion
    rest_offsets: Tensor,         # (B, 22, 3)
    gt_joints: Tensor,            # (B, T, 22, 3) world frame
    seq_mask: Tensor,             # (B, T)
    target_joints: tuple[int, ...] = (
        J_NECK, J_HEAD, J_L_SHOULDER, J_R_SHOULDER,
        J_L_ELBOW, J_R_ELBOW, J_L_WRIST, J_R_WRIST,
    ),
    joint_weights: tuple[float, ...] | None = None,
    contact_state: Tensor | None = None,      # (B, T, 5) — 0/1 hand+foot+pelvis
    contact_mask_mode: str = "off",            # "off" | "reweight" | "hard"
    contact_active_weight: float = 4.0,         # multiplier on contact-active frames
    add_velocity: bool = False,
    velocity_weight: float = 0.5,
    beta_cm: float = 1.0,
) -> Tensor:
    """V8 — extend V7-C's partial FK supervision down the arm chain to wrist.

    Parameters
    ----------
    pelvis_rot6d_pred, spine3_rot6d_pred, root_world_pred
        Predicted pieces of Stage-1's 23-D output (un-z-scored). Same
        substitution pattern as V7-C: these replace the joint-0 and joint-9
        rotations in the GT-rot6d chain; root_world replaces motion[..., 132:135].
    gt_motion_135, rest_offsets, gt_joints, seq_mask
        Ground truth scaffold.
    target_joints
        Joint indices the loss evaluates. Default = (neck, head, L_sh,
        R_sh, L_el, R_el, L_wr, R_wr). At minimum {neck, head, L_sh,
        R_sh} reproduces V7-C; adding wrists is the V8 axis.
    joint_weights
        Per-target-joint scalar weight. If None, equal weighting. To
        replicate PB1 hand_endpoint_weight=2.0, pass weights with 2.0
        on the wrist indices and 1.0 elsewhere.
    contact_state
        (B, T, 5). Surface ``contact_state`` from the batch (it is the
        same field the PB1 trainer uses). Required when contact_mask_mode
        != "off".
    contact_mask_mode
        - "off": all frames supervised equally.
        - "reweight": frames where contact_state[..., {LH, RH}] >= 0.5
          get x ``contact_active_weight``; others stay at 1.0.
        - "hard": only contact-active frames (any of LH or RH active)
          enter the loss; non-contact frames are zeroed.
    contact_active_weight
        Multiplier used in "reweight" mode (ignored in "hard").
    add_velocity
        When True, also computes a cm-space SmoothL1 on the 1-frame
        finite difference of predicted target-joint positions vs GT.
    velocity_weight
        Multiplier on the velocity term inside the loss (the combined
        return value is pos + velocity_weight * vel).
    beta_cm
        SmoothL1 beta (in cm); 1 cm = half-quadratic boundary.

    Returns
    -------
    Scalar in cm² (SmoothL1 reduction).
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

    K = len(target_joints)
    idx = torch.tensor(target_joints, device=joints_pred.device,
                       dtype=torch.long)
    pred_sel = joints_pred.index_select(2, idx)                    # (B, T, K, 3)
    gt_sel = gt_joints.float().index_select(2, idx)                # (B, T, K, 3)

    # Per-joint weights, default 1.0.
    if joint_weights is None:
        jw = torch.ones(K, device=joints_pred.device,
                        dtype=joints_pred.dtype)
    else:
        if len(joint_weights) != K:
            raise ValueError(
                f"joint_weights length {len(joint_weights)} must match "
                f"target_joints length {K}."
            )
        jw = torch.tensor(joint_weights, device=joints_pred.device,
                          dtype=joints_pred.dtype)

    # Frame mask. Combine seq_mask with optional contact reweighting / hard.
    if contact_mask_mode == "off":
        frame_w = seq_mask                                          # (B, T)
    elif contact_mask_mode in ("reweight", "hard"):
        if contact_state is None or contact_state.shape[-1] < 2:
            raise ValueError(
                "contact_mask_mode='reweight'/'hard' requires contact_state "
                "with at least 2 hand channels (CONTACT_IDX_LEFT_HAND, "
                "CONTACT_IDX_RIGHT_HAND)."
            )
        hand_active = (
            (contact_state[..., CONTACT_IDX_LEFT_HAND] >= 0.5)
            | (contact_state[..., CONTACT_IDX_RIGHT_HAND] >= 0.5)
        ).to(seq_mask.dtype)                                       # (B, T)
        if contact_mask_mode == "reweight":
            # 1.0 on non-contact, contact_active_weight on contact frames.
            frame_w = seq_mask * (1.0 + (contact_active_weight - 1.0) * hand_active)
        else:  # "hard"
            frame_w = seq_mask * hand_active
    else:
        raise ValueError(
            f"contact_mask_mode must be 'off' | 'reweight' | 'hard'; "
            f"got {contact_mask_mode!r}."
        )

    # Position term — cm-scale SmoothL1.
    diff_pos_cm = (pred_sel - gt_sel) * 100.0                      # (B, T, K, 3)
    # weight = frame_w (B, T) * jw (K), broadcast to (B, T, K).
    weight_bt_k = (
        frame_w.unsqueeze(-1) * jw.view(1, 1, K)
    )                                                               # (B, T, K)
    weight_bt_k_xyz = weight_bt_k.unsqueeze(-1).expand_as(diff_pos_cm)  # (B, T, K, 3)
    w_sum = weight_bt_k_xyz.sum().clamp_min(1.0)
    # SmoothL1 with weight reduction = manual: sum(|x|<beta: 0.5 x²/beta; else |x|-0.5*beta) * w / w_sum.
    abs_d = diff_pos_cm.abs()
    sl1 = torch.where(
        abs_d < beta_cm,
        0.5 * diff_pos_cm.pow(2) / beta_cm,
        abs_d - 0.5 * beta_cm,
    )                                                               # (B, T, K, 3)
    loss_pos = (sl1 * weight_bt_k_xyz).sum() / w_sum

    if not add_velocity or T < 2:
        return loss_pos

    # Velocity term — same masking & weighting on 1-frame finite diff.
    vel_pred = pred_sel[:, 1:] - pred_sel[:, :-1]                  # (B, T-1, K, 3)
    vel_gt = gt_sel[:, 1:] - gt_sel[:, :-1]
    diff_vel_cm = (vel_pred - vel_gt) * 100.0
    pair_w = (frame_w[:, 1:] * frame_w[:, :-1]).unsqueeze(-1) * jw.view(1, 1, K)
    pair_w_xyz = pair_w.unsqueeze(-1).expand_as(diff_vel_cm)
    w_sum_v = pair_w_xyz.sum().clamp_min(1.0)
    abs_dv = diff_vel_cm.abs()
    sl1_v = torch.where(
        abs_dv < beta_cm,
        0.5 * diff_vel_cm.pow(2) / beta_cm,
        abs_dv - 0.5 * beta_cm,
    )
    loss_vel = (sl1_v * pair_w_xyz).sum() / w_sum_v

    return loss_pos + velocity_weight * loss_vel


# ──────────────────────────────────────────────────────────────────────────
# R31 V8 — frame-0 anchor signal (init_pose) and frame-0 consistency loss
# ──────────────────────────────────────────────────────────────────────────
#
# Phase 1 audit showed Stage-1 violates the frame-0 invariant
# (root_local channel [0:3] rms_at_t0 = 9.8 cm; should be exactly 0 by
# construction). PB1 was trained with init_pose=GT joints_22 frame 0 but
# Stage-1 currently has no equivalent input.
#
# Two F-mode variants tested:
#
#   F1 (init_pose_dim=135): the full motion_135 frame-0 slice = 22 rot6d
#     (132) + root world (3). All Stage-1 frame-0 information available;
#     redundant on channels Stage-1 doesn't predict but no harm.
#
#   F2 (init_pose_dim=14): only the channels Stage-1 outputs:
#     pelvis_rot6d (6) + spine3_rot6d (6) + head_height (1) +
#     shoulder_center_h (1) = 14. Pulled from motion_135[:, 0, :] in
#     raw space, then z-scored against the same stage1_coarse_norm stats
#     the rest of the trainer uses.
#
# The init_pose vector goes through ``V12InputProjection.init_pose_proj``
# (zero-init Linear), so step-0 forward equals the V0 forward
# bit-identically — V8 training starts from the same noise floor and
# the init_pose contribution grows during optimisation only if it helps.


# F2 channel indices into stage1_coarse 23-D (matches stage1_coarse_oracle.py).
INIT_POSE_F2_INDICES: tuple[int, ...] = tuple(
    list(range(9, 21))   # pelvis_rot6d (9:15) + spine3_rot6d (15:21)
    + [21, 22]            # head_height, shoulder_center_h
)
"""14 channel indices used by F2 init_pose mode (must match the channels
Stage-1 emits at t=0 in raw space)."""

INIT_POSE_F2_DIM: int = 14


def build_init_pose_f1(motion_135: Tensor) -> Tensor:
    """F1 — full motion_135 frame-0 slice (B, 135).

    Reads ``motion_135[:, 0, :]``, returns raw-space (no normalisation).
    The init_pose_proj zero-init Linear absorbs the scale at training
    time; no need to z-score F1.
    """
    if motion_135.dim() != 3 or motion_135.shape[-1] != 135:
        raise ValueError(
            f"motion_135 must be (B, T, 135); got {tuple(motion_135.shape)}"
        )
    return motion_135[:, 0, :].float()                                # (B, 135)


def build_init_pose_f2(
    stage1_coarse_raw: Tensor,        # (B, T, 23) — un-z-scored oracle
    mean_t: Tensor,                   # (23,) — stage1_coarse_norm mean
    std_t: Tensor,                    # (23,) — stage1_coarse_norm std
) -> Tensor:
    """F2 — frame-0 slice of the 14 channels Stage-1 actually outputs,
    z-scored against the stage1_coarse stats.

    Z-scoring is important here: the rest of the trainer feeds Stage-1
    z-scored targets, so the model's internal scale is z-scored. F2's
    init_pose is z-scored to share that scale; the F1 alternative
    contains 22 rot6d (different scale per joint) and is left raw,
    relying on the zero-init Linear to learn the right scale.
    """
    if stage1_coarse_raw.dim() != 3 or stage1_coarse_raw.shape[-1] != 23:
        raise ValueError(
            f"stage1_coarse_raw must be (B, T, 23); got "
            f"{tuple(stage1_coarse_raw.shape)}"
        )
    frame0 = stage1_coarse_raw[:, 0, :]                                # (B, 23)
    # Z-score before slicing — keeps indexing clean.
    frame0_z = (frame0 - mean_t.view(1, -1)) / std_t.view(1, -1)
    idx = torch.tensor(
        INIT_POSE_F2_INDICES,
        device=frame0_z.device, dtype=torch.long,
    )
    return frame0_z.index_select(-1, idx).float()                      # (B, 14)


def frame0_consistency_loss(
    stage1_pred_zscored: Tensor,      # (B, T, 23) — pred in z-scored space
    init_pose_targets_zscored: Tensor,  # (B, 14) — F2 init_pose targets
) -> Tensor:
    """Force Stage-1's pred at t=0 on the same 14 channels to equal the
    init_pose target. Operates in z-scored space (matches main x0-MSE
    so the gradient scale is uniform).

    Returns scalar MSE.
    """
    if stage1_pred_zscored.shape[-1] != 23:
        raise ValueError(
            f"stage1_pred_zscored must end in 23; got "
            f"{tuple(stage1_pred_zscored.shape)}"
        )
    pred_frame0 = stage1_pred_zscored[:, 0, :]                         # (B, 23)
    idx = torch.tensor(
        INIT_POSE_F2_INDICES,
        device=pred_frame0.device, dtype=torch.long,
    )
    pred_sel = pred_frame0.index_select(-1, idx)                       # (B, 14)
    err = (pred_sel - init_pose_targets_zscored).pow(2)
    return err.mean()
