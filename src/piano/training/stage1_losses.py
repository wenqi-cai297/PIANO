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


# ──────────────────────────────────────────────────────────────────────────
# R40 — per-channel weight helper (used by the Stage-1 trainer's x0/vel MSE)
# ──────────────────────────────────────────────────────────────────────────


def build_channel_weight_tensor(
    weights,
    *,
    expected_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> Tensor | None:
    """Validate a per-channel weight list and convert to a broadcast tensor.

    R40 lets configs reduce exact-GT pressure on under-determined channels
    (root, vel, yaw, pelvis_rot6d) by passing a 23-long weight list. An empty
    or omitted list returns ``None``, which the caller treats as "all ones"
    so the old MSE behavior is preserved bit-for-bit.

    Parameters
    ----------
    weights
        Per-channel scalar multipliers (list, tuple, or ``None``). Empty
        or ``None`` → no weighting.
    expected_dim
        Required length when non-empty (23 for stage1_coarse).
    device, dtype
        Where/how to materialise the tensor.
    name
        Identifier used in the ValueError message.

    Returns
    -------
    Tensor of shape ``(1, 1, expected_dim)`` ready to broadcast onto
    ``(B, T, expected_dim)`` per-dim squared error, or ``None`` when no
    weighting was requested.
    """
    if weights is None:
        return None
    if not isinstance(weights, (list, tuple)):
        raise ValueError(
            f"{name} must be a list/tuple or None; got {type(weights).__name__}."
        )
    if len(weights) == 0:
        return None
    if len(weights) != expected_dim:
        raise ValueError(
            f"{name} must have length {expected_dim}; got {len(weights)}."
        )
    return torch.tensor(
        list(weights), device=device, dtype=dtype,
    ).view(1, 1, expected_dim)


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


def temporal_derivative_mse_loss(
    pred: Tensor,
    gt: Tensor,
    seq_mask: Tensor,
    *,
    order: int = 1,
    channel_subset: tuple[int, ...] | None = None,
    normalize_by_gt_std: bool = True,
) -> Tensor:
    """Masked raw-space finite-difference MSE for velocity/acceleration.

    ``order=1`` compares frame-to-frame velocity. ``order=2`` compares
    acceleration. The helper is intentionally generic so Stage-1 can apply it
    to the 23-D raw coarse target and Stage-1.5 can reuse the same convention
    on C41 via its own thin wrapper.
    """
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs gt {gt.shape}")
    if pred.ndim != 3:
        raise ValueError(f"expected (B, T, C), got {pred.shape}")
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order}")

    B, T, D = pred.shape
    if seq_mask.shape != (B, T):
        raise ValueError(
            f"seq_mask shape {tuple(seq_mask.shape)} != (B, T) = {(B, T)}"
        )
    if T <= order:
        return pred.sum() * 0.0

    if channel_subset is None:
        pred_sel = pred
        gt_sel = gt
        n_ch = D
    else:
        idx = torch.tensor(
            channel_subset, device=pred.device, dtype=torch.long,
        )
        pred_sel = pred.index_select(-1, idx)
        gt_sel = gt.index_select(-1, idx)
        n_ch = int(idx.numel())
        if n_ch == 0:
            return pred.sum() * 0.0

    d_pred = pred_sel
    d_gt = gt_sel
    valid = seq_mask.float()
    for _ in range(order):
        d_pred = d_pred[:, 1:] - d_pred[:, :-1]
        d_gt = d_gt[:, 1:] - d_gt[:, :-1]
        valid = valid[:, 1:] * valid[:, :-1]

    err = (d_pred - d_gt).pow(2)
    if normalize_by_gt_std:
        # Per-channel variance under the same derivative mask. This prevents
        # high-magnitude channels from drowning out small but important ones.
        flat_gt = d_gt.reshape(-1, n_ch)
        flat_valid = valid.reshape(-1).to(dtype=flat_gt.dtype)
        w_sum = flat_valid.sum().clamp_min(1.0)
        mean_gt = (flat_gt * flat_valid.unsqueeze(-1)).sum(0) / w_sum
        var_gt = (
            (flat_gt - mean_gt).pow(2) * flat_valid.unsqueeze(-1)
        ).sum(0) / w_sum
        err = err / var_gt.clamp_min(1e-6).view(1, 1, n_ch)

    denom = valid.sum().clamp_min(1.0) * float(n_ch)
    return (err * valid.unsqueeze(-1).to(err.dtype)).sum() / denom


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


# ──────────────────────────────────────────────────────────────────────────
# R40 — Stage-1 plan-invariant loss (plan-energy)
# ──────────────────────────────────────────────────────────────────────────
#
# Stage-1's training objective so far has been per-frame GT regression
# (w_x0 + w_vel + V7 anti-collapse moments). R35 / R31 dynamics audits
# showed Stage-1 still collapses on multi-modal channels (root path,
# facing yaw, pelvis posture): generated stage1_coarse has vel_ratio
# 0.379 on velocity_xzy and 0.474 on pelvis_rot6d vs GT, the canonical
# mode-collapse fingerprint of MSE under multimodal-conditional sampling.
#
# The fix in R40 mirrors what got Stage-2 PB1 out of the same problem
# (R29 PB1, train_anchordiff.py:1106): instead of forcing the per-frame
# GT mode, supervise plan-level invariants that any valid sample should
# satisfy (speed envelope, arc length, turn activity, root-object radial
# distance, rotation activity, height envelope, plus a light smoothness
# anchor). Combined with x0_channel_weights downweighting the ambiguous
# channels in the exact-GT term, the model is free to pick a mode rather
# than averaging.
#
# Mode-invariance design: all GT comparisons use moments (mean/std)
# or cross-segment statistics, not signed per-frame values. So a CCW
# vs CW path of the same magnitude, or left vs right routing past the
# object, do not produce different penalties.


_R40_DEFAULT_COMPONENT_WEIGHTS: dict[str, float] = {
    "root_speed": 1.0,
    "root_arc": 1.0,
    "root_displacement": 0.5,
    "root_object_radial": 1.0,
    "yaw_activity": 1.0,
    "rot_activity": 0.5,
    "height_envelope": 0.5,
    "smoothness": 0.05,
}

# bf16-safe epsilon for ``sqrt(x + _SQRT_EPS)``. The gradient of sqrt at
# x is 1/(2*sqrt(x)), so at x=eps it equals 1/(2*sqrt(eps)). With eps=1e-6
# the per-element grad floor is 500 (vs 5e5 for 1e-12) — orders of
# magnitude safer in bf16, which only has ~3 decimal digits of mantissa
# precision. C3's step-50 NaN cascade in R40 came from sqrt(clamp_min(1e-12))
# back-propagating undefined gradients through degenerate inputs (e.g.
# pred velocity ≈ 0 on a frozen-root mode-collapse step), so every value-
# domain sqrt in the plan-invariant loss uses this floor.
_SQRT_EPS: float = 1e-6


def _masked_mean_std(
    x: Tensor, mask_btc: Tensor,
) -> tuple[Tensor, Tensor]:
    """Per-batch-element mean and std of ``x`` under ``mask_btc``.

    Parameters
    ----------
    x : (B, T) or (B, T-1) — scalar-per-frame quantity.
    mask_btc : same shape as ``x`` — float mask (0/1).

    Returns
    -------
    mean : (B,) — masked mean per element. Zero when no valid frames.
    std  : (B,) — masked std (population, sqrt of variance). Zero when
                 fewer than 2 valid frames or constant.
    """
    w_sum = mask_btc.sum(dim=-1).clamp_min(1.0)                # (B,)
    x_sum = (x * mask_btc).sum(dim=-1)                         # (B,)
    mean = x_sum / w_sum
    centered = (x - mean.unsqueeze(-1)) * mask_btc             # (B, T)
    var = (centered.pow(2)).sum(dim=-1) / w_sum                # (B,)
    # bf16-safe: clamping at 0 then sqrt leaves the gradient as 1/(2*0)
    # = inf at degenerate (constant) inputs. Adding _SQRT_EPS floors the
    # backward grad at ~500 instead, which bf16 can represent without
    # overflow.
    std = (var.clamp_min(0.0) + _SQRT_EPS).sqrt()              # (B,)
    return mean, std


def _final_valid_index(seq_mask: Tensor) -> Tensor:
    """Index of the last valid frame per clip. Returns 0 when no valid frame.

    seq_mask : (B, T) float 0/1.
    """
    B, T = seq_mask.shape
    idx = torch.arange(T, device=seq_mask.device, dtype=seq_mask.dtype)
    masked_idx = idx.view(1, T) * seq_mask                     # (B, T)
    return masked_idx.argmax(dim=-1).long()                    # (B,)


def stage1_plan_invariant_loss(
    stage1_raw_pred: Tensor,           # (B, T, 23) raw (un-z-scored)
    stage1_raw_gt: Tensor,             # (B, T, 23) raw (un-z-scored)
    object_world_traj: Tensor,         # (B, T, 9) — COM(3, world xyz) + rot6d(6)
    root_world_t0: Tensor,             # (B, 1, 3) — motion[:, :1, 132:135], world xyz
    seq_mask: Tensor,                  # (B, T) float
    *,
    component_weights: dict[str, float] | None = None,
    beta: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """R40 plan-invariant ("plan-energy") loss.

    Returns
    -------
    total : scalar — weighted sum of per-component SmoothL1 losses.
    components : dict[str, Tensor] — the raw (unweighted) scalar per
        component, detached. Includes only the components that were
        actually evaluated.

    Stage-1 channel order (oracle_v1):
        [0:3]  root_local x, z, y     (offset from root_world_t0)
        [3:6]  vel x, z, y            (world-frame finite diff)
        [6:8]  yaw_sin, yaw_cos
        [8]    yaw_vel (unwrapped)
        [9:15] pelvis_rot6d
        [15:21] spine3_rot6d
        [21]   head_height_y
        [22]   shoulder_center_h_y

    Notes
    -----
    - All GT summary stats are detached (target should not back-prop).
    - Root world XZ is reconstructed by adding root_world_t0's (x, z) to
      the predicted root_local (x, z) channels (indices [0] and [1] in
      the oracle layout — y is at index [2]).
    - Object world XZ is ``object_world_traj[..., [0, 2]]`` (world COM
      is stored as x, y, z).
    - All reductions respect ``seq_mask`` (frame mask) and pair-masks
      (where a finite-difference pair requires both frames valid).
    - When the entire batch lacks valid frames/pairs, the function
      returns a gradient-safe zero on ``stage1_raw_pred``.
    """
    if stage1_raw_pred.shape != stage1_raw_gt.shape:
        raise ValueError(
            f"pred/gt shape mismatch: {stage1_raw_pred.shape} vs "
            f"{stage1_raw_gt.shape}"
        )
    if stage1_raw_pred.ndim != 3 or stage1_raw_pred.shape[-1] != 23:
        raise ValueError(
            f"expected (B, T, 23); got {tuple(stage1_raw_pred.shape)}"
        )
    if object_world_traj.shape[:2] != stage1_raw_pred.shape[:2]:
        raise ValueError(
            f"object_world_traj batch/seq mismatch: "
            f"{tuple(object_world_traj.shape)} vs {tuple(stage1_raw_pred.shape)}"
        )
    if root_world_t0.shape[1] != 1 or root_world_t0.shape[-1] != 3:
        raise ValueError(
            f"root_world_t0 must be (B, 1, 3); got {tuple(root_world_t0.shape)}"
        )
    if seq_mask.shape != stage1_raw_pred.shape[:2]:
        raise ValueError(
            f"seq_mask shape {tuple(seq_mask.shape)} != (B, T)"
        )

    B, T, _ = stage1_raw_pred.shape
    device = stage1_raw_pred.device
    dtype = stage1_raw_pred.dtype

    weights = dict(_R40_DEFAULT_COMPONENT_WEIGHTS)
    if component_weights is not None:
        for k, v in component_weights.items():
            if k not in weights:
                raise ValueError(
                    f"unknown plan-invariant component: {k!r}; "
                    f"expected one of {sorted(weights.keys())}"
                )
            weights[k] = float(v)

    seq_mask_f = seq_mask.to(dtype=dtype)
    pair_mask = (
        seq_mask_f[:, 1:] * seq_mask_f[:, :-1]
        if T >= 2 else seq_mask_f[:, :0]
    )                                                            # (B, T-1)
    triple_mask = (
        seq_mask_f[:, 2:] * seq_mask_f[:, 1:-1] * seq_mask_f[:, :-2]
        if T >= 3 else seq_mask_f[:, :0]
    )                                                            # (B, T-2)

    # ─── Build world-frame root XZ from root_local + t0 ──────────────
    # Stage-1 layout: root_local (x, z, y); root_world_t0 (x, y, z).
    rx_pred = stage1_raw_pred[..., 0]                            # (B, T)
    rz_pred = stage1_raw_pred[..., 1]                            # (B, T)
    rx_gt = stage1_raw_gt[..., 0]
    rz_gt = stage1_raw_gt[..., 1]
    t0_world_x = root_world_t0[..., 0]                           # (B, 1)
    t0_world_z = root_world_t0[..., 2]                           # (B, 1)
    rwx_pred = rx_pred + t0_world_x
    rwz_pred = rz_pred + t0_world_z
    rwx_gt = rx_gt + t0_world_x
    rwz_gt = rz_gt + t0_world_z

    # ─── Object XZ ──
    obj_x = object_world_traj[..., 0].to(dtype=dtype)            # (B, T)
    obj_z = object_world_traj[..., 2].to(dtype=dtype)            # (B, T)

    components: dict[str, Tensor] = {}

    def _sl1_mean(diff: Tensor) -> Tensor:
        """Mean SmoothL1(diff, 0) with beta."""
        abs_d = diff.abs()
        return torch.where(
            abs_d < beta,
            0.5 * diff.pow(2) / beta,
            abs_d - 0.5 * beta,
        ).mean()

    # ── (1) root_speed: SmoothL1 on (mean, std) of frame-to-frame XZ speed ──
    if T >= 2 and weights["root_speed"] != 0:
        dx_p = rwx_pred[:, 1:] - rwx_pred[:, :-1]
        dz_p = rwz_pred[:, 1:] - rwz_pred[:, :-1]
        sp_p = (dx_p.pow(2) + dz_p.pow(2)).clamp_min(0.0).add(_SQRT_EPS).sqrt()
        dx_g = rwx_gt[:, 1:] - rwx_gt[:, :-1]
        dz_g = rwz_gt[:, 1:] - rwz_gt[:, :-1]
        sp_g = (dx_g.pow(2) + dz_g.pow(2)).clamp_min(0.0).add(_SQRT_EPS).sqrt()
        mean_p, std_p = _masked_mean_std(sp_p, pair_mask)
        mean_g, std_g = _masked_mean_std(sp_g, pair_mask)
        loss_rs = (
            _sl1_mean(mean_p - mean_g.detach())
            + _sl1_mean(std_p - std_g.detach())
        )
        components["root_speed"] = loss_rs

    # ── (2) root_arc: SmoothL1 on cumulative XZ speed sum per clip ──
    if T >= 2 and weights["root_arc"] != 0:
        dx_p = rwx_pred[:, 1:] - rwx_pred[:, :-1]
        dz_p = rwz_pred[:, 1:] - rwz_pred[:, :-1]
        sp_p = (dx_p.pow(2) + dz_p.pow(2)).clamp_min(0.0).add(_SQRT_EPS).sqrt()
        dx_g = rwx_gt[:, 1:] - rwx_gt[:, :-1]
        dz_g = rwz_gt[:, 1:] - rwz_gt[:, :-1]
        sp_g = (dx_g.pow(2) + dz_g.pow(2)).clamp_min(0.0).add(_SQRT_EPS).sqrt()
        arc_p = (sp_p * pair_mask).sum(dim=-1)                   # (B,)
        arc_g = (sp_g * pair_mask).sum(dim=-1)
        components["root_arc"] = _sl1_mean(arc_p - arc_g.detach())

    # ── (3) root_displacement: final-frame XZ offset from t=0 ──
    if weights["root_displacement"] != 0:
        last = _final_valid_index(seq_mask_f)
        gathered_pred_x = rwx_pred.gather(1, last.view(B, 1)).squeeze(1)
        gathered_pred_z = rwz_pred.gather(1, last.view(B, 1)).squeeze(1)
        gathered_gt_x = rwx_gt.gather(1, last.view(B, 1)).squeeze(1)
        gathered_gt_z = rwz_gt.gather(1, last.view(B, 1)).squeeze(1)
        d_pred = (
            (gathered_pred_x - t0_world_x.squeeze(-1)).pow(2)
            + (gathered_pred_z - t0_world_z.squeeze(-1)).pow(2)
        ).clamp_min(0.0).add(_SQRT_EPS).sqrt()
        d_gt = (
            (gathered_gt_x - t0_world_x.squeeze(-1)).pow(2)
            + (gathered_gt_z - t0_world_z.squeeze(-1)).pow(2)
        ).clamp_min(0.0).add(_SQRT_EPS).sqrt()
        components["root_displacement"] = _sl1_mean(d_pred - d_gt.detach())

    # ── (4) root_object_radial: distance-to-object profile stats ──
    if weights["root_object_radial"] != 0:
        dist_p = (
            (rwx_pred - obj_x).pow(2) + (rwz_pred - obj_z).pow(2)
        ).clamp_min(0.0).add(_SQRT_EPS).sqrt()                                # (B, T)
        dist_g = (
            (rwx_gt - obj_x).pow(2) + (rwz_gt - obj_z).pow(2)
        ).clamp_min(0.0).add(_SQRT_EPS).sqrt()
        mean_p, std_p = _masked_mean_std(dist_p, seq_mask_f)
        mean_g, std_g = _masked_mean_std(dist_g, seq_mask_f)
        # Min: replace invalid frames with +inf so they don't dominate.
        big = torch.full_like(dist_p, float(1e9))
        valid_b = seq_mask_f > 0.5
        min_p = torch.where(valid_b, dist_p, big).min(dim=-1).values
        min_g = torch.where(valid_b, dist_g, big).min(dim=-1).values
        # Final (last valid frame).
        last = _final_valid_index(seq_mask_f)
        final_p = dist_p.gather(1, last.view(B, 1)).squeeze(1)
        final_g = dist_g.gather(1, last.view(B, 1)).squeeze(1)
        loss_ror = (
            _sl1_mean(mean_p - mean_g.detach())
            + _sl1_mean(std_p - std_g.detach())
            + _sl1_mean(min_p - min_g.detach())
            + _sl1_mean(final_p - final_g.detach())
        )
        components["root_object_radial"] = loss_ror

    # ── (5) yaw_activity: |Δyaw_unwrapped| moments + cumulative range ──
    if T >= 2 and weights["yaw_activity"] != 0:
        yaw_p = torch.atan2(stage1_raw_pred[..., CH_YAW_SIN],
                            stage1_raw_pred[..., CH_YAW_COS])
        yaw_g = torch.atan2(stage1_raw_gt[..., CH_YAW_SIN],
                            stage1_raw_gt[..., CH_YAW_COS])
        twopi = 2.0 * 3.141592653589793
        d_p = yaw_p[:, 1:] - yaw_p[:, :-1]
        d_g = yaw_g[:, 1:] - yaw_g[:, :-1]
        d_p = (d_p + 3.141592653589793) % twopi - 3.141592653589793
        d_g = (d_g + 3.141592653589793) % twopi - 3.141592653589793
        abs_dp = d_p.abs()
        abs_dg = d_g.abs()
        mean_p, std_p = _masked_mean_std(abs_dp, pair_mask)
        mean_g, std_g = _masked_mean_std(abs_dg, pair_mask)
        # Cumulative yaw range from cumsum of wrapped diffs (avoids
        # the atan2 ±π discontinuity).
        cs_p = torch.cumsum(d_p, dim=-1)
        cs_g = torch.cumsum(d_g, dim=-1)
        valid_b = (pair_mask > 0.5)
        big = float(1e6)
        cp_max = torch.where(
            valid_b, cs_p, cs_p.new_full((), -big)
        ).max(dim=-1).values
        cp_min = torch.where(
            valid_b, cs_p, cs_p.new_full((), big)
        ).min(dim=-1).values
        cg_max = torch.where(
            valid_b, cs_g, cs_g.new_full((), -big)
        ).max(dim=-1).values
        cg_min = torch.where(
            valid_b, cs_g, cs_g.new_full((), big)
        ).min(dim=-1).values
        range_p = (cp_max - cp_min).clamp_min(0.0)
        range_g = (cg_max - cg_min).clamp_min(0.0)
        loss_yact = (
            _sl1_mean(mean_p - mean_g.detach())
            + _sl1_mean(std_p - std_g.detach())
            + _sl1_mean(range_p - range_g.detach())
        )
        components["yaw_activity"] = loss_yact

    # ── (6) rot_activity: |Δrot6d| moments for pelvis + spine3 ──
    if T >= 2 and weights["rot_activity"] != 0:
        # Per-frame finite-diff magnitude across the 6 channels of the
        # rot6d block (vector L2 norm of the 6-D diff). Not strictly
        # angular velocity but a representation-level activity stat.
        pel_p = stage1_raw_pred[..., CH_PELVIS_ROT6D]
        pel_g = stage1_raw_gt[..., CH_PELVIS_ROT6D]
        sp_p = stage1_raw_pred[..., CH_SPINE3_ROT6D]
        sp_g = stage1_raw_gt[..., CH_SPINE3_ROT6D]
        d_pel_p = (pel_p[:, 1:] - pel_p[:, :-1]).pow(2).sum(-1).clamp_min(0.0).add(_SQRT_EPS).sqrt()
        d_pel_g = (pel_g[:, 1:] - pel_g[:, :-1]).pow(2).sum(-1).clamp_min(0.0).add(_SQRT_EPS).sqrt()
        d_sp_p = (sp_p[:, 1:] - sp_p[:, :-1]).pow(2).sum(-1).clamp_min(0.0).add(_SQRT_EPS).sqrt()
        d_sp_g = (sp_g[:, 1:] - sp_g[:, :-1]).pow(2).sum(-1).clamp_min(0.0).add(_SQRT_EPS).sqrt()
        m_pel_p, s_pel_p = _masked_mean_std(d_pel_p, pair_mask)
        m_pel_g, s_pel_g = _masked_mean_std(d_pel_g, pair_mask)
        m_sp_p, s_sp_p = _masked_mean_std(d_sp_p, pair_mask)
        m_sp_g, s_sp_g = _masked_mean_std(d_sp_g, pair_mask)
        loss_ract = (
            _sl1_mean(m_pel_p - m_pel_g.detach())
            + _sl1_mean(s_pel_p - s_pel_g.detach())
            + _sl1_mean(m_sp_p - m_sp_g.detach())
            + _sl1_mean(s_sp_p - s_sp_g.detach())
        )
        components["rot_activity"] = loss_ract

    # ── (7) height_envelope: head/shoulder height (mean, min, max) ──
    if weights["height_envelope"] != 0:
        head_p = stage1_raw_pred[..., CH_HEAD_HEIGHT]
        head_g = stage1_raw_gt[..., CH_HEAD_HEIGHT]
        sh_p = stage1_raw_pred[..., CH_SHOULDER_H]
        sh_g = stage1_raw_gt[..., CH_SHOULDER_H]
        m_head_p, _ = _masked_mean_std(head_p, seq_mask_f)
        m_head_g, _ = _masked_mean_std(head_g, seq_mask_f)
        m_sh_p, _ = _masked_mean_std(sh_p, seq_mask_f)
        m_sh_g, _ = _masked_mean_std(sh_g, seq_mask_f)
        big = torch.full_like(head_p, float(1e9))
        valid_b = seq_mask_f > 0.5
        min_head_p = torch.where(valid_b, head_p, big).min(dim=-1).values
        min_head_g = torch.where(valid_b, head_g, big).min(dim=-1).values
        max_head_p = torch.where(valid_b, head_p, -big).max(dim=-1).values
        max_head_g = torch.where(valid_b, head_g, -big).max(dim=-1).values
        min_sh_p = torch.where(valid_b, sh_p, big).min(dim=-1).values
        min_sh_g = torch.where(valid_b, sh_g, big).min(dim=-1).values
        max_sh_p = torch.where(valid_b, sh_p, -big).max(dim=-1).values
        max_sh_g = torch.where(valid_b, sh_g, -big).max(dim=-1).values
        loss_he = (
            _sl1_mean(m_head_p - m_head_g.detach())
            + _sl1_mean(m_sh_p - m_sh_g.detach())
            + _sl1_mean(min_head_p - min_head_g.detach())
            + _sl1_mean(max_head_p - max_head_g.detach())
            + _sl1_mean(min_sh_p - min_sh_g.detach())
            + _sl1_mean(max_sh_p - max_sh_g.detach())
        )
        components["height_envelope"] = loss_he

    # ── (8) smoothness (pred-only): root XZ accel + yaw accel magnitude ──
    if T >= 3 and weights["smoothness"] != 0:
        vx = rwx_pred[:, 1:] - rwx_pred[:, :-1]                  # (B, T-1)
        vz = rwz_pred[:, 1:] - rwz_pred[:, :-1]
        ax = vx[:, 1:] - vx[:, :-1]                              # (B, T-2)
        az = vz[:, 1:] - vz[:, :-1]
        acc_mag = (ax.pow(2) + az.pow(2)).clamp_min(0.0).add(_SQRT_EPS).sqrt()
        yaw_p = torch.atan2(stage1_raw_pred[..., CH_YAW_SIN],
                            stage1_raw_pred[..., CH_YAW_COS])
        d_yaw = yaw_p[:, 1:] - yaw_p[:, :-1]
        twopi = 2.0 * 3.141592653589793
        d_yaw = (d_yaw + 3.141592653589793) % twopi - 3.141592653589793
        a_yaw = (d_yaw[:, 1:] - d_yaw[:, :-1]).abs()
        if triple_mask.numel() > 0:
            denom = triple_mask.sum().clamp_min(1.0)
            acc_term = (acc_mag * triple_mask).sum() / denom
            yaw_term = (a_yaw * triple_mask).sum() / denom
            components["smoothness"] = acc_term + yaw_term
        else:
            components["smoothness"] = stage1_raw_pred.sum() * 0.0
    elif weights["smoothness"] != 0:
        components["smoothness"] = stage1_raw_pred.sum() * 0.0

    # Aggregate weighted components.
    if not components:
        return stage1_raw_pred.sum() * 0.0, {}
    total = stage1_raw_pred.new_zeros(())
    for name, val in components.items():
        total = total + weights[name] * val
    return total, components
