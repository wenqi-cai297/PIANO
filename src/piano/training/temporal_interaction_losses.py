"""Tier-0B / v28 temporal interaction losses for PIANO Stage-2.

Per piano_stage2_full_architecture_roadmap.md §7. Five losses that
encode temporal hand-object contact persistence and foot-ground gait
support — failure modes the v27 per-frame anchor MSE could not address.

Contact losses (hand-object):

1. ``loss_contact_rel_offset_smoothl1``  (§7.5)
   SmoothL1 on the object-local relative wrist offset
   ``r = R_obj.T @ (wrist - obj_pos)``, masked to hand-contact frames.
   Avoids the joint-centre vs object-surface 30 cm geometric offset
   that broke the absolute ``contact_target_xyz`` loss (Round-24
   diagnostic).

2. ``loss_contact_drift_smoothl1``  (§7.6)
   SmoothL1 on the segment-level drift of the object-local offset:
   ``(r_pred[t] - r_pred[t0]) - (r_gt[t] - r_gt[t0])`` for t in a
   contact segment [t0, t1]. Penalises predictions that start near
   the object then drift away while GT stays close.

3. ``loss_contact_tracking_projection``  (§7.7)
   ``relu(track_gt - track_pred - margin)^2`` along the object
   displacement unit vector. Targets the "hand follows box only
   partway" failure that absolute / drift losses do not directly
   encode.

Gait losses (foot-ground):

4. ``loss_gait_both_airborne``  (§7.8)
   Penalises ``(1 - L_grounded) * (1 - R_grounded)`` on walking
   frames, with grounded probability derived from predicted ankle
   height vs a sample-specific floor estimate.

5. ``loss_gait_stance_velocity``  (§7.9)
   Penalises stance-foot horizontal velocity on walking frames.
   Blocks the trivial "both feet low but sliding" solution.

Conventions
-----------

- SMPL-22 joint indices (pelvis=0, ankles=7/8, wrists=20/21).
- ``contact_state[:, :, 0:2]`` = hand contact mask (L, R).
- ``object_rotations`` is world-frame axis-angle ``(B, T, 3)``.
- Up axis = Y.
- ``walking_mask`` and ``foot_stance_gt`` are derived from GT joints
  by ``piano.data.interaction_hint`` (NOT InterAct foot-object
  pseudo-labels — roadmap §16-3).
- All losses return a 0-D scalar tensor. They internally clamp the
  mask-frame count with ``.clamp_min(1.0)`` to stay finite on batches
  with zero contact / zero walking frames.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


# Joint indices — duplicated from ``piano.data.interaction_hint`` so this
# module has no inbound dep on the data side (avoids cycles + makes the
# losses importable in tests without dataset code).
LEFT_WRIST_IDX: int = 20
RIGHT_WRIST_IDX: int = 21
LEFT_ANKLE_IDX: int = 7
RIGHT_ANKLE_IDX: int = 8
ROOT_IDX: int = 0
LEFT_KNEE_IDX: int = 4
RIGHT_KNEE_IDX: int = 5
NECK_IDX: int = 12
CONTACT_LEFT_HAND_COL: int = 0
CONTACT_RIGHT_HAND_COL: int = 1
BODY_ACTION_KEY_JOINT_INDICES: tuple[int, ...] = (
    LEFT_WRIST_IDX,
    RIGHT_WRIST_IDX,
    LEFT_KNEE_IDX,
    RIGHT_KNEE_IDX,
    NECK_IDX,
    ROOT_IDX,
)


@dataclass(slots=True)
class TemporalInteractionLossConfig:
    """Per-term weights. Set to 0 to disable a term."""

    contact_rel_offset_weight: float = 0.0
    contact_drift_weight: float = 0.0
    contact_tracking_weight: float = 0.0
    gait_both_airborne_weight: float = 0.0
    gait_stance_velocity_weight: float = 0.0

    # Round-28 consistency losses (prompt §7.3 / §7.4). Small per-frame
    # SmoothL1 between pred-derived signal and oracle hint target on the
    # masked subset (hand_contact frames or active body-action joints).
    # Recommended initial weights: 0.25 - 0.5; never start at 1.0.
    hint_contact_consistency_weight: float = 0.0
    body_action_consistency_weight: float = 0.0

    # Hand-object thresholds.
    contact_threshold: float = 0.5
    """Frames with hand_contact < threshold are excluded."""

    contact_rel_clamp_m: float = 2.0
    """Per-element clamp on |r| (object-local offset) before SmoothL1.
    Prevents a single bad frame's huge magnitude from dominating."""

    tracking_margin_m: float = 0.03
    """Slack in the tracking projection loss; pred only needs to keep
    up with GT minus this margin."""

    tracking_min_obj_disp_m: float = 0.05
    """Only score tracking on frames where the object actually moved
    more than this from segment start (avoids divide-by-tiny noise on
    near-stationary objects)."""

    # Gait thresholds.
    floor_quantile: float = 0.05
    """Quantile of GT ankle y used as the sample-specific floor."""

    grounded_threshold_above_floor_m: float = 0.10
    grounded_softness_m: float = 0.03
    """Sigmoid: grounded_prob = sigmoid((threshold - h) / softness)."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _axis_angle_to_matrix_t(aa: Tensor) -> Tensor:
    """Rodrigues on a ``(..., 3)`` axis-angle tensor.

    Matches ``piano.utils.canonical_frame.axis_angle_to_matrix_np``
    semantics (NaN-safe at θ ≈ 0). Stays in torch so gradients can flow
    if the caller ever differentiates through object pose (we do not
    today; object_rotations is treated as a known condition).
    """
    theta = torch.linalg.norm(aa, dim=-1, keepdim=True)                 # (..., 1)
    safe_theta = torch.where(theta < 1e-8, torch.ones_like(theta), theta)
    axis = aa / safe_theta                                              # (..., 3)
    cos_t = torch.cos(theta).unsqueeze(-1)                              # (..., 1, 1)
    sin_t = torch.sin(theta).unsqueeze(-1)
    one_m_cos = 1.0 - cos_t

    zeros = torch.zeros_like(axis[..., 0])
    K = torch.stack(
        [
            torch.stack([zeros, -axis[..., 2], axis[..., 1]], dim=-1),
            torch.stack([axis[..., 2], zeros, -axis[..., 0]], dim=-1),
            torch.stack([-axis[..., 1], axis[..., 0], zeros], dim=-1),
        ],
        dim=-2,
    )                                                                   # (..., 3, 3)
    eye = torch.eye(3, dtype=aa.dtype, device=aa.device).expand_as(K)
    K2 = torch.matmul(K, K)
    R = eye + sin_t * K + one_m_cos * K2

    # NaN-safe: at θ ≈ 0, return identity.
    near_zero = (theta < 1e-8).unsqueeze(-1).expand_as(R)
    R = torch.where(near_zero, eye, R)
    return R


def _wrist_world_pred_gt(
    pred_joints: Tensor,    # (B, T, 22, 3)
    gt_joints: Tensor,      # (B, T, 22, 3)
) -> tuple[Tensor, Tensor]:
    """Stack L+R wrist for both pred and GT."""
    pw = torch.stack(
        [pred_joints[..., LEFT_WRIST_IDX, :], pred_joints[..., RIGHT_WRIST_IDX, :]],
        dim=2,
    )                                                                   # (B, T, 2, 3)
    gw = torch.stack(
        [gt_joints[..., LEFT_WRIST_IDX, :], gt_joints[..., RIGHT_WRIST_IDX, :]],
        dim=2,
    )
    return pw, gw


def _wrist_object_local(
    wrist_world: Tensor,        # (B, T, 2, 3)
    obj_pos_world: Tensor,      # (B, T, 3)
    R_obj_world: Tensor,        # (B, T, 3, 3)
) -> Tensor:
    """world -> object-local: ``r = R_obj.T @ (wrist - obj_pos)``."""
    R_T = R_obj_world.transpose(-1, -2)                                 # (B, T, 3, 3)
    delta = wrist_world - obj_pos_world.unsqueeze(-2)                   # (B, T, 2, 3)
    rel = torch.einsum("btij,bthj->bthi", R_T, delta)
    return rel                                                          # (B, T, 2, 3)


def _hand_contact_mask(
    contact_state: Tensor,      # (B, T, 5)
    threshold: float,
    seq_mask: Tensor | None,    # (B, T) bool/float, optional valid-frame mask
) -> Tensor:
    """Binary (B, T, 2) mask: 1 where hand_contact > threshold AND
    frame is within seq_len (when seq_mask is provided)."""
    hand_contact = contact_state[..., [CONTACT_LEFT_HAND_COL, CONTACT_RIGHT_HAND_COL]]
    mask = (hand_contact > threshold).to(dtype=contact_state.dtype)
    if seq_mask is not None:
        if seq_mask.dim() == 2:
            sm = seq_mask.to(dtype=mask.dtype).unsqueeze(-1)            # (B, T, 1)
        else:
            sm = seq_mask.to(dtype=mask.dtype)
        mask = mask * sm
    return mask                                                         # (B, T, 2) float


# ---------------------------------------------------------------------------
# Contact losses
# ---------------------------------------------------------------------------


def loss_contact_rel_offset_smoothl1(
    pred_joints: Tensor,        # (B, T, 22, 3)
    gt_joints: Tensor,
    object_positions: Tensor,   # (B, T, 3)
    object_rotations: Tensor,   # (B, T, 3) axis-angle
    contact_state: Tensor,      # (B, T, 5)
    cfg: TemporalInteractionLossConfig,
    seq_mask: Tensor | None = None,
) -> Tensor:
    """§7.5. SmoothL1 on object-local wrist offset, masked to
    hand-contact frames.

    Replaces the legacy absolute ``pred_joint - contact_target_xyz``
    loss whose target was 30 cm away from any reachable GT joint
    position.
    """
    R_obj = _axis_angle_to_matrix_t(object_rotations.to(pred_joints.dtype))
    pw, gw = _wrist_world_pred_gt(pred_joints, gt_joints)
    r_pred = _wrist_object_local(pw, object_positions, R_obj)
    r_gt = _wrist_object_local(gw, object_positions, R_obj)

    # Clamp magnitude to avoid one-frame catastrophes (matches
    # AnchorConsistencyConfig.max_distance_m pattern).
    clamp = float(cfg.contact_rel_clamp_m)
    r_pred = r_pred.clamp(-clamp, clamp)
    r_gt = r_gt.clamp(-clamp, clamp)

    mask = _hand_contact_mask(contact_state, cfg.contact_threshold, seq_mask)
    mask3 = mask.unsqueeze(-1)                                          # (B, T, 2, 1)
    diff = F.smooth_l1_loss(r_pred, r_gt, reduction="none")             # (B, T, 2, 3)
    diff = diff.sum(dim=-1, keepdim=True)                               # (B, T, 2, 1) sum xyz
    num = (diff * mask3).sum()
    den = mask3.sum().clamp_min(1.0)
    return num / den


def loss_contact_drift_smoothl1(
    pred_joints: Tensor,
    gt_joints: Tensor,
    object_positions: Tensor,
    object_rotations: Tensor,
    contact_state: Tensor,
    cfg: TemporalInteractionLossConfig,
    seq_mask: Tensor | None = None,
) -> Tensor:
    """§7.6. SmoothL1 on segment-level drift of object-local offset.

    For each contact segment ``[t0, t1]`` of a given hand:

        drift_pred[t] = r_pred[t] - r_pred[t0]
        drift_gt[t]   = r_gt[t]   - r_gt[t0]
        loss = SmoothL1(drift_pred - drift_gt)

    Implemented vectorised across all (b, hand) sequences by anchoring
    each contact-run's first contact frame as t0. Anchor is recomputed
    independently per hand and per batch element.
    """
    R_obj = _axis_angle_to_matrix_t(object_rotations.to(pred_joints.dtype))
    pw, gw = _wrist_world_pred_gt(pred_joints, gt_joints)
    r_pred = _wrist_object_local(pw, object_positions, R_obj).clamp(
        -cfg.contact_rel_clamp_m, cfg.contact_rel_clamp_m,
    )                                                                   # (B, T, 2, 3)
    r_gt = _wrist_object_local(gw, object_positions, R_obj).clamp(
        -cfg.contact_rel_clamp_m, cfg.contact_rel_clamp_m,
    )

    mask = _hand_contact_mask(contact_state, cfg.contact_threshold, seq_mask)
    # Identify each contact-run's first frame: the frame in which mask
    # transitions 0 -> 1. We then segment-cumsum the run id and look
    # up r_pred / r_gt at each segment's first-contact-frame index.
    B, T, H = mask.shape                                                # H = 2
    mask_b = (mask > 0.5)                                               # (B, T, 2) bool
    prev = torch.zeros_like(mask_b)
    prev[:, 1:, :] = mask_b[:, :-1, :]
    run_start = mask_b & (~prev)                                        # (B, T, 2) bool

    # Per-hand cumulative segment id. seg_id[b, t, h] = how many runs of
    # contact have STARTED by frame t (inclusive). Frames with mask=0
    # therefore have seg_id pointing at the LAST run, which is fine —
    # we mask those out at the loss-aggregation step.
    seg_id = run_start.cumsum(dim=1).to(torch.long)                     # (B, T, 2)

    # First-frame-of-segment time index per (b, t, h). For frames that
    # belong to no segment (seg_id == 0), we set anchor_t = t (so drift
    # is 0); the loss is masked anyway.
    safe_seg_id = seg_id.clamp_min(1)                                   # (B, T, 2)

    # Build (B, n_segs_max, 2) lookup: time index of each segment's start.
    # ``n_segs_max`` is at most T // 2 + 1 since runs alternate with gaps.
    # Use a sentinel for "no such segment" slots.
    # Allocate at least 2 columns so ``safe_seg_id.clamp_min(1)`` is
    # always a valid gather index (handles the all-zero-contact batch).
    n_segs_max = max(int(seg_id.amax().item()) + 1, 2)
    # Gather start-times: for each (b, h), the first occurrence of each
    # run_start. We do this with a small per-hand-per-batch sort:
    #   collect (b, h, seg_id, t) for run_start positions, then scatter.
    start_t = torch.zeros(B, n_segs_max, H, dtype=torch.long, device=mask.device)
    # We anchor seg_id=0 (the pre-first-run state) at t=0 too; doesn't
    # affect loss because those frames are masked out.
    rs_idx = run_start.nonzero(as_tuple=False)                          # (N_runs, 3) [b, t, h]
    if rs_idx.numel() > 0:
        bb, tt, hh = rs_idx[:, 0], rs_idx[:, 1], rs_idx[:, 2]
        # seg_id at a run_start frame == segment number for that run (1-based).
        sid = seg_id[bb, tt, hh]                                        # (N_runs,)
        start_t[bb, sid, hh] = tt

    # For every frame we anchor at start_t[b, seg_id, h].
    anchor_t = start_t.gather(1, safe_seg_id)                           # (B, T, 2)

    # Gather r_pred / r_gt at anchor frames per hand: need
    # r[b, anchor_t[b, t, h], h, :]. Expand anchor_t to (B, T, 2, 3).
    anchor_t_exp = anchor_t.unsqueeze(-1).expand(B, T, 2, 3)            # (B, T, 2, 3)
    # Reshape r_pred / r_gt to (B, T, 2, 3); gather along time axis = 1.
    r_pred_anchor = torch.gather(r_pred, 1, anchor_t_exp)               # (B, T, 2, 3)
    r_gt_anchor = torch.gather(r_gt, 1, anchor_t_exp)

    drift_pred = r_pred - r_pred_anchor
    drift_gt = r_gt - r_gt_anchor

    diff = F.smooth_l1_loss(drift_pred, drift_gt, reduction="none").sum(
        dim=-1, keepdim=True,
    )                                                                   # (B, T, 2, 1)
    mask3 = mask.unsqueeze(-1)
    num = (diff * mask3).sum()
    den = mask3.sum().clamp_min(1.0)
    return num / den


def loss_contact_tracking_projection(
    pred_joints: Tensor,
    gt_joints: Tensor,
    object_positions: Tensor,
    object_rotations: Tensor,  # noqa: ARG001 — kept for signature parity / future use
    contact_state: Tensor,
    cfg: TemporalInteractionLossConfig,
    seq_mask: Tensor | None = None,
) -> Tensor:
    """§7.7. Penalise pred hand for failing to follow GT-relative
    object motion along the object displacement direction.

    For each contact segment [t0, t1] of each hand:

        obj_disp[t]  = obj_pos[t] - obj_pos[t0]
        u[t]         = obj_disp[t] / |obj_disp[t]|  (with mask |obj_disp|>tau)

        hand_disp_pred[t] = pred_wrist[t] - pred_wrist[t0]   (world)
        hand_disp_gt[t]   = gt_wrist[t]   - gt_wrist[t0]

        track_pred[t] = <hand_disp_pred[t], u[t]>
        track_gt[t]   = <hand_disp_gt[t],   u[t]>

        loss = mean( relu(track_gt - track_pred - margin)^2 )

    The asymmetric ReLU only penalises the "hand fell behind GT"
    direction — there is no penalty for pred over-following the
    object (which would itself be GT-faithful).
    """
    pw, gw = _wrist_world_pred_gt(pred_joints, gt_joints)               # (B, T, 2, 3)
    mask = _hand_contact_mask(contact_state, cfg.contact_threshold, seq_mask)
    B, T, H = mask.shape
    mask_b = (mask > 0.5)
    prev = torch.zeros_like(mask_b)
    prev[:, 1:, :] = mask_b[:, :-1, :]
    run_start = mask_b & (~prev)
    seg_id = run_start.cumsum(dim=1).to(torch.long)
    # Allocate at least 2 columns so ``safe_seg_id.clamp_min(1)`` is
    # always a valid gather index (handles the all-zero-contact batch).
    n_segs_max = max(int(seg_id.amax().item()) + 1, 2)

    # Per-hand start-time table.
    start_t_h = torch.zeros(B, n_segs_max, H, dtype=torch.long, device=mask.device)
    rs_idx = run_start.nonzero(as_tuple=False)
    if rs_idx.numel() > 0:
        bb, tt, hh = rs_idx[:, 0], rs_idx[:, 1], rs_idx[:, 2]
        start_t_h[bb, seg_id[bb, tt, hh], hh] = tt

    # Object-displacement is hand-independent — but each (b, hand) may
    # have a different segment start. Build per-hand obj anchor.
    safe_seg_id = seg_id.clamp_min(1)
    anchor_t_h = start_t_h.gather(1, safe_seg_id)                       # (B, T, 2)
    anchor_t_exp_obj = anchor_t_h.unsqueeze(-1).expand(B, T, 2, 3)      # (B, T, 2, 3)
    obj_pos_exp = object_positions.unsqueeze(2).expand(B, T, 2, 3)      # (B, T, 2, 3)
    obj_anchor = torch.gather(obj_pos_exp, 1, anchor_t_exp_obj)         # (B, T, 2, 3)
    obj_disp = obj_pos_exp - obj_anchor                                 # (B, T, 2, 3)
    obj_disp_norm = obj_disp.norm(dim=-1, keepdim=True)                 # (B, T, 2, 1)
    # Unit vector with a noise floor; outside the "object moved enough"
    # mask we just zero u out — its dot products will then be 0.
    u = obj_disp / obj_disp_norm.clamp_min(1e-8)                        # (B, T, 2, 3)
    mover_mask = (obj_disp_norm.squeeze(-1) > cfg.tracking_min_obj_disp_m).to(mask.dtype)

    # Hand displacements anchored at same start frame.
    pw_anchor = torch.gather(pw, 1, anchor_t_exp_obj)                   # (B, T, 2, 3)
    gw_anchor = torch.gather(gw, 1, anchor_t_exp_obj)
    hand_disp_pred = pw - pw_anchor
    hand_disp_gt = gw - gw_anchor

    track_pred = (hand_disp_pred * u).sum(dim=-1)                       # (B, T, 2)
    track_gt = (hand_disp_gt * u).sum(dim=-1)
    margin = float(cfg.tracking_margin_m)
    penalty = F.relu(track_gt - track_pred - margin).pow(2)             # (B, T, 2)

    effective_mask = mask * mover_mask                                  # (B, T, 2)
    num = (penalty * effective_mask).sum()
    den = effective_mask.sum().clamp_min(1.0)
    return num / den


# ---------------------------------------------------------------------------
# Gait losses
# ---------------------------------------------------------------------------


def _per_clip_floor_y(
    gt_joints: Tensor,          # (B, T, 22, 3)
    seq_mask: Tensor | None,    # (B, T) bool/float, optional
    quantile: float,
) -> Tensor:
    """Sample-specific floor estimate per batch element.

    Uses the chosen quantile of (L_ankle_y, R_ankle_y) across the valid
    frames of each clip. ``torch.quantile`` cannot run on a per-row
    variable-length axis with a single call, so we fall back to a small
    Python loop over batch. B is at most ~64 in our trainer; cheap.
    """
    B, T, _, _ = gt_joints.shape
    out = gt_joints.new_zeros(B)
    for b in range(B):
        if seq_mask is not None:
            valid = (seq_mask[b] > 0.5)
            if valid.dim() == 0 or not valid.any():
                out[b] = 0.0
                continue
            yy = torch.cat(
                [
                    gt_joints[b, valid, LEFT_ANKLE_IDX, 1],
                    gt_joints[b, valid, RIGHT_ANKLE_IDX, 1],
                ]
            )
        else:
            yy = torch.cat(
                [
                    gt_joints[b, :, LEFT_ANKLE_IDX, 1],
                    gt_joints[b, :, RIGHT_ANKLE_IDX, 1],
                ]
            )
        if yy.numel() == 0:
            out[b] = 0.0
        else:
            out[b] = torch.quantile(yy, float(quantile))
    return out                                                          # (B,)


def loss_gait_both_airborne(
    pred_joints: Tensor,        # (B, T, 22, 3)
    gt_joints: Tensor,          # (B, T, 22, 3) — for the floor estimate
    walking_mask: Tensor,       # (B, T, 1) float in {0, 1}
    cfg: TemporalInteractionLossConfig,
    seq_mask: Tensor | None = None,
) -> Tensor:
    """§7.8. Penalty for both feet airborne during walking.

    ``grounded_prob = sigmoid((floor_y + threshold - ankle_y) / softness)``

    Returns 0 if no walking frames in batch (clamp denom).
    """
    floor_y = _per_clip_floor_y(gt_joints, seq_mask, cfg.floor_quantile)  # (B,)
    threshold = float(cfg.grounded_threshold_above_floor_m)
    softness = float(cfg.grounded_softness_m)

    l_y = pred_joints[..., LEFT_ANKLE_IDX, 1]                           # (B, T)
    r_y = pred_joints[..., RIGHT_ANKLE_IDX, 1]
    L_grounded = torch.sigmoid((floor_y.unsqueeze(-1) + threshold - l_y) / softness)
    R_grounded = torch.sigmoid((floor_y.unsqueeze(-1) + threshold - r_y) / softness)

    both_airborne = (1.0 - L_grounded) * (1.0 - R_grounded)             # (B, T)
    wm = walking_mask.squeeze(-1).to(pred_joints.dtype)                 # (B, T)
    if seq_mask is not None:
        wm = wm * seq_mask.to(pred_joints.dtype)
    num = (wm * both_airborne).sum()
    den = wm.sum().clamp_min(1.0)
    return num / den


def loss_gait_stance_velocity(
    pred_joints: Tensor,        # (B, T, 22, 3)
    foot_stance_gt: Tensor,     # (B, T, 2) in [0, 1] — from interaction_hint
    walking_mask: Tensor,       # (B, T, 1) in {0, 1}
    fps: float = 20.0,
    seq_mask: Tensor | None = None,
) -> Tensor:
    """§7.9. Penalty for stance-foot horizontal velocity on walking
    frames. Blocks the "both feet low but sliding" trivial solution.

    Loss = mean( walking * stance_gt * ||ankle_xz_vel||^2 ).
    """
    # XZ frame-difference (ankle_xz[t] - ankle_xz[t-1]). First frame
    # has zero velocity (matches piano.utils.smpl_utils convention).
    l_xz = pred_joints[..., LEFT_ANKLE_IDX, :][..., [0, 2]]             # (B, T, 2)
    r_xz = pred_joints[..., RIGHT_ANKLE_IDX, :][..., [0, 2]]
    l_v = torch.zeros_like(l_xz)
    r_v = torch.zeros_like(r_xz)
    l_v[:, 1:] = (l_xz[:, 1:] - l_xz[:, :-1]) * float(fps)
    r_v[:, 1:] = (r_xz[:, 1:] - r_xz[:, :-1]) * float(fps)
    v2 = torch.stack([l_v.pow(2).sum(-1), r_v.pow(2).sum(-1)], dim=-1)  # (B, T, 2)

    wm = walking_mask.squeeze(-1).to(pred_joints.dtype).unsqueeze(-1)   # (B, T, 1)
    if seq_mask is not None:
        wm = wm * seq_mask.to(pred_joints.dtype).unsqueeze(-1)
    stance = foot_stance_gt.to(pred_joints.dtype)                       # (B, T, 2)
    weight = wm * stance                                                # (B, T, 2)

    num = (weight * v2).sum()
    den = weight.sum().clamp_min(1.0)
    return num / den


# ---------------------------------------------------------------------------
# Convenience aggregator
# ---------------------------------------------------------------------------


def compute_all_temporal_losses(
    pred_joints: Tensor,
    gt_joints: Tensor,
    object_positions: Tensor,
    object_rotations: Tensor,
    contact_state: Tensor,
    walking_mask: Tensor,
    foot_stance_gt: Tensor,
    cfg: TemporalInteractionLossConfig,
    seq_mask: Tensor | None = None,
    fps: float = 20.0,
) -> dict[str, Tensor]:
    """One-shot compute that returns a dict of {name → 0-D tensor}.

    Trainer can pick which terms to weight by the per-term ``*_weight``
    fields of ``cfg``. The dict always has all 5 keys so logging is
    consistent across configs.
    """
    return {
        "loss_contact_rel_offset": loss_contact_rel_offset_smoothl1(
            pred_joints, gt_joints, object_positions, object_rotations,
            contact_state, cfg, seq_mask=seq_mask,
        ),
        "loss_contact_drift": loss_contact_drift_smoothl1(
            pred_joints, gt_joints, object_positions, object_rotations,
            contact_state, cfg, seq_mask=seq_mask,
        ),
        "loss_contact_tracking_projection": loss_contact_tracking_projection(
            pred_joints, gt_joints, object_positions, object_rotations,
            contact_state, cfg, seq_mask=seq_mask,
        ),
        "loss_gait_both_airborne": loss_gait_both_airborne(
            pred_joints, gt_joints, walking_mask, cfg, seq_mask=seq_mask,
        ),
        "loss_gait_stance_velocity": loss_gait_stance_velocity(
            pred_joints, foot_stance_gt, walking_mask, fps=fps, seq_mask=seq_mask,
        ),
    }


# ===========================================================================
# Round-28 consistency losses (prompt §7.3 / §7.4)
# ===========================================================================


def loss_hint_contact_consistency(
    pred_joints: Tensor,                # (B, T, 22, 3)
    oracle_interaction_hint: Tensor,    # (B, T, D_hint), D_hint ∈ {8, 13}
    object_positions: Tensor,           # (B, T, 3)
    object_rotations: Tensor,           # (B, T, 3) axis-angle
    contact_state: Tensor,              # (B, T, 5)
    cfg: TemporalInteractionLossConfig,
    seq_mask: Tensor | None = None,
    hand_offset_clamp_m: float = 2.0,
) -> Tensor:
    """§7.3. SmoothL1 between predicted object-local wrist offset and the
    ``hand_object_local_offset`` channel of the oracle interaction hint,
    masked to hand-contact frames.

    Layout of ``oracle_interaction_hint`` (per ``interaction_hint.py``
    ``build_oracle_interaction_hint`` doc):

        [:, :, 0:2]  hand_contact_mask        (L, R)
        [:, :, 2:8]  hand_object_local_offset (L_xyz, R_xyz) — clamped
                     then scaled to [-1, 1] by ``hand_offset_clamp_m``.

    The "hand" or "full" variants share this slice layout, so the loss
    works for either as long as ``D_hint >= 8``.

    Smaller than the T0-B contact losses; this is a CONSISTENCY pull
    toward what we asked the model to consume, not a hard pose pull.
    """
    if oracle_interaction_hint.shape[-1] < 8:
        raise ValueError(
            "loss_hint_contact_consistency requires the 'hand' or 'full' "
            "oracle hint variant (D_hint >= 8); got "
            f"D_hint={oracle_interaction_hint.shape[-1]}"
        )
    # Compute pred object-local offset on the same scheme as the hint.
    R_obj = _axis_angle_to_matrix_t(object_rotations.to(pred_joints.dtype))
    pw, _ = _wrist_world_pred_gt(pred_joints, pred_joints)                # ignore GT
    r_pred = _wrist_object_local(pw, object_positions, R_obj)             # (B, T, 2, 3)
    r_pred = r_pred.clamp(-hand_offset_clamp_m, hand_offset_clamp_m)
    r_pred_scaled = r_pred / float(hand_offset_clamp_m)                   # match hint scaling

    # Pull the hint slice and reshape (B, T, 2, 3).
    hint_off = oracle_interaction_hint[..., 2:8].reshape(
        *oracle_interaction_hint.shape[:2], 2, 3,
    )                                                                     # (B, T, 2, 3)

    mask = _hand_contact_mask(contact_state, cfg.contact_threshold, seq_mask)
    mask3 = mask.unsqueeze(-1)                                            # (B, T, 2, 1)
    diff = F.smooth_l1_loss(r_pred_scaled, hint_off, reduction="none")    # (B, T, 2, 3)
    diff = diff.sum(dim=-1, keepdim=True)                                 # (B, T, 2, 1)
    num = (diff * mask3).sum()
    den = mask3.sum().clamp_min(1.0)
    return num / den


def loss_body_action_consistency(
    pred_joints: Tensor,                # (B, T, 22, 3)
    body_action_hint: Tensor,           # (B, T, 24)
    seq_mask: Tensor | None = None,
) -> Tensor:
    """§7.4. SmoothL1 between predicted six-joint body-action delta and
    the ``body_action_hint``'s GT delta slice, weighted by the hint's
    joint mask.

    The hint layout (``build_body_action_oracle_hint``) is:

        [:, :, 0:6]   joint_mask[6] — broadcast per joint (per-clip mask
                                       repeated across T).
        [:, :, 6:24]  joint_delta_local[6, 3] flat (root-yaw-canonical
                                       pelvis-local for non-pelvis joints,
                                       displacement-from-frame-0 for pelvis).

    To keep this loss differentiable through ``pred_joints``, the
    pred-side delta uses the SAME root-yaw-canonical definition as the
    numpy oracle hint builder: non-pelvis joints are pelvis-translated,
    rotated by frame-0 facing yaw, and differenced from frame 0; pelvis
    uses displacement from frame-0 pelvis in the same canonical yaw
    frame.
    """
    if body_action_hint.shape[-1] < 24:
        raise ValueError(
            f"body_action_hint must have 24 channels; got {body_action_hint.shape[-1]}"
        )
    B, T = pred_joints.shape[0], pred_joints.shape[1]
    J = len(BODY_ACTION_KEY_JOINT_INDICES)
    pelvis = pred_joints[..., ROOT_IDX, :]                                # (B, T, 3)
    R_root0_T = _root0_world_to_canonical_yaw_matrix(pred_joints)          # (B, 3, 3)

    # Build (B, T, J, 3) of pelvis-translated joints + pelvis trace.
    deltas = []
    for j_pos, j_idx in enumerate(BODY_ACTION_KEY_JOINT_INDICES[:-1]):
        jw = pred_joints[..., j_idx, :]                                   # (B, T, 3)
        rel = jw - pelvis                                                 # (B, T, 3)
        rel = torch.matmul(rel, R_root0_T.transpose(-1, -2))               # (B, T, 3)
        rel_anchor = rel[:, 0:1, :]                                       # (B, 1, 3)
        deltas.append(rel - rel_anchor)
    # Pelvis: displacement from frame 0 in the same canonical-yaw frame.
    pelvis_anchor = pelvis[:, 0:1, :]                                     # (B, 1, 3)
    pelvis_delta = torch.matmul(
        pelvis - pelvis_anchor,
        R_root0_T.transpose(-1, -2),
    )
    deltas.append(pelvis_delta)
    pred_delta = torch.stack(deltas, dim=2)                               # (B, T, J, 3)

    # Both tensors are in the same root-yaw-canonical frame.
    hint_mask = body_action_hint[..., :J]                                 # (B, T, J)
    hint_delta = body_action_hint[..., J:].reshape(B, T, J, 3)            # (B, T, J, 3)

    if seq_mask is not None:
        seq_mask_btj = seq_mask.float().unsqueeze(-1)                     # (B, T, 1)
        mask = (hint_mask * seq_mask_btj).unsqueeze(-1)                   # (B, T, J, 1)
    else:
        mask = hint_mask.unsqueeze(-1)                                    # (B, T, J, 1)

    diff = F.smooth_l1_loss(pred_delta, hint_delta, reduction="none")     # (B, T, J, 3)
    diff = diff.sum(dim=-1, keepdim=True)                                 # (B, T, J, 1)
    num = (diff * mask).sum()
    den = mask.sum().clamp_min(1.0)
    return num / den


def _root0_world_to_canonical_yaw_matrix(pred_joints: Tensor) -> Tensor:
    """Torch equivalent of interaction_hint._facing_angle_y + R_y(-yaw)."""
    j0 = pred_joints[:, 0]                                                # (B, 22, 3)
    across = (j0[:, 17] - j0[:, 16]) + (j0[:, 2] - j0[:, 1])
    forward_x = -across[:, 2]
    forward_z = across[:, 0]
    yaw = torch.atan2(forward_x, forward_z)
    angle = -yaw
    c = torch.cos(angle)
    s = torch.sin(angle)
    zeros = torch.zeros_like(c)
    ones = torch.ones_like(c)
    return torch.stack(
        [
            torch.stack([c, zeros, s], dim=-1),
            torch.stack([zeros, ones, zeros], dim=-1),
            torch.stack([-s, zeros, c], dim=-1),
        ],
        dim=-2,
    )
