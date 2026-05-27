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

    # Round-29 loss-strategy ablation (analyses/2026-05-27_round29_loss_strategy_ablation_prompt_for_claude_code.md).
    # These three losses replace absolute-GT anchor losses with condition-
    # consistency losses: instead of pulling joints toward GT world
    # positions (which over-penalises equivalent valid modes like
    # left-hand-first vs right-hand-first), they pull pred motion toward
    # the geometric relations that the R29 I3/S4 conditions specify.
    # Recommended initial weights: 0.25 - 0.75 (see prompt §7.2).
    r29_interaction_consistency_weight: float = 0.0
    r29_support_both_airborne_weight: float = 0.0
    r29_support_stance_velocity_weight: float = 0.0

    # Round-29 swing clearance loss (per Codex review of v1 loss-strategy
    # ablation, 2026-05-27). Penalises low ankle height during walking
    # frames where the foot is NOT in stance (i.e. swing foot must lift).
    # Targets the "both feet planted" pathology v1 produced when
    # both_airborne suppressed double-airborne without ensuring stepping.
    # Active only on walking_mask * (1 - foot_stance) frames.
    r29_swing_clearance_weight: float = 0.0
    r29_swing_clearance_m: float = 0.05
    """Minimum ankle-above-floor height (metres) for a swing foot.
    Frames where the ankle is below this contribute relu(clearance - h)^2."""

    # Round-29 failure-targeted ablation R2 — behavior-level gait losses.
    # Per analyses/2026-05-27_round29_failure_targeted_ablation_prompt_for_claude_code.md
    # §R2: use S4 only for walking_mask + validity, NOT for GT left/right
    # phase target. Respects multimodal left/right equivalence.
    r29_gait_one_foot_support_weight: float = 0.0
    r29_gait_pred_stance_velocity_weight: float = 0.0
    r29_gait_ankle_smooth_weight: float = 0.0
    r29_gait_antiphase_corr_weight: float = 0.0
    r29_gait_antiphase_min_walking_frames: int = 10
    """Sequences with fewer walking frames than this contribute zero to
    the antiphase-correlation loss (avoids meaningless 2-frame correlations)."""

    # Round-29 failure-targeted ablation R3 — exact S4 execution losses.
    # Per prompt §R3: BCE against stage2_support[..., 0:2] (left/right
    # stance) + SmoothL1 against stage2_support[..., 9:13] (footstep
    # target local XZ). Mask by walking_mask + seq_mask.
    r29_s4_stance_bce_weight: float = 0.0
    r29_s4_footstep_target_weight: float = 0.0

    # Round-29 next-baseline ablation G1 — phase-free gait losses.
    # Per analyses/2026-05-27_round29_next_ablation_execution_prompt_for_claude_code.md §G1:
    # avoid R2's height-only loophole by combining ankle height with horizontal
    # speed (a foot is "stance-like" only if low AND slow). Use S4 stance
    # channels for target rates / duty / both-state without copying per-frame
    # left/right phase. All four losses are phase-invariant under L<->R swap.
    r29_gait_soft_stance_velocity_weight: float = 0.0
    r29_gait_transition_rate_weight: float = 0.0
    r29_gait_duty_cycle_weight: float = 0.0
    r29_gait_both_state_match_weight: float = 0.0
    r29_gait_soft_stance_speed_threshold_mps: float = 0.30
    r29_gait_soft_stance_speed_softness_mps: float = 0.10
    """Speed probability is ``sigmoid((threshold - speed_mps) / softness)``;
    foot is "slow enough to be stance-like" when speed is below threshold."""

    # Round-29 failure-targeted ablation R4 / R5 — contact-lock losses.
    # Per prompt §R4: keep strong baseline absolute stabilizers, add
    # object-relative offset + segment-drift + tracking losses driven
    # by the I3 (R4) or I5 (R5) interaction condition.
    r29_contact_lock_offset_weight: float = 0.0
    r29_contact_lock_segment_drift_weight: float = 0.0
    r29_contact_lock_tracking_weight: float = 0.0

    # Round-29 interaction-consistency normalization clamp. Must match
    # the value used by ``piano.data.stage2_oracle_conditions.build_interaction_condition``
    # (clamp before normalization → values in [-1, 1]). Configured via
    # ``data.r29_hand_offset_clamp_m`` in the YAML. Pre-existing
    # ``contact_rel_clamp_m`` above is for the R27 contact_rel_offset
    # loss and is independent.
    r29_hand_offset_clamp_m: float = 2.0

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


# ===========================================================================
# Round-29 condition-consistency losses
# Per analyses/2026-05-27_round29_loss_strategy_ablation_prompt_for_claude_code.md
#
# These three losses pull pred motion toward the geometric relations
# specified by the R29 typed conditions (I3 / S4), NOT toward absolute
# GT world joint positions. The motivation: absolute-GT auxiliary
# losses over-penalise equivalent valid modes (e.g. left-hand-first
# vs right-hand-first under an ambiguous condition). By contrast,
# condition-consistency losses ask whether the predicted motion
# *realises the condition*.
#
# Design rule (prompt §3): when the R29 condition specifies side
# (I3 left/right hand, S4 left/right foot), the loss must enforce
# that side. These losses are NOT permutation-invariant — they read
# the side directly from the condition channels.
# ===========================================================================


def _r29_walking_mask_from_support(
    stage2_support: Tensor,                 # (B, T, >=5)
) -> Tensor:
    """Pull the walking_mask channel out of the R29 S-family condition.

    S1/S2/S3/S4 all share the same first 5 channels: ``[L_stance, R_stance,
    L_height_norm, R_height_norm, walking_mask]``. Returns ``(B, T, 1)``
    float in [0, 1].
    """
    if stage2_support.shape[-1] < 5:
        raise ValueError(
            "stage2_support must have at least 5 channels (S1+ layout: "
            "[L_stance, R_stance, L_height_norm, R_height_norm, walking_mask]); "
            f"got dim={stage2_support.shape[-1]}. Use S1/S2/S3/S4 variants."
        )
    return stage2_support[..., 4:5]                                       # (B, T, 1)


def _r29_foot_stance_from_support(
    stage2_support: Tensor,                 # (B, T, >=5)
) -> Tensor:
    """Pull the L/R foot-stance channels out of the R29 S-family condition.

    Returns ``(B, T, 2)`` float in [0, 1] — soft stance probabilities from
    ``derive_foot_stance_from_gt`` (height + xz-velocity sigmoid).
    """
    if stage2_support.shape[-1] < 5:
        raise ValueError(
            "stage2_support must have at least 5 channels for foot_stance "
            "extraction; got dim={stage2_support.shape[-1]}."
        )
    return stage2_support[..., 0:2]                                       # (B, T, 2)


def loss_r29_interaction_consistency(
    pred_joints: Tensor,                # (B, T, 22, 3)
    object_positions: Tensor,           # (B, T, 3)
    object_rotations: Tensor,           # (B, T, 3), axis-angle
    stage2_interaction: Tensor,         # (B, T, 8) — I3 layout only for P0
    cfg: TemporalInteractionLossConfig,
    seq_mask: Tensor | None = None,
    hand_offset_clamp_m: float = 2.0,
) -> Tensor:
    """R29 P0 — SmoothL1 between pred object-local wrist offset and the
    I3 ``target_offset`` channel, masked to contact frames.

    Layout of I3 ``stage2_interaction`` (per
    ``piano.data.stage2_oracle_conditions.build_interaction_condition``):

        [:, :, 0:2]  hand_contact  (L, R) — soft 0..1
        [:, :, 2:8]  target_offset (L_xyz, R_xyz), clamped to ±clamp_m
                     and divided by clamp_m → values in [-1, 1].
                     Multiplied by hand_contact so non-contact frames
                     are exactly zero in the channel.

    Loss:
        pred_offset_norm = clamp(R_obj.T @ (pred_wrist - obj_pos)) / clamp_m
        SmoothL1(pred_offset_norm, stage2_interaction[..., 2:8])
        averaged over (hand_contact > threshold) frames.

    Unlike ``loss_hint_contact_consistency`` (R28), this loss is
    self-contained on the R29 condition — it does NOT need
    ``contact_state`` from the dataset; the mask comes from the I3
    ``hand_contact`` channel itself.
    """
    expected_dim = 8
    if stage2_interaction.shape[-1] != expected_dim:
        raise ValueError(
            "loss_r29_interaction_consistency P0 supports only the I3 "
            f"layout (dim={expected_dim}); got "
            f"dim={stage2_interaction.shape[-1]}. To enable other I-variants, "
            "extend this loss with their channel layout."
        )

    R_obj = _axis_angle_to_matrix_t(object_rotations.to(pred_joints.dtype))
    pw, _ = _wrist_world_pred_gt(pred_joints, pred_joints)                 # (B, T, 2, 3)
    r_pred = _wrist_object_local(pw, object_positions, R_obj)              # (B, T, 2, 3)
    r_pred = r_pred.clamp(-hand_offset_clamp_m, hand_offset_clamp_m)
    pred_offset_norm = r_pred / float(hand_offset_clamp_m)                 # (B, T, 2, 3)

    hand_contact = stage2_interaction[..., 0:2]                            # (B, T, 2)
    target_offset = stage2_interaction[..., 2:8].reshape(
        *stage2_interaction.shape[:2], 2, 3,
    )                                                                       # (B, T, 2, 3)
    # I3 stores masked_offset = offset_norm * hand_contact. The mask is
    # applied separately below, so active soft-contact frames need the
    # unmasked geometric target.
    target_offset = torch.where(
        hand_contact.unsqueeze(-1) > 0.0,
        target_offset / hand_contact.clamp_min(1e-6).unsqueeze(-1),
        torch.zeros_like(target_offset),
    ).clamp(-1.0, 1.0)

    mask = (hand_contact > cfg.contact_threshold).to(dtype=pred_joints.dtype)
    if seq_mask is not None:
        if seq_mask.dim() == 2:
            sm = seq_mask.to(dtype=mask.dtype).unsqueeze(-1)               # (B, T, 1)
        else:
            sm = seq_mask.to(dtype=mask.dtype)
        mask = mask * sm
    mask3 = mask.unsqueeze(-1)                                              # (B, T, 2, 1)

    diff = F.smooth_l1_loss(pred_offset_norm, target_offset, reduction="none")  # (B, T, 2, 3)
    diff = diff.sum(dim=-1, keepdim=True)                                   # (B, T, 2, 1)
    num = (diff * mask3).sum()
    den = mask3.sum().clamp_min(1.0)
    return num / den


def loss_r29_support_both_airborne(
    pred_joints: Tensor,                # (B, T, 22, 3)
    gt_joints: Tensor,                  # (B, T, 22, 3) — only for floor estimate
    stage2_support: Tensor,             # (B, T, >=5) — walking_mask at [..., 4:5]
    cfg: TemporalInteractionLossConfig,
    seq_mask: Tensor | None = None,
) -> Tensor:
    """R29 P0 — Penalty for both feet airborne on walking frames, using
    the R29 S-family ``walking_mask`` channel instead of a dataset aux
    field.

    Same grounded-probability formula as ``loss_gait_both_airborne``:
        grounded_prob = sigmoid((floor_y + threshold - ankle_y) / softness)
        loss          = mean( walking * (1 - L_g) * (1 - R_g) )

    Returns 0 if no walking frames (clamp denominator).
    """
    walking_mask = _r29_walking_mask_from_support(stage2_support)          # (B, T, 1)
    return loss_gait_both_airborne(
        pred_joints=pred_joints,
        gt_joints=gt_joints,
        walking_mask=walking_mask,
        cfg=cfg,
        seq_mask=seq_mask,
    )


def loss_r29_support_stance_velocity(
    pred_joints: Tensor,                # (B, T, 22, 3)
    stage2_support: Tensor,             # (B, T, >=5)
    fps: float = 20.0,
    seq_mask: Tensor | None = None,
) -> Tensor:
    """R29 P0 — Penalty for stance-foot horizontal velocity on walking
    frames, using R29 S-family ``foot_stance`` and ``walking_mask``
    channels.

    Same formula as ``loss_gait_stance_velocity``, but stance + walking
    come from ``stage2_support`` instead of dataset aux fields.
    """
    foot_stance = _r29_foot_stance_from_support(stage2_support)            # (B, T, 2)
    walking_mask = _r29_walking_mask_from_support(stage2_support)          # (B, T, 1)
    return loss_gait_stance_velocity(
        pred_joints=pred_joints,
        foot_stance_gt=foot_stance,
        walking_mask=walking_mask,
        fps=fps,
        seq_mask=seq_mask,
    )


def loss_r29_swing_clearance(
    pred_joints: Tensor,                # (B, T, 22, 3)
    gt_joints: Tensor,                  # (B, T, 22, 3) — for floor estimate
    stage2_support: Tensor,             # (B, T, >=5)
    cfg: TemporalInteractionLossConfig,
    seq_mask: Tensor | None = None,
) -> Tensor:
    """R29 P0+ — Penalise low swing ankle on walking frames.

    Per Codex review of v1 loss-strategy ablation: the existing
    ``loss_r29_support_both_airborne`` only suppresses double-airborne and
    ``loss_r29_support_stance_velocity`` only freezes stance foot. Neither
    forces the swing foot to actually lift, so the model's easy equilibrium
    becomes "both feet planted" (v1 showed ``frac_both_stance`` going from
    0.16 to 0.44+ when relative_behavior was enabled).

    This loss says: during walking, for any foot that is NOT in stance
    (i.e. the GT-tagged swing foot), its ankle should rise above
    ``cfg.r29_swing_clearance_m`` (default 5 cm above floor).

        floor_y = per-clip quantile(GT ankle y) — same estimate as
                  loss_gait_both_airborne uses (sample-specific floor).
        h_above_floor = ankle_y - floor_y                    (B, T, 2)
        swing_mask = walking_mask * (1 - foot_stance)        (B, T, 2)
        penalty    = relu(clearance - h_above_floor)^2       (B, T, 2)
        loss       = sum(swing_mask * penalty) / sum(swing_mask).clamp_min(1)

    Returns 0 if no swing-during-walking frames in batch.
    """
    if stage2_support.shape[-1] < 5:
        raise ValueError(
            "loss_r29_swing_clearance requires stage2_support dim >= 5 "
            f"(S1+ layout: [L_stance, R_stance, ..., walking_mask]); got "
            f"dim={stage2_support.shape[-1]}."
        )

    floor_y = _per_clip_floor_y(gt_joints, seq_mask, cfg.floor_quantile)    # (B,)
    clearance = float(cfg.r29_swing_clearance_m)

    l_y = pred_joints[..., LEFT_ANKLE_IDX, 1]                              # (B, T)
    r_y = pred_joints[..., RIGHT_ANKLE_IDX, 1]
    h_lr = torch.stack(
        [l_y - floor_y.unsqueeze(-1), r_y - floor_y.unsqueeze(-1)],
        dim=-1,
    )                                                                       # (B, T, 2)

    foot_stance = _r29_foot_stance_from_support(stage2_support).to(pred_joints.dtype)  # (B, T, 2)
    walking = _r29_walking_mask_from_support(stage2_support).to(pred_joints.dtype)     # (B, T, 1)
    # Swing = walking AND NOT stance. ``foot_stance`` is soft in [0, 1],
    # so (1 - foot_stance) is a soft swing weight.
    swing_mask = walking * (1.0 - foot_stance)                              # (B, T, 2)
    if seq_mask is not None:
        sm = seq_mask.to(pred_joints.dtype).unsqueeze(-1)                   # (B, T, 1)
        swing_mask = swing_mask * sm

    penalty = F.relu(clearance - h_lr).pow(2)                               # (B, T, 2)
    num = (swing_mask * penalty).sum()
    den = swing_mask.sum().clamp_min(1.0)
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


# ===========================================================================
# Round-29 failure-targeted ablation losses (R2 behavior gait, R3 exact S4,
# R4/R5 contact-lock). Per
# ``analyses/2026-05-27_round29_failure_targeted_ablation_prompt_for_claude_code.md``.
# ===========================================================================


def _pred_grounded_prob(
    pred_joints: Tensor,                # (B, T, 22, 3)
    gt_joints: Tensor,                  # (B, T, 22, 3) — for floor estimate
    cfg: TemporalInteractionLossConfig,
    seq_mask: Tensor | None = None,
) -> Tensor:
    """Soft grounded probability per ankle from the sample-specific floor.

    ``grounded = sigmoid((floor_y + threshold - ankle_y) / softness)`` —
    matches the convention in ``loss_gait_both_airborne`` /
    ``loss_r29_swing_clearance``. Returns ``(B, T, 2)`` where the last
    dim is (left, right).
    """
    floor_y = _per_clip_floor_y(gt_joints, seq_mask, cfg.floor_quantile)    # (B,)
    threshold = float(cfg.grounded_threshold_above_floor_m)
    softness = float(cfg.grounded_softness_m)
    l_y = pred_joints[..., LEFT_ANKLE_IDX, 1]                              # (B, T)
    r_y = pred_joints[..., RIGHT_ANKLE_IDX, 1]
    L_g = torch.sigmoid((floor_y.unsqueeze(-1) + threshold - l_y) / softness)
    R_g = torch.sigmoid((floor_y.unsqueeze(-1) + threshold - r_y) / softness)
    return torch.stack([L_g, R_g], dim=-1)                                  # (B, T, 2)


# --------------------------------------------------------------------------- #
# R2 — behavior-level gait losses (no GT phase target).
# --------------------------------------------------------------------------- #


def loss_r29_gait_one_foot_support(
    pred_joints: Tensor,                # (B, T, 22, 3)
    gt_joints: Tensor,                  # (B, T, 22, 3) — for floor
    stage2_support: Tensor,             # (B, T, >=5)
    cfg: TemporalInteractionLossConfig,
    seq_mask: Tensor | None = None,
) -> Tensor:
    """R2 — penalise both-airborne and both-stance on walking frames
    without choosing which foot is which.

        loss = mean( walking * (L_g + R_g - 1)^2 )

    L_g / R_g are predicted grounded probabilities (sigmoid of ankle-
    above-floor). This rewards exactly one foot on the ground per frame,
    which is the necessary condition for walking gait, but never picks
    left-first vs right-first (multimodal-safe).
    """
    grounded = _pred_grounded_prob(
        pred_joints, gt_joints, cfg, seq_mask=seq_mask,
    )                                                                       # (B, T, 2)
    walking = _r29_walking_mask_from_support(stage2_support).to(pred_joints.dtype)
    walking = walking.squeeze(-1)                                           # (B, T)
    if seq_mask is not None:
        walking = walking * seq_mask.to(pred_joints.dtype)
    diff = (grounded.sum(dim=-1) - 1.0).pow(2)                              # (B, T)
    num = (walking * diff).sum()
    den = walking.sum().clamp_min(1.0)
    return num / den


def loss_r29_gait_pred_stance_velocity(
    pred_joints: Tensor,                # (B, T, 22, 3)
    gt_joints: Tensor,                  # (B, T, 22, 3) — for floor
    stage2_support: Tensor,             # (B, T, >=5)
    cfg: TemporalInteractionLossConfig,
    fps: float = 20.0,
    seq_mask: Tensor | None = None,
) -> Tensor:
    """R2 — penalise horizontal velocity of whatever foot the model
    chose to plant (soft, via ``pred_grounded_prob``). Discourages
    sliding without copying GT stance assignment.

        weight  = walking * pred_grounded_prob[..., k]           (B, T, 2)
        v_lr    = horizontal ankle velocity                       (B, T, 2)
        loss    = mean( weight * ||v||^2 )
    """
    grounded = _pred_grounded_prob(
        pred_joints, gt_joints, cfg, seq_mask=seq_mask,
    )                                                                       # (B, T, 2)
    walking = _r29_walking_mask_from_support(stage2_support).to(pred_joints.dtype)
    walking = walking.squeeze(-1).unsqueeze(-1)                             # (B, T, 1)
    if seq_mask is not None:
        walking = walking * seq_mask.to(pred_joints.dtype).unsqueeze(-1)

    l_xz = pred_joints[..., LEFT_ANKLE_IDX, :][..., [0, 2]]                 # (B, T, 2)
    r_xz = pred_joints[..., RIGHT_ANKLE_IDX, :][..., [0, 2]]
    l_v = torch.zeros_like(l_xz)
    r_v = torch.zeros_like(r_xz)
    l_v[:, 1:] = (l_xz[:, 1:] - l_xz[:, :-1]) * float(fps)
    r_v[:, 1:] = (r_xz[:, 1:] - r_xz[:, :-1]) * float(fps)
    v2 = torch.stack([l_v.pow(2).sum(-1), r_v.pow(2).sum(-1)], dim=-1)      # (B, T, 2)

    weight = walking * grounded                                              # (B, T, 2)
    num = (weight * v2).sum()
    den = weight.sum().clamp_min(1.0)
    return num / den


def loss_r29_gait_ankle_smooth(
    pred_joints: Tensor,                # (B, T, 22, 3)
    stage2_support: Tensor,             # (B, T, >=5)
    seq_mask: Tensor | None = None,
) -> Tensor:
    """R2 — SmoothL1 on ankle acceleration during walking. Targets
    ankle twist / flicker without copying GT.

        accel[t] = ankle[t+1] - 2*ankle[t] + ankle[t-1]    (B, T-2, 2, 3)
        loss     = mean( walking * SmoothL1(accel, 0) ) over T-2 frames
    """
    l_xyz = pred_joints[..., LEFT_ANKLE_IDX, :]                             # (B, T, 3)
    r_xyz = pred_joints[..., RIGHT_ANKLE_IDX, :]
    ankles = torch.stack([l_xyz, r_xyz], dim=-2)                            # (B, T, 2, 3)
    if ankles.shape[1] < 3:
        return ankles.new_zeros(())
    accel = ankles[:, 2:] - 2.0 * ankles[:, 1:-1] + ankles[:, :-2]          # (B, T-2, 2, 3)
    walking = _r29_walking_mask_from_support(stage2_support).to(pred_joints.dtype)
    walking = walking.squeeze(-1)                                           # (B, T)
    if seq_mask is not None:
        walking = walking * seq_mask.to(pred_joints.dtype)
    # Walking weight for accel[t] uses walking[t+1] (center frame).
    w = walking[:, 1:-1].unsqueeze(-1).unsqueeze(-1)                        # (B, T-2, 1, 1)
    diff = F.smooth_l1_loss(accel, torch.zeros_like(accel), reduction="none")  # (B, T-2, 2, 3)
    diff = diff.sum(dim=-1, keepdim=True)                                   # (B, T-2, 2, 1)
    num = (diff * w).sum()
    # Denominator: count (frame, foot) pairs that are within the walking mask.
    den = (w.expand_as(diff) > 0).to(diff.dtype).sum().clamp_min(1.0)
    return num / den


def loss_r29_gait_antiphase_corr(
    pred_joints: Tensor,                # (B, T, 22, 3)
    stage2_support: Tensor,             # (B, T, >=5)
    cfg: TemporalInteractionLossConfig,
    seq_mask: Tensor | None = None,
) -> Tensor:
    """R2 — masked correlation of L/R ankle height during walking.

    Penalise ``relu(corr + 0.15)`` so corr <= -0.15 is accepted.
    Sequences with fewer than ``cfg.r29_gait_antiphase_min_walking_frames``
    walking frames contribute zero.
    """
    B, T = pred_joints.shape[:2]
    l_y = pred_joints[..., LEFT_ANKLE_IDX, 1]                              # (B, T)
    r_y = pred_joints[..., RIGHT_ANKLE_IDX, 1]
    walking = _r29_walking_mask_from_support(stage2_support).to(pred_joints.dtype)
    walking = walking.squeeze(-1)                                          # (B, T)
    if seq_mask is not None:
        walking = walking * seq_mask.to(pred_joints.dtype)

    min_frames = int(cfg.r29_gait_antiphase_min_walking_frames)
    pieces: list[Tensor] = []
    for b in range(B):
        m = walking[b] > 0.5
        if int(m.sum()) < min_frames:
            continue
        l = l_y[b][m]                                                       # (n,)
        r = r_y[b][m]
        l_c = l - l.mean()
        r_c = r - r.mean()
        denom = (l_c.pow(2).sum().clamp_min(1e-9) * r_c.pow(2).sum().clamp_min(1e-9)).sqrt()
        corr = (l_c * r_c).sum() / denom
        pieces.append(F.relu(corr + 0.15))
    if not pieces:
        return l_y.new_zeros(())
    return torch.stack(pieces).mean()


# --------------------------------------------------------------------------- #
# G1 — phase-free gait losses (no per-frame L/R alignment to GT).
#
# Per analyses/2026-05-27_round29_next_ablation_execution_prompt_for_claude_code.md
# §G1: R2's one-foot-support loss has a height-only loophole — the model
# satisfied it by keeping one foot constantly above the floor (frac_both_swing
# = 0.872 on the val matrix). G1 combines ankle height with horizontal speed
# so a foot is only "stance-like" when low AND slow, then derives transition
# density / duty cycle / both-state from those soft stance probabilities.
#
# All four losses are phase-invariant: swapping L<->R for the prediction does
# not change the loss, because the per-frame label of left vs right is never
# compared. Only aggregate statistics (transition density, sorted duty cycle,
# both-state) are matched to S4 targets.
# --------------------------------------------------------------------------- #


def _pred_soft_stance_prob(
    pred_joints: Tensor,                # (B, T, 22, 3)
    gt_joints: Tensor,                  # (B, T, 22, 3) — for floor estimate
    cfg: TemporalInteractionLossConfig,
    fps: float = 20.0,
    seq_mask: Tensor | None = None,
) -> Tensor:
    """Soft stance probability per ankle. Returns ``(B, T, 2)``.

        height_grounded = sigmoid((floor + threshold - ankle_y) / softness)
        speed_prob      = sigmoid((threshold_mps - speed_mps) / softness_mps)
        soft_stance     = height_grounded * speed_prob

    Velocity is computed on XZ in metres/second (the same fps convention as
    ``loss_r29_gait_pred_stance_velocity``). The shared helper lets the four
    G1 losses share a single soft-stance tensor.
    """
    grounded = _pred_grounded_prob(
        pred_joints, gt_joints, cfg, seq_mask=seq_mask,
    )                                                                       # (B, T, 2)
    l_xz = pred_joints[..., LEFT_ANKLE_IDX, :][..., [0, 2]]                 # (B, T, 2)
    r_xz = pred_joints[..., RIGHT_ANKLE_IDX, :][..., [0, 2]]
    l_v = torch.zeros_like(l_xz)
    r_v = torch.zeros_like(r_xz)
    l_v[:, 1:] = (l_xz[:, 1:] - l_xz[:, :-1]) * float(fps)
    r_v[:, 1:] = (r_xz[:, 1:] - r_xz[:, :-1]) * float(fps)
    l_speed = l_v.pow(2).sum(-1).clamp_min(1e-12).sqrt()                    # (B, T)
    r_speed = r_v.pow(2).sum(-1).clamp_min(1e-12).sqrt()
    speed_mps = torch.stack([l_speed, r_speed], dim=-1)                     # (B, T, 2)
    thr = float(cfg.r29_gait_soft_stance_speed_threshold_mps)
    soft = float(cfg.r29_gait_soft_stance_speed_softness_mps)
    speed_prob = torch.sigmoid((thr - speed_mps) / max(soft, 1e-6))         # (B, T, 2)
    return grounded * speed_prob                                            # (B, T, 2)


def loss_r29_gait_soft_stance_velocity(
    pred_joints: Tensor,                # (B, T, 22, 3)
    gt_joints: Tensor,                  # (B, T, 22, 3) — for floor
    stage2_support: Tensor,             # (B, T, >=5)
    cfg: TemporalInteractionLossConfig,
    fps: float = 20.0,
    seq_mask: Tensor | None = None,
) -> Tensor:
    """G1 — penalise horizontal speed weighted by height-grounded probability
    only (not full soft-stance). Using full soft-stance would let the loss
    hide by making speed high (which lowers the weight and the loss).

        grounded = _pred_grounded_prob(...)                  # (B, T, 2)
        v2       = ||delta_xz * fps||^2                      # (B, T, 2)
        loss     = mean( walking * grounded * v2 )

    This is similar to old ``loss_r29_gait_pred_stance_velocity`` but kept as
    a separate named loss so the G1 bundle is auditable.
    """
    grounded = _pred_grounded_prob(
        pred_joints, gt_joints, cfg, seq_mask=seq_mask,
    )                                                                       # (B, T, 2)
    walking = _r29_walking_mask_from_support(stage2_support).to(pred_joints.dtype)
    walking = walking.squeeze(-1).unsqueeze(-1)                             # (B, T, 1)
    if seq_mask is not None:
        walking = walking * seq_mask.to(pred_joints.dtype).unsqueeze(-1)

    l_xz = pred_joints[..., LEFT_ANKLE_IDX, :][..., [0, 2]]
    r_xz = pred_joints[..., RIGHT_ANKLE_IDX, :][..., [0, 2]]
    l_v = torch.zeros_like(l_xz)
    r_v = torch.zeros_like(r_xz)
    l_v[:, 1:] = (l_xz[:, 1:] - l_xz[:, :-1]) * float(fps)
    r_v[:, 1:] = (r_xz[:, 1:] - r_xz[:, :-1]) * float(fps)
    v2 = torch.stack([l_v.pow(2).sum(-1), r_v.pow(2).sum(-1)], dim=-1)      # (B, T, 2)
    weight = walking * grounded
    num = (weight * v2).sum()
    den = weight.sum().clamp_min(1.0)
    return num / den


def loss_r29_gait_transition_rate(
    pred_joints: Tensor,                # (B, T, 22, 3)
    gt_joints: Tensor,                  # (B, T, 22, 3) — for floor
    stage2_support: Tensor,             # (B, T, >=5)
    cfg: TemporalInteractionLossConfig,
    fps: float = 20.0,
    seq_mask: Tensor | None = None,
) -> Tensor:
    """G1 — phase-free transition rate. Match how often L/R alternates without
    aligning per-frame phase. Uses S4 stance channels as target.

        pred_alt   = pred_soft_stance[..., 0] - pred_soft_stance[..., 1]
        target_alt = target_stance[..., 0]   - target_stance[..., 1]
        rate_proxy = mean( |alt[t] - alt[t-1]| ) over walking-adjacent frames
        loss       = SmoothL1(pred_rate, target_rate.detach())

    This encourages switching density without saying whether left or right
    should step first.
    """
    if stage2_support.shape[-1] < 5:
        raise ValueError(
            "loss_r29_gait_transition_rate requires stage2_support with at "
            f"least 5 channels (got {stage2_support.shape[-1]})."
        )
    pred_soft = _pred_soft_stance_prob(
        pred_joints, gt_joints, cfg, fps=fps, seq_mask=seq_mask,
    )                                                                       # (B, T, 2)
    target = stage2_support[..., 0:2].to(pred_joints.dtype).clamp(0.0, 1.0)  # (B, T, 2)
    walking = _r29_walking_mask_from_support(stage2_support).to(pred_joints.dtype)
    walking = walking.squeeze(-1)                                           # (B, T)
    if seq_mask is not None:
        walking = walking * seq_mask.to(pred_joints.dtype)
    if pred_soft.shape[1] < 2:
        return pred_soft.new_zeros(())

    pred_alt = pred_soft[..., 0] - pred_soft[..., 1]                        # (B, T)
    target_alt = target[..., 0] - target[..., 1]                            # (B, T)
    d_pred = (pred_alt[:, 1:] - pred_alt[:, :-1]).abs()                     # (B, T-1)
    d_tgt = (target_alt[:, 1:] - target_alt[:, :-1]).abs()                  # (B, T-1)
    # Walking-adjacent: both endpoints of the diff must be walking frames.
    w_pair = walking[:, 1:] * walking[:, :-1]                               # (B, T-1)

    pieces_pred: list[Tensor] = []
    pieces_tgt: list[Tensor] = []
    min_frames = int(cfg.r29_gait_antiphase_min_walking_frames)
    B = pred_joints.shape[0]
    for b in range(B):
        m = w_pair[b] > 0.5
        if int(m.sum()) < min_frames:
            continue
        pieces_pred.append(d_pred[b][m].mean())
        pieces_tgt.append(d_tgt[b][m].mean())
    if not pieces_pred:
        return pred_soft.new_zeros(())
    p = torch.stack(pieces_pred)
    t = torch.stack(pieces_tgt).detach()
    return F.smooth_l1_loss(p, t, reduction="mean")


def loss_r29_gait_duty_cycle(
    pred_joints: Tensor,                # (B, T, 22, 3)
    gt_joints: Tensor,                  # (B, T, 22, 3) — for floor
    stage2_support: Tensor,             # (B, T, >=5)
    cfg: TemporalInteractionLossConfig,
    fps: float = 20.0,
    seq_mask: Tensor | None = None,
) -> Tensor:
    """G1 — duty-cycle match, phase-safe under L<->R swap.

        pred_duty   = mean(pred_soft_stance over walking)           # (B, 2)
        target_duty = mean(target_stance over walking)              # (B, 2)
        loss        = SmoothL1(sort(pred_duty), sort(target_duty))

    Sorting the 2-vector makes the comparison invariant to which foot the
    model chose to plant longer.
    """
    if stage2_support.shape[-1] < 5:
        raise ValueError(
            "loss_r29_gait_duty_cycle requires stage2_support with at "
            f"least 5 channels (got {stage2_support.shape[-1]})."
        )
    pred_soft = _pred_soft_stance_prob(
        pred_joints, gt_joints, cfg, fps=fps, seq_mask=seq_mask,
    )                                                                       # (B, T, 2)
    target = stage2_support[..., 0:2].to(pred_joints.dtype).clamp(0.0, 1.0)
    walking = _r29_walking_mask_from_support(stage2_support).to(pred_joints.dtype)
    walking = walking.squeeze(-1)                                           # (B, T)
    if seq_mask is not None:
        walking = walking * seq_mask.to(pred_joints.dtype)
    min_frames = int(cfg.r29_gait_antiphase_min_walking_frames)

    pieces: list[Tensor] = []
    B = pred_joints.shape[0]
    for b in range(B):
        m = walking[b] > 0.5
        if int(m.sum()) < min_frames:
            continue
        pred_duty = pred_soft[b][m].mean(dim=0)                             # (2,)
        tgt_duty = target[b][m].mean(dim=0).detach()                        # (2,)
        pred_sorted, _ = torch.sort(pred_duty)
        tgt_sorted, _ = torch.sort(tgt_duty)
        pieces.append(F.smooth_l1_loss(pred_sorted, tgt_sorted, reduction="mean"))
    if not pieces:
        return pred_soft.new_zeros(())
    return torch.stack(pieces).mean()


def loss_r29_gait_both_state_match(
    pred_joints: Tensor,                # (B, T, 22, 3)
    gt_joints: Tensor,                  # (B, T, 22, 3) — for floor
    stage2_support: Tensor,             # (B, T, >=5)
    cfg: TemporalInteractionLossConfig,
    fps: float = 20.0,
    seq_mask: Tensor | None = None,
) -> Tensor:
    """G1 — both-state aggregate match (directly addresses R2's collapse to
    ``frac_both_swing=0.872``).

        pred_both_stance = mean(pL * pR)        on walking frames
        pred_both_swing  = mean((1-pL)*(1-pR))  on walking frames
        target_*         = same from S4 stance channels
        loss             = SmoothL1([pred_both_stance, pred_both_swing],
                                    [tgt_both_stance,  tgt_both_swing].detach())
    """
    if stage2_support.shape[-1] < 5:
        raise ValueError(
            "loss_r29_gait_both_state_match requires stage2_support with at "
            f"least 5 channels (got {stage2_support.shape[-1]})."
        )
    pred_soft = _pred_soft_stance_prob(
        pred_joints, gt_joints, cfg, fps=fps, seq_mask=seq_mask,
    )                                                                       # (B, T, 2)
    target = stage2_support[..., 0:2].to(pred_joints.dtype).clamp(0.0, 1.0)
    walking = _r29_walking_mask_from_support(stage2_support).to(pred_joints.dtype)
    walking = walking.squeeze(-1)                                           # (B, T)
    if seq_mask is not None:
        walking = walking * seq_mask.to(pred_joints.dtype)
    min_frames = int(cfg.r29_gait_antiphase_min_walking_frames)

    pL = pred_soft[..., 0]
    pR = pred_soft[..., 1]
    tL = target[..., 0]
    tR = target[..., 1]
    pred_both_stance = pL * pR
    pred_both_swing = (1.0 - pL) * (1.0 - pR)
    tgt_both_stance = tL * tR
    tgt_both_swing = (1.0 - tL) * (1.0 - tR)

    pieces: list[Tensor] = []
    B = pred_joints.shape[0]
    for b in range(B):
        m = walking[b] > 0.5
        if int(m.sum()) < min_frames:
            continue
        p_agg = torch.stack([pred_both_stance[b][m].mean(),
                             pred_both_swing[b][m].mean()])
        t_agg = torch.stack([tgt_both_stance[b][m].mean(),
                             tgt_both_swing[b][m].mean()]).detach()
        pieces.append(F.smooth_l1_loss(p_agg, t_agg, reduction="mean"))
    if not pieces:
        return pred_soft.new_zeros(())
    return torch.stack(pieces).mean()


# --------------------------------------------------------------------------- #
# R3 — exact S4 execution losses.
# --------------------------------------------------------------------------- #


def loss_r29_s4_stance_bce(
    pred_joints: Tensor,                # (B, T, 22, 3)
    gt_joints: Tensor,                  # (B, T, 22, 3) — for floor
    stage2_support: Tensor,             # (B, T, >=5)
    cfg: TemporalInteractionLossConfig,
    seq_mask: Tensor | None = None,
) -> Tensor:
    """R3 — Binary cross-entropy between predicted grounded prob and
    the S4 left/right stance target, masked by walking + seq.

        pred_grounded = sigmoid((floor_y + threshold - ankle_y) / softness)
        target        = stage2_support[..., 0:2]
        loss          = mean_over_walking_frames( BCE(pred_grounded, target) )
    """
    if stage2_support.shape[-1] < 13:
        raise ValueError(
            "loss_r29_s4_stance_bce requires S4 layout (dim >= 13); got "
            f"dim={stage2_support.shape[-1]}."
        )
    grounded = _pred_grounded_prob(
        pred_joints, gt_joints, cfg, seq_mask=seq_mask,
    )                                                                       # (B, T, 2)
    target = stage2_support[..., 0:2].to(pred_joints.dtype).clamp(0.0, 1.0)  # (B, T, 2)
    walking = _r29_walking_mask_from_support(stage2_support).to(pred_joints.dtype)
    walking = walking.squeeze(-1).unsqueeze(-1)                              # (B, T, 1)
    if seq_mask is not None:
        walking = walking * seq_mask.to(pred_joints.dtype).unsqueeze(-1)

    eps = 1e-6
    g = grounded.clamp(eps, 1.0 - eps)
    bce = -(target * torch.log(g) + (1.0 - target) * torch.log(1.0 - g))    # (B, T, 2)
    num = (walking * bce).sum()
    den = (walking.expand_as(bce) > 0).to(bce.dtype).sum().clamp_min(1.0)
    return num / den


def loss_r29_s4_footstep_target(
    pred_joints: Tensor,                # (B, T, 22, 3)
    gt_joints: Tensor,                  # (B, T, 22, 3) — for pelvis_0 + yaw_0
    stage2_support: Tensor,             # (B, T, >=13)
    seq_mask: Tensor | None = None,
) -> Tensor:
    """R3 — SmoothL1 between predicted ankle local XZ (root0-yaw
    canonical, normalised /3.0) and the S4 footstep-target channel.

    Matches the coordinate convention in
    ``stage2_oracle_conditions._footstep_target_local_xz``: pelvis at
    frame 0 + yaw at frame 0 as the canonical frame. We use GT for the
    frame-0 pelvis + yaw (oracle conditions), so the loss differentiates
    purely through ``pred_joints[..., ankle_idx, :]``.
    """
    if stage2_support.shape[-1] < 13:
        raise ValueError(
            "loss_r29_s4_footstep_target requires S4 layout (dim >= 13); got "
            f"dim={stage2_support.shape[-1]}."
        )
    R_T0 = _root0_world_to_canonical_yaw_matrix(gt_joints)                  # (B, 3, 3)
    pelvis_0 = gt_joints[:, 0, ROOT_IDX, :]                                 # (B, 3)
    l_xyz = pred_joints[..., LEFT_ANKLE_IDX, :]                             # (B, T, 3)
    r_xyz = pred_joints[..., RIGHT_ANKLE_IDX, :]
    ankles = torch.stack([l_xyz, r_xyz], dim=-2)                            # (B, T, 2, 3)
    rel = ankles - pelvis_0[:, None, None, :]                               # (B, T, 2, 3)
    # local = R_T0 @ (ankle - pelvis_0). einsum over the world dim.
    local = torch.einsum("bij,btfj->btfi", R_T0, rel)                       # (B, T, 2, 3)
    pred_xz = local[..., [0, 2]]                                            # (B, T, 2, 2)
    pred_xz_norm = (pred_xz.clamp(-3.0, 3.0) / 3.0).reshape(
        *pred_xz.shape[:2], 4,
    )                                                                       # (B, T, 4)
    target = stage2_support[..., 9:13].to(pred_joints.dtype)                # (B, T, 4)

    walking = _r29_walking_mask_from_support(stage2_support).to(pred_joints.dtype)
    walking = walking.squeeze(-1).unsqueeze(-1)                              # (B, T, 1)
    if seq_mask is not None:
        walking = walking * seq_mask.to(pred_joints.dtype).unsqueeze(-1)

    diff = F.smooth_l1_loss(pred_xz_norm, target, reduction="none")          # (B, T, 4)
    diff = diff.sum(dim=-1, keepdim=True)                                    # (B, T, 1)
    num = (walking * diff).sum()
    den = (walking.expand_as(diff) > 0).to(diff.dtype).sum().clamp_min(1.0)
    return num / den


# --------------------------------------------------------------------------- #
# R4 / R5 — contact-lock losses (generalised over I3 dim=8 / I5 dim=20).
# --------------------------------------------------------------------------- #


def _parse_i_channel(
    stage2_interaction: Tensor,         # (B, T, 8) for I3 or (B, T, 20) for I5
) -> tuple[Tensor, Tensor, tuple[int, ...]]:
    """Extract per-part contact + per-part target offset from an I-channel.

    Returns:
        contacts        : (B, T, P)
        target_offsets  : (B, T, P, 3). Active-contact frames are recovered
                          from the builder's masked-offset layout back to the
                          unmasked, normalised object-local offset in [-1, 1].
        part_indices    : tuple of SMPL joint indices for each part.
    """
    D = stage2_interaction.shape[-1]
    if D == 8:
        # I3 layout: 2 contact + 2*3 = 6 offset. Parts: L_wrist, R_wrist.
        contacts = stage2_interaction[..., 0:2]
        offsets = stage2_interaction[..., 2:8].reshape(
            *stage2_interaction.shape[:2], 2, 3,
        )
        part_idx = (LEFT_WRIST_IDX, RIGHT_WRIST_IDX)
    elif D == 20:
        # I5 layout: 5 contact + 5*3 = 15 offset.
        # Parts: L_wrist, R_wrist, L_ankle, R_ankle, pelvis.
        contacts = stage2_interaction[..., 0:5]
        offsets = stage2_interaction[..., 5:20].reshape(
            *stage2_interaction.shape[:2], 5, 3,
        )
        part_idx = (
            LEFT_WRIST_IDX, RIGHT_WRIST_IDX,
            LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX, ROOT_IDX,
        )
    else:
        raise ValueError(
            "contact-lock losses support only I3 (dim=8) or I5 (dim=20); "
            f"got dim={D}."
        )

    # I3/I5 store ``offset_norm * contact`` so inactive frames are exactly
    # zero. The lock losses below already use ``contacts`` as the mask, so on
    # active frames they must compare against the unmasked target. Otherwise
    # soft pseudo-labels like contact=0.75 would incorrectly pull a GT-perfect
    # prediction toward 0.75 * offset.
    denom = contacts.clamp_min(1e-6).unsqueeze(-1)
    offsets = torch.where(
        contacts.unsqueeze(-1) > 0.0,
        offsets / denom,
        torch.zeros_like(offsets),
    ).clamp(-1.0, 1.0)
    return contacts, offsets, part_idx


def _pred_part_object_local_offset(
    pred_joints: Tensor,                # (B, T, 22, 3)
    object_positions: Tensor,           # (B, T, 3)
    object_rotations: Tensor,           # (B, T, 3) axis-angle
    part_indices: tuple[int, ...],
    clamp_m: float,
) -> Tensor:
    """Compute predicted per-part object-local offset normalised to [-1, 1].

    Mirrors ``_hand_object_local_offset`` / ``_allpart_object_local_offset``
    on the dataset side. Returns ``(B, T, P, 3)``.
    """
    R_obj = _axis_angle_to_matrix_t(object_rotations.to(pred_joints.dtype))  # (B, T, 3, 3)
    R_T = R_obj.transpose(-1, -2)                                            # (B, T, 3, 3)
    parts_world = torch.stack(
        [pred_joints[..., idx, :] for idx in part_indices], dim=-2,
    )                                                                        # (B, T, P, 3)
    delta = parts_world - object_positions.unsqueeze(-2)                     # (B, T, P, 3)
    rel = torch.einsum("btij,btpj->btpi", R_T, delta)                        # (B, T, P, 3)
    return (rel.clamp(-clamp_m, clamp_m) / float(clamp_m))                   # (B, T, P, 3)


def loss_r29_contact_lock_offset(
    pred_joints: Tensor,                # (B, T, 22, 3)
    object_positions: Tensor,           # (B, T, 3)
    object_rotations: Tensor,           # (B, T, 3) axis-angle
    stage2_interaction: Tensor,         # (B, T, 8) or (B, T, 20)
    cfg: TemporalInteractionLossConfig,
    seq_mask: Tensor | None = None,
    hand_offset_clamp_m: float = 2.0,
) -> Tensor:
    """R4 / R5 — SmoothL1 between predicted per-part object-local offset
    and the I-channel target offset, masked to contact frames.

    Generalises ``loss_r29_interaction_consistency`` to both I3 (2 parts)
    and I5 (5 parts). Uses the I-channel ``contacts`` as the mask
    (self-contained on the R29 condition; no dataset contact_state needed).
    """
    contacts, target_offsets, part_idx = _parse_i_channel(stage2_interaction)
    pred_norm = _pred_part_object_local_offset(
        pred_joints, object_positions, object_rotations,
        part_indices=part_idx, clamp_m=float(hand_offset_clamp_m),
    )                                                                        # (B, T, P, 3)

    mask = (contacts > float(cfg.contact_threshold)).to(pred_joints.dtype)   # (B, T, P)
    if seq_mask is not None:
        sm = seq_mask.to(pred_joints.dtype).unsqueeze(-1)                    # (B, T, 1)
        mask = mask * sm
    mask3 = mask.unsqueeze(-1)                                               # (B, T, P, 1)
    diff = F.smooth_l1_loss(pred_norm, target_offsets, reduction="none")     # (B, T, P, 3)
    diff = diff.sum(dim=-1, keepdim=True)                                    # (B, T, P, 1)
    num = (diff * mask3).sum()
    den = mask3.sum().clamp_min(1.0)
    return num / den


def loss_r29_contact_lock_segment_drift(
    pred_joints: Tensor,                # (B, T, 22, 3)
    object_positions: Tensor,           # (B, T, 3)
    object_rotations: Tensor,           # (B, T, 3) axis-angle
    stage2_interaction: Tensor,         # (B, T, 8) or (B, T, 20)
    cfg: TemporalInteractionLossConfig,
    seq_mask: Tensor | None = None,
    hand_offset_clamp_m: float = 2.0,
) -> Tensor:
    """R4 / R5 — Per-segment relative drift loss. For each contiguous
    contact segment per part, compare:

        pred_delta = pred_rel[t] - pred_rel[t0]
        gt_delta   = target_rel[t] - target_rel[t0]
        SmoothL1(pred_delta, gt_delta)

    averaged over (segment-frame, part). Loops over batch + part because
    segment boundaries vary per clip; T <= 196 and training-only, cheap.
    """
    contacts, target_offsets, part_idx = _parse_i_channel(stage2_interaction)
    pred_norm = _pred_part_object_local_offset(
        pred_joints, object_positions, object_rotations,
        part_indices=part_idx, clamp_m=float(hand_offset_clamp_m),
    )                                                                        # (B, T, P, 3)

    B, T, P = contacts.shape
    contact_bool = (contacts > float(cfg.contact_threshold))                 # (B, T, P)
    if seq_mask is not None:
        sm = seq_mask.bool() if seq_mask.dtype == torch.bool else seq_mask > 0.5
        contact_bool = contact_bool & sm.unsqueeze(-1)

    num = pred_joints.new_zeros(())
    den_count: int = 0
    for b in range(B):
        for p in range(P):
            active = contact_bool[b, :, p]
            if int(active.sum()) < 2:
                continue
            # Walk contiguous True runs.
            run_start = -1
            t0_pred: Tensor | None = None
            t0_tgt: Tensor | None = None
            for t in range(T):
                if active[t]:
                    if run_start < 0:
                        run_start = t
                        t0_pred = pred_norm[b, t, p]
                        t0_tgt = target_offsets[b, t, p]
                    else:
                        d_pred = pred_norm[b, t, p] - t0_pred
                        d_tgt = target_offsets[b, t, p] - t0_tgt
                        num = num + F.smooth_l1_loss(
                            d_pred, d_tgt, reduction="sum",
                        )
                        den_count += 1
                else:
                    run_start = -1
                    t0_pred = None
                    t0_tgt = None
    den = max(den_count * 3, 1)                                              # 3 = xyz dims
    return num / float(den)


def loss_r29_contact_lock_tracking(
    pred_joints: Tensor,                # (B, T, 22, 3)
    object_positions: Tensor,           # (B, T, 3)
    stage2_interaction: Tensor,         # (B, T, 8) or (B, T, 20)
    cfg: TemporalInteractionLossConfig,
    seq_mask: Tensor | None = None,
) -> Tensor:
    """R4 / R5 — Per-segment object-motion projection. For each contact
    segment per part, penalise the gap between how far the predicted
    part moved in the object's motion direction vs how far the object
    moved (with a small margin).

        u            = obj_disp / |obj_disp|              segment-level direction
        pred_align   = (pred_part[t1] - pred_part[t0]) · u
        target_align = |obj_disp|                          (assume part should
                                                            follow object 1:1)
        penalty      = relu(target_align - pred_align - margin) ** 2

    Only scored on segments where |obj_disp| > tracking_min_obj_disp_m
    (skips near-stationary segments where tracking is undefined).
    """
    contacts, _, part_idx = _parse_i_channel(stage2_interaction)
    parts_world = torch.stack(
        [pred_joints[..., idx, :] for idx in part_idx], dim=-2,
    )                                                                        # (B, T, P, 3)
    B, T, P = contacts.shape
    contact_bool = (contacts > float(cfg.contact_threshold))                 # (B, T, P)
    if seq_mask is not None:
        sm = seq_mask.bool() if seq_mask.dtype == torch.bool else seq_mask > 0.5
        contact_bool = contact_bool & sm.unsqueeze(-1)

    margin = float(cfg.tracking_margin_m)
    min_disp = float(cfg.tracking_min_obj_disp_m)
    num = pred_joints.new_zeros(())
    den_count: int = 0
    for b in range(B):
        for p in range(P):
            active = contact_bool[b, :, p]
            if int(active.sum()) < 2:
                continue
            # Find each contiguous True run and score (t0, t1=last frame
            # in run). Only one penalty per segment (segment-level).
            run_start = -1
            last = -1
            for t in range(T):
                if active[t]:
                    if run_start < 0:
                        run_start = t
                    last = t
                if not active[t] or t == T - 1:
                    if run_start >= 0 and last > run_start:
                        obj_disp = object_positions[b, last] - object_positions[b, run_start]
                        disp_norm = obj_disp.norm()
                        if float(disp_norm) > min_disp:
                            u = obj_disp / disp_norm.clamp_min(1e-6)
                            pred_disp = parts_world[b, last, p] - parts_world[b, run_start, p]
                            pred_align = (pred_disp * u).sum()
                            target_align = disp_norm
                            penalty = F.relu(target_align - pred_align - margin).pow(2)
                            num = num + penalty
                            den_count += 1
                    if not active[t]:
                        run_start = -1
                        last = -1
    return num / float(max(den_count, 1))
