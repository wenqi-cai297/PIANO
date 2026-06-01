"""R41 cascade loss building blocks — mirrors PB1's motion-space loss
family for use in Stage-1's cascade training loop.

Why this file exists
--------------------
``train_anchordiff.step_fn`` (lines 319-1230) computes PB1's training
loss inline against ``x0_pred`` / ``x0_target`` / ``joints``. R41
cascade training reuses the same loss family on PB1's frozen forward
output but inside Stage-1's trainer. Refactoring the PB1 trainer's
800-line step_fn into a public function is R42 work. For R41 we
re-implement the four critical loss formulas as standalone helpers
that take only the tensors they actually need.

The helpers are intentionally:

  - small: each one ≤ 50 lines, the math matches the inline PB1
    counterpart line-for-line (PB1 src lines in each docstring);
  - pure: no batch dict, no config object, no module state — every
    input is an explicit Tensor or scalar so cascade calibration can
    feed them with synthetic inputs;
  - parameterless: they emit raw loss values (no weights baked in);
    weighting + balance happens in the cascade trainer step_fn so the
    R41 launcher can do per-cell calibration (see the calibration
    routine in scripts/stage_a_generator/round41_make_stage1_cascade_configs.py).

The PB1 ship-loss family (anchordiff_r29_pb_a1_adaln_s4.yaml weights)
across rows 1-4 has well-defined ratios; the R41 cascade trainer
respects them and lets the user pick a single ``w_cascade_total`` (B1
calibration approach) per cell.

References
----------
- ``src/piano/training/train_anchordiff.py`` (PB1 trainer; per-loss
  inline computation at the line ranges noted in each helper).
- ``configs/training/anchordiff_r29_pb_a1_adaln_s4.yaml`` (PB1 ship
  weights — A1-A4 reuse these ratios after calibration).
- ``src/piano/training/anchordiff_geometric_losses.py`` (re-used
  feature_velocity_loss).
- ``src/piano/training/smpl_kinematics.py`` (re-used
  rotation_6d_to_matrix + fk_from_global_rotations).
- ``src/piano/training/anchor_consistency_loss.py:58`` (re-used
  PART_TO_JOINT constant).

R41 cascade loss cells
----------------------
The launcher composes these helpers into the 4 ablation cells:

  - A1 motion_mse_only:    motion_mse(min-SNR weighted)
  - A2 + world_joint_vel:  A1 + world_joint_velocity_loss
  - A3 + L_pos_full:       A2 + L_pos_full (FK-derived joint MSE,
                            hand/foot endpoint reweighted)
  - A4 + anchor_joint_pos: A3 + anchor_joint_pos (contact-active wrist
                            anchor — the closest analog to PB1's
                            anchor_joint_pos_weight=10.0 supervision)
"""
from __future__ import annotations

import torch
from torch import Tensor

from piano.training.anchor_consistency_loss import PART_TO_JOINT
from piano.training.anchordiff_geometric_losses import feature_velocity_loss
from piano.training.smpl_kinematics import (
    fk_from_global_rotations,
    rotation_6d_to_matrix,
)


# Hand/foot joint indices in the 22-joint SMPL layout. Matches
# train_anchordiff.py:520-523 where PB1 reweights L_pos by these.
J_LEFT_WRIST = 20
J_RIGHT_WRIST = 21
J_LEFT_FOOT = 10
J_RIGHT_FOOT = 11

# Smallest variance / area floor to keep gradients finite under bf16
# at degenerate inputs. Conservative; never tripped under healthy
# training but exists so cascade-loop NaN guards stay quiet.
_EPS_BF16 = 1e-4


# ──────────────────────────────────────────────────────────────────────────
# Building block 1 — masked motion-space MSE (with optional min-SNR-γ)
# ──────────────────────────────────────────────────────────────────────────


def masked_motion_mse_loss(
    pred: Tensor,          # (B, T, D)  — PB1 x0_pred under cascade
    target: Tensor,        # (B, T, D)  — motion_gt
    seq_mask: Tensor,      # (B, T)     — float
    *,
    min_snr_weight: Tensor | None = None,   # (B,) — optional per-sample SNR weight
) -> Tensor:
    """Masked MSE on motion vectors, optional min-SNR-γ per-sample weight.

    Equivalent to:
        sum_per_dim = (pred - target) ** 2          # (B, T, D)
        per_frame   = sum_per_dim.sum(-1)           # (B, T)
        per_frame   = per_frame * min_snr_weight    # broadcast (B, 1)
        loss        = (per_frame * seq_mask).sum() / seq_mask.sum().clamp_min(1.0)

    Mirrors the structure used by ``train_anchordiff.py:438-465`` (PB1's
    own min-SNR-weighted diffusion MSE). R41 reuses it because:

    - bf16-safe (sum then mean, no per-channel sqrt);
    - the masked-mean denominator keeps gradient scale invariant to
      sequence length, matching PB1's training-time gradient scale;
    - min-SNR weighting reverses the t-bias surfaced by P0 check 7
      (cascade grad norm at high t was 9x larger than at low t) so the
      cascade-loop gradient is balanced across the diffusion schedule.

    Parameters
    ----------
    pred, target
        Same shape (B, T, D). For R41 cascade D=135 (PB1 motion dim).
    seq_mask
        Per-frame validity mask. Float, 0/1.
    min_snr_weight
        Per-sample weight tensor, shape (B,). When None, no weighting.
        Should be the normalized min(SNR_t, γ) over the sampled t_pb1.
        Passed in (rather than computed here) so the cascade trainer can
        reuse the same t_pb1 + α_bar arrays it already has.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"masked_motion_mse_loss shape mismatch: "
            f"pred {tuple(pred.shape)} vs target {tuple(target.shape)}"
        )
    if seq_mask.shape != pred.shape[:2]:
        raise ValueError(
            f"seq_mask shape {tuple(seq_mask.shape)} != (B, T) "
            f"= {tuple(pred.shape[:2])}"
        )
    per_dim = (pred - target).pow(2)                         # (B, T, D)
    per_frame = per_dim.sum(-1)                              # (B, T)
    if min_snr_weight is not None:
        if min_snr_weight.shape != (pred.shape[0],):
            raise ValueError(
                f"min_snr_weight shape {tuple(min_snr_weight.shape)} != "
                f"(B,) = ({pred.shape[0]},)"
            )
        per_frame = per_frame * min_snr_weight.view(-1, 1)
    return (per_frame * seq_mask).sum() / seq_mask.sum().clamp_min(1.0)


# ──────────────────────────────────────────────────────────────────────────
# Building block 2 — masked world-joint velocity MSE (135-D)
# ──────────────────────────────────────────────────────────────────────────


def world_joint_velocity_loss(
    pred: Tensor,          # (B, T, D)
    target: Tensor,        # (B, T, D)
    seq_mask: Tensor,      # (B, T)
) -> Tensor:
    """Wrapper around the existing PB1 helper to make A2 self-contained.

    The math mirrors ``train_anchordiff.py:502-506``: feature-vector MSE
    on the 1-frame finite difference, masked to valid pairs. No SNR
    weighting (PB1 doesn't apply min-SNR here either).
    """
    return feature_velocity_loss(pred, target, seq_mask)


# ──────────────────────────────────────────────────────────────────────────
# Building block 3 — FK from PB1 x0_pred → 22-joint positions
# ──────────────────────────────────────────────────────────────────────────


def fk_motion_135_to_joints_22(
    motion: Tensor,        # (B, T, 135) — 132 rot6d + 3 root world xyz
    rest_offsets: Tensor,  # (B, 22, 3)
) -> Tensor:
    """Recover 22-joint world positions from a 135-D motion tensor.

    Mirrors ``train_anchordiff.py:481-488``. Used by both
    ``l_pos_full_loss`` and ``anchor_joint_pos_loss`` below; factored
    out so we don't run FK twice when a cascade cell needs both losses.

    Returns: (B, T, 22, 3) joint positions in world frame.
    """
    if motion.dim() != 3 or motion.shape[-1] != 135:
        raise ValueError(
            f"motion must be (B, T, 135); got {tuple(motion.shape)}"
        )
    if rest_offsets.shape[-2:] != (22, 3):
        raise ValueError(
            f"rest_offsets must end in (22, 3); got {tuple(rest_offsets.shape)}"
        )
    B, T, _ = motion.shape
    rot6d = motion[..., :132].view(B, T, 22, 6).float()
    root_world = motion[..., 132:135].float()                # (B, T, 3)
    rot_mat_global = rotation_6d_to_matrix(rot6d)            # (B, T, 22, 3, 3)
    rest_per_frame = rest_offsets.unsqueeze(1).expand(B, T, 22, 3).float()
    return fk_from_global_rotations(
        rot_mat_global, rest_per_frame, root_world,
    )                                                        # (B, T, 22, 3)


# ──────────────────────────────────────────────────────────────────────────
# Building block 4 — full-body L_pos (FK joint MSE, hand/foot reweighted)
# ──────────────────────────────────────────────────────────────────────────


def l_pos_full_loss(
    jpos_pred: Tensor,     # (B, T, 22, 3) — FK from pred motion
    joints_gt: Tensor,     # (B, T, 22, 3)
    seq_mask: Tensor,      # (B, T)
    *,
    hand_endpoint_weight: float = 2.0,
    foot_endpoint_weight: float = 2.0,
) -> Tensor:
    """Full-body L_pos = MSE between FK-derived predicted joints and GT
    joints, all 22 joints × all valid frames, with per-joint weighting
    on hand/foot endpoints.

    Mirrors ``train_anchordiff.py:511-529``. PB1 ship cfg uses
    ``hand_endpoint_weight=2.0, foot_endpoint_weight=2.0,
    pos_loss_weight=5.0`` (the 5.0 multiplier is applied by the cascade
    trainer's calibration, not here).

    The dense per-frame supervision (MDM Eq. 3, Tevet et al. ICLR 2023)
    is what PB1 relies on for anti-frozen training: it directly attacks
    the channel-mean failure mode by penalizing any joint that drifts
    in world-frame position, not just the rot6d.
    """
    if jpos_pred.shape != joints_gt.shape:
        raise ValueError(
            f"l_pos_full_loss shape mismatch: pred {tuple(jpos_pred.shape)} "
            f"vs gt {tuple(joints_gt.shape)}"
        )
    if jpos_pred.shape[-2:] != (22, 3):
        raise ValueError(
            f"l_pos_full_loss expected (B, T, 22, 3); "
            f"got {tuple(jpos_pred.shape)}"
        )
    err = (jpos_pred.float() - joints_gt.float()).pow(2).sum(-1)   # (B, T, 22)
    if hand_endpoint_weight != 1.0 or foot_endpoint_weight != 1.0:
        jw = torch.ones(22, device=err.device, dtype=err.dtype)
        jw[J_LEFT_WRIST] = float(hand_endpoint_weight)
        jw[J_RIGHT_WRIST] = float(hand_endpoint_weight)
        jw[J_LEFT_FOOT] = float(foot_endpoint_weight)
        jw[J_RIGHT_FOOT] = float(foot_endpoint_weight)
        err = err * jw
        weight_sum = jw.sum()
    else:
        weight_sum = err.new_tensor(22.0)
    denom = (seq_mask.sum() * weight_sum).clamp_min(1.0)
    return (err * seq_mask.unsqueeze(-1).float()).sum() / denom


# ──────────────────────────────────────────────────────────────────────────
# Building block 5 — contact-active anchor joint pos loss
# ──────────────────────────────────────────────────────────────────────────


def anchor_joint_pos_loss(
    jpos_pred: Tensor,           # (B, T, 22, 3)
    joints_gt: Tensor,           # (B, T, 22, 3)
    contact_state: Tensor,       # (B, T, 5) — float, soft 0/1 contact per part
    seq_mask: Tensor,            # (B, T)
    *,
    part_weights: tuple[float, ...] = (2.0, 2.0, 0.0, 0.0, 0.5),
    contact_threshold: float = 0.5,
) -> Tensor:
    """Anchor-joint position loss on contact-active frames per part.

    Mirrors ``train_anchordiff.py:539-564``. PB1 ship cfg uses
    ``anchor_joint_pos_weight=10.0`` (the largest weight in the stack),
    with part weights (L_hand=2, R_hand=2, L_foot=0, R_foot=0,
    pelvis=0.5).

    Why R41 includes this in A4:
        PB1's L_pos averages over all 22 joints × all frames; hands can
        stay poor while the average looks fine. anchor_joint_pos targets
        only contact-active wrist frames (the hardest visual failure
        mode) with 10x the dense weight. It is the supervision signal
        Stage-1 has never seen in any form, so A4 tests whether
        explicit contact-aware anchor signal moves the cascade metric
        beyond what dense L_pos already does.
    """
    if jpos_pred.shape != joints_gt.shape:
        raise ValueError(
            f"anchor_joint_pos_loss shape mismatch: pred {tuple(jpos_pred.shape)} "
            f"vs gt {tuple(joints_gt.shape)}"
        )
    if contact_state.shape[-1] != len(PART_TO_JOINT):
        raise ValueError(
            f"contact_state last dim {contact_state.shape[-1]} != "
            f"len(PART_TO_JOINT)={len(PART_TO_JOINT)}"
        )
    if len(part_weights) != len(PART_TO_JOINT):
        raise ValueError(
            f"part_weights length {len(part_weights)} != "
            f"len(PART_TO_JOINT)={len(PART_TO_JOINT)}"
        )
    part_to_joint_t = torch.tensor(
        PART_TO_JOINT, device=jpos_pred.device, dtype=torch.long,
    )
    pred_part = jpos_pred.float().index_select(2, part_to_joint_t)  # (B, T, P, 3)
    gt_part = joints_gt.float().index_select(2, part_to_joint_t)
    active_part = (
        (contact_state >= float(contact_threshold))
        & seq_mask.bool().unsqueeze(-1)
    )                                                            # (B, T, P) bool
    active_f = active_part.float()
    part_w = torch.tensor(
        list(part_weights), device=jpos_pred.device, dtype=pred_part.dtype,
    )
    weighted_active = active_f * part_w                          # (B, T, P)
    err_p = (pred_part - gt_part).pow(2).sum(-1)                 # (B, T, P)
    denom = weighted_active.sum().clamp_min(1.0)
    return (err_p * weighted_active).sum() / denom


# ──────────────────────────────────────────────────────────────────────────
# Building block 6 — min-SNR-γ per-sample weight
# ──────────────────────────────────────────────────────────────────────────


def compute_min_snr_weight(
    t: Tensor,                    # (B,) — sampled diffusion timesteps
    alphas_cumprod: Tensor,       # (num_steps,) — diffusion noise schedule
    *,
    gamma: float = 5.0,
) -> Tensor:
    """Per-sample min(SNR_t, γ) weight, normalized so mean(w)=1.

    Mirrors ``train_anchordiff.py:448-460``. The normalization preserves
    overall loss scale so the cascade trainer's other-loss weights are
    unaffected — only the per-sample timestep balance changes.

    Why R41 cascade should use this:
        P0 check 7 showed cascade grad norm at t=900-1000 was 9x
        larger than at t=0-100. Min-SNR-γ inverts this bias: high-t
        samples get small weight (their cond signal is noise-dominated),
        low-t samples get large weight (clean signal, informative
        gradient).
    """
    if t.dim() != 1:
        raise ValueError(f"t must be (B,); got {tuple(t.shape)}")
    if alphas_cumprod.dim() != 1:
        raise ValueError(
            f"alphas_cumprod must be (num_steps,); got {tuple(alphas_cumprod.shape)}"
        )
    alpha_bar = alphas_cumprod.gather(0, t)                  # (B,)
    snr = alpha_bar / (1.0 - alpha_bar + 1e-8)               # (B,)
    snr_clamped = torch.clamp_max(snr, float(gamma))         # (B,)
    return snr_clamped / snr_clamped.mean().clamp_min(1e-8)  # (B,) normalized
