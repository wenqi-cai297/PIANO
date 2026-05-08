"""V5 (OMOMO/CHOIS-style) auxiliary losses for AnchorDiff.

The 198-D representation packs `(global_jpos: 22*3, global_rot_6d: 22*6)`.
The two halves are redundant: positions can be re-derived from rotations
via forward kinematics through the SMPL-22 chain. CHOIS adds a "FK
consistency" loss that ties the two together, preventing the network
from drifting one half away from the other (which is exactly the
"stretched arm" failure mode v4 exhibited on bat-swing clips).

Reference:
    Li et al. *Controllable Human-Object Interaction Synthesis* (CHOIS),
    ECCV 2024. Loss implementation at
    ``lijiaman/chois_release@manip/model/transformer_object_motion_cond_diffusion.py:1448-1475``
    re-derives joint positions from predicted 6D rotations via
    ``quat_ik_torch -> quat_fk_torch`` and L1-compares against the
    predicted positions.

Our module differs from CHOIS in two minor ways, both grounded in the
OMOMO source:

1. We use ``fk_from_global_rotations`` (skips the IK step) because we
   predict GLOBAL rotations directly (matching OMOMO's
   ``query['global_rot_6d']`` at hand_foot_dataset.py:742). CHOIS
   stores LOCAL rotations and IKs first; the choice is equivalent in
   information content but avoids a `quat_ik_torch` chain pass.

2. We use squared L2 instead of L1 because squared L2 plays nicer with
   the diffusion x_0 MSE objective and matches MDM's ``L_pos`` term
   convention (Eq. 3 of Tevet et al. ICLR 2023).
"""
from __future__ import annotations

import torch
from torch import Tensor

from piano.training.smpl_kinematics import (
    fk_from_global_rotations,
    rotation_6d_to_matrix,
)


def fk_consistency_loss(
    jpos_pred: Tensor,             # (B, T, 22, 3) predicted world joints
    rot_6d_pred: Tensor,           # (B, T, 22, 6) predicted global per-joint 6D
    rest_offsets: Tensor,          # (B, 22, 3) per-clip T-pose bone offsets
    seq_mask: Tensor,              # (B, T) — True/1.0 = valid frame
) -> Tensor:
    """Squared L2 distance between predicted positions and FK-derived
    positions through predicted rotations.

    Forces the redundant 198-D rep to be internally consistent: any
    movement of a wrist must be reachable via the SMPL-22 kinematic
    chain from root + global rotations + (clip-fixed) bone offsets.
    Eliminates the "limb stretches to satisfy anchor" failure mode.

    Returns
    -------
    scalar loss tensor
    """
    B, T = jpos_pred.shape[:2]
    # 6D -> 3x3 rotation matrices (batched).
    rot_mat = rotation_6d_to_matrix(rot_6d_pred)               # (B, T, 22, 3, 3)

    # rest_offsets is per-clip; broadcast to per-frame.
    rest_per_frame = rest_offsets.unsqueeze(1).expand(B, T, 22, 3)

    # Use the predicted root joint as the FK origin per frame. This means
    # FK consistency cannot constrain root TRANSLATION (root_jpos is
    # taken as-is); only relative joint positions are constrained.
    # Combined with the L_simple MSE on jpos[:, :, 0], the anchor loss,
    # and the world joint velocity loss, root translation gets enough
    # supervision elsewhere.
    root_pos = jpos_pred[:, :, 0, :]                           # (B, T, 3)
    fk_jpos = fk_from_global_rotations(rot_mat, rest_per_frame, root_pos)

    sq = (fk_jpos - jpos_pred).pow(2).sum(dim=-1)              # (B, T, 22)
    valid = seq_mask.float().unsqueeze(-1)                     # (B, T, 1)
    denom = (valid.sum() * 22).clamp_min(1.0)
    return (sq * valid).sum() / denom
