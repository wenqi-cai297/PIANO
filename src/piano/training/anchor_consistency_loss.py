"""Anchor consistency loss for PIANO-AnchorDiff.

OMOMO-inspired. On contact frames the predicted body part position
(in world frame) should match the object-anchored target:

    pred_body_part_world[t, p]
        ≈ object_world_pose[t] @ contact_target_xyz_object_local[t, p]

Loss is masked to frames + parts where ``contact_state[t, p] >= 0.5``.

We work from the predicted x_0 (denoiser's implicit clean sample), not
from the noisy x_t, because the anchor relationship is only defined on
the clean motion. The diffusion's ``predict_x0_from_eps`` returns x_0
in HumanML3D 263-d canonical coords; we then run MoMask's
``recover_from_ric`` to lift to canonical-frame joints.

For M0 the loss is always computed in **canonical (body) frame**, not
in true world frame. Reasons:
    1. ``contact_target_xyz`` is stored in object-local coords; the
       Stage B dataset already canonicalises object world pose into the
       same body-canonical frame as motion_263 (see
       ``HOIDataset._compute_canonical_object_pose``).
    2. World-frame supervision would need to undo the canonical frame
       rotation per-frame, which is doable but adds complexity. Defer
       to M1+ once the loss-shape is known to fire.

Reference:
    src/piano/models/backbones/momask/utils/motion_process.py:400  recover_from_ric
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


# Body-part → SMPL-22 joint index mapping. Matches
# ``BODY_PART_INDICES`` in ``src/piano/utils/smpl_utils.py`` and the
# 5-part contact_state layout used by Stage A v10:
#     0=left_hand, 1=right_hand, 2=left_foot, 3=right_foot, 4=pelvis
PART_TO_JOINT: tuple[int, ...] = (20, 21, 10, 11, 0)


@dataclass(slots=True)
class AnchorConsistencyConfig:
    weight: float = 1.0
    contact_threshold: float = 0.5
    """Frames+parts with contact_state below this are excluded."""

    eps_l2: float = 1e-3
    """Small constant inside L2 to keep gradient finite at zero."""


def _quat_apply(q: Tensor, v: Tensor) -> Tensor:
    """Apply unit quaternion q (w,x,y,z) to vector v.

    q: (..., 4), v: (..., 3). Both broadcast.
    """
    qw, qx, qy, qz = q.unbind(-1)
    vx, vy, vz = v.unbind(-1)
    # standard quat-vector rotation expanded
    tx = 2 * (qy * vz - qz * vy)
    ty = 2 * (qz * vx - qx * vz)
    tz = 2 * (qx * vy - qy * vx)
    rx = vx + qw * tx + (qy * tz - qz * ty)
    ry = vy + qw * ty + (qz * tx - qx * tz)
    rz = vz + qw * tz + (qx * ty - qy * tx)
    return torch.stack([rx, ry, rz], dim=-1)


def lift_motion263_to_joints(motion_263: Tensor, num_joints: int = 22) -> Tensor:
    """Wrap MoMask's ``recover_from_ric`` for canonical-frame joints.

    motion_263: (B, T, 263)
    returns:    (B, T, J, 3)  in body-canonical frame.

    The MoMask backbone is vendored under ``src/piano/models/backbones/momask/``
    with absolute-style imports (``from utils.motion_process import ...``,
    ``from common.skeleton import ...``). Importing
    ``piano.models.backbones.momask_adapter`` injects the right paths into
    ``sys.path`` so those imports resolve. Same pattern as
    ``src/piano/inference/visualize_motion.py:92``.
    """
    import piano.models.backbones.momask_adapter  # noqa: F401 — sys.path side-effect
    from utils.motion_process import recover_from_ric

    return recover_from_ric(motion_263, num_joints)


def lift_object_local_to_canonical(
    target_local: Tensor,           # (B, T, P, 3)
    obj_com_canonical: Tensor,      # (B, T, 3)
    obj_rot6d_canonical: Tensor,    # (B, T, 6)
) -> Tensor:
    """Map a contact target from object-local frame to body-canonical
    frame using the per-frame canonical object pose surfaced by
    ``HOIDataset._compute_canonical_object_pose``.

    target_canon = R_canonical @ target_local + t_canonical.
    """
    # 6D rot → 3x3
    a1 = obj_rot6d_canonical[..., :3]
    a2 = obj_rot6d_canonical[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    R = torch.stack([b1, b2, b3], dim=-1)            # (B, T, 3, 3)

    # rotate target_local by R
    target_local_b = target_local                     # (B, T, P, 3)
    R_b = R.unsqueeze(2)                              # (B, T, 1, 3, 3)
    rotated = (R_b @ target_local_b.unsqueeze(-1)).squeeze(-1)  # (B, T, P, 3)
    return rotated + obj_com_canonical.unsqueeze(2)   # broadcast over P


def anchor_consistency_loss(
    x0_pred: Tensor,                # (B, T, 263) predicted clean motion
    contact_state_gt: Tensor,       # (B, T, P)   GT contact (or sigmoid)
    contact_target_xyz_local: Tensor,  # (B, T, P, 3)
    obj_com_canonical: Tensor,      # (B, T, 3)
    obj_rot6d_canonical: Tensor,    # (B, T, 6)
    cfg: AnchorConsistencyConfig,
    seq_mask: Tensor | None = None, # (B, T) — True = valid frame, optional
) -> Tensor:
    """Masked L2 between predicted body-part position and object-anchored
    target, both in body-canonical frame. Returns a scalar.
    """
    joints_pred = lift_motion263_to_joints(x0_pred)            # (B, T, J, 3)
    target_canon = lift_object_local_to_canonical(
        contact_target_xyz_local, obj_com_canonical, obj_rot6d_canonical,
    )                                                           # (B, T, P, 3)

    # gather predicted joint per body part
    part_idx = torch.tensor(
        PART_TO_JOINT, device=joints_pred.device, dtype=torch.long,
    )                                                           # (P,)
    pred_part = joints_pred.index_select(2, part_idx)          # (B, T, P, 3)

    diff = pred_part - target_canon                             # (B, T, P, 3)
    sq = (diff * diff).sum(-1)                                 # (B, T, P)
    l2 = torch.sqrt(sq + cfg.eps_l2)                           # (B, T, P)

    contact_mask = (contact_state_gt >= cfg.contact_threshold).float()  # (B, T, P)
    if seq_mask is not None:
        contact_mask = contact_mask * seq_mask.unsqueeze(-1).float()

    denom = contact_mask.sum().clamp_min(1.0)
    return cfg.weight * (l2 * contact_mask).sum() / denom
