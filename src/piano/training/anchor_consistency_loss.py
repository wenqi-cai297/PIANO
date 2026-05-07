"""Anchor consistency loss for PIANO-AnchorDiff (world-frame variant).

OMOMO-inspired. On contact frames the predicted body part position
(in world frame) should match the object-anchored target:

    pred_body_part_world[t, p]
        ≈ object_world_pose[t] @ contact_target_xyz_object_local[t, p]

Loss is masked to frames + parts where ``contact_state[t, p] >= 0.5``
AND ``part_weight[p] > 0`` (foot is dropped by default — see Bug 2
in analyses/2026-05-08_anchordiff_frame_bug_fix.md: v18 pseudo-label
"foot contact" actually fires on knee/shin markers for chairs/imhd/
neuraldome subsets, with mean target-vs-toe distance > 60 cm).

We work in **world frame** because:

1. ``joints`` (raw-skel) and ``contact_target_xyz`` (object-local from
   raw-skel markers) are in consistent skeleton scale.
   ``recover_from_ric(motion_263)`` is uniform-skel; comparing
   uniform-skel canonical against raw-skel canonical introduced a
   systematic 5-10 cm bias. Lifting uniform-skel to world via the
   per-clip canonicalisation transform makes the bias a single
   world-frame translation that the loss tolerates.
2. ``contact_target_xyz`` is stored in object-local. We lift to
   world via ``object_positions`` (world COM) and ``object_rotations``
   (world axis-angle). This avoids the ``obj_*_canonical`` machinery
   that depended on a brittle frame-0 alignment.

Inputs the caller must provide:
    x0_pred                 (B, T, 263)   denoiser's x₀ prediction
    contact_state_gt        (B, T, P)     pseudo-label or Stage A sigmoid
    contact_target_xyz_local(B, T, P, 3)  pseudo-label in object-local
    object_positions        (B, T, 3)     world COM
    object_rotations        (B, T, 3)     world axis-angle
    R_y                     (B,)          canonical→world Y-rotation angle
    T_xz                    (B, 2)        canonical→world XZ translation
    T_y                     (B,)          canonical→world Y translation

The (R_y, T_xz, T_y) tuple is per-clip and is computed by
``piano.utils.canonical_frame.get_canonicalize_transform_from_clip``
inside the trainer's step_fn (cheap; one frame-0 evaluation per clip).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


# Body-part → SMPL-22 joint index mapping. 5-part contact_state layout:
#     0=left_hand, 1=right_hand, 2=left_foot, 3=right_foot, 4=pelvis
# foot maps to SMPL-22 toe-tip joints 10/11; this is conventionally
# correct, but v18 pseudo-label "foot contact" semantics on chairs etc.
# don't always anchor to toe-tip. We keep the mapping but mask foot off
# by default via ``AnchorConsistencyConfig.part_weights``.
PART_TO_JOINT: tuple[int, ...] = (20, 21, 10, 11, 0)


@dataclass(slots=True)
class AnchorConsistencyConfig:
    weight: float = 1.0
    contact_threshold: float = 0.5
    """Frames+parts with contact_state below this are excluded."""

    # Per-part weight in the L2 sum. Foot weight = 0 by default because
    # v18 foot-marker semantics don't reliably correspond to
    # SMPL-22 toe-tip on chairs/imhd/neuraldome. Re-enable after a v19
    # foot-marker fix.
    part_weights: tuple[float, ...] = (1.0, 1.0, 0.0, 0.0, 1.0)

    eps_l2: float = 1e-3
    """Small constant inside L2 to keep gradient finite at zero."""

    max_distance_m: float = 0.5
    """Per-element cap on the L2 distance. Caps catastrophic outliers
    where motion_263 canonicalisation has accumulated cumsum drift over
    many frames (~1 % of v18 clips have late-frame canonical→world
    error > 1 m). Clamping at 0.5 m keeps the gradient direction
    correct (pull body toward target) without letting one bad frame
    dominate the batch."""


def lift_motion263_to_joints(motion_263: Tensor, num_joints: int = 22) -> Tensor:
    """Wrap MoMask's ``recover_from_ric`` for canonical-frame joints.

    motion_263: (B, T, 263)
    returns:    (B, T, J, 3)  in body-canonical frame, **uniform skeleton**.

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


def _aa_to_rotation_matrix(aa: Tensor) -> Tensor:
    """Rodrigues. ``aa`` shape (..., 3) → R shape (..., 3, 3)."""
    theta = aa.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    k = aa / theta
    K = torch.zeros(aa.shape[:-1] + (3, 3), device=aa.device, dtype=aa.dtype)
    kx, ky, kz = k.unbind(-1)
    K[..., 0, 1] = -kz; K[..., 0, 2] = ky
    K[..., 1, 0] = kz;  K[..., 1, 2] = -kx
    K[..., 2, 0] = -ky; K[..., 2, 1] = kx
    eye = torch.eye(3, device=aa.device, dtype=aa.dtype).expand_as(K)
    sin = theta.unsqueeze(-1).sin()
    cos = theta.unsqueeze(-1).cos()
    return eye + sin * K + (1 - cos) * (K @ K)


def lift_object_local_to_world(
    target_local: Tensor,           # (B, T, P, 3)
    obj_pos_world: Tensor,          # (B, T, 3)
    obj_rot_world_aa: Tensor,       # (B, T, 3) axis-angle
) -> Tensor:
    """Map an object-local target into world frame using world object pose."""
    R = _aa_to_rotation_matrix(obj_rot_world_aa)              # (B, T, 3, 3)
    rotated = (R.unsqueeze(2) @ target_local.unsqueeze(-1)).squeeze(-1)
    return rotated + obj_pos_world.unsqueeze(2)


def lift_canonical_joints_to_world(
    joints_canon: Tensor,           # (B, T, J, 3) uniform-skel canonical
    R_y: Tensor,                    # (B,) rotation angle (canonical→world)
    T_xz: Tensor,                   # (B, 2) XZ translation
    T_y: Tensor,                    # (B,) Y translation
) -> Tensor:
    """Apply the per-clip canonical→world transform recovered by
    ``get_canonicalize_transform_from_clip`` to lift uniform-skel
    canonical joints into world frame.

    world_joints = R_y(canon_joints) + (T_xz[0], T_y, T_xz[1]).
    """
    cos = R_y.cos()
    sin = R_y.sin()
    zero = torch.zeros_like(cos)
    one = torch.ones_like(cos)
    # R_y rotation matrix (around +Y by R_y angle), per clip
    R = torch.stack([
        torch.stack([cos, zero, sin], dim=-1),
        torch.stack([zero, one, zero], dim=-1),
        torch.stack([-sin, zero, cos], dim=-1),
    ], dim=-2)                                                   # (B, 3, 3)
    # Apply rotation: R_y @ joints (per-clip rotation)
    rotated = torch.einsum("bij,btnj->btni", R, joints_canon)   # (B, T, J, 3)
    rotated[..., 0] = rotated[..., 0] + T_xz[..., 0:1].unsqueeze(-1)
    rotated[..., 1] = rotated[..., 1] + T_y.view(-1, 1, 1)
    rotated[..., 2] = rotated[..., 2] + T_xz[..., 1:2].unsqueeze(-1)
    return rotated


def anchor_consistency_loss(
    x0_pred: Tensor,                # (B, T, 263) predicted clean motion
    contact_state_gt: Tensor,       # (B, T, P)   GT contact (or sigmoid)
    contact_target_xyz_local: Tensor,  # (B, T, P, 3)
    object_positions: Tensor,       # (B, T, 3) world COM
    object_rotations: Tensor,       # (B, T, 3) world axis-angle
    R_y: Tensor,                    # (B,) canonical→world Y-rot
    T_xz: Tensor,                   # (B, 2) canonical→world XZ-trans
    T_y: Tensor,                    # (B,) canonical→world Y-trans
    cfg: AnchorConsistencyConfig,
    seq_mask: Tensor | None = None, # (B, T) — True = valid frame, optional
) -> Tensor:
    """Masked L2 between predicted body-part position (lifted to world)
    and object-anchored target (also in world). Returns a scalar.

    Per-part weights mask off parts whose pseudo-label semantics aren't
    reliable (default: foot weight = 0 due to v18 marker bug).
    """
    joints_canon = lift_motion263_to_joints(x0_pred)              # (B, T, J, 3)
    joints_world_pred = lift_canonical_joints_to_world(
        joints_canon, R_y, T_xz, T_y,
    )                                                              # (B, T, J, 3)
    target_world = lift_object_local_to_world(
        contact_target_xyz_local, object_positions, object_rotations,
    )                                                              # (B, T, P, 3)

    # gather predicted joint per body part
    part_idx = torch.tensor(
        PART_TO_JOINT, device=joints_world_pred.device, dtype=torch.long,
    )                                                              # (P,)
    pred_part = joints_world_pred.index_select(2, part_idx)        # (B, T, P, 3)

    diff = pred_part - target_world                                # (B, T, P, 3)
    sq = (diff * diff).sum(-1)                                    # (B, T, P)
    l2 = torch.sqrt(sq + cfg.eps_l2)                              # (B, T, P)
    # Cap per-element to limit pathological-clip contribution.
    l2 = l2.clamp(max=cfg.max_distance_m)

    contact_mask = (contact_state_gt >= cfg.contact_threshold).float()
    part_w = torch.tensor(
        cfg.part_weights, device=l2.device, dtype=l2.dtype,
    ).view(1, 1, -1)                                              # (1, 1, P)
    contact_mask = contact_mask * part_w
    if seq_mask is not None:
        contact_mask = contact_mask * seq_mask.unsqueeze(-1).float()

    denom = contact_mask.sum().clamp_min(1.0)
    return cfg.weight * (l2 * contact_mask).sum() / denom
