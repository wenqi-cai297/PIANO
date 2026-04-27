"""Left-right mirror utilities for the HumanML3D 263-d motion repr +
the world-frame object pose carried alongside it in PIANO Stage B.

Existing :func:`piano.data.dataset.HOIDataset._apply_augmentation`
already mirrors ``joints_22``, ``contact_state``, ``contact_target_xyz``,
and the text prompt — sufficient for Stage A predictor training (which
takes ``joints_22`` as input). Stage B's encoder consumes
``motion_263`` (HumanML3D rep) and the per-frame world-frame object
pose; without mirroring those, ``mirror_prob > 0`` produces an
input/label mismatch (text says "right" but motion shows "left"). This
module fills that gap.

Mirror convention: reflection through the body's sagittal plane —
equivalently, ``x → -x`` in body-canonical (``motion_263``) and world
(``object_*``) frames. All formulas verified against the reflection
identity ``R' = M R M`` for ``M = diag(-1, 1, 1)``.

Sources:
- Guo, C. et al. *Generating Diverse and Natural 3D Human Motions
  from Text (HumanML3D).* **CVPR 2022**. arXiv:2204.14109. — defines
  the 263-d rep used by MoMask + ours (4 root + 21·3 ric + 21·6
  cont6d rot + 22·3 local-vel + 4 foot-contact).
- Zhou, Y. et al. *On the Continuity of Rotation Representations in
  Neural Networks.* **CVPR 2019**. arXiv:1812.07035. — cont6d rep
  whose mirror formula derives from columns of ``M R M``.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# SMPL-22 left/right joint pairs (matches dataset.py:_SMPL22_LR_PAIRS)
# ---------------------------------------------------------------------------

# (left_idx, right_idx) in SMPL-22 numbering. Pelvis (0), spines (3, 6, 9),
# neck (12), head (15) are midline → no L/R swap.
_SMPL22_LR_PAIRS = (
    (1, 2),     # hips
    (4, 5),     # knees
    (7, 8),     # ankles
    (10, 11),   # feet (mid-foot, used by INTERACTION_BODY_PARTS since v9)
    (13, 14),   # collars
    (16, 17),   # shoulders
    (18, 19),   # elbows
    (20, 21),   # wrists (= hands per INTERACTION_BODY_PARTS)
)

# In ric_data / rot_data, joints 1..21 are stored, indexed by (joint - 1).
_RIC_LR_PAIRS = tuple((li - 1, ri - 1) for li, ri in _SMPL22_LR_PAIRS)


# ---------------------------------------------------------------------------
# motion_263 mirror (HumanML3D rep, num_joints=22)
# ---------------------------------------------------------------------------

# Layout (verified against
# ``src/piano/models/backbones/momask/utils/motion_process.py::extract_features``):
#   [0:1]    root_y_rot_velocity      (1)
#   [1:3]    root_xz_velocity         (2)
#   [3:4]    root_y_height            (1)
#   [4:67]   ric_data                 (21 joints × 3 = 63) — local pos in canonical
#   [67:193] rot_data                 (21 joints × 6 = 126) — cont6d local rotations
#   [193:259] local_velocity          (22 joints × 3 = 66) — vel in canonical
#   [259:263] foot_contact            (4) — [feet_l × 2, feet_r × 2]
_RIC_START, _RIC_END = 4, 67
_ROT_START, _ROT_END = 67, 193
_VEL_START, _VEL_END = 193, 259
_FEET_START, _FEET_END = 259, 263


def mirror_motion_263(motion: np.ndarray) -> np.ndarray:
    """Mirror a HumanML3D 263-d motion sequence through the sagittal plane.

    Parameters
    ----------
    motion : (T, 263) float32

    Returns
    -------
    mirrored : (T, 263) float32 — a copy with all left/right channels
        swapped + x-coords / x-velocities / y- and z- components of
        cont6d rotations sign-flipped according to ``R' = M R M`` for
        ``M = diag(-1, 1, 1)``. Round-trip safe:
        ``mirror_motion_263(mirror_motion_263(m)) == m``.

    Notes
    -----
    The cont6d formula:
        cont6d = [b1_x, b1_y, b1_z, b2_x, b2_y, b2_z] (first two cols of R)
        mirror → [b1_x, -b1_y, -b1_z, -b2_x, b2_y, b2_z]
    derives from R' = M R M, taking columns 1 and 2 of R'.

    Local rotations (HumanML3D ``rot_data``) transform under the same
    formula as global rotations because the mirror conjugates *both*
    the joint and its parent — so the local relative rotation
    transforms by ``R_local' = M R_local M`` identically.
    """
    if motion.shape[-1] != 263:
        raise ValueError(
            f"mirror_motion_263 expects last dim 263 (HumanML3D), got {motion.shape[-1]}",
        )
    m = np.asarray(motion, dtype=np.float32).copy()
    T_axes = m.shape[:-1]

    # ---- Root features (4) ----
    m[..., 0] *= -1.0      # root y-rotation velocity: opposite spin
    m[..., 1] *= -1.0      # root x-linear velocity: flip
    # m[..., 2] (root z-vel) and m[..., 3] (root y-height) unchanged.

    # ---- ric_data: (..., 21, 3) — joint local positions (joints 1..21) ----
    ric = m[..., _RIC_START:_RIC_END].reshape(*T_axes, 21, 3)
    ric[..., 0] *= -1.0    # flip x of every joint
    for li, ri in _RIC_LR_PAIRS:
        tmp = ric[..., li, :].copy()
        ric[..., li, :] = ric[..., ri, :]
        ric[..., ri, :] = tmp
    m[..., _RIC_START:_RIC_END] = ric.reshape(*T_axes, 63)

    # ---- rot_data: (..., 21, 6) — cont6d local rotations (joints 1..21) ----
    rot = m[..., _ROT_START:_ROT_END].reshape(*T_axes, 21, 6)
    # Per-joint cont6d mirror: signs [+, -, -, -, +, +]
    rot[..., 1] *= -1.0    # b1_y
    rot[..., 2] *= -1.0    # b1_z
    rot[..., 3] *= -1.0    # b2_x
    for li, ri in _RIC_LR_PAIRS:
        tmp = rot[..., li, :].copy()
        rot[..., li, :] = rot[..., ri, :]
        rot[..., ri, :] = tmp
    m[..., _ROT_START:_ROT_END] = rot.reshape(*T_axes, 126)

    # ---- local_velocity: (..., 22, 3) — joint velocities (joints 0..21) ----
    vel = m[..., _VEL_START:_VEL_END].reshape(*T_axes, 22, 3)
    vel[..., 0] *= -1.0    # flip x
    for li, ri in _SMPL22_LR_PAIRS:
        tmp = vel[..., li, :].copy()
        vel[..., li, :] = vel[..., ri, :]
        vel[..., ri, :] = tmp
    m[..., _VEL_START:_VEL_END] = vel.reshape(*T_axes, 66)

    # ---- foot_contact (4): swap [l_ankle, l_toe] ↔ [r_ankle, r_toe] ----
    feet = m[..., _FEET_START:_FEET_END].copy()
    m[..., _FEET_START:_FEET_START + 2] = feet[..., 2:4]
    m[..., _FEET_START + 2:_FEET_END] = feet[..., 0:2]

    return m


# ---------------------------------------------------------------------------
# World-frame object pose mirror
# ---------------------------------------------------------------------------

def mirror_object_world_pose(
    positions: np.ndarray,           # (..., 3) world-frame COM
    rotations: np.ndarray,           # (..., 3) world-frame axis-angle
) -> tuple[np.ndarray, np.ndarray]:
    """Mirror per-frame world-frame object position + axis-angle rotation
    through the body's sagittal plane (i.e., ``x → -x``).

    Position: flip x.

    Axis-angle: under reflection ``M = diag(-1, 1, 1)``, an axis-angle
    vector ``v = θ ω`` maps to ``v' = (v_x, -v_y, -v_z)``. Derivation:
    the mirrored axis is ``M ω = (-ω_x, ω_y, ω_z)``, but reflection
    reverses handedness so the mirrored angle becomes ``-θ``; hence
    ``v' = -θ · M ω = (θ ω_x, -θ ω_y, -θ ω_z)``.
    """
    pos_m = np.asarray(positions, dtype=np.float32).copy()
    pos_m[..., 0] *= -1.0

    rot_m = np.asarray(rotations, dtype=np.float32).copy()
    rot_m[..., 1] *= -1.0
    rot_m[..., 2] *= -1.0

    return pos_m, rot_m


# ---------------------------------------------------------------------------
# Sanity-check API: verify mirror is an involution on motion_263.
# ---------------------------------------------------------------------------

def _round_trip_max_error(motion: np.ndarray) -> float:
    """Helper for tests / smoke-checks: ``|m - mirror(mirror(m))|_∞``.

    Should be < 1e-6 on float32 after two mirrors (each is an exact
    permutation + sign flip).
    """
    return float(np.abs(motion - mirror_motion_263(mirror_motion_263(motion))).max())
