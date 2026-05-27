"""Round-29 Stage-2 oracle condition library.

Per ``analyses/2026-05-26_stage2_cond_injection_ablation_claude_code_prompt.md``,
the Stage-2 oracle condition has FOUR independent families, each with its
own variant search space:

    C : coarse scaffold       (C23, C38-current, C41-current, C38-root0, C41-root0)
    I : hand/object interaction (I0, I1, I2, I3, I4)
    S : support/gait           (S0, S1, S2, S3, S4)
    B : body-action refinement (B0, B1, B2, B3, B4)

This module exposes one builder per family plus a bundle entry point.
Each builder returns ``(tensor, info_dict)`` where ``info_dict`` reports
finite rate, shape, validity-mask rates etc. so callers (smoke test,
condition_stats script, summarizer) can detect degenerate conditions
(all-zero phase, always-on masks).

All math operates on NumPy. Builders are called per-clip inside the
dataset (CPU side), the same place Round-28's ``build_oracle_interaction_hint``
and ``build_body_action_oracle_hint`` already run. The Stage-2 trainer
just consumes the produced tensors via the existing ``cond[...]``
plumbing.

Conventions
-----------
- joints_22 : (T, 22, 3) world-frame SMPL-22 joints, metres. Y is up.
- object_positions : (T, 3) world-frame object centre, metres.
- object_rotations : (T, 3) world-frame axis-angle.
- contact_state   : (T, 5) per-part contact pseudo-label.
- fps : float, default 20 (project default).
- Y up. SMPL-22 indices reused from ``piano.data.interaction_hint``.

Frames (per the Round-29 prompt §3):
    current-yaw pelvis-local : local_j[t] = R_yaw(t).T @ (j_world[t] - pelvis_world[t])
    root0-yaw canonical      : local_j[t] = R_yaw(0).T @ (j_world[t] - pelvis_world[t])
    pelvis_delta             : pelvis_delta[t] = R_yaw(0).T @ (pelvis_world[t] - pelvis_world[0])

Pelvis MUST NOT use the pelvis-translated frame for body deltas
(that's identically zero by definition). See prompt §3.1 "Important".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from piano.data.interaction_hint import (
    LEFT_ANKLE_IDX,
    LEFT_KNEE_IDX,
    LEFT_WRIST_IDX,
    NECK_IDX,
    RIGHT_ANKLE_IDX,
    RIGHT_KNEE_IDX,
    RIGHT_WRIST_IDX,
    ROOT_IDX,
    derive_foot_stance_from_gt,
    derive_walking_mask_from_gt,
)
from piano.utils.canonical_frame import (
    _facing_angle_y,
    axis_angle_to_matrix_np,
    y_rotation_matrix,
)


DEFAULT_FPS: float = 20.0

# ---------------------------------------------------------------------------
# Family C — coarse scaffold key joints
# ---------------------------------------------------------------------------

# 5 non-pelvis key joints. Order is stable across all coarse variants.
COARSE_KEY_JOINT_INDICES: tuple[int, ...] = (
    LEFT_WRIST_IDX,
    RIGHT_WRIST_IDX,
    LEFT_KNEE_IDX,
    RIGHT_KNEE_IDX,
    NECK_IDX,
)
COARSE_KEY_JOINT_NAMES: tuple[str, ...] = (
    "left_wrist", "right_wrist", "left_knee", "right_knee", "neck",
)
NUM_COARSE_KEY_JOINTS: int = len(COARSE_KEY_JOINT_INDICES)  # 5
COARSE_KEY_JOINT_DIM: int = NUM_COARSE_KEY_JOINTS * 3       # 15
PELVIS_DELTA_DIM: int = 3                                    # 3

COARSE_VARIANT_DIMS: dict[str, int] = {
    "C23":          0,    # coarse_v1 only, no key-joint channels
    "C38-current":  COARSE_KEY_JOINT_DIM,                     # 15
    "C41-current":  COARSE_KEY_JOINT_DIM + PELVIS_DELTA_DIM,  # 18
    "C38-root0":    COARSE_KEY_JOINT_DIM,                     # 15
    "C41-root0":    COARSE_KEY_JOINT_DIM + PELVIS_DELTA_DIM,  # 18
}
"""Per-variant width of the EXTRA channel emitted by ``build_coarse_condition``.
``C23`` returns an empty (T, 0) channel — the Stage-1 Coarse-v1 (23-D)
stream still flows via the existing ``stage1_coarse`` cond key."""


# ---------------------------------------------------------------------------
# Family I — interaction
# ---------------------------------------------------------------------------

INTERACTION_VARIANT_DIMS: dict[str, int] = {
    "I0": 0,
    "I1-contact": 2,
    "I2-offset-masked": 6,
    "I3-contact-offset-masked": 8,
    "I4-contact-offset-unmasked": 8,
    # R29 failure-targeted ablation R5 (per
    # ``analyses/2026-05-27_round29_failure_targeted_ablation_prompt_for_claude_code.md`` §R5):
    # all 5 contact parts (L hand, R hand, L foot, R foot, pelvis) instead
    # of hands-only I3. 5 contact channels + 5 parts × 3 = 20D.
    "I5-allpart-contact-offset-masked": 20,
}

# Part-to-joint mapping for I5 (matches contact_state column order).
# Same body_part_indices used by piano.utils.smpl_utils.BODY_PART_INDICES.
ALLPART_CONTACT_JOINT_INDICES: tuple[int, ...] = (
    LEFT_WRIST_IDX,    # 0 left_hand
    RIGHT_WRIST_IDX,   # 1 right_hand
    LEFT_ANKLE_IDX,    # 2 left_foot
    RIGHT_ANKLE_IDX,   # 3 right_foot
    ROOT_IDX,          # 4 pelvis
)
NUM_ALLPART_CONTACT: int = len(ALLPART_CONTACT_JOINT_INDICES)  # 5


# ---------------------------------------------------------------------------
# Family S — support/gait
# ---------------------------------------------------------------------------

SUPPORT_VARIANT_DIMS: dict[str, int] = {
    "S0": 0,
    "S1-stance-height-walking": 5,            # 2 + 2 + 1
    "S2-S1-phase":              5 + 4,         # 9
    "S3-S1-footstep-target":    5 + 4,         # 9
    "S4-S1-phase-footstep":     5 + 4 + 4,    # 13
}


# ---------------------------------------------------------------------------
# Family B — body-action refinement
# ---------------------------------------------------------------------------

BODY_VARIANT_DIMS: dict[str, int] = {
    "B0": 0,
    "B1-mask-only":              NUM_COARSE_KEY_JOINTS,
    "B2-absolute-delta":         COARSE_KEY_JOINT_DIM,
    "B3-lowpass-residual":       COARSE_KEY_JOINT_DIM,
    "B4-lowpass-residual-mask":  NUM_COARSE_KEY_JOINTS + COARSE_KEY_JOINT_DIM,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _yaw_matrices_per_frame(joints_22: np.ndarray) -> np.ndarray:
    """Per-frame R_yaw(t).T (world -> body) from the cross-line at frame t.

    Returns (T, 3, 3). Uses the same facing-angle convention as
    ``_facing_angle_y`` but applied per-frame; this matches the
    "current-yaw pelvis-local" definition in prompt §3.1.
    """
    T = int(joints_22.shape[0])
    out = np.zeros((T, 3, 3), dtype=np.float32)
    for t in range(T):
        yaw_t = _facing_angle_y(joints_22[t])
        # ``y_rotation_matrix(yaw_t)`` rotates canonical (forward=+Z) into
        # world. We want world -> canonical (i.e. R_yaw(t).T) which is
        # ``y_rotation_matrix(-yaw_t)``.
        out[t] = y_rotation_matrix(-yaw_t)
    return out


def _frame0_yaw_matrix_T(joints_22: np.ndarray) -> np.ndarray:
    """R_yaw(0).T (world -> canonical) from frame 0 cross-line."""
    yaw0 = _facing_angle_y(joints_22[0])
    return y_rotation_matrix(-yaw0)


def _coarse_key_joint_deltas(
    joints_22: np.ndarray,
    coord_frame: str,
) -> np.ndarray:
    """Per-frame 5-joint pelvis-local delta (T, 5, 3) in the requested frame.

    ``coord_frame``:
        "current" — per-frame yaw R_yaw(t).T (current-yaw pelvis-local).
        "root0"   — frame-0 yaw R_yaw(0).T (root0-yaw canonical).

    delta_j[t] = local_j[t] - local_j[0] (by construction, t=0 row is 0).
    """
    T = int(joints_22.shape[0])
    pelvis_world = joints_22[:, ROOT_IDX, :].astype(np.float32)         # (T, 3)
    joints_key = joints_22[:, list(COARSE_KEY_JOINT_INDICES), :].astype(np.float32)  # (T, 5, 3)
    j_rel = joints_key - pelvis_world[:, None, :]                       # (T, 5, 3)

    if coord_frame == "current":
        R_T = _yaw_matrices_per_frame(joints_22)                       # (T, 3, 3)
        # local[t, j] = R_T[t] @ rel[t, j]
        local = np.einsum("tij,tkj->tki", R_T, j_rel).astype(np.float32)
    elif coord_frame == "root0":
        R_T0 = _frame0_yaw_matrix_T(joints_22)                          # (3, 3)
        local = np.einsum("ij,tkj->tki", R_T0, j_rel).astype(np.float32)
    else:
        raise ValueError(
            f"coord_frame must be 'current' or 'root0'; got {coord_frame!r}"
        )

    delta = local - local[0:1, :, :]                                    # (T, 5, 3)
    return delta.astype(np.float32)


def _pelvis_delta_root0(joints_22: np.ndarray) -> np.ndarray:
    """Pelvis displacement from frame-0, rotated into the frame-0 yaw frame.

    Returns (T, 3). Uses R_yaw(0).T regardless of variant — by §3.1's
    definition the pelvis cannot be expressed in the per-frame yaw frame
    (would be identically zero). Both C41-current and C41-root0 use the
    same root0-frame pelvis displacement; the "-current" / "-root0"
    distinction only affects the 5 non-pelvis joints.
    """
    R_T0 = _frame0_yaw_matrix_T(joints_22)
    pelvis_world = joints_22[:, ROOT_IDX, :].astype(np.float32)
    disp = pelvis_world - pelvis_world[0:1, :]                          # (T, 3)
    return (disp @ R_T0.T).astype(np.float32)


def _gaussian_temporal_smooth(
    x: np.ndarray, window: int = 9, sigma: float | None = None,
) -> np.ndarray:
    """1-D Gaussian temporal smoothing along axis 0.

    ``x``: (T, ...) — any trailing shape. Applied independently per
    channel via reflect-padded convolution. ``sigma`` defaults to
    ``window / 6`` (Gaussian effective support roughly = window).

    Used for B3/B4 low-pass residuals. Window = 9 frames at 20 fps is
    ~0.45s, slightly above typical step period — captures the smooth
    coarse body trajectory.
    """
    if window <= 1:
        return x.astype(np.float32, copy=True)
    sigma = float(sigma) if sigma is not None else float(window) / 6.0
    radius = window // 2
    t = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (t / sigma) ** 2)
    kernel = (kernel / kernel.sum()).astype(np.float32)                # (window,)

    # Convolve along axis 0 with reflect padding.
    pad = [(radius, radius)] + [(0, 0)] * (x.ndim - 1)
    xp = np.pad(x.astype(np.float32), pad, mode="reflect")
    # Manual stride-and-sum (no scipy dependency).
    out = np.zeros_like(x, dtype=np.float32)
    for i, w in enumerate(kernel):
        slc = (slice(i, i + x.shape[0]),) + (slice(None),) * (x.ndim - 1)
        out += float(w) * xp[slc]
    return out


# ---------------------------------------------------------------------------
# Public dataclass for the bundle
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Stage2ConditionBundle:
    """Container for the Round-29 condition bundle.

    Each field is ``None`` when the corresponding family is disabled
    (variant ``"<X>0"`` or coarse_variant=="C23"). Otherwise it is a
    ``(T, dim_family)`` float32 array. ``info`` mirrors the validity
    / shape / stat metadata each builder emits, keyed by family name.
    """
    coarse_extra: np.ndarray | None = None
    interaction: np.ndarray | None = None
    support: np.ndarray | None = None
    body_refine: np.ndarray | None = None
    info: dict[str, dict[str, Any]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# C — Coarse condition
# ---------------------------------------------------------------------------

def build_coarse_condition(
    joints_22: np.ndarray,
    variant: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build coarse scaffold EXTRA channel of shape (T, dim) for ``variant``.

    The base Stage-1 Coarse-v1 (23-D) channel is NOT emitted here — it
    is already produced by ``piano.data.stage1_coarse_oracle`` and
    plumbed via ``cond["stage1_coarse"]``. This builder only emits the
    additional key-joint / pelvis-delta channels that distinguish
    C38/C41 variants from the C23 baseline.

    Returns:
        coarse_extra : (T, dim) float32. dim per ``COARSE_VARIANT_DIMS``.
        info : dict with shape, finite_frac, mean/std summary.
    """
    if variant not in COARSE_VARIANT_DIMS:
        raise ValueError(
            f"coarse variant must be one of {sorted(COARSE_VARIANT_DIMS)}; "
            f"got {variant!r}"
        )
    T = int(joints_22.shape[0])
    dim = COARSE_VARIANT_DIMS[variant]
    info: dict[str, Any] = {
        "variant": variant, "T": T, "dim": dim,
    }
    if dim == 0:
        out = np.zeros((T, 0), dtype=np.float32)
        info.update(finite_frac=1.0, mean=0.0, std=0.0)
        return out, info

    if variant in ("C38-current", "C41-current"):
        delta = _coarse_key_joint_deltas(joints_22, "current")           # (T, 5, 3)
    else:  # C38-root0, C41-root0
        delta = _coarse_key_joint_deltas(joints_22, "root0")
    out = delta.reshape(T, COARSE_KEY_JOINT_DIM)                         # (T, 15)

    if variant in ("C41-current", "C41-root0"):
        pelvis_d = _pelvis_delta_root0(joints_22)                        # (T, 3)
        out = np.concatenate([out, pelvis_d], axis=-1)                   # (T, 18)

    if out.shape != (T, dim):
        raise AssertionError(
            f"coarse builder shape {out.shape} != ({T}, {dim}); variant={variant!r}"
        )

    finite = np.isfinite(out)
    info.update(
        finite_frac=float(finite.mean()),
        mean=float(np.nan_to_num(out).mean()),
        std=float(np.nan_to_num(out).std()),
        max_abs=float(np.abs(np.nan_to_num(out)).max()),
        delta_t0_max_abs=float(np.abs(out[0]).max()) if T > 0 else 0.0,
    )
    return out.astype(np.float32, copy=False), info


# ---------------------------------------------------------------------------
# I — Interaction condition
# ---------------------------------------------------------------------------

def _hand_object_local_offset(
    joints_22: np.ndarray,
    object_positions: np.ndarray,
    object_rotations: np.ndarray,
    hand_offset_clamp_m: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute object-local hand offset and the L/R contact-side absence
    mask is the caller's job. Returns ``rel`` of shape (T, 2, 3),
    unclamped/uncentred raw object-local position, plus a clamped
    version normalised to [-1, 1]."""
    T = int(joints_22.shape[0])
    R_obj = axis_angle_to_matrix_np(object_rotations.astype(np.float32))   # (T, 3, 3)
    R_obj_T = R_obj.transpose(0, 2, 1)
    wrist_world = np.stack(
        [joints_22[:, LEFT_WRIST_IDX, :], joints_22[:, RIGHT_WRIST_IDX, :]],
        axis=1,
    ).astype(np.float32)                                                    # (T, 2, 3)
    obj_pos = object_positions[:, None, :].astype(np.float32)               # (T, 1, 3)
    rel = np.einsum("tij,thj->thi", R_obj_T, wrist_world - obj_pos).astype(np.float32)
    rel_norm = np.clip(rel, -hand_offset_clamp_m, hand_offset_clamp_m) / hand_offset_clamp_m
    return rel.astype(np.float32, copy=False), rel_norm.astype(np.float32, copy=False)


def _allpart_object_local_offset(
    joints_22: np.ndarray,
    object_positions: np.ndarray,
    object_rotations: np.ndarray,
    offset_clamp_m: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Object-local offset for the 5 contact parts (L hand, R hand, L foot,
    R foot, pelvis).

    Generalises ``_hand_object_local_offset`` from 2 hand joints to all 5
    contact-bearing joints. Returns ``(rel, rel_norm)`` where ``rel`` is the
    raw object-local position and ``rel_norm`` is clamped to ``±clamp_m`` and
    divided by ``clamp_m`` to land in ``[-1, 1]``.

    Note: the prompt asked us to keep the config key name
    ``r29_hand_offset_clamp_m`` and reuse it for all 5 parts (no broad rename).
    """
    T = int(joints_22.shape[0])
    R_obj = axis_angle_to_matrix_np(object_rotations.astype(np.float32))   # (T, 3, 3)
    R_obj_T = R_obj.transpose(0, 2, 1)
    parts_world = np.stack(
        [joints_22[:, idx, :] for idx in ALLPART_CONTACT_JOINT_INDICES],
        axis=1,
    ).astype(np.float32)                                                    # (T, 5, 3)
    obj_pos = object_positions[:, None, :].astype(np.float32)               # (T, 1, 3)
    rel = np.einsum("tij,thj->thi", R_obj_T, parts_world - obj_pos).astype(np.float32)
    rel_norm = np.clip(rel, -offset_clamp_m, offset_clamp_m) / offset_clamp_m
    return rel.astype(np.float32, copy=False), rel_norm.astype(np.float32, copy=False)


def build_interaction_condition(
    joints_22: np.ndarray,
    object_positions: np.ndarray | None,
    object_rotations: np.ndarray | None,
    contact_state: np.ndarray | None,
    variant: str,
    hand_offset_clamp_m: float = 2.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build interaction channel for ``variant`` ∈ {I0..I5}.

    All non-I0 variants require ``object_positions``, ``object_rotations``,
    ``contact_state`` to be present. The caller (dataset) is responsible
    for guarding via ``data.use_interaction_condition`` / variant.

    I5-allpart (R29 failure-targeted ablation): all 5 contact parts
    (L hand, R hand, L foot, R foot, pelvis) instead of hands-only I3.
    Layout: ``[contacts (5), offsets_masked (5×3 = 15)]`` → 20D.
    Per-part contact mask gates each part's offset to zero on non-contact
    frames (same masking semantics as I3). ``hand_offset_clamp_m`` is
    reused as the universal clamp for all 5 parts.
    """
    if variant not in INTERACTION_VARIANT_DIMS:
        raise ValueError(
            f"interaction variant must be one of "
            f"{sorted(INTERACTION_VARIANT_DIMS)}; got {variant!r}"
        )
    T = int(joints_22.shape[0])
    dim = INTERACTION_VARIANT_DIMS[variant]
    info: dict[str, Any] = {"variant": variant, "T": T, "dim": dim}
    if dim == 0:
        out = np.zeros((T, 0), dtype=np.float32)
        info.update(finite_frac=1.0, contact_frame_frac=0.0)
        return out, info

    if object_positions is None or object_rotations is None or contact_state is None:
        raise ValueError(
            f"interaction variant {variant!r} requires object_positions, "
            "object_rotations, and contact_state (got at least one None)."
        )

    pieces: list[np.ndarray] = []
    if variant == "I5-allpart-contact-offset-masked":
        # 5-part contact (L hand, R hand, L foot, R foot, pelvis).
        # Reuse hand_offset_clamp_m as the universal clamp per prompt §R5.
        all_contact = np.clip(
            contact_state[:, 0:NUM_ALLPART_CONTACT].astype(np.float32), 0.0, 1.0,
        )                                                                   # (T, 5)
        _, rel_norm5 = _allpart_object_local_offset(
            joints_22, object_positions, object_rotations,
            offset_clamp_m=hand_offset_clamp_m,
        )                                                                   # (T, 5, 3)
        offset_masked = rel_norm5 * all_contact[:, :, None]                 # (T, 5, 3)
        pieces.append(all_contact)
        pieces.append(offset_masked.reshape(T, NUM_ALLPART_CONTACT * 3))
        # info-dict aggregates differ from I1-I4; computed below.
        contact_mask = (all_contact > 0.5).astype(np.float32)               # (T, 5)
        hand_contact_for_info = all_contact[:, :2]                           # backward-compat
    else:
        # I1/I2/I3/I4 — hands only.
        hand_contact_for_info = np.clip(
            contact_state[:, [0, 1]].astype(np.float32), 0.0, 1.0,
        )                                                                   # (T, 2)
        contact_mask = (hand_contact_for_info > 0.5).astype(np.float32)     # (T, 2)
        if variant == "I1-contact":
            pieces.append(hand_contact_for_info)
        else:
            # I2/I3/I4 need offset.
            _, rel_norm = _hand_object_local_offset(
                joints_22, object_positions, object_rotations,
                hand_offset_clamp_m=hand_offset_clamp_m,
            )                                                               # (T, 2, 3)
            if variant == "I2-offset-masked":
                offset = rel_norm * hand_contact_for_info[:, :, None]
                pieces.append(offset.reshape(T, 6))
            elif variant == "I3-contact-offset-masked":
                offset = rel_norm * hand_contact_for_info[:, :, None]
                pieces.append(hand_contact_for_info)
                pieces.append(offset.reshape(T, 6))
            elif variant == "I4-contact-offset-unmasked":
                pieces.append(hand_contact_for_info)
                pieces.append(rel_norm.reshape(T, 6))
            else:
                raise AssertionError(f"unreachable I-variant {variant!r}")

    out = np.concatenate(pieces, axis=-1) if pieces else np.zeros((T, 0), dtype=np.float32)
    if out.shape != (T, dim):
        raise AssertionError(
            f"interaction builder shape {out.shape} != ({T}, {dim}); "
            f"variant={variant!r}"
        )
    if not np.isfinite(out).all():
        raise FloatingPointError(
            f"interaction variant {variant!r} produced non-finite values."
        )
    info.update(
        finite_frac=float(np.isfinite(out).mean()),
        contact_frame_frac=float(contact_mask.max(axis=-1).mean()),
        left_contact_frac=float(contact_mask[:, 0].mean()),
        right_contact_frac=float(contact_mask[:, 1].mean()),
        max_abs=float(np.abs(out).max() if out.size else 0.0),
    )
    if variant == "I5-allpart-contact-offset-masked":
        # Additional per-part contact fractions (foot + pelvis) so smoke
        # tests / condition_stats can sanity-check the new channels.
        info.update(
            left_foot_contact_frac=float(contact_mask[:, 2].mean()),
            right_foot_contact_frac=float(contact_mask[:, 3].mean()),
            pelvis_contact_frac=float(contact_mask[:, 4].mean()),
        )
    return out.astype(np.float32, copy=False), info


# ---------------------------------------------------------------------------
# S — Support / gait
# ---------------------------------------------------------------------------

def _foot_phase_sincos_from_stance(
    foot_stance: np.ndarray, fps: float, walking_threshold: float = 0.2,
) -> tuple[np.ndarray, float]:
    """Derive per-foot phase (sin, cos) from stance/swing alternation.

    foot_stance : (T, 2) soft stance in [0, 1].
    Returns:
        sincos : (T, 2, 2) — (sin(phi), cos(phi)) per foot.
        valid_frac : fraction of frames where a stance/swing transition
            was actually observed (i.e. there was a gait cycle to phase).

    Implementation: assign phase 0 at each stance-onset frame, phase π at
    each swing-onset frame, and linearly interpolate between. Frames
    before the first detected transition (or after the last) inherit
    the nearest event's phase. When NO transitions are detected for a
    foot, that foot's sincos is set to zero and counted as invalid.
    """
    T, F = foot_stance.shape
    if F != 2:
        raise ValueError(f"foot_stance must be (T, 2); got {foot_stance.shape!r}")
    sincos = np.zeros((T, 2, 2), dtype=np.float32)
    valid_per_foot = np.zeros(2, dtype=np.float32)

    for f_idx in range(2):
        stance = foot_stance[:, f_idx] > 0.5                                # bool (T,)
        # Stance-onset: stance frame whose previous frame was non-stance.
        events = []   # list of (frame, phase) — phase=0 for stance-onset, pi for swing-onset
        for t in range(1, T):
            if stance[t] and not stance[t - 1]:
                events.append((t, 0.0))
            elif not stance[t] and stance[t - 1]:
                events.append((t, float(np.pi)))
        if not events:
            continue  # leave sincos = 0; valid stays 0

        valid_per_foot[f_idx] = 1.0
        # Build a per-frame phase by linear interpolation between events.
        # Outside [first_event, last_event] we hold the nearest event's
        # phase constant (the prompt allows zeros with a warning when no
        # robust phase can be derived; clip start/end with a constant is
        # acceptable and avoids edge discontinuity).
        ev_frames = np.array([e[0] for e in events], dtype=np.float32)
        ev_phases = np.array([e[1] for e in events], dtype=np.float32)
        # Unwrap phases: each transition advances by pi.
        for i in range(1, len(ev_phases)):
            # ensure monotonic with +pi increments
            if ev_phases[i] <= ev_phases[i - 1]:
                ev_phases[i] += float(np.pi)
        # Interpolate (frame -> phase). For frames outside the event
        # range, np.interp clips to ev_phases[0] / ev_phases[-1].
        phase = np.interp(
            np.arange(T, dtype=np.float32), ev_frames, ev_phases,
        ).astype(np.float32)
        sincos[:, f_idx, 0] = np.sin(phase)
        sincos[:, f_idx, 1] = np.cos(phase)
    valid_frac = float(valid_per_foot.mean())
    return sincos, valid_frac


def _footstep_target_local_xz(
    joints_22: np.ndarray,
    foot_stance: np.ndarray,
) -> tuple[np.ndarray, float]:
    """For each frame and foot, the next stance-contact foot location in
    pelvis-local XZ at frame 0 (root0-yaw canonical) coords.

    Returns:
        target : (T, 2, 2) — (x, z) per foot, root0-canonical at frame 0.
        valid_frac : per-foot mean validity (1 when ≥1 stance segment).
    """
    T, F = foot_stance.shape
    if F != 2:
        raise ValueError(f"foot_stance must be (T, 2); got {foot_stance.shape!r}")
    R_T0 = _frame0_yaw_matrix_T(joints_22)
    pelvis0 = joints_22[0, ROOT_IDX, :].astype(np.float32)
    ankle_idx = (LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX)
    target = np.zeros((T, 2, 2), dtype=np.float32)
    valid_per_foot = np.zeros(2, dtype=np.float32)

    for f_idx in range(2):
        stance = foot_stance[:, f_idx] > 0.5                                # bool
        if not stance.any():
            continue
        valid_per_foot[f_idx] = 1.0
        # Identify stance segments [start, end] and assign each frame its
        # NEXT stance segment's representative position (centroid frame
        # in pelvis-local XZ at frame 0). Frames already inside a stance
        # segment use that segment's own centroid; frames after the last
        # segment hold the last segment's centroid (clip end fallback).
        seg_starts = []
        seg_ends = []
        in_seg = False
        for t in range(T):
            if stance[t] and not in_seg:
                seg_starts.append(t); in_seg = True
            elif not stance[t] and in_seg:
                seg_ends.append(t - 1); in_seg = False
        if in_seg:
            seg_ends.append(T - 1)
        if not seg_starts:
            continue
        # Per-segment centroid in pelvis-local XZ at frame 0.
        centroids_world = []
        for s, e in zip(seg_starts, seg_ends):
            ankle_world = joints_22[s:e + 1, ankle_idx[f_idx], :].astype(np.float32)
            centroids_world.append(ankle_world.mean(axis=0))                 # (3,)
        centroids_world_arr = np.stack(centroids_world, axis=0)              # (S, 3)
        centroids_local = ((centroids_world_arr - pelvis0[None, :]) @ R_T0.T)
        centroids_xz = centroids_local[:, [0, 2]]                             # (S, 2)

        # Assign each frame to the nearest upcoming stance segment.
        for t in range(T):
            future_segs = [i for i, s in enumerate(seg_starts) if s >= t]
            # If frame t lies INSIDE a stance segment, use that segment.
            inside = None
            for i, (s, e) in enumerate(zip(seg_starts, seg_ends)):
                if s <= t <= e:
                    inside = i
                    break
            if inside is not None:
                target[t, f_idx, :] = centroids_xz[inside]
            elif future_segs:
                target[t, f_idx, :] = centroids_xz[future_segs[0]]
            else:
                target[t, f_idx, :] = centroids_xz[-1]
    valid_frac = float(valid_per_foot.mean())
    return target, valid_frac


def build_support_condition(
    joints_22: np.ndarray,
    variant: str,
    fps: float = DEFAULT_FPS,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build support/gait channel for ``variant`` ∈ {S0..S4}.

    Only depends on ``joints_22`` (the foot-stance, height, walking,
    phase and footstep_target are all derived from GT ankles +
    pelvis). Per prompt §3.3, foot stance is from GT ankle and NOT
    from the InterAct foot-object pseudo-label.
    """
    if variant not in SUPPORT_VARIANT_DIMS:
        raise ValueError(
            f"support variant must be one of {sorted(SUPPORT_VARIANT_DIMS)}; "
            f"got {variant!r}"
        )
    T = int(joints_22.shape[0])
    dim = SUPPORT_VARIANT_DIMS[variant]
    info: dict[str, Any] = {"variant": variant, "T": T, "dim": dim}
    if dim == 0:
        out = np.zeros((T, 0), dtype=np.float32)
        info.update(finite_frac=1.0)
        return out, info

    foot_stance, ankle_height_norm = derive_foot_stance_from_gt(
        joints_22, fps=fps,
    )                                                                       # (T, 2), (T, 2)
    walking_mask = derive_walking_mask_from_gt(joints_22, fps=fps)          # (T, 1)
    s1 = np.concatenate(
        [foot_stance, ankle_height_norm, walking_mask], axis=-1,
    ).astype(np.float32)                                                    # (T, 5)

    pieces: list[np.ndarray] = [s1]
    phase_valid = 1.0
    footstep_valid = 1.0
    if variant in ("S2-S1-phase", "S4-S1-phase-footstep"):
        phase_sincos, phase_valid = _foot_phase_sincos_from_stance(
            foot_stance, fps=fps,
        )                                                                   # (T, 2, 2)
        pieces.append(phase_sincos.reshape(T, 4))
    if variant in ("S3-S1-footstep-target", "S4-S1-phase-footstep"):
        target, footstep_valid = _footstep_target_local_xz(
            joints_22, foot_stance,
        )                                                                   # (T, 2, 2)
        # Normalise XZ to [-1, 1] via clamp at ±3 m (typical stride/range).
        target_norm = np.clip(target, -3.0, 3.0) / 3.0
        pieces.append(target_norm.reshape(T, 4))

    out = np.concatenate(pieces, axis=-1)
    if out.shape != (T, dim):
        raise AssertionError(
            f"support builder shape {out.shape} != ({T}, {dim}); "
            f"variant={variant!r}"
        )
    if not np.isfinite(out).all():
        raise FloatingPointError(
            f"support variant {variant!r} produced non-finite values."
        )
    info.update(
        finite_frac=float(np.isfinite(out).mean()),
        stance_frame_frac_left=float((foot_stance[:, 0] > 0.5).mean()),
        stance_frame_frac_right=float((foot_stance[:, 1] > 0.5).mean()),
        walking_frame_frac=float(walking_mask.mean()),
        phase_valid_frame_frac=float(phase_valid),
        footstep_target_valid_frame_frac=float(footstep_valid),
        max_abs=float(np.abs(out).max() if out.size else 0.0),
    )
    return out.astype(np.float32, copy=False), info


# ---------------------------------------------------------------------------
# B — Body refinement
# ---------------------------------------------------------------------------

def build_body_refinement_condition(
    joints_22: np.ndarray,
    variant: str,
    coord_frame: str = "current",
    energy_threshold: float = 0.05,
    lowpass_window: int = 9,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build body-action refinement channel for ``variant`` ∈ {B0..B4}.

    ``coord_frame``: "current" or "root0" — should match the active
    coarse variant's frame so coarse and refinement live in the same
    coordinate convention (prompt §3.4 + §3.5 FULL-ROOT0 sanity-check).

    Codex post-review (2026-05-26) design-check note on B3 / B4:
    --------------------------------------------------------------
    Per the prompt §3.4, B3/B4 are described as the "Stage-1.5-like
    body refinement = full_delta − lowpass(full_delta)" half of a
    lowpass + residual factorization (Stage-1 would emit lowpass).

    In this implementation the coarse variants (C38 / C41) emit the
    FULL key-joint delta — NOT the lowpass component — and B3 / B4
    emit the high-frequency residual. So when a Stage-2 variant
    combines C38/C41 with B3/B4, the model receives:

        coarse_extra : full delta
        body_refine  : residual = full delta − lowpass(full delta)

    Which is a residual SIDE-CHANNEL, not a pure lowpass+residual
    factorization (Stage-2 sees both full and residual, and the
    redundancy is left to the model to compress).

    A pure factorization (Stage-1 emits lowpass, Stage-1.5 emits
    residual) would require a dedicated lowpass coarse variant — that
    is intentionally NOT in the R29 matrix to avoid an unreviewed
    expansion; it would land in a follow-up round if B3/B4 proves
    valuable on its own merits. See r29_e3 / r29_e4 manifest purpose
    text for the explicit caveat.
    """
    if variant not in BODY_VARIANT_DIMS:
        raise ValueError(
            f"body variant must be one of {sorted(BODY_VARIANT_DIMS)}; "
            f"got {variant!r}"
        )
    T = int(joints_22.shape[0])
    dim = BODY_VARIANT_DIMS[variant]
    info: dict[str, Any] = {
        "variant": variant, "T": T, "dim": dim,
        "coord_frame": coord_frame, "lowpass_window": int(lowpass_window),
    }
    if dim == 0:
        out = np.zeros((T, 0), dtype=np.float32)
        info.update(finite_frac=1.0)
        return out, info

    delta = _coarse_key_joint_deltas(joints_22, coord_frame)                # (T, 5, 3)
    # Per-joint energy (mean ||delta|| over T) — used by B1/B4 mask.
    energy = np.linalg.norm(delta, axis=-1).mean(axis=0)                    # (5,)
    active_mask = (energy > float(energy_threshold)).astype(np.float32)     # (5,)
    mask_t = np.broadcast_to(active_mask[None, :], (T, NUM_COARSE_KEY_JOINTS)).astype(np.float32)

    if variant == "B1-mask-only":
        out = mask_t                                                        # (T, 5)
    elif variant == "B2-absolute-delta":
        out = delta.reshape(T, COARSE_KEY_JOINT_DIM)                        # (T, 15)
    elif variant in ("B3-lowpass-residual", "B4-lowpass-residual-mask"):
        smoothed = _gaussian_temporal_smooth(delta, window=lowpass_window)  # (T, 5, 3)
        residual = (delta - smoothed).reshape(T, COARSE_KEY_JOINT_DIM)
        if variant == "B3-lowpass-residual":
            out = residual
        else:  # B4: prepend mask
            out = np.concatenate([mask_t, residual], axis=-1)               # (T, 5+15)
    else:
        raise AssertionError(f"unreachable B-variant {variant!r}")

    if out.shape != (T, dim):
        raise AssertionError(
            f"body builder shape {out.shape} != ({T}, {dim}); variant={variant!r}"
        )
    if not np.isfinite(out).all():
        raise FloatingPointError(
            f"body variant {variant!r} produced non-finite values."
        )
    info.update(
        finite_frac=float(np.isfinite(out).mean()),
        max_abs=float(np.abs(out).max() if out.size else 0.0),
        active_joint_frac=float(active_mask.mean()),
        per_joint_energy=energy.astype(np.float32).tolist(),
    )
    return out.astype(np.float32, copy=False), info


# ---------------------------------------------------------------------------
# Bundle entry
# ---------------------------------------------------------------------------

def build_stage2_condition_bundle(
    joints_22: np.ndarray,
    *,
    coarse_variant: str = "C23",
    interaction_variant: str = "I0",
    support_variant: str = "S0",
    body_variant: str = "B0",
    body_coord_frame: str | None = None,
    body_energy_threshold: float = 0.05,
    body_lowpass_window: int = 9,
    object_positions: np.ndarray | None = None,
    object_rotations: np.ndarray | None = None,
    contact_state: np.ndarray | None = None,
    fps: float = DEFAULT_FPS,
    hand_offset_clamp_m: float = 2.0,
) -> Stage2ConditionBundle:
    """Build all four condition families for one clip.

    ``body_coord_frame`` defaults to "current" when the coarse variant
    is C38-current / C41-current / C23, and to "root0" when the coarse
    variant is C38-root0 / C41-root0. Explicit override is accepted for
    sanity-check ablations.
    """
    if body_coord_frame is None:
        if coarse_variant.endswith("root0"):
            body_coord_frame = "root0"
        else:
            body_coord_frame = "current"

    bundle = Stage2ConditionBundle()

    coarse_extra, c_info = build_coarse_condition(joints_22, coarse_variant)
    bundle.coarse_extra = coarse_extra if coarse_extra.shape[-1] > 0 else None
    bundle.info["coarse"] = c_info

    interaction, i_info = build_interaction_condition(
        joints_22, object_positions, object_rotations, contact_state,
        variant=interaction_variant,
        hand_offset_clamp_m=hand_offset_clamp_m,
    )
    bundle.interaction = interaction if interaction.shape[-1] > 0 else None
    bundle.info["interaction"] = i_info

    support, s_info = build_support_condition(joints_22, support_variant, fps=fps)
    bundle.support = support if support.shape[-1] > 0 else None
    bundle.info["support"] = s_info

    body, b_info = build_body_refinement_condition(
        joints_22, body_variant,
        coord_frame=body_coord_frame,
        energy_threshold=body_energy_threshold,
        lowpass_window=body_lowpass_window,
    )
    bundle.body_refine = body if body.shape[-1] > 0 else None
    bundle.info["body_refine"] = b_info

    return bundle


# ---------------------------------------------------------------------------
# Dim helpers — used by config generator and smoke test
# ---------------------------------------------------------------------------

def coarse_dim(variant: str) -> int:
    """EXTRA channel width for the coarse variant (not counting Stage-1 23-D)."""
    return COARSE_VARIANT_DIMS[variant]


def interaction_dim(variant: str) -> int:
    return INTERACTION_VARIANT_DIMS[variant]


def support_dim(variant: str) -> int:
    return SUPPORT_VARIANT_DIMS[variant]


def body_dim(variant: str) -> int:
    return BODY_VARIANT_DIMS[variant]


__all__ = [
    "Stage2ConditionBundle",
    "build_coarse_condition",
    "build_interaction_condition",
    "build_support_condition",
    "build_body_refinement_condition",
    "build_stage2_condition_bundle",
    "coarse_dim",
    "interaction_dim",
    "support_dim",
    "body_dim",
    "COARSE_VARIANT_DIMS",
    "INTERACTION_VARIANT_DIMS",
    "SUPPORT_VARIANT_DIMS",
    "BODY_VARIANT_DIMS",
    "COARSE_KEY_JOINT_INDICES",
    "COARSE_KEY_JOINT_NAMES",
    "NUM_COARSE_KEY_JOINTS",
]
