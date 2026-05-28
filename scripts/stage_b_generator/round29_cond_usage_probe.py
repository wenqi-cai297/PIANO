"""Round-29 condition-usage probe — Phase 0 (mandatory pre-PB).

Per Codex review of `analyses/2026-05-29_round29_cond_injection_prior_review_for_codex.md`
§3, before any architectural ablation (PB1/PB2) we must directly measure
whether each R29 condition family (coarse_extra / interaction / support /
body_refine) is actually consumed by a trained Stage-2 ckpt — not just
inferred from paired-bootstrap comparisons across runs.

Mechanism:
  1. For each (ckpt, family, perturbation) combination, run the model's
     sampler on the same val 48-clip selection.
  2. Compare generated joints against an unperturbed baseline produced
     with the SAME random seed, so the resulting delta is the model's
     condition response, not denoising noise.
  3. Aggregate per-family deltas and emit Codex's heuristic triage labels:
       ignored, weakly_used, actively_used, temporally_used.

Perturbations (Codex §3.1):
  - baseline       : no perturbation; reference sample with seed
  - zero           : family tensor multiplied by 0
  - time_shuffle   : within each clip, permute the valid_T frames of the
                     family (padded frames untouched); see if the model
                     uses temporal structure
  - batch_shuffle  : swap the family tensor across clips inside a mini-
                     batch of 2 (rejects batch_size=1 if requested)
  - scale_0.5      : family tensor scaled by 0.5
  - scale_2.0      : family tensor scaled by 2.0

We only ever mutate the 4 R29 cond keys:
    stage2_coarse_extra, stage2_interaction, stage2_support, stage2_body_refine
Everything else — Stage-1 Coarse-v1, object tokens, init_pose, text — is
left exactly as the dataset / oracle produced it.

Outputs:
    <output-dir>/cond_usage_stats.json   # full aggregate with all rows
    <output-dir>/cond_usage_summary.md   # human-readable triage labels

Usage:
    python scripts/stage_b_generator/round29_cond_usage_probe.py \\
        --config configs/training/anchordiff_r29_ns_a1_c41_s4_g1.yaml \\
        --ckpt   runs/training/stageB_anchordiff_r29_ns_a1_c41_s4_g1/final.pt \\
        --selection-json analyses/round29_val_diag_indices_48_balanced.json \\
        --bucket val \\
        --output-dir analyses/round29_cond_usage_a1_val \\
        --variant-id r29_ns_a1_c41_s4_g1

Each run probes ONE ckpt. The shell launcher loops over A1/R0/G1.

Pure helpers (perturbation primitives + label thresholds) are importable
without torch/omegaconf so unit tests can validate the math on synthetic
arrays.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# R29 condition family keys in the cond dict. Mirrors
# src/piano/models/round29_cond_injection.py FAMILY_NAMES.
FAMILY_OF_KEY: dict[str, str] = {
    "stage2_coarse_extra": "coarse_extra",
    "stage2_interaction": "interaction",
    "stage2_support": "support",
    "stage2_body_refine": "body_refine",
}
KEY_OF_FAMILY: dict[str, str] = {v: k for k, v in FAMILY_OF_KEY.items()}

# SMPL-22 indices for the 6 key joints in the summary.
KEY_JOINT_INDICES: dict[str, int] = {
    "left_wrist": 20, "right_wrist": 21,
    "left_ankle": 7, "right_ankle": 8,
    "neck": 12, "pelvis": 0,
}

# Codex §3.3 thresholds (heuristic triage labels, not paper claims).
THRESH_IGNORED_KEY_CM: float = 1.0       # zeroing changes key joints by <1 cm
THRESH_IGNORED_RELATIVE: float = 0.05    # <5% relative target metric change
THRESH_WEAK_KEY_CM: float = 3.0
THRESH_WEAK_RELATIVE: float = 0.15
THRESH_TEMPORALLY_USED_FRACTION: float = 1.20  # time_shuffle hurts ≥20% more than zero


# --------------------------------------------------------------------------- #
# Perturbation primitives (pure, importable without torch).
# --------------------------------------------------------------------------- #


def perturbation_zero(family_tensor: np.ndarray) -> np.ndarray:
    """Zero perturbation: replace the family tensor with zeros of same shape."""
    return np.zeros_like(family_tensor)


def perturbation_scale(family_tensor: np.ndarray, k: float) -> np.ndarray:
    """Scale family tensor by k."""
    return family_tensor * float(k)


def perturbation_time_shuffle(
    family_tensor: np.ndarray,
    valid_T: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Within each clip, permute only the valid_T frames; leave padded
    frames (indices >= valid_T) unchanged.

    family_tensor shape: (B, T, D)
    """
    if family_tensor.ndim != 3:
        raise ValueError(
            f"time_shuffle expects (B, T, D) tensor, got shape "
            f"{family_tensor.shape}"
        )
    if valid_T < 2:
        # Nothing to permute meaningfully; return as-is.
        return family_tensor.copy()
    out = family_tensor.copy()
    for b in range(family_tensor.shape[0]):
        perm = rng.permutation(min(valid_T, family_tensor.shape[1]))
        out[b, :len(perm)] = family_tensor[b, perm]
    return out


def _derange_indices(
    n: int, rng: np.random.RandomState, max_tries: int = 16,
) -> list[int]:
    """Return a length-n list of indices that is a *derangement* of
    ``range(n)`` — every position is mapped to a different value.

    Used by both ``perturbation_batch_shuffle`` (on the batch axis of a
    family tensor) and by the cond-usage probe ``main()`` loop (on the
    clip-index axis for batch_shuffle pairing). Centralising the
    derangement logic guarantees the unit-tested invariants apply to
    the runtime pairing too.

    Strategy: try random permutations until we get one with no fixed
    point; fall back to a roll-by-1 (always a valid derangement for
    ``n >= 2``). Raises if ``n < 2`` because no derangement exists.
    """
    if n < 2:
        raise ValueError(
            f"_derange_indices requires n >= 2 (got {n}); a derangement "
            "is undefined for n < 2."
        )
    base = np.arange(n)
    for _ in range(max_tries):
        perm = rng.permutation(n)
        if not (perm == base).any():
            return perm.tolist()
    return np.roll(base, 1).tolist()


def perturbation_batch_shuffle(
    family_tensor: np.ndarray,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Permute the batch dimension. Requires B >= 2 (caller responsibility).

    family_tensor shape: (B, T, D); returns shape (B, T, D) with rows shuffled.
    Internally delegates to ``_derange_indices`` so the same derangement
    semantics apply at the clip-index level in ``main()``.
    """
    if family_tensor.ndim != 3:
        raise ValueError(
            f"batch_shuffle expects (B, T, D) tensor, got shape "
            f"{family_tensor.shape}"
        )
    B = family_tensor.shape[0]
    if B < 2:
        raise ValueError(
            "batch_shuffle requires batch size >= 2; got 1. "
            "Caller must build a mini-batch of >= 2 clips for this "
            "perturbation."
        )
    perm = _derange_indices(B, rng)
    return family_tensor[perm].copy()


# --------------------------------------------------------------------------- #
# Aggregation + labelling (pure, importable).
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class PerClipPerturbationResult:
    """One row: (variant_id, family, perturbation, clip) → delta metrics.

    The task-metric deltas (sustained_contact / gait / body_action) are
    cheap proxies of the diag scripts' full per-segment metrics, kept
    purely on per-frame whole-clip joint statistics. They are recorded
    as **fractional change** (|pert − base| / |base|) so the labeller's
    relative thresholds (Codex §3.3, 5 % / 15 %) read directly.

    `nan` means the metric could not be computed for this clip (e.g. no
    contact frames for sustained_contact; no walking frames for gait).
    """
    variant_id: str
    bucket: str
    family: str
    perturbation: str
    subset: str
    seq_id: str
    pred_delta_joints_cm_mean: float
    pred_delta_joints_cm_p95: float
    key_joint_delta_cm: dict[str, float]
    # R4: task-metric proxies expressed as fractional change vs baseline.
    sustained_contact_delta_rel: float
    gait_delta_rel: float
    body_action_delta_rel: float


def _per_joint_delta_cm(
    joints_a: np.ndarray, joints_b: np.ndarray,
) -> np.ndarray:
    """Per-frame, per-joint Euclidean error in cm. Shape (T, 22)."""
    T = min(joints_a.shape[0], joints_b.shape[0])
    diff = joints_a[:T] - joints_b[:T]
    err_m = np.linalg.norm(diff, axis=-1)
    return err_m * 100.0


def _aa_to_rotation_matrix_np(aa: np.ndarray) -> np.ndarray:
    """Rodrigues, numpy version. ``aa`` shape (..., 3) → R shape (..., 3, 3).

    Mirrors ``piano.training.anchor_consistency_loss._aa_to_rotation_matrix``
    but stays in numpy so the probe stays importable without torch.
    """
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)
    theta_safe = np.maximum(theta, 1e-12)
    k = aa / theta_safe
    K = np.zeros(aa.shape[:-1] + (3, 3), dtype=aa.dtype)
    kx = k[..., 0]; ky = k[..., 1]; kz = k[..., 2]
    K[..., 0, 1] = -kz; K[..., 0, 2] = ky
    K[..., 1, 0] = kz;  K[..., 1, 2] = -kx
    K[..., 2, 0] = -ky; K[..., 2, 1] = kx
    eye = np.broadcast_to(np.eye(3, dtype=aa.dtype), K.shape)
    sin_t = np.sin(theta)[..., None]
    cos_t = np.cos(theta)[..., None]
    return eye + sin_t * K + (1.0 - cos_t) * (K @ K)


def _lift_object_local_to_world_np(
    target_local: np.ndarray,      # (T, P, 3) object-local frame
    obj_pos_world: np.ndarray,     # (T, 3) world translation
    obj_rot_world_aa: np.ndarray,  # (T, 3) axis-angle world rotation
) -> np.ndarray:
    """Map an object-local target into world frame using the same SE(3)
    the trainer uses (see ``lift_object_local_to_world`` in
    ``src/piano/training/anchor_consistency_loss.py``).

    Returns (T, P, 3) in world frame, metres.

    Per Phase 0 review §N1: the dataset emits ``contact_target_xyz`` in
    object-local frame (see ``src/piano/data/dataset.py:820-826`` —
    "closest-surface-point on the mesh in object-local frame"), and the
    trainer rotates it to world before any loss
    (``src/piano/training/train_anchordiff.py:493``). The probe was
    previously computing ``wrist_world − target_local`` directly, which
    is geometrically nonsense and produced biased sc% values.
    """
    R = _aa_to_rotation_matrix_np(obj_rot_world_aa)              # (T, 3, 3)
    # einsum gives (T, P, 3) = R @ target_local along the last axis.
    rotated = np.einsum("tij,tpj->tpi", R, target_local)
    return rotated + obj_pos_world[:, None, :]                   # (T, P, 3)


def _proxy_sustained_contact_cm(
    joints: np.ndarray,                      # (T, 22, 3) world frame, metres
    contact_target_xyz: np.ndarray,          # (T, 2, 3) world frame, metres — must be rotated upstream via _lift_object_local_to_world_np
    contact_state: np.ndarray,               # (T, 5) contact mask [L_hand, R_hand, L_foot, R_foot, pelvis]
    contact_threshold: float = 0.5,
) -> float:
    """Mean per-frame wrist-vs-contact-target distance over hand-contact frames,
    in cm. Returns NaN if there are no contact frames.

    This is the cheapest proxy of the sustained_contact diag's drift metric
    that fits in the per-perturbation loop without re-running the full
    segment detector. Lower = wrist closer to contact target = better contact.

    The caller MUST pass ``contact_target_xyz`` already lifted into world
    frame — the dataset stores it in object-local frame. See
    ``_lift_object_local_to_world_np``.
    """
    T = min(joints.shape[0], contact_target_xyz.shape[0], contact_state.shape[0])
    if T < 1:
        return float("nan")
    # Hand contact columns: 0 = left, 1 = right.
    lh_mask = contact_state[:T, 0] > contact_threshold
    rh_mask = contact_state[:T, 1] > contact_threshold
    vals: list[float] = []
    if lh_mask.any():
        lh_dist_m = np.linalg.norm(
            joints[:T, KEY_JOINT_INDICES["left_wrist"]] - contact_target_xyz[:T, 0],
            axis=-1,
        )
        vals.extend((lh_dist_m[lh_mask] * 100.0).tolist())
    if rh_mask.any():
        rh_dist_m = np.linalg.norm(
            joints[:T, KEY_JOINT_INDICES["right_wrist"]] - contact_target_xyz[:T, 1],
            axis=-1,
        )
        vals.extend((rh_dist_m[rh_mask] * 100.0).tolist())
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def _proxy_gait_velocity_score(
    joints: np.ndarray,                      # (T, 22, 3)
    walking_mask: np.ndarray | None,         # (T,) or None
    fps: float = 20.0,
) -> float:
    """Mean absolute L-R ankle horizontal-speed difference on walking
    frames, in cm/s. Higher = more L↔R alternation = healthier gait.
    Returns NaN if there are no walking frames.

    This captures the same axis as round26_gait_diag's
    transitions_per_second + L_R_height_corr without per-segment detection.
    """
    T = joints.shape[0]
    if T < 2:
        return float("nan")
    la = joints[:, KEY_JOINT_INDICES["left_ankle"], [0, 2]]   # (T, 2) xz, metres
    ra = joints[:, KEY_JOINT_INDICES["right_ankle"], [0, 2]]
    l_speed_mps = np.linalg.norm(np.diff(la, axis=0), axis=-1) * fps   # (T-1,)
    r_speed_mps = np.linalg.norm(np.diff(ra, axis=0), axis=-1) * fps
    diff_mps = np.abs(l_speed_mps - r_speed_mps)  # (T-1,)
    if walking_mask is not None:
        wm = walking_mask[:T - 1].astype(bool)
        if not wm.any():
            return float("nan")
        diff_mps = diff_mps[wm]
    return float(diff_mps.mean() * 100.0)   # cm/s


def _proxy_body_action_motion_energy_cm(
    joints: np.ndarray,                      # (T, 22, 3)
) -> float:
    """RMS per-frame whole-body joint velocity in cm/frame. Captures the
    "amount of body action" — proxy for body_action_diag's delta_err
    when comparing pred-vs-pred. Higher = more total motion.
    """
    T = joints.shape[0]
    if T < 2:
        return float("nan")
    vel_m = np.diff(joints, axis=0)         # (T-1, 22, 3)
    speed_m = np.linalg.norm(vel_m, axis=-1)  # (T-1, 22)
    return float(np.sqrt((speed_m ** 2).mean()) * 100.0)


# Per-proxy minimum baseline magnitudes for `_fractional_change`. Baselines
# below these floors are treated as degenerate (proxy meaningless on that
# clip) and return NaN rather than producing exploding fractions.
#
# - sustained_contact (cm): 0.01 cm = 0.1 mm wrist-vs-target distance. A
#   baseline below this is in the noise floor of FK reconstruction and the
#   fractional change is meaningless.
# - gait (cm/s): 0.05 cm/s mean |L-R| ankle speed difference. Below this
#   the clip has effectively no gait signal (stationary or both feet
#   moving identically).
# - body_action (cm/frame): 0.05 cm/frame = 0.5 mm/frame whole-body RMS
#   velocity. A model that produces this little motion is in a degenerate
#   "frozen pose" regime; doubling near-zero noise should not register as
#   a 1000% change. Without this floor, baselines around 1e-6 cm/frame
#   produced fractional changes > 2000 % under time_shuffle (see
#   analyses/2026-05-29_round29_cond_usage_probe_code_review_v2.md, §N2).
_FRACTIONAL_CHANGE_MIN_BASELINE = {
    "sustained_contact_cm": 0.01,
    "gait_cm_per_s": 0.05,
    "body_action_cm_per_frame": 0.05,
}


def _fractional_change(
    base: float, pert: float, *, min_baseline: float = 1e-6,
) -> float:
    """|pert − base| / max(|base|, min_baseline). NaN-safe. Returns NaN if
    either input is NaN or if ``|base| < min_baseline`` (proxy degenerate
    on this clip).

    ``min_baseline`` defaults to 1e-6 for back-compat; callers should pass
    the proxy-specific floor from ``_FRACTIONAL_CHANGE_MIN_BASELINE`` so
    that an effectively-zero baseline does not produce an exploded ratio.
    """
    if not math.isfinite(base) or not math.isfinite(pert):
        return float("nan")
    if abs(base) < min_baseline:
        return float("nan")
    return float(abs(pert - base) / abs(base))


def compute_clip_delta(
    base_joints: np.ndarray, pert_joints: np.ndarray,
    *,
    contact_target_xyz: np.ndarray | None = None,
    contact_state: np.ndarray | None = None,
    walking_mask: np.ndarray | None = None,
    fps: float = 20.0,
) -> tuple[float, float, dict[str, float], float, float, float]:
    """Whole-body delta + per-key-joint delta + 3 task-metric fractional
    changes (sustained_contact / gait / body_action).

    Task-metric deltas are NaN when the metric is undefined for that
    clip (e.g. no contact frames). The labeller is responsible for
    skipping NaN values.
    """
    err = _per_joint_delta_cm(base_joints, pert_joints)   # (T, 22)
    if err.size == 0:
        return (
            float("nan"), float("nan"),
            {n: float("nan") for n in KEY_JOINT_INDICES},
            float("nan"), float("nan"), float("nan"),
        )
    mean_cm = float(err.mean())
    p95_cm = float(np.percentile(err, 95))
    key_joint = {}
    for name, idx in KEY_JOINT_INDICES.items():
        key_joint[name] = float(err[:, idx].mean())

    # Task-metric proxies as fractional change vs baseline.
    sc_rel = float("nan")
    gait_rel = float("nan")
    body_rel = float("nan")
    if contact_target_xyz is not None and contact_state is not None:
        sc_base = _proxy_sustained_contact_cm(
            base_joints, contact_target_xyz, contact_state,
        )
        sc_pert = _proxy_sustained_contact_cm(
            pert_joints, contact_target_xyz, contact_state,
        )
        sc_rel = _fractional_change(
            sc_base, sc_pert,
            min_baseline=_FRACTIONAL_CHANGE_MIN_BASELINE["sustained_contact_cm"],
        )
    gait_base = _proxy_gait_velocity_score(base_joints, walking_mask, fps=fps)
    gait_pert = _proxy_gait_velocity_score(pert_joints, walking_mask, fps=fps)
    gait_rel = _fractional_change(
        gait_base, gait_pert,
        min_baseline=_FRACTIONAL_CHANGE_MIN_BASELINE["gait_cm_per_s"],
    )
    body_base = _proxy_body_action_motion_energy_cm(base_joints)
    body_pert = _proxy_body_action_motion_energy_cm(pert_joints)
    body_rel = _fractional_change(
        body_base, body_pert,
        min_baseline=_FRACTIONAL_CHANGE_MIN_BASELINE["body_action_cm_per_frame"],
    )
    return mean_cm, p95_cm, key_joint, sc_rel, gait_rel, body_rel


def _nanmean(vals: list[float]) -> float | None:
    """Like np.nanmean but returns None instead of NaN on all-NaN input."""
    finite = [float(v) for v in vals if math.isfinite(float(v))]
    if not finite:
        return None
    return float(np.mean(finite))


def aggregate_per_family(
    rows: list[PerClipPerturbationResult],
) -> dict[str, Any]:
    """Aggregate (family, perturbation) over clips. NaN values per row are
    dropped before averaging — so a metric undefined for a clip does not
    poison the mean.
    """
    out: dict[str, dict[str, Any]] = {}
    by_fam: dict[str, list[PerClipPerturbationResult]] = {}
    for r in rows:
        by_fam.setdefault(r.family, []).append(r)
    for family, frows in by_fam.items():
        by_pert: dict[str, list[PerClipPerturbationResult]] = {}
        for r in frows:
            by_pert.setdefault(r.perturbation, []).append(r)
        out[family] = {}
        for pert, prows in by_pert.items():
            means = [r.pred_delta_joints_cm_mean for r in prows]
            p95s = [r.pred_delta_joints_cm_p95 for r in prows]
            key_means: dict[str, list[float]] = {n: [] for n in KEY_JOINT_INDICES}
            for r in prows:
                for n, v in r.key_joint_delta_cm.items():
                    if math.isfinite(v):
                        key_means[n].append(v)
            sc_vals = [r.sustained_contact_delta_rel for r in prows]
            gait_vals = [r.gait_delta_rel for r in prows]
            body_vals = [r.body_action_delta_rel for r in prows]
            out[family][pert] = {
                "n_clips": len(prows),
                "pred_delta_joints_cm_mean": _nanmean(means),
                "pred_delta_joints_cm_p95": _nanmean(p95s),
                "key_joint_delta_cm_mean": {
                    n: (float(np.mean(vs)) if vs else None)
                    for n, vs in key_means.items()
                },
                # R4: per-perturbation aggregate of task-metric fractional changes.
                "sustained_contact_delta_rel_mean": _nanmean(sc_vals),
                "gait_delta_rel_mean": _nanmean(gait_vals),
                "body_action_delta_rel_mean": _nanmean(body_vals),
                "task_metric_n_clips": {
                    "sustained_contact": sum(
                        1 for v in sc_vals if math.isfinite(v)
                    ),
                    "gait": sum(1 for v in gait_vals if math.isfinite(v)),
                    "body_action": sum(
                        1 for v in body_vals if math.isfinite(v)
                    ),
                },
            }
    return out


def _bucket_label_from_key(zero_key_joint_max: float) -> str:
    if zero_key_joint_max < THRESH_IGNORED_KEY_CM:
        return "ignored"
    if zero_key_joint_max < THRESH_WEAK_KEY_CM:
        return "weakly_used"
    return "actively_used"


def _bucket_label_from_relative(rel_max: float) -> str:
    """Codex §3.3 relative-metric arm:
        ignored      <5%
        weakly_used  5-15%
        actively_used >15%
    """
    if rel_max < THRESH_IGNORED_RELATIVE:
        return "ignored"
    if rel_max < THRESH_WEAK_RELATIVE:
        return "weakly_used"
    return "actively_used"


def _strongest_label(*labels: str) -> str:
    """OR-combine bucket labels: actively_used wins over weakly_used wins
    over ignored. NaN entries pass through as 'unknown'."""
    order = {"unknown": -1, "ignored": 0, "weakly_used": 1, "actively_used": 2}
    best = "unknown"
    for lab in labels:
        if order.get(lab, -1) > order.get(best, -1):
            best = lab
    return best


def label_family_usage(
    family_agg: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Apply Codex §3.3 thresholds, FULL OR judge: key-joint magnitude OR
    relative target-metric change. ``family_agg`` is one family's
    perturbation dict from ``aggregate_per_family``.

    Returns a dict containing:
      label             : ignored / weakly_used / actively_used / unknown
      temporally_used   : True if time_shuffle delta > 1.20 × zero delta
      scale_linearity_ratio : (delta(scale_2.0) - delta(scale_0.5)) / delta(zero),
                              indicating whether the family responds
                              linearly (≈ 1.5) or saturates (≈ 0) to
                              magnitude changes — distinguishing PB1
                              AdaLN gating (works for linear) from PB2
                              cross-attention (needed when saturated).
      reasons           : list of human-readable reasons (every gate gets one)
    """
    reasons: list[str] = []
    zero = family_agg.get("zero", {})
    zero_mean = zero.get("pred_delta_joints_cm_mean")
    zero_key_joint_max = None
    kj = zero.get("key_joint_delta_cm_mean") or {}
    finite_kj = [v for v in kj.values() if v is not None and math.isfinite(v)]
    if finite_kj:
        zero_key_joint_max = max(finite_kj)

    if zero_mean is None or zero_key_joint_max is None:
        return {
            "label": "unknown",
            "temporally_used": False,
            "scale_linearity_ratio": None,
            "reasons": ["zero perturbation result missing"],
        }

    # Arm 1: key-joint magnitude bucket.
    key_label = _bucket_label_from_key(zero_key_joint_max)
    reasons.append(
        f"zero key-joint max = {zero_key_joint_max:.2f} cm → key-arm `{key_label}` "
        f"(thresholds {THRESH_IGNORED_KEY_CM:.2f} / {THRESH_WEAK_KEY_CM:.2f} cm)"
    )

    # Arm 2: relative target-metric bucket. Take the MAX over the 3
    # task-metric proxies — any one of them moving ≥ threshold is enough
    # to count as "the family changed model behaviour on a task axis."
    rel_label: str | None = None
    finite_rels: dict[str, float] = {}
    for key in (
        "sustained_contact_delta_rel_mean",
        "gait_delta_rel_mean",
        "body_action_delta_rel_mean",
    ):
        v = zero.get(key)
        if v is not None and math.isfinite(float(v)):
            finite_rels[key] = float(v)
    if finite_rels:
        rel_max_name, rel_max = max(finite_rels.items(), key=lambda kv: kv[1])
        rel_label = _bucket_label_from_relative(rel_max)
        reasons.append(
            f"zero {rel_max_name} = {rel_max * 100:.1f}% → relative-arm "
            f"`{rel_label}` (thresholds {THRESH_IGNORED_RELATIVE * 100:.0f}% / "
            f"{THRESH_WEAK_RELATIVE * 100:.0f}%)"
        )
    else:
        reasons.append(
            "no finite task-metric proxy available on the zero "
            "perturbation; relative-arm SKIPPED"
        )

    # OR judge: take the strongest of the two bucket labels.
    if rel_label is None:
        label = key_label
    else:
        label = _strongest_label(key_label, rel_label)
    reasons.append(
        f"OR judge → label = `{label}` "
        f"(key-arm `{key_label}`"
        + (f", relative-arm `{rel_label}`" if rel_label else "")
        + ")"
    )

    # Temporally-used flag from time_shuffle vs zero.
    time_shuffle = family_agg.get("time_shuffle", {})
    time_shuffle_mean = time_shuffle.get("pred_delta_joints_cm_mean")
    temporally_used = False
    if (time_shuffle_mean is not None and zero_mean is not None
            and zero_mean > 1e-6):
        ratio = float(time_shuffle_mean) / float(zero_mean)
        if ratio > THRESH_TEMPORALLY_USED_FRACTION:
            temporally_used = True
            reasons.append(
                f"time_shuffle / zero = {ratio:.2f} > "
                f"{THRESH_TEMPORALLY_USED_FRACTION:.2f} → temporally used"
            )
        else:
            reasons.append(
                f"time_shuffle / zero = {ratio:.2f}; not temporally used"
            )

    # R3 scale linearity ratio: (delta(scale_2.0) − delta(scale_0.5)) / delta(zero).
    #   ~1.5: family responds proportionally to magnitude → AdaLN gating likely sufficient (PB1).
    #   ~0  : family saturates / non-monotone → structured cross-attention more likely needed (PB2).
    scale_lin_ratio: float | None = None
    scale_lo = family_agg.get("scale_0.5", {})
    scale_hi = family_agg.get("scale_2.0", {})
    d_lo = scale_lo.get("pred_delta_joints_cm_mean")
    d_hi = scale_hi.get("pred_delta_joints_cm_mean")
    d_zero = zero_mean
    if (d_lo is not None and d_hi is not None
            and d_zero is not None and abs(d_zero) > 1e-6):
        scale_lin_ratio = float((d_hi - d_lo) / d_zero)
        if abs(scale_lin_ratio - 1.5) < 0.5:
            reasons.append(
                f"scale_linearity_ratio = {scale_lin_ratio:.2f} "
                f"(≈ 1.5 → linear response; AdaLN-friendly)"
            )
        elif abs(scale_lin_ratio) < 0.3:
            reasons.append(
                f"scale_linearity_ratio = {scale_lin_ratio:.2f} "
                f"(≈ 0 → saturated; cross-attention likely needed)"
            )
        else:
            reasons.append(
                f"scale_linearity_ratio = {scale_lin_ratio:.2f} "
                f"(neither linear nor saturated; mixed)"
            )

    return {
        "label": label,
        "key_arm_label": key_label,
        "relative_arm_label": rel_label,
        "temporally_used": temporally_used,
        "scale_linearity_ratio": scale_lin_ratio,
        "reasons": reasons,
    }


# --------------------------------------------------------------------------- #
# Diagnostic main (uses torch + dataset).
# --------------------------------------------------------------------------- #


PERTURBATIONS_DEFAULT: tuple[str, ...] = (
    "baseline", "zero", "time_shuffle", "batch_shuffle", "scale_0.5", "scale_2.0",
)


def _parse_perturbations(s: str) -> list[str]:
    out = [p.strip() for p in s.split(",") if p.strip()]
    for p in out:
        if p not in PERTURBATIONS_DEFAULT:
            raise ValueError(
                f"unknown perturbation {p!r}; must be one of "
                f"{PERTURBATIONS_DEFAULT}"
            )
    return out


def _parse_families(s: str | None, active: list[str]) -> list[str]:
    """Returns the families to probe. If ``s`` is None, probe every
    active family. Otherwise comma-separated list, validated against
    active list."""
    if s is None:
        return active
    out = [f.strip() for f in s.split(",") if f.strip()]
    for f in out:
        if f not in FAMILY_OF_KEY.values():
            raise ValueError(f"unknown family {f!r}")
        if f not in active:
            raise ValueError(
                f"family {f!r} not active in this config; active families "
                f"are {active}"
            )
    return out


def _apply_perturbation(
    cond_b1: dict, key: str, pert: str, valid_T: int,
    rng: np.random.RandomState, cond_b2: dict | None = None,
) -> dict:
    """Return a copy of cond_b1 with the given family key perturbed.

    For batch_shuffle, also requires cond_b2 (a different clip's cond
    bundle); the family tensor on cond_b1 is replaced with cond_b2's.
    Other keys are unchanged.
    """
    import torch
    out = dict(cond_b1)
    if key not in cond_b1:
        return out   # family not present — nothing to perturb
    t = cond_b1[key]                                  # (1, T, D)
    if pert == "baseline":
        return out
    if pert == "zero":
        out[key] = torch.zeros_like(t)
        return out
    if pert == "scale_0.5":
        out[key] = t * 0.5
        return out
    if pert == "scale_2.0":
        out[key] = t * 2.0
        return out
    if pert == "time_shuffle":
        np_t = t.detach().cpu().numpy()
        perturbed_np = perturbation_time_shuffle(np_t, valid_T, rng)
        out[key] = torch.from_numpy(perturbed_np).to(t.device).to(t.dtype)
        return out
    if pert == "batch_shuffle":
        if cond_b2 is None or key not in cond_b2:
            raise ValueError(
                "batch_shuffle requires a second clip's cond bundle "
                "(cond_b2) with the same family key"
            )
        out[key] = cond_b2[key].clone()
        return out
    raise AssertionError(f"unreachable perturbation {pert!r}")


def _active_families_from_cfg(cfg) -> list[str]:
    """Return the family names with dim > 0 according to the model config."""
    den = cfg.model.denoiser
    active: list[str] = []
    if int(den.get("r29_coarse_extra_dim", 0)) > 0:
        active.append("coarse_extra")
    if int(den.get("r29_interaction_dim", 0)) > 0:
        active.append("interaction")
    if int(den.get("r29_support_dim", 0)) > 0:
        active.append("support")
    if int(den.get("r29_body_refine_dim", 0)) > 0:
        active.append("body_refine")
    return active


def _write_summary_md(
    out_path: Path,
    variant_id: str,
    bucket: str,
    ckpt: str,
    perturbations: list[str],
    aggregate: dict[str, dict[str, Any]],
    family_labels: dict[str, dict[str, Any]],
    n_clips: int,
) -> None:
    L: list[str] = []
    a = L.append

    def _fmt(x, prec=2):
        if x is None:
            return "—"
        try:
            f = float(x)
        except (TypeError, ValueError):
            return str(x)
        if not math.isfinite(f):
            return "—"
        return f"{f:.{prec}f}"

    def _fmt_pct(x):
        if x is None:
            return "—"
        try:
            f = float(x)
        except (TypeError, ValueError):
            return str(x)
        if not math.isfinite(f):
            return "—"
        return f"{100.0 * f:.1f}%"

    a(f"# Round-29 cond-usage probe — `{variant_id}` ({bucket})")
    a("")
    a(f"**Ckpt:** `{ckpt}`")
    a(f"**Clips:** {n_clips}")
    a(f"**Perturbations:** {', '.join(perturbations)}")
    a("")
    a("## Triage labels (per family)")
    a("")
    a("Per Codex §3.3, label = `OR(key-joint arm, relative-target-metric arm)`. "
      "Each arm independently buckets the family into ignored / weakly_used / "
      "actively_used; the stronger of the two wins. `temporally_used` is a "
      "separate flag from `time_shuffle / zero > "
      f"{THRESH_TEMPORALLY_USED_FRACTION:.2f}`. `scale_linearity_ratio` "
      "(`(Δ_scale_2.0 − Δ_scale_0.5) / Δ_zero`) ≈ 1.5 ⇒ linear response "
      "(AdaLN-friendly, PB1 candidate); ≈ 0 ⇒ saturated (likely needs PB2 "
      "cross-attention).")
    a("")
    a("| family | label | key-arm | rel-arm | temporally_used? | scale_linearity | reasons |")
    a("| --- | --- | --- | --- | :---: | ---: | --- |")
    for family, lab in family_labels.items():
        reasons_str = "; ".join(lab.get("reasons", [])) or "—"
        tu = "✓" if lab.get("temporally_used") else "—"
        rel_arm = lab.get("relative_arm_label") or "—"
        slr = lab.get("scale_linearity_ratio")
        slr_str = _fmt(slr, 2) if slr is not None else "—"
        a(
            f"| `{family}` | **{lab.get('label', 'unknown')}** | "
            f"`{lab.get('key_arm_label', '—')}` | `{rel_arm}` | "
            f"{tu} | {slr_str} | {reasons_str} |"
        )
    a("")
    a("## Per-family per-perturbation aggregate")
    a("")
    a("Each row: per-clip mean across clips. `whole_body_mean (cm)` is the "
      "mean Euclidean delta over **all 22 SMPL joints × valid frames** vs "
      "the baseline sample (same random seed). Key-joint columns "
      "(LW / RW / LA / RA / Neck / Pelvis) are the mean across clips of "
      "per-clip per-key-joint mean. The triage label's key-arm uses the "
      "**max** of the 6 per-key-joint means, NOT `whole_body_mean`. "
      "`sc%` / `gait%` / `body%` are mean fractional change of the three "
      "task-metric proxies vs baseline; the relative-arm uses the max "
      "of those three.")
    a("")
    for family, perts in aggregate.items():
        a(f"### `{family}`")
        a("")
        a("| perturbation | n_clips | whole_body_mean (cm) | p95 (cm) | LW | RW | LA | RA | Neck | Pelvis | sc% | gait% | body% |")
        a("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for pert in perturbations:
            row = perts.get(pert)
            if row is None:
                a(f"| {pert} | — | — | — | — | — | — | — | — | — | — | — | — |")
                continue
            kj = row.get("key_joint_delta_cm_mean") or {}
            a(
                f"| {pert} | {row.get('n_clips', 0)} | "
                f"{_fmt(row.get('pred_delta_joints_cm_mean'))} | "
                f"{_fmt(row.get('pred_delta_joints_cm_p95'))} | "
                f"{_fmt(kj.get('left_wrist'))} | "
                f"{_fmt(kj.get('right_wrist'))} | "
                f"{_fmt(kj.get('left_ankle'))} | "
                f"{_fmt(kj.get('right_ankle'))} | "
                f"{_fmt(kj.get('neck'))} | "
                f"{_fmt(kj.get('pelvis'))} | "
                f"{_fmt_pct(row.get('sustained_contact_delta_rel_mean'))} | "
                f"{_fmt_pct(row.get('gait_delta_rel_mean'))} | "
                f"{_fmt_pct(row.get('body_action_delta_rel_mean'))} |"
            )
        a("")
    a("## How to read")
    a("")
    a("### Per-family label (Codex §3.3 OR judge)")
    a("")
    a(f"- `key-arm` thresholds on the **max** of 6 per-key-joint mean deltas "
      f"after `zero`: < {THRESH_IGNORED_KEY_CM:.2f} cm → ignored, "
      f"[{THRESH_IGNORED_KEY_CM:.2f}, {THRESH_WEAK_KEY_CM:.2f}] → weakly_used, "
      f"≥ {THRESH_WEAK_KEY_CM:.2f} → actively_used.")
    a(f"- `rel-arm` thresholds on the **max** of the 3 task-metric "
      f"fractional changes after `zero`: < {THRESH_IGNORED_RELATIVE*100:.0f}% "
      f"→ ignored, [{THRESH_IGNORED_RELATIVE*100:.0f}%, {THRESH_WEAK_RELATIVE*100:.0f}%] "
      f"→ weakly_used, ≥ {THRESH_WEAK_RELATIVE*100:.0f}% → actively_used.")
    a("- The final label is the **stronger** of the two arms. A family can "
      "qualify as `actively_used` by either large key-joint movement OR "
      "large task-metric movement.")
    a("")
    a("### Independent flags")
    a("")
    a(f"- `temporally_used`: whether `time_shuffle / zero > "
      f"{THRESH_TEMPORALLY_USED_FRACTION:.2f}`. If True, the model uses the "
      "family's **temporal structure**, not just its average magnitude. "
      "Implication: a pooled AdaLN summary (PB1) would lose this signal; "
      "cross-attn (PB2) preserves it.")
    a("- `scale_linearity_ratio`: "
      "`(Δ(scale_2.0) − Δ(scale_0.5)) / Δ(zero)` from whole-body mean. "
      "Roughly 1.5 ⇒ proportional response ⇒ AdaLN gating sufficient. "
      "Roughly 0 ⇒ saturation ⇒ structured cross-attention more likely "
      "needed. Negative ⇒ non-monotone (rare; suspicious).")
    a("")
    a("### Task-metric proxies (sc% / gait% / body%)")
    a("")
    a("These are cheap whole-clip proxies of the diag scripts' full per-"
      "segment metrics, kept inside the per-perturbation loop to avoid "
      "re-running segment detectors:")
    a("- `sc%` (sustained_contact_delta_rel): mean per-frame wrist-vs-"
      "contact_target distance on hand-contact frames. Fractional change "
      "vs baseline; NaN when no contact frames.")
    a("- `gait%` (gait_delta_rel): mean |L − R| ankle XZ-speed on walking "
      "frames. Fractional change vs baseline; NaN when no walking frames.")
    a("- `body%` (body_action_delta_rel): RMS whole-body per-frame joint "
      "velocity. Fractional change vs baseline.")
    a("")
    a("### Batch_shuffle pairing")
    a("")
    a("`batch_shuffle` uses a derangement-guaranteed random pairing of the "
      "clip-index list (see `_derange_indices`), seeded from the run "
      "`--seed`. This avoids the failure mode where adjacent clips in the "
      "selection JSON share subset / object and therefore have similar "
      "cond bundles, which would suppress the perturbation signal. The "
      "concrete pairing used is recorded in `batch_shuffle_pairing` in the "
      "JSON output for reproducibility.")
    a("")
    out_path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Round-29 cond-usage probe — measures whether each active R29 "
            "condition family is consumed by a trained ckpt via zero / "
            "time_shuffle / batch_shuffle / scale perturbations."
        ),
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bucket", default="val", choices=["train", "val"])
    parser.add_argument("--variant-id", required=True,
                        help="Label written into the JSON / MD output.")
    parser.add_argument("--families", default=None,
                        help="Comma-separated family names to probe; default "
                             "= all active families in the config.")
    parser.add_argument(
        "--perturbations", default=",".join(PERTURBATIONS_DEFAULT),
        help=f"Comma-separated; valid: {','.join(PERTURBATIONS_DEFAULT)}",
    )
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Heavy imports deferred so pure helpers above remain importable in tests.
    import torch
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from piano.data.dataset import collate_hoi
    from piano.inference.diagnostic_helpers import (
        _build_cond, _build_dataset, _build_model, _fk_22joints,
        _stage1_norm_for_cfg, extract_train_time_meta,
    )
    from piano.utils.clip_utils import load_clip_text_encoder

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    perturbations = _parse_perturbations(args.perturbations)
    active_families = _active_families_from_cfg(cfg)
    if not active_families:
        raise SystemExit(
            f"no active R29 families in config {args.config} — nothing to "
            f"probe. Check r29_coarse_extra_dim / r29_interaction_dim / "
            f"r29_support_dim / r29_body_refine_dim."
        )
    families = _parse_families(args.families, active_families)
    print(f"[cond_probe] variant={args.variant_id} bucket={args.bucket}")
    print(f"[cond_probe] active families: {active_families}")
    print(f"[cond_probe] probing families: {families}")
    print(f"[cond_probe] perturbations: {perturbations}")

    # Selection JSON.
    sel_obj = json.loads(args.selection_json.read_text("utf-8"))
    selection = (
        sel_obj.get("selected") or sel_obj.get("candidates")
        or sel_obj.get("clips") or []
    )
    if not selection:
        raise SystemExit(f"empty selection: {args.selection_json}")
    sel_pairs = {(e["subset"], e["seq_id"]) for e in selection}
    print(f"[cond_probe] selection: {len(sel_pairs)} clips")

    dataset = _build_dataset(cfg, args.bucket, augment=False)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        collate_fn=collate_hoi, num_workers=0,
    )

    model, object_encoder = _build_model(cfg, device)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    train_meta = extract_train_time_meta(state)
    model_state = state.get("model", state)
    model.load_state_dict(model_state)
    if "object_encoder" in state:
        object_encoder.load_state_dict(state["object_encoder"])
    elif "extra_modules" in state and "object_encoder" in state["extra_modules"]:
        object_encoder.load_state_dict(state["extra_modules"]["object_encoder"])
    else:
        # Fail-closed (per code-review R5): a missing object_encoder state
        # would silently leave object features at their random init, making
        # every sample meaningless while the script reports success.
        raise SystemExit(
            f"FATAL: ckpt {args.ckpt} has no object_encoder state. The probe "
            "refuses to run a randomly-initialised object encoder, which "
            "would produce junk samples that look superficially valid. "
            "Confirm the ckpt was saved by the current trainer's "
            "_save_checkpoint path."
        )
    if int(cfg.model.denoiser.get("text_dim", 0)) > 0:
        clip_model = load_clip_text_encoder(
            device=device,
            model_name=str(cfg.model.text_encoder.clip_version),
            download_root=str(cfg.model.text_encoder.get(
                "download_root", "cache/clip")),
        )
    else:
        clip_model = None
    stage1_norm = _stage1_norm_for_cfg(cfg, device)
    model.eval()

    # Pre-collect ALL selected batches (single-clip) and their cond dicts
    # so batch_shuffle can pair them later.
    selected_batches: list[tuple[str, str, dict, dict, int]] = []
    for batch in loader:
        subset = str(batch["subset"][0]); seq_id = str(batch["seq_id"][0])
        if (subset, seq_id) not in sel_pairs:
            continue
        with torch.no_grad():
            cond, T = _build_cond(
                batch, model, object_encoder, clip_model, cfg, device,
                stage1_norm=stage1_norm,
            )
        seq_len = int(batch["seq_len"][0].item())
        valid_T = min(T, seq_len)
        selected_batches.append((subset, seq_id, batch, cond, valid_T))
        if len(selected_batches) % 8 == 0:
            print(f"  [cond_probe] precomputed {len(selected_batches)} clips")
    if not selected_batches:
        raise SystemExit("no clips matched selection in dataset")
    print(f"[cond_probe] precomputed {len(selected_batches)} cond bundles")

    # Helper to sample with seed reset.
    def _sample(cond_in: dict, T: int) -> torch.Tensor:
        torch.manual_seed(args.seed)
        with torch.no_grad():
            return model.sample(
                cond=cond_in, seq_length=T, cfg_scale=args.cfg_scale,
            )

    rng = np.random.RandomState(args.seed)
    rows: list[PerClipPerturbationResult] = []

    # R2 fix: batch_shuffle pairing is randomised + derangement-guaranteed
    # so we don't always pair clip i with its structurally-adjacent
    # neighbour (which in the 12/12/12/12 balanced selection JSON shares
    # subset and often object, suppressing the perturbation signal).
    # The full-derangement helper used here mirrors
    # ``perturbation_batch_shuffle`` semantics on the clip-index level so
    # the unit-tested invariants (no row maps to itself, every row is a
    # valid original row) carry over.
    n_clips = len(selected_batches)
    pair_idx = _derange_indices(n_clips, rng)

    fps = float(cfg.data.get("fps", 20.0))   # used by gait proxy

    for i, (subset, seq_id, batch, cond, valid_T) in enumerate(selected_batches):
        T = cond["object_world_traj"].shape[1]
        rest_offsets = batch["rest_offsets"].to(device).float()

        # Extract per-clip task-metric inputs ONCE (avoid CPU↔GPU shuttle
        # on every perturbation). All shapes truncated to valid_T.
        #
        # Per Phase 0 review §N1: contact_target_xyz is emitted by the
        # dataset in OBJECT-LOCAL frame (see
        # ``src/piano/data/dataset.py:820-826``). The trainer rotates it
        # to world via ``lift_object_local_to_world`` before any loss.
        # We MUST do the same here, otherwise the sc% proxy mixes
        # world-frame wrist with object-local target and produces biased
        # distances. Requires ``object_positions`` (B,T,3) and
        # ``object_rotations`` (B,T,3 axis-angle) in the batch, same as
        # ``diagnostic_helpers._build_cond``.
        contact_target_xyz_np: np.ndarray | None = None
        contact_state_np: np.ndarray | None = None
        walking_mask_np: np.ndarray | None = None
        if "contact_target_xyz" in batch:
            target_local = batch[
                "contact_target_xyz"
            ][0, :valid_T].cpu().numpy()                # (T, P, 3)
            if (
                "object_positions" not in batch
                or "object_rotations" not in batch
            ):
                raise KeyError(
                    "round29_cond_usage_probe: batch has "
                    "'contact_target_xyz' but is missing "
                    "'object_positions' / 'object_rotations' — cannot "
                    "lift target to world frame. Update the dataset to "
                    "surface both."
                )
            obj_pos_world_np = batch[
                "object_positions"
            ][0, :valid_T].cpu().numpy()                # (T, 3)
            obj_rot_world_aa_np = batch[
                "object_rotations"
            ][0, :valid_T].cpu().numpy()                # (T, 3) axis-angle
            contact_target_xyz_np = _lift_object_local_to_world_np(
                target_local, obj_pos_world_np, obj_rot_world_aa_np,
            )
        if "contact_state" in batch:
            contact_state_np = batch[
                "contact_state"
            ][0, :valid_T].cpu().numpy()
        if "walking_mask" in batch:
            wm = batch["walking_mask"][0, :valid_T].cpu().numpy()
            # walking_mask may be (T, 1) or (T,) depending on dataset.
            walking_mask_np = wm.squeeze(-1) if wm.ndim == 2 else wm

        # Baseline sample for this clip — anchored by seed.
        pred_motion_base = _sample(cond, T)
        base_joints = _fk_22joints(
            pred_motion_base, rest_offsets,
        )[0, :valid_T].cpu().numpy()

        for family in families:
            cond_key = KEY_OF_FAMILY[family]
            if cond_key not in cond:
                # Family active in config but the dataset didn't emit it
                # (shouldn't happen on R29 configs — sanity-skip).
                continue
            for pert in perturbations:
                if pert == "baseline":
                    # zero delta against itself
                    rows.append(PerClipPerturbationResult(
                        variant_id=args.variant_id, bucket=args.bucket,
                        family=family, perturbation=pert,
                        subset=subset, seq_id=seq_id,
                        pred_delta_joints_cm_mean=0.0,
                        pred_delta_joints_cm_p95=0.0,
                        key_joint_delta_cm={n: 0.0 for n in KEY_JOINT_INDICES},
                        sustained_contact_delta_rel=0.0,
                        gait_delta_rel=0.0,
                        body_action_delta_rel=0.0,
                    ))
                    continue
                cond_b2 = (
                    selected_batches[pair_idx[i]][3]
                    if pert == "batch_shuffle" else None
                )
                cond_pert = _apply_perturbation(
                    cond, cond_key, pert, valid_T, rng, cond_b2=cond_b2,
                )
                pred_motion_pert = _sample(cond_pert, T)
                pert_joints = _fk_22joints(
                    pred_motion_pert, rest_offsets,
                )[0, :valid_T].cpu().numpy()
                mean_cm, p95_cm, kj, sc_rel, gait_rel, body_rel = (
                    compute_clip_delta(
                        base_joints, pert_joints,
                        contact_target_xyz=contact_target_xyz_np,
                        contact_state=contact_state_np,
                        walking_mask=walking_mask_np,
                        fps=fps,
                    )
                )
                rows.append(PerClipPerturbationResult(
                    variant_id=args.variant_id, bucket=args.bucket,
                    family=family, perturbation=pert,
                    subset=subset, seq_id=seq_id,
                    pred_delta_joints_cm_mean=mean_cm,
                    pred_delta_joints_cm_p95=p95_cm,
                    key_joint_delta_cm=kj,
                    sustained_contact_delta_rel=sc_rel,
                    gait_delta_rel=gait_rel,
                    body_action_delta_rel=body_rel,
                ))

        if (i + 1) % 4 == 0:
            print(
                f"  [cond_probe] {i + 1}/{n_clips} clips, "
                f"{len(rows)} rows so far"
            )

    print(f"[cond_probe] collected {len(rows)} rows total")

    aggregate = aggregate_per_family(rows)
    family_labels = {f: label_family_usage(aggregate[f]) for f in aggregate}

    out = {
        "variant_id": args.variant_id,
        "bucket": args.bucket,
        "config": str(args.config),
        "ckpt": str(args.ckpt),
        "selection_json": str(args.selection_json),
        "cfg_scale": args.cfg_scale,
        "seed": args.seed,
        "n_clips": n_clips,
        "active_families": active_families,
        "probed_families": families,
        "perturbations": perturbations,
        "train_time": train_meta,
        "thresholds": {
            "ignored_key_cm": THRESH_IGNORED_KEY_CM,
            "weak_key_cm": THRESH_WEAK_KEY_CM,
            "ignored_relative": THRESH_IGNORED_RELATIVE,
            "weak_relative": THRESH_WEAK_RELATIVE,
            "temporally_used_fraction": THRESH_TEMPORALLY_USED_FRACTION,
        },
        "batch_shuffle_pairing": pair_idx,
        "aggregate": aggregate,
        "family_labels": family_labels,
        "rows": [
            {
                "variant_id": r.variant_id, "bucket": r.bucket,
                "family": r.family, "perturbation": r.perturbation,
                "subset": r.subset, "seq_id": r.seq_id,
                "pred_delta_joints_cm_mean": r.pred_delta_joints_cm_mean,
                "pred_delta_joints_cm_p95": r.pred_delta_joints_cm_p95,
                "key_joint_delta_cm": r.key_joint_delta_cm,
                # R4 task-metric deltas (fractional change vs baseline).
                "sustained_contact_delta_rel": r.sustained_contact_delta_rel,
                "gait_delta_rel": r.gait_delta_rel,
                "body_action_delta_rel": r.body_action_delta_rel,
            }
            for r in rows
        ],
    }
    out_json = args.output_dir / "cond_usage_stats.json"
    out_md = args.output_dir / "cond_usage_summary.md"
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    _write_summary_md(
        out_md, args.variant_id, args.bucket, str(args.ckpt),
        perturbations, aggregate, family_labels, n_clips,
    )
    print(f"[cond_probe] wrote {out_json}")
    print(f"[cond_probe] wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
