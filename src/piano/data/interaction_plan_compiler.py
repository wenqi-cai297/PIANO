"""Interaction Plan Compiler — converts dense per-frame ``z_int`` evidence
into a sparse semantic interaction program (anchors + segments).

Design source of truth:
    analyses/piano_interaction_plan_pipeline_reframe_for_claude_code.md

Why this module exists
----------------------
Stage A predicts dense per-frame ``z_int`` (contact_state, contact_target,
phase, support). Feeding ``z_int(t)`` directly into Stage B produced two
documented failure modes (see analyses/2026-05-09_stageB_design_journey.md
and 2026-05-10_v9_4_hard_observation_report.md):

1. Dense per-frame conditioning collapses Stage B to "rest pose + root
   translation along the trajectory".
2. Concatenating sparse keyframes as a side-channel (CondMDI-style) does
   pin observed frames but the model does not propagate that information
   to unobserved frames (sensitivity test: identical to 4 dp across GT,
   zeros, shuffled, wrong-clip cond_motion).

The compiler is the missing layer between the two: it converts dense
evidence into a *sparse, semantically typed* program of anchors (event
constraints) + segments (sustained intervals). Stage B is then asked to
realize this program rather than reconstruct dense per-frame conditions.

The same compiler is used at training (compile from GT pseudo-labels) and
at inference (compile from Stage A predictions). This keeps the train-test
representation consistent.

Outputs are pure numpy arrays in a fixed-shape padded format suitable for
PyTorch dataloader collation. See ``InteractionPlan`` for the schema and
``collate_interaction_plans`` for the batch-stacking convention.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.ndimage import uniform_filter1d


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Anchor type IDs. The encoder embeds these with a small categorical
# embedding; downstream metrics group by type. Keep stable across releases.
ANCHOR_TYPE_ONSET: int = 0          # contact begins
ANCHOR_TYPE_STABLE: int = 1         # mid-segment, most stable frame
ANCHOR_TYPE_RELEASE: int = 2        # contact ends
ANCHOR_TYPE_PHASE_CHANGE: int = 3   # phase transition (e.g. approach→contact)
ANCHOR_TYPE_SUPPORT_CHANGE: int = 4 # support transition (e.g. none→object)
NUM_ANCHOR_TYPES: int = 5

# Body-part identities follow piano.utils.smpl_utils.INTERACTION_BODY_PARTS:
#   0 = left_hand   (SMPL-22 joint 20)
#   1 = right_hand  (SMPL-22 joint 21)
#   2 = left_foot   (SMPL-22 joint 10)
#   3 = right_foot  (SMPL-22 joint 11)
#   4 = pelvis      (SMPL-22 joint 0)
NUM_PARTS_DEFAULT: int = 5

# Body-part priority weights for anchor budgeting. Hands first because
# contact targets there are the highest-information for downstream
# motion synthesis (HOI literature consensus). Pelvis matters for
# sit/lie poses; feet/head only contribute when explicitly active.
PART_PRIORITY_DEFAULT: tuple[float, ...] = (1.0, 1.0, 0.6, 0.6, 0.8)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class InteractionPlanCompilerConfig:
    """Hyperparameters for the compiler.

    Defaults are tuned for 20 fps motion (PIANO's preprocessing rate). At
    other rates scale the temporal windows / durations proportionally.

    Audit any change with the audit script (see
    ``analyses/2026-05-10_interaction_plan_compiler_audit.md``); the
    rejection counts and per-clip anchor histogram are sensitive to the
    contact thresholds and minimum duration.
    """

    # Number of body parts in contact_state. Set to 5 for the default
    # PIANO INTERACTION_BODY_PARTS layout.
    num_parts: int = NUM_PARTS_DEFAULT
    num_phase_classes: int = 3
    num_support_classes: int = 3

    # Smoothing windows (frames). 5 frames at 20 fps = 0.25 s — enough
    # to suppress single-frame flickers, not enough to wash out real
    # transitions.
    contact_smooth_window: int = 5
    target_smooth_window: int = 5
    phase_smooth_window: int = 5
    support_smooth_window: int = 5

    # Hysteresis thresholds for binarising the smoothed contact prob.
    # Asymmetric (enter > exit) prevents oscillation around a single
    # threshold.
    contact_enter_threshold: float = 0.55
    contact_exit_threshold: float = 0.35
    min_contact_duration: int = 4
    gap_merge: int = 3

    # Temporal merging / NMS of generated anchors.
    merge_window: int = 4
    temporal_nms_window: int = 5

    # Anchor budget. K_MIN guarantees we don't return an empty plan
    # for clips with weak evidence — fillers are added uniformly.
    # K_MAX caps memory in the dataloader (padded tensors).
    k_min: int = 3
    k_max: int = 12
    s_max: int = 12

    # Stable-frame scoring weights (within a contact segment).
    stable_target_velocity_weight: float = 1.0
    phase_entropy_weight: float = 0.2
    support_entropy_weight: float = 0.2

    # Anchor-priority scoring weights (used for K_MAX budgeting).
    score_contact_conf_weight: float = 1.0
    score_duration_weight: float = 0.3
    score_stability_weight: float = 0.5
    score_change_weight: float = 0.5

    # Per-part priority multiplier for anchor scoring. Length must
    # equal ``num_parts``. See ``PART_PRIORITY_DEFAULT`` for the
    # default heuristic (hands > pelvis > feet > head).
    part_priority: tuple[float, ...] = PART_PRIORITY_DEFAULT

    def __post_init__(self) -> None:
        if len(self.part_priority) != self.num_parts:
            raise ValueError(
                f"part_priority length {len(self.part_priority)} != "
                f"num_parts {self.num_parts}"
            )
        if not (0 < self.contact_exit_threshold < self.contact_enter_threshold < 1):
            raise ValueError(
                "thresholds must satisfy 0 < exit < enter < 1, got "
                f"enter={self.contact_enter_threshold}, "
                f"exit={self.contact_exit_threshold}"
            )
        if self.k_min < 1 or self.k_min > self.k_max:
            raise ValueError(f"k_min={self.k_min} k_max={self.k_max}")


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------
#
# All tensors are zero-padded to a fixed shape so the dataloader can
# stack them across the batch. ``anchor_mask`` / ``segment_mask`` mark
# which slots are valid.

InteractionPlan = dict[str, np.ndarray]


def _empty_plan(cfg: InteractionPlanCompilerConfig) -> InteractionPlan:
    K, S, P = cfg.k_max, cfg.s_max, cfg.num_parts
    return {
        "anchor_time": np.zeros(K, dtype=np.int64),
        "anchor_part": np.zeros((K, P), dtype=np.float32),
        "anchor_target_local": np.zeros((K, P, 3), dtype=np.float32),
        "anchor_target_world": np.zeros((K, P, 3), dtype=np.float32),
        "anchor_type": np.zeros(K, dtype=np.int64),
        "anchor_phase": np.zeros(K, dtype=np.int64),
        "anchor_support": np.zeros(K, dtype=np.int64),
        "anchor_conf": np.zeros(K, dtype=np.float32),
        "anchor_mask": np.zeros(K, dtype=bool),
        "segment_start": np.zeros(S, dtype=np.int64),
        "segment_end": np.zeros(S, dtype=np.int64),
        "segment_part": np.zeros((S, P), dtype=np.float32),
        "segment_target_summary_local": np.zeros((S, P, 3), dtype=np.float32),
        "segment_phase": np.zeros(S, dtype=np.int64),
        "segment_support": np.zeros(S, dtype=np.int64),
        "segment_conf": np.zeros(S, dtype=np.float32),
        "segment_mask": np.zeros(S, dtype=bool),
    }


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------


def smooth_contact(
    contact_prob: np.ndarray,           # (T, P) float in [0, 1]
    window: int,
) -> np.ndarray:
    """Temporal moving-average smoother for per-part contact probability.

    A short window (default 5 frames at 20 fps) suppresses single-frame
    flickers without washing out real transitions. ``mode='nearest'``
    avoids edge artefacts at clip start / end where finger contact often
    matters most.
    """
    if window <= 1:
        return contact_prob.astype(np.float32, copy=True)
    return uniform_filter1d(
        contact_prob.astype(np.float32, copy=False),
        size=int(window),
        axis=0,
        mode="nearest",
    )


def smooth_target_local(
    target_local: np.ndarray,           # (T, P, 3)
    contact_smooth: np.ndarray,         # (T, P) — used as a soft mask
    window: int,
) -> np.ndarray:
    """Smooth per-part target_local with a contact-weighted moving average.

    Frames where the part is not contacting carry no meaningful target —
    averaging in those frames would pull the smoothed target toward the
    noisy "off-contact" target_xyz output. We weight by contact_smooth so
    that off-contact frames contribute near-zero to the average.
    """
    if window <= 1:
        return target_local.astype(np.float32, copy=True)
    T, P, _ = target_local.shape
    weights = contact_smooth[..., None].astype(np.float32)  # (T, P, 1)
    weighted = target_local.astype(np.float32) * weights
    num = uniform_filter1d(weighted, size=window, axis=0, mode="nearest")
    den = uniform_filter1d(weights, size=window, axis=0, mode="nearest")
    return num / np.clip(den, 1e-6, None)


def smooth_categorical_softmax(
    logits_or_softmax: np.ndarray,       # (T, C)
    window: int,
) -> np.ndarray:
    """Smooth a per-frame softmax distribution with a moving average.

    Re-normalises after the average so each row still sums to 1 (the
    moving average preserves the sum-to-1 property up to floating error
    but we re-normalise defensively).
    """
    if window <= 1:
        return logits_or_softmax.astype(np.float32, copy=True)
    sm = uniform_filter1d(
        logits_or_softmax.astype(np.float32, copy=False),
        size=int(window),
        axis=0,
        mode="nearest",
    )
    sm = sm / np.clip(sm.sum(axis=-1, keepdims=True), 1e-6, None)
    return sm


# ---------------------------------------------------------------------------
# Hysteresis segmentation
# ---------------------------------------------------------------------------


def hysteresis_segments(
    prob: np.ndarray,                    # (T,) — single body part
    enter: float,
    exit: float,
    min_duration: int,
    gap_merge: int,
) -> list[tuple[int, int]]:
    """Hysteresis thresholding → list of [start, end] inclusive intervals.

    Once ``prob`` rises above ``enter`` we are in-contact until it falls
    below ``exit`` (asymmetric thresholds prevent flicker around a single
    boundary). Segments shorter than ``min_duration`` are discarded.
    Segments separated by a gap < ``gap_merge`` are merged.
    """
    T = len(prob)
    in_seg = False
    raw: list[list[int]] = []
    cur_start = 0
    for t in range(T):
        if not in_seg:
            if prob[t] >= enter:
                in_seg = True
                cur_start = t
        else:
            if prob[t] < exit:
                raw.append([cur_start, t - 1])
                in_seg = False
    if in_seg:
        raw.append([cur_start, T - 1])

    # Drop too-short
    pruned = [(s, e) for s, e in raw if (e - s + 1) >= min_duration]

    # Merge close segments
    if not pruned:
        return []
    merged: list[list[int]] = [list(pruned[0])]
    for s, e in pruned[1:]:
        if s - merged[-1][1] - 1 <= gap_merge:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


# ---------------------------------------------------------------------------
# Anchor candidates
# ---------------------------------------------------------------------------


def _entropy(p: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-row entropy of a softmax distribution."""
    return -np.sum(p * np.log(np.clip(p, eps, None)), axis=-1)


def _select_stable_frame(
    s: int,
    e: int,
    p: int,
    contact_smooth: np.ndarray,         # (T, P)
    target_smooth: np.ndarray,          # (T, P, 3)
    phase_entropy: np.ndarray,          # (T,)
    support_entropy: np.ndarray,        # (T,)
    cfg: InteractionPlanCompilerConfig,
) -> tuple[int, float]:
    """Return (stable_frame_idx, stability_score) maximising the score:

        contact_conf - λ_v * target_velocity - λ_p * phase_H - λ_s * support_H

    Argmax in [s, e].
    """
    span = slice(s, e + 1)
    conf = contact_smooth[span, p]                                 # (L,)
    tgt = target_smooth[span, p, :]                                # (L, 3)
    if len(tgt) > 1:
        vel = np.linalg.norm(np.diff(tgt, axis=0), axis=-1)         # (L-1,)
        vel = np.concatenate([vel[:1], vel])                        # pad to len L
    else:
        vel = np.zeros(1, dtype=np.float32)
    score = (
        conf
        - cfg.stable_target_velocity_weight * vel
        - cfg.phase_entropy_weight * phase_entropy[span]
        - cfg.support_entropy_weight * support_entropy[span]
    )
    rel = int(np.argmax(score))
    return s + rel, float(score[rel])


def _find_categorical_change_frames(
    soft: np.ndarray,                   # (T, C) softmax
) -> list[tuple[int, int, int]]:
    """Detect frames where the argmax category transitions.

    Returns a list of ``(t, from_label, to_label)`` tuples for each
    transition. The frame index is the FIRST frame of the new label.
    """
    if len(soft) < 2:
        return []
    labels = np.argmax(soft, axis=-1)
    diffs = np.where(np.diff(labels) != 0)[0]
    return [(int(t + 1), int(labels[t]), int(labels[t + 1])) for t in diffs]


# ---------------------------------------------------------------------------
# Anchor & segment construction
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _CandidateAnchor:
    """Internal representation of an anchor candidate before NMS / budget.

    Multi-hot ``parts`` allows merging anchors that fire on multiple body
    parts at the same time (e.g. both hands grasping a chair).
    """
    time: int
    parts: np.ndarray                   # (P,) multi-hot float
    target_local: np.ndarray            # (P, 3)
    type_id: int
    phase: int
    support: int
    confidence: float
    duration: int
    stability: float = 0.0
    is_change: bool = False


def _make_part_onehot(p: int, num_parts: int) -> np.ndarray:
    arr = np.zeros(num_parts, dtype=np.float32)
    arr[p] = 1.0
    return arr


def _build_contact_anchors(
    contact_smooth: np.ndarray,
    target_smooth: np.ndarray,
    phase_smooth: np.ndarray,
    support_smooth: np.ndarray,
    phase_entropy: np.ndarray,
    support_entropy: np.ndarray,
    cfg: InteractionPlanCompilerConfig,
) -> tuple[list[_CandidateAnchor], list[dict]]:
    """Per-part contact-segment scan → onset/stable/release anchor candidates.

    Also returns the list of contact-segment dicts (used to build segment
    tokens later).
    """
    candidates: list[_CandidateAnchor] = []
    segments_out: list[dict] = []

    for p in range(cfg.num_parts):
        intervals = hysteresis_segments(
            contact_smooth[:, p],
            enter=cfg.contact_enter_threshold,
            exit=cfg.contact_exit_threshold,
            min_duration=cfg.min_contact_duration,
            gap_merge=cfg.gap_merge,
        )
        for s, e in intervals:
            duration = e - s + 1
            mean_conf = float(contact_smooth[s : e + 1, p].mean())
            stable_t, stable_score = _select_stable_frame(
                s, e, p, contact_smooth, target_smooth,
                phase_entropy, support_entropy, cfg,
            )

            # Segment record (for segment-token branch)
            seg_target = target_smooth[s : e + 1, p, :]             # (L, 3)
            target_summary = seg_target.mean(axis=0)                # (3,)
            phase_majority = int(
                np.bincount(np.argmax(phase_smooth[s : e + 1], axis=-1)).argmax()
            )
            support_majority = int(
                np.bincount(np.argmax(support_smooth[s : e + 1], axis=-1)).argmax()
            )
            segments_out.append({
                "start": s,
                "end": e,
                "part": p,
                "target_summary_local": target_summary.astype(np.float32),
                "phase": phase_majority,
                "support": support_majority,
                "conf": mean_conf,
                "duration": duration,
            })

            # Onset / stable / release anchors for this segment.
            for time_idx, type_id, conf in (
                (s, ANCHOR_TYPE_ONSET, float(contact_smooth[s, p])),
                (stable_t, ANCHOR_TYPE_STABLE, mean_conf),
                (e, ANCHOR_TYPE_RELEASE, float(contact_smooth[e, p])),
            ):
                # contact_target at the anchor frame (object-local coords).
                tloc = np.zeros((cfg.num_parts, 3), dtype=np.float32)
                tloc[p] = target_smooth[time_idx, p].astype(np.float32)
                candidates.append(_CandidateAnchor(
                    time=time_idx,
                    parts=_make_part_onehot(p, cfg.num_parts),
                    target_local=tloc,
                    type_id=type_id,
                    phase=int(np.argmax(phase_smooth[time_idx])),
                    support=int(np.argmax(support_smooth[time_idx])),
                    confidence=conf,
                    duration=duration,
                    stability=stable_score if type_id == ANCHOR_TYPE_STABLE else 0.0,
                ))

    return candidates, segments_out


def _build_change_anchors(
    phase_smooth: np.ndarray,
    support_smooth: np.ndarray,
    cfg: InteractionPlanCompilerConfig,
) -> list[_CandidateAnchor]:
    """Phase / support transition → CHANGE anchor candidates.

    These anchors carry no contact target (zero target_local). The model
    sees them as "something semantic just changed at this frame".
    """
    candidates: list[_CandidateAnchor] = []
    for t, _from, to in _find_categorical_change_frames(phase_smooth):
        candidates.append(_CandidateAnchor(
            time=t,
            parts=np.zeros(cfg.num_parts, dtype=np.float32),
            target_local=np.zeros((cfg.num_parts, 3), dtype=np.float32),
            type_id=ANCHOR_TYPE_PHASE_CHANGE,
            phase=int(to),
            support=int(np.argmax(support_smooth[t])),
            confidence=float(phase_smooth[t, to]),
            duration=1,
            is_change=True,
        ))
    for t, _from, to in _find_categorical_change_frames(support_smooth):
        candidates.append(_CandidateAnchor(
            time=t,
            parts=np.zeros(cfg.num_parts, dtype=np.float32),
            target_local=np.zeros((cfg.num_parts, 3), dtype=np.float32),
            type_id=ANCHOR_TYPE_SUPPORT_CHANGE,
            phase=int(np.argmax(phase_smooth[t])),
            support=int(to),
            confidence=float(support_smooth[t, to]),
            duration=1,
            is_change=True,
        ))
    return candidates


# ---------------------------------------------------------------------------
# Merging & budgeting
# ---------------------------------------------------------------------------


def merge_nearby_anchors(
    candidates: list[_CandidateAnchor],
    cfg: InteractionPlanCompilerConfig,
) -> list[_CandidateAnchor]:
    """Merge anchors whose time indices are within ``merge_window``.

    Two cases:
      1. Same-type (e.g. two onset anchors firing within 4 frames on
         different body parts) — merge into a multi-part anchor.
      2. Cross-type (e.g. contact onset coincides with phase change) —
         keep the higher-priority type as the merged type, but union
         the parts/targets.

    Priority for type merging: STABLE > ONSET > RELEASE > PHASE_CHANGE >
    SUPPORT_CHANGE. STABLE is highest because it carries the most
    information (mid-segment, most stable frame).
    """
    if not candidates:
        return []
    type_priority = {
        ANCHOR_TYPE_STABLE: 4,
        ANCHOR_TYPE_ONSET: 3,
        ANCHOR_TYPE_RELEASE: 2,
        ANCHOR_TYPE_PHASE_CHANGE: 1,
        ANCHOR_TYPE_SUPPORT_CHANGE: 0,
    }
    sorted_c = sorted(candidates, key=lambda c: c.time)
    merged: list[_CandidateAnchor] = []
    cluster: list[_CandidateAnchor] = [sorted_c[0]]
    for c in sorted_c[1:]:
        if c.time - cluster[-1].time <= cfg.merge_window:
            cluster.append(c)
        else:
            merged.append(_merge_cluster(cluster, type_priority))
            cluster = [c]
    merged.append(_merge_cluster(cluster, type_priority))
    return merged


def _merge_cluster(
    cluster: list[_CandidateAnchor],
    type_priority: dict[int, int],
) -> _CandidateAnchor:
    """Confidence-weighted merge of a contiguous cluster of candidates."""
    if len(cluster) == 1:
        return cluster[0]
    confs = np.array([c.confidence for c in cluster], dtype=np.float32)
    w = confs / max(confs.sum(), 1e-6)
    time = int(round(float(np.dot(w, [c.time for c in cluster]))))

    # Union of parts (multi-hot OR), confidence-weighted target_local
    parts = np.zeros_like(cluster[0].parts)
    target_local = np.zeros_like(cluster[0].target_local)
    for c, wi in zip(cluster, w):
        parts = np.maximum(parts, c.parts)
        target_local = target_local + wi * c.target_local

    # Highest-priority type wins
    type_id = max((c.type_id for c in cluster), key=lambda tid: type_priority[tid])
    # Phase / support: pick from highest-priority candidate
    primary = max(cluster, key=lambda c: type_priority[c.type_id])
    return _CandidateAnchor(
        time=time,
        parts=parts,
        target_local=target_local.astype(np.float32),
        type_id=type_id,
        phase=primary.phase,
        support=primary.support,
        confidence=float(np.max(confs)),
        duration=int(max(c.duration for c in cluster)),
        stability=float(max(c.stability for c in cluster)),
        is_change=any(c.is_change for c in cluster),
    )


def _score_for_budget(
    cand: _CandidateAnchor, cfg: InteractionPlanCompilerConfig,
) -> float:
    """Compute the priority score for K_MAX budgeting.

    A higher score = more likely to survive the K_MAX cut.
    """
    part_priority_arr = np.asarray(cfg.part_priority, dtype=np.float32)
    part_score = float(np.dot(cand.parts, part_priority_arr))
    return (
        cfg.score_contact_conf_weight * cand.confidence
        + cfg.score_duration_weight * np.log1p(cand.duration)
        + cfg.score_stability_weight * cand.stability
        + cfg.score_change_weight * (1.0 if cand.is_change else 0.0)
        + part_score
    )


def temporal_nms_budget(
    candidates: list[_CandidateAnchor],
    cfg: InteractionPlanCompilerConfig,
) -> list[_CandidateAnchor]:
    """Greedy temporal NMS down to ``k_max`` anchors.

    Sort by score descending, then walk the list keeping each anchor if
    no higher-scoring anchor has already been kept within
    ``temporal_nms_window`` frames.

    If we end up with fewer than ``k_min`` anchors, pad with uniformly-
    spaced filler frames from the unselected candidates (or the time
    grid if none remain) so downstream Stage B always has a non-empty
    plan to attend to.
    """
    if not candidates:
        return []
    scored = sorted(
        candidates,
        key=lambda c: _score_for_budget(c, cfg),
        reverse=True,
    )
    kept: list[_CandidateAnchor] = []
    suppressed: list[_CandidateAnchor] = []
    for c in scored:
        if len(kept) >= cfg.k_max:
            suppressed.append(c)
            continue
        if any(abs(k.time - c.time) < cfg.temporal_nms_window for k in kept):
            suppressed.append(c)
            continue
        kept.append(c)

    # Sort kept anchors by time for downstream consumption
    kept.sort(key=lambda c: c.time)
    return kept


# ---------------------------------------------------------------------------
# Object-local → world lifting (numpy)
# ---------------------------------------------------------------------------


def _axis_angle_to_rotmat_np(aa: np.ndarray) -> np.ndarray:
    """Rodrigues' formula in numpy. ``aa`` shape (..., 3) → (..., 3, 3)."""
    angle = np.linalg.norm(aa, axis=-1, keepdims=True)
    safe = np.where(angle < 1e-8, 1.0, angle)
    axis = aa / safe
    s = np.sin(angle).squeeze(-1)
    c = np.cos(angle).squeeze(-1)
    one_c = 1.0 - c
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    R = np.stack([
        np.stack([c + x * x * one_c,        x * y * one_c - z * s,  x * z * one_c + y * s], axis=-1),
        np.stack([y * x * one_c + z * s,    c + y * y * one_c,      y * z * one_c - x * s], axis=-1),
        np.stack([z * x * one_c - y * s,    z * y * one_c + x * s,  c + z * z * one_c    ], axis=-1),
    ], axis=-2)
    # When angle≈0 the formula is identity (since one_c≈0, s≈0, c≈1) —
    # the np.where above only protects the divide by ``angle``.
    return R


def lift_target_local_to_world_np(
    target_local: np.ndarray,           # (T, P, 3) object-local
    object_pos_world: np.ndarray,       # (T, 3)
    object_rot_world_aa: np.ndarray,    # (T, 3) axis-angle
) -> np.ndarray:
    """Numpy port of ``anchor_consistency_loss.lift_object_local_to_world``.

    The compiler runs offline / inside the dataloader where torch isn't
    on the GPU, so we keep the numpy version local. Convention matches
    the trainer's lifting (same Rodrigues + R @ p + t)."""
    R = _axis_angle_to_rotmat_np(object_rot_world_aa)               # (T, 3, 3)
    rotated = np.einsum("tij,tpj->tpi", R, target_local)             # (T, P, 3)
    return rotated + object_pos_world[:, None, :]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compile_interaction_plan(
    contact_prob: np.ndarray,           # (T, P) [0,1]
    target_local: np.ndarray,           # (T, P, 3) object-local
    phase_softmax: np.ndarray,          # (T, num_phase)
    support_softmax: np.ndarray,        # (T, num_support)
    object_pos_world: np.ndarray,       # (T, 3)
    object_rot_world_aa: np.ndarray,    # (T, 3) axis-angle
    seq_len: int,
    cfg: InteractionPlanCompilerConfig | None = None,
) -> InteractionPlan:
    """Compile dense per-frame z_int evidence into a sparse interaction plan.

    The same function is used at training time (with GT pseudo-labels) and
    at inference time (with Stage A predicted z_int). The caller is
    responsible for shaping the inputs into the canonical formats above
    before calling.

    Inputs MUST be valid frames only — pass ``arr[:seq_len]`` if the upstream
    has already padded. Padding is handled internally on the way out.

    Returns an ``InteractionPlan`` dict with all arrays padded to fixed
    shape (``cfg.k_max`` / ``cfg.s_max``) and zero-filled in invalid slots.
    Use ``anchor_mask`` / ``segment_mask`` to identify valid entries.
    """
    cfg = cfg or InteractionPlanCompilerConfig()
    plan = _empty_plan(cfg)
    if seq_len < 2:
        return plan
    T = min(seq_len, contact_prob.shape[0])

    # 1. Smooth dense evidence
    contact_smooth = smooth_contact(contact_prob[:T], cfg.contact_smooth_window)
    phase_smooth = smooth_categorical_softmax(phase_softmax[:T], cfg.phase_smooth_window)
    support_smooth = smooth_categorical_softmax(support_softmax[:T], cfg.support_smooth_window)
    target_smooth = smooth_target_local(target_local[:T], contact_smooth, cfg.target_smooth_window)
    phase_entropy = _entropy(phase_smooth)
    support_entropy = _entropy(support_smooth)

    # 2. Build candidates (contact + change)
    contact_cands, segments = _build_contact_anchors(
        contact_smooth, target_smooth, phase_smooth, support_smooth,
        phase_entropy, support_entropy, cfg,
    )
    change_cands = _build_change_anchors(phase_smooth, support_smooth, cfg)
    all_cands = contact_cands + change_cands

    # 3. Merge nearby
    merged = merge_nearby_anchors(all_cands, cfg)

    # 4. Temporal NMS + budget to k_max
    kept = temporal_nms_budget(merged, cfg)

    # 5. K_MIN guarantee — uniform fillers if we have too few anchors.
    # Fillers are STABLE-type with no contact part / target so the
    # encoder can recognise them as "context placeholders" via the type
    # embedding.
    if len(kept) < cfg.k_min:
        existing_times = {c.time for c in kept}
        n_needed = cfg.k_min - len(kept)
        filler_times = np.linspace(0, T - 1, n_needed + 2, dtype=int)[1:-1]
        for ft in filler_times:
            if int(ft) not in existing_times:
                kept.append(_CandidateAnchor(
                    time=int(ft),
                    parts=np.zeros(cfg.num_parts, dtype=np.float32),
                    target_local=np.zeros((cfg.num_parts, 3), dtype=np.float32),
                    type_id=ANCHOR_TYPE_STABLE,
                    phase=int(np.argmax(phase_smooth[int(ft)])),
                    support=int(np.argmax(support_smooth[int(ft)])),
                    confidence=0.0,
                    duration=1,
                ))
        kept.sort(key=lambda c: c.time)

    # 6. Lift target_local to world for the entire smoothed clip ONCE,
    #    then index by anchor time. Cheaper than per-anchor lift + numerically
    #    identical to the trainer's (R @ p + t) convention.
    target_world_full = lift_target_local_to_world_np(
        target_smooth, object_pos_world[:T], object_rot_world_aa[:T],
    )                                                                # (T, P, 3)

    # 7. Pack anchors into the padded output dict.
    K_MAX, S_MAX, P = cfg.k_max, cfg.s_max, cfg.num_parts
    n_anch = min(len(kept), K_MAX)
    for i, c in enumerate(kept[:n_anch]):
        plan["anchor_time"][i] = int(c.time)
        plan["anchor_part"][i] = c.parts
        plan["anchor_target_local"][i] = c.target_local
        # World target only valid where parts are active. For inactive
        # parts we leave zeros (encoder gates by parts mask anyway).
        plan["anchor_target_world"][i] = c.parts[:, None] * target_world_full[c.time]
        plan["anchor_type"][i] = int(c.type_id)
        plan["anchor_phase"][i] = int(c.phase)
        plan["anchor_support"][i] = int(c.support)
        plan["anchor_conf"][i] = float(c.confidence)
        plan["anchor_mask"][i] = True

    # 8. Pack segments. Multi-part segments are encoded as separate slots
    #    (one segment per (part, interval) pair); merging would lose
    #    per-part target identity.
    n_seg = min(len(segments), S_MAX)
    # Sort by confidence, then start
    segments.sort(key=lambda s: (-s["conf"], s["start"]))
    for i, seg in enumerate(segments[:n_seg]):
        p = seg["part"]
        plan["segment_start"][i] = int(seg["start"])
        plan["segment_end"][i] = int(seg["end"])
        plan["segment_part"][i, p] = 1.0
        plan["segment_target_summary_local"][i, p] = seg["target_summary_local"]
        plan["segment_phase"][i] = int(seg["phase"])
        plan["segment_support"][i] = int(seg["support"])
        plan["segment_conf"][i] = float(seg["conf"])
        plan["segment_mask"][i] = True

    return plan


# ---------------------------------------------------------------------------
# Stats / audit helpers
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CompilerStats:
    """Per-clip and aggregate statistics returned by the audit script."""
    n_clips: int = 0
    n_zero_anchor_clips: int = 0
    n_zero_segment_clips: int = 0
    anchor_count_histogram: list[int] = field(default_factory=lambda: [0] * 13)
    segment_count_histogram: list[int] = field(default_factory=lambda: [0] * 13)
    anchor_type_counts: list[int] = field(default_factory=lambda: [0] * NUM_ANCHOR_TYPES)
    anchor_part_counts: list[int] = field(default_factory=lambda: [0] * NUM_PARTS_DEFAULT)
    anchor_time_normalized_sum: float = 0.0    # sum of (time / seq_len)
    anchor_time_normalized_n: int = 0


def update_stats(stats: CompilerStats, plan: InteractionPlan, seq_len: int) -> None:
    """Accumulate per-clip stats into the aggregate ``CompilerStats``."""
    n_a = int(plan["anchor_mask"].sum())
    n_s = int(plan["segment_mask"].sum())
    stats.n_clips += 1
    if n_a == 0:
        stats.n_zero_anchor_clips += 1
    if n_s == 0:
        stats.n_zero_segment_clips += 1
    if n_a < len(stats.anchor_count_histogram):
        stats.anchor_count_histogram[n_a] += 1
    else:
        stats.anchor_count_histogram[-1] += 1
    if n_s < len(stats.segment_count_histogram):
        stats.segment_count_histogram[n_s] += 1
    else:
        stats.segment_count_histogram[-1] += 1
    for i in range(n_a):
        stats.anchor_type_counts[int(plan["anchor_type"][i])] += 1
        for p in range(plan["anchor_part"].shape[1]):
            if plan["anchor_part"][i, p] > 0:
                stats.anchor_part_counts[p] += 1
        if seq_len > 0:
            stats.anchor_time_normalized_sum += float(plan["anchor_time"][i]) / float(seq_len)
            stats.anchor_time_normalized_n += 1


# ---------------------------------------------------------------------------
# Collation for PyTorch dataloader
# ---------------------------------------------------------------------------


def collate_interaction_plans(
    plans: Sequence[InteractionPlan],
) -> dict[str, np.ndarray]:
    """Stack a list of per-clip plans into batched arrays.

    All entries share the same ``k_max`` / ``s_max`` (compiler-config
    constraint) so a simple stack works. Returned dict has the same keys
    as a single plan, with a leading batch dim.
    """
    if not plans:
        raise ValueError("collate_interaction_plans needs ≥ 1 plan")
    keys = list(plans[0].keys())
    return {k: np.stack([p[k] for p in plans], axis=0) for k in keys}
