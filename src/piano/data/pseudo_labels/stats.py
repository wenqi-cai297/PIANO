"""Per-sequence + aggregate stats for pseudo-label quality evaluation.

Turns raw extracted labels into numeric summaries that can be written to
``summary.json`` by ``run_all.py`` or computed post-hoc from a finished
``pseudo_labels/`` directory.

Design goals:
    - Cheap enough to run inline inside the extraction loop: uses only the
      already-loaded label arrays plus a tiny geometric sanity check on
      joints/object_positions.
    - JSON-able output. No numpy arrays in the returned dicts (everything
      cast to ``float`` / ``int`` / lists of primitives).
    - Produces human-readable ``quality_flags`` strings so review does not
      require eyeballing numeric tables.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from piano.data.pseudo_labels.extract_phase import PHASE_NAMES
from piano.data.pseudo_labels.extract_support import (
    SUPPORT_BOTH_FEET,
    SUPPORT_NAMES,
)
from piano.utils.smpl_utils import (
    BODY_PART_INDICES,
    BODY_PART_NAMES,
    NUM_BODY_PARTS,
)


# Thresholds used by the stats pass. Keep in sync with run-time configs:
# CONTACT_THRESHOLD matches the binarization used everywhere downstream
# (visualize_pseudo_labels, extract_support).
CONTACT_THRESHOLD: float = 0.5

# Mean target entropy below this on a sequence's contact frames means the
# soft assignment has effectively collapsed to hard nearest-patch. With
# K=16 the theoretical max is ln(16) ≈ 2.77; anything under ~0.3 is almost
# a one-hot distribution.
TARGET_DEGENERATE_ENTROPY: float = 0.3

# A sequence whose minimum hand-to-object-center distance exceeds this
# (over the whole sequence) is flagged as a preprocessing outlier — a
# well-behaved InterAct clip should have hands near the object at some
# point.
GEOMETRIC_OUTLIER_DIST_M: float = 2.0


@dataclass(slots=True)
class PerSequenceStats:
    """Numeric stats extracted from one sequence's labels + motion arrays."""

    seq_id: str
    num_frames: int
    # contact
    contact_frames: np.ndarray            # (B,) int  — frames with soft > threshold
    contact_any_frame: np.ndarray         # (B,) bool — this part ever contacted
    # phase
    phase_counts: np.ndarray              # (P,) int
    phase_reached: np.ndarray             # (P,) bool
    phase_transitions: int
    # support
    support_counts: np.ndarray            # (S,) int
    support_entered: np.ndarray           # (S,) bool
    support_only_both_feet: bool
    # target
    target_entropy_per_frame: np.ndarray  # (M,) float — M = # contact frames with nonzero target
    target_argmax_histogram: np.ndarray   # (K,) int   — how often each patch was argmax
    # geometric sanity
    min_hand_to_obj_center_dist_m: float


def _target_entropy(row: np.ndarray) -> float:
    """Shannon entropy (natural log) of a soft distribution row.

    Rows that sum to 0 return 0 (rather than NaN).
    """
    p = row.astype(np.float64)
    s = p.sum()
    if s <= 1e-12:
        return 0.0
    p = p / s
    nz = p > 1e-12
    return float(-(p[nz] * np.log(p[nz])).sum())


def compute_seq_stats(
    seq_id: str,
    labels: dict[str, np.ndarray],
    joints_22: np.ndarray | None = None,
    object_positions: np.ndarray | None = None,
) -> PerSequenceStats:
    """Compute all per-sequence stats from one sequence's labels.

    ``labels`` must contain ``contact_state`` (T,B), ``contact_target``
    (T,B,K), ``phase`` (T,), ``support`` (T,). ``joints_22`` and
    ``object_positions`` are optional — if provided, a coarse
    hand-to-object-center distance sanity check is included.
    """
    contact = labels["contact_state"]       # (T, B) soft
    target = labels["contact_target"]       # (T, B, K) soft
    phase = labels["phase"].astype(np.int64)
    support = labels["support"].astype(np.int64)

    T, B = contact.shape
    K = target.shape[-1]

    # ---- contact ----
    contact_binary = contact > CONTACT_THRESHOLD                       # (T, B)
    contact_frames = contact_binary.sum(axis=0).astype(np.int64)       # (B,)
    contact_any_frame = contact_binary.any(axis=0)                     # (B,)

    # ---- phase ----
    phase_counts = np.bincount(phase, minlength=len(PHASE_NAMES)).astype(np.int64)
    # Clip in case phase has out-of-range values (defensive; shouldn't happen)
    phase_counts = phase_counts[: len(PHASE_NAMES)]
    phase_reached = phase_counts > 0
    phase_transitions = int((np.diff(phase) != 0).sum()) if T > 1 else 0

    # ---- support ----
    support_counts = np.bincount(support, minlength=len(SUPPORT_NAMES)).astype(np.int64)
    support_counts = support_counts[: len(SUPPORT_NAMES)]
    support_entered = support_counts > 0
    support_only_both_feet = bool(
        support_entered.sum() == 1 and support_entered[SUPPORT_BOTH_FEET]
    )

    # ---- target (only on contact frames) ----
    entropies: list[float] = []
    argmax_hist = np.zeros(K, dtype=np.int64)
    contact_tb = np.argwhere(contact_binary)    # (M, 2) — [t, b] pairs
    for t, b in contact_tb:
        row = target[t, b]
        s = float(row.sum())
        if s <= 1e-6:
            # Body-part contacted but target wasn't set — skip (shouldn't
            # happen when contact_threshold is aligned with extractor)
            continue
        entropies.append(_target_entropy(row))
        argmax_hist[int(np.argmax(row))] += 1
    target_entropy_per_frame = np.array(entropies, dtype=np.float64)

    # ---- geometric sanity: min hand-to-object-center distance ----
    min_dist = float("inf")
    if joints_22 is not None and object_positions is not None:
        hand_idx = [BODY_PART_INDICES[0], BODY_PART_INDICES[1]]
        T_min = min(len(joints_22), len(object_positions))
        if T_min > 0:
            hands = joints_22[:T_min, hand_idx, :]                      # (T, 2, 3)
            obj = object_positions[:T_min, None, :]                     # (T, 1, 3)
            d = np.linalg.norm(hands - obj, axis=-1)                    # (T, 2)
            if d.size:
                min_dist = float(d.min())

    return PerSequenceStats(
        seq_id=seq_id,
        num_frames=T,
        contact_frames=contact_frames,
        contact_any_frame=contact_any_frame,
        phase_counts=phase_counts,
        phase_reached=phase_reached,
        phase_transitions=phase_transitions,
        support_counts=support_counts,
        support_entered=support_entered,
        support_only_both_feet=support_only_both_feet,
        target_entropy_per_frame=target_entropy_per_frame,
        target_argmax_histogram=argmax_hist,
        min_hand_to_obj_center_dist_m=min_dist,
    )


def aggregate_stats(
    per_seq: list[PerSequenceStats],
    num_patches: int = 16,
) -> dict[str, Any]:
    """Combine per-sequence stats into a JSON-able summary dict."""
    N = len(per_seq)
    if N == 0:
        return {"num_sequences": 0}

    total_frames = int(sum(s.num_frames for s in per_seq))

    # ---- contact ----
    rates = np.stack([
        s.contact_frames / max(s.num_frames, 1) for s in per_seq
    ])   # (N, B)
    had_contact = np.stack([s.contact_any_frame for s in per_seq])   # (N, B)

    per_body_part: dict[str, dict[str, float]] = {}
    for i, name in enumerate(BODY_PART_NAMES):
        per_body_part[name] = {
            "frame_rate_mean": float(rates[:, i].mean()),
            "frame_rate_std": float(rates[:, i].std()),
            "seq_without_contact_fraction": float(1.0 - had_contact[:, i].mean()),
        }

    # "Active-part" rate: per-seq max across parts, then mean across seqs.
    # Gives a coarse sense of overall activity; robust to one-part dominance.
    any_part_rate_mean = float(rates.max(axis=1).mean())
    zero_contact_seq = int((~had_contact.any(axis=1)).sum())

    contact_stats = {
        "per_body_part": per_body_part,
        "any_part_frame_rate_mean": any_part_rate_mean,
        "zero_contact_seq_count": zero_contact_seq,
        "zero_contact_seq_fraction": float(zero_contact_seq / N),
    }

    # ---- phase ----
    phase_total = np.sum([s.phase_counts for s in per_seq], axis=0).astype(np.int64)
    phase_reached_total = np.sum([s.phase_reached for s in per_seq], axis=0).astype(np.int64)
    transitions = np.array([s.phase_transitions for s in per_seq])
    seq_no_trans = int((transitions == 0).sum())
    phase_stats = {
        "frame_distribution": {
            name: float(phase_total[i] / max(total_frames, 1))
            for i, name in enumerate(PHASE_NAMES)
        },
        "seq_reached_phase_fraction": {
            name: float(phase_reached_total[i] / N)
            for i, name in enumerate(PHASE_NAMES)
        },
        "mean_transitions_per_seq": float(transitions.mean()),
        "median_transitions_per_seq": float(np.median(transitions)),
        "seq_with_zero_transitions_count": seq_no_trans,
        "seq_with_zero_transitions_fraction": float(seq_no_trans / N),
    }

    # ---- support ----
    support_total = np.sum([s.support_counts for s in per_seq], axis=0).astype(np.int64)
    support_entered_total = np.sum([s.support_entered for s in per_seq], axis=0).astype(np.int64)
    stuck = int(sum(1 for s in per_seq if s.support_only_both_feet))
    support_stats = {
        "frame_distribution": {
            name: float(support_total[i] / max(total_frames, 1))
            for i, name in enumerate(SUPPORT_NAMES)
        },
        "seq_entered_support_fraction": {
            name: float(support_entered_total[i] / N)
            for i, name in enumerate(SUPPORT_NAMES)
        },
        "seq_stuck_in_both_feet_count": stuck,
        "seq_stuck_in_both_feet_fraction": float(stuck / N),
    }

    # ---- target ----
    per_seq_mean_entropy = np.array([
        s.target_entropy_per_frame.mean() if s.target_entropy_per_frame.size else np.nan
        for s in per_seq
    ])
    has_contact_frames = ~np.isnan(per_seq_mean_entropy)
    all_entropies = np.concatenate([
        s.target_entropy_per_frame for s in per_seq if s.target_entropy_per_frame.size
    ]) if has_contact_frames.any() else np.array([])

    degenerate_seq = int((per_seq_mean_entropy[has_contact_frames] < TARGET_DEGENERATE_ENTROPY).sum())
    total_contact_seq = int(has_contact_frames.sum())

    patch_hist = np.sum([s.target_argmax_histogram for s in per_seq], axis=0).astype(np.int64)
    target_stats: dict[str, Any] = {
        "num_sequences_with_contact_frames": total_contact_seq,
        "num_contact_frames_with_target": int(all_entropies.size),
        "entropy_max_possible": float(np.log(num_patches)),
        "degenerate_seq_count": degenerate_seq,
        "degenerate_seq_fraction": float(degenerate_seq / max(total_contact_seq, 1)),
        "patch_argmax_histogram": patch_hist.tolist(),
        "patch_coverage_fraction": float((patch_hist > 0).sum() / num_patches),
    }
    if all_entropies.size:
        target_stats.update({
            "entropy_mean": float(all_entropies.mean()),
            "entropy_median": float(np.median(all_entropies)),
            "entropy_std": float(all_entropies.std()),
            "entropy_p10": float(np.percentile(all_entropies, 10)),
            "entropy_p90": float(np.percentile(all_entropies, 90)),
        })
    else:
        target_stats.update({
            "entropy_mean": 0.0,
            "entropy_median": 0.0,
            "entropy_std": 0.0,
            "entropy_p10": 0.0,
            "entropy_p90": 0.0,
        })

    # ---- geometric sanity ----
    dists = np.array([s.min_hand_to_obj_center_dist_m for s in per_seq])
    finite = dists[np.isfinite(dists)]
    if finite.size:
        geometric_sanity = {
            "min_hand_to_obj_center_dist_m": {
                "median": float(np.median(finite)),
                "p90": float(np.percentile(finite, 90)),
                "max": float(finite.max()),
                "outlier_seq_count_gt_2m": int((finite > GEOMETRIC_OUTLIER_DIST_M).sum()),
            },
            "num_sequences_with_object_data": int(finite.size),
        }
    else:
        geometric_sanity = {
            "min_hand_to_obj_center_dist_m": None,
            "num_sequences_with_object_data": 0,
        }

    return {
        "num_sequences": N,
        "total_frames": total_frames,
        "contact_stats": contact_stats,
        "phase_stats": phase_stats,
        "support_stats": support_stats,
        "target_stats": target_stats,
        "geometric_sanity": geometric_sanity,
    }


def make_quality_flags(agg: dict[str, Any], subset_hint: str | None = None) -> list[str]:
    """Human-readable warnings derived from aggregated stats.

    ``subset_hint`` lets flags specialise to a known subset (e.g. chairs
    must have non-trivial ``sitting`` support to be trustworthy).
    """
    flags: list[str] = []
    if agg.get("num_sequences", 0) == 0:
        flags.append("empty aggregate — nothing to evaluate")
        return flags

    label = subset_hint or "subset"

    # Support: chairs-specific + generic "stuck in both_feet" check
    sitting_frac = agg["support_stats"]["frame_distribution"].get("sitting", 0.0)
    stuck_frac = agg["support_stats"]["seq_stuck_in_both_feet_fraction"]
    if subset_hint == "chairs" and sitting_frac < 0.05:
        flags.append(
            f"[{label}] `sitting` support frame fraction = {sitting_frac * 100:.1f}% "
            "— support label likely broken for chairs (expected >> 30%)"
        )
    if stuck_frac > 0.5:
        flags.append(
            f"[{label}] {stuck_frac * 100:.1f}% of sequences never leave `both_feet` — "
            "support extractor may be defaulting too aggressively"
        )

    # Phase: manipulation reach + zero-transition degenerate
    manip_reach = agg["phase_stats"]["seq_reached_phase_fraction"].get("manipulation", 0.0)
    if manip_reach < 0.15:
        flags.append(
            f"[{label}] `manipulation` phase reached in only {manip_reach * 100:.1f}% of sequences "
            "— phase velocity threshold may be too strict (or contact velocity is wrong frame)"
        )
    zero_trans_frac = agg["phase_stats"]["seq_with_zero_transitions_fraction"]
    if zero_trans_frac > 0.2:
        flags.append(
            f"[{label}] {zero_trans_frac * 100:.1f}% of sequences have zero phase transitions "
            "— phase may be saturated to one class"
        )

    # Contact: per-part absence + overall zero-contact
    for name, stats in agg["contact_stats"]["per_body_part"].items():
        nc = stats["seq_without_contact_fraction"]
        if nc > 0.95:
            flags.append(
                f"[{label}] {name} never contacts in {nc * 100:.1f}% of sequences — "
                "thresholds or joint tracking may be wrong"
            )
    zcs = agg["contact_stats"]["zero_contact_seq_fraction"]
    if zcs > 0.05:
        flags.append(
            f"[{label}] {zcs * 100:.1f}% of sequences have zero contact anywhere — suspicious"
        )

    # Target: degenerate soft-assign
    deg = agg["target_stats"]["degenerate_seq_fraction"]
    if deg > 0.5:
        max_ent = agg["target_stats"]["entropy_max_possible"]
        flags.append(
            f"[{label}] {deg * 100:.1f}% of sequences have near-hard target "
            f"(mean entropy < {TARGET_DEGENERATE_ENTROPY:.2f} / max {max_ent:.2f}) "
            "— soft-assign kernel may be too sharp"
        )

    # Geometric outliers
    geo = agg["geometric_sanity"].get("min_hand_to_obj_center_dist_m")
    if geo and geo.get("outlier_seq_count_gt_2m", 0) > 0:
        flags.append(
            f"[{label}] {geo['outlier_seq_count_gt_2m']} sequences with min "
            f"hand-to-obj-center distance > {GEOMETRIC_OUTLIER_DIST_M} m — "
            "possible preprocessing bug or unrelated-object sequence"
        )

    return flags