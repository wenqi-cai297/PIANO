"""Shared utilities for no-training Stage-B diagnostics.

This module intentionally lives next to the diagnostic entry scripts.  It
does not change the model/training path; it only centralizes loading,
batch merging, event extraction, and metric aggregation used by the offline
diagnostics added on 2026-05-14.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from dynamics_diagnostic import (
    LOCAL_JOINTS,
    PART_JOINT,
    _balanced_subset_indices,
    _build_dataset,
    _fft_band_energy,
    _joint_velocity_acceleration,
    _per_joint_vel_stats,
    _stats_from_magnitudes,
)
from piano.data.dataset import collate_hoi


PLAN_KEYS = [
    "anchor_time",
    "anchor_part",
    "anchor_target_local",
    "anchor_target_world",
    "anchor_type",
    "anchor_phase",
    "anchor_support",
    "anchor_conf",
    "anchor_mask",
    "segment_start",
    "segment_end",
    "segment_part",
    "segment_target_summary_local",
    "segment_phase",
    "segment_support",
    "segment_conf",
    "segment_mask",
]

HAND_SPECS = (
    ("L_hand", 20, 0),
    ("R_hand", 21, 1),
)


def safe_div(num: float, den: float) -> float:
    return float(num / den) if abs(float(den)) > 1e-12 else 0.0


def mean_numeric(rows: Iterable[dict[str, Any]]) -> dict[str, float]:
    rows = list(rows)
    keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float)) and np.isfinite(float(value))
        }
    )
    return {
        key: float(np.mean([float(row[key]) for row in rows if key in row]))
        for key in keys
    }


def stats_list(values: Iterable[float]) -> dict[str, float | int]:
    arr = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0, "p95": 0.0, "std": 0.0, "n": 0}
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "std": float(arr.std()),
        "n": int(arr.size),
    }


def format_md_table(rows: list[list[Any]]) -> str:
    if not rows:
        return ""
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]
    out: list[str] = []
    for r, row in enumerate(rows):
        out.append("| " + " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(row)) + " |")
        if r == 0:
            out.append("| " + " | ".join("-" * widths[i] for i in range(len(widths))) + " |")
    return "\n".join(out)


def extract_plan(batch: dict[str, Any], device: torch.device) -> dict[str, Tensor]:
    return {key: batch[f"plan_{key}"].to(device) for key in PLAN_KEYS}


def load_checkpoint(model: torch.nn.Module, object_encoder: torch.nn.Module, ckpt: Path) -> dict[str, Any]:
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("model", state))
    if "object_encoder" in state:
        object_encoder.load_state_dict(state["object_encoder"])
    elif "extra_modules" in state and "object_encoder" in state["extra_modules"]:
        object_encoder.load_state_dict(state["extra_modules"]["object_encoder"])
    return state


def make_seq_mask(seq_len: Tensor, total_t: int, device: torch.device) -> Tensor:
    t = torch.arange(int(total_t), device=device).view(1, int(total_t))
    return t < seq_len.to(device).long().view(-1, 1)


def merge_single_batches(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("merge_single_batches needs at least one batch")
    out: dict[str, Any] = {}
    keys = items[0].keys()
    for key in keys:
        first = items[0][key]
        if isinstance(first, Tensor):
            out[key] = torch.cat([item[key] for item in items], dim=0)
        elif isinstance(first, list):
            vals: list[Any] = []
            for item in items:
                vals.extend(item[key])
            out[key] = vals
        else:
            out[key] = [item[key] for item in items]
    return out


def selected_balanced_batches(
    cfg,
    *,
    bucket: str,
    num_clips: int,
    num_candidates: int,
    balanced_subsets: bool = True,
) -> list[dict[str, Any]]:
    dataset = _build_dataset(cfg, bucket)
    if balanced_subsets:
        indices = _balanced_subset_indices(dataset, int(num_candidates))
    else:
        indices = list(range(min(int(num_candidates), len(dataset))))
    subset = Subset(dataset, indices[: int(num_candidates)])
    loader = DataLoader(subset, batch_size=1, shuffle=False, collate_fn=collate_hoi, num_workers=0)
    return [batch for _, batch in zip(range(int(num_clips)), loader)]


def event_records_from_contact(
    contact_state: Tensor,
    seq_len: Tensor,
    *,
    threshold: float = 0.5,
    hands_only: bool = True,
) -> list[dict[str, Any]]:
    """Return onset/release events using the repo's transition convention."""
    contact = contact_state.detach().cpu()
    lengths = seq_len.detach().cpu().long().tolist()
    specs = HAND_SPECS if hands_only else (
        ("L_hand", 20, 0), ("R_hand", 21, 1), ("L_foot", 10, 2), ("R_foot", 11, 3), ("pelvis", 0, 4),
    )
    events: list[dict[str, Any]] = []
    for b, valid in enumerate(lengths):
        valid = int(valid)
        if valid < 2:
            continue
        for part, joint, part_idx in specs:
            c = contact[b, :valid, part_idx] > float(threshold)
            onset = torch.where(c[1:] & ~c[:-1])[0] + 1
            release = torch.where(~c[1:] & c[:-1])[0] + 1
            for t in onset.tolist():
                events.append({"batch": b, "kind": "onset", "part": part, "joint": joint, "part_idx": part_idx, "frame": int(t)})
            for t in release.tolist():
                events.append({"batch": b, "kind": "release", "part": part, "joint": joint, "part_idx": part_idx, "frame": int(t)})
    return events


def first_event_per_clip(events: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for ev in events:
        out.setdefault(int(ev["batch"]), ev)
    return out


def body_jerk_p95(joints: Tensor, seq_mask: Tensor, *, scale_to_cm: float = 100.0) -> float:
    if joints.shape[1] < 4:
        return 0.0
    _vel_w, vel_local, _acc_w, acc_local = _joint_velocity_acceleration(joints, seq_mask)
    jerk = acc_local[:, 1:] - acc_local[:, :-1]
    jerk_mag = torch.linalg.vector_norm(jerk[:, :, LOCAL_JOINTS], dim=-1) * float(scale_to_cm)
    mask = seq_mask[:, :-3] & seq_mask[:, 1:-2] & seq_mask[:, 2:-1] & seq_mask[:, 3:]
    return float(_stats_from_magnitudes(jerk_mag, mask.unsqueeze(-1).expand(-1, -1, len(LOCAL_JOINTS)))["p95"])


def dynamics_metrics(
    joints: Tensor,
    seq_mask: Tensor,
    *,
    gt_joints: Tensor | None = None,
    fps: float = 20.0,
) -> dict[str, float]:
    stats = _per_joint_vel_stats(joints, seq_mask)
    fft = _fft_band_energy(joints, seq_mask, fps=fps)
    out = {
        "body_velocity_cm_per_frame": float(stats["body_local_vel_cm_per_frame"]["mean"]),
        "body_acc_p95_cm_per_frame2": float(stats["body_local_acc_cm_per_frame"]["p95"]),
        "body_jerk_p95_cm_per_frame3": body_jerk_p95(joints, seq_mask),
        "fft_low": float(fft["fraction_low"]),
        "fft_mid": float(fft["fraction_mid"]),
        "fft_high": float(fft["fraction_high"]),
        "L_hand_velocity_cm_per_frame": float(stats["per_joint_vel_cm_per_frame"]["L_hand"]["mean"]),
        "R_hand_velocity_cm_per_frame": float(stats["per_joint_vel_cm_per_frame"]["R_hand"]["mean"]),
        "L_foot_velocity_cm_per_frame": float(stats["per_joint_vel_cm_per_frame"]["L_foot"]["mean"]),
        "R_foot_velocity_cm_per_frame": float(stats["per_joint_vel_cm_per_frame"]["R_foot"]["mean"]),
    }
    out["hand_velocity_cm_per_frame"] = 0.5 * (
        out["L_hand_velocity_cm_per_frame"] + out["R_hand_velocity_cm_per_frame"]
    )
    out["foot_velocity_cm_per_frame"] = 0.5 * (
        out["L_foot_velocity_cm_per_frame"] + out["R_foot_velocity_cm_per_frame"]
    )
    if gt_joints is not None:
        gt = dynamics_metrics(gt_joints, seq_mask, gt_joints=None, fps=fps)
        for key in (
            "body_velocity_cm_per_frame",
            "hand_velocity_cm_per_frame",
            "foot_velocity_cm_per_frame",
            "body_acc_p95_cm_per_frame2",
            "body_jerk_p95_cm_per_frame3",
        ):
            out[f"{key}_over_gt"] = safe_div(out[key], gt[key])
    return out


def transition_metrics(
    joints: Tensor,
    object_positions: Tensor,
    contact_state: Tensor,
    seq_mask: Tensor,
    *,
    gt_joints: Tensor | None = None,
    window_k: int = 10,
    threshold: float = 0.5,
    metric_version: str = "v1",
    object_pc: Tensor | None = None,
    object_rotations: Tensor | None = None,
    edge_margin: int = 5,
    min_gt_change_cm: float = 2.0,
    flicker_max_frames: int = 2,
    m5_clip_cms: tuple[float, ...] = (2.0, 5.0),
    m5_ratio_cap: float = 5.0,
) -> dict[str, Any]:
    """Hand-object transition metrics.

    metric_version="v1" (default, backward compatible) — uses object COM
    distance, ``onset_positive_closing_cm`` / ``release_positive_opening_cm``
    summary stats, and divides by GT mean when gt_joints is supplied.
    Same numerical output as before the v2 patch.

    metric_version="v2" — uses object SURFACE distance when ``object_pc``
    and ``object_rotations`` are provided (COM fallback otherwise);
    reports M2 (slope cm/frame), M3 (signed pre-post Δ cm), M5 (robust
    clipped ratio gen/max(gt, threshold_cm)); applies per-event validity
    flags so denominator-unstable events do not contaminate aggregates.
    Round 6 found 65.2% of events are denominator-unstable under v1;
    v2 is the proposed replacement.
    """
    if metric_version not in {"v1", "v2"}:
        raise ValueError(f"metric_version must be v1 or v2, got {metric_version!r}")
    if metric_version == "v1":
        return _transition_metrics_v1(
            joints, object_positions, contact_state, seq_mask,
            gt_joints=gt_joints, window_k=window_k, threshold=threshold,
        )
    return _transition_metrics_v2(
        joints, object_positions, contact_state, seq_mask,
        gt_joints=gt_joints, window_k=window_k, threshold=threshold,
        object_pc=object_pc, object_rotations=object_rotations,
        edge_margin=edge_margin, min_gt_change_cm=min_gt_change_cm,
        flicker_max_frames=flicker_max_frames,
        m5_clip_cms=m5_clip_cms, m5_ratio_cap=m5_ratio_cap,
    )


def _transition_metrics_v1(
    joints: Tensor,
    object_positions: Tensor,
    contact_state: Tensor,
    seq_mask: Tensor,
    *,
    gt_joints: Tensor | None = None,
    window_k: int = 10,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """V1 (legacy) — kept for backward compatibility.

    Round-6 Diag C shows this metric is denominator-unstable on 65%+ of
    events. New callers should pass metric_version="v2".
    """
    events = event_records_from_contact(contact_state, seq_mask.sum(dim=1), threshold=threshold, hands_only=True)
    obj = object_positions
    rows: list[dict[str, float]] = []
    onset_positive: list[float] = []
    release_positive: list[float] = []
    onset_raw: list[float] = []
    release_raw: list[float] = []
    rel_vels: list[float] = []
    for ev in events:
        b = int(ev["batch"])
        t = int(ev["frame"])
        valid = int(seq_mask[b].sum().item())
        joint = int(ev["joint"])
        if t < 0 or t >= valid:
            continue
        hand = joints[b, :valid, joint]
        dist = torch.linalg.vector_norm(hand - obj[b, :valid], dim=-1) * 100.0
        if ev["kind"] == "onset":
            start = max(0, t - int(window_k))
            raw = float(dist[start].item() - dist[t].item())
            lo, hi = start, t
            onset_raw.append(raw)
            onset_positive.append(max(0.0, raw))
        else:
            end = min(valid - 1, t + int(window_k))
            raw = float(dist[end].item() - dist[t].item())
            lo, hi = t, end
            release_raw.append(raw)
            release_positive.append(max(0.0, raw))
        if hi > lo:
            hvel = hand[lo + 1 : hi + 1] - hand[lo:hi]
            ovel = obj[b, lo + 1 : hi + 1] - obj[b, lo:hi]
            rel = float(torch.linalg.vector_norm(hvel - ovel, dim=-1).mean().item() * 100.0)
        else:
            rel = 0.0
        rel_vels.append(rel)
        rows.append({
            "batch": float(b),
            "frame": float(t),
            "positive_distance_change_cm": max(0.0, raw),
            "raw_distance_change_cm": raw,
            "transition_relative_velocity_cm_per_frame": rel,
        })
    out: dict[str, Any] = {
        "metric_version": "v1",
        "n_events": len(events),
        "n_onsets": len(onset_positive),
        "n_releases": len(release_positive),
        "onset_positive_closing_cm": stats_list(onset_positive),
        "release_positive_opening_cm": stats_list(release_positive),
        "onset_raw_closing_cm": stats_list(onset_raw),
        "release_raw_opening_cm": stats_list(release_raw),
        "transition_relative_velocity_cm_per_frame": stats_list(rel_vels),
    }
    if gt_joints is not None:
        gt = _transition_metrics_v1(
            gt_joints,
            object_positions,
            contact_state,
            seq_mask,
            gt_joints=None,
            window_k=window_k,
            threshold=threshold,
        )
        out["ratios_over_gt"] = {
            "onset_positive_closing": safe_div(
                float(out["onset_positive_closing_cm"]["mean"]),
                float(gt["onset_positive_closing_cm"]["mean"]),
            ),
            "release_positive_opening": safe_div(
                float(out["release_positive_opening_cm"]["mean"]),
                float(gt["release_positive_opening_cm"]["mean"]),
            ),
            "transition_relative_velocity": safe_div(
                float(out["transition_relative_velocity_cm_per_frame"]["mean"]),
                float(gt["transition_relative_velocity_cm_per_frame"]["mean"]),
            ),
        }
    return out


# ---------------------------------------------------------------------------
# v2 helpers
# ---------------------------------------------------------------------------


def _axis_angle_to_rot_torch(aa: Tensor) -> Tensor:
    """Rodrigues. ``aa`` (..., 3) -> (..., 3, 3)."""
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


def _segment_pairs_np(c_bool: np.ndarray) -> list[tuple[int, int]]:
    """Pair onsets and releases into (start, end_exclusive) segments."""
    if c_bool.size < 2:
        return []
    onset = (c_bool[1:] & ~c_bool[:-1])
    release = (~c_bool[1:] & c_bool[:-1])
    onset_idx = (np.where(onset)[0] + 1).tolist()
    release_idx = (np.where(release)[0] + 1).tolist()
    segments: list[tuple[int, int]] = []
    ri = 0
    for o in onset_idx:
        while ri < len(release_idx) and release_idx[ri] <= o:
            ri += 1
        if ri < len(release_idx):
            segments.append((o, release_idx[ri]))
            ri += 1
        else:
            segments.append((o, int(c_bool.size)))
    return segments


def _v2_metrics_for_curve(
    dist_cm: np.ndarray, kind: str, t: int, seq_len: int, window_k: int,
) -> dict[str, float]:
    """Compute M1 raw, M1 positive, M2 slope, M3 signed pre-post for one event."""
    if kind == "onset":
        lo, hi = max(0, t - window_k), int(t)
    else:
        lo, hi = int(t), min(seq_len - 1, t + window_k)
    if kind == "onset":
        raw = float(dist_cm[lo] - dist_cm[int(t)])
    else:
        raw = float(dist_cm[hi] - dist_cm[int(t)])
    m1_pos = max(0.0, raw)
    n = hi - lo + 1
    if n >= 3:
        xs = np.arange(n)
        ys = dist_cm[lo:hi + 1]
        m2 = float(np.polyfit(xs, ys, 1)[0])
    else:
        m2 = 0.0
    pre_lo = max(0, t - window_k)
    post_hi = min(seq_len - 1, t + window_k)
    pre_mean = float(dist_cm[pre_lo:t + 1].mean()) if t > pre_lo else float(dist_cm[int(t)])
    post_mean = float(dist_cm[t:post_hi + 1].mean()) if post_hi > t else float(dist_cm[int(t)])
    m3 = pre_mean - post_mean if kind == "onset" else post_mean - pre_mean
    return {
        "m1_raw_change_cm": raw,
        "m1_positive_change_cm": m1_pos,
        "m2_slope_cm_per_frame": m2,
        "m3_signed_diff_cm": float(m3),
    }


def _transition_metrics_v2(
    joints: Tensor,
    object_positions: Tensor,
    contact_state: Tensor,
    seq_mask: Tensor,
    *,
    gt_joints: Tensor | None,
    window_k: int,
    threshold: float,
    object_pc: Tensor | None,
    object_rotations: Tensor | None,
    edge_margin: int,
    min_gt_change_cm: float,
    flicker_max_frames: int,
    m5_clip_cms: tuple[float, ...],
    m5_ratio_cap: float,
) -> dict[str, Any]:
    """V2 — surface-aware, denominator-stable transition metrics.

    Reports M2 (slope), M3 (signed pre-post), M5 (clipped ratio) plus
    per-event validity flags. M2 / M3 do not depend on GT-denominator
    stability; M5 does.
    """
    joints_np = joints.detach().cpu().numpy().astype(np.float32)
    obj_np = object_positions.detach().cpu().numpy().astype(np.float32)
    cs_np = contact_state.detach().cpu().numpy().astype(np.float32)
    seq_len_arr = seq_mask.sum(dim=1).detach().cpu().long().tolist()
    B, T_pad = cs_np.shape[:2]

    # Optional surface support
    use_surface = (object_pc is not None) and (object_rotations is not None)
    if use_surface:
        pc_np = object_pc.detach().cpu().numpy().astype(np.float32)         # (B, N, 3)
        rot_np = object_rotations.detach().cpu().numpy().astype(np.float32) # (B, T, 3)

    def _surface_curve(b: int, valid: int, joint: int) -> np.ndarray | None:
        if not use_surface:
            return None
        aa = rot_np[b, :valid]                                   # (T, 3)
        theta = np.linalg.norm(aa, axis=-1, keepdims=True).clip(min=1e-12)
        k = aa / theta
        K = np.zeros(aa.shape + (3,), dtype=np.float32)
        K[..., 0, 1] = -k[..., 2]; K[..., 0, 2] = k[..., 1]
        K[..., 1, 0] = k[..., 2];  K[..., 1, 2] = -k[..., 0]
        K[..., 2, 0] = -k[..., 1]; K[..., 2, 1] = k[..., 0]
        eye = np.broadcast_to(np.eye(3, dtype=np.float32), K.shape)
        s = np.sin(theta)[..., None]
        c = np.cos(theta)[..., None]
        R = eye + s * K + (1 - c) * (K @ K)                     # (T, 3, 3)
        # transform pc into world
        pc_world = np.einsum("tij,nj->tni", R, pc_np[b]) + obj_np[b, :valid, None, :]  # (T, N, 3)
        h = joints_np[b, :valid, joint][:, None, :]
        d = np.linalg.norm(h - pc_world, axis=-1).min(axis=-1) * 100.0
        return d.astype(np.float32)

    def _com_curve(b: int, valid: int, joint: int) -> np.ndarray:
        h = joints_np[b, :valid, joint]
        o = obj_np[b, :valid]
        return (np.linalg.norm(h - o, axis=-1) * 100.0).astype(np.float32)

    # Build per-clip events with paired (segment_start, segment_end) so we
    # know duration → flicker filter.
    gen_event_rows: list[dict[str, Any]] = []
    gt_event_rows: list[dict[str, Any]] = []
    for b in range(B):
        valid = int(seq_len_arr[b])
        if valid < 2:
            continue
        for part, joint, p_idx in HAND_SPECS:
            c_bool = cs_np[b, :valid, p_idx] > float(threshold)
            segments = _segment_pairs_np(c_bool)
            d_com_gen = _com_curve(b, valid, joint)
            d_surf_gen = _surface_curve(b, valid, joint)
            d_com_gt = None
            d_surf_gt = None
            if gt_joints is not None:
                gt_np = gt_joints.detach().cpu().numpy().astype(np.float32)
                hg = gt_np[b, :valid, joint]
                d_com_gt = (np.linalg.norm(hg - obj_np[b, :valid], axis=-1) * 100.0).astype(np.float32)
                if use_surface and d_surf_gen is not None:
                    # GT surface curve uses same object pose / same pc; replace hand
                    aa = rot_np[b, :valid]
                    theta = np.linalg.norm(aa, axis=-1, keepdims=True).clip(min=1e-12)
                    k = aa / theta
                    K = np.zeros(aa.shape + (3,), dtype=np.float32)
                    K[..., 0, 1] = -k[..., 2]; K[..., 0, 2] = k[..., 1]
                    K[..., 1, 0] = k[..., 2];  K[..., 1, 2] = -k[..., 0]
                    K[..., 2, 0] = -k[..., 1]; K[..., 2, 1] = k[..., 0]
                    eye = np.broadcast_to(np.eye(3, dtype=np.float32), K.shape)
                    s = np.sin(theta)[..., None]
                    c = np.cos(theta)[..., None]
                    R = eye + s * K + (1 - c) * (K @ K)
                    pc_world = np.einsum("tij,nj->tni", R, pc_np[b]) + obj_np[b, :valid, None, :]
                    d_surf_gt = (np.linalg.norm(hg[:, None, :] - pc_world, axis=-1).min(axis=-1) * 100.0).astype(np.float32)
            for s, e in segments:
                duration = int(max(1, e - s))
                for kind, t_ev in (("onset", s), ("release", e)):
                    t_ev = int(min(max(0, t_ev), valid - 1))
                    # validity flags
                    in_pre_range = (kind == "onset" and t_ev - window_k >= 0) or kind == "release"
                    in_post_range = (kind == "release" and t_ev + window_k <= valid - 1) or kind == "onset"
                    away_from_edge = (t_ev >= int(edge_margin) and t_ev <= valid - 1 - int(edge_margin))
                    not_flicker = duration > int(flicker_max_frames)
                    # compute metric curves
                    gen_curve = d_surf_gen if d_surf_gen is not None else d_com_gen
                    gen_source = "surface" if d_surf_gen is not None else "com_fallback"
                    com_gen_metrics = _v2_metrics_for_curve(d_com_gen, kind, t_ev, valid, window_k)
                    surf_gen_metrics = (
                        _v2_metrics_for_curve(d_surf_gen, kind, t_ev, valid, window_k)
                        if d_surf_gen is not None else None
                    )
                    gen_event_rows.append({
                        "batch": int(b), "part": part, "joint": int(joint),
                        "kind": kind, "frame": int(t_ev),
                        "segment_start": int(s), "segment_end": int(e),
                        "duration": int(duration),
                        "is_flicker": bool(not not_flicker),
                        "in_pre_range": bool(in_pre_range),
                        "in_post_range": bool(in_post_range),
                        "away_from_edge": bool(away_from_edge),
                        "distance_source": gen_source,
                        "com": com_gen_metrics,
                        "surface": surf_gen_metrics,
                    })
                    if d_com_gt is not None:
                        com_gt_metrics = _v2_metrics_for_curve(d_com_gt, kind, t_ev, valid, window_k)
                        surf_gt_metrics = (
                            _v2_metrics_for_curve(d_surf_gt, kind, t_ev, valid, window_k)
                            if d_surf_gt is not None else None
                        )
                        gt_event_rows.append({
                            "batch": int(b), "part": part, "joint": int(joint),
                            "kind": kind, "frame": int(t_ev),
                            "duration": int(duration),
                            "com": com_gt_metrics,
                            "surface": surf_gt_metrics,
                        })

    # Pair gen↔gt event rows for denominator + ratio metrics
    def _gt_lookup(b: int, part: str, kind: str, frame: int) -> dict[str, Any] | None:
        for g in gt_event_rows:
            if (g["batch"] == b and g["part"] == part and g["kind"] == kind
                    and g["frame"] == frame):
                return g
        return None

    # Per-event derived fields
    for ev in gen_event_rows:
        gt_row = _gt_lookup(ev["batch"], ev["part"], ev["kind"], ev["frame"]) if gt_event_rows else None
        if gt_row is not None:
            ev["gt_com"] = gt_row["com"]
            ev["gt_surface"] = gt_row["surface"]
            denom_basis_com = float(gt_row["com"]["m1_positive_change_cm"])
            denom_basis_surf = (
                float(gt_row["surface"]["m1_positive_change_cm"]) if gt_row["surface"] is not None else 0.0
            )
            ev["denom_stable_2cm"] = bool(denom_basis_com >= 2.0)
            ev["denom_stable_5cm"] = bool(denom_basis_com >= 5.0)
            ev["denom_stable_2cm_surf"] = bool(denom_basis_surf >= 2.0)
            # M5 ratios
            gen_change = float(ev["com"]["m1_positive_change_cm"])
            m5_ratios = {}
            for cm in m5_clip_cms:
                r = gen_change / max(denom_basis_com, float(cm))
                m5_ratios[f"m5_ratio_com_clip_{int(cm)}cm"] = float(np.clip(r, -float(m5_ratio_cap), float(m5_ratio_cap)))
            ev["m5_ratios"] = m5_ratios
        else:
            # No GT — use self-denom for stability flag (gen positive change)
            gen_change = float(ev["com"]["m1_positive_change_cm"])
            ev["denom_stable_2cm"] = bool(gen_change >= 2.0)
            ev["denom_stable_5cm"] = bool(gen_change >= 5.0)
            ev["denom_stable_2cm_surf"] = (
                bool(float(ev["surface"]["m1_positive_change_cm"]) >= 2.0)
                if ev["surface"] is not None else False
            )
        # Tier flags
        in_range = ev["in_pre_range"] and ev["in_post_range"] and ev["away_from_edge"]
        not_flick = not ev["is_flicker"]
        ev["valid_v2_slope"] = bool(in_range and not_flick)
        ev["valid_v2_signed"] = bool(in_range and not_flick)
        ev["valid_v2_ratio_2cm"] = bool(in_range and not_flick and ev["denom_stable_2cm"])
        ev["valid_v2_ratio_5cm"] = bool(in_range and not_flick and ev["denom_stable_5cm"])

    # Aggregates
    n_total = len(gen_event_rows)
    n_boundary = sum(1 for e in gen_event_rows if not e["away_from_edge"])
    n_flicker = sum(1 for e in gen_event_rows if e["is_flicker"])
    n_denom_unstable_2cm = sum(1 for e in gen_event_rows if not e["denom_stable_2cm"])
    n_denom_unstable_5cm = sum(1 for e in gen_event_rows if not e["denom_stable_5cm"])
    n_valid_slope = sum(1 for e in gen_event_rows if e["valid_v2_slope"])
    n_valid_signed = sum(1 for e in gen_event_rows if e["valid_v2_signed"])
    n_valid_ratio_2cm = sum(1 for e in gen_event_rows if e["valid_v2_ratio_2cm"])
    n_valid_ratio_5cm = sum(1 for e in gen_event_rows if e["valid_v2_ratio_5cm"])

    def _vec(rows: list[dict[str, Any]], path: str) -> list[float]:
        parts = path.split(".")
        out = []
        for r in rows:
            d = r
            for p in parts:
                d = d.get(p) if isinstance(d, dict) else None
                if d is None:
                    break
            if isinstance(d, (int, float)):
                out.append(float(d))
        return out

    slope_events = [e for e in gen_event_rows if e["valid_v2_slope"]]
    signed_events = [e for e in gen_event_rows if e["valid_v2_signed"]]
    ratio_2_events = [e for e in gen_event_rows if e["valid_v2_ratio_2cm"]]
    ratio_5_events = [e for e in gen_event_rows if e["valid_v2_ratio_5cm"]]

    # Direction scores
    onset_slope_com = stats_list([e["com"]["m2_slope_cm_per_frame"] for e in slope_events if e["kind"] == "onset"])
    release_slope_com = stats_list([e["com"]["m2_slope_cm_per_frame"] for e in slope_events if e["kind"] == "release"])
    onset_direction_score = stats_list([-e["com"]["m2_slope_cm_per_frame"] for e in slope_events if e["kind"] == "onset"])
    release_direction_score = stats_list([e["com"]["m2_slope_cm_per_frame"] for e in slope_events if e["kind"] == "release"])
    onset_signed_com = stats_list([e["com"]["m3_signed_diff_cm"] for e in signed_events if e["kind"] == "onset"])
    release_signed_com = stats_list([e["com"]["m3_signed_diff_cm"] for e in signed_events if e["kind"] == "release"])

    # Surface availability + disagreement
    n_with_surface = sum(1 for e in gen_event_rows if e["surface"] is not None)
    if n_with_surface > 0:
        com_vals = np.asarray([e["com"]["m1_positive_change_cm"] for e in gen_event_rows if e["surface"] is not None])
        surf_vals = np.asarray([e["surface"]["m1_positive_change_cm"] for e in gen_event_rows if e["surface"] is not None])
        com_surf_mean_offset_cm = float((com_vals - surf_vals).mean())
        if com_vals.std() > 1e-8 and surf_vals.std() > 1e-8 and com_vals.size > 2:
            com_surf_corr = float(np.corrcoef(com_vals, surf_vals)[0, 1])
        else:
            com_surf_corr = 0.0
    else:
        com_surf_mean_offset_cm = 0.0
        com_surf_corr = 0.0

    out: dict[str, Any] = {
        "metric_version": "v2",
        "window_k": int(window_k),
        "edge_margin": int(edge_margin),
        "min_gt_change_cm": float(min_gt_change_cm),
        "flicker_max_frames": int(flicker_max_frames),
        "use_surface": bool(use_surface),
        "n_events_total": int(n_total),
        "n_boundary": int(n_boundary),
        "n_flicker": int(n_flicker),
        "n_denom_unstable_2cm": int(n_denom_unstable_2cm),
        "n_denom_unstable_5cm": int(n_denom_unstable_5cm),
        "n_valid_slope": int(n_valid_slope),
        "n_valid_signed": int(n_valid_signed),
        "n_valid_ratio_2cm": int(n_valid_ratio_2cm),
        "n_valid_ratio_5cm": int(n_valid_ratio_5cm),
        # M2 slope summaries on valid_slope events
        "onset_slope_cm_per_frame": onset_slope_com,
        "release_slope_cm_per_frame": release_slope_com,
        "onset_direction_score_cm_per_frame": onset_direction_score,
        "release_direction_score_cm_per_frame": release_direction_score,
        # M3 signed summaries on valid_signed events
        "onset_signed_diff_cm": onset_signed_com,
        "release_signed_diff_cm": release_signed_com,
        # surface vs com
        "surface_vs_com_corr_m1_positive_change": com_surf_corr,
        "surface_vs_com_mean_offset_cm": com_surf_mean_offset_cm,
        "n_events_with_surface": int(n_with_surface),
    }
    # M5 ratio summaries on valid_ratio events
    if ratio_2_events:
        for cm in m5_clip_cms:
            key = f"m5_ratio_com_clip_{int(cm)}cm"
            vec = [e["m5_ratios"][key] for e in ratio_2_events if "m5_ratios" in e]
            out[f"m5_ratio_clip_{int(cm)}cm"] = stats_list(vec)
    # Full event log surface (compact)
    out["events"] = gen_event_rows
    return out


def clip_metadata(batch: dict[str, Any], events: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    first_event = first_event_per_clip(events or [])
    rows: list[dict[str, Any]] = []
    n = len(batch.get("seq_id", []))
    for i in range(n):
        row = {
            "batch": i,
            "subset": str(batch["subset"][i]),
            "seq_id": str(batch["seq_id"][i]),
            "object_id": str(batch.get("object_id", [""] * n)[i]),
            "text": str(batch["text"][i]),
            "seq_len": int(batch["seq_len"][i].item()),
        }
        if i in first_event:
            ev = first_event[i]
            row["event"] = {
                "kind": ev["kind"],
                "part": ev["part"],
                "frame": int(ev["frame"]),
            }
        rows.append(row)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

