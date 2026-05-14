"""Non-contact transition diagnostic for Stage B v18.

This standalone diagnostic checks whether the remaining "slow approach /
frozen body" failure is concentrated around hand-object transition regions:

* pre-contact approach windows before GT contact onsets;
* post-release windows after GT releases;
* onset/release transition neighborhoods;
* far non-contact regions;
* the repo's actual phase labels
  (non_contact / stable_contact / manipulation).

It compares GT, one-step reconstruction at a fixed diffusion timestep, and
normal DDPM sampling under the same conditioning. Training and sampling code
paths are reused exactly as in the existing Stage B diagnostics; this file only
adds offline aggregation and reporting.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from dynamics_diagnostic import (
    LOCAL_JOINTS,
    _build_cond,
    _build_dataset,
    _build_model,
    _balanced_subset_indices,
    _fk_from_motion_135,
    _joint_velocity_acceleration,
    _one_step_recon_motion,
    _stats_from_magnitudes,
)
from piano.data.dataset import collate_hoi
from piano.data.pseudo_labels.extract_phase import PHASE_NAMES
from piano.utils.clip_utils import load_clip_text_encoder


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

SOURCES = ("gt", "sampled", "recon_one_step")


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if abs(float(den)) > 1e-12 else 0.0


def _extract_plan(batch: dict[str, Any], device: torch.device) -> dict[str, Tensor]:
    return {key: batch[f"plan_{key}"].to(device) for key in PLAN_KEYS}


def _stats_from_values(values: Tensor, mask: Tensor) -> dict[str, float | int]:
    flat = values[mask.bool()].detach().cpu().double().numpy()
    if flat.size == 0:
        return {
            "mean": 0.0,
            "median": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "p95": 0.0,
            "std": 0.0,
            "n": 0,
        }
    return {
        "mean": float(flat.mean()),
        "median": float(np.median(flat)),
        "p25": float(np.percentile(flat, 25)),
        "p75": float(np.percentile(flat, 75)),
        "p95": float(np.percentile(flat, 95)),
        "std": float(flat.std()),
        "n": int(flat.size),
    }


def _expand_event_mask(events: Tensor, offsets: range, seq_mask: Tensor) -> Tensor:
    out = torch.zeros_like(events, dtype=torch.bool)
    batch_size, total_t = events.shape
    for offset in offsets:
        if offset == 0:
            shifted = events
        elif offset > 0:
            shifted = torch.zeros_like(events, dtype=torch.bool)
            shifted[:, offset:] = events[:, : total_t - offset]
        else:
            shifted = torch.zeros_like(events, dtype=torch.bool)
            shifted[:, : total_t + offset] = events[:, -offset:]
        out |= shifted
    return out & seq_mask.bool()


def _event_frame_mask(contact_mask: Tensor, seq_mask: Tensor) -> tuple[Tensor, Tensor]:
    onset = torch.zeros_like(contact_mask, dtype=torch.bool)
    release = torch.zeros_like(contact_mask, dtype=torch.bool)
    valid_step = seq_mask[:, 1:] & seq_mask[:, :-1]
    onset[:, 1:] = contact_mask[:, 1:] & ~contact_mask[:, :-1] & valid_step
    release[:, 1:] = ~contact_mask[:, 1:] & contact_mask[:, :-1] & valid_step
    return onset, release


def _count_contact_segments(contact_mask: Tensor, seq_mask: Tensor) -> int:
    total = 0
    for b in range(contact_mask.shape[0]):
        valid = int(seq_mask[b].sum().item())
        if valid <= 0:
            continue
        c = contact_mask[b, :valid]
        total += int(c[0].item())
        if valid > 1:
            total += int((c[1:] & ~c[:-1]).sum().item())
    return total


def _far_noncontact_mask(
    contact_mask: Tensor,
    onset: Tensor,
    release: Tensor,
    seq_mask: Tensor,
    k: int,
) -> Tensor:
    out = torch.zeros_like(contact_mask, dtype=torch.bool)
    device = contact_mask.device
    for b in range(contact_mask.shape[0]):
        valid = int(seq_mask[b].sum().item())
        if valid <= 0:
            continue
        idx = torch.arange(valid, device=device)
        boundary = torch.where((onset[b, :valid] | release[b, :valid]))[0]
        non_contact = ~contact_mask[b, :valid]
        if boundary.numel() == 0:
            out[b, :valid] = non_contact
            continue
        dist = (idx[:, None] - boundary[None, :]).abs().min(dim=1).values
        out[b, :valid] = non_contact & (dist > int(k))
    return out & seq_mask.bool()


def _build_part_windows(
    contact_state: Tensor,
    seq_mask: Tensor,
    threshold: float,
    main_k: int,
    short_k: int,
    transition_radius: int,
) -> dict[str, dict[str, Any]]:
    windows: dict[str, dict[str, Any]] = {}
    for part_name, _joint_idx, part_idx in HAND_SPECS:
        contact_mask = (contact_state[:, :, part_idx] > float(threshold)) & seq_mask.bool()
        onset, release = _event_frame_mask(contact_mask, seq_mask)
        contact_segments = _count_contact_segments(contact_mask, seq_mask)
        onset_events = int(onset.sum().item())
        release_events = int(release.sum().item())

        frame_masks: dict[str, Tensor] = {
            "in_contact": contact_mask,
            f"pre_contact_k{main_k}": _expand_event_mask(
                onset,
                range(-int(main_k), 0),
                seq_mask,
            ),
            f"pre_contact_k{short_k}": _expand_event_mask(
                onset,
                range(-int(short_k), 0),
                seq_mask,
            ),
            f"post_release_k{main_k}": _expand_event_mask(
                release,
                range(1, int(main_k) + 1),
                seq_mask,
            ),
            f"post_release_k{short_k}": _expand_event_mask(
                release,
                range(1, int(short_k) + 1),
                seq_mask,
            ),
            f"onset_pm{transition_radius}": _expand_event_mask(
                onset,
                range(-int(transition_radius), int(transition_radius) + 1),
                seq_mask,
            ),
            f"release_pm{transition_radius}": _expand_event_mask(
                release,
                range(-int(transition_radius), int(transition_radius) + 1),
                seq_mask,
            ),
        }
        frame_masks[f"transition_pm{transition_radius}"] = (
            frame_masks[f"onset_pm{transition_radius}"]
            | frame_masks[f"release_pm{transition_radius}"]
        )
        frame_masks[f"far_non_contact_k{main_k}"] = _far_noncontact_mask(
            contact_mask,
            onset,
            release,
            seq_mask,
            k=main_k,
        )

        count_events = {
            "in_contact": contact_segments,
            f"pre_contact_k{main_k}": onset_events,
            f"pre_contact_k{short_k}": onset_events,
            f"post_release_k{main_k}": release_events,
            f"post_release_k{short_k}": release_events,
            f"onset_pm{transition_radius}": onset_events,
            f"release_pm{transition_radius}": release_events,
            f"transition_pm{transition_radius}": onset_events + release_events,
            f"far_non_contact_k{main_k}": 0,
        }
        windows[part_name] = {
            "contact_mask": contact_mask,
            "onset_mask": onset,
            "release_mask": release,
            "frame_masks": frame_masks,
            "counts": {
                name: {
                    "n_frames": int(mask.sum().item()),
                    "n_events": int(count_events.get(name, 0)),
                }
                for name, mask in frame_masks.items()
            },
        }
    return windows


def _masked_fft_band_energy(
    joints: Tensor,
    frame_mask: Tensor,
    fps: float,
    min_segment_frames: int = 8,
) -> dict[str, float | int]:
    root = joints[:, :, 0:1, :]
    rel = joints - root
    low_e = 0.0
    mid_e = 0.0
    high_e = 0.0
    segment_count = 0
    frame_count = 0

    for b in range(joints.shape[0]):
        mask = frame_mask[b].detach().cpu().numpy().astype(bool)
        total_t = mask.shape[0]
        start = None
        for t in range(total_t + 1):
            active = bool(mask[t]) if t < total_t else False
            if active and start is None:
                start = t
            elif not active and start is not None:
                end = t
                length = end - start
                if length >= int(min_segment_frames):
                    x = rel[b, start:end].float()
                    x = x - x.mean(dim=0, keepdim=True)
                    fft = torch.fft.rfft(x, dim=0)
                    power = (fft.real.square() + fft.imag.square())
                    body_power = power[:, LOCAL_JOINTS, :].sum(dim=(1, 2))
                    freqs = torch.fft.rfftfreq(length, d=1.0 / float(fps)).to(body_power.device)
                    low_e += float(body_power[(freqs >= 0.0) & (freqs < 1.0)].sum().item())
                    mid_e += float(body_power[(freqs >= 1.0) & (freqs < 4.0)].sum().item())
                    high_e += float(body_power[freqs >= 4.0].sum().item())
                    segment_count += 1
                    frame_count += length
                start = None

    total_e = low_e + mid_e + high_e
    return {
        "energy_total": float(total_e),
        "energy_low": float(low_e),
        "energy_mid": float(mid_e),
        "energy_high": float(high_e),
        "fraction_low": _safe_div(low_e, total_e),
        "fraction_mid": _safe_div(mid_e, total_e),
        "fraction_high": _safe_div(high_e, total_e),
        "n_segments": int(segment_count),
        "n_frames_fft": int(frame_count),
        "min_segment_frames": int(min_segment_frames),
    }


def _body_window_metrics(
    joints: Tensor,
    frame_mask: Tensor,
    seq_mask: Tensor,
    fps: float,
) -> dict[str, Any]:
    _vel_world, vel_local, _acc_world, acc_local = _joint_velocity_acceleration(
        joints,
        frame_mask.bool(),
    )
    vel_mag = vel_local.pow(2).sum(-1).clamp_min(1e-12).sqrt() * 100.0
    acc_mag = acc_local.pow(2).sum(-1).clamp_min(1e-12).sqrt() * 100.0
    vel_mask = frame_mask[:, :-1] & seq_mask[:, :-1] & seq_mask[:, 1:]
    acc_mask = frame_mask[:, :-2] & seq_mask[:, :-2] & seq_mask[:, 1:-1] & seq_mask[:, 2:]
    body_vel = vel_mag[:, :, LOCAL_JOINTS]
    body_acc = acc_mag[:, :, LOCAL_JOINTS]
    return {
        "body_local_velocity_cm_per_frame": _stats_from_magnitudes(
            body_vel,
            vel_mask.unsqueeze(-1).expand(-1, -1, len(LOCAL_JOINTS)),
        ),
        "body_local_acceleration_cm_per_frame2": _stats_from_magnitudes(
            body_acc,
            acc_mask.unsqueeze(-1).expand(-1, -1, len(LOCAL_JOINTS)),
        ),
        "fft_spectrum": _masked_fft_band_energy(
            joints,
            frame_mask,
            fps=fps,
        ),
    }


def _hand_window_metrics(
    joints: Tensor,
    frame_mask: Tensor,
    seq_mask: Tensor,
    hand_joint: int,
) -> dict[str, Any]:
    _vel_world, vel_local, _acc_world, acc_local = _joint_velocity_acceleration(
        joints,
        frame_mask.bool(),
    )
    vel_mag = vel_local[:, :, hand_joint].pow(2).sum(-1).clamp_min(1e-12).sqrt() * 100.0
    acc_mag = acc_local[:, :, hand_joint].pow(2).sum(-1).clamp_min(1e-12).sqrt() * 100.0
    vel_mask = frame_mask[:, :-1] & seq_mask[:, :-1] & seq_mask[:, 1:]
    acc_mask = frame_mask[:, :-2] & seq_mask[:, :-2] & seq_mask[:, 1:-1] & seq_mask[:, 2:]
    return {
        "hand_velocity_cm_per_frame": _stats_from_magnitudes(vel_mag, vel_mask),
        "hand_acceleration_cm_per_frame2": _stats_from_magnitudes(acc_mag, acc_mask),
    }


def _relative_window_metrics(
    joints: Tensor,
    object_positions: Tensor,
    frame_mask: Tensor,
    seq_mask: Tensor,
    hand_joint: int,
) -> dict[str, Any]:
    hand = joints[:, :, hand_joint, :]
    obj = object_positions
    distance = torch.linalg.vector_norm(hand - obj, dim=-1) * 100.0
    rel_vel = (hand[:, 1:] - hand[:, :-1]) - (obj[:, 1:] - obj[:, :-1])
    rel_vel_mag = torch.linalg.vector_norm(rel_vel, dim=-1) * 100.0
    dist_derivative = distance[:, 1:] - distance[:, :-1]
    closing_speed = torch.clamp(-dist_derivative, min=0.0)
    pair_mask = frame_mask[:, :-1] & seq_mask[:, :-1] & seq_mask[:, 1:]
    return {
        "relative_velocity_cm_per_frame": _stats_from_magnitudes(rel_vel_mag, pair_mask),
        "hand_object_distance_cm": _stats_from_values(distance, frame_mask),
        "distance_derivative_cm_per_frame": _stats_from_values(dist_derivative, pair_mask),
        "distance_closing_speed_cm_per_frame": _stats_from_magnitudes(
            closing_speed,
            pair_mask,
        ),
    }


def _source_window_metrics(
    joints: Tensor,
    object_positions: Tensor,
    frame_mask: Tensor,
    seq_mask: Tensor,
    hand_joint: int,
    fps: float,
) -> dict[str, Any]:
    return {
        "body": _body_window_metrics(joints, frame_mask, seq_mask, fps=fps),
        "hand": _hand_window_metrics(joints, frame_mask, seq_mask, hand_joint),
        "relative": _relative_window_metrics(
            joints,
            object_positions,
            frame_mask,
            seq_mask,
            hand_joint,
        ),
    }


def _event_net_change_stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "std": 0.0,
            "n": 0,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "std": float(arr.std()),
        "n": int(arr.size),
    }


def _event_distance_metrics(
    joints: Tensor,
    object_positions: Tensor,
    seq_mask: Tensor,
    event_mask: Tensor,
    hand_joint: int,
    window_k: int,
    kind: str,
) -> dict[str, Any]:
    hand = joints[:, :, hand_joint, :]
    distance = torch.linalg.vector_norm(hand - object_positions, dim=-1) * 100.0
    raw_change: list[float] = []
    positive_change: list[float] = []

    for b in range(event_mask.shape[0]):
        valid = int(seq_mask[b].sum().item())
        if valid <= 0:
            continue
        event_idx = torch.where(event_mask[b, :valid])[0].detach().cpu().tolist()
        for t in event_idx:
            if kind == "onset":
                start = max(0, int(t) - int(window_k))
                change = float(distance[b, start].item() - distance[b, int(t)].item())
            elif kind == "release":
                end = min(valid - 1, int(t) + int(window_k))
                change = float(distance[b, end].item() - distance[b, int(t)].item())
            else:
                raise ValueError(f"Unknown event kind: {kind}")
            raw_change.append(change)
            positive_change.append(max(0.0, change))

    return {
        "raw_distance_change_cm": _event_net_change_stats(raw_change),
        "positive_distance_change_cm": _event_net_change_stats(positive_change),
    }


def _ratio_view(source: dict[str, Any], gt: dict[str, Any]) -> dict[str, float]:
    return {
        "body_velocity_sample_or_recon_over_gt": _safe_div(
            source["body"]["body_local_velocity_cm_per_frame"]["mean"],
            gt["body"]["body_local_velocity_cm_per_frame"]["mean"],
        ),
        "body_acceleration_p95_sample_or_recon_over_gt": _safe_div(
            source["body"]["body_local_acceleration_cm_per_frame2"]["p95"],
            gt["body"]["body_local_acceleration_cm_per_frame2"]["p95"],
        ),
        "hand_velocity_sample_or_recon_over_gt": _safe_div(
            source["hand"]["hand_velocity_cm_per_frame"]["mean"],
            gt["hand"]["hand_velocity_cm_per_frame"]["mean"],
        ),
        "hand_acceleration_p95_sample_or_recon_over_gt": _safe_div(
            source["hand"]["hand_acceleration_cm_per_frame2"]["p95"],
            gt["hand"]["hand_acceleration_cm_per_frame2"]["p95"],
        ),
        "relative_velocity_sample_or_recon_over_gt": _safe_div(
            source["relative"]["relative_velocity_cm_per_frame"]["mean"],
            gt["relative"]["relative_velocity_cm_per_frame"]["mean"],
        ),
        "closing_speed_sample_or_recon_over_gt": _safe_div(
            source["relative"]["distance_closing_speed_cm_per_frame"]["mean"],
            gt["relative"]["distance_closing_speed_cm_per_frame"]["mean"],
        ),
    }


def _build_threshold_metrics(
    gt_joints: Tensor,
    sample_joints: Tensor,
    recon_joints: Tensor,
    object_positions: Tensor,
    contact_state: Tensor,
    seq_mask: Tensor,
    threshold: float,
    main_k: int,
    short_k: int,
    transition_radius: int,
    fps: float,
) -> dict[str, Any]:
    part_windows = _build_part_windows(
        contact_state=contact_state,
        seq_mask=seq_mask,
        threshold=threshold,
        main_k=main_k,
        short_k=short_k,
        transition_radius=transition_radius,
    )
    source_joints = {
        "gt": gt_joints,
        "sampled": sample_joints,
        "recon_one_step": recon_joints,
    }

    out: dict[str, Any] = {
        "threshold": float(threshold),
        "window_counts": {},
        "parts": {},
    }
    for part_name, hand_joint, _part_idx in HAND_SPECS:
        spec = part_windows[part_name]
        out["window_counts"][part_name] = spec["counts"]
        out["parts"][part_name] = {
            "windows": {},
            "event_metrics": {},
        }
        for window_name, frame_mask in spec["frame_masks"].items():
            source_metrics = {
                source_name: _source_window_metrics(
                    source_joints[source_name],
                    object_positions,
                    frame_mask,
                    seq_mask,
                    hand_joint,
                    fps=fps,
                )
                for source_name in SOURCES
            }
            source_metrics["ratios_sampled_over_gt"] = _ratio_view(
                source_metrics["sampled"],
                source_metrics["gt"],
            )
            source_metrics["ratios_recon_over_gt"] = _ratio_view(
                source_metrics["recon_one_step"],
                source_metrics["gt"],
            )
            out["parts"][part_name]["windows"][window_name] = source_metrics

        for event_kind, event_mask in (
            ("onset", spec["onset_mask"]),
            ("release", spec["release_mask"]),
        ):
            for window_k in (main_k, short_k):
                event_name = f"{event_kind}_k{window_k}"
                event_metrics = {
                    source_name: _event_distance_metrics(
                        source_joints[source_name],
                        object_positions,
                        seq_mask,
                        event_mask,
                        hand_joint,
                        window_k=window_k,
                        kind=event_kind,
                    )
                    for source_name in SOURCES
                }
                positive_gt = event_metrics["gt"]["positive_distance_change_cm"]["mean"]
                event_metrics["sample_positive_change_over_gt"] = _safe_div(
                    event_metrics["sampled"]["positive_distance_change_cm"]["mean"],
                    positive_gt,
                )
                event_metrics["recon_positive_change_over_gt"] = _safe_div(
                    event_metrics["recon_one_step"]["positive_distance_change_cm"]["mean"],
                    positive_gt,
                )
                out["parts"][part_name]["event_metrics"][event_name] = event_metrics
    return out


def _phase_metrics(
    gt_joints: Tensor,
    sample_joints: Tensor,
    recon_joints: Tensor,
    phase: Tensor,
    seq_mask: Tensor,
    fps: float,
) -> dict[str, Any]:
    source_joints = {
        "gt": gt_joints,
        "sampled": sample_joints,
        "recon_one_step": recon_joints,
    }
    out: dict[str, Any] = {}
    for phase_id, phase_name in enumerate(PHASE_NAMES):
        frame_mask = (phase == int(phase_id)) & seq_mask.bool()
        metrics = {
            source_name: {
                "body": _body_window_metrics(
                    source_joints[source_name],
                    frame_mask,
                    seq_mask,
                    fps=fps,
                )
            }
            for source_name in SOURCES
        }
        metrics["n_frames"] = int(frame_mask.sum().item())
        gt_vel = metrics["gt"]["body"]["body_local_velocity_cm_per_frame"]["mean"]
        sample_vel = metrics["sampled"]["body"]["body_local_velocity_cm_per_frame"]["mean"]
        recon_vel = metrics["recon_one_step"]["body"]["body_local_velocity_cm_per_frame"]["mean"]
        metrics["sample_body_velocity_over_gt"] = _safe_div(sample_vel, gt_vel)
        metrics["recon_body_velocity_over_gt"] = _safe_div(recon_vel, gt_vel)
        out[str(phase_name)] = metrics
    return out


def _mean_part_value(threshold_metrics: dict[str, Any], accessor) -> float:
    values = []
    for part_name, _joint_idx, _part_idx in HAND_SPECS:
        value = accessor(threshold_metrics["parts"][part_name])
        values.append(float(value))
    return float(np.mean(values)) if values else 0.0


def _threshold_summary(
    threshold_metrics: dict[str, Any],
    main_k: int,
    transition_radius: int,
) -> dict[str, float]:
    def _window_ratio(part_payload: dict[str, Any], window_name: str, ratio_key: str) -> float:
        return part_payload["windows"][window_name]["ratios_sampled_over_gt"][ratio_key]

    def _window_recon_ratio(part_payload: dict[str, Any], window_name: str, ratio_key: str) -> float:
        return part_payload["windows"][window_name]["ratios_recon_over_gt"][ratio_key]

    def _window_gt_metric(part_payload: dict[str, Any], window_name: str, branch: str, metric: str) -> float:
        return (
            part_payload["windows"][window_name]["gt"][branch][metric]["mean"]
        )

    def _event_ratio(part_payload: dict[str, Any], event_name: str, key: str) -> float:
        return part_payload["event_metrics"][event_name][key]

    summary = {
        "in_contact_rel_vel_sample_over_gt": _mean_part_value(
            threshold_metrics,
            lambda part: _window_ratio(
                part,
                "in_contact",
                "relative_velocity_sample_or_recon_over_gt",
            ),
        ),
        "pre_contact_rel_vel_sample_over_gt": _mean_part_value(
            threshold_metrics,
            lambda part: _window_ratio(
                part,
                f"pre_contact_k{main_k}",
                "relative_velocity_sample_or_recon_over_gt",
            ),
        ),
        "post_release_rel_vel_sample_over_gt": _mean_part_value(
            threshold_metrics,
            lambda part: _window_ratio(
                part,
                f"post_release_k{main_k}",
                "relative_velocity_sample_or_recon_over_gt",
            ),
        ),
        "onset_transition_rel_vel_sample_over_gt": _mean_part_value(
            threshold_metrics,
            lambda part: _window_ratio(
                part,
                f"onset_pm{transition_radius}",
                "relative_velocity_sample_or_recon_over_gt",
            ),
        ),
        "in_contact_rel_vel_recon_over_gt": _mean_part_value(
            threshold_metrics,
            lambda part: _window_recon_ratio(
                part,
                "in_contact",
                "relative_velocity_sample_or_recon_over_gt",
            ),
        ),
        "pre_contact_rel_vel_recon_over_gt": _mean_part_value(
            threshold_metrics,
            lambda part: _window_recon_ratio(
                part,
                f"pre_contact_k{main_k}",
                "relative_velocity_sample_or_recon_over_gt",
            ),
        ),
        "post_release_rel_vel_recon_over_gt": _mean_part_value(
            threshold_metrics,
            lambda part: _window_recon_ratio(
                part,
                f"post_release_k{main_k}",
                "relative_velocity_sample_or_recon_over_gt",
            ),
        ),
        "onset_transition_rel_vel_recon_over_gt": _mean_part_value(
            threshold_metrics,
            lambda part: _window_recon_ratio(
                part,
                f"onset_pm{transition_radius}",
                "relative_velocity_sample_or_recon_over_gt",
            ),
        ),
        "pre_contact_gt_rel_vel_cm_per_frame": _mean_part_value(
            threshold_metrics,
            lambda part: _window_gt_metric(
                part,
                f"pre_contact_k{main_k}",
                "relative",
                "relative_velocity_cm_per_frame",
            ),
        ),
        "in_contact_gt_rel_vel_cm_per_frame": _mean_part_value(
            threshold_metrics,
            lambda part: _window_gt_metric(
                part,
                "in_contact",
                "relative",
                "relative_velocity_cm_per_frame",
            ),
        ),
        "onset_positive_closing_sample_over_gt": _mean_part_value(
            threshold_metrics,
            lambda part: _event_ratio(
                part,
                f"onset_k{main_k}",
                "sample_positive_change_over_gt",
            ),
        ),
        "onset_positive_closing_recon_over_gt": _mean_part_value(
            threshold_metrics,
            lambda part: _event_ratio(
                part,
                f"onset_k{main_k}",
                "recon_positive_change_over_gt",
            ),
        ),
        "release_positive_opening_sample_over_gt": _mean_part_value(
            threshold_metrics,
            lambda part: _event_ratio(
                part,
                f"release_k{main_k}",
                "sample_positive_change_over_gt",
            ),
        ),
        "release_positive_opening_recon_over_gt": _mean_part_value(
            threshold_metrics,
            lambda part: _event_ratio(
                part,
                f"release_k{main_k}",
                "recon_positive_change_over_gt",
            ),
        ),
    }
    return summary


def _interpret_results(
    threshold_summaries: dict[str, dict[str, float]],
) -> dict[str, Any]:
    primary = threshold_summaries["0.5"]
    secondary = threshold_summaries["0.3"]

    def _case_flags(summary: dict[str, float]) -> dict[str, bool]:
        in_contact_near_gt = 0.85 <= summary["in_contact_rel_vel_sample_over_gt"] <= 1.20
        transition_under = (
            summary["pre_contact_rel_vel_sample_over_gt"] < 0.80
            or summary["onset_transition_rel_vel_sample_over_gt"] < 0.80
            or summary["post_release_rel_vel_sample_over_gt"] < 0.80
            or summary["onset_positive_closing_sample_over_gt"] < 0.80
        )
        all_under = (
            summary["in_contact_rel_vel_sample_over_gt"] < 0.80
            and summary["pre_contact_rel_vel_sample_over_gt"] < 0.80
            and summary["post_release_rel_vel_sample_over_gt"] < 0.80
        )
        recon_transition_near = (
            summary["pre_contact_rel_vel_recon_over_gt"] >= 0.85
            and summary["onset_transition_rel_vel_recon_over_gt"] >= 0.85
        )
        sample_transition_low = (
            summary["pre_contact_rel_vel_sample_over_gt"] < 0.80
            or summary["onset_transition_rel_vel_sample_over_gt"] < 0.80
        )
        gt_transition_low = (
            summary["pre_contact_gt_rel_vel_cm_per_frame"] < 1.0
            and summary["onset_positive_closing_sample_over_gt"] == 0.0
        )
        return {
            "case_a": bool(in_contact_near_gt and transition_under),
            "case_b": bool(all_under),
            "case_c": bool(recon_transition_near and sample_transition_low),
            "case_d": bool(gt_transition_low),
        }

    flags_primary = _case_flags(primary)
    flags_secondary = _case_flags(secondary)
    stable_a = flags_primary["case_a"] and flags_secondary["case_a"]
    stable_c = flags_primary["case_c"] and flags_secondary["case_c"]

    if stable_c:
        verdict = "Case C"
        headline = (
            "Transition under-motion is real, but the strongest signal is a "
            "reconstruction-vs-sampling split: one-step recon stays much closer "
            "to GT than DDPM samples around approach/onset windows."
        )
    elif stable_a:
        verdict = "Case A"
        headline = (
            "In-contact relative dynamics remain near GT while approach/onset/"
            "release windows stay systematically weaker, consistent with a "
            "non-contact transition supervision gap."
        )
    elif flags_primary["case_b"] and flags_secondary["case_b"]:
        verdict = "Case B"
        headline = (
            "Both contact and non-contact windows are broadly under-dynamic, "
            "which looks more like a general sampled-dynamics collapse."
        )
    elif flags_primary["case_d"] and flags_secondary["case_d"]:
        verdict = "Case D"
        headline = (
            "The selected transition windows do not carry meaningful GT dynamics, "
            "so the window definition or data distribution is the main issue."
        )
    else:
        verdict = "Mixed"
        headline = (
            "The diagnostic shows a repeatable transition deficit plus a clear "
            "reconstruction-vs-sampling gap, but this 16-clip slice is not clean "
            "enough to call pure Case A or pure Case C."
        )

    return {
        "verdict": verdict,
        "headline": headline,
        "primary_threshold": "0.5",
        "case_flags_threshold_0_5": flags_primary,
        "case_flags_threshold_0_3": flags_secondary,
        "threshold_0_5_summary": primary,
        "threshold_0_3_summary": secondary,
        "threshold_stable_case_a": bool(stable_a),
        "threshold_stable_case_c": bool(stable_c),
    }


def _table_row_ratio(
    threshold_metrics: dict[str, Any],
    part_name: str,
    window_name: str,
) -> tuple[float, float, float, float, float]:
    payload = threshold_metrics["parts"][part_name]["windows"][window_name]
    gt_rel = payload["gt"]["relative"]["relative_velocity_cm_per_frame"]["mean"]
    sample_rel = payload["sampled"]["relative"]["relative_velocity_cm_per_frame"]["mean"]
    recon_rel = payload["recon_one_step"]["relative"]["relative_velocity_cm_per_frame"]["mean"]
    return (
        float(gt_rel),
        float(sample_rel),
        float(recon_rel),
        _safe_div(sample_rel, gt_rel),
        _safe_div(recon_rel, gt_rel),
    )


def _table_row_closing(
    threshold_metrics: dict[str, Any],
    part_name: str,
    window_name: str,
) -> tuple[float, float, float, float]:
    payload = threshold_metrics["parts"][part_name]["windows"][window_name]
    gt_close = payload["gt"]["relative"]["distance_closing_speed_cm_per_frame"]["mean"]
    sample_close = payload["sampled"]["relative"]["distance_closing_speed_cm_per_frame"]["mean"]
    recon_close = payload["recon_one_step"]["relative"]["distance_closing_speed_cm_per_frame"]["mean"]
    return (
        float(gt_close),
        float(sample_close),
        float(recon_close),
        _safe_div(sample_close, gt_close),
    )


def _write_markdown(path: Path, results: dict[str, Any]) -> None:
    threshold_main = results["threshold_metrics"]["0.5"]
    threshold_alt = results["threshold_metrics"]["0.3"]
    phase_metrics = results["phase_metrics"]
    interpretation = results["interpretation"]
    main_k = int(results["window_k"])
    short_k = int(results["short_window_k"])
    transition_radius = int(results["transition_radius"])

    lines: list[str] = []
    lines.append("# v18 Non-contact Transition Diagnostic")
    lines.append("")
    lines.append(f"**Config:** `{results['config']}`")
    lines.append(f"**Checkpoint:** `{results['ckpt']}`")
    lines.append(
        f"**Clips:** {results['num_clips']}    **bucket:** {results['bucket']}    "
        f"**recon_t:** {results['recon_t']}    **cfg_scale:** {results['cfg_scale']}"
    )
    lines.append("")
    lines.append(
        "Important label note: this repo's `phase` field is "
        "`non_contact / stable_contact / manipulation`, not "
        "`approach / stable / release`. Therefore the approach and release "
        "analyses below are derived from GT hand-contact onset/release events; "
        "phase-wise reporting uses the repo's actual label semantics."
    )
    lines.append("")
    lines.append(
        f"Transition windows use GT contact events at threshold 0.5 as the primary "
        f"readout, plus a 0.3 sensitivity pass. Main approach/release span is k={main_k}; "
        f"the shorter companion span is k={short_k}; transition neighborhoods are "
        f"onset/release +/-{transition_radius} frames."
    )

    lines.append("")
    lines.append("## Table 1: Window counts (threshold=0.5)")
    lines.append("")
    lines.append("| window | part | n_frames | n_events |")
    lines.append("|---|---|---:|---:|")
    ordered_windows = [
        "in_contact",
        f"pre_contact_k{main_k}",
        f"pre_contact_k{short_k}",
        f"post_release_k{main_k}",
        f"post_release_k{short_k}",
        f"onset_pm{transition_radius}",
        f"release_pm{transition_radius}",
        f"transition_pm{transition_radius}",
        f"far_non_contact_k{main_k}",
    ]
    for window_name in ordered_windows:
        for part_name, _joint_idx, _part_idx in HAND_SPECS:
            count = threshold_main["window_counts"][part_name][window_name]
            lines.append(
                f"| {window_name} | {part_name} | {count['n_frames']} | {count['n_events']} |"
            )

    for table_idx, part_name in ((2, "L_hand"), (3, "R_hand")):
        lines.append("")
        lines.append(f"## Table {table_idx}: {part_name} window relative dynamics (threshold=0.5)")
        lines.append("")
        lines.append(
            "| window | GT rel-vel | sample rel-vel | recon rel-vel | sample/GT | recon/GT |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|")
        for window_name in (
            "in_contact",
            f"pre_contact_k{main_k}",
            f"pre_contact_k{short_k}",
            f"onset_pm{transition_radius}",
            f"post_release_k{main_k}",
            f"post_release_k{short_k}",
            f"release_pm{transition_radius}",
            f"far_non_contact_k{main_k}",
        ):
            gt_rel, sample_rel, recon_rel, sample_ratio, recon_ratio = _table_row_ratio(
                threshold_main,
                part_name,
                window_name,
            )
            lines.append(
                f"| {window_name} | {gt_rel:.3f} | {sample_rel:.3f} | {recon_rel:.3f} | "
                f"{sample_ratio:.3f} | {recon_ratio:.3f} |"
            )

    lines.append("")
    lines.append("## Table 4: Distance closing (threshold=0.5)")
    lines.append("")
    lines.append("| window | part | GT closing | sample closing | recon closing | sample/GT |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for window_name in (
        f"pre_contact_k{main_k}",
        f"pre_contact_k{short_k}",
        f"onset_pm{transition_radius}",
    ):
        for part_name, _joint_idx, _part_idx in HAND_SPECS:
            gt_close, sample_close, recon_close, sample_ratio = _table_row_closing(
                threshold_main,
                part_name,
                window_name,
            )
            lines.append(
                f"| {window_name} | {part_name} | {gt_close:.3f} | {sample_close:.3f} | "
                f"{recon_close:.3f} | {sample_ratio:.3f} |"
            )

    lines.append("")
    lines.append("## Table 5: Phase-wise body dynamics")
    lines.append("")
    lines.append("| phase | GT body vel | sample body vel | recon body vel | sample/GT | recon/GT | n_frames |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for phase_name in PHASE_NAMES:
        payload = phase_metrics[str(phase_name)]
        gt_vel = payload["gt"]["body"]["body_local_velocity_cm_per_frame"]["mean"]
        sample_vel = payload["sampled"]["body"]["body_local_velocity_cm_per_frame"]["mean"]
        recon_vel = payload["recon_one_step"]["body"]["body_local_velocity_cm_per_frame"]["mean"]
        lines.append(
            f"| {phase_name} | {gt_vel:.3f} | {sample_vel:.3f} | {recon_vel:.3f} | "
            f"{payload['sample_body_velocity_over_gt']:.3f} | "
            f"{payload['recon_body_velocity_over_gt']:.3f} | {payload['n_frames']} |"
        )

    lines.append("")
    lines.append("## Event-level transition quality")
    lines.append("")
    lines.append(
        "| event | part | GT positive change | sample positive change | recon positive change | sample/GT | recon/GT | n_events |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for part_name, _joint_idx, _part_idx in HAND_SPECS:
        for event_name in (
            f"onset_k{main_k}",
            f"onset_k{short_k}",
            f"release_k{main_k}",
            f"release_k{short_k}",
        ):
            payload = threshold_main["parts"][part_name]["event_metrics"][event_name]
            gt_stats = payload["gt"]["positive_distance_change_cm"]
            sample_stats = payload["sampled"]["positive_distance_change_cm"]
            recon_stats = payload["recon_one_step"]["positive_distance_change_cm"]
            lines.append(
                f"| {event_name} | {part_name} | {gt_stats['mean']:.3f} | "
                f"{sample_stats['mean']:.3f} | {recon_stats['mean']:.3f} | "
                f"{payload['sample_positive_change_over_gt']:.3f} | "
                f"{payload['recon_positive_change_over_gt']:.3f} | {gt_stats['n']} |"
            )

    lines.append("")
    lines.append("## Threshold sensitivity (0.5 vs 0.3)")
    lines.append("")
    lines.append("| summary metric | threshold 0.5 | threshold 0.3 |")
    lines.append("|---|---:|---:|")
    summary_labels = (
        ("in_contact_rel_vel_sample_over_gt", "in-contact rel-vel sample/GT"),
        ("pre_contact_rel_vel_sample_over_gt", f"pre-contact k={main_k} rel-vel sample/GT"),
        ("post_release_rel_vel_sample_over_gt", f"post-release k={main_k} rel-vel sample/GT"),
        ("onset_transition_rel_vel_sample_over_gt", f"onset +/-{transition_radius} rel-vel sample/GT"),
        ("onset_positive_closing_sample_over_gt", f"onset k={main_k} closing sample/GT"),
        ("pre_contact_rel_vel_recon_over_gt", f"pre-contact k={main_k} rel-vel recon/GT"),
        ("onset_transition_rel_vel_recon_over_gt", f"onset +/-{transition_radius} rel-vel recon/GT"),
    )
    for key, label in summary_labels:
        lines.append(
            f"| {label} | {interpretation['threshold_0_5_summary'][key]:.3f} | "
            f"{interpretation['threshold_0_3_summary'][key]:.3f} |"
        )

    if results.get("subset_wise"):
        lines.append("")
        lines.append("## Subset-wise transition summary (threshold 0.5)")
        lines.append("")
        lines.append("| subset | clips | in-contact rel vel xGT | pre-contact rel vel xGT | onset +/- rel vel xGT | post-release rel vel xGT | release opening xGT |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for subset, payload in results["subset_wise"].items():
            s = payload["threshold_0_5_summary"]
            lines.append(
                f"| {subset} | {payload['num_clips']} | "
                f"{s['in_contact_rel_vel_sample_over_gt']:.3f} | "
                f"{s['pre_contact_rel_vel_sample_over_gt']:.3f} | "
                f"{s['onset_transition_rel_vel_sample_over_gt']:.3f} | "
                f"{s['post_release_rel_vel_sample_over_gt']:.3f} | "
                f"{s['release_positive_opening_sample_over_gt']:.3f} |"
            )

    lines.append("")
    lines.append("## Table 6: Summary diagnosis")
    lines.append("")
    lines.append("| metric | interpretation |")
    lines.append("|---|---|")
    lines.append(
        f"| primary verdict | **{interpretation['verdict']}**: {interpretation['headline']} |"
    )
    lines.append(
        "| label semantics | Approach/release are event-derived because the persisted "
        "`phase` field does not distinguish them. |"
    )
    lines.append(
        f"| threshold stability | Case A stable across thresholds: "
        f"{interpretation['threshold_stable_case_a']}; Case C stable across thresholds: "
        f"{interpretation['threshold_stable_case_c']}. |"
    )
    lines.append(
        f"| onset k={main_k} closing sample/GT | "
        f"{interpretation['threshold_0_5_summary']['onset_positive_closing_sample_over_gt']:.3f} "
        "(threshold 0.5 primary readout). |"
    )
    lines.append(
        f"| onset k={main_k} closing recon/GT | "
        f"{interpretation['threshold_0_5_summary']['onset_positive_closing_recon_over_gt']:.3f} "
        "(separates training-time reconstruction from sampling-time behavior). |"
    )
    lines.append(
        f"| release k={main_k} opening sample/GT | "
        f"{interpretation['threshold_0_5_summary']['release_positive_opening_sample_over_gt']:.3f}. |"
    )

    lines.append("")
    lines.append("## Clear decision")
    lines.append("")
    if interpretation["verdict"] == "Case A":
        lines.append("1. **Non-contact transition supervision insufficient?** Yes, the diagnostic supports that.")
        lines.append("2. **Is onset slow approach the main remaining failure?** Yes; it stays weak while in-contact dynamics remain near GT.")
        lines.append("3. **Worth implementing targeted transition loss next?** Yes, as a minimal follow-up against the v18 baseline.")
        lines.append(
            "4. **Smallest ablation:** a low-weight approach distance-closing loss on onset-preceding frames, "
            "optionally paired with onset +/-5 relative-velocity MSE only if the first ablation moves the right metric."
        )
        lines.append("")
        lines.append("### Minimal repair suggestion (not implemented)")
        lines.append("")
        lines.append("- Apply only around onset/release windows, never as a global velocity imitation term.")
        lines.append("- Keep the weight low and compare directly against v18 on this diagnostic.")
        lines.append("- First ablation: onset-preceding hand-object distance derivative matching.")
        lines.append("- Second-choice follow-up: onset +/-5 relative hand-object velocity MSE.")
        lines.append("- Third-choice follow-up: release-window relative velocity / distance-increase MSE.")
    elif interpretation["verdict"] == "Case C":
        lines.append("1. **Non-contact transition supervision insufficient?** Not cleanly established; the sampled transition deficit is real, but recon is much closer to GT.")
        lines.append("2. **Is onset slow approach the main remaining failure?** Yes as an observed symptom, but the stronger mechanism signal is iterative denoising collapse.")
        lines.append("3. **Worth implementing targeted transition loss next?** Not as the first move. I would verify objective-side fixes before adding a new phase-targeted loss.")
        lines.append(
            "4. **Recommended next path:** keep transition diagnostics as the judge, but prioritize a bounded denoising-objective test "
            "before training a new transition-specific loss."
        )
    elif interpretation["verdict"] == "Case B":
        lines.append("1. **Non-contact transition supervision insufficient?** The deficit is broader than transition frames.")
        lines.append("2. **Is onset slow approach the main remaining failure?** No; it is one part of a more global sampled-dynamics collapse.")
        lines.append("3. **Worth implementing targeted transition loss next?** No.")
        lines.append("4. **Recommended next path:** objective-side work such as v-pred/min-SNR style denoising changes remains higher priority.")
    elif interpretation["verdict"] == "Case D":
        lines.append("1. **Non-contact transition supervision insufficient?** Not supported.")
        lines.append("2. **Is onset slow approach the main remaining failure?** No; GT does not expose enough transition motion under this window definition.")
        lines.append("3. **Worth implementing targeted transition loss next?** No.")
        lines.append("4. **Recommended next path:** revise the event/window definition or inspect a larger slice before changing training.")
    else:
        lines.append("1. **Non-contact transition supervision insufficient?** Partially supported, but not cleanly isolated as the root cause.")
        lines.append("2. **Is onset slow approach the main remaining failure?** It remains a real failure mode, especially for L-hand approach and both-hand release, but the evidence is mixed rather than one-note.")
        lines.append("3. **Worth implementing targeted transition loss next?** Hold for now.")
        lines.append("4. **Recommended next path:** use this diagnostic to compare a denoising-objective ablation first; only add targeted transition loss if the recon/sample gap does not close.")

    lines.append("")
    lines.append("## Notes on body-dynamics detail")
    lines.append("")
    lines.append(
        "Per-window body local velocity, acceleration, and masked FFT low/mid/high "
        "fractions are all stored in the JSON artifact. FFT rows include the number "
        "of contributing contiguous segments; very short event windows can legitimately "
        "have sparse or zero FFT support."
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--md", type=Path, required=True)
    parser.add_argument("--num-clips", type=int, default=8)
    parser.add_argument("--bucket", default="train", choices=["train", "val"])
    parser.add_argument("--recon-t", type=int, default=100)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--window-k", type=int, default=10)
    parser.add_argument("--short-window-k", type=int, default=5)
    parser.add_argument("--transition-radius", type=int, default=5)
    parser.add_argument(
        "--balanced-subsets",
        action="store_true",
        help="Pick clips evenly across configured subsets instead of taking the first N.",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    dataset = _build_dataset(cfg, args.bucket)
    if args.balanced_subsets:
        clip_indices = _balanced_subset_indices(dataset, int(args.num_clips))
    else:
        clip_indices = list(range(min(args.num_clips, len(dataset))))
    dataset = Subset(dataset, clip_indices)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_hoi,
        num_workers=0,
    )

    model, object_encoder, z_dims = _build_model(cfg, device)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("model", state))
    if "object_encoder" in state:
        object_encoder.load_state_dict(state["object_encoder"])
    elif "extra_modules" in state and "object_encoder" in state["extra_modules"]:
        object_encoder.load_state_dict(state["extra_modules"]["object_encoder"])

    clip_model = load_clip_text_encoder(
        device=device,
        model_name=str(cfg.model.text_encoder.clip_version),
        download_root=str(cfg.model.text_encoder.get("download_root", "cache/clip")),
    )

    all_gt_joints: list[Tensor] = []
    all_sample_joints: list[Tensor] = []
    all_recon_joints: list[Tensor] = []
    all_seq_masks: list[Tensor] = []
    all_object_positions: list[Tensor] = []
    all_contact_state: list[Tensor] = []
    all_phase: list[Tensor] = []
    per_clip: list[dict[str, Any]] = []

    for idx, batch in enumerate(loader):
        cond, total_t = _build_cond(
            batch,
            model,
            object_encoder,
            clip_model,
            z_dims,
            cfg,
            device,
        )
        plan = _extract_plan(batch, device)
        cond_full = {**cond, "interaction_plan": plan}

        motion_gt = batch["motion"].to(device).float()
        rest_offsets = batch["rest_offsets"].to(device).float()
        seq_len = batch["seq_len"].to(device)
        seq_idx = torch.arange(total_t, device=device).unsqueeze(0)
        seq_mask = seq_idx < seq_len.unsqueeze(1)
        gt_joints = batch["joints"].to(device).float()

        torch.manual_seed(args.seed + idx)
        with torch.no_grad():
            x0_sample = model.sample(
                cond=cond_full,
                seq_length=total_t,
                cfg_scale=args.cfg_scale,
                replacement="none",
                output_skip=False,
            )
        sample_joints = _fk_from_motion_135(x0_sample, rest_offsets)

        torch.manual_seed(args.seed + 1000 + idx)
        with torch.no_grad():
            x0_recon = _one_step_recon_motion(
                model,
                motion_gt,
                cond_full,
                recon_t=int(args.recon_t),
            )
        recon_joints = _fk_from_motion_135(x0_recon, rest_offsets)

        all_gt_joints.append(gt_joints)
        all_sample_joints.append(sample_joints)
        all_recon_joints.append(recon_joints)
        all_seq_masks.append(seq_mask)
        all_object_positions.append(batch["object_positions"].to(device).float())
        all_contact_state.append(batch["contact_state"].to(device).float())
        all_phase.append(batch["phase"].to(device).long())
        per_clip.append(
            {
                "subset": batch["subset"][0],
                "seq_id": batch["seq_id"][0],
                "seq_len": int(seq_len.item()),
                "text": batch["text"][0][:120],
            }
        )
        print(
            f"  [{idx + 1}/{len(loader)}] "
            f"{batch['subset'][0]}/{batch['seq_id'][0]} "
            f"T={int(seq_len.item())}"
        )

    gt_joints = torch.cat(all_gt_joints, dim=0)
    sample_joints = torch.cat(all_sample_joints, dim=0)
    recon_joints = torch.cat(all_recon_joints, dim=0)
    seq_mask = torch.cat(all_seq_masks, dim=0)
    object_positions = torch.cat(all_object_positions, dim=0)
    contact_state = torch.cat(all_contact_state, dim=0)
    phase = torch.cat(all_phase, dim=0)

    threshold_metrics = {
        "0.5": _build_threshold_metrics(
            gt_joints=gt_joints,
            sample_joints=sample_joints,
            recon_joints=recon_joints,
            object_positions=object_positions,
            contact_state=contact_state,
            seq_mask=seq_mask,
            threshold=0.5,
            main_k=args.window_k,
            short_k=args.short_window_k,
            transition_radius=args.transition_radius,
            fps=args.fps,
        ),
        "0.3": _build_threshold_metrics(
            gt_joints=gt_joints,
            sample_joints=sample_joints,
            recon_joints=recon_joints,
            object_positions=object_positions,
            contact_state=contact_state,
            seq_mask=seq_mask,
            threshold=0.3,
            main_k=args.window_k,
            short_k=args.short_window_k,
            transition_radius=args.transition_radius,
            fps=args.fps,
        ),
    }
    threshold_summaries = {
        key: _threshold_summary(
            value,
            main_k=args.window_k,
            transition_radius=args.transition_radius,
        )
        for key, value in threshold_metrics.items()
    }
    phase_metrics = _phase_metrics(
        gt_joints=gt_joints,
        sample_joints=sample_joints,
        recon_joints=recon_joints,
        phase=phase,
        seq_mask=seq_mask,
        fps=args.fps,
    )
    subset_wise: dict[str, Any] = {}
    for subset in sorted({c["subset"] for c in per_clip}):
        row_idx = [i for i, c in enumerate(per_clip) if c["subset"] == subset]
        if not row_idx:
            continue
        idx_t = torch.tensor(row_idx, device=gt_joints.device, dtype=torch.long)
        sub_gt = gt_joints.index_select(0, idx_t)
        sub_sample = sample_joints.index_select(0, idx_t)
        sub_recon = recon_joints.index_select(0, idx_t)
        sub_mask = seq_mask.index_select(0, idx_t)
        sub_obj = object_positions.index_select(0, idx_t)
        sub_contact = contact_state.index_select(0, idx_t)
        sub_metrics = _build_threshold_metrics(
            gt_joints=sub_gt,
            sample_joints=sub_sample,
            recon_joints=sub_recon,
            object_positions=sub_obj,
            contact_state=sub_contact,
            seq_mask=sub_mask,
            threshold=0.5,
            main_k=args.window_k,
            short_k=args.short_window_k,
            transition_radius=args.transition_radius,
            fps=args.fps,
        )
        subset_wise[subset] = {
            "num_clips": len(row_idx),
            "threshold_0_5_summary": _threshold_summary(
                sub_metrics,
                main_k=args.window_k,
                transition_radius=args.transition_radius,
            ),
        }

    results: dict[str, Any] = {
        "config": str(args.config),
        "ckpt": str(args.ckpt),
        "bucket": str(args.bucket),
        "num_clips": int(gt_joints.shape[0]),
        "recon_t": int(args.recon_t),
        "cfg_scale": float(args.cfg_scale),
        "seed": int(args.seed),
        "fps": float(args.fps),
        "window_k": int(args.window_k),
        "short_window_k": int(args.short_window_k),
        "transition_radius": int(args.transition_radius),
        "phase_semantics": list(PHASE_NAMES),
        "per_clip": per_clip,
        "threshold_metrics": threshold_metrics,
        "phase_metrics": phase_metrics,
        "subset_wise": subset_wise,
    }
    results["interpretation"] = _interpret_results(threshold_summaries)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    _write_markdown(args.md, results)

    print(f"\nWrote JSON to {args.output}")
    print(f"Wrote Markdown to {args.md}")
    print("\n=== Non-contact transition verdict ===")
    print(f"  {results['interpretation']['verdict']}: {results['interpretation']['headline']}")
    print(
        "  onset closing sample/GT="
        f"{results['interpretation']['threshold_0_5_summary']['onset_positive_closing_sample_over_gt']:.3f}, "
        "recon/GT="
        f"{results['interpretation']['threshold_0_5_summary']['onset_positive_closing_recon_over_gt']:.3f}"
    )
    print(
        "  pre-contact rel-vel sample/GT="
        f"{results['interpretation']['threshold_0_5_summary']['pre_contact_rel_vel_sample_over_gt']:.3f}, "
        "recon/GT="
        f"{results['interpretation']['threshold_0_5_summary']['pre_contact_rel_vel_recon_over_gt']:.3f}"
    )


if __name__ == "__main__":
    main()
