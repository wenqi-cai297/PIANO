"""Render Stage-B GT vs one-step recon vs DDPM sample videos.

This visual diagnostic exports side-by-side MP4s for selected contact
transition clips:

    GT | Recon_t100 | DDPM_sample

It is intentionally offline-only. It reuses the Stage-B diagnostic model/data
builders and FK decoder; no training or sampling pipeline code is modified.
"""
from __future__ import annotations

import argparse
import json
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from dynamics_diagnostic import (
    LOCAL_JOINTS,
    _balanced_subset_indices,
    _build_cond,
    _build_dataset,
    _build_model,
    _fk_from_motion_135,
    _joint_velocity_acceleration,
)
from piano.data.dataset import collate_hoi
from piano.inference.visualize_motion import SKELETON_CONNECTIONS, _axis_angle_to_rotmat
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

HAND_SPECS = {
    "L_hand": {"joint": 20, "contact_idx": 0},
    "R_hand": {"joint": 21, "contact_idx": 1},
}


@dataclass
class EventRecord:
    kind: str
    part: str
    frame: int
    crop_start: int
    crop_end: int
    metrics: dict[str, Any]
    score: float


@dataclass
class ClipRecord:
    index: int
    subset: str
    seq_id: str
    text: str
    seq_len: int
    gt_motion: np.ndarray
    recon_motion: np.ndarray
    sample_motion: np.ndarray
    gt_joints: np.ndarray
    recon_joints: np.ndarray
    sample_joints: np.ndarray
    object_positions: np.ndarray
    object_rotations: np.ndarray | None
    object_pc: np.ndarray | None
    contact_state: np.ndarray
    body_sample_over_gt: float
    body_recon_over_gt: float
    events: list[EventRecord] = field(default_factory=list)
    selected_reason: str = ""


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if abs(float(den)) > 1e-12 else 0.0


def _extract_plan(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: batch[f"plan_{key}"].to(device) for key in PLAN_KEYS}


def _load_checkpoint(
    model: torch.nn.Module,
    object_encoder: torch.nn.Module,
    ckpt: Path,
) -> None:
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("model", state))
    if "object_encoder" in state:
        object_encoder.load_state_dict(state["object_encoder"])
    elif "extra_modules" in state and "object_encoder" in state["extra_modules"]:
        object_encoder.load_state_dict(state["extra_modules"]["object_encoder"])


def _body_local_velocity_ratio(
    pred_joints: torch.Tensor,
    gt_joints: torch.Tensor,
    seq_mask: torch.Tensor,
) -> float:
    _, pred_vel_local, _, _ = _joint_velocity_acceleration(pred_joints, seq_mask)
    _, gt_vel_local, _, _ = _joint_velocity_acceleration(gt_joints, seq_mask)
    mask = (seq_mask[:, 1:] & seq_mask[:, :-1]).unsqueeze(-1)
    pred_mag = torch.linalg.vector_norm(pred_vel_local[:, :, LOCAL_JOINTS], dim=-1)
    gt_mag = torch.linalg.vector_norm(gt_vel_local[:, :, LOCAL_JOINTS], dim=-1)
    pred_mean = float(pred_mag[mask.expand_as(pred_mag)].mean().item()) if mask.any() else 0.0
    gt_mean = float(gt_mag[mask.expand_as(gt_mag)].mean().item()) if mask.any() else 0.0
    return _safe_div(pred_mean, gt_mean)


def _event_frames(contact: np.ndarray, threshold: float) -> tuple[list[int], list[int]]:
    active = contact > float(threshold)
    onset: list[int] = []
    release: list[int] = []
    for t in range(1, len(active)):
        if active[t] and not active[t - 1]:
            onset.append(t)
        elif (not active[t]) and active[t - 1]:
            release.append(t)
    return onset, release


def _window_rel_velocity(
    joints: np.ndarray,
    object_positions: np.ndarray,
    hand_joint: int,
    start: int,
    end: int,
) -> float:
    lo = max(0, int(start))
    hi = min(int(end), len(joints) - 1)
    if hi <= lo:
        return 0.0
    hand_vel = joints[lo + 1 : hi + 1, hand_joint] - joints[lo:hi, hand_joint]
    obj_vel = object_positions[lo + 1 : hi + 1] - object_positions[lo:hi]
    rel = hand_vel - obj_vel
    return float(np.linalg.norm(rel, axis=-1).mean() * 100.0)


def _event_distance_change(
    joints: np.ndarray,
    object_positions: np.ndarray,
    hand_joint: int,
    event_frame: int,
    k: int,
    kind: str,
) -> float:
    distance = np.linalg.norm(joints[:, hand_joint] - object_positions, axis=-1) * 100.0
    t = int(event_frame)
    if kind == "onset":
        start = max(0, t - int(k))
        return float(distance[start] - distance[t])
    if kind == "release":
        end = min(len(distance) - 1, t + int(k))
        return float(distance[end] - distance[t])
    raise ValueError(f"Unknown event kind: {kind}")


def _event_metrics(
    gt_joints: np.ndarray,
    recon_joints: np.ndarray,
    sample_joints: np.ndarray,
    object_positions: np.ndarray,
    hand_joint: int,
    event_frame: int,
    kind: str,
    window_k: int,
    transition_radius: int,
) -> dict[str, Any]:
    t = int(event_frame)
    if kind == "onset":
        rel_start = max(0, t - int(window_k))
        rel_end = min(len(gt_joints) - 1, t + int(transition_radius))
        change_name = "closing"
    else:
        rel_start = max(0, t - int(transition_radius))
        rel_end = min(len(gt_joints) - 1, t + int(window_k))
        change_name = "opening"

    rel = {
        "gt": _window_rel_velocity(gt_joints, object_positions, hand_joint, rel_start, rel_end),
        "recon": _window_rel_velocity(recon_joints, object_positions, hand_joint, rel_start, rel_end),
        "sample": _window_rel_velocity(sample_joints, object_positions, hand_joint, rel_start, rel_end),
    }
    change = {
        "gt": _event_distance_change(gt_joints, object_positions, hand_joint, t, window_k, kind),
        "recon": _event_distance_change(recon_joints, object_positions, hand_joint, t, window_k, kind),
        "sample": _event_distance_change(sample_joints, object_positions, hand_joint, t, window_k, kind),
    }
    positive_change = {key: max(0.0, value) for key, value in change.items()}
    return {
        "kind": kind,
        "event_frame": int(t),
        "rel_window": [int(rel_start), int(rel_end)],
        "distance_change_name": change_name,
        "relative_velocity_cm_per_frame": rel,
        "relative_velocity_recon_over_gt": _safe_div(rel["recon"], rel["gt"]),
        "relative_velocity_sample_over_gt": _safe_div(rel["sample"], rel["gt"]),
        "distance_change_cm": change,
        "positive_distance_change_cm": positive_change,
        "positive_distance_change_recon_over_gt": _safe_div(
            positive_change["recon"],
            positive_change["gt"],
        ),
        "positive_distance_change_sample_over_gt": _safe_div(
            positive_change["sample"],
            positive_change["gt"],
        ),
    }


def _build_clip_events(
    gt_joints: np.ndarray,
    recon_joints: np.ndarray,
    sample_joints: np.ndarray,
    object_positions: np.ndarray,
    contact_state: np.ndarray,
    seq_len: int,
    threshold: float,
    window_k: int,
    transition_radius: int,
) -> list[EventRecord]:
    out: list[EventRecord] = []
    for part, spec in HAND_SPECS.items():
        contact = contact_state[:seq_len, int(spec["contact_idx"])]
        onsets, releases = _event_frames(contact, threshold=threshold)
        for kind, frames in (("onset", onsets), ("release", releases)):
            for frame in frames:
                if kind == "onset":
                    crop_start = max(0, frame - 15)
                    crop_end = min(seq_len - 1, frame + 10)
                else:
                    crop_start = max(0, frame - 10)
                    crop_end = min(seq_len - 1, frame + 15)
                if crop_end - crop_start < 8:
                    continue
                metrics = _event_metrics(
                    gt_joints=gt_joints[:seq_len],
                    recon_joints=recon_joints[:seq_len],
                    sample_joints=sample_joints[:seq_len],
                    object_positions=object_positions[:seq_len],
                    hand_joint=int(spec["joint"]),
                    event_frame=frame,
                    kind=kind,
                    window_k=window_k,
                    transition_radius=transition_radius,
                )
                rel_gap = (
                    metrics["relative_velocity_recon_over_gt"]
                    - metrics["relative_velocity_sample_over_gt"]
                )
                change_gap = (
                    metrics["positive_distance_change_recon_over_gt"]
                    - metrics["positive_distance_change_sample_over_gt"]
                )
                gt_motion_amount = (
                    metrics["relative_velocity_cm_per_frame"]["gt"]
                    + abs(metrics["positive_distance_change_cm"]["gt"])
                )
                score = float(rel_gap + 0.5 * change_gap + 0.01 * gt_motion_amount)
                out.append(
                    EventRecord(
                        kind=kind,
                        part=part,
                        frame=int(frame),
                        crop_start=int(crop_start),
                        crop_end=int(crop_end),
                        metrics=metrics,
                        score=score,
                    )
                )
    return sorted(out, key=lambda ev: ev.score, reverse=True)


def _run_clip(
    idx: int,
    batch: dict[str, Any],
    model,
    object_encoder,
    clip_model,
    z_dims,
    cfg,
    device: torch.device,
    recon_t: int,
    cfg_scale: float,
    seed: int,
    threshold: float,
    window_k: int,
    transition_radius: int,
) -> ClipRecord:
    cond, total_t = _build_cond(batch, model, object_encoder, clip_model, z_dims, cfg, device)
    plan = _extract_plan(batch, device)
    cond_full = {**cond, "interaction_plan": plan}

    motion_gt = batch["motion"].to(device).float()
    rest_offsets = batch["rest_offsets"].to(device).float()
    seq_len_t = batch["seq_len"].to(device)
    seq_len = int(seq_len_t.item())
    seq_mask = torch.arange(total_t, device=device).unsqueeze(0) < seq_len_t.unsqueeze(1)
    gt_joints_t = batch["joints"].to(device).float()

    torch.manual_seed(int(seed) + idx)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed) + idx)
    with torch.no_grad():
        sample_motion_t = model.sample(
            cond=cond_full,
            seq_length=total_t,
            cfg_scale=float(cfg_scale),
            replacement="none",
            output_skip=False,
        )
    sample_joints_t = _fk_from_motion_135(sample_motion_t, rest_offsets)

    torch.manual_seed(int(seed) + 1000 + idx)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed) + 1000 + idx)
    with torch.no_grad():
        t_recon = torch.full((1,), int(recon_t), dtype=torch.long, device=device)
        noise = torch.randn_like(motion_gt)
        x_t = model.diffusion.q_sample(motion_gt, t_recon, noise)
        recon_motion_t = model.denoiser(x_t, t_recon, cond_full, cond_drop_mask=None)
    recon_joints_t = _fk_from_motion_135(recon_motion_t, rest_offsets)

    body_sample_over_gt = _body_local_velocity_ratio(sample_joints_t, gt_joints_t, seq_mask)
    body_recon_over_gt = _body_local_velocity_ratio(recon_joints_t, gt_joints_t, seq_mask)

    gt_joints = gt_joints_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
    recon_joints = recon_joints_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
    sample_joints = sample_joints_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
    object_positions = (
        batch["object_positions"].squeeze(0).detach().cpu().numpy().astype(np.float32)
    )
    object_rotations = (
        batch["object_rotations"].squeeze(0).detach().cpu().numpy().astype(np.float32)
        if "object_rotations" in batch
        else None
    )
    object_pc = (
        batch["object_pc"].squeeze(0).detach().cpu().numpy().astype(np.float32)
        if "object_pc" in batch
        else None
    )
    contact_state = (
        batch["contact_state"].squeeze(0).detach().cpu().numpy().astype(np.float32)
    )

    events = _build_clip_events(
        gt_joints=gt_joints,
        recon_joints=recon_joints,
        sample_joints=sample_joints,
        object_positions=object_positions,
        contact_state=contact_state,
        seq_len=seq_len,
        threshold=threshold,
        window_k=window_k,
        transition_radius=transition_radius,
    )
    return ClipRecord(
        index=int(idx),
        subset=str(batch["subset"][0]),
        seq_id=str(batch["seq_id"][0]),
        text=str(batch["text"][0]),
        seq_len=seq_len,
        gt_motion=motion_gt.squeeze(0).detach().cpu().numpy().astype(np.float32),
        recon_motion=recon_motion_t.squeeze(0).detach().cpu().numpy().astype(np.float32),
        sample_motion=sample_motion_t.squeeze(0).detach().cpu().numpy().astype(np.float32),
        gt_joints=gt_joints,
        recon_joints=recon_joints,
        sample_joints=sample_joints,
        object_positions=object_positions,
        object_rotations=object_rotations,
        object_pc=object_pc,
        contact_state=contact_state,
        body_sample_over_gt=float(body_sample_over_gt),
        body_recon_over_gt=float(body_recon_over_gt),
        events=events,
    )


def _selected_event_for_category(
    candidates: list[ClipRecord],
    kind: str | None = None,
    part: str | None = None,
    subset: str | None = None,
    exclude: set[str] | None = None,
) -> tuple[ClipRecord, EventRecord] | None:
    exclude = exclude or set()
    best: tuple[float, ClipRecord, EventRecord] | None = None
    for clip in candidates:
        if clip.seq_id in exclude:
            continue
        if subset is not None and clip.subset != subset:
            continue
        for event in clip.events:
            if kind is not None and event.kind != kind:
                continue
            if part is not None and event.part != part:
                continue
            score = event.score
            if best is None or score > best[0]:
                best = (score, clip, event)
    if best is None:
        return None
    return best[1], best[2]


def _choose_clips(candidates: list[ClipRecord], max_clips: int) -> list[ClipRecord]:
    selected: list[ClipRecord] = []
    selected_ids: set[str] = set()

    def add(clip: ClipRecord, reason: str, event: EventRecord | None = None) -> None:
        if clip.seq_id in selected_ids or len(selected) >= int(max_clips):
            return
        clip.selected_reason = reason
        if event is not None:
            clip.events = [event] + [ev for ev in clip.events if ev is not event]
        selected.append(clip)
        selected_ids.add(clip.seq_id)

    categories = [
        ("L_hand onset", {"kind": "onset", "part": "L_hand"}),
        ("R_hand onset", {"kind": "onset", "part": "R_hand"}),
        ("L_hand release", {"kind": "release", "part": "L_hand"}),
        ("R_hand release", {"kind": "release", "part": "R_hand"}),
        ("chair transition", {"subset": "chairs"}),
        ("object manipulation transition", {"subset": None}),
    ]
    for reason, filters in categories:
        pool = candidates
        if reason == "object manipulation transition":
            pool = [c for c in candidates if c.subset != "chairs"]
            filters = {}
        hit = _selected_event_for_category(pool, exclude=selected_ids, **filters)
        if hit is not None:
            add(hit[0], reason, hit[1])

    remaining = [c for c in candidates if c.seq_id not in selected_ids]
    if remaining and len(selected) < int(max_clips):
        failure = min(remaining, key=lambda c: c.body_sample_over_gt)
        add(failure, "v18 sample frozen/low body velocity")
    remaining = [c for c in candidates if c.seq_id not in selected_ids]
    if remaining and len(selected) < int(max_clips):
        good = max(remaining, key=lambda c: c.body_sample_over_gt)
        add(good, "v18 comparatively better sample")

    for clip in sorted(candidates, key=lambda c: c.events[0].score if c.events else -1.0, reverse=True):
        if len(selected) >= int(max_clips):
            break
        if clip.seq_id not in selected_ids and clip.events:
            add(clip, "top transition score", clip.events[0])
    return selected


def _downsample_object_pc(object_pc: np.ndarray | None, max_points: int, seed: int) -> np.ndarray | None:
    if object_pc is None:
        return None
    if len(object_pc) <= int(max_points):
        return object_pc
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(len(object_pc), size=int(max_points), replace=False)
    return object_pc[idx]


def _precompute_object_cloud(
    object_pc: np.ndarray | None,
    object_positions: np.ndarray,
    object_rotations: np.ndarray | None,
) -> np.ndarray | None:
    if object_pc is None:
        return None
    total_t = len(object_positions)
    out = np.empty((total_t, len(object_pc), 3), dtype=np.float32)
    for t in range(total_t):
        if object_rotations is not None and t < len(object_rotations):
            rot = _axis_angle_to_rotmat(object_rotations[t])
            out[t] = object_pc @ rot.T + object_positions[t]
        else:
            out[t] = object_pc + object_positions[t]
    return out


def _axis_limits(
    joint_series: list[np.ndarray],
    object_positions: np.ndarray,
    object_cloud: np.ndarray | None,
    start: int,
    end: int,
) -> tuple[np.ndarray, float]:
    chunks = [j[start : end + 1].reshape(-1, 3) for j in joint_series]
    chunks.append(object_positions[start : end + 1])
    if object_cloud is not None:
        chunks.append(object_cloud[start : end + 1].reshape(-1, 3))
    all_pos = np.concatenate(chunks, axis=0)
    center = all_pos.mean(axis=0)
    max_range = max(float((all_pos.max(axis=0) - all_pos.min(axis=0)).max()) / 2.0 * 1.15, 0.5)
    return center.astype(np.float32), float(max_range)


def render_side_by_side_video(
    clip: ClipRecord,
    output_path: Path,
    start: int,
    end: int,
    title: str,
    event: EventRecord | None,
    fps: float,
    dpi: int,
    object_points: int,
    seed: int,
    elev: float = 15.0,
    azim: float = -60.0,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    start = max(0, int(start))
    end = min(int(end), clip.seq_len - 1)
    frames = list(range(start, end + 1))
    object_pc = _downsample_object_pc(clip.object_pc, object_points, seed=seed)
    object_cloud = _precompute_object_cloud(
        object_pc,
        clip.object_positions[: clip.seq_len],
        clip.object_rotations[: clip.seq_len] if clip.object_rotations is not None else None,
    )
    center, max_range = _axis_limits(
        [clip.gt_joints, clip.recon_joints, clip.sample_joints],
        clip.object_positions[: clip.seq_len],
        object_cloud,
        start,
        end,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(15.5, 5.6))
    axes = [fig.add_subplot(1, 3, i + 1, projection="3d") for i in range(3)]
    names = ["GT", "Recon_t100", "DDPM_sample"]
    joint_sources = [clip.gt_joints, clip.recon_joints, clip.sample_joints]
    artists: list[dict[str, Any]] = []

    for ax, name in zip(axes, names):
        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[2] - max_range, center[2] + max_range)
        ax.set_zlim(center[1] - max_range, center[1] + max_range)
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_zlabel("Y")
        ax.view_init(elev=elev, azim=azim)
        scatter = ax.scatter([], [], [], c="#1f77b4", s=18)
        lines = [ax.plot([], [], [], c="0.35", linewidth=1.2)[0] for _ in SKELETON_CONNECTIONS]
        obj = ax.scatter([], [], [], c="#d62728", s=3 if object_cloud is not None else 32, alpha=0.55)
        title_artist = ax.set_title(name)
        artists.append({"scatter": scatter, "lines": lines, "obj": obj, "title": title_artist})

    event_text = ""
    if event is not None:
        event_text = f" | {event.kind} {event.part} @ frame {event.frame}"
    wrapped_text = "\n".join(textwrap.wrap(clip.text, width=135))
    fig.suptitle(
        f"{title}{event_text}\n{clip.subset}/{clip.seq_id} | {wrapped_text}",
        fontsize=10,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.86))

    def update(frame_idx: int) -> list[Any]:
        frame = frames[frame_idx]
        updated: list[Any] = []
        for source, ax_art, name in zip(joint_sources, artists, names):
            joints = source[frame]
            ax_art["scatter"]._offsets3d = (joints[:, 0], joints[:, 2], joints[:, 1])
            updated.append(ax_art["scatter"])
            for (i, j), line in zip(SKELETON_CONNECTIONS, ax_art["lines"]):
                line.set_data([joints[i, 0], joints[j, 0]], [joints[i, 2], joints[j, 2]])
                line.set_3d_properties([joints[i, 1], joints[j, 1]])
                updated.append(line)
            if object_cloud is not None:
                obj = object_cloud[frame]
                ax_art["obj"]._offsets3d = (obj[:, 0], obj[:, 2], obj[:, 1])
            else:
                obj_pos = clip.object_positions[frame]
                ax_art["obj"]._offsets3d = ([obj_pos[0]], [obj_pos[2]], [obj_pos[1]])
            updated.append(ax_art["obj"])
            marker = ""
            if event is not None and frame == event.frame:
                marker = " | EVENT"
            ax_art["title"].set_text(f"{name}\nframe {frame}{marker}")
            updated.append(ax_art["title"])
        return updated

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=False, repeat=False)
    try:
        anim.save(str(output_path), writer="ffmpeg", fps=fps, dpi=dpi)
    finally:
        plt.close(fig)
    print(f"  saved {output_path}")


def save_contact_sheet(
    clip: ClipRecord,
    output_path: Path,
    start: int,
    end: int,
    title: str,
    event: EventRecord | None,
    dpi: int,
    object_points: int,
    seed: int,
    elev: float = 15.0,
    azim: float = -60.0,
) -> None:
    import matplotlib.pyplot as plt

    start = max(0, int(start))
    end = min(int(end), clip.seq_len - 1)
    if event is not None:
        frames = [
            start,
            max(start, event.frame - 5),
            event.frame,
            min(end, event.frame + 5),
            end,
        ]
    else:
        frames = np.linspace(start, end, num=5).round().astype(int).tolist()
    object_pc = _downsample_object_pc(clip.object_pc, object_points, seed=seed)
    object_cloud = _precompute_object_cloud(
        object_pc,
        clip.object_positions[: clip.seq_len],
        clip.object_rotations[: clip.seq_len] if clip.object_rotations is not None else None,
    )
    center, max_range = _axis_limits(
        [clip.gt_joints, clip.recon_joints, clip.sample_joints],
        clip.object_positions[: clip.seq_len],
        object_cloud,
        start,
        end,
    )
    sources = [clip.gt_joints, clip.recon_joints, clip.sample_joints]
    names = ["GT", "Recon_t100", "DDPM_sample"]
    fig = plt.figure(figsize=(15.5, 12.5))
    for row, frame in enumerate(frames):
        for col, (source, name) in enumerate(zip(sources, names)):
            ax = fig.add_subplot(len(frames), 3, row * 3 + col + 1, projection="3d")
            joints = source[frame]
            ax.scatter(joints[:, 0], joints[:, 2], joints[:, 1], c="#1f77b4", s=12)
            for i, j in SKELETON_CONNECTIONS:
                ax.plot(
                    [joints[i, 0], joints[j, 0]],
                    [joints[i, 2], joints[j, 2]],
                    [joints[i, 1], joints[j, 1]],
                    c="0.35",
                    linewidth=0.9,
                )
            if object_cloud is not None:
                obj = object_cloud[frame]
                ax.scatter(obj[:, 0], obj[:, 2], obj[:, 1], c="#d62728", s=1, alpha=0.45)
            else:
                obj_pos = clip.object_positions[frame]
                ax.scatter([obj_pos[0]], [obj_pos[2]], [obj_pos[1]], c="#d62728", s=20)
            ax.set_xlim(center[0] - max_range, center[0] + max_range)
            ax.set_ylim(center[2] - max_range, center[2] + max_range)
            ax.set_zlim(center[1] - max_range, center[1] + max_range)
            ax.view_init(elev=elev, azim=azim)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            marker = " EVENT" if event is not None and frame == event.frame else ""
            ax.set_title(f"{name} | frame {frame}{marker}", fontsize=8)
    fig.suptitle(title, fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print(f"  saved {output_path}")


def _save_tensors(clip: ClipRecord, tensors_dir: Path, selected_events: list[EventRecord]) -> Path:
    tensors_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{clip.index:02d}_{clip.subset}_{clip.seq_id}"
    npz_path = tensors_dir / f"{stem}.npz"
    np.savez_compressed(
        npz_path,
        gt_motion=clip.gt_motion[: clip.seq_len],
        recon_x0_pred_t100=clip.recon_motion[: clip.seq_len],
        sample_motion=clip.sample_motion[: clip.seq_len],
        gt_joints=clip.gt_joints[: clip.seq_len],
        recon_joints=clip.recon_joints[: clip.seq_len],
        sample_joints=clip.sample_joints[: clip.seq_len],
        object_positions=clip.object_positions[: clip.seq_len],
        object_rotations=(
            clip.object_rotations[: clip.seq_len]
            if clip.object_rotations is not None
            else np.zeros((clip.seq_len, 3), dtype=np.float32)
        ),
        contact_state=clip.contact_state[: clip.seq_len],
    )
    meta = {
        "subset": clip.subset,
        "seq_id": clip.seq_id,
        "text": clip.text,
        "seq_len": clip.seq_len,
        "body_sample_over_gt": clip.body_sample_over_gt,
        "body_recon_over_gt": clip.body_recon_over_gt,
        "selected_reason": clip.selected_reason,
        "events": [
            {
                "kind": ev.kind,
                "part": ev.part,
                "frame": ev.frame,
                "crop_start": ev.crop_start,
                "crop_end": ev.crop_end,
                "score": ev.score,
                "metrics": ev.metrics,
            }
            for ev in selected_events
        ],
    }
    (tensors_dir / f"{stem}_metadata.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )
    return npz_path


def _short_text(text: str, n: int = 90) -> str:
    text = " ".join(str(text).split())
    return text[:n] + ("..." if len(text) > n else "")


def _render_report(
    path: Path,
    args: argparse.Namespace,
    selected: list[ClipRecord],
    video_rows: list[dict[str, Any]],
    full_count: int,
    crop_count: int,
) -> None:
    lines: list[str] = []
    lines.append("# v18 recon vs sample visual diagnostic\n")
    lines.append(f"**Config:** `{args.config}`  ")
    lines.append(f"**Checkpoint:** `{args.ckpt}`  ")
    lines.append(f"**Output dir:** `{args.output_dir}`  ")
    lines.append(f"**Recon:** one-step x0 prediction at t={args.recon_t}  ")
    lines.append(f"**DDPM sample:** cfg_scale={args.cfg_scale}, seed={args.seed}  ")
    lines.append(f"**Rendered:** {full_count} full videos, {crop_count} crop videos\n")

    lines.append("## Rendered Video List\n")
    lines.append("| clip | subset | seq_id | text short | event | part | video path |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in video_rows:
        lines.append(
            f"| {row['clip']} | {row['subset']} | `{row['seq_id']}` | "
            f"{row['text_short']} | {row['event']} | {row['part']} | `{row['path']}` |"
        )

    lines.append("\n## Visual Checklist\n")
    lines.append(
        "These entries are generated beside the videos to make naked-eye review "
        "consistent. The auto verdict uses event dynamics as a proxy; final QA "
        "should still watch the MP4s."
    )
    lines.append("")
    lines.append("| clip | reason | recon pose/root | recon faster than sample | onset approach | release leave | jitter/body distortion risk | sample smooth/frozen risk |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for clip in selected:
        best_onset = next((ev for ev in clip.events if ev.kind == "onset"), None)
        best_release = next((ev for ev in clip.events if ev.kind == "release"), None)
        recon_body_ok = clip.body_recon_over_gt >= 0.75
        sample_frozen = clip.body_sample_over_gt < 0.75
        recon_adv = any(
            ev.metrics["relative_velocity_recon_over_gt"]
            > ev.metrics["relative_velocity_sample_over_gt"] + 0.08
            for ev in clip.events[:2]
        )
        onset_ok = (
            best_onset is not None
            and best_onset.metrics["positive_distance_change_recon_over_gt"] >= 0.75
        )
        release_ok = (
            best_release is not None
            and best_release.metrics["positive_distance_change_recon_over_gt"] >= 0.75
        )
        jitter_risk = "low/needs visual review" if recon_body_ok else "medium"
        lines.append(
            f"| `{clip.seq_id}` | {clip.selected_reason} | "
            f"{'likely yes' if recon_body_ok else 'unclear'} "
            f"(body recon/GT {clip.body_recon_over_gt:.2f}) | "
            f"{'yes' if recon_adv else 'mixed'} | "
            f"{'yes' if onset_ok else 'n/a or mixed'} | "
            f"{'yes' if release_ok else 'n/a or mixed'} | "
            f"{jitter_risk} | "
            f"{'yes' if sample_frozen else 'less clear'} "
            f"(sample/GT {clip.body_sample_over_gt:.2f}) |"
        )

    lines.append("\n## Quantitative Metadata Beside Each Event\n")
    lines.append("| clip | event | part | frame | GT rel-vel | recon rel-vel | sample rel-vel | recon/GT | sample/GT | GT change | recon change | sample change | recon/GT change | sample/GT change |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    event_rows: list[dict[str, Any]] = []
    for clip in selected:
        for ev in clip.events[:2]:
            m = ev.metrics
            rel = m["relative_velocity_cm_per_frame"]
            ch = m["positive_distance_change_cm"]
            event_rows.append(
                {
                    "recon_rel_ratio": m["relative_velocity_recon_over_gt"],
                    "sample_rel_ratio": m["relative_velocity_sample_over_gt"],
                    "recon_change_ratio": m["positive_distance_change_recon_over_gt"],
                    "sample_change_ratio": m["positive_distance_change_sample_over_gt"],
                }
            )
            lines.append(
                f"| `{clip.seq_id}` | {ev.kind} | {ev.part} | {ev.frame} | "
                f"{rel['gt']:.2f} | {rel['recon']:.2f} | {rel['sample']:.2f} | "
                f"{m['relative_velocity_recon_over_gt']:.2f} | "
                f"{m['relative_velocity_sample_over_gt']:.2f} | "
                f"{ch['gt']:.2f} | {ch['recon']:.2f} | {ch['sample']:.2f} | "
                f"{m['positive_distance_change_recon_over_gt']:.2f} | "
                f"{m['positive_distance_change_sample_over_gt']:.2f} |"
            )

    if event_rows:
        recon_rel = float(np.mean([r["recon_rel_ratio"] for r in event_rows]))
        sample_rel = float(np.mean([r["sample_rel_ratio"] for r in event_rows]))
        recon_change = float(np.mean([r["recon_change_ratio"] for r in event_rows]))
        sample_change = float(np.mean([r["sample_change_ratio"] for r in event_rows]))
    else:
        recon_rel = sample_rel = recon_change = sample_change = 0.0

    lines.append("\n## Final Visual Verdict\n")
    if recon_rel > sample_rel + 0.08 and recon_change >= sample_change:
        lines.append(
            "**Auto-assisted verdict:** supports the hypothesis. The selected "
            f"events have mean recon rel-vel/GT {recon_rel:.2f} vs sample "
            f"{sample_rel:.2f}, and recon distance-change/GT {recon_change:.2f} "
            f"vs sample {sample_change:.2f}. Watch the MP4s to confirm pose/root "
            "quality, but the rendered event set is consistent with one-step recon "
            "recovering more transition motion than DDPM rollout."
        )
    elif recon_rel > sample_rel:
        lines.append(
            "**Auto-assisted verdict:** mixed but leaning toward recon advantage. "
            f"Mean recon rel-vel/GT {recon_rel:.2f} exceeds sample {sample_rel:.2f}, "
            f"while distance-change ratios are recon {recon_change:.2f} vs sample "
            f"{sample_change:.2f}. The MP4s are needed to decide whether the visual "
            "difference is strong enough."
        )
    else:
        lines.append(
            "**Auto-assisted verdict:** does not cleanly support the hypothesis. "
            f"Mean recon rel-vel/GT {recon_rel:.2f}, sample {sample_rel:.2f}; "
            "if the videos look similar, the transition metrics are insufficient "
            "or the recon path is also visually weak."
        )
    lines.append("")
    lines.append("Answers:")
    lines.append("1. One-step recon visually recovers transition? **Pending naked-eye MP4 review; auto metrics indicate the expected comparison set.**")
    lines.append("2. Recon advantage only numeric? **Use crop videos and contact sheets to verify; static metrics alone are not treated as final proof.**")
    lines.append("3. DDPM sample more smooth/frozen? **Likely on clips with body sample/GT < 0.75; inspect the full videos.**")
    lines.append("4. If recon looks good and sample bad: **supports denoising rollout collapse.**")
    lines.append("5. If recon also looks bad: **metric is misleading or transition representation/supervision is insufficient.**")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--num-candidates", type=int, default=16)
    parser.add_argument("--max-clips", type=int, default=8)
    parser.add_argument("--bucket", choices=["train", "val"], default="train")
    parser.add_argument("--balanced-subsets", action="store_true")
    parser.add_argument("--recon-t", type=int, default=100)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--window-k", type=int, default=10)
    parser.add_argument("--transition-radius", type=int, default=5)
    parser.add_argument("--dpi", type=int, default=72)
    parser.add_argument("--object-points", type=int, default=192)
    parser.add_argument("--max-crop-events-per-clip", type=int, default=2)
    parser.add_argument("--skip-full-render", action="store_true")
    parser.add_argument("--skip-crops", action="store_true")
    parser.add_argument("--save-contact-sheets", action="store_true")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    dataset = _build_dataset(cfg, args.bucket)
    clip_indices = (
        _balanced_subset_indices(dataset, int(args.num_candidates))
        if args.balanced_subsets
        else list(range(min(int(args.num_candidates), len(dataset))))
    )
    dataset = Subset(dataset, clip_indices)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_hoi, num_workers=0)

    model, object_encoder, z_dims = _build_model(cfg, device)
    _load_checkpoint(model, object_encoder, args.ckpt)
    clip_model = load_clip_text_encoder(
        device=device,
        model_name=str(cfg.model.text_encoder.clip_version),
        download_root=str(cfg.model.text_encoder.get("download_root", "cache/clip")),
    )

    print(f"Running {len(loader)} candidate clips on {device}...")
    candidates: list[ClipRecord] = []
    for idx, batch in enumerate(loader):
        clip = _run_clip(
            idx=idx,
            batch=batch,
            model=model,
            object_encoder=object_encoder,
            clip_model=clip_model,
            z_dims=z_dims,
            cfg=cfg,
            device=device,
            recon_t=int(args.recon_t),
            cfg_scale=float(args.cfg_scale),
            seed=int(args.seed),
            threshold=float(args.threshold),
            window_k=int(args.window_k),
            transition_radius=int(args.transition_radius),
        )
        candidates.append(clip)
        print(
            f"  [{idx + 1}/{len(loader)}] {clip.subset}/{clip.seq_id} "
            f"T={clip.seq_len} events={len(clip.events)} "
            f"body sample/GT={clip.body_sample_over_gt:.2f} "
            f"recon/GT={clip.body_recon_over_gt:.2f}"
        )

    selected = _choose_clips(candidates, max_clips=int(args.max_clips))
    print(f"Selected {len(selected)} clips.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tensors_dir = args.output_dir / "tensors"
    sheets_dir = args.output_dir / "contact_sheets"
    video_rows: list[dict[str, Any]] = []
    full_count = 0
    crop_count = 0
    for out_idx, clip in enumerate(selected, start=1):
        selected_events = clip.events[: max(1, int(args.max_crop_events_per_clip))]
        _save_tensors(clip, tensors_dir, selected_events)
        stem = f"{out_idx:02d}_{clip.subset}_{clip.seq_id}"

        if not args.skip_full_render:
            full_path = args.output_dir / f"{stem}_full_gt_recon_sample.mp4"
            render_side_by_side_video(
                clip=clip,
                output_path=full_path,
                start=0,
                end=clip.seq_len - 1,
                title=f"full sequence | reason: {clip.selected_reason}",
                event=None,
                fps=float(args.fps),
                dpi=int(args.dpi),
                object_points=int(args.object_points),
                seed=int(args.seed) + out_idx,
            )
            full_count += 1
            video_rows.append(
                {
                    "clip": out_idx,
                    "subset": clip.subset,
                    "seq_id": clip.seq_id,
                    "text_short": _short_text(clip.text),
                    "event": "full",
                    "part": "-",
                    "path": str(full_path),
                }
            )

        if not args.skip_crops:
            for ev_idx, event in enumerate(selected_events, start=1):
                crop_path = (
                    args.output_dir
                    / f"{stem}_{event.part}_{event.kind}_crop{ev_idx}.mp4"
                )
                render_side_by_side_video(
                    clip=clip,
                    output_path=crop_path,
                    start=event.crop_start,
                    end=event.crop_end,
                    title=f"{event.kind} crop | reason: {clip.selected_reason}",
                    event=event,
                    fps=float(args.fps),
                    dpi=int(args.dpi),
                    object_points=int(args.object_points),
                    seed=int(args.seed) + out_idx + ev_idx,
                )
                crop_count += 1
                video_rows.append(
                    {
                        "clip": out_idx,
                        "subset": clip.subset,
                        "seq_id": clip.seq_id,
                        "text_short": _short_text(clip.text),
                        "event": event.kind,
                        "part": event.part,
                        "path": str(crop_path),
                    }
                )
                if args.save_contact_sheets:
                    sheet_path = (
                        sheets_dir
                        / f"{stem}_{event.part}_{event.kind}_crop{ev_idx}.png"
                    )
                    save_contact_sheet(
                        clip=clip,
                        output_path=sheet_path,
                        start=event.crop_start,
                        end=event.crop_end,
                        title=f"{clip.subset}/{clip.seq_id} {event.kind} {event.part} @ {event.frame}",
                        event=event,
                        dpi=int(args.dpi),
                        object_points=int(args.object_points),
                        seed=int(args.seed) + out_idx + ev_idx,
                    )

    selection_meta = {
        "config": str(args.config),
        "ckpt": str(args.ckpt),
        "output_dir": str(args.output_dir),
        "seed": int(args.seed),
        "recon_t": int(args.recon_t),
        "cfg_scale": float(args.cfg_scale),
        "num_candidates": len(candidates),
        "selected": [
            {
                "subset": clip.subset,
                "seq_id": clip.seq_id,
                "seq_len": clip.seq_len,
                "text": clip.text,
                "selected_reason": clip.selected_reason,
                "body_sample_over_gt": clip.body_sample_over_gt,
                "body_recon_over_gt": clip.body_recon_over_gt,
                "events": [
                    {
                        "kind": ev.kind,
                        "part": ev.part,
                        "frame": ev.frame,
                        "crop_start": ev.crop_start,
                        "crop_end": ev.crop_end,
                        "score": ev.score,
                        "metrics": ev.metrics,
                    }
                    for ev in clip.events[: max(1, int(args.max_crop_events_per_clip))]
                ],
            }
            for clip in selected
        ],
        "videos": video_rows,
    }
    (args.output_dir / "selection_metadata.json").write_text(
        json.dumps(selection_meta, indent=2),
        encoding="utf-8",
    )
    _render_report(
        path=args.report,
        args=args,
        selected=selected,
        video_rows=video_rows,
        full_count=full_count,
        crop_count=crop_count,
    )
    print(f"Wrote report to {args.report}")
    print(f"Output dir: {args.output_dir}")
    print(f"Rendered full={full_count}, crops={crop_count}")


if __name__ == "__main__":
    main()
