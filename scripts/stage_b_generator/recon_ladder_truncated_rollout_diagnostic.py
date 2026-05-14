"""Stage-B recon ladder and truncated-rollout diagnostic.

This script separates three possible causes of frozen transition dynamics:

1. high-noise one-step x0 prediction failure;
2. multi-step reverse-rollout collapse from a real q(x_t | x0) state;
3. pure-noise full-DDPM mode-selection failure.

It is eval/render only and does not modify training code.
"""
from __future__ import annotations

import argparse
import json
import math
import textwrap
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from dynamics_diagnostic import (
    LOCAL_JOINTS,
    _balanced_subset_indices,
    _build_cond,
    _build_dataset,
    _build_model,
    _fk_from_motion_135,
)
from piano.data.dataset import collate_hoi
from piano.inference.visualize_motion import SKELETON_CONNECTIONS
from piano.models.motion_anchordiff import _extract
from piano.utils.clip_utils import load_clip_text_encoder
from render_recon_vs_sample import (
    EventRecord,
    _axis_limits,
    _downsample_object_pc,
    _event_frames,
    _extract_plan,
    _load_checkpoint,
    _precompute_object_cloud,
    _safe_div,
    _short_text,
)


TIMESTEPS_DEFAULT = (100, 300, 500, 700, 900)
LOG_TIMESTEPS_DEFAULT = (900, 700, 500, 300, 100, 0)
HAND_SPECS = {
    "L_hand": {"joint": 20, "contact_idx": 0},
    "R_hand": {"joint": 21, "contact_idx": 1},
}


@dataclass
class SelectedEvent:
    kind: str
    part: str
    frame: int
    crop_start: int
    crop_end: int
    reason: str = ""


@dataclass
class ClipOutputs:
    index: int
    subset: str
    seq_id: str
    text: str
    seq_len: int
    event: SelectedEvent
    gt_motion: np.ndarray
    gt_joints: np.ndarray
    object_positions: np.ndarray
    object_rotations: np.ndarray | None
    object_pc: np.ndarray | None
    one_step_motion: dict[int, np.ndarray]
    one_step_joints: dict[int, np.ndarray]
    trunc_motion: dict[int, np.ndarray]
    trunc_joints: dict[int, np.ndarray]
    full_ddpm_motion: np.ndarray
    full_ddpm_joints: np.ndarray
    full_intermediate_motion: dict[int, np.ndarray]
    full_intermediate_joints: dict[int, np.ndarray]
    ddim_motion: np.ndarray | None
    ddim_joints: np.ndarray | None
    metrics: dict[str, Any]


def _parse_ints(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _noise_metadata(diffusion, timesteps: list[int]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for t in timesteps:
        alpha_bar = float(diffusion.alphas_cumprod[int(t)].detach().cpu().item())
        sqrt_alpha = math.sqrt(max(alpha_bar, 0.0))
        sigma = math.sqrt(max(0.0, 1.0 - alpha_bar))
        snr = _safe_div(alpha_bar, 1.0 - alpha_bar)
        log_snr = float(math.log(max(snr, 1e-30)))
        out[str(int(t))] = {
            "t": int(t),
            "alpha_bar": alpha_bar,
            "sqrt_alpha_bar": sqrt_alpha,
            "sigma": sigma,
            "snr": snr,
            "log_snr": log_snr,
        }
    return out


def _predict_x0(
    model,
    x: Tensor,
    t: Tensor,
    cond: dict[str, Any],
    cfg_scale: float,
    self_cond: Tensor | None = None,
) -> Tensor:
    pred_cond = model.denoiser(
        x, t, cond, cond_drop_mask=None, self_cond=self_cond,
    )
    if float(cfg_scale) != 1.0:
        drop = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
        pred_uncond = model.denoiser(
            x, t, cond, cond_drop_mask=drop, self_cond=self_cond,
        )
    else:
        pred_uncond = None

    if model.diffusion.prediction_target == "v":
        x0_cond = model.diffusion.predict_x0_from_v(x, t, pred_cond)
        x0_uncond = (
            model.diffusion.predict_x0_from_v(x, t, pred_uncond)
            if pred_uncond is not None
            else None
        )
    else:
        x0_cond = pred_cond
        x0_uncond = pred_uncond

    if x0_uncond is None:
        return x0_cond
    return x0_uncond + float(cfg_scale) * (x0_cond - x0_uncond)


def _self_cond_for_t(model, previous_x0: Tensor | None, t_int: int) -> Tensor | None:
    dcfg = getattr(model.denoiser, "cfg", None)
    if not bool(getattr(dcfg, "use_self_conditioning", False)):
        return None
    mode = str(getattr(dcfg, "self_conditioning_mode", "standard"))
    if mode == "standard":
        return previous_x0
    if mode == "late_start":
        t_max = int(getattr(dcfg, "self_conditioning_t_max", 700))
        return previous_x0 if int(t_int) <= t_max else None
    raise ValueError(f"Unknown self_conditioning_mode: {mode!r}")


@torch.no_grad()
def _one_step_recon(
    model,
    motion_gt: Tensor,
    cond: dict[str, Any],
    t_value: int,
    cfg_scale: float,
    seed: int,
) -> Tensor:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    t = torch.full((motion_gt.shape[0],), int(t_value), device=motion_gt.device, dtype=torch.long)
    noise = torch.randn_like(motion_gt)
    x_t = model.diffusion.q_sample(motion_gt, t, noise)
    return _predict_x0(model, x_t, t, cond, cfg_scale=cfg_scale)


@torch.no_grad()
def _truncated_rollout(
    model,
    motion_gt: Tensor,
    cond: dict[str, Any],
    t_start: int,
    cfg_scale: float,
    seed: int,
    sampler: str = "ddpm",
) -> Tensor:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    t_init = torch.full((motion_gt.shape[0],), int(t_start), device=motion_gt.device, dtype=torch.long)
    noise = torch.randn_like(motion_gt)
    x = model.diffusion.q_sample(motion_gt, t_init, noise)
    self_cond: Tensor | None = None

    for t_int in reversed(range(int(t_start) + 1)):
        t = torch.full((motion_gt.shape[0],), t_int, device=motion_gt.device, dtype=torch.long)
        curr_self_cond = _self_cond_for_t(model, self_cond, t_int)
        x0 = _predict_x0(
            model, x, t, cond, cfg_scale=cfg_scale, self_cond=curr_self_cond,
        )
        self_cond = x0.detach()
        if sampler == "ddim_eta0":
            if t_int == 0:
                x = x0
            else:
                eps = (
                    x - _extract(model.diffusion.sqrt_alphas_cumprod, t, x.shape) * x0
                ) / _extract(model.diffusion.sqrt_one_minus_alphas_cumprod, t, x.shape)
                t_prev = (t - 1).clamp(min=0)
                sqrt_a_prev = _extract(model.diffusion.sqrt_alphas_cumprod, t_prev, x.shape)
                sqrt_om_a_prev = _extract(
                    model.diffusion.sqrt_one_minus_alphas_cumprod,
                    t_prev,
                    x.shape,
                )
                x = sqrt_a_prev * x0 + sqrt_om_a_prev * eps
        else:
            mean = model.diffusion.posterior_mean_from_x0(x0, x, t)
            if sampler == "ddpm_det" or t_int == 0:
                x = mean
            else:
                step_noise = torch.randn_like(x)
                log_var = _extract(model.diffusion.posterior_log_variance_clipped, t, x.shape)
                x = mean + (0.5 * log_var).exp() * step_noise
    return x


@torch.no_grad()
def _full_sample_with_intermediates(
    model,
    cond: dict[str, Any],
    seq_length: int,
    cfg_scale: float,
    seed: int,
    log_timesteps: list[int],
    sampler: str = "ddpm",
) -> tuple[Tensor, dict[int, Tensor]]:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    batch_size = cond["z_int"].shape[0]
    shape = (batch_size, int(seq_length), model.cfg.denoiser.motion_dim)
    x = torch.randn(shape, device=cond["z_int"].device)
    logs: dict[int, Tensor] = {}
    log_set = {int(t) for t in log_timesteps}
    self_cond: Tensor | None = None
    for t_int in reversed(range(model.diffusion.num_steps)):
        t = torch.full((batch_size,), t_int, device=x.device, dtype=torch.long)
        curr_self_cond = _self_cond_for_t(model, self_cond, t_int)
        x0 = _predict_x0(
            model, x, t, cond, cfg_scale=cfg_scale, self_cond=curr_self_cond,
        )
        self_cond = x0.detach()
        if t_int in log_set:
            logs[int(t_int)] = x0.detach().clone()
        if sampler == "ddim_eta0":
            if t_int == 0:
                x = x0
            else:
                eps = (
                    x - _extract(model.diffusion.sqrt_alphas_cumprod, t, x.shape) * x0
                ) / _extract(model.diffusion.sqrt_one_minus_alphas_cumprod, t, x.shape)
                t_prev = (t - 1).clamp(min=0)
                sqrt_a_prev = _extract(model.diffusion.sqrt_alphas_cumprod, t_prev, x.shape)
                sqrt_om_a_prev = _extract(
                    model.diffusion.sqrt_one_minus_alphas_cumprod,
                    t_prev,
                    x.shape,
                )
                x = sqrt_a_prev * x0 + sqrt_om_a_prev * eps
        else:
            mean = model.diffusion.posterior_mean_from_x0(x0, x, t)
            if sampler == "ddpm_det" or t_int == 0:
                x = mean
            else:
                noise = torch.randn_like(x)
                log_var = _extract(model.diffusion.posterior_log_variance_clipped, t, x.shape)
                x = mean + (0.5 * log_var).exp() * noise
    if 0 not in logs:
        logs[0] = x.detach().clone()
    return x, logs


def _body_stats_np(joints: np.ndarray, seq_len: int, fps: float) -> dict[str, float]:
    j = joints[: int(seq_len)].astype(np.float32)
    if len(j) < 3:
        return {
            "body_local_velocity_cm_per_frame": 0.0,
            "body_local_acceleration_p95_cm_per_frame2": 0.0,
            "fft_low": 0.0,
            "fft_mid": 0.0,
            "fft_high": 0.0,
        }
    vel_world = j[1:] - j[:-1]
    root_vel = j[1:, 0:1] - j[:-1, 0:1]
    vel_local = vel_world - root_vel
    acc_local = vel_local[1:] - vel_local[:-1]
    vel_mag = np.linalg.norm(vel_local[:, LOCAL_JOINTS], axis=-1) * 100.0
    acc_mag = np.linalg.norm(acc_local[:, LOCAL_JOINTS], axis=-1) * 100.0

    rel = j - j[:, 0:1]
    x = rel - rel.mean(axis=0, keepdims=True)
    fft = np.fft.rfft(x, axis=0)
    power = (fft.real ** 2 + fft.imag ** 2)[:, LOCAL_JOINTS, :].sum(axis=(1, 2))
    freqs = np.fft.rfftfreq(len(j), d=1.0 / float(fps))
    low = float(power[(freqs >= 0.0) & (freqs < 1.0)].sum())
    mid = float(power[(freqs >= 1.0) & (freqs < 4.0)].sum())
    high = float(power[freqs >= 4.0].sum())
    total = low + mid + high
    return {
        "body_local_velocity_cm_per_frame": float(vel_mag.mean()),
        "body_local_acceleration_p95_cm_per_frame2": float(np.percentile(acc_mag, 95)),
        "fft_low": _safe_div(low, total),
        "fft_mid": _safe_div(mid, total),
        "fft_high": _safe_div(high, total),
    }


def _hand_velocity_np(joints: np.ndarray, seq_len: int, hand_joint: int) -> float:
    j = joints[: int(seq_len)].astype(np.float32)
    if len(j) < 2:
        return 0.0
    hand_vel = j[1:, hand_joint] - j[:-1, hand_joint]
    root_vel = j[1:, 0] - j[:-1, 0]
    local = hand_vel - root_vel
    return float(np.linalg.norm(local, axis=-1).mean() * 100.0)


def _transition_raw_np(
    joints: np.ndarray,
    object_positions: np.ndarray,
    seq_len: int,
    event: SelectedEvent,
    window_k: int,
    transition_radius: int,
) -> dict[str, float]:
    spec = HAND_SPECS[event.part]
    hand_joint = int(spec["joint"])
    j = joints[: int(seq_len)]
    obj = object_positions[: int(seq_len)]
    t = int(event.frame)
    if event.kind == "onset":
        lo = max(0, t - int(window_k))
        hi = min(int(seq_len) - 1, t + int(transition_radius))
        change_name = "closing"
        change_start = max(0, t - int(window_k))
        raw_change = float(
            np.linalg.norm(j[change_start, hand_joint] - obj[change_start]) * 100.0
            - np.linalg.norm(j[t, hand_joint] - obj[t]) * 100.0
        )
    else:
        lo = max(0, t - int(transition_radius))
        hi = min(int(seq_len) - 1, t + int(window_k))
        change_name = "opening"
        change_end = min(int(seq_len) - 1, t + int(window_k))
        raw_change = float(
            np.linalg.norm(j[change_end, hand_joint] - obj[change_end]) * 100.0
            - np.linalg.norm(j[t, hand_joint] - obj[t]) * 100.0
        )
    if hi <= lo:
        rel_vel = 0.0
    else:
        hand_vel = j[lo + 1 : hi + 1, hand_joint] - j[lo:hi, hand_joint]
        obj_vel = obj[lo + 1 : hi + 1] - obj[lo:hi]
        rel_vel = float(np.linalg.norm(hand_vel - obj_vel, axis=-1).mean() * 100.0)
    return {
        "transition_relative_velocity_cm_per_frame": rel_vel,
        "distance_change_cm": raw_change,
        "positive_distance_change_cm": max(0.0, raw_change),
        "distance_change_name": change_name,
    }


def _source_metrics_np(
    source_joints: np.ndarray,
    gt_joints: np.ndarray,
    object_positions: np.ndarray,
    seq_len: int,
    event: SelectedEvent,
    fps: float,
    window_k: int,
    transition_radius: int,
) -> dict[str, float]:
    hand_joint = int(HAND_SPECS[event.part]["joint"])
    body = _body_stats_np(source_joints, seq_len, fps=fps)
    body_gt = _body_stats_np(gt_joints, seq_len, fps=fps)
    hand = _hand_velocity_np(source_joints, seq_len, hand_joint)
    hand_gt = _hand_velocity_np(gt_joints, seq_len, hand_joint)
    trans = _transition_raw_np(
        source_joints,
        object_positions,
        seq_len,
        event,
        window_k=window_k,
        transition_radius=transition_radius,
    )
    trans_gt = _transition_raw_np(
        gt_joints,
        object_positions,
        seq_len,
        event,
        window_k=window_k,
        transition_radius=transition_radius,
    )
    return {
        **body,
        "body_velocity_over_gt": _safe_div(
            body["body_local_velocity_cm_per_frame"],
            body_gt["body_local_velocity_cm_per_frame"],
        ),
        "hand_velocity_cm_per_frame": hand,
        "hand_velocity_over_gt": _safe_div(hand, hand_gt),
        "transition_relative_velocity_cm_per_frame": trans[
            "transition_relative_velocity_cm_per_frame"
        ],
        "transition_relative_velocity_over_gt": _safe_div(
            trans["transition_relative_velocity_cm_per_frame"],
            trans_gt["transition_relative_velocity_cm_per_frame"],
        ),
        "positive_distance_change_cm": trans["positive_distance_change_cm"],
        "positive_distance_change_over_gt": _safe_div(
            trans["positive_distance_change_cm"],
            trans_gt["positive_distance_change_cm"],
        ),
        "distance_change_cm": trans["distance_change_cm"],
        "distance_change_name": trans["distance_change_name"],
    }


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({k for row in rows for k, v in row.items() if isinstance(v, (int, float))})
    out: dict[str, float] = {}
    for key in keys:
        vals = [float(row[key]) for row in rows if key in row and np.isfinite(float(row[key]))]
        out[key] = float(np.mean(vals)) if vals else 0.0
    return out


def _event_from_metadata(entry: dict[str, Any]) -> SelectedEvent | None:
    if "event" in entry:
        ev = entry["event"]
    else:
        events = entry.get("events") or []
        if not events:
            return None
        ev = events[0]
    return SelectedEvent(
        kind=str(ev["kind"]),
        part=str(ev["part"]),
        frame=int(ev["frame"]),
        crop_start=int(ev.get("crop_start", max(0, int(ev["frame"]) - 15))),
        crop_end=int(ev.get("crop_end", int(ev["frame"]) + 15)),
        reason=str(ev.get("reason", entry.get("selected_reason", ""))),
    )


def _fallback_event_from_batch(
    contact_state: np.ndarray,
    seq_len: int,
    threshold: float,
) -> SelectedEvent | None:
    best: SelectedEvent | None = None
    for part, spec in HAND_SPECS.items():
        contact = contact_state[:seq_len, int(spec["contact_idx"])]
        onsets, releases = _event_frames(contact, threshold=threshold)
        for kind, frames in (("onset", onsets), ("release", releases)):
            for frame in frames:
                if kind == "onset":
                    start, end = max(0, frame - 15), min(seq_len - 1, frame + 10)
                else:
                    start, end = max(0, frame - 10), min(seq_len - 1, frame + 15)
                cand = SelectedEvent(kind=kind, part=part, frame=int(frame), crop_start=start, crop_end=end)
                if best is None:
                    best = cand
    return best


def _load_selection(selection_json: Path | None, max_clips: int) -> dict[str, SelectedEvent]:
    if selection_json is None or not Path(selection_json).exists():
        return {}
    payload = json.loads(Path(selection_json).read_text(encoding="utf-8"))
    out: dict[str, SelectedEvent] = {}
    for entry in payload.get("selected", payload.get("selected_clips", [])):
        event = _event_from_metadata(entry)
        if event is None:
            continue
        out[str(entry["seq_id"])] = event
        if len(out) >= int(max_clips):
            break
    return out


def _build_selected_batches(
    cfg,
    bucket: str,
    balanced_subsets: bool,
    num_candidates: int,
    selection: dict[str, SelectedEvent],
    max_clips: int,
    threshold: float,
) -> list[tuple[int, dict[str, Any], SelectedEvent]]:
    dataset = _build_dataset(cfg, bucket)
    indices = (
        _balanced_subset_indices(dataset, int(num_candidates))
        if balanced_subsets
        else list(range(min(int(num_candidates), len(dataset))))
    )
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=1, shuffle=False, collate_fn=collate_hoi, num_workers=0)
    selected: list[tuple[int, dict[str, Any], SelectedEvent]] = []
    fallback: list[tuple[int, dict[str, Any], SelectedEvent]] = []
    for idx, batch in enumerate(loader):
        seq_id = str(batch["seq_id"][0])
        if seq_id in selection:
            selected.append((idx, batch, selection[seq_id]))
        elif len(selected) + len(fallback) < int(max_clips):
            seq_len = int(batch["seq_len"][0].item())
            contact_state = batch["contact_state"].squeeze(0).cpu().numpy()
            ev = _fallback_event_from_batch(contact_state, seq_len, threshold=threshold)
            if ev is not None:
                fallback.append((idx, batch, ev))
    if len(selected) < int(max_clips):
        seen = {str(batch["seq_id"][0]) for _idx, batch, _ev in selected}
        for item in fallback:
            if str(item[1]["seq_id"][0]) in seen:
                continue
            selected.append(item)
            seen.add(str(item[1]["seq_id"][0]))
            if len(selected) >= int(max_clips):
                break
    return selected[: int(max_clips)]


def _run_clip(
    ordinal: int,
    batch_idx: int,
    batch: dict[str, Any],
    event: SelectedEvent,
    model,
    object_encoder,
    clip_model,
    z_dims,
    cfg,
    device: torch.device,
    timesteps: list[int],
    log_timesteps: list[int],
    cfg_scale: float,
    seed: int,
    fps: float,
    window_k: int,
    transition_radius: int,
    run_ddim: bool,
) -> ClipOutputs:
    cond, total_t = _build_cond(batch, model, object_encoder, clip_model, z_dims, cfg, device)
    cond = {**cond, "interaction_plan": _extract_plan(batch, device)}
    motion_gt = batch["motion"].to(device).float()
    rest_offsets = batch["rest_offsets"].to(device).float()
    seq_len = int(batch["seq_len"][0].item())
    gt_joints_t = batch["joints"].to(device).float()

    one_step_motion_t: dict[int, Tensor] = {}
    one_step_joints_t: dict[int, Tensor] = {}
    for t in timesteps:
        x0 = _one_step_recon(
            model,
            motion_gt,
            cond,
            t_value=int(t),
            cfg_scale=cfg_scale,
            seed=int(seed) + batch_idx * 10000 + int(t),
        )
        one_step_motion_t[int(t)] = x0
        one_step_joints_t[int(t)] = _fk_from_motion_135(x0, rest_offsets)

    trunc_motion_t: dict[int, Tensor] = {}
    trunc_joints_t: dict[int, Tensor] = {}
    for t in timesteps:
        x0 = _truncated_rollout(
            model,
            motion_gt,
            cond,
            t_start=int(t),
            cfg_scale=cfg_scale,
            seed=int(seed) + batch_idx * 10000 + 2000 + int(t),
            sampler="ddpm",
        )
        trunc_motion_t[int(t)] = x0
        trunc_joints_t[int(t)] = _fk_from_motion_135(x0, rest_offsets)

    full_motion_t, intermediate_t = _full_sample_with_intermediates(
        model,
        cond,
        seq_length=total_t,
        cfg_scale=cfg_scale,
        seed=int(seed) + batch_idx * 10000 + 900000,
        log_timesteps=log_timesteps,
        sampler="ddpm",
    )
    full_joints_t = _fk_from_motion_135(full_motion_t, rest_offsets)
    intermediate_joints_t = {
        int(t): _fk_from_motion_135(x0, rest_offsets)
        for t, x0 in intermediate_t.items()
    }

    ddim_motion_t = None
    ddim_joints_t = None
    if run_ddim:
        ddim_motion_t, _ = _full_sample_with_intermediates(
            model,
            cond,
            seq_length=total_t,
            cfg_scale=cfg_scale,
            seed=int(seed) + batch_idx * 10000 + 910000,
            log_timesteps=[],
            sampler="ddim_eta0",
        )
        ddim_joints_t = _fk_from_motion_135(ddim_motion_t, rest_offsets)

    gt_joints = gt_joints_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
    gt_motion = motion_gt.squeeze(0).detach().cpu().numpy().astype(np.float32)
    obj_pos = batch["object_positions"].squeeze(0).detach().cpu().numpy().astype(np.float32)
    obj_rot = batch["object_rotations"].squeeze(0).detach().cpu().numpy().astype(np.float32)
    obj_pc = batch["object_pc"].squeeze(0).detach().cpu().numpy().astype(np.float32)

    one_step_motion = {
        int(t): x.squeeze(0).detach().cpu().numpy().astype(np.float32)
        for t, x in one_step_motion_t.items()
    }
    one_step_joints = {
        int(t): x.squeeze(0).detach().cpu().numpy().astype(np.float32)
        for t, x in one_step_joints_t.items()
    }
    trunc_motion = {
        int(t): x.squeeze(0).detach().cpu().numpy().astype(np.float32)
        for t, x in trunc_motion_t.items()
    }
    trunc_joints = {
        int(t): x.squeeze(0).detach().cpu().numpy().astype(np.float32)
        for t, x in trunc_joints_t.items()
    }
    full_motion = full_motion_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
    full_joints = full_joints_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
    intermediate_motion = {
        int(t): x.squeeze(0).detach().cpu().numpy().astype(np.float32)
        for t, x in intermediate_t.items()
    }
    intermediate_joints = {
        int(t): x.squeeze(0).detach().cpu().numpy().astype(np.float32)
        for t, x in intermediate_joints_t.items()
    }
    ddim_motion = (
        ddim_motion_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
        if ddim_motion_t is not None
        else None
    )
    ddim_joints = (
        ddim_joints_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
        if ddim_joints_t is not None
        else None
    )

    source_metrics: dict[str, dict[str, float]] = {
        "gt": _source_metrics_np(
            gt_joints,
            gt_joints,
            obj_pos,
            seq_len,
            event,
            fps=fps,
            window_k=window_k,
            transition_radius=transition_radius,
        ),
        "full_ddpm": _source_metrics_np(
            full_joints,
            gt_joints,
            obj_pos,
            seq_len,
            event,
            fps=fps,
            window_k=window_k,
            transition_radius=transition_radius,
        ),
    }
    for t in timesteps:
        source_metrics[f"one_step_t{t}"] = _source_metrics_np(
            one_step_joints[int(t)],
            gt_joints,
            obj_pos,
            seq_len,
            event,
            fps=fps,
            window_k=window_k,
            transition_radius=transition_radius,
        )
        source_metrics[f"trunc_t{t}"] = _source_metrics_np(
            trunc_joints[int(t)],
            gt_joints,
            obj_pos,
            seq_len,
            event,
            fps=fps,
            window_k=window_k,
            transition_radius=transition_radius,
        )
    for t in log_timesteps:
        if int(t) in intermediate_joints:
            source_metrics[f"full_x0pred_t{t}"] = _source_metrics_np(
                intermediate_joints[int(t)],
                gt_joints,
                obj_pos,
                seq_len,
                event,
                fps=fps,
                window_k=window_k,
                transition_radius=transition_radius,
            )
    if ddim_joints is not None:
        source_metrics["ddim_eta0"] = _source_metrics_np(
            ddim_joints,
            gt_joints,
            obj_pos,
            seq_len,
            event,
            fps=fps,
            window_k=window_k,
            transition_radius=transition_radius,
        )

    return ClipOutputs(
        index=int(ordinal),
        subset=str(batch["subset"][0]),
        seq_id=str(batch["seq_id"][0]),
        text=str(batch["text"][0]),
        seq_len=seq_len,
        event=event,
        gt_motion=gt_motion,
        gt_joints=gt_joints,
        object_positions=obj_pos,
        object_rotations=obj_rot,
        object_pc=obj_pc,
        one_step_motion=one_step_motion,
        one_step_joints=one_step_joints,
        trunc_motion=trunc_motion,
        trunc_joints=trunc_joints,
        full_ddpm_motion=full_motion,
        full_ddpm_joints=full_joints,
        full_intermediate_motion=intermediate_motion,
        full_intermediate_joints=intermediate_joints,
        ddim_motion=ddim_motion,
        ddim_joints=ddim_joints,
        metrics=source_metrics,
    )


def _render_ladder_video(
    clip: ClipOutputs,
    sources: "OrderedDict[str, np.ndarray]",
    output_path: Path,
    start: int,
    end: int,
    title: str,
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
        list(sources.values()),
        clip.object_positions[: clip.seq_len],
        object_cloud,
        start,
        end,
    )
    n_cols = len(sources)
    fig = plt.figure(figsize=(3.15 * n_cols, 5.2))
    axes = [fig.add_subplot(1, n_cols, i + 1, projection="3d") for i in range(n_cols)]
    artists: list[dict[str, Any]] = []
    for ax, label in zip(axes, sources.keys()):
        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[2] - max_range, center[2] + max_range)
        ax.set_zlim(center[1] - max_range, center[1] + max_range)
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_zlabel("Y")
        ax.view_init(elev=elev, azim=azim)
        scatter = ax.scatter([], [], [], c="#1f77b4", s=14)
        lines = [ax.plot([], [], [], c="0.35", linewidth=1.0)[0] for _ in SKELETON_CONNECTIONS]
        obj = ax.scatter([], [], [], c="#d62728", s=2 if object_cloud is not None else 26, alpha=0.55)
        title_artist = ax.set_title(label, fontsize=8)
        artists.append({"scatter": scatter, "lines": lines, "obj": obj, "title": title_artist})
    wrapped_text = "\n".join(textwrap.wrap(_short_text(clip.text, 145), width=145))
    event = clip.event
    fig.suptitle(
        f"{title} | {clip.subset}/{clip.seq_id} | {event.kind} {event.part} @ frame {event.frame}\n{wrapped_text}",
        fontsize=9,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.84))

    def update(frame_idx: int) -> list[Any]:
        frame = frames[frame_idx]
        updated: list[Any] = []
        for joints, label, ax_art in zip(sources.values(), sources.keys(), artists):
            j = joints[frame]
            ax_art["scatter"]._offsets3d = (j[:, 0], j[:, 2], j[:, 1])
            updated.append(ax_art["scatter"])
            for (a, b), line in zip(SKELETON_CONNECTIONS, ax_art["lines"]):
                line.set_data([j[a, 0], j[b, 0]], [j[a, 2], j[b, 2]])
                line.set_3d_properties([j[a, 1], j[b, 1]])
                updated.append(line)
            if object_cloud is not None:
                obj = object_cloud[frame]
                ax_art["obj"]._offsets3d = (obj[:, 0], obj[:, 2], obj[:, 1])
            else:
                p = clip.object_positions[frame]
                ax_art["obj"]._offsets3d = ([p[0]], [p[2]], [p[1]])
            updated.append(ax_art["obj"])
            marker = " | EVENT" if frame == event.frame else ""
            ax_art["title"].set_text(f"{label}\nframe {frame}{marker}")
            updated.append(ax_art["title"])
        return updated

    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=False, repeat=False)
    try:
        anim.save(str(output_path), writer="ffmpeg", fps=fps, dpi=dpi)
    finally:
        plt.close(fig)
    print(f"  saved {output_path}")


def _source_label(prefix: str, t: int, noise_meta: dict[str, dict[str, float]]) -> str:
    m = noise_meta[str(int(t))]
    return f"{prefix}{t}\nSNR {m['snr']:.2g} log {m['log_snr']:.1f}"


def _save_tensors(clip: ClipOutputs, tensor_dir: Path) -> None:
    tensor_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{clip.index:02d}_{clip.subset}_{clip.seq_id}_{clip.event.part}_{clip.event.kind}"
    payload: dict[str, Any] = {
        "gt_motion": clip.gt_motion[: clip.seq_len],
        "full_ddpm_sample": clip.full_ddpm_motion[: clip.seq_len],
        "object_positions": clip.object_positions[: clip.seq_len],
        "object_rotations": clip.object_rotations[: clip.seq_len] if clip.object_rotations is not None else np.zeros((clip.seq_len, 3), dtype=np.float32),
    }
    for t, motion in clip.one_step_motion.items():
        payload[f"recon_x0_pred_t{t}"] = motion[: clip.seq_len]
    for t, motion in clip.trunc_motion.items():
        payload[f"trunc_rollout_t{t}"] = motion[: clip.seq_len]
    for t, motion in clip.full_intermediate_motion.items():
        payload[f"full_ddpm_x0_pred_t{t}"] = motion[: clip.seq_len]
    if clip.ddim_motion is not None:
        payload["ddim_eta0_sample"] = clip.ddim_motion[: clip.seq_len]
    np.savez_compressed(tensor_dir / f"{stem}.npz", **payload)
    meta = {
        "subset": clip.subset,
        "seq_id": clip.seq_id,
        "text": clip.text,
        "seq_len": clip.seq_len,
        "event": {
            "kind": clip.event.kind,
            "part": clip.event.part,
            "frame": clip.event.frame,
            "crop_start": clip.event.crop_start,
            "crop_end": clip.event.crop_end,
            "reason": clip.event.reason,
        },
    }
    (tensor_dir / f"{stem}_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _aggregate_by_source(clips: list[ClipOutputs]) -> dict[str, dict[str, float]]:
    rows_by_source: dict[str, list[dict[str, float]]] = {}
    for clip in clips:
        for name, row in clip.metrics.items():
            rows_by_source.setdefault(name, []).append(row)
    return {name: _mean_metrics(rows) for name, rows in rows_by_source.items()}


def _metric_row(metrics: dict[str, float]) -> str:
    return (
        f"{metrics.get('body_velocity_over_gt', 0.0):.3f} | "
        f"{metrics.get('hand_velocity_over_gt', 0.0):.3f} | "
        f"{metrics.get('transition_relative_velocity_over_gt', 0.0):.3f} | "
        f"{metrics.get('positive_distance_change_over_gt', 0.0):.3f} | "
        f"{metrics.get('fft_mid', 0.0):.3f} | "
        f"{metrics.get('body_local_acceleration_p95_cm_per_frame2', 0.0):.3f}"
    )


def _classify(aggregate: dict[str, dict[str, float]], timesteps: list[int]) -> dict[str, Any]:
    def trans(source: str) -> float:
        return float(aggregate.get(source, {}).get("transition_relative_velocity_over_gt", 0.0))

    def body(source: str) -> float:
        return float(aggregate.get(source, {}).get("body_velocity_over_gt", 0.0))

    one = {t: trans(f"one_step_t{t}") for t in timesteps}
    trunc = {t: trans(f"trunc_t{t}") for t in timesteps}
    full = trans("full_ddpm")
    full_body = body("full_ddpm")
    high_one_ok = np.mean([one[t] for t in timesteps if t >= 500]) >= 0.75
    high_one_fail = np.mean([one[t] for t in timesteps if t >= 500]) < 0.65
    trunc_high_ok = np.mean([trunc[t] for t in timesteps if t >= 500]) >= 0.75
    trunc_worse_than_one = np.mean([
        trunc[t] - one[t] for t in timesteps if t >= 500
    ]) < -0.15
    t100_ok = one.get(100, 0.0) >= 0.80
    t300_bad = one.get(300, 0.0) < 0.65
    full_bad = (full < 0.70) or (full_body < 0.70)
    cases: list[str] = []
    if high_one_fail:
        cases.append("Case A: high-noise one-step failure")
    if high_one_ok and trunc_worse_than_one:
        cases.append("Case B: multi-step rollout collapse")
    if trunc_high_ok and full_bad:
        cases.append("Case C: pure-noise initialization / mode-selection failure")
    if t100_ok and t300_bad:
        cases.append("Case D: low-noise-only reconstruction artifact")
    if not cases:
        cases.append("Mixed / no single dominant case")
    return {
        "cases": cases,
        "one_step_transition_xgt": one,
        "truncated_transition_xgt": trunc,
        "full_transition_xgt": full,
        "full_body_xgt": full_body,
        "high_one_ok": bool(high_one_ok),
        "trunc_high_ok": bool(trunc_high_ok),
        "full_bad": bool(full_bad),
    }


def _write_report(
    path: Path,
    args: argparse.Namespace,
    noise_meta: dict[str, dict[str, float]],
    clips: list[ClipOutputs],
    aggregate: dict[str, dict[str, float]],
    classification: dict[str, Any],
    video_rows: list[dict[str, str]],
    timesteps: list[int],
    log_timesteps: list[int],
) -> None:
    lines: list[str] = []
    lines.append("# v18 recon ladder and truncated rollout diagnostic\n")
    lines.append(f"**Config:** `{args.config}`  ")
    lines.append(f"**Checkpoint:** `{args.ckpt}`  ")
    lines.append(f"**Output dir:** `{args.output_dir}`  ")
    lines.append(f"**cfg_scale:** {args.cfg_scale}  ")
    lines.append(f"**Timesteps:** {timesteps}  ")
    lines.append(f"**Clips:** {len(clips)}\n")

    lines.append("## 1. Noise Schedule Table\n")
    lines.append("| t | alpha_bar | sqrt_alpha_bar | sigma | SNR | logSNR |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for t in timesteps:
        m = noise_meta[str(t)]
        lines.append(
            f"| {t} | {m['alpha_bar']:.6f} | {m['sqrt_alpha_bar']:.6f} | "
            f"{m['sigma']:.6f} | {m['snr']:.4f} | {m['log_snr']:.3f} |"
        )
    lines.append("")

    lines.append("## 2. Selected Clips / Events\n")
    lines.append("| clip | subset | seq_id | event | part | frame | crop | text short |")
    lines.append("|---:|---|---|---|---|---:|---|---|")
    for clip in clips:
        ev = clip.event
        lines.append(
            f"| {clip.index} | {clip.subset} | `{clip.seq_id}` | {ev.kind} | "
            f"{ev.part} | {ev.frame} | [{ev.crop_start}, {ev.crop_end}] | "
            f"{_short_text(clip.text)} |"
        )
    lines.append("")

    lines.append("## 3. One-Step Ladder Metrics\n")
    lines.append("| source | body vel xGT | hand vel xGT | transition rel-vel xGT | closing/opening xGT | FFT mid | acc p95 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for t in timesteps:
        key = f"one_step_t{t}"
        lines.append(f"| recon_t{t} | {_metric_row(aggregate.get(key, {}))} |")
    lines.append("")

    lines.append("## 4. Truncated Rollout Ladder Metrics\n")
    lines.append("| source | body vel xGT | hand vel xGT | transition rel-vel xGT | closing/opening xGT | FFT mid | acc p95 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for t in timesteps:
        key = f"trunc_t{t}"
        lines.append(f"| trunc_t{t} | {_metric_row(aggregate.get(key, {}))} |")
    lines.append("")

    lines.append("## 5. One-Step vs Truncated vs Full DDPM\n")
    lines.append("| source | body vel xGT | hand vel xGT | transition rel-vel xGT | closing/opening xGT | FFT mid | acc p95 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for key in ["one_step_t100", "one_step_t500", "one_step_t900", "trunc_t100", "trunc_t500", "trunc_t900", "full_ddpm", "ddim_eta0"]:
        if key in aggregate:
            lines.append(f"| {key} | {_metric_row(aggregate.get(key, {}))} |")
    lines.append("")

    lines.append("## 6. Full DDPM Intermediate x0 Trajectory\n")
    lines.append("| sampling t | body vel xGT | hand vel xGT | transition rel-vel xGT | closing/opening xGT | FFT mid | acc p95 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for t in log_timesteps:
        key = f"full_x0pred_t{t}"
        if key in aggregate:
            lines.append(f"| {t} | {_metric_row(aggregate.get(key, {}))} |")
    lines.append(f"| final full DDPM | {_metric_row(aggregate.get('full_ddpm', {}))} |")
    lines.append("")

    lines.append("## 7. Rendered Video List\n")
    lines.append("| clip | type | video path |")
    lines.append("|---:|---|---|")
    for row in video_rows:
        lines.append(f"| {row['clip']} | {row['type']} | `{row['path']}` |")
    lines.append("")

    lines.append("## 8. One-Step Ladder Visual Verdict\n")
    lines.append("| clip | t100 | t300 | t500 | t700 | t900 | estimated transition loss point |")
    lines.append("|---:|---|---|---|---|---|---|")
    for clip in clips:
        vals = {
            t: clip.metrics[f"one_step_t{t}"]["transition_relative_velocity_over_gt"]
            for t in timesteps
        }
        loss = "not lost through t900"
        for t in timesteps:
            if vals[t] < 0.65:
                loss = f"around t{t}"
                break
        lines.append(
            f"| {clip.index} | "
            + " | ".join(f"{vals[t]:.2f}" for t in timesteps)
            + f" | {loss} |"
        )
    lines.append("")

    lines.append("## 9. Truncated Rollout Visual Verdict\n")
    lines.append("| clip | trunc_t100 | trunc_t300 | trunc_t500 | trunc_t700 | trunc_t900 | reading |")
    lines.append("|---:|---:|---:|---:|---:|---:|---|")
    for clip in clips:
        vals = {
            t: clip.metrics[f"trunc_t{t}"]["transition_relative_velocity_over_gt"]
            for t in timesteps
        }
        one_vals = {
            t: clip.metrics[f"one_step_t{t}"]["transition_relative_velocity_over_gt"]
            for t in timesteps
        }
        reading = "multi-step holds transition"
        if np.mean([vals[t] - one_vals[t] for t in timesteps if t >= 500]) < -0.15:
            reading = "multi-step smooths vs one-step"
        lines.append(
            f"| {clip.index} | "
            + " | ".join(f"{vals[t]:.2f}" for t in timesteps)
            + f" | {reading} |"
        )
    lines.append("")

    lines.append("## 10. Full DDPM Trajectory Verdict\n")
    full_rows = [aggregate.get(f"full_x0pred_t{t}", {}) for t in log_timesteps]
    if full_rows:
        start = aggregate.get("full_x0pred_t900", {}).get("transition_relative_velocity_over_gt", 0.0)
        mid = aggregate.get("full_x0pred_t500", {}).get("transition_relative_velocity_over_gt", 0.0)
        end = aggregate.get("full_ddpm", {}).get("transition_relative_velocity_over_gt", 0.0)
        if start < 0.65 and end < 0.75:
            lines.append(
                f"Full sampling x0 estimates are already smooth at high noise "
                f"(t900 transition rel-vel xGT {start:.2f}) and remain weak by "
                f"the final sample ({end:.2f})."
            )
        elif mid > start and end < mid - 0.15:
            lines.append(
                f"Full sampling improves by mid trajectory but loses dynamics late "
                f"(t500 {mid:.2f} -> final {end:.2f})."
            )
        else:
            lines.append(
                f"Full sampling trajectory is mixed: t900 {start:.2f}, "
                f"t500 {mid:.2f}, final {end:.2f}."
            )
    lines.append("")

    lines.append("## 11. Final Root-Cause Classification\n")
    for case in classification["cases"]:
        lines.append(f"- **{case}**")
    lines.append("")
    lines.append(
        f"Aggregate one-step transition xGT: "
        + ", ".join(f"t{t}={classification['one_step_transition_xgt'][t]:.2f}" for t in timesteps)
    )
    lines.append(
        f"Aggregate truncated transition xGT: "
        + ", ".join(f"t{t}={classification['truncated_transition_xgt'][t]:.2f}" for t in timesteps)
    )
    lines.append(
        f"Full DDPM final transition xGT={classification['full_transition_xgt']:.2f}, "
        f"body xGT={classification['full_body_xgt']:.2f}."
    )
    lines.append("")

    lines.append("## 12. Next-Step Recommendation\n")
    cases = " ".join(classification["cases"])
    if "Case A" in cases or "Case D" in cases:
        lines.append(
            "Prioritize v-pred / objective geometry / high-noise conditioning. "
            "Do not cite t100 recon as strong evidence by itself."
        )
    elif "Case B" in cases:
        lines.append(
            "Prioritize sampler / schedule / DDIM / self-conditioning or consistency-style training, "
            "because one-step can recover more transition than the reverse trajectory preserves."
        )
    elif "Case C" in cases:
        lines.append(
            "Prioritize better initialization / plan-conditioned prior / coarse-to-fine or anchor-conditioned sampling, "
            "because truncated rollout from real q(x_t|x0) is healthier than pure-noise sampling."
        )
    else:
        lines.append(
            "Result is mixed. Review the MP4s first; if visuals disagree with metrics, repair the transition metric/renderer before changing objective."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path, default=Path("analyses/visuals/2026-05-13_v18_recon_vs_sample/selection_metadata.json"))
    parser.add_argument("--num-candidates", type=int, default=16)
    parser.add_argument("--max-clips", type=int, default=4)
    parser.add_argument("--bucket", choices=["train", "val"], default="train")
    parser.add_argument("--balanced-subsets", action="store_true")
    parser.add_argument("--timesteps", type=str, default="100,300,500,700,900")
    parser.add_argument("--log-timesteps", type=str, default="900,700,500,300,100,0")
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--window-k", type=int, default=10)
    parser.add_argument("--transition-radius", type=int, default=5)
    parser.add_argument("--dpi", type=int, default=60)
    parser.add_argument("--object-points", type=int, default=96)
    parser.add_argument("--run-ddim", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    args = parser.parse_args()

    timesteps = _parse_ints(args.timesteps)
    log_timesteps = _parse_ints(args.log_timesteps)
    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    selection = _load_selection(args.selection_json, max_clips=int(args.max_clips))
    selected_batches = _build_selected_batches(
        cfg,
        bucket=args.bucket,
        balanced_subsets=bool(args.balanced_subsets),
        num_candidates=int(args.num_candidates),
        selection=selection,
        max_clips=int(args.max_clips),
        threshold=float(args.threshold),
    )
    model, object_encoder, z_dims = _build_model(cfg, device)
    _load_checkpoint(model, object_encoder, args.ckpt)
    clip_model = load_clip_text_encoder(
        device=device,
        model_name=str(cfg.model.text_encoder.clip_version),
        download_root=str(cfg.model.text_encoder.get("download_root", "cache/clip")),
    )
    noise_meta = _noise_metadata(model.diffusion, timesteps)
    print(f"Selected {len(selected_batches)} clips. Running on {device}.")

    clips: list[ClipOutputs] = []
    for ordinal, (batch_idx, batch, event) in enumerate(selected_batches, start=1):
        print(
            f"[{ordinal}/{len(selected_batches)}] {batch['subset'][0]}/{batch['seq_id'][0]} "
            f"{event.kind} {event.part}@{event.frame}"
        )
        clip = _run_clip(
            ordinal=ordinal,
            batch_idx=batch_idx,
            batch=batch,
            event=event,
            model=model,
            object_encoder=object_encoder,
            clip_model=clip_model,
            z_dims=z_dims,
            cfg=cfg,
            device=device,
            timesteps=timesteps,
            log_timesteps=log_timesteps,
            cfg_scale=float(args.cfg_scale),
            seed=int(args.seed),
            fps=float(args.fps),
            window_k=int(args.window_k),
            transition_radius=int(args.transition_radius),
            run_ddim=bool(args.run_ddim),
        )
        clips.append(clip)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tensor_dir = args.output_dir / "tensors"
    video_rows: list[dict[str, str]] = []
    for clip in clips:
        _save_tensors(clip, tensor_dir)
        if args.skip_render:
            continue
        stem = f"{clip.index:02d}_{clip.subset}_{clip.seq_id}_{clip.event.part}_{clip.event.kind}"
        one_sources: "OrderedDict[str, np.ndarray]" = OrderedDict()
        one_sources["GT"] = clip.gt_joints
        for t in timesteps:
            one_sources[_source_label("recon_t", t, noise_meta)] = clip.one_step_joints[int(t)]
        trunc_sources: "OrderedDict[str, np.ndarray]" = OrderedDict()
        trunc_sources["GT"] = clip.gt_joints
        for t in timesteps:
            trunc_sources[_source_label("trunc_t", t, noise_meta)] = clip.trunc_joints[int(t)]

        for kind, sources, start, end, suffix in (
            ("one-step full", one_sources, 0, clip.seq_len - 1, "onestep_ladder.mp4"),
            ("one-step crop", one_sources, clip.event.crop_start, clip.event.crop_end, "onestep_ladder_crop.mp4"),
            ("truncated full", trunc_sources, 0, clip.seq_len - 1, "truncated_rollout_ladder.mp4"),
            ("truncated crop", trunc_sources, clip.event.crop_start, clip.event.crop_end, "truncated_rollout_ladder_crop.mp4"),
        ):
            out_path = args.output_dir / f"{stem}_{suffix}"
            _render_ladder_video(
                clip,
                sources,
                output_path=out_path,
                start=start,
                end=end,
                title=kind,
                fps=float(args.fps),
                dpi=int(args.dpi),
                object_points=int(args.object_points),
                seed=int(args.seed) + clip.index,
            )
            video_rows.append({"clip": str(clip.index), "type": kind, "path": str(out_path)})

    aggregate = _aggregate_by_source(clips)
    classification = _classify(aggregate, timesteps=timesteps)
    payload = {
        "config": str(args.config),
        "ckpt": str(args.ckpt),
        "output_dir": str(args.output_dir),
        "timesteps": timesteps,
        "log_timesteps": log_timesteps,
        "noise_metadata": noise_meta,
        "selected_clips": [
            {
                "index": c.index,
                "subset": c.subset,
                "seq_id": c.seq_id,
                "text": c.text,
                "seq_len": c.seq_len,
                "event": {
                    "kind": c.event.kind,
                    "part": c.event.part,
                    "frame": c.event.frame,
                    "crop_start": c.event.crop_start,
                    "crop_end": c.event.crop_end,
                    "reason": c.event.reason,
                },
                "metrics": c.metrics,
            }
            for c in clips
        ],
        "aggregate": aggregate,
        "classification": classification,
        "videos": video_rows,
        "ddim": "enabled" if args.run_ddim else "skipped",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_report(
        args.report,
        args,
        noise_meta=noise_meta,
        clips=clips,
        aggregate=aggregate,
        classification=classification,
        video_rows=video_rows,
        timesteps=timesteps,
        log_timesteps=log_timesteps,
    )
    print(f"Wrote JSON to {args.output_json}")
    print(f"Wrote report to {args.report}")
    print(f"Output dir: {args.output_dir}")
    print("Classification:", "; ".join(classification["cases"]))


if __name__ == "__main__":
    main()
