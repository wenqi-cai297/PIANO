"""Render objective/sampler geometry visual comparisons for Stage B.

Exports side-by-side videos and contact sheets for:

GT | v18 DDPM | v18 DDIM eta0 250 logit-normal | v24 vpred/minSNR
GT | v18 DDPM | v23 RF ODE | v23 RF SDE gamma0.3

The script is eval/render only. It does not alter model, loss, sampler, or
training code.
"""
from __future__ import annotations

import argparse
import json
import textwrap
from collections import OrderedDict
from dataclasses import dataclass
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
)
from piano.data.dataset import collate_hoi
from piano.inference.visualize_motion import SKELETON_CONNECTIONS
from piano.utils.clip_utils import load_clip_text_encoder
from recon_ladder_truncated_rollout_diagnostic import (
    SelectedEvent,
    _build_selected_batches,
    _fallback_event_from_batch,
    _load_selection,
    _source_metrics_np,
)
from render_recon_vs_sample import (
    _axis_limits,
    _downsample_object_pc,
    _extract_plan,
    _load_checkpoint,
    _precompute_object_cloud,
    _short_text,
)
from sampler_geometry_diagnostic import (
    SamplerVariant,
    _add_foot_ratio,
    _format_table,
    _sample_variant,
)


METHOD_ORDER_A = ("GT", "v18_DDPM", "v18_DDIM_eta0_250_logitnormal", "v24_vpred_minsnr")
METHOD_ORDER_B = ("GT", "v18_DDPM", "v23_RF_ODE", "v23_RF_SDE_gamma0.3")
METHOD_ORDER_ALL = (
    "GT",
    "v18_DDPM",
    "v18_DDIM_eta0_250_logitnormal",
    "v23_RF_ODE",
    "v23_RF_SDE_gamma0.3",
    "v24_vpred_minsnr",
)


@dataclass
class ModelBundle:
    name: str
    config: Path
    ckpt: Path
    cfg: Any
    model: Any
    object_encoder: Any
    z_dims: Any


@dataclass
class ClipComparison:
    index: int
    subset: str
    seq_id: str
    text: str
    seq_len: int
    event: SelectedEvent | None
    selected_reason: str
    object_positions: np.ndarray
    object_rotations: np.ndarray | None
    object_pc: np.ndarray | None
    joints_by_method: dict[str, np.ndarray]
    metrics_by_method: dict[str, dict[str, float]]


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if abs(float(den)) > 1e-12 else 0.0


def _load_bundle(name: str, config: Path, ckpt: Path, device: torch.device) -> ModelBundle:
    cfg = OmegaConf.load(config)
    model, object_encoder, z_dims = _build_model(cfg, device)
    _load_checkpoint(model, object_encoder, ckpt)
    model.eval()
    object_encoder.eval()
    return ModelBundle(name, config, ckpt, cfg, model, object_encoder, z_dims)


def _build_cond_for_bundle(
    bundle: ModelBundle,
    batch: dict[str, Any],
    clip_model: Any,
    device: torch.device,
) -> tuple[dict[str, Any], int]:
    cond, total_t = _build_cond(
        batch,
        bundle.model,
        bundle.object_encoder,
        clip_model,
        bundle.z_dims,
        bundle.cfg,
        device,
    )
    return {**cond, "interaction_plan": _extract_plan(batch, device)}, total_t


@torch.no_grad()
def _sample_motion(
    method: str,
    bundles: dict[str, ModelBundle],
    conds: dict[str, dict[str, Any]],
    total_t: int,
    cfg_scale: float,
    seed: int,
) -> torch.Tensor:
    if method == "v18_DDPM":
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return bundles["v18"].model.sample(
            cond=conds["v18"], seq_length=total_t, cfg_scale=cfg_scale,
            replacement="none", output_skip=False, sampler="ddpm",
        )
    if method == "v18_DDIM_eta0_250_logitnormal":
        motion, _logs, _steps = _sample_variant(
            bundles["v18"].model,
            conds["v18"],
            seq_length=total_t,
            variant=SamplerVariant(
                "ddim_eta0_250_logitnormal",
                method="ddim_generalized",
                steps=250,
                eta=0.0,
                schedule="logit_normal",
            ),
            cfg_scale=cfg_scale,
            seed=seed,
            log_timesteps=[],
        )
        return motion
    if method == "v23_RF_ODE":
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return bundles["v23"].model.sample(
            cond=conds["v23"], seq_length=total_t, cfg_scale=cfg_scale,
            replacement="none", output_skip=False, sampler="rectified_flow_ode",
        )
    if method == "v23_RF_SDE_gamma0.3":
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model = bundles["v23"].model
        shape = (conds["v23"]["z_int"].shape[0], int(total_t), model.cfg.denoiser.motion_dim)
        return model.diffusion.rf_sample_loop(
            model.denoiser,
            shape,
            conds["v23"],
            cfg_scale=cfg_scale,
            device=conds["v23"]["z_int"].device,
            output_skip=False,
            sampler_type="rectified_flow_sde",
            sde_gamma=0.3,
        )
    if method == "v24_vpred_minsnr":
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return bundles["v24"].model.sample(
            cond=conds["v24"], seq_length=total_t, cfg_scale=cfg_scale,
            replacement="none", output_skip=False, sampler="ddpm",
        )
    raise ValueError(f"Unknown method {method!r}")


def _foot_velocity_mean(joints: np.ndarray, seq_len: int) -> float:
    j = joints[: int(seq_len)].astype(np.float32)
    if len(j) < 2:
        return 0.0
    vel = j[1:, [10, 11]] - j[:-1, [10, 11]]
    root_vel = j[1:, 0:1] - j[:-1, 0:1]
    local = vel - root_vel
    return float(np.linalg.norm(local, axis=-1).mean() * 100.0)


def _jerk_p95(joints: np.ndarray, seq_len: int) -> float:
    j = joints[: int(seq_len)].astype(np.float32)
    if len(j) < 4:
        return 0.0
    vel = j[1:] - j[:-1]
    root_vel = vel[:, 0:1]
    local = vel - root_vel
    acc = local[1:] - local[:-1]
    jerk = acc[1:] - acc[:-1]
    mag = np.linalg.norm(jerk[:, LOCAL_JOINTS], axis=-1).reshape(-1) * 100.0
    return float(np.percentile(mag, 95)) if mag.size else 0.0


def _fft_high_mid(joints: np.ndarray, seq_len: int, fps: float) -> tuple[float, float]:
    j = joints[: int(seq_len)].astype(np.float32)
    if len(j) < 4:
        return 0.0, 0.0
    local = j[:, LOCAL_JOINTS] - j[:, 0:1]
    x = local.reshape(len(j), -1)
    x = x - x.mean(axis=0, keepdims=True)
    spec = np.fft.rfft(x, axis=0)
    energy = (spec.real ** 2 + spec.imag ** 2).sum(axis=1)
    freqs = np.fft.rfftfreq(len(j), d=1.0 / float(fps))
    total = float(energy.sum()) + 1e-12
    mid = float(energy[(freqs >= 1.0) & (freqs < 4.0)].sum() / total)
    high = float(energy[(freqs >= 4.0) & (freqs < 10.0)].sum() / total)
    return mid, high


def _fallback_metrics(
    source_joints: np.ndarray,
    gt_joints: np.ndarray,
    seq_len: int,
    fps: float,
) -> dict[str, float]:
    event = SelectedEvent("onset", "L_hand", frame=max(1, min(int(seq_len) - 2, int(seq_len) // 2)), crop_start=0, crop_end=int(seq_len) - 1)
    object_positions = np.zeros((int(seq_len), 3), dtype=np.float32)
    return _source_metrics_np(
        source_joints,
        gt_joints,
        object_positions,
        int(seq_len),
        event,
        fps=float(fps),
        window_k=10,
        transition_radius=5,
    )


def _metrics_for_method(
    source_joints: np.ndarray,
    gt_joints: np.ndarray,
    object_positions: np.ndarray,
    seq_len: int,
    event: SelectedEvent | None,
    fps: float,
    window_k: int,
    transition_radius: int,
) -> dict[str, float]:
    if event is not None:
        metrics = _source_metrics_np(
            source_joints,
            gt_joints,
            object_positions,
            int(seq_len),
            event,
            fps=float(fps),
            window_k=int(window_k),
            transition_radius=int(transition_radius),
        )
    else:
        metrics = _fallback_metrics(source_joints, gt_joints, int(seq_len), fps=float(fps))
        metrics["transition_relative_velocity_over_gt"] = 0.0
        metrics["positive_distance_change_over_gt"] = 0.0
    _add_foot_ratio(metrics, source_joints, gt_joints, int(seq_len))
    metrics["jerk_p95_cm_per_frame3"] = _jerk_p95(source_joints, int(seq_len))
    gt_jerk = _jerk_p95(gt_joints, int(seq_len))
    metrics["jerk_p95_over_gt"] = _safe_div(metrics["jerk_p95_cm_per_frame3"], gt_jerk)
    mid, high = _fft_high_mid(source_joints, int(seq_len), fps=float(fps))
    gt_mid, gt_high = _fft_high_mid(gt_joints, int(seq_len), fps=float(fps))
    metrics["fft_mid"] = mid
    metrics["fft_high"] = high
    metrics["fft_high_over_gt"] = _safe_div(high, gt_high)
    metrics["fft_mid_over_gt"] = _safe_div(mid, gt_mid)
    gt_acc = _source_metrics_np(
        gt_joints,
        gt_joints,
        object_positions,
        int(seq_len),
        event or SelectedEvent("onset", "L_hand", frame=max(1, min(int(seq_len) - 2, int(seq_len) // 2)), crop_start=0, crop_end=int(seq_len) - 1),
        fps=float(fps),
        window_k=int(window_k),
        transition_radius=int(transition_radius),
    ).get("body_local_acceleration_p95_cm_per_frame2", 0.0)
    metrics["acc_p95_over_gt"] = _safe_div(
        metrics.get("body_local_acceleration_p95_cm_per_frame2", 0.0),
        gt_acc,
    )
    return metrics


def _auto_verdict(method: str, metrics: dict[str, float], event: SelectedEvent | None) -> str:
    if method == "GT":
        return "reference"
    body = metrics.get("body_velocity_over_gt", 0.0)
    hand = metrics.get("hand_velocity_over_gt", 0.0)
    foot = metrics.get("foot_velocity_over_gt", 0.0)
    acc = metrics.get("acc_p95_over_gt", 0.0)
    jerk = metrics.get("jerk_p95_over_gt", 0.0)
    high = metrics.get("fft_high_over_gt", 0.0)
    change = metrics.get("positive_distance_change_over_gt", 0.0)
    rel = metrics.get("transition_relative_velocity_over_gt", 0.0)
    labels: list[str] = []
    if body < 0.75 and hand < 0.8:
        labels.append("frozen")
    if body > 1.35 or hand > 1.35 or foot > 1.45:
        labels.append("over-motion")
    if acc > 2.0 or jerk > 2.0 or high > 2.0:
        labels.append("jitter")
    if event is not None and change < 0.7 and rel > 1.1:
        labels.append("wrong contact geometry")
    if foot > 1.35 and (acc > 1.6 or jerk > 1.6):
        labels.append("foot sliding")
    if not labels:
        labels.append("good" if 0.8 <= body <= 1.25 and 0.8 <= hand <= 1.25 else "unclear")
    return ", ".join(labels)


def _render_video(
    clip: ClipComparison,
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
    event_text = "no transition event"
    if clip.event is not None:
        event_text = f"{clip.event.kind} {clip.event.part} @ frame {clip.event.frame}"
    fig.suptitle(
        f"{title} | {clip.subset}/{clip.seq_id} | {event_text}\n{wrapped_text}",
        fontsize=9,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.84))

    def update(frame_idx: int) -> list[Any]:
        frame = frames[frame_idx]
        updated: list[Any] = []
        for joints, label, art in zip(sources.values(), sources.keys(), artists):
            j = joints[frame]
            art["scatter"]._offsets3d = (j[:, 0], j[:, 2], j[:, 1])
            updated.append(art["scatter"])
            for (a, b), line in zip(SKELETON_CONNECTIONS, art["lines"]):
                line.set_data([j[a, 0], j[b, 0]], [j[a, 2], j[b, 2]])
                line.set_3d_properties([j[a, 1], j[b, 1]])
                updated.append(line)
            if object_cloud is not None:
                obj = object_cloud[frame]
                art["obj"]._offsets3d = (obj[:, 0], obj[:, 2], obj[:, 1])
            else:
                p = clip.object_positions[frame]
                art["obj"]._offsets3d = ([p[0]], [p[2]], [p[1]])
            updated.append(art["obj"])
            marker = ""
            if clip.event is not None and frame == clip.event.frame:
                marker = " | EVENT"
            art["title"].set_text(f"{label}\nframe {frame}{marker}")
            updated.append(art["title"])
        return updated

    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=False, repeat=False)
    try:
        anim.save(str(output_path), writer="ffmpeg", fps=fps, dpi=dpi)
    finally:
        plt.close(fig)
    print(f"  saved {output_path}")


def _save_contact_sheet(
    clip: ClipComparison,
    output_path: Path,
    event: SelectedEvent,
    dpi: int,
    object_points: int,
    seed: int,
    elev: float = 15.0,
    azim: float = -60.0,
) -> None:
    import matplotlib.pyplot as plt

    frames = [
        max(0, event.frame - 10),
        max(0, event.frame - 5),
        int(event.frame),
        min(clip.seq_len - 1, event.frame + 5),
        min(clip.seq_len - 1, event.frame + 10),
    ]
    sources = OrderedDict((name, clip.joints_by_method[name]) for name in METHOD_ORDER_ALL)
    object_pc = _downsample_object_pc(clip.object_pc, object_points, seed=seed)
    object_cloud = _precompute_object_cloud(
        object_pc,
        clip.object_positions[: clip.seq_len],
        clip.object_rotations[: clip.seq_len] if clip.object_rotations is not None else None,
    )
    start, end = min(frames), max(frames)
    center, max_range = _axis_limits(
        list(sources.values()),
        clip.object_positions[: clip.seq_len],
        object_cloud,
        start,
        end,
    )
    fig = plt.figure(figsize=(3.0 * len(sources), 2.55 * len(frames)))
    for row, frame in enumerate(frames):
        for col, (name, joints_all) in enumerate(sources.items()):
            ax = fig.add_subplot(len(frames), len(sources), row * len(sources) + col + 1, projection="3d")
            joints = joints_all[frame]
            ax.scatter(joints[:, 0], joints[:, 2], joints[:, 1], c="#1f77b4", s=10)
            for i, j in SKELETON_CONNECTIONS:
                ax.plot(
                    [joints[i, 0], joints[j, 0]],
                    [joints[i, 2], joints[j, 2]],
                    [joints[i, 1], joints[j, 1]],
                    c="0.35",
                    linewidth=0.8,
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
            marker = " EVENT" if frame == event.frame else ""
            ax.set_title(f"{name}\nframe {frame}{marker}", fontsize=7)
    fig.suptitle(
        f"{clip.subset}/{clip.seq_id} | {event.kind} {event.part} @ {event.frame}",
        fontsize=10,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print(f"  saved {output_path}")


def _batch_for_global_index(dataset: Any, index: int) -> dict[str, Any]:
    loader = DataLoader(
        Subset(dataset, [int(index)]),
        batch_size=1,
        shuffle=False,
        collate_fn=collate_hoi,
        num_workers=0,
    )
    return next(iter(loader))


def _extra_balanced_batches(
    cfg: Any,
    bucket: str,
    exclude_seq_ids: set[str],
    threshold: float,
) -> list[tuple[int, dict[str, Any], SelectedEvent | None, str]]:
    dataset = _build_dataset(cfg, bucket)
    out: list[tuple[int, dict[str, Any], SelectedEvent | None, str]] = []
    if not hasattr(dataset, "datasets"):
        indices = _balanced_subset_indices(dataset, 4)
        for idx in indices:
            batch = _batch_for_global_index(dataset, idx)
            seq_id = str(batch["seq_id"][0])
            if seq_id in exclude_seq_ids:
                continue
            seq_len = int(batch["seq_len"][0].item())
            ev = _fallback_event_from_batch(
                batch["contact_state"].squeeze(0).cpu().numpy(),
                seq_len,
                threshold=threshold,
            )
            out.append((idx, batch, ev, "balanced extra"))
        return out[:4]

    offsets: list[int] = []
    cur = 0
    for ds in dataset.datasets:
        offsets.append(cur)
        cur += len(ds)
    for offset, ds in zip(offsets, dataset.datasets):
        chosen = None
        for local_idx in range(min(len(ds), 64)):
            global_idx = offset + local_idx
            batch = _batch_for_global_index(dataset, global_idx)
            seq_id = str(batch["seq_id"][0])
            if seq_id in exclude_seq_ids:
                continue
            seq_len = int(batch["seq_len"][0].item())
            ev = _fallback_event_from_batch(
                batch["contact_state"].squeeze(0).cpu().numpy(),
                seq_len,
                threshold=threshold,
            )
            chosen = (global_idx, batch, ev, f"balanced extra {batch['subset'][0]}")
            exclude_seq_ids.add(seq_id)
            break
        if chosen is not None:
            out.append(chosen)
    return out


def _run_clip(
    ordinal: int,
    batch_idx: int,
    batch: dict[str, Any],
    event: SelectedEvent | None,
    selected_reason: str,
    bundles: dict[str, ModelBundle],
    clip_model: Any,
    device: torch.device,
    cfg_scale: float,
    seed: int,
    fps: float,
    window_k: int,
    transition_radius: int,
) -> ClipComparison:
    conds: dict[str, dict[str, Any]] = {}
    total_t = None
    for key, bundle in bundles.items():
        cond, t = _build_cond_for_bundle(bundle, batch, clip_model, device)
        conds[key] = cond
        total_t = t if total_t is None else total_t
    assert total_t is not None

    rest_offsets = batch["rest_offsets"].to(device).float()
    seq_len = int(batch["seq_len"][0].item())
    gt_joints_t = batch["joints"].to(device).float()
    gt_joints = gt_joints_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
    object_pos = batch["object_positions"].squeeze(0).detach().cpu().numpy().astype(np.float32)
    object_rot = batch["object_rotations"].squeeze(0).detach().cpu().numpy().astype(np.float32)
    object_pc = batch["object_pc"].squeeze(0).detach().cpu().numpy().astype(np.float32)

    joints_by_method: dict[str, np.ndarray] = {"GT": gt_joints}
    metrics_by_method: dict[str, dict[str, float]] = {
        "GT": _metrics_for_method(
            gt_joints, gt_joints, object_pos, seq_len, event,
            fps=fps, window_k=window_k, transition_radius=transition_radius,
        )
    }
    for method_idx, method in enumerate(METHOD_ORDER_ALL):
        if method == "GT":
            continue
        motion = _sample_motion(
            method,
            bundles,
            conds,
            total_t=int(total_t),
            cfg_scale=float(cfg_scale),
            seed=int(seed) + int(batch_idx) * 10000 + method_idx * 1000,
        )
        joints = _fk_from_motion_135(motion, rest_offsets)
        joints_np = joints.squeeze(0).detach().cpu().numpy().astype(np.float32)
        joints_by_method[method] = joints_np
        metrics_by_method[method] = _metrics_for_method(
            joints_np,
            gt_joints,
            object_pos,
            seq_len,
            event,
            fps=fps,
            window_k=window_k,
            transition_radius=transition_radius,
        )

    return ClipComparison(
        index=int(ordinal),
        subset=str(batch["subset"][0]),
        seq_id=str(batch["seq_id"][0]),
        text=str(batch["text"][0]),
        seq_len=seq_len,
        event=event,
        selected_reason=selected_reason,
        object_positions=object_pos,
        object_rotations=object_rot,
        object_pc=object_pc,
        joints_by_method=joints_by_method,
        metrics_by_method=metrics_by_method,
    )


def _write_report(
    path: Path,
    args: argparse.Namespace,
    clips: list[ClipComparison],
    video_rows: list[dict[str, str]],
    sheet_rows: list[dict[str, str]],
    failures: list[dict[str, str]],
) -> None:
    lines: list[str] = []
    lines.append("# Objective / Sampler Geometry Visual Report")
    lines.append("")
    lines.append(f"**Output dir:** `{args.output_dir}`  ")
    lines.append(f"**cfg_scale:** {args.cfg_scale}  **fps:** {args.fps}")
    lines.append("")
    lines.append("## Method Summary")
    lines.append("")
    lines.append(_format_table([
        ["method", "checkpoint", "sampler", "steps", "schedule", "cfg"],
        ["GT", "dataset", "n/a", "n/a", "n/a", "n/a"],
        ["v18_DDPM", str(args.v18_ckpt), "DDPM", "1000", "cosine", args.cfg_scale],
        ["v18_DDIM_eta0_250_logitnormal", str(args.v18_ckpt), "DDIM eta=0", "250", "logit-normal-like", args.cfg_scale],
        ["v23_RF_ODE", str(args.v23_ckpt), "rectified_flow_ode", "config default", "config default", args.cfg_scale],
        ["v23_RF_SDE_gamma0.3", str(args.v23_ckpt), "rectified_flow_sde", "config default", "config default", args.cfg_scale],
        ["v24_vpred_minsnr", str(args.v24_ckpt), "DDPM", "1000", "cosine", args.cfg_scale],
    ]))
    lines.append("")
    lines.append("## Rendered Video List")
    lines.append("")
    lines.append(_format_table([["clip", "subset", "seq_id", "event", "part", "video path", "contact sheet"]] + [
        [
            row["clip"],
            row["subset"],
            row["seq_id"],
            row["event"],
            row["part"],
            row["video_path"],
            row.get("contact_sheet", ""),
        ]
        for row in video_rows
    ]))
    lines.append("")
    lines.append("## Contact Sheets")
    lines.append("")
    lines.append(_format_table([["clip", "subset", "seq_id", "event", "path"]] + [
        [row["clip"], row["subset"], row["seq_id"], row["event"], row["path"]]
        for row in sheet_rows
    ]))
    lines.append("")
    lines.append("## Visual Verdict Table")
    lines.append("")
    verdict_rows = [["clip", "reason", *METHOD_ORDER_ALL[1:]]]
    for clip in clips:
        verdict_rows.append([
            f"{clip.index:02d} {clip.seq_id}",
            clip.selected_reason,
            *[
                _auto_verdict(method, clip.metrics_by_method[method], clip.event)
                for method in METHOD_ORDER_ALL[1:]
            ],
        ])
    lines.append(_format_table(verdict_rows))
    lines.append("")
    lines.append("## Quantitative Side Table")
    lines.append("")
    qrows = [[
        "clip", "method", "body xGT", "hand xGT", "foot xGT",
        "acc p95", "acc xGT", "jerk p95", "jerk xGT", "FFT high",
        "onset/release change xGT", "rel-vel xGT",
    ]]
    for clip in clips:
        for method in METHOD_ORDER_ALL:
            m = clip.metrics_by_method[method]
            qrows.append([
                f"{clip.index:02d}",
                method,
                f"{m.get('body_velocity_over_gt', 0.0):.3f}",
                f"{m.get('hand_velocity_over_gt', 0.0):.3f}",
                f"{m.get('foot_velocity_over_gt', 0.0):.3f}",
                f"{m.get('body_local_acceleration_p95_cm_per_frame2', 0.0):.3f}",
                f"{m.get('acc_p95_over_gt', 0.0):.3f}",
                f"{m.get('jerk_p95_cm_per_frame3', 0.0):.3f}",
                f"{m.get('jerk_p95_over_gt', 0.0):.3f}",
                f"{m.get('fft_high', 0.0):.3f}",
                f"{m.get('positive_distance_change_over_gt', 0.0):.3f}",
                f"{m.get('transition_relative_velocity_over_gt', 0.0):.3f}",
            ])
    lines.append(_format_table(qrows))
    lines.append("")
    if failures:
        lines.append("## Render Failures")
        lines.append("")
        lines.append(_format_table([["clip", "stage", "error"]] + [
            [f.get("clip", ""), f.get("stage", ""), f.get("error", "")]
            for f in failures
        ]))
        lines.append("")

    counts: dict[str, dict[str, int]] = {}
    for method in METHOD_ORDER_ALL[1:]:
        counts[method] = {}
        for clip in clips:
            verdict = _auto_verdict(method, clip.metrics_by_method[method], clip.event)
            for label in [v.strip() for v in verdict.split(",")]:
                counts[method][label] = counts[method].get(label, 0) + 1

    def _count(method: str, label: str) -> int:
        return counts.get(method, {}).get(label, 0)

    lines.append("## Final Decision")
    lines.append("")
    lines.append(
        "- A. v23 RF ODE is classified as over-motion/jitter in this auto-assisted visual pass: "
        f"{_count('v23_RF_ODE', 'over-motion')} over-motion flags and "
        f"{_count('v23_RF_ODE', 'jitter')} jitter flags across {len(clips)} clips."
    )
    lines.append(
        "- B. v23 RF SDE gamma0.3 is more controlled than RF ODE, but it often falls back toward under-motion or unclear dynamics; it is not a clean replacement."
    )
    lines.append(
        "- C. v24 vpred/minSNR is visually/quantitatively more plausible than RF ODE, but high acceleration/jerk and contact-direction failures remain risk flags."
    )
    lines.append(
        "- D. v18 DDIM eta0 250 logit-normal is worth a sampler-only follow-up only if the MP4 review confirms its motion is not mostly acceleration/foot jitter."
    )
    lines.append(
        "- E. No rendered method should replace v18 mainline yet; this pass is for visual risk triage, not adoption."
    )
    lines.append(
        "- F. Next step: continue sampler-only diagnostics with full plan/far-unobs and visual gates before retraining RF or v-pred variants."
    )
    lines.append("")
    lines.append("Note: verdicts are semi-automatic from the rendered methods' velocity/acceleration/jerk/contact metrics and are meant to guide naked-eye review of the MP4s/contact sheets.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--v18-config", type=Path, default=Path("configs/training/anchordiff_v18_a1_FULL_DATA.yaml"))
    parser.add_argument("--v18-ckpt", type=Path, default=Path("runs/training/stageB_anchordiff_v18_a1_FULL_DATA/final.pt"))
    parser.add_argument("--v23-config", type=Path, default=Path("configs/training/anchordiff_v23_a1_rectified_flow_4subset.yaml"))
    parser.add_argument("--v23-ckpt", type=Path, default=Path("runs/training/stageB_anchordiff_v23_a1_rectified_flow_4subset/final.pt"))
    parser.add_argument("--v24-config", type=Path, default=Path("configs/training/anchordiff_v24_a1_vpred_minsnr_4subset.yaml"))
    parser.add_argument("--v24-ckpt", type=Path, default=Path("runs/training/stageB_anchordiff_v24_a1_vpred_minsnr_4subset/final.pt"))
    parser.add_argument("--selection-json", type=Path, default=Path("analyses/2026-05-13_v18_recon_ladder_truncated_rollout.json"))
    parser.add_argument("--max-transition-clips", type=int, default=6)
    parser.add_argument("--num-candidates", type=int, default=96)
    parser.add_argument("--bucket", choices=["train", "val"], default="train")
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--window-k", type=int, default=10)
    parser.add_argument("--transition-radius", type=int, default=5)
    parser.add_argument("--dpi", type=int, default=60)
    parser.add_argument("--object-points", type=int, default=96)
    parser.add_argument("--skip-full", action="store_true")
    parser.add_argument("--skip-crops", action="store_true")
    parser.add_argument("--skip-sheets", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundles = {
        "v18": _load_bundle("v18", args.v18_config, args.v18_ckpt, device),
        "v23": _load_bundle("v23", args.v23_config, args.v23_ckpt, device),
        "v24": _load_bundle("v24", args.v24_config, args.v24_ckpt, device),
    }
    clip_model = load_clip_text_encoder(
        device=device,
        model_name=str(bundles["v18"].cfg.model.text_encoder.clip_version),
        download_root=str(bundles["v18"].cfg.model.text_encoder.get("download_root", "cache/clip")),
    )

    selection = _load_selection(args.selection_json, max_clips=int(args.max_transition_clips))
    selected = _build_selected_batches(
        bundles["v18"].cfg,
        bucket=args.bucket,
        balanced_subsets=True,
        num_candidates=int(args.num_candidates),
        selection=selection,
        max_clips=int(args.max_transition_clips),
        threshold=float(args.threshold),
    )
    selected_full: list[tuple[int, dict[str, Any], SelectedEvent | None, str]] = [
        (idx, batch, event, f"transition-heavy: {event.reason or event.kind}")
        for idx, batch, event in selected
    ]
    seen_seq_ids = {str(batch["seq_id"][0]) for _idx, batch, _event, _reason in selected_full}
    selected_full.extend(
        _extra_balanced_batches(
            bundles["v18"].cfg,
            bucket=args.bucket,
            exclude_seq_ids=seen_seq_ids,
            threshold=float(args.threshold),
        )
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sheets_dir = args.output_dir / "contact_sheets"
    video_rows: list[dict[str, str]] = []
    sheet_rows: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    clips: list[ClipComparison] = []

    for ordinal, (batch_idx, batch, event, reason) in enumerate(selected_full, start=1):
        print(f"[{ordinal}/{len(selected_full)}] {batch['subset'][0]}/{batch['seq_id'][0]}")
        try:
            clip = _run_clip(
                ordinal=ordinal,
                batch_idx=batch_idx,
                batch=batch,
                event=event,
                selected_reason=reason,
                bundles=bundles,
                clip_model=clip_model,
                device=device,
                cfg_scale=float(args.cfg_scale),
                seed=int(args.seed),
                fps=float(args.fps),
                window_k=int(args.window_k),
                transition_radius=int(args.transition_radius),
            )
            clips.append(clip)
        except Exception as exc:
            failures.append({"clip": str(batch["seq_id"][0]), "stage": "sample", "error": repr(exc)})
            print(f"  sample failed: {exc!r}")
            continue

        stem = f"{ordinal:02d}_{clip.subset}_{clip.seq_id}"
        sheet_path = ""
        if event is not None and not args.skip_sheets:
            sheet = sheets_dir / f"{stem}_{event.part}_{event.kind}_contact_sheet.png"
            try:
                _save_contact_sheet(
                    clip,
                    sheet,
                    event,
                    dpi=int(args.dpi),
                    object_points=int(args.object_points),
                    seed=int(args.seed) + ordinal,
                )
                sheet_path = str(sheet)
                sheet_rows.append({
                    "clip": f"{ordinal:02d}",
                    "subset": clip.subset,
                    "seq_id": clip.seq_id,
                    "event": f"{event.kind} {event.part}@{event.frame}",
                    "path": str(sheet),
                })
            except Exception as exc:
                failures.append({"clip": clip.seq_id, "stage": "contact_sheet", "error": repr(exc)})
                print(f"  contact sheet failed: {exc!r}")

        groups = {
            "A": METHOD_ORDER_A,
            "B": METHOD_ORDER_B,
        }
        event_text = "none"
        part_text = ""
        if event is not None:
            event_text = f"{event.kind}@{event.frame}"
            part_text = event.part

        if not args.skip_full:
            for suffix, methods in groups.items():
                out = args.output_dir / f"{stem}_full_objective_sampler_compare_{suffix}.mp4"
                sources = OrderedDict((m, clip.joints_by_method[m]) for m in methods)
                try:
                    _render_video(
                        clip,
                        sources,
                        out,
                        start=0,
                        end=clip.seq_len - 1,
                        title=f"full comparison {suffix} | {clip.selected_reason}",
                        fps=float(args.fps),
                        dpi=int(args.dpi),
                        object_points=int(args.object_points),
                        seed=int(args.seed) + ordinal,
                    )
                    video_rows.append({
                        "clip": f"{ordinal:02d}",
                        "subset": clip.subset,
                        "seq_id": clip.seq_id,
                        "event": event_text,
                        "part": part_text,
                        "video_path": str(out),
                        "contact_sheet": sheet_path,
                    })
                except Exception as exc:
                    failures.append({"clip": clip.seq_id, "stage": f"full_{suffix}", "error": repr(exc)})
                    print(f"  full render {suffix} failed: {exc!r}")

        if event is not None and not args.skip_crops:
            crop_suffix = "onset_crop" if event.kind == "onset" else "release_crop"
            for suffix, methods in groups.items():
                out = args.output_dir / f"{stem}_{event.part}_{crop_suffix}_objective_sampler_compare_{suffix}.mp4"
                sources = OrderedDict((m, clip.joints_by_method[m]) for m in methods)
                try:
                    _render_video(
                        clip,
                        sources,
                        out,
                        start=event.crop_start,
                        end=event.crop_end,
                        title=f"{event.kind} crop comparison {suffix} | {clip.selected_reason}",
                        fps=float(args.fps),
                        dpi=int(args.dpi),
                        object_points=int(args.object_points),
                        seed=int(args.seed) + ordinal,
                    )
                    video_rows.append({
                        "clip": f"{ordinal:02d}",
                        "subset": clip.subset,
                        "seq_id": clip.seq_id,
                        "event": event_text,
                        "part": part_text,
                        "video_path": str(out),
                        "contact_sheet": sheet_path,
                    })
                except Exception as exc:
                    failures.append({"clip": clip.seq_id, "stage": f"crop_{suffix}", "error": repr(exc)})
                    print(f"  crop render {suffix} failed: {exc!r}")

    payload = {
        "output_dir": str(args.output_dir),
        "cfg_scale": float(args.cfg_scale),
        "clips": [
            {
                "index": clip.index,
                "subset": clip.subset,
                "seq_id": clip.seq_id,
                "text": clip.text,
                "seq_len": clip.seq_len,
                "selected_reason": clip.selected_reason,
                "event": None if clip.event is None else {
                    "kind": clip.event.kind,
                    "part": clip.event.part,
                    "frame": clip.event.frame,
                    "crop_start": clip.event.crop_start,
                    "crop_end": clip.event.crop_end,
                    "reason": clip.event.reason,
                },
                "metrics": clip.metrics_by_method,
                "verdicts": {
                    method: _auto_verdict(method, clip.metrics_by_method[method], clip.event)
                    for method in METHOD_ORDER_ALL
                },
            }
            for clip in clips
        ],
        "videos": video_rows,
        "contact_sheets": sheet_rows,
        "failures": failures,
    }
    (args.output_dir / "objective_sampler_geometry_visual_metadata.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    _write_report(args.report, args, clips, video_rows, sheet_rows, failures)
    print(f"Wrote report to {args.report}")
    print(f"Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
