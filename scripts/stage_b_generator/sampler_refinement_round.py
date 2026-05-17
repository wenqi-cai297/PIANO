"""Stage-B sampler-only refinement sweep with stability/contact gates.

This script keeps the v18 checkpoint fixed. It evaluates sampler/schedule
variants on the transition-heavy visual clips, filters them with hard
motion/stability/contact/plan gates, and renders a small side-by-side visual
set for the risky baseline and best candidates.
"""
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from dynamics_diagnostic import (
    _build_cond,
    _build_dataset,
    _build_model,
    _fk_from_motion_135,
)
from piano.data.dataset import collate_hoi
from piano.sampling import SamplerConfig, default_sampler_sweep, sample_with_config
from piano.utils.clip_utils import load_clip_text_encoder
from plan_condition_diagnostics import (
    _compute_metrics as _compute_plan_metrics,
    _gt_plan,
    _part_swapped_plan,
    _reversed_plan,
    _shuffled_plan,
    _target_perturbed_plan,
    _wrong_clip_plan,
    _zero_plan,
)
from recon_ladder_truncated_rollout_diagnostic import (
    LOG_TIMESTEPS_DEFAULT,
    SelectedEvent,
    _build_selected_batches,
    _fallback_event_from_batch,
    _source_metrics_np,
)
from render_objective_sampler_comparison import (
    ClipComparison,
    _auto_verdict,
    _metrics_for_method,
    _render_video,
)
from render_recon_vs_sample import (
    _axis_limits,
    _downsample_object_pc,
    _extract_plan,
    _load_checkpoint,
    _precompute_object_cloud,
    _short_text,
)
from piano.inference.visualize_motion import SKELETON_CONNECTIONS


PLAN_VARIANTS = (
    "gt",
    "zero",
    "wrong_clip",
    "shuffled_time",
    "reversed_time",
    "target_perturbed",
    "part_swapped",
)


EXTERNAL_REFERENCE_FILES = {
    "StableMoFusion": [
        "external/StableMoFusion/config/diffuser_params.yaml",
        "external/StableMoFusion/models/gaussian_diffusion_w_footskate_cleanup.py",
        "external/StableMoFusion/utils/footskate_clean.py",
    ],
    "GMD": [
        "external/guided-motion-diffusion/diffusion/gaussian_diffusion.py",
        "external/guided-motion-diffusion/sample/condition.py",
        "external/guided-motion-diffusion/sample/generate.py",
    ],
    "OmniControl": [
        "external/OmniControl/diffusion/gaussian_diffusion.py",
        "external/OmniControl/sample/generate.py",
    ],
    "k-diffusion": [
        "external/k-diffusion/k_diffusion/sampling.py",
    ],
}


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if abs(float(den)) > 1e-12 else 0.0


def _mean_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({k for row in rows for k, v in row.items() if isinstance(v, (int, float))})
    out: dict[str, float] = {}
    for key in keys:
        vals = [float(row[key]) for row in rows if key in row and np.isfinite(float(row[key]))]
        out[key] = float(np.mean(vals)) if vals else 0.0
    return out


def _format_table(rows: list[list[Any]]) -> str:
    if not rows:
        return ""
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]
    out: list[str] = []
    for r, row in enumerate(rows):
        out.append("| " + " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(row)) + " |")
        if r == 0:
            out.append("| " + " | ".join("-" * widths[i] for i in range(len(widths))) + " |")
    return "\n".join(out)


def _load_visual_selection(path: Path, max_clips: int) -> dict[str, SelectedEvent]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, SelectedEvent] = {}
    for entry in raw.get("clips", []):
        ev = entry.get("event") or {}
        seq_id = str(entry.get("seq_id", ""))
        if seq_id and ev:
            out[seq_id] = SelectedEvent(
                kind=str(ev["kind"]),
                part=str(ev["part"]),
                frame=int(ev["frame"]),
                crop_start=int(ev.get("crop_start", 0)),
                crop_end=int(ev.get("crop_end", int(ev["frame"]) + 1)),
                reason=str(ev.get("reason", entry.get("selected_reason", ""))),
            )
        if len(out) >= int(max_clips):
            break
    return out


def _selected_batches(
    cfg: Any,
    bucket: str,
    selection_json: Path,
    max_clips: int,
    num_candidates: int,
) -> list[tuple[int, dict[str, Any], SelectedEvent]]:
    selection = _load_visual_selection(selection_json, max_clips=max_clips)
    selected = _build_selected_batches(
        cfg,
        bucket=bucket,
        balanced_subsets=True,
        num_candidates=int(num_candidates),
        selection=selection,
        max_clips=int(max_clips),
        threshold=0.5,
    )
    return selected


def _gt_joints_np(batch: dict[str, Any]) -> np.ndarray:
    return batch["joints"].squeeze(0).detach().cpu().numpy().astype(np.float32)


def _object_pos_np(batch: dict[str, Any]) -> np.ndarray:
    return batch["object_positions"].squeeze(0).detach().cpu().numpy().astype(np.float32)


def _seq_mask(batch: dict[str, Any], device: torch.device) -> torch.Tensor:
    total_t = int(batch["motion"].shape[1])
    seq_len = batch["seq_len"].to(device).long()
    grid = torch.arange(total_t, device=device).view(1, total_t)
    return grid < seq_len.view(-1, 1)


def _part_to_joint(device: torch.device) -> torch.Tensor:
    return torch.tensor([20, 21, 10, 11, 0], dtype=torch.long, device=device)


def _clip_meta(batch_idx: int, batch: dict[str, Any], event: SelectedEvent) -> dict[str, Any]:
    return {
        "dataset_index": int(batch_idx),
        "subset": str(batch["subset"][0]),
        "seq_id": str(batch["seq_id"][0]),
        "text": str(batch["text"][0]),
        "seq_len": int(batch["seq_len"][0].item()),
        "event": {
            "kind": event.kind,
            "part": event.part,
            "frame": int(event.frame),
            "crop_start": int(event.crop_start),
            "crop_end": int(event.crop_end),
            "reason": event.reason,
        },
    }


def _metrics_for_joints(
    source_joints: np.ndarray,
    gt_joints: np.ndarray,
    object_pos: np.ndarray,
    seq_len: int,
    event: SelectedEvent | None,
    fps: float,
) -> dict[str, float]:
    return _metrics_for_method(
        source_joints,
        gt_joints,
        object_pos,
        int(seq_len),
        event,
        fps=float(fps),
        window_k=10,
        transition_radius=5,
    )


def _event_kind_aggregates(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for kind in ("onset", "release"):
        vals = [r["metrics"] for r in rows if r.get("event_kind") == kind]
        out[kind] = _mean_rows(vals)
    return out


def _build_plan_variants(
    plan_gt: dict[str, torch.Tensor],
    plan_other: dict[str, torch.Tensor],
    seq_len: int,
    seed: int,
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        "gt": _gt_plan(plan_gt),
        "zero": _zero_plan(plan_gt),
        "wrong_clip": _wrong_clip_plan(plan_gt, plan_other),
        "shuffled_time": _shuffled_plan(plan_gt, seed=seed),
        "reversed_time": _reversed_plan(plan_gt, T=int(seq_len)),
        "target_perturbed": _target_perturbed_plan(plan_gt, sigma_m=0.10, seed=seed),
        "part_swapped": _part_swapped_plan(plan_gt),
    }


def _repeat_cond_batch(cond: dict[str, Any], repeats: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in cond.items():
        if torch.is_tensor(value) and value.shape[0] == 1:
            out[key] = value.repeat(repeats, *([1] * (value.ndim - 1)))
        else:
            out[key] = value
    return out


def _stack_plan_batch(
    plans: dict[str, dict[str, torch.Tensor]],
    names: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    keys = plans[names[0]].keys()
    return {
        key: torch.cat([plans[name][key] for name in names], dim=0)
        for key in keys
    }


@torch.no_grad()
def _run_plan_probe(
    model: Any,
    object_encoder: Any,
    clip_model: Any,
    z_dims: Any,
    cfg: Any,
    selected: list[tuple[int, dict[str, Any], SelectedEvent]],
    sampler_cfg: SamplerConfig,
    device: torch.device,
    cfg_scale: float,
    seed: int,
    fps: float,
    max_plan_clips: int,
) -> dict[str, Any]:
    del fps
    variant_rows: dict[str, list[dict[str, float]]] = {name: [] for name in PLAN_VARIANTS}
    motion_delta_rows: dict[str, list[float]] = {name: [] for name in PLAN_VARIANTS if name != "gt"}
    plan_clip_count = min(int(max_plan_clips), len(selected))
    for ordinal in range(plan_clip_count):
        _batch_idx, batch, _event = selected[ordinal]
        other = selected[(ordinal + 1) % len(selected)][1]
        cond_base, total_t = _build_cond(batch, model, object_encoder, clip_model, z_dims, cfg, device)
        plan_gt = _extract_plan(batch, device)
        plan_other = _extract_plan(other, device)
        plans = _build_plan_variants(
            plan_gt,
            plan_other,
            seq_len=int(batch["seq_len"][0].item()),
            seed=int(seed) + ordinal * 31,
        )
        num_plan_variants = len(PLAN_VARIANTS)
        plan_batch = _stack_plan_batch(plans, PLAN_VARIANTS)
        cond = {
            **_repeat_cond_batch(cond_base, num_plan_variants),
            "interaction_plan": plan_batch,
        }
        sample, _logs, _meta = sample_with_config(
            model,
            cond,
            seq_length=total_t,
            config=sampler_cfg,
            cfg_scale=float(cfg_scale),
            seed=int(seed) + ordinal * 1000,
            log_timesteps=(),
        )
        rest_offsets = batch["rest_offsets"].to(device).float().repeat(
            num_plan_variants, 1, 1,
        )
        joints = _fk_from_motion_135(sample, rest_offsets)
        gt_joints_all = batch["joints"].to(device).float().repeat(
            num_plan_variants, 1, 1, 1,
        )
        seq_mask_all = _seq_mask(batch, device).repeat(num_plan_variants, 1)
        part_to_joint = _part_to_joint(device)
        gt_motion = sample[0:1].detach()
        for p_idx, plan_name in enumerate(PLAN_VARIANTS):
            plan_one = {
                key: value[p_idx : p_idx + 1]
                for key, value in plan_batch.items()
            }
            metrics = _compute_plan_metrics(
                joints[p_idx : p_idx + 1],
                gt_joints_all[p_idx : p_idx + 1],
                seq_mask_all[p_idx : p_idx + 1],
                plan_one["anchor_time"].long(),
                plan_one["anchor_mask"].bool(),
                plan_one["anchor_part"].float(),
                plan_one["anchor_target_world"].float(),
                part_to_joint,
                window=3,
            )
            variant_rows[plan_name].append({
                k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))
            })
            if plan_name != "gt":
                motion_delta_rows[plan_name].append(
                    float((sample[p_idx : p_idx + 1] - gt_motion).pow(2).mean().sqrt().item())
                )

    aggregate = {name: _mean_rows(rows) for name, rows in variant_rows.items()}
    delta = {
        name: float(np.mean(vals)) if vals else 0.0
        for name, vals in motion_delta_rows.items()
    }
    far_gt = aggregate.get("gt", {}).get("far_unobserved_error_cm", 0.0)
    far_zero = aggregate.get("zero", {}).get("far_unobserved_error_cm", 0.0)
    far_wrong = aggregate.get("wrong_clip", {}).get("far_unobserved_error_cm", 0.0)
    return {
        "num_plan_clips": plan_clip_count,
        "variants": aggregate,
        "motion_135_delta_vs_gt": delta,
        "far_unobs_gt_cm": far_gt,
        "gt_zero_gap_cm": far_zero - far_gt,
        "gt_wrong_gap_cm": far_wrong - far_gt,
        "anchor_realization_gt_cm": aggregate.get("gt", {}).get("plan_anchor_contact_realization_cm", 0.0),
    }


def _filter_variant(
    name: str,
    agg: dict[str, float],
    event_agg: dict[str, dict[str, float]],
    plan: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    baseline_agg = baseline["aggregate"]
    baseline_events = baseline["events"]
    baseline_plan = baseline.get("plan", {})
    body = agg.get("body_velocity_over_gt", 0.0)
    hand = agg.get("hand_velocity_over_gt", 0.0)
    foot = agg.get("foot_velocity_over_gt", 0.0)
    acc = agg.get("acc_p95_over_gt", 0.0)
    jerk = agg.get("jerk_p95_over_gt", 0.0)
    fft_high = agg.get("fft_high", 0.0)
    base_high = baseline_agg.get("fft_high", 0.0)
    onset_close = event_agg.get("onset", {}).get("positive_distance_change_over_gt", 0.0)
    base_onset = baseline_events.get("onset", {}).get("positive_distance_change_over_gt", 0.0)
    release_open = event_agg.get("release", {}).get("positive_distance_change_over_gt", 0.0)
    base_release = baseline_events.get("release", {}).get("positive_distance_change_over_gt", 0.0)
    motion_gate = 0.75 <= body <= 1.05 and 0.75 <= hand <= 1.15 and foot <= 1.35
    stability_gate = acc <= 2.0 and jerk <= 3.0 and (
        base_high <= 1e-12 or fft_high <= 2.0 * base_high
    )
    transition_gate = (
        onset_close + 1e-6 >= base_onset
        and (release_open + 1e-6 >= base_release or release_open >= 0.75)
    )
    far = plan.get("far_unobs_gt_cm", 0.0)
    base_far = baseline_plan.get("far_unobs_gt_cm", far)
    plan_gate = (
        plan.get("gt_zero_gap_cm", 0.0) > 0.0
        and plan.get("gt_wrong_gap_cm", 0.0) > -1.0
        and (base_far <= 1e-12 or far <= 1.15 * base_far)
    )
    status = "candidate" if (motion_gate and stability_gate and transition_gate and plan_gate) else "reject"
    if name == "A0_ddpm_1000_cosine_default":
        status = "baseline"
    return {
        "motion_gate": motion_gate,
        "stability_gate": stability_gate,
        "transition_gate": transition_gate,
        "plan_contact_gate": plan_gate,
        "visual_gate": "needs visual review",
        "status": status,
        "notes": {
            "body_xgt": body,
            "hand_xgt": hand,
            "foot_xgt": foot,
            "acc_xgt": acc,
            "jerk_xgt": jerk,
            "fft_high": fft_high,
            "baseline_fft_high": base_high,
            "onset_closing_xgt": onset_close,
            "baseline_onset_closing_xgt": base_onset,
            "release_opening_xgt": release_open,
            "baseline_release_opening_xgt": base_release,
        },
    }


def _choose_render_variants(filters: dict[str, dict[str, Any]], aggregate: dict[str, dict[str, float]]) -> list[str]:
    chosen = ["A0_ddpm_1000_cosine_default", "A1_ddim_eta0_250_logit_normal"]
    candidates = [name for name, f in filters.items() if f["status"] == "candidate"]
    if candidates:
        best = max(candidates, key=lambda n: aggregate[n].get("body_velocity_over_gt", 0.0))
        chosen.append(best)
    else:
        non_baseline = [n for n in aggregate if n not in chosen]
        low_jitter = sorted(
            non_baseline,
            key=lambda n: (
                abs(aggregate[n].get("body_velocity_over_gt", 0.0) - 0.9),
                aggregate[n].get("jerk_p95_over_gt", 99.0),
            ),
        )
        if low_jitter:
            chosen.append(low_jitter[0])
    dpmpp = [
        n for n in aggregate
        if "dpmpp" in n and n not in chosen
    ]
    if dpmpp:
        chosen.append(min(dpmpp, key=lambda n: aggregate[n].get("jerk_p95_over_gt", 99.0)))
    out: list[str] = []
    for name in chosen:
        if name in aggregate and name not in out:
            out.append(name)
    return out[:4]


def _apply_visual_gates(filters: dict[str, dict[str, Any]], visuals: list[dict[str, Any]]) -> None:
    bad_tokens = ("jitter", "over-motion", "wrong contact geometry", "foot sliding")
    by_variant: dict[str, list[str]] = {}
    for row in visuals:
        for name, verdict in row.get("verdicts", {}).items():
            if name == "GT":
                continue
            by_variant.setdefault(name, []).append(str(verdict))
    for name, verdicts in by_variant.items():
        bad_count = sum(any(tok in v for tok in bad_tokens) for v in verdicts)
        if bad_count > 0:
            filters[name]["visual_gate"] = f"fail ({bad_count}/{len(verdicts)} clips flagged)"
            if filters[name]["status"] == "candidate":
                filters[name]["status"] = "reject"
        else:
            filters[name]["visual_gate"] = "pass"
    for name, f in filters.items():
        if f["status"] == "candidate" and name not in by_variant:
            f["visual_gate"] = "not rendered"
            f["status"] = "needs visual review"


def _render_selected(
    render_variants: list[str],
    sample_cache: dict[tuple[str, str], dict[str, Any]],
    selected: list[tuple[int, dict[str, Any], SelectedEvent]],
    visual_dir: Path,
    fps: float,
    dpi: int,
    object_points: int,
    seed: int,
    max_render_clips: int,
) -> list[dict[str, Any]]:
    visual_rows: list[dict[str, Any]] = []
    for ordinal, (_batch_idx, batch, event) in enumerate(selected[: int(max_render_clips)], start=1):
        subset = str(batch["subset"][0])
        seq_id = str(batch["seq_id"][0])
        seq_len = int(batch["seq_len"][0].item())
        gt_joints = _gt_joints_np(batch)
        object_positions = _object_pos_np(batch)
        object_rot = batch.get("object_rotations")
        object_pc = batch.get("object_pc")
        object_rot_np = (
            object_rot.squeeze(0).detach().cpu().numpy().astype(np.float32)
            if object_rot is not None else None
        )
        object_pc_np = (
            object_pc.squeeze(0).detach().cpu().numpy().astype(np.float32)
            if object_pc is not None else None
        )
        joints_by_method: dict[str, np.ndarray] = {"GT": gt_joints}
        metrics_by_method: dict[str, dict[str, float]] = {
            "GT": _metrics_for_joints(gt_joints, gt_joints, object_positions, seq_len, event, fps)
        }
        for variant in render_variants:
            cached = sample_cache[(seq_id, variant)]
            joints_by_method[variant] = cached["joints"]
            metrics_by_method[variant] = cached["metrics"]
        clip = ClipComparison(
            index=ordinal,
            subset=subset,
            seq_id=seq_id,
            text=str(batch["text"][0]),
            seq_len=seq_len,
            event=event,
            selected_reason=event.reason,
            object_positions=object_positions,
            object_rotations=object_rot_np,
            object_pc=object_pc_np,
            joints_by_method=joints_by_method,
            metrics_by_method=metrics_by_method,
        )
        stem = f"{ordinal:02d}_{subset}_{seq_id}".replace("/", "_").replace("\\", "_")
        sources = OrderedDict([("GT", gt_joints)])
        for variant in render_variants:
            sources[variant] = joints_by_method[variant]
        full_path = visual_dir / f"{stem}_full_sampler_refinement_compare.mp4"
        _render_video(
            clip,
            sources,
            full_path,
            start=0,
            end=seq_len - 1,
            title="Sampler refinement",
            fps=float(fps),
            dpi=int(dpi),
            object_points=int(object_points),
            seed=int(seed) + ordinal,
        )
        crop_path = None
        sheet_path = None
        if event is not None:
            crop_start = max(0, int(event.crop_start))
            crop_end = min(seq_len - 1, int(event.crop_end))
            crop_suffix = "onset_crop" if event.kind == "onset" else "release_crop"
            crop_path = visual_dir / f"{stem}_{event.part}_{crop_suffix}_sampler_refinement_compare.mp4"
            _render_video(
                clip,
                sources,
                crop_path,
                start=crop_start,
                end=crop_end,
                title="Sampler refinement crop",
                fps=float(fps),
                dpi=int(dpi),
                object_points=int(object_points),
                seed=int(seed) + ordinal + 900,
            )
            sheet_path = visual_dir / f"{stem}_{event.part}_{event.kind}_contact_sheet.png"
            _save_dynamic_contact_sheet(
                clip,
                sources,
                sheet_path,
                event,
                dpi=int(dpi),
                object_points=int(object_points),
                seed=int(seed) + ordinal + 1800,
            )
        verdicts = {
            name: _auto_verdict(name, metrics_by_method[name], event)
            for name in ["GT", *render_variants]
        }
        visual_rows.append({
            "clip": ordinal,
            "subset": subset,
            "seq_id": seq_id,
            "event": None if event is None else {
                "kind": event.kind,
                "part": event.part,
                "frame": int(event.frame),
            },
            "full_video": str(full_path),
            "crop_video": str(crop_path) if crop_path is not None else "",
            "contact_sheet": str(sheet_path) if sheet_path is not None else "",
            "verdicts": verdicts,
        })
    return visual_rows


def _save_dynamic_contact_sheet(
    clip: ClipComparison,
    sources: "OrderedDict[str, np.ndarray]",
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
                ax.scatter(obj[:, 0], obj[:, 2], obj[:, 1], c="#d62728", s=1.5, alpha=0.5)
            else:
                p = clip.object_positions[frame]
                ax.scatter([p[0]], [p[2]], [p[1]], c="#d62728", s=24)
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
        fontsize=9,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(output_path, dpi=dpi)
    finally:
        plt.close(fig)
    print(f"  saved {output_path}")


def _write_report(payload: dict[str, Any], md_path: Path) -> None:
    agg = payload["aggregate"]
    filters = payload["candidate_filters"]
    plan = payload["plan_condition"]
    event = payload["event_aggregate"]
    rows = [[
        "variant", "body", "hand", "foot", "acc", "jerk", "FFT mid", "FFT high",
        "onset close", "release open", "far", "GT-zero", "GT-wrong",
    ]]
    for name, metrics in agg.items():
        p = plan.get(name, {})
        ev = event.get(name, {})
        rows.append([
            name,
            f"{metrics.get('body_velocity_over_gt', 0.0):.3f}",
            f"{metrics.get('hand_velocity_over_gt', 0.0):.3f}",
            f"{metrics.get('foot_velocity_over_gt', 0.0):.3f}",
            f"{metrics.get('acc_p95_over_gt', 0.0):.3f}",
            f"{metrics.get('jerk_p95_over_gt', 0.0):.3f}",
            f"{metrics.get('fft_mid', 0.0):.3f}",
            f"{metrics.get('fft_high', 0.0):.3f}",
            f"{ev.get('onset', {}).get('positive_distance_change_over_gt', 0.0):.3f}",
            f"{ev.get('release', {}).get('positive_distance_change_over_gt', 0.0):.3f}",
            f"{p.get('far_unobs_gt_cm', 0.0):.2f}",
            f"{p.get('gt_zero_gap_cm', 0.0):.2f}",
            f"{p.get('gt_wrong_gap_cm', 0.0):.2f}",
        ])

    gate_rows = [[
        "variant", "motion", "stability", "transition", "plan/contact", "visual", "status",
    ]]
    for name, f in filters.items():
        gate_rows.append([
            name,
            "pass" if f["motion_gate"] else "fail",
            "pass" if f["stability_gate"] else "fail",
            "pass" if f["transition_gate"] else "fail",
            "pass" if f["plan_contact_gate"] else "fail",
            f["visual_gate"],
            f["status"],
        ])

    visual_rows = [["clip", "subset", "seq_id", "event", "video", "contact sheet", "verdicts"]]
    for row in payload.get("visuals", []):
        ev = row.get("event") or {}
        verdict = "; ".join(f"{k}: {v}" for k, v in row.get("verdicts", {}).items())
        visual_rows.append([
            row["clip"],
            row["subset"],
            row["seq_id"],
            f"{ev.get('kind', '')} {ev.get('part', '')} {ev.get('frame', '')}".strip(),
            row["full_video"],
            row["contact_sheet"],
            verdict,
        ])

    candidate_names = [
        name for name, f in filters.items()
        if f.get("status") == "candidate"
    ]
    if candidate_names:
        decision = (
            "B. Adopt the best passing sampler only as an eval option pending naked-eye review; "
            "do not replace v18 training/mainline yet."
        )
    else:
        decision = "A. Keep v18 DDPM as default; no sampler passes all hard gates in this sweep."

    lines = [
        "# Stage B Sampler-Only Refinement Round",
        "",
        "## Background",
        "",
        f"- Mainline config: `{payload['config']}`",
        f"- Mainline checkpoint: `{payload['ckpt']}`",
        "- v18 remains the training mainline. v23/v24 were rejected as replacements because visual and quantitative checks showed over-motion, high acceleration/jerk, or contact-direction failures.",
        "- This round changes sampling only. No training, loss, model architecture, all-7 data, self-conditioning, RF, v-pred, temporal conv, or transition loss was used.",
        "",
        "## External References",
        "",
        "- StableMoFusion: inspected diffusers scheduler config plus late footskate cleanup; used as caution that cleanup/guidance should be gated late and checked for foot artifacts.",
        "- GMD: inspected `cond_fn` / key-location guidance pattern; relevant only for a future oracle event-guidance prototype.",
        "- OmniControl: inspected spatial guidance and variance-scaled gradient update; reinforces small, variance-aware sampling-time guidance rather than global velocity forcing.",
        "- k-diffusion: inspected Karras sigma schedules and DPM++/Heun update structure; adapted only the algorithmic sigma-space idea to this VP x0-pred model.",
        "",
        "Files inspected:",
        "",
        *[f"- `{path}`" for paths in EXTERNAL_REFERENCE_FILES.values() for path in paths],
        "",
        "## Sampler Implementation Summary",
        "",
        "- Added a config-driven sampler registry in `src/piano/sampling/samplers.py`.",
        "- Implemented DDPM baseline, generalized DDIM eta sweep, logSNR/logit-normal/mild-logit/Karras timestep allocation, Heun, DPM++ 2M, and an approximate DPM++ 2M SDE.",
        "- DPM++ variants operate in adapted sigma space `y=x_t/sqrt(alpha_t)` and call the existing denoiser through nearest VP timestep indices.",
        "- DPM++ SDE uses local Gaussian noise rather than k-diffusion Brownian trees to avoid adding external runtime dependencies.",
        "",
        "## Full Metrics Table",
        "",
        _format_table(rows),
        "",
        "## Candidate Filtering Table",
        "",
        _format_table(gate_rows),
        "",
        "## Visual Report Summary",
        "",
        _format_table(visual_rows),
        "",
        "## Optional Event Guidance",
        "",
        "Not implemented in this run. The first sweep still needed a clean sampler/stability gate; event guidance remains a diagnostic-only next step if a low-jitter sampler improves body motion but fails onset/release closing/opening.",
        "",
        "## Manual Spot Check",
        "",
        payload.get("manual_visual_spot_check", {}).get(
            "summary",
            "No manual spot-check notes were recorded.",
        ),
        "",
        "## Final Decision",
        "",
        decision,
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/training/anchordiff_v18_a1_FULL_DATA.yaml"))
    parser.add_argument("--ckpt", type=Path, default=Path("runs/training/stageB_anchordiff_v18_a1_FULL_DATA/final.pt"))
    parser.add_argument("--output", type=Path, default=Path("analyses/2026-05-14_sampler_refinement_round_report.json"))
    parser.add_argument("--md", type=Path, default=Path("analyses/2026-05-14_sampler_refinement_round_report.md"))
    parser.add_argument("--visual-dir", type=Path, default=Path("analyses/visuals/2026-05-14_sampler_refinement_visuals"))
    parser.add_argument("--selection-json", type=Path, default=Path("analyses/visuals/2026-05-14_objective_sampler_geometry_visuals/objective_sampler_geometry_visual_metadata.json"))
    parser.add_argument("--bucket", type=str, default="train")
    parser.add_argument("--variant-set", choices=("minimum", "full"), default="minimum")
    parser.add_argument("--max-clips", type=int, default=10)
    parser.add_argument("--max-render-clips", type=int, default=10)
    parser.add_argument("--num-candidates", type=int, default=128)
    parser.add_argument("--plan-clips", type=int, default=2)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--render-fps", type=float, default=12.0)
    parser.add_argument("--render-dpi", type=int, default=90)
    parser.add_argument("--object-points", type=int, default=120)
    parser.add_argument("--skip-render", action="store_true")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, object_encoder, z_dims = _build_model(cfg, device)
    _load_checkpoint(model, object_encoder, args.ckpt)
    model.eval()
    object_encoder.eval()
    clip_model = load_clip_text_encoder(
        device=device,
        model_name=str(cfg.model.text_encoder.clip_version),
        download_root=str(cfg.model.text_encoder.get("download_root", "cache/clip")),
    )
    selected = _selected_batches(
        cfg,
        bucket=args.bucket,
        selection_json=args.selection_json,
        max_clips=int(args.max_clips),
        num_candidates=int(args.num_candidates),
    )
    variants = default_sampler_sweep(
        include_dpmpp=True,
        minimum_only=args.variant_set == "minimum",
    )

    per_variant_rows: dict[str, list[dict[str, Any]]] = {v.name: [] for v in variants}
    aggregate_rows: dict[str, list[dict[str, float]]] = {v.name: [] for v in variants}
    event_aggregate: dict[str, dict[str, dict[str, float]]] = {}
    plan_condition: dict[str, dict[str, Any]] = {}
    sampler_meta: dict[str, dict[str, Any]] = {}
    clips_meta = [_clip_meta(idx, batch, event) for idx, batch, event in selected]
    sample_cache: dict[tuple[str, str], dict[str, Any]] = {}

    for ordinal, (_batch_idx, batch, event) in enumerate(selected, start=1):
        print(f"[clip {ordinal}/{len(selected)}] {batch['subset'][0]} {batch['seq_id'][0]}")
        cond_base, total_t = _build_cond(batch, model, object_encoder, clip_model, z_dims, cfg, device)
        cond = {**cond_base, "interaction_plan": _extract_plan(batch, device)}
        rest_offsets = batch["rest_offsets"].to(device).float()
        seq_len = int(batch["seq_len"][0].item())
        gt_joints = _gt_joints_np(batch)
        object_pos = _object_pos_np(batch)
        seq_id = str(batch["seq_id"][0])
        for v_idx, variant in enumerate(variants):
            seed = int(args.seed) + ordinal * 10000 + v_idx * 1000
            motion, logs, meta = sample_with_config(
                model,
                cond,
                seq_length=total_t,
                config=variant,
                cfg_scale=float(args.cfg_scale),
                seed=seed,
                log_timesteps=LOG_TIMESTEPS_DEFAULT,
            )
            if variant.name not in sampler_meta:
                sampler_meta[variant.name] = {
                    **asdict(variant),
                    "actual_steps": int(meta.get("actual_steps", 0)),
                }
            joints = _fk_from_motion_135(motion, rest_offsets)
            joints_np = joints.squeeze(0).detach().cpu().numpy().astype(np.float32)
            metrics = _metrics_for_joints(
                joints_np,
                gt_joints,
                object_pos,
                seq_len,
                event,
                fps=float(args.fps),
            )
            aggregate_rows[variant.name].append(metrics)
            per_variant_rows[variant.name].append({
                "clip": ordinal,
                "subset": str(batch["subset"][0]),
                "seq_id": seq_id,
                "event_kind": event.kind,
                "event_part": event.part,
                "event_frame": int(event.frame),
                "metrics": metrics,
            })
            sample_cache[(seq_id, variant.name)] = {
                "motion": motion.detach().cpu(),
                "joints": joints_np,
                "metrics": metrics,
                "intermediate_keys": sorted(int(k) for k in logs.keys()),
            }

    aggregate = {name: _mean_rows(rows) for name, rows in aggregate_rows.items()}
    event_aggregate = {
        name: _event_kind_aggregates(rows)
        for name, rows in per_variant_rows.items()
    }

    print("[plan] running plan-condition probe")
    for variant in variants:
        plan_condition[variant.name] = _run_plan_probe(
            model,
            object_encoder,
            clip_model,
            z_dims,
            cfg,
            selected,
            variant,
            device,
            cfg_scale=float(args.cfg_scale),
            seed=int(args.seed) + 404,
            fps=float(args.fps),
            max_plan_clips=int(args.plan_clips),
        )

    baseline_name = "A0_ddpm_1000_cosine_default"
    baseline = {
        "aggregate": aggregate.get(baseline_name, {}),
        "events": event_aggregate.get(baseline_name, {}),
        "plan": plan_condition.get(baseline_name, {}),
    }
    candidate_filters = {
        name: _filter_variant(
            name,
            aggregate.get(name, {}),
            event_aggregate.get(name, {}),
            plan_condition.get(name, {}),
            baseline,
        )
        for name in aggregate
    }
    render_variants = _choose_render_variants(candidate_filters, aggregate)
    visuals: list[dict[str, Any]] = []
    if not args.skip_render:
        print(f"[render] variants: {', '.join(render_variants)}")
        visuals = _render_selected(
            render_variants,
            sample_cache,
            selected,
            args.visual_dir,
            fps=float(args.render_fps),
            dpi=int(args.render_dpi),
            object_points=int(args.object_points),
            seed=int(args.seed),
            max_render_clips=int(args.max_render_clips),
        )
    _apply_visual_gates(candidate_filters, visuals)

    payload = {
        "config": str(args.config),
        "ckpt": str(args.ckpt),
        "cfg_scale": float(args.cfg_scale),
        "device": str(device),
        "variant_set": args.variant_set,
        "variants": sampler_meta,
        "clips": clips_meta,
        "aggregate": aggregate,
        "event_aggregate": event_aggregate,
        "plan_condition": plan_condition,
        "candidate_filters": candidate_filters,
        "render_variants": render_variants,
        "visuals": visuals,
        "external_reference_files": EXTERNAL_REFERENCE_FILES,
        "skipped": {
            "event_guidance": "not implemented; reserved for diagnostic follow-up after sampler gates",
            "all7": "not used; behave/grab/intercap onboarding remains blocked",
            "training": "not run",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_report(payload, args.md)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.md}")


if __name__ == "__main__":
    main()
