"""Stage-B sampler geometry diagnostic for the v18 DDPM checkpoint.

This is a sampler-only experiment: no weights are changed. It compares the
default ancestral DDPM trajectory against generalized DDIM/DDPM-style updates
with different step counts and timestep allocations, inspired by ELF's explicit
separation of training and sampling configs.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from dynamics_diagnostic import (
    _build_cond,
    _build_model,
    _fk_from_motion_135,
)
from piano.models.motion_anchordiff import _extract
from piano.utils.clip_utils import load_clip_text_encoder
from recon_ladder_truncated_rollout_diagnostic import (
    LOG_TIMESTEPS_DEFAULT,
    SelectedEvent,
    _build_selected_batches,
    _source_metrics_np,
)
from render_recon_vs_sample import _extract_plan, _load_checkpoint, _short_text


@dataclass(frozen=True)
class SamplerVariant:
    name: str
    method: str
    steps: int
    eta: float = 0.0
    schedule: str = "uniform"


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if abs(float(den)) > 1e-12 else 0.0


def _make_timestep_list(
    diffusion,
    steps: int,
    schedule: str,
    seed: int,
) -> list[int]:
    max_t = int(diffusion.num_steps) - 1
    steps = max(2, int(steps))
    if schedule == "uniform":
        raw = np.linspace(max_t, 0, steps)
    elif schedule == "low_noise_dense":
        u = np.linspace(0.0, 1.0, steps)
        raw = max_t * (1.0 - u) ** 2
    elif schedule == "high_noise_dense":
        u = np.linspace(0.0, 1.0, steps)
        raw = max_t * (1.0 - u ** 2)
    elif schedule == "quadratic":
        u = np.linspace(0.0, 1.0, steps)
        raw = max_t * (1.0 - (0.5 * u + 0.5 * u ** 2))
    elif schedule == "logsnr_uniform":
        alpha = diffusion.alphas_cumprod.detach().cpu().numpy().astype(np.float64)
        logsnr = np.log(np.maximum(alpha, 1e-30)) - np.log(np.maximum(1.0 - alpha, 1e-30))
        targets = np.linspace(logsnr[max_t], logsnr[0], steps)
        raw = np.array([int(np.argmin(np.abs(logsnr - v))) for v in targets], dtype=np.int64)
    elif schedule == "logit_normal":
        rng = np.random.default_rng(int(seed))
        inner = 1.0 / (1.0 + np.exp(-(rng.standard_normal(max(0, steps - 2)) * 0.8 - 1.5)))
        # ELF time u=0 is noise, u=1 is data; DDPM index is noisedness.
        raw = np.concatenate([[max_t], np.sort((1.0 - inner) * max_t)[::-1], [0]])
    else:
        raise ValueError(f"Unknown timestep schedule={schedule!r}")

    idx = [int(np.clip(round(float(v)), 0, max_t)) for v in raw]
    idx = sorted(set(idx), reverse=True)
    if idx[0] != max_t:
        idx.insert(0, max_t)
    if idx[-1] != 0:
        idx.append(0)
    return idx


def _predict_x0(model, x: torch.Tensor, t: torch.Tensor, cond: dict[str, Any], cfg_scale: float) -> torch.Tensor:
    pred_cond = model.denoiser(x, t, cond, cond_drop_mask=None, self_cond=None)
    if float(cfg_scale) != 1.0:
        drop = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
        pred_uncond = model.denoiser(x, t, cond, cond_drop_mask=drop, self_cond=None)
    else:
        pred_uncond = None
    if model.diffusion.prediction_target == "v":
        x0_cond = model.diffusion.predict_x0_from_v(x, t, pred_cond)
        x0_uncond = (
            model.diffusion.predict_x0_from_v(x, t, pred_uncond)
            if pred_uncond is not None else None
        )
    else:
        x0_cond = pred_cond
        x0_uncond = pred_uncond
    if x0_uncond is None:
        return x0_cond
    return x0_uncond + float(cfg_scale) * (x0_cond - x0_uncond)


@torch.no_grad()
def _sample_variant(
    model,
    cond: dict[str, Any],
    seq_length: int,
    variant: SamplerVariant,
    cfg_scale: float,
    seed: int,
    log_timesteps: list[int],
) -> tuple[torch.Tensor, dict[int, torch.Tensor], list[int]]:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    device = cond["z_int"].device
    batch_size = cond["z_int"].shape[0]
    shape = (batch_size, int(seq_length), model.cfg.denoiser.motion_dim)
    x = torch.randn(shape, device=device)
    logs: dict[int, torch.Tensor] = {}
    best_logs: dict[int, tuple[int, torch.Tensor]] = {}

    if variant.method == "ddpm_full":
        timesteps = list(range(model.diffusion.num_steps - 1, -1, -1))
        for t_int in timesteps:
            t = torch.full((batch_size,), int(t_int), device=device, dtype=torch.long)
            x0 = _predict_x0(model, x, t, cond, cfg_scale=cfg_scale)
            for target in log_timesteps:
                diff = abs(int(t_int) - int(target))
                if target not in best_logs or diff < best_logs[target][0]:
                    best_logs[int(target)] = (diff, x0.detach().clone())
            mean = model.diffusion.posterior_mean_from_x0(x0, x, t)
            if t_int == 0:
                x = mean
            else:
                noise = torch.randn_like(x)
                log_var = _extract(model.diffusion.posterior_log_variance_clipped, t, x.shape)
                x = mean + (0.5 * log_var).exp() * noise
        logs = {int(k): v for k, (_d, v) in best_logs.items()}
        logs[0] = x.detach().clone()
        return x, logs, timesteps

    timesteps = _make_timestep_list(
        model.diffusion,
        steps=int(variant.steps),
        schedule=str(variant.schedule),
        seed=int(seed),
    )
    for i, t_int in enumerate(timesteps[:-1]):
        prev_int = int(timesteps[i + 1])
        t = torch.full((batch_size,), int(t_int), device=device, dtype=torch.long)
        prev = torch.full((batch_size,), prev_int, device=device, dtype=torch.long)
        x0 = _predict_x0(model, x, t, cond, cfg_scale=cfg_scale)
        for target in log_timesteps:
            diff = abs(int(t_int) - int(target))
            if target not in best_logs or diff < best_logs[target][0]:
                best_logs[int(target)] = (diff, x0.detach().clone())
        alpha_t = _extract(model.diffusion.alphas_cumprod, t, x.shape)
        alpha_prev = _extract(model.diffusion.alphas_cumprod, prev, x.shape)
        sqrt_alpha_t = alpha_t.sqrt()
        eps = (x - sqrt_alpha_t * x0) / (1.0 - alpha_t).sqrt().clamp_min(1e-8)
        if prev_int == 0 and i == len(timesteps) - 2:
            # Keep the final formula active too; alpha_prev is close to 1 but
            # not exactly 1 under the stored cosine buffer.
            pass
        eta = float(variant.eta)
        sigma = (
            eta
            * ((1.0 - alpha_prev) / (1.0 - alpha_t)).sqrt()
            * (1.0 - alpha_t / alpha_prev).clamp_min(0.0).sqrt()
        )
        dir_scale = (1.0 - alpha_prev - sigma.pow(2)).clamp_min(0.0).sqrt()
        noise = torch.randn_like(x) if eta > 0.0 and prev_int > 0 else torch.zeros_like(x)
        x = alpha_prev.sqrt() * x0 + dir_scale * eps + sigma * noise
    logs = {int(k): v for k, (_d, v) in best_logs.items()}
    logs[0] = x.detach().clone()
    return x, logs, timesteps


def _foot_velocity_mean(joints: np.ndarray, seq_len: int) -> float:
    j = joints[: int(seq_len)].astype(np.float32)
    if len(j) < 2:
        return 0.0
    vel = j[1:, [10, 11]] - j[:-1, [10, 11]]
    root_vel = j[1:, 0:1] - j[:-1, 0:1]
    local = vel - root_vel
    return float(np.linalg.norm(local, axis=-1).mean() * 100.0)


def _add_foot_ratio(metrics: dict[str, float], source_joints: np.ndarray, gt_joints: np.ndarray, seq_len: int) -> None:
    foot = _foot_velocity_mean(source_joints, seq_len)
    foot_gt = _foot_velocity_mean(gt_joints, seq_len)
    metrics["foot_velocity_cm_per_frame"] = foot
    metrics["foot_velocity_over_gt"] = _safe_div(foot, foot_gt)


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
    out = []
    for r, row in enumerate(rows):
        out.append("| " + " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(row)) + " |")
        if r == 0:
            out.append("| " + " | ".join("-" * widths[i] for i in range(len(widths))) + " |")
    return "\n".join(out)


def _write_report(payload: dict[str, Any], md_path: Path) -> None:
    variants = payload["aggregate"]
    rows = [[
        "sampler", "steps", "schedule", "eta", "body xGT", "hand xGT",
        "foot xGT", "trans rel xGT", "open/close xGT", "FFT mid", "acc p95",
    ]]
    for name, agg in variants.items():
        meta = payload["variants"][name]
        rows.append([
            name,
            meta["actual_steps_mean"],
            meta["schedule"],
            meta["eta"],
            f"{agg.get('body_velocity_over_gt', 0.0):.3f}",
            f"{agg.get('hand_velocity_over_gt', 0.0):.3f}",
            f"{agg.get('foot_velocity_over_gt', 0.0):.3f}",
            f"{agg.get('transition_relative_velocity_over_gt', 0.0):.3f}",
            f"{agg.get('positive_distance_change_over_gt', 0.0):.3f}",
            f"{agg.get('fft_mid', 0.0):.3f}",
            f"{agg.get('body_local_acceleration_p95_cm_per_frame2', 0.0):.3f}",
        ])

    traj_rows = [["sampler", "t", "body xGT", "hand xGT", "trans xGT", "FFT mid", "acc p95"]]
    for name, by_t in payload["intermediate_aggregate"].items():
        for t in ["900", "700", "500", "300", "100", "0"]:
            if t not in by_t:
                continue
            m = by_t[t]
            traj_rows.append([
                name, t,
                f"{m.get('body_velocity_over_gt', 0.0):.3f}",
                f"{m.get('hand_velocity_over_gt', 0.0):.3f}",
                f"{m.get('transition_relative_velocity_over_gt', 0.0):.3f}",
                f"{m.get('fft_mid', 0.0):.3f}",
                f"{m.get('body_local_acceleration_p95_cm_per_frame2', 0.0):.3f}",
            ])

    best_body = max(variants.items(), key=lambda kv: kv[1].get("body_velocity_over_gt", 0.0))
    best_trans = max(variants.items(), key=lambda kv: kv[1].get("transition_relative_velocity_over_gt", 0.0))
    baseline = variants.get("ddpm_1000", {})
    body_gain = best_body[1].get("body_velocity_over_gt", 0.0) - baseline.get("body_velocity_over_gt", 0.0)
    if body_gain > 0.04:
        case = "Case S1/S4: sampler geometry can improve final dynamics; inspect jitter and plan metrics before adopting."
    elif best_trans[1].get("transition_relative_velocity_over_gt", 0.0) > baseline.get("transition_relative_velocity_over_gt", 0.0) + 0.08:
        case = "Case S2: sampler mainly improves transition relative velocity, not a clean body-dynamics fix."
    else:
        case = "Case S3: sampler-only changes do not cleanly fix body/hand velocity damping."

    lines = [
        "# v18 Sampler Geometry Diagnostic",
        "",
        f"**Config:** `{payload['config']}`  ",
        f"**Checkpoint:** `{payload['ckpt']}`  ",
        f"**Clips:** {len(payload['clips'])} transition-heavy selected clips  ",
        "",
        "## ELF Reference",
        "",
        "ELF separates training and sampling configs and uses continuous ODE/SDE samplers over a noise-to-data trajectory. "
        "This diagnostic keeps the trained v18 DDPM model fixed and only changes the reverse trajectory discretization.",
        "",
        "## Final Sample Metrics",
        "",
        _format_table(rows),
        "",
        "## Intermediate x0 Trajectory",
        "",
        _format_table(traj_rows),
        "",
        "## Selected Clips",
        "",
        _format_table([["subset", "seq_id", "event", "part", "frame", "text"]] + [
            [
                c["subset"], c["seq_id"], c["event"]["kind"], c["event"]["part"],
                c["event"]["frame"], _short_text(c["text"], 72),
            ]
            for c in payload["clips"]
        ]),
        "",
        "## Interpretation",
        "",
        f"- Best body velocity variant: `{best_body[0]}` at xGT={best_body[1].get('body_velocity_over_gt', 0.0):.3f}.",
        f"- Best transition rel-vel variant: `{best_trans[0]}` at xGT={best_trans[1].get('transition_relative_velocity_over_gt', 0.0):.3f}.",
        f"- Decision: {case}",
        "- `far_unobs` and plan GT-zero/GT-wrong are skipped here because this script is sampler-only on selected clips; full plan diagnostics are reserved for trained v23/v24 checkpoints.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--md", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path, default=Path("analyses/2026-05-13_v18_recon_ladder_truncated_rollout.json"))
    parser.add_argument("--bucket", type=str, default="train")
    parser.add_argument("--max-clips", type=int, default=6)
    parser.add_argument("--num-candidates", type=int, default=96)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--fps", type=float, default=20.0)
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
    selection: dict[str, SelectedEvent] = {}
    if args.selection_json.exists():
        raw = json.loads(args.selection_json.read_text(encoding="utf-8"))
        for entry in raw.get("selected_clips", []):
            ev = entry.get("event") or {}
            if ev:
                selection[str(entry["seq_id"])] = SelectedEvent(
                    kind=str(ev["kind"]),
                    part=str(ev["part"]),
                    frame=int(ev["frame"]),
                    crop_start=int(ev.get("crop_start", 0)),
                    crop_end=int(ev.get("crop_end", int(ev["frame"]) + 1)),
                    reason=str(ev.get("reason", "")),
                )
            if len(selection) >= args.max_clips:
                break
    selected = _build_selected_batches(
        cfg,
        bucket=args.bucket,
        balanced_subsets=True,
        num_candidates=int(args.num_candidates),
        selection=selection,
        max_clips=int(args.max_clips),
        threshold=0.5,
    )

    variants = [
        SamplerVariant("ddpm_1000", "ddpm_full", 1000, 1.0, "uniform"),
        SamplerVariant("ddpm_eta1_500", "ddim_generalized", 500, 1.0, "uniform"),
        SamplerVariant("ddpm_eta1_250", "ddim_generalized", 250, 1.0, "uniform"),
        SamplerVariant("ddpm_eta1_100", "ddim_generalized", 100, 1.0, "uniform"),
        SamplerVariant("ddim_eta0_250", "ddim_generalized", 250, 0.0, "uniform"),
        SamplerVariant("ddim_eta05_250", "ddim_generalized", 250, 0.5, "uniform"),
        SamplerVariant("ddim_eta1_250", "ddim_generalized", 250, 1.0, "uniform"),
        SamplerVariant("ddim_eta0_250_low_noise_dense", "ddim_generalized", 250, 0.0, "low_noise_dense"),
        SamplerVariant("ddim_eta0_250_high_noise_dense", "ddim_generalized", 250, 0.0, "high_noise_dense"),
        SamplerVariant("ddim_eta0_250_logsnr_uniform", "ddim_generalized", 250, 0.0, "logsnr_uniform"),
        SamplerVariant("ddim_eta0_250_logit_normal", "ddim_generalized", 250, 0.0, "logit_normal"),
    ]

    per_variant_rows: dict[str, list[dict[str, float]]] = {v.name: [] for v in variants}
    per_variant_intermediate: dict[str, dict[str, list[dict[str, float]]]] = {
        v.name: {str(t): [] for t in LOG_TIMESTEPS_DEFAULT} for v in variants
    }
    variant_steps: dict[str, list[int]] = {v.name: [] for v in variants}
    clips_meta: list[dict[str, Any]] = []

    for ordinal, (batch_idx, batch, event) in enumerate(selected, start=1):
        cond, total_t = _build_cond(batch, model, object_encoder, clip_model, z_dims, cfg, device)
        cond = {**cond, "interaction_plan": _extract_plan(batch, device)}
        gt_joints = batch["joints"].to(device).float()
        rest_offsets = batch["rest_offsets"].to(device).float()
        seq_len = int(batch["seq_len"][0].item())
        gt_joints_np = gt_joints.squeeze(0).detach().cpu().numpy().astype(np.float32)
        object_pos = batch["object_positions"].squeeze(0).detach().cpu().numpy().astype(np.float32)
        clips_meta.append({
            "subset": str(batch["subset"][0]),
            "seq_id": str(batch["seq_id"][0]),
            "text": str(batch["text"][0]),
            "seq_len": seq_len,
            "event": {
                "kind": event.kind,
                "part": event.part,
                "frame": event.frame,
                "crop_start": event.crop_start,
                "crop_end": event.crop_end,
                "reason": event.reason,
            },
        })

        for v_idx, variant in enumerate(variants):
            motion, logs, tlist = _sample_variant(
                model,
                cond,
                seq_length=total_t,
                variant=variant,
                cfg_scale=float(args.cfg_scale),
                seed=int(args.seed) + ordinal * 10000 + v_idx * 1000,
                log_timesteps=list(LOG_TIMESTEPS_DEFAULT),
            )
            variant_steps[variant.name].append(len(tlist))
            joints = _fk_from_motion_135(motion, rest_offsets)
            joints_np = joints.squeeze(0).detach().cpu().numpy().astype(np.float32)
            m = _source_metrics_np(
                joints_np,
                gt_joints_np,
                object_pos,
                seq_len,
                event,
                fps=float(args.fps),
                window_k=10,
                transition_radius=5,
            )
            _add_foot_ratio(m, joints_np, gt_joints_np, seq_len)
            per_variant_rows[variant.name].append(m)

            for t_key, x0 in logs.items():
                log_joints = _fk_from_motion_135(x0, rest_offsets)
                log_np = log_joints.squeeze(0).detach().cpu().numpy().astype(np.float32)
                lm = _source_metrics_np(
                    log_np,
                    gt_joints_np,
                    object_pos,
                    seq_len,
                    event,
                    fps=float(args.fps),
                    window_k=10,
                    transition_radius=5,
                )
                _add_foot_ratio(lm, log_np, gt_joints_np, seq_len)
                per_variant_intermediate[variant.name].setdefault(str(int(t_key)), []).append(lm)

    aggregate = {name: _mean_rows(rows) for name, rows in per_variant_rows.items()}
    intermediate_aggregate = {
        name: {t: _mean_rows(rows) for t, rows in by_t.items() if rows}
        for name, by_t in per_variant_intermediate.items()
    }
    variant_meta = {
        v.name: {
            "method": v.method,
            "requested_steps": v.steps,
            "actual_steps_mean": float(np.mean(variant_steps[v.name])) if variant_steps[v.name] else 0.0,
            "eta": v.eta,
            "schedule": v.schedule,
        }
        for v in variants
    }
    payload = {
        "config": str(args.config),
        "ckpt": str(args.ckpt),
        "cfg_scale": float(args.cfg_scale),
        "variants": variant_meta,
        "clips": clips_meta,
        "aggregate": aggregate,
        "intermediate_aggregate": intermediate_aggregate,
        "skipped": {
            "far_unobs": "not computed in selected-clip sampler-only diagnostic",
            "plan_condition": "not computed here; reserved for trained v23/v24 diagnostics",
            "exact_skipped_ddpm": "step-count DDPM rows use generalized DDIM eta=1, not exact skipped posterior",
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
