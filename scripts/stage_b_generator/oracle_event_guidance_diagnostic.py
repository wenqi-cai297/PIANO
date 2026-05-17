"""Oracle event-derived sampling-time guidance diagnostic for Stage B v18.

This is a diagnostic-only script.  It uses GT hand contact onset/release
windows to steer the predicted x0 locally during DDPM sampling, then measures
whether transition direction improves without RF/DDIM-like jitter.  It does
not train or modify the denoiser.
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

from diagnostic_common import (
    clip_metadata,
    dynamics_metrics,
    event_records_from_contact,
    extract_plan,
    format_md_table,
    load_checkpoint,
    make_seq_mask,
    merge_single_batches,
    safe_div,
    transition_metrics,
    write_json,
)
from dynamics_diagnostic import (
    _build_cond,
    _build_model,
    _fk_from_motion_135,
)
from piano.models.motion_anchordiff import _extract
from piano.utils.clip_utils import load_clip_text_encoder
from recon_ladder_truncated_rollout_diagnostic import (
    _build_selected_batches,
    _load_selection,
)


def _parse_floats(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _guidance_loss(
    x0: Tensor,
    *,
    rest_offsets: Tensor,
    object_positions: Tensor,
    seq_mask: Tensor,
    events: list[dict[str, Any]],
    pre_k: int,
    post_k: int,
    margin_m: float,
    max_events_per_clip: int,
) -> Tensor:
    joints = _fk_from_motion_135(x0, rest_offsets)
    loss = x0.new_zeros(())
    used_per_clip: dict[int, int] = {}
    for ev in events:
        b = int(ev["batch"])
        if used_per_clip.get(b, 0) >= int(max_events_per_clip):
            continue
        valid = int(seq_mask[b].sum().item())
        t = int(ev["frame"])
        if t < 0 or t >= valid:
            continue
        joint = int(ev["joint"])
        dist = torch.linalg.vector_norm(joints[b, :valid, joint] - object_positions[b, :valid], dim=-1)
        if ev["kind"] == "onset":
            start = max(0, t - int(pre_k))
            loss = loss + torch.relu(x0.new_tensor(float(margin_m)) + dist[t] - dist[start])
            if t > start:
                d = dist[start : t + 1]
                loss = loss + torch.relu(d[1:] - d[:-1] + float(margin_m)).mean()
        else:
            end = min(valid - 1, t + int(post_k))
            loss = loss + torch.relu(x0.new_tensor(float(margin_m)) + dist[t] - dist[end])
            if end > t:
                d = dist[t : end + 1]
                loss = loss + torch.relu(d[:-1] - d[1:] + float(margin_m)).mean()
        used_per_clip[b] = used_per_clip.get(b, 0) + 1
    denom = max(1, sum(used_per_clip.values()))
    return loss / float(denom)


def _guide_x0(
    x0: Tensor,
    *,
    weight: float,
    rest_offsets: Tensor,
    object_positions: Tensor,
    seq_mask: Tensor,
    events: list[dict[str, Any]],
    pre_k: int,
    post_k: int,
    margin_m: float,
    grad_clip: float,
    delta_cap: float,
    max_events_per_clip: int,
) -> tuple[Tensor, dict[str, float]]:
    if float(weight) <= 0.0 or not events:
        return x0, {"loss": 0.0, "grad_norm_mean": 0.0, "delta_norm_mean": 0.0}
    with torch.enable_grad():
        x_req = x0.detach().clone().requires_grad_(True)
        loss = _guidance_loss(
            x_req,
            rest_offsets=rest_offsets,
            object_positions=object_positions,
            seq_mask=seq_mask,
            events=events,
            pre_k=pre_k,
            post_k=post_k,
            margin_m=margin_m,
            max_events_per_clip=max_events_per_clip,
        )
        grad = torch.autograd.grad(loss, x_req, retain_graph=False, create_graph=False)[0]
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        flat = grad.reshape(grad.shape[0], -1)
        norm = torch.linalg.vector_norm(flat, dim=-1).clamp_min(1e-12)
        if float(grad_clip) > 0:
            scale = torch.clamp(float(grad_clip) / norm, max=1.0).view(-1, 1, 1)
            grad = grad * scale
        delta = -float(weight) * grad
        if float(delta_cap) > 0:
            dnorm = torch.linalg.vector_norm(delta.reshape(delta.shape[0], -1), dim=-1).clamp_min(1e-12)
            dscale = torch.clamp(float(delta_cap) / dnorm, max=1.0).view(-1, 1, 1)
            delta = delta * dscale
        guided = (x_req + delta).detach()
    meta = {
        "loss": float(loss.detach().cpu().item()),
        "grad_norm_mean": float(norm.detach().mean().cpu().item()),
        "delta_norm_mean": float(torch.linalg.vector_norm(delta.reshape(delta.shape[0], -1), dim=-1).mean().cpu().item()),
    }
    return guided, meta


@torch.no_grad()
def _predict_x0(model, x: Tensor, t: Tensor, cond: dict[str, Any], cfg_scale: float) -> Tensor:
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
            if pred_uncond is not None
            else None
        )
    else:
        x0_cond = pred_cond
        x0_uncond = pred_uncond
    return x0_cond if x0_uncond is None else x0_uncond + float(cfg_scale) * (x0_cond - x0_uncond)


def _sample_guided_ddpm(
    model,
    cond: dict[str, Any],
    *,
    seq_length: int,
    cfg_scale: float,
    seed: int,
    weight: float,
    guide_t_max: int,
    rest_offsets: Tensor,
    object_positions: Tensor,
    seq_mask: Tensor,
    events: list[dict[str, Any]],
    pre_k: int,
    post_k: int,
    margin_m: float,
    grad_clip: float,
    delta_cap: float,
    max_events_per_clip: int,
) -> tuple[Tensor, dict[str, Any]]:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    device = cond["z_int"].device
    shape = (cond["z_int"].shape[0], int(seq_length), model.cfg.denoiser.motion_dim)
    x = torch.randn(shape, device=device)
    guidance_logs: list[dict[str, float]] = []
    applied_steps = 0
    for t_int in reversed(range(model.diffusion.num_steps)):
        t = torch.full((shape[0],), int(t_int), device=device, dtype=torch.long)
        x0 = _predict_x0(model, x, t, cond, cfg_scale=cfg_scale)
        if float(weight) > 0.0 and int(t_int) <= int(guide_t_max):
            x0, gmeta = _guide_x0(
                x0,
                weight=float(weight),
                rest_offsets=rest_offsets,
                object_positions=object_positions,
                seq_mask=seq_mask,
                events=events,
                pre_k=pre_k,
                post_k=post_k,
                margin_m=margin_m,
                grad_clip=grad_clip,
                delta_cap=delta_cap,
                max_events_per_clip=max_events_per_clip,
            )
            guidance_logs.append({"t": float(t_int), **gmeta})
            applied_steps += 1
        mean = model.diffusion.posterior_mean_from_x0(x0, x, t)
        if t_int == 0:
            x = mean
        else:
            noise = torch.randn_like(x)
            log_var = _extract(model.diffusion.posterior_log_variance_clipped, t, x.shape)
            x = mean + (0.5 * log_var).exp() * noise
    return x, {
        "applied_steps": applied_steps,
        "guidance_loss_mean": float(np.mean([r["loss"] for r in guidance_logs])) if guidance_logs else 0.0,
        "grad_norm_mean": float(np.mean([r["grad_norm_mean"] for r in guidance_logs])) if guidance_logs else 0.0,
        "delta_norm_mean": float(np.mean([r["delta_norm_mean"] for r in guidance_logs])) if guidance_logs else 0.0,
    }


def _write_report(payload: dict[str, Any], path: Path) -> None:
    rows = [["setting", "body xGT", "hand xGT", "acc p95 xGT", "jerk p95 xGT", "trans xGT", "onset xGT", "release xGT", "FFT mid", "verdict"]]
    for setting in payload["settings"]:
        m = setting["metrics"]
        tr = setting["transition"]
        rows.append([
            setting["name"],
            f"{m.get('body_velocity_cm_per_frame_over_gt', 0.0):.3f}",
            f"{m.get('hand_velocity_cm_per_frame_over_gt', 0.0):.3f}",
            f"{m.get('body_acc_p95_cm_per_frame2_over_gt', 0.0):.3f}",
            f"{m.get('body_jerk_p95_cm_per_frame3_over_gt', 0.0):.3f}",
            f"{tr.get('ratios_over_gt', {}).get('transition_relative_velocity', 0.0):.3f}",
            f"{tr.get('ratios_over_gt', {}).get('onset_positive_closing', 0.0):.3f}",
            f"{tr.get('ratios_over_gt', {}).get('release_positive_opening', 0.0):.3f}",
            f"{m.get('fft_mid', 0.0):.3f}",
            setting["verdict"],
        ])
    lines = [
        "# Oracle Event-Derived Sampling-Time Guidance Diagnostic",
        "",
        "This is an oracle diagnostic: GT contact onset/release windows are used during sampling. It is not deployable as-is.",
        "",
        f"- Config: `{payload['config']}`",
        f"- Checkpoint: `{payload['ckpt']}`",
        f"- Seed: `{payload['seed']}`",
        f"- Clips: {len(payload['selected_clips'])}",
        f"- Event count: {payload['event_count']} hand onset/release events",
        f"- Guidance location: after x0 prediction / CFG blend, before posterior mean",
        "",
        "## Contract Note",
        "",
        payload["contract_note"],
        "",
        "## Aggregate Metrics",
        "",
        format_md_table(rows),
        "",
        "## Interpretation",
        "",
        payload["interpretation"],
        "",
        "## Selected Clips",
        "",
    ]
    clip_rows = [["subset", "seq_id", "event", "seq_len", "text"]]
    for row in payload["selected_clips"]:
        ev = row.get("event", {})
        clip_rows.append([
            row["subset"],
            row["seq_id"],
            f"{ev.get('kind', 'none')} {ev.get('part', '')}@{ev.get('frame', '')}",
            row["seq_len"],
            row["text"][:80],
        ])
    lines.append(format_md_table(clip_rows))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/training/anchordiff_v18_a1_FULL_DATA.yaml"))
    parser.add_argument("--ckpt", type=Path, default=Path("runs/training/stageB_anchordiff_v18_a1_FULL_DATA/final.pt"))
    parser.add_argument("--output", type=Path, default=Path("analyses/2026-05-14_oracle_event_guidance_diagnostic.json"))
    parser.add_argument("--md", type=Path, default=Path("analyses/2026-05-14_oracle_event_guidance_diagnostic.md"))
    parser.add_argument("--selection-json", type=Path, default=Path("analyses/2026-05-13_v18_recon_ladder_truncated_rollout.json"))
    parser.add_argument("--max-clips", type=int, default=4)
    parser.add_argument("--num-candidates", type=int, default=96)
    parser.add_argument("--bucket", choices=["train", "val"], default="train")
    parser.add_argument("--balanced-subsets", action="store_true")
    parser.add_argument("--weights", type=str, default="0.0,0.02,0.05,0.10,0.20")
    parser.add_argument("--guide-t-max", type=int, default=700)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--pre-k", type=int, default=10)
    parser.add_argument("--post-k", type=int, default=5)
    parser.add_argument("--margin-cm", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--delta-cap", type=float, default=0.02)
    parser.add_argument("--max-events-per-clip", type=int, default=4)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selection = _load_selection(args.selection_json, max_clips=int(args.max_clips))
    selected = _build_selected_batches(
        cfg,
        bucket=args.bucket,
        balanced_subsets=bool(args.balanced_subsets),
        num_candidates=int(args.num_candidates),
        selection=selection,
        max_clips=int(args.max_clips),
        threshold=float(args.threshold),
    )
    if not selected:
        raise RuntimeError("No clips with hand contact events were selected")
    batch = merge_single_batches([item[1] for item in selected])

    model, object_encoder, z_dims = _build_model(cfg, device)
    load_checkpoint(model, object_encoder, args.ckpt)
    clip_model = load_clip_text_encoder(
        device=device,
        model_name=str(cfg.model.text_encoder.clip_version),
        download_root=str(cfg.model.text_encoder.get("download_root", "cache/clip")),
    )
    cond, total_t = _build_cond(batch, model, object_encoder, clip_model, z_dims, cfg, device)
    cond = {**cond, "interaction_plan": extract_plan(batch, device)}
    rest_offsets = batch["rest_offsets"].to(device).float()
    object_positions = batch["object_positions"].to(device).float()
    gt_joints = batch["joints"].to(device).float()
    seq_mask = make_seq_mask(batch["seq_len"], total_t, device)
    contact_state = batch["contact_state"].to(device).float()
    events = event_records_from_contact(contact_state, batch["seq_len"], threshold=float(args.threshold), hands_only=True)
    if not events:
        raise RuntimeError("Selected clips do not contain hand onset/release events")

    gt_dyn = dynamics_metrics(gt_joints, seq_mask, gt_joints=None, fps=float(args.fps))
    gt_trans = transition_metrics(
        gt_joints,
        object_positions,
        contact_state,
        seq_mask,
        window_k=int(args.pre_k),
        threshold=float(args.threshold),
    )
    settings: list[dict[str, Any]] = []
    for weight in _parse_floats(args.weights):
        print(f"Sampling oracle guidance weight={weight:.3f} on {len(selected)} clips")
        motion, guide_meta = _sample_guided_ddpm(
            model,
            cond,
            seq_length=total_t,
            cfg_scale=float(args.cfg_scale),
            seed=int(args.seed),
            weight=float(weight),
            guide_t_max=int(args.guide_t_max),
            rest_offsets=rest_offsets,
            object_positions=object_positions,
            seq_mask=seq_mask,
            events=events,
            pre_k=int(args.pre_k),
            post_k=int(args.post_k),
            margin_m=float(args.margin_cm) / 100.0,
            grad_clip=float(args.grad_clip),
            delta_cap=float(args.delta_cap),
            max_events_per_clip=int(args.max_events_per_clip),
        )
        joints = _fk_from_motion_135(motion, rest_offsets)
        dyn = dynamics_metrics(joints, seq_mask, gt_joints=gt_joints, fps=float(args.fps))
        trans = transition_metrics(
            joints,
            object_positions,
            contact_state,
            seq_mask,
            gt_joints=gt_joints,
            window_k=int(args.pre_k),
            threshold=float(args.threshold),
        )
        ratios = trans.get("ratios_over_gt", {})
        safe = (
            dyn.get("body_acc_p95_cm_per_frame2_over_gt", 99.0) <= 1.35
            and dyn.get("body_jerk_p95_cm_per_frame3_over_gt", 99.0) <= 1.50
        )
        direction_good = (
            ratios.get("onset_positive_closing", 0.0) >= 0.90
            and ratios.get("release_positive_opening", 0.0) >= 0.70
        )
        if float(weight) == 0.0:
            verdict = "baseline"
        elif direction_good and safe:
            verdict = "success_candidate"
        elif direction_good and not safe:
            verdict = "unsafe_direction_success"
        else:
            verdict = "failure"
        settings.append({
            "name": f"w{weight:g}_tmax{int(args.guide_t_max)}",
            "weight": float(weight),
            "guide_t_max": int(args.guide_t_max),
            "metrics": dyn,
            "transition": trans,
            "guidance": guide_meta,
            "verdict": verdict,
        })

    baseline = settings[0]
    best = max(
        settings[1:] or settings,
        key=lambda row: (
            row["transition"].get("ratios_over_gt", {}).get("onset_positive_closing", 0.0)
            + row["transition"].get("ratios_over_gt", {}).get("release_positive_opening", 0.0)
        ),
    )
    bbase = baseline["transition"].get("ratios_over_gt", {})
    bbest = best["transition"].get("ratios_over_gt", {})
    if best["verdict"] == "success_candidate":
        interpretation = (
            "Oracle event guidance produced a safe success candidate: transition direction improved without excessive acc/jerk. "
            "A deployable next step would need predicted or plan-derived event windows."
        )
    elif best["verdict"] == "unsafe_direction_success":
        interpretation = (
            "Oracle guidance improved transition direction but crossed stability gates, so it is not a safe fix as configured."
        )
    else:
        interpretation = (
            "Oracle guidance did not deliver a safe transition-direction improvement. "
            f"Baseline onset/release xGT=({bbase.get('onset_positive_closing', 0.0):.3f}, {bbase.get('release_positive_opening', 0.0):.3f}); "
            f"best guided=({bbest.get('onset_positive_closing', 0.0):.3f}, {bbest.get('release_positive_opening', 0.0):.3f})."
        )

    payload = {
        "config": str(args.config),
        "ckpt": str(args.ckpt),
        "device": str(device),
        "seed": int(args.seed),
        "cfg_scale": float(args.cfg_scale),
        "weights": _parse_floats(args.weights),
        "event_count": len(events),
        "gt_metrics": gt_dyn,
        "gt_transition": gt_trans,
        "selected_clips": clip_metadata(batch, events),
        "settings": settings,
        "contract_note": (
            "`z_int` is packed as 5 contact + 15 local target xyz + 3 phase + 3 support. "
            "`object_world_traj` is built in `_build_cond` as obj_com + obj_rot6d + lifted target_world. "
            "FK maps motion_135 global rot6d + root_world_pos to SMPL-22 joints through differentiable torch ops; "
            "the guidance detaches denoiser x0 and differentiates only through this FK/object-distance loss."
        ),
        "interpretation": interpretation,
    }
    write_json(args.output, payload)
    _write_report(payload, args.md)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.md}")


if __name__ == "__main__":
    main()

