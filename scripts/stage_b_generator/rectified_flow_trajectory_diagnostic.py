"""Rectified-flow trajectory diagnostic for Stage B.

Logs intermediate samples along the RF noise-to-data ODE/SDE path and computes
the same transition-heavy metrics used by the DDPM recon-ladder diagnostic.
This is eval-only; it does not alter training code or checkpoints.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from dynamics_diagnostic import _build_cond, _build_model, _fk_from_motion_135
from recon_ladder_truncated_rollout_diagnostic import (
    SelectedEvent,
    _build_selected_batches,
    _source_metrics_np,
)
from render_recon_vs_sample import _extract_plan, _load_checkpoint, _short_text
from sampler_geometry_diagnostic import _add_foot_ratio, _format_table, _mean_rows
from piano.utils.clip_utils import load_clip_text_encoder


def _load_selection(selection_json: Path, max_clips: int) -> dict[str, SelectedEvent]:
    if not selection_json.exists():
        return {}
    payload = json.loads(selection_json.read_text(encoding="utf-8"))
    entries = payload.get("selected_clips", payload.get("selected", []))
    out: dict[str, SelectedEvent] = {}
    for entry in entries:
        ev = entry.get("event") or (entry.get("events") or [{}])[0]
        if not ev:
            continue
        out[str(entry["seq_id"])] = SelectedEvent(
            kind=str(ev["kind"]),
            part=str(ev["part"]),
            frame=int(ev["frame"]),
            crop_start=int(ev.get("crop_start", max(0, int(ev["frame"]) - 15))),
            crop_end=int(ev.get("crop_end", int(ev["frame"]) + 15)),
            reason=str(ev.get("reason", entry.get("selected_reason", ""))),
        )
        if len(out) >= int(max_clips):
            break
    return out


def _write_report(payload: dict[str, Any], md_path: Path) -> None:
    final = payload["aggregate"]["final"]
    rows = [[
        "source", "body xGT", "hand xGT", "foot xGT", "trans rel xGT",
        "open/close xGT", "FFT mid", "acc p95",
    ]]
    rows.append([
        "final",
        f"{final.get('body_velocity_over_gt', 0.0):.3f}",
        f"{final.get('hand_velocity_over_gt', 0.0):.3f}",
        f"{final.get('foot_velocity_over_gt', 0.0):.3f}",
        f"{final.get('transition_relative_velocity_over_gt', 0.0):.3f}",
        f"{final.get('positive_distance_change_over_gt', 0.0):.3f}",
        f"{final.get('fft_mid', 0.0):.3f}",
        f"{final.get('body_local_acceleration_p95_cm_per_frame2', 0.0):.3f}",
    ])

    traj_rows = [["rf_t", "body xGT", "hand xGT", "trans xGT", "FFT mid", "acc p95"]]
    for t_key in sorted(payload["aggregate"]["trajectory"], key=lambda v: float(v)):
        m = payload["aggregate"]["trajectory"][t_key]
        traj_rows.append([
            t_key,
            f"{m.get('body_velocity_over_gt', 0.0):.3f}",
            f"{m.get('hand_velocity_over_gt', 0.0):.3f}",
            f"{m.get('transition_relative_velocity_over_gt', 0.0):.3f}",
            f"{m.get('fft_mid', 0.0):.3f}",
            f"{m.get('body_local_acceleration_p95_cm_per_frame2', 0.0):.3f}",
        ])

    t_keys = sorted(payload["aggregate"]["trajectory"], key=lambda v: float(v))
    first_body = payload["aggregate"]["trajectory"].get(t_keys[0], {}).get("body_velocity_over_gt", 0.0) if t_keys else 0.0
    last_body = final.get("body_velocity_over_gt", 0.0)
    damping = last_body - first_body
    if last_body > 1.1 and final.get("body_local_acceleration_p95_cm_per_frame2", 0.0) > 2.0:
        verdict = "Velocity is high, but acceleration p95 indicates likely jitter/over-dynamics."
    elif damping < -0.05:
        verdict = "RF trajectory still damps velocity over the path."
    else:
        verdict = "RF trajectory does not show strong monotonic velocity damping on the selected clips."

    lines = [
        "# Rectified-Flow Trajectory Diagnostic",
        "",
        f"**Config:** `{payload['config']}`  ",
        f"**Checkpoint:** `{payload['ckpt']}`  ",
        f"**Sampler:** `{payload['sampler_type']}`  ",
        f"**Steps:** {payload['num_steps']}  **schedule:** `{payload['time_schedule']}`  **sde_gamma:** {payload['sde_gamma']}  ",
        f"**Clips:** {len(payload['clips'])}",
        "",
        "## Final Sample",
        "",
        _format_table(rows),
        "",
        "## RF Intermediate Trajectory",
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
        f"- Body velocity change from first logged RF time to final: {damping:+.3f} xGT.",
        f"- Verdict: {verdict}",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--md", type=Path, required=True)
    parser.add_argument(
        "--selection-json",
        type=Path,
        default=Path("analyses/2026-05-13_v18_recon_ladder_truncated_rollout.json"),
    )
    parser.add_argument("--num-candidates", type=int, default=96)
    parser.add_argument("--max-clips", type=int, default=6)
    parser.add_argument("--bucket", choices=["train", "val"], default="train")
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--sampler-type", default=None)
    parser.add_argument("--time-schedule", default=None)
    parser.add_argument("--sde-gamma", type=float, default=None)
    parser.add_argument("--time-points", default="0.1,0.3,0.5,0.7,0.9,1.0")
    parser.add_argument("--window-k", type=int, default=10)
    parser.add_argument("--transition-radius", type=int, default=5)
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
    if getattr(model.diffusion, "objective", "ddpm") != "rectified_flow":
        raise ValueError("rectified_flow_trajectory_diagnostic requires objective='rectified_flow'")

    time_points = tuple(float(v.strip()) for v in args.time_points.split(",") if v.strip())
    selection = _load_selection(args.selection_json, max_clips=int(args.max_clips))
    selected = _build_selected_batches(
        cfg,
        bucket=args.bucket,
        balanced_subsets=True,
        num_candidates=int(args.num_candidates),
        selection=selection,
        max_clips=int(args.max_clips),
        threshold=0.5,
    )

    final_rows: list[dict[str, float]] = []
    trajectory_rows: dict[str, list[dict[str, float]]] = {f"{t:.1f}": [] for t in time_points}
    clips_meta: list[dict[str, Any]] = []

    for ordinal, (_batch_idx, batch, event) in enumerate(selected, start=1):
        cond, total_t = _build_cond(batch, model, object_encoder, clip_model, z_dims, cfg, device)
        cond = {**cond, "interaction_plan": _extract_plan(batch, device)}
        rest_offsets = batch["rest_offsets"].to(device).float()
        gt_joints = batch["joints"].to(device).float()
        gt_np = gt_joints.squeeze(0).detach().cpu().numpy().astype(np.float32)
        object_pos = batch["object_positions"].squeeze(0).detach().cpu().numpy().astype(np.float32)
        seq_len = int(batch["seq_len"][0].item())
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

        torch.manual_seed(int(args.seed) + ordinal * 1000)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed) + ordinal * 1000)
        with torch.no_grad():
            final_motion, logs = model.diffusion.rf_sample_loop(
                model.denoiser,
                (1, int(total_t), int(model.cfg.denoiser.motion_dim)),
                cond,
                cfg_scale=float(args.cfg_scale),
                device=device,
                output_skip=False,
                num_steps=args.num_steps,
                sampler_type=args.sampler_type,
                time_schedule=args.time_schedule,
                sde_gamma=args.sde_gamma,
                return_intermediates=time_points,
            )

        final_joints = _fk_from_motion_135(final_motion, rest_offsets)
        final_np = final_joints.squeeze(0).detach().cpu().numpy().astype(np.float32)
        final_m = _source_metrics_np(
            final_np, gt_np, object_pos, seq_len, event,
            fps=float(args.fps), window_k=int(args.window_k),
            transition_radius=int(args.transition_radius),
        )
        _add_foot_ratio(final_m, final_np, gt_np, seq_len)
        final_rows.append(final_m)

        for t_val, motion in logs.items():
            key = f"{float(t_val):.1f}"
            joints = _fk_from_motion_135(motion, rest_offsets)
            j_np = joints.squeeze(0).detach().cpu().numpy().astype(np.float32)
            m = _source_metrics_np(
                j_np, gt_np, object_pos, seq_len, event,
                fps=float(args.fps), window_k=int(args.window_k),
                transition_radius=int(args.transition_radius),
            )
            _add_foot_ratio(m, j_np, gt_np, seq_len)
            trajectory_rows.setdefault(key, []).append(m)
        print(
            f"  [{ordinal}/{len(selected)}] {batch['subset'][0]}/{batch['seq_id'][0]} "
            f"{event.part} {event.kind}@{event.frame}"
        )

    payload = {
        "config": str(args.config),
        "ckpt": str(args.ckpt),
        "cfg_scale": float(args.cfg_scale),
        "sampler_type": str(args.sampler_type or model.diffusion.rf_sampler_type),
        "num_steps": int(args.num_steps or model.diffusion.rf_num_sampling_steps),
        "time_schedule": str(args.time_schedule or model.diffusion.rf_time_schedule),
        "sde_gamma": float(args.sde_gamma if args.sde_gamma is not None else model.diffusion.rf_sde_gamma),
        "time_points": list(time_points),
        "clips": clips_meta,
        "aggregate": {
            "final": _mean_rows(final_rows),
            "trajectory": {k: _mean_rows(v) for k, v in trajectory_rows.items() if v},
        },
        "per_clip_final": final_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_report(payload, args.md)
    print(f"Wrote JSON to {args.output}")
    print(f"Wrote report to {args.md}")


if __name__ == "__main__":
    main()
