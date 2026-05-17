"""Targeted A9 sampler visual follow-up for the Stage B v18 checkpoint.

This script intentionally evaluates only four sampler settings:

GT | A0 DDPM | A1 DDIM eta0 logit-normal | A2 DDIM eta0 logSNR | A9 DDIM eta0.2 logSNR

It does not train, change the model, or expand the sampler sweep.
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

from dynamics_diagnostic import _build_cond, _build_model, _fk_from_motion_135
from piano.sampling import SamplerConfig, sample_with_config
from piano.utils.clip_utils import load_clip_text_encoder
from render_objective_sampler_comparison import ClipComparison, _auto_verdict, _render_video
from render_recon_vs_sample import _extract_plan, _load_checkpoint
from sampler_refinement_round import (
    LOG_TIMESTEPS_DEFAULT,
    _clip_meta,
    _event_kind_aggregates,
    _format_table,
    _gt_joints_np,
    _mean_rows,
    _metrics_for_joints,
    _object_pos_np,
    _save_dynamic_contact_sheet,
    _selected_batches,
)


TARGET_VARIANTS: tuple[tuple[str, str, SamplerConfig], ...] = (
    (
        "A0_DDPM",
        "A0_ddpm_1000_cosine_default",
        SamplerConfig(
            name="A0_DDPM",
            sampler_type="ddpm",
            num_sampling_steps=1000,
            sampling_schedule="cosine_default",
            ddim_eta=1.0,
        ),
    ),
    (
        "A1_DDIM_eta0_logitnormal",
        "A1_ddim_eta0_250_logit_normal",
        SamplerConfig(
            name="A1_DDIM_eta0_logitnormal",
            sampler_type="ddim",
            num_sampling_steps=250,
            sampling_schedule="logit_normal",
            ddim_eta=0.0,
        ),
    ),
    (
        "A2_DDIM_eta0_logSNR",
        "A2_ddim_eta0_250_logsnr_uniform",
        SamplerConfig(
            name="A2_DDIM_eta0_logSNR",
            sampler_type="ddim",
            num_sampling_steps=250,
            sampling_schedule="logsnr_uniform",
            ddim_eta=0.0,
        ),
    ),
    (
        "A9_DDIM_eta0p2_logSNR",
        "A9_ddim_eta0p2_250_logsnr_uniform",
        SamplerConfig(
            name="A9_DDIM_eta0p2_logSNR",
            sampler_type="ddim",
            num_sampling_steps=250,
            sampling_schedule="logsnr_uniform",
            ddim_eta=0.2,
        ),
    ),
)


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if abs(float(den)) > 1e-12 else 0.0


def _short_metrics(metrics: dict[str, float]) -> dict[str, float]:
    keys = (
        "body_velocity_over_gt",
        "hand_velocity_over_gt",
        "foot_velocity_over_gt",
        "acc_p95_over_gt",
        "jerk_p95_over_gt",
        "fft_mid",
        "fft_high",
        "fft_high_over_gt",
        "transition_relative_velocity_over_gt",
        "positive_distance_change_over_gt",
    )
    return {key: float(metrics.get(key, 0.0)) for key in keys}


def _load_prior_plan(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    plan = raw.get("plan_condition", {})
    out: dict[str, dict[str, float]] = {}
    for alias, original, _config in TARGET_VARIANTS:
        row = plan.get(original, {})
        if row:
            out[alias] = {
                "far_unobs_gt_cm": float(row.get("far_unobs_gt_cm", 0.0)),
                "gt_zero_gap_cm": float(row.get("gt_zero_gap_cm", 0.0)),
                "gt_wrong_gap_cm": float(row.get("gt_wrong_gap_cm", 0.0)),
            }
    return out


def _render_followup(
    selected: list[tuple[int, dict[str, Any], Any]],
    sample_cache: dict[tuple[str, str], dict[str, Any]],
    visual_dir: Path,
    fps: float,
    dpi: int,
    object_points: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    visual_dir.mkdir(parents=True, exist_ok=True)
    aliases = [alias for alias, _original, _config in TARGET_VARIANTS]
    for ordinal, (_batch_idx, batch, event) in enumerate(selected, start=1):
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
        for alias in aliases:
            cached = sample_cache[(seq_id, alias)]
            joints_by_method[alias] = cached["joints"]
            metrics_by_method[alias] = cached["metrics"]

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
        for alias in aliases:
            sources[alias] = joints_by_method[alias]

        full_path = visual_dir / f"{stem}_a9_sampler_followup_full.mp4"
        _render_video(
            clip,
            sources,
            full_path,
            start=0,
            end=seq_len - 1,
            title="A9 sampler visual follow-up",
            fps=float(fps),
            dpi=int(dpi),
            object_points=int(object_points),
            seed=int(seed) + ordinal,
        )

        crop_path = ""
        sheet_path = ""
        if event is not None:
            crop_start = max(0, int(event.crop_start))
            crop_end = min(seq_len - 1, int(event.crop_end))
            crop_path_obj = visual_dir / (
                f"{stem}_{event.part}_{event.kind}_a9_sampler_followup_crop.mp4"
            )
            _render_video(
                clip,
                sources,
                crop_path_obj,
                start=crop_start,
                end=crop_end,
                title="A9 sampler visual follow-up crop",
                fps=float(fps),
                dpi=int(dpi),
                object_points=int(object_points),
                seed=int(seed) + ordinal + 900,
            )
            sheet_path_obj = visual_dir / (
                f"{stem}_{event.part}_{event.kind}_a9_sampler_followup_contact_sheet.png"
            )
            _save_dynamic_contact_sheet(
                clip,
                sources,
                sheet_path_obj,
                event,
                dpi=int(dpi),
                object_points=int(object_points),
                seed=int(seed) + ordinal + 1800,
            )
            crop_path = str(crop_path_obj)
            sheet_path = str(sheet_path_obj)

        verdicts = {
            name: _auto_verdict(name, metrics_by_method[name], event)
            for name in ["GT", *aliases]
        }
        rows.append({
            "clip": ordinal,
            "subset": subset,
            "seq_id": seq_id,
            "event": {
                "kind": event.kind,
                "part": event.part,
                "frame": int(event.frame),
            } if event is not None else None,
            "full_video": str(full_path),
            "crop_video": crop_path,
            "contact_sheet": sheet_path,
            "verdicts": verdicts,
        })
    return rows


def _gate_summary(payload: dict[str, Any]) -> dict[str, Any]:
    agg = payload["aggregate"].get("A9_DDIM_eta0p2_logSNR", {})
    baseline = payload["aggregate"].get("A0_DDPM", {})
    events = payload["event_aggregate"].get("A9_DDIM_eta0p2_logSNR", {})
    base_events = payload["event_aggregate"].get("A0_DDPM", {})
    visuals = payload.get("visuals", [])
    a9_verdicts = [
        str(row.get("verdicts", {}).get("A9_DDIM_eta0p2_logSNR", ""))
        for row in visuals
    ]
    bad_tokens = (
        "jitter",
        "wrong contact geometry",
        "wrong contact direction",
        "foot sliding",
        "over-motion",
        "root drift",
        "body pose distortion",
    )
    bad_count = sum(any(tok in v for tok in bad_tokens) for v in a9_verdicts)
    motion_gate = (
        agg.get("body_velocity_over_gt", 0.0) > baseline.get("body_velocity_over_gt", 0.0)
        and 0.75 <= agg.get("body_velocity_over_gt", 0.0) <= 1.05
        and 0.75 <= agg.get("hand_velocity_over_gt", 0.0) <= 1.15
    )
    stability_gate = (
        agg.get("acc_p95_over_gt", 999.0) <= 2.0
        and agg.get("jerk_p95_over_gt", 999.0) <= 3.0
    )
    onset = events.get("onset", {}).get("positive_distance_change_over_gt", 0.0)
    base_onset = base_events.get("onset", {}).get("positive_distance_change_over_gt", 0.0)
    release = events.get("release", {}).get("positive_distance_change_over_gt", 0.0)
    base_release = base_events.get("release", {}).get("positive_distance_change_over_gt", 0.0)
    transition_gate = onset + 1e-6 >= base_onset and release + 1e-6 >= base_release
    contact_gate = bad_count <= 3
    visual_gate = bad_count <= 3
    if motion_gate and stability_gate and transition_gate and contact_gate and visual_gate:
        decision = "A9 passes as optional evaluation sampler"
    elif motion_gate and stability_gate and bad_count < 5:
        decision = "A9 mixed; keep v18 default and use A9 only for selected debug/visual cases"
    elif motion_gate and stability_gate and release >= base_release and bad_count >= 5:
        decision = "A9 improves dynamics numerically but fails visual/contact gate; move to oracle event guidance"
    else:
        decision = "A9 fails; stop sampler-only sweep and move to oracle event guidance"
    return {
        "motion_gate": motion_gate,
        "stability_gate": stability_gate,
        "transition_gate": transition_gate,
        "contact_gate": contact_gate,
        "visual_gate": visual_gate,
        "visual_flagged_clips": int(bad_count),
        "visual_total_clips": int(len(a9_verdicts)),
        "decision": decision,
    }


def _write_report(payload: dict[str, Any], md_path: Path) -> None:
    method_rows = [["method", "sampler", "eta", "steps", "schedule", "checkpoint", "cfg_scale"]]
    for alias, _original, config in TARGET_VARIANTS:
        method_rows.append([
            alias,
            config.sampler_type,
            config.ddim_eta,
            config.num_sampling_steps,
            config.sampling_schedule,
            payload["ckpt"],
            payload["cfg_scale"],
        ])

    video_rows = [["clip", "subset", "seq_id", "event", "part", "full video", "crop video", "contact sheet"]]
    for row in payload.get("visuals", []):
        ev = row.get("event") or {}
        video_rows.append([
            row["clip"],
            row["subset"],
            row["seq_id"],
            ev.get("kind", ""),
            ev.get("part", ""),
            row["full_video"],
            row["crop_video"],
            row["contact_sheet"],
        ])

    quant_rows = [[
        "clip", "method", "body", "hand", "foot", "acc", "jerk",
        "FFT high", "onset closing", "release opening", "rel-vel",
    ]]
    for row in payload.get("per_clip", []):
        ev_kind = row["event_kind"]
        for method, metrics in row["metrics"].items():
            if method == "GT":
                continue
            quant_rows.append([
                row["clip"],
                method,
                f"{metrics.get('body_velocity_over_gt', 0.0):.3f}",
                f"{metrics.get('hand_velocity_over_gt', 0.0):.3f}",
                f"{metrics.get('foot_velocity_over_gt', 0.0):.3f}",
                f"{metrics.get('acc_p95_over_gt', 0.0):.3f}",
                f"{metrics.get('jerk_p95_over_gt', 0.0):.3f}",
                f"{metrics.get('fft_high', 0.0):.4f}",
                f"{metrics.get('positive_distance_change_over_gt', 0.0):.3f}" if ev_kind == "onset" else "",
                f"{metrics.get('positive_distance_change_over_gt', 0.0):.3f}" if ev_kind == "release" else "",
                f"{metrics.get('transition_relative_velocity_over_gt', 0.0):.3f}",
            ])

    verdict_rows = [["clip", "A0 verdict", "A1 verdict", "A2 verdict", "A9 verdict"]]
    for row in payload.get("visuals", []):
        v = row.get("verdicts", {})
        verdict_rows.append([
            row["clip"],
            v.get("A0_DDPM", ""),
            v.get("A1_DDIM_eta0_logitnormal", ""),
            v.get("A2_DDIM_eta0_logSNR", ""),
            v.get("A9_DDIM_eta0p2_logSNR", ""),
        ])

    aggregate_rows = [[
        "method", "body", "hand", "foot", "acc", "jerk", "FFT mid",
        "FFT high", "onset closing", "release opening",
    ]]
    for alias, _original, _config in TARGET_VARIANTS:
        agg = payload["aggregate"].get(alias, {})
        ev = payload["event_aggregate"].get(alias, {})
        aggregate_rows.append([
            alias,
            f"{agg.get('body_velocity_over_gt', 0.0):.3f}",
            f"{agg.get('hand_velocity_over_gt', 0.0):.3f}",
            f"{agg.get('foot_velocity_over_gt', 0.0):.3f}",
            f"{agg.get('acc_p95_over_gt', 0.0):.3f}",
            f"{agg.get('jerk_p95_over_gt', 0.0):.3f}",
            f"{agg.get('fft_mid', 0.0):.3f}",
            f"{agg.get('fft_high', 0.0):.3f}",
            f"{ev.get('onset', {}).get('positive_distance_change_over_gt', 0.0):.3f}",
            f"{ev.get('release', {}).get('positive_distance_change_over_gt', 0.0):.3f}",
        ])

    plan_rows = [["method", "far_unobs_gt_cm", "GT-zero gap cm", "GT-wrong gap cm"]]
    for alias, _original, _config in TARGET_VARIANTS:
        p = payload.get("prior_plan_condition", {}).get(alias, {})
        plan_rows.append([
            alias,
            f"{p.get('far_unobs_gt_cm', 0.0):.2f}" if p else "n/a",
            f"{p.get('gt_zero_gap_cm', 0.0):.2f}" if p else "n/a",
            f"{p.get('gt_wrong_gap_cm', 0.0):.2f}" if p else "n/a",
        ])

    gates = payload["a9_gate_summary"]
    gate_rows = [["gate", "result"]]
    for key in ("motion_gate", "stability_gate", "transition_gate", "contact_gate", "visual_gate"):
        gate_rows.append([key, "pass" if gates.get(key) else "fail"])
    gate_rows.append(["visual flagged clips", f"{gates['visual_flagged_clips']}/{gates['visual_total_clips']}"])
    gate_rows.append(["decision", gates["decision"]])

    lines = [
        "# A9 Sampler Visual Follow-up",
        "",
        "## Background",
        "",
        "- v18 DDPM remains the default mainline.",
        "- A1/A2/A13 were rejected in the previous sampler-only round because higher velocity did not pass visual/contact gates.",
        "- A9 numerically passed the previous gates but was not rendered, so this run validates only A9 against A0/A1/A2.",
        "- No training, model, loss, all-7 data, or new sampler variants were used.",
        "",
        "## Method Summary",
        "",
        _format_table(method_rows),
        "",
        "## Rendered Video List",
        "",
        _format_table(video_rows),
        "",
        "## Aggregate Quantitative Summary",
        "",
        _format_table(aggregate_rows),
        "",
        "## Prior Plan Metadata",
        "",
        "Plan metrics are copied from the previous sampler refinement report for the same sampler definitions; this targeted visual run did not rerun the plan sweep.",
        "",
        _format_table(plan_rows),
        "",
        "## Quantitative Table",
        "",
        _format_table(quant_rows),
        "",
        "## Visual Verdict Table",
        "",
        _format_table(verdict_rows),
        "",
        "## A9 Pass/Fail Gate Summary",
        "",
        _format_table(gate_rows),
        "",
        "## Final Decision",
        "",
        gates["decision"],
    ]
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/training/anchordiff_v18_a1_FULL_DATA.yaml"))
    parser.add_argument("--ckpt", type=Path, default=Path("runs/training/stageB_anchordiff_v18_a1_FULL_DATA/final.pt"))
    parser.add_argument("--output", type=Path, default=Path("analyses/2026-05-14_a9_sampler_visual_followup_report.json"))
    parser.add_argument("--md", type=Path, default=Path("analyses/2026-05-14_a9_sampler_visual_followup_report.md"))
    parser.add_argument("--visual-dir", type=Path, default=Path("analyses/visuals/2026-05-14_a9_sampler_visual_followup"))
    parser.add_argument("--selection-json", type=Path, default=Path("analyses/visuals/2026-05-14_objective_sampler_geometry_visuals/objective_sampler_geometry_visual_metadata.json"))
    parser.add_argument("--prior-report", type=Path, default=Path("analyses/2026-05-14_sampler_refinement_round_report.json"))
    parser.add_argument("--bucket", type=str, default="train")
    parser.add_argument("--max-clips", type=int, default=10)
    parser.add_argument("--num-candidates", type=int, default=160)
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

    per_variant_rows: dict[str, list[dict[str, Any]]] = {
        alias: [] for alias, _original, _config in TARGET_VARIANTS
    }
    aggregate_rows: dict[str, list[dict[str, float]]] = {
        alias: [] for alias, _original, _config in TARGET_VARIANTS
    }
    sample_cache: dict[tuple[str, str], dict[str, Any]] = {}
    sampler_meta: dict[str, dict[str, Any]] = {}
    per_clip: list[dict[str, Any]] = []

    for ordinal, (_batch_idx, batch, event) in enumerate(selected, start=1):
        print(f"[clip {ordinal}/{len(selected)}] {batch['subset'][0]} {batch['seq_id'][0]}")
        cond_base, total_t = _build_cond(batch, model, object_encoder, clip_model, z_dims, cfg, device)
        cond = {**cond_base, "interaction_plan": _extract_plan(batch, device)}
        rest_offsets = batch["rest_offsets"].to(device).float()
        seq_len = int(batch["seq_len"][0].item())
        gt_joints = _gt_joints_np(batch)
        object_pos = _object_pos_np(batch)
        seq_id = str(batch["seq_id"][0])
        clip_metrics: dict[str, dict[str, float]] = {}
        for v_idx, (alias, original_name, variant) in enumerate(TARGET_VARIANTS):
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
            if alias not in sampler_meta:
                sampler_meta[alias] = {
                    **asdict(variant),
                    "original_variant_id": original_name,
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
            aggregate_rows[alias].append(metrics)
            per_variant_rows[alias].append({
                "clip": ordinal,
                "subset": str(batch["subset"][0]),
                "seq_id": seq_id,
                "event_kind": event.kind,
                "event_part": event.part,
                "event_frame": int(event.frame),
                "metrics": metrics,
            })
            sample_cache[(seq_id, alias)] = {
                "motion": motion.detach().cpu(),
                "joints": joints_np,
                "metrics": metrics,
                "intermediate_keys": sorted(int(k) for k in logs.keys()),
            }
            clip_metrics[alias] = _short_metrics(metrics)
        per_clip.append({
            "clip": ordinal,
            "subset": str(batch["subset"][0]),
            "seq_id": seq_id,
            "event_kind": event.kind,
            "event_part": event.part,
            "event_frame": int(event.frame),
            "metrics": clip_metrics,
        })

    aggregate = {name: _mean_rows(rows) for name, rows in aggregate_rows.items()}
    event_aggregate = {
        name: _event_kind_aggregates(rows)
        for name, rows in per_variant_rows.items()
    }

    visuals: list[dict[str, Any]] = []
    if not args.skip_render:
        visuals = _render_followup(
            selected,
            sample_cache,
            args.visual_dir,
            fps=float(args.render_fps),
            dpi=int(args.render_dpi),
            object_points=int(args.object_points),
            seed=int(args.seed),
        )

    payload = {
        "config": str(args.config),
        "ckpt": str(args.ckpt),
        "cfg_scale": float(args.cfg_scale),
        "device": str(device),
        "variants": sampler_meta,
        "clips": [_clip_meta(idx, batch, event) for idx, batch, event in selected],
        "aggregate": aggregate,
        "event_aggregate": event_aggregate,
        "prior_plan_condition": _load_prior_plan(args.prior_report),
        "per_clip": per_clip,
        "visuals": visuals,
        "a9_gate_summary": {},
        "skipped": {
            "training": "not run",
            "new_sampler_variants": "not run",
            "model_or_loss_changes": "not made",
            "all7": "not used",
        },
    }
    payload["a9_gate_summary"] = _gate_summary(payload)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_report(payload, args.md)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.md}")


if __name__ == "__main__":
    main()
