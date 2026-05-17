"""Inference-only target-route ablation diagnostic for Stage B v18."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from diagnostic_common import (
    clip_metadata,
    dynamics_metrics,
    extract_plan,
    format_md_table,
    load_checkpoint,
    make_seq_mask,
    merge_single_batches,
    safe_div,
    transition_metrics,
    write_json,
)
from dynamics_diagnostic import _build_cond, _build_model, _fk_from_motion_135
from piano.utils.clip_utils import load_clip_text_encoder
from plan_condition_diagnostics import _compute_metrics as _compute_plan_metrics
from recon_ladder_truncated_rollout_diagnostic import _build_selected_batches, _load_selection


PART_TO_JOINT = torch.tensor([20, 21, 10, 11, 0], dtype=torch.long)


def _clone_cond(cond: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in cond.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.clone()
        elif isinstance(value, dict):
            out[key] = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in value.items()}
        else:
            out[key] = value
    return out


def _zero_zint_target(z_int: torch.Tensor, num_parts: int = 5) -> torch.Tensor:
    out = z_int.clone()
    start = int(num_parts)
    end = start + int(num_parts) * 3
    out[..., start:end] = 0.0
    return out


def _zero_plan_targets(plan: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {k: v.clone() for k, v in plan.items()}
    out["anchor_target_local"][:] = 0.0
    out["anchor_target_world"][:] = 0.0
    out["segment_target_summary_local"][:] = 0.0
    return out


def _variant_cond(base: dict[str, Any], variant: str) -> dict[str, Any]:
    cond = _clone_cond(base)
    if variant in {"no_zint_target", "plan_target_only", "dense_target_only", "timing_only"}:
        cond["z_int"] = _zero_zint_target(cond["z_int"])
    if variant in {"no_dense_target", "plan_target_only", "timing_only"}:
        cond["object_world_traj"][..., 9:] = 0.0
    if variant in {"no_plan_target", "dense_target_only", "timing_only"}:
        cond["interaction_plan"] = _zero_plan_targets(cond["interaction_plan"])
    return cond


def _write_report(payload: dict[str, Any], path: Path) -> None:
    rows = [["variant", "motion delta", "far unobs", "near anchor", "anchor cm", "body xGT", "hand xGT", "onset xGT", "release xGT", "verdict"]]
    for row in payload["variants"]:
        pm = row["plan_metrics"]
        dyn = row["dynamics"]
        tr = row["transition"].get("ratios_over_gt", {})
        rows.append([
            row["variant"],
            f"{row['motion_delta_vs_full']:.4f}",
            f"{pm['far_unobserved_error_cm']:.2f}",
            f"{pm['near_anchor_window_error_cm']:.2f}",
            f"{pm['plan_anchor_contact_realization_cm']:.2f}",
            f"{dyn.get('body_velocity_cm_per_frame_over_gt', 0.0):.3f}",
            f"{dyn.get('hand_velocity_cm_per_frame_over_gt', 0.0):.3f}",
            f"{tr.get('onset_positive_closing', 0.0):.3f}",
            f"{tr.get('release_positive_opening', 0.0):.3f}",
            row["verdict"],
        ])
    lines = [
        "# Target-Route Ablation Diagnostic",
        "",
        f"- Config: `{payload['config']}`",
        f"- Checkpoint: `{payload['ckpt']}`",
        f"- Clips: {len(payload['selected_clips'])}",
        "",
        "## Verdict",
        "",
        payload["verdict"],
        "",
        "## Metrics",
        "",
        format_md_table(rows),
        "",
        "## Route Definitions",
        "",
        "- `full`: z_int target + dense lifted target_world + plan target.",
        "- `no_zint_target`: zero only z_int contact_target_xyz.",
        "- `no_dense_target`: zero `object_world_traj[..., 9:]`.",
        "- `no_plan_target`: zero plan target local/world and segment target summary.",
        "- `plan_target_only`: keep plan target, zero z_int target and dense target.",
        "- `dense_target_only`: keep dense target, zero z_int target and plan target.",
        "- `timing_only`: keep anchor time/type/part, zero all target xyz routes.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/training/anchordiff_v18_a1_FULL_DATA.yaml"))
    parser.add_argument("--ckpt", type=Path, default=Path("runs/training/stageB_anchordiff_v18_a1_FULL_DATA/final.pt"))
    parser.add_argument("--output", type=Path, default=Path("analyses/2026-05-14_target_route_ablation_diagnostic.json"))
    parser.add_argument("--md", type=Path, default=Path("analyses/2026-05-14_target_route_ablation_diagnostic.md"))
    parser.add_argument("--selection-json", type=Path, default=Path("analyses/2026-05-13_v18_recon_ladder_truncated_rollout.json"))
    parser.add_argument("--max-clips", type=int, default=4)
    parser.add_argument("--num-candidates", type=int, default=96)
    parser.add_argument("--bucket", choices=["train", "val"], default="train")
    parser.add_argument("--balanced-subsets", action="store_true")
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
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
        raise RuntimeError("No clips selected for target-route ablation")
    batch = merge_single_batches([item[1] for item in selected])
    model, object_encoder, z_dims = _build_model(cfg, device)
    load_checkpoint(model, object_encoder, args.ckpt)
    clip_model = load_clip_text_encoder(
        device=device,
        model_name=str(cfg.model.text_encoder.clip_version),
        download_root=str(cfg.model.text_encoder.get("download_root", "cache/clip")),
    )
    cond, total_t = _build_cond(batch, model, object_encoder, clip_model, z_dims, cfg, device)
    plan_gt = extract_plan(batch, device)
    base_cond = {**cond, "interaction_plan": plan_gt}
    gt_joints = batch["joints"].to(device).float()
    rest_offsets = batch["rest_offsets"].to(device).float()
    object_positions = batch["object_positions"].to(device).float()
    contact_state = batch["contact_state"].to(device).float()
    seq_mask = make_seq_mask(batch["seq_len"], total_t, device)
    part_to_joint = PART_TO_JOINT.to(device)

    variants = [
        "full",
        "no_zint_target",
        "no_dense_target",
        "no_plan_target",
        "plan_target_only",
        "dense_target_only",
        "timing_only",
    ]
    rows: list[dict[str, Any]] = []
    full_motion: torch.Tensor | None = None
    for variant in variants:
        print(f"Sampling target-route variant={variant}")
        torch.manual_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))
        c = base_cond if variant == "full" else _variant_cond(base_cond, variant)
        with torch.no_grad():
            motion = model.sample(cond=c, seq_length=total_t, cfg_scale=float(args.cfg_scale))
        if full_motion is None:
            full_motion = motion.detach()
        joints = _fk_from_motion_135(motion, rest_offsets)
        plan_metrics = _compute_plan_metrics(
            jpos_pred=joints,
            jpos_gt=gt_joints,
            seq_mask=seq_mask,
            anchor_time=plan_gt["anchor_time"],
            anchor_mask=plan_gt["anchor_mask"],
            anchor_part=plan_gt["anchor_part"],
            anchor_target_world=plan_gt["anchor_target_world"],
            part_to_joint=part_to_joint,
            window=3,
        )
        dyn = dynamics_metrics(joints, seq_mask, gt_joints=gt_joints, fps=float(args.fps))
        trans = transition_metrics(
            joints,
            object_positions,
            contact_state,
            seq_mask,
            gt_joints=gt_joints,
            window_k=10,
            threshold=float(args.threshold),
        )
        delta = float(torch.linalg.vector_norm((motion - full_motion).reshape(motion.shape[0], -1), dim=-1).mean().item())
        if variant == "full":
            verdict = "baseline"
        elif delta < 0.05:
            verdict = "weak_route_effect"
        elif plan_metrics["far_unobserved_error_cm"] < rows[0]["plan_metrics"]["far_unobserved_error_cm"] - 1.0:
            verdict = "improves_vs_full"
        else:
            verdict = "changes_output"
        rows.append({
            "variant": variant,
            "motion_delta_vs_full": delta,
            "plan_metrics": plan_metrics,
            "dynamics": dyn,
            "transition": trans,
            "verdict": verdict,
        })

    full_row = rows[0]
    weak = [r["variant"] for r in rows[1:] if r["verdict"] == "weak_route_effect"]
    better = [r["variant"] for r in rows[1:] if r["verdict"] == "improves_vs_full"]
    if better:
        verdict = (
            f"Some target-route ablations improve far-unobserved error versus full condition ({better}); duplicated target routes may conflict and should be investigated before training changes."
        )
    elif len(weak) >= 4:
        verdict = (
            f"Most route ablations barely changed output ({weak}); target routing may be weak relative to sampling/model priors."
        )
    else:
        verdict = (
            "Target route ablations changed outputs but did not clearly improve the full condition; no target-route simplification is justified yet."
        )

    payload = {
        "config": str(args.config),
        "ckpt": str(args.ckpt),
        "seed": int(args.seed),
        "selected_clips": clip_metadata(batch),
        "variants": rows,
        "full_baseline": full_row,
        "verdict": verdict,
    }
    write_json(args.output, payload)
    _write_report(payload, args.md)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.md}")


if __name__ == "__main__":
    main()

