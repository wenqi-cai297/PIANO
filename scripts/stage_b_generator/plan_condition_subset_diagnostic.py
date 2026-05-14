"""Subset-aware plan-condition diagnostic for Stage B AnchorDiff.

This is a small wrapper around ``plan_condition_diagnostics.py`` helpers. It
evaluates the same plan variants on a balanced set of clips so the round report
can include subset-wise far-unobserved and GT-vs-zero/wrong plan gaps.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from dynamics_diagnostic import _balanced_subset_indices
from piano.data.dataset import collate_hoi
from piano.utils.clip_utils import load_clip_text_encoder
from plan_condition_diagnostics import (
    _build_cond,
    _build_dataset,
    _build_model,
    _compute_metrics,
    _fk_from_global,
    _gt_plan,
    _part_swapped_plan,
    _reversed_plan,
    _rot6d_to_mat,
    _shuffled_plan,
    _target_perturbed_plan,
    _wrong_clip_plan,
    _zero_plan,
)


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


def _extract_plan(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: batch[f"plan_{k}"].to(device) for k in PLAN_KEYS}


def _mean_dict(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    out: dict[str, Any] = {}
    keys = sorted(set().union(*(row.keys() for row in rows)))
    for key in keys:
        vals = [row[key] for row in rows if key in row]
        if not vals:
            continue
        if isinstance(vals[0], dict):
            out[key] = _mean_dict([v for v in vals if isinstance(v, dict)])
        elif isinstance(vals[0], (int, float, np.floating)):
            out[key] = float(np.mean(vals))
    return out


def _run_one_clip(
    *,
    main_batch: dict[str, Any],
    secondary_batch: dict[str, Any],
    model,
    object_encoder,
    clip_model,
    z_dims,
    cfg,
    device: torch.device,
    seed: int,
    cfg_scale: float,
) -> dict[str, dict[str, float]]:
    cond_main, total_t = _build_cond(
        main_batch, model, object_encoder, clip_model, z_dims, cfg, device,
    )
    plan_gt = _extract_plan(main_batch, device)
    plan_other = _extract_plan(secondary_batch, device)
    variants = {
        "gt": _gt_plan(plan_gt),
        "zero": _zero_plan(plan_gt),
        "wrong_clip": _wrong_clip_plan(plan_gt, plan_other),
        "shuffled_time": _shuffled_plan(plan_gt, seed=seed),
        "reversed_time": _reversed_plan(plan_gt, T=total_t),
        "target_perturbed": _target_perturbed_plan(plan_gt, sigma_m=0.10, seed=seed),
        "part_swapped": _part_swapped_plan(plan_gt),
    }

    part_to_joint = torch.tensor([20, 21, 10, 11, 0], dtype=torch.long, device=device)
    motion_gt = main_batch["motion"].to(device)
    rest_offsets = main_batch["rest_offsets"].to(device).float()
    seq_len = main_batch["seq_len"].to(device)
    seq_idx = torch.arange(total_t, device=device).unsqueeze(0)
    seq_mask = seq_idx < seq_len.unsqueeze(1)
    joints_gt = main_batch["joints"].to(device).float()

    metrics_by_variant: dict[str, dict[str, float]] = {}
    motion_outputs: dict[str, torch.Tensor] = {}
    for name, plan in variants.items():
        torch.manual_seed(seed)
        cond = {**cond_main, "interaction_plan": plan}
        with torch.no_grad():
            x0_pred = model.sample(
                cond=cond,
                seq_length=total_t,
                cfg_scale=cfg_scale,
                replacement="none",
                output_skip=False,
            )
        rot_6d = x0_pred[..., :132].view(1, total_t, 22, 6).float()
        root_world = x0_pred[..., 132:135].float()
        rot_mat = _rot6d_to_mat(rot_6d)
        rest_per_frame = rest_offsets.unsqueeze(1).expand(1, total_t, 22, 3)
        jpos_pred = _fk_from_global(rot_mat, rest_per_frame, root_world)
        metrics_by_variant[name] = _compute_metrics(
            jpos_pred=jpos_pred,
            jpos_gt=joints_gt,
            seq_mask=seq_mask,
            anchor_time=plan["anchor_time"],
            anchor_mask=plan["anchor_mask"],
            anchor_part=plan["anchor_part"],
            anchor_target_world=plan["anchor_target_world"],
            part_to_joint=part_to_joint,
            window=3,
        )
        motion_outputs[name] = x0_pred.detach().cpu()

    base = motion_outputs["gt"]
    for name, x0 in motion_outputs.items():
        metrics_by_variant[name]["motion_135_delta_vs_gt"] = float(
            (x0 - base).pow(2).sum(-1).sqrt().mean().item()
        )
    return metrics_by_variant


def _gap(metrics: dict[str, Any], variant: str, key: str) -> float:
    return float(metrics[variant][key] - metrics["gt"][key])


def _write_markdown(path: Path, results: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# v20 plan-condition subset diagnostic")
    lines.append("")
    lines.append(f"**Config:** `{results['config']}`")
    lines.append(f"**Ckpt:** `{results['ckpt']}`")
    lines.append(f"**Clips:** {results['num_clips']} balanced across subsets")
    lines.append("")
    lines.append("## Aggregate Plan Variants")
    lines.append("")
    lines.append("| variant | far_unobs cm | anchor realization cm | transition jump cm/fr | motion_135 delta |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, payload in results["metrics"].items():
        lines.append(
            f"| {name} | {payload['far_unobserved_error_cm']:.2f} | "
            f"{payload['plan_anchor_contact_realization_cm']:.2f} | "
            f"{payload['transition_local_vel_jump_cm_per_frame']:.2f} | "
            f"{payload['motion_135_delta_vs_gt']:.3f} |"
        )

    lines.append("")
    lines.append("## Subset-wise Plan Sensitivity")
    lines.append("")
    lines.append("| subset | clips | GT far | zero far | wrong far | GT-zero gap | GT-wrong gap | GT anchor real |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for subset, payload in results["subset_wise"].items():
        m = payload["metrics"]
        lines.append(
            f"| {subset} | {payload['num_clips']} | "
            f"{m['gt']['far_unobserved_error_cm']:.2f} | "
            f"{m['zero']['far_unobserved_error_cm']:.2f} | "
            f"{m['wrong_clip']['far_unobserved_error_cm']:.2f} | "
            f"{_gap(m, 'zero', 'far_unobserved_error_cm'):.2f} | "
            f"{_gap(m, 'wrong_clip', 'far_unobserved_error_cm'):.2f} | "
            f"{m['gt']['plan_anchor_contact_realization_cm']:.2f} |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--md", type=Path, required=True)
    parser.add_argument("--num-clips", type=int, default=4)
    parser.add_argument("--bucket", default="train", choices=["train", "val"])
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = _build_dataset(cfg, args.bucket, augment=False)
    indices = _balanced_subset_indices(dataset, int(args.num_clips))
    dataset = Subset(dataset, indices)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_hoi,
        num_workers=0,
    )
    batches = list(loader)
    if not batches:
        raise RuntimeError("No batches selected for plan-condition subset diagnostic")

    model, object_encoder, z_dims = _build_model(cfg, device)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model_state = state.get("model", state)
    model.load_state_dict(model_state)
    if "object_encoder" in state:
        object_encoder.load_state_dict(state["object_encoder"])
    elif "extra_modules" in state and "object_encoder" in state["extra_modules"]:
        object_encoder.load_state_dict(state["extra_modules"]["object_encoder"])
    model.eval()
    object_encoder.eval()

    clip_model = load_clip_text_encoder(
        device=device,
        model_name=str(cfg.model.text_encoder.clip_version),
        download_root=str(cfg.model.text_encoder.get("download_root", "cache/clip")),
    )

    rows_by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows_by_subset: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    per_clip: list[dict[str, Any]] = []
    for i, batch in enumerate(batches):
        secondary = batches[(i + 1) % len(batches)]
        subset = str(batch["subset"][0])
        metrics = _run_one_clip(
            main_batch=batch,
            secondary_batch=secondary,
            model=model,
            object_encoder=object_encoder,
            clip_model=clip_model,
            z_dims=z_dims,
            cfg=cfg,
            device=device,
            seed=int(args.seed) + i,
            cfg_scale=float(args.cfg_scale),
        )
        for variant, payload in metrics.items():
            rows_by_variant[variant].append(payload)
            rows_by_subset[subset][variant].append(payload)
        per_clip.append(
            {
                "subset": subset,
                "seq_id": str(batch["seq_id"][0]),
                "seq_len": int(batch["seq_len"].item()),
            }
        )
        print(f"[{i+1}/{len(batches)}] {subset}/{batch['seq_id'][0]}")

    aggregate = {variant: _mean_dict(rows) for variant, rows in rows_by_variant.items()}
    subset_wise = {
        subset: {
            "num_clips": len(next(iter(variant_rows.values()))) if variant_rows else 0,
            "metrics": {
                variant: _mean_dict(rows)
                for variant, rows in variant_rows.items()
            },
        }
        for subset, variant_rows in rows_by_subset.items()
    }
    results = {
        "config": str(args.config),
        "ckpt": str(args.ckpt),
        "bucket": str(args.bucket),
        "cfg_scale": float(args.cfg_scale),
        "seed": int(args.seed),
        "num_clips": len(batches),
        "per_clip": per_clip,
        "metrics": aggregate,
        "subset_wise": subset_wise,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    _write_markdown(args.md, results)
    print(f"Wrote JSON to {args.output}")
    print(f"Wrote Markdown to {args.md}")


if __name__ == "__main__":
    main()
