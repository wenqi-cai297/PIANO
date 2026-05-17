"""Condition branch norm / scale / sensitivity diagnostic for Stage B v18."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from diagnostic_common import (
    clip_metadata,
    extract_plan,
    format_md_table,
    load_checkpoint,
    merge_single_batches,
    selected_balanced_batches,
    stats_list,
    write_json,
)
from dynamics_diagnostic import _build_cond, _build_model
from piano.utils.clip_utils import load_clip_text_encoder


def _parse_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _branch_stats(branches: dict[str, torch.Tensor]) -> dict[str, float]:
    norms = {
        name: torch.linalg.vector_norm(tensor.float(), dim=-1).detach().cpu().numpy().reshape(-1)
        for name, tensor in branches.items()
    }
    out: dict[str, float] = {}
    for name, values in norms.items():
        out[f"{name}_norm_mean"] = float(values.mean())
        out[f"{name}_norm_p95"] = float(np.percentile(values, 95))
    motion_mean = out.get("motion_norm_mean", 0.0)
    for name in ("zint", "object", "hint"):
        out[f"{name}_over_motion_norm_mean"] = float(out.get(f"{name}_norm_mean", 0.0) / motion_mean) if motion_mean > 1e-12 else 0.0

    hm = branches["motion"].float().reshape(-1, branches["motion"].shape[-1])
    for name in ("zint", "object", "hint"):
        hx = branches[name].float().reshape(-1, branches[name].shape[-1])
        cos = torch.nn.functional.cosine_similarity(hm, hx, dim=-1)
        out[f"cos_motion_{name}_mean"] = float(cos.mean().detach().cpu().item())
        out[f"cos_motion_{name}_abs_mean"] = float(cos.abs().mean().detach().cpu().item())
    return out


def _write_report(payload: dict[str, Any], path: Path) -> None:
    rows = [["t", "motion", "z_int", "object", "hint", "z/m", "o/m", "h/m", "cos m-z", "cos m-o", "cos m-h"]]
    for row in payload["by_timestep"]:
        rows.append([
            row["t"],
            f"{row['motion_norm_mean']:.3f}",
            f"{row['zint_norm_mean']:.3f}",
            f"{row['object_norm_mean']:.3f}",
            f"{row['hint_norm_mean']:.3f}",
            f"{row['zint_over_motion_norm_mean']:.3f}",
            f"{row['object_over_motion_norm_mean']:.3f}",
            f"{row['hint_over_motion_norm_mean']:.3f}",
            f"{row['cos_motion_zint_mean']:.3f}",
            f"{row['cos_motion_object_mean']:.3f}",
            f"{row['cos_motion_hint_mean']:.3f}",
        ])
    subset_rows = [["subset", "clips", "z/m", "o/m", "h/m"]]
    for subset, row in payload["by_subset"].items():
        subset_rows.append([
            subset,
            row["clips"],
            f"{row['zint_over_motion_norm_mean']:.3f}",
            f"{row['object_over_motion_norm_mean']:.3f}",
            f"{row['hint_over_motion_norm_mean']:.3f}",
        ])
    lines = [
        "# Condition Branch Signal Diagnostic",
        "",
        f"- Config: `{payload['config']}`",
        f"- Checkpoint: `{payload['ckpt']}`",
        f"- Clips: {len(payload['selected_clips'])}",
        "",
        "## Verdict",
        "",
        payload["verdict"],
        "",
        "## Branch Norms by Timestep",
        "",
        format_md_table(rows),
        "",
        "## Subset Average Ratios",
        "",
        format_md_table(subset_rows),
        "",
        "## Selected Clips",
        "",
        format_md_table([["subset", "seq_id", "seq_len", "text"]] + [[r["subset"], r["seq_id"], r["seq_len"], r["text"][:80]] for r in payload["selected_clips"]]),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/training/anchordiff_v18_a1_FULL_DATA.yaml"))
    parser.add_argument("--ckpt", type=Path, default=Path("runs/training/stageB_anchordiff_v18_a1_FULL_DATA/final.pt"))
    parser.add_argument("--output", type=Path, default=Path("analyses/2026-05-14_condition_branch_signal_diagnostic.json"))
    parser.add_argument("--md", type=Path, default=Path("analyses/2026-05-14_condition_branch_signal_diagnostic.md"))
    parser.add_argument("--bucket", choices=["train", "val"], default="train")
    parser.add_argument("--num-clips", type=int, default=8)
    parser.add_argument("--num-candidates", type=int, default=96)
    parser.add_argument("--timesteps", type=str, default="100,300,500,700,900")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    batches = selected_balanced_batches(
        cfg,
        bucket=args.bucket,
        num_clips=int(args.num_clips),
        num_candidates=int(args.num_candidates),
        balanced_subsets=True,
    )
    batch = merge_single_batches(batches)
    model, object_encoder, z_dims = _build_model(cfg, device)
    load_checkpoint(model, object_encoder, args.ckpt)
    clip_model = load_clip_text_encoder(
        device=device,
        model_name=str(cfg.model.text_encoder.clip_version),
        download_root=str(cfg.model.text_encoder.get("download_root", "cache/clip")),
    )
    cond, total_t = _build_cond(batch, model, object_encoder, clip_model, z_dims, cfg, device)
    cond = {**cond, "interaction_plan": extract_plan(batch, device)}
    motion = batch["motion"].to(device).float()

    den = model.denoiser
    if not hasattr(den, "v12_input_proj") or den.plan_encoder is None:
        raise RuntimeError("This diagnostic expects v18 V12InputProjection + plan_encoder")
    with torch.no_grad():
        plan_tokens, plan_mask, plan_hint = den.plan_encoder(cond["interaction_plan"], total_t)
        del plan_tokens, plan_mask
    by_timestep: list[dict[str, Any]] = []
    per_subset_rows: dict[str, list[dict[str, float]]] = {}
    subsets = list(batch["subset"])
    for t_int in _parse_ints(args.timesteps):
        t = torch.full((motion.shape[0],), int(t_int), device=device, dtype=torch.long)
        noise = torch.randn_like(motion)
        x_t = model.diffusion.q_sample(motion, t, noise)
        with torch.no_grad():
            branches = {
                "motion": den.v12_input_proj.motion_proj(x_t),
                "zint": den.v12_input_proj.zint_proj(cond["z_int"]),
                "object": den.v12_input_proj.obj_proj(cond["object_world_traj"]),
                "hint": den.v12_input_proj.hint_proj(plan_hint),
            }
        row = {"t": int(t_int), **_branch_stats(branches)}
        by_timestep.append(row)
        for i, subset in enumerate(subsets):
            one = {name: tensor[i : i + 1] for name, tensor in branches.items()}
            per_subset_rows.setdefault(str(subset), []).append(_branch_stats(one))

    by_subset: dict[str, Any] = {}
    for subset, rows in per_subset_rows.items():
        by_subset[subset] = {
            "clips": int(sum(1 for s in subsets if s == subset)),
            "zint_over_motion_norm_mean": float(np.mean([r["zint_over_motion_norm_mean"] for r in rows])),
            "object_over_motion_norm_mean": float(np.mean([r["object_over_motion_norm_mean"] for r in rows])),
            "hint_over_motion_norm_mean": float(np.mean([r["hint_over_motion_norm_mean"] for r in rows])),
        }

    min_aux = min(
        min(row["zint_over_motion_norm_mean"], row["object_over_motion_norm_mean"], row["hint_over_motion_norm_mean"])
        for row in by_timestep
    )
    if min_aux < 0.05:
        verdict = (
            "At least one auxiliary branch is below 5% of the motion branch norm at some timestep; condition injection weakness remains plausible and should be localized to the weak branch before architectural changes."
        )
    else:
        verdict = (
            "Auxiliary branch norms are not near-zero relative to the motion branch; this diagnostic does not support blaming simple summed-projection scale collapse."
        )

    payload = {
        "config": str(args.config),
        "ckpt": str(args.ckpt),
        "seed": int(args.seed),
        "timesteps": _parse_ints(args.timesteps),
        "selected_clips": clip_metadata(batch),
        "by_timestep": by_timestep,
        "by_subset": by_subset,
        "ratio_stats": {
            "zint_over_motion": stats_list([row["zint_over_motion_norm_mean"] for row in by_timestep]),
            "object_over_motion": stats_list([row["object_over_motion_norm_mean"] for row in by_timestep]),
            "hint_over_motion": stats_list([row["hint_over_motion_norm_mean"] for row in by_timestep]),
        },
        "verdict": verdict,
    }
    write_json(args.output, payload)
    _write_report(payload, args.md)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.md}")


if __name__ == "__main__":
    main()

