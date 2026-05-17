"""Plan-hint local/event distribution audit (Diagnostic 3 of the 2026-05-14
condition-route round).

The previous condition_branch_signal_diagnostic showed plan_hint / motion
projected-branch norm is 0.035-0.052 globally. This script asks the
follow-up question: is plan_hint at least *locally* stronger near
anchors or event windows? If yes, the global ratio understates causal
relevance.

For each selected clip and each timestep t∈{100,300,500,700,900}:
  - plan_hint raw norm per frame (from plan_encoder)
  - hint_proj(plan_hint) projected norm per frame
  - motion_proj(x_t) projected norm per frame
  - hint_proj / motion_proj per-frame ratio

Per-frame stats are aggregated over five region masks:
  - near-anchor       : within ±anchor_window frames of any valid anchor
  - near-event        : within onset_pre / release_post window of any
                        hand contact onset/release
  - stable-contact    : frames within a stable_contact segment
  - far               : not near anchor AND not near event
  - all               : sequence-mask only

No training, no model state mutation. Forward hooks are removed in a
``finally`` block.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import Tensor

from diagnostic_common import (
    clip_metadata,
    event_records_from_contact,
    extract_plan,
    format_md_table,
    load_checkpoint,
    make_seq_mask,
    merge_single_batches,
    write_json,
)
from dynamics_diagnostic import _build_cond, _build_model
from piano.utils.clip_utils import load_clip_text_encoder
from recon_ladder_truncated_rollout_diagnostic import (
    _build_selected_batches,
    _load_selection,
)


def _parse_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _region_masks(
    *,
    plan: dict[str, Tensor],
    events: list[dict[str, Any]],
    seq_mask: Tensor,
    anchor_window: int,
    onset_pre: int,
    release_post: int,
    stable_phase_idx: int = 1,
) -> dict[str, Tensor]:
    """Build (B, T) boolean masks. Frames outside seq_mask are False."""
    B, T = seq_mask.shape
    device = seq_mask.device
    t_grid = torch.arange(T, device=device).view(1, 1, T)

    # near-anchor
    a_t = plan["anchor_time"].long().view(B, -1, 1)
    a_mask = plan["anchor_mask"].bool().view(B, -1, 1)
    near_anchor = (
        (a_t - int(anchor_window) <= t_grid)
        & (t_grid <= a_t + int(anchor_window))
        & a_mask
    ).any(dim=1) & seq_mask  # (B, T)

    # near-event (from extracted hand contact onset/release events)
    near_event = torch.zeros((B, T), dtype=torch.bool, device=device)
    for ev in events:
        b = int(ev["batch"])
        frame = int(ev["frame"])
        if ev["kind"] == "onset":
            lo, hi = max(0, frame - int(onset_pre)), min(T - 1, frame)
        else:
            lo, hi = max(0, frame), min(T - 1, frame + int(release_post))
        near_event[b, lo : hi + 1] = True
    near_event = near_event & seq_mask

    # stable-contact (from segment_phase == stable_phase_idx)
    stable_mask = torch.zeros((B, T), dtype=torch.bool, device=device)
    if "segment_start" in plan and "segment_phase" in plan:
        s_start = plan["segment_start"].long()
        s_end = plan["segment_end"].long()
        s_phase = plan["segment_phase"].long()
        s_mask_pl = plan["segment_mask"].bool()
        for b in range(B):
            for k in range(s_start.shape[1]):
                if not bool(s_mask_pl[b, k].item()):
                    continue
                if int(s_phase[b, k].item()) != int(stable_phase_idx):
                    continue
                lo = int(s_start[b, k].item())
                hi = int(s_end[b, k].item())
                if hi < lo or lo < 0 or lo >= T:
                    continue
                hi = min(hi, T - 1)
                stable_mask[b, lo : hi + 1] = True
    stable_mask = stable_mask & seq_mask

    far_mask = seq_mask & ~near_anchor & ~near_event

    return {
        "near_anchor": near_anchor,
        "near_event": near_event,
        "stable_contact": stable_mask,
        "far": far_mask,
        "all": seq_mask,
    }


def _masked_mean(x: Tensor, mask: Tensor) -> float:
    if mask.dtype != torch.bool:
        mask = mask.bool()
    if mask.sum() == 0:
        return 0.0
    return float(x[mask].mean().item())


def _per_region_stats(
    per_frame_values: dict[str, Tensor],
    masks: dict[str, Tensor],
) -> dict[str, dict[str, float]]:
    """For each region mask, compute mean of each per-frame value tensor."""
    out: dict[str, dict[str, float]] = {}
    for region, mask in masks.items():
        out[region] = {
            name: _masked_mean(values, mask)
            for name, values in per_frame_values.items()
        }
        out[region]["n_frames"] = float(int(mask.sum().item()))
    return out


@torch.no_grad()
def _compute_per_timestep(
    model,
    motion_gt: Tensor,
    plan_dict: dict[str, Tensor],
    *,
    t_int: int,
    seed: int,
    seq_mask: Tensor,
    masks: dict[str, Tensor],
) -> dict[str, Any]:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    device = motion_gt.device
    t = torch.full((motion_gt.shape[0],), int(t_int), device=device, dtype=torch.long)
    noise = torch.randn_like(motion_gt)
    x_t = model.diffusion.q_sample(motion_gt, t, noise)

    den = model.denoiser
    # Run plan encoder once to get plan_hint (B, T, d_hint).
    _, _, plan_hint = den.plan_encoder(plan_dict, motion_gt.shape[1])

    motion_proj_out = den.v12_input_proj.motion_proj(x_t)           # (B, T, D)
    hint_proj_out = den.v12_input_proj.hint_proj(plan_hint)         # (B, T, D)
    # We only need motion / hint / raw_hint norms here. Other branches are
    # measured in condition_branch_signal_diagnostic (already on disk).

    hint_raw_norm = torch.linalg.vector_norm(plan_hint.float(), dim=-1)         # (B, T)
    hint_proj_norm = torch.linalg.vector_norm(hint_proj_out.float(), dim=-1)    # (B, T)
    motion_proj_norm = torch.linalg.vector_norm(motion_proj_out.float(), dim=-1)
    hint_over_motion = hint_proj_norm / motion_proj_norm.clamp_min(1e-12)
    # Per-frame cosine between projected hint and projected motion
    hp = hint_proj_out.float().reshape(-1, hint_proj_out.shape[-1])
    mp = motion_proj_out.float().reshape(-1, motion_proj_out.shape[-1])
    cos_per_frame = torch.nn.functional.cosine_similarity(hp, mp, dim=-1).reshape(
        hint_proj_out.shape[0], hint_proj_out.shape[1],
    )

    per_frame = {
        "hint_raw_norm": hint_raw_norm,
        "hint_proj_norm": hint_proj_norm,
        "motion_proj_norm": motion_proj_norm,
        "hint_over_motion_proj_norm": hint_over_motion,
        "cos_motion_hint_proj": cos_per_frame,
    }
    region_stats = _per_region_stats(per_frame, masks)
    return {
        "t": int(t_int),
        "regions": region_stats,
        "global_means": {
            name: float(v[seq_mask].mean().item()) if seq_mask.sum() > 0 else 0.0
            for name, v in per_frame.items()
        },
    }


def _verdict(payload: dict[str, Any]) -> str:
    # Compare hint/motion ratio in near_event/near_anchor vs far for the middle timestep.
    timesteps = payload["timesteps"]
    if not timesteps:
        return "No timesteps configured."
    mid = timesteps[len(timesteps) // 2]
    block = None
    for entry in payload["per_timestep"]:
        if int(entry["t"]) == int(mid):
            block = entry
            break
    if block is None:
        return "Mid timestep not found in payload."
    near_anchor = float(block["regions"]["near_anchor"].get("hint_over_motion_proj_norm", 0.0))
    near_event = float(block["regions"]["near_event"].get("hint_over_motion_proj_norm", 0.0))
    far = float(block["regions"]["far"].get("hint_over_motion_proj_norm", 0.0))
    if max(near_anchor, near_event) > far * 1.5 and max(near_anchor, near_event) >= 0.08:
        return (
            f"Hint/motion ratio is locally stronger near anchors/events "
            f"(anchor={near_anchor:.3f}, event={near_event:.3f}) than far "
            f"({far:.3f}). Local channel is present even though global ratio is small."
        )
    if max(near_anchor, near_event) <= far * 1.2 and far > 0.0:
        return (
            f"Hint/motion ratio is globally flat (anchor={near_anchor:.3f}, "
            f"event={near_event:.3f}, far={far:.3f}). Plan hint is not "
            f"temporally localized; target-aware hint may not be encoded effectively."
        )
    return (
        f"Mixed locality result (anchor={near_anchor:.3f}, event={near_event:.3f}, "
        f"far={far:.3f}). Read tables before drawing a verdict."
    )


def _write_report(payload: dict[str, Any], path: Path) -> None:
    lines: list[str] = [
        "# Plan-Hint Local / Event Distribution Audit",
        "",
        f"- Config: `{payload['config']}`",
        f"- Checkpoint: `{payload['ckpt']}`",
        f"- Seed: {payload['seed']}",
        f"- Clips: {len(payload['selected_clips'])}",
        f"- Anchor window: ±{payload['anchor_window']} frames",
        f"- Event windows: onset pre-{payload['onset_pre']}, release post-{payload['release_post']}",
        "",
        "## Distribution-shift caveat",
        "",
        "All measurements are taken under q(x_t | x_0) at fixed seeds. No "
        "training, no model state changes. Per-frame norms come from "
        "`V12InputProjection.motion_proj` / `hint_proj` and the plan encoder "
        "output `plan_hint`.",
        "",
    ]

    # Region table per timestep
    for entry in payload["per_timestep"]:
        t = int(entry["t"])
        lines.extend([f"## t={t}", ""])
        rows = [["region", "n_frames", "||hint_raw||", "||motion_proj||", "||hint_proj||", "hint/motion", "cos(motion,hint)"]]
        for region in ("near_anchor", "near_event", "stable_contact", "far", "all"):
            r = entry["regions"][region]
            rows.append([
                region,
                f"{int(r.get('n_frames', 0))}",
                f"{r['hint_raw_norm']:.3f}",
                f"{r['motion_proj_norm']:.3f}",
                f"{r['hint_proj_norm']:.3f}",
                f"{r['hint_over_motion_proj_norm']:.4f}",
                f"{r['cos_motion_hint_proj']:.3f}",
            ])
        lines.append(format_md_table(rows))
        lines.append("")

    lines.extend(["## Selected Clips", ""])
    rows = [["subset", "seq_id", "seq_len", "text"]]
    for c in payload["selected_clips"]:
        rows.append([c["subset"], c["seq_id"], c["seq_len"], c["text"][:80]])
    lines.append(format_md_table(rows))
    lines.append("")

    lines.extend(["## Verdict", "", payload["verdict"], ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/training/anchordiff_v18_a1_FULL_DATA.yaml"))
    parser.add_argument("--ckpt", type=Path, default=Path("runs/training/stageB_anchordiff_v18_a1_FULL_DATA/final.pt"))
    parser.add_argument("--output", type=Path, default=Path("analyses/2026-05-14_plan_hint_locality_audit.json"))
    parser.add_argument("--md", type=Path, default=Path("analyses/2026-05-14_plan_hint_locality_audit.md"))
    parser.add_argument("--selection-json", type=Path, default=Path("analyses/2026-05-13_v18_recon_ladder_truncated_rollout.json"))
    parser.add_argument("--max-clips", type=int, default=8)
    parser.add_argument("--num-candidates", type=int, default=96)
    parser.add_argument("--bucket", choices=["train", "val"], default="train")
    parser.add_argument("--balanced-subsets", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--timesteps", type=str, default="100,300,500,700,900")
    parser.add_argument("--anchor-window", type=int, default=3)
    parser.add_argument("--onset-pre", type=int, default=10)
    parser.add_argument("--release-post", type=int, default=10)
    parser.add_argument("--stable-phase-idx", type=int, default=1,
                        help="phase index for stable_contact in segment_phase encoding.")
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
        raise RuntimeError("No clips selected for plan-hint locality audit")
    batch = merge_single_batches([item[1] for item in selected])

    model, object_encoder, z_dims = _build_model(cfg, device)
    load_checkpoint(model, object_encoder, args.ckpt)
    clip_model = load_clip_text_encoder(
        device=device,
        model_name=str(cfg.model.text_encoder.clip_version),
        download_root=str(cfg.model.text_encoder.get("download_root", "cache/clip")),
    )
    _, total_t = _build_cond(batch, model, object_encoder, clip_model, z_dims, cfg, device)
    plan_dict = extract_plan(batch, device)
    motion_gt = batch["motion"].to(device).float()
    contact_state = batch["contact_state"].to(device).float()
    seq_mask = make_seq_mask(batch["seq_len"], total_t, device)
    events = event_records_from_contact(
        contact_state, batch["seq_len"], threshold=float(args.threshold), hands_only=True,
    )
    masks = _region_masks(
        plan=plan_dict,
        events=events,
        seq_mask=seq_mask,
        anchor_window=int(args.anchor_window),
        onset_pre=int(args.onset_pre),
        release_post=int(args.release_post),
        stable_phase_idx=int(args.stable_phase_idx),
    )

    timesteps = _parse_ints(args.timesteps)
    per_timestep: list[dict[str, Any]] = []
    for t_int in timesteps:
        entry = _compute_per_timestep(
            model, motion_gt, plan_dict,
            t_int=t_int, seed=int(args.seed) + int(t_int),
            seq_mask=seq_mask, masks=masks,
        )
        per_timestep.append(entry)

    payload = {
        "config": str(args.config),
        "ckpt": str(args.ckpt),
        "seed": int(args.seed),
        "timesteps": timesteps,
        "anchor_window": int(args.anchor_window),
        "onset_pre": int(args.onset_pre),
        "release_post": int(args.release_post),
        "n_events": len(events),
        "selected_clips": clip_metadata(batch, events),
        "region_frame_counts": {
            region: int(mask.sum().item())
            for region, mask in masks.items()
        },
        "per_timestep": per_timestep,
        "caveat": (
            "Per-frame projected-branch norms measured under q(x_t|x_0). "
            "Diagnostic-only; no inference-time deployment proposed."
        ),
    }
    payload["verdict"] = _verdict(payload)

    write_json(args.output, payload)
    _write_report(payload, args.md)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.md}")
    print(f"Verdict: {payload['verdict']}")


if __name__ == "__main__":
    main()
