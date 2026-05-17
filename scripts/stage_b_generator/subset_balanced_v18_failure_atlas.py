"""Subset-balanced v18 failure atlas (Round 9, Task 3).

Loads clips by direct dataset_global_index from a selection JSON
(produced by build_subset_balanced_failure_selection.py), runs v18
default DDPM rollout at seed 42, and emits per-clip 3D strip + curve
PNGs and an index markdown with manual-review checkboxes.

This is a thin wrapper around v18_failure_atlas helpers that bypasses
``_build_selected_batches`` (which falls back to chairs when seq_ids
are not in the first-N-per-subset scan window).
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
from torch.utils.data import DataLoader, Subset

from condition_route_causal_sensitivity_diagnostic import _full_rollout
from diagnostic_common import (
    event_records_from_contact, extract_plan, load_checkpoint, make_seq_mask,
    merge_single_batches,
)
from dynamics_diagnostic import _build_cond, _build_dataset, _build_model, _fk_from_motion_135
from piano.data.dataset import collate_hoi
from piano.utils.clip_utils import load_clip_text_encoder
from v18_failure_atlas import HAND_SPECS, _strip_plot, _distance_curves_plot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=Path("configs/training/anchordiff_v18_a1_FULL_DATA.yaml"))
    parser.add_argument("--ckpt", type=Path,
                        default=Path("runs/training/stageB_anchordiff_v18_a1_FULL_DATA/final.pt"))
    parser.add_argument("--selection-json", type=Path,
                        default=Path("analyses/2026-05-19_subset_balanced_failure_selection.json"))
    parser.add_argument("--md", type=Path,
                        default=Path("analyses/2026-05-19_subset_balanced_v18_failure_atlas.md"))
    parser.add_argument("--visuals-dir", type=Path,
                        default=Path("analyses/visuals/2026-05-19_subset_balanced_v18_failure_atlas"))
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--bucket", choices=["train", "val"], default="train")
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n-strip-frames", type=int, default=5)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sel_payload = json.loads(args.selection_json.read_text(encoding="utf-8"))
    entries = sel_payload.get("selected", [])
    if not entries:
        raise SystemExit("Empty selection")
    if not all("dataset_global_index" in e for e in entries):
        raise SystemExit("Selection must contain dataset_global_index for each entry")

    full_ds = _build_dataset(cfg, args.bucket)
    indices = [int(e["dataset_global_index"]) for e in entries]
    subset_ds = Subset(full_ds, indices)
    loader = DataLoader(subset_ds, batch_size=1, shuffle=False, collate_fn=collate_hoi, num_workers=0)
    clip_batches = list(loader)
    if not clip_batches:
        raise SystemExit("Selection produced no batches.")
    batch = merge_single_batches(clip_batches)
    B = int(batch["motion"].shape[0])
    print(f"Atlas: {B} clips loaded directly by dataset_global_index.")

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

    motion_gt = batch["motion"].to(device).float()
    gt_joints = batch["joints"].to(device).float()
    rest_offsets = batch["rest_offsets"].to(device).float()
    object_positions = batch["object_positions"].to(device).float()
    contact_state = batch["contact_state"].to(device).float()
    seq_lens = batch["seq_len"].to(device)
    seq_mask = make_seq_mask(seq_lens, total_t, device)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    seq_ids = list(batch.get("seq_id", []))
    subsets = list(batch.get("subset", []))

    events = event_records_from_contact(
        contact_state, batch["seq_len"], threshold=float(args.threshold), hands_only=True,
    )
    first_ev_per_clip: dict[int, dict[str, Any]] = {}
    for ev in events:
        first_ev_per_clip.setdefault(int(ev["batch"]), ev)

    gen_motions: dict[int, Tensor] = {}
    for s in seeds:
        m = _full_rollout(
            model, base_cond, seq_length=total_t, seed=int(s),
            cfg_scale=float(args.cfg_scale), alpha_hint=1.0, sampler="ddpm",
        )
        gen_motions[s] = m.detach()
        print(f"  rolled out seed={s}")

    args.visuals_dir.mkdir(parents=True, exist_ok=True)
    rows_md: list[str] = [
        "# Subset-Balanced v18 Visual Failure Atlas (Round 9, Task 3)",
        "",
        f"- Config: `{args.config}`",
        f"- Checkpoint: `{args.ckpt}`",
        f"- Selection: `{args.selection_json}`",
        f"- Seeds: {seeds}",
        f"- Clips: {B}",
        "",
        "## Manual-review categories (checkbox per clip)",
        "",
        "- [ ] visually acceptable",
        "- [ ] frozen / under-motion",
        "- [ ] over-motion / jitter",
        "- [ ] wrong contact direction",
        "- [ ] contact timing wrong",
        "- [ ] root drift / global placement issue",
        "- [ ] object geometry / transform suspicious",
        "- [ ] pseudo-label/event suspicious",
        "- [ ] metric artifact",
        "- [ ] text/action mismatch",
        "- [ ] subset-specific failure",
        "- [ ] needs video review",
        "",
        "## Per-clip strips and curves",
        "",
        "| subset | seq_id | seed | event | text short | strip | curves |",
        "|---|---|---:|---|---|---|---|",
    ]

    # Extract plan anchors for curve plot
    plan_anchor_time = batch["plan_anchor_time"].cpu().numpy()
    plan_anchor_mask = batch["plan_anchor_mask"].cpu().numpy()
    plan_anchor_part = batch["plan_anchor_part"].cpu().numpy()
    for b in range(B):
        sid = str(seq_ids[b]) if b < len(seq_ids) else f"clip{b}"
        sub = str(subsets[b]) if b < len(subsets) else "?"
        seq_len_b = int(seq_lens[b].item())
        text = str(batch["text"][b]) if b < len(batch.get("text", [])) else ""
        ev = first_ev_per_clip.get(b, None)
        if ev is not None:
            ev_frame = int(ev["frame"])
            ev_kind = str(ev["kind"])
            ev_part = str(ev["part"])
        else:
            ev_frame = max(1, seq_len_b // 2)
            ev_kind = "none"
            ev_part = "—"
        # ±n_strip_frames//2 + 1 evenly-spaced frames around event_frame
        n_frames = int(args.n_strip_frames)
        half = n_frames // 2
        lo = max(0, ev_frame - half - 2)
        hi = min(seq_len_b - 1, ev_frame + half + 2)
        if hi <= lo:
            hi = min(seq_len_b - 1, lo + 1)
        frames = np.linspace(lo, hi, n_frames, dtype=int).tolist()

        gt_j = gt_joints[b].detach().cpu().numpy()
        obj_pos = object_positions[b].detach().cpu().numpy()
        cs = contact_state[b].detach().cpu().numpy()

        # Get active plan anchor times for this clip
        anc_times = []
        for k in range(plan_anchor_mask.shape[1]):
            if bool(plan_anchor_mask[b, k]):
                anc_times.append(int(plan_anchor_time[b, k]))
        # Per-seed: render strip + curves
        for s in seeds:
            gen_motion = gen_motions[s]
            gen_j_full = _fk_from_motion_135(gen_motion, rest_offsets).detach().cpu().numpy()
            gen_j = gen_j_full[b]
            tag = f"{sub}_{sid}_seed{s}"
            strip_path = args.visuals_dir / f"{tag}_strip.png"
            curves_path = args.visuals_dir / f"{tag}_curves.png"
            _strip_plot(
                title=f"{sub}/{sid} | seed={s} | {ev_kind} {ev_part} @ f={ev_frame}\n{text[:80]}",
                frames=frames,
                sources={"GT": gt_j, f"v18 seed{s}": gen_j},
                object_positions=obj_pos, out_path=strip_path,
                event_frame=ev_frame,
            )
            _distance_curves_plot(
                gt_joints=gt_j, gen_joints=gen_j, contact_state=cs,
                obj_positions=obj_pos, seq_len=seq_len_b,
                plan_anchor_times=anc_times,
                plan_anchor_parts_active=[],
                out_path=curves_path,
                title=f"{sub}/{sid} | curves | seed={s}",
            )
            rows_md.append(
                f"| {sub} | {sid} | {s} | {ev_kind} {ev_part} f={ev_frame} | "
                f"{text[:90].replace('|', '/')} | "
                f"`{strip_path.as_posix()}` | `{curves_path.as_posix()}` |"
            )

    rows_md.append("")
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text("\n".join(rows_md) + "\n", encoding="utf-8")
    print(f"Wrote {args.md}")


if __name__ == "__main__":
    main()
