"""v18 visual failure atlas (round 5, Diag 4).

Compact human-reviewable artifact for v18 failure analysis. For the
16 transition-heavy clips × 1 seed (default 42) under default DDPM,
saves:

  * per-clip metadata + first-event marker
  * per-clip frame strips (5 evenly-spaced frames spanning the event
    window, rows: GT vs v18 baseline α=1) — reuses the same strip-plot
    style as z_target_visual_sanity_review.py
  * per-clip hand-object distance + contact_state curves with anchor
    markers (saved alongside the strips)
  * index markdown listing every PNG and the dominant numeric failure
    flags (under-motion / over-motion / wrong-direction proxies)

NOT a metric replacement — atlas is meant for manual review.
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

from condition_route_causal_sensitivity_diagnostic import _full_rollout
from diagnostic_common import (
    event_records_from_contact,
    extract_plan,
    load_checkpoint,
    make_seq_mask,
    merge_single_batches,
)
from dynamics_diagnostic import _build_cond, _build_model, _fk_from_motion_135
from piano.inference.visualize_motion import SKELETON_CONNECTIONS
from piano.utils.clip_utils import load_clip_text_encoder
from recon_ladder_truncated_rollout_diagnostic import (
    _build_selected_batches,
    _load_selection,
)


HAND_SPECS = (("L_hand", 20, 0), ("R_hand", 21, 1))


def _strip_plot(
    *, title: str, frames: list[int],
    sources: dict[str, np.ndarray], object_positions: np.ndarray,
    out_path: Path, event_frame: int | None,
    elev: float = 15.0, azim: float = -60.0,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows = len(sources)
    n_cols = len(frames)
    all_pts: list[np.ndarray] = []
    for arr in sources.values():
        for f in frames:
            if 0 <= f < arr.shape[0]:
                all_pts.append(arr[f])
                all_pts.append(object_positions[f : f + 1])
    if not all_pts:
        return
    stack = np.concatenate(all_pts, axis=0)
    center = (stack.min(0) + stack.max(0)) * 0.5
    max_range = float((stack.max(0) - stack.min(0)).max()) * 0.55 + 0.2

    fig = plt.figure(figsize=(2.4 * n_cols, 2.7 * n_rows))
    for r, (label, joints) in enumerate(sources.items()):
        for c, f in enumerate(frames):
            ax = fig.add_subplot(n_rows, n_cols, r * n_cols + c + 1, projection="3d")
            if 0 <= f < joints.shape[0]:
                j = joints[f]
                ax.scatter(j[:, 0], j[:, 2], j[:, 1], c="#1f77b4", s=6)
                for a, b in SKELETON_CONNECTIONS:
                    ax.plot([j[a, 0], j[b, 0]], [j[a, 2], j[b, 2]], [j[a, 1], j[b, 1]],
                            c="0.4", linewidth=0.6)
                p = object_positions[f]
                ax.scatter([p[0]], [p[2]], [p[1]], c="#d62728", s=24, alpha=0.9)
            marker = " *" if event_frame is not None and f == int(event_frame) else ""
            ax.set_title(f"{label} f={f}{marker}", fontsize=7)
            ax.set_xlim(center[0] - max_range, center[0] + max_range)
            ax.set_ylim(center[2] - max_range, center[2] + max_range)
            ax.set_zlim(center[1] - max_range, center[1] + max_range)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            ax.view_init(elev=elev, azim=azim)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _distance_curves_plot(
    *,
    gt_joints: np.ndarray, gen_joints: np.ndarray, contact_state: np.ndarray,
    obj_positions: np.ndarray, seq_len: int,
    plan_anchor_times: list[int], plan_anchor_parts_active: list[list[int]],
    out_path: Path, title: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    # row 0: contact_state
    for part, _j, idx in HAND_SPECS:
        axes[0].plot(contact_state[:seq_len, idx], label=part)
    axes[0].set_ylabel("contact_state")
    axes[0].legend(fontsize=7)
    axes[0].set_ylim(-0.1, 1.1)
    # row 1: hand-objectCOM distance (GT vs gen)
    for part, joint, _idx in HAND_SPECS:
        d_gt = np.linalg.norm(gt_joints[:seq_len, joint] - obj_positions[:seq_len], axis=-1) * 100.0
        d_gn = np.linalg.norm(gen_joints[:seq_len, joint] - obj_positions[:seq_len], axis=-1) * 100.0
        axes[1].plot(d_gt, label=f"GT {part}", linestyle="-")
        axes[1].plot(d_gn, label=f"gen {part}", linestyle="--")
    axes[1].set_ylabel("hand→objCOM (cm)")
    axes[1].legend(fontsize=7)
    # row 2: hand local velocity (root-aligned) — gen vs gt
    def _local_vel(j: np.ndarray, joint: int) -> np.ndarray:
        if len(j) < 2: return np.zeros(0)
        root_v = j[1:, 0] - j[:-1, 0]
        v = j[1:, joint] - j[:-1, joint] - root_v
        return np.linalg.norm(v, axis=-1) * 100.0
    for part, joint, _idx in HAND_SPECS:
        axes[2].plot(_local_vel(gt_joints[:seq_len], joint), label=f"GT {part}", linestyle="-")
        axes[2].plot(_local_vel(gen_joints[:seq_len], joint), label=f"gen {part}", linestyle="--")
    axes[2].set_ylabel("hand local vel (cm/frame)")
    axes[2].set_xlabel("frame")
    axes[2].legend(fontsize=7)
    for t in plan_anchor_times:
        for ax in axes:
            ax.axvline(t, c="k", ls=":", alpha=0.3)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _parse_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/training/anchordiff_v18_a1_FULL_DATA.yaml"))
    parser.add_argument("--ckpt", type=Path, default=Path("runs/training/stageB_anchordiff_v18_a1_FULL_DATA/final.pt"))
    parser.add_argument("--selection-json", type=Path, default=Path("analyses/2026-05-13_v18_recon_ladder_truncated_rollout.json"))
    parser.add_argument("--md", type=Path, default=Path("analyses/2026-05-15_v18_failure_atlas.md"))
    parser.add_argument("--visuals-dir", type=Path, default=Path("analyses/visuals/2026-05-15_v18_failure_atlas"))
    parser.add_argument("--max-clips", type=int, default=16)
    parser.add_argument("--num-candidates", type=int, default=256)
    parser.add_argument("--bucket", choices=["train", "val"], default="train")
    parser.add_argument("--balanced-subsets", action="store_true", default=True)
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    selection = _load_selection(args.selection_json, max_clips=int(args.max_clips))
    selected = _build_selected_batches(
        cfg, bucket=args.bucket, balanced_subsets=bool(args.balanced_subsets),
        num_candidates=int(args.num_candidates), selection=selection,
        max_clips=int(args.max_clips), threshold=float(args.threshold),
    )
    if not selected:
        raise SystemExit("No clips selected")
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

    motion_gt = batch["motion"].to(device).float()
    gt_joints = batch["joints"].to(device).float()
    rest_offsets = batch["rest_offsets"].to(device).float()
    object_positions = batch["object_positions"].to(device).float()
    contact_state = batch["contact_state"].to(device).float()
    seq_lens = batch["seq_len"].to(device)
    seq_mask = make_seq_mask(seq_lens, total_t, device)
    seeds = _parse_ints(args.seeds)
    seq_ids = list(batch.get("seq_id", []))
    subsets = list(batch.get("subset", []))

    events = event_records_from_contact(
        contact_state, batch["seq_len"], threshold=float(args.threshold), hands_only=True,
    )
    first_ev_per_clip: dict[int, dict[str, Any]] = {}
    for ev in events:
        first_ev_per_clip.setdefault(int(ev["batch"]), ev)

    # one rollout per seed
    gen_motions: dict[int, Tensor] = {}
    for s in seeds:
        m = _full_rollout(
            model, base_cond, seq_length=total_t,
            seed=int(s), cfg_scale=float(args.cfg_scale), alpha_hint=1.0,
            sampler="ddpm",
        )
        gen_motions[s] = m.detach()
        print(f"  rolled out seed={s}")

    args.visuals_dir.mkdir(parents=True, exist_ok=True)
    rows_md: list[str] = [
        "# v18 Visual Failure Atlas",
        "",
        f"- Config: `{args.config}`",
        f"- Checkpoint: `{args.ckpt}`",
        f"- Sampler: v18 default DDPM",
        f"- Seeds: {seeds}",
        f"- Clips: {len(seq_ids)}",
        "",
        "## Per-clip strips and distance curves",
        "",
        "| subset | seq_id | seed | event | text short | strip | curves |",
        "|---|---|---:|---|---|---|---|",
    ]

    B = motion_gt.shape[0]
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
        lo = max(0, ev_frame - 12)
        hi = min(seq_len_b - 1, ev_frame + 12)
        frames = sorted(set(int(round(x)) for x in np.linspace(lo, hi, num=5)))

        gt_b = gt_joints[b].detach().cpu().numpy().astype(np.float32)
        obj_b = object_positions[b].detach().cpu().numpy().astype(np.float32)
        cs_b = contact_state[b].detach().cpu().numpy().astype(np.float32)

        # plan_anchor_times for this clip
        plan_anchor_times = []
        plan_anchor_parts: list[list[int]] = []
        a_mask = plan_gt["anchor_mask"][b].detach().cpu().numpy().astype(bool)
        a_time = plan_gt["anchor_time"][b].detach().cpu().numpy().astype(int)
        a_part = plan_gt["anchor_part"][b].detach().cpu().numpy().astype(np.float32)
        for k in range(a_mask.shape[0]):
            if bool(a_mask[k]):
                plan_anchor_times.append(int(a_time[k]))
                plan_anchor_parts.append([int(p) for p in np.where(a_part[k] > 0)[0]])

        for s in seeds:
            gen_motion_s = gen_motions[s]
            gen_joints_s = _fk_from_motion_135(gen_motion_s[b:b+1], rest_offsets[b:b+1]).squeeze(0).detach().cpu().numpy().astype(np.float32)

            strip_path = args.visuals_dir / f"{sub}_{sid}_seed{s}_strip.png"
            curves_path = args.visuals_dir / f"{sub}_{sid}_seed{s}_curves.png"

            sources: dict[str, np.ndarray] = {"GT": gt_b, f"v18_seed{s}": gen_joints_s}
            title = (
                f"{sub}/{sid}  seed={s}  event={ev_kind} {ev_part}@{ev_frame}  T={seq_len_b}"
            )
            _strip_plot(
                title=title, frames=frames, sources=sources,
                object_positions=obj_b, out_path=strip_path, event_frame=ev_frame,
            )

            _distance_curves_plot(
                gt_joints=gt_b, gen_joints=gen_joints_s, contact_state=cs_b,
                obj_positions=obj_b, seq_len=seq_len_b,
                plan_anchor_times=plan_anchor_times, plan_anchor_parts_active=plan_anchor_parts,
                out_path=curves_path,
                title=f"{sub}/{sid} seed={s} | {text[:100]}",
            )

            text_short = (text[:60] + "…") if len(text) > 60 else text
            rows_md.append(
                f"| {sub} | `{sid}` | {s} | {ev_kind} {ev_part}@{ev_frame} | {text_short} | "
                f"`{strip_path.as_posix()}` | `{curves_path.as_posix()}` |"
            )

    rows_md.extend([
        "",
        "## Manual review categories (per spec)",
        "",
        "For each clip, tag with one or more of:",
        "",
        "- under-motion / frozen",
        "- over-motion / jitter",
        "- wrong contact direction",
        "- root / object misalignment",
        "- pseudo-label / event wrong (cross-reference Diag 2 / Diag 3 reports)",
        "- object geometry suspicious (cross-reference Diag 2 object-transform table — esp. imhd bat clips with high rotation jumps)",
        "- metric artifact (cross-reference Diag 3 denominator-instability flag)",
        "- visually acceptable despite metric failure",
        "",
        "Cross-references:",
        "- `analyses/2026-05-15_pseudolabel_plan_object_geometry_audit.md` for plan-target / event / object-transform per-clip flags",
        "- `analyses/2026-05-15_transition_metric_reliability_audit.md` for per-clip event validity counts",
        "",
    ])
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text("\n".join(rows_md) + "\n", encoding="utf-8")
    print(f"Wrote {args.md}")


if __name__ == "__main__":
    main()
