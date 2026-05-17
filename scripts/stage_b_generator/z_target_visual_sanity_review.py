"""Visual sanity review for the α_z_target seed-sensitivity question.

For the 4 chairs clips used in round 3, re-runs the v18 model under
default DDPM at seeds 42/43/44 × α_z_target ∈ {0.0, 1.0} and writes
static side-by-side frame strips (GT vs α=1 vs α=0) at the event window
of each clip.

Purpose: check whether the round-3 metric divergence between seed 42
(favorable to α=0) and seeds 43/44 (unfavorable) corresponds to
visually meaningful contact-direction differences or is a metric
artifact.

Outputs (per clip × per seed):
  analyses/visuals/2026-05-15_z_target_seed_sensitivity/
    {subset}_{seq_id}_seed{s}_strip.png

Plus an index file:
  analyses/visuals/2026-05-15_z_target_seed_sensitivity/index.md

No training, no model state changes. Uses the default v18 DDPM
sampler. α_z_target=0 is an inference-only diagnostic perturbation.
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

from condition_route_causal_sensitivity_diagnostic import (
    _apply_zint_target_scale,
    _full_rollout,
)
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


def _strip_plot(
    *,
    title: str,
    frames: list[int],
    sources: dict[str, np.ndarray],
    object_positions: np.ndarray,
    out_path: Path,
    event_frame: int | None,
    elev: float = 15.0,
    azim: float = -60.0,
) -> None:
    """Render a static (rows=sources, cols=frames) 3-D scatter strip."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows = len(sources)
    n_cols = len(frames)

    # Compute axis limits across all sources / all selected frames
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
                    ax.plot(
                        [j[a, 0], j[b, 0]],
                        [j[a, 2], j[b, 2]],
                        [j[a, 1], j[b, 1]],
                        c="0.4", linewidth=0.6,
                    )
                p = object_positions[f]
                ax.scatter([p[0]], [p[2]], [p[1]], c="#d62728", s=24, alpha=0.9)
            marker = " *" if event_frame is not None and f == int(event_frame) else ""
            ax.set_title(f"{label} f={f}{marker}", fontsize=7)
            ax.set_xlim(center[0] - max_range, center[0] + max_range)
            ax.set_ylim(center[2] - max_range, center[2] + max_range)
            ax.set_zlim(center[1] - max_range, center[1] + max_range)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.view_init(elev=elev, azim=azim)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/training/anchordiff_v18_a1_FULL_DATA.yaml"))
    parser.add_argument("--ckpt", type=Path, default=Path("runs/training/stageB_anchordiff_v18_a1_FULL_DATA/final.pt"))
    parser.add_argument("--selection-json", type=Path, default=Path("analyses/2026-05-13_v18_recon_ladder_truncated_rollout.json"))
    parser.add_argument("--max-clips", type=int, default=4)
    parser.add_argument("--num-candidates", type=int, default=96)
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--alphas", type=str, default="0.0,1.0")
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=Path("analyses/visuals/2026-05-15_z_target_seed_sensitivity"))
    parser.add_argument("--md", type=Path, default=Path("analyses/2026-05-15_z_target_visual_sanity_review.md"))
    parser.add_argument("--bucket", choices=["train", "val"], default="train")
    parser.add_argument("--balanced-subsets", action="store_true", default=True)
    args = parser.parse_args()

    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]
    alphas = [float(a) for a in str(args.alphas).split(",") if a.strip()]

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
    seq_mask = make_seq_mask(batch["seq_len"], total_t, device)
    seq_lens = batch["seq_len"].cpu().tolist()
    seq_ids = list(batch.get("seq_id", []))
    subsets = list(batch.get("subset", []))

    events = event_records_from_contact(
        contact_state, batch["seq_len"], threshold=float(args.threshold), hands_only=True,
    )
    first_ev_per_clip: dict[int, dict[str, Any]] = {}
    for ev in events:
        first_ev_per_clip.setdefault(int(ev["batch"]), ev)

    rollouts: dict[tuple[int, float], Tensor] = {}
    for s in seeds:
        for a in alphas:
            cond_a = _apply_zint_target_scale(base_cond, a) if abs(a - 1.0) > 1e-9 else base_cond
            motion = _full_rollout(
                model, cond_a, seq_length=total_t,
                seed=int(s), cfg_scale=float(args.cfg_scale), alpha_hint=1.0,
                sampler="ddpm",
            )
            rollouts[(int(s), float(a))] = motion.detach()
            print(f"  rolled out seed={s} alpha={a:.2f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_md: list[str] = [
        "# α_z_target Visual Sanity Review (seeds 42/43/44, α ∈ {0.0, 1.0})",
        "",
        f"- Config: `{args.config}`",
        f"- Checkpoint: `{args.ckpt}`",
        f"- Sampler: v18 default DDPM",
        f"- Seeds: {seeds}",
        f"- α_z_target: {alphas}",
        f"- Frames per strip: 5 evenly spaced spanning the event window",
        "",
        "## Caveat",
        "",
        "α_z_target=0 is a diagnostic-only perturbation; it is not a deployable change. "
        "These visuals are intended to test whether the round-3 metric divergence corresponds "
        "to visually meaningful contact-direction differences.",
        "",
        "## Outputs",
        "",
        "| subset | seq_id | seed | event | png |",
        "|---|---|---:|---|---|",
    ]

    B = motion_gt.shape[0]
    for b in range(B):
        sid = str(seq_ids[b]) if b < len(seq_ids) else f"clip{b}"
        sub = str(subsets[b]) if b < len(subsets) else "?"
        seq_len_b = int(seq_lens[b])
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

        # GT joints
        gt_b = gt_joints[b].detach().cpu().numpy().astype(np.float32)
        obj_b = object_positions[b].detach().cpu().numpy().astype(np.float32)

        for s in seeds:
            sources: dict[str, np.ndarray] = {"GT": gt_b}
            for a in alphas:
                motion_sa = rollouts[(int(s), float(a))]
                joints_sa = _fk_from_motion_135(
                    motion_sa[b : b + 1], rest_offsets[b : b + 1],
                )
                sources[f"α={a:.1f}"] = joints_sa.squeeze(0).detach().cpu().numpy().astype(np.float32)

            out_png = args.output_dir / f"{sub}_{sid}_seed{s}_strip.png"
            title = (
                f"{sub}/{sid}  seed={s}  event={ev_kind} {ev_part}@{ev_frame}  "
                f"frames={frames}"
            )
            _strip_plot(
                title=title,
                frames=frames,
                sources=sources,
                object_positions=obj_b,
                out_path=out_png,
                event_frame=ev_frame,
            )
            rows_md.append(
                f"| {sub} | `{sid}` | {s} | {ev_kind} {ev_part}@{ev_frame} | `{out_png.as_posix()}` |"
            )

    rows_md.extend([
        "",
        "## Review questions (per spec)",
        "",
        "1. Does seed 42 α=0 visibly improve onset/release, or only metric?",
        "2. Does seed 43 α=0 visibly worsen release as metric says?",
        "3. Does seed 44 α=0 visibly worsen onset?",
        "4. Are far/anchor regressions visually meaningful?",
        "5. Are there object geometry / contact artifacts not captured by metrics?",
        "6. Are event windows correct for clips with onset at frame 2 (Sub0001_Obj116_Seg0_600)?",
        "",
        "Each PNG shows 5 frames (rows: GT, α=1.0, α=0.0) around the event frame "
        "(marked with '*'). Red dot = object position at that frame. Skeleton = SMPL-22 joints.",
        "",
    ])
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text("\n".join(rows_md) + "\n", encoding="utf-8")
    print(f"Wrote {args.md}")


if __name__ == "__main__":
    main()
