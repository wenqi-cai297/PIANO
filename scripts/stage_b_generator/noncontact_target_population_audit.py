"""Non-contact target population audit (round 7, Task 5).

Round 6 Diag A found that 100% of non-contact frames carry populated
`contact_target_xyz` because the official-marker prior writes per-frame
nearest-object data even when contact is not active. The question is
whether (a) the non-contact target is still anchored to the object
surface (so dense-route consumption is benign), or (b) it jumps around
unpredictably during non-contact (so dense route receives noise).

This audit answers:
  - % of non-contact frames with populated target
  - distance of non-contact target_world to object surface
  - target temporal jump (cm/frame) in non-contact vs contact frames
  - whether dense z_int receives the non-contact value (yes — z_int
    is fed `contact_target_xyz` verbatim slot [5:20])
  - whether object_world_traj receives the non-contact value (yes —
    `object_world_traj[9:24]` is the lifted target for all 5 parts)
  - whether plan compiler masks it via contact-weighted smoothing
    (yes — target_smooth = num/den with den = contact_smooth; target
    contribution is gated to ~0 in non-contact frames)
  - contact-vs-non-contact target stability comparison

Outputs:
  analyses/2026-05-17_noncontact_target_population_audit.json
  analyses/2026-05-17_noncontact_target_population_audit.md

This is audit-only. Do NOT change pseudo-label data. Do NOT mask
non-contact frames in the dense route. The output informs whether to
build a separate no-training masking diagnostic next round.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from diagnostic_common import format_md_table, merge_single_batches, stats_list
from recon_ladder_truncated_rollout_diagnostic import (
    _build_selected_batches,
    _load_selection,
)


HAND_SPECS = (("L_hand", 20, 0), ("R_hand", 21, 1))
PART_NAMES = ("left_hand", "right_hand", "left_foot", "right_foot", "pelvis")


def _axis_angle_to_rot(aa: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(aa, axis=-1, keepdims=True).clip(min=1e-12)
    k = aa / theta
    K = np.zeros(aa.shape[:-1] + (3, 3), dtype=aa.dtype)
    K[..., 0, 1] = -k[..., 2]; K[..., 0, 2] = k[..., 1]
    K[..., 1, 0] = k[..., 2];  K[..., 1, 2] = -k[..., 0]
    K[..., 2, 0] = -k[..., 1]; K[..., 2, 1] = k[..., 0]
    eye = np.broadcast_to(np.eye(3), K.shape)
    s = np.sin(theta)[..., None]
    c = np.cos(theta)[..., None]
    return eye + s * K + (1 - c) * (K @ K)


def _audit_clip(
    batch: dict[str, Any], b: int,
    *, threshold: float, surface_samples: int,
) -> dict[str, Any]:
    seq_len = int(batch["seq_len"][b].item())
    subset = str(batch["subset"][b])
    seq_id = str(batch["seq_id"][b])
    contact_state = batch["contact_state"][b].detach().cpu().numpy().astype(np.float32)[:seq_len]
    target_local = batch["contact_target_xyz"][b].detach().cpu().numpy().astype(np.float32)[:seq_len]
    obj_positions = batch["object_positions"][b].detach().cpu().numpy().astype(np.float32)[:seq_len]
    obj_rotations = batch["object_rotations"][b].detach().cpu().numpy().astype(np.float32)[:seq_len]
    obj_pc = batch["object_pc"][b].detach().cpu().numpy().astype(np.float32)

    R = _axis_angle_to_rot(obj_rotations)
    if obj_pc.shape[0] > surface_samples:
        idx = np.random.RandomState(0).choice(obj_pc.shape[0], surface_samples, replace=False)
        pc_sub = obj_pc[idx]
    else:
        pc_sub = obj_pc
    obj_pc_world = np.einsum("tij,nj->tni", R, pc_sub) + obj_positions[:, None, :]

    # World-lifted target (T, 5, 3)
    target_world = np.einsum("tij,tpj->tpi", R, target_local) + obj_positions[:, None, :]

    per_part_results: dict[str, dict[str, Any]] = {}
    for part_name, joint, p_idx in HAND_SPECS:
        c_bool = contact_state[:, p_idx] > float(threshold)
        in_contact = c_bool
        non_contact = ~c_bool

        # populated mask = local target has non-zero magnitude
        target_norm_cm = np.linalg.norm(target_local[:, p_idx, :], axis=-1) * 100.0
        populated = target_norm_cm > 0.1  # 1 mm
        populated_in_contact = int((populated & in_contact).sum())
        populated_in_non_contact = int((populated & non_contact).sum())

        # target-to-surface distance per frame (cm)
        d_target_surf = np.linalg.norm(
            obj_pc_world - target_world[:, p_idx, None, :], axis=-1
        ).min(axis=-1) * 100.0

        d_target_surf_in_contact = (
            float(d_target_surf[in_contact].mean()) if in_contact.any() else 0.0
        )
        d_target_surf_in_non_contact = (
            float(d_target_surf[non_contact].mean()) if non_contact.any() else 0.0
        )

        # Frame-to-frame jump magnitudes
        diff_local = np.linalg.norm(np.diff(target_local[:, p_idx, :], axis=0), axis=-1) * 100.0
        diff_world = np.linalg.norm(np.diff(target_world[:, p_idx, :], axis=0), axis=-1) * 100.0
        mask_pair_in = in_contact[:-1] & in_contact[1:]
        mask_pair_nc = non_contact[:-1] & non_contact[1:]
        jump_in_contact_local = (
            float(diff_local[mask_pair_in].mean()) if mask_pair_in.any() else 0.0
        )
        jump_in_non_contact_local = (
            float(diff_local[mask_pair_nc].mean()) if mask_pair_nc.any() else 0.0
        )
        jump_in_contact_world = (
            float(diff_world[mask_pair_in].mean()) if mask_pair_in.any() else 0.0
        )
        jump_in_non_contact_world = (
            float(diff_world[mask_pair_nc].mean()) if mask_pair_nc.any() else 0.0
        )

        # P95 jumps in non-contact
        p95_jump_nc_local = (
            float(np.percentile(diff_local[mask_pair_nc], 95)) if mask_pair_nc.any() else 0.0
        )
        p95_jump_nc_world = (
            float(np.percentile(diff_world[mask_pair_nc], 95)) if mask_pair_nc.any() else 0.0
        )

        # Distribution of target_to_surface in non-contact
        d_nc_stats = (
            stats_list(d_target_surf[non_contact].tolist()) if non_contact.any() else stats_list([])
        )
        d_in_stats = (
            stats_list(d_target_surf[in_contact].tolist()) if in_contact.any() else stats_list([])
        )

        per_part_results[part_name] = {
            "n_frames_total": int(seq_len),
            "n_contact_frames": int(in_contact.sum()),
            "n_non_contact_frames": int(non_contact.sum()),
            "n_populated_in_contact": populated_in_contact,
            "n_populated_in_non_contact": populated_in_non_contact,
            "pct_non_contact_populated": (
                100.0 * populated_in_non_contact / max(1, int(non_contact.sum()))
            ),
            "target_to_surface_in_contact_cm_mean": d_target_surf_in_contact,
            "target_to_surface_in_non_contact_cm_mean": d_target_surf_in_non_contact,
            "target_to_surface_in_non_contact_cm_p95": d_nc_stats["p95"],
            "jump_local_in_contact_cm": jump_in_contact_local,
            "jump_local_in_non_contact_cm": jump_in_non_contact_local,
            "jump_world_in_contact_cm": jump_in_contact_world,
            "jump_world_in_non_contact_cm": jump_in_non_contact_world,
            "jump_local_p95_non_contact_cm": p95_jump_nc_local,
            "jump_world_p95_non_contact_cm": p95_jump_nc_world,
            "ratio_jump_world_non_contact_over_contact": (
                jump_in_non_contact_world / jump_in_contact_world
                if jump_in_contact_world > 1e-3 else 0.0
            ),
            "target_to_surface_in_contact_distribution": d_in_stats,
            "target_to_surface_in_non_contact_distribution": d_nc_stats,
        }

    return {
        "subset": subset, "seq_id": seq_id, "seq_len": seq_len,
        "per_part": per_part_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=Path("configs/training/anchordiff_v18_a1_FULL_DATA.yaml"))
    parser.add_argument("--selection-json", type=Path,
                        default=Path("analyses/2026-05-13_v18_recon_ladder_truncated_rollout.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("analyses/2026-05-17_noncontact_target_population_audit.json"))
    parser.add_argument("--md", type=Path,
                        default=Path("analyses/2026-05-17_noncontact_target_population_audit.md"))
    parser.add_argument("--max-clips", type=int, default=16)
    parser.add_argument("--num-candidates", type=int, default=256)
    parser.add_argument("--bucket", choices=["train", "val"], default="train")
    parser.add_argument("--balanced-subsets", action="store_true", default=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--surface-samples", type=int, default=512)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    selection = _load_selection(args.selection_json, max_clips=int(args.max_clips))
    selected = _build_selected_batches(
        cfg, bucket=args.bucket, balanced_subsets=bool(args.balanced_subsets),
        num_candidates=int(args.num_candidates), selection=selection,
        max_clips=int(args.max_clips), threshold=float(args.threshold),
    )
    if not selected:
        raise SystemExit("No clips selected")
    batch = merge_single_batches([item[1] for item in selected])
    B = int(batch["motion"].shape[0])

    audits = [
        _audit_clip(
            batch, b,
            threshold=float(args.threshold),
            surface_samples=int(args.surface_samples),
        )
        for b in range(B)
    ]

    # Aggregate across clips and hand parts
    all_part_rows = []
    for clip in audits:
        for part_name, st in clip["per_part"].items():
            all_part_rows.append({"clip": clip["seq_id"], "subset": clip["subset"],
                                  "part": part_name, **st})

    def _mean(field: str) -> float:
        vals = [r[field] for r in all_part_rows if isinstance(r.get(field), (int, float))]
        return float(np.mean(vals)) if vals else 0.0

    aggregate = {
        "n_clips": B,
        "pct_non_contact_populated_mean": _mean("pct_non_contact_populated"),
        "target_to_surface_in_contact_cm_mean": _mean("target_to_surface_in_contact_cm_mean"),
        "target_to_surface_in_non_contact_cm_mean": _mean("target_to_surface_in_non_contact_cm_mean"),
        "target_to_surface_in_non_contact_cm_p95_mean": _mean("target_to_surface_in_non_contact_cm_p95"),
        "jump_local_in_contact_cm_mean": _mean("jump_local_in_contact_cm"),
        "jump_local_in_non_contact_cm_mean": _mean("jump_local_in_non_contact_cm"),
        "jump_world_in_contact_cm_mean": _mean("jump_world_in_contact_cm"),
        "jump_world_in_non_contact_cm_mean": _mean("jump_world_in_non_contact_cm"),
        "jump_world_p95_non_contact_cm_mean": _mean("jump_world_p95_non_contact_cm"),
        "ratio_jump_world_non_contact_over_contact_mean": _mean(
            "ratio_jump_world_non_contact_over_contact"
        ),
    }

    # Verdict heuristics
    nc_surface = aggregate["target_to_surface_in_non_contact_cm_mean"]
    c_surface = aggregate["target_to_surface_in_contact_cm_mean"]
    nc_jump = aggregate["jump_world_in_non_contact_cm_mean"]
    c_jump = aggregate["jump_world_in_contact_cm_mean"]
    if nc_surface < 5.0 and nc_jump < c_jump * 1.5:
        verdict = (
            f"**Non-contact target is benign**: surface offset {nc_surface:.2f} cm "
            f"(vs in-contact {c_surface:.2f}); world jump {nc_jump:.2f} cm "
            f"(vs in-contact {c_jump:.2f}). Target stays on / near object surface even "
            "during non-contact. The dense route consuming non-contact target should not "
            "produce supervision noise. (Case F.)"
        )
    elif nc_surface > 10.0 or nc_jump > c_jump * 3:
        verdict = (
            f"**Non-contact target IS suspicious**: surface offset {nc_surface:.2f} cm, "
            f"world jump {nc_jump:.2f} cm vs in-contact {c_jump:.2f}. Recommend a "
            "dense-route masking diagnostic next round. (Case G.)"
        )
    else:
        verdict = (
            f"Non-contact target is borderline: surface offset {nc_surface:.2f} cm, "
            f"world jump {nc_jump:.2f} cm vs in-contact {c_jump:.2f}. Mark inconclusive; "
            "investigate further only if metric-v2 replays implicate the dense route."
        )

    aggregate["verdict"] = verdict

    payload = {
        "config": str(args.config),
        "selection_json": str(args.selection_json),
        "n_clips": B,
        "aggregate": aggregate,
        "clips": audits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")

    # Markdown
    lines = [
        "# Non-contact Target Population Audit (Round 7, Task 5)",
        "",
        f"- Config: `{args.config}`",
        f"- Clips: {B}",
        "",
        "## Why this audit",
        "",
        "Round 6 Diag A found 100% of non-contact frames have populated "
        "`contact_target_xyz` because the official-marker prior writes "
        "per-frame nearest-object data for every frame. The dense "
        "z_int route and `object_world_traj` consume this verbatim. "
        "If the non-contact target is benign (on the object surface, "
        "stable across frames), the dense-route gating via contact "
        "weights is unnecessary; if it is noisy, downstream "
        "conditioning may be polluted.",
        "",
        "## Where these signals are consumed",
        "",
        "- **Dense z_int** (`pack_z_int` in `src/piano/models/motion_anchordiff.py`): slot [5:20] = `contact_target_xyz` flattened across 5 parts. Receives non-contact target verbatim.",
        "- **`object_world_traj`** (`_build_object_traj` in `src/piano/training/train_anchordiff.py`): channels [9:24] = lifted `contact_target_xyz` per part. Receives non-contact target verbatim.",
        "- **Plan compiler**: `smooth_target_local` is contact-weighted (num / clip(den, 1e-6)); off-contact frames contribute ~zero. Plan-anchor route is naturally gated.",
        "",
        "## Aggregate findings",
        "",
        f"- % non-contact frames with populated target: **{aggregate['pct_non_contact_populated_mean']:.1f}%**",
        f"- target → surface mean (in contact): {aggregate['target_to_surface_in_contact_cm_mean']:.2f} cm",
        f"- target → surface mean (non contact): **{aggregate['target_to_surface_in_non_contact_cm_mean']:.2f} cm**",
        f"- target → surface p95 (non contact, mean across clips): {aggregate['target_to_surface_in_non_contact_cm_p95_mean']:.2f} cm",
        f"- frame-to-frame jump in contact (world): {aggregate['jump_world_in_contact_cm_mean']:.2f} cm",
        f"- frame-to-frame jump non contact (world): **{aggregate['jump_world_in_non_contact_cm_mean']:.2f} cm**",
        f"- frame-to-frame jump non contact p95: {aggregate['jump_world_p95_non_contact_cm_mean']:.2f} cm",
        f"- ratio (jump non-contact / jump in-contact): {aggregate['ratio_jump_world_non_contact_over_contact_mean']:.2f}",
        "",
        "## Verdict",
        "",
        verdict,
        "",
        "## Per-clip / per-hand summary",
        "",
    ]
    rows = [[
        "subset", "seq_id", "part", "n_nc", "%pop NC",
        "surf C cm", "surf NC cm", "jump NC cm", "ratio NC/C",
    ]]
    for clip in audits:
        for part_name, st in clip["per_part"].items():
            rows.append([
                clip["subset"], clip["seq_id"], part_name,
                int(st["n_non_contact_frames"]),
                f"{st['pct_non_contact_populated']:.0f}",
                f"{st['target_to_surface_in_contact_cm_mean']:.2f}",
                f"{st['target_to_surface_in_non_contact_cm_mean']:.2f}",
                f"{st['jump_world_in_non_contact_cm']:.2f}",
                f"{st['ratio_jump_world_non_contact_over_contact']:.2f}",
            ])
    lines.append(format_md_table(rows))
    lines.append("")
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.md}")


if __name__ == "__main__":
    main()
