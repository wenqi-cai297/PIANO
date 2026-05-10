"""GT-motion upper bound for ``plan_anchor_contact_realization_cm``.

Per analyses/claude_code_v10_plan_tokens_next_steps.md §4: before treating
the v10 diagnostic's 33.57 cm anchor realisation as a model-training
failure, we need to compute the same metric on **GT motion** rather than
model prediction. The metric aligns a SMPL joint center (e.g. wrist for
"hand contact") to a pseudo-label contact target. If pseudo-labels
encode a surface point on the object — not the joint center itself —
the metric has a built-in floor.

Decision rule (§4):
    if GT_anchor_realization > 20 cm:
        metric / loss target is partially miscalibrated; do not blindly
        raise plan_anchor_weight.
    else:
        the model should be able to reach < 20 cm.

Usage::

    python scripts/stage_b_generator/gt_anchor_realization_upper_bound.py \\
        --config configs/training/anchordiff_v10_plan_tokens_gt_overfit.yaml \\
        --bucket train --max-clips 1 \\
        --output analyses/2026-05-10_gt_anchor_realization_upper_bound.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, DataLoader, Subset

from piano.data.dataset import (
    HOIDataset, build_subject_split, collate_hoi, extract_subject_id,
)
from piano.utils.io_utils import load_json


def _build_dataset(cfg, bucket: str) -> ConcatDataset:
    keys: set[tuple[str, str]] = set()
    for entry in cfg.data.datasets:
        meta = load_json(Path(entry.root) / "metadata_clean.json")
        for m in meta:
            sid = extract_subject_id(Path(entry.root).name, m.get("seq_id", ""))
            if sid is not None:
                keys.add((Path(entry.root).name, sid))
    splits = build_subject_split(
        sorted(keys),
        train_pct=int(cfg.data.subject_split.train_pct),
        val_pct=int(cfg.data.subject_split.val_pct),
        seed=int(cfg.data.subject_split.seed),
    )
    subj_filter = splits[bucket]
    pseudo_label_subdir = cfg.data.get("pseudo_label_subdir", None)
    datasets = []
    for entry in cfg.data.datasets:
        sub_dir = (
            str(Path(entry.root) / pseudo_label_subdir)
            if pseudo_label_subdir is not None else None
        )
        ds = HOIDataset(
            root=entry.root,
            pseudo_label_dir=sub_dir,
            max_seq_length=int(cfg.data.max_seq_length),
            subject_id_filter=subj_filter,
            subsample_n_per_object=int(cfg.data.subsample_n_per_object),
            subsample_seed=int(cfg.data.subsample_seed),
            support_collapse_hand_support=True,
            surface_obj_pose=True,
            force_world_frame=bool(cfg.data.get("force_world_frame", False)),
            motion_representation=str(cfg.data.motion_representation),
        )
        datasets.append(ds)
    return ConcatDataset(datasets)


def _gt_anchor_realization_cm(
    joints_gt: torch.Tensor,             # (B, T, 22, 3)
    plan: dict[str, torch.Tensor],
    part_to_joint: torch.Tensor,         # (P,) long
) -> tuple[float, dict]:
    """Mirror ``_compute_metrics`` plan_anchor_contact_realization_cm.

    Returns (mean_cm, per_part_breakdown).
    """
    anchor_time = plan["anchor_time"].long()
    anchor_mask = plan["anchor_mask"].bool()
    anchor_part = plan["anchor_part"].float()
    anchor_target_world = plan["anchor_target_world"].float()
    B, K, P = anchor_part.shape
    T = joints_gt.shape[1]

    if not anchor_mask.any():
        return 0.0, {}

    t_idx = (
        anchor_time.clamp(0, T - 1)
        .view(B, K, 1, 1)
        .expand(B, K, 22, 3)
    )
    fk_at_anchor = torch.gather(joints_gt, 1, t_idx)                   # (B, K, 22, 3)
    joint_at_part = fk_at_anchor[:, :, part_to_joint, :]               # (B, K, P, 3)
    err = (joint_at_part - anchor_target_world).pow(2).sum(-1).sqrt()  # (B, K, P) in m
    act = anchor_mask.unsqueeze(-1).float() * anchor_part              # (B, K, P)
    denom = act.sum().clamp_min(1.0)
    mean_cm = float((err * act).sum() / denom) * 100.0

    # Per-part breakdown
    per_part: dict[str, dict] = {}
    part_names = ("L_hand", "R_hand", "L_foot", "R_foot", "pelvis")
    for p in range(P):
        act_p = act[..., p]
        if act_p.sum() < 0.5:
            per_part[part_names[p]] = {"mean_cm": 0.0, "n_active": 0}
            continue
        err_p = err[..., p]
        m = float((err_p * act_p).sum() / act_p.sum()) * 100.0
        per_part[part_names[p]] = {
            "mean_cm": m, "n_active": int(act_p.sum().item()),
        }
    return mean_cm, per_part


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bucket", default="train", choices=["train", "val"])
    parser.add_argument("--max-clips", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    dataset = _build_dataset(cfg, args.bucket)
    overfit_n = int(cfg.data.get("overfit_n_clips", 0))
    if overfit_n > 0:
        dataset = Subset(dataset, list(range(min(overfit_n, len(dataset)))))
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=collate_hoi, num_workers=0,
    )

    part_to_joint = torch.tensor([20, 21, 10, 11, 0], dtype=torch.long)
    part_names = ("L_hand", "R_hand", "L_foot", "R_foot", "pelvis")

    rows: list[dict] = []
    for i, batch in enumerate(loader):
        if i >= args.max_clips:
            break
        joints_gt = batch["joints"].float()
        plan = {
            "anchor_time": batch["plan_anchor_time"],
            "anchor_part": batch["plan_anchor_part"],
            "anchor_target_world": batch["plan_anchor_target_world"],
            "anchor_mask": batch["plan_anchor_mask"],
        }
        mean_cm, per_part = _gt_anchor_realization_cm(joints_gt, plan, part_to_joint)
        rows.append({
            "subset": batch["subset"][0],
            "seq_id": batch["seq_id"][0],
            "n_anchors_valid": int(batch["plan_anchor_mask"].sum().item()),
            "mean_cm": mean_cm,
            "per_part": per_part,
        })

    # Aggregate over clips
    valid_means = [r["mean_cm"] for r in rows if r["n_anchors_valid"] > 0]
    agg_mean = sum(valid_means) / max(len(valid_means), 1)
    threshold_cm = 20.0
    verdict = (
        "METRIC_TARGET_MISMATCH (>20 cm)" if agg_mean > threshold_cm
        else "METRIC_OK (<= 20 cm)"
    )

    # Markdown report
    md: list[str] = []
    md.append("# GT motion plan_anchor_contact_realization_cm — upper bound\n")
    md.append("**Date:** 2026-05-10  ")
    md.append(f"**Config:** `{args.config}`  ")
    md.append(f"**Bucket:** `{args.bucket}`  ")
    md.append(f"**Clips:** {len(rows)}\n")
    md.append("Per spec [analyses/claude_code_v10_plan_tokens_next_steps.md](analyses/claude_code_v10_plan_tokens_next_steps.md) §4: ")
    md.append("compute the same metric on GT motion to check whether 33 cm in the v10 diagnostic ")
    md.append("is a model-training failure or a metric-target mismatch.\n")
    md.append("## Aggregate\n")
    md.append(f"- Mean across {len(valid_means)} clip(s): **{agg_mean:.2f} cm**")
    md.append(f"- Threshold (spec §4): {threshold_cm:.0f} cm")
    md.append(f"- Verdict: **{verdict}**\n")
    md.append("## Per-clip\n")
    md.append("| subset | seq_id | n_anchors | mean_cm | L_hand | R_hand | L_foot | R_foot | pelvis |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        cells = []
        for p in part_names:
            pp = r["per_part"].get(p, {"mean_cm": 0.0, "n_active": 0})
            if pp["n_active"] == 0:
                cells.append("—")
            else:
                cells.append(f"{pp['mean_cm']:.1f} ({pp['n_active']})")
        md.append(
            f"| {r['subset']} | {r['seq_id']} | {r['n_anchors_valid']} | "
            f"{r['mean_cm']:.2f} | " + " | ".join(cells) + " |"
        )
    md.append("\n## Decision rule\n")
    if agg_mean > threshold_cm:
        md.append(
            "GT motion does NOT achieve < 20 cm anchor realisation. The metric has a "
            "built-in floor — the pseudo-label contact target encodes an object "
            "surface point, not the SMPL joint centre. The 33 cm in the v10 "
            "diagnostic is partially structural; do not blindly raise "
            "`plan_anchor_weight`."
        )
    else:
        md.append(
            "GT motion achieves < 20 cm anchor realisation. The metric is well-"
            "calibrated; the model's 33 cm is a training-quality gap and "
            "stronger `plan_anchor_weight` (or longer training) is justified."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote: {args.output}")
    print(f"Aggregate mean: {agg_mean:.2f} cm  → {verdict}")
    for r in rows:
        print(f"  {r['subset']}/{r['seq_id']}: {r['mean_cm']:.2f} cm "
              f"({r['n_anchors_valid']} anchors)")


if __name__ == "__main__":
    main()
