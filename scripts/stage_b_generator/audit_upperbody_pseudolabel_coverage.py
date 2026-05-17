"""Upper-body pseudo-label / coverage audit for Stage B v18.

This no-training audit checks whether lying/reclining/leaning clips contain
GT upper-body geometry that the current 5-part plan schema cannot explicitly
condition: hands, feet, and pelvis are tracked, but shoulders/head/chest are
not plan parts.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from diagnostic_common import (
    clip_metadata,
    extract_plan,
    format_md_table,
    load_checkpoint,
    make_seq_mask,
    merge_single_batches,
    safe_div,
    stats_list,
    write_json,
)
from dynamics_diagnostic import (
    _build_cond,
    _build_dataset,
    _build_model,
    _fk_from_motion_135,
)
from piano.data.dataset import collate_hoi
from piano.utils.clip_utils import load_clip_text_encoder


KEYWORDS = (
    "lie", "lying", "lay", "laying", "recline", "reclining", "lean", "leaning",
    "rest", "sofa", "bed", "chair", "sit", "sitting",
)
UPPER_JOINTS = [3, 6, 9, 12, 13, 14, 15, 16, 17]
LOWER_TRACKED_JOINTS = [10, 11]


def _keyword_hit(text: str) -> bool:
    low = text.lower()
    return any(k in low for k in KEYWORDS)


def _select_keyword_and_controls(cfg, *, bucket: str, max_keyword: int, max_controls: int) -> list[int]:
    dataset = _build_dataset(cfg, bucket)
    if not hasattr(dataset, "datasets"):
        return list(range(min(max_keyword + max_controls, len(dataset))))
    selected_keywords: list[int] = []
    selected_controls: list[int] = []
    offset = 0
    subdatasets = list(dataset.datasets)
    per_kw = max(1, int(np.ceil(max_keyword / max(1, len(subdatasets)))))
    per_ctrl = max(1, int(np.ceil(max_controls / max(1, len(subdatasets)))))
    for ds in subdatasets:
        kw_count = 0
        ctrl_count = 0
        for local_idx, meta in enumerate(getattr(ds, "metadata", [])):
            text = str(meta.get("text", ""))
            if _keyword_hit(text) and kw_count < per_kw:
                selected_keywords.append(offset + local_idx)
                kw_count += 1
            elif (not _keyword_hit(text)) and ctrl_count < per_ctrl:
                selected_controls.append(offset + local_idx)
                ctrl_count += 1
            if kw_count >= per_kw and ctrl_count >= per_ctrl:
                break
        offset += len(ds)
    return (selected_keywords[:max_keyword] + selected_controls[:max_controls])


def _masked_np(values: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    flat = values[mask.bool()].detach().cpu().double().numpy()
    return flat[np.isfinite(flat)]


def _upper_metrics(joints: torch.Tensor, object_positions: torch.Tensor, seq_mask: torch.Tensor) -> dict[str, Any]:
    up = joints.new_tensor([0.0, 1.0, 0.0])
    pelvis = joints[:, :, 0]
    shoulder_center = 0.5 * (joints[:, :, 16] + joints[:, :, 17])
    head = joints[:, :, 15]
    chest = joints[:, :, 12]
    torso_vec = shoulder_center - pelvis
    head_vec = head - pelvis
    torso_norm = torch.linalg.vector_norm(torso_vec, dim=-1).clamp_min(1e-8)
    head_norm = torch.linalg.vector_norm(head_vec, dim=-1).clamp_min(1e-8)
    torso_cos = (torso_vec * up).sum(-1) / torso_norm
    head_cos = (head_vec * up).sum(-1) / head_norm
    torso_angle = torch.rad2deg(torch.acos(torso_cos.clamp(-1.0, 1.0)))
    pelvis_head_angle = torch.rad2deg(torch.acos(head_cos.clamp(-1.0, 1.0)))

    if joints.shape[1] >= 2:
        vel = joints[:, 1:] - joints[:, :-1]
        root_vel = joints[:, 1:, 0:1] - joints[:, :-1, 0:1]
        local_vel = vel - root_vel
        vmask = seq_mask[:, 1:] & seq_mask[:, :-1]
        upper_vel = torch.linalg.vector_norm(local_vel[:, :, UPPER_JOINTS], dim=-1) * 100.0
        lower_vel = torch.linalg.vector_norm(local_vel[:, :, LOWER_TRACKED_JOINTS], dim=-1) * 100.0
        upper_vel_vals = _masked_np(upper_vel, vmask.unsqueeze(-1).expand(-1, -1, len(UPPER_JOINTS)))
        lower_vel_vals = _masked_np(lower_vel, vmask.unsqueeze(-1).expand(-1, -1, len(LOWER_TRACKED_JOINTS)))
    else:
        upper_vel_vals = np.asarray([], dtype=np.float64)
        lower_vel_vals = np.asarray([], dtype=np.float64)

    prox = {
        "head_to_object_com_cm": torch.linalg.vector_norm(head - object_positions, dim=-1) * 100.0,
        "shoulder_to_object_com_cm": torch.linalg.vector_norm(shoulder_center - object_positions, dim=-1) * 100.0,
        "chest_to_object_com_cm": torch.linalg.vector_norm(chest - object_positions, dim=-1) * 100.0,
    }

    per_clip: list[dict[str, float]] = []
    for b in range(joints.shape[0]):
        valid = int(seq_mask[b].sum().item())
        if valid <= 1:
            continue
        clip_mask = seq_mask[b, :valid]
        row = {
            "torso_recline_angle_deg_mean": float(torso_angle[b, :valid][clip_mask].mean().item()),
            "torso_recline_angle_deg_p75": float(torch.quantile(torso_angle[b, :valid][clip_mask], 0.75).item()),
            "pelvis_to_head_angle_deg_mean": float(pelvis_head_angle[b, :valid][clip_mask].mean().item()),
            "head_height_range_cm": float((head[b, :valid, 1].max() - head[b, :valid, 1].min()).item() * 100.0),
            "shoulder_height_range_cm": float((shoulder_center[b, :valid, 1].max() - shoulder_center[b, :valid, 1].min()).item() * 100.0),
            "min_head_object_com_cm": float(prox["head_to_object_com_cm"][b, :valid][clip_mask].min().item()),
            "min_shoulder_object_com_cm": float(prox["shoulder_to_object_com_cm"][b, :valid][clip_mask].min().item()),
            "min_chest_object_com_cm": float(prox["chest_to_object_com_cm"][b, :valid][clip_mask].min().item()),
        }
        if valid >= 2:
            vm = vmask[b, : valid - 1]
            uv = upper_vel[b, : valid - 1][vm]
            lv = lower_vel[b, : valid - 1][vm]
            row["upper_body_velocity_cm_per_frame"] = float(uv.mean().item()) if uv.numel() else 0.0
            row["lower_tracked_velocity_cm_per_frame"] = float(lv.mean().item()) if lv.numel() else 0.0
            row["upper_lower_velocity_ratio"] = safe_div(
                row["upper_body_velocity_cm_per_frame"],
                row["lower_tracked_velocity_cm_per_frame"],
            )
        per_clip.append(row)

    return {
        "per_clip": per_clip,
        "aggregate": {
            "torso_recline_angle_deg": stats_list(_masked_np(torso_angle, seq_mask)),
            "pelvis_to_head_angle_deg": stats_list(_masked_np(pelvis_head_angle, seq_mask)),
            "head_to_object_com_cm": stats_list(_masked_np(prox["head_to_object_com_cm"], seq_mask)),
            "shoulder_to_object_com_cm": stats_list(_masked_np(prox["shoulder_to_object_com_cm"], seq_mask)),
            "chest_to_object_com_cm": stats_list(_masked_np(prox["chest_to_object_com_cm"], seq_mask)),
            "upper_body_velocity_cm_per_frame": stats_list(upper_vel_vals),
            "lower_tracked_velocity_cm_per_frame": stats_list(lower_vel_vals),
            "upper_lower_velocity_ratio": safe_div(
                float(upper_vel_vals.mean()) if upper_vel_vals.size else 0.0,
                float(lower_vel_vals.mean()) if lower_vel_vals.size else 0.0,
            ),
        },
    }


def _plan_coverage(plan: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    part_names = ["L_hand", "R_hand", "L_foot", "R_foot", "pelvis"]
    rows: list[dict[str, Any]] = []
    mask = plan["anchor_mask"].bool().detach().cpu()
    parts = plan["anchor_part"].detach().cpu()
    times = plan["anchor_time"].detach().cpu()
    types = plan["anchor_type"].detach().cpu()
    for b in range(mask.shape[0]):
        active = mask[b]
        counts = {name: int(((parts[b, :, i] > 0.0) & active).sum().item()) for i, name in enumerate(part_names)}
        rows.append({
            "n_anchors": int(active.sum().item()),
            "anchor_part_counts": counts,
            "anchor_times": [int(v) for v in times[b, active].tolist()],
            "anchor_types": [int(v) for v in types[b, active].tolist()],
            "has_upper_body_plan_channel": False,
        })
    return rows


def _write_report(payload: dict[str, Any], path: Path) -> None:
    rows = [["group", "clips", "GT torso deg", "sample torso deg", "torso delta", "upper vel xGT", "head prox delta cm", "verdict"]]
    for group, item in payload["group_summary"].items():
        rows.append([
            group,
            item["clips"],
            f"{item['gt_torso_deg']:.2f}",
            f"{item['sample_torso_deg']:.2f}",
            f"{item['sample_minus_gt_torso_deg']:.2f}",
            f"{item['upper_velocity_over_gt']:.3f}",
            f"{item['sample_minus_gt_min_head_object_cm']:.2f}",
            item["verdict"],
        ])
    clip_rows = [["subset", "seq_id", "keyword", "GT torso", "sample torso", "upper xGT", "anchors", "text"]]
    for row in payload["clips"]:
        clip_rows.append([
            row["subset"],
            row["seq_id"],
            row["keyword_group"],
            f"{row['gt']['torso_recline_angle_deg_mean']:.1f}",
            f"{row['sample']['torso_recline_angle_deg_mean']:.1f}",
            f"{row['upper_velocity_over_gt']:.2f}",
            row["plan"]["n_anchors"],
            row["text"][:70],
        ])
    lines = [
        "# Upper-Body Pseudo-Label / Coverage Audit",
        "",
        f"- Config: `{payload['config']}`",
        f"- Checkpoint: `{payload['ckpt']}`",
        f"- Seed: `{payload['seed']}`",
        f"- Clips: {len(payload['clips'])}",
        "",
        "## Verdict",
        "",
        payload["verdict"],
        "",
        "## Group Summary",
        "",
        format_md_table(rows),
        "",
        "## Clip Details",
        "",
        format_md_table(clip_rows),
        "",
        "## Contract Interpretation",
        "",
        "The 5 current pseudo-label / plan parts are L_hand, R_hand, L_foot, R_foot, and pelvis. They are conditioning/plan tracked parts, not output joints. The model still predicts full-body SMPL motion_dim=135. This audit therefore tests a conditioning coverage gap, not an output-dimensionality gap.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/training/anchordiff_v18_a1_FULL_DATA.yaml"))
    parser.add_argument("--ckpt", type=Path, default=Path("runs/training/stageB_anchordiff_v18_a1_FULL_DATA/final.pt"))
    parser.add_argument("--output", type=Path, default=Path("analyses/2026-05-14_upperbody_pseudolabel_audit.json"))
    parser.add_argument("--md", type=Path, default=Path("analyses/2026-05-14_upperbody_pseudolabel_audit.md"))
    parser.add_argument("--bucket", choices=["train", "val"], default="train")
    parser.add_argument("--max-keyword-clips", type=int, default=8)
    parser.add_argument("--max-control-clips", type=int, default=4)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    indices = _select_keyword_and_controls(
        cfg,
        bucket=args.bucket,
        max_keyword=int(args.max_keyword_clips),
        max_controls=int(args.max_control_clips),
    )
    if not indices:
        raise RuntimeError("No clips selected for upper-body audit")
    dataset = _build_dataset(cfg, args.bucket)
    loader = DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False, collate_fn=collate_hoi, num_workers=0)
    batches = [batch for batch in loader]
    batch = merge_single_batches(batches)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    model, object_encoder, z_dims = _build_model(cfg, device)
    load_checkpoint(model, object_encoder, args.ckpt)
    clip_model = load_clip_text_encoder(
        device=device,
        model_name=str(cfg.model.text_encoder.clip_version),
        download_root=str(cfg.model.text_encoder.get("download_root", "cache/clip")),
    )
    cond, total_t = _build_cond(batch, model, object_encoder, clip_model, z_dims, cfg, device)
    plan = extract_plan(batch, device)
    cond = {**cond, "interaction_plan": plan}
    with torch.no_grad():
        sample = model.sample(cond=cond, seq_length=total_t, cfg_scale=float(args.cfg_scale))
    rest_offsets = batch["rest_offsets"].to(device).float()
    sample_joints = _fk_from_motion_135(sample, rest_offsets)
    gt_joints = batch["joints"].to(device).float()
    object_positions = batch["object_positions"].to(device).float()
    seq_mask = make_seq_mask(batch["seq_len"], total_t, device)

    gt_metrics = _upper_metrics(gt_joints, object_positions, seq_mask)
    sample_metrics = _upper_metrics(sample_joints, object_positions, seq_mask)
    coverage = _plan_coverage(plan)
    meta = clip_metadata(batch)
    clip_rows: list[dict[str, Any]] = []
    for i, row in enumerate(meta):
        gt = gt_metrics["per_clip"][i]
        smp = sample_metrics["per_clip"][i]
        keyword_group = "keyword" if _keyword_hit(row["text"]) else "control"
        clip_rows.append({
            **row,
            "keyword_group": keyword_group,
            "gt": gt,
            "sample": smp,
            "upper_velocity_over_gt": safe_div(
                smp.get("upper_body_velocity_cm_per_frame", 0.0),
                gt.get("upper_body_velocity_cm_per_frame", 0.0),
            ),
            "plan": coverage[i],
            "sample_minus_gt_torso_deg": smp["torso_recline_angle_deg_mean"] - gt["torso_recline_angle_deg_mean"],
            "sample_minus_gt_min_head_object_cm": smp["min_head_object_com_cm"] - gt["min_head_object_com_cm"],
        })

    group_summary: dict[str, Any] = {}
    for group in ("keyword", "control"):
        rows = [r for r in clip_rows if r["keyword_group"] == group]
        if not rows:
            continue
        gt_torso = float(np.mean([r["gt"]["torso_recline_angle_deg_mean"] for r in rows]))
        smp_torso = float(np.mean([r["sample"]["torso_recline_angle_deg_mean"] for r in rows]))
        ux = float(np.mean([r["upper_velocity_over_gt"] for r in rows]))
        head_delta = float(np.mean([r["sample_minus_gt_min_head_object_cm"] for r in rows]))
        supports = (gt_torso >= 35.0 and smp_torso <= gt_torso - 8.0) or (ux < 0.75 and gt_torso >= 30.0)
        group_summary[group] = {
            "clips": len(rows),
            "gt_torso_deg": gt_torso,
            "sample_torso_deg": smp_torso,
            "sample_minus_gt_torso_deg": smp_torso - gt_torso,
            "upper_velocity_over_gt": ux,
            "sample_minus_gt_min_head_object_cm": head_delta,
            "verdict": "supports_coverage_gap" if supports else "mixed_or_not_supported",
        }

    keyword_verdict = group_summary.get("keyword", {}).get("verdict", "mixed_or_not_supported")
    if keyword_verdict == "supports_coverage_gap":
        verdict = (
            "Supports the upper-body coverage-gap hypothesis: keyword clips show GT upper-body recline/dynamics that v18 samples under-realize while the plan has no shoulders/head/chest channels. v25 upper-body pseudo-label extension is justified only after reviewing representative clips."
        )
    else:
        verdict = (
            "Mixed / not supported: this run does not show a clean keyword-group upper-body gap strong enough to justify v25 training by itself."
        )

    payload = {
        "config": str(args.config),
        "ckpt": str(args.ckpt),
        "seed": int(args.seed),
        "keywords": list(KEYWORDS),
        "indices": indices,
        "gt_aggregate": gt_metrics["aggregate"],
        "sample_aggregate": sample_metrics["aggregate"],
        "clips": clip_rows,
        "group_summary": group_summary,
        "verdict": verdict,
    }
    write_json(args.output, payload)
    _write_report(payload, args.md)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.md}")


if __name__ == "__main__":
    main()

