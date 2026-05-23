"""Round-25 D2: multi-sample diversity diagnostic.

For each clip in the D1 multimodal eval subset, sample the v26 denoiser
N times with different random seeds (same condition each time) and
measure how diverse the limb-endpoint trajectories are.

Discriminates between:
    H1 TRUE mode collapse  — APD ≈ 0 across N samples (all collapse to
                             same mean limb pose)
    not-H1                 — APD large; multi-mode IS being sampled but
                             individual samples are still wrong on
                             anchor_pose_error_cm.

Design source:
    analyses/2026-05-23_round25_diagnostic_bundle_design.md §D2.

Usage:
    conda run -n piano python scripts/stage_b_generator/round25_d2_diversity_diagnostic.py \
        --config configs/training/anchordiff_v26_FULL_DATA_local.yaml \
        --ckpt   runs/training/stageB_anchordiff_v25_round23_noplan_clean_alibi_FULL_DATA/final.pt \
        --selection-json analyses/round25_multimodal_eval_subset.json \
        --n-samples 8 \
        --cfg-scale 1.0 \
        --output analyses/round25_d2_diversity_stats.json

Output:
    {output}.json + sibling .md report.

Key metrics per clip:
    - APD_hand_endpoint_cm  : average pairwise distance on wrist
                              positions, averaged over time, averaged
                              over the C(N, 2) pairs.
    - APD_foot_endpoint_cm  : same for ankle positions.
    - APD_22joint_cm        : same on the full 22-joint pose (mean
                              over joints).
    - best_of_N_anchor_pose_error_cm : min over N samples of the
                              anchor_pose_error metric defined in
                              the Round-24 redefine.
    - mean_anchor_pose_error_cm : mean over N samples.

Interpretation:
    | APD_hand    | best-of-N  | reading                                    |
    |-------------|------------|--------------------------------------------|
    | <2 cm       | high       | mode collapse confirmed                    |
    | >10 cm      | low        | multi-mode sampled; selection issue        |
    | >10 cm      | high       | multi-mode sampled but all wrong; other    |
    |             |            | bottleneck (capacity / data / FK)          |
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from plan_condition_diagnostics import (  # noqa: E402
    _build_cond, _build_dataset, _build_model, _stage1_norm_for_cfg,
)

from piano.data.dataset import collate_hoi  # noqa: E402
from piano.training.smpl_kinematics import (  # noqa: E402
    fk_from_global_rotations, rotation_6d_to_matrix,
)
from piano.utils.clip_utils import load_clip_text_encoder  # noqa: E402
from piano.utils.smpl_utils import (  # noqa: E402
    BODY_PART_INDICES, BODY_PART_NAMES,
)


# Indices of wrist and ankle in the 22-joint SMPL skeleton.
# (matches BODY_PART_INDICES order: L_hand, R_hand, L_foot, R_foot, pelvis)
HAND_JOINT_IDXS = (BODY_PART_INDICES[0], BODY_PART_INDICES[1])   # L+R wrist
FOOT_JOINT_IDXS = (BODY_PART_INDICES[2], BODY_PART_INDICES[3])   # L+R ankle


def _fk_22joints(motion: torch.Tensor, rest_offsets: torch.Tensor) -> torch.Tensor:
    """motion: (B, T, 135) → joints: (B, T, 22, 3) world frame."""
    B, T, _ = motion.shape
    rot6d = motion[..., :132].reshape(B, T, 22, 6).float()
    root_world = motion[..., 132:135].float()
    rot_mat = rotation_6d_to_matrix(rot6d)
    rest_per_frame = rest_offsets.unsqueeze(1).expand(B, T, 22, 3).float()
    return fk_from_global_rotations(rot_mat, rest_per_frame, root_world)


def _apd_on_joints(joint_traj_list: list[np.ndarray],
                   joint_idxs: tuple[int, ...]) -> float:
    """Average pairwise distance over a subset of joints (cm).

    joint_traj_list : list of (T, 22, 3) arrays, length N.
    joint_idxs      : which joint columns to include.
    Returns mean over { time × selected joints × C(N,2) pairs } of L2
    distance in cm.
    """
    if len(joint_traj_list) < 2:
        return 0.0
    selected = [jt[:, joint_idxs, :] for jt in joint_traj_list]
    # Truncate to min T to align.
    min_T = min(s.shape[0] for s in selected)
    selected = [s[:min_T] for s in selected]
    pair_dists: list[float] = []
    for i, j in itertools.combinations(range(len(selected)), 2):
        diff = selected[i] - selected[j]                       # (T, J, 3)
        d = np.linalg.norm(diff, axis=-1)                      # (T, J)
        pair_dists.append(float(d.mean()))
    return float(np.mean(pair_dists)) * 100.0                  # m → cm


def _anchor_pose_error_cm(pred_motion: torch.Tensor,
                          gt_motion: torch.Tensor,
                          rest_offsets: torch.Tensor,
                          plan: dict) -> float:
    """|pred_joint - gt_joint| at active anchor (k, part) frames; cm.

    Mirrors the Round-24 anchor_pose_error_cm definition from
    scripts/stage_b_generator/plan_condition_diagnostics.py:298-313.
    """
    pred_joints = _fk_22joints(pred_motion, rest_offsets)[0]     # (T, 22, 3)
    gt_joints = _fk_22joints(gt_motion, rest_offsets)[0]         # (T, 22, 3)
    T = pred_joints.shape[0]

    anchor_time = plan["anchor_time"][0].long()
    anchor_mask = plan["anchor_mask"][0].bool()
    anchor_part = plan["anchor_part"][0]

    errs: list[float] = []
    for k in range(anchor_time.shape[0]):
        if not bool(anchor_mask[k]):
            continue
        t = int(anchor_time[k].item())
        if t < 0 or t >= T:
            continue
        for p in range(anchor_part.shape[1]):
            if float(anchor_part[k, p].item()) <= 0.5:
                continue
            joint_idx = BODY_PART_INDICES[p]
            d = torch.linalg.norm(
                pred_joints[t, joint_idx] - gt_joints[t, joint_idx],
            ).item()
            errs.append(d * 100.0)
    return float(np.mean(errs)) if errs else 0.0


def _filter_dataset_by_selection(
    dataset, selection: list[dict],
) -> tuple[Subset, list[dict]]:
    """Return a Subset of dataset containing only clips in selection
    (matched by (subset, seq_id))."""
    sel_pairs = {(e["subset"], e["seq_id"]): e for e in selection}
    indices: list[int] = []
    matched_entries: list[dict] = []
    for i in range(len(dataset)):
        sample = dataset[i]
        key = (str(sample["subset"]), str(sample["seq_id"]))
        if key in sel_pairs:
            indices.append(i)
            matched_entries.append(sel_pairs[key])
            if len(indices) >= len(sel_pairs):
                break
    return Subset(dataset, indices), matched_entries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path, required=True,
                        help="D1 output: analyses/round25_multimodal_eval_subset.json. "
                             "Must contain a top-level 'selected' list with "
                             "{subset, seq_id, mode_category} entries.")
    parser.add_argument("--bucket", default="val", choices=["train", "val"])
    parser.add_argument("--n-samples", type=int, default=8,
                        help="Number of distinct seeds per clip.")
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed-base", type=int, default=42,
                        help="First sampler seed; subsequent seeds are seed_base + i.")
    parser.add_argument("--output", type=Path,
                        default=Path("analyses/round25_d2_diversity_stats.json"))
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load selection ----
    sel_obj = json.loads(args.selection_json.read_text("utf-8"))
    selection = sel_obj.get("selected", sel_obj.get("candidates", []))
    if not selection:
        raise SystemExit(f"[d2] no clips in {args.selection_json}")
    print(f"[d2] selection: {len(selection)} clips")

    # ---- Dataset ----
    full_dataset = _build_dataset(cfg, args.bucket, augment=False)
    subset_ds, matched = _filter_dataset_by_selection(full_dataset, selection)
    if not matched:
        raise SystemExit("[d2] none of the selection clips matched the val bucket")
    print(f"[d2] matched {len(matched)} / {len(selection)} clips in {args.bucket} bucket")
    loader = DataLoader(subset_ds, batch_size=1, shuffle=False,
                        collate_fn=collate_hoi, num_workers=0)

    # ---- Model + extras ----
    model, object_encoder, z_dims = _build_model(cfg, device)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model_state = state.get("model", state)
    model.load_state_dict(model_state)
    if "object_encoder" in state:
        object_encoder.load_state_dict(state["object_encoder"])
    elif "extra_modules" in state and "object_encoder" in state["extra_modules"]:
        object_encoder.load_state_dict(state["extra_modules"]["object_encoder"])
    clip_model = load_clip_text_encoder(
        device=device,
        model_name=str(cfg.model.text_encoder.clip_version),
        download_root=str(cfg.model.text_encoder.get("download_root", "cache/clip")),
    )
    stage1_norm = _stage1_norm_for_cfg(cfg, device)
    model.eval()

    plan_keys = [
        "anchor_time", "anchor_part", "anchor_target_local",
        "anchor_target_world", "anchor_type", "anchor_phase",
        "anchor_support", "anchor_conf", "anchor_mask",
        "segment_start", "segment_end", "segment_part",
        "segment_target_summary_local", "segment_phase",
        "segment_support", "segment_conf", "segment_mask",
    ]

    # ---- Loop over clips ----
    per_clip: list[dict] = []
    matched_iter = iter(matched)
    for batch in loader:
        sel_entry = next(matched_iter)
        subset = str(batch["subset"][0])
        seq_id = str(batch["seq_id"][0])
        text = str(batch["text"][0])

        cond, T = _build_cond(
            batch, model, object_encoder, clip_model, z_dims, cfg, device,
            stage1_norm=stage1_norm,
        )
        cond["interaction_plan"] = {
            k: batch[f"plan_{k}"].to(device) for k in plan_keys
        }
        plan_local = cond["interaction_plan"]

        gt_motion = batch["motion"][:, :T].to(device).float()
        rest_offsets = batch["rest_offsets"].to(device).float()
        seq_len = int(batch["seq_len"][0].item())
        valid_T = min(T, seq_len)

        # Sample N times with different seeds.
        joint_trajs: list[np.ndarray] = []
        per_sample_err: list[float] = []
        for i in range(args.n_samples):
            torch.manual_seed(args.seed_base + i)
            with torch.no_grad():
                pred_motion = model.sample(
                    cond=cond, seq_length=T, cfg_scale=args.cfg_scale,
                    replacement="none", output_skip=False,
                )
            joints = _fk_22joints(pred_motion[:, :valid_T], rest_offsets)[0]
            joint_trajs.append(joints.detach().cpu().numpy())
            err = _anchor_pose_error_cm(
                pred_motion[:, :valid_T], gt_motion[:, :valid_T],
                rest_offsets, plan_local,
            )
            per_sample_err.append(err)

        # Diversity (APD) per category.
        apd_hand = _apd_on_joints(joint_trajs, HAND_JOINT_IDXS)
        apd_foot = _apd_on_joints(joint_trajs, FOOT_JOINT_IDXS)
        apd_full = _apd_on_joints(joint_trajs, tuple(range(22)))

        per_clip.append({
            "subset": subset,
            "seq_id": seq_id,
            "text": text,
            "mode_category": sel_entry.get("mode_category",
                                            sel_entry.get("mode_category_guess", "unknown")),
            "T": valid_T,
            "n_samples": args.n_samples,
            "APD_hand_endpoint_cm": float(apd_hand),
            "APD_foot_endpoint_cm": float(apd_foot),
            "APD_22joint_cm": float(apd_full),
            "best_of_N_anchor_pose_error_cm": float(min(per_sample_err)) if per_sample_err else 0.0,
            "mean_anchor_pose_error_cm": float(np.mean(per_sample_err)) if per_sample_err else 0.0,
            "max_anchor_pose_error_cm": float(max(per_sample_err)) if per_sample_err else 0.0,
            "per_sample_anchor_pose_error_cm": per_sample_err,
        })
        print(f"  [d2 {len(per_clip)}/{len(matched)}] {subset}/{seq_id}  "
              f"APD_hand={apd_hand:.2f}cm  best-of-N={min(per_sample_err):.2f}cm  "
              f"mean={np.mean(per_sample_err):.2f}cm")

    # ---- Aggregate per mode_category and per subset ----
    by_category: dict[str, list[dict]] = {}
    by_subset: dict[str, list[dict]] = {}
    for r in per_clip:
        by_category.setdefault(r["mode_category"], []).append(r)
        by_subset.setdefault(r["subset"], []).append(r)

    def _agg(rows: list[dict]) -> dict:
        if not rows:
            return {}
        return {
            "n": len(rows),
            "APD_hand_endpoint_cm_mean": float(np.mean([r["APD_hand_endpoint_cm"] for r in rows])),
            "APD_foot_endpoint_cm_mean": float(np.mean([r["APD_foot_endpoint_cm"] for r in rows])),
            "APD_22joint_cm_mean": float(np.mean([r["APD_22joint_cm"] for r in rows])),
            "best_of_N_mean": float(np.mean([r["best_of_N_anchor_pose_error_cm"] for r in rows])),
            "mean_error_mean": float(np.mean([r["mean_anchor_pose_error_cm"] for r in rows])),
        }

    summary = {
        "config": str(args.config),
        "ckpt": str(args.ckpt),
        "selection_json": str(args.selection_json),
        "bucket": args.bucket,
        "n_samples_per_clip": args.n_samples,
        "cfg_scale": args.cfg_scale,
        "seed_base": args.seed_base,
        "overall": _agg(per_clip),
        "by_mode_category": {k: _agg(v) for k, v in by_category.items()},
        "by_subset": {k: _agg(v) for k, v in by_subset.items()},
        "per_clip": per_clip,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    print(f"\n[d2] wrote JSON to {args.output}")

    # ---- Markdown ----
    md = args.output.with_suffix(".md")
    lines: list[str] = []
    lines.append("# Round-25 D2 diversity diagnostic\n")
    lines.append(f"**Ckpt:** `{args.ckpt}`")
    lines.append(f"**Selection:** `{args.selection_json}` ({len(per_clip)} clips × N={args.n_samples})")
    lines.append(f"**cfg_scale:** {args.cfg_scale}    **seed_base:** {args.seed_base}\n")
    lines.append("## Decision rule\n")
    lines.append("| APD_hand_endpoint_cm | best-of-N error | reading |")
    lines.append("|---:|---:|---|")
    lines.append("| < 2 | high | **mode collapse confirmed** → P1 mode mechanism |")
    lines.append("| > 10 | low | multi-mode sampled, ranking issue → CFG tuning |")
    lines.append("| > 10 | high | multi-mode but all wrong → other bottleneck |\n")
    ov = summary["overall"]
    if ov:
        lines.append("## Overall\n")
        lines.append("| metric | mean |")
        lines.append("|---|---:|")
        for k, v in ov.items():
            if k == "n":
                continue
            lines.append(f"| {k} | {v:.3f} |")
    lines.append("\n## By mode category\n")
    lines.append("| category | n | APD_hand | APD_foot | APD_22joint | best-of-N | mean error |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for cat, agg in summary["by_mode_category"].items():
        lines.append(
            f"| {cat} | {agg['n']} | {agg['APD_hand_endpoint_cm_mean']:.2f} | "
            f"{agg['APD_foot_endpoint_cm_mean']:.2f} | {agg['APD_22joint_cm_mean']:.2f} | "
            f"{agg['best_of_N_mean']:.2f} | {agg['mean_error_mean']:.2f} |"
        )
    lines.append("\n## By subset\n")
    lines.append("| subset | n | APD_hand | APD_foot | best-of-N | mean error |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for sub, agg in summary["by_subset"].items():
        lines.append(
            f"| {sub} | {agg['n']} | {agg['APD_hand_endpoint_cm_mean']:.2f} | "
            f"{agg['APD_foot_endpoint_cm_mean']:.2f} | "
            f"{agg['best_of_N_mean']:.2f} | {agg['mean_error_mean']:.2f} |"
        )
    md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[d2] wrote Markdown to {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
