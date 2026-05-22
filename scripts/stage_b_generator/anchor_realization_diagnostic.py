"""Round-24 P1 anchor realization diagnostic.

Why this exists
---------------
The Round-23 P1 fullscale diagnostic showed both v25 R23 ckpts (with-plan
and no-plan) FAIL the anchor gate: ``plan_anchor_contact_realization_cm
≈ 32 cm`` at GT plan + GT route, on both models identically. That tells
us anchor placement is broken INDEPENDENT of whether plan is conditioning
the model — it's a Stage-2 decoder / data / FK issue. This script
classifies the failure into one of four root causes by direct measurement
per anchor frame.

The four hypotheses being tested
--------------------------------

Q1. Is the model's full body pose at anchor frames correct,
    or is it only the anchor body part (hand/foot/pelvis) that's wrong?
    - measure full 22-joint L2 vs anchor-part-only L2 at anchor frames

Q2. What's the spatial structure of the hand-position error?
    - delta = pred_hand_world - target_world; report magnitude / direction
      / per-part / per-subset

Q3. Is contact_target_xyz GT label itself noisy?
    - Distance comparison: pred_to_target, GT_to_target, pred_to_GT
    - If GT_to_target is also high → label noise (GT motion doesn't itself
      reach the contact target)
    - If GT_to_target is low but pred_to_target is high → decoder issue

Q4. Is anchor-frame velocity off?
    - Compare local joint velocity in ±3 frames around anchor

Input + output
--------------
Input:
  --config       a v25 R23 config (with-plan or no-plan)
  --ckpt         the trained ckpt
  --selection    JSON with the clip list to evaluate (defaults to R19 selection)
  --output-dir   per-clip anchor stats + summary

Output:
  <output_dir>/anchor_stats.json     full per-anchor table
  <output_dir>/anchor_summary.md     human-readable summary (Q1-Q4)
  <output_dir>/per_subset_bars.png   bar chart per subset
  <output_dir>/distance_scatter.png  pred_to_target vs GT_to_target scatter
  <output_dir>/per_part_histogram.png  histogram per body-part

Wallclock: ~20 min for 32 clips × DDPM sample on cuda:0.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from omegaconf import OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader, Subset

# Reuse helpers from existing diagnostic.
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from plan_condition_diagnostics import (  # noqa: E402
    _build_cond, _build_dataset, _build_model, _stage1_norm_for_cfg,
)

from piano.data.dataset import collate_hoi  # noqa: E402
from piano.training.anchor_consistency_loss import (  # noqa: E402
    lift_object_local_to_world,
)
from piano.training.smpl_kinematics import (  # noqa: E402
    fk_from_global_rotations,
    rotation_6d_to_matrix,
)
from piano.utils.clip_utils import load_clip_text_encoder  # noqa: E402
from piano.utils.smpl_utils import (  # noqa: E402
    BODY_PART_INDICES,
    BODY_PART_NAMES,
    INTERACTION_BODY_PARTS,
)


# ---------------------------------------------------------------------------
# FK from motion_135
# ---------------------------------------------------------------------------


def _fk_22joints(motion: Tensor, rest_offsets: Tensor) -> Tensor:
    """motion: (B, T, 135) → joints: (B, T, 22, 3) world frame."""
    B, T, _ = motion.shape
    rot6d = motion[..., :132].reshape(B, T, 22, 6).float()
    root_world = motion[..., 132:135].float()
    rot_mat = rotation_6d_to_matrix(rot6d)
    rest_per_frame = rest_offsets.unsqueeze(1).expand(B, T, 22, 3).float()
    return fk_from_global_rotations(rot_mat, rest_per_frame, root_world)


# ---------------------------------------------------------------------------
# Per-anchor scoring
# ---------------------------------------------------------------------------


def _score_one_clip(
    *,
    pred_motion: Tensor,         # (1, T, 135)
    gt_motion: Tensor,           # (1, T, 135)
    rest_offsets: Tensor,        # (1, 22, 3)
    plan: dict[str, Tensor],
    obj_pos_world: Tensor,       # (1, T, 3)
    obj_rot_world: Tensor,       # (1, T, 3) axis-angle
    subset: str,
    seq_id: str,
) -> list[dict[str, Any]]:
    """Per anchor (k) × per active part (p): score the four metrics."""
    pred_joints = _fk_22joints(pred_motion, rest_offsets)[0]    # (T, 22, 3)
    gt_joints = _fk_22joints(gt_motion, rest_offsets)[0]        # (T, 22, 3)
    T = pred_joints.shape[0]

    anchor_time = plan["anchor_time"][0].long()                  # (K,)
    anchor_mask = plan["anchor_mask"][0].bool()                  # (K,)
    anchor_part = plan["anchor_part"][0]                         # (K, P)
    anchor_target_local = plan["anchor_target_local"][0]         # (K, P, 3)

    rows: list[dict[str, Any]] = []
    for k in range(anchor_time.shape[0]):
        if not bool(anchor_mask[k]):
            continue
        t = int(anchor_time[k].item())
        if t < 0 or t >= T:
            continue
        # Lift target_local at this anchor's time to world.
        # Use OBJECT pose at THE ANCHOR FRAME (t), not at compile time.
        target_local_kp = anchor_target_local[k]                 # (P, 3)
        # Lift each part: target_world = R(t) @ target_local + obj_pos(t)
        R_t = _aa_matrix(obj_rot_world[0, t])                    # (3, 3)
        target_world_kp = (R_t @ target_local_kp.T).T + obj_pos_world[0, t]  # (P, 3)

        for p in range(anchor_part.shape[1]):
            active = float(anchor_part[k, p].item()) > 0.5
            if not active:
                continue
            joint_idx = BODY_PART_INDICES[p]
            pred_joint = pred_joints[t, joint_idx]                # (3,)
            gt_joint = gt_joints[t, joint_idx]                    # (3,)
            target = target_world_kp[p]                           # (3,)

            pred_to_target = float(torch.linalg.norm(pred_joint - target).item())
            gt_to_target = float(torch.linalg.norm(gt_joint - target).item())
            pred_to_gt = float(torch.linalg.norm(pred_joint - gt_joint).item())

            # Q1: full body L2 at this frame (pred vs GT, 22-joint sum).
            full_body_l2 = float(
                torch.linalg.norm(pred_joints[t] - gt_joints[t], dim=-1).mean().item()
            )

            # Q4: local velocity at ±3 frames (mean speed delta).
            w = 3
            t_lo = max(0, t - w)
            t_hi = min(T - 1, t + w)
            if t_hi - t_lo >= 2:
                pred_vel = torch.diff(pred_joints[t_lo:t_hi + 1, joint_idx], dim=0)
                gt_vel = torch.diff(gt_joints[t_lo:t_hi + 1, joint_idx], dim=0)
                pred_speed_mean = float(torch.linalg.norm(pred_vel, dim=-1).mean().item())
                gt_speed_mean = float(torch.linalg.norm(gt_vel, dim=-1).mean().item())
            else:
                pred_speed_mean = gt_speed_mean = 0.0

            # Spatial offset components (for direction analysis).
            offset = (pred_joint - target).cpu().numpy()

            rows.append({
                "subset": subset,
                "seq_id": seq_id,
                "anchor_idx": k,
                "anchor_time": t,
                "part_idx": p,
                "part_name": BODY_PART_NAMES[p],
                "joint_idx": joint_idx,
                "pred_to_target_cm": pred_to_target * 100.0,
                "gt_to_target_cm": gt_to_target * 100.0,
                "pred_to_gt_cm": pred_to_gt * 100.0,
                "full_body_l2_cm": full_body_l2 * 100.0,
                "pred_speed_cm_per_frame": pred_speed_mean * 100.0,
                "gt_speed_cm_per_frame": gt_speed_mean * 100.0,
                "offset_x_cm": float(offset[0]) * 100.0,
                "offset_y_cm": float(offset[1]) * 100.0,
                "offset_z_cm": float(offset[2]) * 100.0,
            })
    return rows


def _aa_matrix(aa: Tensor) -> Tensor:
    """Axis-angle (3,) → rotation matrix (3, 3). Rodrigues, batched-able."""
    aa = aa.float()
    theta = torch.linalg.norm(aa).clamp_min(1e-9)
    k = aa / theta
    K = torch.zeros(3, 3, device=aa.device, dtype=aa.dtype)
    K[0, 1] = -k[2]; K[0, 2] = k[1]
    K[1, 0] = k[2];  K[1, 2] = -k[0]
    K[2, 0] = -k[1]; K[2, 1] = k[0]
    I3 = torch.eye(3, device=aa.device, dtype=aa.dtype)
    return I3 + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)


# ---------------------------------------------------------------------------
# Aggregation + plotting
# ---------------------------------------------------------------------------


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0, "mean": float("nan"), "median": float("nan"),
                "p95": float("nan"), "min": float("nan"), "max": float("nan")}
    a = np.asarray(values, dtype=np.float64)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p95": float(np.percentile(a, 95)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def _plot_scatter_pred_vs_gt(rows: list[dict], out_path: Path) -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pred_vals = np.asarray([r["pred_to_target_cm"] for r in rows])
    gt_vals = np.asarray([r["gt_to_target_cm"] for r in rows])
    parts = np.asarray([r["part_idx"] for r in rows])

    fig, ax = plt.subplots(figsize=(7, 7))
    cmap = plt.get_cmap("tab10")
    for p_idx in sorted(set(parts.tolist())):
        m = parts == p_idx
        ax.scatter(
            gt_vals[m], pred_vals[m],
            label=f"{BODY_PART_NAMES[p_idx]} (n={int(m.sum())})",
            color=cmap(p_idx), alpha=0.5, s=18,
        )
    lim = max(pred_vals.max(), gt_vals.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.7, alpha=0.5, label="y = x (pred = GT)")
    ax.set_xlabel("GT_to_target (cm) — does GT motion reach the contact target?")
    ax.set_ylabel("pred_to_target (cm) — does our model reach the contact target?")
    ax.set_title(
        "Anchor realization: model vs GT distance to contact_target\n"
        "Points BELOW y=x: model better than GT (label noise / model lucky)\n"
        "Points ABOVE y=x: model worse than GT (decoder issue OR GT also poor)"
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _plot_per_part_histogram(rows: list[dict], out_path: Path) -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    cmap = plt.get_cmap("tab10")

    metrics = [
        ("pred_to_target_cm", "pred → target (model anchor error)"),
        ("gt_to_target_cm", "GT → target (label fidelity)"),
        ("pred_to_gt_cm", "pred → GT (model pose error at anchor frame)"),
    ]
    for ax, (key, title) in zip(axes, metrics):
        for p_idx in sorted({r["part_idx"] for r in rows}):
            vals = [r[key] for r in rows if r["part_idx"] == p_idx]
            if not vals:
                continue
            ax.hist(vals, bins=30, alpha=0.45, label=f"{BODY_PART_NAMES[p_idx]} (n={len(vals)})",
                    color=cmap(p_idx))
        ax.set_xlabel(f"{title} (cm)")
        ax.set_ylabel("count")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _plot_per_subset_bars(rows: list[dict], out_path: Path) -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subset_to_metric: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"pred_to_target": [], "gt_to_target": [], "pred_to_gt": []}
    )
    for r in rows:
        s = r["subset"]
        subset_to_metric[s]["pred_to_target"].append(r["pred_to_target_cm"])
        subset_to_metric[s]["gt_to_target"].append(r["gt_to_target_cm"])
        subset_to_metric[s]["pred_to_gt"].append(r["pred_to_gt_cm"])

    subsets = sorted(subset_to_metric.keys())
    x = np.arange(len(subsets))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    means_pt = [float(np.mean(subset_to_metric[s]["pred_to_target"])) for s in subsets]
    means_gt = [float(np.mean(subset_to_metric[s]["gt_to_target"])) for s in subsets]
    means_pg = [float(np.mean(subset_to_metric[s]["pred_to_gt"])) for s in subsets]
    ax.bar(x - width, means_pt, width, label="pred → target")
    ax.bar(x, means_gt, width, label="GT → target (label fidelity)")
    ax.bar(x + width, means_pg, width, label="pred → GT (pose error)")
    ax.axhline(20.0, color="red", linewidth=1, linestyle="--", alpha=0.5,
               label="§9.2 anchor gate (< 20 cm)")
    ax.set_xticks(x); ax.set_xticklabels(subsets, rotation=15)
    ax.set_ylabel("mean error (cm)")
    ax.set_title("Anchor realization per subset (mean per body-part per anchor)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument(
        "--selection-json", type=Path,
        default=Path("analyses/2026-05-20_round19_eval_selection.json"),
        help="Reuse the R19/R20 selection (32 clips) for cross-round comparability.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bucket", default="val", choices=["train", "val"])
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-clips", type=int, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Selection ----
    if args.selection_json.exists():
        sel = json.loads(args.selection_json.read_text("utf-8"))
        sel_pairs = {(e["subset"], e["seq_id"]) for e in sel.get("selected", [])}
        print(f"[anchor-diag] using {len(sel_pairs)} clips from {args.selection_json}")
    else:
        sel_pairs = None
        print(f"[anchor-diag] no selection JSON — using first --max-clips val clips")

    # ---- Dataset ----
    dataset = _build_dataset(cfg, args.bucket, augment=False)
    overfit_n = int(cfg.data.get("overfit_n_clips", 0))
    if overfit_n > 0:
        dataset = Subset(dataset, list(range(min(overfit_n, len(dataset)))))
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=collate_hoi, num_workers=0,
    )

    # ---- Model ----
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

    # ---- Iterate ----
    all_rows: list[dict[str, Any]] = []
    n_clips = 0
    for batch in loader:
        subset = str(batch["subset"][0])
        seq_id = str(batch["seq_id"][0])
        if sel_pairs is not None and (subset, seq_id) not in sel_pairs:
            continue
        if args.max_clips is not None and n_clips >= args.max_clips:
            break
        cond, T = _build_cond(
            batch, model, object_encoder, clip_model, z_dims, cfg, device,
            stage1_norm=stage1_norm,
        )
        plan_keys = [
            "anchor_time", "anchor_part", "anchor_target_local",
            "anchor_target_world", "anchor_type", "anchor_phase",
            "anchor_support", "anchor_conf", "anchor_mask",
            "segment_start", "segment_end", "segment_part",
            "segment_target_summary_local", "segment_phase",
            "segment_support", "segment_conf", "segment_mask",
        ]
        plan = {k: batch[f"plan_{k}"].to(device) for k in plan_keys}
        cond["interaction_plan"] = plan

        torch.manual_seed(args.seed)
        with torch.no_grad():
            # Stage-2 MotionAnchorDiff.sample signature matches the
            # plan_condition_diagnostics call site (no inpaint_frame0 arg).
            pred_motion = model.sample(
                cond=cond, seq_length=T, cfg_scale=args.cfg_scale,
                replacement="none", output_skip=False,
            )                                                       # (1, T, 135)

        gt_motion = batch["motion"][:, :T].to(device).float()
        rest_offsets = batch["rest_offsets"].to(device).float()
        obj_pos_world = batch["object_positions"][:, :T].to(device).float()
        obj_rot_world = batch["object_rotations"][:, :T].to(device).float()

        rows = _score_one_clip(
            pred_motion=pred_motion, gt_motion=gt_motion,
            rest_offsets=rest_offsets, plan=plan,
            obj_pos_world=obj_pos_world, obj_rot_world=obj_rot_world,
            subset=subset, seq_id=seq_id,
        )
        all_rows.extend(rows)
        n_clips += 1
        if n_clips % 4 == 0:
            print(f"  [anchor-diag] processed {n_clips} clips, {len(all_rows)} anchors so far")

    if not all_rows:
        raise RuntimeError("no anchors collected — check selection JSON / bucket")

    # ---- Aggregate ----
    overall = {
        "pred_to_target_cm": _stats([r["pred_to_target_cm"] for r in all_rows]),
        "gt_to_target_cm": _stats([r["gt_to_target_cm"] for r in all_rows]),
        "pred_to_gt_cm": _stats([r["pred_to_gt_cm"] for r in all_rows]),
        "full_body_l2_cm": _stats([r["full_body_l2_cm"] for r in all_rows]),
        "pred_speed_cm_per_frame": _stats([r["pred_speed_cm_per_frame"] for r in all_rows]),
        "gt_speed_cm_per_frame": _stats([r["gt_speed_cm_per_frame"] for r in all_rows]),
    }
    per_subset = {}
    per_part = {}
    for s in sorted({r["subset"] for r in all_rows}):
        subset_rows = [r for r in all_rows if r["subset"] == s]
        per_subset[s] = {
            "n_anchors": len(subset_rows),
            "pred_to_target_cm": _stats([r["pred_to_target_cm"] for r in subset_rows]),
            "gt_to_target_cm": _stats([r["gt_to_target_cm"] for r in subset_rows]),
            "pred_to_gt_cm": _stats([r["pred_to_gt_cm"] for r in subset_rows]),
        }
    for p_idx, p_name in enumerate(BODY_PART_NAMES):
        part_rows = [r for r in all_rows if r["part_idx"] == p_idx]
        if not part_rows:
            continue
        per_part[p_name] = {
            "n_anchors": len(part_rows),
            "pred_to_target_cm": _stats([r["pred_to_target_cm"] for r in part_rows]),
            "gt_to_target_cm": _stats([r["gt_to_target_cm"] for r in part_rows]),
            "pred_to_gt_cm": _stats([r["pred_to_gt_cm"] for r in part_rows]),
        }

    # ---- Save JSON + markdown ----
    json_payload = {
        "config": str(args.config),
        "ckpt": str(args.ckpt),
        "selection_json": str(args.selection_json),
        "bucket": args.bucket,
        "cfg_scale": args.cfg_scale,
        "seed": args.seed,
        "n_clips": n_clips,
        "n_anchors": len(all_rows),
        "overall": overall,
        "per_subset": per_subset,
        "per_part": per_part,
        "rows": all_rows,
    }
    (args.output_dir / "anchor_stats.json").write_text(
        json.dumps(json_payload, indent=2), encoding="utf-8"
    )

    md = ["# Anchor realization diagnostic\n",
          f"**Ckpt:** `{args.ckpt}`",
          f"**Selection:** `{args.selection_json}` ({n_clips} clips, {len(all_rows)} anchors)",
          f"**cfg_scale:** {args.cfg_scale}  **seed:** {args.seed}  **bucket:** {args.bucket}\n",
          "## Q1+Q2+Q3 overall (cm)\n",
          "| metric | n | mean | median | p95 | min | max |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for key, label in [
        ("pred_to_target_cm", "pred → target (model anchor error)"),
        ("gt_to_target_cm", "GT → target (label fidelity)"),
        ("pred_to_gt_cm", "pred → GT (pose error at anchor)"),
        ("full_body_l2_cm", "full 22-joint L2 at anchor frame"),
    ]:
        s = overall[key]
        md.append(f"| {label} | {s['n']} | {s['mean']:.2f} | {s['median']:.2f} | "
                  f"{s['p95']:.2f} | {s['min']:.2f} | {s['max']:.2f} |")

    md.append("\n## Per body-part\n")
    md.append("| part | n | pred→target mean | GT→target mean | pred→GT mean |")
    md.append("|---|---:|---:|---:|---:|")
    for p_name, st in per_part.items():
        md.append(
            f"| {p_name} | {st['n_anchors']} | "
            f"{st['pred_to_target_cm']['mean']:.2f} | "
            f"{st['gt_to_target_cm']['mean']:.2f} | "
            f"{st['pred_to_gt_cm']['mean']:.2f} |"
        )

    md.append("\n## Per subset\n")
    md.append("| subset | n | pred→target mean | GT→target mean | pred→GT mean |")
    md.append("|---|---:|---:|---:|---:|")
    for s_name, st in per_subset.items():
        md.append(
            f"| {s_name} | {st['n_anchors']} | "
            f"{st['pred_to_target_cm']['mean']:.2f} | "
            f"{st['gt_to_target_cm']['mean']:.2f} | "
            f"{st['pred_to_gt_cm']['mean']:.2f} |"
        )

    md.append("\n## Q4 velocity (cm/frame, ±3 window around anchor)\n")
    md.append("| metric | n | mean | median | p95 |")
    md.append("|---|---:|---:|---:|---:|")
    for key, label in [
        ("pred_speed_cm_per_frame", "pred local speed"),
        ("gt_speed_cm_per_frame", "GT local speed"),
    ]:
        s = overall[key]
        md.append(f"| {label} | {s['n']} | {s['mean']:.2f} | {s['median']:.2f} | {s['p95']:.2f} |")

    md.append("\n## Interpretation key\n")
    md.append("- **`GT → target` low + `pred → target` high** → decoder issue: model can't reach contact points even though GT motion does.")
    md.append("- **`GT → target` high too** → label noise: GT motion itself doesn't reach the labeled contact_target.")
    md.append("- **`pred → GT` low + `pred → target` high** → pose at anchor frame is correct but target geometry is wrong.")
    md.append("- **`pred → GT` high** → general decoder issue, not specific to contact.")
    md.append("- Compare per-part: hands are the typical contact body parts; feet/pelvis matter for sit/lie.")
    md.append("- Compare per-subset to see if certain HOI datasets dominate the failure.")
    (args.output_dir / "anchor_summary.md").write_text("\n".join(md), encoding="utf-8")

    # ---- Plots ----
    _plot_scatter_pred_vs_gt(all_rows, args.output_dir / "distance_scatter.png")
    _plot_per_part_histogram(all_rows, args.output_dir / "per_part_histogram.png")
    _plot_per_subset_bars(all_rows, args.output_dir / "per_subset_bars.png")

    # ---- Stdout summary ----
    print(f"\n[anchor-diag] processed {n_clips} clips, {len(all_rows)} anchors")
    print(f"  overall mean cm — pred→target: {overall['pred_to_target_cm']['mean']:.2f}  "
          f"GT→target: {overall['gt_to_target_cm']['mean']:.2f}  "
          f"pred→GT: {overall['pred_to_gt_cm']['mean']:.2f}")
    print(f"\nWrote:")
    print(f"  {args.output_dir / 'anchor_stats.json'}")
    print(f"  {args.output_dir / 'anchor_summary.md'}")
    print(f"  {args.output_dir / 'distance_scatter.png'}")
    print(f"  {args.output_dir / 'per_part_histogram.png'}")
    print(f"  {args.output_dir / 'per_subset_bars.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
