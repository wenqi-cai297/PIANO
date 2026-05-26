"""Round-26 sustained contact diagnostic.

For each pseudo-labeled contact segment (contiguous frames where
``contact_state[..., part] >= 0.5``), measure whether the predicted
limb maintains GT-like spatial relationship to the object throughout
the segment — not just at sparse anchor frames.

This captures the user-reported failure mode:

    "把箱子举高时，人的手只会跟着箱子往上抬一段，不会真的用手把箱子抬到对应的高度"

A per-frame anchor metric (Round-24 / our `anchor_realization_diagnostic`)
only samples one moment per anchor; it misses temporal drift within a
contact window.

Per-segment metrics (units cm unless noted):

    drift_max_cm     = max_t (pred_to_target[t] - gt_to_target[t])
    drift_end_cm     = (pred_to_target[t1] - gt_to_target[t1])
                       - (pred_to_target[t0] - gt_to_target[t0])
                       (positive = pred fell behind GT by end of segment)
    pred_align_cm    = (pred_part[t1] - pred_part[t0]) · obj_disp_dir
                       (how far the predicted joint moved in the direction
                       the object moved during the segment)
    gt_align_cm      = (gt_part[t1] - gt_part[t0]) · obj_disp_dir
    tracking_fraction= pred_align / gt_align    (= 1.0 if pred tracks
                       object exactly as GT does; 0.5 = "only went halfway";
                       reported only when obj displacement > 10 cm)
    rel_var_ratio    = var(pred_part - obj_pos, axis=0).sum()
                       / var(gt_part - obj_pos, axis=0).sum()
                       (large = pred's part-relative-to-object wanders;
                       small = pred tracks object more tightly than GT)

Usage::

    conda run -n piano python scripts/stage_b_generator/round26_sustained_contact_diag.py \
        --config configs/training/anchordiff_v27_stage2_anchoraware_FULL_DATA_local.yaml \
        --ckpt   runs/training/stageB_anchordiff_v27_stage2_anchoraware_FULL_DATA/final.pt \
        --selection-json analyses/round25_multimodal_eval_subset.json \
        --output-dir analyses/round26_sustained_contact_v27_final

To establish the GT-self baseline (should give drift≈0, tracking≈1.0)::

    ... --use-gt-as-pred --output-dir analyses/round26_sustained_contact_gt_reference
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from plan_condition_diagnostics import (  # noqa: E402
    extract_train_time_meta,
    _build_cond, _build_dataset, _build_model, _stage1_norm_for_cfg,
)
from anchor_realization_diagnostic import _aa_matrix, _fk_22joints  # noqa: E402

from piano.data.dataset import collate_hoi  # noqa: E402
from piano.utils.clip_utils import load_clip_text_encoder  # noqa: E402
from piano.utils.smpl_utils import BODY_PART_INDICES, BODY_PART_NAMES  # noqa: E402


CONTACT_THRESHOLD = 0.5
MIN_SEGMENT_LEN_DEFAULT = 5      # frames
OBJ_MOTION_MIN_CM = 10.0         # below this we don't compute tracking_fraction


def find_contact_segments(contact_state_p: np.ndarray,
                          seq_mask: np.ndarray,
                          min_len: int) -> list[tuple[int, int]]:
    """Return list of (t0, t1_inclusive) contiguous intervals where
    contact_state_p[t] >= CONTACT_THRESHOLD and seq_mask[t] is True."""
    active = (contact_state_p >= CONTACT_THRESHOLD) & seq_mask.astype(bool)
    segments: list[tuple[int, int]] = []
    in_seg = False
    t0 = 0
    T = len(active)
    for t in range(T):
        if active[t] and not in_seg:
            t0 = t; in_seg = True
        elif not active[t] and in_seg:
            t1 = t - 1
            if t1 - t0 + 1 >= min_len:
                segments.append((t0, t1))
            in_seg = False
    if in_seg:
        t1 = T - 1
        if t1 - t0 + 1 >= min_len:
            segments.append((t0, t1))
    return segments


def sustained_metrics_for_segment(
    pred_part_xyz: np.ndarray,           # (Tseg, 3) world frame
    gt_part_xyz: np.ndarray,             # (Tseg, 3)
    obj_pos_world: np.ndarray,           # (Tseg, 3)
    obj_rot_aa: np.ndarray,              # (Tseg, 3) axis-angle
    contact_target_local: np.ndarray,    # (3,) object-local target for this part
) -> dict[str, Any]:
    Tseg = len(pred_part_xyz)
    # Per-frame world target
    target_world = np.zeros((Tseg, 3), dtype=np.float64)
    for t in range(Tseg):
        aa = torch.from_numpy(obj_rot_aa[t]).float()
        R = _aa_matrix(aa).cpu().numpy().astype(np.float64)
        target_world[t] = R @ contact_target_local + obj_pos_world[t]

    pred_to_target = np.linalg.norm(pred_part_xyz - target_world, axis=-1)  # (Tseg,)
    gt_to_target = np.linalg.norm(gt_part_xyz - target_world, axis=-1)
    gap = pred_to_target - gt_to_target                                      # (Tseg,)

    drift_max = float(gap.max())
    drift_mean = float(gap.mean())
    drift_end = float(gap[-1] - gap[0])

    # Tracking: how far did pred/gt move in the direction the object moved?
    obj_disp = obj_pos_world[-1] - obj_pos_world[0]
    obj_disp_norm = float(np.linalg.norm(obj_disp))
    pred_disp = pred_part_xyz[-1] - pred_part_xyz[0]
    gt_disp = gt_part_xyz[-1] - gt_part_xyz[0]
    if obj_disp_norm > 1e-6:
        u = obj_disp / obj_disp_norm
        pred_align = float(pred_disp @ u)
        gt_align = float(gt_disp @ u)
    else:
        pred_align = gt_align = 0.0
    tracking_fraction: float | None
    if obj_disp_norm * 100 > OBJ_MOTION_MIN_CM and abs(gt_align) > 0.01:
        tracking_fraction = pred_align / gt_align
    else:
        tracking_fraction = None  # object barely moved; tracking undefined

    # Variance of part-relative-to-object position (does pred grip stay fixed?)
    pred_rel = pred_part_xyz - obj_pos_world                                 # (Tseg, 3)
    gt_rel = gt_part_xyz - obj_pos_world
    pred_rel_var = float(np.var(pred_rel, axis=0).sum())
    gt_rel_var = float(np.var(gt_rel, axis=0).sum())
    rel_var_ratio = (pred_rel_var / max(gt_rel_var, 1e-9)) if gt_rel_var > 1e-9 else None

    return {
        "segment_length": int(Tseg),
        "drift_max_cm": drift_max * 100.0,
        "drift_mean_cm": drift_mean * 100.0,
        "drift_end_cm": drift_end * 100.0,
        "obj_disp_cm": obj_disp_norm * 100.0,
        "pred_align_cm": pred_align * 100.0,
        "gt_align_cm": gt_align * 100.0,
        "tracking_fraction": float(tracking_fraction) if tracking_fraction is not None else None,
        "pred_rel_var_cm2": pred_rel_var * 10000.0,
        "gt_rel_var_cm2": gt_rel_var * 10000.0,
        "rel_var_ratio": float(rel_var_ratio) if rel_var_ratio is not None else None,
    }


def aggregate(rows: list[dict]) -> dict:
    """Aggregate per-part and per-subset."""
    overall = _stats_block(rows)
    per_part: dict[str, dict] = {}
    for p_idx, p_name in enumerate(BODY_PART_NAMES):
        part_rows = [r for r in rows if r["part_idx"] == p_idx]
        if not part_rows: continue
        per_part[p_name] = _stats_block(part_rows)
    per_subset: dict[str, dict] = {}
    for s in sorted({r["subset"] for r in rows}):
        sub_rows = [r for r in rows if r["subset"] == s]
        per_subset[s] = _stats_block(sub_rows)
    return {"overall": overall, "per_part": per_part, "per_subset": per_subset}


def _stats_block(rows: list[dict]) -> dict:
    drift_max = np.array([r["drift_max_cm"] for r in rows])
    drift_end = np.array([r["drift_end_cm"] for r in rows])
    drift_mean = np.array([r["drift_mean_cm"] for r in rows])
    tracking = [r["tracking_fraction"] for r in rows if r["tracking_fraction"] is not None]
    tracking_a = np.array(tracking) if tracking else np.array([])
    rel_var = [r["rel_var_ratio"] for r in rows if r["rel_var_ratio"] is not None]
    rel_var_a = np.array(rel_var) if rel_var else np.array([])
    n_failed = int((np.array(tracking) < 0.5).sum()) if tracking else 0
    n_drift_5cm = int((drift_max > 5).sum())
    n_drift_10cm = int((drift_max > 10).sum())
    return {
        "n_segments": len(rows),
        "drift_max_cm": {
            "mean": float(drift_max.mean()) if len(drift_max) else 0.0,
            "median": float(np.median(drift_max)) if len(drift_max) else 0.0,
            "p75": float(np.percentile(drift_max, 75)) if len(drift_max) else 0.0,
            "p95": float(np.percentile(drift_max, 95)) if len(drift_max) else 0.0,
            "max": float(drift_max.max()) if len(drift_max) else 0.0,
        },
        "drift_mean_cm": {
            "mean": float(drift_mean.mean()) if len(drift_mean) else 0.0,
            "median": float(np.median(drift_mean)) if len(drift_mean) else 0.0,
        },
        "drift_end_cm": {
            "mean": float(drift_end.mean()) if len(drift_end) else 0.0,
            "median": float(np.median(drift_end)) if len(drift_end) else 0.0,
        },
        "tracking_fraction": {
            "n_with_obj_motion": int(tracking_a.size),
            "mean": float(tracking_a.mean()) if tracking_a.size else None,
            "median": float(np.median(tracking_a)) if tracking_a.size else None,
            "n_below_0.5": n_failed,
            "rate_below_0.5": (n_failed / tracking_a.size) if tracking_a.size else None,
        },
        "rel_var_ratio": {
            "n": int(rel_var_a.size),
            "mean": float(rel_var_a.mean()) if rel_var_a.size else None,
            "median": float(np.median(rel_var_a)) if rel_var_a.size else None,
        },
        "n_drift_max_above_5cm": n_drift_5cm,
        "n_drift_max_above_10cm": n_drift_10cm,
    }


def write_summary_md(stats: dict, ckpt_path: Path, out_path: Path,
                     use_gt_as_pred: bool) -> None:
    label = "GT-self reference" if use_gt_as_pred else f"ckpt={ckpt_path.name}"
    L = [
        "# Round-26 sustained contact diagnostic",
        "",
        f"**Source:** `{ckpt_path}`",
        f"**Mode:** {'GT used in place of model sample (sanity)' if use_gt_as_pred else 'model sample'}",
        f"**Min segment length:** {MIN_SEGMENT_LEN_DEFAULT} frames",
        "",
        "## Overall",
        "",
    ]
    o = stats["overall"]
    L.append(f"- n_segments = {o['n_segments']}")
    L.append(f"- drift_max_cm:   mean={o['drift_max_cm']['mean']:.2f}  median={o['drift_max_cm']['median']:.2f}  p75={o['drift_max_cm']['p75']:.2f}  p95={o['drift_max_cm']['p95']:.2f}  max={o['drift_max_cm']['max']:.2f}")
    L.append(f"- drift_end_cm:   mean={o['drift_end_cm']['mean']:+.2f}  median={o['drift_end_cm']['median']:+.2f}")
    if o['tracking_fraction']['mean'] is not None:
        L.append(f"- tracking_fraction (obj moved >10cm, n={o['tracking_fraction']['n_with_obj_motion']}): "
                 f"mean={o['tracking_fraction']['mean']:.3f}  median={o['tracking_fraction']['median']:.3f}  "
                 f"%<0.5={100*o['tracking_fraction']['rate_below_0.5']:.1f}%  "
                 f"(n_below_0.5={o['tracking_fraction']['n_below_0.5']})")
    L.append(f"- segments with drift_max > 5cm:  {o['n_drift_max_above_5cm']} / {o['n_segments']} ({100*o['n_drift_max_above_5cm']/max(o['n_segments'],1):.1f}%)")
    L.append(f"- segments with drift_max > 10cm: {o['n_drift_max_above_10cm']} / {o['n_segments']} ({100*o['n_drift_max_above_10cm']/max(o['n_segments'],1):.1f}%)")
    L.append("")
    L.append("## Per-part")
    L.append("")
    L.append("| part | n | drift_max mean | drift_max p95 | drift_end mean | track frac mean | %<0.5 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for part, st in stats["per_part"].items():
        tf = st["tracking_fraction"]
        tf_mean = f"{tf['mean']:.3f}" if tf['mean'] is not None else "—"
        tf_pct = f"{100*tf['rate_below_0.5']:.1f}%" if tf['rate_below_0.5'] is not None else "—"
        L.append(
            f"| {part} | {st['n_segments']} | "
            f"{st['drift_max_cm']['mean']:.2f} | {st['drift_max_cm']['p95']:.2f} | "
            f"{st['drift_end_cm']['mean']:+.2f} | {tf_mean} | {tf_pct} |"
        )
    L.append("")
    L.append("## Per-subset")
    L.append("")
    L.append("| subset | n | drift_max mean | drift_end mean | track frac mean |")
    L.append("|---|---:|---:|---:|---:|")
    for sub, st in stats["per_subset"].items():
        tf = st["tracking_fraction"]
        tf_mean = f"{tf['mean']:.3f}" if tf['mean'] is not None else "—"
        L.append(
            f"| {sub} | {st['n_segments']} | "
            f"{st['drift_max_cm']['mean']:.2f} | {st['drift_end_cm']['mean']:+.2f} | {tf_mean} |"
        )
    L.append("")
    L.append("## Interpretation key")
    L.append("")
    L.append("- `drift_max > 5cm` = at some point in the contact window, pred was 5cm farther from the contact target than GT was. High rate suggests sustained-contact failure.")
    L.append("- `drift_end > 0` = pred drifted away by the end of the segment (didn't end with hand on object).")
    L.append("- `tracking_fraction = pred_align / gt_align` along object displacement direction. 1.0 = pred matches GT's tracking of object motion; 0.5 = pred only went halfway with the object; 0 = pred didn't move with object at all. The user's 'hand only goes partway up with the box' failure → tracking_fraction << 1.")
    L.append("- `rel_var_ratio` = variance of (pred_part - obj_pos) / variance of (gt_part - obj_pos). >>1 = pred's part swings around relative to object (loose grip).")
    L.append("")
    out_path.write_text("\n".join(L), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path, required=True,
                        help="Same 48-clip subset used for D2/D3/anchor-diag.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bucket", default="val", choices=["train", "val"])
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-segment-len", type=int, default=MIN_SEGMENT_LEN_DEFAULT)
    parser.add_argument("--use-gt-as-pred", action="store_true",
                        help="Sanity mode: use GT motion as the 'prediction' so all drift "
                        "should be ~0 and tracking_fraction~1.0. Validates the script.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Selection
    # Accept three legacy schemas: eval_selection uses `selected`,
    # older diagnostic files used `candidates`, and the train_indices
    # builders emit the same {subset, seq_id} pairs under `clips`.
    sel_obj = json.loads(args.selection_json.read_text("utf-8"))
    selection = (
        sel_obj.get("selected")
        or sel_obj.get("candidates")
        or sel_obj.get("clips")
        or []
    )
    if not selection:
        raise SystemExit(f"empty selection: {args.selection_json}")
    sel_pairs = {(e["subset"], e["seq_id"]) for e in selection}
    print(f"[sustained] selection: {len(sel_pairs)} clips")

    # --- Dataset
    dataset = _build_dataset(cfg, args.bucket, augment=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        collate_fn=collate_hoi, num_workers=0)

    # --- Model
    model, object_encoder, z_dims = _build_model(cfg, device)
    train_meta: dict[str, Any] = {}
    if not args.use_gt_as_pred:
        state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        train_meta = extract_train_time_meta(state)
        model_state = state.get("model", state)
        model.load_state_dict(model_state)
        if "object_encoder" in state:
            object_encoder.load_state_dict(state["object_encoder"])
        elif "extra_modules" in state and "object_encoder" in state["extra_modules"]:
            object_encoder.load_state_dict(state["extra_modules"]["object_encoder"])
    # Skip CLIP load when model was trained with text_dim=0 (Tier-2 ablation).
    if int(cfg.model.denoiser.get("text_dim", 0)) > 0:
        clip_model = load_clip_text_encoder(
            device=device,
            model_name=str(cfg.model.text_encoder.clip_version),
            download_root=str(cfg.model.text_encoder.get("download_root", "cache/clip")),
        )
    else:
        clip_model = None
    stage1_norm = _stage1_norm_for_cfg(cfg, device)
    model.eval()

    rows: list[dict] = []
    n_processed = 0
    for batch in loader:
        subset = str(batch["subset"][0])
        seq_id = str(batch["seq_id"][0])
        if (subset, seq_id) not in sel_pairs:
            continue
        n_processed += 1

        cond, T = _build_cond(
            batch, model, object_encoder, clip_model, z_dims, cfg, device,
            stage1_norm=stage1_norm,
        )

        gt_motion = batch["motion"][:, :T].to(device).float()
        if args.use_gt_as_pred:
            pred_motion = gt_motion
        else:
            torch.manual_seed(args.seed)
            with torch.no_grad():
                pred_motion = model.sample(
                    cond=cond, seq_length=T, cfg_scale=args.cfg_scale,
                    replacement="none", output_skip=False,
                )

        rest_offsets = batch["rest_offsets"].to(device).float()
        pred_joints = _fk_22joints(pred_motion, rest_offsets)[0].cpu().numpy()  # (T,22,3)
        gt_joints = _fk_22joints(gt_motion, rest_offsets)[0].cpu().numpy()

        seq_len = int(batch["seq_len"][0].item())
        valid_T = min(T, seq_len)
        seq_mask = np.zeros(T, dtype=bool); seq_mask[:valid_T] = True

        contact_state = batch["contact_state"][0, :T].cpu().numpy()             # (T, 5)
        contact_target_local = batch["contact_target_xyz"][0, :T].cpu().numpy() # (T, 5, 3)
        obj_pos_world = batch["object_positions"][0, :T].cpu().numpy()          # (T, 3)
        obj_rot_world = batch["object_rotations"][0, :T].cpu().numpy()          # (T, 3) axis-angle

        n_segs_this_clip = 0
        for p_idx in range(5):
            joint_idx = BODY_PART_INDICES[p_idx]
            segs = find_contact_segments(contact_state[:, p_idx], seq_mask, args.min_segment_len)
            for (t0, t1) in segs:
                # Use object pose at first frame of segment for the target_local
                # (object surface contact point in object-local frame is roughly
                # constant within a single segment; the dataset already provides
                # per-frame contact_target_xyz, so take its mean within the
                # segment to be robust to per-frame noise).
                target_local_seg = contact_target_local[t0:t1 + 1, p_idx, :].mean(axis=0)
                pred_part = pred_joints[t0:t1 + 1, joint_idx, :]
                gt_part = gt_joints[t0:t1 + 1, joint_idx, :]
                obj_pos_seg = obj_pos_world[t0:t1 + 1, :]
                obj_rot_seg = obj_rot_world[t0:t1 + 1, :]
                m = sustained_metrics_for_segment(
                    pred_part_xyz=pred_part, gt_part_xyz=gt_part,
                    obj_pos_world=obj_pos_seg, obj_rot_aa=obj_rot_seg,
                    contact_target_local=target_local_seg,
                )
                m.update({
                    "subset": subset, "seq_id": seq_id,
                    "part_idx": p_idx, "part_name": BODY_PART_NAMES[p_idx],
                    "t0": t0, "t1": t1,
                })
                rows.append(m)
                n_segs_this_clip += 1
        if n_processed % 4 == 0:
            print(f"  [sustained {n_processed}/{len(sel_pairs)}] {subset}/{seq_id}  "
                  f"+{n_segs_this_clip} segments (total {len(rows)})")

    if not rows:
        print("[sustained] no contact segments collected.")
        return 1

    stats = aggregate(rows)
    out_json = args.output_dir / "sustained_contact_stats.json"
    out_md = args.output_dir / "sustained_contact_summary.md"
    out_json.write_text(json.dumps({
        "config": str(args.config),
        "ckpt": str(args.ckpt),
        "use_gt_as_pred": args.use_gt_as_pred,
        "train_time": train_meta,
        "n_clips_processed": n_processed,
        "n_segments": len(rows),
        "min_segment_len": args.min_segment_len,
        "obj_motion_min_cm": OBJ_MOTION_MIN_CM,
        "overall": stats["overall"],
        "per_part": stats["per_part"],
        "per_subset": stats["per_subset"],
        "rows": rows,
    }, indent=2), "utf-8")
    print(f"wrote {out_json}")
    if train_meta.get("train_wallclock_hms"):
        print(f"  train wallclock: {train_meta['train_wallclock_hms']} "
              f"({train_meta['train_wallclock_seconds']:.1f}s)")
    write_summary_md(stats, args.ckpt, out_md, args.use_gt_as_pred)
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
