"""Pseudo-label / plan / object-geometry consistency audit (round 5, Diag 2).

Pure CPU/GPU-light analysis on GT data + interaction plan + object
transforms. No denoiser inference. Verifies whether the supervision
the model is being trained on is internally consistent before any
model-side change.

Subdiagnostics:
  2A. Contact event validity (per clip, per hand)
  2B. Hand-object distance consistency (using GT joints + object COM
      + plan anchor target_world)
  2C. Plan anchor consistency (anchor_target_world vs GT hand vs
      object surface)
  2D. Object transform consistency (transform object_pc by world
      pose; centroid vs object_positions; rotation continuity)

Outputs:
  analyses/2026-05-15_pseudolabel_plan_object_geometry_audit.{json,md}
  analyses/visuals/2026-05-15_pseudolabel_geometry/*.png
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

from diagnostic_common import (
    clip_metadata,
    event_records_from_contact,
    extract_plan,
    format_md_table,
    make_seq_mask,
    merge_single_batches,
)
from dynamics_diagnostic import _build_dataset, _balanced_subset_indices
from piano.data.dataset import collate_hoi
from recon_ladder_truncated_rollout_diagnostic import (
    _build_selected_batches,
    _load_selection,
)


HAND_SPECS = (("L_hand", 20, 0), ("R_hand", 21, 1))


def _axis_angle_to_rot(aa: np.ndarray) -> np.ndarray:
    """Rodrigues axis-angle (..., 3) → rotation matrix (..., 3, 3)."""
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


def _event_validity_for_clip(
    contact_state: np.ndarray, seq_len: int, *, threshold: float = 0.5,
    edge_margin: int = 5,
) -> dict[str, Any]:
    """Per-clip event-validity flags."""
    flags: dict[str, Any] = {"per_part": {}, "n_total": 0, "n_boundary": 0, "n_flicker": 0, "n_valid": 0}
    for part, _joint, idx in HAND_SPECS:
        c = contact_state[:seq_len, idx] > float(threshold)
        # transitions
        onset = (c[1:] & ~c[:-1])
        release = (~c[1:] & c[:-1])
        onset_idx = (np.where(onset)[0] + 1).tolist()
        release_idx = (np.where(release)[0] + 1).tolist()
        # segments: pair onset → next release
        segments: list[tuple[int, int]] = []
        ri = 0
        for o in onset_idx:
            while ri < len(release_idx) and release_idx[ri] <= o:
                ri += 1
            if ri < len(release_idx):
                segments.append((o, release_idx[ri]))
                ri += 1
            else:
                segments.append((o, seq_len - 1))
        n_total = len(onset_idx) + len(release_idx)
        n_boundary = sum(1 for o in onset_idx if o < int(edge_margin)) \
            + sum(1 for r in release_idx if r > seq_len - 1 - int(edge_margin))
        n_flicker = sum(1 for (s, e) in segments if (e - s) <= 2)
        durations = [e - s for (s, e) in segments]
        valid_events = max(0, n_total - n_boundary - 2 * n_flicker)
        flags["per_part"][part] = {
            "onset_frames": onset_idx,
            "release_frames": release_idx,
            "n_onset": len(onset_idx),
            "n_release": len(release_idx),
            "n_boundary": n_boundary,
            "n_flicker": n_flicker,
            "segments": [{"start": int(s), "end": int(e), "duration": int(e - s)} for (s, e) in segments],
            "segment_duration_min": int(min(durations)) if durations else 0,
            "segment_duration_mean": float(np.mean(durations)) if durations else 0.0,
            "segment_duration_max": int(max(durations)) if durations else 0,
        }
        flags["n_total"] += n_total
        flags["n_boundary"] += n_boundary
        flags["n_flicker"] += n_flicker
        flags["n_valid"] += valid_events
    return flags


def _hand_obj_distance_curve(
    gt_joints: np.ndarray, obj_positions: np.ndarray, seq_len: int, joint: int,
) -> np.ndarray:
    """Per-frame hand-to-objectCOM distance in cm."""
    h = gt_joints[:seq_len, joint]
    o = obj_positions[:seq_len]
    return np.linalg.norm(h - o, axis=-1) * 100.0


def _hand_pseudo_target_distance(
    gt_joints: np.ndarray, target_world: np.ndarray, seq_len: int, joint: int, part_idx: int,
) -> np.ndarray:
    """Per-frame distance from GT hand to pseudo target_world (cm)."""
    h = gt_joints[:seq_len, joint]
    # target_world is (T, P*3); reshape and slice the part
    tw = target_world[:seq_len].reshape(seq_len, -1, 3)[:, part_idx, :]
    return np.linalg.norm(h - tw, axis=-1) * 100.0


def _audit_clip(
    batch: dict[str, Any], b: int,
    *, threshold: float, edge_margin: int,
) -> dict[str, Any]:
    seq_len = int(batch["seq_len"][b].item())
    gt_joints = batch["joints"][b].detach().cpu().numpy().astype(np.float32)
    contact_state = batch["contact_state"][b].detach().cpu().numpy().astype(np.float32)
    obj_positions = batch["object_positions"][b].detach().cpu().numpy().astype(np.float32)
    obj_rotations = batch["object_rotations"][b].detach().cpu().numpy().astype(np.float32)
    obj_pc = batch["object_pc"][b].detach().cpu().numpy().astype(np.float32)
    # contact_target_xyz is in object-local; target_world lifted = R(t) @ local + t(t)
    contact_target_local = batch["contact_target_xyz"][b].detach().cpu().numpy().astype(np.float32)  # (T, 5, 3)
    # plan
    plan_keys = ["anchor_time", "anchor_part", "anchor_target_local", "anchor_target_world", "anchor_type", "anchor_mask", "anchor_conf"]
    plan = {k: batch[f"plan_{k}"][b].detach().cpu().numpy() for k in plan_keys}
    # text / metadata
    seq_id = str(batch["seq_id"][b])
    subset = str(batch["subset"][b])
    text = str(batch["text"][b])

    # ---- 2A event validity ----
    events = _event_validity_for_clip(
        contact_state, seq_len, threshold=threshold, edge_margin=edge_margin,
    )

    # ---- 2B hand-object distance consistency ----
    # lift pseudo target_world from object pose
    R = _axis_angle_to_rot(obj_rotations[:seq_len])      # (T, 3, 3)
    t = obj_positions[:seq_len]                          # (T, 3)
    # target_world(T, 5, 3) = (R @ local^T)^T + t
    target_world_lifted = (
        np.einsum("tij,tkj->tki", R, contact_target_local[:seq_len])
        + t[:, None, :]
    )

    per_hand_curves: dict[str, dict[str, list[float]]] = {}
    for part, joint, part_idx in HAND_SPECS:
        d_obj = _hand_obj_distance_curve(gt_joints, obj_positions, seq_len, joint).tolist()
        d_pt = np.linalg.norm(gt_joints[:seq_len, joint] - target_world_lifted[:, part_idx, :], axis=-1) * 100.0
        # at contact frames, pseudo target should be near hand and near object surface
        cs = contact_state[:seq_len, part_idx] > float(threshold)
        if cs.any():
            d_pt_contact_mean = float(d_pt[cs].mean())
            d_obj_contact_mean = float(np.array(d_obj)[cs].mean())
        else:
            d_pt_contact_mean = 0.0
            d_obj_contact_mean = 0.0
        per_hand_curves[part] = {
            "distance_hand_to_objectCOM_cm": d_obj,
            "distance_hand_to_pseudo_target_cm": d_pt.tolist(),
            "n_contact_frames": int(cs.sum()),
            "mean_hand_to_pseudo_target_in_contact_cm": d_pt_contact_mean,
            "mean_hand_to_objectCOM_in_contact_cm": d_obj_contact_mean,
        }

    # ---- 2C plan anchor consistency (round 7: per-type separated) ----
    anchor_mask = plan["anchor_mask"].astype(bool)        # (K,)
    K = anchor_mask.shape[0]
    anchor_time = plan["anchor_time"].astype(int)         # (K,)
    anchor_part = plan["anchor_part"].astype(np.float32)  # (K, P)
    anchor_target_world = plan["anchor_target_world"].astype(np.float32)  # (K, P, 3)
    anchor_target_local = plan["anchor_target_local"].astype(np.float32)  # (K, P, 3)
    # plan_anchor_type may not be in the audit batch dict — fall back to "stable"
    anchor_type_arr = batch.get("plan_anchor_type", None)
    if anchor_type_arr is not None:
        anchor_type_arr = anchor_type_arr[b].detach().cpu().numpy().astype(int)
    else:
        anchor_type_arr = np.full(K, 1, dtype=int)  # ANCHOR_TYPE_STABLE default
    type_names = {0: "onset", 1: "stable", 2: "release", 3: "phase_change", 4: "support_change"}

    part_joint = [20, 21, 10, 11, 0]
    anchor_rows: list[dict[str, Any]] = []
    # Per-type buckets
    type_target_err: dict[str, list[float]] = {n: [] for n in type_names.values()}
    type_time_err: dict[str, list[float]] = {n: [] for n in type_names.values()}
    # Pooled (kept for backward compat but flagged as artifact)
    target_to_hand_errors: list[float] = []
    target_to_obj_centroid_errors: list[float] = []
    # Per-part onset/release frames for type-correct time-error comparison
    onset_release_by_part: dict[int, dict[str, list[int]]] = {}
    for part, _j, idx in HAND_SPECS:
        onset_release_by_part[idx] = {
            "onset": list(events["per_part"][part]["onset_frames"]),
            "release": list(events["per_part"][part]["release_frames"]),
        }
    # Segment representative frames for STABLE anchor comparison
    segment_mid_by_part: dict[int, list[int]] = {}
    for part, _j, idx in HAND_SPECS:
        segments_part = events["per_part"][part]["segments"]
        segment_mid_by_part[idx] = [int((s["start"] + s["end"]) // 2) for s in segments_part]

    for k in range(K):
        if not bool(anchor_mask[k]):
            continue
        t_a = int(min(max(0, anchor_time[k]), seq_len - 1))
        t_id = int(anchor_type_arr[k])
        t_name = type_names.get(t_id, "stable")
        # for each active part on this anchor
        active = np.where(anchor_part[k] > 0)[0]
        for p in active:
            joint = int(part_joint[int(p)])
            hand_gt = gt_joints[t_a, joint]
            target_w = anchor_target_world[k, int(p)]
            err = float(np.linalg.norm(hand_gt - target_w) * 100.0)
            err_centroid = float(np.linalg.norm(obj_positions[t_a] - target_w) * 100.0)
            target_to_hand_errors.append(err)
            target_to_obj_centroid_errors.append(err_centroid)
            type_target_err.setdefault(t_name, []).append(err)
            anchor_rows.append({
                "anchor_idx": int(k),
                "frame": int(t_a),
                "part": int(p),
                "type": t_name,
                "target_to_GT_hand_cm": err,
                "target_to_objectCOM_cm": err_centroid,
            })
        # Per-type time error reference
        if int(p) <= 1:
            part_ev = onset_release_by_part.get(int(p), {"onset": [], "release": []})
            if t_name == "onset" and part_ev["onset"]:
                time_err = min(abs(t_a - f) for f in part_ev["onset"])
            elif t_name == "release" and part_ev["release"]:
                time_err = min(abs(t_a - f) for f in part_ev["release"])
            elif t_name == "stable" and segment_mid_by_part.get(int(p)):
                time_err = min(abs(t_a - f) for f in segment_mid_by_part[int(p)])
            elif part_ev["onset"] or part_ev["release"]:
                all_ev = part_ev["onset"] + part_ev["release"]
                time_err = min(abs(t_a - f) for f in all_ev)
            else:
                time_err = -1
            if time_err >= 0:
                type_time_err.setdefault(t_name, []).append(float(time_err))

    pct_over = lambda arr, thr: float(np.mean([e > thr for e in arr]) * 100.0) if arr else 0.0
    summary_2c = {
        "n_active_anchors": int(anchor_mask.sum()),
        "n_active_anchor_parts": len(anchor_rows),
        # NOTE: pooled fields below mix anchor types and are an aggregation artifact;
        # see per-type breakdown below (Round-7 mandatory protocol).
        "anchor_target_to_GT_hand_cm_mean": float(np.mean(target_to_hand_errors)) if target_to_hand_errors else 0.0,
        "anchor_target_to_GT_hand_cm_median": float(np.median(target_to_hand_errors)) if target_to_hand_errors else 0.0,
        "anchor_target_to_GT_hand_cm_p95": float(np.percentile(target_to_hand_errors, 95)) if target_to_hand_errors else 0.0,
        "pct_target_to_hand_over_10cm": pct_over(target_to_hand_errors, 10.0),
        "pct_target_to_hand_over_20cm": pct_over(target_to_hand_errors, 20.0),
        "pct_target_to_hand_over_40cm": pct_over(target_to_hand_errors, 40.0),
        "anchor_target_to_objectCOM_cm_mean": float(np.mean(target_to_obj_centroid_errors)) if target_to_obj_centroid_errors else 0.0,
        "anchor_time_error_to_nearest_event_mean": float(np.mean(
            [v for vs in type_time_err.values() for v in vs]
        )) if any(type_time_err.values()) else 0.0,
        "per_type": {
            t_name: {
                "n_anchor_parts": len(type_target_err[t_name]),
                "target_to_GT_hand_cm_mean": float(np.mean(type_target_err[t_name])) if type_target_err[t_name] else 0.0,
                "target_to_GT_hand_cm_median": float(np.median(type_target_err[t_name])) if type_target_err[t_name] else 0.0,
                "target_to_GT_hand_cm_p95": float(np.percentile(type_target_err[t_name], 95)) if type_target_err[t_name] else 0.0,
                "time_err_frames_mean": float(np.mean(type_time_err[t_name])) if type_time_err[t_name] else 0.0,
                "time_err_frames_max": float(np.max(type_time_err[t_name])) if type_time_err[t_name] else 0.0,
            }
            for t_name in type_names.values()
        },
        "anchor_rows": anchor_rows,
    }

    # ---- 2D object transform consistency ----
    # transform object_pc by per-frame world pose and compare centroid vs object_positions
    # object_pc is (N, 3) in canonical frame
    pc = obj_pc.astype(np.float32)
    obj_pc_world = np.einsum("tij,nj->tni", R, pc) + obj_positions[:seq_len, None, :]
    centroid = obj_pc_world.mean(axis=1)
    centroid_err = np.linalg.norm(centroid - obj_positions[:seq_len], axis=-1) * 100.0
    # rotation continuity: frame-to-frame trace of R_t R_{t-1}^T
    R_now = R[1:]; R_prev = R[:-1]
    rel = np.einsum("tij,tkj->tik", R_now, R_prev)
    cos = (np.trace(rel, axis1=-2, axis2=-1) - 1.0) / 2.0
    cos = np.clip(cos, -1.0, 1.0)
    rot_step_deg = np.degrees(np.arccos(cos))
    pos_step_cm = np.linalg.norm(obj_positions[1:seq_len] - obj_positions[:seq_len-1], axis=-1) * 100.0
    summary_2d = {
        "centroid_offset_to_objectCOM_cm_mean": float(centroid_err.mean()),
        "centroid_offset_to_objectCOM_cm_p95": float(np.percentile(centroid_err, 95)),
        "rotation_step_deg_mean": float(rot_step_deg.mean()),
        "rotation_step_deg_p95": float(np.percentile(rot_step_deg, 95)),
        "rotation_step_deg_max": float(rot_step_deg.max()),
        "object_step_cm_mean": float(pos_step_cm.mean()),
        "object_step_cm_p95": float(np.percentile(pos_step_cm, 95)),
        "object_step_cm_max": float(pos_step_cm.max()),
        "n_rot_jumps_over_45deg": int((rot_step_deg > 45.0).sum()),
        "n_pos_jumps_over_30cm": int((pos_step_cm > 30.0).sum()),
    }

    return {
        "subset": subset,
        "seq_id": seq_id,
        "text": text[:120],
        "seq_len": seq_len,
        "events": events,
        "hand_curves": per_hand_curves,
        "plan_anchor_consistency": summary_2c,
        "object_transform_consistency": summary_2d,
    }


def _plot_clip(
    clip: dict[str, Any], out_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    seq_len = int(clip["seq_len"])
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    for part in ("L_hand", "R_hand"):
        d_obj = clip["hand_curves"][part]["distance_hand_to_objectCOM_cm"]
        d_pt = clip["hand_curves"][part]["distance_hand_to_pseudo_target_cm"]
        axes[0].plot(d_obj, label=f"{part} → objCOM")
        axes[1].plot(d_pt, label=f"{part} → pseudoTarget")
        for f in clip["events"]["per_part"][part]["onset_frames"]:
            axes[0].axvline(f, c="g", ls="--", alpha=0.4)
            axes[1].axvline(f, c="g", ls="--", alpha=0.4)
        for f in clip["events"]["per_part"][part]["release_frames"]:
            axes[0].axvline(f, c="r", ls="--", alpha=0.4)
            axes[1].axvline(f, c="r", ls="--", alpha=0.4)
    for anc in clip["plan_anchor_consistency"]["anchor_rows"]:
        axes[1].axvline(anc["frame"], c="k", ls=":", alpha=0.3)
    axes[0].set_ylabel("hand → objCOM (cm)")
    axes[0].legend(fontsize=7)
    axes[0].set_title(f"{clip['subset']}/{clip['seq_id']}  T={seq_len}\n{clip['text']}", fontsize=8)
    axes[1].set_ylabel("hand → pseudoTarget (cm)")
    axes[1].set_xlabel("frame")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/training/anchordiff_v18_a1_FULL_DATA.yaml"))
    parser.add_argument("--selection-json", type=Path, default=Path("analyses/2026-05-13_v18_recon_ladder_truncated_rollout.json"))
    parser.add_argument("--output", type=Path, default=Path("analyses/2026-05-15_pseudolabel_plan_object_geometry_audit.json"))
    parser.add_argument("--md", type=Path, default=Path("analyses/2026-05-15_pseudolabel_plan_object_geometry_audit.md"))
    parser.add_argument("--visuals-dir", type=Path, default=Path("analyses/visuals/2026-05-15_pseudolabel_geometry"))
    parser.add_argument("--max-clips", type=int, default=16)
    parser.add_argument("--num-candidates", type=int, default=256)
    parser.add_argument("--bucket", choices=["train", "val"], default="train")
    parser.add_argument("--balanced-subsets", action="store_true", default=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--edge-margin", type=int, default=5)
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

    audits: list[dict[str, Any]] = []
    for b in range(B):
        a = _audit_clip(batch, b, threshold=float(args.threshold), edge_margin=int(args.edge_margin))
        audits.append(a)
        out_png = args.visuals_dir / f"{a['subset']}_{a['seq_id']}.png"
        _plot_clip(a, out_png)

    # aggregate stats
    def _flatten(key_path: list[str]) -> list[float]:
        out: list[float] = []
        for a in audits:
            d = a
            for k in key_path:
                d = d.get(k, {})
            if isinstance(d, (int, float)):
                out.append(float(d))
        return out

    agg = {
        "n_clips": B,
        "subset_counts": {},
        "events": {
            "n_total_events": sum(a["events"]["n_total"] for a in audits),
            "n_boundary_events": sum(a["events"]["n_boundary"] for a in audits),
            "n_flicker_events": sum(a["events"]["n_flicker"] for a in audits),
            "n_valid_events": sum(a["events"]["n_valid"] for a in audits),
        },
        "anchor_target_to_GT_hand_cm_mean_mean": float(np.mean(_flatten(["plan_anchor_consistency", "anchor_target_to_GT_hand_cm_mean"]))) if audits else 0.0,
        "anchor_target_to_GT_hand_cm_p95_mean": float(np.mean(_flatten(["plan_anchor_consistency", "anchor_target_to_GT_hand_cm_p95"]))) if audits else 0.0,
        "pct_target_to_hand_over_20cm_mean": float(np.mean(_flatten(["plan_anchor_consistency", "pct_target_to_hand_over_20cm"]))) if audits else 0.0,
        "pct_target_to_hand_over_40cm_mean": float(np.mean(_flatten(["plan_anchor_consistency", "pct_target_to_hand_over_40cm"]))) if audits else 0.0,
        "anchor_time_error_to_nearest_event_mean_mean": float(np.mean(_flatten(["plan_anchor_consistency", "anchor_time_error_to_nearest_event_mean"]))) if audits else 0.0,
        "centroid_offset_to_objectCOM_cm_mean_mean": float(np.mean(_flatten(["object_transform_consistency", "centroid_offset_to_objectCOM_cm_mean"]))) if audits else 0.0,
        "rotation_step_deg_p95_mean": float(np.mean(_flatten(["object_transform_consistency", "rotation_step_deg_p95"]))) if audits else 0.0,
        "n_rot_jumps_over_45deg_total": sum(a["object_transform_consistency"]["n_rot_jumps_over_45deg"] for a in audits),
        "n_pos_jumps_over_30cm_total": sum(a["object_transform_consistency"]["n_pos_jumps_over_30cm"] for a in audits),
        "mean_hand_to_pseudo_target_in_contact_cm_mean": float(np.mean([
            a["hand_curves"][part]["mean_hand_to_pseudo_target_in_contact_cm"]
            for a in audits for part in ("L_hand", "R_hand")
            if a["hand_curves"][part]["n_contact_frames"] > 0
        ])) if audits else 0.0,
        "mean_hand_to_objectCOM_in_contact_cm_mean": float(np.mean([
            a["hand_curves"][part]["mean_hand_to_objectCOM_in_contact_cm"]
            for a in audits for part in ("L_hand", "R_hand")
            if a["hand_curves"][part]["n_contact_frames"] > 0
        ])) if audits else 0.0,
    }
    for a in audits:
        agg["subset_counts"][a["subset"]] = agg["subset_counts"].get(a["subset"], 0) + 1

    payload = {
        "config": str(args.config),
        "n_clips": B,
        "aggregate": agg,
        "clips": audits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")

    # Markdown report
    lines: list[str] = [
        "# Pseudo-label / Plan / Object-Geometry Consistency Audit",
        "",
        f"- Config: `{args.config}`",
        f"- Clips: {B}",
        f"- Subset composition: {agg['subset_counts']}",
        "",
        "## 2A Event validity (aggregate across clips)",
        "",
        f"- Total events: {agg['events']['n_total_events']}",
        f"- Boundary events (within {args.edge_margin} frames of clip edge): {agg['events']['n_boundary_events']}",
        f"- Flicker events (segment length ≤ 2 frames): {agg['events']['n_flicker_events']}",
        f"- Valid events: {agg['events']['n_valid_events']}",
        "",
        "## 2B Hand-object distance consistency",
        "",
        f"- Mean hand→pseudo_target distance during contact: **{agg['mean_hand_to_pseudo_target_in_contact_cm_mean']:.2f} cm**",
        f"- Mean hand→objectCOM distance during contact: **{agg['mean_hand_to_objectCOM_in_contact_cm_mean']:.2f} cm**",
        "",
        "(Lower hand→target during contact → pseudo target lands near the hand. Compare hand→target vs hand→objectCOM to see if target is on object surface or far away.)",
        "",
        "## 2C Plan anchor consistency",
        "",
        f"- Mean anchor target_world → GT hand error: **{agg['anchor_target_to_GT_hand_cm_mean_mean']:.2f} cm**",
        f"- P95 anchor target_world → GT hand error: **{agg['anchor_target_to_GT_hand_cm_p95_mean']:.2f} cm**",
        f"- % anchors with target→hand > 20 cm: **{agg['pct_target_to_hand_over_20cm_mean']:.1f} %**",
        f"- % anchors with target→hand > 40 cm: **{agg['pct_target_to_hand_over_40cm_mean']:.1f} %**",
        f"- Anchor-time error to nearest GT event (mean): **{agg['anchor_time_error_to_nearest_event_mean_mean']:.2f} frames**",
        "",
        "## 2D Object transform consistency",
        "",
        f"- Mean object_pc centroid offset to obj_positions COM: **{agg['centroid_offset_to_objectCOM_cm_mean_mean']:.3f} cm**",
        f"- P95 frame-to-frame rotation step: **{agg['rotation_step_deg_p95_mean']:.2f}°**",
        f"- Rotation jumps > 45° (total): {agg['n_rot_jumps_over_45deg_total']}",
        f"- Position jumps > 30 cm (total): {agg['n_pos_jumps_over_30cm_total']}",
        "",
        "## Per-clip summary",
        "",
    ]
    rows = [["subset", "seq_id", "T", "n_total_ev", "n_boundary", "n_flicker", "anchor→hand mean cm", "p95 cm", ">20cm %", "rot p95 °", "centroid err cm"]]
    for a in audits:
        rows.append([
            a["subset"], a["seq_id"], a["seq_len"],
            a["events"]["n_total"], a["events"]["n_boundary"], a["events"]["n_flicker"],
            f"{a['plan_anchor_consistency']['anchor_target_to_GT_hand_cm_mean']:.2f}",
            f"{a['plan_anchor_consistency']['anchor_target_to_GT_hand_cm_p95']:.2f}",
            f"{a['plan_anchor_consistency']['pct_target_to_hand_over_20cm']:.0f}",
            f"{a['object_transform_consistency']['rotation_step_deg_p95']:.2f}",
            f"{a['object_transform_consistency']['centroid_offset_to_objectCOM_cm_mean']:.3f}",
        ])
    lines.append(format_md_table(rows))
    lines.append("")
    lines.append("## Visualization files")
    lines.append("")
    for a in audits:
        png = args.visuals_dir / f"{a['subset']}_{a['seq_id']}.png"
        lines.append(f"- `{png.as_posix()}`")
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.md}")


if __name__ == "__main__":
    main()
