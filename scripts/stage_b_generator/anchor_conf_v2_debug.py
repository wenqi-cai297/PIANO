"""Anchor-confidence v2 debug side-channel (round 7).

Reads the existing plan compiler output for each clip, computes a
candidate ``anchor_conf_v2`` field per anchor as a debug side-channel,
and measures whether v2 correlates better with target-to-hand error
than the current ``anchor_conf``.

This script does NOT modify the compiler or the model. It is a pure
audit-side experiment on the same 16-clip Round-5 selection.

Formula (geometric mean):

  anchor_conf_v2 = exp((1/5) * sum(log(eps + c_i)))

where the five factors are:

  1. contact_score       = anchor_conf (current value at anchor time)
  2. surface_score       = 1 - clip(target_to_surface_cm / 10cm, 0, 1)
  3. duration_score      = min(1, segment_duration / 8)
  4. stability_score     = 1 / (1 + std_target_local_cm_in_segment)
  5. boundary_flicker    = 0.5 if (is_boundary or is_flicker) else 1.0

Outputs:
  analyses/2026-05-17_anchor_conf_v2_debug.json
  analyses/2026-05-17_anchor_conf_v2_debug.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from diagnostic_common import format_md_table, merge_single_batches, stats_list
from piano.data.interaction_plan_compiler import (
    ANCHOR_TYPE_ONSET, ANCHOR_TYPE_STABLE, ANCHOR_TYPE_RELEASE,
    ANCHOR_TYPE_PHASE_CHANGE, ANCHOR_TYPE_SUPPORT_CHANGE,
    InteractionPlanCompilerConfig,
    _build_change_anchors, _build_contact_anchors, _entropy,
    merge_nearby_anchors, smooth_categorical_softmax, smooth_contact,
    smooth_target_local, temporal_nms_budget,
    lift_target_local_to_world_np,
)
from recon_ladder_truncated_rollout_diagnostic import (
    _build_selected_batches,
    _load_selection,
)


ANCHOR_TYPE_NAMES = {
    ANCHOR_TYPE_ONSET: "onset",
    ANCHOR_TYPE_STABLE: "stable",
    ANCHOR_TYPE_RELEASE: "release",
    ANCHOR_TYPE_PHASE_CHANGE: "phase_change",
    ANCHOR_TYPE_SUPPORT_CHANGE: "support_change",
}

HAND_PARTS = (("L_hand", 0, 20), ("R_hand", 1, 21))
PART_JOINT = (20, 21, 10, 11, 0)


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


def _per_anchor_conf_v2(
    anchor_time: int,
    anchor_part: int,
    contact_conf: float,
    target_to_surface_cm: float,
    segment_start: int,
    segment_end: int,
    contact_state: np.ndarray,
    target_local: np.ndarray,
    seq_len: int,
    edge_margin: int,
    flicker_max_frames: int,
    eps: float = 1e-3,
) -> tuple[float, dict[str, float]]:
    """Compute anchor_conf_v2 + the five contributing factors."""
    duration = max(1, segment_end - segment_start)
    duration_score = float(min(1.0, duration / 8.0))
    # surface_score: target on surface => high
    surface_score = float(np.clip(1.0 - target_to_surface_cm / 10.0, 0.0, 1.0))
    # stability_score: low intra-segment target std => high
    seg_lo = max(0, int(segment_start))
    seg_hi = min(seq_len, int(segment_end) + 1)
    if seg_hi > seg_lo + 1:
        seg_targets = target_local[seg_lo:seg_hi, anchor_part, :]  # (L, 3)
        std_cm = float(seg_targets.std(axis=0).mean() * 100.0)
    else:
        std_cm = 0.0
    stability_score = float(1.0 / (1.0 + std_cm))
    is_boundary = bool(
        anchor_time < int(edge_margin)
        or anchor_time > seq_len - 1 - int(edge_margin)
    )
    is_flicker = bool(duration <= int(flicker_max_frames))
    boundary_flicker = 0.5 if (is_boundary or is_flicker) else 1.0
    factors = {
        "contact_score": float(contact_conf),
        "surface_score": surface_score,
        "duration_score": duration_score,
        "stability_score": stability_score,
        "boundary_flicker_score": float(boundary_flicker),
    }
    # geometric mean
    vals = np.asarray([
        max(eps, factors[k]) for k in (
            "contact_score", "surface_score", "duration_score",
            "stability_score", "boundary_flicker_score",
        )
    ], dtype=np.float64)
    conf_v2 = float(np.exp(np.log(vals).mean()))
    return conf_v2, factors


def _audit_clip(
    batch: dict[str, Any], b: int,
    *, threshold: float, edge_margin: int, flicker_max_frames: int,
    surface_samples: int,
) -> dict[str, Any]:
    seq_len = int(batch["seq_len"][b].item())
    subset = str(batch["subset"][b])
    seq_id = str(batch["seq_id"][b])
    gt_joints = batch["joints"][b].detach().cpu().numpy().astype(np.float32)
    contact_state = batch["contact_state"][b].detach().cpu().numpy().astype(np.float32)
    target_local = batch["contact_target_xyz"][b].detach().cpu().numpy().astype(np.float32)
    obj_positions = batch["object_positions"][b].detach().cpu().numpy().astype(np.float32)
    obj_rotations = batch["object_rotations"][b].detach().cpu().numpy().astype(np.float32)
    obj_pc = batch["object_pc"][b].detach().cpu().numpy().astype(np.float32)
    phase = batch["phase"][b].detach().cpu().numpy()
    support = batch["support"][b].detach().cpu().numpy()

    # Replay compiler
    cfg = InteractionPlanCompilerConfig(num_parts=int(contact_state.shape[1]))
    contact_prob = contact_state[:seq_len].astype(np.float32)
    target_loc = target_local[:seq_len].astype(np.float32)
    obj_pos_seq = obj_positions[:seq_len]
    obj_rot_seq = obj_rotations[:seq_len]
    contact_smooth_arr = smooth_contact(contact_prob, cfg.contact_smooth_window)
    num_phase = max(int(phase.max()) + 1, cfg.num_phase_classes)
    num_support = cfg.num_support_classes
    phase_softmax = np.zeros((seq_len, num_phase), dtype=np.float32)
    support_softmax = np.zeros((seq_len, num_support), dtype=np.float32)
    phase_softmax[np.arange(seq_len), np.clip(phase[:seq_len].astype(np.int64), 0, num_phase - 1)] = 1.0
    support_softmax[np.arange(seq_len), np.clip(support[:seq_len].astype(np.int64), 0, num_support - 1)] = 1.0
    phase_smooth = smooth_categorical_softmax(phase_softmax, cfg.phase_smooth_window)
    support_smooth = smooth_categorical_softmax(support_softmax, cfg.support_smooth_window)
    target_smooth_local = smooth_target_local(target_loc, contact_smooth_arr, cfg.target_smooth_window)
    phase_entropy = _entropy(phase_smooth)
    support_entropy = _entropy(support_smooth)
    contact_cands, segments = _build_contact_anchors(
        contact_smooth_arr, target_smooth_local, phase_smooth, support_smooth,
        phase_entropy, support_entropy, cfg,
    )
    change_cands = _build_change_anchors(phase_smooth, support_smooth, cfg)
    merged = merge_nearby_anchors(contact_cands + change_cands, cfg)
    kept = temporal_nms_budget(merged, cfg)

    # Per-anchor segment lookup (for duration / stability)
    seg_lookup: dict[tuple[int, int], dict[str, Any]] = {}
    for seg in segments:
        seg_lookup[(int(seg["part"]), int(seg["start"]))] = seg

    R = _axis_angle_to_rot(obj_rot_seq)
    if obj_pc.shape[0] > surface_samples:
        idx = np.random.RandomState(0).choice(obj_pc.shape[0], surface_samples, replace=False)
        pc_sub = obj_pc[idx]
    else:
        pc_sub = obj_pc
    obj_pc_world = np.einsum("tij,nj->tni", R, pc_sub) + obj_pos_seq[:, None, :]
    target_world_full = lift_target_local_to_world_np(target_smooth_local, obj_pos_seq, obj_rot_seq)

    # For each kept anchor, compute conf_v2 per part
    anchor_records: list[dict[str, Any]] = []
    for ai, c in enumerate(kept):
        type_name = ANCHOR_TYPE_NAMES.get(c.type_id, str(c.type_id))
        active_parts = np.where(c.parts > 0)[0].tolist()
        t_a = int(min(max(0, c.time), seq_len - 1))
        for p in active_parts:
            joint = PART_JOINT[p]
            target_w = target_world_full[t_a, p]
            hand_world = gt_joints[t_a, joint]
            err_to_hand = float(np.linalg.norm(target_w - hand_world) * 100.0)
            target_to_surface_cm = float(
                np.linalg.norm(obj_pc_world[t_a] - target_w[None, :], axis=-1).min() * 100.0
            )
            # Find the source contact segment for this part containing t_a
            seg_match = None
            for seg in segments:
                if int(seg["part"]) == int(p):
                    if int(seg["start"]) <= t_a <= int(seg["end"]):
                        seg_match = seg
                        break
            if seg_match is None:
                # Best-effort: fall back to nearest segment of same part
                same_part = [s for s in segments if int(s["part"]) == int(p)]
                if same_part:
                    seg_match = min(same_part, key=lambda s: abs(int(s["start"]) - t_a))
                else:
                    seg_match = {"start": t_a, "end": t_a, "duration": 1, "part": p}
            conf_v2, factors = _per_anchor_conf_v2(
                anchor_time=t_a,
                anchor_part=int(p),
                contact_conf=float(c.confidence),
                target_to_surface_cm=target_to_surface_cm,
                segment_start=int(seg_match["start"]),
                segment_end=int(seg_match["end"]),
                contact_state=contact_state,
                target_local=target_loc,
                seq_len=seq_len,
                edge_margin=edge_margin,
                flicker_max_frames=flicker_max_frames,
            )
            anchor_records.append({
                "anchor_idx": int(ai),
                "anchor_type": type_name,
                "part": int(p),
                "anchor_time": t_a,
                "anchor_conf_v1": float(c.confidence),
                "anchor_conf_v2": float(conf_v2),
                "target_to_GT_hand_cm": err_to_hand,
                "target_to_surface_cm": target_to_surface_cm,
                "factors": factors,
                "segment_start": int(seg_match["start"]),
                "segment_end": int(seg_match["end"]),
            })

    return {
        "subset": subset, "seq_id": seq_id, "seq_len": seq_len,
        "anchors": anchor_records,
    }


def _corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3:
        return 0.0
    from scipy.stats import spearmanr
    try:
        s, _ = spearmanr(a, b)
        if not np.isfinite(s):
            return 0.0
        return float(s)
    except Exception:
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=Path("configs/training/anchordiff_v18_a1_FULL_DATA.yaml"))
    parser.add_argument("--selection-json", type=Path,
                        default=Path("analyses/2026-05-13_v18_recon_ladder_truncated_rollout.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("analyses/2026-05-17_anchor_conf_v2_debug.json"))
    parser.add_argument("--md", type=Path,
                        default=Path("analyses/2026-05-17_anchor_conf_v2_debug.md"))
    parser.add_argument("--max-clips", type=int, default=16)
    parser.add_argument("--num-candidates", type=int, default=256)
    parser.add_argument("--bucket", choices=["train", "val"], default="train")
    parser.add_argument("--balanced-subsets", action="store_true", default=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--edge-margin", type=int, default=5)
    parser.add_argument("--flicker-max-frames", type=int, default=2)
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
            edge_margin=int(args.edge_margin),
            flicker_max_frames=int(args.flicker_max_frames),
            surface_samples=int(args.surface_samples),
        )
        for b in range(B)
    ]

    # Aggregate across all hand anchors (parts 0/1)
    all_anchors = [a for clip in audits for a in clip["anchors"] if a["part"] <= 1]
    n_total = len(all_anchors)

    def _corr_per_type(anchor_type: str | None) -> dict[str, float]:
        rows = all_anchors if anchor_type is None else [a for a in all_anchors if a["anchor_type"] == anchor_type]
        if len(rows) < 3:
            return {"n": len(rows), "pearson_v1": 0.0, "pearson_v2": 0.0, "spearman_v1": 0.0, "spearman_v2": 0.0}
        conf1 = np.asarray([a["anchor_conf_v1"] for a in rows], dtype=np.float64)
        conf2 = np.asarray([a["anchor_conf_v2"] for a in rows], dtype=np.float64)
        err = np.asarray([a["target_to_GT_hand_cm"] for a in rows], dtype=np.float64)
        return {
            "n": len(rows),
            "pearson_v1": _corrcoef(conf1, err),
            "pearson_v2": _corrcoef(conf2, err),
            "spearman_v1": _spearman(conf1, err),
            "spearman_v2": _spearman(conf2, err),
        }

    aggregate = {
        "n_clips": B,
        "n_hand_anchors": n_total,
        "correlation_overall": _corr_per_type(None),
        "correlation_by_type": {
            t: _corr_per_type(t) for t in ("onset", "stable", "release")
        },
        "correlation_by_part": {},
    }
    for part_name, part_idx, _joint in HAND_PARTS:
        rows = [a for a in all_anchors if a["part"] == part_idx]
        if len(rows) >= 3:
            conf1 = np.asarray([a["anchor_conf_v1"] for a in rows], dtype=np.float64)
            conf2 = np.asarray([a["anchor_conf_v2"] for a in rows], dtype=np.float64)
            err = np.asarray([a["target_to_GT_hand_cm"] for a in rows], dtype=np.float64)
            aggregate["correlation_by_part"][part_name] = {
                "n": len(rows),
                "pearson_v1": _corrcoef(conf1, err),
                "pearson_v2": _corrcoef(conf2, err),
                "spearman_v1": _spearman(conf1, err),
                "spearman_v2": _spearman(conf2, err),
            }
        else:
            aggregate["correlation_by_part"][part_name] = {"n": len(rows)}

    # Per-clip correlation (helps see if any one clip dominates)
    per_clip_corr = []
    for clip in audits:
        rows = [a for a in clip["anchors"] if a["part"] <= 1]
        if len(rows) < 3:
            per_clip_corr.append({
                "subset": clip["subset"], "seq_id": clip["seq_id"], "n": len(rows),
                "pearson_v1": 0.0, "pearson_v2": 0.0,
            })
            continue
        conf1 = np.asarray([a["anchor_conf_v1"] for a in rows], dtype=np.float64)
        conf2 = np.asarray([a["anchor_conf_v2"] for a in rows], dtype=np.float64)
        err = np.asarray([a["target_to_GT_hand_cm"] for a in rows], dtype=np.float64)
        per_clip_corr.append({
            "subset": clip["subset"], "seq_id": clip["seq_id"], "n": len(rows),
            "pearson_v1": _corrcoef(conf1, err),
            "pearson_v2": _corrcoef(conf2, err),
        })

    aggregate["per_clip_correlation"] = per_clip_corr

    payload = {
        "config": str(args.config),
        "selection_json": str(args.selection_json),
        "n_clips": B,
        "aggregate": aggregate,
        "clips": audits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")

    # Verdict
    co = aggregate["correlation_overall"]
    if co["pearson_v2"] <= -0.10 and co["pearson_v2"] < co["pearson_v1"]:
        verdict = (
            f"**anchor_conf_v2 is useful as a debug reliability weight**: "
            f"Pearson r = {co['pearson_v2']:.3f} (negative direction) vs current "
            f"anchor_conf r = {co['pearson_v1']:+.3f}. Spearman v2 = {co['spearman_v2']:.3f}."
        )
    elif co["pearson_v2"] >= co["pearson_v1"] - 0.05:
        verdict = (
            f"**anchor_conf_v2 does NOT improve reliability**: "
            f"v2 Pearson r = {co['pearson_v2']:+.3f} vs v1 r = {co['pearson_v1']:+.3f}. "
            "Reject confidence-weighting for now (Case E)."
        )
    else:
        verdict = (
            f"anchor_conf_v2 improves Pearson r from {co['pearson_v1']:+.3f} to "
            f"{co['pearson_v2']:+.3f}; weak negative correlation but not robust. "
            "Keep as debug only; do NOT promote into model input."
        )

    lines = [
        "# anchor_conf_v2 Debug Side-Channel (Round 7)",
        "",
        f"- Config: `{args.config}`",
        f"- Selection: `{args.selection_json}`",
        f"- Clips: {B}, hand anchors: {n_total}",
        "",
        "## Formula",
        "",
        "geometric mean of:",
        "1. contact_score = anchor_conf (current value)",
        "2. surface_score = 1 - clip(target_to_surface_cm / 10cm, 0, 1)",
        "3. duration_score = min(1, segment_duration / 8)",
        "4. stability_score = 1 / (1 + std_target_local_cm_in_segment)",
        "5. boundary_flicker = 0.5 if (is_boundary or is_flicker) else 1.0",
        "",
        "## Overall correlation (anchor_conf vs target → GT hand error)",
        "",
        f"- n = {co['n']} hand anchors",
        f"- Pearson r (v1 conf): **{co['pearson_v1']:+.3f}**",
        f"- Pearson r (v2 conf): **{co['pearson_v2']:+.3f}** (sign target: negative = lower error)",
        f"- Spearman ρ (v1): {co['spearman_v1']:+.3f}",
        f"- Spearman ρ (v2): {co['spearman_v2']:+.3f}",
        "",
        "## Verdict",
        "",
        verdict,
        "",
        "## Correlation by anchor type",
        "",
        "| type | n | v1 Pearson | v2 Pearson | v1 Spearman | v2 Spearman |",
        "|------|---|------------|------------|--------------|--------------|",
    ]
    for t_name, st in aggregate["correlation_by_type"].items():
        lines.append(
            f"| {t_name} | {st['n']} | "
            f"{st['pearson_v1']:+.3f} | {st['pearson_v2']:+.3f} | "
            f"{st['spearman_v1']:+.3f} | {st['spearman_v2']:+.3f} |"
        )
    lines += [
        "",
        "## Correlation by hand part",
        "",
        "| part | n | v1 Pearson | v2 Pearson | v1 Spearman | v2 Spearman |",
        "|------|---|------------|------------|--------------|--------------|",
    ]
    for p_name, st in aggregate["correlation_by_part"].items():
        if st.get("n", 0) >= 3:
            lines.append(
                f"| {p_name} | {st['n']} | "
                f"{st['pearson_v1']:+.3f} | {st['pearson_v2']:+.3f} | "
                f"{st['spearman_v1']:+.3f} | {st['spearman_v2']:+.3f} |"
            )
    lines += [
        "",
        "## Per-clip correlation",
        "",
    ]
    rows = [["subset", "seq_id", "n", "v1 Pearson", "v2 Pearson"]]
    for r in per_clip_corr:
        rows.append([r["subset"], r["seq_id"], r["n"],
                     f"{r['pearson_v1']:+.3f}", f"{r['pearson_v2']:+.3f}"])
    lines.append(format_md_table(rows))
    lines.append("")
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.md}")


if __name__ == "__main__":
    main()
