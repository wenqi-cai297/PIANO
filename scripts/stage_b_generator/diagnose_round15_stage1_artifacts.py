"""Round-15 lightweight diagnostics on existing Round-14 artifacts.

Reads the per-clip GT / S1-A / S1-B coarse-v1 trajectories saved by
``render_round15_visual_review.py`` and the 12 Round-14 eval JSONs to
answer Part-D diagnostic questions WITHOUT retraining or generating new
checkpoints:

1. ``root_acc / root_jerk by frame position mod block_size=16`` —
   does S1-B's jerk failure align with the block-causal block boundaries?
2. ``root velocity stored channels vs root_local_trans derivative``
   consistency — does the model emit a root_vel channel (dims 3:6)
   that disagrees with diff(root_local_trans) (dims 0:3)?
3. ``yaw_vel stored vs yaw_sin/cos derivative`` consistency — same
   question for yaw.
4. ``clip dominance`` — is S1-B's per-subset jerk failure driven by a
   single clip or sign-consistent across all 6 clips per subset?

Output:
    analyses/2026-05-23_stage1_round15_artifact_diagnostics.json
    analyses/2026-05-23_stage1_round15_artifact_diagnostics.md

This script does NOT call the model. All evidence comes from already-
generated trajectories / eval JSONs.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ANALYSES = Path("analyses")
TRAJ_DIR = ANALYSES / "round15_stage1_visual_review" / "trajectories"
EVAL_DATE_TAG = "2026-05-22"
CKPT_SEEDS = (42, 43, 44, 45, 46, 47)
SUBSETS = ("chairs", "imhd", "neuraldome", "omomo_correct_v2")
BLOCK_SIZE = 16


def _deriv(arr: np.ndarray, order: int) -> np.ndarray:
    out = arr
    for _ in range(order):
        out = np.diff(out, axis=0)
    return out


def _norm_deriv(arr: np.ndarray, order: int) -> np.ndarray:
    return np.linalg.norm(_deriv(arr, order), axis=-1)


def _per_clip_mod_block_breakdown(
    coarse: np.ndarray, block_size: int = BLOCK_SIZE,
) -> dict[str, list[float]]:
    """Compute root jerk magnitude at each frame, then average by
    `frame mod block_size`.

    Frame indices for `||Δ³root||` are 3..T-1 (length T-3). To map back
    to the original frame index we use the LAST of the 4 frames involved
    (index i corresponds to ||r[i] - 3 r[i-1] + 3 r[i-2] - r[i-3]||,
    so we tag with i = 3..T-1). The same offset is applied for jerk
    and acceleration.

    Returns the per-mod-bin mean (16 entries each).
    """
    T = coarse.shape[0]
    out: dict[str, list[float]] = {}
    root = coarse[:, 0:3]
    if T >= 4:
        jerk = np.linalg.norm(_deriv(root, 3), axis=-1)            # length T-3
        frame_idx_jerk = np.arange(3, T)
        bins = frame_idx_jerk % block_size
        means = np.zeros(block_size, dtype=np.float64)
        counts = np.zeros(block_size, dtype=np.int64)
        for j, b in zip(jerk, bins):
            means[b] += j
            counts[b] += 1
        means = np.where(counts > 0, means / np.maximum(counts, 1), 0.0)
        out["jerk_by_mod"] = [float(v) for v in means]
    if T >= 3:
        acc = np.linalg.norm(_deriv(root, 2), axis=-1)              # length T-2
        frame_idx_acc = np.arange(2, T)
        bins = frame_idx_acc % block_size
        means = np.zeros(block_size, dtype=np.float64)
        counts = np.zeros(block_size, dtype=np.int64)
        for a, b in zip(acc, bins):
            means[b] += a
            counts[b] += 1
        means = np.where(counts > 0, means / np.maximum(counts, 1), 0.0)
        out["acc_by_mod"] = [float(v) for v in means]
    return out


def _consistency_root_vel_vs_root_pos_deriv(
    coarse: np.ndarray,
) -> tuple[float, float, float]:
    """Compare the stored root_vel channels (dims 3:6) with
    diff(root_local_trans) (dims 0:3).

    Returns (mean_abs_disagreement, mean_abs_stored, mean_abs_derived).
    """
    root_trans = coarse[:, 0:3]
    root_vel_stored = coarse[:, 3:6]
    if root_trans.shape[0] < 2:
        return 0.0, 0.0, 0.0
    root_vel_derived = np.diff(root_trans, axis=0)
    # Compare on overlapping frames. The extractor's convention is to
    # use diff(prepend=first) so length matches; we'll just compare on
    # frames 1..T-1.
    stored = root_vel_stored[1:]
    derived = root_vel_derived
    disagree = float(np.mean(np.linalg.norm(stored - derived, axis=-1)))
    mean_stored = float(np.mean(np.linalg.norm(stored, axis=-1)))
    mean_derived = float(np.mean(np.linalg.norm(derived, axis=-1)))
    return disagree, mean_stored, mean_derived


def _consistency_yaw_vel_vs_sincos_deriv(
    coarse: np.ndarray,
) -> tuple[float, float, float]:
    """Compare the stored yaw_vel channel (dim 8) with the finite
    difference of unwrap(atan2(sin, cos)) from dims 6:8."""
    if coarse.shape[0] < 2:
        return 0.0, 0.0, 0.0
    yaw = np.unwrap(np.arctan2(coarse[:, 6], coarse[:, 7]))
    yaw_derived = np.diff(yaw, prepend=yaw[:1])
    yaw_stored = coarse[:, 8]
    disagree = float(np.mean(np.abs(yaw_stored - yaw_derived)))
    mean_stored = float(np.mean(np.abs(yaw_stored)))
    mean_derived = float(np.mean(np.abs(yaw_derived)))
    return disagree, mean_stored, mean_derived


# ============================================================================
# Clip-dominance analysis off eval JSONs
# ============================================================================


def _per_clip_max_xgt_by_mode(
    metric: str,
) -> dict[tuple[str, str], dict[str, list[float]]]:
    """For every (subset, seq_id), list the per-ckpt-seed mean xGT
    (averaged over 3 sampler seeds) for the requested metric, per mode.

    Returns ``{(sub, sid) -> {mode -> [xgt_seed42, ..., xgt_seed47]}}``.
    """
    per_key: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"s1a": [], "s1b": []}
    )
    for mode in ("s1a", "s1b"):
        for cs in CKPT_SEEDS:
            path = ANALYSES / f"{EVAL_DATE_TAG}_stage1_eval_round14_{mode}_ckptseed{cs}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            # group by (sub, seq_id), average over 3 sampler seeds.
            by_clip: dict[tuple[str, str], list[float]] = defaultdict(list)
            for rec in payload["per_clip"]:
                x = rec["xGT"].get(f"xGT.{metric}", float("nan"))
                if isinstance(x, (int, float)) and math.isfinite(x):
                    by_clip[(rec["subset"], rec["seq_id"])].append(float(x))
            for key, vs in by_clip.items():
                per_key[key][mode].append(float(np.mean(vs)) if vs else float("nan"))
    return per_key


def _clip_dominance(metric: str) -> dict[str, Any]:
    """For each subset, return per-clip mean xGT for S1-B and the
    fraction of subset-level jerk concentration accounted for by the
    worst-2 clips out of 6.
    """
    by_key = _per_clip_max_xgt_by_mode(metric)
    out: dict[str, Any] = {"per_clip_means": {}, "per_subset_concentration": {}}
    by_subset: dict[str, list[tuple[str, str, float, float]]] = defaultdict(list)
    for (sub, sid), modes in by_key.items():
        a = float(np.nanmean(modes["s1a"])) if modes["s1a"] else float("nan")
        b = float(np.nanmean(modes["s1b"])) if modes["s1b"] else float("nan")
        by_subset[sub].append((sub, sid, a, b))
    for sub, rows in by_subset.items():
        rows_sorted = sorted(rows, key=lambda r: r[3], reverse=True)
        per_clip: list[dict[str, Any]] = []
        for sub_, sid, a, b in rows_sorted:
            per_clip.append({"seq_id": sid, "s1a_xGT": a, "s1b_xGT": b})
        out["per_clip_means"][sub] = per_clip
        # Concentration: fraction of total S1-B jerk contributed by the
        # top-2 clips (out of 6). >0.5 = a couple of bad clips drive the
        # subset average; ≈0.33 = uniformly distributed.
        s1b_vals = [r[3] for r in rows_sorted if math.isfinite(r[3])]
        if len(s1b_vals) >= 2 and sum(s1b_vals) > 1e-6:
            top2_share = float(sum(s1b_vals[:2]) / sum(s1b_vals))
        else:
            top2_share = float("nan")
        out["per_subset_concentration"][sub] = {
            "n_clips": len(s1b_vals),
            "s1b_top2_share_of_total": top2_share,
            "s1b_mean_xGT_per_clip": s1b_vals,
        }
    return out


# ============================================================================
# Main
# ============================================================================


def main() -> int:
    if not TRAJ_DIR.exists():
        raise SystemExit(
            f"[r15-diag] trajectories dir missing: {TRAJ_DIR}. "
            "Run render_round15_visual_review.py first."
        )

    traj_files = sorted(TRAJ_DIR.glob("*.npz"))
    if not traj_files:
        raise SystemExit(f"[r15-diag] no trajectory .npz under {TRAJ_DIR}")

    per_clip_diag: list[dict[str, Any]] = []
    # Subset aggregates for the mod-16 plot.
    mod_jerk_agg: dict[str, dict[str, np.ndarray]] = {
        sub: {"gt": np.zeros(BLOCK_SIZE), "s1a": np.zeros(BLOCK_SIZE),
              "s1b": np.zeros(BLOCK_SIZE)} for sub in SUBSETS
    }
    mod_jerk_counts: dict[str, int] = {sub: 0 for sub in SUBSETS}

    for path in traj_files:
        npz = np.load(path, allow_pickle=False)
        gt = np.asarray(npz["gt"], dtype=np.float32)
        s1a = np.asarray(npz["s1a"], dtype=np.float32)
        s1b = np.asarray(npz["s1b"], dtype=np.float32)
        T = int(npz["T"])
        stem = path.stem      # e.g. "neuraldome_subject01_chair_499_1"
        # Recover subset by prefix match (subset name appears at the
        # start of the stem; longest prefix match wins so
        # "omomo_correct_v2" beats "omomo").
        sub = next(
            (s for s in sorted(SUBSETS, key=len, reverse=True) if stem.startswith(s + "_")),
            None,
        )
        if sub is None:
            print(f"[r15-diag] could not infer subset for {stem}; skipping")
            continue
        seq_id = stem[len(sub) + 1:]

        # Mod-16 root acc/jerk breakdown.
        mod_gt = _per_clip_mod_block_breakdown(gt)
        mod_a = _per_clip_mod_block_breakdown(s1a)
        mod_b = _per_clip_mod_block_breakdown(s1b)
        if mod_b.get("jerk_by_mod"):
            mod_jerk_agg[sub]["gt"] += np.asarray(mod_gt["jerk_by_mod"])
            mod_jerk_agg[sub]["s1a"] += np.asarray(mod_a["jerk_by_mod"])
            mod_jerk_agg[sub]["s1b"] += np.asarray(mod_b["jerk_by_mod"])
            mod_jerk_counts[sub] += 1

        # Channel consistency.
        rv_dis_gt, rv_st_gt, rv_dr_gt = _consistency_root_vel_vs_root_pos_deriv(gt)
        rv_dis_a, rv_st_a, rv_dr_a = _consistency_root_vel_vs_root_pos_deriv(s1a)
        rv_dis_b, rv_st_b, rv_dr_b = _consistency_root_vel_vs_root_pos_deriv(s1b)
        yv_dis_gt, yv_st_gt, yv_dr_gt = _consistency_yaw_vel_vs_sincos_deriv(gt)
        yv_dis_a, yv_st_a, yv_dr_a = _consistency_yaw_vel_vs_sincos_deriv(s1a)
        yv_dis_b, yv_st_b, yv_dr_b = _consistency_yaw_vel_vs_sincos_deriv(s1b)

        per_clip_diag.append({
            "subset": sub, "seq_id": seq_id, "T": T,
            "mod_jerk_s1b_max_over_min": float(
                max(mod_b["jerk_by_mod"]) / max(min(mod_b["jerk_by_mod"]), 1e-9)
            ) if mod_b.get("jerk_by_mod") else float("nan"),
            "mod_jerk_s1b_argmax": int(np.argmax(mod_b["jerk_by_mod"])) if mod_b.get("jerk_by_mod") else -1,
            "mod_jerk_s1b_argmin": int(np.argmin(mod_b["jerk_by_mod"])) if mod_b.get("jerk_by_mod") else -1,
            "root_vel_consistency": {
                "gt_disagreement_mean":  rv_dis_gt,
                "s1a_disagreement_mean": rv_dis_a,
                "s1b_disagreement_mean": rv_dis_b,
                "gt_stored_mean":  rv_st_gt,  "gt_derived_mean":  rv_dr_gt,
                "s1a_stored_mean": rv_st_a,   "s1a_derived_mean": rv_dr_a,
                "s1b_stored_mean": rv_st_b,   "s1b_derived_mean": rv_dr_b,
            },
            "yaw_vel_consistency": {
                "gt_disagreement_mean":  yv_dis_gt,
                "s1a_disagreement_mean": yv_dis_a,
                "s1b_disagreement_mean": yv_dis_b,
                "gt_stored_mean":  yv_st_gt,  "gt_derived_mean":  yv_dr_gt,
                "s1a_stored_mean": yv_st_a,   "s1a_derived_mean": yv_dr_a,
                "s1b_stored_mean": yv_st_b,   "s1b_derived_mean": yv_dr_b,
            },
        })

    # Average mod-16 across the clips in each subset.
    mod_jerk_avg: dict[str, dict[str, list[float]]] = {}
    for sub in SUBSETS:
        n = mod_jerk_counts[sub]
        if n == 0:
            continue
        mod_jerk_avg[sub] = {
            mode: (mod_jerk_agg[sub][mode] / n).tolist() for mode in ("gt", "s1a", "s1b")
        }

    # Clip dominance for root_jerk_p95 (the failing metric).
    dom_jerk = _clip_dominance("root_jerk_p95")
    dom_acc = _clip_dominance("root_acc_p95")
    # Sanity check: same analysis for the metric S1-B wins on.
    dom_pelvis_rot = _clip_dominance("pelvis_rot6d_vel_mean")

    out_json = {
        "block_size": BLOCK_SIZE,
        "n_clips_diagnosed": len(per_clip_diag),
        "per_clip": per_clip_diag,
        "mod_block_root_jerk_subset_avg": mod_jerk_avg,
        "mod_block_summary": {
            sub: {
                "argmax_bin_s1b": int(np.argmax(mod_jerk_avg[sub]["s1b"])),
                "argmin_bin_s1b": int(np.argmin(mod_jerk_avg[sub]["s1b"])),
                "max_over_min_s1b": float(
                    max(mod_jerk_avg[sub]["s1b"])
                    / max(min(mod_jerk_avg[sub]["s1b"]), 1e-9)
                ),
                "argmax_bin_s1a": int(np.argmax(mod_jerk_avg[sub]["s1a"])),
                "argmin_bin_s1a": int(np.argmin(mod_jerk_avg[sub]["s1a"])),
                "max_over_min_s1a": float(
                    max(mod_jerk_avg[sub]["s1a"])
                    / max(min(mod_jerk_avg[sub]["s1a"]), 1e-9)
                ),
            }
            for sub in mod_jerk_avg
        },
        "clip_dominance_root_jerk_p95": dom_jerk,
        "clip_dominance_root_acc_p95": dom_acc,
        "clip_dominance_pelvis_rot6d_vel": dom_pelvis_rot,
    }
    out_path = ANALYSES / "2026-05-23_stage1_round15_artifact_diagnostics.json"
    out_path.write_text(json.dumps(out_json, indent=2, default=float), encoding="utf-8")
    print(f"[r15-diag] wrote {out_path}")

    # ─── MD report ──
    lines: list[str] = []
    lines.append("# Round-15 Stage-1 Artifact Diagnostics")
    lines.append("")
    lines.append(
        "Per-clip diagnostics on the 10-clip Round-15 visual-review trajectories, "
        "plus per-subset clip dominance on the full Round-14 eval JSONs (24 clips × "
        "6 ckpt seeds × 3 sampler seeds)."
    )
    lines.append("")
    lines.append("## 1. Root jerk by frame index mod block_size=16 (subset average)")
    lines.append("")
    lines.append(
        "If block-causal K=16 is the source of S1-B's root jitter, jerk should "
        "spike at specific bins (e.g. the first or last frame of each 16-frame "
        "block). If the failure mode is uniform high-frequency root noise, "
        "the max/min ratio across the 16 bins will be small."
    )
    lines.append("")
    lines.append("| subset | S1-A max/min | S1-A argmax | S1-B max/min | S1-B argmax |")
    lines.append("|---|---:|---:|---:|---:|")
    for sub in SUBSETS:
        if sub not in out_json["mod_block_summary"]:
            continue
        s = out_json["mod_block_summary"][sub]
        lines.append(
            f"| {sub} | {s['max_over_min_s1a']:.2f} | {s['argmax_bin_s1a']} | "
            f"{s['max_over_min_s1b']:.2f} | {s['argmax_bin_s1b']} |"
        )
    lines.append("")
    lines.append("**Interpretation key:**")
    lines.append("")
    lines.append("- `max/min ≈ 1.0–1.5` → root jerk is approximately uniform across")
    lines.append("  the 16 mod-bins → S1-B failure is broadband high-frequency noise,")
    lines.append("  NOT block-boundary aliasing.")
    lines.append("- `max/min >> 2.0` AND `argmax` lands consistently on bin 0 or 15")
    lines.append("  → block-boundary artefact.")
    lines.append("- Bin position is the frame index of the LATEST of the 4 frames")
    lines.append("  involved in the third difference; block boundaries land between")
    lines.append("  bins 15 → 0 (or 0 → 15 if you read the other direction).")
    lines.append("")
    lines.append("Per-subset normalized profile (S1-B jerk by mod-bin / mean):")
    lines.append("")
    for sub in SUBSETS:
        if sub not in mod_jerk_avg:
            continue
        s1b = np.asarray(mod_jerk_avg[sub]["s1b"])
        gt = np.asarray(mod_jerk_avg[sub]["gt"])
        s1b_norm = s1b / max(float(s1b.mean()), 1e-9)
        gt_norm = gt / max(float(gt.mean()), 1e-9)
        lines.append(f"### {sub}")
        lines.append("")
        lines.append("| mod-bin | GT (norm) | S1-A (norm) | S1-B (norm) |")
        lines.append("|---:|---:|---:|---:|")
        s1a = np.asarray(mod_jerk_avg[sub]["s1a"])
        s1a_norm = s1a / max(float(s1a.mean()), 1e-9)
        for i in range(BLOCK_SIZE):
            lines.append(
                f"| {i} | {gt_norm[i]:.2f} | {s1a_norm[i]:.2f} | {s1b_norm[i]:.2f} |"
            )
        lines.append("")
    lines.append("## 2. Per-clip channel-consistency on Round-15 trajectories")
    lines.append("")
    lines.append(
        "Compares the stored root_vel (dims 3:6) against diff(root_local_trans) "
        "(dims 0:3), and stored yaw_vel (dim 8) against diff(unwrap(atan2(sin, cos))) "
        "(dims 6:8). For GT these disagreements should be exactly 0 by construction. "
        "For samples, large disagreement means the model is emitting internally "
        "inconsistent channels."
    )
    lines.append("")
    lines.append(
        "| clip | GT rv_dis | S1-A rv_dis | S1-B rv_dis | "
        "GT yv_dis | S1-A yv_dis | S1-B yv_dis |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in per_clip_diag:
        rv = r["root_vel_consistency"]; yv = r["yaw_vel_consistency"]
        lines.append(
            f"| {r['subset']}/{r['seq_id'][:32]} | "
            f"{rv['gt_disagreement_mean']:.4f} | "
            f"{rv['s1a_disagreement_mean']:.4f} | "
            f"{rv['s1b_disagreement_mean']:.4f} | "
            f"{yv['gt_disagreement_mean']:.4f} | "
            f"{yv['s1a_disagreement_mean']:.4f} | "
            f"{yv['s1b_disagreement_mean']:.4f} |"
        )
    lines.append("")
    lines.append("## 3. Clip-dominance: root_jerk_p95 xGT per clip (S1-B)")
    lines.append("")
    lines.append(
        "Are S1-B's per-subset jerk averages driven by 1–2 outlier clips, or are "
        "all 6 clips in each subset bad? `top2_share` = sum of worst-2 clips / "
        "subset total. ≈0.33 = uniform; > 0.5 = a couple of bad clips dominate."
    )
    lines.append("")
    for sub in SUBSETS:
        block = dom_jerk["per_subset_concentration"].get(sub, {})
        if not block:
            continue
        share = block.get("s1b_top2_share_of_total", float("nan"))
        lines.append(f"### {sub} — top-2 share = {share:.2f}")
        lines.append("")
        lines.append("| seq_id | S1-A xGT | S1-B xGT |")
        lines.append("|---|---:|---:|")
        for row in dom_jerk["per_clip_means"][sub]:
            lines.append(
                f"| {row['seq_id'][:48]} | {row['s1a_xGT']:.2f} | {row['s1b_xGT']:.2f} |"
            )
        lines.append("")

    out_md = ANALYSES / "2026-05-23_stage1_round15_artifact_diagnostics.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[r15-diag] wrote {out_md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
