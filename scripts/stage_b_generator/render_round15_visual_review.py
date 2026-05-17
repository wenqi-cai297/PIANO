"""Round-15 Stage-1 visual review.

Selects 8–12 representative clips from the Round-9 fixed selection
(``analyses/2026-05-19_subset_balanced_failure_selection.json``) using
the per-clip evidence in the 12 Round-14 eval JSONs, then for each
selected clip samples Coarse-v1 from:

- GT (loaded from the Stage-1 cache);
- S1-A ckpt seed 42 (``runs/training/stage1_s1a_seed42/final.pt``);
- S1-B ckpt seed 42 (``runs/training/stage1_s1b_seed42/final.pt``);

both with sampler seed 42 and full 1 000-step DDPM (cfg_scale = 1.0).

Outputs:

- ``analyses/round15_stage1_visual_review/selected_clips.json``
- ``analyses/round15_stage1_visual_review/metrics_table.csv``
- ``analyses/round15_stage1_visual_review/plots/<subset>_<seq_id>.png``
- ``analyses/round15_stage1_visual_review/trajectories/<subset>_<seq_id>.npz``

The plots are 9-panel curve panels (root XZ traj, root height, root
vel, root acc, root jerk, yaw, pelvis rot vel, head height, shoulder
height). No full-body video renderer is invoked — coarse motion is 23
scalar channels; curve plots are the right primitive for this review.

Usage
-----

    $env:PYTHONIOENCODING="utf-8"
    conda run -n piano python scripts/stage_b_generator/render_round15_visual_review.py
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch

from piano.models.coarse_motion_prior import (
    CoarsePriorConfig, CoarsePriorDenoiserConfig, CoarsePriorDiff,
)
from piano.models.motion_anchordiff import DiffusionConfig


CKPT_SEEDS = (42, 43, 44, 45, 46, 47)
SAMPLER_SEEDS = (42, 43, 44)
SUBSETS_ORDER = ("chairs", "imhd", "neuraldome", "omomo_correct_v2")

ANALYSES = Path("analyses")
EVAL_DATE_TAG = "2026-05-22"

CACHE_ROOT = Path("cache/stage1_coarse_v1_full")
S1A_CKPT = Path("runs/training/stage1_s1a_seed42/final.pt")
S1B_CKPT = Path("runs/training/stage1_s1b_seed42/final.pt")

OUT_DIR = ANALYSES / "round15_stage1_visual_review"
PLOTS_DIR = OUT_DIR / "plots"
TRAJ_DIR = OUT_DIR / "trajectories"

COARSE_DIM = 23


# ============================================================================
# Eval data -> per-clip features for clip selection
# ============================================================================


def _load_eval_payloads() -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {"s1a": [], "s1b": []}
    for mode in ("s1a", "s1b"):
        for cs in CKPT_SEEDS:
            path = ANALYSES / f"{EVAL_DATE_TAG}_stage1_eval_round14_{mode}_ckptseed{cs}.json"
            out[mode].append(json.loads(path.read_text(encoding="utf-8")))
    return out


def _per_clip_aggregate_xgt(
    payloads_by_mode: dict[str, list[dict[str, Any]]],
) -> dict[tuple[str, str], dict[str, dict[str, float]]]:
    """Average xGT over (6 ckpt seeds × 3 sampler seeds) per clip.

    Returns ``{(subset, seq_id) -> {mode -> {metric -> mean_xGT}}}``.
    """
    by_key: dict[tuple[str, str], dict[str, defaultdict[str, list[float]]]] = (
        defaultdict(lambda: {"s1a": defaultdict(list), "s1b": defaultdict(list)})
    )
    for mode, payloads in payloads_by_mode.items():
        for payload in payloads:
            for rec in payload["per_clip"]:
                sub = rec["subset"]
                sid = rec["seq_id"]
                for k, v in rec["xGT"].items():
                    if not k.startswith("xGT."):
                        continue
                    metric = k[len("xGT."):]
                    if isinstance(v, (int, float)) and math.isfinite(v):
                        by_key[(sub, sid)][mode][metric].append(float(v))
    out: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
    for key, m in by_key.items():
        out[key] = {
            mode: {
                metric: float(np.mean(vals)) if vals else float("nan")
                for metric, vals in d.items()
            }
            for mode, d in m.items()
        }
    return out


def _select_clips(
    agg: dict[tuple[str, str], dict[str, dict[str, float]]],
) -> list[dict[str, Any]]:
    """Pick representative clips per subset per the Round-15 visual-review
    protocol.

    chairs (2): largest S1-B paired-improvement in pelvis_rot6d_vel
                + spine3_rot6d_vel (rotation wins).
    imhd (3): biggest split — S1-B rotation win AND biggest S1-B
              root_jerk blow-up on the same clips.
    neuraldome (3): worst S1-B root_jerk blow-up.
    omomo (2): worst S1-B root_jerk blow-up.

    Total: 10 clips.
    """
    by_subset: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for (sub, sid) in agg.keys():
        by_subset[sub].append((sub, sid))

    selected: list[dict[str, Any]] = []
    # Sort each pool, take top-k. Score depends on subset.
    def _score(metrics: dict[str, dict[str, float]], name: str, sign: int) -> float:
        a = metrics["s1a"].get(name, float("nan"))
        b = metrics["s1b"].get(name, float("nan"))
        if not (math.isfinite(a) and math.isfinite(b)):
            return float("-inf")
        # Δerr = err(S1-A) − err(S1-B); positive = S1-B closer (S1-B win)
        de = abs(a - 1.0) - abs(b - 1.0)
        return sign * de

    def _safety_overshoot(metrics: dict[str, dict[str, float]]) -> float:
        b_acc = metrics["s1b"].get("root_acc_p95", float("nan"))
        b_jerk = metrics["s1b"].get("root_jerk_p95", float("nan"))
        if not (math.isfinite(b_acc) and math.isfinite(b_jerk)):
            return float("-inf")
        return max(b_acc, b_jerk)

    # chairs: 2 clips by S1-B pelvis/spine3 rotation-win.
    chairs_pool = sorted(
        by_subset["chairs"],
        key=lambda key: (
            _score(agg[key], "pelvis_rot6d_vel_mean", +1)
            + _score(agg[key], "spine3_rot6d_vel_mean", +1)
        ),
        reverse=True,
    )
    for key in chairs_pool[:2]:
        m = agg[key]
        selected.append({
            "subset": key[0],
            "seq_id": key[1],
            "rule": "chairs_top_s1b_rotation_win",
            "s1b_root_acc_xGT": m["s1b"].get("root_acc_p95", float("nan")),
            "s1b_root_jerk_xGT": m["s1b"].get("root_jerk_p95", float("nan")),
            "s1a_pelvis_rot_xGT": m["s1a"].get("pelvis_rot6d_vel_mean", float("nan")),
            "s1b_pelvis_rot_xGT": m["s1b"].get("pelvis_rot6d_vel_mean", float("nan")),
        })
    # imhd: 3 by combined rotation-win + S1-B safety blow-up (so we see
    # both effects in the same clip).
    imhd_pool = sorted(
        by_subset["imhd"],
        key=lambda key: (
            _score(agg[key], "pelvis_rot6d_vel_mean", +1)
            + 0.05 * _safety_overshoot(agg[key])
        ),
        reverse=True,
    )
    for key in imhd_pool[:3]:
        m = agg[key]
        selected.append({
            "subset": key[0],
            "seq_id": key[1],
            "rule": "imhd_combined_rotation_win_plus_safety_blowup",
            "s1b_root_acc_xGT": m["s1b"].get("root_acc_p95", float("nan")),
            "s1b_root_jerk_xGT": m["s1b"].get("root_jerk_p95", float("nan")),
            "s1a_pelvis_rot_xGT": m["s1a"].get("pelvis_rot6d_vel_mean", float("nan")),
            "s1b_pelvis_rot_xGT": m["s1b"].get("pelvis_rot6d_vel_mean", float("nan")),
        })
    # neuraldome: 3 worst S1-B root_jerk.
    neur_pool = sorted(
        by_subset["neuraldome"],
        key=lambda key: _safety_overshoot(agg[key]),
        reverse=True,
    )
    for key in neur_pool[:3]:
        m = agg[key]
        selected.append({
            "subset": key[0],
            "seq_id": key[1],
            "rule": "neuraldome_worst_s1b_jerk",
            "s1b_root_acc_xGT": m["s1b"].get("root_acc_p95", float("nan")),
            "s1b_root_jerk_xGT": m["s1b"].get("root_jerk_p95", float("nan")),
            "s1a_pelvis_rot_xGT": m["s1a"].get("pelvis_rot6d_vel_mean", float("nan")),
            "s1b_pelvis_rot_xGT": m["s1b"].get("pelvis_rot6d_vel_mean", float("nan")),
        })
    # omomo: 2 worst S1-B root_jerk.
    omo_pool = sorted(
        by_subset["omomo_correct_v2"],
        key=lambda key: _safety_overshoot(agg[key]),
        reverse=True,
    )
    for key in omo_pool[:2]:
        m = agg[key]
        selected.append({
            "subset": key[0],
            "seq_id": key[1],
            "rule": "omomo_worst_s1b_jerk",
            "s1b_root_acc_xGT": m["s1b"].get("root_acc_p95", float("nan")),
            "s1b_root_jerk_xGT": m["s1b"].get("root_jerk_p95", float("nan")),
            "s1a_pelvis_rot_xGT": m["s1a"].get("pelvis_rot6d_vel_mean", float("nan")),
            "s1b_pelvis_rot_xGT": m["s1b"].get("pelvis_rot6d_vel_mean", float("nan")),
        })
    return selected


# ============================================================================
# Cache + model loading
# ============================================================================


def _load_cache(cache_root: Path) -> dict[str, Any]:
    manifest = [
        json.loads(line)
        for line in (cache_root / "manifest_train.jsonl").read_text("utf-8").splitlines()
        if line.strip()
    ]
    clip_npz = np.load(cache_root / "text_embeddings_clip_vit_b32.npz", allow_pickle=True)
    clip_emb = clip_npz["embeddings"]
    text_index = json.loads(
        (cache_root / "text_embeddings_index.json").read_text("utf-8")
    )["index"]
    norm = json.loads((cache_root / "normalization_train.json").read_text("utf-8"))
    mean = np.asarray(norm["global"]["mean"], dtype=np.float32)
    std = np.asarray(norm["global"]["std_clamped"], dtype=np.float32)
    lookup: dict[tuple[str, str], int] = {}
    for i, r in enumerate(manifest):
        lookup[(r["subset"], r["seq_id"])] = i
    return {
        "manifest": manifest,
        "clip_emb": clip_emb,
        "text_index": text_index,
        "mean": mean,
        "std": std,
        "lookup": lookup,
    }


def _build_model_from_ckpt(ckpt_path: Path, device: torch.device) -> CoarsePriorDiff:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_d = ckpt["config"]
    diff = DiffusionConfig(
        num_steps=int(cfg_d["model"]["diffusion"]["num_steps"]),
        schedule=str(cfg_d["model"]["diffusion"]["schedule"]),
        objective="ddpm",
        prediction_target="x0",
    )
    den_d = cfg_d["model"]["denoiser"]
    den = CoarsePriorDenoiserConfig(
        coarse_dim=int(den_d["coarse_dim"]),
        text_dim=int(den_d["text_dim"]),
        init_pose_dim=int(den_d["init_pose_dim"]),
        d_model=int(den_d["d_model"]),
        n_layers=int(den_d["n_layers"]),
        n_heads=int(den_d["n_heads"]),
        ff_mult=int(den_d["ff_mult"]),
        dropout=float(den_d.get("dropout", 0.1)),
        max_seq_length=int(den_d["max_seq_length"]),
        attention_mode=str(den_d["attention_mode"]),
        block_size=int(den_d.get("block_size", 16)),
    )
    model = CoarsePriorDiff(CoarsePriorConfig(diffusion=diff, denoiser=den))
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model.to(device)


# ============================================================================
# Per-clip sampling
# ============================================================================


def _sample_clip(
    model: CoarsePriorDiff, cache: dict[str, Any], rec: dict[str, Any],
    sampler_seed: int, device: torch.device, max_T: int,
) -> np.ndarray:
    """Sample one denormalized Coarse-v1 trajectory for a given clip."""
    npz = np.load(cache["manifest_dir"] / rec["npz_path"], allow_pickle=False)
    init = npz["init_coarse_v1"].astype(np.float32)
    init_norm = (init - cache["mean"]) / cache["std"]
    T = min(int(rec["seq_len"]), max_T)

    text = rec.get("text", "")
    text_row = cache["text_index"].get(text, None)
    text_pool = (
        cache["clip_emb"][int(text_row)].astype(np.float32)
        if text_row is not None
        else np.zeros((512,), dtype=np.float32)
    )

    torch.manual_seed(sampler_seed)
    valid_mask = torch.ones(1, T, dtype=torch.bool, device=device)
    cond = {
        "text_pool": torch.from_numpy(text_pool).unsqueeze(0).to(device),
        "init_coarse": torch.from_numpy(init_norm).unsqueeze(0).to(device),
        "valid_mask": valid_mask,
    }
    with torch.no_grad():
        gen_norm = model.sample(
            shape=(1, T, COARSE_DIM), cond=cond, cfg_scale=1.0, device=device,
        )
    gen_np = gen_norm.squeeze(0).cpu().numpy()
    return gen_np * cache["std"] + cache["mean"]


def _gt_clip(cache: dict[str, Any], rec: dict[str, Any], max_T: int) -> np.ndarray:
    npz = np.load(cache["manifest_dir"] / rec["npz_path"], allow_pickle=False)
    gt = npz["coarse_v1"].astype(np.float32)
    T = min(int(rec["seq_len"]), gt.shape[0], max_T)
    return gt[:T]


# ============================================================================
# Plotting (9-panel curve plot)
# ============================================================================


def _root_deriv_norm(root: np.ndarray, order: int) -> np.ndarray:
    """Return ||d^k root|| of length T-k for k = order."""
    cur = root
    for _ in range(order):
        cur = np.diff(cur, axis=0)
    return np.linalg.norm(cur, axis=-1)


def _plot_clip(
    gt: np.ndarray, s1a: np.ndarray, s1b: np.ndarray,
    *, title: str, subtitle: str, out_path: Path,
) -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Tgt, Ta, Tb = gt.shape[0], s1a.shape[0], s1b.shape[0]
    T = min(Tgt, Ta, Tb)
    gt = gt[:T]; s1a = s1a[:T]; s1b = s1b[:T]

    fig, axes = plt.subplots(3, 3, figsize=(15, 11))

    # (0,0): root XZ trajectory  (x=ch0, z=ch2)
    ax = axes[0, 0]
    ax.plot(gt[:, 0], gt[:, 2], label="GT", color="C0", lw=1.5)
    ax.plot(s1a[:, 0], s1a[:, 2], label="S1-A", color="C1", ls="--", lw=1.2)
    ax.plot(s1b[:, 0], s1b[:, 2], label="S1-B", color="C2", ls=":", lw=1.2)
    ax.scatter([gt[0, 0]], [gt[0, 2]], color="C0", s=18, marker="o", zorder=5)
    ax.scatter([s1a[0, 0]], [s1a[0, 2]], color="C1", s=18, marker="o", zorder=5)
    ax.scatter([s1b[0, 0]], [s1b[0, 2]], color="C2", s=18, marker="o", zorder=5)
    ax.set_xlabel("root local trans X"); ax.set_ylabel("root local trans Z")
    ax.set_title("Root XZ trajectory (dot = frame 0)")
    ax.legend(fontsize=7); ax.set_aspect("equal", "datalim")
    ax.grid(True, alpha=0.3)

    # (0,1): root height (Y, channel 1)
    ax = axes[0, 1]
    ax.plot(gt[:, 1], label="GT", color="C0"); ax.plot(s1a[:, 1], label="S1-A", color="C1", ls="--")
    ax.plot(s1b[:, 1], label="S1-B", color="C2", ls=":")
    ax.set_xlabel("frame"); ax.set_ylabel("root_local_trans Y")
    ax.set_title("Root height (Y, ch1)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # (0,2): facing yaw
    yaw_gt = np.unwrap(np.arctan2(gt[:, 6], gt[:, 7]))
    yaw_a = np.unwrap(np.arctan2(s1a[:, 6], s1a[:, 7]))
    yaw_b = np.unwrap(np.arctan2(s1b[:, 6], s1b[:, 7]))
    ax = axes[0, 2]
    ax.plot(yaw_gt, label="GT", color="C0"); ax.plot(yaw_a, label="S1-A", color="C1", ls="--")
    ax.plot(yaw_b, label="S1-B", color="C2", ls=":")
    ax.set_xlabel("frame"); ax.set_ylabel("yaw (rad, unwrapped)")
    ax.set_title("Facing yaw (from sin/cos)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # (1,0): root velocity norm (frame-to-frame)
    rv_gt = _root_deriv_norm(gt[:, 0:3], 1)
    rv_a = _root_deriv_norm(s1a[:, 0:3], 1)
    rv_b = _root_deriv_norm(s1b[:, 0:3], 1)
    ax = axes[1, 0]
    ax.plot(rv_gt, label="GT", color="C0"); ax.plot(rv_a, label="S1-A", color="C1", ls="--")
    ax.plot(rv_b, label="S1-B", color="C2", ls=":")
    ax.set_xlabel("frame"); ax.set_ylabel("‖Δroot‖")
    ax.set_title("Root velocity (||Δroot||)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # (1,1): root acceleration
    ra_gt = _root_deriv_norm(gt[:, 0:3], 2)
    ra_a = _root_deriv_norm(s1a[:, 0:3], 2)
    ra_b = _root_deriv_norm(s1b[:, 0:3], 2)
    ax = axes[1, 1]
    ax.plot(ra_gt, label="GT", color="C0"); ax.plot(ra_a, label="S1-A", color="C1", ls="--")
    ax.plot(ra_b, label="S1-B", color="C2", ls=":")
    ax.set_xlabel("frame"); ax.set_ylabel("‖Δ²root‖")
    ax.set_title("Root acceleration")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # (1,2): root jerk
    rj_gt = _root_deriv_norm(gt[:, 0:3], 3)
    rj_a = _root_deriv_norm(s1a[:, 0:3], 3)
    rj_b = _root_deriv_norm(s1b[:, 0:3], 3)
    ax = axes[1, 2]
    ax.plot(rj_gt, label="GT", color="C0"); ax.plot(rj_a, label="S1-A", color="C1", ls="--")
    ax.plot(rj_b, label="S1-B", color="C2", ls=":")
    # Mark block boundaries every block_size=16 frames for the eye.
    for x in range(16, T, 16):
        ax.axvline(x, color="gray", alpha=0.18, lw=0.7)
    ax.set_xlabel("frame"); ax.set_ylabel("‖Δ³root‖")
    ax.set_title("Root jerk (gray = block-causal block boundaries)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # (2,0): pelvis rot6d frame-to-frame change norm (ch [9:15])
    pr_gt = np.linalg.norm(np.diff(gt[:, 9:15], axis=0), axis=-1) if T >= 2 else np.zeros(0)
    pr_a = np.linalg.norm(np.diff(s1a[:, 9:15], axis=0), axis=-1) if T >= 2 else np.zeros(0)
    pr_b = np.linalg.norm(np.diff(s1b[:, 9:15], axis=0), axis=-1) if T >= 2 else np.zeros(0)
    ax = axes[2, 0]
    ax.plot(pr_gt, label="GT", color="C0"); ax.plot(pr_a, label="S1-A", color="C1", ls="--")
    ax.plot(pr_b, label="S1-B", color="C2", ls=":")
    ax.set_xlabel("frame"); ax.set_ylabel("‖Δpelvis rot6d‖")
    ax.set_title("Pelvis rotation velocity")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # (2,1): head height (ch 21)
    ax = axes[2, 1]
    ax.plot(gt[:, 21], label="GT", color="C0"); ax.plot(s1a[:, 21], label="S1-A", color="C1", ls="--")
    ax.plot(s1b[:, 21], label="S1-B", color="C2", ls=":")
    ax.set_xlabel("frame"); ax.set_ylabel("head_height")
    ax.set_title("Head height (ch21)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # (2,2): shoulder height (ch 22)
    ax = axes[2, 2]
    ax.plot(gt[:, 22], label="GT", color="C0"); ax.plot(s1a[:, 22], label="S1-A", color="C1", ls="--")
    ax.plot(s1b[:, 22], label="S1-B", color="C2", ls=":")
    ax.set_xlabel("frame"); ax.set_ylabel("shoulder_height")
    ax.set_title("Shoulder height (ch22)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    fig.suptitle(title + "\n" + subtitle, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--s1a-ckpt", type=Path, default=S1A_CKPT)
    parser.add_argument("--s1b-ckpt", type=Path, default=S1B_CKPT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--sampler-seed", type=int, default=42)
    parser.add_argument("--max-T", type=int, default=196)
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    plots_dir = out_dir / "plots"
    traj_dir = out_dir / "trajectories"
    plots_dir.mkdir(parents=True, exist_ok=True)
    traj_dir.mkdir(parents=True, exist_ok=True)

    # ─── Clip selection (off the 12 eval JSONs) ──────────────────────
    print("[r15-vis] loading 12 eval JSONs for selection…")
    eval_payloads = _load_eval_payloads()
    per_clip_agg = _per_clip_aggregate_xgt(eval_payloads)
    selected = _select_clips(per_clip_agg)
    print(f"[r15-vis] selected {len(selected)} clips")

    selected_path = out_dir / "selected_clips.json"
    selected_path.write_text(
        json.dumps({
            "n_selected": len(selected),
            "sampler_seed": int(args.sampler_seed),
            "s1a_ckpt": str(args.s1a_ckpt),
            "s1b_ckpt": str(args.s1b_ckpt),
            "selected": selected,
        }, indent=2, default=float),
        encoding="utf-8",
    )
    print(f"[r15-vis] wrote {selected_path}")

    # ─── Load cache + ckpts ──────────────────────────────────────────
    print(f"[r15-vis] loading cache {args.cache_root}…")
    cache = _load_cache(args.cache_root)
    cache["manifest_dir"] = args.cache_root

    device = torch.device(args.device)
    print(f"[r15-vis] loading S1-A ckpt {args.s1a_ckpt}…")
    s1a_model = _build_model_from_ckpt(args.s1a_ckpt, device)
    print(f"[r15-vis] loading S1-B ckpt {args.s1b_ckpt}…")
    s1b_model = _build_model_from_ckpt(args.s1b_ckpt, device)

    # ─── Per-clip: sample + plot ─────────────────────────────────────
    csv_rows: list[str] = [
        "subset,seq_id,T,"
        "gt_root_jerk_p95,s1a_root_jerk_p95,s1b_root_jerk_p95,"
        "s1a_root_jerk_xGT,s1b_root_jerk_xGT,"
        "gt_root_acc_p95,s1a_root_acc_p95,s1b_root_acc_p95,"
        "s1a_root_acc_xGT,s1b_root_acc_xGT,"
        "rule"
    ]
    t0 = time.time()
    for i, sel in enumerate(selected):
        sub = sel["subset"]; sid = sel["seq_id"]
        key = (sub, sid)
        if key not in cache["lookup"]:
            print(f"  [warn] {sub}/{sid} missing from cache; skipping")
            continue
        idx = cache["lookup"][key]
        rec = cache["manifest"][idx]
        T_eff = min(int(rec["seq_len"]), args.max_T)
        gt = _gt_clip(cache, rec, args.max_T)
        print(f"  [r15-vis] clip {i + 1}/{len(selected)}  {sub}/{sid}  T={T_eff}")
        gen_a = _sample_clip(s1a_model, cache, rec, args.sampler_seed, device, args.max_T)
        gen_b = _sample_clip(s1b_model, cache, rec, args.sampler_seed, device, args.max_T)

        # Save trajectories so the report (and future Stage-2 work) can re-plot
        # without re-running the sampler.
        safe_sid = sid.replace("/", "_").replace(" ", "_")
        npz_path = traj_dir / f"{sub}_{safe_sid}.npz"
        np.savez_compressed(
            npz_path,
            gt=gt.astype(np.float32),
            s1a=gen_a[:T_eff].astype(np.float32),
            s1b=gen_b[:T_eff].astype(np.float32),
            text=str(rec.get("text", "")),
            T=int(T_eff),
        )

        # Aggregate per-clip xGT for the CSV.
        def _pct95_jerk(x: np.ndarray) -> float:
            T = x.shape[0]
            if T < 4: return 0.0
            return float(np.percentile(_root_deriv_norm(x[:, 0:3], 3), 95))
        def _pct95_acc(x: np.ndarray) -> float:
            T = x.shape[0]
            if T < 3: return 0.0
            return float(np.percentile(_root_deriv_norm(x[:, 0:3], 2), 95))
        gt_jerk = _pct95_jerk(gt); a_jerk = _pct95_jerk(gen_a[:T_eff]); b_jerk = _pct95_jerk(gen_b[:T_eff])
        gt_acc = _pct95_acc(gt); a_acc = _pct95_acc(gen_a[:T_eff]); b_acc = _pct95_acc(gen_b[:T_eff])
        a_jerk_x = a_jerk / gt_jerk if gt_jerk > 1e-6 else float("nan")
        b_jerk_x = b_jerk / gt_jerk if gt_jerk > 1e-6 else float("nan")
        a_acc_x = a_acc / gt_acc if gt_acc > 1e-6 else float("nan")
        b_acc_x = b_acc / gt_acc if gt_acc > 1e-6 else float("nan")
        csv_rows.append(
            f"{sub},{sid},{T_eff},"
            f"{gt_jerk:.5f},{a_jerk:.5f},{b_jerk:.5f},"
            f"{a_jerk_x:.3f},{b_jerk_x:.3f},"
            f"{gt_acc:.5f},{a_acc:.5f},{b_acc:.5f},"
            f"{a_acc_x:.3f},{b_acc_x:.3f},"
            f"{sel['rule']}"
        )

        # Plot.
        text = str(rec.get("text", ""))[:120]
        title = f"{sub}/{sid}   T={T_eff}   sampler_seed={args.sampler_seed}"
        subtitle = (
            f"rule={sel['rule']}  "
            f"S1-A jerk xGT={a_jerk_x:.2f}  S1-B jerk xGT={b_jerk_x:.2f}  "
            f"S1-A acc xGT={a_acc_x:.2f}  S1-B acc xGT={b_acc_x:.2f}\n"
            f"text: {text}"
        )
        plot_path = plots_dir / f"{sub}_{safe_sid}.png"
        _plot_clip(gt, gen_a[:T_eff], gen_b[:T_eff],
                   title=title, subtitle=subtitle, out_path=plot_path)
    elapsed = time.time() - t0
    print(f"[r15-vis] sampling+plotting done in {elapsed:.1f}s")

    # ─── CSV ─────────────────────────────────────────────────────────
    csv_path = out_dir / "metrics_table.csv"
    csv_path.write_text("\n".join(csv_rows) + "\n", encoding="utf-8")
    print(f"[r15-vis] wrote {csv_path}")

    # ─── README ──────────────────────────────────────────────────────
    readme_lines = [
        "# Round-15 Stage-1 Visual Review",
        "",
        "Selected 10 clips from the Round-9 fixed selection to compare GT vs",
        "S1-A vs S1-B coarse-motion trajectories (per-channel curves).",
        "",
        f"- Sampler seed: {args.sampler_seed}",
        f"- DDPM steps: 1 000 (config default)",
        f"- ckpt: S1-A `{args.s1a_ckpt}` / S1-B `{args.s1b_ckpt}`",
        f"- Cache: `{args.cache_root}`",
        f"- Wall-clock: {elapsed:.1f}s",
        "",
        "## Files",
        "",
        "- `selected_clips.json` — selection rule + per-clip eval-aggregate xGT.",
        "- `metrics_table.csv` — per-clip GT/S1-A/S1-B root acc/jerk + xGT (single sampler seed).",
        "- `plots/<subset>_<seq_id>.png` — 9-panel curve panels.",
        "- `trajectories/<subset>_<seq_id>.npz` — raw GT + S1-A + S1-B coarse-v1 arrays.",
        "",
        "## Selection rules",
        "",
        "- chairs (2): largest S1-B paired-improvement on pelvis_rot6d_vel + spine3_rot6d_vel",
        "  (the metric class S1-B wins on chairs).",
        "- imhd (3): combined S1-B rotation-win AND S1-B root_jerk blow-up (worst-case",
        "  for the safety gate).",
        "- neuraldome (3): worst S1-B root_jerk blow-up (where the safety gate fails hardest).",
        "- omomo (2): worst S1-B root_jerk blow-up.",
        "",
        "## What to look for in the plots",
        "",
        "- **Root jerk panel** (centre-right): S1-B should show extreme high-frequency",
        "  oscillation on imhd/neuraldome/omomo clips. Grey vertical lines mark the",
        "  block-causal block boundaries (every 16 frames); spikes aligned with",
        "  boundaries would be the textbook block-causal artefact, but if spikes are",
        "  uniformly distributed the failure mode is not boundary-driven.",
        "- **Pelvis rotation velocity panel** (bottom-left): S1-A flat / GT-undershooting",
        "  is the v18 frozen-body behaviour; S1-B catching up to GT magnitude is the",
        "  intended block-causal win.",
        "- **Root XZ trajectory**: S1-A under-traversing GT path = under-motion failure.",
        "  S1-B oscillating around GT path = root-noise failure.",
        "",
    ]
    (out_dir / "README_visual_review.md").write_text(
        "\n".join(readme_lines), encoding="utf-8",
    )
    print(f"[r15-vis] wrote {out_dir / 'README_visual_review.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
