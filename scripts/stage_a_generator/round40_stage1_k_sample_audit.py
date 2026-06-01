"""Round-40 Stage-1 K-sample diversity audit.

Sample a Stage-1 ckpt K times with different seeds (same cond), then
measure pairwise diversity to determine whether the diffusion noise is
being used as a mode-selector or whether all seeds converge to the same
plan (i.e. classic mode collapse).

If pairwise diversity is near zero for every variant, plan-energy alone
is not enough — a later round needs an explicit mode token / latent /
best-of-K training.

Cache layout per seed (matches ``sample_substitute_conds``):
    <out-dir>/samples/seed<S>/<bucket>/<subset>/<seq_id>.npz
        stage1_coarse : (T, 23) z-scored
        valid_T       : int

Outputs:
    <out-dir>/k_sample_stats.json
    <out-dir>/k_sample_summary.md

Run:
    python -u scripts/stage_a_generator/round40_stage1_k_sample_audit.py \\
        --config configs/training/stage1_r40_c2_plan_energy.yaml \\
        --ckpt   runs/training/stage1_r40_c2_plan_energy/final.pt \\
        --selection-json analyses/round29_val_diag_indices_48_balanced.json \\
        --bucket val \\
        --out-dir analyses/round40_stage1_kdiv_c2_plan_energy \\
        --num-samples 8 \\
        --seeds 41,42,43,44,45,46,47,48 \\
        --cfg-scale 1.0 \\
        --sampler ddim_eta0
"""
from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from piano.data.stage1_coarse_oracle import load_stage1_coarse_norm
from piano.inference.sample_substitute_conds import (
    _read_selection,
    sample_substitute_conds,
)


STAGE1_COARSE_DIM = 23


def _parse_seeds(s: str | None, num_samples: int) -> list[int]:
    if s is None or not s.strip():
        return list(range(41, 41 + num_samples))
    seeds = [int(x.strip()) for x in s.split(",") if x.strip()]
    if len(seeds) != num_samples:
        raise SystemExit(
            f"--seeds has {len(seeds)} entries but --num-samples is "
            f"{num_samples}"
        )
    return seeds


def _load_seed_cache(
    out_dir: Path, seed: int, bucket: str, subset: str, seq_id: str,
) -> tuple[np.ndarray, int] | None:
    p = (
        out_dir / "samples" / f"seed{seed}" / bucket / subset / f"{seq_id}.npz"
    )
    if not p.exists():
        return None
    data = np.load(p)
    if "stage1_coarse" not in data.files:
        return None
    arr = data["stage1_coarse"].astype(np.float32)
    valid_T = (
        int(data["valid_T"]) if "valid_T" in data.files else arr.shape[0]
    )
    return arr, valid_T


def _pairwise_clip_metrics(
    samples: list[np.ndarray],            # K samples, each (T_valid, 23) z-scored
    raw_samples: list[np.ndarray],        # K samples, raw un-z-scored
) -> dict[str, float]:
    """Pairwise diversity stats over K samples for one clip.

    Returns dict with means + maxes of pairwise distances.
    """
    K = len(samples)
    if K < 2:
        return {}
    rms_root_path = []
    rms_final_disp = []
    yaw_range_diff = []
    rms_pelvis_rot = []
    for i, j in combinations(range(K), 2):
        a = raw_samples[i]
        b = raw_samples[j]
        T = min(a.shape[0], b.shape[0])
        if T < 2:
            continue
        # Root XZ path in oracle order (root_local x at 0, z at 1).
        path_a = a[:T, :2]
        path_b = b[:T, :2]
        rms_root_path.append(
            float(np.sqrt(np.mean(np.square(path_a - path_b))))
        )
        rms_final_disp.append(
            float(
                math.hypot(
                    path_a[-1, 0] - path_b[-1, 0],
                    path_a[-1, 1] - path_b[-1, 1],
                )
            )
        )
        # Yaw cumulative range (unwrapped).
        twopi = 2.0 * math.pi
        ya = np.arctan2(a[:T, 6], a[:T, 7])
        yb = np.arctan2(b[:T, 6], b[:T, 7])
        d_ya = np.diff(ya)
        d_yb = np.diff(yb)
        d_ya = (d_ya + math.pi) % twopi - math.pi
        d_yb = (d_yb + math.pi) % twopi - math.pi
        cs_a = np.cumsum(d_ya)
        cs_b = np.cumsum(d_yb)
        range_a = float(cs_a.max() - cs_a.min()) if cs_a.size else 0.0
        range_b = float(cs_b.max() - cs_b.min()) if cs_b.size else 0.0
        yaw_range_diff.append(abs(range_a - range_b))
        # Pelvis rot6d block RMS.
        rms_pelvis_rot.append(
            float(np.sqrt(np.mean(np.square(a[:T, 9:15] - b[:T, 9:15]))))
        )

    def _summary(xs: list[float]) -> tuple[float, float]:
        if not xs:
            return 0.0, 0.0
        arr = np.array(xs, dtype=np.float64)
        return float(arr.mean()), float(arr.max())

    rms_path_mean, rms_path_max = _summary(rms_root_path)
    final_mean, final_max = _summary(rms_final_disp)
    yaw_mean, yaw_max = _summary(yaw_range_diff)
    pel_mean, pel_max = _summary(rms_pelvis_rot)
    return {
        "pair_root_path_rms_mean": rms_path_mean,
        "pair_root_path_rms_max": rms_path_max,
        "pair_final_disp_mean": final_mean,
        "pair_final_disp_max": final_max,
        "pair_yaw_range_diff_mean": yaw_mean,
        "pair_yaw_range_diff_max": yaw_max,
        "pair_pelvis_rot6d_rms_mean": pel_mean,
        "pair_pelvis_rot6d_rms_max": pel_max,
    }


def _aggregate(per_clip: list[dict[str, float]]) -> dict[str, float]:
    if not per_clip:
        return {}
    out: dict[str, float] = {}
    keys = sorted({k for c in per_clip for k in c.keys()})
    for k in keys:
        vals = np.array(
            [c[k] for c in per_clip if k in c], dtype=np.float64,
        )
        out[k + "_mean"] = float(vals.mean()) if vals.size else 0.0
        out[k + "_max"] = float(vals.max()) if vals.size else 0.0
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--selection-json", type=Path, required=True)
    ap.add_argument("--bucket", choices=["train", "val"], default="val")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument(
        "--seeds", type=str, default="41,42,43,44,45,46,47,48",
        help="Comma-separated list of seeds; length must match --num-samples.",
    )
    ap.add_argument("--cfg-scale", type=float, default=1.0)
    ap.add_argument(
        "--sampler", choices=["ddpm", "ddim_eta0", "ddpm_det"],
        default="ddim_eta0",
    )
    ap.add_argument(
        "--skip-sample", action="store_true",
        help="Reuse existing per-seed sample dirs (re-run metrics only).",
    )
    args = ap.parse_args()

    cfg = OmegaConf.load(str(args.config))
    seeds = _parse_seeds(args.seeds, args.num_samples)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    samples_root = args.out_dir / "samples"
    samples_root.mkdir(parents=True, exist_ok=True)

    # Sample K times with different seeds. Each call uses a per-seed dir.
    for seed in seeds:
        seed_dir = samples_root / f"seed{seed}" / args.bucket
        marker = seed_dir / ".done"
        if args.skip_sample and marker.exists():
            print(f"[kdiv] seed {seed}: --skip-sample, marker present, reusing")
            continue
        seed_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[kdiv] sampling seed {seed} → {seed_dir}  "
            f"(cfg_scale={args.cfg_scale}, sampler={args.sampler})"
        )
        n_written = sample_substitute_conds(
            config_path=args.config,
            ckpt_path=args.ckpt,
            selection_json=args.selection_json,
            out_dir=seed_dir,
            bucket=args.bucket,
            stage="stage1",
            upstream_dir=None,
            seed=seed,
            cfg_scale=args.cfg_scale,
            sampler=args.sampler,
        )
        marker.write_text(f"n_written={n_written}\n", encoding="utf-8")

    # Load mean/std for un-z-scoring (kept for raw-space pairwise metrics).
    mean_np, std_np = load_stage1_coarse_norm(
        str(cfg.data.stage1_coarse_cache_root),
    )
    mean = mean_np.astype(np.float32).reshape(1, STAGE1_COARSE_DIM)
    std = std_np.astype(np.float32).reshape(1, STAGE1_COARSE_DIM)

    # Walk the selection and compute pairwise diversity per clip.
    sel_pairs = _read_selection(args.selection_json)
    per_clip_pair: list[dict[str, float]] = []
    n_clips_with_full_K = 0
    for subset, seq_id in sorted(sel_pairs):
        samples: list[np.ndarray] = []
        raw_samples: list[np.ndarray] = []
        for seed in seeds:
            payload = _load_seed_cache(
                args.out_dir, seed, args.bucket, subset, seq_id,
            )
            if payload is None:
                continue
            arr_z, vt = payload
            arr_z = arr_z[:vt]
            arr_raw = arr_z * std + mean
            samples.append(arr_z)
            raw_samples.append(arr_raw)
        if len(samples) < 2:
            continue
        if len(samples) == len(seeds):
            n_clips_with_full_K += 1
        per_clip_pair.append(_pairwise_clip_metrics(samples, raw_samples))

    summary = _aggregate(per_clip_pair)
    stats = {
        "config": str(args.config),
        "ckpt": str(args.ckpt),
        "selection_json": str(args.selection_json),
        "bucket": args.bucket,
        "num_samples": args.num_samples,
        "seeds": seeds,
        "cfg_scale": args.cfg_scale,
        "sampler": args.sampler,
        "n_clips_with_full_K": n_clips_with_full_K,
        "n_clips_evaluated": len(per_clip_pair),
        "pairwise_summary": summary,
    }

    out_json = args.out_dir / "k_sample_stats.json"
    out_md = args.out_dir / "k_sample_summary.md"
    out_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    md = [
        "# Round-40 Stage-1 K-sample diversity audit",
        "",
        f"- config: `{args.config}`",
        f"- ckpt: `{args.ckpt}`",
        f"- bucket: {args.bucket}",
        f"- num_samples: {args.num_samples}",
        f"- seeds: {seeds}",
        f"- cfg_scale: {args.cfg_scale}",
        f"- sampler: {args.sampler}",
        f"- n_clips_with_full_K: {n_clips_with_full_K}",
        f"- n_clips_evaluated: {len(per_clip_pair)}",
        "",
        "## Pairwise diversity (averaged across clip × pair)",
        "",
        "| metric | mean | max |",
        "|---|---:|---:|",
    ]
    for k in (
        "pair_root_path_rms",
        "pair_final_disp",
        "pair_yaw_range_diff",
        "pair_pelvis_rot6d_rms",
    ):
        if f"{k}_mean_mean" not in summary:
            continue
        md.append(
            f"| {k} | {summary[k + '_mean_mean']:.4f} | "
            f"{summary[k + '_max_max']:.4f} |"
        )

    md.extend([
        "",
        "## Decision",
        "",
        "Read the numbers above with the R40 handoff §8.3 thresholds:",
        "",
        "- If pair_root_path_rms is < ~0.05 m and pair_final_disp_max is < ~0.10 m,",
        "  the K samples converge — diffusion noise is not being used as a",
        "  mode-selector. R41+ needs an explicit mode token / best-of-K.",
        "- If pair_root_path_rms is > ~0.10 m AND plan-energy stats separate",
        "  good/bad samples, a later reranking path is viable.",
    ])
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {out_md}")
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
