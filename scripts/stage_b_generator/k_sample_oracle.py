#!/usr/bin/env python
"""K-sample contact oracle for Stage B.

This no-retrain diagnostic asks whether the current generator distribution
already contains good-contact samples. For each fixed validation clip, it
generates K full-condition samples with different RNG seeds, scores each sample
with the same body-to-object contact metric used by Stage B evaluation, and
reports single-sample versus best-of-K contact.

If best-of-K approaches GT roundtrip while sample #0 remains poor, the next
strategy should be reranking/guidance. If best-of-K remains near the current
single-sample band, the learned distribution itself is wrong.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

import piano.models.backbones.momask_adapter  # noqa: F401
from piano.data.eval_sampling import (
    describe_eval_clip_selection,
    resolve_eval_clip_count,
    select_eval_clip_indices,
)
from piano.data.humanml3d_repr import load_motion_stats
from piano.training.contact_eval import compute_clip_contact_distance
from piano.utils.io_utils import ensure_dir, save_json

# Reuse the load-bearing Stage B offline eval helpers. This file lives in the
# same directory as qual_eval.py, so direct script execution puts that directory
# on sys.path.
from qual_eval import (  # type: ignore
    _build_model,
    _build_val_dataset,
    _generate,
    _get_canon_to_world_transform,
    _save_condition_dir,
    _tokenize_z_int,
)


def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cm(x_m: float) -> float:
    return round(float(x_m) * 100.0, 4)


def _summary_stats(values_m: list[float]) -> dict[str, float]:
    if not values_m:
        return {
            "mean_m": 0.0,
            "median_m": 0.0,
            "p25_m": 0.0,
            "p75_m": 0.0,
            "mean_cm": 0.0,
            "median_cm": 0.0,
            "p25_cm": 0.0,
            "p75_cm": 0.0,
        }
    arr = np.asarray(values_m, dtype=np.float64)
    return {
        "mean_m": round(float(arr.mean()), 6),
        "median_m": round(float(np.median(arr)), 6),
        "p25_m": round(float(np.percentile(arr, 25)), 6),
        "p75_m": round(float(np.percentile(arr, 75)), 6),
        "mean_cm": _cm(float(arr.mean())),
        "median_cm": _cm(float(np.median(arr))),
        "p25_cm": _cm(float(np.percentile(arr, 25))),
        "p75_cm": _cm(float(np.percentile(arr, 75))),
    }


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    single = [float(r["single_sample_dist_m"]) for r in rows]
    best = [float(r["best_dist_m"]) for r in rows]
    sample_means = [float(r["sample_mean_dist_m"]) for r in rows]
    improvements = [s - b for s, b in zip(single, best)]

    return {
        "n_clips": len(rows),
        "single_sample": _summary_stats(single),
        "sample_mean": _summary_stats(sample_means),
        "best_of_k": _summary_stats(best),
        "improvement_single_minus_best": _summary_stats(improvements),
        "best_under_25cm_frac": (
            round(float(np.mean(np.asarray(best) <= 0.25)), 4) if best else 0.0
        ),
        "best_under_22cm_frac": (
            round(float(np.mean(np.asarray(best) <= 0.22)), 4) if best else 0.0
        ),
    }


def _aggregate_by_subset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_subset: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_subset.setdefault(str(row["subset"]), []).append(row)
    return {
        subset: _aggregate_rows(sub_rows)
        for subset, sub_rows in sorted(by_subset.items())
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "runs/sweeps/stageB_v12_decoded_contact_weight_sweep/configs/"
            "generator_v12_decoded_contact_w02_diagnostics.yaml",
        ),
        help="Training config that matches the checkpoint.",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=Path("runs/training/generator_v12_decoded_contact_w02_diagnostics/best_val.pt"),
        help="Stage B checkpoint to diagnose.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/eval/stageB_v0_12_w02_bv_k_sample_oracle"))
    parser.add_argument("--num-clips", type=int, default=80)
    parser.add_argument(
        "--num-clips-per-subset",
        type=int,
        default=20,
        help="When metadata is available, overrides --num-clips as N per subset.",
    )
    parser.add_argument("--k", type=int, default=16, help="Number of samples per clip.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--w-text", type=float, default=4.0)
    parser.add_argument("--w-int", type=float, default=2.0)
    parser.add_argument("--timesteps", type=int, default=10)
    parser.add_argument("--res-cond-scale", type=float, default=2.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--save-best",
        action="store_true",
        help="Save best-of-K motions as output-dir/best/generated.npz for visualization.",
    )
    args = parser.parse_args()

    if args.k <= 0:
        raise ValueError("--k must be positive")

    _set_all_seeds(args.seed)
    ensure_dir(args.output_dir)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = OmegaConf.load(args.config)

    print(f"Loading model from {args.ckpt} on {device} ...")
    transformer, vq_model, res_transformer, token_stride = _build_model(cfg, args.ckpt, device)
    motion_mean, motion_std = load_motion_stats(cfg.model.checkpoints.vq_vae)

    val_dataset = _build_val_dataset(cfg)
    num_clips = resolve_eval_clip_count(
        val_dataset,
        num_clips=int(args.num_clips),
        num_clips_per_subset=(
            int(args.num_clips_per_subset)
            if args.num_clips_per_subset is not None and int(args.num_clips_per_subset) > 0
            else None
        ),
    )
    sampled_idx = select_eval_clip_indices(val_dataset, num_clips, seed=args.seed)
    selected_rows = describe_eval_clip_selection(val_dataset, sampled_idx)
    samples = [val_dataset[i] for i in sampled_idx]

    print(f"Selected {len(samples)} stratified clips; K={args.k}")
    for row in selected_rows:
        print(
            "  "
            f"idx={row['index']} subset={row['subset']} "
            f"object={row['object_id']} seq={row['seq_id']}"
        )

    from utils.motion_process import recover_from_ric

    per_clip: list[dict[str, Any]] = []
    best_save_rows: list[dict[str, dict[str, Any]]] = []
    texts: list[str] = []
    seq_ids: list[str] = []
    seq_lens_frames: list[int] = []
    object_pcs: list[np.ndarray] = []
    object_positions: list[np.ndarray] = []
    object_rotations: list[np.ndarray] = []
    world_R_y: list[float] = []
    world_T_xz: list[np.ndarray] = []

    for clip_i, sample in enumerate(samples):
        seq_len = int(sample["seq_len"].item())
        if seq_len < token_stride:
            print(f"  [{clip_i + 1}/{len(samples)}] skip short clip seq_len={seq_len}")
            continue

        text = str(sample["text"])
        seq_id = str(sample["seq_id"])
        subset = str(selected_rows[clip_i]["subset"])
        object_id = str(selected_rows[clip_i]["object_id"])
        m_lens_tok = torch.tensor([max(1, seq_len // token_stride)], dtype=torch.long, device=device)
        int_kv, int_pad = _tokenize_z_int(transformer, sample, device)

        joints_world = sample["joints"].cpu().numpy().astype(np.float32)
        motion_src = sample["motion"].cpu().numpy().astype(np.float32)
        R_y, T_xz = _get_canon_to_world_transform(joints_world[:seq_len], motion_src[:seq_len])
        obj_pc = sample["object_pc"].cpu().numpy().astype(np.float32)
        obj_pos = (
            sample["object_positions"].cpu().numpy().astype(np.float32)
            if "object_positions" in sample
            else np.zeros((seq_len, 3), dtype=np.float32)
        )
        obj_rot = (
            sample["object_rotations"].cpu().numpy().astype(np.float32)
            if "object_rotations" in sample
            else np.zeros((seq_len, 3), dtype=np.float32)
        )

        sample_rows: list[dict[str, Any]] = []
        best_motion: np.ndarray | None = None
        best_base: np.ndarray | None = None
        best_dist = float("inf")
        best_sample_idx = -1
        best_seed = -1

        print(f"  [{clip_i + 1}/{len(samples)}] {seq_id} subset={subset}")
        for k_i in range(args.k):
            sample_seed = int(args.seed + clip_i * 100_000 + k_i)
            _set_all_seeds(sample_seed)
            motion_gen, base_ids = _generate(
                transformer,
                vq_model,
                res_transformer,
                text=text,
                int_kv=int_kv,
                int_pad=int_pad,
                m_lens_tok=m_lens_tok,
                w_text=float(args.w_text),
                w_int=float(args.w_int),
                motion_mean=motion_mean,
                motion_std=motion_std,
                timesteps=int(args.timesteps),
                res_cond_scale=float(args.res_cond_scale),
                device=device,
            )
            dist_m = compute_clip_contact_distance(
                motion_263_generated=motion_gen,
                R_y_angle=R_y,
                T_xz=T_xz,
                object_pc_local=obj_pc,
                object_positions=obj_pos,
                object_rotations=obj_rot,
                seq_len=seq_len,
                recover_from_ric_fn=recover_from_ric,
            )
            sample_rows.append({
                "sample_index": k_i,
                "seed": sample_seed,
                "dist_m": round(float(dist_m), 6),
                "dist_cm": _cm(float(dist_m)),
            })
            if dist_m < best_dist:
                best_dist = float(dist_m)
                best_sample_idx = k_i
                best_seed = sample_seed
                best_motion = motion_gen
                best_base = base_ids

        dists = [float(r["dist_m"]) for r in sample_rows]
        row = {
            "index": int(selected_rows[clip_i]["index"]),
            "subset": subset,
            "object_id": object_id,
            "seq_id": seq_id,
            "text": text,
            "seq_len": seq_len,
            "k": int(args.k),
            "samples": sample_rows,
            "single_sample_dist_m": round(float(dists[0]), 6),
            "single_sample_dist_cm": _cm(float(dists[0])),
            "sample_mean_dist_m": round(float(np.mean(dists)), 6),
            "sample_mean_dist_cm": _cm(float(np.mean(dists))),
            "sample_median_dist_m": round(float(np.median(dists)), 6),
            "sample_median_dist_cm": _cm(float(np.median(dists))),
            "best_sample_index": int(best_sample_idx),
            "best_seed": int(best_seed),
            "best_dist_m": round(float(best_dist), 6),
            "best_dist_cm": _cm(float(best_dist)),
            "improvement_single_minus_best_m": round(float(dists[0] - best_dist), 6),
            "improvement_single_minus_best_cm": _cm(float(dists[0] - best_dist)),
        }
        per_clip.append(row)
        print(
            f"    single={row['single_sample_dist_cm']:.2f}cm "
            f"mean={row['sample_mean_dist_cm']:.2f}cm "
            f"best={row['best_dist_cm']:.2f}cm "
            f"(sample {best_sample_idx}, seed {best_seed})"
        )

        if args.save_best:
            assert best_motion is not None
            assert best_base is not None
            best_save_rows.append({
                "full": {
                    "motion": best_motion,
                    "base": best_base,
                    "swap_from": None,
                },
            })
            texts.append(text)
            seq_ids.append(seq_id)
            seq_lens_frames.append(seq_len)
            object_pcs.append(obj_pc)
            object_positions.append(obj_pos)
            object_rotations.append(obj_rot)
            world_R_y.append(float(R_y))
            world_T_xz.append(np.asarray(T_xz, dtype=np.float32))

    summary = {
        "diagnostic": "stageB_k_sample_oracle",
        "config": str(args.config),
        "ckpt": str(args.ckpt),
        "seed": int(args.seed),
        "k": int(args.k),
        "num_clips_requested": int(args.num_clips),
        "num_clips_per_subset": int(args.num_clips_per_subset),
        "num_clips_evaluated": len(per_clip),
        "w_text": float(args.w_text),
        "w_int": float(args.w_int),
        "timesteps": int(args.timesteps),
        "res_cond_scale": float(args.res_cond_scale),
        "clip_selection": selected_rows,
        "aggregate": _aggregate_rows(per_clip),
        "by_subset": _aggregate_by_subset(per_clip),
        "per_clip": per_clip,
    }
    save_json(args.output_dir / "summary.json", summary)
    print(f"Wrote {args.output_dir / 'summary.json'}")

    if args.save_best and best_save_rows:
        best_dir = args.output_dir / "best"
        _save_condition_dir(
            best_dir,
            best_save_rows,
            "full",
            texts,
            seq_lens_frames,
            seq_ids,
            object_pcs=object_pcs,
            object_positions=object_positions,
            object_rotations=object_rotations,
            world_R_y=world_R_y,
            world_T_xz=world_T_xz,
        )
        print(f"Wrote best-of-K visualization run to {best_dir}")

    agg = summary["aggregate"]
    print("\n=== K-sample oracle ===")
    print(f"clips: {agg['n_clips']}  K: {args.k}")
    print(f"single sample mean: {agg['single_sample']['mean_cm']:.2f} cm")
    print(f"sample mean:        {agg['sample_mean']['mean_cm']:.2f} cm")
    print(f"best-of-K mean:     {agg['best_of_k']['mean_cm']:.2f} cm")
    print(
        "single-best gain:  "
        f"{agg['improvement_single_minus_best']['mean_cm']:.2f} cm"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
