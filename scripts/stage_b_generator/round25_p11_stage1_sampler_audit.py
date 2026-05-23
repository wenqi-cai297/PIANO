"""Round-25 P1.1: Stage-1 sampler procedure audit.

For each clip in the D1 multimodal eval subset, sample Stage-1 S1-O
under a matrix of (cfg_scale_stage1, num_steps), then measure the
sampled-vs-oracle coarse-v1 reconstruction quality (23-D L2) per
(subset, mode_category).

Goal: identify whether the +12.7cm D3 oracle-vs-sampled gap is fixable
by changing inference-time sampler config, or whether Stage-1 needs
retraining (much more expensive).

Decision rule:
  - If some (cfg, steps) combination shrinks the gap by >30%,
    cheap fix possible: ship updated inference config for v26.
  - If all (cfg, steps) combinations give similar gap, Stage-1 model
    itself is the limit → retrain candidate (P1.1 step 4).

Outputs:
  - analyses/round25_p11_stage1_sampler_audit.json
    Per (clip, cfg, steps): coarse_v1 L2 (sampled vs GT-derived),
    plus per-channel breakdown.
  - analyses/round25_p11_stage1_sampler_audit.md
    Heatmap-style table per subset × per (cfg, steps), best cell
    highlighted.

Design source:
  analyses/2026-05-23_round25_p0_synthesis.md §7 P1.1.

Usage (server):
  conda run --no-capture-output -n piano python -u \
      scripts/stage_b_generator/round25_p11_stage1_sampler_audit.py \
      --config configs/training/anchordiff_v26_FULL_DATA_local.yaml \
      --stage1-ckpt runs/training/stage1_s1o_round20_seed42/final.pt \
      --stage1-cache-root cache/stage1_coarse_v1_objtraj_root0_world_round18_fix \
      --selection-json analyses/round25_multimodal_eval_subset.json \
      --cfg-scale-grid 0.5,1.0,2.0,4.0 \
      --num-steps-grid 10,50,100,200,1000 \
      --output analyses/round25_p11_stage1_sampler_audit.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from plan_condition_diagnostics import _build_dataset  # noqa: E402
from eval_stage1_coarse_prior import (  # noqa: E402
    COARSE_DIM as STAGE1_COARSE_DIM,
    _build_model_from_ckpt as build_stage1_model,
)
from round25_d2_diversity_diagnostic import _filter_dataset_by_selection  # noqa: E402
from round25_d3_oracle_vs_sampled import _load_stage1_cache  # noqa: E402

from piano.data.dataset import collate_hoi  # noqa: E402
from piano.data.stage1_coarse_oracle import extract_coarse_v1_batched  # noqa: E402


def _l2_per_channel(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-channel RMS error. pred, gt: (T, 23) raw (un-normalized)."""
    # match T length defensively
    T = min(pred.shape[0], gt.shape[0])
    diff = pred[:T] - gt[:T]
    return np.sqrt((diff ** 2).mean(axis=0))   # (23,)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True,
                        help="v26 (Stage-2) config — used only for dataset construction.")
    parser.add_argument("--stage1-ckpt", type=Path, required=True)
    parser.add_argument("--stage1-cache-root", type=Path,
                        default=Path("cache/stage1_coarse_v1_objtraj_root0_world_round18_fix"))
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--bucket", default="val", choices=["train", "val"])
    parser.add_argument("--cfg-scale-grid", default="0.5,1.0,2.0,4.0",
                        help="Comma-separated cfg_scale values.")
    parser.add_argument("--num-steps-grid", default="50,100,200,1000",
                        help="Comma-separated num_steps values. 1000 = the Stage-1 ckpt's native count.")
    parser.add_argument("--seeds", default="42,43,44",
                        help="Comma-separated seeds (each (cfg, steps) sampled per seed; report mean).")
    parser.add_argument("--max-clips", type=int, default=0,
                        help="0 = use all matched clips; >0 caps for quick smoke.")
    parser.add_argument("--output", type=Path,
                        default=Path("analyses/round25_p11_stage1_sampler_audit.json"))
    args = parser.parse_args()

    cfg_scales = [float(x) for x in args.cfg_scale_grid.split(",")]
    nstep_grid = [int(x) for x in args.num_steps_grid.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    n_combos = len(cfg_scales) * len(nstep_grid) * len(seeds)

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load Stage-1 model + cache ----
    print(f"[p11] loading Stage-1 ckpt {args.stage1_ckpt} ...")
    stage1_ckpt = torch.load(args.stage1_ckpt, map_location="cpu", weights_only=False)
    stage1_model, loaded_ema = build_stage1_model(stage1_ckpt, prefer_ema=True)
    stage1_model = stage1_model.to(device).eval()
    print(f"[p11] Stage-1 EMA loaded: {loaded_ema}")
    s1_obj_traj_dim = int(stage1_ckpt["config"]["model"]["denoiser"].get("obj_traj_dim", 0))
    print(f"[p11] Stage-1 obj_traj_dim = {s1_obj_traj_dim}")

    # Check if model supports configurable num_steps at inference.
    has_num_steps_arg = "num_inference_steps" in stage1_model.sample.__doc__ if stage1_model.sample.__doc__ else False
    # Fall back: introspect via inspect.
    import inspect
    sample_sig = inspect.signature(stage1_model.sample)
    sample_params = list(sample_sig.parameters.keys())
    has_num_steps_arg = any(p in sample_params for p in
                             ("num_inference_steps", "num_steps", "n_steps"))
    print(f"[p11] Stage-1 sample() supports configurable num_steps: {has_num_steps_arg}")
    print(f"[p11] sample() params: {sample_params}")

    s1_cache = _load_stage1_cache(args.stage1_cache_root, split=args.bucket)
    s1_mean = s1_cache["s1_mean"]
    s1_std = s1_cache["s1_std"]
    s1_mean_t = torch.from_numpy(s1_mean).to(device)
    s1_std_t = torch.from_numpy(s1_std).to(device)
    if s1_obj_traj_dim > 0 and s1_cache["obj_mean"] is not None:
        s1_obj_mean_t = torch.from_numpy(s1_cache["obj_mean"]).to(device)
        s1_obj_std_t = torch.from_numpy(s1_cache["obj_std"]).to(device)
    else:
        s1_obj_mean_t = s1_obj_std_t = None

    # ---- Selection ----
    sel_obj = json.loads(args.selection_json.read_text("utf-8"))
    selection = sel_obj.get("selected", sel_obj.get("candidates", []))
    full_dataset = _build_dataset(cfg, args.bucket, augment=False)
    subset_ds, matched = _filter_dataset_by_selection(full_dataset, selection)
    if args.max_clips > 0:
        matched = matched[: args.max_clips]
        subset_ds = Subset(subset_ds.dataset, subset_ds.indices[: args.max_clips])
    print(f"[p11] matched {len(matched)} clips in {args.bucket} bucket")
    loader = DataLoader(subset_ds, batch_size=1, shuffle=False,
                        collate_fn=collate_hoi, num_workers=0)

    # ---- Audit loop ----
    per_clip: list[dict] = []
    total_evals = len(matched) * n_combos
    eval_i = 0
    matched_iter = iter(matched)
    t_start = time.time()

    for batch in loader:
        sel_entry = next(matched_iter)
        subset = str(batch["subset"][0])
        seq_id = str(batch["seq_id"][0])
        cat = sel_entry.get("mode_category",
                             sel_entry.get("mode_category_guess", "unknown"))
        key = (subset, seq_id)
        if key not in s1_cache["by_key"]:
            print(f"  [SKIP] {subset}/{seq_id} not in Stage-1 cache")
            continue
        rec_idx = s1_cache["by_key"][key]
        rec = s1_cache["manifest"][rec_idx]
        npz = np.load(s1_cache["cache_root"] / rec["npz_path"], allow_pickle=False)

        # GT coarse (oracle) for residual computation.
        T_clip = int(batch["seq_len"][0].item())
        gt_motion = batch["motion"][:, :T_clip].to(device).float()
        rest_offsets = batch["rest_offsets"].to(device).float()
        oracle_coarse = extract_coarse_v1_batched(gt_motion, rest_offsets)[0]  # (T, 23)
        oracle_np = oracle_coarse.detach().cpu().numpy()

        # Stage-1 sampling inputs.
        init_norm_np = (oracle_np[0] - s1_mean) / s1_std
        init_norm = torch.from_numpy(init_norm_np).unsqueeze(0).to(device)  # (1, 23)

        # text_pool from Stage-1 cache.
        text = str(batch["text"][0])
        text_row = s1_cache["text_index"].get(text, None)
        if text_row is None:
            print(f"  [SKIP] text not in Stage-1 CLIP index: {subset}/{seq_id}")
            continue
        text_pool_np = s1_cache["clip_emb"][int(text_row)].astype(np.float32)
        text_pool = torch.from_numpy(text_pool_np).unsqueeze(0).to(device)

        # obj_traj from Stage-1 cache npz.
        s1_cond_base = {
            "text_pool": text_pool,
            "init_coarse": init_norm,
            "valid_mask": torch.ones(1, T_clip, dtype=torch.bool, device=device),
        }
        if s1_obj_traj_dim > 0:
            obj_field = None
            for cand in ("obj_traj_root0_world", "obj_traj_canonical"):
                if cand in npz.files:
                    obj_field = cand
                    break
            if obj_field is None:
                print(f"  [SKIP] no obj_traj field: {subset}/{seq_id}")
                continue
            obj_raw = npz[obj_field].astype(np.float32)
            if obj_raw.shape[0] < T_clip:
                obj_raw = np.concatenate(
                    [obj_raw, np.tile(obj_raw[-1:], (T_clip - obj_raw.shape[0], 1))],
                    axis=0,
                )
            obj_raw = obj_raw[:T_clip]
            obj_norm_np = (obj_raw - s1_cache["obj_mean"]) / s1_cache["obj_std"]
            s1_cond_base["obj_traj"] = torch.from_numpy(obj_norm_np).unsqueeze(0).to(device)

        clip_results: list[dict] = []
        for cfg_scale in cfg_scales:
            for nstep in nstep_grid:
                seed_l2_means = []
                seed_per_channel = []
                for seed in seeds:
                    torch.manual_seed(seed)
                    sample_kwargs = dict(
                        shape=(1, T_clip, STAGE1_COARSE_DIM),
                        cond=s1_cond_base,
                        cfg_scale=cfg_scale,
                        device=device,
                        inpaint_frame0=True,
                    )
                    # Try to plumb num_steps if supported.
                    if has_num_steps_arg:
                        # Try common arg names.
                        for k in ("num_inference_steps", "num_steps", "n_steps"):
                            if k in sample_params:
                                sample_kwargs[k] = nstep
                                break

                    with torch.no_grad():
                        sampled_norm = stage1_model.sample(**sample_kwargs)
                    sampled_raw = sampled_norm.detach().cpu().numpy()[0] * s1_std + s1_mean  # (T, 23)
                    l2_per_ch = _l2_per_channel(sampled_raw, oracle_np)
                    seed_l2_means.append(float(l2_per_ch.mean()))
                    seed_per_channel.append(l2_per_ch)
                eval_i += 1
                if eval_i % 5 == 0 or eval_i == 1:
                    elapsed = time.time() - t_start
                    eta = elapsed / eval_i * (total_evals - eval_i)
                    print(f"  [{eval_i}/{total_evals}] {subset}/{seq_id}  "
                          f"cfg={cfg_scale} steps={nstep}  "
                          f"L2_mean={np.mean(seed_l2_means):.4f}  "
                          f"(elapsed {elapsed:.0f}s, ETA {eta:.0f}s)")
                clip_results.append({
                    "cfg_scale": cfg_scale,
                    "num_steps": nstep,
                    "n_seeds": len(seeds),
                    "L2_per_seed": seed_l2_means,
                    "L2_mean_over_seeds": float(np.mean(seed_l2_means)),
                    "L2_per_channel_mean": np.stack(seed_per_channel).mean(axis=0).tolist(),
                })

        per_clip.append({
            "subset": subset,
            "seq_id": seq_id,
            "mode_category": cat,
            "T": T_clip,
            "results": clip_results,
        })

    # ---- Aggregate ----
    # Per (cfg, steps) cell across all clips, mean L2.
    by_cell: dict[tuple, list[float]] = {}
    for clip in per_clip:
        for r in clip["results"]:
            cell = (r["cfg_scale"], r["num_steps"])
            by_cell.setdefault(cell, []).append(r["L2_mean_over_seeds"])

    # Per (subset, cfg, steps) cell.
    by_subset_cell: dict[tuple, list[float]] = {}
    for clip in per_clip:
        sub = clip["subset"]
        for r in clip["results"]:
            key = (sub, r["cfg_scale"], r["num_steps"])
            by_subset_cell.setdefault(key, []).append(r["L2_mean_over_seeds"])

    # Per (mode_category, cfg, steps).
    by_cat_cell: dict[tuple, list[float]] = {}
    for clip in per_clip:
        cat = clip["mode_category"]
        for r in clip["results"]:
            key = (cat, r["cfg_scale"], r["num_steps"])
            by_cat_cell.setdefault(key, []).append(r["L2_mean_over_seeds"])

    summary = {
        "config": str(args.config),
        "stage1_ckpt": str(args.stage1_ckpt),
        "stage1_cache_root": str(args.stage1_cache_root),
        "selection_json": str(args.selection_json),
        "bucket": args.bucket,
        "cfg_scales": cfg_scales,
        "nstep_grid": nstep_grid,
        "seeds": seeds,
        "n_clips": len(per_clip),
        "n_evals": eval_i,
        "supports_num_steps_arg": bool(has_num_steps_arg),
        "overall_cell_L2_mean": {
            f"cfg{cfg}_steps{n}": float(np.mean(v))
            for (cfg, n), v in sorted(by_cell.items())
        },
        "by_subset_cell": {
            f"{sub}_cfg{cfg}_steps{n}": float(np.mean(v))
            for (sub, cfg, n), v in sorted(by_subset_cell.items())
        },
        "by_mode_category_cell": {
            f"{cat}_cfg{cfg}_steps{n}": float(np.mean(v))
            for (cat, cfg, n), v in sorted(by_cat_cell.items())
        },
        "per_clip": per_clip,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[p11] wrote JSON to {args.output}")

    # ---- Markdown ----
    md_path = args.output.with_suffix(".md")
    md: list[str] = []
    md.append("# Round-25 P1.1 — Stage-1 sampler audit\n")
    md.append(f"**Stage-1 ckpt:** `{args.stage1_ckpt}`")
    md.append(f"**Cache:** `{args.stage1_cache_root}`")
    md.append(f"**Selection:** `{args.selection_json}` ({len(per_clip)} clips × {n_combos} (cfg,steps,seed) combos)")
    md.append(f"**Sampler supports num_steps arg:** {has_num_steps_arg}")
    md.append("")
    md.append("## Overall (mean L2 of sampled-vs-oracle coarse-v1, raw units)\n")
    md.append("| cfg \\ steps | " + " | ".join(str(n) for n in nstep_grid) + " |")
    md.append("|---|" + "|".join(["---"] * len(nstep_grid)) + "|")
    for cfg_v in cfg_scales:
        row = [f"cfg={cfg_v}"]
        for n in nstep_grid:
            val = float(np.mean(by_cell.get((cfg_v, n), [float("nan")])))
            row.append(f"{val:.4f}")
        md.append("| " + " | ".join(row) + " |")

    md.append("\n## Per-subset breakdown (rows=subset, cols=cfg×steps)\n")
    subsets = sorted({c["subset"] for c in per_clip})
    md.append("| subset | " + " | ".join(f"cfg{cfg}_n{n}" for cfg in cfg_scales for n in nstep_grid) + " |")
    md.append("|---|" + "|".join(["---"] * (len(cfg_scales) * len(nstep_grid))) + "|")
    for sub in subsets:
        row = [sub]
        for cfg_v in cfg_scales:
            for n in nstep_grid:
                v = float(np.mean(by_subset_cell.get((sub, cfg_v, n), [float("nan")])))
                row.append(f"{v:.3f}")
        md.append("| " + " | ".join(row) + " |")

    md.append("\n## Per-mode_category breakdown\n")
    cats = sorted({c["mode_category"] for c in per_clip})
    md.append("| category | " + " | ".join(f"cfg{cfg}_n{n}" for cfg in cfg_scales for n in nstep_grid) + " |")
    md.append("|---|" + "|".join(["---"] * (len(cfg_scales) * len(nstep_grid))) + "|")
    for cat in cats:
        row = [cat]
        for cfg_v in cfg_scales:
            for n in nstep_grid:
                v = float(np.mean(by_cat_cell.get((cat, cfg_v, n), [float("nan")])))
                row.append(f"{v:.3f}")
        md.append("| " + " | ".join(row) + " |")

    md.append("\n## Interpretation\n")
    md.append(
        "- **Lower L2 = closer sampled coarse to oracle (GT-derived).**\n"
        "- If one (cfg, steps) cell is materially lower than (cfg=1.0, steps=1000) "
        "(the default D3 used), the gap is fixable by inference-time tuning.\n"
        "- If all cells are similar, Stage-1 model itself is the limit → consider retrain "
        "with subset-balanced data or higher capacity (see PLAN.md §1.1 step 4).\n"
        "- Compare subset rows: chairs typically smallest residual (training-dominant); "
        "imhd / neuraldome / omomo show whether non-chair behaviors are sampler-config-rescuable "
        "or training-data-balance limited."
    )
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"[p11] wrote Markdown to {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
