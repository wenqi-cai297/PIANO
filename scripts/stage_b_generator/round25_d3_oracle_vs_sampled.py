"""Round-25 D3: oracle vs sampled Stage-1 comparison.

Stage-2 (v26) was trained with GT-derived ORACLE Stage-1 coarse as
conditioning. At inference, the deployable pipeline uses a SAMPLED
Stage-1 coarse from the Stage-1 ckpt. This diagnostic measures the
gap.

Discriminates between:
    H4 (Stage-1 sampler bottleneck): sampled is materially worse than
        oracle → the limb gap is not Stage-2 mode collapse but Stage-1
        sampler quality. Tier A (23→31D) may not help unless Stage-1
        itself stops averaging.
    not-H4: oracle ≈ sampled → Stage-2 is the bottleneck, Stage-1
        sampler is fine.

Design source:
    analyses/2026-05-23_round25_diagnostic_bundle_design.md §D3.

Usage:
    conda run -n piano python scripts/stage_b_generator/round25_d3_oracle_vs_sampled.py \
        --config configs/training/anchordiff_v26_FULL_DATA_local.yaml \
        --ckpt   runs/training/stageB_anchordiff_v25_round23_noplan_clean_alibi_FULL_DATA/final.pt \
        --stage1-ckpt runs/training/stage1_s1o_round20_seed42/final.pt \
        --selection-json analyses/round25_multimodal_eval_subset.json \
        --output analyses/round25_d3_oracle_vs_sampled.json

Output:
    {output}.json + sibling .md report.

Key metric per clip:
    - anchor_pose_error_oracle_cm  : v26 sampled with GT-derived
                                      Stage-1 coarse (training-time
                                      condition)
    - anchor_pose_error_sampled_cm : v26 sampled with Stage-1-sampler
                                      output (deployment-time condition)
    - gap_cm                        : sampled − oracle. Positive = worse
                                      under sampled Stage-1.

Decision rule:
    | mean(gap_cm) | reading                                              |
    |--------------|------------------------------------------------------|
    | ≈ 0          | Stage-1 sampler is fine; not the bottleneck          |
    | > 5 cm       | Stage-1 sampler is the limit; Tier A 23→31D illusory |
    | < -2 cm      | Sampled better than oracle (unexpected, investigate) |
"""
from __future__ import annotations

import argparse
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
from eval_stage1_coarse_prior import (  # noqa: E402
    COARSE_DIM as STAGE1_COARSE_DIM,
    _build_model_from_ckpt as build_stage1_model,
)
from round25_d2_diversity_diagnostic import (  # noqa: E402
    _anchor_pose_error_cm,
    _filter_dataset_by_selection,
)

from piano.data.dataset import collate_hoi  # noqa: E402
from piano.data.stage1_coarse_oracle import (  # noqa: E402
    extract_coarse_v1_batched,
    load_stage1_coarse_norm,
)
from piano.utils.clip_utils import load_clip_text_encoder  # noqa: E402


def _stage1_init_pose_from_motion(motion_135: torch.Tensor) -> torch.Tensor:
    """Replicate Stage-1's init_coarse_v1 = coarse_v1[0] convention.

    The Stage-1 trainer + eval use ``init_coarse_v1 = coarse_v1[0]``
    (frame-0 of the 23-D coarse) as init_pose conditioning. Here we
    compute this from GT motion to mirror what the Stage-1 sampler
    would see in deployment.
    """
    raise NotImplementedError(
        "_stage1_init_pose_from_motion: see inline call site; we compute "
        "init via extract_coarse_v1_batched(motion[:, :1], ...).squeeze(1)"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True,
                        help="v26 (Stage-2) training config.")
    parser.add_argument("--ckpt", type=Path, required=True,
                        help="v26 Stage-2 checkpoint.")
    parser.add_argument("--stage1-ckpt", type=Path, required=True,
                        help="Stage-1 S1-O ckpt (Round-20 SHIP).")
    parser.add_argument("--stage1-cache-root", type=Path,
                        default=Path("cache/stage1_coarse_v1_full"),
                        help="Used to load Stage-1 normalization_train.json (mean, std).")
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--bucket", default="val", choices=["train", "val"])
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--cfg-scale-stage1", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path,
                        default=Path("analyses/round25_d3_oracle_vs_sampled.json"))
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load Stage-1 ckpt and its normalizer ----
    print(f"[d3] loading Stage-1 ckpt {args.stage1_ckpt} ...")
    stage1_ckpt = torch.load(args.stage1_ckpt, map_location="cpu", weights_only=False)
    stage1_model, loaded_ema = build_stage1_model(stage1_ckpt, prefer_ema=True)
    stage1_model = stage1_model.to(device).eval()
    print(f"[d3] Stage-1 EMA loaded: {loaded_ema}")

    s1_mean, s1_std = load_stage1_coarse_norm(args.stage1_cache_root)
    s1_mean_t = torch.from_numpy(s1_mean).to(device)
    s1_std_t = torch.from_numpy(s1_std).to(device)

    # Stage-1's obj_traj dim (may be 0 for older ckpts).
    s1_obj_traj_dim = int(stage1_ckpt["config"]["model"]["denoiser"].get("obj_traj_dim", 0))
    print(f"[d3] Stage-1 obj_traj_dim = {s1_obj_traj_dim}")

    # Stage-1's obj_traj normalization (if obj_traj_dim > 0).
    s1_obj_mean = s1_obj_std = None
    if s1_obj_traj_dim > 0:
        # Mirror what eval_stage1_coarse_prior loads: per-cache obj_traj
        # mean/std from normalization_train.json under "obj_traj" key.
        norm_path = args.stage1_cache_root / "normalization_train.json"
        norm = json.loads(norm_path.read_text("utf-8"))
        if "obj_traj" in norm:
            s1_obj_mean = np.asarray(norm["obj_traj"]["mean"], dtype=np.float32)
            s1_obj_std = np.asarray(norm["obj_traj"]["std_clamped"], dtype=np.float32)
        else:
            print("[d3] WARNING: stage1 obj_traj_dim > 0 but no obj_traj norm "
                  "in cache; using zero mean / unit std")
            s1_obj_mean = np.zeros((s1_obj_traj_dim,), dtype=np.float32)
            s1_obj_std = np.ones((s1_obj_traj_dim,), dtype=np.float32)

    # ---- Selection ----
    sel_obj = json.loads(args.selection_json.read_text("utf-8"))
    selection = sel_obj.get("selected", sel_obj.get("candidates", []))
    full_dataset = _build_dataset(cfg, args.bucket, augment=False)
    subset_ds, matched = _filter_dataset_by_selection(full_dataset, selection)
    print(f"[d3] matched {len(matched)} / {len(selection)} clips in {args.bucket}")
    loader = DataLoader(subset_ds, batch_size=1, shuffle=False,
                        collate_fn=collate_hoi, num_workers=0)

    # ---- Stage-2 model + extras ----
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

    per_clip: list[dict] = []
    matched_iter = iter(matched)
    for batch in loader:
        sel_entry = next(matched_iter)
        subset = str(batch["subset"][0])
        seq_id = str(batch["seq_id"][0])
        text = str(batch["text"][0])

        # ---- Build cond for oracle path (stage1_coarse derived from GT) ----
        cond_oracle, T = _build_cond(
            batch, model, object_encoder, clip_model, z_dims, cfg, device,
            stage1_norm=stage1_norm,
        )
        cond_oracle["interaction_plan"] = {
            k: batch[f"plan_{k}"].to(device) for k in plan_keys
        }
        plan_local = cond_oracle["interaction_plan"]

        gt_motion = batch["motion"][:, :T].to(device).float()
        rest_offsets = batch["rest_offsets"].to(device).float()
        seq_len = int(batch["seq_len"][0].item())
        valid_T = min(T, seq_len)

        # ---- Sampled Stage-1 path: sample 23-D coarse from Stage-1 model ----
        # init_coarse for Stage-1 = frame-0 of GT's coarse (oracle init).
        # This is what Stage-1 eval does: text + init_coarse + (optional obj_traj).
        with torch.no_grad():
            oracle_coarse_full = extract_coarse_v1_batched(
                gt_motion, rest_offsets,
            )                                                 # (1, T, 23) raw
            init_coarse = oracle_coarse_full[:, 0]            # (1, 23)
            init_coarse_norm = (init_coarse - s1_mean_t) / s1_std_t   # (1, 23)

            text_pool = batch.get("text_pool")
            if text_pool is None:
                # Re-encode via clip_model: mirror _build_cond pathway.
                texts = [str(t) for t in batch["text"]]
                text_pool = clip_model.encode_text(texts).to(device).float()
            else:
                text_pool = text_pool.to(device).float()

            s1_cond: dict[str, torch.Tensor] = {
                "text_pool": text_pool,
                "init_coarse": init_coarse_norm,
                "valid_mask": torch.ones(1, T, dtype=torch.bool, device=device),
            }
            if s1_obj_traj_dim > 0:
                # obj_traj from batch — should be (1, T, 9). Normalize with Stage-1 norm.
                obj_traj_raw = batch.get("obj_traj")
                if obj_traj_raw is None:
                    # Fallback: derive from object_positions + object_rotations + size.
                    obj_traj_raw = torch.cat(
                        [batch["object_positions"].to(device),
                         batch["object_rotations"].to(device)],
                        dim=-1,
                    )[:, :T]   # 6-D fallback; only valid if Stage-1 uses 6-D obj
                else:
                    obj_traj_raw = obj_traj_raw[:, :T].to(device)
                obj_mean_t = torch.from_numpy(s1_obj_mean).to(device)
                obj_std_t = torch.from_numpy(s1_obj_std).to(device)
                s1_cond["obj_traj"] = (obj_traj_raw.float() - obj_mean_t) / obj_std_t

            torch.manual_seed(args.seed)
            sampled_norm = stage1_model.sample(
                shape=(1, T, STAGE1_COARSE_DIM), cond=s1_cond,
                cfg_scale=args.cfg_scale_stage1, device=device,
                inpaint_frame0=True,
            )                                                  # (1, T, 23) normalized
            # Build cond_sampled by REPLACING the stage1_coarse_norm of cond_oracle.
            # cond_oracle["stage1_coarse_norm"] is already normalized in _build_cond.
            # The sampled output is also in the same normalization (same s1_mean/s1_std).
            cond_sampled = {k: v for k, v in cond_oracle.items()}
            cond_sampled["stage1_coarse_norm"] = sampled_norm

        # ---- Sample Stage-2 under each path ----
        torch.manual_seed(args.seed)
        with torch.no_grad():
            pred_oracle = model.sample(
                cond=cond_oracle, seq_length=T, cfg_scale=args.cfg_scale,
                replacement="none", output_skip=False,
            )
        err_oracle = _anchor_pose_error_cm(
            pred_oracle[:, :valid_T], gt_motion[:, :valid_T],
            rest_offsets, plan_local,
        )

        torch.manual_seed(args.seed)
        with torch.no_grad():
            pred_sampled = model.sample(
                cond=cond_sampled, seq_length=T, cfg_scale=args.cfg_scale,
                replacement="none", output_skip=False,
            )
        err_sampled = _anchor_pose_error_cm(
            pred_sampled[:, :valid_T], gt_motion[:, :valid_T],
            rest_offsets, plan_local,
        )

        gap = err_sampled - err_oracle
        per_clip.append({
            "subset": subset,
            "seq_id": seq_id,
            "text": text,
            "mode_category": sel_entry.get("mode_category",
                                            sel_entry.get("mode_category_guess", "unknown")),
            "T": valid_T,
            "anchor_pose_error_oracle_cm": float(err_oracle),
            "anchor_pose_error_sampled_cm": float(err_sampled),
            "gap_cm": float(gap),
        })
        print(f"  [d3 {len(per_clip)}/{len(matched)}] {subset}/{seq_id}  "
              f"oracle={err_oracle:.2f}cm  sampled={err_sampled:.2f}cm  gap={gap:+.2f}cm")

    # ---- Aggregate ----
    by_subset: dict[str, list[dict]] = {}
    by_category: dict[str, list[dict]] = {}
    for r in per_clip:
        by_subset.setdefault(r["subset"], []).append(r)
        by_category.setdefault(r["mode_category"], []).append(r)

    def _agg(rows: list[dict]) -> dict:
        if not rows:
            return {}
        return {
            "n": len(rows),
            "oracle_mean": float(np.mean([r["anchor_pose_error_oracle_cm"] for r in rows])),
            "sampled_mean": float(np.mean([r["anchor_pose_error_sampled_cm"] for r in rows])),
            "gap_mean": float(np.mean([r["gap_cm"] for r in rows])),
            "gap_p50": float(np.median([r["gap_cm"] for r in rows])),
            "gap_p95": float(np.percentile([r["gap_cm"] for r in rows], 95)),
        }

    summary = {
        "config": str(args.config),
        "ckpt": str(args.ckpt),
        "stage1_ckpt": str(args.stage1_ckpt),
        "selection_json": str(args.selection_json),
        "bucket": args.bucket,
        "cfg_scale_stage2": args.cfg_scale,
        "cfg_scale_stage1": args.cfg_scale_stage1,
        "seed": args.seed,
        "overall": _agg(per_clip),
        "by_mode_category": {k: _agg(v) for k, v in by_category.items()},
        "by_subset": {k: _agg(v) for k, v in by_subset.items()},
        "per_clip": per_clip,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    print(f"\n[d3] wrote JSON to {args.output}")

    # ---- Markdown ----
    md = args.output.with_suffix(".md")
    lines: list[str] = []
    lines.append("# Round-25 D3 oracle vs sampled Stage-1 diagnostic\n")
    lines.append(f"**Stage-2 ckpt:** `{args.ckpt}`")
    lines.append(f"**Stage-1 ckpt:** `{args.stage1_ckpt}`")
    lines.append(f"**Selection:** `{args.selection_json}` ({len(per_clip)} clips)")
    lines.append(f"**cfg_scale (stage1 / stage2):** {args.cfg_scale_stage1} / {args.cfg_scale}    **seed:** {args.seed}\n")
    lines.append("## Decision rule\n")
    lines.append("| mean(gap_cm) | reading |")
    lines.append("|---:|---|")
    lines.append("| ≈ 0 | Stage-1 sampler is fine; not the bottleneck |")
    lines.append("| > 5 cm | Stage-1 sampler is the limit; Tier A 23→31D may be illusory |")
    lines.append("| < -2 cm | Unexpected — sampled better than oracle |\n")
    ov = summary["overall"]
    if ov:
        lines.append("## Overall\n")
        lines.append(f"- n = {ov['n']}")
        lines.append(f"- oracle mean = {ov['oracle_mean']:.2f} cm")
        lines.append(f"- sampled mean = {ov['sampled_mean']:.2f} cm")
        lines.append(f"- gap mean = {ov['gap_mean']:+.2f} cm    (p50 = {ov['gap_p50']:+.2f}, p95 = {ov['gap_p95']:+.2f})\n")
    lines.append("## By subset\n")
    lines.append("| subset | n | oracle | sampled | gap |")
    lines.append("|---|---:|---:|---:|---:|")
    for sub, agg in summary["by_subset"].items():
        lines.append(
            f"| {sub} | {agg['n']} | {agg['oracle_mean']:.2f} | "
            f"{agg['sampled_mean']:.2f} | {agg['gap_mean']:+.2f} |"
        )
    lines.append("\n## By mode category\n")
    lines.append("| category | n | oracle | sampled | gap |")
    lines.append("|---|---:|---:|---:|---:|")
    for cat, agg in summary["by_mode_category"].items():
        lines.append(
            f"| {cat} | {agg['n']} | {agg['oracle_mean']:.2f} | "
            f"{agg['sampled_mean']:.2f} | {agg['gap_mean']:+.2f} |"
        )
    md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[d3] wrote Markdown to {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
