"""Round-25 D3: oracle vs sampled Stage-1 comparison (REVISED post-Codex audit).

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

Codex audit fixes (analyses/2026-05-23_codex_round25_p0_implementation_review.md §3):
  B1: write sampled coarse to ``cond["stage1_coarse"]`` (NOT
      ``stage1_coarse_norm``) — that is the key the Stage-2 model reads.
  B2: default Stage-1 cache root to the actual S1-O ckpt cache
      ``cache/stage1_coarse_v1_objtraj_root0_world_round18_fix``.
  B4: read 9-D ``obj_traj_root0_world`` directly from the Stage-1 cache
      .npz (NOT a 6-D fallback that conflates axis-angle with rot6d).
  text_pool: read pooled CLIP embeddings from the Stage-1 cache
      (NOT call OpenAI ``encode_text(strings)`` which expects token IDs).
  Cross-cache norm: Stage-1 sampler output is in Stage-1 cache norm
      space; Stage-2 v26 expects v26's own cache norm space. Denorm
      with Stage-1 stats then re-norm with Stage-2 stats before
      writing to ``cond["stage1_coarse"]``.

Usage:
    conda run -n piano python scripts/stage_b_generator/round25_d3_oracle_vs_sampled.py \
        --config configs/training/anchordiff_v26_FULL_DATA_local.yaml \
        --ckpt   runs/training/stageB_anchordiff_v25_round23_noplan_clean_alibi_FULL_DATA/final.pt \
        --stage1-ckpt runs/training/stage1_s1o_round20_seed42/final.pt \
        --stage1-cache-root cache/stage1_coarse_v1_objtraj_root0_world_round18_fix \
        --selection-json analyses/round25_multimodal_eval_subset.json \
        --output analyses/round25_d3_oracle_vs_sampled.json
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
from piano.utils.clip_utils import load_clip_text_encoder  # noqa: E402


def _load_stage1_cache(cache_root: Path, split: str = "val") -> dict[str, Any]:
    """Load Stage-1 cache: manifest + norm + CLIP pooled embeddings."""
    manifest = [
        json.loads(line) for line in
        (cache_root / f"manifest_{split}.jsonl").read_text("utf-8").splitlines()
        if line.strip()
    ]
    norm = json.loads((cache_root / "normalization_train.json").read_text("utf-8"))
    g = norm["global"]
    # Stage-1 23-D coarse stats (legacy schema, still present in round18_fix
    # cache for back-compat with eval_stage1_coarse_prior.py:601-602).
    s1_mean = np.asarray(g["mean"], dtype=np.float32)
    s1_std = np.asarray(g["std_clamped"], dtype=np.float32)
    # Stage-1 obj_traj_root0_world (9-D) stats — new-schema nested block,
    # per eval_stage1_coarse_prior.py:603-617.
    obj_block = g.get("obj_traj_root0_world", None)
    if obj_block is None:
        obj_block = g.get("obj_traj_canonical", None)
    if obj_block is not None:
        obj_mean = np.asarray(obj_block["mean"], dtype=np.float32)
        obj_std = np.asarray(obj_block["std_clamped"], dtype=np.float32)
    else:
        obj_mean = obj_std = None
    # Pooled CLIP embeddings (text → 512-D), per
    # eval_stage1_coarse_prior.py:597-599.
    clip_npz = np.load(cache_root / "text_embeddings_clip_vit_b32.npz",
                       allow_pickle=True)
    clip_emb = clip_npz["embeddings"]
    text_index = json.loads(
        (cache_root / "text_embeddings_index.json").read_text("utf-8"),
    )["index"]
    # Build (subset, seq_id) → manifest index.
    by_key: dict[tuple[str, str], int] = {}
    for i, r in enumerate(manifest):
        by_key[(r["subset"], r["seq_id"])] = i
    return {
        "manifest": manifest,
        "s1_mean": s1_mean,
        "s1_std": s1_std,
        "obj_mean": obj_mean,
        "obj_std": obj_std,
        "clip_emb": clip_emb,
        "text_index": text_index,
        "by_key": by_key,
        "cache_root": cache_root,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True,
                        help="v26 (Stage-2) training config.")
    parser.add_argument("--ckpt", type=Path, required=True,
                        help="v26 Stage-2 checkpoint.")
    parser.add_argument("--stage1-ckpt", type=Path, required=True,
                        help="Stage-1 S1-O ckpt (Round-20 SHIP).")
    parser.add_argument("--stage1-cache-root", type=Path,
                        default=Path("cache/stage1_coarse_v1_objtraj_root0_world_round18_fix"),
                        help="Stage-1 cache dir matching the Stage-1 ckpt training. "
                             "Default = the round18_fix cache used by S1-O.")
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

    # ---- Load Stage-1 ckpt and cache ----
    print(f"[d3] loading Stage-1 ckpt {args.stage1_ckpt} ...")
    stage1_ckpt = torch.load(args.stage1_ckpt, map_location="cpu", weights_only=False)
    stage1_model, loaded_ema = build_stage1_model(stage1_ckpt, prefer_ema=True)
    stage1_model = stage1_model.to(device).eval()
    print(f"[d3] Stage-1 EMA loaded: {loaded_ema}")
    s1_obj_traj_dim = int(
        stage1_ckpt["config"]["model"]["denoiser"].get("obj_traj_dim", 0),
    )
    print(f"[d3] Stage-1 obj_traj_dim = {s1_obj_traj_dim}")

    print(f"[d3] loading Stage-1 cache from {args.stage1_cache_root} ...")
    s1_cache = _load_stage1_cache(args.stage1_cache_root, split=args.bucket)
    s1_mean_t = torch.from_numpy(s1_cache["s1_mean"]).to(device)        # (23,)
    s1_std_t = torch.from_numpy(s1_cache["s1_std"]).to(device)          # (23,)
    if s1_obj_traj_dim > 0:
        if s1_cache["obj_mean"] is None:
            raise SystemExit(
                "[d3] Stage-1 obj_traj_dim > 0 but cache has no "
                "global.obj_traj_root0_world (or _canonical) norm block. "
                f"Check {args.stage1_cache_root}/normalization_train.json"
            )
        s1_obj_mean_t = torch.from_numpy(s1_cache["obj_mean"]).to(device)  # (9,)
        s1_obj_std_t = torch.from_numpy(s1_cache["obj_std"]).to(device)
    else:
        s1_obj_mean_t = s1_obj_std_t = None

    # ---- Stage-2 v26 normalizer for stage1_coarse condition ----
    stage1_norm = _stage1_norm_for_cfg(cfg, device)
    if stage1_norm is None:
        raise SystemExit("[d3] v26 config has stage1_coarse_dim=0; nothing to compare")
    v26_mean_t, v26_std_t = stage1_norm                                # (1, 1, 23)

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
    skipped: list[dict] = []
    for batch in loader:
        sel_entry = next(matched_iter)
        subset = str(batch["subset"][0])
        seq_id = str(batch["seq_id"][0])
        text = str(batch["text"][0])

        # ---- Stage-1 cache lookup for this clip ----
        key = (subset, seq_id)
        if key not in s1_cache["by_key"]:
            print(f"  [d3 SKIP] {subset}/{seq_id} not in Stage-1 cache manifest")
            skipped.append({"subset": subset, "seq_id": seq_id,
                            "reason": "not in Stage-1 cache manifest"})
            continue
        idx = s1_cache["by_key"][key]
        rec = s1_cache["manifest"][idx]
        npz = np.load(s1_cache["cache_root"] / rec["npz_path"], allow_pickle=False)

        # ---- Build cond_oracle (Stage-2 v26 oracle path, ground truth Stage-1) ----
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

        # ---- Stage-1 sampling: build cond from Stage-1 cache .npz ----
        # init_coarse: take frame-0 of the cached coarse (matches
        # eval_stage1_coarse_prior.py:698-699 convention).
        init = npz["init_coarse_v1"].astype(np.float32)                  # (23,)
        init_norm_np = (init - s1_cache["s1_mean"]) / s1_cache["s1_std"]
        init_norm = torch.from_numpy(init_norm_np).unsqueeze(0).to(device)  # (1, 23)

        # text_pool: read pooled CLIP from Stage-1 cache (NOT call CLIP on strings).
        text_row = s1_cache["text_index"].get(text, None)
        if text_row is None:
            print(f"  [d3 SKIP] {subset}/{seq_id} text not in Stage-1 CLIP index")
            skipped.append({"subset": subset, "seq_id": seq_id,
                            "reason": "text not in Stage-1 CLIP index"})
            continue
        text_pool_np = s1_cache["clip_emb"][int(text_row)].astype(np.float32)  # (512,)
        text_pool = torch.from_numpy(text_pool_np).unsqueeze(0).to(device)

        # Stage-1 T may differ from Stage-2 T (cache stores its own seq_len).
        # Pin to Stage-2's T so the produced (1, T, 23) matches what
        # cond_oracle["stage1_coarse"] expects.
        T_for_stage1 = T

        s1_cond: dict[str, torch.Tensor] = {
            "text_pool": text_pool,
            "init_coarse": init_norm,
            "valid_mask": torch.ones(1, T_for_stage1, dtype=torch.bool, device=device),
        }
        if s1_obj_traj_dim > 0:
            # obj_traj_root0_world (9-D = root0-world position 3 + rot6d 6),
            # NOT obj_com_canonical + obj_rot6d_canonical from the v26 batch.
            # Read from the Stage-1 cache npz directly.
            obj_field = None
            for cand in ("obj_traj_root0_world", "obj_traj_canonical"):
                if cand in npz.files:
                    obj_field = cand
                    break
            if obj_field is None:
                print(f"  [d3 SKIP] {subset}/{seq_id} cache has no obj_traj field")
                skipped.append({"subset": subset, "seq_id": seq_id,
                                "reason": "cache has no obj_traj_root0_world"})
                continue
            obj_raw = npz[obj_field].astype(np.float32)                  # (T_s1, 9)
            if obj_raw.shape[0] < T_for_stage1:
                # Pad with last-frame repeat — defensive, shouldn't normally hit
                pad_len = T_for_stage1 - obj_raw.shape[0]
                obj_raw = np.concatenate(
                    [obj_raw, np.tile(obj_raw[-1:], (pad_len, 1))], axis=0,
                )
            obj_raw = obj_raw[:T_for_stage1]
            obj_norm_np = (obj_raw - s1_cache["obj_mean"]) / s1_cache["obj_std"]
            s1_cond["obj_traj"] = torch.from_numpy(obj_norm_np).unsqueeze(0).to(device)

        torch.manual_seed(args.seed)
        with torch.no_grad():
            sampled_norm = stage1_model.sample(
                shape=(1, T_for_stage1, STAGE1_COARSE_DIM), cond=s1_cond,
                cfg_scale=args.cfg_scale_stage1, device=device,
                inpaint_frame0=True,
            )                                                            # (1, T, 23) in Stage-1 norm

        # ---- Cross-cache renormalization ----
        # Stage-1 output is in Stage-1 cache's mean/std space.
        # Stage-2 v26 expects stage1_coarse_norm in v26's own cache space
        # (cache/stage1_coarse_v1_full), which has DIFFERENT stats.
        # Denorm with Stage-1 stats then re-norm with Stage-2 stats.
        sampled_raw = sampled_norm * s1_std_t.view(1, 1, -1) + s1_mean_t.view(1, 1, -1)
        sampled_v26_norm = (sampled_raw - v26_mean_t) / v26_std_t       # (1, T, 23)

        # cond_sampled: shallow-copy oracle cond, OVERWRITE stage1_coarse.
        cond_sampled = {k: v for k, v in cond_oracle.items()}
        cond_sampled["stage1_coarse"] = sampled_v26_norm

        # ---- Stage-2 sampling under each condition ----
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
        "stage1_cache_root": str(args.stage1_cache_root),
        "selection_json": str(args.selection_json),
        "bucket": args.bucket,
        "cfg_scale_stage2": args.cfg_scale,
        "cfg_scale_stage1": args.cfg_scale_stage1,
        "seed": args.seed,
        "n_matched": len(matched),
        "n_processed": len(per_clip),
        "n_skipped": len(skipped),
        "skipped": skipped,
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
    lines.append(f"**Stage-1 cache:** `{args.stage1_cache_root}`")
    lines.append(f"**Selection:** `{args.selection_json}` ({len(per_clip)}/{len(matched)} clips processed; {len(skipped)} skipped)")
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
    if skipped:
        lines.append(f"\n## Skipped clips ({len(skipped)})\n")
        for s in skipped[:20]:
            lines.append(f"- {s['subset']}/{s['seq_id']}: {s['reason']}")
    md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[d3] wrote Markdown to {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
