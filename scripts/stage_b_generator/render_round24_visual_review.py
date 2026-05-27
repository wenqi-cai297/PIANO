"""Round-24 quick visual review — render GT vs predicted motion side-by-side
for N selected clips. Uses the R23 no-plan ckpt (v26-equivalent) by default.

Why minimal:
    plan_condition_diagnostics.py also renders MP4 but runs the full 14-variant
    DDPM sweep per clip (~5 min each). For a hand-review of N clips, we just
    need GT + one sample, so ~30 s per clip.

Usage:
    conda run -n piano python scripts/stage_b_generator/render_round24_visual_review.py \\
        --config configs/training/anchordiff_v25_round23_noplan_clean_alibi_FULL_DATA.yaml \\
        --ckpt   runs/training/stageB_anchordiff_v25_round23_noplan_clean_alibi_FULL_DATA/final.pt \\
        --output-dir analyses/round24_visual_review \\
        --bucket val --n-clips 8 --cfg-scale 1.0 --seed 42

Output:
    <output-dir>/clip{NN}_<subset>_<seq_id>_gt.mp4
    <output-dir>/clip{NN}_<subset>_<seq_id>_pred.mp4
    <output-dir>/summary.md  (table linking clips + caption + per-clip metrics)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from plan_condition_diagnostics import (  # noqa: E402
    _build_cond, _build_dataset, _build_model, _stage1_norm_for_cfg,
)

from piano.data.dataset import collate_hoi  # noqa: E402
from piano.inference.visualize_motion import render_motion_video  # noqa: E402
from piano.training.smpl_kinematics import (  # noqa: E402
    fk_from_global_rotations,
    rotation_6d_to_matrix,
)
from piano.utils.clip_utils import load_clip_text_encoder  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bucket", default="val", choices=["train", "val"])
    parser.add_argument("--n-clips", type=int, default=8)
    parser.add_argument("--start-clip", type=int, default=0,
                        help="Skip this many clips from the start of the bucket.")
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--selection-json", type=Path,
                        default=Path("analyses/2026-05-20_round19_eval_selection.json"),
                        help="Optional: filter to clips listed in this selection JSON.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Selection filter ----
    # Two schemas in the wild:
    #   - old (R27 tier0_eval_selection_balanced.json): {"selected": [...], "n_clips": N}
    #   - new (R27 build_tier0_train_indices.py output, used by R29
    #     build_val_diag_subset): {"clips": [...], "n_found": N, "indices": [...]}
    # Both list dicts with {"subset", "seq_id", ...}; we accept either.
    sel_pairs = None
    if args.selection_json.exists():
        sel = json.loads(args.selection_json.read_text("utf-8"))
        entries = sel.get("selected") or sel.get("clips") or []
        sel_pairs = {(e["subset"], e["seq_id"]) for e in entries}
        print(f"[viz] selection JSON: {len(sel_pairs)} clips")

    # ---- Dataset ----
    dataset = _build_dataset(cfg, args.bucket, augment=False)
    overfit_n = int(cfg.data.get("overfit_n_clips", 0))
    if overfit_n > 0:
        dataset = Subset(dataset, list(range(min(overfit_n, len(dataset)))))
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=collate_hoi, num_workers=0,
    )

    # ---- Model + ckpt + text encoder ----
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

    # ---- Loop over clips ----
    rendered_rows = []
    matched = 0
    for clip_global_idx, batch in enumerate(loader):
        if clip_global_idx < args.start_clip:
            continue
        subset = str(batch["subset"][0])
        seq_id = str(batch["seq_id"][0])
        if sel_pairs is not None and (subset, seq_id) not in sel_pairs:
            continue
        if matched >= args.n_clips:
            break

        cond, T = _build_cond(
            batch, model, object_encoder, clip_model, z_dims, cfg, device,
            stage1_norm=stage1_norm,
        )
        plan_keys = [
            "anchor_time", "anchor_part", "anchor_target_local",
            "anchor_target_world", "anchor_type", "anchor_phase",
            "anchor_support", "anchor_conf", "anchor_mask",
            "segment_start", "segment_end", "segment_part",
            "segment_target_summary_local", "segment_phase",
            "segment_support", "segment_conf", "segment_mask",
        ]
        cond["interaction_plan"] = {
            k: batch[f"plan_{k}"].to(device) for k in plan_keys
        }

        torch.manual_seed(args.seed)
        with torch.no_grad():
            pred_motion = model.sample(
                cond=cond, seq_length=T, cfg_scale=args.cfg_scale,
                replacement="none", output_skip=False,
            )                                                       # (1, T, 135)

        gt_motion = batch["motion"][:, :T].to(device).float()
        rest_offsets = batch["rest_offsets"].to(device).float()
        seq_len = int(batch["seq_len"][0].item())
        valid_T = min(T, seq_len)
        text = str(batch["text"][0])

        # FK both motions.
        def _fk(motion):
            B, Tm, _ = motion.shape
            rot6d = motion[..., :132].view(B, Tm, 22, 6).float()
            root_world = motion[..., 132:135].float()
            rot_mat = rotation_6d_to_matrix(rot6d)
            rest_per_frame = rest_offsets.unsqueeze(1).expand(B, Tm, 22, 3)
            return fk_from_global_rotations(rot_mat, rest_per_frame, root_world)

        gt_joints = _fk(gt_motion).squeeze(0).cpu().numpy()[:valid_T]
        pred_joints = _fk(pred_motion).squeeze(0).cpu().numpy()[:valid_T]

        obj_pos_np = batch["object_positions"].squeeze(0).cpu().numpy()[:valid_T]
        obj_rot_np = batch["object_rotations"].squeeze(0).cpu().numpy()[:valid_T]
        obj_pc_np = batch["object_pc"].squeeze(0).cpu().numpy()

        clip_label = f"clip{matched:02d}_{subset}_{seq_id}"
        text_short = (text[:80] + "…") if len(text) > 80 else text
        gt_out = args.output_dir / f"{clip_label}_gt.mp4"
        pred_out = args.output_dir / f"{clip_label}_pred.mp4"

        print(f"  [{matched + 1}/{args.n_clips}] {subset}/{seq_id} (T={valid_T}) — rendering GT + pred...")
        render_motion_video(
            joints=gt_joints, output_path=gt_out, fps=args.fps,
            title=f"{subset}/{seq_id} [GT]\n{text_short}",
            object_positions=obj_pos_np, object_rotations=obj_rot_np,
            object_pc=obj_pc_np,
        )
        render_motion_video(
            joints=pred_joints, output_path=pred_out, fps=args.fps,
            title=f"{subset}/{seq_id} [pred no-plan ckpt, cfg={args.cfg_scale}]\n{text_short}",
            object_positions=obj_pos_np, object_rotations=obj_rot_np,
            object_pc=obj_pc_np,
        )
        rendered_rows.append({
            "clip_label": clip_label,
            "subset": subset, "seq_id": seq_id, "T": valid_T,
            "text": text,
            "gt_mp4": str(gt_out.name),
            "pred_mp4": str(pred_out.name),
        })
        matched += 1

    # ---- Summary markdown ----
    md = [
        "# Round-24 visual review",
        "",
        f"**Ckpt:** `{args.ckpt}`",
        f"**Config:** `{args.config}`",
        f"**Bucket:** {args.bucket}    **cfg_scale:** {args.cfg_scale}    **seed:** {args.seed}    **fps:** {args.fps}",
        f"**Clips rendered:** {len(rendered_rows)}",
        "",
        "| # | subset | seq_id | T | GT video | predicted video | text |",
        "|---|---|---|---:|---|---|---|",
    ]
    for i, r in enumerate(rendered_rows):
        md.append(
            f"| {i:02d} | {r['subset']} | `{r['seq_id']}` | {r['T']} | "
            f"`{r['gt_mp4']}` | `{r['pred_mp4']}` | {r['text'][:120]} |"
        )
    (args.output_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n[viz] {len(rendered_rows)} clips rendered → {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
