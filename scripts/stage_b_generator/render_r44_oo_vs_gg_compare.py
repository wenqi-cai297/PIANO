"""R44 visualization — render side-by-side OO vs GG PB1 outputs.

For each selected clip, sample PB1 twice:

  - **OO** (oracle Stage-1 + oracle C41/S4): the standard cond, all
    cond keys extracted from GT inside ``_build_cond``. PB1 floor
    ~7.5 cm sustained drift.
  - **GG** (generated Stage-1 + generated Stage-1.5): substitute
    stage1_coarse + stage2_coarse_extra + stage2_support from a
    pre-sampled cache (R42 produces this at
    ``analyses/round42_cond_2x2_<stamp>/substitute_conds/merged_generated_s1_generated_s1p5/<bucket>``).
    Current best generated baseline = R42 GG ~39 cm; the R43 P0 retrain
    made it slightly worse (~41 cm), so the R42 generated cache stays
    the "best generated condition" reference.

Each clip's two motions render into ONE side-by-side mp4 named
``clip{NN}_<subset>_<seq_id>_oo_vs_gg.mp4`` (left pane = OO, right
pane = GG). A ``summary.md`` indexes the rendered files and the
per-clip metadata (T, text caption, sustained drift if precomputed).

Usage::

    python scripts/stage_b_generator/render_r44_oo_vs_gg_compare.py \\
        --config configs/training/anchordiff_r29_pb_a1_adaln_s4.yaml \\
        --ckpt runs/training/stageB_anchordiff_r29_pb_a1_adaln_s4/final.pt \\
        --gg-substitute-dir analyses/round42_cond_2x2_<stamp>/substitute_conds/merged_generated_s1_generated_s1p5/val \\
        --output-dir analyses/round44_visualize_oo_vs_gg \\
        --bucket val \\
        --selection-json analyses/round29_val_diag_indices_48_balanced.json \\
        --n-clips 12 --cfg-scale 1.0 --seed 42

The OO label and GG label appear above each pane. ``object_positions``,
``object_rotations`` and ``object_pc`` are shared across both panes so
"is the body interacting with the object?" reads the same on both
sides.

This script does NOT introspect the cache's PB1 prediction tarballs
because R42's GG was an on-the-fly sample (diag scripts didn't dump the
PB1 motion to disk). Re-sampling is cheap (~3 s/clip) and guarantees
the OO/GG pair shares everything except the cond. Total wall-clock:
~7 s/clip x N clips on a 5080.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from piano.data.dataset import collate_hoi
from piano.inference.diagnostic_helpers import (
    _build_cond,
    _build_dataset,
    _build_model,
    _fk_22joints,
    _stage1_norm_for_cfg,
    load_substitute_conds_for_clip,
)
from piano.inference.visualize_motion import render_motion_pair_video
from piano.utils.clip_utils import load_clip_text_encoder


def _read_selection(path: Path | None) -> set[tuple[str, str]] | None:
    if path is None or not path.exists():
        return None
    obj = json.loads(path.read_text("utf-8"))
    entries = obj.get("selected") or obj.get("candidates") or obj.get("clips") or []
    return {(str(e["subset"]), str(e["seq_id"])) for e in entries}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True,
                    help="PB1 cfg (e.g. anchordiff_r29_pb_a1_adaln_s4.yaml).")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--gg-substitute-dir", type=Path, required=True,
                    help=(
                        "Directory of per-clip <subset>/<seq_id>.npz with "
                        "keys stage1_coarse + stage2_coarse_extra + "
                        "stage2_support. Produced by R42's "
                        "run_round42_cond_2x2_diag.sh under "
                        "substitute_conds/merged_generated_s1_generated_s1p5/<bucket>/."
                    ))
    ap.add_argument("--bucket", default="val", choices=["train", "val"])
    ap.add_argument("--n-clips", type=int, default=12)
    ap.add_argument("--start-clip", type=int, default=0,
                    help="Skip this many filter-matching clips from the start.")
    ap.add_argument("--selection-json", type=Path, default=None,
                    help=(
                        "Optional: restrict the rendered clips to the "
                        "(subset, seq_id) pairs listed in this selection "
                        "JSON. When omitted, render the first --n-clips "
                        "of the bucket."
                    ))
    ap.add_argument("--cfg-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--dpi", type=int, default=80)
    ap.add_argument(
        "--gg-label", default="GG (generated S1 + generated S1.5)",
        help="Right-pane subtitle.",
    )
    ap.add_argument(
        "--oo-label", default="OO (oracle S1 + oracle S1.5)",
        help="Left-pane subtitle.",
    )
    args = ap.parse_args()

    if not args.gg_substitute_dir.is_dir():
        raise SystemExit(
            f"[viz] --gg-substitute-dir not found: {args.gg_substitute_dir}. "
            "Re-run R42's pipeline first (it leaves the merged cache under "
            "analyses/round42_cond_2x2_<stamp>/substitute_conds/merged_...)."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load(str(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[viz] device={device}")

    sel_pairs = _read_selection(args.selection_json)
    if sel_pairs is not None:
        print(f"[viz] selection: {len(sel_pairs)} clips")

    # ─── Dataset
    dataset = _build_dataset(cfg, args.bucket, augment=False)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        collate_fn=collate_hoi, num_workers=0,
    )

    # ─── Model + text encoder + Stage-1 norm
    model, object_encoder = _build_model(cfg, device)
    state = torch.load(str(args.ckpt), map_location="cpu", weights_only=False)
    model_state = state.get("model", state)
    model.load_state_dict(model_state)
    if "object_encoder" in state:
        object_encoder.load_state_dict(state["object_encoder"])
    elif "extra_modules" in state and "object_encoder" in state["extra_modules"]:
        object_encoder.load_state_dict(state["extra_modules"]["object_encoder"])
    if int(cfg.model.denoiser.get("text_dim", 0)) > 0:
        clip_model = load_clip_text_encoder(
            device=device,
            model_name=str(cfg.model.text_encoder.clip_version),
            download_root=str(
                cfg.model.text_encoder.get("download_root", "cache/clip"),
            ),
        )
    else:
        clip_model = None
    stage1_norm = _stage1_norm_for_cfg(cfg, device)
    model.eval()

    rendered_rows: list[dict] = []
    matched = 0
    for clip_global_idx, batch in enumerate(loader):
        subset = str(batch["subset"][0])
        seq_id = str(batch["seq_id"][0])
        if sel_pairs is not None and (subset, seq_id) not in sel_pairs:
            continue
        if matched < args.start_clip:
            matched_skip = True
        else:
            matched_skip = False
        if not matched_skip and len(rendered_rows) >= args.n_clips:
            break

        T_for_sub = int(batch["motion"].shape[1])

        # OO: no substitute conds → all cond keys extracted from GT.
        cond_oo, T_oo = _build_cond(
            batch, model, object_encoder, clip_model, cfg, device,
            stage1_norm=stage1_norm,
            substitute_conds=None,
        )
        # GG: substitute stage1_coarse + stage2_coarse_extra + stage2_support
        # from the pre-sampled merged cache.
        sub = load_substitute_conds_for_clip(
            args.gg_substitute_dir, subset, seq_id, T_for_sub, device,
        )
        if sub is None or not {
            "stage1_coarse", "stage2_coarse_extra", "stage2_support",
        }.issubset(sub):
            present = set(sub.keys()) if sub else set()
            print(
                f"  [skip] {subset}/{seq_id}: merged cache missing keys "
                f"(present={sorted(present)}). "
                "Re-run R42 to produce the merged cache."
            )
            continue
        cond_gg, T_gg = _build_cond(
            batch, model, object_encoder, clip_model, cfg, device,
            stage1_norm=stage1_norm,
            substitute_conds=sub,
        )
        T = min(T_oo, T_gg)

        # Sample twice with the same seed so the noise schedule is shared —
        # only the cond differs.
        torch.manual_seed(args.seed)
        with torch.no_grad():
            pred_oo = model.sample(
                cond=cond_oo, seq_length=T_oo, cfg_scale=args.cfg_scale,
            )
        torch.manual_seed(args.seed)
        with torch.no_grad():
            pred_gg = model.sample(
                cond=cond_gg, seq_length=T_gg, cfg_scale=args.cfg_scale,
            )

        rest_offsets = batch["rest_offsets"].to(device).float()
        joints_oo = _fk_22joints(pred_oo, rest_offsets)[0].cpu().numpy()
        joints_gg = _fk_22joints(pred_gg, rest_offsets)[0].cpu().numpy()
        valid_T = min(T, int(batch["seq_len"][0].item()))
        joints_oo = joints_oo[:valid_T]
        joints_gg = joints_gg[:valid_T]

        text = str(batch["text"][0])
        text_short = (text[:80] + "…") if len(text) > 80 else text
        obj_pos_np = batch["object_positions"][0, :valid_T].cpu().numpy()
        obj_rot_np = batch["object_rotations"][0, :valid_T].cpu().numpy()
        obj_pc_np = batch["object_pc"][0].cpu().numpy()

        if matched_skip:
            matched += 1
            continue

        clip_label = f"clip{len(rendered_rows):02d}_{subset}_{seq_id}"
        out_path = args.output_dir / f"{clip_label}_oo_vs_gg.mp4"
        suptitle = (
            f"{subset}/{seq_id}    T={valid_T}    "
            f"cfg={args.cfg_scale}    seed={args.seed}\n{text_short}"
        )
        print(
            f"  [{len(rendered_rows) + 1}/{args.n_clips}] {subset}/{seq_id} "
            f"(T={valid_T}) — rendering side-by-side OO vs GG..."
        )
        render_motion_pair_video(
            joints_left=joints_oo,
            joints_right=joints_gg,
            output_path=out_path,
            fps=args.fps,
            title_left=args.oo_label,
            title_right=args.gg_label,
            suptitle=suptitle,
            object_positions=obj_pos_np,
            object_rotations=obj_rot_np,
            object_pc=obj_pc_np,
            dpi=args.dpi,
        )
        rendered_rows.append({
            "clip_label": clip_label,
            "subset": subset, "seq_id": seq_id, "T": valid_T,
            "text": text,
            "mp4": out_path.name,
        })
        matched += 1

    # Summary
    md_lines = [
        "# R44 OO vs GG side-by-side visualization",
        "",
        f"- PB1 config: `{args.config}`",
        f"- PB1 ckpt:   `{args.ckpt}`",
        f"- GG substitute cache: `{args.gg_substitute_dir}`",
        f"- bucket: {args.bucket}    cfg_scale: {args.cfg_scale}    "
        f"seed: {args.seed}    fps: {args.fps}",
        f"- clips rendered: {len(rendered_rows)}",
        "",
        "Each MP4 has two panes: **LEFT = OO** (oracle Stage-1 + oracle "
        "Stage-1.5), **RIGHT = GG** (generated Stage-1 + generated "
        "Stage-1.5). Both panes share the same object trajectory and "
        "PB1 ckpt; only the cond differs.",
        "",
        "| # | subset | seq_id | T | mp4 | text |",
        "|---|---|---|---:|---|---|",
    ]
    for i, r in enumerate(rendered_rows):
        md_lines.append(
            f"| {i:02d} | {r['subset']} | `{r['seq_id']}` | {r['T']} | "
            f"`{r['mp4']}` | {r['text'][:120]} |"
        )
    summary_md = args.output_dir / "summary.md"
    summary_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(
        f"\n[viz] {len(rendered_rows)} clips rendered → {args.output_dir}\n"
        f"[viz] summary at {summary_md}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
