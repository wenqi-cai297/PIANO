"""Round-19 Stage-1 visual review.

Samples Coarse-v1 from Plan A (seed 47 ckpt-040000) and S1-O (seed 45
ckpt-030000) — the per-mode val-loss champions across 6 training seeds —
on the Round-19 fixed eval selection, then renders 6-panel curve PNGs
comparing GT vs Plan A vs S1-O per clip.

Stage-1 output is 23-D coarse skeleton (root + trunk orientation +
heights), NOT a full SMPL skeleton — so curve plots are the right
primitive, not full-body video.

Per-clip panel (3×2):

    +---------------------+---------------------+
    | root XZ trajectory  | root height over t  |
    +---------------------+---------------------+
    | facing yaw over t   | pelvis rot velocity |
    +---------------------+---------------------+
    | head height over t  | shoulder height     |
    +---------------------+---------------------+

Outputs:
    analyses/round19_visual_review/
      plots/<subset>_<seq_id>__cfg<X>.png   (N_clips × N_cfg files)
      trajectories/<subset>_<seq_id>__cfg<X>.npz  (gt + plan_a + s1o stacked)
      summary.md

Usage
-----

    $env:PYTHONIOENCODING="utf-8"
    conda run -n piano python scripts/stage_b_generator/render_round19_visual_review.py \\
        --plan-a-ckpt runs/training/stage1_s1a_cmc_round19_seed47/ckpt-040000.pt \\
        --s1o-ckpt runs/training/stage1_s1o_round19_seed45/ckpt-030000.pt \\
        --selection-json analyses/2026-05-20_round19_eval_selection.json \\
        --cache-plan-a cache/stage1_coarse_v1_full \\
        --cache-s1o cache/stage1_coarse_v1_objtraj_root0_world_round18_fix \\
        --cfg-scales 2.5 \\
        --sampler-seed 42

Note: Plan A and S1-O were trained from DIFFERENT init seeds (47 vs 45)
because they're the per-mode val-loss champions, not paired. The
intended use is "best representative of each mode" — paired bit-exact
fairness analysis lives in the eval pipeline, not this viz.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch

from piano.models.coarse_motion_prior import (
    CoarsePriorConfig, CoarsePriorDenoiserConfig, CoarsePriorDiff,
)
from piano.models.motion_anchordiff import DiffusionConfig


COARSE_DIM = 23


# ============================================================================
# Ckpt loading (copied from eval_stage1_coarse_prior.py for self-containment)
# ============================================================================


def _build_model_from_ckpt(
    ckpt: dict[str, Any], *, prefer_ema: bool = True,
) -> tuple[CoarsePriorDiff, bool]:
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
        obj_traj_dim=int(den_d.get("obj_traj_dim", 0)),
        obj_traj_hint_hidden_mult=int(den_d.get("obj_traj_hint_hidden_mult", 1)),
    )
    model = CoarsePriorDiff(CoarsePriorConfig(diffusion=diff, denoiser=den))
    model.load_state_dict(ckpt["model"], strict=True)
    loaded_ema = False
    if prefer_ema and "ema" in ckpt:
        own = model.state_dict()
        for k, v in ckpt["ema"].items():
            if k in own and own[k].shape == v.shape:
                own[k] = v.to(own[k].dtype)
        model.load_state_dict(own, strict=True)
        loaded_ema = True
    return model, loaded_ema


# ============================================================================
# Cache + clip loading
# ============================================================================


def _load_cache(cache_root: Path, split: str = "val") -> dict[str, Any]:
    manifest = [
        json.loads(line) for line in
        (cache_root / f"manifest_{split}.jsonl").read_text("utf-8").splitlines()
        if line.strip()
    ]
    clip_npz = np.load(cache_root / "text_embeddings_clip_vit_b32.npz", allow_pickle=True)
    clip_emb = clip_npz["embeddings"]
    text_index = json.loads(
        (cache_root / "text_embeddings_index.json").read_text("utf-8"),
    )["index"]
    norm = json.loads((cache_root / "normalization_train.json").read_text("utf-8"))
    global_block = norm["global"]
    mean = np.asarray(global_block["mean"], dtype=np.float32)
    std = np.asarray(global_block["std_clamped"], dtype=np.float32)
    obj_block = (
        global_block.get("obj_traj_root0_world")
        or global_block.get("obj_traj_canonical")
    )
    if obj_block is not None:
        obj_mean = np.asarray(obj_block["mean"], dtype=np.float32)
        obj_std = np.asarray(obj_block["std_clamped"], dtype=np.float32)
    else:
        obj_mean = obj_std = None
    return {
        "manifest": manifest,
        "manifest_idx": {(r["subset"], r["seq_id"]): i for i, r in enumerate(manifest)},
        "clip_emb": clip_emb,
        "text_index": text_index,
        "mean": mean,
        "std": std,
        "obj_mean": obj_mean,
        "obj_std": obj_std,
    }


def _load_clip(cache_root: Path, cache: dict, subset: str, seq_id: str, max_T: int):
    idx = cache["manifest_idx"].get((subset, seq_id))
    if idx is None:
        return None
    r = cache["manifest"][idx]
    npz = np.load(cache_root / r["npz_path"], allow_pickle=False)
    gt = npz["coarse_v1"].astype(np.float32)
    T = min(int(r["seq_len"]), gt.shape[0], max_T)
    gt = gt[:T]
    init = npz["init_coarse_v1"].astype(np.float32)
    init_norm = (init - cache["mean"]) / cache["std"]
    text = r.get("text", "")
    text_row = cache["text_index"].get(text)
    text_pool = (
        cache["clip_emb"][int(text_row)].astype(np.float32)
        if text_row is not None
        else np.zeros((512,), dtype=np.float32)
    )
    obj_traj_raw: np.ndarray | None = None
    obj_traj_norm: np.ndarray | None = None
    for cand in ("obj_traj_root0_world", "obj_traj_canonical"):
        if cand in npz.files:
            obj_traj_raw = npz[cand].astype(np.float32)[:T]
            break
    if obj_traj_raw is not None and cache["obj_mean"] is not None:
        obj_traj_norm = (obj_traj_raw - cache["obj_mean"]) / cache["obj_std"]
    return {
        "T": T,
        "gt": gt,
        "init_norm": init_norm,
        "text_pool": text_pool,
        "obj_traj_raw": obj_traj_raw,
        "obj_traj_norm": obj_traj_norm,
        "text": text,
    }


# ============================================================================
# Sample one trajectory
# ============================================================================


def _sample(
    model: CoarsePriorDiff,
    clip_data: dict,
    cfg_scale: float,
    sampler_seed: int,
    cache: dict,
    device: torch.device,
) -> np.ndarray:
    T = clip_data["T"]
    torch.manual_seed(sampler_seed)
    valid_mask = torch.ones(1, T, dtype=torch.bool, device=device)
    cond = {
        "text_pool": torch.from_numpy(clip_data["text_pool"]).unsqueeze(0).to(device),
        "init_coarse": torch.from_numpy(clip_data["init_norm"]).unsqueeze(0).to(device),
        "valid_mask": valid_mask,
    }
    if int(model.cfg.denoiser.obj_traj_dim) > 0:
        if clip_data["obj_traj_norm"] is None:
            raise RuntimeError("S1-O ckpt needs obj_traj but clip has none")
        cond["obj_traj"] = torch.from_numpy(
            clip_data["obj_traj_norm"]
        ).unsqueeze(0).to(device)
    with torch.no_grad():
        gen_norm = model.sample(
            shape=(1, T, COARSE_DIM), cond=cond,
            cfg_scale=cfg_scale, device=device,
            inpaint_frame0=True,
        )
    gen_norm = gen_norm.squeeze(0).cpu().numpy()
    return gen_norm * cache["std"] + cache["mean"]


# ============================================================================
# Plot one clip's panel
# ============================================================================


def _unwrap_yaw_from_sincos(coarse: np.ndarray) -> np.ndarray:
    s, c = coarse[:, 6], coarse[:, 7]
    return np.unwrap(np.arctan2(s, c))


def _plot_clip(
    gt: np.ndarray, plan_a: np.ndarray, s1o: np.ndarray,
    title: str, out_path: Path,
) -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = min(gt.shape[0], plan_a.shape[0], s1o.shape[0])
    fig, axes = plt.subplots(3, 2, figsize=(13, 11))

    # (0,0) root XZ trajectory (top-down)
    ax = axes[0, 0]
    ax.plot(gt[:T, 0], gt[:T, 2], label="GT", color="C0", linewidth=2)
    ax.plot(plan_a[:T, 0], plan_a[:T, 2], label="Plan A", color="C1", linestyle="--")
    ax.plot(s1o[:T, 0], s1o[:T, 2], label="S1-O", color="C2", linestyle=":")
    ax.scatter(gt[0, 0], gt[0, 2], marker="o", color="black", zorder=10, label="start")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)"); ax.set_title("Root XZ trajectory")
    ax.legend(fontsize=8); ax.set_aspect("equal", "datalim"); ax.grid(alpha=0.3)

    # (0,1) root height (Y) over time
    ax = axes[0, 1]
    ax.plot(gt[:T, 1], label="GT", color="C0", linewidth=2)
    ax.plot(plan_a[:T, 1], label="Plan A", color="C1", linestyle="--")
    ax.plot(s1o[:T, 1], label="S1-O", color="C2", linestyle=":")
    ax.set_xlabel("frame"); ax.set_ylabel("root Y (m)"); ax.set_title("Root height")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (1,0) facing yaw (unwrapped from sin/cos)
    ax = axes[1, 0]
    ax.plot(_unwrap_yaw_from_sincos(gt[:T]), label="GT", color="C0", linewidth=2)
    ax.plot(_unwrap_yaw_from_sincos(plan_a[:T]), label="Plan A", color="C1", linestyle="--")
    ax.plot(_unwrap_yaw_from_sincos(s1o[:T]), label="S1-O", color="C2", linestyle=":")
    ax.set_xlabel("frame"); ax.set_ylabel("yaw (rad, unwrapped)"); ax.set_title("Facing yaw")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (1,1) pelvis rot6d frame-to-frame change norm
    ax = axes[1, 1]
    for arr, lab, col, sty in [
        (gt, "GT", "C0", "-"), (plan_a, "Plan A", "C1", "--"), (s1o, "S1-O", "C2", ":"),
    ]:
        pr = arr[:T, 9:15]
        if T >= 2:
            ax.plot(np.linalg.norm(np.diff(pr, axis=0), axis=-1),
                    label=lab, color=col, linestyle=sty,
                    linewidth=2 if lab == "GT" else 1)
    ax.set_xlabel("frame"); ax.set_ylabel("‖Δpelvis_rot6d‖")
    ax.set_title("Pelvis rotation velocity")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (2,0) head height
    ax = axes[2, 0]
    ax.plot(gt[:T, 21], label="GT", color="C0", linewidth=2)
    ax.plot(plan_a[:T, 21], label="Plan A", color="C1", linestyle="--")
    ax.plot(s1o[:T, 21], label="S1-O", color="C2", linestyle=":")
    ax.set_xlabel("frame"); ax.set_ylabel("head height (m)"); ax.set_title("Head height")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (2,1) shoulder height
    ax = axes[2, 1]
    ax.plot(gt[:T, 22], label="GT", color="C0", linewidth=2)
    ax.plot(plan_a[:T, 22], label="Plan A", color="C1", linestyle="--")
    ax.plot(s1o[:T, 22], label="S1-O", color="C2", linestyle=":")
    ax.set_xlabel("frame"); ax.set_ylabel("shoulder height (m)"); ax.set_title("Shoulder height")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan-a-ckpt", type=Path, required=True,
        help="Plan A ckpt (e.g. seed 47 ckpt-040000.pt — val-loss champion across 6 seeds)",
    )
    parser.add_argument(
        "--s1o-ckpt", type=Path, required=True,
        help="S1-O ckpt (e.g. seed 45 ckpt-030000.pt — val-loss champion across 6 seeds)",
    )
    parser.add_argument(
        "--cache-plan-a", type=Path,
        default=Path("cache/stage1_coarse_v1_full"),
    )
    parser.add_argument(
        "--cache-s1o", type=Path,
        default=Path("cache/stage1_coarse_v1_objtraj_root0_world_round18_fix"),
    )
    parser.add_argument(
        "--selection-json", type=Path,
        default=Path("analyses/2026-05-20_round19_eval_selection.json"),
    )
    parser.add_argument(
        "--cfg-scales", type=float, nargs="+", default=[2.5],
        help="Sampler cfg_scale values. Per literature audit + Round-19 eval, "
             "cfg=2.5 is the sweet spot (closer to GT velocity). cfg=1.0 = no CFG; "
             "cfg=5.0 tends to overshoot in our data.",
    )
    parser.add_argument("--sampler-seed", type=int, default=42)
    parser.add_argument("--max-T", type=int, default=196)
    parser.add_argument(
        "--max-clips", type=int, default=None,
        help="Cap on number of selection clips (default = all in selection).",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("analyses/round19_visual_review"),
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    # ---- load ckpts ----
    print(f"[viz] loading Plan A ckpt: {args.plan_a_ckpt}")
    ckpt_a = torch.load(args.plan_a_ckpt, map_location="cpu", weights_only=False)
    model_a, loaded_ema_a = _build_model_from_ckpt(ckpt_a, prefer_ema=True)
    if int(model_a.cfg.denoiser.obj_traj_dim) != 0:
        raise SystemExit(
            f"Plan A ckpt has obj_traj_dim={model_a.cfg.denoiser.obj_traj_dim} != 0; wrong ckpt?"
        )
    print(f"[viz]   loaded_ema={loaded_ema_a}, obj_traj_dim=0 (object-free)")

    print(f"[viz] loading S1-O ckpt: {args.s1o_ckpt}")
    ckpt_o = torch.load(args.s1o_ckpt, map_location="cpu", weights_only=False)
    model_o, loaded_ema_o = _build_model_from_ckpt(ckpt_o, prefer_ema=True)
    if int(model_o.cfg.denoiser.obj_traj_dim) != 9:
        raise SystemExit(
            f"S1-O ckpt has obj_traj_dim={model_o.cfg.denoiser.obj_traj_dim} != 9; wrong ckpt?"
        )
    print(f"[viz]   loaded_ema={loaded_ema_o}, obj_traj_dim=9 (root0_world)")

    device = torch.device(args.device)
    model_a = model_a.to(device).eval()
    model_o = model_o.to(device).eval()

    # ---- load caches ----
    print(f"[viz] loading Plan A cache: {args.cache_plan_a}")
    cache_a = _load_cache(args.cache_plan_a, split="val")
    print(f"[viz] loading S1-O cache:   {args.cache_s1o}")
    cache_o = _load_cache(args.cache_s1o, split="val")

    # ---- load selection ----
    sel = json.loads(args.selection_json.read_text(encoding="utf-8"))
    entries = sel["selected"]
    if args.max_clips is not None:
        entries = entries[:args.max_clips]
    print(f"[viz] selection: {len(entries)} clips")

    # ---- output dirs ----
    plots_dir = args.output_dir / "plots"
    traj_dir = args.output_dir / "trajectories"
    plots_dir.mkdir(parents=True, exist_ok=True)
    traj_dir.mkdir(parents=True, exist_ok=True)

    # ---- run per clip × cfg ----
    t_start = time.time()
    summary_rows: list[dict[str, Any]] = []
    for e in entries:
        subset = e["subset"]
        seq_id = e["seq_id"]
        clip_a = _load_clip(args.cache_plan_a, cache_a, subset, seq_id, args.max_T)
        clip_o = _load_clip(args.cache_s1o, cache_o, subset, seq_id, args.max_T)
        if clip_a is None or clip_o is None:
            print(f"[viz]   SKIP {subset}/{seq_id} (not in cache)")
            continue
        # Sanity: same GT trajectory in both caches.
        T = min(clip_a["T"], clip_o["T"])
        gt = clip_a["gt"][:T]

        for cfg in args.cfg_scales:
            t0 = time.time()
            plan_a_traj = _sample(model_a, clip_a, cfg, args.sampler_seed, cache_a, device)
            s1o_traj = _sample(model_o, clip_o, cfg, args.sampler_seed, cache_o, device)
            elapsed = time.time() - t0

            tag = f"{subset}_{seq_id}__cfg{cfg:.1f}".replace(".", "_")
            png_path = plots_dir / f"{tag}.png"
            npz_path = traj_dir / f"{tag}.npz"

            _plot_clip(
                gt, plan_a_traj, s1o_traj,
                title=f"{subset}/{seq_id}  T={T}  cfg={cfg:.1f}  text={clip_a['text'][:60]!r}",
                out_path=png_path,
            )
            np.savez_compressed(
                npz_path,
                gt=gt, plan_a=plan_a_traj[:T], s1o=s1o_traj[:T],
                text=clip_a["text"],
                cfg_scale=float(cfg),
                sampler_seed=int(args.sampler_seed),
                T=int(T),
                subset=subset, seq_id=seq_id,
            )
            print(f"[viz]   {tag}  ({elapsed:.1f}s)  → {png_path.name}")
            summary_rows.append({
                "subset": subset, "seq_id": seq_id, "cfg_scale": cfg,
                "T": int(T), "elapsed_s": float(elapsed),
                "png": str(png_path), "npz": str(npz_path),
            })

    elapsed = time.time() - t_start
    print(f"[viz] done — {len(summary_rows)} panels in {elapsed:.1f}s")

    # ---- summary md ----
    md_lines: list[str] = []
    md_lines.append("# Round-19 Visual Review — Plan A (seed 47, ckpt-040000) vs S1-O (seed 45, ckpt-030000)")
    md_lines.append("")
    md_lines.append(f"- Plan A ckpt: `{args.plan_a_ckpt}`  (loaded_ema={loaded_ema_a})")
    md_lines.append(f"- S1-O ckpt:   `{args.s1o_ckpt}`  (loaded_ema={loaded_ema_o})")
    md_lines.append(f"- Selection:   `{args.selection_json}` (N={len(entries)})")
    md_lines.append(f"- cfg_scales:  {args.cfg_scales}")
    md_lines.append(f"- sampler seed: {args.sampler_seed}")
    md_lines.append(f"- frame-0 inpainting: True (init pose anchoring per ObjTrajHintBlock zero-init contract)")
    md_lines.append("")
    md_lines.append("## Per-clip panels")
    md_lines.append("")
    for r in summary_rows:
        rel = Path(r["png"]).relative_to(args.output_dir.parent)
        md_lines.append(f"### {r['subset']} / {r['seq_id']} — cfg {r['cfg_scale']:.1f}")
        md_lines.append("")
        md_lines.append(f"![panel]({rel.as_posix()})")
        md_lines.append("")
    (args.output_dir / "summary.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[viz] wrote {args.output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
