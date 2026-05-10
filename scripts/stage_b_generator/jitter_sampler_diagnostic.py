"""Sampler-vs-stability diagnostic — Step B of
analyses/claude_code_v10_after_fkposfix_strategy.md §6.

Question being answered: is the residual high-frequency wobble in v10
fkposfix predictions caused by DDPM ancestral sampling stochasticity
(per-step noise injection compounding through the FK chain), or is it
the learned model / loss?

Approach: same checkpoint + same seed + same conditioning + same plan
(GT plan), only the reverse-diffusion sampler changes:

    "ddpm"      : ancestral DDPM (current default; stochastic)
    "ddim_eta0" : DDIM with η=0 (Song et al., ICLR 2021); deterministic
    "ddpm_det"  : DDPM posterior mean without noise injection (cheap
                  deterministic variant)

Metrics emphasised in §6.3 (stable-support jitter on pseudo-label
sitting / object-support segments). Plus video renders of all three
samplers for direct visual comparison vs GT.

Decision rule (§6.4):
    DDIM ≈ GT-stable   → residual jitter is sampler stochasticity;
                          prefer deterministic sampling for visual
                          generation; don't add training-side stability
                          loss yet.
    DDIM still jittery → residual jitter is learned/loss-side; add
                          support-aware stability loss only on
                          stable-segment frames.

Usage::

    python scripts/stage_b_generator/jitter_sampler_diagnostic.py \\
        --config configs/training/anchordiff_v10_plan_tokens_gt_overfit_fkposfix.yaml \\
        --ckpt   runs/training/stageB_anchordiff_v10_plan_tokens_gt_overfit_fkposfix/final.pt \\
        --output analyses/2026-05-10_v10_fkposfix_sampler_jitter.json \\
        --md     analyses/2026-05-10_v10_fkposfix_sampler_jitter_report.md \\
        --render-dir runs/visualizations/anchordiff_v10_fkposfix_sampler_jitter \\
        --seeds 42,43,44
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plan_condition_diagnostics import (  # type: ignore[import-not-found]
    _build_cond, _build_dataset, _build_model,
)
from piano.data.dataset import collate_hoi
from piano.training.smpl_kinematics import (
    fk_from_global_rotations as _fk_from_global,
    rotation_6d_to_matrix as _rot6d_to_mat,
)
from piano.utils.clip_utils import load_clip_text_encoder


# ---------------------------------------------------------------------------
# Stable-support jitter metrics (§6.3)
# ---------------------------------------------------------------------------


def _stable_support_mask(
    support: torch.Tensor,           # (B, T) int  (0 = none, 1 = object, 2 = hand_on_obj)
    contact_state: torch.Tensor,     # (B, T, P) float
    seq_mask: torch.Tensor,          # (B, T) bool
    min_run: int = 8,
) -> torch.Tensor:
    """Build a (B, T) bool mask marking frames inside stable-support runs.

    A frame is considered stable-support when:
      * support label != 0 (the body is being supported by the object), or
      * pelvis contact_state > 0.5 (sitting / leaning).
    AND the surrounding ±min_run/2 window is also stable (suppresses
    transition frames).
    """
    B, T = support.shape
    device = support.device
    raw = (support != 0)                                                    # (B, T)
    if contact_state is not None and contact_state.shape[-1] >= 5:
        pelvis = contact_state[..., 4] > 0.5                                # (B, T)
        raw = raw | pelvis
    raw = raw & seq_mask
    if min_run <= 1:
        return raw
    # Erosion-style: keep only frames whose ±half window is fully stable.
    half = min_run // 2
    out = raw.clone()
    for shift in range(1, half + 1):
        shifted_left = torch.roll(raw, shifts=-shift, dims=-1)
        shifted_right = torch.roll(raw, shifts=shift, dims=-1)
        # Edge frames where the roll wraps — clear them.
        if shift > 0:
            shifted_left[..., -shift:] = False
            shifted_right[..., :shift] = False
        out = out & shifted_left & shifted_right
    return out


def _jitter_metrics(
    jpos_pred: torch.Tensor,           # (B, T, 22, 3) world joints
    stable_mask: torch.Tensor,         # (B, T) bool — frames considered stable-support
    seq_mask: torch.Tensor,            # (B, T) bool
) -> dict[str, float]:
    """Compute the five stable-support jitter metrics from §6.3.

    All values returned in cm or cm/frame.
    """
    if not seq_mask.any():
        return {k: 0.0 for k in (
            "stable_support_root_vel_rms_cm_per_frame",
            "stable_support_root_acc_rms_cm_per_frame2",
            "stable_support_local_vel_rms_cm_per_frame",
            "stable_support_local_acc_rms_cm_per_frame2",
            "global_root_vel_rms_cm_per_frame",
        )}

    root = jpos_pred[..., 0, :]                                             # (B, T, 3)
    # Local joints: subtract root per frame for shape-only motion.
    local = jpos_pred - root.unsqueeze(-2)                                  # (B, T, 22, 3)

    # Frame-difference velocity / acceleration (in metres / frame, ×100 → cm).
    root_vel = root[..., 1:, :] - root[..., :-1, :]                         # (B, T-1, 3)
    root_acc = root_vel[..., 1:, :] - root_vel[..., :-1, :]                 # (B, T-2, 3)
    local_vel = local[..., 1:, :, :] - local[..., :-1, :, :]                # (B, T-1, 22, 3)
    local_acc = local_vel[..., 1:, :, :] - local_vel[..., :-1, :, :]        # (B, T-2, 22, 3)

    # Stable-mask alignment: vel covers (T-1) edges between consecutive
    # frames; require both endpoints to be stable. Acc covers (T-2) edges.
    vel_mask = stable_mask[..., 1:] & stable_mask[..., :-1]                 # (B, T-1)
    acc_mask = vel_mask[..., 1:] & vel_mask[..., :-1]                       # (B, T-2)

    def _rms(x: torch.Tensor, m: torch.Tensor) -> float:
        if not m.any():
            return 0.0
        # RMS over the masked frames (and over joint axis for local).
        sq = x.pow(2).sum(dim=-1)                                           # (..., 22) or (..., )
        if sq.dim() == m.dim() + 1:
            sq = sq.mean(dim=-1)                                            # average over joints
        rms = (sq[m].mean()).sqrt().item() if m.any() else 0.0
        return float(rms) * 100.0                                           # m → cm

    return {
        "stable_support_root_vel_rms_cm_per_frame": _rms(root_vel, vel_mask),
        "stable_support_root_acc_rms_cm_per_frame2": _rms(root_acc, acc_mask),
        "stable_support_local_vel_rms_cm_per_frame": _rms(local_vel, vel_mask),
        "stable_support_local_acc_rms_cm_per_frame2": _rms(local_acc, acc_mask),
        "global_root_vel_rms_cm_per_frame": _rms(root_vel, seq_mask[..., 1:] & seq_mask[..., :-1]),
        "stable_frames_total": int(stable_mask.sum().item()),
    }


# ---------------------------------------------------------------------------
# FK helper (135 → joints)
# ---------------------------------------------------------------------------


def _fk_decode_135(x0: torch.Tensor, rest_offsets: torch.Tensor, T: int) -> torch.Tensor:
    rot_6d = x0[..., :132].view(1, T, 22, 6).float()
    root_world = x0[..., 132:135].float()
    rot_mat = _rot6d_to_mat(rot_6d)
    rest_per_frame = rest_offsets.unsqueeze(1).expand(1, T, 22, 3)
    return _fk_from_global(rot_mat, rest_per_frame, root_world)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--md", type=Path, required=True)
    parser.add_argument("--render-dir", type=Path, default=None)
    parser.add_argument("--seeds", type=str, default="42,43,44",
                        help="Comma-separated DDPM seeds for stochasticity check.")
    parser.add_argument("--clip-idx", type=int, default=0)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--bucket", default="train", choices=["train", "val"])
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--stable-min-run", type=int, default=8,
                        help="Min stable-support run length (frames) for the jitter mask.")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- Dataset / model ----------
    ds = _build_dataset(cfg, args.bucket, augment=False)
    overfit_n = int(cfg.data.get("overfit_n_clips", 0))
    if overfit_n > 0:
        ds = Subset(ds, list(range(min(overfit_n, len(ds)))))
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_hoi, num_workers=0)
    main_batch = None
    for i, b in enumerate(loader):
        if i == args.clip_idx:
            main_batch = b
            break
    if main_batch is None:
        raise RuntimeError("Could not find main clip")

    model, object_encoder, z_dims = _build_model(cfg, device)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("model", state))
    if "object_encoder" in state:
        object_encoder.load_state_dict(state["object_encoder"])
    elif "extra_modules" in state and "object_encoder" in state["extra_modules"]:
        object_encoder.load_state_dict(state["extra_modules"]["object_encoder"])

    clip_model = load_clip_text_encoder(
        device=device,
        model_name=str(cfg.model.text_encoder.clip_version),
        download_root=str(cfg.model.text_encoder.get("download_root", "cache/clip")),
    )

    cond_main, T = _build_cond(main_batch, model, object_encoder, clip_model, z_dims, cfg, device)

    # GT plan (we only use GT plan in this diagnostic — sampler is the only variable)
    plan_keys = [
        "anchor_time", "anchor_part", "anchor_target_local",
        "anchor_target_world", "anchor_type", "anchor_phase",
        "anchor_support", "anchor_conf", "anchor_mask",
        "segment_start", "segment_end", "segment_part",
        "segment_target_summary_local", "segment_phase",
        "segment_support", "segment_conf", "segment_mask",
    ]
    plan_gt = {k: main_batch[f"plan_{k}"].to(device) for k in plan_keys}
    cond_main["interaction_plan"] = plan_gt

    # GT geometry
    rest_offsets = main_batch["rest_offsets"].to(device).float()
    seq_len = main_batch["seq_len"].to(device)
    seq_idx = torch.arange(T, device=device).unsqueeze(0)
    seq_mask = (seq_idx < seq_len.unsqueeze(1))
    joints_gt = main_batch["joints"].to(device).float()
    contact_state = main_batch["contact_state"].to(device).float()
    support = main_batch["support"].to(device).long()
    stable_mask = _stable_support_mask(
        support, contact_state, seq_mask, min_run=args.stable_min_run,
    )

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    # ---------- Sample under each (sampler, seed) combination ----------
    results: dict[str, dict] = {}
    motion_outputs: dict[str, torch.Tensor] = {}
    print(f"Stable-support frames: {int(stable_mask.sum())} / {int(seq_mask.sum())}")

    samplers_to_run: list[tuple[str, int]] = []
    for s in seeds:
        samplers_to_run.append(("ddpm", s))
    samplers_to_run.append(("ddim_eta0", seeds[0]))
    samplers_to_run.append(("ddpm_det", seeds[0]))

    for samp, seed in samplers_to_run:
        torch.manual_seed(seed)
        with torch.no_grad():
            x0 = model.sample(
                cond=cond_main, seq_length=T, cfg_scale=args.cfg_scale,
                replacement="none", output_skip=False, sampler=samp,
            )
        jpos = _fk_decode_135(x0, rest_offsets, T)
        m = _jitter_metrics(jpos, stable_mask, seq_mask)
        # Also accuracy reference vs GT
        err = (jpos - joints_gt).pow(2).sum(-1).sqrt().mean(-1)             # (B, T)
        m["mean_joint_err_cm"] = float((err * seq_mask.float()).sum() / seq_mask.float().sum().clamp_min(1.0)) * 100.0
        # Reproducibility marker for DDPM seeds.
        m["sampler"] = samp
        m["seed"] = seed
        key = f"{samp}_seed{seed}"
        results[key] = m
        motion_outputs[key] = x0.cpu()
        print(f"  {key:18s}  root_vel_rms={m['stable_support_root_vel_rms_cm_per_frame']:.3f}  "
              f"local_vel_rms={m['stable_support_local_vel_rms_cm_per_frame']:.3f}  "
              f"local_acc_rms={m['stable_support_local_acc_rms_cm_per_frame2']:.3f}  "
              f"err={m['mean_joint_err_cm']:.2f}")

    # ---------- GT-motion baseline (jitter on GT itself) ----------
    gt_jitter = _jitter_metrics(joints_gt, stable_mask, seq_mask)
    gt_jitter["mean_joint_err_cm"] = 0.0
    gt_jitter["sampler"] = "GT"
    gt_jitter["seed"] = -1
    results["GT"] = gt_jitter
    print(f"  {'GT':18s}  root_vel_rms={gt_jitter['stable_support_root_vel_rms_cm_per_frame']:.3f}  "
          f"local_vel_rms={gt_jitter['stable_support_local_vel_rms_cm_per_frame']:.3f}  "
          f"local_acc_rms={gt_jitter['stable_support_local_acc_rms_cm_per_frame2']:.3f}")

    # ---------- DDPM-seed dispersion (cross-seed variability) ----------
    if len(seeds) > 1:
        # Variance across seeds at each frame on root pos
        seed_keys = [f"ddpm_seed{s}" for s in seeds]
        stack = torch.stack([motion_outputs[k] for k in seed_keys], dim=0)  # (S, 1, T, 135)
        root_stack = stack[..., 132:135]                                     # (S, 1, T, 3)
        root_std_per_frame = root_stack.std(dim=0).squeeze(0)                # (T, 3)
        valid_T = int(seq_len[0].item())
        results["ddpm_seed_dispersion"] = {
            "root_xyz_std_cm_mean": float(root_std_per_frame[:valid_T].pow(2).sum(-1).sqrt().mean()) * 100.0,
            "root_xyz_std_cm_max": float(root_std_per_frame[:valid_T].pow(2).sum(-1).sqrt().max()) * 100.0,
        }

    # ---------- Optional rendering ----------
    if args.render_dir is not None:
        from piano.inference.visualize_motion import render_motion_video
        args.render_dir.mkdir(parents=True, exist_ok=True)
        valid_T = int(seq_len[0].item())
        seq_id = main_batch["seq_id"][0]
        subset = main_batch["subset"][0]
        text = main_batch["text"][0]
        obj_pos_np = main_batch["object_positions"].squeeze(0).cpu().numpy()[:valid_T]
        obj_rot_np = main_batch["object_rotations"].squeeze(0).cpu().numpy()[:valid_T]
        obj_pc_np = main_batch["object_pc"].squeeze(0).cpu().numpy()
        joints_gt_np = joints_gt.squeeze(0).cpu().numpy()[:valid_T]

        gt_out = args.render_dir / f"{subset}_{seq_id}_gt.mp4"
        print(f"  rendering GT → {gt_out.name}")
        render_motion_video(
            joints=joints_gt_np, output_path=gt_out, fps=args.fps,
            title=f"{subset}/{seq_id}\n[GT]\ntext: {text[:80]}",
            object_positions=obj_pos_np, object_rotations=obj_rot_np, object_pc=obj_pc_np,
        )
        # Render one DDPM seed (the first), the deterministic DDIM, and ddpm_det
        for samp, seed in [("ddpm", seeds[0]), ("ddim_eta0", seeds[0]), ("ddpm_det", seeds[0])]:
            key = f"{samp}_seed{seed}"
            x0 = motion_outputs[key].to(device)
            jpos = _fk_decode_135(x0, rest_offsets, T)
            jpos_np = jpos.squeeze(0).cpu().numpy()[:valid_T]
            r = results[key]
            tag = key.replace("/", "__")
            pred_out = args.render_dir / f"{subset}_{seq_id}_predicted_{tag}.mp4"
            title = (
                f"{subset}/{seq_id}\n[{samp} seed={seed}]\n"
                f"local_vel_rms={r['stable_support_local_vel_rms_cm_per_frame']:.2f} "
                f"err={r['mean_joint_err_cm']:.2f}cm"
            )
            print(f"  rendering {key:25s} → {pred_out.name}")
            render_motion_video(
                joints=jpos_np, output_path=pred_out, fps=args.fps, title=title,
                object_positions=obj_pos_np, object_rotations=obj_rot_np, object_pc=obj_pc_np,
            )

    # ---------- JSON ----------
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote JSON to {args.output}")

    # ---------- Markdown ----------
    md: list[str] = []
    md.append("# v10 fkposfix — sampler jitter diagnostic\n")
    md.append("**Date:** 2026-05-10  ")
    md.append(f"**Config:** `{args.config}`  ")
    md.append(f"**Checkpoint:** `{args.ckpt}`  ")
    md.append(f"**cfg_scale:** {args.cfg_scale}  ")
    md.append(f"**Seeds (DDPM):** {seeds}  ")
    md.append(f"**Stable-support frames in clip:** {gt_jitter['stable_frames_total']} / {int(seq_mask.sum().item())}\n")
    md.append("Per spec [analyses/claude_code_v10_after_fkposfix_strategy.md](analyses/claude_code_v10_after_fkposfix_strategy.md) §6: ")
    md.append("isolate whether residual high-frequency wobble in v10 fkposfix is sampler-side ")
    md.append("(DDPM ancestral noise injection) or learned/loss-side. Same checkpoint, same plan, ")
    md.append("only sampler changes.\n")
    md.append("## Stable-support jitter table\n")
    md.append("| sampler | seed | root vel rms cm/fr | root acc rms cm/fr² | local vel rms cm/fr | local acc rms cm/fr² | global root vel rms cm/fr | mean err cm |")
    md.append("|---|---|---|---|---|---|---|---|")
    order = ["GT"] + [f"ddpm_seed{s}" for s in seeds] + [f"ddim_eta0_seed{seeds[0]}", f"ddpm_det_seed{seeds[0]}"]
    for k in order:
        r = results[k]
        md.append(
            f"| {r['sampler']} | {r['seed']} | "
            f"{r['stable_support_root_vel_rms_cm_per_frame']:.3f} | "
            f"{r['stable_support_root_acc_rms_cm_per_frame2']:.3f} | "
            f"{r['stable_support_local_vel_rms_cm_per_frame']:.3f} | "
            f"{r['stable_support_local_acc_rms_cm_per_frame2']:.3f} | "
            f"{r['global_root_vel_rms_cm_per_frame']:.3f} | "
            f"{r['mean_joint_err_cm']:.2f} |"
        )
    md.append("")
    if "ddpm_seed_dispersion" in results:
        d = results["ddpm_seed_dispersion"]
        md.append(f"\n## DDPM seed dispersion (root-position std across {len(seeds)} seeds)\n")
        md.append(f"- mean per-frame root-xyz std: **{d['root_xyz_std_cm_mean']:.2f} cm**")
        md.append(f"- max per-frame root-xyz std:  **{d['root_xyz_std_cm_max']:.2f} cm**\n")
        md.append("If this is > a few cm, DDPM stochasticity meaningfully shifts the global trajectory across seeds.")

    # Pass / fail per §6.4
    gt_local = results["GT"]["stable_support_local_vel_rms_cm_per_frame"]
    ddpm_local = results[f"ddpm_seed{seeds[0]}"]["stable_support_local_vel_rms_cm_per_frame"]
    ddim_local = results[f"ddim_eta0_seed{seeds[0]}"]["stable_support_local_vel_rms_cm_per_frame"]
    ddpm_det_local = results[f"ddpm_det_seed{seeds[0]}"]["stable_support_local_vel_rms_cm_per_frame"]

    if gt_jitter["stable_frames_total"] == 0:
        md.append("\n## Verdict\n")
        md.append("**INSUFFICIENT_DATA**: no stable-support frames in this clip's pseudo-labels. "
                  "Re-run on a clip with sustained sitting/contact intervals.")
    else:
        ratio_ddpm = ddpm_local / max(gt_local, 1e-6)
        ratio_ddim = ddim_local / max(gt_local, 1e-6)
        ratio_ddpm_det = ddpm_det_local / max(gt_local, 1e-6)
        md.append("\n## Verdict (§6.4)\n")
        md.append(f"- GT stable-support local-vel RMS:    **{gt_local:.3f} cm/frame** (lower bound)")
        md.append(f"- DDPM stable-support local-vel RMS:  {ddpm_local:.3f} cm/frame (× {ratio_ddpm:.2f} GT)")
        md.append(f"- DDIM(η=0) stable-support local-vel: {ddim_local:.3f} cm/frame (× {ratio_ddim:.2f} GT)")
        md.append(f"- DDPM-det stable-support local-vel:  {ddpm_det_local:.3f} cm/frame (× {ratio_ddpm_det:.2f} GT)\n")
        if ratio_ddim < 0.7 * ratio_ddpm:
            md.append("**Reading:** DDIM(η=0) local-vel RMS is < 70% of DDPM's. Most of the residual jitter "
                      "is **sampler stochasticity**. `FAIL_JITTER_SAMPLER`. Prefer deterministic sampling "
                      "for visual generation; do not add training-side stability loss yet.")
        elif ratio_ddim > 0.95 * ratio_ddpm:
            md.append("**Reading:** DDIM(η=0) jitter is essentially the same as DDPM. The wobble is "
                      "**learned / loss-side**. `FAIL_JITTER_LEARNED`. Add support-aware stability loss "
                      "on stable-segment frames (per spec §6.5). Do not raise the global velocity weight.")
        else:
            md.append("**Reading:** DDIM(η=0) is in between — partial sampler contribution. Mixed verdict; "
                      "consider both deterministic sampling and a small support-aware stability loss.")

    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote MD to {args.md}")


if __name__ == "__main__":
    main()
