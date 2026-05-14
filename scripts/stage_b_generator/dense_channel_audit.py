"""Dense-channel audit on a PLAN_ONLY-trained checkpoint.

Per analyses/claude_code_v11_n10_followup_dense_audit_full_baseline.md
Task A. Three checks:

* **A.3 Batch zeroing check** — load one train batch, reproduce the
  trainer's cond construction with `zero_z_int_for_stageB=true` and
  `zero_dense_contact_target_for_stageB=true`, and report norms before
  and after the zeroing. Confirms training ran in the intended
  PLAN_ONLY mode.

* **A.4 Input-projection weight-norm audit** — inspect the denoiser's
  ``in_proj`` weight matrix per input block (x_t / z_int / object_pose /
  object_dense_target / plan_hint). If z_int or dense-target columns
  have larger-than-expected norms, the model may have adapted to those
  channels despite the zeroing.

* **A.5 Dense-scale sweep** — sample with α·real_z_int + α·real_dense_target
  for α ∈ {0.0, 0.25, 0.5, 1.0}, GT plan, fixed seed. Reports
  far-unobs / root-aligned / motion-135 delta / stable-support metrics.
  Tells us whether the OOD-FULL improvement is monotonic (stable bias)
  or only fires at α=1 (artefact).

Usage::

    python scripts/stage_b_generator/dense_channel_audit.py \\
      --config configs/training/anchordiff_v11_per_part_tokens_planonly_N10_stableloss.yaml \\
      --ckpt   runs/training/stageB_anchordiff_v11_per_part_tokens_planonly_N10_stableloss/final.pt \\
      --output analyses/2026-05-10_v11_planonly_N10_dense_audit.json \\
      --md     analyses/2026-05-10_v11_planonly_N10_dense_audit.md
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
    _build_cond, _build_dataset, _build_model, _compute_metrics,
)
from piano.data.dataset import collate_hoi
from piano.training.smpl_kinematics import (
    fk_from_global_rotations as _fk_from_global,
    rotation_6d_to_matrix as _rot6d_to_mat,
)
from piano.utils.clip_utils import load_clip_text_encoder


def _fk_decode_135(
    x0: torch.Tensor, rest_offsets: torch.Tensor, T: int,
) -> torch.Tensor:
    rot_6d = x0[..., :132].view(1, T, 22, 6).float()
    root_world = x0[..., 132:135].float()
    rot_mat = _rot6d_to_mat(rot_6d)
    rest_per_frame = rest_offsets.unsqueeze(1).expand(1, T, 22, 3)
    return _fk_from_global(rot_mat, rest_per_frame, root_world)


# ---------------------------------------------------------------------------
# A.3 batch zeroing check
# ---------------------------------------------------------------------------


def _batch_zeroing_check(cond: dict, cfg) -> dict:
    """Reproduce trainer's zero_z_int_for_stageB / zero_dense_contact_target_for_stageB
    logic and report norms before and after zeroing."""
    z_int = cond["z_int"]
    object_traj = cond["object_world_traj"]

    pose_dims = 9
    dense_target_dims = max(object_traj.shape[-1] - pose_dims, 0)

    z_int_before = float(z_int.norm())
    pose_before = float(object_traj[..., :pose_dims].norm())
    dense_target_before = float(object_traj[..., pose_dims:].norm()) if dense_target_dims else 0.0

    zero_z_int = bool(cfg.model.get("zero_z_int_for_stageB", False))
    zero_dense_target = bool(
        cfg.model.get("zero_dense_contact_target_for_stageB", False)
    )

    z_int_after = z_int.clone()
    object_traj_after = object_traj.clone()
    if zero_z_int:
        z_int_after = torch.zeros_like(z_int_after)
    if zero_dense_target and object_traj_after.shape[-1] >= 24:
        object_traj_after[..., pose_dims:] = 0.0

    z_int_after_norm = float(z_int_after.norm())
    pose_after = float(object_traj_after[..., :pose_dims].norm())
    dense_target_after = float(object_traj_after[..., pose_dims:].norm()) if dense_target_dims else 0.0

    plan_dict = cond.get("interaction_plan", {})
    anchor_mask = plan_dict.get("anchor_mask")
    segment_mask = plan_dict.get("segment_mask")

    return {
        "config_zero_z_int_for_stageB": zero_z_int,
        "config_zero_dense_contact_target_for_stageB": zero_dense_target,
        "z_int_norm_before_zeroing": z_int_before,
        "z_int_norm_after_zeroing": z_int_after_norm,
        "object_traj_pose_norm_before_zeroing": pose_before,
        "object_traj_pose_norm_after_zeroing": pose_after,
        "object_traj_dense_target_norm_before_zeroing": dense_target_before,
        "object_traj_dense_target_norm_after_zeroing": dense_target_after,
        "anchor_mask_sum": (
            int(anchor_mask.sum().item()) if anchor_mask is not None else None
        ),
        "segment_mask_sum": (
            int(segment_mask.sum().item()) if segment_mask is not None else None
        ),
        "plan_token_count_per_clip": (
            int(anchor_mask.sum(-1).item())
            if anchor_mask is not None and anchor_mask.shape[0] == 1
            else None
        ),
    }


# ---------------------------------------------------------------------------
# A.4 input projection weight-norm audit
# ---------------------------------------------------------------------------


def _input_projection_audit(model, cfg) -> dict:
    """Compute per-block weight Frobenius norms of denoiser.in_proj.

    Layout is built by AnchorDenoiser.__init__ as:
        cat(x_t, cond_motion, z_int, object_world_traj, [plan_hint])
    The total input width is `per_frame_in`; we slice column-blocks to
    isolate each contributing channel.
    """
    denoiser_cfg = model.cfg.denoiser
    motion_dim = int(denoiser_cfg.motion_dim)
    cond_motion_dim = int(denoiser_cfg.cond_motion_dim)
    z_int_dim = int(denoiser_cfg.z_int.total)
    object_traj_dim = int(denoiser_cfg.object_traj_dim)
    use_plan = bool(denoiser_cfg.use_interaction_plan)
    use_hint = bool(denoiser_cfg.plan_use_context_hint)
    d_hint = int(denoiser_cfg.plan_d_hint) if use_plan and use_hint else 0
    pose_dims = 9  # COM 3 + rot6d 6

    W = model.denoiser.in_proj.weight.detach().cpu()                # (d_model, per_frame_in)

    # Block boundaries
    a = 0
    x_t_slice = slice(a, a + motion_dim);                  a += motion_dim
    cond_motion_slice = slice(a, a + cond_motion_dim);     a += cond_motion_dim
    z_int_slice = slice(a, a + z_int_dim);                 a += z_int_dim
    object_traj_slice = slice(a, a + object_traj_dim);
    object_pose_slice = slice(a, a + pose_dims)
    object_dense_target_slice = slice(a + pose_dims, a + object_traj_dim)
    a += object_traj_dim
    plan_hint_slice = slice(a, a + d_hint);                a += d_hint
    total = a
    assert W.shape[1] == total, (
        f"input width mismatch: W has {W.shape[1]}, expected {total}"
    )

    def _frob(s: slice) -> float:
        if s.start == s.stop:
            return 0.0
        return float(W[:, s].norm().item())

    def _per_dim(s: slice) -> float:
        if s.start == s.stop:
            return 0.0
        return float(W[:, s].norm(dim=0).mean().item())

    return {
        "in_proj_total_input_width": int(W.shape[1]),
        "x_t_block_norm": _frob(x_t_slice),
        "x_t_per_dim_norm": _per_dim(x_t_slice),
        "cond_motion_block_norm": _frob(cond_motion_slice),
        "z_int_block_norm": _frob(z_int_slice),
        "z_int_per_dim_norm": _per_dim(z_int_slice),
        "object_pose_block_norm": _frob(object_pose_slice),
        "object_pose_per_dim_norm": _per_dim(object_pose_slice),
        "object_dense_target_block_norm": _frob(object_dense_target_slice),
        "object_dense_target_per_dim_norm": _per_dim(object_dense_target_slice),
        "plan_hint_block_norm": _frob(plan_hint_slice),
        "plan_hint_per_dim_norm": _per_dim(plan_hint_slice),
    }


# ---------------------------------------------------------------------------
# A.5 dense-scale sweep
# ---------------------------------------------------------------------------


def _dense_scale_cond(base_cond: dict, alpha: float, pose_dims: int = 9) -> dict:
    """Scale the dense (z_int) and the dense-target portion of object_traj
    by ``alpha``. Object pose dims (COM + rot6d) are kept intact.

    α=0 reproduces the trainer's PLAN_ONLY zeroing exactly.
    α=1 injects the full real dense channels (matches diagnostic FULL mode).
    α∈(0,1) linearly interpolates.
    """
    out = dict(base_cond)
    out["z_int"] = base_cond["z_int"] * float(alpha)
    obj_traj = base_cond["object_world_traj"].clone()
    if obj_traj.shape[-1] > pose_dims:
        obj_traj[..., pose_dims:] = obj_traj[..., pose_dims:] * float(alpha)
    out["object_world_traj"] = obj_traj
    return out


def _stable_support_jitter(
    jpos: torch.Tensor,           # (B, T, 22, 3)
    jpos_gt: torch.Tensor,
    seq_mask: torch.Tensor,       # (B, T) bool
    support: torch.Tensor,        # (B, T) long
    contact_state: torch.Tensor,  # (B, T, P)
) -> dict[str, float]:
    """Subset of the metrics in jitter_sampler_diagnostic._jitter_metrics.
    Computed against the same stable-support mask used by training."""
    pelvis = contact_state[..., 4] > 0.5
    raw = ((support != 0) | pelvis) & seq_mask
    half = 2
    mask_t = raw.clone()
    for shift in range(1, half + 1):
        left = torch.roll(raw, shifts=-shift, dims=-1)
        right = torch.roll(raw, shifts=shift, dims=-1)
        if shift > 0:
            left[..., -shift:] = False
            right[..., :shift] = False
        mask_t = mask_t & left & right

    root = jpos[..., 0, :]
    root_gt = jpos_gt[..., 0, :]
    local = jpos - root.unsqueeze(-2)
    local_gt = jpos_gt - root_gt.unsqueeze(-2)

    vel_root = root[..., 1:, :] - root[..., :-1, :]
    vel_root_gt = root_gt[..., 1:, :] - root_gt[..., :-1, :]
    vel_local = local[..., 1:, :, :] - local[..., :-1, :, :]
    vel_local_gt = local_gt[..., 1:, :, :] - local_gt[..., :-1, :, :]
    vel_mask = mask_t[..., 1:] & mask_t[..., :-1]

    if not vel_mask.any():
        return {
            "stable_root_vel_rms_pred_cm_per_frame": 0.0,
            "stable_root_vel_rms_gt_cm_per_frame": 0.0,
            "stable_local_vel_rms_pred_cm_per_frame": 0.0,
            "stable_local_vel_rms_gt_cm_per_frame": 0.0,
            "stable_frames_total": 0,
        }

    def _rms(x: torch.Tensor, m: torch.Tensor) -> float:
        sq = x.pow(2).sum(dim=-1)
        if sq.dim() == m.dim() + 1:
            sq = sq.mean(dim=-1)
        return float(sq[m].mean().sqrt()) * 100.0

    return {
        "stable_root_vel_rms_pred_cm_per_frame": _rms(vel_root, vel_mask),
        "stable_root_vel_rms_gt_cm_per_frame": _rms(vel_root_gt, vel_mask),
        "stable_local_vel_rms_pred_cm_per_frame": _rms(vel_local, vel_mask),
        "stable_local_vel_rms_gt_cm_per_frame": _rms(vel_local_gt, vel_mask),
        "stable_frames_total": int(mask_t.sum().item()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--md", type=Path, required=True)
    parser.add_argument("--clip-idx", type=int, default=0)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bucket", default="train", choices=["train", "val"])
    parser.add_argument(
        "--alphas", type=str, default="0.0,0.25,0.5,1.0",
        help="Comma-separated dense-scale α values for the §A.5 sweep.",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- Dataset / model ----------
    ds = _build_dataset(cfg, args.bucket, augment=False)
    overfit_n = int(cfg.data.get("overfit_n_clips", 0))
    if overfit_n > 0:
        ds = Subset(ds, list(range(min(overfit_n, len(ds)))))
    loader = DataLoader(
        ds, batch_size=1, shuffle=False, collate_fn=collate_hoi, num_workers=0,
    )
    main_batch = None
    for i, b in enumerate(loader):
        if i == args.clip_idx:
            main_batch = b
            break
    if main_batch is None:
        raise RuntimeError("could not find main clip")

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

    cond_main, T = _build_cond(
        main_batch, model, object_encoder, clip_model, z_dims, cfg, device,
    )
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

    # ---------- A.3 batch zeroing check ----------
    a3_results = _batch_zeroing_check(cond_main, cfg)
    print("[A.3] Batch zeroing check:")
    for k, v in a3_results.items():
        print(f"  {k}: {v}")

    # ---------- A.4 input projection weight-norm audit ----------
    a4_results = _input_projection_audit(model, cfg)
    print("\n[A.4] Input projection weight-norm audit:")
    for k, v in a4_results.items():
        print(f"  {k}: {v}")

    # ---------- A.5 dense-scale sweep ----------
    rest_offsets = main_batch["rest_offsets"].to(device).float()
    seq_len = main_batch["seq_len"].to(device)
    seq_idx = torch.arange(T, device=device).unsqueeze(0)
    seq_mask = (seq_idx < seq_len.unsqueeze(1))
    joints_gt = main_batch["joints"].to(device).float()
    contact_state = main_batch["contact_state"].to(device).float()
    support = main_batch["support"].to(device).long()
    part_to_joint = torch.tensor([20, 21, 10, 11, 0], dtype=torch.long, device=device)

    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]
    sweep_results: list[dict] = []
    motion_outputs: dict[str, torch.Tensor] = {}

    print(f"\n[A.5] Dense-scale sweep over α ∈ {alphas}:")
    for alpha in alphas:
        torch.manual_seed(args.seed)
        cond_alpha = _dense_scale_cond(cond_main, alpha, pose_dims=9)
        with torch.no_grad():
            x0 = model.sample(
                cond=cond_alpha, seq_length=T, cfg_scale=args.cfg_scale,
                replacement="none", output_skip=False, sampler="ddpm",
            )
        jpos = _fk_decode_135(x0, rest_offsets, T)
        m = _compute_metrics(
            jpos_pred=jpos, jpos_gt=joints_gt, seq_mask=seq_mask,
            anchor_time=plan_gt["anchor_time"], anchor_mask=plan_gt["anchor_mask"],
            anchor_part=plan_gt["anchor_part"],
            anchor_target_world=plan_gt["anchor_target_world"],
            part_to_joint=part_to_joint, window=3,
        )
        jit = _stable_support_jitter(
            jpos, joints_gt, seq_mask, support, contact_state,
        )
        m.update(jit)
        m["alpha"] = alpha
        sweep_results.append(m)
        motion_outputs[f"alpha_{alpha}"] = x0.cpu()
        print(
            f"  α={alpha:>4}  far_unobs={m['far_unobserved_error_cm']:7.3f} cm  "
            f"root_aligned={m['root_aligned_joint_error_cm']:6.3f}  "
            f"global={m['global_joint_error_cm']:6.3f}  "
            f"local_vel_rms={jit['stable_local_vel_rms_pred_cm_per_frame']:.3f}  "
            f"root_vel_rms={jit['stable_root_vel_rms_pred_cm_per_frame']:.3f}"
        )

    # Cross-α motion-135 delta vs α=0
    base = motion_outputs[f"alpha_{alphas[0]}"]
    for r in sweep_results:
        x0 = motion_outputs[f"alpha_{r['alpha']}"]
        delta = (x0 - base).pow(2).sum(-1).sqrt().mean().item()
        r["motion_135_delta_vs_alpha0"] = float(delta)

    # ---------- write outputs ----------
    summary = {
        "config": str(args.config),
        "ckpt": str(args.ckpt),
        "seed": args.seed,
        "cfg_scale": args.cfg_scale,
        "T": T,
        "a3_batch_zeroing_check": a3_results,
        "a4_input_projection_audit": a4_results,
        "a5_dense_scale_sweep": sweep_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote JSON to {args.output}")

    # Markdown
    md: list[str] = []
    md.append("# v11 PLAN_ONLY N=10 — dense-channel audit\n")
    md.append("**Date:** 2026-05-10  ")
    md.append(f"**Config:** `{args.config}`  ")
    md.append(f"**Checkpoint:** `{args.ckpt}`  ")
    md.append(f"**seed:** {args.seed}    **cfg_scale:** {args.cfg_scale}    **T:** {T}\n")
    md.append("Per spec [analyses/claude_code_v11_n10_followup_dense_audit_full_baseline.md](analyses/claude_code_v11_n10_followup_dense_audit_full_baseline.md) §A.\n")

    md.append("## A.3 Batch zeroing check\n")
    md.append("| field | value |")
    md.append("|---|---|")
    for k, v in a3_results.items():
        md.append(f"| `{k}` | {v} |")
    md.append("")
    z_int_after = a3_results["z_int_norm_after_zeroing"]
    dt_after = a3_results["object_traj_dense_target_norm_after_zeroing"]
    pose_after = a3_results["object_traj_pose_norm_after_zeroing"]
    md.append(
        f"**Reading:** z_int after zeroing = {z_int_after:.6f} "
        f"(expect 0); dense target after zeroing = {dt_after:.6f} "
        f"(expect 0); object pose after zeroing = {pose_after:.4f} "
        f"(expect > 0). "
        + ("✓ training was actually PLAN_ONLY"
           if z_int_after < 1e-6 and dt_after < 1e-6 and pose_after > 0
           else "**✗ zeroing failed — training condition was not actually PLAN_ONLY**")
        + ".\n"
    )

    md.append("\n## A.4 Input projection weight-norm audit\n")
    md.append(
        "Per-block Frobenius norm and per-dim mean norm of the denoiser's "
        "first Linear (`AnchorDenoiser.in_proj`). Per-dim norm divides "
        "by sqrt(d_model=512) implicitly via the column reduction. If the "
        "dense channels (z_int, dense target) were truly always zero at "
        "training, their per-dim norm should reflect random-init magnitude "
        "(~ √(2/in_features)), not large task-adapted weights.\n"
    )
    md.append("| block | block norm (Frobenius) | per-dim norm (mean) |")
    md.append("|---|---|---|")
    for label, key_block, key_per in [
        ("x_t (motion)", "x_t_block_norm", "x_t_per_dim_norm"),
        ("z_int", "z_int_block_norm", "z_int_per_dim_norm"),
        ("object_pose (COM+rot6d)", "object_pose_block_norm", "object_pose_per_dim_norm"),
        ("object_dense_target", "object_dense_target_block_norm", "object_dense_target_per_dim_norm"),
        ("plan_hint", "plan_hint_block_norm", "plan_hint_per_dim_norm"),
    ]:
        md.append(f"| {label} | {a4_results[key_block]:.4f} | {a4_results[key_per]:.4f} |")
    md.append("")
    z_int_per = a4_results["z_int_per_dim_norm"]
    pose_per = a4_results["object_pose_per_dim_norm"]
    target_per = a4_results["object_dense_target_per_dim_norm"]
    x_t_per = a4_results["x_t_per_dim_norm"]
    md.append(
        f"**Reading:** x_t per-dim {x_t_per:.3f}, object_pose per-dim "
        f"{pose_per:.3f}, plan_hint per-dim {a4_results['plan_hint_per_dim_norm']:.3f} "
        f"(actively used channels). "
        f"z_int per-dim {z_int_per:.3f}, dense target per-dim {target_per:.3f} "
        f"(zeroed channels). "
        + ("If zeroed channels are within ~30% of x_t, that's random-init magnitude — "
           "consistent with the model not adapting to them. If they're substantially larger, "
           "the model adapted to a non-zero value during training (= zeroing bug).\n")
    )

    md.append("\n## A.5 Dense-scale sweep (α applied to z_int + dense-target portion of object_traj)\n")
    md.append("All other inputs identical (same plan, same seed, same x_T noise).\n")
    md.append("| α | far-unobs cm | root-aligned cm | global cm | model−GT anchor cm | stable_root_vel_rms cm/fr | stable_local_vel_rms cm/fr | Δmotion-135 vs α=0 |")
    md.append("|---|---|---|---|---|---|---|---|")
    for r in sweep_results:
        md.append(
            f"| {r['alpha']} | "
            f"{r['far_unobserved_error_cm']:.3f} | "
            f"{r['root_aligned_joint_error_cm']:.3f} | "
            f"{r['global_joint_error_cm']:.3f} | "
            f"{r['anchor_realization_minus_gt_cm']:.3f} | "
            f"{r['stable_root_vel_rms_pred_cm_per_frame']:.3f} | "
            f"{r['stable_local_vel_rms_pred_cm_per_frame']:.3f} | "
            f"{r['motion_135_delta_vs_alpha0']:.3f} |"
        )
    md.append("")
    far_alpha_0 = sweep_results[0]["far_unobserved_error_cm"]
    far_alpha_1 = next(
        (r["far_unobserved_error_cm"] for r in sweep_results if r["alpha"] == 1.0),
        sweep_results[-1]["far_unobserved_error_cm"],
    )
    monotonic = all(
        sweep_results[i + 1]["far_unobserved_error_cm"] <= sweep_results[i]["far_unobserved_error_cm"] + 0.5
        for i in range(len(sweep_results) - 1)
    )
    md.append(
        f"**Reading:** α=0 (training-distribution PLAN_ONLY) → far-unobs "
        f"{far_alpha_0:.2f} cm. α=1 (FULL-mode injection) → "
        f"{far_alpha_1:.2f} cm. Improvement: {far_alpha_0 - far_alpha_1:.2f} cm "
        f"(positive = α=1 improves over α=0). Monotonicity: "
        + ("✓ monotonic (smooth helpful bias)" if monotonic else "✗ not monotonic (likely OOD artefact)")
        + ".\n"
    )
    md.append(
        "If α=1 helps and α∈(0,1) interpolates monotonically, dense channels "
        "carry a stable useful signal even though the model was not trained "
        "with them. If only α=1 helps (non-monotonic), the FULL-mode result "
        "is an OOD artefact and should not be trusted as evidence for a "
        "FULL mainline. The decisive comparison is the FULL-trained baseline "
        "(spec §B).\n"
    )


    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote MD to {args.md}")


if __name__ == "__main__":
    main()
