"""v10 phase-2 diagnostic — four-way causality + strengthened plan semantics.

Per analyses/claude_code_v10_plan_tokens_next_steps.md §6 (four-way
causality ablation) and §7 (strengthened semantic plan variants).
Builds on top of plan_condition_diagnostics.py — reuses the metric
function, model loader, and plan extractor — but adds:

1. **Four-way dense-vs-plan causality ablation** (§6):
     FULL        : dense z_int + dense object target + GT plan
     DENSE_ONLY  : dense z_int + dense object target + zero plan
     PLAN_ONLY   : zero z_int + object pose only (zero target_world) + GT plan
     NONE        : zero z_int + object pose only + zero plan
   Tests whether the model uses plan tokens vs dense conditioning shortcuts.

2. **Strengthened plan-variant battery** (§7) under FULL conditioning:
     gt
     zero
     shuffled_time / reversed_time
     wrong_clip (only when ≥ 2 clips available; else marked INVALID)
     target_perturbed at sigma 10 / 30 / 50 cm (perturb in object-local;
       recompute target_world via the per-clip object pose at the anchor
       frame so the two channels stay consistent — §7.2)
     part_only_swapped / target_only_swapped (§7.3)
     part_target_mismatched (assign hand target to foot anchor)
     phase_shuffled / support_shuffled / phase_support_zeroed
     support_object_to_none (§7.4)

Output: JSON + Markdown table + selected MP4 renderings of (mode, variant) pairs.
"""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

# Make the sibling ``plan_condition_diagnostics`` importable when this
# script is invoked directly as a file. scripts/ is not a python package
# in this repo (no __init__.py).
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reuse helpers from the original diagnostic
from plan_condition_diagnostics import (  # type: ignore[import-not-found]
    _build_cond,
    _build_dataset,
    _build_model,
    _compute_metrics,
    _gt_plan,
    _zero_plan,
    _shuffled_plan,
    _wrong_clip_plan,
    _reversed_plan,
)
from piano.data.dataset import collate_hoi
from piano.data.interaction_plan_compiler import (
    lift_target_local_to_world_np,
)
from piano.training.smpl_kinematics import (
    fk_from_global_rotations as _fk_from_global,
    rotation_6d_to_matrix as _rot6d_to_mat,
)
from piano.utils.clip_utils import load_clip_text_encoder


# ---------------------------------------------------------------------------
# Strengthened plan variants (§7)
# ---------------------------------------------------------------------------


def _target_perturbed_consistent(
    plan: dict[str, torch.Tensor],
    sigma_m: float,
    object_pos_world: torch.Tensor,    # (B, T, 3)
    object_rot_world: torch.Tensor,    # (B, T, 3) axis-angle
    seed: int,
) -> dict[str, torch.Tensor]:
    """§7.2: perturb target_local with σ noise, then recompute target_world
    via the object pose at the anchor frame. Keeps the two channels
    consistent — perturbing them independently makes the encoder receive
    contradictory tokens."""
    out = {k: v.clone() for k, v in plan.items()}
    device = out["anchor_target_local"].device
    rng = torch.Generator(device="cpu").manual_seed(int(seed))
    n_local = (
        torch.randn(out["anchor_target_local"].shape, generator=rng) * sigma_m
    ).to(device=device, dtype=out["anchor_target_local"].dtype)
    out["anchor_target_local"] = out["anchor_target_local"] + n_local

    # For each anchor, lift perturbed target_local through the object pose
    # at that anchor's frame.
    B, K, P, _ = out["anchor_target_local"].shape
    T = object_pos_world.shape[1]
    a_t = out["anchor_time"].clamp(0, T - 1)
    obj_pos_at = torch.gather(
        object_pos_world, 1, a_t.unsqueeze(-1).expand(-1, -1, 3),
    )                                                                       # (B, K, 3)
    obj_rot_at = torch.gather(
        object_rot_world, 1, a_t.unsqueeze(-1).expand(-1, -1, 3),
    )                                                                       # (B, K, 3)
    # Numpy lift expects (T, P, 3); use (K, P, 3) per batch.
    new_world = torch.zeros_like(out["anchor_target_world"])
    for b in range(B):
        tl_np = out["anchor_target_local"][b].detach().cpu().numpy()        # (K, P, 3)
        op_np = obj_pos_at[b].detach().cpu().numpy()                        # (K, 3)
        or_np = obj_rot_at[b].detach().cpu().numpy()                        # (K, 3)
        tw_np = lift_target_local_to_world_np(tl_np, op_np, or_np)          # (K, P, 3)
        new_world[b] = torch.from_numpy(tw_np).to(device=device, dtype=new_world.dtype)
    out["anchor_target_world"] = new_world
    return out


def _part_only_swapped(plan: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Roll anchor_part one slot; keep targets fixed."""
    out = {k: v.clone() for k, v in plan.items()}
    out["anchor_part"] = torch.roll(out["anchor_part"], shifts=1, dims=-1)
    return out


def _target_only_swapped(plan: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Roll target_local + target_world along the part axis; keep parts fixed."""
    out = {k: v.clone() for k, v in plan.items()}
    out["anchor_target_local"] = torch.roll(
        out["anchor_target_local"], shifts=1, dims=-2,
    )
    out["anchor_target_world"] = torch.roll(
        out["anchor_target_world"], shifts=1, dims=-2,
    )
    return out


def _part_target_mismatched(plan: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Force a hand target onto a foot/pelvis anchor by overriding
    anchor_part to a single non-active part. We pick part=2 (L_foot) for
    every active anchor — if the original anchor was a hand, the model is
    now told the foot should reach what was the hand target."""
    out = {k: v.clone() for k, v in plan.items()}
    new_part = torch.zeros_like(out["anchor_part"])
    new_part[..., 2] = 1.0  # L_foot
    # Only do this for valid anchors; padded slots stay all-zero.
    mask = out["anchor_mask"].unsqueeze(-1).float()
    out["anchor_part"] = new_part * mask
    return out


def _phase_shuffled(plan: dict[str, torch.Tensor], seed: int) -> dict[str, torch.Tensor]:
    out = {k: v.clone() for k, v in plan.items()}
    device = out["anchor_phase"].device
    rng = torch.Generator(device="cpu").manual_seed(int(seed))
    B, K = out["anchor_phase"].shape
    for b in range(B):
        valid = int(out["anchor_mask"][b].sum().item())
        if valid >= 2:
            perm = torch.randperm(valid, generator=rng).to(device)
            out["anchor_phase"][b, :valid] = out["anchor_phase"][b, perm]
    return out


def _support_shuffled(plan: dict[str, torch.Tensor], seed: int) -> dict[str, torch.Tensor]:
    out = {k: v.clone() for k, v in plan.items()}
    device = out["anchor_support"].device
    rng = torch.Generator(device="cpu").manual_seed(int(seed) + 1)
    B, K = out["anchor_support"].shape
    for b in range(B):
        valid = int(out["anchor_mask"][b].sum().item())
        if valid >= 2:
            perm = torch.randperm(valid, generator=rng).to(device)
            out["anchor_support"][b, :valid] = out["anchor_support"][b, perm]
    return out


def _phase_support_zeroed(plan: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {k: v.clone() for k, v in plan.items()}
    out["anchor_phase"] = torch.zeros_like(out["anchor_phase"])
    out["anchor_support"] = torch.zeros_like(out["anchor_support"])
    return out


def _support_object_to_none(plan: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Force every anchor's support to 0 ('none') regardless of source label."""
    out = {k: v.clone() for k, v in plan.items()}
    out["anchor_support"] = torch.zeros_like(out["anchor_support"])
    return out


# ---------------------------------------------------------------------------
# Causality ablation: dense conditioning toggles (§6)
# ---------------------------------------------------------------------------


def _zero_z_int(cond: dict) -> dict:
    """Replace dense z_int with zeros. The model's null_zint param is the
    LEARNED null embedding used for CFG; for ablation we want a clean
    'no semantic evidence' signal — zeros are the right primitive."""
    out = dict(cond)
    out["z_int"] = torch.zeros_like(cond["z_int"])
    return out


def _drop_object_target_world(cond: dict, motion_dim_obj_pose: int = 9) -> dict:
    """object_traj is (B, T, 24) = (COM 3 + rot6d 6 + 5 anchors world * 3).
    For PLAN_ONLY / NONE modes, zero the last 15 dims so the dense lifted
    contact_target shortcut is removed but the object's geometric
    trajectory (COM + rotation) remains."""
    out = dict(cond)
    obj_traj = cond["object_world_traj"].clone()
    if obj_traj.shape[-1] > motion_dim_obj_pose:
        obj_traj[..., motion_dim_obj_pose:] = 0.0
    out["object_world_traj"] = obj_traj
    return out


# ---------------------------------------------------------------------------
# FK helper
# ---------------------------------------------------------------------------


def _fk_decode_135(x0: torch.Tensor, rest_offsets: torch.Tensor, T: int) -> torch.Tensor:
    """135-D motion → (1, T, 22, 3) world joints. Same as visualize/diagnostic."""
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
    parser.add_argument(
        "--render-pairs", type=str,
        default="FULL/gt,FULL/zero,FULL/part_target_mismatched,DENSE_ONLY/zero,PLAN_ONLY/gt,NONE/zero",
        help="Comma-separated mode/variant pairs to render.",
    )
    parser.add_argument("--clip-idx", type=int, default=0)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bucket", default="train", choices=["train", "val"])
    parser.add_argument("--fps", type=float, default=20.0)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------- Dataset / model -----------
    ds = _build_dataset(cfg, args.bucket, augment=False)
    overfit_n = int(cfg.data.get("overfit_n_clips", 0))
    if overfit_n > 0:
        ds = Subset(ds, list(range(min(overfit_n, len(ds)))))
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_hoi, num_workers=0)

    main_batch = None
    secondary_batch = None
    for i, batch in enumerate(loader):
        if i == args.clip_idx:
            main_batch = batch
        elif main_batch is not None and secondary_batch is None:
            secondary_batch = batch
            break
    if main_batch is None:
        raise RuntimeError("Could not find main clip for diagnostic")
    has_real_second_clip = secondary_batch is not None
    if not has_real_second_clip:
        secondary_batch = main_batch  # placeholder; wrong_clip marked INVALID below

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

    cond_main, T = _build_cond(main_batch, model, object_encoder, clip_model, z_dims, cfg, device)
    cond_sec, _ = _build_cond(secondary_batch, model, object_encoder, clip_model, z_dims, cfg, device)

    # ----------- Plan dicts -----------
    def _extract_plan(batch: dict) -> dict[str, torch.Tensor]:
        keys = [
            "anchor_time", "anchor_part", "anchor_target_local",
            "anchor_target_world", "anchor_type", "anchor_phase",
            "anchor_support", "anchor_conf", "anchor_mask",
            "segment_start", "segment_end", "segment_part",
            "segment_target_summary_local", "segment_phase",
            "segment_support", "segment_conf", "segment_mask",
        ]
        return {k: batch[f"plan_{k}"].to(device) for k in keys}

    plan_gt = _extract_plan(main_batch)
    plan_other = _extract_plan(secondary_batch)

    obj_pos_world = main_batch["object_positions"].to(device)
    obj_rot_world = main_batch["object_rotations"].to(device)

    # ----------- Strengthened semantic variants (§7) -----------
    plan_variants: dict[str, dict[str, torch.Tensor]] = {
        "gt": _gt_plan(plan_gt),
        "zero": _zero_plan(plan_gt),
        "shuffled_time": _shuffled_plan(plan_gt, seed=args.seed),
        "reversed_time": _reversed_plan(plan_gt, T=T),
        "target_perturbed_10cm": _target_perturbed_consistent(
            plan_gt, sigma_m=0.10,
            object_pos_world=obj_pos_world, object_rot_world=obj_rot_world,
            seed=args.seed,
        ),
        "target_perturbed_30cm": _target_perturbed_consistent(
            plan_gt, sigma_m=0.30,
            object_pos_world=obj_pos_world, object_rot_world=obj_rot_world,
            seed=args.seed + 1,
        ),
        "target_perturbed_50cm": _target_perturbed_consistent(
            plan_gt, sigma_m=0.50,
            object_pos_world=obj_pos_world, object_rot_world=obj_rot_world,
            seed=args.seed + 2,
        ),
        "part_only_swapped": _part_only_swapped(plan_gt),
        "target_only_swapped": _target_only_swapped(plan_gt),
        "part_target_mismatched": _part_target_mismatched(plan_gt),
        "phase_shuffled": _phase_shuffled(plan_gt, seed=args.seed),
        "support_shuffled": _support_shuffled(plan_gt, seed=args.seed),
        "phase_support_zeroed": _phase_support_zeroed(plan_gt),
        "support_object_to_none": _support_object_to_none(plan_gt),
    }
    if has_real_second_clip:
        plan_variants["wrong_clip"] = _wrong_clip_plan(plan_gt, plan_other)

    # ----------- Causality modes -----------
    def _cond_for_mode(mode: str, base: dict) -> dict:
        if mode == "FULL":
            return base
        if mode == "DENSE_ONLY":
            return base  # dense kept; plan will be set to zero variant
        if mode == "PLAN_ONLY":
            c = _zero_z_int(base)
            c = _drop_object_target_world(c, motion_dim_obj_pose=9)
            return c
        if mode == "NONE":
            c = _zero_z_int(base)
            c = _drop_object_target_world(c, motion_dim_obj_pose=9)
            return c
        raise ValueError(f"Unknown mode: {mode}")

    # The 4-way table uses GT plan for FULL/PLAN_ONLY and zero plan for
    # DENSE_ONLY/NONE — that's the spec §6 design.
    causality_specs = [
        ("FULL", "gt"),
        ("DENSE_ONLY", "zero"),
        ("PLAN_ONLY", "gt"),
        ("NONE", "zero"),
    ]

    # SMPL part-to-joint
    part_to_joint = torch.tensor([20, 21, 10, 11, 0], dtype=torch.long, device=device)

    # GT joints / mask
    rest_offsets = main_batch["rest_offsets"].to(device).float()
    seq_len = main_batch["seq_len"].to(device)
    seq_idx = torch.arange(T, device=device).unsqueeze(0)
    seq_mask = (seq_idx < seq_len.unsqueeze(1))
    joints_gt = main_batch["joints"].to(device).float()

    # ----------- Sample loop -----------
    all_results: dict[str, dict] = {}
    motion_outputs: dict[str, torch.Tensor] = {}

    # 1. Plan-variant sweep under FULL conditioning
    for vname, plan in plan_variants.items():
        torch.manual_seed(args.seed)
        cond = {**cond_main, "interaction_plan": plan}
        with torch.no_grad():
            x0 = model.sample(cond=cond, seq_length=T, cfg_scale=args.cfg_scale)
        jpos_pred = _fk_decode_135(x0, rest_offsets, T)
        m = _compute_metrics(
            jpos_pred=jpos_pred, jpos_gt=joints_gt, seq_mask=seq_mask,
            anchor_time=plan["anchor_time"], anchor_mask=plan["anchor_mask"],
            anchor_part=plan["anchor_part"],
            anchor_target_world=plan["anchor_target_world"],
            part_to_joint=part_to_joint, window=3,
        )
        key = f"FULL/{vname}"
        all_results[key] = m
        motion_outputs[key] = x0.cpu()

    # 2. Causality 4-way
    for mode, plan_name in causality_specs:
        if mode == "FULL" and plan_name == "gt":
            continue  # already done
        torch.manual_seed(args.seed)
        plan = plan_variants[plan_name]
        cond = _cond_for_mode(mode, {**cond_main, "interaction_plan": plan})
        with torch.no_grad():
            x0 = model.sample(cond=cond, seq_length=T, cfg_scale=args.cfg_scale)
        jpos_pred = _fk_decode_135(x0, rest_offsets, T)
        m = _compute_metrics(
            jpos_pred=jpos_pred, jpos_gt=joints_gt, seq_mask=seq_mask,
            anchor_time=plan["anchor_time"], anchor_mask=plan["anchor_mask"],
            anchor_part=plan["anchor_part"],
            anchor_target_world=plan["anchor_target_world"],
            part_to_joint=part_to_joint, window=3,
        )
        key = f"{mode}/{plan_name}"
        all_results[key] = m
        motion_outputs[key] = x0.cpu()

    # 3. Cross-key motion-135 delta (vs FULL/gt baseline)
    base = motion_outputs["FULL/gt"]
    for key, x0 in motion_outputs.items():
        delta = (x0 - base).pow(2).sum(-1).sqrt().mean().item()
        all_results[key]["motion_135_delta_vs_full_gt"] = float(delta)

    # ----------- Pass / fail -----------
    far = lambda k: all_results[k]["far_unobserved_error_cm"]
    pass_unobs = (far("FULL/zero") - far("FULL/gt")) >= 5.0
    pass_anchor = all_results["FULL/gt"]["plan_anchor_contact_realization_cm"] < 20.0
    pass_trans = all_results["FULL/gt"]["transition_local_vel_jump_cm_per_frame"] < 3.0
    # New §6 causality readouts
    plan_only_lift = far("NONE/zero") - far("PLAN_ONLY/gt")          # higher = plan helps even without dense
    dense_only_lift = far("NONE/zero") - far("DENSE_ONLY/zero")      # higher = dense helps even without plan
    full_vs_plan_only = far("PLAN_ONLY/gt") - far("FULL/gt")          # higher = dense complementary

    # ----------- Optional rendering -----------
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
        gt_title = f"{subset}/{seq_id}\n[GT]\ntext: {text[:80]}"
        print(f"  rendering GT → {gt_out.name}")
        render_motion_video(
            joints=joints_gt_np, output_path=gt_out, fps=args.fps, title=gt_title,
            object_positions=obj_pos_np, object_rotations=obj_rot_np, object_pc=obj_pc_np,
        )
        pairs = [p.strip() for p in args.render_pairs.split(",") if p.strip()]
        for pair in pairs:
            if pair not in motion_outputs:
                print(f"  skip render '{pair}' (not in motion_outputs)")
                continue
            x0 = motion_outputs[pair].to(device)
            jpos_pred = _fk_decode_135(x0, rest_offsets, T)
            jpos_pred_np = jpos_pred.squeeze(0).cpu().numpy()[:valid_T]
            far_err = all_results[pair]["far_unobserved_error_cm"]
            delta = all_results[pair]["motion_135_delta_vs_full_gt"]
            tag = pair.replace("/", "__")
            pred_out = args.render_dir / f"{subset}_{seq_id}_predicted_{tag}.mp4"
            title = (
                f"{subset}/{seq_id}\n[{pair}]\n"
                f"far-unobs={far_err:.2f} cm  Δmotion-135={delta:.3f}"
            )
            print(f"  rendering {pair:35s} → {pred_out.name}")
            render_motion_video(
                joints=jpos_pred_np, output_path=pred_out, fps=args.fps, title=title,
                object_positions=obj_pos_np, object_rotations=obj_rot_np,
                object_pc=obj_pc_np,
            )

    # ----------- JSON -----------
    summary = {
        "config": str(args.config),
        "ckpt": str(args.ckpt),
        "cfg_scale": args.cfg_scale,
        "seed": args.seed,
        "T": T,
        "wrong_clip_valid": has_real_second_clip,
        "pass_gates": {
            "gt_better_than_zero_unobs_5cm": bool(pass_unobs),
            "anchor_contact_realization_under_20cm": bool(pass_anchor),
            "transition_vel_jump_under_3cm_per_frame": bool(pass_trans),
        },
        "causality": {
            "plan_only_lift_vs_none_cm": float(plan_only_lift),
            "dense_only_lift_vs_none_cm": float(dense_only_lift),
            "full_vs_plan_only_cm": float(full_vs_plan_only),
        },
        "results": all_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote JSON to {args.output}")

    # ----------- Markdown -----------
    md: list[str] = []
    md.append("# v10 phase-2 diagnostic — FK-fix + causality + strengthened semantics\n")
    md.append("**Date:** 2026-05-10  ")
    md.append(f"**Config:** `{args.config}`  ")
    md.append(f"**Checkpoint:** `{args.ckpt}`  ")
    md.append(f"**cfg_scale:** {args.cfg_scale}    **seed:** {args.seed}    **T:** {T}\n")
    md.append("## Pass gates\n")
    md.append(f"- GT plan ≥ 5 cm better than zero on far-unobs: {'✓' if pass_unobs else '✗'}  "
              f"(gt={far('FULL/gt'):.2f}, zero={far('FULL/zero'):.2f})")
    md.append(f"- Anchor realisation < 20 cm: {'✓' if pass_anchor else '✗'}  "
              f"({all_results['FULL/gt']['plan_anchor_contact_realization_cm']:.2f}; see GT upper-bound report)")
    md.append(f"- Transition vel jump < 3 cm/frame: {'✓' if pass_trans else '✗'}  "
              f"({all_results['FULL/gt']['transition_local_vel_jump_cm_per_frame']:.2f})\n")
    md.append("## §6 four-way causality ablation\n")
    md.append("| mode / plan | far-unobs cm | root-aligned cm | anchor realisation cm | Δ motion-135 vs FULL/gt |")
    md.append("|---|---|---|---|---|")
    for k in ["FULL/gt", "DENSE_ONLY/zero", "PLAN_ONLY/gt", "NONE/zero"]:
        r = all_results[k]
        md.append(
            f"| {k} | {r['far_unobserved_error_cm']:.3f} | "
            f"{r['root_aligned_joint_error_cm']:.3f} | "
            f"{r['plan_anchor_contact_realization_cm']:.3f} | "
            f"{r['motion_135_delta_vs_full_gt']:.3f} |"
        )
    md.append("")
    md.append(f"- plan-only lift vs NONE: **{plan_only_lift:+.2f} cm** (>0 = plan helps even without dense)")
    md.append(f"- dense-only lift vs NONE: **{dense_only_lift:+.2f} cm** (>0 = dense helps even without plan)")
    md.append(f"- FULL vs PLAN_ONLY: **{full_vs_plan_only:+.2f} cm** (>0 = dense is complementary)\n")
    md.append("## §7 strengthened plan-variant battery (FULL conditioning)\n")
    cols = [
        "global_joint_error_cm",
        "root_aligned_joint_error_cm",
        "near_anchor_window_error_cm",
        "far_unobserved_error_cm",
        "transition_local_vel_jump_cm_per_frame",
        "plan_anchor_contact_realization_cm",
        "gt_anchor_realization_cm",
        "anchor_realization_minus_gt_cm",
        "motion_135_delta_vs_full_gt",
    ]
    md.append("| variant | " + " | ".join(c.replace("_", " ") for c in cols) + " |")
    md.append("|" + "|".join(["---"] * (len(cols) + 1)) + "|")
    for vname in plan_variants.keys():
        key = f"FULL/{vname}"
        row = [vname] + [f"{all_results[key][c]:.3f}" for c in cols]
        md.append("| " + " | ".join(row) + " |")
    if not has_real_second_clip:
        md.append("\n*Note (§7.1): `wrong_clip` variant is INVALID in this run "
                  "(only one clip in overfit). Skipped instead of falling back to GT.*")
    md.append("")

    # Per-part anchor realisation breakdown (per spec §E)
    pp_model = all_results["FULL/gt"].get("anchor_realization_per_part_model", {})
    pp_gt = all_results["FULL/gt"].get("anchor_realization_per_part_gt", {})
    pp_diff = all_results["FULL/gt"].get("anchor_realization_per_part_diff", {})
    if pp_model:
        md.append("## Per-part anchor realisation (FULL/gt) — GT-normalised\n")
        md.append("Spec §E: pseudo-label `contact_target_xyz` encodes object surface points, "
                  "so the absolute `< 20 cm` gate is unreachable (GT motion itself scores ~34 cm). "
                  "Report model − GT per part instead so the structural floor doesn't dominate.\n")
        md.append("| body part | model cm | GT cm | model − GT cm |")
        md.append("|---|---|---|---|")
        for p in ["L_hand", "R_hand", "L_foot", "R_foot", "pelvis"]:
            if p in pp_model:
                md.append(
                    f"| {p} | {pp_model[p]:.2f} | {pp_gt[p]:.2f} | {pp_diff[p]:+.2f} |"
                )
        md.append("")
        md.append(
            "If `model − GT` is consistently negative, the model is at or below "
            "the metric floor (no further training pressure justified). If it's "
            "positive, the model is genuinely worse than GT on that part."
        )
        md.append("")

    md.append("## Interpretation rubric (§6.2)\n")
    md.append("- FULL ≈ DENSE_ONLY and PLAN_ONLY poor → dense conditioning dominates.")
    md.append("- PLAN_ONLY clearly better than NONE → plan tokens carry meaningful information.")
    md.append("- FULL > PLAN_ONLY and FULL > DENSE_ONLY → dense + plan are complementary.")
    md.append("- DENSE_ONLY ≈ PLAN_ONLY → both pathways carry similar information.\n")
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote MD to {args.md}")

    # ----------- Console summary -----------
    print("\nFour-way causality far-unobs (cm):")
    for k in ["FULL/gt", "DENSE_ONLY/zero", "PLAN_ONLY/gt", "NONE/zero"]:
        print(f"  {k:25s}  {all_results[k]['far_unobserved_error_cm']:.3f}  "
              f"Δmotion-135={all_results[k]['motion_135_delta_vs_full_gt']:.3f}")
    print(f"\nplan_only_lift_vs_none = {plan_only_lift:+.2f} cm")
    print(f"dense_only_lift_vs_none = {dense_only_lift:+.2f} cm")
    print(f"full_vs_plan_only      = {full_vs_plan_only:+.2f} cm")


if __name__ == "__main__":
    main()
