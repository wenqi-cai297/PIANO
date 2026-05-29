"""G1 soft-stance diagnostic.

Per analyses/2026-05-28_round29_next_step_ablation_execution_prompt_for_claude_code.md §4.

Checks whether a G1-trained model is actually producing meaningful
alternating stance, or merely satisfying its aggregate-statistic losses
(transition_rate / duty_cycle / both_state_match) through a
constant-mid soft probability (pL ≈ pR ≈ 0.5 always), low alt-amplitude
(`std(pL - pR)` low), or other degenerate solutions that the standard
gait diag would not flag.

For each walking segment (detected GT-side, same convention as
`round26_gait_diag.py`):

Hard metrics (from round26_gait_diag — for comparison):
  - both_stance, both_swing, L_only_stance, R_only_stance
  - transitions/sec
  - L_R_height_corr
  - step_period_frames

Soft metrics (using the G1 `_pred_soft_stance_prob` definition):
  - mean / std of pL, pR over walking frames
  - mean / std of soft_alt = pL - pR
  - soft transition density: mean(|soft_alt[t] - soft_alt[t-1]|)
  - soft both_stance: mean(pL · pR)
  - soft both_swing: mean((1-pL) · (1-pR))

Degeneracy flags:
  - constant_mid_rate: fraction of walking frames with both 0.4<pL<0.6
    AND 0.4<pR<0.6
  - low_alt_amplitude: segment-level flag when std(pL - pR) < 0.15
  - low_transition: soft transition density much lower than GT target
    density (computed from GT's hard alternating signal on the same
    walking frames)
  - high_both_swing: same hard-side threshold as the gait report;
    reported alongside the soft both_swing for cross-check

Outputs:
  - <output_dir>/g1_soft_stance_stats.json
  - <output_dir>/g1_soft_stance_summary.md

Usage:
  conda run -n piano python scripts/stage_b_generator/round29_g1_soft_stance_diag.py \\
    --config configs/training/anchordiff_r29_ns_a0_c41_g1_loss_s4.yaml \\
    --ckpt   runs/training/stageB_anchordiff_r29_ns_a0_c41_g1_loss_s4/final.pt \\
    --selection-json analyses/round29_val_diag_indices_48_balanced.json \\
    --output-dir analyses/round29_r29_ns_a0_c41_g1_loss_s4_diag_g1_soft_stance_val \\
    --bucket val
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

# Joint indices used by the pure metric helpers — defined locally so the
# helpers stay importable without pulling in torch / omegaconf / clip etc.
# Must match piano.training.temporal_interaction_losses.LEFT_ANKLE_IDX /
# RIGHT_ANKLE_IDX (SMPL-22).
LEFT_ANKLE_IDX: int = 7
RIGHT_ANKLE_IDX: int = 8


# Re-use round26_gait_diag's defaults.
ROOT_WALK_SPEED_M_PER_FRAME_DEFAULT = 0.005
MIN_WALK_LEN_FRAMES_DEFAULT = 20
STANCE_HEIGHT_M_DEFAULT = 0.12
STANCE_SPEED_M_PER_FRAME_DEFAULT = 0.015

# Degeneracy thresholds (per prompt §4).
CONSTANT_MID_LO = 0.4
CONSTANT_MID_HI = 0.6
LOW_ALT_AMPLITUDE_STD = 0.15
LOW_TRANSITION_RATIO = 0.40  # soft / GT < this → low_transition

# Soft-stance helper config — must match the G1 training defaults.
SOFT_STANCE_DEFAULT_THRESHOLD_MPS = 0.30
SOFT_STANCE_DEFAULT_SOFTNESS_MPS = 0.10
SOFT_STANCE_FLOOR_QUANTILE = 0.05
SOFT_STANCE_GROUNDED_THRESHOLD_ABOVE_FLOOR_M = 0.10
SOFT_STANCE_GROUNDED_SOFTNESS_M = 0.03


# Hard gait metrics — duplicated from round26_gait_diag.py to keep this
# module importable in unit tests without pulling in torch + OmegaConf.
# Keep these in sync with scripts/stage_b_generator/round26_gait_diag.py.

J_PELVIS = 0
J_L_FOOT = 7   # left ankle
J_R_FOOT = 8


def detect_walking_segments_gt(
    gt_joints: np.ndarray,
    seq_mask: np.ndarray,
    fps: float,
    min_speed: float,
    min_length: int,
) -> list[tuple[int, int]]:
    """Return [(t0, t1_inclusive)] walking segments from GT root motion."""
    T = len(gt_joints)
    root_xz = gt_joints[:, J_PELVIS, [0, 2]]
    horiz_vel = np.concatenate(
        [[0], np.linalg.norm(np.diff(root_xz, axis=0), axis=-1)]
    )
    is_walk = (horiz_vel > min_speed) & seq_mask.astype(bool)
    segments: list[tuple[int, int]] = []
    in_seg = False
    t0 = 0
    for t in range(T):
        if is_walk[t] and not in_seg:
            t0 = t; in_seg = True
        elif not is_walk[t] and in_seg:
            t1 = t - 1
            if t1 - t0 + 1 >= min_length:
                segments.append((t0, t1))
            in_seg = False
    if in_seg:
        t1 = T - 1
        if t1 - t0 + 1 >= min_length:
            segments.append((t0, t1))
    return segments


def _stance_mask(foot_xyz: np.ndarray,
                 stance_height: float,
                 stance_speed: float) -> np.ndarray:
    height = foot_xyz[:, 1]
    horiz_vel = np.concatenate(
        [[0], np.linalg.norm(np.diff(foot_xyz[:, [0, 2]], axis=0), axis=-1)]
    )
    return (height < stance_height) & (horiz_vel < stance_speed)


def _step_period(L_foot_y: np.ndarray) -> int | None:
    if len(L_foot_y) < 10:
        return None
    y = L_foot_y - L_foot_y.mean()
    if np.std(y) < 1e-4:
        return None
    ac = np.correlate(y, y, mode="full")[len(y) - 1:]
    ac = ac / max(ac[0], 1e-9)
    dip_found = False
    for lag in range(2, min(len(ac), 60)):
        if not dip_found and ac[lag] < 0.2:
            dip_found = True
        if dip_found and ac[lag] > 0.4 and ac[lag] >= ac[lag - 1] and (
            lag + 1 >= len(ac) or ac[lag] >= ac[lag + 1]
        ):
            return int(lag)
    return None


def gait_metrics_for_segment(
    joints: np.ndarray, fps: float,
    stance_height: float, stance_speed: float,
) -> dict[str, Any]:
    L_foot = joints[:, J_L_FOOT, :]
    R_foot = joints[:, J_R_FOOT, :]
    L_stance = _stance_mask(L_foot, stance_height, stance_speed)
    R_stance = _stance_mask(R_foot, stance_height, stance_speed)
    T = len(L_stance)
    both_stance = (L_stance & R_stance).mean()
    both_swing = (~L_stance & ~R_stance).mean()
    L_only = (L_stance & ~R_stance).mean()
    R_only = (~L_stance & R_stance).mean()
    state = np.where(L_stance & ~R_stance, 0,
                     np.where(~L_stance & R_stance, 1, -1))
    transitions = 0
    last_solid = -1
    for s in state:
        if s == -1:
            continue
        if last_solid != -1 and s != last_solid:
            transitions += 1
        last_solid = int(s)
    L_R_corr = (
        float(np.corrcoef(L_foot[:, 1], R_foot[:, 1])[0, 1]) if T > 5 else None
    )
    period = _step_period(L_foot[:, 1])
    return {
        "n_frames": int(T),
        "frac_both_stance": float(both_stance),
        "frac_both_swing": float(both_swing),
        "frac_L_only_stance": float(L_only),
        "frac_R_only_stance": float(R_only),
        "n_stance_transitions": int(transitions),
        "transitions_per_second": float(transitions * fps / max(T, 1)),
        "L_R_height_corr": L_R_corr,
        "step_period_frames": period,
    }


def _soft_stance_for_clip(
    pred_joints_torch: "torch.Tensor",    # (1, T, 22, 3)
    gt_joints_torch: "torch.Tensor",      # (1, T, 22, 3) — for floor
    cfg: "TemporalInteractionLossConfig",
    fps: float,
) -> np.ndarray:
    """Compute pL, pR (soft stance) for one clip. Returns (T, 2) numpy
    where columns are (pL, pR).
    """
    import torch  # noqa: F401  — only needed when running diag, not tests
    from piano.training.temporal_interaction_losses import _pred_soft_stance_prob
    with torch.no_grad():
        soft = _pred_soft_stance_prob(
            pred_joints_torch.float(), gt_joints_torch.float(),
            cfg=cfg, fps=fps,
        )                                                       # (1, T, 2)
    return soft[0].cpu().numpy()


def _hard_alt_from_joints(
    joints: np.ndarray,                 # (Tseg, 22, 3)
    stance_height: float,
    stance_speed: float,
) -> np.ndarray:
    """Hard alternating signal: L_stance - R_stance ∈ {-1, 0, +1}."""
    L_foot = joints[:, LEFT_ANKLE_IDX, :]
    R_foot = joints[:, RIGHT_ANKLE_IDX, :]
    L_low = (L_foot[:, 1] < stance_height)
    R_low = (R_foot[:, 1] < stance_height)
    L_slow_ = (np.concatenate([[0], np.linalg.norm(
        np.diff(L_foot[:, [0, 2]], axis=0), axis=-1)]) < stance_speed)
    R_slow_ = (np.concatenate([[0], np.linalg.norm(
        np.diff(R_foot[:, [0, 2]], axis=0), axis=-1)]) < stance_speed)
    L_stance = (L_low & L_slow_).astype(np.float32)
    R_stance = (R_low & R_slow_).astype(np.float32)
    return L_stance - R_stance                                  # (Tseg,)


def _soft_metrics_for_segment(
    soft_seg: np.ndarray,                # (Tseg, 2) pred soft (pL, pR)
    gt_hard_alt: np.ndarray,             # (Tseg,) GT hard L - R ∈ {-1, 0, 1}
) -> dict[str, Any]:
    pL = soft_seg[:, 0]
    pR = soft_seg[:, 1]
    soft_alt = pL - pR
    Tseg = len(pL)
    if Tseg < 2:
        return {
            "n_frames": Tseg,
            "pL_mean": float(pL.mean()) if Tseg else None,
            "pR_mean": float(pR.mean()) if Tseg else None,
            "pL_std": float(pL.std()) if Tseg else None,
            "pR_std": float(pR.std()) if Tseg else None,
            "soft_alt_mean": float(soft_alt.mean()) if Tseg else None,
            "soft_alt_std": float(soft_alt.std()) if Tseg else None,
            "soft_transition_density": None,
            "soft_both_stance": float((pL * pR).mean()) if Tseg else None,
            "soft_both_swing": float(((1 - pL) * (1 - pR)).mean()) if Tseg else None,
            "gt_transition_density": None,
            "constant_mid_rate": None,
            "low_alt_amplitude": None,
            "low_transition": None,
        }

    soft_trans = float(np.abs(np.diff(soft_alt)).mean())
    gt_trans = float(np.abs(np.diff(gt_hard_alt)).mean())

    constant_mid = (
        (pL > CONSTANT_MID_LO) & (pL < CONSTANT_MID_HI)
        & (pR > CONSTANT_MID_LO) & (pR < CONSTANT_MID_HI)
    )
    constant_mid_rate = float(constant_mid.mean())
    low_alt_amp = bool(soft_alt.std() < LOW_ALT_AMPLITUDE_STD)
    if gt_trans > 1e-6:
        low_trans = bool(soft_trans / gt_trans < LOW_TRANSITION_RATIO)
    else:
        low_trans = False

    return {
        "n_frames": Tseg,
        "pL_mean": float(pL.mean()),
        "pR_mean": float(pR.mean()),
        "pL_std": float(pL.std()),
        "pR_std": float(pR.std()),
        "soft_alt_mean": float(soft_alt.mean()),
        "soft_alt_std": float(soft_alt.std()),
        "soft_transition_density": soft_trans,
        "soft_both_stance": float((pL * pR).mean()),
        "soft_both_swing": float(((1 - pL) * (1 - pR)).mean()),
        "gt_transition_density": gt_trans,
        "constant_mid_rate": constant_mid_rate,
        "low_alt_amplitude": low_alt_amp,
        "low_transition": low_trans,
    }


def _aggregate_soft(rows: list[dict]) -> dict[str, Any]:
    if not rows:
        return {
            "n_segments": 0,
            "constant_mid_rate": None,
            "low_alt_amplitude_rate": None,
            "low_transition_rate": None,
        }
    means = {
        k: float(np.mean([r[k] for r in rows if r.get(k) is not None]))
        for k in (
            "pL_mean", "pR_mean", "pL_std", "pR_std",
            "soft_alt_mean", "soft_alt_std",
            "soft_transition_density", "soft_both_stance", "soft_both_swing",
            "gt_transition_density", "constant_mid_rate",
        )
        if any(r.get(k) is not None for r in rows)
    }
    flag_rate = lambda k: float(  # noqa: E731
        np.mean([1.0 if r.get(k) else 0.0 for r in rows])
    )
    return {
        "n_segments": len(rows),
        **means,
        "low_alt_amplitude_rate": flag_rate("low_alt_amplitude"),
        "low_transition_rate": flag_rate("low_transition"),
    }


def _aggregate_hard(rows: list[dict]) -> dict[str, Any]:
    if not rows:
        return {"n_segments": 0}
    out: dict[str, Any] = {"n_segments": len(rows)}
    for k in (
        "frac_both_stance", "frac_both_swing",
        "frac_L_only_stance", "frac_R_only_stance",
        "transitions_per_second",
    ):
        vals = np.array([r[k] for r in rows])
        out[k] = {
            "mean": float(vals.mean()),
            "median": float(np.median(vals)),
        }
    corrs = [r["L_R_height_corr"] for r in rows if r["L_R_height_corr"] is not None]
    out["L_R_height_corr"] = {
        "n": len(corrs),
        "mean": float(np.mean(corrs)) if corrs else None,
    }
    return out


def _write_summary_md(
    out_path: Path, hard_pred: dict, hard_gt: dict, soft_pred: dict,
    n_walking_segments: int, ckpt: Path,
) -> None:
    L: list[str] = []
    L.append("# G1 soft-stance diagnostic")
    L.append("")
    L.append(f"**ckpt:** `{ckpt}`")
    L.append(f"**walking segments:** {n_walking_segments}")
    L.append("")
    L.append("## Hard alternation (round26-style, for comparison)")
    L.append("")
    L.append("| metric | GT | pred |")
    L.append("| --- | ---: | ---: |")
    for k in (
        "frac_both_stance", "frac_both_swing",
        "frac_L_only_stance", "frac_R_only_stance",
        "transitions_per_second",
    ):
        gt_v = hard_gt.get(k, {}).get("mean")
        pr_v = hard_pred.get(k, {}).get("mean")
        L.append(
            f"| {k} | "
            f"{('-' if gt_v is None else f'{gt_v:.3f}')} | "
            f"{('-' if pr_v is None else f'{pr_v:.3f}')} |"
        )
    L_R = soft_pred.get("L_R_height_corr") or hard_pred.get("L_R_height_corr") or {}
    L.append("")
    L.append("## Soft stance (G1 training loss perspective)")
    L.append("")
    L.append("| metric | value | interpretation |")
    L.append("| --- | ---: | --- |")
    L.append(
        f"| pL_mean | {soft_pred.get('pL_mean', float('nan')):.3f} | "
        f"target ≈ GT duty cycle (~0.55-0.70) |"
    )
    L.append(
        f"| pR_mean | {soft_pred.get('pR_mean', float('nan')):.3f} | same |"
    )
    L.append(
        f"| soft_alt_std | {soft_pred.get('soft_alt_std', float('nan')):.3f} | "
        f"low (<0.15) indicates constant-mid degeneracy |"
    )
    L.append(
        f"| soft_transition_density | "
        f"{soft_pred.get('soft_transition_density', float('nan')):.4f} | "
        f"vs GT {soft_pred.get('gt_transition_density', float('nan')):.4f} |"
    )
    L.append(
        f"| soft_both_stance | "
        f"{soft_pred.get('soft_both_stance', float('nan')):.3f} | "
        f"close to GT both_stance (~0.15-0.19) when healthy |"
    )
    L.append(
        f"| soft_both_swing | "
        f"{soft_pred.get('soft_both_swing', float('nan')):.3f} | "
        f"close to GT both_swing (~0.25-0.30) when healthy |"
    )
    L.append("")
    L.append("## Degeneracy flags")
    L.append("")
    L.append("| flag | rate |")
    L.append("| --- | ---: |")
    L.append(
        f"| constant_mid_rate | "
        f"{soft_pred.get('constant_mid_rate', float('nan')):.3f} |"
    )
    L.append(
        f"| low_alt_amplitude_rate (seg) | "
        f"{soft_pred.get('low_alt_amplitude_rate', float('nan')):.3f} |"
    )
    L.append(
        f"| low_transition_rate (seg) | "
        f"{soft_pred.get('low_transition_rate', float('nan')):.3f} |"
    )
    L.append("")
    L.append("### Reading")
    L.append("")
    L.append(
        "- `constant_mid_rate > 0.40`: model parks both feet near "
        "soft-stance ≈ 0.5; G1 losses are satisfied as aggregate "
        "statistics but the soft probability does not encode real "
        "alternation. Hard L_R_height_corr / transitions/sec might still "
        "look OK because the foot position is fine, but soft stance "
        "isn't doing its job."
    )
    L.append(
        "- `low_alt_amplitude_rate > 0.50`: most walking segments have "
        "std(pL - pR) < 0.15. The soft stance is not separating L from R."
    )
    L.append(
        "- `low_transition_rate > 0.50`: soft transition density is "
        f"< {int(LOW_TRANSITION_RATIO * 100)}% of GT. Aggregate stat is "
        "matched by under-switching."
    )
    out_path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "G1 soft-stance diagnostic: checks whether a G1-trained model "
            "is actually predicting meaningful alternating stance or "
            "satisfying the aggregate-statistic losses degenerately."
        ),
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bucket", default="val", choices=["train", "val"])
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--substitute-conds-dir", type=Path, default=None,
        help=(
            "Optional dir of per-clip .npz that override oracle cond keys; "
            "used for R31/R32 downstream-coupling diag."
        ),
    )
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--walk-speed-m", type=float,
                        default=ROOT_WALK_SPEED_M_PER_FRAME_DEFAULT)
    parser.add_argument("--min-walk-frames", type=int,
                        default=MIN_WALK_LEN_FRAMES_DEFAULT)
    parser.add_argument("--stance-height-m", type=float,
                        default=STANCE_HEIGHT_M_DEFAULT)
    parser.add_argument("--stance-speed-m", type=float,
                        default=STANCE_SPEED_M_PER_FRAME_DEFAULT)
    parser.add_argument("--use-gt-as-pred", action="store_true")
    args = parser.parse_args()

    # Heavy imports deferred so the metric helpers (above) remain importable
    # in unit tests that only need numpy.
    import torch
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from piano.data.dataset import collate_hoi
    from piano.inference.diagnostic_helpers import (
        _build_cond, _build_dataset, _build_model, _fk_22joints,
        _stage1_norm_for_cfg, extract_train_time_meta,
        load_substitute_conds_for_clip,
    )
    from piano.training.temporal_interaction_losses import (
        TemporalInteractionLossConfig,
    )
    from piano.utils.clip_utils import load_clip_text_encoder

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sel_obj = json.loads(args.selection_json.read_text("utf-8"))
    selection = (
        sel_obj.get("selected") or sel_obj.get("candidates")
        or sel_obj.get("clips") or []
    )
    if not selection:
        raise SystemExit(f"empty selection: {args.selection_json}")
    sel_pairs = {(e["subset"], e["seq_id"]) for e in selection}
    print(f"[g1_soft] selection: {len(sel_pairs)} clips, bucket={args.bucket}")

    dataset = _build_dataset(cfg, args.bucket, augment=False)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        collate_fn=collate_hoi, num_workers=0,
    )

    model, object_encoder = _build_model(cfg, device)
    train_meta: dict[str, Any] = {}
    if not args.use_gt_as_pred:
        state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        train_meta = extract_train_time_meta(state)
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
            download_root=str(cfg.model.text_encoder.get(
                "download_root", "cache/clip")),
        )
    else:
        clip_model = None
    stage1_norm = _stage1_norm_for_cfg(cfg, device)
    model.eval()

    # Build TemporalInteractionLossConfig that matches G1's soft-stance helper.
    tl_cfg = TemporalInteractionLossConfig(
        r29_gait_soft_stance_speed_threshold_mps=(
            float(cfg.loss.get("temporal_interaction", {}).get(
                "r29_gait_soft_stance_speed_threshold_mps",
                SOFT_STANCE_DEFAULT_THRESHOLD_MPS,
            ))
        ),
        r29_gait_soft_stance_speed_softness_mps=(
            float(cfg.loss.get("temporal_interaction", {}).get(
                "r29_gait_soft_stance_speed_softness_mps",
                SOFT_STANCE_DEFAULT_SOFTNESS_MPS,
            ))
        ),
        floor_quantile=SOFT_STANCE_FLOOR_QUANTILE,
        grounded_threshold_above_floor_m=SOFT_STANCE_GROUNDED_THRESHOLD_ABOVE_FLOOR_M,
        grounded_softness_m=SOFT_STANCE_GROUNDED_SOFTNESS_M,
    )

    hard_pred_rows: list[dict] = []
    hard_gt_rows: list[dict] = []
    soft_rows: list[dict] = []
    seg_records: list[dict] = []
    n_processed = 0
    for batch in loader:
        subset = str(batch["subset"][0])
        seq_id = str(batch["seq_id"][0])
        if (subset, seq_id) not in sel_pairs:
            continue
        n_processed += 1

        T_for_sub = int(batch["motion"].shape[1])
        substitute_conds = load_substitute_conds_for_clip(
            args.substitute_conds_dir, subset, seq_id, T_for_sub, device,
        )
        cond, T = _build_cond(
            batch, model, object_encoder, clip_model, cfg, device,
            stage1_norm=stage1_norm,
            substitute_conds=substitute_conds,
        )
        gt_motion = batch["motion"][:, :T].to(device).float()
        if args.use_gt_as_pred:
            pred_motion = gt_motion
        else:
            torch.manual_seed(args.seed)
            with torch.no_grad():
                pred_motion = model.sample(
                    cond=cond, seq_length=T, cfg_scale=args.cfg_scale,
                )

        rest_offsets = batch["rest_offsets"].to(device).float()
        gt_joints_torch = _fk_22joints(gt_motion, rest_offsets)
        pred_joints_torch = _fk_22joints(pred_motion, rest_offsets)
        gt_joints = gt_joints_torch[0].cpu().numpy()
        pred_joints = pred_joints_torch[0].cpu().numpy()

        seq_len = int(batch["seq_len"][0].item())
        valid_T = min(T, seq_len)
        seq_mask = np.zeros(T, dtype=bool); seq_mask[:valid_T] = True

        segs = detect_walking_segments_gt(
            gt_joints, seq_mask, fps=args.fps,
            min_speed=args.walk_speed_m, min_length=args.min_walk_frames,
        )
        if not segs:
            continue

        # Compute soft stance once per clip (uses GT for floor).
        soft = _soft_stance_for_clip(
            pred_joints_torch, gt_joints_torch, tl_cfg, args.fps,
        )                                                       # (T, 2)

        for (t0, t1) in segs:
            gt_m = gait_metrics_for_segment(
                gt_joints[t0:t1 + 1], fps=args.fps,
                stance_height=args.stance_height_m,
                stance_speed=args.stance_speed_m,
            )
            pred_m = gait_metrics_for_segment(
                pred_joints[t0:t1 + 1], fps=args.fps,
                stance_height=args.stance_height_m,
                stance_speed=args.stance_speed_m,
            )
            gt_hard_alt = _hard_alt_from_joints(
                gt_joints[t0:t1 + 1],
                args.stance_height_m, args.stance_speed_m,
            )
            soft_m = _soft_metrics_for_segment(
                soft[t0:t1 + 1], gt_hard_alt,
            )

            seg_id = {"subset": subset, "seq_id": seq_id, "t0": int(t0), "t1": int(t1)}
            gt_m.update(seg_id); pred_m.update(seg_id); soft_m.update(seg_id)
            hard_pred_rows.append(pred_m)
            hard_gt_rows.append(gt_m)
            soft_rows.append(soft_m)
            seg_records.append({**seg_id, "hard_gt": gt_m, "hard_pred": pred_m,
                                "soft_pred": soft_m})

        if n_processed % 4 == 0:
            print(f"  [g1_soft {n_processed}/{len(sel_pairs)}] {subset}/{seq_id}  "
                  f"+{len(segs)} walking segments (total {len(soft_rows)})")

    if not soft_rows:
        print("[g1_soft] no walking segments produced soft-stance rows.")
        return 1

    hard_pred_agg = _aggregate_hard(hard_pred_rows)
    hard_gt_agg = _aggregate_hard(hard_gt_rows)
    soft_agg = _aggregate_soft(soft_rows)

    stats = {
        "config": str(args.config),
        "ckpt": str(args.ckpt),
        "use_gt_as_pred": args.use_gt_as_pred,
        "bucket": args.bucket,
        "train_time": train_meta,
        "fps": args.fps,
        "stance_height_m": args.stance_height_m,
        "stance_speed_m_per_frame": args.stance_speed_m,
        "walk_speed_m_per_frame": args.walk_speed_m,
        "n_clips_processed": n_processed,
        "n_walking_segments": len(soft_rows),
        "soft_aggregate": soft_agg,
        "hard_pred_aggregate": hard_pred_agg,
        "hard_gt_aggregate": hard_gt_agg,
        "per_segment": seg_records,
    }
    out_json = args.output_dir / "g1_soft_stance_stats.json"
    out_md = args.output_dir / "g1_soft_stance_summary.md"
    out_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    _write_summary_md(
        out_md, hard_pred_agg, hard_gt_agg, soft_agg,
        len(soft_rows), args.ckpt,
    )
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
