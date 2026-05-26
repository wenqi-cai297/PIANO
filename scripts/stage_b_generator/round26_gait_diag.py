"""Round-26 gait diagnostic.

Detects walking segments (root horizontal speed sustained above threshold)
and within each segment characterizes foot alternation. Captures the user-
reported failure mode:

    "走路时人物的双脚没有有序地交替行动"

A per-frame anchor metric or per-frame foot position MSE cannot detect this
because it scores each frame independently and ignores the temporal
alternation structure (one foot planted while the other swings).

Per-segment metrics:

    n_frames                    number of frames in this walking segment
    frac_both_stance            both feet planted simultaneously (shuffle)
    frac_both_swing             both feet airborne simultaneously (run/jump)
    frac_L_only_stance          only L planted (R swinging) — gait state
    frac_R_only_stance          only R planted (L swinging) — gait state
    n_stance_transitions        L<->R alternation events
    transitions_per_second      density of alternations
    L_R_height_corr             corr(L_foot_y, R_foot_y); should be NEGATIVE
                                (anti-phase) for proper gait
    step_period_frames          dominant period via L_foot_y autocorrelation,
                                or None if no clear period

Detection thresholds (root walking, foot stance) are intentionally lenient —
they accept slow walking + heavy footfall datasets.

Comparison: walking segments are detected ON GT motion only (since GT is
the trusted reference for "this is a walking segment"). Both GT and pred
metrics are then computed over those same segment time windows. If pred is
doing something other than walking during those windows, the gait metrics
will reflect that mismatch — which IS the failure we want to catch.

Usage::

    conda run -n piano python scripts/stage_b_generator/round26_gait_diag.py \
        --config configs/training/anchordiff_v27_stage2_anchoraware_FULL_DATA_local.yaml \
        --ckpt   runs/training/stageB_anchordiff_v27_stage2_anchoraware_FULL_DATA/final.pt \
        --selection-json analyses/round25_multimodal_eval_subset.json \
        --output-dir analyses/round26_gait_v27_final

GT-self reference::

    ... --use-gt-as-pred --output-dir analyses/round26_gait_gt_reference
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
from torch.utils.data import DataLoader

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from plan_condition_diagnostics import (  # noqa: E402
    _build_cond, _build_dataset, _build_model, _stage1_norm_for_cfg,
    extract_train_time_meta,
)
from anchor_realization_diagnostic import _fk_22joints  # noqa: E402

from piano.data.dataset import collate_hoi  # noqa: E402
from piano.utils.clip_utils import load_clip_text_encoder  # noqa: E402


# SMPL-22 joint indices
J_PELVIS = 0
J_L_FOOT = 7   # left ankle; keep gait metric aligned with Tier-0B loss/hint
J_R_FOOT = 8   # right ankle; SMPL 10/11 are mid-foot contact proxies

# Thresholds (tunable via CLI)
ROOT_WALK_SPEED_M_PER_FRAME_DEFAULT = 0.005   # ~10 cm/sec at 20fps
MIN_WALK_LEN_FRAMES_DEFAULT = 20              # 1 second at 20fps
STANCE_HEIGHT_M_DEFAULT = 0.12                # foot below 12cm = on ground
STANCE_SPEED_M_PER_FRAME_DEFAULT = 0.015      # foot moving < 30 cm/sec


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
    horiz_vel = np.concatenate([[0], np.linalg.norm(np.diff(root_xz, axis=0), axis=-1)])
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
    """Boolean (Tseg,) — True if foot is planted (low + slow)."""
    height = foot_xyz[:, 1]
    horiz_vel = np.concatenate([[0], np.linalg.norm(np.diff(foot_xyz[:, [0, 2]], axis=0), axis=-1)])
    return (height < stance_height) & (horiz_vel < stance_speed)


def _step_period(L_foot_y: np.ndarray) -> int | None:
    """Estimate dominant step period via autocorrelation of L foot height.
    Returns period in frames, or None if no clear peak."""
    if len(L_foot_y) < 10:
        return None
    y = L_foot_y - L_foot_y.mean()
    if np.std(y) < 1e-4:
        return None
    # autocorrelation
    ac = np.correlate(y, y, mode="full")[len(y) - 1:]
    ac = ac / max(ac[0], 1e-9)
    # search for first peak after a dip
    dip_found = False
    for lag in range(2, min(len(ac), 60)):  # period <= 3 seconds at 20fps
        if not dip_found and ac[lag] < 0.2:
            dip_found = True
        if dip_found and ac[lag] > 0.4 and ac[lag] >= ac[lag - 1] and (lag + 1 >= len(ac) or ac[lag] >= ac[lag + 1]):
            return int(lag)
    return None


def gait_metrics_for_segment(
    joints: np.ndarray,             # (Tseg, 22, 3)
    fps: float,
    stance_height: float,
    stance_speed: float,
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

    # State: 0 = L-only, 1 = R-only, -1 = transient
    state = np.where(L_stance & ~R_stance, 0,
            np.where(~L_stance & R_stance, 1, -1))
    transitions = 0
    last_solid = -1
    for s in state:
        if s == -1: continue
        if last_solid != -1 and s != last_solid:
            transitions += 1
        last_solid = int(s)

    L_R_corr = float(np.corrcoef(L_foot[:, 1], R_foot[:, 1])[0, 1]) if T > 5 else None
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


def aggregate(rows: list[dict]) -> dict:
    keys_means = [
        "frac_both_stance", "frac_both_swing",
        "frac_L_only_stance", "frac_R_only_stance",
        "transitions_per_second",
    ]
    out = {"n_segments": len(rows)}
    for k in keys_means:
        vals = np.array([r[k] for r in rows])
        out[k] = {
            "mean": float(vals.mean()) if len(vals) else 0.0,
            "median": float(np.median(vals)) if len(vals) else 0.0,
            "p25": float(np.percentile(vals, 25)) if len(vals) else 0.0,
            "p75": float(np.percentile(vals, 75)) if len(vals) else 0.0,
        }
    # L_R corr (filter None)
    corrs = [r["L_R_height_corr"] for r in rows if r["L_R_height_corr"] is not None]
    out["L_R_height_corr"] = {
        "n": len(corrs),
        "mean": float(np.mean(corrs)) if corrs else None,
        "median": float(np.median(corrs)) if corrs else None,
        "n_below_minus_0.3": int(sum(1 for c in corrs if c < -0.3)),
        "rate_below_minus_0.3": (sum(1 for c in corrs if c < -0.3) / len(corrs)) if corrs else None,
    }
    # Step period (filter None)
    periods = [r["step_period_frames"] for r in rows if r["step_period_frames"] is not None]
    out["step_period_frames"] = {
        "n_segments_with_period": len(periods),
        "mean": float(np.mean(periods)) if periods else None,
        "median": float(np.median(periods)) if periods else None,
        "rate_with_period": (len(periods) / max(len(rows), 1)),
    }
    return out


def write_summary_md(stats_pred: dict, stats_gt: dict, out_path: Path,
                     ckpt: Path, use_gt_as_pred: bool,
                     stance_h: float, stance_s: float, walk_speed: float) -> None:
    L = [
        "# Round-26 gait diagnostic",
        "",
        f"**Source:** `{ckpt}`",
        f"**Mode:** {'GT used as pred (sanity baseline)' if use_gt_as_pred else 'model sample'}",
        "",
        f"Walking detection: root horizontal speed > {walk_speed*100:.1f} cm/frame, "
        f"sustained ≥ 20 frames.",
        f"Stance detection: foot height < {stance_h*100:.0f} cm AND foot horiz speed < {stance_s*100:.1f} cm/frame.",
        "",
        "## Headline comparison: pred vs GT on same walking segments",
        "",
        "| metric | GT | pred | GT-pred Δ | interpretation |",
        "|---|---:|---:|---:|---|",
    ]
    def fmt(d, key, prec=3):
        return f"{d[key]['mean']:.{prec}f}"
    def fmt_delta(d_pred, d_gt, key, prec=3):
        return f"{d_pred[key]['mean'] - d_gt[key]['mean']:+.{prec}f}"

    sp, sg = stats_pred, stats_gt
    L.append(f"| frac_both_stance | {fmt(sg,'frac_both_stance')} | {fmt(sp,'frac_both_stance')} | {fmt_delta(sp,sg,'frac_both_stance')} | both feet planted; high = shuffling, not stepping |")
    L.append(f"| frac_both_swing | {fmt(sg,'frac_both_swing')} | {fmt(sp,'frac_both_swing')} | {fmt_delta(sp,sg,'frac_both_swing')} | both feet airborne; high = running or floating |")
    L.append(f"| frac_L_only_stance | {fmt(sg,'frac_L_only_stance')} | {fmt(sp,'frac_L_only_stance')} | {fmt_delta(sp,sg,'frac_L_only_stance')} | proper L-foot stance during R-foot swing |")
    L.append(f"| frac_R_only_stance | {fmt(sg,'frac_R_only_stance')} | {fmt(sp,'frac_R_only_stance')} | {fmt_delta(sp,sg,'frac_R_only_stance')} | proper R-foot stance during L-foot swing |")
    L.append(f"| transitions/sec | {fmt(sg,'transitions_per_second',2)} | {fmt(sp,'transitions_per_second',2)} | {fmt_delta(sp,sg,'transitions_per_second',2)} | L↔R alternation density (Hz); 2*step-cadence |")

    L.append("")
    L.append("## L-R foot height correlation (anti-phase indicator)")
    L.append("")
    L.append("In proper gait L and R foot heights alternate: when L is up (swing), R is down (stance) and vice versa.")
    L.append("Correlation should be **NEGATIVE**, ideally < -0.3.")
    L.append("")
    L.append("| | GT | pred |")
    L.append("|---|---:|---:|")
    L.append(f"| mean L_R_height_corr | {sg['L_R_height_corr']['mean']:.3f} | {sp['L_R_height_corr']['mean']:.3f} |")
    L.append(f"| % segments with corr < -0.3 | {100*sg['L_R_height_corr']['rate_below_minus_0.3']:.1f}% | {100*sp['L_R_height_corr']['rate_below_minus_0.3']:.1f}% |")
    L.append("")
    L.append("## Step period")
    L.append("")
    L.append("| | GT | pred |")
    L.append("|---|---:|---:|")
    L.append(f"| segments with detected period | {100*sg['step_period_frames']['rate_with_period']:.1f}% | {100*sp['step_period_frames']['rate_with_period']:.1f}% |")
    if sg['step_period_frames']['mean']:
        sp_mean = sp['step_period_frames']['mean']
        sp_str = f"{sp_mean:.1f}" if sp_mean is not None else "n/a"
        L.append(f"| mean period (frames) | {sg['step_period_frames']['mean']:.1f} | {sp_str} |")
    L.append("")
    L.append("## Headline finding")
    L.append("")
    L.append("**The user reported '走路时双脚没有有序地交替行动'. The numeric signals of that failure are:**")
    L.append("- `frac_both_stance` >> GT's value (shuffling instead of stepping)")
    L.append("- `frac_both_swing` >> GT's value (no foot grounded)")
    L.append("- `transitions/sec` < GT's (fewer alternations)")
    L.append("- `L_R_height_corr` > GT's (less anti-phase, possibly +ve = in-phase)")
    L.append("- `step_period_frames` undetected on pred but present on GT")
    L.append("")
    out_path.write_text("\n".join(L), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bucket", default="val", choices=["train", "val"])
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sel_obj = json.loads(args.selection_json.read_text("utf-8"))
    selection = (
        sel_obj.get("selected")
        or sel_obj.get("candidates")
        or sel_obj.get("clips")
        or []
    )
    if not selection:
        raise SystemExit(f"empty selection: {args.selection_json}")
    sel_pairs = {(e["subset"], e["seq_id"]) for e in selection}
    print(f"[gait] selection: {len(sel_pairs)} clips")

    dataset = _build_dataset(cfg, args.bucket, augment=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        collate_fn=collate_hoi, num_workers=0)

    model, object_encoder, z_dims = _build_model(cfg, device)
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
            download_root=str(cfg.model.text_encoder.get("download_root", "cache/clip")),
        )
    else:
        clip_model = None
    stage1_norm = _stage1_norm_for_cfg(cfg, device)
    model.eval()

    pred_rows: list[dict] = []
    gt_rows: list[dict] = []
    seg_records: list[dict] = []
    n_processed = 0
    for batch in loader:
        subset = str(batch["subset"][0])
        seq_id = str(batch["seq_id"][0])
        if (subset, seq_id) not in sel_pairs:
            continue
        n_processed += 1

        cond, T = _build_cond(
            batch, model, object_encoder, clip_model, z_dims, cfg, device,
            stage1_norm=stage1_norm,
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
        gt_joints = _fk_22joints(gt_motion, rest_offsets)[0].cpu().numpy()
        pred_joints = _fk_22joints(pred_motion, rest_offsets)[0].cpu().numpy()

        seq_len = int(batch["seq_len"][0].item())
        valid_T = min(T, seq_len)
        seq_mask = np.zeros(T, dtype=bool); seq_mask[:valid_T] = True

        segs = detect_walking_segments_gt(
            gt_joints, seq_mask, fps=args.fps,
            min_speed=args.walk_speed_m, min_length=args.min_walk_frames,
        )

        for (t0, t1) in segs:
            gt_m = gait_metrics_for_segment(
                gt_joints[t0:t1 + 1], fps=args.fps,
                stance_height=args.stance_height_m, stance_speed=args.stance_speed_m,
            )
            pred_m = gait_metrics_for_segment(
                pred_joints[t0:t1 + 1], fps=args.fps,
                stance_height=args.stance_height_m, stance_speed=args.stance_speed_m,
            )
            gt_m.update({"subset": subset, "seq_id": seq_id, "t0": t0, "t1": t1})
            pred_m.update({"subset": subset, "seq_id": seq_id, "t0": t0, "t1": t1})
            gt_rows.append(gt_m); pred_rows.append(pred_m)
            seg_records.append({
                "subset": subset, "seq_id": seq_id, "t0": t0, "t1": t1,
                "gt": gt_m, "pred": pred_m,
            })

        if n_processed % 4 == 0:
            print(f"  [gait {n_processed}/{len(sel_pairs)}] {subset}/{seq_id}  "
                  f"+{len(segs)} walking segments (total {len(pred_rows)})")

    if not pred_rows:
        print("[gait] no walking segments detected in selection.")
        return 1

    stats_pred = aggregate(pred_rows)
    stats_gt = aggregate(gt_rows)

    out_json = args.output_dir / "gait_stats.json"
    out_md = args.output_dir / "gait_summary.md"
    out_json.write_text(json.dumps({
        "config": str(args.config),
        "ckpt": str(args.ckpt),
        "use_gt_as_pred": args.use_gt_as_pred,
        "train_time": train_meta,
        "fps": args.fps,
        "walk_speed_m_per_frame": args.walk_speed_m,
        "min_walk_frames": args.min_walk_frames,
        "stance_height_m": args.stance_height_m,
        "stance_speed_m_per_frame": args.stance_speed_m,
        "n_clips_processed": n_processed,
        "n_walking_segments": len(pred_rows),
        "pred_aggregate": stats_pred,
        "gt_aggregate": stats_gt,
        "per_segment": seg_records,
    }, indent=2), "utf-8")
    print(f"wrote {out_json}")
    if train_meta.get("train_wallclock_hms"):
        print(f"  train wallclock: {train_meta['train_wallclock_hms']} "
              f"({train_meta['train_wallclock_seconds']:.1f}s)")
    write_summary_md(
        stats_pred, stats_gt, out_md,
        args.ckpt, args.use_gt_as_pred,
        args.stance_height_m, args.stance_speed_m, args.walk_speed_m,
    )
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
