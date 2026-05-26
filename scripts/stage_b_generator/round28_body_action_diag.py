"""Round-28 body-action diagnostic.

Evaluates how well a model's predicted motion reproduces six key joints'
body-only deltas in the per-clip root-yaw-canonical frame anchored at
frame 0:

    left_wrist, right_wrist, left_knee, right_knee, neck, pelvis

For each clip we forward both GT and pred motion through the same
body-action hint constructor (``build_body_action_oracle_hint``) and
compare the resulting per-joint per-frame deltas. Per joint we report:

    delta_error_cm        mean_t || pred_delta - gt_delta ||
    amp_pred_cm           mean_t || pred_delta ||
    amp_gt_cm             mean_t || gt_delta ||
    amp_ratio             amp_pred / (amp_gt + eps)
    direction_cosine      mean cos(pred_delta, gt_delta) over frames
                          with ||gt_delta|| > 0.01 m  (only "active" frames)
    active_frame_frac     fraction of frames considered active
    energy_mask_pred      per-joint energy mask (==1 if amp_pred >
                          energy_threshold), broadcast over the clip
    energy_mask_gt        same for GT (the "ground-truth high-energy" joints)

The diagnostic mirrors the CLI shape of
``round26_sustained_contact_diag.py`` / ``round26_gait_diag.py``:

    conda run -n piano python scripts/stage_b_generator/round28_body_action_diag.py \\
        --config configs/training/<cfg>.yaml \\
        --ckpt   runs/training/<run>/final.pt \\
        --selection-json analyses/<selection>.json \\
        --output-dir analyses/round28_body_action_<variant>

Optional flags:
    --use-gt-as-pred   sanity baseline; pred=GT → delta_error≈0,
                       amp_ratio≈1, direction_cosine≈1.
    --energy-threshold ENERGY threshold for the energy mask report (m).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from plan_condition_diagnostics import (  # noqa: E402
    _build_cond, _build_dataset, _build_model, _stage1_norm_for_cfg,
)
from anchor_realization_diagnostic import _fk_22joints  # noqa: E402

from piano.data.dataset import collate_hoi  # noqa: E402
from piano.data.interaction_hint import (  # noqa: E402
    BODY_ACTION_KEY_JOINT_NAMES,
    NUM_BODY_ACTION_JOINTS,
    build_body_action_oracle_hint,
)
from piano.utils.clip_utils import load_clip_text_encoder  # noqa: E402


def _per_joint_metrics(
    pred_joints: np.ndarray,
    gt_joints: np.ndarray,
    valid_T: int,
    energy_threshold_m: float,
    active_threshold_m: float = 0.01,
    eps: float = 1e-6,
) -> dict:
    """Compute per-joint amp / direction / error stats for ONE clip."""
    pred_hint = build_body_action_oracle_hint(
        pred_joints[:valid_T].astype(np.float32),
        mask_mode="all_on",
    )
    gt_hint = build_body_action_oracle_hint(
        gt_joints[:valid_T].astype(np.float32),
        mask_mode="all_on",
    )
    # shape: (T, 6, 3)
    pred_delta = pred_hint[:, 6:].reshape(valid_T, NUM_BODY_ACTION_JOINTS, 3)
    gt_delta = gt_hint[:, 6:].reshape(valid_T, NUM_BODY_ACTION_JOINTS, 3)

    out: dict[str, dict] = {}
    for j, name in enumerate(BODY_ACTION_KEY_JOINT_NAMES):
        pd = pred_delta[:, j, :]                                          # (T, 3)
        gd = gt_delta[:, j, :]
        d_err = np.linalg.norm(pd - gd, axis=-1)                          # (T,)
        amp_p = np.linalg.norm(pd, axis=-1)
        amp_g = np.linalg.norm(gd, axis=-1)
        amp_p_mean = float(amp_p.mean())
        amp_g_mean = float(amp_g.mean())
        active = amp_g > active_threshold_m
        if active.any():
            pd_n = pd[active]; gd_n = gd[active]
            pd_norm = pd_n / (np.linalg.norm(pd_n, axis=-1, keepdims=True) + eps)
            gd_norm = gd_n / (np.linalg.norm(gd_n, axis=-1, keepdims=True) + eps)
            cos = float((pd_norm * gd_norm).sum(axis=-1).mean())
        else:
            cos = float("nan")
        out[name] = {
            "delta_error_cm": float(d_err.mean()) * 100.0,
            "delta_error_p95_cm": float(np.quantile(d_err, 0.95)) * 100.0,
            "amp_pred_cm": amp_p_mean * 100.0,
            "amp_gt_cm": amp_g_mean * 100.0,
            "amp_ratio": amp_p_mean / (amp_g_mean + eps),
            "direction_cosine": cos,
            "active_frame_frac": float(active.mean()),
            "energy_mask_pred": int(amp_p_mean > energy_threshold_m),
            "energy_mask_gt": int(amp_g_mean > energy_threshold_m),
        }
    return out


def _aggregate(rows: list[dict]) -> dict:
    """Average per-joint metrics across all clips."""
    if not rows:
        return {name: {} for name in BODY_ACTION_KEY_JOINT_NAMES}
    out: dict[str, dict] = {}
    keys = ("delta_error_cm", "delta_error_p95_cm", "amp_pred_cm",
            "amp_gt_cm", "amp_ratio", "direction_cosine",
            "active_frame_frac", "energy_mask_pred", "energy_mask_gt")
    for name in BODY_ACTION_KEY_JOINT_NAMES:
        joint_rows = [r[name] for r in rows]
        agg: dict = {}
        for k in keys:
            vals = [float(jr[k]) for jr in joint_rows if jr.get(k) is not None
                    and not (isinstance(jr[k], float) and np.isnan(jr[k]))]
            if vals:
                agg[f"{k}_mean"] = float(np.mean(vals))
                agg[f"{k}_median"] = float(np.median(vals))
            else:
                agg[f"{k}_mean"] = None
                agg[f"{k}_median"] = None
        out[name] = agg
    return out


def _write_summary_md(
    stats: dict,
    out_path: Path,
    ckpt: Path,
    use_gt_as_pred: bool,
    energy_threshold_m: float,
    n_clips: int,
) -> None:
    L = [
        "# Round-28 body-action diagnostic",
        "",
        f"**Source:** `{ckpt}`",
        f"**Mode:** {'GT used as pred (sanity baseline)' if use_gt_as_pred else 'model sample'}",
        f"**Clips evaluated:** {n_clips}",
        f"**Energy threshold:** {energy_threshold_m*100:.1f} cm (mean per-clip joint amplitude)",
        "",
        "## Per-joint metrics (mean over clips)",
        "",
        "Delta = position in per-clip root-yaw-canonical pelvis-local frame, anchored at frame 0.",
        "For pelvis: displacement-from-frame-0 in the same canonical frame.",
        "",
        "| joint | delta_error [cm] | amp_pred [cm] | amp_gt [cm] | amp_ratio | direction_cos | active_frac |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in BODY_ACTION_KEY_JOINT_NAMES:
        s = stats.get(name, {})
        def fmt(k, prec=2):
            v = s.get(k)
            return f"{v:.{prec}f}" if v is not None else "n/a"
        L.append(
            f"| {name} | {fmt('delta_error_cm_mean')} | "
            f"{fmt('amp_pred_cm_mean')} | {fmt('amp_gt_cm_mean')} | "
            f"{fmt('amp_ratio_mean', 3)} | "
            f"{fmt('direction_cosine_mean', 3)} | "
            f"{fmt('active_frame_frac_mean', 3)} |"
        )
    L.append("")
    L.append("## Interpretation guide")
    L.append("")
    L.append("- **delta_error_cm**: distance between pred and GT per-joint")
    L.append("  delta vectors (canonical frame). Lower = pred follows GT body motion.")
    L.append("- **amp_ratio**: 1.0 ≈ pred matches GT body-motion magnitude;")
    L.append("  < 1 = under-articulating (stiff body); > 1 = over-articulating.")
    L.append("- **direction_cosine**: 1.0 = pred delta direction matches GT;")
    L.append("  computed only on frames where ||gt_delta|| > 1 cm (active frames).")
    L.append("  Negative values mean pred moves the joint in the OPPOSITE direction.")
    L.append("- **active_frame_frac**: fraction of frames where the GT joint moved")
    L.append("  more than 1 cm (i.e. how 'active' this joint is in the clip).")
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
    parser.add_argument("--energy-threshold", type=float, default=0.05)
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
    print(f"[body-action] selection: {len(sel_pairs)} clips")

    dataset = _build_dataset(cfg, args.bucket, augment=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        collate_fn=collate_hoi, num_workers=0)

    model, object_encoder, z_dims = _build_model(cfg, device)
    if not args.use_gt_as_pred:
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

    clip_rows: list[dict] = []
    per_clip_records: list[dict] = []
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
        cond["interaction_plan"] = {
            k: batch[f"plan_{k}"].to(device) for k in plan_keys
        }

        gt_motion = batch["motion"][:, :T].to(device).float()
        if args.use_gt_as_pred:
            pred_motion = gt_motion
        else:
            torch.manual_seed(args.seed)
            with torch.no_grad():
                pred_motion = model.sample(
                    cond=cond, seq_length=T, cfg_scale=args.cfg_scale,
                    replacement="none", output_skip=False,
                )

        rest_offsets = batch["rest_offsets"].to(device).float()
        gt_joints = _fk_22joints(gt_motion, rest_offsets)[0].cpu().numpy()
        pred_joints = _fk_22joints(pred_motion, rest_offsets)[0].cpu().numpy()

        seq_len = int(batch["seq_len"][0].item())
        valid_T = min(T, seq_len)

        metrics = _per_joint_metrics(
            pred_joints, gt_joints, valid_T,
            energy_threshold_m=args.energy_threshold,
        )
        clip_rows.append(metrics)
        per_clip_records.append({
            "subset": subset, "seq_id": seq_id,
            "valid_T": valid_T,
            "per_joint": metrics,
        })

        if n_processed % 4 == 0:
            wr_err = metrics.get("left_wrist", {}).get("delta_error_cm", float("nan"))
            print(
                f"  [body-action {n_processed}/{len(sel_pairs)}] {subset}/{seq_id}  "
                f"L_wrist_err={wr_err:.2f}cm"
            )

    if not clip_rows:
        print("[body-action] no clips matched the selection.")
        return 1

    stats = _aggregate(clip_rows)

    out_json = args.output_dir / "body_action_stats.json"
    out_md = args.output_dir / "body_action_summary.md"
    out_json.write_text(json.dumps({
        "config": str(args.config),
        "ckpt": str(args.ckpt),
        "use_gt_as_pred": args.use_gt_as_pred,
        "energy_threshold_m": args.energy_threshold,
        "n_clips_processed": n_processed,
        "aggregate": stats,
        "per_clip": per_clip_records,
    }, indent=2), "utf-8")
    print(f"wrote {out_json}")
    _write_summary_md(
        stats, out_md, args.ckpt, args.use_gt_as_pred,
        args.energy_threshold, n_processed,
    )
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
