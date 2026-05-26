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
    extract_train_time_meta,
)
from anchor_realization_diagnostic import _fk_22joints  # noqa: E402

from piano.data.dataset import collate_hoi  # noqa: E402
from piano.utils.canonical_frame import (  # noqa: E402
    _facing_angle_y,
    y_rotation_matrix,
)
from piano.utils.clip_utils import load_clip_text_encoder  # noqa: E402


# ----------------------------------------------------------------------
# Body-action oracle hint (diag-local copy).
#
# This is a measurement-only construction lifted from the deleted
# piano.data.interaction_hint.build_body_action_oracle_hint (Tier-1
# cleanup, commit 79c894b). The function is not used by the model
# anywhere in the R29 path, so it does not belong in src/. It stays
# inline here so this diag still runs.
# ----------------------------------------------------------------------

# SMPL-22 joint indices.
_LEFT_WRIST_IDX, _RIGHT_WRIST_IDX = 20, 21
_LEFT_KNEE_IDX, _RIGHT_KNEE_IDX = 4, 5
_NECK_IDX = 12
_ROOT_IDX = 0

BODY_ACTION_KEY_JOINT_INDICES: tuple[int, ...] = (
    _LEFT_WRIST_IDX, _RIGHT_WRIST_IDX,
    _LEFT_KNEE_IDX, _RIGHT_KNEE_IDX,
    _NECK_IDX, _ROOT_IDX,         # pelvis must be LAST
)
BODY_ACTION_KEY_JOINT_NAMES: tuple[str, ...] = (
    "left_wrist", "right_wrist", "left_knee", "right_knee", "neck", "pelvis",
)
NUM_BODY_ACTION_JOINTS: int = len(BODY_ACTION_KEY_JOINT_INDICES)  # 6
HINT_DIM_BODY_ACTION: int = NUM_BODY_ACTION_JOINTS + NUM_BODY_ACTION_JOINTS * 3


def build_body_action_oracle_hint(
    joints_22: np.ndarray,
    mask_mode: str = "all_on",
    energy_threshold: float = 0.05,
    joint_indices: tuple[int, ...] = BODY_ACTION_KEY_JOINT_INDICES,
) -> np.ndarray:
    """24D body-action hint (T, 24): mask[6] + delta_local[6, 3].

    Root-yaw-canonical frame at t=0; pelvis-translated for non-pelvis
    joints; pelvis is the displacement from frame 0 in the same frame.
    Frame-0 invariant: hint[0, 6:24] == 0 by construction.
    """
    T = int(joints_22.shape[0])
    J = len(joint_indices)
    if J != NUM_BODY_ACTION_JOINTS:
        raise ValueError(f"joint_indices must have length {NUM_BODY_ACTION_JOINTS}; got {J}")
    if joints_22.shape != (T, 22, 3):
        raise ValueError(f"joints_22 must be (T, 22, 3); got {joints_22.shape!r}")
    if mask_mode not in {"all_on", "energy"}:
        raise ValueError(f"mask_mode must be 'all_on' or 'energy'; got {mask_mode!r}")

    joints = joints_22.astype(np.float32)
    yaw0 = _facing_angle_y(joints[0])
    R_root0_T = y_rotation_matrix(-yaw0)
    pelvis_world = joints[:, _ROOT_IDX, :]

    delta_local = np.zeros((T, J, 3), dtype=np.float32)
    for j_pos, j_idx in enumerate(joint_indices[:-1]):
        joint_world = joints[:, j_idx, :]
        joint_rel = joint_world - pelvis_world
        joint_local = joint_rel @ R_root0_T.T
        delta_local[:, j_pos, :] = joint_local - joint_local[0:1, :]

    pelvis_disp = pelvis_world - pelvis_world[0:1, :]
    delta_local[:, -1, :] = pelvis_disp @ R_root0_T.T

    if mask_mode == "all_on":
        joint_mask = np.ones((J,), dtype=np.float32)
    else:
        energy = np.linalg.norm(delta_local, axis=-1).mean(axis=0)
        joint_mask = (energy > float(energy_threshold)).astype(np.float32)
    joint_mask_t = np.broadcast_to(joint_mask[None, :], (T, J)).astype(np.float32)

    hint = np.concatenate(
        [joint_mask_t, delta_local.reshape(T, J * 3)],
        axis=-1,
    ).astype(np.float32)
    if hint.shape != (T, HINT_DIM_BODY_ACTION):
        raise AssertionError(f"hint shape {hint.shape!r} != ({T}, {HINT_DIM_BODY_ACTION})")
    if not np.isfinite(hint).all():
        raise FloatingPointError("non-finite values in body-action hint")
    return hint


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
    train_meta: dict = {}
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
        "train_time": train_meta,
        "energy_threshold_m": args.energy_threshold,
        "n_clips_processed": n_processed,
        "aggregate": stats,
        "per_clip": per_clip_records,
    }, indent=2), "utf-8")
    print(f"wrote {out_json}")
    if train_meta.get("train_wallclock_hms"):
        print(f"  train wallclock: {train_meta['train_wallclock_hms']} "
              f"({train_meta['train_wallclock_seconds']:.1f}s)")
    _write_summary_md(
        stats, out_md, args.ckpt, args.use_gt_as_pred,
        args.energy_threshold, n_processed,
    )
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
