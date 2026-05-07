"""Scan all clips for motion_263 cumsum-drift.

For each clip:
    1. Re-encode the saved joints_22 via HumanML3DEncoder → produces
       process_file's drift-free `global_positions`.
    2. Decode the saved motion_263 via recover_from_ric → drifted canonical.
    3. Per-frame pelvis distance between them = the cumsum drift caused
       by motion_263's Y-only-rotation-velocity encoding losing X/Z
       body-twist info (analyses/2026-05-08_anchordiff_frame_bug_fix.md
       Bug 3).

Clips with max-pelvis-drift > threshold are flagged for exclusion from
AnchorDiff training. Their motion_263 → recover_from_ric path is so
inconsistent with the GT body that anchor consistency loss would
push the model toward an unreachable target.

Output: ``analyses/<date>_motion_263_drift_scan/<subset>/drift_*.json``
plus a top-level ``exclude_drifty_clips.json`` listing seq_ids to
exclude. Run ``apply_caption_exclusions.py``-style update of
metadata_clean.json afterwards.

Usage:
    python scripts/stage1_pseudo_labels/scan_motion_263_drift.py \\
        --config configs/training/anchordiff_v1.yaml \\
        --threshold-m 0.30 \\
        --output analyses/2026-05-08_motion_263_drift_scan
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from piano.data.humanml3d_encoder import HumanML3DEncoder
from piano.training.anchor_consistency_loss import lift_motion263_to_joints


def _first_usable_joints(motions_dir: Path) -> np.ndarray:
    for npz_path in sorted(motions_dir.glob("*.npz")):
        data = np.load(npz_path)
        joints = data["joints_22"]
        if joints.shape[0] >= 2:
            return joints[0]
    raise RuntimeError(f"No usable npz with joints_22 in {motions_dir}")


def _scan_subset(
    subset_root: Path,
    threshold_m: float,
    output_dir: Path,
) -> dict:
    motions_dir = subset_root / "motions"
    if not motions_dir.exists():
        raise FileNotFoundError(f"missing motions dir: {motions_dir}")

    # Use the same per-subset reference as the rebuild script did.
    reference = _first_usable_joints(motions_dir)
    encoder = HumanML3DEncoder(reference_joints=reference, feet_thre=0.002)

    npz_paths = sorted(motions_dir.glob("*.npz"))
    per_clip: list[dict] = []
    drifty: list[dict] = []
    t0 = time.time()

    for npz_path in tqdm(npz_paths, desc=f"{subset_root.name}"):
        seq_id = npz_path.stem
        try:
            data = np.load(npz_path)
            joints = data["joints_22"]
            motion = data["motion_263"]

            # Re-encode to get drift-free global_positions.
            features_re, global_positions = encoder.encode(joints)
            T = min(features_re.shape[0], motion.shape[0])
            if T < 2:
                continue

            # Decode saved motion_263.
            recovered = lift_motion263_to_joints(
                torch.from_numpy(motion[:T]).float().unsqueeze(0)
            ).squeeze(0).cpu().numpy()                             # (T, 22, 3)

            # Per-frame pelvis drift (joint 0).
            drift = np.linalg.norm(
                global_positions[:T, 0] - recovered[:T, 0],
                axis=-1,
            )                                                       # (T,)
            max_drift = float(drift.max())
            mean_drift = float(drift.mean())
            t_argmax = int(drift.argmax())

            row = {
                "seq_id": seq_id,
                "subset": subset_root.name,
                "T": int(T),
                "max_drift_m": max_drift,
                "mean_drift_m": mean_drift,
                "max_drift_frame": t_argmax,
            }
            per_clip.append(row)
            if max_drift > threshold_m:
                drifty.append(row)
        except Exception as e:
            per_clip.append({
                "seq_id": seq_id,
                "subset": subset_root.name,
                "error": str(e),
            })

    elapsed = time.time() - t0

    # Write outputs
    out_sub = output_dir / subset_root.name
    out_sub.mkdir(parents=True, exist_ok=True)
    (out_sub / "drift_per_clip.json").write_text(
        json.dumps(per_clip, indent=2)
    )
    (out_sub / "drift_above_threshold.json").write_text(
        json.dumps(drifty, indent=2)
    )

    n_total = sum(1 for r in per_clip if "error" not in r)
    n_drifty = len(drifty)
    pct = (n_drifty / n_total * 100.0) if n_total else 0.0
    print(
        f"[{subset_root.name}] {n_drifty}/{n_total} "
        f"({pct:.2f}%) above {threshold_m*100:.0f}cm, "
        f"elapsed {elapsed:.1f}s"
    )
    return {
        "subset": subset_root.name,
        "n_total": n_total,
        "n_drifty": n_drifty,
        "drifty": drifty,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--threshold-m", type=float, default=0.30,
                        help="max pelvis-drift threshold (m)")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    all_drifty: list[dict] = []
    for entry in cfg.data.datasets:
        subset_root = Path(entry.root)
        result = _scan_subset(subset_root, args.threshold_m, output_dir)
        summary.append({k: v for k, v in result.items() if k != "drifty"})
        all_drifty.extend(result["drifty"])

    # Top-level: complete drifty list + summary
    (output_dir / "exclude_drifty_clips.json").write_text(json.dumps(
        {
            "threshold_m": args.threshold_m,
            "n_excluded": len(all_drifty),
            "by_subset": summary,
            "exclusions": all_drifty,
        }, indent=2,
    ))

    print(f"\nTotal flagged: {len(all_drifty)} clips. Output: {output_dir}/")


if __name__ == "__main__":
    main()
