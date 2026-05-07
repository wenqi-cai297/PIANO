"""GT roundtrip diagnostic for PIANO-AnchorDiff.

Two variants run side-by-side:

    canonical (the path AnchorDiff training uses):
        1. GT motion_263                          ─── recover_from_ric → joints_canon
        2. GT contact_target_xyz (object-local)
           + GT obj_com_canonical                 ─── lift to body-canonical → target_canon
           + GT obj_rot6d_canonical
        3. distance(joints_canon[PART_TO_JOINT], target_canon)

    world (sanity check independent of motion_263 / canonical machinery):
        1. GT joints_world (already in dataset)
        2. GT contact_target_xyz (object-local)
           + GT object_positions (world)          ─── lift to world → target_world
           + GT object_rotations (axis-angle, world)
        3. distance(joints_world[PART_TO_JOINT], target_world)

Both should be small on contact frames (≤ v18 threshold). If world
is small but canonical is big, the bug is in the canonical-frame
machinery (motion_263 ↔ obj_*_canonical mismatch). If both are big,
the bug is in pseudo-label extraction or in PART_TO_JOINT.

Usage:
    python scripts/stage_b_generator/anchordiff_gt_roundtrip.py \\
        --config configs/training/anchordiff_v1.yaml \\
        --num-clips 200 \\
        --output runs/eval/anchordiff_gt_roundtrip
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from piano.data.dataset import HOIDataset, collate_hoi
from piano.training.anchor_consistency_loss import (
    PART_TO_JOINT,
    lift_motion263_to_joints,
    lift_object_local_to_canonical,
)


def _axis_angle_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    """Rodrigues. ``aa`` shape (..., 3) → R shape (..., 3, 3)."""
    theta = aa.norm(dim=-1, keepdim=True).clamp_min(1e-12)         # (..., 1)
    k = aa / theta                                                  # (..., 3)
    K = torch.zeros(aa.shape[:-1] + (3, 3), device=aa.device, dtype=aa.dtype)
    kx, ky, kz = k.unbind(-1)
    zero = torch.zeros_like(kx)
    K[..., 0, 1] = -kz; K[..., 0, 2] = ky
    K[..., 1, 0] = kz;  K[..., 1, 2] = -kx
    K[..., 2, 0] = -ky; K[..., 2, 1] = kx
    eye = torch.eye(3, device=aa.device, dtype=aa.dtype).expand_as(K)
    sin = theta.unsqueeze(-1).sin()
    cos = theta.unsqueeze(-1).cos()
    return eye + sin * K + (1 - cos) * (K @ K)


def lift_object_local_to_world(
    target_local: torch.Tensor,        # (B, T, P, 3)
    object_positions: torch.Tensor,    # (B, T, 3)
    object_rotations: torch.Tensor,    # (B, T, 3) axis-angle
) -> torch.Tensor:
    """Lift a contact target from object-local to world frame using
    the world object pose (axis-angle convention from preprocess)."""
    R = _axis_angle_to_matrix(object_rotations)                     # (B, T, 3, 3)
    rotated = (R.unsqueeze(2) @ target_local.unsqueeze(-1)).squeeze(-1)  # (B, T, P, 3)
    return rotated + object_positions.unsqueeze(2)


PART_NAMES = ("left_hand", "right_hand", "left_foot", "right_foot", "pelvis")
PART_THRESHOLD_M = {
    "left_hand": 0.10, "right_hand": 0.10,
    "left_foot": 0.05, "right_foot": 0.05,
    "pelvis": 0.20,
}


def _load_subsets(cfg, n: int) -> list[tuple[str, dict]]:
    """Sample n clips uniformly from the four-subset training set."""
    rng = np.random.default_rng(seed=42)
    pseudo_label_subdir = cfg.data.get("pseudo_label_subdir", None)
    datasets = []
    sub_names = []
    for entry in cfg.data.datasets:
        sub_dir = (
            str(Path(entry.root) / pseudo_label_subdir)
            if pseudo_label_subdir is not None else None
        )
        ds = HOIDataset(
            root=entry.root,
            pseudo_label_dir=sub_dir,
            max_seq_length=cfg.data.max_seq_length,
            augment=None,
            support_collapse_hand_support=True,
            surface_obj_pose=True,
        )
        datasets.append(ds)
        sub_names.append(entry.name)
    sizes = [len(d) for d in datasets]
    print(f"Per-subset sizes: {dict(zip(sub_names, sizes))}")

    pool = []
    for ds_idx, ds in enumerate(datasets):
        for clip_idx in range(len(ds)):
            pool.append((ds_idx, clip_idx, sub_names[ds_idx]))
    sampled = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
    return [(datasets[pool[i][0]], pool[i][1], pool[i][2]) for i in sampled]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num-clips", type=int, default=200)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Sampling {args.num_clips} clips...")
    triples = _load_subsets(cfg, args.num_clips)

    # Per-(frame, subset, part) accumulators: list of frame distances.
    # frame ∈ {"canonical", "world"}.
    per_subset_part: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    overall_part: dict[tuple[str, str], list[float]] = defaultdict(list)
    per_clip_summary: list[dict] = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Process clip by clip (B=1) to keep memory low and easy bookkeeping.
    n_done = 0
    part_idx = torch.tensor(PART_TO_JOINT, device=device, dtype=torch.long)
    for ds, clip_idx, subset in triples:
        sample = ds[clip_idx]
        batch = collate_hoi([sample])
        motion = batch["motion"].to(device)                   # (1, T, 263)
        joints_world = batch["joints"].to(device)             # (1, T, 22, 3)
        contact_state = batch["contact_state"].to(device)     # (1, T, 5)
        contact_target_xyz = batch["contact_target_xyz"].to(device)  # (1, T, 5, 3)
        obj_com = batch["obj_com_canonical"].to(device)       # (1, T, 3)
        obj_rot6d = batch["obj_rot6d_canonical"].to(device)   # (1, T, 6)
        obj_pos_world = batch["object_positions"].to(device)  # (1, T, 3)
        obj_rot_world = batch["object_rotations"].to(device)  # (1, T, 3) axis-angle
        seq_len = int(batch["seq_len"][0])

        # --- Canonical-frame variant ---
        joints_canon = lift_motion263_to_joints(motion)        # (1, T, 22, 3)
        target_canon = lift_object_local_to_canonical(
            contact_target_xyz, obj_com, obj_rot6d,
        )                                                       # (1, T, 5, 3)
        pred_canon = joints_canon.index_select(2, part_idx)    # (1, T, 5, 3)
        dist_canon = (pred_canon - target_canon).pow(2).sum(-1).sqrt().squeeze(0)

        # --- World-frame variant ---
        target_world = lift_object_local_to_world(
            contact_target_xyz, obj_pos_world, obj_rot_world,
        )                                                       # (1, T, 5, 3)
        pred_world = joints_world.index_select(2, part_idx)    # (1, T, 5, 3)
        dist_world = (pred_world - target_world).pow(2).sum(-1).sqrt().squeeze(0)

        cs = contact_state.squeeze(0)                           # (T, 5)

        clip_part_summary: dict[str, dict[str, float | None]] = {}
        for p_idx, part_name in enumerate(PART_NAMES):
            mask = (cs[:seq_len, p_idx] >= 0.5)
            if mask.sum() == 0:
                clip_part_summary[part_name] = {"canonical": None, "world": None}
                continue
            d_canon = dist_canon[:seq_len, p_idx][mask].cpu().numpy()
            d_world = dist_world[:seq_len, p_idx][mask].cpu().numpy()
            per_subset_part[("canonical", subset, part_name)].extend(d_canon.tolist())
            per_subset_part[("world", subset, part_name)].extend(d_world.tolist())
            overall_part[("canonical", part_name)].extend(d_canon.tolist())
            overall_part[("world", part_name)].extend(d_world.tolist())
            clip_part_summary[part_name] = {
                "canonical_mean_m": float(d_canon.mean()),
                "world_mean_m": float(d_world.mean()),
            }

        per_clip_summary.append({
            "seq_id": batch["seq_id"][0],
            "subset": subset,
            "seq_len": seq_len,
            "per_part": clip_part_summary,
        })

        n_done += 1
        if n_done % 50 == 0:
            print(f"  processed {n_done}/{len(triples)}")

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    def stats(arr: list[float]) -> dict:
        if not arr:
            return {"n": 0}
        a = np.array(arr)
        return {
            "n": int(a.size),
            "mean_m": float(a.mean()),
            "median_m": float(np.median(a)),
            "p90_m": float(np.percentile(a, 90)),
            "p99_m": float(np.percentile(a, 99)),
            "max_m": float(a.max()),
        }

    overall_summary: dict[str, dict[str, dict]] = {}
    for frame in ("canonical", "world"):
        overall_summary[frame] = {
            part: {
                **stats(overall_part[(frame, part)]),
                "v18_threshold_m": PART_THRESHOLD_M[part],
                "median_within_threshold": (
                    bool(np.median(overall_part[(frame, part)]) <= PART_THRESHOLD_M[part])
                    if overall_part[(frame, part)] else None
                ),
            }
            for part in PART_NAMES
        }

    per_subset_summary: dict[str, dict[str, dict[str, dict]]] = {}
    for (frame, subset, part), values in per_subset_part.items():
        per_subset_summary.setdefault(frame, {}).setdefault(subset, {})[part] = stats(values)

    summary = {
        "config": str(args.config),
        "num_clips": len(per_clip_summary),
        "overall_per_part": overall_summary,
        "per_subset_per_part": per_subset_summary,
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "per_clip.json").write_text(json.dumps(per_clip_summary, indent=2))

    def _print_table(frame: str) -> None:
        print(f"\n=== GT roundtrip floor — {frame} frame ===\n")
        print(f"{'part':>11} | {'n_frames':>9} | {'mean':>7} | {'median':>7} | "
              f"{'p90':>7} | {'p99':>7} | {'thresh':>6} | within?")
        print("-" * 80)
        for part in PART_NAMES:
            s = overall_summary[frame][part]
            if s["n"] == 0:
                print(f"{part:>11} |   no contact frames")
                continue
            print(
                f"{part:>11} | {s['n']:>9d} | "
                f"{s['mean_m']*100:>6.2f}cm | {s['median_m']*100:>6.2f}cm | "
                f"{s['p90_m']*100:>6.2f}cm | {s['p99_m']*100:>6.2f}cm | "
                f"{s['v18_threshold_m']*100:>5.1f}cm | "
                f"{'YES' if s['median_within_threshold'] else 'NO'}"
            )

    _print_table("canonical")
    _print_table("world")

    print("\n=== Per-subset median distance (canonical / world) ===\n")
    subsets = sorted(per_subset_summary.get("canonical", {}).keys())
    header = f"{'part':>11} | " + " | ".join(f"{s:>20}" for s in subsets)
    print(header)
    print("-" * len(header))
    for part in PART_NAMES:
        cells = []
        for s in subsets:
            c = per_subset_summary.get("canonical", {}).get(s, {}).get(part)
            w = per_subset_summary.get("world", {}).get(s, {}).get(part)
            if c is None or c.get("n", 0) == 0:
                cells.append("--")
            else:
                c_med = c["median_m"] * 100
                w_med = w["median_m"] * 100 if w else float("nan")
                cells.append(f"{c_med:5.1f} / {w_med:5.1f}")
        print(f"{part:>11} | " + " | ".join(f"{c:>20}" for c in cells))

    print(f"\nFull JSON: {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
