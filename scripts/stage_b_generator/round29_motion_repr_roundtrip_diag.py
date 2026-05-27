"""Motion representation round-trip floor diagnostic.

Per analyses/2026-05-28_round29_next_step_ablation_execution_prompt_for_claude_code.md §3.

Measures whether the current ``smpl_pose_135_plan`` representation +
``_fk_22joints`` reconstruction already introduces hand/foot/pelvis error
comparable to the model's contact drift. The existing diagnostic
``--use-gt-as-pred`` mode uses FK(motion) as BOTH pred and GT and
therefore HIDES the representation floor — because the model is operating
on the same FK-reconstructed signal, any residual gap between FK(motion)
and raw ``joints_22`` from the source NPZ is a hard floor on what the
model can reach.

Workflow:

  1. For each clip in the 48-clip selection JSON:
     - Load the dataset batch (with the same condition + normalization
       as a real training run).
     - Compute ``fk_joints = _fk_22joints(batch["motion"], batch["rest_offsets"])``.
     - Independently load the raw NPZ: ``motion_data["joints_22"]``.
     - Compute per-joint Euclidean error in cm between FK and raw.
     - On the same contact segments the model is evaluated on, compute
       sustained-contact-style drift metrics with FK as "pred" and raw as
       "GT". This is the contact floor.
  2. Emit two diag dirs (one per bucket: train + val):
     - ``analyses/round29_repr_floor_<bucket>/repr_floor_stats.json``
     - ``analyses/round29_repr_floor_<bucket>/repr_floor_summary.md``

Interpretation thresholds (per prompt §3):

  - If hand/wrist mean drift floor is comparable to current model hand
    drift (~10-13 cm), motion representation is on the critical path.
  - If hand mean < 2 cm and p95 < 5 cm, representation is not the
    bottleneck; capacity/objective become more likely.

This script does NOT load a checkpoint. It uses no model. Inputs:
``--config`` (for dataset construction + selection of subsets/data root),
``--selection-json``, ``--bucket``, ``--output-dir``.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# SMPL-22 indices we surface in tables.
SMPL_JOINT_NAMES: tuple[str, ...] = (
    "pelvis", "left_hip", "right_hip", "spine1",
    "left_knee", "right_knee", "spine2", "left_ankle",
    "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
)
assert len(SMPL_JOINT_NAMES) == 22

KEY_JOINTS_FOR_SUMMARY: tuple[str, ...] = (
    "pelvis", "left_wrist", "right_wrist", "left_ankle", "right_ankle",
    "left_foot", "right_foot", "neck",
)

# Per-part to joint-index for sustained-contact-style floor (matches
# round26_sustained_contact_diag conventions).
PART_TO_JOINT_IDX: dict[str, int] = {
    "left_hand": 20,    # left_wrist
    "right_hand": 21,
    "left_foot": 7,     # using ankle as the contact "foot" reference
    "right_foot": 8,
    "pelvis": 0,
}


@dataclass(slots=True)
class PerClipRow:
    subset: str
    seq_id: str
    n_frames: int
    # Per-joint mean Euclidean error (cm), shape (22,) — list for JSON.
    per_joint_mean_cm: list[float]
    per_joint_p95_cm: list[float]
    # Per-part sustained-contact-style floor (FK vs raw, on contact frames).
    per_part_floor_cm: dict[str, dict[str, float]]


def _percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, q))


def _per_joint_error_cm(
    fk: np.ndarray,           # (T, 22, 3) world frame, m
    raw: np.ndarray,           # (T, 22, 3) world frame, m
) -> np.ndarray:
    """Per-frame per-joint Euclidean error in cm. Shape (T, 22)."""
    T = min(fk.shape[0], raw.shape[0])
    diff = fk[:T] - raw[:T]                                     # (T, 22, 3)
    err_m = np.linalg.norm(diff, axis=-1)                       # (T, 22)
    return err_m * 100.0                                        # cm


def _per_part_contact_floor(
    fk: np.ndarray,            # (T, 22, 3)
    raw: np.ndarray,           # (T, 22, 3)
    contact_state: np.ndarray, # (T, 5) prob in [0, 1]
    contact_threshold: float = 0.5,
) -> dict[str, dict[str, float]]:
    """For each contact part, compute mean / p95 / max Euclidean error
    between FK and raw on the frames where `contact_state > threshold`.
    Returns empty stats for parts with no active contact frames.
    """
    out: dict[str, dict[str, float]] = {}
    T = min(fk.shape[0], raw.shape[0], contact_state.shape[0])
    # contact_state columns match the 5-part ordering used by the dataset
    # (left_hand, right_hand, left_foot, right_foot, pelvis).
    part_to_col = {
        "left_hand": 0, "right_hand": 1,
        "left_foot": 2, "right_foot": 3, "pelvis": 4,
    }
    for part_name, j_idx in PART_TO_JOINT_IDX.items():
        col = part_to_col[part_name]
        mask = (contact_state[:T, col] > contact_threshold)
        if mask.sum() < 2:
            out[part_name] = {
                "n_frames": int(mask.sum()),
                "mean_cm": float("nan"),
                "p95_cm": float("nan"),
                "max_cm": float("nan"),
            }
            continue
        err_m = np.linalg.norm(fk[:T, j_idx] - raw[:T, j_idx], axis=-1)
        err_cm = (err_m * 100.0)[mask]
        out[part_name] = {
            "n_frames": int(mask.sum()),
            "mean_cm": float(err_cm.mean()),
            "p95_cm": _percentile(err_cm, 95),
            "max_cm": float(err_cm.max()),
        }
    return out


def _load_raw_joints_22(
    dataset_root: Path, seq_id: str, valid_T: int,
) -> np.ndarray | None:
    """Load raw ``joints_22`` from the source NPZ, mirroring
    HOIDataset.__getitem__ at src/piano/data/dataset.py:414-417.
    """
    npz_path = dataset_root / "motions" / f"{seq_id}.npz"
    if not npz_path.exists():
        return None
    motion_data = np.load(npz_path, allow_pickle=True)
    if "joints_22" not in motion_data.files:
        return None
    return motion_data["joints_22"].astype(np.float32)[:valid_T]


def _per_subset_dataset_roots(cfg: Any) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for ds in cfg.data.datasets:
        out[str(ds.name)] = Path(str(ds.root))
    return out


def _run_clip(
    batch: "dict[str, Any]",   # values are torch.Tensor in the real path
    subset_roots: dict[str, Path],
) -> PerClipRow | None:
    """Compute FK-vs-raw stats for a single clip's batch. Returns None
    when the raw NPZ is unavailable."""
    subset = batch["subset"][0]
    seq_id = batch["seq_id"][0]
    seq_mask = batch["seq_mask"][0].bool()
    valid_T = int(seq_mask.sum().item())
    if valid_T < 2:
        return None

    # Deferred imports — keep helpers importable without torch in tests.
    import torch
    from piano.inference.diagnostic_helpers import _fk_22joints

    motion = batch["motion"][:, :valid_T].float()                # (1, T, 135)
    rest_offsets = batch["rest_offsets"].float()                 # (1, 22, 3)
    with torch.no_grad():
        fk_joints = _fk_22joints(motion, rest_offsets)            # (1, T, 22, 3)
    fk_np = fk_joints[0].cpu().numpy()                            # (T, 22, 3)

    ds_root = subset_roots.get(subset)
    if ds_root is None:
        return None
    raw_np = _load_raw_joints_22(ds_root, seq_id, valid_T)
    if raw_np is None or raw_np.shape[0] < 2:
        return None

    T_eff = min(fk_np.shape[0], raw_np.shape[0])
    err_cm = _per_joint_error_cm(fk_np[:T_eff], raw_np[:T_eff])  # (T, 22)
    per_joint_mean = err_cm.mean(axis=0).tolist()
    per_joint_p95 = [
        _percentile(err_cm[:, j], 95) for j in range(22)
    ]

    contact_state = batch["contact_state"][0, :T_eff].cpu().numpy()  # (T, 5)
    per_part_floor = _per_part_contact_floor(
        fk_np[:T_eff], raw_np[:T_eff], contact_state,
    )
    return PerClipRow(
        subset=subset, seq_id=seq_id, n_frames=T_eff,
        per_joint_mean_cm=per_joint_mean,
        per_joint_p95_cm=per_joint_p95,
        per_part_floor_cm=per_part_floor,
    )


def _aggregate(rows: list[PerClipRow]) -> dict[str, Any]:
    """Aggregate per-clip rows into per-joint + per-part summary."""
    if not rows:
        return {
            "n_clips": 0,
            "per_joint": {n: {"mean_cm": None, "p95_cm": None}
                          for n in SMPL_JOINT_NAMES},
            "per_part_contact_floor": {p: {"mean_cm": None, "p95_cm": None,
                                            "max_cm": None, "n_frames": 0}
                                       for p in PART_TO_JOINT_IDX},
        }
    # Per-joint: average of per-clip mean, max of per-clip p95.
    n_clips = len(rows)
    joint_means = np.stack([np.array(r.per_joint_mean_cm) for r in rows])      # (N, 22)
    joint_p95s = np.stack([np.array(r.per_joint_p95_cm) for r in rows])         # (N, 22)
    per_joint = {}
    for j, name in enumerate(SMPL_JOINT_NAMES):
        per_joint[name] = {
            "mean_cm": float(joint_means[:, j].mean()),
            "p95_cm": float(np.percentile(joint_p95s[:, j], 95)),
            "max_cm": float(joint_p95s[:, j].max()),
        }
    # Per-part contact floor: aggregate across clips weighted by n_frames.
    per_part: dict[str, dict[str, float]] = {}
    for part_name in PART_TO_JOINT_IDX:
        n_frames = 0
        sum_w_mean = 0.0
        sum_w_p95 = 0.0
        max_cm = 0.0
        for r in rows:
            slot = r.per_part_floor_cm.get(part_name, {})
            nf = int(slot.get("n_frames", 0))
            if nf == 0 or not math.isfinite(slot.get("mean_cm", float("nan"))):
                continue
            n_frames += nf
            sum_w_mean += nf * slot["mean_cm"]
            sum_w_p95 += nf * slot["p95_cm"]
            max_cm = max(max_cm, slot.get("max_cm", 0.0))
        if n_frames == 0:
            per_part[part_name] = {
                "n_frames": 0, "mean_cm": None, "p95_cm": None, "max_cm": None,
            }
        else:
            per_part[part_name] = {
                "n_frames": n_frames,
                "mean_cm": sum_w_mean / n_frames,
                "p95_cm": sum_w_p95 / n_frames,
                "max_cm": max_cm,
            }
    return {
        "n_clips": n_clips,
        "per_joint": per_joint,
        "per_part_contact_floor": per_part,
    }


def _interpretation(agg: dict[str, Any]) -> dict[str, Any]:
    """Decide whether the representation floor is on the critical path.

    Per prompt §3:
      - hand floor mean comparable to current model hand drift (~10-13 cm)
        → representation is critical.
      - hand mean < 2 cm AND p95 < 5 cm → representation is NOT the
        bottleneck.
    """
    per_joint = agg.get("per_joint", {})
    lw = per_joint.get("left_wrist", {})
    rw = per_joint.get("right_wrist", {})

    def _safe(x: Any) -> float | None:
        try:
            f = float(x)
        except (TypeError, ValueError):
            return None
        return f if math.isfinite(f) else None

    lw_mean = _safe(lw.get("mean_cm"))
    rw_mean = _safe(rw.get("mean_cm"))
    lw_p95 = _safe(lw.get("p95_cm"))
    rw_p95 = _safe(rw.get("p95_cm"))

    if None in (lw_mean, rw_mean, lw_p95, rw_p95):
        return {
            "verdict": "unknown",
            "reason": "missing per-joint left/right wrist floor stats",
            "hand_mean_cm": None,
            "hand_p95_cm": None,
        }
    hand_mean = max(lw_mean, rw_mean)
    hand_p95 = max(lw_p95, rw_p95)
    if hand_mean < 2.0 and hand_p95 < 5.0:
        verdict = "representation_floor_low"
        reason = (
            f"hand mean floor < 2 cm ({hand_mean:.2f}) and p95 < 5 cm "
            f"({hand_p95:.2f}); representation is NOT the contact bottleneck"
        )
    elif hand_mean > 5.0 or hand_p95 > 10.0:
        verdict = "representation_floor_critical"
        reason = (
            f"hand mean floor {hand_mean:.2f} cm and p95 {hand_p95:.2f} cm "
            f"are comparable to current model hand drift (~10-13 cm); "
            f"motion representation is on the critical path"
        )
    else:
        verdict = "representation_floor_borderline"
        reason = (
            f"hand mean {hand_mean:.2f} cm / p95 {hand_p95:.2f} cm are in "
            f"between thresholds; representation may contribute but is "
            f"unlikely the dominant bottleneck"
        )
    return {
        "verdict": verdict,
        "reason": reason,
        "hand_mean_cm": hand_mean,
        "hand_p95_cm": hand_p95,
    }


def _write_summary_md(
    out_path: Path,
    agg: dict[str, Any],
    interp: dict[str, Any],
    bucket: str,
) -> None:
    L: list[str] = []
    L.append(f"# Motion representation round-trip floor — `{bucket}`")
    L.append("")
    L.append(
        "FK(motion_135) vs raw ``joints_22`` from source NPZ. This measures "
        "the hard floor of the SMPL-pose-135 representation + FK "
        "reconstruction, BEFORE any model error."
    )
    L.append("")
    L.append(f"**Clips:** {agg['n_clips']}")
    L.append("")
    L.append(f"**Verdict:** **{interp['verdict']}**")
    L.append("")
    L.append(f"{interp['reason']}")
    L.append("")
    L.append("## Per-joint floor (selected key joints)")
    L.append("")
    L.append("| joint | mean (cm) | p95 (cm) | max (cm) |")
    L.append("| --- | ---: | ---: | ---: |")
    per_joint = agg.get("per_joint", {})
    for name in KEY_JOINTS_FOR_SUMMARY:
        if name not in per_joint:
            continue
        s = per_joint[name]
        mean = s.get("mean_cm"); p95 = s.get("p95_cm"); mx = s.get("max_cm")
        L.append(
            f"| {name} | "
            f"{('-' if mean is None else f'{mean:.2f}')} | "
            f"{('-' if p95 is None else f'{p95:.2f}')} | "
            f"{('-' if mx is None else f'{mx:.2f}')} |"
        )
    L.append("")
    L.append("## Per-part sustained-contact floor (FK vs raw on contact frames)")
    L.append("")
    L.append("| part | n_frames | mean (cm) | p95 (cm) | max (cm) |")
    L.append("| --- | ---: | ---: | ---: | ---: |")
    per_part = agg.get("per_part_contact_floor", {})
    for name in PART_TO_JOINT_IDX:
        s = per_part.get(name, {})
        nf = s.get("n_frames", 0)
        mean = s.get("mean_cm"); p95 = s.get("p95_cm"); mx = s.get("max_cm")
        L.append(
            f"| {name} | {int(nf or 0)} | "
            f"{('-' if mean is None else f'{mean:.2f}')} | "
            f"{('-' if p95 is None else f'{p95:.2f}')} | "
            f"{('-' if mx is None else f'{mx:.2f}')} |"
        )
    L.append("")
    L.append("## All 22 joints — mean Euclidean error (cm)")
    L.append("")
    L.append("| joint | mean (cm) | p95 (cm) |")
    L.append("| --- | ---: | ---: |")
    for name in SMPL_JOINT_NAMES:
        if name not in per_joint:
            continue
        s = per_joint[name]
        mean = s.get("mean_cm"); p95 = s.get("p95_cm")
        L.append(
            f"| {name} | "
            f"{('-' if mean is None else f'{mean:.2f}')} | "
            f"{('-' if p95 is None else f'{p95:.2f}')} |"
        )
    out_path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Motion representation round-trip floor diagnostic. Measures "
            "FK(motion_135) vs raw joints_22 to estimate the representational "
            "floor on hand/foot/pelvis accuracy."
        ),
    )
    parser.add_argument("--config", type=Path, required=True,
                        help="A R29 ablation training config — used for "
                             "dataset construction (data roots, normalization).")
    parser.add_argument("--selection-json", type=Path, required=True,
                        help="48-clip selection JSON (same as the model diag).")
    parser.add_argument("--bucket", default="val", choices=["train", "val"])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    # Heavy imports deferred so the pure helpers above stay importable
    # without torch / omegaconf in unit tests.
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from piano.data.dataset import collate_hoi
    from piano.inference.diagnostic_helpers import _build_dataset

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load(args.config)

    # Selection.
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
    print(f"[repr_floor] selection: {len(sel_pairs)} clips, bucket={args.bucket}")

    # Subset roots from the config (raw NPZ live under <root>/motions/<seq>.npz).
    subset_roots = _per_subset_dataset_roots(cfg)

    # Dataset.
    dataset = _build_dataset(cfg, args.bucket, augment=False)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        collate_fn=collate_hoi, num_workers=0,
    )

    rows: list[PerClipRow] = []
    matched = 0
    for batch in loader:
        if not batch.get("subset") or not batch.get("seq_id"):
            continue
        pair = (batch["subset"][0], batch["seq_id"][0])
        if pair not in sel_pairs:
            continue
        matched += 1
        try:
            row = _run_clip(batch, subset_roots)
        except Exception as exc:  # noqa: BLE001
            print(f"[repr_floor] skip {pair}: {exc}")
            continue
        if row is not None:
            rows.append(row)

    print(f"[repr_floor] matched {matched} selection clips, kept {len(rows)} with raw joints_22.")
    if not rows:
        raise SystemExit(
            "no clips produced FK-vs-raw rows; check that the dataset "
            "root contains motions/<seq_id>.npz with joints_22."
        )

    agg = _aggregate(rows)
    interp = _interpretation(agg)

    stats = {
        "config": str(args.config),
        "bucket": args.bucket,
        "selection_json": str(args.selection_json),
        "n_clips_matched": matched,
        "n_clips_kept": len(rows),
        "aggregate": agg,
        "interpretation": interp,
        "rows": [
            {
                "subset": r.subset, "seq_id": r.seq_id, "n_frames": r.n_frames,
                "per_joint_mean_cm": r.per_joint_mean_cm,
                "per_joint_p95_cm": r.per_joint_p95_cm,
                "per_part_floor_cm": r.per_part_floor_cm,
            }
            for r in rows
        ],
    }
    out_json = args.output_dir / "repr_floor_stats.json"
    out_md = args.output_dir / "repr_floor_summary.md"
    out_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    _write_summary_md(out_md, agg, interp, args.bucket)
    print(f"[repr_floor] wrote {out_json}")
    print(f"[repr_floor] wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
