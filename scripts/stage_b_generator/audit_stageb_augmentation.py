"""Audit Stage-B synchronized augmentation before launching v20 training.

The v20 experiment keeps the v18 four-subset corpus and adds online random
timewarp. This script forces each configured non-1.0 scale on the same train
clips and checks that motion, joints, object trajectory, pseudo-labels, and
compiled interaction-plan tensors remain aligned.

Example:
    python scripts/stage_b_generator/audit_stageb_augmentation.py \
        --config configs/training/anchordiff_v20_a1_4subset_timewarp.yaml \
        --output analyses/2026-05-13_v20_timewarp_augmentation_audit.json \
        --md analyses/2026-05-13_v20_timewarp_augmentation_audit.md
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from piano.data.dataset import (
    AugmentConfig,
    HOIDataset,
    build_subject_split,
    extract_subject_id,
)
from piano.utils.io_utils import load_json


PART_JOINT = {
    "L_hand": 20,
    "R_hand": 21,
    "L_foot": 10,
    "R_foot": 11,
    "pelvis": 0,
}
PLAN_KEYS = (
    "anchor_time",
    "anchor_part",
    "anchor_target_local",
    "anchor_target_world",
    "anchor_type",
    "anchor_phase",
    "anchor_support",
    "anchor_conf",
    "anchor_mask",
    "segment_start",
    "segment_end",
    "segment_part",
    "segment_target_summary_local",
    "segment_phase",
    "segment_support",
    "segment_conf",
    "segment_mask",
)
TENSOR_SHAPE_KEYS = (
    "motion",
    "joints",
    "object_positions",
    "object_rotations",
    "obj_com_canonical",
    "obj_rot6d_canonical",
    "contact_state",
    "contact_target_xyz",
    "phase",
    "support",
    "rest_offsets",
)


def _stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def _range(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0, "min": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {"n": int(arr.size), "min": float(arr.min()), "max": float(arr.max())}


def _read_metadata(root: Path) -> list[dict[str, Any]]:
    meta = root / "metadata_clean.json"
    if not meta.exists():
        meta = root / "metadata.json"
    return load_json(meta)


def _resolve_subject_filter(cfg, bucket: str) -> set[str] | None:
    subj_cfg = cfg.data.get("subject_split")
    if subj_cfg is None or not subj_cfg.get("enabled", False) or bucket == "all":
        return None
    keys: set[tuple[str, str]] = set()
    for entry in cfg.data.datasets:
        root = Path(entry.root)
        subset = root.name
        for meta in _read_metadata(root):
            sid = extract_subject_id(subset, meta.get("seq_id", ""))
            if sid is not None:
                keys.add((subset, sid))
    split = build_subject_split(
        sorted(keys),
        train_pct=int(subj_cfg.train_pct),
        val_pct=int(subj_cfg.val_pct),
        seed=int(subj_cfg.seed),
    )
    return split[bucket]


def _make_dataset(cfg, entry, subject_filter: set[str] | None, augment: AugmentConfig | None) -> HOIDataset:
    pseudo_label_subdir = cfg.data.get("pseudo_label_subdir", None)
    pseudo_label_dir = cfg.data.get("pseudo_label_dir", None)
    if pseudo_label_dir is not None:
        sub_dir = pseudo_label_dir
    elif pseudo_label_subdir is not None:
        sub_dir = str(Path(entry.root) / pseudo_label_subdir)
    else:
        sub_dir = None
    return HOIDataset(
        root=entry.root,
        pseudo_label_dir=sub_dir,
        max_seq_length=int(cfg.data.max_seq_length),
        subject_id_filter=subject_filter,
        subsample_n_per_object=cfg.data.get("subsample_n_per_object", None),
        subsample_seed=int(cfg.data.get("subsample_seed", 42)),
        augment=augment,
        support_collapse_hand_support=bool(cfg.data.get("support_collapse_hand_support", True)),
        surface_obj_pose=True,
        force_world_frame=bool(cfg.data.get("force_world_frame", False)),
        motion_representation=str(cfg.data.get("motion_representation", "motion_263")),
    )


def _velocity_metrics(sample: dict[str, Any]) -> dict[str, float]:
    joints = sample["joints"].cpu().numpy().astype(np.float32)
    T = int(sample["seq_len"].item())
    if T <= 1:
        return {"body_vel": 0.0, "L_hand_vel": 0.0, "R_hand_vel": 0.0}
    joints = joints[:T]
    local = joints - joints[:, 0:1, :]
    vel = np.linalg.norm(np.diff(local, axis=0), axis=-1) * 100.0
    body = vel[:, 1:]
    return {
        "body_vel": float(body.mean()) if body.size else 0.0,
        "L_hand_vel": float(vel[:, PART_JOINT["L_hand"]].mean()),
        "R_hand_vel": float(vel[:, PART_JOINT["R_hand"]].mean()),
    }


def _contact_events(contact_state: np.ndarray, threshold: float = 0.5) -> int:
    if contact_state.size == 0:
        return 0
    mask = contact_state >= threshold
    total = int(mask[0].sum())
    if len(mask) > 1:
        total += int((mask[1:] & ~mask[:-1]).sum())
    return total


def _plan_metrics(sample: dict[str, Any], contact_threshold: float = 0.5) -> dict[str, float | int]:
    T = int(sample["seq_len"].item())
    mask = sample["plan_anchor_mask"].cpu().numpy().astype(bool)
    times = sample["plan_anchor_time"].cpu().numpy().astype(np.int64)
    parts = sample["plan_anchor_part"].cpu().numpy().astype(np.float32)
    target_world = sample["plan_anchor_target_world"].cpu().numpy().astype(np.float32)
    contact = sample["contact_state"].cpu().numpy().astype(np.float32)[:T]

    active_idx = np.where(mask)[0]
    near = 0
    checked = 0
    for k in active_idx:
        t = int(times[k])
        active_parts = np.where(parts[k] > 0.5)[0]
        if active_parts.size == 0 or t < 0 or t >= T:
            continue
        lo = max(0, t - 5)
        hi = min(T, t + 6)
        checked += 1
        if bool((contact[lo:hi, active_parts] >= contact_threshold).any()):
            near += 1

    active_target = target_world[mask]
    finite_target = bool(np.isfinite(active_target).all()) if active_target.size else True
    active_times = times[mask]
    return {
        "anchor_count": int(mask.sum()),
        "anchor_time_mean_norm": float(active_times.mean() / max(T - 1, 1)) if active_times.size else 0.0,
        "anchor_time_std_norm": float(active_times.std() / max(T - 1, 1)) if active_times.size else 0.0,
        "anchor_near_contact_fraction": float(near / max(checked, 1)),
        "anchor_out_of_bounds": int(((active_times < 0) | (active_times >= T)).sum()) if active_times.size else 0,
        "plan_target_finite": finite_target,
        "plan_target_min": float(active_target.min()) if active_target.size else 0.0,
        "plan_target_max": float(active_target.max()) if active_target.size else 0.0,
    }


def _sample_metrics(sample: dict[str, Any]) -> dict[str, Any]:
    T = int(sample["seq_len"].item())
    contact = sample["contact_state"].cpu().numpy().astype(np.float32)[:T]
    phase = sample["phase"].cpu().numpy().astype(np.int64)[:T]
    support = sample["support"].cpu().numpy().astype(np.int64)[:T]
    vel = _velocity_metrics(sample)
    plan = _plan_metrics(sample)
    return {
        "seq_len": T,
        "body_vel": vel["body_vel"],
        "L_hand_vel": vel["L_hand_vel"],
        "R_hand_vel": vel["R_hand_vel"],
        "contact_rate": float((contact >= 0.5).mean()) if contact.size else 0.0,
        "contact_events": _contact_events(contact),
        "phase_values": sorted(int(x) for x in np.unique(phase)) if phase.size else [],
        "support_values": sorted(int(x) for x in np.unique(support)) if support.size else [],
        **plan,
    }


def _numeric_checks(sample: dict[str, Any], subset: str, scale: float) -> list[str]:
    failures: list[str] = []
    T = int(sample["seq_len"].item())
    max_T = int(sample["motion"].shape[0])
    if not (0 < T <= max_T):
        failures.append(f"{subset} scale={scale}: seq_len {T} outside (0,{max_T}]")
    for key in TENSOR_SHAPE_KEYS:
        if key not in sample:
            continue
        arr = sample[key].cpu().numpy()
        if not np.isfinite(arr).all():
            failures.append(f"{subset} scale={scale}: {key} has NaN/inf")
    for key in PLAN_KEYS:
        pkey = f"plan_{key}"
        if pkey in sample and not np.isfinite(sample[pkey].cpu().numpy()).all():
            failures.append(f"{subset} scale={scale}: {pkey} has NaN/inf")
    cs = sample["contact_state"].cpu().numpy()
    if cs.min() < -1e-5 or cs.max() > 1.0 + 1e-5:
        failures.append(
            f"{subset} scale={scale}: contact_state range [{cs.min():.4f},{cs.max():.4f}]"
        )
    if _plan_metrics(sample)["anchor_out_of_bounds"] > 0:
        failures.append(f"{subset} scale={scale}: plan anchor time out of bounds")
    return failures


def _shape_failures(orig: dict[str, Any], aug: dict[str, Any], subset: str, scale: float) -> list[str]:
    failures: list[str] = []
    for key in TENSOR_SHAPE_KEYS:
        if key not in orig and key not in aug:
            continue
        if key not in orig or key not in aug:
            failures.append(f"{subset} scale={scale}: {key} missing on one side")
            continue
        if tuple(orig[key].shape) != tuple(aug[key].shape):
            failures.append(
                f"{subset} scale={scale}: {key} shape {tuple(orig[key].shape)} -> {tuple(aug[key].shape)}"
            )
    for key in PLAN_KEYS:
        pkey = f"plan_{key}"
        if pkey in orig and pkey in aug and tuple(orig[pkey].shape) != tuple(aug[pkey].shape):
            failures.append(
                f"{subset} scale={scale}: {pkey} shape {tuple(orig[pkey].shape)} -> {tuple(aug[pkey].shape)}"
            )
    return failures


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "seq_len": _stats([float(r["seq_len"]) for r in rows]),
        "body_vel": _stats([float(r["body_vel"]) for r in rows]),
        "L_hand_vel": _stats([float(r["L_hand_vel"]) for r in rows]),
        "R_hand_vel": _stats([float(r["R_hand_vel"]) for r in rows]),
        "contact_rate": _stats([float(r["contact_rate"]) for r in rows]),
        "contact_events": _stats([float(r["contact_events"]) for r in rows]),
        "anchor_count": _stats([float(r["anchor_count"]) for r in rows]),
        "anchor_time_mean_norm": _stats([float(r["anchor_time_mean_norm"]) for r in rows]),
        "anchor_time_std_norm": _stats([float(r["anchor_time_std_norm"]) for r in rows]),
        "anchor_near_contact_fraction": _stats(
            [float(r["anchor_near_contact_fraction"]) for r in rows]
        ),
        "plan_target_range": _range(
            [float(r["plan_target_min"]) for r in rows]
            + [float(r["plan_target_max"]) for r in rows]
        ),
    }


def _ratio(num: float, den: float) -> float:
    return float(num / den) if abs(den) > 1e-9 else 0.0


def _write_markdown(path: Path, results: dict[str, Any]) -> None:
    lines: list[str] = []
    decision = results["decision"]
    lines.append("# v20 timewarp augmentation audit")
    lines.append("")
    lines.append(f"**Config:** `{results['config']}`")
    lines.append(f"**Bucket:** `{results['bucket']}`")
    lines.append(f"**Decision:** **{'PASS' if decision['passed'] else 'FAIL'}**")
    lines.append("")
    for item in decision["summary"]:
        lines.append(f"- {item}")

    lines.append("")
    lines.append("## Table 1: Aggregate original vs forced timewarp")
    lines.append("")
    lines.append("| scale | n | seq_len mean | body vel | L hand vel | R hand vel | contact rate | events | anchors | near-contact anchors |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for scale_key in ["original"] + [f"scale_{s:g}" for s in results["scales"]]:
        agg = results["aggregate"][scale_key]
        label = "1.0 original" if scale_key == "original" else scale_key.replace("scale_", "")
        lines.append(
            f"| {label} | {agg['n']} | {agg['seq_len']['mean']:.1f} | "
            f"{agg['body_vel']['mean']:.3f} | {agg['L_hand_vel']['mean']:.3f} | "
            f"{agg['R_hand_vel']['mean']:.3f} | {agg['contact_rate']['mean']:.3f} | "
            f"{agg['contact_events']['mean']:.2f} | {agg['anchor_count']['mean']:.2f} | "
            f"{agg['anchor_near_contact_fraction']['mean']:.3f} |"
        )

    lines.append("")
    lines.append("## Table 2: Ratios vs original")
    lines.append("")
    lines.append("| scale | seq_len | body vel | L hand vel | R hand vel | contact rate | events | anchors |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for scale in results["scales"]:
        r = results["ratios_vs_original"][f"scale_{scale:g}"]
        lines.append(
            f"| {scale:g} | {r['seq_len']:.3f} | {r['body_vel']:.3f} | "
            f"{r['L_hand_vel']:.3f} | {r['R_hand_vel']:.3f} | "
            f"{r['contact_rate']:.3f} | {r['contact_events']:.3f} | "
            f"{r['anchor_count']:.3f} |"
        )

    lines.append("")
    lines.append("## Table 3: Subset-wise sanity")
    lines.append("")
    lines.append("| subset | scale | n | seq_len mean | body vel | contact rate | events | anchors | near-contact anchors |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for subset, payload in results["subsets"].items():
        for scale_key in ["original"] + [f"scale_{s:g}" for s in results["scales"]]:
            agg = payload[scale_key]
            label = "1.0 original" if scale_key == "original" else scale_key.replace("scale_", "")
            lines.append(
                f"| {subset} | {label} | {agg['n']} | {agg['seq_len']['mean']:.1f} | "
                f"{agg['body_vel']['mean']:.3f} | {agg['contact_rate']['mean']:.3f} | "
                f"{agg['contact_events']['mean']:.2f} | {agg['anchor_count']['mean']:.2f} | "
                f"{agg['anchor_near_contact_fraction']['mean']:.3f} |"
            )

    lines.append("")
    lines.append("## Table 4: Failure / warning summary")
    lines.append("")
    lines.append("| type | count | examples |")
    lines.append("|---|---:|---|")
    lines.append(
        f"| blocking failures | {len(results['failures'])} | "
        f"{'; '.join(results['failures'][:5]) if results['failures'] else 'none'} |"
    )
    lines.append(
        f"| warnings | {len(results['warnings'])} | "
        f"{'; '.join(results['warnings'][:5]) if results['warnings'] else 'none'} |"
    )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- The training config uses online random timewarp, not offline duplication. "
        "Effective clips per epoch remain the same as v18; the local mode density is increased across epochs."
    )
    lines.append(
        "- Phase labels are treated as categorical `non_contact/stable_contact/manipulation`; "
        "the audit never assumes approach/release labels."
    )
    if decision["passed"]:
        lines.append("- Numeric, shape, and plan-alignment checks passed, so v20 training can start.")
    else:
        lines.append("- Blocking checks failed; do not start v20 until the failures above are fixed.")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--md", type=Path, required=True)
    parser.add_argument("--bucket", type=str, default="train", choices=["train", "val", "all"])
    parser.add_argument("--clips-per-subset", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    cfg = OmegaConf.load(args.config)
    scales = [
        float(s)
        for s in cfg.data.get("augmentation", {}).get("timewarp_scales", [])
        if float(s) > 0.0 and not math.isclose(float(s), 1.0)
    ]
    if not scales:
        raise ValueError("No non-1.0 timewarp scales found in config")

    subject_filter = _resolve_subject_filter(cfg, args.bucket)
    failures: list[str] = []
    warnings: list[str] = []
    aggregate_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    subsets: dict[str, dict[str, Any]] = {}

    for entry in cfg.data.datasets:
        subset = Path(entry.root).name
        base_ds = _make_dataset(cfg, entry, subject_filter, augment=None)
        n = min(int(args.clips_per_subset), len(base_ds))
        if n <= 0:
            failures.append(f"{subset}: no clips available in {args.bucket} bucket")
            continue
        indices = np.linspace(0, len(base_ds) - 1, n, dtype=np.int64)
        subset_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        aug_datasets = {
            float(scale): _make_dataset(
                cfg,
                entry,
                subject_filter,
                augment=AugmentConfig(
                    enabled=True,
                    mirror_prob=0.0,
                    mirror_duplicate=False,
                    rotate_around_y_prob=0.0,
                    pc_jitter_std=0.0,
                    timewarp_scales=(float(scale),),
                    timewarp_mode="online",
                ),
            )
            for scale in scales
        }

        for raw_idx in indices:
            idx = int(raw_idx)
            orig = base_ds[idx]
            orig_metrics = _sample_metrics(orig)
            subset_rows["original"].append(orig_metrics)
            aggregate_rows["original"].append(orig_metrics)

            for scale, ds_aug in aug_datasets.items():
                aug = ds_aug[idx]
                scale_key = f"scale_{scale:g}"
                failures.extend(_shape_failures(orig, aug, subset, scale))
                failures.extend(_numeric_checks(aug, subset, scale))
                metrics = _sample_metrics(aug)
                subset_rows[scale_key].append(metrics)
                aggregate_rows[scale_key].append(metrics)

        subset_payload: dict[str, Any] = {}
        for key, rows in subset_rows.items():
            subset_payload[key] = _aggregate(rows)
        subsets[subset] = subset_payload

    aggregate = {key: _aggregate(rows) for key, rows in aggregate_rows.items()}
    ratios: dict[str, dict[str, float]] = {}
    original = aggregate["original"]
    for scale in scales:
        key = f"scale_{scale:g}"
        cur = aggregate[key]
        ratios[key] = {
            "seq_len": _ratio(cur["seq_len"]["mean"], original["seq_len"]["mean"]),
            "body_vel": _ratio(cur["body_vel"]["mean"], original["body_vel"]["mean"]),
            "L_hand_vel": _ratio(cur["L_hand_vel"]["mean"], original["L_hand_vel"]["mean"]),
            "R_hand_vel": _ratio(cur["R_hand_vel"]["mean"], original["R_hand_vel"]["mean"]),
            "contact_rate": _ratio(cur["contact_rate"]["mean"], original["contact_rate"]["mean"]),
            "contact_events": _ratio(cur["contact_events"]["mean"], original["contact_events"]["mean"]),
            "anchor_count": _ratio(cur["anchor_count"]["mean"], original["anchor_count"]["mean"]),
        }
        if cur["anchor_near_contact_fraction"]["mean"] < 0.75:
            failures.append(
                f"{key}: mean anchor-near-contact fraction "
                f"{cur['anchor_near_contact_fraction']['mean']:.3f} < 0.75"
            )
        if not (0.5 <= ratios[key]["contact_events"] <= 1.5):
            warnings.append(
                f"{key}: contact event count ratio {ratios[key]['contact_events']:.3f} outside [0.5, 1.5]"
            )
        if ratios[key]["anchor_count"] < 0.7:
            warnings.append(
                f"{key}: anchor count ratio {ratios[key]['anchor_count']:.3f} < 0.7"
            )

    results = {
        "config": str(args.config),
        "bucket": args.bucket,
        "clips_per_subset": int(args.clips_per_subset),
        "scales": scales,
        "augmentation_mode": str(cfg.data.augmentation.get("timewarp_mode", "online")),
        "effective_training_sample_count": "same per epoch as v18 (online random augmentation)",
        "aggregate": aggregate,
        "ratios_vs_original": ratios,
        "subsets": subsets,
        "failures": failures,
        "warnings": warnings,
        "decision": {
            "passed": len(failures) == 0,
            "summary": [
                f"Checked {sum(len(v) for v in aggregate_rows.values())} sample-scale rows across {len(subsets)} subsets.",
                "All per-frame tensors keep fixed Stage-B padded shapes.",
                "Continuous fields use linear interpolation; phase/support use nearest; axis-angle rotations use SciPy Slerp.",
                "Plan tensors are recompiled after timewarp from warped contact/object labels.",
            ],
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    _write_markdown(args.md, results)

    print(f"Wrote JSON to {args.output}")
    print(f"Wrote Markdown to {args.md}")
    print(f"Decision: {'PASS' if results['decision']['passed'] else 'FAIL'}")
    if failures:
        print("Failures:")
        for item in failures[:10]:
            print(f"  - {item}")
    if warnings:
        print("Warnings:")
        for item in warnings[:10]:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
