#!/usr/bin/env python
"""Per-frame diagnostics for generated body/object contact failures.

This script compares a generated Stage B condition against the source clip:

- original body-to-object distance from the processed dataset,
- generated body-to-object distance after the saved canonical->world lift,
- pseudo-label contact coverage and object speed,
- generated part-to-target error in object-local coordinates.

It is intentionally diagnostic-only: it reads an existing generated.npz and
writes CSV/JSON/PNG reports under the requested output directory.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

import piano.models.backbones.momask_adapter  # noqa: F401
from piano.data.pseudo_labels.extract_contact import ContactConfig, _kinematic_contact_score
from piano.training.contact_eval import (
    _lift_canonical_to_world,
    _per_frame_body_to_object_distance,
)
from piano.training.decoded_contact_loss import body_canonical_to_object_local_torch
from piano.utils.io_utils import ensure_dir, save_json
from piano.utils.smpl_utils import BODY_PART_INDICES, BODY_PART_NAMES
from utils.motion_process import recover_from_ric

# qual_eval.py is in the same directory as this script.
from qual_eval import _build_val_dataset  # type: ignore


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _round(x: float | None, ndigits: int = 4) -> float | None:
    if x is None or not np.isfinite(float(x)):
        return None
    return round(float(x), ndigits)


def _mean(values: np.ndarray, mask: np.ndarray | None = None) -> float | None:
    if mask is not None:
        values = values[mask]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(values.mean())


def _object_motion_speed(
    object_positions: np.ndarray,
    object_rotations: np.ndarray,
    cfg: ContactConfig,
) -> np.ndarray:
    speed = np.zeros(len(object_positions), dtype=np.float32)
    if len(object_positions) <= 1:
        return speed
    trans = np.linalg.norm(np.diff(object_positions, axis=0), axis=-1) * float(cfg.fps)
    rot = np.linalg.norm(np.diff(object_rotations, axis=0), axis=-1) * float(cfg.fps)
    speed[1:] = trans + float(cfg.kin_radius_proxy) * rot
    return speed


def _dataset_index_by_seq(generated_dir: Path) -> dict[str, int]:
    parent_summary = generated_dir.parent / "summary.json"
    if parent_summary.exists():
        data = _load_json(parent_summary)
    else:
        diagnostic_summary = generated_dir.parent / "diagnostic_summary.json"
        if not diagnostic_summary.exists():
            return {}
        data = _load_json(diagnostic_summary)
    out: dict[str, int] = {}
    for row in data.get("clip_selection", []):
        seq_id = str(row.get("seq_id", ""))
        if not seq_id:
            continue
        out[seq_id] = int(row["index"])
    return out


def _part_names(mask_row: np.ndarray) -> str:
    return "|".join(name for name, active in zip(BODY_PART_NAMES, mask_row) if active)


def _segment_rows(
    min_dist_gen: np.ndarray,
    min_dist_orig: np.ndarray,
    target_error: np.ndarray,
    contact_any: np.ndarray,
    moving: np.ndarray,
    close_any: np.ndarray,
    coupled_any: np.ndarray,
) -> list[dict[str, Any]]:
    T = len(min_dist_gen)
    cuts = [0, T // 3, (2 * T) // 3, T]
    names = ["early", "middle", "late"]
    rows: list[dict[str, Any]] = []
    for name, start, end in zip(names, cuts[:-1], cuts[1:]):
        sl = slice(start, end)
        seg_moving = moving[sl]
        rows.append({
            "segment": name,
            "frame_start": int(start),
            "frame_end_exclusive": int(end),
            "generated_mean_min_dist_cm": _round(_mean(min_dist_gen[sl]) * 100.0, 2),
            "original_mean_min_dist_cm": _round(_mean(min_dist_orig[sl]) * 100.0, 2),
            "generated_target_error_cm": _round(_mean(target_error[sl]) * 100.0, 2),
            "label_contact_frac": _round(float(contact_any[sl].mean())),
            "generated_close_frac": _round(float(close_any[sl].mean())),
            "moving_frame_frac": _round(float(seg_moving.mean())),
            "moving_coupled_frac": (
                _round(float(coupled_any[sl][seg_moving].mean()))
                if int(seg_moving.sum()) > 0
                else None
            ),
        })
    return rows


def _write_plot(
    out_path: Path,
    *,
    seq_id: str,
    min_dist_gen: np.ndarray,
    min_dist_orig: np.ndarray,
    target_error: np.ndarray,
    object_speed: np.ndarray,
    contact_any: np.ndarray,
    close_any: np.ndarray,
    coupled_any: np.ndarray,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional diagnostic nicety
        print(f"matplotlib unavailable; skipping plot for {seq_id}: {exc}")
        return

    frames = np.arange(len(min_dist_gen))
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(frames, min_dist_gen * 100.0, label="generated min distance", color="#d62728")
    axes[0].plot(frames, min_dist_orig * 100.0, label="original min distance", color="#1f77b4")
    axes[0].set_ylabel("cm")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.25)

    axes[1].plot(frames, target_error * 100.0, color="#9467bd", label="generated active-part target error")
    axes[1].fill_between(
        frames,
        0,
        np.nanmax(target_error * 100.0) if np.isfinite(target_error).any() else 1.0,
        where=contact_any,
        color="#ffbb78",
        alpha=0.25,
        label="pseudo-label contact",
    )
    axes[1].set_ylabel("cm")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.25)

    axes[2].plot(frames, object_speed, color="#2ca02c", label="object speed")
    axes[2].fill_between(frames, 0, max(float(object_speed.max()), 1e-3), where=close_any, color="#d62728", alpha=0.16, label="generated close")
    axes[2].fill_between(frames, 0, max(float(object_speed.max()), 1e-3), where=coupled_any, color="#1f77b4", alpha=0.20, label="generated coupled")
    axes[2].set_ylabel("m/s")
    axes[2].set_xlabel("frame")
    axes[2].legend(loc="upper right")
    axes[2].grid(alpha=0.25)

    fig.suptitle(seq_id)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def diagnose_seq(
    *,
    seq_id: str,
    dataset_index: int,
    generated_index: int,
    dataset: Any,
    generated: dict[str, np.ndarray],
    output_dir: Path,
    fps: float,
    contact_threshold: float,
    moving_speed_threshold: float,
    coupling_threshold: float,
) -> dict[str, Any]:
    sample = dataset[dataset_index]
    seq_len = int(sample["seq_len"].item())
    T = min(seq_len, int(generated["motion_263"].shape[1]))

    motion_gen = generated["motion_263"][generated_index, :T].astype(np.float32)
    motion_t = torch.from_numpy(motion_gen).float().unsqueeze(0)
    canon_gen = recover_from_ric(motion_t, 22).squeeze(0).cpu().numpy().astype(np.float32)
    world_gen = _lift_canonical_to_world(
        canon_gen,
        float(generated["world_R_y_angle"][generated_index]),
        generated["world_T_xz"][generated_index],
    )

    object_pc = sample["object_pc"].cpu().numpy().astype(np.float32)
    object_positions = sample["object_positions"].cpu().numpy().astype(np.float32)[:T]
    object_rotations = sample["object_rotations"].cpu().numpy().astype(np.float32)[:T]
    orig_joints = sample["joints"].cpu().numpy().astype(np.float32)[:T]

    d_gen = _per_frame_body_to_object_distance(
        world_gen[:, BODY_PART_INDICES, :],
        object_pc,
        object_positions,
        object_rotations,
    )
    d_orig = _per_frame_body_to_object_distance(
        orig_joints[:, BODY_PART_INDICES, :],
        object_pc,
        object_positions,
        object_rotations,
    )
    min_dist_gen = d_gen.min(axis=1)
    min_dist_orig = d_orig.min(axis=1)
    closest_part_gen = d_gen.argmin(axis=1)
    closest_part_orig = d_orig.argmin(axis=1)

    cfg = ContactConfig(fps=float(fps))
    close_thresholds = np.array(
        [cfg.distance_thresholds[name] for name in BODY_PART_NAMES],
        dtype=np.float32,
    )
    close_any = (d_gen <= close_thresholds[None, :]).any(axis=1)
    kin_scores = np.stack([
        _kinematic_contact_score(
            world_gen[:, BODY_PART_INDICES[p], :],
            object_positions,
            object_rotations,
            cfg,
        )
        for p in range(len(BODY_PART_NAMES))
    ], axis=1)
    coupled_any = kin_scores.max(axis=1) >= float(coupling_threshold)
    object_speed = _object_motion_speed(object_positions, object_rotations, cfg)
    moving = object_speed >= float(moving_speed_threshold)

    contact_state = sample["contact_state"].cpu().numpy().astype(np.float32)[:T]
    contact_mask = contact_state >= float(contact_threshold)
    contact_any = contact_mask.any(axis=1)

    contact_target = sample["contact_target_xyz"].cpu().numpy().astype(np.float32)[:T]
    obj_com = sample["obj_com_canonical"][:T].unsqueeze(0).float()
    obj_rot6d = sample["obj_rot6d_canonical"][:T].unsqueeze(0).float()
    body_canon = torch.from_numpy(canon_gen[:, BODY_PART_INDICES, :]).float().unsqueeze(0)
    body_local = (
        body_canonical_to_object_local_torch(body_canon, obj_com, obj_rot6d)
        .squeeze(0)
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    target_dist = np.linalg.norm(body_local - contact_target, axis=-1)
    target_error = np.full(T, np.nan, dtype=np.float32)
    for t in range(T):
        if contact_mask[t].any():
            weights = np.clip(contact_state[t], 0.0, 1.0) * contact_mask[t].astype(np.float32)
            denom = float(weights.sum())
            if denom > 1e-6:
                target_error[t] = float((target_dist[t] * weights).sum() / denom)

    csv_path = output_dir / f"{seq_id}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame",
            "generated_min_dist_m",
            "original_min_dist_m",
            "generated_target_error_m",
            "object_speed_mps",
            "moving",
            "label_contact_any",
            "label_contact_parts",
            "generated_close_any",
            "generated_coupled_any",
            "generated_closest_part",
            "original_closest_part",
        ])
        for t in range(T):
            writer.writerow([
                t,
                f"{float(min_dist_gen[t]):.6f}",
                f"{float(min_dist_orig[t]):.6f}",
                "" if not np.isfinite(target_error[t]) else f"{float(target_error[t]):.6f}",
                f"{float(object_speed[t]):.6f}",
                int(bool(moving[t])),
                int(bool(contact_any[t])),
                _part_names(contact_mask[t]),
                int(bool(close_any[t])),
                int(bool(coupled_any[t])),
                BODY_PART_NAMES[int(closest_part_gen[t])],
                BODY_PART_NAMES[int(closest_part_orig[t])],
            ])

    png_path = output_dir / f"{seq_id}.png"
    _write_plot(
        png_path,
        seq_id=seq_id,
        min_dist_gen=min_dist_gen,
        min_dist_orig=min_dist_orig,
        target_error=target_error,
        object_speed=object_speed,
        contact_any=contact_any,
        close_any=close_any,
        coupled_any=coupled_any,
    )

    moving_mask = moving
    summary = {
        "seq_id": seq_id,
        "dataset_index": int(dataset_index),
        "generated_index": int(generated_index),
        "seq_len": int(seq_len),
        "T_evaluated": int(T),
        "text": str(sample["text"]),
        "generated_mean_min_dist_cm": _round(float(min_dist_gen.mean()) * 100.0, 2),
        "original_mean_min_dist_cm": _round(float(min_dist_orig.mean()) * 100.0, 2),
        "generated_target_error_cm": _round(_mean(target_error) * 100.0, 2),
        "generated_late_minus_early_dist_cm": _round(
            (float(min_dist_gen[(2 * T) // 3 :].mean()) - float(min_dist_gen[: T // 3].mean()))
            * 100.0,
            2,
        ),
        "label_contact_frame_frac": _round(float(contact_any.mean())),
        "generated_close_frame_frac": _round(float(close_any.mean())),
        "moving_frame_frac": _round(float(moving.mean())),
        "moving_generated_close_frac": (
            _round(float(close_any[moving_mask].mean()))
            if int(moving_mask.sum()) > 0
            else None
        ),
        "moving_generated_coupled_frac": (
            _round(float(coupled_any[moving_mask].mean()))
            if int(moving_mask.sum()) > 0
            else None
        ),
        "dominant_label_contact_parts": {
            name: _round(float(contact_mask[:, i].mean()))
            for i, name in enumerate(BODY_PART_NAMES)
        },
        "dominant_generated_closest_parts": {
            name: _round(float((closest_part_gen == i).mean()))
            for i, name in enumerate(BODY_PART_NAMES)
        },
        "segments": _segment_rows(
            min_dist_gen,
            min_dist_orig,
            target_error,
            contact_any,
            moving,
            close_any,
            coupled_any,
        ),
        "csv": str(csv_path),
        "plot": str(png_path),
    }
    save_json(output_dir / f"{seq_id}.summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seq-id", action="append", required=True)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--contact-threshold", type=float, default=0.5)
    parser.add_argument("--moving-speed-threshold", type=float, default=0.15)
    parser.add_argument("--coupling-threshold", type=float, default=0.5)
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    cfg = OmegaConf.load(args.config)
    dataset = _build_val_dataset(cfg)

    generated_path = args.generated_dir / "generated.npz"
    summary_path = args.generated_dir / "summary.json"
    if not generated_path.exists():
        raise FileNotFoundError(generated_path)
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    generated_summary = _load_json(summary_path)
    generated_seq_ids = [str(s) for s in generated_summary["seq_ids"]]
    generated_index_by_seq = {sid: i for i, sid in enumerate(generated_seq_ids)}
    dataset_index_by_seq = _dataset_index_by_seq(args.generated_dir)

    with np.load(generated_path) as npz:
        generated = {k: npz[k] for k in npz.files}

    summaries = []
    for seq_id in args.seq_id:
        if seq_id not in generated_index_by_seq:
            raise KeyError(f"{seq_id!r} not found in {summary_path}")
        if seq_id not in dataset_index_by_seq:
            raise KeyError(
                f"{seq_id!r} not found in {args.generated_dir.parent / 'summary.json'} "
                f"or {args.generated_dir.parent / 'diagnostic_summary.json'}"
            )
        print(f"diagnosing {seq_id}")
        summaries.append(diagnose_seq(
            seq_id=seq_id,
            dataset_index=dataset_index_by_seq[seq_id],
            generated_index=generated_index_by_seq[seq_id],
            dataset=dataset,
            generated=generated,
            output_dir=args.output_dir,
            fps=float(args.fps),
            contact_threshold=float(args.contact_threshold),
            moving_speed_threshold=float(args.moving_speed_threshold),
            coupling_threshold=float(args.coupling_threshold),
        ))

    save_json(args.output_dir / "summary.json", {"clips": summaries})
    print(f"wrote {args.output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
