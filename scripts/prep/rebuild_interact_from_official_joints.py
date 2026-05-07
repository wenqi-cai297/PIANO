"""Rebuild PIANO InterAct roots from official InterAct canonical joints.

This is a corrective path for InterAct subsets that already ship
``sequences_canonical/<seq_id>/joints.npy``. The original
``piano.data.preprocess_interact`` recomputes joints from ``human.npz`` using
one SMPL-X FK path for every subset. The official InterAct processing uses
different body models per source subset, so for subsets with official joints
available, those joints are the safer source of truth.

The script writes a fresh PIANO-format root and never mutates the source
InterAct directory.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

from piano.data.humanml3d_encoder import HumanML3DEncoder
from piano.data.preprocess_omomo import (
    NUM_OBJECT_POINTS,
    SOURCE_FPS,
    TARGET_FPS,
    downsample_temporal,
    sample_object_point_cloud,
)
from piano.utils.io_utils import ensure_dir, save_json, save_npz


DEFAULT_SUBSETS: tuple[str, ...] = ("chairs", "imhd", "neuraldome")


@dataclass(slots=True)
class Config:
    interact_root: Path
    output_root: Path
    current_piano_root: Path | None
    subsets: tuple[str, ...]
    source_fps: float = SOURCE_FPS
    target_fps: float = TARGET_FPS
    num_object_points: int = NUM_OBJECT_POINTS
    min_frames: int = 5
    limit: int | None = None


def _sequence_root(subset_dir: Path) -> Path:
    candidates = (
        subset_dir / "sequences_canonical",
        subset_dir / "sequences" / "sequences_canonical",
    )
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Cannot find sequences_canonical under {subset_dir}")


def _object_name(raw: np.ndarray | str) -> str:
    if isinstance(raw, np.ndarray):
        if raw.shape == ():
            return str(raw.item())
        return str(raw.reshape(-1)[0])
    return str(raw)


def _read_text(seq_dir: Path) -> str:
    text_path = seq_dir / "text.txt"
    if not text_path.exists():
        return ""
    raw = text_path.read_text(encoding="utf-8", errors="replace").strip()
    return raw.split("#")[0].strip()


def _load_official_joints22(seq_dir: Path, cfg: Config) -> np.ndarray | None:
    path = seq_dir / "joints.npy"
    if not path.exists():
        return None
    joints = np.load(path).astype(np.float32)
    if joints.ndim != 3 or joints.shape[1] < 22 or joints.shape[2] != 3:
        raise ValueError(f"Unexpected joints.npy shape at {path}: {joints.shape}")
    joints_22 = joints[:, :22, :]
    return downsample_temporal(joints_22, cfg.source_fps, cfg.target_fps).astype(np.float32)


def _load_object_pose(seq_dir: Path, cfg: Config) -> tuple[np.ndarray, np.ndarray, str]:
    path = seq_dir / "object.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing object.npz: {seq_dir}")
    with np.load(path, allow_pickle=True) as data:
        angles = data["angles"].astype(np.float32)
        trans = data["trans"].astype(np.float32)
        name = _object_name(data["name"])
    angles_ds = downsample_temporal(angles, cfg.source_fps, cfg.target_fps).astype(np.float32)
    trans_ds = downsample_temporal(trans, cfg.source_fps, cfg.target_fps).astype(np.float32)
    return trans_ds, angles_ds, name


def _load_human_params(seq_dir: Path, cfg: Config) -> dict[str, np.ndarray | str]:
    path = seq_dir / "human.npz"
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=True) as data:
        out: dict[str, np.ndarray | str] = {}
        if "poses" in data.files:
            out["smplx_poses"] = downsample_temporal(
                data["poses"].astype(np.float32), cfg.source_fps, cfg.target_fps
            ).astype(np.float32)
        if "trans" in data.files:
            out["smplx_trans"] = downsample_temporal(
                data["trans"].astype(np.float32), cfg.source_fps, cfg.target_fps
            ).astype(np.float32)
        if "betas" in data.files:
            out["smplx_betas"] = data["betas"].astype(np.float32)
        if "gender" in data.files:
            out["gender"] = _object_name(data["gender"])
    return out


def _subsample_points(points: np.ndarray, n: int, seed: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if len(points) == n:
        return points
    rng = np.random.default_rng(seed)
    replace = len(points) < n
    idx = rng.choice(len(points), size=n, replace=replace)
    return points[idx].astype(np.float32)


def _find_object_mesh(obj_dir: Path, obj_name: str) -> Path | None:
    candidates = (
        obj_dir / f"{obj_name}.obj",
        obj_dir / f"{obj_name}_face1000.obj",
        obj_dir / f"{obj_name}_simplified.obj",
        obj_dir / "mesh.obj",
    )
    for p in candidates:
        if p.exists():
            return p
    objs = sorted(obj_dir.glob("*.obj"))
    return objs[0] if objs else None


def _write_object_clouds(subset: str, cfg: Config, object_names: set[str], out_root: Path) -> None:
    objects_out = ensure_dir(out_root / "objects")
    src_objects = cfg.interact_root / subset / "objects"
    current_objects = (
        cfg.current_piano_root / subset / "objects"
        if cfg.current_piano_root is not None
        else None
    )

    for obj_name in sorted(object_names):
        out_path = objects_out / f"{obj_name}.npy"
        if out_path.exists():
            continue

        obj_dir = src_objects / obj_name
        points_path = obj_dir / "2048.npy"
        if not points_path.exists():
            points_path = obj_dir / "sample_points.npy"
        if points_path.exists():
            points = np.load(points_path)
            np.save(out_path, _subsample_points(points, cfg.num_object_points, seed=42))
            continue

        if current_objects is not None:
            current_path = current_objects / f"{obj_name}.npy"
            if current_path.exists():
                points = np.load(current_path)
                np.save(out_path, _subsample_points(points, cfg.num_object_points, seed=42))
                continue

        mesh_path = _find_object_mesh(obj_dir, obj_name)
        if mesh_path is None:
            raise FileNotFoundError(f"Cannot find points or mesh for {subset}/{obj_name}")
        np.save(out_path, sample_object_point_cloud(mesh_path, cfg.num_object_points))


def _first_reference_joints(seq_dirs: list[Path], cfg: Config) -> np.ndarray:
    for seq_dir in seq_dirs:
        joints = _load_official_joints22(seq_dir, cfg)
        if joints is not None and len(joints) >= 2:
            return joints[0]
    raise RuntimeError("No sequence with usable official joints.npy found")


def _process_subset(subset: str, cfg: Config) -> dict:
    t0 = time.time()
    subset_dir = cfg.interact_root / subset
    seq_root = _sequence_root(subset_dir)
    seq_dirs = sorted(p for p in seq_root.iterdir() if p.is_dir())
    if cfg.limit is not None:
        seq_dirs = seq_dirs[: cfg.limit]

    out_root = ensure_dir(cfg.output_root / subset)
    motions_out = ensure_dir(out_root / "motions")

    reference = _first_reference_joints(seq_dirs, cfg)
    encoder = HumanML3DEncoder(reference_joints=reference, feet_thre=0.002)

    metadata: list[dict] = []
    object_names: set[str] = set()
    skipped: list[dict[str, str]] = []

    for seq_dir in tqdm(seq_dirs, desc=f"{subset} official-joints"):
        sid = seq_dir.name
        try:
            joints = _load_official_joints22(seq_dir, cfg)
            if joints is None:
                skipped.append({"seq_id": sid, "reason": "missing_joints.npy"})
                continue
            obj_pos, obj_rot, obj_name = _load_object_pose(seq_dir, cfg)
            human_params = _load_human_params(seq_dir, cfg)
            features, _ = encoder.encode(joints)
            T = features.shape[0]
            if T < cfg.min_frames:
                skipped.append({
                    "seq_id": sid,
                    "reason": f"too_short_after_encode_{T}_lt_{cfg.min_frames}",
                })
                continue

            arrays: dict[str, np.ndarray] = {
                "joints_22": joints[:T].astype(np.float32),
                "motion_263": features.astype(np.float32),
                "object_positions": obj_pos[:T].astype(np.float32),
                "object_rotations": obj_rot[:T].astype(np.float32),
            }
            for key in ("smplx_poses", "smplx_trans"):
                value = human_params.get(key)
                if isinstance(value, np.ndarray):
                    arrays[key] = value[:T].astype(np.float32)
            if isinstance(human_params.get("smplx_betas"), np.ndarray):
                arrays["smplx_betas"] = human_params["smplx_betas"].astype(np.float32)

            save_npz(motions_out / f"{sid}.npz", **arrays)
            object_names.add(obj_name)
            metadata.append(
                {
                    "seq_id": sid,
                    "subset": subset,
                    "split": "train",
                    "object_id": obj_name,
                    "gender": str(human_params.get("gender", "neutral")),
                    "text": _read_text(seq_dir),
                    "num_frames": int(T),
                }
            )
        except Exception as exc:
            skipped.append({"seq_id": sid, "reason": repr(exc)})

    _write_object_clouds(subset, cfg, object_names, out_root)
    save_json(out_root / "metadata.json", metadata)
    summary = {
        "timestamp": datetime.now().isoformat(),
        "subset": subset,
        "source": "official InterAct sequences_canonical/joints.npy",
        "num_sequences_total": len(seq_dirs),
        "num_processed": len(metadata),
        "num_skipped": len(skipped),
        "num_objects": len(object_names),
        "source_fps": float(cfg.source_fps),
        "target_fps": float(cfg.target_fps),
        "min_frames": int(cfg.min_frames),
        "elapsed_sec": round(time.time() - t0, 2),
        "skipped": skipped[:100],
    }
    save_json(out_root / "summary.json", summary)
    return summary


def run(cfg: Config) -> None:
    t0 = time.time()
    ensure_dir(cfg.output_root)
    summaries = [_process_subset(subset, cfg) for subset in cfg.subsets]
    save_json(
        cfg.output_root / "summary.json",
        {
            "timestamp": datetime.now().isoformat(),
            "interact_root": str(cfg.interact_root),
            "output_root": str(cfg.output_root),
            "current_piano_root": str(cfg.current_piano_root) if cfg.current_piano_root else None,
            "subsets": list(cfg.subsets),
            "min_frames": int(cfg.min_frames),
            "totals": {
                "num_processed": sum(s["num_processed"] for s in summaries),
                "num_skipped": sum(s["num_skipped"] for s in summaries),
                "elapsed_sec": round(time.time() - t0, 2),
            },
            "per_subset": summaries,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interact-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--current-piano-root", type=Path, default=None)
    parser.add_argument("--subsets", nargs="+", default=list(DEFAULT_SUBSETS))
    parser.add_argument("--source-fps", type=float, default=SOURCE_FPS)
    parser.add_argument("--target-fps", type=float, default=TARGET_FPS)
    parser.add_argument("--num-object-points", type=int, default=NUM_OBJECT_POINTS)
    parser.add_argument(
        "--min-frames",
        type=int,
        default=5,
        help="Skip clips shorter than this after HumanML3D encoding. Default 5 "
             "matches the pseudo-label contact min-duration filter.",
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(
        Config(
            interact_root=args.interact_root,
            output_root=args.output_root,
            current_piano_root=args.current_piano_root,
            subsets=tuple(args.subsets),
            source_fps=args.source_fps,
            target_fps=args.target_fps,
            num_object_points=args.num_object_points,
            min_frames=args.min_frames,
            limit=args.limit,
        )
    )


if __name__ == "__main__":
    main()
