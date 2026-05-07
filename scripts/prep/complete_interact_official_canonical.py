"""Build a clean 4-subset InterAct canonical root using official processing.

This script is intentionally a thin wrapper around the upstream InterAct
process code. It never calls our old SMPL-X-only preprocessing path.

For subsets that already ship official ``joints.npy``, ``markers.npy``, and
``motion.npy`` in the downloaded InterAct archive, it copies those canonical
files into a clean 4-subset root. For subsets that only ship canonical
``human.npz`` / ``object.npz`` / ``text.txt`` (notably OMOMO in the local
release), it completes the missing official artifacts by using upstream
``process.canonicalize_human.visualize_smpl`` and the upstream motion
representation logic.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm


DEFAULT_SUBSETS: tuple[str, ...] = (
    "chairs",
    "imhd",
    "neuraldome",
    "omomo_correct_v2",
)
REQUIRED_CANONICAL_FILES: tuple[str, ...] = (
    "human.npz",
    "object.npz",
    "text.txt",
    "joints.npy",
    "markers.npy",
    "motion.npy",
)


@dataclass
class Config:
    interact_root: Path
    output_root: Path
    official_repo: Path
    subsets: tuple[str, ...] = DEFAULT_SUBSETS
    limit: int | None = None
    force_missing: bool = False
    device: str = "cuda"


def _copytree(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    shutil.copytree(src, dst, dirs_exist_ok=True)


def _copy_subset_scaffold(subset: str, cfg: Config) -> None:
    src_subset = cfg.interact_root / subset
    dst_subset = cfg.output_root / subset
    dst_subset.mkdir(parents=True, exist_ok=True)
    for name in ("objects", "objects_bps"):
        _copytree(src_subset / name, dst_subset / name)
    src_seq_root = src_subset / "sequences_canonical"
    dst_seq_root = dst_subset / "sequences_canonical"
    dst_seq_root.mkdir(parents=True, exist_ok=True)
    seq_dirs = sorted(p for p in src_seq_root.iterdir() if p.is_dir())
    if cfg.limit is not None:
        seq_dirs = seq_dirs[: cfg.limit]
    for seq_dir in seq_dirs:
        _copytree(seq_dir, dst_seq_root / seq_dir.name)


def _object_name(raw: np.ndarray | str) -> str:
    if isinstance(raw, np.ndarray):
        if raw.shape == ():
            return str(raw.item())
        return str(raw.reshape(-1)[0])
    return str(raw)


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _dataset_spec(subset: str) -> tuple[str, int, list[int]]:
    # Mirrors upstream process/canonicalize_human.py and
    # process/motion_representation.py.
    from process.markerset import markerset_smplh, markerset_smplx

    key = subset.lower()
    if key in {"imhd", "neuraldome"}:
        return "smplh", 16, markerset_smplh
    if key == "chairs":
        return "smplx", 10, markerset_smplx
    if key in {"omomo", "omomo_correct_v1", "omomo_correct_v2"}:
        return "smplx", 16, markerset_smplx
    raise ValueError(f"Unsupported InterAct subset for official processing: {subset}")


def _matrix_to_rotation_6d_np(mat: np.ndarray) -> np.ndarray:
    batch_dim = mat.shape[:-2]
    return mat[..., :2, :].reshape(batch_dim + (6,))


def _rotation_6d_to_matrix_np(d6: np.ndarray) -> np.ndarray:
    from pytorch3d.transforms import rotation_6d_to_matrix

    return rotation_6d_to_matrix(torch.from_numpy(d6)).numpy()


def _contact_detect_torch(
    verts: np.ndarray,
    obj_points: np.ndarray,
    freeze_obj_points: np.ndarray,
    *,
    device: str,
) -> np.ndarray:
    verts_t = torch.tensor(verts, device=device)
    obj_points_t = torch.tensor(obj_points, device=device)
    freeze_obj_points_t = torch.tensor(freeze_obj_points, device=device)

    contact_vector = obj_points_t[:, None, :, :] - verts_t[:, :, None, :]
    contact = torch.norm(contact_vector, dim=-1)
    _, contact_label = torch.min(contact, dim=-1, keepdim=True)

    expanded = contact_label.unsqueeze(-1).expand(-1, -1, -1, 3)
    freeze_obj_verts = freeze_obj_points_t.expand(
        contact_label.size(0), contact_label.size(1), -1, -1
    )
    selected_contact_vector = torch.gather(contact_vector, 2, expanded).squeeze()
    selected_can_obj_verts = torch.gather(freeze_obj_verts, 2, expanded).squeeze()
    contact_data = torch.cat([selected_contact_vector, selected_can_obj_verts], dim=-1)
    return contact_data.detach().cpu().numpy()


def _get_representation_canonical(
    markers: np.ndarray,
    obj_data: np.ndarray,
    obj_points: np.ndarray,
    *,
    device: str,
) -> np.ndarray:
    # Exact feature layout from upstream process/motion_representation.py:
    # positions, velocity, foot heights, object pose/velocity, dense
    # marker-to-object nearest vectors + canonical object points.
    fid_r = [61, 52, 53, 40, 34, 49, 40]
    fid_l = [29, 30, 18, 19, 7, 2, 15]

    obj_angles = obj_data[:, :6]
    obj_trans = obj_data[:, 6:9]
    ground_marker = obj_trans[:, None, :].copy()
    ground_marker[..., 1] = 0
    angle_matrix = _rotation_6d_to_matrix_np(obj_angles)
    obj_verts = obj_points[None, ...]
    freeze_obj_verts = obj_verts[None, ...]
    obj_verts = (
        np.matmul(obj_verts, np.transpose(angle_matrix, (0, 2, 1)))
        + obj_trans[:, None, :]
    )

    velocity = markers[1:] - markers[:-1]
    velocity_obj = obj_data[1:] - obj_data[:-1]
    velocity = np.concatenate([velocity, np.zeros(velocity.shape[1:])[None]], axis=0)
    velocity_obj = np.concatenate(
        [velocity_obj, np.zeros(velocity_obj.shape[1:])[None]], axis=0
    )

    feet_l = markers[:, fid_l, 1]
    feet_r = markers[:, fid_r, 1]
    contact_data = _contact_detect_torch(
        np.concatenate([markers, ground_marker], axis=1),
        obj_verts,
        freeze_obj_verts,
        device=device,
    )

    velocity = velocity.reshape(len(velocity), -1)
    positions = markers.reshape(len(markers), -1)
    contact_data = contact_data.reshape(len(markers), -1)
    return np.concatenate(
        [positions, velocity, feet_l, feet_r, obj_data, velocity_obj, contact_data],
        axis=-1,
    )


def _load_official_modules(official_repo: Path):
    sys.path.insert(0, str(official_repo))
    old_cwd = Path.cwd()
    os.chdir(official_repo)
    try:
        from process import canonicalize_human
    finally:
        os.chdir(old_cwd)
    return canonicalize_human


def _complete_sequence(
    seq_dir: Path,
    seq_root: Path,
    objects_root: Path,
    subset: str,
    canonicalize_human,
    *,
    cfg: Config,
) -> str:
    missing = [name for name in REQUIRED_CANONICAL_FILES if not (seq_dir / name).exists()]
    if not missing and not cfg.force_missing:
        return "already_complete"

    model_type, num_betas, marker_indices = _dataset_spec(subset)
    verts, _faces, joints = canonicalize_human.visualize_smpl(
        seq_dir.name,
        str(seq_root),
        model_type,
        num_betas,
    )
    verts_np = _to_numpy(verts)
    joints_np = _to_numpy(joints)
    markers_np = verts_np[:, marker_indices]

    np.save(seq_dir / "joints.npy", joints_np)
    np.save(seq_dir / "markers.npy", markers_np)

    with np.load(seq_dir / "object.npz", allow_pickle=True) as data:
        obj_angles = data["angles"]
        obj_trans = data["trans"]
        obj_name = _object_name(data["name"])
    angle_matrix = Rotation.from_rotvec(obj_angles).as_matrix()
    obj_rot6d = _matrix_to_rotation_6d_np(angle_matrix)
    obj_data = np.concatenate([obj_rot6d, obj_trans], axis=-1)
    points_path = objects_root / obj_name / "sample_points.npy"
    if not points_path.exists():
        points_path = objects_root / obj_name / "2048.npy"
    obj_points = np.load(points_path)
    motion = _get_representation_canonical(
        markers_np,
        obj_data,
        obj_points,
        device=cfg.device,
    )
    np.save(seq_dir / "motion.npy", motion)
    return "completed"


def _complete_subset(subset: str, cfg: Config, canonicalize_human) -> dict:
    t0 = time.time()
    subset_root = cfg.output_root / subset
    seq_root = subset_root / "sequences_canonical"
    objects_root = subset_root / "objects"
    seq_dirs = sorted(p for p in seq_root.iterdir() if p.is_dir())
    if cfg.limit is not None:
        seq_dirs = seq_dirs[: cfg.limit]

    counts = {"already_complete": 0, "completed": 0, "failed": 0}
    failures: list[dict[str, str]] = []
    for seq_dir in tqdm(seq_dirs, desc=f"{subset} official canonical"):
        try:
            status = _complete_sequence(
                seq_dir,
                seq_root,
                objects_root,
                subset,
                canonicalize_human,
                cfg=cfg,
            )
            counts[status] = counts.get(status, 0) + 1
        except Exception as exc:
            counts["failed"] += 1
            failures.append({"seq_id": seq_dir.name, "reason": repr(exc)})

    return {
        "subset": subset,
        "num_sequences": len(seq_dirs),
        "counts": counts,
        "elapsed_sec": round(time.time() - t0, 2),
        "failures": failures[:100],
    }


def run(cfg: Config) -> None:
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Official motion representation requested CUDA, but CUDA is unavailable")

    cfg.output_root.mkdir(parents=True, exist_ok=True)
    for subset in cfg.subsets:
        _copy_subset_scaffold(subset, cfg)

    canonicalize_human = _load_official_modules(cfg.official_repo)
    summaries = [_complete_subset(subset, cfg, canonicalize_human) for subset in cfg.subsets]
    report = {
        "timestamp": datetime.now().isoformat(),
        "interact_root": str(cfg.interact_root),
        "output_root": str(cfg.output_root),
        "official_repo": str(cfg.official_repo),
        "subsets": list(cfg.subsets),
        "limit": cfg.limit,
        "device": cfg.device,
        "per_subset": summaries,
    }
    (cfg.output_root / "official_process_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    failed = sum(item["counts"].get("failed", 0) for item in summaries)
    if failed:
        raise RuntimeError(f"Official canonical completion failed for {failed} sequences")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interact-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--subsets", nargs="+", default=list(DEFAULT_SUBSETS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force-missing", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(
        Config(
            interact_root=args.interact_root,
            output_root=args.output_root,
            official_repo=args.official_repo,
            subsets=tuple(args.subsets),
            limit=args.limit,
            force_missing=args.force_missing,
            device=args.device,
        )
    )


if __name__ == "__main__":
    main()
