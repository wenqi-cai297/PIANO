"""Build Stage-1 coarse alpha-residual substitute caches.

For each selected clip, compute oracle z-scored ``stage1_coarse`` from GT
motion and mix it with generated Stage-1 output:

    mixed = oracle + alpha * (generated - oracle)

The output is a set of substitute_conds dirs suitable as
``ROUND32_DS_UPSTREAM_DIR`` for ``run_round32_stage1p5_downstream_diag.sh``.

Output schema::

    <out-root>/alpha050/<bucket>/<subset>/<seq_id>.npz
        stage1_coarse: (T, 23), z-scored
        valid_T: int
        alpha: float
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from piano.data.dataset import collate_hoi
from piano.data.stage1_coarse_oracle import (
    extract_coarse_v1_batched,
    load_stage1_coarse_norm,
)
from piano.inference.sample_substitute_conds import _read_selection
from piano.training.train_anchordiff import _build_dataset


STAGE1_COARSE_DIM = 23


def alpha_tag(alpha: float) -> str:
    """Stable sortable tag, e.g. 0.50 -> alpha050."""
    return f"alpha{int(round(alpha * 100)):03d}"


def _cache_bucket_root(root: Path, bucket: str) -> Path:
    return root / bucket if (root / bucket).is_dir() else root


def _load_generated(root: Path, bucket: str, subset: str, seq_id: str) -> tuple[np.ndarray, int]:
    p = _cache_bucket_root(root, bucket) / subset / f"{seq_id}.npz"
    if not p.exists():
        raise FileNotFoundError(p)
    data = np.load(p)
    if "stage1_coarse" not in data.files:
        raise KeyError(f"{p}: missing stage1_coarse (keys={list(data.files)})")
    arr = data["stage1_coarse"].astype(np.float32)
    if arr.ndim != 2 or arr.shape[-1] != STAGE1_COARSE_DIM:
        raise ValueError(f"{p}: expected (T, 23), got {arr.shape}")
    valid_T = int(data["valid_T"]) if "valid_T" in data.files else arr.shape[0]
    return arr, valid_T


def _save_stage1(path: Path, arr: np.ndarray, valid_T: int, alpha: float) -> None:
    if arr.ndim != 2 or arr.shape[-1] != STAGE1_COARSE_DIM:
        raise ValueError(f"expected (T, 23), got {arr.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        stage1_coarse=arr.astype(np.float32),
        valid_T=np.int32(valid_T),
        seed=np.int32(0),
        alpha=np.float32(alpha),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--generated-dir", type=Path, required=True)
    ap.add_argument("--selection-json", type=Path, required=True)
    ap.add_argument("--bucket", choices=["train", "val"], default="val")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument(
        "--alphas", default="0.00,0.25,0.50,0.75,1.00",
        help="comma-separated alpha values in [0, 1]",
    )
    args = ap.parse_args()

    alphas = tuple(float(s.strip()) for s in args.alphas.split(",") if s.strip())
    for a in alphas:
        if not (0.0 <= a <= 1.0):
            raise SystemExit(f"alpha outside [0, 1]: {a}")

    cfg = OmegaConf.load(str(args.config))
    sel_pairs = _read_selection(args.selection_json)
    mean_np, std_np = load_stage1_coarse_norm(str(cfg.data.stage1_coarse_cache_root))
    mean = mean_np.astype(np.float32).reshape(1, STAGE1_COARSE_DIM)
    std = std_np.astype(np.float32).reshape(1, STAGE1_COARSE_DIM)

    dataset = _build_dataset(cfg, args.bucket, augment=False)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=collate_hoi, num_workers=0,
    )

    counts = {alpha_tag(a): 0 for a in alphas}
    for batch in loader:
        subset = str(batch["subset"][0])
        seq_id = str(batch["seq_id"][0])
        if (subset, seq_id) not in sel_pairs:
            continue

        gen, vt_gen = _load_generated(args.generated_dir, args.bucket, subset, seq_id)
        motion = batch["motion"].float()
        rest_offsets = batch["rest_offsets"].float()
        oracle_raw = extract_coarse_v1_batched(
            motion=motion, rest_offsets=rest_offsets,
        )[0].cpu().numpy().astype(np.float32)
        oracle = (oracle_raw - mean) / std

        if gen.shape[0] != oracle.shape[0]:
            raise RuntimeError(
                f"({subset}, {seq_id}) T mismatch generated {gen.shape} vs oracle {oracle.shape}"
            )
        valid_T = min(int(batch["seq_len"][0].item()), vt_gen, gen.shape[0])
        residual = gen - oracle

        for a in alphas:
            tag = alpha_tag(a)
            mixed = oracle + float(a) * residual
            out_path = args.out_root / tag / args.bucket / subset / f"{seq_id}.npz"
            _save_stage1(out_path, mixed, valid_T=valid_T, alpha=a)
            counts[tag] += 1

    print("[stage1_residual] DONE")
    for tag in sorted(counts):
        print(f"  {tag}: {counts[tag]} clips -> {args.out_root / tag / args.bucket}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
