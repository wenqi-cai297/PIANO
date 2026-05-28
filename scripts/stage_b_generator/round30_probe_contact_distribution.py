"""Round-30 E0 forensic — explain why ILD = 2 / 8283.

Walk every subset, sample 30 clips per subset randomly, read each clip's
pseudo_label npz, and dump:

  - left/right hand contact frame fraction (cs[:, 0] > 0.5).mean()
  - the keyword regex hit on the text
  - whether the clip is_stationary (root XZ p95 + walking_frac)

This tells us whether the 0.03 % verdict is a real data sparsity result
or a pseudo-label false-positive (e.g. chair clips with contact_state
always = 1 on hand columns because sitting + hand-on-armrest looks
like "hand contact" to the extractor).

Usage on the server:
    python scripts/stage_b_generator/round30_probe_contact_distribution.py \\
        --config configs/training/anchordiff_r29_ns_a1_c41_s4_g1.yaml \\
        --samples-per-subset 30 --seed 0
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "stage_b_generator"
sys.path.insert(0, str(SCRIPTS))
from round30_build_ild_subset import (  # noqa: E402
    IDLE_LOCAL_DETAIL_KEYWORDS,
    _resolve_npz_path,
    _root_xz_p95_m,
    _walking_frac,
)


def _hand_contact_frac(cs: np.ndarray) -> tuple[float, float]:
    if cs is None or cs.ndim != 2 or cs.shape[-1] < 2:
        return float("nan"), float("nan")
    lh = float((cs[:, 0] > 0.5).mean())
    rh = float((cs[:, 1] > 0.5).mean())
    return lh, rh


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--samples-per-subset", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.config)

    rng = random.Random(args.seed)
    compiled = [re.compile(pat, re.IGNORECASE) for pat in IDLE_LOCAL_DETAIL_KEYWORDS]

    pseudo_label_subdir = cfg.data.get("pseudo_label_subdir", None)
    pseudo_label_dir_cfg = cfg.data.get("pseudo_label_dir", None)

    print(
        f"{'subset':<22} {'lh%':>5} {'rh%':>5} {'either%':>7} "
        f"{'root_xz_cm':>10} {'walk%':>6} {'kw':>3} "
        f"{'seq_id':<32} text"
    )
    print("-" * 140)
    bucket_summary: dict[str, dict] = {}
    for entry in cfg.data.datasets:
        root = Path(entry.root)
        subset_name = root.name
        if pseudo_label_dir_cfg is not None:
            pl_root = Path(pseudo_label_dir_cfg)
        elif pseudo_label_subdir:
            pl_root = root / pseudo_label_subdir
        else:
            pl_root = root / "pseudo_labels"
        if not pl_root.exists():
            print(f"!! {subset_name}: pseudo_label_root not found at {pl_root}")
            continue

        meta_path = root / "metadata_clean.json"
        if not meta_path.exists():
            meta_path = root / "metadata.json"
        meta = json.loads(meta_path.read_text("utf-8"))
        n = min(args.samples_per_subset, len(meta))
        sample = rng.sample(meta, n)

        lh_all: list[float] = []
        rh_all: list[float] = []
        either_all: list[float] = []
        kw_hits = 0
        for m in sample:
            seq_id = str(m["seq_id"])
            text = str(m.get("text", ""))
            npz = _resolve_npz_path(root, seq_id)
            if npz is None:
                continue
            try:
                data = np.load(npz, allow_pickle=False)
            except Exception:
                continue
            joints = (
                data["joints_22"].astype(np.float32) if "joints_22" in data.files
                else (
                    data["joints"].astype(np.float32) if "joints" in data.files
                    else None
                )
            )
            if joints is None:
                continue
            T = min(int(m.get("num_frames", joints.shape[0])), joints.shape[0])
            root_xz_cm = _root_xz_p95_m(joints, T) * 100.0
            wf = _walking_frac(joints, T)
            kw = any(p.search(text) for p in compiled)
            if kw:
                kw_hits += 1
            pl = pl_root / f"{seq_id}.npz"
            lh = rh = either = float("nan")
            if pl.exists():
                try:
                    pld = np.load(pl, allow_pickle=False)
                    if "contact_state" in pld.files:
                        cs = pld["contact_state"][:T].astype(np.float32)
                        lh, rh = _hand_contact_frac(cs)
                        either = float(
                            ((cs[:, 0] > 0.5) | (cs[:, 1] > 0.5)).mean()
                        )
                except Exception:
                    pass
            print(
                f"{subset_name:<22} "
                f"{lh*100:>4.0f}% {rh*100:>4.0f}% {either*100:>6.0f}% "
                f"{root_xz_cm:>10.1f} {wf*100:>5.0f}% "
                f"{'Y' if kw else '-':>3} "
                f"{seq_id:<32} {text[:60]}"
            )
            for lst, v in ((lh_all, lh), (rh_all, rh), (either_all, either)):
                if not np.isnan(v):
                    lst.append(v)
        bucket_summary[subset_name] = {
            "n": n,
            "kw_hit_frac": kw_hits / n if n else 0.0,
            "lh_mean": float(np.mean(lh_all)) if lh_all else None,
            "rh_mean": float(np.mean(rh_all)) if rh_all else None,
            "either_mean": float(np.mean(either_all)) if either_all else None,
            "either_below_30pct_count": sum(1 for v in either_all if v < 0.30),
        }

    print()
    print("=" * 80)
    print("Per-subset summary")
    print("=" * 80)
    print(
        f"{'subset':<22} {'n':>4} {'kw%':>5} "
        f"{'mean_lh%':>9} {'mean_rh%':>9} {'mean_either%':>13} "
        f"{'#(either<30%)':>13}"
    )
    for s, st in bucket_summary.items():
        def _fmt(x):
            return "—" if x is None else f"{x*100:>5.1f}%"
        print(
            f"{s:<22} {st['n']:>4} {st['kw_hit_frac']*100:>4.0f}% "
            f"{_fmt(st['lh_mean']):>9} {_fmt(st['rh_mean']):>9} "
            f"{_fmt(st['either_mean']):>13} "
            f"{st['either_below_30pct_count']:>13}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
