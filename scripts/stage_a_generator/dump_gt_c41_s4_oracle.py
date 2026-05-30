"""Dump oracle GT (C41, S4) tensors as substitute_conds-format npz.

For the same selection that the Stage-1.5 downstream-coupling diag uses,
this script computes the SAME C41 / S4 the dataset / trainer / PB1
inference would surface (via the canonical builders in
``piano.data.stage2_oracle_conditions``) and writes them as per-clip
``.npz`` files with the schema the substitute_conds loader expects:

    <out-dir>/<bucket>/<subset>/<seq_id>.npz
        stage2_coarse_extra : (T_padded, 18) RAW (C41-current)
        stage2_support      : (T_padded, 13) RAW (S4-S1-phase-footstep)
        valid_T             : int32
        seed                : int32 (always 0 for GT)

This becomes the GT reference for D1 spectral-swap and D2 residual
sensitivity. The output is interchangeable with any
``--substitute-conds-dir`` consumer (round26_sustained_contact_diag,
round26_gait_diag, round28_body_action_diag, round29_g1_soft_stance_diag)
because the npz schema matches what
``src/piano/inference/sample_substitute_conds.py:423-432`` writes for
stage1p5 sampling.

CPU only. ~3-5 min on 48 clips. Idempotent.

Run:
    python -u scripts/stage_a_generator/dump_gt_c41_s4_oracle.py \\
        --cfg configs/training/anchordiff_r29_pb_a1_adaln_s4.yaml \\
        --selection-json analyses/round29_val_diag_indices_48_balanced.json \\
        --bucket val \\
        --out-dir analyses/2026-05-31_stage1p5_wrist_d1_oracle_dump
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from piano.data.dataset import collate_hoi
from piano.training.train_anchordiff import _build_dataset


def _read_selection(path: Path) -> set[tuple[str, str]]:
    sel = json.loads(path.read_text("utf-8"))
    items = sel.get("selected") or sel.get("candidates") or sel.get("clips") or []
    if not items:
        raise SystemExit(f"empty selection: {path}")
    return {(e["subset"], e["seq_id"]) for e in items}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=Path, required=True,
                    help="PB1 cfg whose data section sets "
                         "r29_coarse_variant=C41-current + "
                         "r29_support_variant=S4-S1-phase-footstep.")
    ap.add_argument("--selection-json", type=Path, required=True)
    ap.add_argument("--bucket", choices=["train", "val"], default="val")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.cfg)
    coarse_v = str(cfg.data.get("r29_coarse_variant", "C0"))
    support_v = str(cfg.data.get("r29_support_variant", "S0"))
    if coarse_v != "C41-current":
        raise SystemExit(
            f"[dump_gt] cfg has r29_coarse_variant={coarse_v!r}; "
            "this dump expects 'C41-current' (PB1 contract). Use the "
            "PB1 cfg, not a Stage-1.5 cfg."
        )
    if support_v != "S4-S1-phase-footstep":
        raise SystemExit(
            f"[dump_gt] cfg has r29_support_variant={support_v!r}; "
            "this dump expects 'S4-S1-phase-footstep' (PB1 contract)."
        )

    sel_pairs = _read_selection(args.selection_json)
    print(f"[dump_gt] selection size = {len(sel_pairs)}")

    dataset = _build_dataset(cfg, args.bucket, augment=False)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        collate_fn=collate_hoi, num_workers=0,
    )

    out_root = args.out_dir / args.bucket
    out_root.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_seen = 0
    for batch in loader:
        n_seen += 1
        subset = str(batch["subset"][0])
        seq_id = str(batch["seq_id"][0])
        if (subset, seq_id) not in sel_pairs:
            continue

        if "stage2_coarse_extra" not in batch or "stage2_support" not in batch:
            raise KeyError(
                "batch is missing stage2_coarse_extra/stage2_support — "
                "did dataset get configured with the right cfg.data variants?"
            )

        c41 = batch["stage2_coarse_extra"][0].cpu().numpy().astype(np.float32)  # (T_pad, 18)
        s4 = batch["stage2_support"][0].cpu().numpy().astype(np.float32)        # (T_pad, 13)
        valid_T = int(batch["seq_len"][0].item())
        T_pad = c41.shape[0]
        if valid_T > T_pad:
            raise RuntimeError(
                f"[dump_gt] ({subset}, {seq_id}) valid_T={valid_T} > T_pad={T_pad}"
            )
        if c41.shape[-1] != 18 or s4.shape[-1] != 13:
            raise RuntimeError(
                f"[dump_gt] ({subset}, {seq_id}) wrong shapes: c41 {c41.shape} s4 {s4.shape}"
            )

        out_sub = out_root / subset
        out_sub.mkdir(parents=True, exist_ok=True)
        save_path = out_sub / f"{seq_id}.npz"
        np.savez(
            save_path,
            stage2_coarse_extra=c41,
            stage2_support=s4,
            valid_T=np.int32(valid_T),
            seed=np.int32(0),
        )
        n_written += 1
        if n_written % 8 == 0:
            print(f"[dump_gt] wrote {n_written}/{len(sel_pairs)}")

    if n_written != len(sel_pairs):
        missed = len(sel_pairs) - n_written
        print(
            f"[dump_gt] WARN: only wrote {n_written}/{len(sel_pairs)} "
            f"selection entries (missing {missed}; seen {n_seen} dataset items)."
        )
    print(f"[dump_gt] DONE: {n_written} clips → {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
