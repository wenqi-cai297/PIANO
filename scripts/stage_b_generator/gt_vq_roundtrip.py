"""GT-VQ roundtrip diagnostic — distinguishes codebook bottleneck from model failure.

After v0.4 (encoder normalization fix landed) showed body now translates with
z_int but contact precision is still off (per user 2026-04-27 night
qualitative review on
``runs/eval/stageB_v0_4_qual_{w4,w8}/*/vis/``), the residual failure mode
could be in either:

- **Codebook capacity**: HumanML3D-trained 512-code RVQ may not have tokens
  for precise HOI contact (grip handle, hand on chair seat, finger on
  object surface). Model approximates with non-HOI tokens that look like
  "standing near object" instead of "gripping object".

- **Model prediction**: Codebook can express precise contact but the
  MaskTransformer hasn't learned to predict the right tokens, e.g. due
  to insufficient training capacity or train/inference distribution
  mismatch.

This diagnostic isolates the two by saving two parallel motion versions
for the same stratified qual_eval val clips:

- ``gt_original/``: real GT motion_263 from HOIDataset (no VQ touch). The
  upper bound on motion fidelity given perfect generation.
- ``gt_roundtrip/``: GT motion_263 → normalize → ``vq.encode`` (all 6
  RVQ layers) → ``vq.forward_decoder`` → denormalize. This is what the
  codebook can actually preserve.

Both are saved in ``visualize_motion`` ``generated``-subcommand-compatible
format (motion_263, object_pc, object_positions, object_rotations,
world_R_y_angle, world_T_xz), so you can render mp4s of each and compare
side by side using the same pipeline as qual eval mp4s.

Verdict matrix from rendering:

| gt_original | gt_roundtrip | meaning |
|-------------|--------------|---------|
| contact OK  | contact OK   | codebook fine; v0.4 visual failure is in MODEL predictions
                              → continue training / Stage 4 joint finetune /
                              architectural changes (v0.3-δ trainable copy)
| contact OK  | contact off  | codebook is bottleneck for precise HOI →
                              v0.3-γ MANDATORY (retrain RVQ on InterAct +
                              HumanML3D union, ~3-4h VQ + ~2d MaskTransformer)
| contact off | contact off  | GT pseudo-labels themselves are noisy (the
                              17cm Stage A train-target floor); fix is
                              Stage A v7 (per-vertex heatmap target head)

Pure measurement — no training, no model forward. Just VQ-VAE forward +
data save. ~10 sec on 1 A6000.

Usage::

    python scripts/stage_b_generator/gt_vq_roundtrip.py \\
        --config configs/training/generator_v04_normalize.yaml \\
        --num-clips 20 \\
        --output-dir runs/eval/stageB_v0_4_gt_roundtrip

Then render both:

    for cond in gt_original gt_roundtrip; do
        python -m piano.inference.visualize_motion generated \\
            --run-dir runs/eval/stageB_v0_4_gt_roundtrip/$cond \\
            --output-dir runs/eval/stageB_v0_4_gt_roundtrip/$cond/vis
    done
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset

from piano.data.dataset import HOIDataset
from piano.data.eval_sampling import (
    describe_eval_clip_selection,
    select_eval_clip_indices,
)
from piano.data.humanml3d_repr import load_motion_stats
from piano.data.split import build_subject_split, extract_subject_id
from piano.models.backbones.momask_adapter import load_momask_vqvae
from piano.utils.io_utils import ensure_dir, load_json


# ============================================================================
# Dataset + sampling (mirror qual_eval so we hit the same stratified clips)
# ============================================================================

def _read_metadata(roots: list) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    for entry in roots:
        root = Path(entry.root)
        meta_path = root / "metadata_clean.json"
        if not meta_path.exists():
            meta_path = root / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata not found in {root}")
        for m in load_json(meta_path):
            out.append((root.name, m))
    return out


def _collect_subject_keys(roots: list) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    for subset_name, m in _read_metadata(roots):
        raw_id = extract_subject_id(subset_name, m.get("seq_id", ""))
        if raw_id is not None:
            seen.add((subset_name, raw_id))
    return sorted(seen)


def _build_val_dataset(cfg) -> ConcatDataset:
    subj_cfg = cfg.data.subject_split
    subject_keys = _collect_subject_keys(cfg.data.datasets)
    splits = build_subject_split(
        subject_keys,
        train_pct=subj_cfg.train_pct,
        val_pct=subj_cfg.val_pct,
        seed=subj_cfg.seed,
    )
    val_filter = splits["val"]
    pseudo_label_dir = cfg.data.get("pseudo_label_dir", None)
    force_world_frame = bool(cfg.data.get("force_world_frame", False))
    datasets = []
    for entry in cfg.data.datasets:
        ds = HOIDataset(
            root=entry.root,
            pseudo_label_dir=pseudo_label_dir,
            max_seq_length=cfg.data.max_seq_length,
            subject_id_filter=val_filter,
            augment=None,
            surface_obj_pose=False,    # don't need z_int — only use motion + object overlay
            force_world_frame=force_world_frame,
        )
        datasets.append(ds)
    return ConcatDataset(datasets)


# ============================================================================
# Canon→world transform (re-export from qual_eval, but qual_eval is a
# script not a module; copy the function rather than import-via-importlib)
# ============================================================================

def _get_canon_to_world_transform(
    joints_world: np.ndarray,         # (T, 22, 3) raw world joints
    motion_263: np.ndarray,           # (T, 263) HumanML3D canonical
) -> tuple[float, np.ndarray]:
    """Recover (R_y_angle, T_xz) such that world = R_y @ canonical + T_xz.

    Frame-0 anchored: T_xz = world_pelvis_xz - R_y(canonical_pelvis_xz);
    canonical_pelvis_xz at frame 0 is (0, 0) by definition (HumanML3D
    canonicalization), so T_xz = world_pelvis_xz at frame 0. R_y_angle
    comes from the hip-line direction match.
    """
    import torch as _torch
    import piano.models.backbones.momask_adapter  # noqa: F401 — sys.path side-effect
    from utils.motion_process import recover_from_ric

    canonical_joints = (
        recover_from_ric(
            _torch.from_numpy(motion_263).float().unsqueeze(0),
            joints_num=22,
        )
        .squeeze(0).cpu().numpy().astype(np.float32)
    )
    world_t0 = joints_world[0]                     # (22, 3)
    canon_t0 = canonical_joints[0]                  # (22, 3)
    T_xz = world_t0[0, [0, 2]].copy()               # (2,) — pelvis xz
    hip_world = world_t0[2] - world_t0[1]
    hip_canon = canon_t0[2] - canon_t0[1]
    angle_world = float(math.atan2(hip_world[0], hip_world[2]))
    angle_canon = float(math.atan2(hip_canon[0], hip_canon[2]))
    R_y_angle = angle_world - angle_canon
    return R_y_angle, T_xz.astype(np.float32)


# ============================================================================
# VQ roundtrip per clip (matches the diagnose_vq_pipeline NORM path)
# ============================================================================

@torch.no_grad()
def _vq_roundtrip_motion(
    motion_raw: np.ndarray,                # (T, 263) raw HumanML3D
    vq_model: torch.nn.Module,
    motion_mean: np.ndarray,
    motion_std: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Normalize → VQ encode (all RVQ layers) → decode → denormalize."""
    motion_norm = (motion_raw - motion_mean) / np.clip(motion_std, 1e-8, None)
    motion_t = torch.from_numpy(motion_norm).float().unsqueeze(0).to(device)
    code_idx, _ = vq_model.encode(motion_t)             # (1, S, Q)
    decoded = vq_model.forward_decoder(code_idx)         # (1, T, 263) normalized
    decoded_np = decoded.squeeze(0).cpu().numpy().astype(np.float32)
    decoded_denorm = decoded_np * motion_std + motion_mean
    return decoded_denorm.astype(np.float32)


# ============================================================================
# Save in visualize_motion-compatible format
# ============================================================================

def _pad_to_max_T(arr: np.ndarray, T_max: int) -> np.ndarray:
    """Right-pad first axis with zeros to length T_max."""
    if arr.shape[0] >= T_max:
        return arr[:T_max]
    pad_width = [(0, T_max - arr.shape[0])] + [(0, 0)] * (arr.ndim - 1)
    return np.pad(arr, pad_width, mode="constant", constant_values=0)


def _save_dir(
    out_dir: Path,
    motions: list[np.ndarray],                       # list of (T_clip, 263)
    object_pcs: list[np.ndarray],                    # list of (1024, 3)
    object_positions: list[np.ndarray],              # list of (T_clip, 3)
    object_rotations: list[np.ndarray],              # list of (T_clip, 3)
    world_R_y_angles: list[float],
    world_T_xzs: list[np.ndarray],                   # list of (2,)
    seq_ids: list[str],
    texts: list[str],
    seq_lens_frames: list[int],
) -> None:
    """Write generated.npz + summary.json under ``out_dir``, matching the
    schema qual_eval._save_condition_dir produces (so the existing
    visualize_motion ``generated`` subcommand can render this directly).
    """
    ensure_dir(out_dir)
    T_max = max(len(m) for m in motions)
    motion_arr = np.stack([_pad_to_max_T(m, T_max) for m in motions], axis=0)
    obj_pos_arr = np.stack(
        [_pad_to_max_T(p, T_max) for p in object_positions], axis=0,
    )
    obj_rot_arr = np.stack(
        [_pad_to_max_T(r, T_max) for r in object_rotations], axis=0,
    )
    obj_pc_arr = np.stack(object_pcs, axis=0)
    R_y_arr = np.array(world_R_y_angles, dtype=np.float32)
    T_xz_arr = np.stack(world_T_xzs, axis=0).astype(np.float32)

    np.savez(
        out_dir / "generated.npz",
        motion_263=motion_arr,
        object_pc=obj_pc_arr,
        object_positions=obj_pos_arr,
        object_rotations=obj_rot_arr,
        world_R_y_angle=R_y_arr,
        world_T_xz=T_xz_arr,
        seq_lens=np.array(seq_lens_frames, dtype=np.int32),
    )
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "seq_ids": seq_ids,
                "texts": texts,
                # Key MUST be "seq_lens" — that's what
                # ``visualize_motion.load_generated_samples`` reads. With the
                # wrong key, the visualizer falls back to motion.shape[1]
                # (= padded T_max) and renders zero-padded frames as a
                # collapsed body at the end of the mp4.
                "seq_lens": seq_lens_frames,
            },
            f, indent=2,
        )


# ============================================================================
# Main
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path,
                        default=Path("configs/training/generator_v04_normalize.yaml"))
    parser.add_argument("--num-clips", type=int, default=20,
                        help="match qual_eval default to compare same clips.")
    parser.add_argument("--seed", type=int, default=42,
                        help="match qual_eval seed to hit identical sample.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", "-o", type=Path,
                        default=Path("runs/eval/stageB_v0_4_gt_roundtrip"))
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    cfg = OmegaConf.load(args.config)
    model_cfg = OmegaConf.load(cfg.model.config)

    print(f"Loading frozen MoMask VQ-VAE...")
    vq_model = load_momask_vqvae(
        cfg.model.checkpoints.vq_vae,
        input_width=model_cfg.vq_vae.input_width,
        nb_code=model_cfg.vq_vae.nb_code,
        code_dim=model_cfg.vq_vae.code_dim,
        output_emb_width=model_cfg.vq_vae.code_dim,
        down_t=model_cfg.vq_vae.down_t,
        stride_t=model_cfg.vq_vae.stride_t,
        width=model_cfg.vq_vae.width,
        depth=model_cfg.vq_vae.depth,
        dilation_growth_rate=model_cfg.vq_vae.dilation_growth_rate,
        num_quantizers=model_cfg.vq_vae.num_quantizers,
        device=str(device),
    )
    motion_mean, motion_std = load_motion_stats(cfg.model.checkpoints.vq_vae)
    print(f"  motion stats: mean.shape={motion_mean.shape}")

    val_dataset = _build_val_dataset(cfg)
    sampled_idx = select_eval_clip_indices(
        val_dataset,
        args.num_clips,
        seed=args.seed,
    )
    selected_rows = describe_eval_clip_selection(val_dataset, sampled_idx)
    print(f"Sampling {len(sampled_idx)} stratified clips from {len(val_dataset)} val: {sampled_idx}")
    for row in selected_rows:
        print(
            "  "
            f"idx={row['index']} subset={row['subset']} "
            f"object={row['object_id']} seq={row['seq_id']}",
        )

    samples = [val_dataset[i] for i in sampled_idx]
    seq_lens_frames = [int(s["seq_len"].item()) for s in samples]
    seq_ids = [str(s["seq_id"]) for s in samples]
    texts = [str(s["text"]) for s in samples]

    # VQ stride-4 truncation: encode/decode round-trip emits floor(T/4)*4
    # frames. Truncate seq_len before any per-clip processing to keep all
    # arrays length-consistent.
    TOKEN_STRIDE = 4
    seq_lens_frames_trunc = [(L // TOKEN_STRIDE) * TOKEN_STRIDE for L in seq_lens_frames]

    motions_orig: list[np.ndarray] = []
    motions_rt: list[np.ndarray] = []
    object_pcs: list[np.ndarray] = []
    object_positions: list[np.ndarray] = []
    object_rotations: list[np.ndarray] = []
    R_y_list: list[float] = []
    T_xz_list: list[np.ndarray] = []

    for i, sample in enumerate(samples):
        T_full = seq_lens_frames[i]
        T_use = seq_lens_frames_trunc[i]
        if T_use < TOKEN_STRIDE:
            print(f"  [skip] clip {i} ({seq_ids[i]}): seq_len {T_full} < {TOKEN_STRIDE}")
            continue

        motion_raw = sample["motion"].numpy().astype(np.float32)[:T_use]
        joints_world = sample["joints"].numpy().astype(np.float32)[:T_use]
        obj_pc = sample["object_pc"].numpy().astype(np.float32)
        obj_pos = sample["object_positions"].numpy().astype(np.float32)[:T_use]
        obj_rot = sample["object_rotations"].numpy().astype(np.float32)[:T_use]

        # GT motion already in canonical scale (HOIDataset returns raw
        # process_file output). Just use as-is for gt_original.
        motions_orig.append(motion_raw)

        # Roundtrip — normalize → encode → decode → denormalize.
        motion_rt = _vq_roundtrip_motion(
            motion_raw, vq_model, motion_mean, motion_std, device,
        )
        motions_rt.append(motion_rt)

        # Object overlay + world transform from this clip.
        object_pcs.append(obj_pc)
        object_positions.append(obj_pos)
        object_rotations.append(obj_rot)
        R_y, T_xz = _get_canon_to_world_transform(joints_world, motion_raw)
        R_y_list.append(R_y)
        T_xz_list.append(T_xz)

        print(
            f"  {i+1}/{len(samples)} {seq_ids[i]:<32} T={T_use}  "
            f"orig||={np.linalg.norm(motion_raw):8.2f}  "
            f"rt||={np.linalg.norm(motion_rt):8.2f}  "
            f"R_y={R_y:+.3f}  T_xz=({T_xz[0]:+.2f},{T_xz[1]:+.2f})"
        )

    if not motions_orig:
        print("ERROR: no usable clips after truncation.")
        return 1

    # Save both subdirs.
    print(f"\nSaving to {args.output_dir}/{{gt_original,gt_roundtrip}}/")
    _save_dir(
        args.output_dir / "gt_original",
        motions_orig, object_pcs, object_positions, object_rotations,
        R_y_list, T_xz_list, seq_ids, texts,
        [m.shape[0] for m in motions_orig],
    )
    _save_dir(
        args.output_dir / "gt_roundtrip",
        motions_rt, object_pcs, object_positions, object_rotations,
        R_y_list, T_xz_list, seq_ids, texts,
        [m.shape[0] for m in motions_rt],
    )
    print(f"\nDone. Render mp4s with:")
    print(f"  for cond in gt_original gt_roundtrip; do")
    print(f"      python -m piano.inference.visualize_motion generated \\")
    print(f"          --run-dir {args.output_dir}/$cond \\")
    print(f"          --output-dir {args.output_dir}/$cond/vis")
    print(f"  done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
