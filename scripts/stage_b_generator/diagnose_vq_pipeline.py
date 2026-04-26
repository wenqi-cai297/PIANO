"""Stage B v0.3-β: VQ-VAE round-trip diagnostic — encoder normalization + codebook capacity.

After the v0.3-α + denorm-bug discovery (analyses/2026-04-27_v0_3_root_cause_research.md
§"Denormalization-bug discovery"), v0.2 and v0.3-α both showed body still in-place
post-denorm. Two remaining suspects in the VQ pipeline:

1. **Encoder normalization (suspected bug)**: MoMask VQ-VAE was trained on
   ``motion = (raw - mean) / std`` normalized features
   (verified at ``src/piano/models/backbones/momask/data/t2m_dataset.py:85``),
   but our pipeline (HOIDataset → train_generator step_fn → ``vq_model.encode``)
   feeds raw motion. If encoder operates on out-of-distribution scale, it
   produces token IDs that don't match what the pretrained MaskTransformer
   was taught those IDs mean → text → token prior is broken → generated
   motion lacks coherent trajectory.

2. **Codebook capacity (Hypothesis C)**: even with correct normalization, the
   512-code HumanML3D-trained codebook may lack tokens for "drift body root
   to obj_com over T frames". InterMask retrained VQ-VAE on InterHuman/X for
   this reason.

This script tests both at once via the **VQ round-trip**: take GT motion,
encode → decode → recover joints, compare to GT joints. Two paths:

- **Path RAW** (current pipeline): ``vq.encode(raw_motion)`` → decode → denorm
- **Path NORM** (proposed fix): ``vq.encode((raw - mean) / std)`` → decode → denorm

For each path, measure root-xz drift over T frames. Compare to GT root drift.

Decision matrix:

- Both paths preserve GT drift ≈ → encoder normalization doesn't matter; codebook
  also fine for HOI; the v0.3 visual failure is in the MaskTransformer's
  *predictions*, not in the VQ pipeline. Move to v0.3-δ (architecture).
- Path NORM preserves GT drift, Path RAW doesn't → **encoder bug is real**.
  Fix: normalize motion at training and inference time, retrain Stage B.
- Both paths lose GT drift → **codebook can't represent HOI translation**.
  Hypothesis C confirmed; need v0.3-γ (VQ-VAE retrain on InterAct + HumanML3D).

Pure measurement — no training, no checkpoint writes. Reads val data + the
frozen MoMask VQ-VAE. ~30 sec on 1 A6000.

Usage::

    python scripts/stage_b_generator/diagnose_vq_pipeline.py \\
        --config configs/training/generator.yaml \\
        --num-clips 20 \\
        --output-dir runs/eval/stageB_v0_3_beta_vq
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import Tensor
from torch.utils.data import ConcatDataset

from piano.data.dataset import HOIDataset
from piano.data.split import build_subject_split, extract_subject_id
from piano.models.backbones.momask_adapter import load_momask_vqvae
from piano.utils.io_utils import ensure_dir, load_json


# ============================================================================
# Dataset assembly (same val bucket as train_generator + qual_eval)
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
    datasets = []
    for entry in cfg.data.datasets:
        ds = HOIDataset(
            root=entry.root,
            pseudo_label_dir=pseudo_label_dir,
            max_seq_length=cfg.data.max_seq_length,
            subject_id_filter=val_filter,
            augment=None,
            surface_obj_pose=False,    # don't need it for VQ round-trip
        )
        datasets.append(ds)
    return ConcatDataset(datasets)


# ============================================================================
# Stats loading (same convention as qual_eval._load_motion_stats)
# ============================================================================

def _load_motion_stats(vq_vae_ckpt: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load HumanML3D mean/std from MoMask co-located meta dir.

    Convention: ``<vq_vae_root>/meta/{mean,std}.npy``. The ``vq_vae_ckpt`` is
    the .tar inside ``<root>/model/``, so ``parent.parent`` gives ``<root>/``.
    """
    vq_vae_dir = Path(vq_vae_ckpt).parent.parent
    mean_path = vq_vae_dir / "meta" / "mean.npy"
    std_path = vq_vae_dir / "meta" / "std.npy"
    if not mean_path.exists() or not std_path.exists():
        raise FileNotFoundError(
            f"HumanML3D motion stats not found at {vq_vae_dir / 'meta'}.",
        )
    return (
        np.load(mean_path).astype(np.float32),
        np.load(std_path).astype(np.float32),
    )


# ============================================================================
# VQ round-trip + joint recovery
# ============================================================================

@torch.no_grad()
def _vq_roundtrip(
    vq_model: torch.nn.Module,
    motion: Tensor,                  # (1, T, 263) float32 on device
) -> np.ndarray:
    """Encode → decode through ALL quantizer layers; returns the decoded
    (1, T, 263) motion in the encoder's input scale (i.e. normalized if input
    was normalized, raw if input was raw).
    """
    code_idx, _ = vq_model.encode(motion)         # (1, S, Q)
    decoded = vq_model.forward_decoder(code_idx)  # (1, T, 263)
    return decoded.squeeze(0).detach().cpu().numpy()


def _recover_joints(motion_263: np.ndarray) -> np.ndarray:
    """``recover_from_ric`` wrapper that returns (T, 22, 3) joints in the
    canonical-frame, real-world-units coordinate system MoMask uses.

    Lazy MoMask path setup (same dance as
    ``HOIDataset._compute_canonical_object_pose``).
    """
    import torch as _torch
    import piano.models.backbones.momask_adapter  # noqa: F401 — sys.path side-effect
    from utils.motion_process import recover_from_ric

    joints = recover_from_ric(
        _torch.from_numpy(motion_263).float().unsqueeze(0),
        joints_num=22,
    )
    return joints.squeeze(0).cpu().numpy().astype(np.float32)


# ============================================================================
# Per-clip metrics
# ============================================================================

def _root_xz_drift(joints: np.ndarray, valid_T: int) -> dict[str, float]:
    """Root xz drift summary over the first ``valid_T`` frames.

    Reports:
      - max_drift:    max ||(root_xz_t - root_xz_0)||_2 over t
      - end_drift:    ||(root_xz_{T-1} - root_xz_0)||_2 (final displacement)
      - path_length:  sum of per-frame xz step lengths (total path travelled)
    """
    root_xz = joints[:valid_T, 0, [0, 2]]                      # (T, 2)
    if len(root_xz) < 2:
        return {"max_drift": 0.0, "end_drift": 0.0, "path_length": 0.0}
    rel = root_xz - root_xz[0:1]                               # (T, 2)
    norms = np.linalg.norm(rel, axis=-1)                       # (T,)
    step_norms = np.linalg.norm(np.diff(root_xz, axis=0), axis=-1)
    return {
        "max_drift": float(norms.max()),
        "end_drift": float(norms[-1]),
        "path_length": float(step_norms.sum()),
    }


def _per_clip_diagnose(
    sample: dict[str, Any],
    vq_model: torch.nn.Module,
    motion_mean: np.ndarray,
    motion_std: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    """Compare GT vs raw-encode-roundtrip vs norm-encode-roundtrip for one clip.

    All three motions get ``recover_from_ric``'d so root-xz drift is
    measured in the same coordinate space. Round-trip motions are
    denormalised back to raw scale before recovery (since the decoder
    output is in normalised space when input was normalised).
    """
    motion_raw = sample["motion"].numpy().astype(np.float32)         # (T, 263) raw
    seq_len = int(sample["seq_len"].item())
    valid_T = min(seq_len, motion_raw.shape[0])
    # MoMask VQ-VAE is stride-4 (down_t=2, stride_t=2 → 2² = 4), so the
    # decoder always emits ``floor(T/4) * 4`` frames. Truncate GT to the
    # same length so all three round-trip arrays have matching first axis
    # (otherwise the L2 / per-frame comparisons hit a shape mismatch when
    # seq_len isn't a multiple of 4).
    TOKEN_STRIDE = 4
    valid_T = (valid_T // TOKEN_STRIDE) * TOKEN_STRIDE
    if valid_T < TOKEN_STRIDE:
        # Degenerate clip (< 4 frames). Skip with zero-drift entries.
        return {
            "seq_id": str(sample.get("seq_id", "?")),
            "subset": str(sample.get("subset", "?")),
            "valid_T": 0,
            "gt": {"max_drift": 0.0, "end_drift": 0.0, "path_length": 0.0},
            "roundtrip_raw_encode": {"max_drift": 0.0, "end_drift": 0.0, "path_length": 0.0},
            "roundtrip_norm_encode": {"max_drift": 0.0, "end_drift": 0.0, "path_length": 0.0},
            "l2_decoded_vs_gt_motion": {"raw_encode": 0.0, "norm_encode": 0.0},
            "ratios": {
                "raw_max_drift_ratio": 0.0, "norm_max_drift_ratio": 0.0,
                "raw_path_ratio": 0.0, "norm_path_ratio": 0.0,
            },
            "skipped": True,
        }

    # Path 0 — GT (no encode at all, just recover joints from raw motion).
    gt_joints = _recover_joints(motion_raw[:valid_T])
    gt_drift = _root_xz_drift(gt_joints, valid_T)

    # The VQ-VAE decoder's weights are fixed — it ALWAYS produces output
    # in the training-distribution scale (normalized HumanML3D), regardless
    # of what scale we fed the encoder. So both paths must denorm the
    # decoder output (`x * std + mean`) before recover_from_ric, which
    # expects raw HumanML3D scale.

    # Path RAW — current pipeline: encode(raw) → wrong codes → decode →
    # output in normalized scale → denorm → joints. The "wrongness" here
    # is in the codebook quantization, not in the decoder output scale.
    motion_t = torch.from_numpy(motion_raw[:valid_T]).float().unsqueeze(0).to(device)
    decoded_raw_norm = _vq_roundtrip(vq_model, motion_t)
    decoded_raw = decoded_raw_norm * motion_std + motion_mean
    raw_joints = _recover_joints(decoded_raw)
    raw_drift = _root_xz_drift(raw_joints, valid_T)

    # Path NORM — proposed fix: encode((raw - mean) / std) → correct codes
    # → decode → output in normalized scale → denorm → joints.
    motion_norm = (motion_raw[:valid_T] - motion_mean) / motion_std
    motion_norm_t = torch.from_numpy(motion_norm).float().unsqueeze(0).to(device)
    decoded_norm_norm = _vq_roundtrip(vq_model, motion_norm_t)
    decoded_norm = decoded_norm_norm * motion_std + motion_mean
    norm_joints = _recover_joints(decoded_norm)
    norm_drift = _root_xz_drift(norm_joints, valid_T)

    # L2 motion-space distance (raw-scale) per path vs GT — secondary
    # signal beside drift. Mean per-frame ||motion - GT|| in 263-d.
    raw_l2 = float(np.sqrt(((decoded_raw - motion_raw[:valid_T]) ** 2).mean()))
    norm_l2 = float(np.sqrt(((decoded_norm - motion_raw[:valid_T]) ** 2).mean()))

    return {
        "seq_id": str(sample.get("seq_id", "?")),
        "subset": str(sample.get("subset", "?")),
        "valid_T": valid_T,
        "gt": gt_drift,
        "roundtrip_raw_encode": raw_drift,
        "roundtrip_norm_encode": norm_drift,
        "l2_decoded_vs_gt_motion": {
            "raw_encode": raw_l2,
            "norm_encode": norm_l2,
        },
        "ratios": {
            # roundtrip / GT — 1.0 = perfect preservation.
            "raw_max_drift_ratio":  raw_drift["max_drift"] / max(gt_drift["max_drift"], 1e-6),
            "norm_max_drift_ratio": norm_drift["max_drift"] / max(gt_drift["max_drift"], 1e-6),
            "raw_path_ratio":       raw_drift["path_length"] / max(gt_drift["path_length"], 1e-6),
            "norm_path_ratio":      norm_drift["path_length"] / max(gt_drift["path_length"], 1e-6),
        },
    }


# ============================================================================
# Verdict
# ============================================================================

def _verdict(per_clip: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply decision matrix from the docstring.

    Aggregates the per-clip max-drift and path-length preservation ratios
    across clips, then maps to one of three categories. Skipped clips
    (too short for VQ stride-4) and zero-GT-drift clips (static poses
    where the ratio is ill-defined) are filtered before aggregation.
    """
    usable = [
        c for c in per_clip
        if not c.get("skipped", False) and c["gt"]["max_drift"] > 0.05
    ]
    if not usable:
        return {
            "category": "no_valid_clips",
            "raw_encode_max_drift_ratio_median": 0.0,
            "norm_encode_max_drift_ratio_median": 0.0,
            "raw_encode_path_length_ratio_median": 0.0,
            "norm_encode_path_length_ratio_median": 0.0,
            "n_usable_clips": 0,
            "n_total_clips": len(per_clip),
            "recommendation": (
                "All sampled clips were either too short (< 4 frames) or "
                "had near-zero GT root drift (static interactions). Re-run "
                "with --num-clips 50 or seed-shuffle to get clips with "
                "meaningful translation."
            ),
        }
    raw_max_ratios = np.array([c["ratios"]["raw_max_drift_ratio"] for c in usable])
    norm_max_ratios = np.array([c["ratios"]["norm_max_drift_ratio"] for c in usable])
    raw_path_ratios = np.array([c["ratios"]["raw_path_ratio"] for c in usable])
    norm_path_ratios = np.array([c["ratios"]["norm_path_ratio"] for c in usable])

    raw_max_med = float(np.median(raw_max_ratios))
    norm_max_med = float(np.median(norm_max_ratios))
    raw_path_med = float(np.median(raw_path_ratios))
    norm_path_med = float(np.median(norm_path_ratios))

    # Heuristic thresholds — preservation > 0.7 is "OK"; < 0.3 is "lost".
    norm_ok = norm_max_med > 0.7 and norm_path_med > 0.7
    raw_ok = raw_max_med > 0.7 and raw_path_med > 0.7

    if norm_ok and raw_ok:
        cat = "vq_pipeline_clean"
        rec = (
            "Both raw-input and normalized-input encoding round-trip preserves "
            "GT root drift (median ratio > 0.7). Encoder normalization is NOT "
            "the bug. Codebook is NOT the bottleneck for translation either. "
            "The visual failure is in the MaskTransformer's predictions — "
            "advance to v0.3-δ (trainable-copy InterControl rebuild) or "
            "investigate train/inference distribution mismatch (Hypothesis F)."
        )
    elif norm_ok and not raw_ok:
        cat = "encoder_normalization_bug"
        rec = (
            "Normalized-input encoding preserves GT drift (median > 0.7) but "
            "raw-input encoding does NOT (median < 0.7). The encoder was "
            "trained on (raw - mean) / std normalized inputs; feeding raw "
            "OOD-scales the input → wrong codebook quantization → token IDs "
            "no longer aligned with what pretrained MaskTransformer learned "
            "those IDs mean. **Fix: normalize motion at training + inference, "
            "retrain Stage B (~13 min).** Run v0.3-β-norm next."
        )
    elif not norm_ok and not raw_ok:
        cat = "codebook_lacks_translation"
        rec = (
            "Both paths LOSE GT drift (median < 0.7). The HumanML3D-trained "
            "codebook can't represent the HOI motion's root translation. "
            "Hypothesis C (codebook bottleneck) confirmed. Fix: retrain "
            "RVQ-VAE on InterAct + HumanML3D union (v0.3-γ, ~3-4h VQ + "
            "~2 days MaskTransformer)."
        )
    else:  # raw_ok but not norm_ok — extremely unlikely
        cat = "anomalous_norm_loses_raw_keeps"
        rec = (
            "Raw-input encoding preserves drift but normalized doesn't — this "
            "is unexpected; check stats files (mean.npy / std.npy) for "
            "corruption or shape mismatch."
        )
    return {
        "category": cat,
        "raw_encode_max_drift_ratio_median": raw_max_med,
        "norm_encode_max_drift_ratio_median": norm_max_med,
        "raw_encode_path_length_ratio_median": raw_path_med,
        "norm_encode_path_length_ratio_median": norm_path_med,
        "n_usable_clips": len(usable),
        "n_total_clips": len(per_clip),
        "recommendation": rec,
    }


# ============================================================================
# Pretty-print
# ============================================================================

def _format_summary(result: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"\n=== Stage B v0.3-β VQ pipeline diagnostic ===")
    lines.append(f"  num_clips:   {result['num_clips']}")
    lines.append(f"  config:      {result['config']}")
    lines.append("")

    lines.append(
        f"  {'#':>2} {'seq_id':<32} {'GT_max':>7} {'RAW_max':>8} {'NORM_max':>9} "
        f"{'GT_path':>8} {'RAW_path':>9} {'NORM_path':>10}"
    )
    lines.append(f"  {'-' * 88}")
    for i, c in enumerate(result["per_clip"]):
        sid = c["seq_id"][:32]
        gtm = c["gt"]["max_drift"]
        gtp = c["gt"]["path_length"]
        rawm = c["roundtrip_raw_encode"]["max_drift"]
        rawp = c["roundtrip_raw_encode"]["path_length"]
        normm = c["roundtrip_norm_encode"]["max_drift"]
        normp = c["roundtrip_norm_encode"]["path_length"]
        lines.append(
            f"  {i:>2} {sid:<32} {gtm:>7.3f} {rawm:>8.3f} {normm:>9.3f} "
            f"{gtp:>8.3f} {rawp:>9.3f} {normp:>10.3f}"
        )

    v = result["verdict"]
    lines.append("")
    lines.append(f"  Median preservation ratios (round-trip / GT):")
    lines.append(
        f"    RAW encode  — max_drift: {v['raw_encode_max_drift_ratio_median']:.3f}, "
        f"path_length: {v['raw_encode_path_length_ratio_median']:.3f}"
    )
    lines.append(
        f"    NORM encode — max_drift: {v['norm_encode_max_drift_ratio_median']:.3f}, "
        f"path_length: {v['norm_encode_path_length_ratio_median']:.3f}"
    )
    lines.append("")
    lines.append(f"  Verdict:        {v['category']}")
    lines.append(f"  Recommendation:")
    for line in v["recommendation"].split(". "):
        if line.strip():
            lines.append(f"    {line.strip().rstrip('.')}.")
    return "\n".join(lines)


# ============================================================================
# Main
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/training/generator.yaml"))
    parser.add_argument("--num-clips", type=int, default=20,
                        help="number of val clips to round-trip (default 20).")
    parser.add_argument("--seed", type=int, default=42,
                        help="seed for clip sampling.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", "-o", type=Path,
                        default=Path("runs/eval/stageB_v0_3_beta_vq"))
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    cfg = OmegaConf.load(args.config)
    model_cfg = OmegaConf.load(cfg.model.config)

    print(f"Loading frozen MoMask VQ-VAE from {cfg.model.checkpoints.vq_vae} ...")
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

    motion_mean, motion_std = _load_motion_stats(cfg.model.checkpoints.vq_vae)
    print(f"  mean.shape={motion_mean.shape} std.shape={motion_std.shape}")
    print(f"  mean range [{motion_mean.min():.3f}, {motion_mean.max():.3f}]")
    print(f"  std range  [{motion_std.min():.3f}, {motion_std.max():.3f}]")

    print(f"Building val dataset ...")
    val_dataset = _build_val_dataset(cfg)
    print(f"  val: {len(val_dataset)} clips total; sampling {args.num_clips}.")

    rng = np.random.default_rng(args.seed)
    pool = list(range(len(val_dataset)))
    rng.shuffle(pool)
    sampled = pool[: args.num_clips]

    print(f"Round-tripping {len(sampled)} clips ...")
    per_clip: list[dict[str, Any]] = []
    for i, idx in enumerate(sampled):
        sample = val_dataset[idx]
        info = _per_clip_diagnose(sample, vq_model, motion_mean, motion_std, device)
        per_clip.append(info)
        if (i + 1) % 5 == 0 or i == len(sampled) - 1:
            print(f"  {i+1}/{len(sampled)} done")

    result: dict[str, Any] = {
        "config": str(args.config),
        "num_clips": len(per_clip),
        "per_clip": per_clip,
        "verdict": _verdict(per_clip),
    }
    report = _format_summary(result)
    print(report)

    ensure_dir(args.output_dir)
    summary_txt = args.output_dir / "summary.txt"
    summary_json = args.output_dir / "summary.json"
    with summary_txt.open("w", encoding="utf-8") as f:
        f.write(report)
        f.write("\n")
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote:")
    print(f"  {summary_txt}")
    print(f"  {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
