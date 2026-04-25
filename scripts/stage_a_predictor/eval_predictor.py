"""Evaluate a Stage A predictor checkpoint on the held-out object split.

Loads the predictor + object_encoder from a checkpoint, runs forward on
the `val` (or `test`, or combined `val+test`) bucket of the config's
object-id split, and reports:

    - Supervision losses (same formula as training, label_smoothing
      included) — comparable to the train-time final values so you can
      quantify the overfitting gap.
    - Contact F1 per body part (sigmoid > 0.5 threshold).
    - Target xyz regression: mean L2 error (cm) and
      percent-within-threshold (5cm / 10cm / 20cm), gated by
      gt_contact > 0.5. Per body part.
    - Phase accuracy, macro-F1, per-class precision/recall/F1, and the
      full confusion matrix.
    - Support same as phase.

Writes a JSON summary next to the checkpoint. Prints a readable table.

Usage:
    python scripts/stage_a_predictor/eval_predictor.py \\
        --config configs/training/predictor.yaml \\
        --checkpoint runs/training/predictor/final.pt \\
        --split val \\
        --output runs/eval/predictor_val.json

Runs on a single GPU (no Accelerate). Uses bf16 autocast to match
training precision; no gradient.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, DataLoader

from piano.data.dataset import (
    AugmentConfig,
    HOIDataset,
    build_object_split,
    collate_hoi,
)
from piano.data.pseudo_labels.extract_phase import PHASE_NAMES
from piano.models.interaction_predictor import InteractionPredictor
from piano.models.object_encoder import ObjectEncoder
from piano.training.losses import PredictorLoss
from piano.utils.clip_utils import encode_text_per_token, load_clip_text_encoder
from piano.utils.io_utils import load_json


# ---------------------------------------------------------------------------
# Split-aware dataset assembly (mirrors train_predictor._build_dataset but
# evaluates on a single named bucket, no augmentation)
# ---------------------------------------------------------------------------

def _collect_object_ids(roots) -> list[str]:
    seen: set[str] = set()
    for entry in roots:
        root = Path(entry.root)
        meta_path = root / "metadata_clean.json"
        if not meta_path.exists():
            meta_path = root / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata not found in {root}")
        for m in load_json(meta_path):
            if (obj := m.get("object_id")) is not None:
                seen.add(obj)
    return sorted(seen)


def _build_eval_dataset(cfg, split: str) -> tuple[ConcatDataset, dict[str, set[str]], set[str]]:
    """Return (ConcatDataset, full_split_map, allowed_object_ids)."""
    object_ids = _collect_object_ids(cfg.data.datasets)
    split_cfg = cfg.data.object_split
    splits = build_object_split(
        object_ids,
        train_pct=split_cfg.train_pct,
        val_pct=split_cfg.val_pct,
        test_pct=split_cfg.test_pct,
        seed=split_cfg.seed,
    )

    if split == "val+test":
        allowed = splits["val"] | splits["test"]
    elif split == "all":
        allowed = splits["train"] | splits["val"] | splits["test"]
    elif split in splits:
        allowed = splits[split]
    else:
        raise ValueError(f"unknown split {split!r}; expected train/val/test/val+test/all")

    pseudo_label_dir = cfg.data.get("pseudo_label_dir", None)
    datasets = [
        HOIDataset(
            root=entry.root,
            pseudo_label_dir=pseudo_label_dir,
            max_seq_length=cfg.data.max_seq_length,
            object_id_filter=allowed,
            augment=AugmentConfig(enabled=False),   # eval → deterministic
        )
        for entry in cfg.data.datasets
    ]
    return ConcatDataset(datasets), splits, allowed


# ---------------------------------------------------------------------------
# Model assembly
# ---------------------------------------------------------------------------

def _build_models(cfg, device: torch.device) -> tuple[InteractionPredictor, ObjectEncoder]:
    model_cfg = OmegaConf.load(cfg.model.config)
    obj_cfg = OmegaConf.load(cfg.model.object_encoder_config)

    predictor = InteractionPredictor(
        d_model=model_cfg.encoder.d_model,
        num_layers=model_cfg.encoder.num_layers,
        num_heads=model_cfg.encoder.num_heads,
        dim_feedforward=model_cfg.encoder.dim_feedforward,
        dropout=model_cfg.encoder.dropout,
        text_dim=model_cfg.input.text_dim,
        pose_dim=model_cfg.input.pose_dim,
        max_seq_length=model_cfg.sequence.max_length,
        num_body_parts=model_cfg.output.num_body_parts,
        num_object_patches=model_cfg.output.num_object_patches,
        num_phases=model_cfg.output.num_phases,
        num_support_states=model_cfg.output.num_support_states,
    ).to(device).eval()

    object_encoder = ObjectEncoder(
        num_input_points=obj_cfg.pointnet.num_input_points,
        num_output_tokens=obj_cfg.pointnet.num_output_tokens,
        feature_dim=obj_cfg.pointnet.feature_dim,
    ).to(device).eval()

    return predictor, object_encoder


def _load_checkpoint(
    ckpt_path: Path,
    predictor: InteractionPredictor,
    object_encoder: ObjectEncoder,
) -> dict:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model" not in ck:
        raise KeyError(f"checkpoint missing 'model' key; got {list(ck)}")
    predictor.load_state_dict(ck["model"])
    if "object_encoder" not in ck:
        raise KeyError(
            "checkpoint missing 'object_encoder' key — older trainer "
            "that didn't save peer modules. Re-train with the current "
            "code (commit 16127e9 or later)."
        )
    object_encoder.load_state_dict(ck["object_encoder"])
    return {"epoch": ck.get("epoch"), "global_step": ck.get("global_step")}


# ---------------------------------------------------------------------------
# Metric computation (manual, no sklearn — keeps the script dep-light)
# ---------------------------------------------------------------------------

def _binary_f1(pred_bool: np.ndarray, gt_bool: np.ndarray) -> dict[str, float]:
    tp = int((pred_bool & gt_bool).sum())
    fp = int((pred_bool & ~gt_bool).sum())
    fn = int((~pred_bool & gt_bool).sum())
    tn = int((~pred_bool & ~gt_bool).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc = (tp + tn) / max(tp + fp + fn + tn, 1)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": prec, "recall": rec, "f1": f1, "accuracy": acc,
    }


def _multiclass_metrics(
    pred_cls: np.ndarray, gt_cls: np.ndarray, num_classes: int, class_names: list[str],
) -> dict:
    """Per-class P/R/F1 + macro + weighted, plus confusion matrix."""
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(gt_cls, pred_cls):
        conf[int(t), int(p)] += 1

    per_class = {}
    f1s, weights = [], []
    for c in range(num_classes):
        tp = int(conf[c, c])
        fp = int(conf[:, c].sum() - tp)
        fn = int(conf[c, :].sum() - tp)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        support = int(conf[c, :].sum())
        per_class[class_names[c]] = {
            "precision": prec, "recall": rec, "f1": f1,
            "support": support,
        }
        f1s.append(f1)
        weights.append(support)

    total_support = sum(weights) or 1
    macro_f1 = float(np.mean(f1s))
    weighted_f1 = float(sum(f * w for f, w in zip(f1s, weights)) / total_support)
    overall_acc = float(np.diag(conf).sum() / max(conf.sum(), 1))

    return {
        "accuracy": overall_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": conf.tolist(),
        "class_names": class_names,
    }


# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_eval(
    cfg,
    ckpt_path: Path,
    split: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> dict:
    # Data
    dataset, splits_map, allowed = _build_eval_dataset(cfg, split)
    print(
        f"[split={split}] train={len(splits_map['train'])} "
        f"val={len(splits_map['val'])} test={len(splits_map['test'])} "
        f"objects in total → using {len(allowed)} objects, {len(dataset)} clips"
    )
    if len(dataset) == 0:
        raise RuntimeError(f"empty dataset for split={split!r}")

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_hoi, num_workers=num_workers, pin_memory=True,
    )

    # Models
    predictor, object_encoder = _build_models(cfg, device)
    meta = _load_checkpoint(ckpt_path, predictor, object_encoder)
    print(f"Loaded checkpoint: epoch={meta['epoch']} global_step={meta['global_step']}")

    # Frozen CLIP
    clip_model = load_clip_text_encoder(
        device=device, model_name=cfg.model.get("text_encoder", "ViT-B/32"),
    )

    # Loss object — same weights + label_smoothing as training, so
    # val losses are directly comparable to wandb's train numbers.
    criterion = PredictorLoss(
        contact_weight=cfg.loss.contact_weight,
        target_weight=cfg.loss.target_weight,
        phase_weight=cfg.loss.phase_weight,
        support_weight=cfg.loss.support_weight,
        label_smoothing=cfg.loss.get("label_smoothing", 0.0),
        focal_gamma=cfg.loss.get("focal_gamma", 0.0),
    )

    # Accumulators
    all_pred_contact: list[np.ndarray] = []
    all_gt_contact: list[np.ndarray] = []
    all_pred_target_xyz: list[np.ndarray] = []
    all_gt_target_xyz: list[np.ndarray] = []
    all_contact_gate: list[np.ndarray] = []              # where to evaluate target
    all_pred_phase: list[np.ndarray] = []
    all_gt_phase: list[np.ndarray] = []
    all_pred_support: list[np.ndarray] = []
    all_gt_support: list[np.ndarray] = []
    loss_sums = {
        "loss": 0.0, "loss_contact": 0.0, "loss_target": 0.0,
        "loss_phase": 0.0, "loss_support": 0.0,
    }
    n_loss_batches = 0
    total_valid_frames = 0
    total_clips = 0

    t0 = time.time()
    for batch_idx, batch in enumerate(dataloader):
        # Move tensors to device
        batch = {
            k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            text_features, text_mask = encode_text_per_token(
                clip_model, batch["text"], device,
            )
            obj_tokens = object_encoder(batch["object_pc"])
            B = batch["joints"].shape[0]
            init_pose = batch["joints"][:, 0, :, :].reshape(B, -1)
            max_T = batch["motion"].shape[1]
            pred = predictor(
                text_features, obj_tokens, init_pose,
                seq_length=max_T, text_key_padding_mask=text_mask,
            )

        seq_len = batch["seq_len"]
        frame_mask = (
            torch.arange(max_T, device=seq_len.device).unsqueeze(0)
            < seq_len.unsqueeze(1)
        )

        # --- losses (use same criterion; cast logits to fp32 for loss math)
        pred_fp32 = {k: (v.float() if isinstance(v, torch.Tensor) else v) for k, v in pred.items()}
        loss_dict = criterion(
            pred_fp32,
            gt_contact=batch["contact_state"],
            gt_target=batch["contact_target_xyz"],
            gt_phase=batch["phase"].long(),
            gt_support=batch["support"].long(),
            mask=frame_mask,
        )
        for k in loss_sums:
            loss_sums[k] += float(loss_dict[k].item())
        n_loss_batches += 1

        # --- predictions (fp32 numpy on CPU)
        mask_np = frame_mask.cpu().numpy()           # (B, T)
        valid = mask_np.reshape(-1)                  # (B*T,)

        # Contact: sigmoid(logits) > 0.5 vs gt > 0.5
        pred_contact = torch.sigmoid(pred_fp32["contact_logits"]).cpu().numpy()   # (B, T, 5)
        gt_contact_np = batch["contact_state"].float().cpu().numpy()              # (B, T, 5)
        pred_contact_v = pred_contact.reshape(-1, 5)[valid]
        gt_contact_v = gt_contact_np.reshape(-1, 5)[valid]
        all_pred_contact.append(pred_contact_v)
        all_gt_contact.append(gt_contact_v)

        # Target xyz regression (in object-local metres). Predicted vs
        # GT are (B, T, 5, 3); we flatten to (B*T, 5, 3) and gate by
        # gt_contact > 0.5 in the metrics step.
        pred_txyz = pred_fp32["contact_target_xyz"].cpu().numpy()                 # (B, T, 5, 3)
        gt_txyz = batch["contact_target_xyz"].float().cpu().numpy()               # (B, T, 5, 3)
        all_pred_target_xyz.append(pred_txyz.reshape(-1, 5, 3)[valid])
        all_gt_target_xyz.append(gt_txyz.reshape(-1, 5, 3)[valid])
        all_contact_gate.append(gt_contact_v > 0.5)

        # Phase
        pred_phase = pred_fp32["phase_logits"].argmax(dim=-1).cpu().numpy()            # (B, T)
        gt_phase = batch["phase"].long().cpu().numpy()
        all_pred_phase.append(pred_phase.reshape(-1)[valid])
        all_gt_phase.append(gt_phase.reshape(-1)[valid])

        # Support
        pred_support = pred_fp32["support_logits"].argmax(dim=-1).cpu().numpy()
        gt_support = batch["support"].long().cpu().numpy()
        all_pred_support.append(pred_support.reshape(-1)[valid])
        all_gt_support.append(gt_support.reshape(-1)[valid])

        total_valid_frames += int(valid.sum())
        total_clips += B
        if batch_idx % 10 == 0:
            print(f"  batch {batch_idx+1}/{len(dataloader)}  ({B} clips, {int(mask_np.sum())} valid frames)")

    elapsed = time.time() - t0
    print(f"Eval done in {elapsed:.1f} s on {total_clips} clips / {total_valid_frames} valid frames")

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------
    pc = np.concatenate(all_pred_contact, axis=0)             # (N_frames, 5)
    gc = np.concatenate(all_gt_contact, axis=0)
    pred_bin = pc > 0.5
    gt_bin = gc > 0.5

    body_parts = ["left_hand", "right_hand", "left_foot", "right_foot", "pelvis"]
    contact_per_part = {
        body_parts[b]: _binary_f1(pred_bin[:, b], gt_bin[:, b]) for b in range(5)
    }
    # Macro across body parts
    contact_macro_f1 = float(np.mean([contact_per_part[b]["f1"] for b in body_parts]))
    contact_any_f1 = _binary_f1(pred_bin.any(-1), gt_bin.any(-1))

    # Target xyz regression, gated by gt_contact > 0.5. L2 error in
    # metres of predicted xyz vs GT xyz (both in object-local frame).
    # Also % within 5 / 10 / 20 cm thresholds.
    pt_xyz = np.concatenate(all_pred_target_xyz, axis=0)    # (N, 5, 3)
    gt_xyz = np.concatenate(all_gt_target_xyz, axis=0)      # (N, 5, 3)
    gate = np.concatenate(all_contact_gate, axis=0)         # (N, 5)
    err = np.linalg.norm(pt_xyz - gt_xyz, axis=-1)          # (N, 5)
    total_gated = int(gate.sum())
    if total_gated > 0:
        err_gated = err[gate]
        target_mean_l2_m = float(err_gated.mean())
        target_pct_5cm = float((err_gated < 0.05).mean())
        target_pct_10cm = float((err_gated < 0.10).mean())
        target_pct_20cm = float((err_gated < 0.20).mean())
    else:
        target_mean_l2_m = None
        target_pct_5cm = target_pct_10cm = target_pct_20cm = None
    target_per_part_xyz = {}
    for b, name in enumerate(body_parts):
        g = gate[:, b]
        n_g = int(g.sum())
        if n_g > 0:
            eb = err[g, b]
            target_per_part_xyz[name] = {
                "mean_l2_m": float(eb.mean()),
                "median_l2_m": float(np.median(eb)),
                "pct_within_5cm": float((eb < 0.05).mean()),
                "pct_within_10cm": float((eb < 0.10).mean()),
                "pct_within_20cm": float((eb < 0.20).mean()),
                "support": n_g,
            }
        else:
            target_per_part_xyz[name] = {
                "mean_l2_m": None, "median_l2_m": None,
                "pct_within_5cm": None, "pct_within_10cm": None,
                "pct_within_20cm": None, "support": 0,
            }

    # Phase. Pull class names from the canonical PHASE_NAMES so the
    # eval is consistent with whatever extract_phase.py defines (3
    # classes as of v5; was 5 in v3-v4).
    ph_pred = np.concatenate(all_pred_phase, axis=0)
    ph_gt = np.concatenate(all_gt_phase, axis=0)
    phase_metrics = _multiclass_metrics(
        ph_pred, ph_gt, num_classes=len(PHASE_NAMES),
        class_names=list(PHASE_NAMES),
    )

    # Support
    su_pred = np.concatenate(all_pred_support, axis=0)
    su_gt = np.concatenate(all_gt_support, axis=0)
    support_metrics = _multiclass_metrics(
        su_pred, su_gt, num_classes=4,
        class_names=["both_feet", "single_foot", "sitting", "hand_support"],
    )

    # Losses → mean over batches
    loss_mean = {k: v / max(n_loss_batches, 1) for k, v in loss_sums.items()}

    report = {
        "checkpoint": str(ckpt_path),
        "epoch": meta["epoch"],
        "global_step": meta["global_step"],
        "split": split,
        "num_objects_total": len(splits_map["train"]) + len(splits_map["val"]) + len(splits_map["test"]),
        "num_objects_train": len(splits_map["train"]),
        "num_objects_val": len(splits_map["val"]),
        "num_objects_test": len(splits_map["test"]),
        "num_objects_used": len(allowed),
        "num_clips": total_clips,
        "num_valid_frames": total_valid_frames,
        "eval_time_sec": round(elapsed, 1),

        "loss": loss_mean,

        "contact": {
            "macro_f1_over_body_parts": contact_macro_f1,
            "any_part_f1": contact_any_f1,
            "per_body_part": contact_per_part,
        },
        "target": {
            "mean_l2_m_overall_gated": target_mean_l2_m,
            "pct_within_5cm_overall_gated": target_pct_5cm,
            "pct_within_10cm_overall_gated": target_pct_10cm,
            "pct_within_20cm_overall_gated": target_pct_20cm,
            "total_gated_frames_x_parts": total_gated,
            "per_body_part": target_per_part_xyz,
        },
        "phase": phase_metrics,
        "support": support_metrics,
    }
    return report


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------

def _print_report(r: dict) -> None:
    print()
    print("=" * 78)
    print(f"Stage A predictor eval — split={r['split']}")
    print(f"  ckpt: {r['checkpoint']}")
    print(f"  epoch={r['epoch']}  global_step={r['global_step']}")
    print(f"  objects: {r['num_objects_used']}/{r['num_objects_total']} "
          f"(train={r['num_objects_train']} val={r['num_objects_val']} test={r['num_objects_test']})")
    print(f"  clips: {r['num_clips']}   valid frames: {r['num_valid_frames']}")
    print("-" * 78)

    print("\n[supervision losses — compare with training wandb]")
    for k, v in r["loss"].items():
        print(f"  {k:<16s} {v:.4f}")

    print("\n[contact — per body part]")
    print(f"  macro-F1 across 5 body parts: {r['contact']['macro_f1_over_body_parts']:.4f}")
    print(f"  any-part F1 (frame has any contact): {r['contact']['any_part_f1']['f1']:.4f}")
    header = f"  {'body part':<14s} {'P':>7s} {'R':>7s} {'F1':>7s} {'gt_pos':>8s} {'pred_pos':>8s}"
    print(header)
    for bp, m in r["contact"]["per_body_part"].items():
        print(f"  {bp:<14s} {m['precision']:>7.4f} {m['recall']:>7.4f} {m['f1']:>7.4f} "
              f"{m['tp']+m['fn']:>8d} {m['tp']+m['fp']:>8d}")

    print("\n[target — xyz regression in object-local frame (gated by gt_contact > 0.5)]")
    t = r["target"]
    if t["mean_l2_m_overall_gated"] is None:
        print(f"  (no gated frames — nothing to evaluate)")
    else:
        print(f"  overall:  mean L2 = {t['mean_l2_m_overall_gated']*100:.1f} cm   "
              f"(<5cm {t['pct_within_5cm_overall_gated']*100:.1f}%, "
              f"<10cm {t['pct_within_10cm_overall_gated']*100:.1f}%, "
              f"<20cm {t['pct_within_20cm_overall_gated']*100:.1f}%)"
              f"   gated cells = {t['total_gated_frames_x_parts']}")
    header = f"  {'body part':<14s} {'L2_mean_cm':>12s} {'L2_med_cm':>11s} {'<5cm':>8s} {'<10cm':>8s} {'<20cm':>8s} {'support':>8s}"
    print(header)
    for bp, m in t["per_body_part"].items():
        if m["mean_l2_m"] is None:
            print(f"  {bp:<14s}  (no gated frames)")
            continue
        print(f"  {bp:<14s} {m['mean_l2_m']*100:>12.2f} {m['median_l2_m']*100:>11.2f} "
              f"{m['pct_within_5cm']*100:>7.1f}% {m['pct_within_10cm']*100:>7.1f}% "
              f"{m['pct_within_20cm']*100:>7.1f}% {m['support']:>8d}")

    for section_name, section in [("phase", r["phase"]), ("support", r["support"])]:
        print(f"\n[{section_name}]")
        print(f"  accuracy:     {section['accuracy']:.4f}")
        print(f"  macro-F1:     {section['macro_f1']:.4f}")
        print(f"  weighted-F1:  {section['weighted_f1']:.4f}")
        print(f"  per-class (precision / recall / f1 / support):")
        for cn in section["class_names"]:
            m = section["per_class"][cn]
            print(f"    {cn:<16s}  {m['precision']:>6.4f}  {m['recall']:>6.4f}  "
                  f"{m['f1']:>6.4f}  {m['support']:>7d}")
        print(f"  confusion matrix (rows=gt, cols=pred):")
        names_row = "                " + " ".join(f"{c[:9]:>10s}" for c in section["class_names"])
        print(names_row)
        for ci, cn in enumerate(section["class_names"]):
            row = "  " + f"{cn[:14]:<14s}" + " ".join(f"{v:>10d}" for v in section["confusion_matrix"][ci])
            print(row)
    print()
    print("=" * 78)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--split", type=str, default="val",
        choices=["train", "val", "test", "val+test", "all"],
        help="which object-id bucket to evaluate on (default: val)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--output", type=Path, default=None,
        help="JSON output path (default: <ckpt>.eval_<split>.json)",
    )
    args = parser.parse_args()

    if not args.config.exists():
        raise FileNotFoundError(f"config not found: {args.config}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    if args.output is None:
        args.output = args.checkpoint.with_suffix(f".eval_{args.split.replace('+','_')}.json")

    cfg = OmegaConf.load(str(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: no CUDA available, falling back to CPU (will be slow)")

    torch.manual_seed(0)
    np.random.seed(0)

    report = run_eval(
        cfg=cfg,
        ckpt_path=args.checkpoint,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    _print_report(report)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"\nWrote report: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
