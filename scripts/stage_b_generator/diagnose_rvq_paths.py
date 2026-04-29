"""Diagnose Stage B soft-hard and RVQ path gaps.

This script is intentionally an eval/generation entry point, not library code.
It loads one Stage B checkpoint, samples the same stratified validation clips as
``qual_eval.py``, and writes several ``generated.npz`` condition directories:

- ``soft_train_full``: differentiable full-RVQ relaxed decode from the training
  MaskTransformer logits.
- ``hard_train_argmax_full``: argmax of those same base logits, then greedy
  residual prediction.
- ``hard_train_argmax_gt_residual``: argmax base logits with GT residual RVQ
  tokens.
- ``mixed_gt_all``: GT VQ roundtrip.
- ``mixed_pred_all``: standard generated base+residual tokens.
- ``mixed_gt_base_pred_residual``: GT base token with generated residual tokens.
- ``mixed_pred_base_gt_residual``: generated base token with GT residual tokens.

The first three quantify the soft-hard gap in the v13 decoded auxiliary loss.
The last four locate whether the remaining error is mostly base-token or
residual-stack prediction.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import Tensor

from piano.data.eval_sampling import describe_eval_clip_selection, select_eval_clip_indices
from piano.data.humanml3d_repr import load_motion_stats
from piano.training.decoded_contact_loss import (
    _base_logits_to_bsv,
    _decode_relaxed_full_rvq_prediction,
    _force_pad_to_token_zero,
    _logits_to_codebook_vocab,
)
from piano.utils.io_utils import ensure_dir

from qual_eval import (
    _build_model,
    _build_val_dataset,
    _get_canon_to_world_transform,
    _save_condition_dir,
    _tokenize_z_int,
)


CONDITION_NAMES = [
    "soft_train_full",
    "hard_train_argmax_full",
    "hard_train_argmax_gt_residual",
    "mixed_gt_all",
    "mixed_pred_all",
    "mixed_gt_base_pred_residual",
    "mixed_pred_base_gt_residual",
]


def _as_stats_tensors(
    motion_mean: np.ndarray,
    motion_std: np.ndarray,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    mean = torch.from_numpy(motion_mean).float().to(device).view(1, 1, -1)
    std = torch.from_numpy(motion_std).float().to(device).view(1, 1, -1)
    return mean, std


def _encode_gt_indices(
    sample: dict[str, Any],
    *,
    vq_model: torch.nn.Module,
    motion_mean_t: Tensor,
    motion_std_t: Tensor,
    token_stride: int,
    device: torch.device,
) -> tuple[Tensor, int]:
    """Return GT all-RVQ ids ``(1, S, Q)`` and token length ``S``."""
    seq_len = int(sample["seq_len"].item())
    m_len_tok = max(1, seq_len // token_stride)
    frames = m_len_tok * token_stride
    motion = sample["motion"][:frames].unsqueeze(0).float().to(device)
    motion_norm = (motion - motion_mean_t) / motion_std_t
    with torch.no_grad():
        code_idx, _ = vq_model.encode(motion_norm)
    return code_idx.long(), m_len_tok


def _decode_ids_to_motion(
    all_ids: Tensor,
    *,
    vq_model: torch.nn.Module,
    motion_mean: np.ndarray,
    motion_std: np.ndarray,
) -> np.ndarray:
    """Decode ``(1, S, Q)`` RVQ ids to denormalized motion_263 numpy."""
    all_for_decode = torch.where(all_ids < 0, torch.zeros_like(all_ids), all_ids)
    with torch.no_grad():
        motion = vq_model.forward_decoder(all_for_decode).squeeze(0)
    out = motion.detach().cpu().numpy()
    return (out * motion_std + motion_mean).astype(np.float32)


def _denorm_motion(
    motion_norm: Tensor,
    *,
    motion_mean: np.ndarray,
    motion_std: np.ndarray,
) -> np.ndarray:
    motion = motion_norm.squeeze(0).detach().cpu().numpy()
    return (motion * motion_std + motion_mean).astype(np.float32)


def _argmax_base_ids_from_logits(
    base_logits: Tensor,
    *,
    m_lens_tok: Tensor,
    vocab_size: int,
) -> Tensor:
    """Argmax base ids from ``InteractionMaskTransformer.forward`` logits."""
    S = int(m_lens_tok.max().item())
    logits_bsv = _base_logits_to_bsv(base_logits, token_count=S)
    logits_bsv = _logits_to_codebook_vocab(logits_bsv, vocab_size)
    token_mask = (
        torch.arange(S, device=base_logits.device).unsqueeze(0)
        < m_lens_tok.to(device=base_logits.device).long().clamp(min=1).unsqueeze(1)
    )
    logits_bsv = _force_pad_to_token_zero(logits_bsv, token_mask)
    ids = logits_bsv.argmax(dim=-1)
    return torch.where(token_mask, ids, torch.full_like(ids, -1))


@torch.no_grad()
def _predict_residual_greedy(
    residual_transformer: torch.nn.Module,
    *,
    base_ids: Tensor,
    text: str,
    m_lens_tok: Tensor,
    int_kv_bfd: Tensor | None,
    int_padding_mask: Tensor | None,
    cond_scale: float,
) -> Tensor:
    """Greedy full-RVQ residual prediction for deterministic diagnostics.

    Mirrors ``ResidualTransformerWithInteraction.generate_with_int`` but uses
    ``argmax`` instead of Gumbel sampling. Output is ``(B, S, Q)`` with ``-1``
    at padded positions.
    """
    r = getattr(residual_transformer, "residual", residual_transformer)
    r.process_embed_proj_weight()
    device = base_ids.device
    seq_len = int(base_ids.shape[1])
    batch_size = 1

    if r.cond_mode == "text":
        cond_vector = r.encode_text([text]).to(device).float()
    elif r.cond_mode == "action":
        cond_vector = r.enc_action([text]).to(device).float()
    elif r.cond_mode == "uncond":
        cond_vector = torch.zeros(batch_size, r.latent_dim, device=device).float()
    else:
        raise NotImplementedError(f"Unsupported cond_mode {r.cond_mode!r}")

    padding_mask = ~(
        torch.arange(seq_len, device=device).unsqueeze(0)
        < m_lens_tok.to(device=device).long().clamp(min=1).unsqueeze(1)
    )
    motion_ids = torch.where(padding_mask, torch.full_like(base_ids, r.pad_id), base_ids)
    motion_ids = torch.where(motion_ids < 0, torch.zeros_like(motion_ids), motion_ids)
    all_indices = [motion_ids]
    history_sum: Tensor | int = 0
    num_quant_layers = int(r.opt.num_quantizers)

    int_kv = None if int_kv_bfd is None else int_kv_bfd.transpose(0, 1).contiguous()
    for q in range(1, num_quant_layers):
        token_embed = r.token_embed_weight[q - 1]
        safe_ids = motion_ids.clamp(min=0, max=int(token_embed.shape[0]) - 1)
        gathered = token_embed.index_select(0, safe_ids.reshape(-1))
        gathered = gathered.view(batch_size, seq_len, token_embed.shape[-1])
        history_sum = history_sum + gathered

        if hasattr(residual_transformer, "forward_with_cond_scale_with_int"):
            logits = residual_transformer.forward_with_cond_scale_with_int(
                history_sum,
                q,
                cond_vector,
                padding_mask,
                int_kv=int_kv,
                int_padding_mask=int_padding_mask,
                cond_scale=cond_scale,
            )
        else:
            logits = r.forward_with_cond_scale(
                history_sum,
                q,
                cond_vector,
                padding_mask,
                cond_scale=cond_scale,
            )
        ids = logits.permute(0, 2, 1).argmax(dim=-1)
        motion_ids = torch.where(padding_mask, torch.full_like(ids, r.pad_id), ids)
        all_indices.append(motion_ids)

    all_indices_t = torch.stack(all_indices, dim=-1)
    return torch.where(all_indices_t == r.pad_id, torch.full_like(all_indices_t, -1), all_indices_t)


@torch.no_grad()
def _generate_all_ids(
    transformer: torch.nn.Module,
    residual_transformer: torch.nn.Module,
    *,
    text: str,
    m_lens_tok: Tensor,
    int_kv_bfd: Tensor | None,
    int_padding_mask: Tensor | None,
    w_text: float,
    w_int: float,
    timesteps: int,
    res_cond_scale: float,
    residual_seed: int | None,
) -> tuple[Tensor, Tensor]:
    cond_vector = transformer.encode_text([text]).to(m_lens_tok.device).float()
    base_ids = transformer.generate(
        cond_vector=cond_vector,
        m_lens_tok=m_lens_tok,
        int_tokens_bf=int_kv_bfd,
        int_padding_mask_bf=int_padding_mask,
        timesteps=timesteps,
        w_text=w_text,
        w_int=w_int,
    )
    base_for_res = torch.where(base_ids < 0, torch.zeros_like(base_ids), base_ids)
    if residual_seed is not None:
        torch.manual_seed(int(residual_seed))

    if hasattr(residual_transformer, "generate_with_int"):
        res_int_kv = None if int_kv_bfd is None else int_kv_bfd.transpose(0, 1).contiguous()
        all_ids = residual_transformer.generate_with_int(
            motion_ids=base_for_res,
            conds=[text],
            m_lens=m_lens_tok,
            int_kv=res_int_kv,
            int_padding_mask=int_padding_mask,
            cond_scale=res_cond_scale,
        )
    else:
        all_ids = residual_transformer.generate(
            motion_ids=base_for_res,
            conds=[text],
            m_lens=m_lens_tok,
            cond_scale=res_cond_scale,
        )
    return base_ids, all_ids


def _valid_hamming(a: Tensor, b: Tensor, valid_len: int) -> float:
    aa = a.detach().cpu().numpy().reshape(-1)[:valid_len]
    bb = b.detach().cpu().numpy().reshape(-1)[:valid_len]
    if len(aa) == 0:
        return 0.0
    return float(np.mean(aa != bb))


def _motion_rms(a: np.ndarray, b: np.ndarray) -> float:
    T = min(len(a), len(b))
    if T == 0:
        return 0.0
    d = a[:T] - b[:T]
    return float(np.sqrt(np.mean(d * d)))


def _row(motion: np.ndarray, base: np.ndarray | None = None) -> dict[str, dict[str, Any]]:
    return {"motion": motion, "base": base if base is not None else np.zeros((1,), dtype=np.int64), "swap_from": None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-clips", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--w-text", type=float, default=4.0)
    parser.add_argument("--w-int", type=float, default=2.0)
    parser.add_argument("--timesteps", type=int, default=10)
    parser.add_argument("--res-cond-scale", type=float, default=2.0)
    parser.add_argument(
        "--residual-seed",
        type=int,
        default=1234,
        help="Seed before standard residual sampling; use -1 to leave RNG free.",
    )
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    residual_seed = None if int(args.residual_seed) < 0 else int(args.residual_seed)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    ensure_dir(args.output_dir)

    cfg = OmegaConf.load(args.config)
    transformer, vq_model, residual_transformer, token_stride = _build_model(
        cfg,
        args.ckpt,
        device,
    )
    motion_mean, motion_std = load_motion_stats(cfg.model.checkpoints.vq_vae)
    motion_mean_t, motion_std_t = _as_stats_tensors(motion_mean, motion_std, device)
    vocab_size = int(vq_model.quantizer.codebooks.shape[1])

    val_dataset = _build_val_dataset(cfg)
    sampled_idx = select_eval_clip_indices(val_dataset, args.num_clips, seed=args.seed)
    selected_rows = describe_eval_clip_selection(val_dataset, sampled_idx)
    samples = [val_dataset[i] for i in sampled_idx]

    texts = [str(s["text"]) for s in samples]
    seq_ids = [str(s["seq_id"]) for s in samples]
    seq_lens_frames = [int(s["seq_len"].item()) for s in samples]
    object_pcs = [s["object_pc"].cpu().numpy() for s in samples]
    object_positions = [s["object_positions"].cpu().numpy() for s in samples]
    object_rotations = [s["object_rotations"].cpu().numpy() for s in samples]
    source_canon_xforms = [
        _get_canon_to_world_transform(
            s["joints"].cpu().numpy(),
            s["motion"].cpu().numpy(),
        )
        for s in samples
    ]

    z_int_per = [_tokenize_z_int(transformer, s, device) for s in samples]
    per_condition: dict[str, list[dict[str, dict[str, Any]]]] = {
        name: [] for name in CONDITION_NAMES
    }
    per_clip_report: list[dict[str, Any]] = []

    print(f"Running RVQ diagnostics for {len(samples)} clips on {device}.")
    for i, sample in enumerate(samples):
        text = texts[i]
        gt_ids, m_len_tok = _encode_gt_indices(
            sample,
            vq_model=vq_model,
            motion_mean_t=motion_mean_t,
            motion_std_t=motion_std_t,
            token_stride=token_stride,
            device=device,
        )
        m_lens_tok = torch.tensor([m_len_tok], dtype=torch.long, device=device)
        int_kv, int_pad = z_int_per[i]
        cond_vector = transformer.encode_text([text]).to(device).float()
        base_ids_gt = gt_ids[..., 0]

        # Training-logit path: one deterministic BERT-mask forward.
        torch.manual_seed(args.seed * 1000 + i)
        train_out = transformer.forward(
            base_ids_gt,
            cond_vector,
            m_lens_tok,
            int_tokens_bf=int_kv,
            int_padding_mask_bf=int_pad,
            cfg_drop_buckets=None,
            return_logits=True,
        )
        base_logits = train_out["logits"]
        res_int_kv = int_kv.transpose(0, 1).contiguous()
        soft_norm, _ = _decode_relaxed_full_rvq_prediction(
            base_logits=base_logits,
            residual_transformer=residual_transformer,
            vq_model=vq_model,
            text=[text],
            m_lens_tok=m_lens_tok,
            int_kv=res_int_kv,
            int_padding_mask=int_pad,
            temperature=1.0,
        )
        motion_soft = _denorm_motion(
            soft_norm,
            motion_mean=motion_mean,
            motion_std=motion_std,
        )
        base_train_argmax = _argmax_base_ids_from_logits(
            base_logits,
            m_lens_tok=m_lens_tok,
            vocab_size=vocab_size,
        )
        hard_full_ids = _predict_residual_greedy(
            residual_transformer,
            base_ids=base_train_argmax,
            text=text,
            m_lens_tok=m_lens_tok,
            int_kv_bfd=int_kv,
            int_padding_mask=int_pad,
            cond_scale=args.res_cond_scale,
        )
        hard_gtres_ids = torch.cat(
            [base_train_argmax.unsqueeze(-1), gt_ids[..., 1:]],
            dim=-1,
        )

        # Standard generation path.
        torch.manual_seed(args.seed * 1000 + 100000 + i)
        pred_base, pred_all = _generate_all_ids(
            transformer,
            residual_transformer,
            text=text,
            m_lens_tok=m_lens_tok,
            int_kv_bfd=int_kv,
            int_padding_mask=int_pad,
            w_text=args.w_text,
            w_int=args.w_int,
            timesteps=args.timesteps,
            res_cond_scale=args.res_cond_scale,
            residual_seed=(None if residual_seed is None else residual_seed + i),
        )

        mixed_gt_all = gt_ids
        mixed_pred_all = pred_all
        mixed_gt_base_pred_res = torch.cat([base_ids_gt.unsqueeze(-1), pred_all[..., 1:]], dim=-1)
        mixed_pred_base_gt_res = torch.cat([pred_base.unsqueeze(-1), gt_ids[..., 1:]], dim=-1)

        motions = {
            "soft_train_full": motion_soft,
            "hard_train_argmax_full": _decode_ids_to_motion(
                hard_full_ids, vq_model=vq_model, motion_mean=motion_mean, motion_std=motion_std,
            ),
            "hard_train_argmax_gt_residual": _decode_ids_to_motion(
                hard_gtres_ids, vq_model=vq_model, motion_mean=motion_mean, motion_std=motion_std,
            ),
            "mixed_gt_all": _decode_ids_to_motion(
                mixed_gt_all, vq_model=vq_model, motion_mean=motion_mean, motion_std=motion_std,
            ),
            "mixed_pred_all": _decode_ids_to_motion(
                mixed_pred_all, vq_model=vq_model, motion_mean=motion_mean, motion_std=motion_std,
            ),
            "mixed_gt_base_pred_residual": _decode_ids_to_motion(
                mixed_gt_base_pred_res, vq_model=vq_model, motion_mean=motion_mean, motion_std=motion_std,
            ),
            "mixed_pred_base_gt_residual": _decode_ids_to_motion(
                mixed_pred_base_gt_res, vq_model=vq_model, motion_mean=motion_mean, motion_std=motion_std,
            ),
        }

        base_np = pred_base.squeeze(0).detach().cpu().numpy()
        for name, motion in motions.items():
            per_condition[name].append({name: _row(motion, base_np)})

        valid = int(m_len_tok)
        per_clip_report.append({
            "index": selected_rows[i]["index"],
            "subset": selected_rows[i]["subset"],
            "object_id": selected_rows[i]["object_id"],
            "seq_id": seq_ids[i],
            "seq_len_frames": seq_lens_frames[i],
            "seq_len_tokens": valid,
            "train_loss_base": float(train_out["loss"].detach().cpu()),
            "train_acc_base": float(train_out["acc"].detach().cpu()),
            "hamming_train_argmax_base_vs_gt": _valid_hamming(base_train_argmax, base_ids_gt, valid),
            "hamming_sample_base_vs_gt": _valid_hamming(pred_base, base_ids_gt, valid),
            "hamming_sample_residual_vs_gt": _valid_hamming(pred_all[..., 1:].reshape(1, -1), gt_ids[..., 1:].reshape(1, -1), valid * (gt_ids.shape[-1] - 1)),
            "rms_soft_vs_hard_argmax_full": _motion_rms(motions["soft_train_full"], motions["hard_train_argmax_full"]),
            "rms_pred_all_vs_gt_all": _motion_rms(motions["mixed_pred_all"], motions["mixed_gt_all"]),
            "rms_gt_base_pred_residual_vs_gt_all": _motion_rms(motions["mixed_gt_base_pred_residual"], motions["mixed_gt_all"]),
            "rms_pred_base_gt_residual_vs_gt_all": _motion_rms(motions["mixed_pred_base_gt_residual"], motions["mixed_gt_all"]),
        })
        print(
            f"  {i + 1:>3}/{len(samples)} {seq_ids[i]} "
            f"S={valid} loss={per_clip_report[-1]['train_loss_base']:.3f} "
            f"base_ham={per_clip_report[-1]['hamming_sample_base_vs_gt']:.3f}",
        )

    for name, rows in per_condition.items():
        _save_condition_dir(
            args.output_dir / name,
            rows,
            name,
            texts,
            seq_lens_frames,
            seq_ids,
            object_pcs=object_pcs,
            object_positions=object_positions,
            object_rotations=object_rotations,
            world_R_y=[x[0] for x in source_canon_xforms],
            world_T_xz=[x[1] for x in source_canon_xforms],
        )

    summary = {
        "config": str(args.config),
        "ckpt": str(args.ckpt),
        "num_clips": len(samples),
        "seed": int(args.seed),
        "w_text": float(args.w_text),
        "w_int": float(args.w_int),
        "conditions": CONDITION_NAMES,
        "clip_selection": selected_rows,
        "per_clip": per_clip_report,
        "aggregate": {
            key: float(np.mean([row[key] for row in per_clip_report]))
            for key in [
                "train_loss_base",
                "train_acc_base",
                "hamming_train_argmax_base_vs_gt",
                "hamming_sample_base_vs_gt",
                "hamming_sample_residual_vs_gt",
                "rms_soft_vs_hard_argmax_full",
                "rms_pred_all_vs_gt_all",
                "rms_gt_base_pred_residual_vs_gt_all",
                "rms_pred_base_gt_residual_vs_gt_all",
            ]
        },
    }
    with (args.output_dir / "diagnostic_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote {args.output_dir}")
    print(f"Summary: {args.output_dir / 'diagnostic_summary.json'}")
    print("Next: run measure_contact_distance.py and measure_temporal_coupling.py on the condition dirs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
