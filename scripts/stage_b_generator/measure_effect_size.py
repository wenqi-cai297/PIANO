"""Stage B v0.3-pre — measure adapter effect size before any architecture change.

The v0.2 retrospective claimed γ_int_abs_mean = 0.025 is "stuck low". The
2026-04-27 root-cause research
(``analyses/2026-04-27_v0_3_root_cause_research.md``) flagged this as
unverified: a per-layer scalar gate of 0.025 against an attention-output
norm of O(1) gives a 2.5% residual contribution, which **may be
saturated rather than dead**. Before committing 80 epochs to v0.3a-revised,
we measure the actual effect size per layer.

For every block in the wrapped MaskTransformer's encoder, compute::

    effect_pre  = ‖γ_int · int_out‖ / ‖src_pre‖
    effect_post = ‖γ_int · int_out‖ / ‖src_post‖

per token, per sample, then aggregate over 100 val batches. ``src_pre``
is the input to ``block.norm_int`` (= post-self-attn residual stream).
``int_out`` is the output of ``block.int_attn`` BEFORE γ scaling AND
before dropout (dropout_int is disabled in eval mode regardless).
``src_post = src_pre + γ_int · int_out`` is what the FFN sees.

Verdict thresholds (per the root-cause research §"Decision tree"):

- max ratio across layers < 0.5%   → adapter_dead       → v0.3a mandatory
- max ratio in [0.5%, 5%)          → adapter_borderline → v0.3a anyway
- max ratio ≥ 5%                   → adapter_contributing → consider skipping
                                       to v0.3c (trainable-copy) or
                                       v0.3d (codebook retrain) since
                                       γ=0.025 is a misleading number

Pure measurement — no training, no checkpoint writes. Reads
``best_val.pt`` and runs forward in eval mode.

Output files (all under ``--output-dir``):

- ``summary.txt`` — pretty-printed per-layer table + verdict + recommendation
  (mirror of what's printed to stdout, so nothing has to be captured from terminal)
- ``summary.json`` — structured result with all per-layer stats and verdict
- ``raw_ratios.npz`` — per-layer concatenated per-token ratio arrays
  (``ratio_pre_layer<i>``, ``ratio_post_layer<i>``) for downstream plotting
  / additional analysis without re-running the script

Usage::

    python scripts/stage_b_generator/measure_effect_size.py \\
        --config configs/training/generator.yaml \\
        --ckpt runs/training/generator/best_val.pt \\
        --num-batches 100 \\
        --output-dir runs/eval/stageB_v0_3_pre
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
from torch.utils.data import ConcatDataset, DataLoader

from piano.data.dataset import HOIDataset, collate_hoi
from piano.data.split import build_subject_split, extract_subject_id
from piano.models.backbones.momask_adapter import (
    load_momask_mask_transformer,
    load_momask_vqvae,
)
from piano.models.interaction_tokenizer import InteractionTokenizer
from piano.models.motion_generator import InteractionMaskTransformer
from piano.utils.io_utils import ensure_dir, load_json


# ============================================================================
# Dataset assembly (mirrors train_generator + qual_eval to hit the exact
# val bucket v0.2 was evaluated on)
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


def _build_val_loader(cfg, batch_size: int) -> DataLoader:
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
            surface_obj_pose=True,           # v0.2 tokenizer wants 36-d z_int
        )
        datasets.append(ds)
    return DataLoader(
        ConcatDataset(datasets),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_hoi,
        num_workers=0,                       # 100 batches is fast; spawn cost
        pin_memory=True,
        drop_last=False,
    )


# ============================================================================
# Model loading (no Residual Transformer / VQ decode — we only need
# MaskTransformer forward to capture pre / post / int_out at each block)
# ============================================================================

def _build_model(cfg, ckpt_path: Path, device: torch.device) -> tuple[
    InteractionMaskTransformer, torch.nn.Module, int,
]:
    model_cfg = OmegaConf.load(cfg.model.config)

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
    mt_cfg = model_cfg.masked_transformer
    base_mt = load_momask_mask_transformer(
        cfg.model.checkpoints.masked_transformer,
        code_dim=mt_cfg.code_dim,
        latent_dim=mt_cfg.latent_dim,
        ff_size=mt_cfg.ff_size,
        num_layers=mt_cfg.num_layers,
        num_heads=mt_cfg.num_heads,
        dropout=mt_cfg.dropout,
        clip_dim=mt_cfg.clip_dim,
        clip_version=cfg.model.get("text_encoder", "ViT-B/32"),
        cond_drop_prob=mt_cfg.cond_drop_prob,
        num_tokens=mt_cfg.num_tokens,
        device=str(device),
    )
    token_stride = int(model_cfg.vq_vae.stride_t ** model_cfg.vq_vae.down_t)
    max_seq_length_frames = int(cfg.data.max_seq_length)
    max_seq_length_tokens = max_seq_length_frames // token_stride

    interaction_tokenizer = InteractionTokenizer(
        d_model=mt_cfg.latent_dim,
        token_stride=token_stride,
        max_seq_length=max_seq_length_frames,
    )
    transformer = InteractionMaskTransformer(
        mask_transformer=base_mt,
        interaction_tokenizer=interaction_tokenizer,
        zero_init_gamma=bool(mt_cfg.interaction_cross_attn.get("zero_init", True)),
        max_token_seq_length=max_seq_length_tokens,
    )
    transformer.to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    missing, unexpected = transformer.load_state_dict(state, strict=False)
    real_missing = [k for k in missing if not k.startswith("mask_transformer.clip_model.")]
    if real_missing:
        print(f"  [warn] missing keys: {real_missing[:5]}{'...' if len(real_missing) > 5 else ''}")
    if unexpected:
        print(f"  [warn] unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    transformer.eval()
    return transformer, vq_model, token_stride


# ============================================================================
# Activation capture via forward hooks
# ============================================================================

class ActivationCapture:
    """Hook every IntXAttn block to capture (src_pre, int_out) tensors.

    The IntXAttn sublayer's forward pass at
    ``MaskTransformerBlockWithInteraction.forward`` is::

        if int_kv is not None:
            q = self.norm_int(src)                                     # src_pre = q's INPUT
            int_out, _ = self.int_attn(query=q, key=int_kv, value=int_kv, ...)
            src = src + self.gamma_int * self.dropout_int(int_out)     # src_post = this

    So:
      - ``norm_int`` forward_pre_hook receives ``src_pre`` as its input.
      - ``int_attn`` forward_hook receives the ``(int_out, attn_weights)``
        tuple as its output.

    Both detached on capture so we don't grow autograd graphs.
    """

    def __init__(self, n_layers: int) -> None:
        self.n_layers = n_layers
        self.src_pre: dict[int, Tensor] = {}
        self.int_out: dict[int, Tensor] = {}
        self._handles: list[Any] = []

    def attach(self, transformer: InteractionMaskTransformer) -> None:
        layers = transformer.mask_transformer.seqTransEncoder.layers
        for i, blk in enumerate(layers):
            self._handles.append(
                blk.norm_int.register_forward_pre_hook(self._make_pre_hook(i)),
            )
            self._handles.append(
                blk.int_attn.register_forward_hook(self._make_attn_hook(i)),
            )

    def _make_pre_hook(self, layer_idx: int):
        def hook(_module: torch.nn.Module, inputs: tuple) -> None:
            self.src_pre[layer_idx] = inputs[0].detach()
        return hook

    def _make_attn_hook(self, layer_idx: int):
        def hook(_module: torch.nn.Module, _inputs: tuple, outputs: tuple) -> None:
            # nn.MultiheadAttention returns (attn_output, attn_weights)
            self.int_out[layer_idx] = outputs[0].detach()
        return hook

    def clear(self) -> None:
        self.src_pre.clear()
        self.int_out.clear()

    def detach_all(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ============================================================================
# Per-batch metric computation
# ============================================================================

def _per_token_norms(t: Tensor, valid_mask: Tensor) -> Tensor:
    """L2 norm along the d (last) axis, masked to valid token positions.

    Parameters
    ----------
    t : (S+1, B, d) — sequence-first activation tensor (matches MoMask convention).
    valid_mask : (B, S+1) bool — True where the token position is real, False if padded.

    Returns
    -------
    norms : (n_valid,) flat list of per-token L2 norms across all valid (B, S+1) positions.
    """
    sb = t.shape[0]
    bs = t.shape[1]
    norms = t.float().norm(dim=-1)               # (S+1, B)
    norms_bf = norms.transpose(0, 1).contiguous()   # (B, S+1)
    return norms_bf[valid_mask]                  # (n_valid,)


def _ratio_per_layer(
    src_pre: Tensor,                             # (S+1, B, d)
    int_out: Tensor,                             # (S+1, B, d)
    gamma_int: float,
    valid_mask: Tensor,                          # (B, S+1) True = valid
) -> dict[str, float]:
    """Compute effect-size ratios for one layer over one batch."""
    effect = gamma_int * int_out                 # (S+1, B, d)
    src_post = src_pre + effect                  # (S+1, B, d)

    eff_norms = _per_token_norms(effect, valid_mask)
    pre_norms = _per_token_norms(src_pre, valid_mask)
    post_norms = _per_token_norms(src_post, valid_mask)

    # Guard against divide-by-zero (very early-training pre-norm could be zero
    # at padded positions; we already mask padded but this is belt-and-suspenders).
    eps = 1e-8
    ratio_pre = (eff_norms / pre_norms.clamp(min=eps)).cpu().numpy()
    ratio_post = (eff_norms / post_norms.clamp(min=eps)).cpu().numpy()

    return {
        "effect_norm_mean": float(eff_norms.mean().item()),
        "src_pre_norm_mean": float(pre_norms.mean().item()),
        "src_post_norm_mean": float(post_norms.mean().item()),
        "ratio_pre": ratio_pre,                  # numpy array, aggregated later
        "ratio_post": ratio_post,
    }


# ============================================================================
# Main measurement loop
# ============================================================================

@torch.no_grad()
def measure(
    transformer: InteractionMaskTransformer,
    vq_model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    num_batches: int,
    token_stride: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run forward on N val batches, capture activations, return aggregated stats + raw ratios.

    Returns
    -------
    summary
        Dict with keys ``per_layer`` (list of length n_layers, each a dict with
        aggregated ratio statistics) and ``num_batches`` / ``num_clips`` totals.
    raw
        Per-layer concatenated per-token ratio arrays. Keys are
        ``ratio_pre_layer{i}`` / ``ratio_post_layer{i}`` for layer index ``i``.
        Useful for histograms / downstream plotting without re-running the
        forward pass.
    """
    n_layers = len(transformer.mask_transformer.seqTransEncoder.layers)
    cap = ActivationCapture(n_layers)
    cap.attach(transformer)

    # γ values are static across batches — read once.
    gammas = [
        float(blk.gamma_int.detach().item())
        for blk in transformer.mask_transformer.seqTransEncoder.layers
    ]

    # Per-layer accumulators. Store all per-token ratios across batches so
    # we can compute robust percentiles at the end. Token count is bounded:
    # 100 batches × 32 samples × 50 tokens ≈ 160k floats per layer — fine.
    pre_ratios: list[list[np.ndarray]] = [[] for _ in range(n_layers)]
    post_ratios: list[list[np.ndarray]] = [[] for _ in range(n_layers)]
    eff_norms: list[list[float]] = [[] for _ in range(n_layers)]
    pre_norms: list[list[float]] = [[] for _ in range(n_layers)]
    post_norms: list[list[float]] = [[] for _ in range(n_layers)]

    n_clips_seen = 0
    n_batches_seen = 0

    for batch in val_loader:
        if n_batches_seen >= num_batches:
            break

        motion = batch["motion"].to(device).float()         # (B, T, 263)
        seq_len = batch["seq_len"].to(device).long()         # (B,)
        text = batch["text"]                                  # list[str]
        B = motion.shape[0]
        n_clips_seen += B
        n_batches_seen += 1

        # GT base tokens (all real, no masking — measures the adapter's
        # effect at "all-context-visible" inference, the cleanest signal).
        code_idx, _ = vq_model.encode(motion)                # (B, S, Q)
        base_ids = code_idx[..., 0].long()                    # (B, S)
        m_lens_tok = (seq_len // token_stride).clamp(min=1).long()

        cond_vector = transformer.encode_text(text).to(device).float()

        int_kv, int_pad = transformer.interaction_tokenizer(
            contact_state=batch["contact_state"].to(device).float(),
            contact_target_xyz=batch["contact_target_xyz"].to(device).float(),
            phase=batch["phase"].to(device).long(),
            support=batch["support"].to(device).long(),
            obj_com_canonical=batch["obj_com_canonical"].to(device).float(),
            obj_rot6d_canonical=batch["obj_rot6d_canonical"].to(device).float(),
            seq_lens=seq_len,
        )

        # token_padding_mask is over the motion tokens (S positions);
        # the wrapper prepends the cond token so the full sequence is S+1
        # with cond at position 0 (always valid).
        S = base_ids.shape[1]
        token_pad_mask = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
        token_pad_mask = token_pad_mask >= m_lens_tok.unsqueeze(1)   # (B, S) True = padded

        cap.clear()
        _ = transformer.trans_forward(
            motion_ids=base_ids,
            cond_vector=cond_vector,
            token_padding_mask=token_pad_mask,
            int_tokens_bf=int_kv,
            int_padding_mask_bf=int_pad,
            drop_text_mask=None,
            drop_int_mask=None,
        )

        # Build the (B, S+1) valid mask: cond token (0) is always valid; motion
        # tokens (1..S) follow ~token_pad_mask.
        cond_valid = torch.ones((B, 1), dtype=torch.bool, device=device)
        valid_mask = torch.cat([cond_valid, ~token_pad_mask], dim=1)   # (B, S+1)

        for i in range(n_layers):
            stats = _ratio_per_layer(
                src_pre=cap.src_pre[i],
                int_out=cap.int_out[i],
                gamma_int=gammas[i],
                valid_mask=valid_mask,
            )
            pre_ratios[i].append(stats["ratio_pre"])
            post_ratios[i].append(stats["ratio_post"])
            eff_norms[i].append(stats["effect_norm_mean"])
            pre_norms[i].append(stats["src_pre_norm_mean"])
            post_norms[i].append(stats["src_post_norm_mean"])

    cap.detach_all()

    # Aggregate across batches into per-layer summaries.
    per_layer_out: list[dict[str, Any]] = []
    for i in range(n_layers):
        pre_all = np.concatenate(pre_ratios[i]) if pre_ratios[i] else np.zeros(0)
        post_all = np.concatenate(post_ratios[i]) if post_ratios[i] else np.zeros(0)
        per_layer_out.append({
            "layer_idx": i,
            "gamma_int": gammas[i],
            "effect_norm_mean": float(np.mean(eff_norms[i])) if eff_norms[i] else 0.0,
            "src_pre_norm_mean": float(np.mean(pre_norms[i])) if pre_norms[i] else 0.0,
            "src_post_norm_mean": float(np.mean(post_norms[i])) if post_norms[i] else 0.0,
            "ratio_pre_mean": float(pre_all.mean()) if pre_all.size > 0 else 0.0,
            "ratio_pre_p50": float(np.percentile(pre_all, 50)) if pre_all.size > 0 else 0.0,
            "ratio_pre_p95": float(np.percentile(pre_all, 95)) if pre_all.size > 0 else 0.0,
            "ratio_post_mean": float(post_all.mean()) if post_all.size > 0 else 0.0,
            "ratio_post_p50": float(np.percentile(post_all, 50)) if post_all.size > 0 else 0.0,
            "ratio_post_p95": float(np.percentile(post_all, 95)) if post_all.size > 0 else 0.0,
        })

    summary = {
        "num_batches": n_batches_seen,
        "num_clips": n_clips_seen,
        "n_layers": n_layers,
        "per_layer": per_layer_out,
    }
    # Raw per-token arrays for downstream plotting / further analysis.
    raw: dict[str, np.ndarray] = {}
    for i in range(n_layers):
        raw[f"ratio_pre_layer{i}"] = (
            np.concatenate(pre_ratios[i]) if pre_ratios[i] else np.zeros(0)
        )
        raw[f"ratio_post_layer{i}"] = (
            np.concatenate(post_ratios[i]) if post_ratios[i] else np.zeros(0)
        )
    return summary, raw


# ============================================================================
# Verdict: classify into adapter_dead / borderline / contributing
# ============================================================================

def _verdict(per_layer: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the analyses doc's decision thresholds to the per-layer ratios.

    We use the **max** of ``ratio_post_mean`` across layers — even if most
    layers are dead, a single layer carrying meaningful contribution
    means the adapter is doing something.
    """
    ratios = [layer["ratio_post_mean"] for layer in per_layer]
    max_ratio = max(ratios) if ratios else 0.0
    mean_ratio = sum(ratios) / len(ratios) if ratios else 0.0

    if max_ratio < 0.005:
        category = "adapter_dead"
        recommendation = (
            "v0.3a-revised mandatory: permanent freeze + remove dropout_int "
            "+ tanh-wrap γ. Adapter contribution is < 0.5% of residual stream "
            "across all layers — γ-gate is structurally not transmitting signal."
        )
    elif max_ratio < 0.05:
        category = "adapter_borderline"
        recommendation = (
            "v0.3a-revised recommended. Adapter is contributing 0.5-5% of "
            "residual stream — measurable but small relative to ControlNet "
            "image-domain ~10-30%. Cheap fix-stack (freeze + remove dropout "
            "+ tanh) likely raises this; if it doesn't, escalate to v0.3c "
            "(trainable-copy InterControl)."
        )
    else:
        category = "adapter_contributing"
        recommendation = (
            "Adapter is meaningfully contributing (≥5% of residual stream). "
            "γ=0.025 was a misleading headline number. The visual failure "
            "is likely NOT a γ-gate problem. Skip v0.3a, jump to v0.3c "
            "(trainable-copy rebuild) or v0.3d (codebook retrain) since the "
            "current adapter is doing what it can within its dof."
        )

    return {
        "category": category,
        "max_layer_ratio_post_mean": max_ratio,
        "mean_layer_ratio_post_mean": mean_ratio,
        "recommendation": recommendation,
    }


# ============================================================================
# Pretty-print + JSON dump
# ============================================================================

def _format_summary(result: dict[str, Any]) -> str:
    """Build the pretty-printed report as a single string.

    Identical content to what previous versions printed to stdout, but
    returnable so we can both ``print`` it and write it to a file
    without diverging.
    """
    lines: list[str] = []
    lines.append(f"\n=== Stage B v0.3-pre effect-size measurement ===")
    lines.append(f"  ckpt:          {result['ckpt']}")
    lines.append(f"  num_batches:   {result['num_batches']}")
    lines.append(f"  num_clips:     {result['num_clips']}")
    lines.append(f"  n_layers:      {result['n_layers']}")
    lines.append("")

    lines.append(
        f"  {'L':>2} {'γ_int':>8} {'‖γ·int_out‖':>12} {'‖src_pre‖':>10} "
        f"{'%pre':>7} {'%post':>7} {'p50post':>8} {'p95post':>8}"
    )
    lines.append(f"  {'-' * 70}")
    for layer in result["per_layer"]:
        i = layer["layer_idx"]
        g = layer["gamma_int"]
        eff = layer["effect_norm_mean"]
        pre = layer["src_pre_norm_mean"]
        rpre = layer["ratio_pre_mean"] * 100.0
        rpost = layer["ratio_post_mean"] * 100.0
        p50 = layer["ratio_post_p50"] * 100.0
        p95 = layer["ratio_post_p95"] * 100.0
        lines.append(
            f"  {i:>2} {g:>+8.4f} {eff:>12.4f} {pre:>10.4f} "
            f"{rpre:>6.2f}% {rpost:>6.2f}% {p50:>7.2f}% {p95:>7.2f}%"
        )

    v = result["verdict"]
    lines.append("")
    lines.append(f"  Verdict:        {v['category']}")
    lines.append(
        f"  Max layer post-residual ratio:  "
        f"{v['max_layer_ratio_post_mean'] * 100:.2f}%"
    )
    lines.append(
        f"  Mean layer post-residual ratio: "
        f"{v['mean_layer_ratio_post_mean'] * 100:.2f}%"
    )
    lines.append("")
    lines.append(f"  Recommendation:")
    for line in v["recommendation"].split(". "):
        if line.strip():
            lines.append(f"    {line.strip().rstrip('.')}.")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/training/generator.yaml"),
        help="training config (drives val-bucket selection + model dims).",
    )
    parser.add_argument(
        "--ckpt", type=Path,
        default=Path("runs/training/generator/best_val.pt"),
        help="Stage B checkpoint to measure.",
    )
    parser.add_argument(
        "--num-batches", type=int, default=100,
        help="number of val batches to forward through. 100 batches × bs=32 "
             "≈ 3200 clips, plenty for stable percentiles.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="val batch size; matches training default.",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="cuda / cpu; auto-detect if omitted.",
    )
    parser.add_argument(
        "--output-dir", "-o", type=Path,
        default=Path("runs/eval/stageB_v0_3_pre"),
        help="output directory; will contain summary.txt + summary.json + raw_ratios.npz.",
    )
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    if not args.ckpt.exists():
        print(f"ERROR: ckpt not found: {args.ckpt}")
        return 1
    if not args.config.exists():
        print(f"ERROR: config not found: {args.config}")
        return 1

    cfg = OmegaConf.load(args.config)
    print(f"Loading model from {args.ckpt} on {device}...")
    transformer, vq_model, token_stride = _build_model(cfg, args.ckpt, device)

    print(f"Building val loader (subject_split, batch_size={args.batch_size})...")
    val_loader = _build_val_loader(cfg, args.batch_size)
    n_total_clips = len(val_loader.dataset)
    print(f"  val: {n_total_clips} clips total; will forward up to {args.num_batches} batches.")

    print(f"Measuring effect size...")
    measurement, raw_ratios = measure(
        transformer=transformer,
        vq_model=vq_model,
        val_loader=val_loader,
        device=device,
        num_batches=args.num_batches,
        token_stride=token_stride,
    )

    result: dict[str, Any] = {
        "ckpt": str(args.ckpt),
        "config": str(args.config),
        **measurement,
        "verdict": _verdict(measurement["per_layer"]),
    }
    report = _format_summary(result)
    print(report)

    # Write all artifacts into the output dir — nothing only-on-stdout.
    ensure_dir(args.output_dir)
    summary_txt = args.output_dir / "summary.txt"
    summary_json = args.output_dir / "summary.json"
    raw_npz = args.output_dir / "raw_ratios.npz"

    with summary_txt.open("w", encoding="utf-8") as f:
        f.write(report)
        f.write("\n")
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    np.savez(raw_npz, **raw_ratios)

    print(f"\nWrote:")
    print(f"  {summary_txt}")
    print(f"  {summary_json}")
    print(f"  {raw_npz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
