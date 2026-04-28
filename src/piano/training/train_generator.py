"""Stage B: train the Motion Generator (interaction-conditioned MoMask).

Loads the pretrained MoMask MaskTransformer + RVQ-VAE, wraps the
MaskTransformer with :class:`piano.models.motion_generator.InteractionMaskTransformer`
(adds per-block IntXAttn sublayers with zero-init γ_int gates + a
learnable ``null_int_kv`` for compositional CFG), and finetunes against
the same masked-CE loss MoMask uses, conditioned on **GT** v11 pseudo-labels
read from each clip's npz.

Design references:
- :doc:`analyses/2026-04-26_stageB_design.md` for the architecture +
  CFG scheme + literature evidence.
- :doc:`analyses/early_setup.md` for MoMask weight-loading gotchas
  (``mu=0.99``, ``share_weight=True``).

Usage::

    accelerate launch --config_file configs/accelerate_config.yaml \\
        -m piano.training.train_generator \\
        --config configs/training/generator.yaml

The console script ``piano-train-generator`` (registered in
``pyproject.toml``) calls :func:`main` which forwards to
:func:`run`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader

from piano.data.dataset import (
    AugmentConfig,
    HOIDataset,
    build_object_split,
    build_subject_split,
    collate_hoi,
    extract_subject_id,
)
from piano.data.eval_sampling import (
    describe_eval_clip_selection,
    resolve_eval_clip_count,
    select_eval_clip_indices,
)
from piano.data.humanml3d_repr import load_motion_stats
from piano.models.backbones.momask_adapter import (
    load_momask_mask_transformer,
    load_momask_residual_transformer,
    load_momask_vqvae,
)
from piano.models.interaction_tokenizer import InteractionTokenizer
from piano.models.motion_generator import InteractionMaskTransformer
from piano.models.motion_generator_residual import ResidualTransformerWithInteraction
from piano.training.contact_eval import build_contact_eval_fn
from piano.training.decoded_contact_loss import decoded_contact_aux_loss
from piano.training.trainer import build_scheduler, run_training_loop
from piano.utils.io_utils import load_json


# ============================================================================
# Dataset assembly (mirrors train_predictor.py to keep splits consistent)
# ============================================================================

def _read_metadata(roots: list) -> list[tuple[str, dict]]:
    """Yield (subset_name, entry) for every metadata row across all roots.

    Reads ``metadata_clean.json`` first (matching what HOIDataset
    consumes by default), falls back to ``metadata.json`` if missing.
    """
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


def _collect_object_ids(roots: list) -> list[str]:
    seen: set[str] = set()
    for _, m in _read_metadata(roots):
        obj_id = m.get("object_id")
        if obj_id is not None:
            seen.add(obj_id)
    return sorted(seen)


def _resolve_split(cfg, split_override: str | None) -> dict:
    """Pick subject vs object split (subject_split takes precedence).

    Stage B should use the **same split** as Stage A so the predictor
    of record (trained on the train bucket) and Stage B's data view
    are consistent. The default config inherits subject_split from
    Stage A (per-subset stratified 85/15, no test bucket, seed=42).
    """
    subj_cfg = cfg.data.get("subject_split")
    obj_cfg = cfg.data.get("object_split")
    bucket = split_override

    if subj_cfg is not None and subj_cfg.get("enabled", False):
        subject_keys = _collect_subject_keys(cfg.data.datasets)
        splits = build_subject_split(
            subject_keys,
            train_pct=subj_cfg.train_pct,
            val_pct=subj_cfg.val_pct,
            seed=subj_cfg.seed,
        )
        b = bucket or subj_cfg.get("split", "train")
        if b == "all":
            allowed = None
        elif b in splits:
            allowed = splits[b]
        else:
            raise ValueError(f"unknown subject_split bucket: {b!r}")
        return {
            "object_id_filter": None,
            "subject_id_filter": allowed,
            "label": f"subject_split[{b}] ({len(allowed) if allowed else 'all'} subjects)",
        }

    if obj_cfg is not None and obj_cfg.get("enabled", False):
        object_ids = _collect_object_ids(cfg.data.datasets)
        splits = build_object_split(
            object_ids,
            train_pct=obj_cfg.train_pct,
            val_pct=obj_cfg.val_pct,
            test_pct=obj_cfg.test_pct,
            seed=obj_cfg.seed,
        )
        b = bucket or obj_cfg.get("split", "train")
        if b == "val+test":
            allowed = splits["val"] | splits["test"]
        elif b == "all":
            allowed = None
        elif b in splits:
            allowed = splits[b]
        else:
            raise ValueError(f"unknown object_split bucket: {b!r}")
        return {
            "object_id_filter": allowed,
            "subject_id_filter": None,
            "label": f"object_split[{b}] ({len(allowed) if allowed else 'all'} objects)",
        }

    return {
        "object_id_filter": None,
        "subject_id_filter": None,
        "label": "no_split (all clips)",
    }


def _build_dataset(cfg, split_override: str | None = None, enable_augment: bool = True) -> ConcatDataset:
    split_info = _resolve_split(cfg, split_override)

    aug_cfg = cfg.data.get("augmentation", None)
    augment = None
    if enable_augment and aug_cfg is not None and aug_cfg.get("enabled", False):
        augment = AugmentConfig(
            enabled=True,
            mirror_prob=float(aug_cfg.get("mirror_prob", 0.0)),
            rotate_around_y_prob=float(aug_cfg.get("rotate_around_y_prob", 0.0)),
            pc_jitter_std=float(aug_cfg.get("pc_jitter_std", 0.0)),
        )

    pseudo_label_dir = cfg.data.get("pseudo_label_dir", None)
    # v0.3-α: when cfg.data.force_world_frame=true, the obj-pose channels
    # are returned in world frame rather than body-canonical. Defaults
    # to false (v0.2 behaviour) when the key is absent.
    force_world_frame = bool(cfg.data.get("force_world_frame", False))
    datasets = []
    for entry in cfg.data.datasets:
        ds = HOIDataset(
            root=entry.root,
            pseudo_label_dir=pseudo_label_dir,
            max_seq_length=cfg.data.max_seq_length,
            object_id_filter=split_info["object_id_filter"],
            subject_id_filter=split_info["subject_id_filter"],
            augment=augment,
            # v0.2: surface body-canonical-frame object pose
            # (obj_com_canonical + obj_rot6d_canonical) so the
            # tokenizer's new 9 channels (per
            # analyses/2026-04-27_object_conditioning_review.md §5.2)
            # have something to consume. Stage A trainer leaves this
            # off — Stage A doesn't need object pose.
            surface_obj_pose=True,
            force_world_frame=force_world_frame,
        )
        datasets.append(ds)
    return ConcatDataset(datasets)


# ============================================================================
# Param-group optimiser (two LR groups: backbone finetune vs new modules)
# ============================================================================

def build_two_group_optimizer(
    new_params: list[nn.Parameter],
    backbone_params: list[nn.Parameter],
    *,
    new_lr: float = 1e-4,
    backbone_lr: float = 5e-5,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.999),
) -> AdamW:
    """AdamW with two LR groups: (a) new IntXAttn / tokenizer / γ /
    null_int_kv at full LR; (b) MoMask backbone finetune at lower LR.

    ViT/T5/MoMask weight-decay convention applies to both groups: no
    decay on biases, LayerNorm weights, positional embeddings, the γ
    scalar, or the null K/V tokens (all 1-D / scale-free).

    The two-rate convention follows ControlNet ICCV'23 + LLaMA-Adapter
    + AdapterFusion: pretrained weights move slowly so they retain their
    learned text-to-motion prior; new weights move fast so the IntXAttn
    sublayers actually become useful within the training budget.

    Returns
    -------
    AdamW optimiser with up to 4 groups: (new-decay, new-no-decay,
    backbone-decay, backbone-no-decay), each with its own LR and the
    appropriate weight_decay.
    """
    def _is_no_decay(p: nn.Parameter) -> bool:
        return p.ndim <= 1

    new_decay, new_no_decay = [], []
    for p in new_params:
        if not p.requires_grad:
            continue
        (new_no_decay if _is_no_decay(p) else new_decay).append(p)

    bb_decay, bb_no_decay = [], []
    for p in backbone_params:
        if not p.requires_grad:
            continue
        (bb_no_decay if _is_no_decay(p) else bb_decay).append(p)

    groups = []
    if new_decay:
        groups.append({"params": new_decay, "lr": new_lr, "weight_decay": weight_decay})
    if new_no_decay:
        groups.append({"params": new_no_decay, "lr": new_lr, "weight_decay": 0.0})
    if bb_decay:
        groups.append({"params": bb_decay, "lr": backbone_lr, "weight_decay": weight_decay})
    if bb_no_decay:
        groups.append({"params": bb_no_decay, "lr": backbone_lr, "weight_decay": 0.0})
    if not groups:
        raise ValueError("no trainable parameters; did you forget to load checkpoints?")
    return AdamW(groups, betas=betas)


# ============================================================================
# Step function
# ============================================================================

def _trainable_params(module: nn.Module) -> tuple[nn.Parameter, ...]:
    seen: set[int] = set()
    params: list[nn.Parameter] = []
    for p in module.parameters():
        if not p.requires_grad:
            continue
        pid = id(p)
        if pid in seen:
            continue
        seen.add(pid)
        params.append(p)
    return tuple(params)


def _grad_l2_norm(loss: Tensor, params: tuple[nn.Parameter, ...]) -> Tensor:
    if not loss.requires_grad or not params:
        return loss.detach().new_zeros(())
    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=True,
        allow_unused=True,
    )
    sq = loss.detach().new_zeros((), dtype=torch.float32)
    for grad in grads:
        if grad is None:
            continue
        sq = sq + grad.detach().float().pow(2).sum()
    return sq.sqrt()


def _weighted_grad_norm_metrics(
    losses: dict[str, Tensor],
    params: tuple[nn.Parameter, ...],
) -> dict[str, Tensor]:
    active = {
        name: loss
        for name, loss in losses.items()
        if isinstance(loss, torch.Tensor) and loss.requires_grad
    }
    if not active:
        return {}

    total = sum(active.values())
    total_norm = _grad_l2_norm(total, params)
    denom = total_norm.clamp(min=1e-12)

    metrics: dict[str, Tensor] = {"grad_norm_total_probe": total_norm}
    for name, loss in active.items():
        norm = _grad_l2_norm(loss, params)
        metrics[f"grad_norm_{name}"] = norm
        metrics[f"grad_ratio_{name}"] = norm / denom
    return metrics


def build_generator_step_fn(
    transformer: InteractionMaskTransformer,
    vq_model: nn.Module,
    *,
    cfg_drop_buckets: tuple[float, float, float],
    residual_transformer: ResidualTransformerWithInteraction | None = None,
    residual_loss_weight: float = 0.0,
    decoded_contact_aux_weight: float = 0.0,
    decoded_contact_aux_mode: str = "metric",
    decoded_contact_aux_temperature: float = 1.0,
    decoded_contact_aux_num_object_points: int = 256,
    decoded_contact_aux_rvq_path: str = "base_gt_residual",
    token_stride: int = 4,
    motion_mean: torch.Tensor | None = None,
    motion_std: torch.Tensor | None = None,
    residual_layer_diagnostics: bool = False,
    gradient_diagnostics_every_steps: int = 0,
):
    """Build the Stage B training step.

    Encodes motion → VQ tokens (frozen), encodes z_int → K/V tokens,
    runs the wrapped MaskTransformer's BERT-style masked-CE forward
    with per-sample CFG drops sampled from the 4-bucket categorical.

    Parameters
    ----------
    transformer
        The wrapped :class:`InteractionMaskTransformer`.
    vq_model
        The frozen MoMask RVQ-VAE; we call ``vq_model.encode(motion)``
        to get base-layer token IDs (matching MoMask's training step
        at ``backbones/momask/models/mask_transformer/transformer_trainer.py:38-56``).
    cfg_drop_buckets
        ``(p_drop_both, p_drop_int_only, p_drop_text_only)`` —
        passed straight to the model's ``forward``. ``None`` from
        the config disables CFG drops for val.
    token_stride
        VQ-VAE temporal downsample (4 for MoMask). Used to compute
        token-space ``m_lens``.
    motion_mean, motion_std
        ``(263,)`` torch tensors on the same device as ``transformer``.
        Required for the encoder normalization fix (v0.3-β-norm). Per
        ``analyses/2026-04-27_v0_3_root_cause_research.md`` v0.3-β
        diagnostic: MoMask VQ-VAE was trained on ``(raw - mean) / std``
        normalized features; feeding raw motion produces OOD-scale
        encoder inputs that quantize to the wrong codes (only 44.5% of
        GT path length preserved on round-trip vs 94.7% with normalized
        input). Apply ``(motion - mean) / std`` before ``vq_model.encode``.
    decoded_contact_aux_weight
        C2 decoded-space auxiliary loss weight. When positive, the step
        requests base logits and decodes a relaxed RVQ stack through the
        frozen VQ-VAE. ``base_gt_residual`` preserves the v0.9 path
        (soft base + GT residual ids); ``full_prediction`` is C2b
        (soft base + differentiable residual rollout).
    residual_layer_diagnostics
        When true, log active-layer CE/accuracy for residual RVQ layers
        q1..q5. This uses the same sampled active layer as training.
    gradient_diagnostics_every_steps
        If positive, compute weighted per-loss gradient L2 norms every N
        optimizer steps and log them as sparse epoch-averaged metrics.
    """
    grad_diag_params = _trainable_params(transformer)

    def step_fn(
        _model: nn.Module,
        batch: dict,
        global_step: int | None = None,
    ) -> dict[str, Tensor]:
        device = next(transformer.parameters()).device

        motion = batch["motion"].to(device).float()       # (B, T, 263) raw
        seq_len = batch["seq_len"].to(device).long()       # (B,)
        text = batch["text"]                               # list[str]

        # ---- Frozen VQ-VAE encode: (B, T, 263) → (B, S=T/4, Q) ----
        # Normalize first: VQ-VAE was trained on (raw - mean) / std
        # features (MoMask t2m_dataset.py:85). Without this, encoder
        # OOD-quantizes to a saturated default-prototype cluster and
        # the MaskTransformer's text→token prior is broken.
        # We follow MoMask's trainer pattern verbatim: use the BASE
        # quantiser layer only (``[..., 0]``); the residual layers are
        # the ResidualTransformer's job and are out of scope for Stage B.
        if motion_mean is None or motion_std is None:
            raise ValueError(
                "build_generator_step_fn requires motion_mean + motion_std "
                "(load via piano.data.humanml3d_repr.load_motion_stats). "
                "Without normalization, the VQ-VAE encoder OOD-quantizes "
                "and Stage B trains on a degraded token distribution.",
            )
        with torch.no_grad():
            motion_norm = (motion - motion_mean) / motion_std.clamp(min=1e-8)
            code_idx, _ = vq_model.encode(motion_norm)     # (B, S, Q)
            base_ids = code_idx[..., 0].long()             # (B, S)

        # Token-space sequence lengths.
        m_lens_tok = (seq_len // token_stride).clamp(min=1).long()

        # ---- Frozen CLIP text encode (pooled, MoMask convention) ----
        # MoMask's MaskTransformer ships a frozen CLIP under
        # ``self.clip_model``; ``encode_text`` returns the pooled (b, 512)
        # vector. We DO NOT use per-token CLIP here — the MoMask cond
        # path is a single prepended token, by design.
        with torch.no_grad():
            cond_vector = transformer.encode_text(text)    # (B, 512)
        cond_vector = cond_vector.to(device).float()

        # ---- Tokenise z_int from GT pseudo-labels ----
        # Per design §4.2: train against the clean GT signal; switch to
        # predictor output only at Stage 4 joint finetune. Decoupled
        # curricula + no predictor noise leakage + cheaper.
        # v0.2: also pass body-canonical-frame object pose channels.
        # HOIDataset(surface_obj_pose=True) computes them per __getitem__
        # using piano.utils.canonical_frame.world_to_canonical_object_pose.
        int_tokens_bf, int_pad_mask_bf = transformer.interaction_tokenizer(
            contact_state=batch["contact_state"].to(device).float(),
            contact_target_xyz=batch["contact_target_xyz"].to(device).float(),
            phase=batch["phase"].to(device).long(),
            support=batch["support"].to(device).long(),
            obj_com_canonical=batch["obj_com_canonical"].to(device).float(),
            obj_rot6d_canonical=batch["obj_rot6d_canonical"].to(device).float(),
            seq_lens=seq_len,
        )

        # ---- Forward + masked-CE loss with bucketed CFG drops ----
        out = transformer(
            ids=base_ids,
            cond_vector=cond_vector,
            m_lens_tok=m_lens_tok,
            int_tokens_bf=int_tokens_bf,
            int_padding_mask_bf=int_pad_mask_bf,
            cfg_drop_buckets=cfg_drop_buckets,
            return_logits=decoded_contact_aux_weight > 0.0,
        )
        out["loss_base"] = out["loss"]
        out["loss_weighted_base"] = out["loss_base"]
        weighted_losses: dict[str, Tensor] = {"base": out["loss_base"]}

        if residual_transformer is not None and residual_loss_weight > 0.0:
            res_int_kv = int_tokens_bf.transpose(0, 1).contiguous()
            res_out = residual_transformer.forward_with_int(
                all_indices=code_idx.long(),
                y=text,
                m_lens=m_lens_tok,
                int_kv=res_int_kv,
                int_padding_mask=int_pad_mask_bf,
                return_layer_metrics=residual_layer_diagnostics,
            )
            if residual_layer_diagnostics:
                res_loss, _res_pred, res_acc, res_layer_metrics = res_out
                out.update(res_layer_metrics)
            else:
                res_loss, _res_pred, res_acc = res_out
            weighted_res_loss = float(residual_loss_weight) * res_loss
            out["loss_residual"] = res_loss
            out["loss_weighted_residual"] = weighted_res_loss
            out["acc_residual"] = torch.as_tensor(
                res_acc, device=device, dtype=out["loss"].dtype,
            )
            out["loss"] = out["loss"] + weighted_res_loss
            weighted_losses["residual_weighted"] = weighted_res_loss

        if decoded_contact_aux_weight > 0.0:
            aux_loss, aux_metrics = decoded_contact_aux_loss(
                base_logits=out["logits"],
                all_indices=code_idx.long(),
                vq_model=vq_model,
                motion_mean=motion_mean,
                motion_std=motion_std,
                batch=batch,
                m_lens_tok=m_lens_tok,
                num_object_points=decoded_contact_aux_num_object_points,
                temperature=decoded_contact_aux_temperature,
                mode=decoded_contact_aux_mode,
                rvq_path=decoded_contact_aux_rvq_path,
                residual_transformer=residual_transformer,
                text=text,
                int_kv=(
                    int_tokens_bf.transpose(0, 1).contiguous()
                    if residual_transformer is not None else None
                ),
                int_padding_mask=int_pad_mask_bf,
            )
            weighted_aux_loss = float(decoded_contact_aux_weight) * aux_loss
            out["loss_decoded_contact"] = aux_loss
            out["loss_weighted_decoded_contact"] = weighted_aux_loss
            out.update(aux_metrics)
            out["loss"] = out["loss"] + weighted_aux_loss
            weighted_losses["decoded_contact_weighted"] = weighted_aux_loss
            out.pop("logits", None)
            out.pop("labels", None)
            out.pop("non_pad_mask", None)

        if (
            gradient_diagnostics_every_steps > 0
            and global_step is not None
            and global_step % gradient_diagnostics_every_steps == 0
            and torch.is_grad_enabled()
        ):
            out.update(_weighted_grad_norm_metrics(weighted_losses, grad_diag_params))
        # Diagnostic: mean γ_int across the 8 layers — useful to track
        # whether the new IntXAttn sublayers are actually learning to
        # contribute. Per design §6 decision tree: if γ_int stays at 0
        # while FID degrades, the new params aren't getting gradient.
        with torch.no_grad():
            gamma_mean = torch.stack([
                blk.gamma_int.detach().abs().mean()
                for blk in transformer.mask_transformer.seqTransEncoder.layers
            ]).mean()
        out["gamma_int_abs_mean"] = gamma_mean
        if residual_transformer is not None:
            with torch.no_grad():
                gamma_res_mean = torch.stack([
                    blk.gamma_int.detach().abs().mean()
                    for blk in residual_transformer.encoder.layers
                ]).mean()
            out["gamma_int_res_abs_mean"] = gamma_res_mean

        return out

    return step_fn


# ============================================================================
# Entrypoint
# ============================================================================

def run(config_path: str) -> None:
    cfg = OmegaConf.load(config_path)
    set_seed(cfg.training.get("seed", 42))

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        mixed_precision=cfg.training.get("mixed_precision", "bf16"),
    )
    device = accelerator.device

    # Sub-config: model architecture (matches MoMask pretrained dims).
    model_cfg = OmegaConf.load(cfg.model.config)

    # ---- Load frozen MoMask RVQ-VAE ----
    # Per analyses/early_setup.md: ``mu=0.99`` is set inside
    # ``build_momask_opt`` (default), so VQ-VAE loads cleanly.
    accelerator.print("Loading MoMask RVQ-VAE...")
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
    # vq_model is already eval+frozen by the loader.

    # ---- Load pretrained MoMask MaskTransformer + wrap with IntXAttn ----
    accelerator.print("Loading MoMask MaskTransformer + adding IntXAttn sublayers...")
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
    # γ-gate kind: training-config override > model-config default > "scalar".
    # v0.6 sets ``cfg.model.gamma_kind: per_head`` for the LLaMA-Adapter-style
    # 48-dof gate (6 heads × 8 layers); v0.1-v0.5 leave both unset and fall
    # through to the scalar 8-dof gate.
    gamma_kind = str(cfg.model.get(
        "gamma_kind",
        mt_cfg.interaction_cross_attn.get("gamma_kind", "scalar"),
    ))
    # Wrapper kind: "v0.6" (default — per-block IntXAttn on a single
    # finetuned encoder) or "v0.3-delta" (trainable-copy InterControl
    # pattern: deepcopy ctrl branch + per-layer zero-init linear
    # connectors + frozen main branch). v0.1-v0.7 leave the key unset
    # → backward-compatible v0.6 path.
    wrapper_kind = str(cfg.model.get("wrapper_kind", "v0.6"))
    accelerator.print(f"γ-gate kind: {gamma_kind}, wrapper_kind: {wrapper_kind}")
    transformer = InteractionMaskTransformer(
        mask_transformer=base_mt,
        interaction_tokenizer=interaction_tokenizer,
        interaction_drop_prob=float(mt_cfg.get("interaction_drop_prob", 0.1)),
        zero_init_gamma=bool(mt_cfg.interaction_cross_attn.get("zero_init", True)),
        max_token_seq_length=max_seq_length_tokens,
        gamma_kind=gamma_kind,
        wrapper_kind=wrapper_kind,
    )
    transformer.to(device)

    residual_wrapper: ResidualTransformerWithInteraction | None = None
    residual_int_cfg = cfg.model.get("residual_int_xattn", None)
    residual_int_enabled = (
        residual_int_cfg is not None
        and bool(residual_int_cfg.get("enabled", False))
    )
    if residual_int_enabled:
        accelerator.print(
            "Loading MoMask ResidualTransformer + adding residual IntXAttn sublayers...",
        )
        res_ckpt = cfg.model.checkpoints.get("residual_transformer", None)
        if res_ckpt is None:
            raise ValueError(
                "model.residual_int_xattn.enabled=true requires "
                "model.checkpoints.residual_transformer.",
            )
        res_base = load_momask_residual_transformer(
            res_ckpt,
            code_dim=model_cfg.residual_transformer.get("code_dim", 512),
            latent_dim=model_cfg.residual_transformer.latent_dim,
            ff_size=model_cfg.residual_transformer.ff_size,
            num_layers=model_cfg.residual_transformer.num_layers,
            num_heads=model_cfg.residual_transformer.num_heads,
            dropout=model_cfg.residual_transformer.dropout,
            cond_drop_prob=model_cfg.residual_transformer.cond_drop_prob,
            num_quantizers=model_cfg.vq_vae.num_quantizers,
            shared_codebook=model_cfg.residual_transformer.shared_codebook,
            share_weight=model_cfg.residual_transformer.share_weight,
            device=str(device),
        )
        residual_wrapper = ResidualTransformerWithInteraction(
            residual_transformer=res_base,
            d_model=model_cfg.residual_transformer.latent_dim,
            num_heads=model_cfg.residual_transformer.num_heads,
            dropout=float(residual_int_cfg.get(
                "dropout", model_cfg.residual_transformer.dropout,
            )),
            zero_init_gamma=bool(residual_int_cfg.get("zero_init_gamma", True)),
            gamma_kind=str(residual_int_cfg.get("gamma_kind", gamma_kind)),
        ).to(device)
        # Make the residual wrapper part of the main module tree so
        # DDP, gradient clipping, .train/.eval, and checkpoints include it.
        transformer.residual_transformer = residual_wrapper

    # ---- Datasets + dataloaders ----
    train_dataset = _build_dataset(cfg, split_override=None, enable_augment=True)
    train_split_info = _resolve_split(cfg, split_override=None)
    accelerator.print(
        f"Train dataset: {len(train_dataset)} clips "
        f"(split={train_split_info['label']})",
    )

    val_dataloader = None
    val_every_epochs = int(cfg.training.get("val_every_epochs", 0))
    if val_every_epochs > 0:
        val_dataset = _build_dataset(cfg, split_override="val", enable_augment=False)
        val_split_info = _resolve_split(cfg, split_override="val")
        accelerator.print(
            f"Val dataset:   {len(val_dataset)} clips "
            f"(split={val_split_info['label']}, augmentation disabled)",
        )

    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg.training.batch_size,
        shuffle=True, collate_fn=collate_hoi,
        num_workers=int(cfg.training.get("num_workers", 4)),
        pin_memory=True, drop_last=True,
    )
    if val_every_epochs > 0:
        val_dataloader = DataLoader(
            val_dataset, batch_size=cfg.training.batch_size,
            shuffle=False, collate_fn=collate_hoi,
            num_workers=int(cfg.training.get("num_workers", 4)),
            pin_memory=True, drop_last=False,
        )

    # ---- Optimiser (two LR groups) + scheduler ----
    new_params = transformer.new_parameters()
    backbone_params = transformer.backbone_parameters()
    if residual_wrapper is not None:
        new_params.extend(residual_wrapper.new_parameters())
        backbone_params.extend(residual_wrapper.backbone_parameters())
    optimizer = build_two_group_optimizer(
        new_params=new_params,
        backbone_params=backbone_params,
        new_lr=float(cfg.training.optimizer.new_lr),
        backbone_lr=float(cfg.training.optimizer.backbone_lr),
        weight_decay=float(cfg.training.optimizer.weight_decay),
        betas=tuple(cfg.training.optimizer.betas),
    )
    accum = cfg.training.gradient_accumulation_steps
    steps_per_epoch = max(1, len(train_dataloader) // accum)
    total_steps = steps_per_epoch * cfg.training.num_epochs
    scheduler = build_scheduler(
        optimizer, cfg.training.scheduler.warmup_steps, total_steps,
    )

    # ---- Accelerate prepare ----
    # vq_model has ``requires_grad=False`` everywhere (no optimiser
    # entries). Still call .to(device) but don't run through .prepare()
    # — Accelerate's DDP wrapper would add overhead with no benefit
    # (HF Accelerate's recommended pattern for frozen sub-modules).
    transformer, optimizer, train_dataloader, scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, scheduler,
    )
    if val_dataloader is not None:
        val_dataloader = accelerator.prepare(val_dataloader)

    # ---- Step fn ----
    cfg_buckets_cfg = cfg.training.get("cfg_drop_buckets", None)
    if cfg_buckets_cfg is None:
        cfg_drop_buckets = (0.10, 0.10, 0.05)
    else:
        cfg_drop_buckets = (
            float(cfg_buckets_cfg.drop_both),
            float(cfg_buckets_cfg.drop_int_only),
            float(cfg_buckets_cfg.drop_text_only),
        )

    # Resolve the inner transformer (Accelerate may have wrapped it in DDP).
    inner_transformer = accelerator.unwrap_model(transformer)
    inner_residual = getattr(inner_transformer, "residual_transformer", None)

    # Load HumanML3D motion mean/std (v0.3-β-norm fix). VQ-VAE was
    # trained on normalized features; without this the encoder OOD-
    # quantizes and Stage B trains on a degraded token distribution.
    # See analyses/2026-04-27_v0_3_root_cause_research.md and
    # scripts/stage_b_generator/diagnose_vq_pipeline.py for the empirical
    # demonstration (raw input preserves 44.5% of GT path length on
    # round-trip; normalized input preserves 94.7%).
    motion_mean_np, motion_std_np = load_motion_stats(cfg.model.checkpoints.vq_vae)
    motion_mean_t = torch.from_numpy(motion_mean_np).float().to(device)
    motion_std_t = torch.from_numpy(motion_std_np).float().to(device)
    accelerator.print(
        f"Loaded HumanML3D motion stats: mean.shape={motion_mean_np.shape}, "
        f"std range [{motion_std_np.min():.3f}, {motion_std_np.max():.3f}]",
    )

    decoded_aux_cfg = cfg.training.get("decoded_contact_aux", None)
    decoded_aux_enabled = (
        decoded_aux_cfg is not None
        and bool(decoded_aux_cfg.get("enabled", False))
    )
    decoded_aux_weight = (
        float(decoded_aux_cfg.get("weight", 0.0))
        if decoded_aux_enabled else 0.0
    )
    if decoded_aux_weight > 0.0:
        accelerator.print(
            "Decoded contact aux enabled: "
            f"weight={decoded_aux_weight}, "
            f"mode={decoded_aux_cfg.get('mode', 'metric')}, "
            f"rvq_path={decoded_aux_cfg.get('rvq_path', 'base_gt_residual')}, "
            f"temperature={float(decoded_aux_cfg.get('temperature', 1.0))}, "
            f"num_object_points={int(decoded_aux_cfg.get('num_object_points', 256))}",
        )

    diagnostics_cfg = cfg.training.get("diagnostics", None)
    grad_diag_every = 0
    residual_layer_diagnostics = False
    if diagnostics_cfg is not None:
        grad_diag_cfg = diagnostics_cfg.get("gradient_norms", None)
        if grad_diag_cfg is not None and bool(grad_diag_cfg.get("enabled", False)):
            grad_diag_every = int(grad_diag_cfg.get("every_n_steps", 0))
            if grad_diag_every <= 0:
                grad_diag_every = len(train_dataloader)
            accelerator.print(
                "Gradient diagnostics enabled: "
                f"every_n_steps={grad_diag_every}",
            )

        residual_layer_cfg = diagnostics_cfg.get("residual_layers", None)
        residual_layer_diagnostics = (
            residual_layer_cfg is not None
            and bool(residual_layer_cfg.get("enabled", False))
        )
        if residual_layer_diagnostics:
            accelerator.print("Residual per-RVQ-layer diagnostics enabled")

    step_fn = build_generator_step_fn(
        transformer=inner_transformer,
        vq_model=vq_model,
        cfg_drop_buckets=cfg_drop_buckets,
        residual_transformer=inner_residual,
        residual_loss_weight=float(cfg.training.get("residual_loss_weight", 0.0)),
        decoded_contact_aux_weight=decoded_aux_weight,
        decoded_contact_aux_mode=(
            str(decoded_aux_cfg.get("mode", "metric"))
            if decoded_aux_cfg is not None else "metric"
        ),
        decoded_contact_aux_temperature=(
            float(decoded_aux_cfg.get("temperature", 1.0))
            if decoded_aux_cfg is not None else 1.0
        ),
        decoded_contact_aux_num_object_points=(
            int(decoded_aux_cfg.get("num_object_points", 256))
            if decoded_aux_cfg is not None else 256
        ),
        decoded_contact_aux_rvq_path=(
            str(decoded_aux_cfg.get("rvq_path", "base_gt_residual"))
            if decoded_aux_cfg is not None else "base_gt_residual"
        ),
        token_stride=token_stride,
        motion_mean=motion_mean_t,
        motion_std=motion_std_t,
        residual_layer_diagnostics=residual_layer_diagnostics,
        gradient_diagnostics_every_steps=grad_diag_every,
    )

    # ---- Wandb ----
    wandb_run = None
    if accelerator.is_main_process:
        try:
            import wandb
            wandb_run = wandb.init(
                project=cfg.logging.project, name=cfg.logging.run_name,
            )
        except ImportError:
            pass

    # ---- Contact-aware checkpointing (B1) ----
    # Per analyses/2026-04-28_v0_3_delta_retrain_and_v0_5_contact.md: the
    # training objective (masked-CE) is empirically decoupled from the
    # ship metric (geometric body-to-object distance). We additionally
    # save best_contact.pt selected on a fixed mini-eval-set's contact
    # distance. Disabled by default for backward compatibility with
    # configs that don't declare training.contact_eval; future Stage B
    # configs should set training.contact_eval.enabled: true.
    contact_eval_cfg = cfg.training.get("contact_eval", None)
    contact_eval_enabled = (
        contact_eval_cfg is not None
        and bool(contact_eval_cfg.get("enabled", False))
        and val_every_epochs > 0
    )
    contact_eval_fn = None
    if contact_eval_enabled:
        if inner_residual is not None:
            res_transformer = inner_residual
            accelerator.print("Contact eval: using trained C1 residual wrapper.")
        else:
            # Backward-compatible B1 path: load the frozen residual
            # transformer only for contact eval.
            accelerator.print("Loading MoMask Residual Transformer for contact eval...")
            res_ckpt = cfg.model.checkpoints.get("residual_transformer", None)
            if res_ckpt is None:
                raise ValueError(
                    "training.contact_eval.enabled=true requires "
                    "model.checkpoints.residual_transformer.",
                )
            res_transformer = load_momask_residual_transformer(
                res_ckpt,
                code_dim=model_cfg.residual_transformer.get("code_dim", 512),
                latent_dim=model_cfg.residual_transformer.latent_dim,
                ff_size=model_cfg.residual_transformer.ff_size,
                num_layers=model_cfg.residual_transformer.num_layers,
                num_heads=model_cfg.residual_transformer.num_heads,
                dropout=model_cfg.residual_transformer.dropout,
                cond_drop_prob=model_cfg.residual_transformer.cond_drop_prob,
                num_quantizers=model_cfg.vq_vae.num_quantizers,
                shared_codebook=model_cfg.residual_transformer.shared_codebook,
                share_weight=model_cfg.residual_transformer.share_weight,
                device=str(device),
            )
        res_transformer.eval()

        # Build a deterministic, type-diverse fixed batch. The sampler
        # balances by dataset subset first and object id second so the
        # best-contact checkpoint is not selected on one narrow slice.
        # Training-time checkpoint selection should use a larger sample
        # than the 20-clip offline visualization set; configs can specify
        # num_clips_per_subset (e.g. 20 x 4 subsets = 80 clips).
        num_clips = resolve_eval_clip_count(
            val_dataset,
            num_clips=int(contact_eval_cfg.get("num_clips", 20)),
            num_clips_per_subset=(
                int(contact_eval_cfg.get("num_clips_per_subset"))
                if contact_eval_cfg.get("num_clips_per_subset", None) is not None
                else None
            ),
        )
        selected_idx = select_eval_clip_indices(
            val_dataset,
            num_clips,
            seed=int(contact_eval_cfg.get("seed", cfg.training.get("seed", 42))),
        )
        fixed_val_batch = collate_hoi([val_dataset[i] for i in selected_idx])
        selected_rows = describe_eval_clip_selection(val_dataset, selected_idx)
        accelerator.print(
            f"Contact eval: {len(selected_idx)} stratified fixed val clips "
            f"(seq_ids={fixed_val_batch.get('seq_id', '?')[:len(selected_idx)]})",
        )
        for row in selected_rows:
            accelerator.print(
                "  contact eval clip "
                f"idx={row['index']} subset={row['subset']} "
                f"object={row['object_id']} seq={row['seq_id']}",
            )

        contact_eval_fn = build_contact_eval_fn(
            transformer=inner_transformer,
            vq_model=vq_model,
            res_transformer=res_transformer,
            fixed_val_batch=fixed_val_batch,
            motion_mean=motion_mean_t,
            motion_std=motion_std_t,
            device=device,
            token_stride=token_stride,
            w_text=float(contact_eval_cfg.get("w_text", 4.0)),
            w_int=float(contact_eval_cfg.get("w_int", 2.0)),
        )

    # ---- Train ----
    run_training_loop(
        accelerator=accelerator,
        model=transformer,
        dataloader=train_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        step_fn=step_fn,
        num_epochs=cfg.training.num_epochs,
        output_dir=cfg.output_dir,
        log_every=cfg.logging.log_every_n_steps,
        save_every_epochs=cfg.logging.save_every_n_epochs,
        max_grad_norm=cfg.training.max_grad_norm,
        wandb_run=wandb_run,
        val_dataloader=val_dataloader,
        val_every_epochs=val_every_epochs,
        val_best_key=cfg.training.get("val_best_key", "loss"),
        contact_eval_fn=contact_eval_fn,
        contact_best_key=str(
            contact_eval_cfg.get("best_key", "mean_min_dist")
            if contact_eval_cfg is not None else "mean_min_dist"
        ),
    )


def main() -> None:
    """CLI entry point for ``piano-train-generator`` (Stage B)."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/training/generator.yaml",
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
