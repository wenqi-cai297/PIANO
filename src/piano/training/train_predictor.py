"""Stage A: Train the Interaction Predictor.

Trains the predictor to map (text, object, init_pose) → interaction latent,
supervised by pseudo-labels extracted from HOI data.

Usage:
    accelerate launch --config_file configs/accelerate_config.yaml \\
        -m piano.training.train_predictor \\
        --config configs/training/predictor.yaml
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
from torch.utils.data import ConcatDataset, DataLoader

from piano.data.dataset import (
    AugmentConfig,
    HOIDataset,
    build_object_split,
    build_subject_split,
    collate_hoi,
    compute_class_priors,
    extract_subject_id,
)
from piano.models.interaction_predictor import InteractionPredictor
from piano.models.object_encoder import ObjectEncoder
from piano.training.losses import PredictorLoss
from piano.training.priors import PhysicalPriors
from piano.training.trainer import (
    build_optimizer_with_decay_groups,
    build_scheduler,
    run_training_loop,
)
from piano.utils.clip_utils import encode_text_per_token, load_clip_text_encoder
from piano.utils.io_utils import load_json


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------

def _read_metadata(roots: list) -> list[tuple[str, dict]]:
    """Read every (subset_name, metadata_entry) pair across all roots.

    Reads each root's ``metadata_clean.json`` (or ``metadata.json``
    fallback — the clean variant is what HOIDataset prefers at train
    time, so the split must be computed on the same universe). Returns
    a flat list of (subset_name, entry) tuples; downstream callers can
    project to object_ids or subject_ids as needed.
    """
    from pathlib import Path

    out: list[tuple[str, dict]] = []
    for entry in roots:
        root = Path(entry.root)
        meta_path = root / "metadata_clean.json"
        if not meta_path.exists():
            meta_path = root / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata not found in {root}")
        subset_name = root.name
        for m in load_json(meta_path):
            out.append((subset_name, m))
    return out


def _collect_object_ids(roots: list) -> list[str]:
    """Sorted unique object_ids across all roots (legacy / object_split path)."""
    seen: set[str] = set()
    for _, m in _read_metadata(roots):
        obj_id = m.get("object_id")
        if obj_id is not None:
            seen.add(obj_id)
    return sorted(seen)


def _collect_subject_keys(roots: list) -> list[tuple[str, str]]:
    """Collect (subset_name, raw_subject_id) across all roots.

    Used to feed ``build_subject_split``. Drops entries whose seq_id
    doesn't parse for the subset (subset has no pattern, or seq_id
    format unexpected) — these would also be dropped at HOIDataset
    filter time, so excluding them from the split universe is consistent.
    Deduped within each subset; outer list preserves duplicates only
    across subsets (which never collide since pattern keys are
    subset-specific).
    """
    seen: set[tuple[str, str]] = set()
    for subset_name, m in _read_metadata(roots):
        raw_id = extract_subject_id(subset_name, m.get("seq_id", ""))
        if raw_id is not None:
            seen.add((subset_name, raw_id))
    return sorted(seen)


def _resolve_split(cfg, split_override: str | None) -> dict:
    """Pick which split to apply (subject vs object) and return the
    filter sets HOIDataset needs.

    Precedence:
      1. ``data.subject_split.enabled = true`` → primary path
         (used as of 2026-04-26 / v6+).
      2. ``data.object_split.enabled = true`` → secondary path,
         kept for the optional novel-object ablation eval.
      3. Neither enabled → no filter (use all clips).

    Returns a dict with keys ``object_id_filter``, ``subject_id_filter``
    (one or both can be None), and a ``label`` for log printing.
    """
    subj_cfg = cfg.data.get("subject_split")
    obj_cfg = cfg.data.get("object_split")
    bucket = split_override

    # Primary: subject_split
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
            raise ValueError(f"unknown subject_split bucket: {b!r}; expected train/val/all")
        return {
            "object_id_filter": None,
            "subject_id_filter": allowed,
            "label": f"subject_split[{b}] ({len(allowed) if allowed else 'all'} subjects)",
        }

    # Secondary: object_split (legacy / ablation only)
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

    # No split configured
    return {
        "object_id_filter": None,
        "subject_id_filter": None,
        "label": "no_split (all clips)",
    }


def _build_dataset(
    cfg,
    split_override: str | None = None,
    enable_augment: bool = True,
) -> ConcatDataset:
    """Build a Stage A dataset from the 4 InterAct subset roots.

    Applies the configured split (subject_split if enabled, else
    object_split, else no filter). ``split_override`` lets the caller
    pick a bucket different from the config default — used by the val
    loader to grab the "val" bucket when training is on "train".
    Augmentation is toggleable independent of the split so the val
    loader can share this builder with augmentation disabled.
    """
    split_info = _resolve_split(cfg, split_override)

    # Augmentation — mirror + Y-rotation + pc jitter (train only by default).
    aug_cfg = cfg.data.get("augmentation", None)
    augment = None
    if enable_augment and aug_cfg is not None and aug_cfg.get("enabled", False):
        augment = AugmentConfig(
            enabled=True,
            mirror_prob=float(aug_cfg.get("mirror_prob", 0.0)),
            mirror_duplicate=bool(aug_cfg.get("mirror_duplicate", False)),
            rotate_around_y_prob=float(aug_cfg.get("rotate_around_y_prob", 0.0)),
            pc_jitter_std=float(aug_cfg.get("pc_jitter_std", 0.0)),
        )

    # Per-subset HOIDataset instances, concatenated.
    #
    # Path resolution priorities (highest first):
    #   1. ``data.pseudo_label_dir`` (absolute, single dir, all subsets share it)
    #   2. ``data.pseudo_label_subdir`` (relative to each subset root —
    #      lets v12_strict / future label versions live as a sibling
    #      directory under each subset's own pseudo_labels tree)
    #   3. None — HOIDataset falls back to ``<root>/pseudo_labels``
    pseudo_label_dir_global = cfg.data.get("pseudo_label_dir", None)
    pseudo_label_subdir = cfg.data.get("pseudo_label_subdir", None)
    datasets = []
    for entry in cfg.data.datasets:
        if pseudo_label_dir_global is not None:
            this_pseudo_label_dir = pseudo_label_dir_global
        elif pseudo_label_subdir is not None:
            this_pseudo_label_dir = str(Path(entry.root) / pseudo_label_subdir)
        else:
            this_pseudo_label_dir = None
        ds = HOIDataset(
            root=entry.root,
            pseudo_label_dir=this_pseudo_label_dir,
            max_seq_length=cfg.data.max_seq_length,
            object_id_filter=split_info["object_id_filter"],
            subject_id_filter=split_info["subject_id_filter"],
            augment=augment,
            # v9.1: collapse hand_support → both_feet at dataloader.
            support_collapse_hand_support=bool(
                cfg.data.get("support_collapse_hand_support", False)
            ),
        )
        datasets.append(ds)

    return ConcatDataset(datasets)


# ---------------------------------------------------------------------------
# Step function
# ---------------------------------------------------------------------------

def build_predictor_step_fn(
    predictor: InteractionPredictor,
    object_encoder: ObjectEncoder,
    clip_model: nn.Module,
    criterion: PredictorLoss,
    priors: PhysicalPriors,
    device: torch.device,
    prior_warmup_steps: int = 0,
    teacher_forcing_schedule: dict | None = None,
    epochs_per_step: float | None = None,
):
    """Build the step function for predictor training.

    The returned callable takes (model, batch, global_step=...) and
    returns a loss dict. ``global_step`` is the optimizer-step counter
    fed in by ``run_training_loop`` — we use it to linearly ramp the
    physical prior contribution from 0 to full weight over the first
    ``prior_warmup_steps`` calls (PhysDiff / CG-HOI convention).

    Parameters
    ----------
    teacher_forcing_schedule : optional dict with keys
        ``high_until_epoch``, ``low_after_epoch``, ``high_prob``,
        ``low_prob``. When set + structured_head is enabled, the
        StructuredHead receives GT contact + GT phase as the
        conditioning input to downstream heads. Probability is
        annealed linearly from ``high_prob`` (epochs ≤ high_until)
        to ``low_prob`` (epochs ≥ low_after). Bengio et al. NeurIPS
        2015 scheduled-sampling convention.
    epochs_per_step : 1 / num_steps_per_epoch — how many epochs each
        global_step represents. Required iff teacher_forcing_schedule
        is set.
    """
    def step_fn(
        _model: nn.Module,
        batch: dict,
        global_step: int = 0,
    ) -> dict[str, Tensor]:
        # Text: CLIP per-token features + padding mask. Returned in
        # CLIP's native dtype (typically fp16 on GPU); the bf16 autocast
        # context handles casting through the predictor's Linear layers.
        text_features, text_mask = encode_text_per_token(
            clip_model, batch["text"], device,
        )

        # Object tokens. v8 needs token positions too for affordance
        # heatmap GT construction; encoder returns (xyz, features) when
        # ``return_xyz=True``. Predictor's structured_head flag decides
        # whether this is consumed.
        use_structured = getattr(predictor, "structured_head", False) or \
            (hasattr(predictor, "module") and getattr(predictor.module, "structured_head", False))
        if use_structured:
            obj_xyz, obj_tokens = object_encoder(batch["object_pc"], return_xyz=True)
        else:
            obj_tokens = object_encoder(batch["object_pc"])
            obj_xyz = None

        # Initial pose: first-frame SMPL-22 joint positions (66-d).
        # HumanML3D 263-d frame 0 has undefined velocities (process_file
        # drops the first frame for velocity computation), so we use
        # the raw joint positions instead.
        B = batch["joints"].shape[0]
        init_pose = batch["joints"][:, 0, :, :].reshape(B, -1)  # (B, 66)

        seq_len = batch["seq_len"]
        max_T = batch["motion"].shape[1]

        # Downstream conditioning input: depends on structured-head mode.
        #
        # downstream_mode="tf" (v8 default): per-batch coin flip on
        # tf_prob (Bengio NeurIPS 2015 scheduled sampling). When True,
        # downstream heads see GT; when False, see pred.
        #
        # downstream_mode="mask" (v8.1, MoMask CVPR 2024): the head
        # itself draws a per-batch mask_ratio ~ Uniform[0, 1] and mixes
        # GT with pred via Bernoulli mask, training on every mix
        # simultaneously. The step_fn just always passes GT; the head
        # ignores the teacher_forcing argument.
        teacher_forcing = False
        sh_module = predictor.module.head if hasattr(predictor, "module") else predictor.head if use_structured else None
        downstream_mode = getattr(sh_module, "downstream_mode", "tf") if sh_module is not None else "tf"
        if use_structured and downstream_mode == "tf" and teacher_forcing_schedule is not None and predictor.training:
            assert epochs_per_step is not None
            current_epoch = float(global_step) * float(epochs_per_step)
            high_until = float(teacher_forcing_schedule.get("high_until_epoch", 50))
            low_after = float(teacher_forcing_schedule.get("low_after_epoch", 80))
            high_prob = float(teacher_forcing_schedule.get("high_prob", 1.0))
            low_prob = float(teacher_forcing_schedule.get("low_prob", 0.5))
            if current_epoch <= high_until:
                tf_prob = high_prob
            elif current_epoch >= low_after:
                tf_prob = low_prob
            else:
                # linear anneal between the two anchors
                alpha = (current_epoch - high_until) / max(low_after - high_until, 1e-6)
                tf_prob = high_prob + alpha * (low_prob - high_prob)
            teacher_forcing = bool(torch.rand(1).item() < tf_prob)

        # v9.2: per-frame joints for motion-aware trunk. Pass in
        # training (predictor's _build_joint_signal applies random
        # masking internally) and during eval (no masking, full info).
        # Stage A standalone inference (Stage B integration) calls
        # predictor without joints → predictor falls back to all-mask.
        underlying = predictor.module if hasattr(predictor, "module") else predictor
        joints_per_frame = batch["joints"] if getattr(underlying, "motion_aware_trunk", False) else None

        # In mask mode, always pass GT — the head's _mix_with_gt does the
        # Bernoulli mix internally. In TF mode, only pass GT when the
        # coin flip says so (matches v8 behaviour).
        if use_structured:
            pass_gt = (downstream_mode == "mask") or teacher_forcing
            pred = predictor(
                text_features, obj_tokens, init_pose,
                seq_length=max_T,
                text_key_padding_mask=text_mask,
                object_xyz=obj_xyz,
                gt_contact=batch["contact_state"] if pass_gt else None,
                gt_phase=batch["phase"].long() if pass_gt else None,
                teacher_forcing=teacher_forcing,
                joints_per_frame=joints_per_frame,
            )
        else:
            pred = predictor(
                text_features, obj_tokens, init_pose,
                seq_length=max_T,
                text_key_padding_mask=text_mask,
                joints_per_frame=joints_per_frame,
            )

        # Frame mask (True for valid, non-padded frames)
        frame_mask = (
            torch.arange(max_T, device=seq_len.device).unsqueeze(0)
            < seq_len.unsqueeze(1)
        )

        # Supervision loss. v7-fix path uses smooth-L1 on xyz; v8 path
        # uses KL-div on attention over object tokens (requires obj_xyz).
        loss_dict = criterion(
            pred,
            gt_contact=batch["contact_state"],
            gt_target=batch["contact_target_xyz"],
            gt_phase=batch["phase"].long(),
            gt_support=batch["support"].long(),
            mask=frame_mask,
            object_xyz=obj_xyz,
        )

        # Physical prior regularization, linearly warmed up from 0. On
        # a random-init predictor, prior gradients would otherwise
        # dominate the first few hundred steps and pull the model away
        # from fitting the pseudo-labels. Ramping lets the data fit lead.
        joints = batch.get("joints")
        prior_dict = priors(pred, joints=joints, mask=frame_mask)
        if prior_warmup_steps > 0:
            prior_scale = min(1.0, float(global_step) / float(prior_warmup_steps))
        else:
            prior_scale = 1.0
        loss_dict["loss_priors"] = prior_dict["loss"]
        loss_dict["prior_scale"] = torch.tensor(
            prior_scale, device=prior_dict["loss"].device,
        )
        loss_dict["loss"] = loss_dict["loss"] + prior_scale * prior_dict["loss"]

        return loss_dict

    return step_fn


# ---------------------------------------------------------------------------
# Training entrypoint
# ---------------------------------------------------------------------------

def run(config_path: str) -> None:
    """Run Stage A training."""
    cfg = OmegaConf.load(config_path)
    set_seed(42)

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        mixed_precision="bf16",
    )
    device = accelerator.device

    # Sub-configs. These are the *model* configs referenced by the
    # training yaml; we read them explicitly so all hyperparameters are
    # auditable from the top-level config tree.
    model_cfg = OmegaConf.load(cfg.model.config)
    obj_cfg = OmegaConf.load(cfg.model.object_encoder_config)

    # v9.1: allow training yaml to override model/output fields (e.g.,
    # num_support_states from 4 → 3 when collapsing hand_support).
    output_cfg = OmegaConf.merge(
        model_cfg.output,
        cfg.model.get("output", {}),
    )

    # Models
    tr_cfg = model_cfg.get("temporal_refine", {})
    # structured_head: model file holds the defaults; training yaml's
    # ``cfg.model.structured_head`` block overrides per-key. This is
    # what enables `predictor_v8_structured.yaml` to flip the flag on
    # without editing the shared model file.
    sh_cfg = OmegaConf.merge(
        model_cfg.get("structured_head", {}),
        cfg.model.get("structured_head", {}),
    )
    accelerator.print(
        f"[predictor cfg] structured_head.enabled = {bool(sh_cfg.get('enabled', False))}"
    )
    # v9.2: motion-aware trunk + ASL contact loss activation log.
    # Resolved here (not later) so a quick startup-log scan tells you
    # which v9.x branch is actually running before the slow data-loading
    # banner. Mirrors the PLAN.md "Startup log must show" checklist.
    motion_cfg_merged = OmegaConf.merge(
        model_cfg.get("motion_aware_trunk", {}),
        cfg.model.get("motion_aware_trunk", {}),
    )
    accelerator.print(
        f"[predictor cfg] motion_aware_trunk.enabled = "
        f"{bool(motion_cfg_merged.get('enabled', False))} "
        f"(joint_input_dim={int(motion_cfg_merged.get('joint_input_dim', 66))})"
    )
    accelerator.print(
        f"[predictor cfg] contact_loss_kind = "
        f"{str(cfg.loss.get('contact_loss_kind', 'bce'))} "
        f"(asl_gamma_pos={float(cfg.loss.get('contact_asl_gamma_pos', 0.0))}, "
        f"asl_gamma_neg={float(cfg.loss.get('contact_asl_gamma_neg', 4.0))}, "
        f"asl_prob_shift={float(cfg.loss.get('contact_asl_prob_shift', 0.05))}, "
        f"use_contact_pos_weight={bool(cfg.loss.get('use_contact_pos_weight', False))})"
    )
    # v9.4: target_attn architectural flags (positional encoding,
    # aux xyz L2, top-K min positives). These three together are
    # Phase 0 of analyses/2026-05-04_target_attn_architectural_optimization.md.
    accelerator.print(
        f"[predictor cfg] target_pos_enc = "
        f"{bool(sh_cfg.get('target_pos_enc', False))} "
        f"(frequencies={int(sh_cfg.get('target_pos_enc_frequencies', 6))}, "
        f"coord_scale={float(sh_cfg.get('target_pos_enc_coord_scale', 1.0))})"
    )
    accelerator.print(
        f"[predictor cfg] target_aux_xyz_weight = "
        f"{float(cfg.loss.get('target_aux_xyz_weight', 0.0))} "
        f"(focal_dice gets aux smooth_l1 spatial gradient when > 0)"
    )
    accelerator.print(
        f"[predictor cfg] target_topk_min_positives = "
        f"{int(cfg.loss.get('target_topk_min_positives', 0))} "
        f"(GT mask = within-tau ∪ top-K nearest; K=1 unlocks IoU ceiling)"
    )
    # v9.5: hierarchical decoder + patch CE.
    accelerator.print(
        f"[predictor cfg] target_attn_kind = "
        f"{str(sh_cfg.get('target_attn_kind', 'single_layer'))} "
        f"(hierarchical_mask_decoder: Mask2Former/Mask3D mask attention "
        f"+ explicit patch head; num_patches="
        f"{int(sh_cfg.get('target_num_patches', 16))})"
    )
    accelerator.print(
        f"[predictor cfg] target_patch_weight = "
        f"{float(cfg.loss.get('target_patch_weight', 0.0))} "
        f"(hierarchical patch CE; only fires under hierarchical_mask_decoder)"
    )
    predictor = InteractionPredictor(
        d_model=model_cfg.encoder.d_model,
        num_layers=model_cfg.encoder.num_layers,
        num_heads=model_cfg.encoder.num_heads,
        dim_feedforward=model_cfg.encoder.dim_feedforward,
        dropout=model_cfg.encoder.dropout,
        text_dim=model_cfg.input.text_dim,
        pose_dim=model_cfg.input.pose_dim,
        max_seq_length=model_cfg.sequence.max_length,
        num_body_parts=int(output_cfg.num_body_parts),
        target_coord_dim=int(output_cfg.get("target_coord_dim", 3)),
        num_phases=int(output_cfg.num_phases),
        num_support_states=int(output_cfg.num_support_states),
        temporal_refine_enabled=bool(tr_cfg.get("enabled", True)),
        temporal_refine_kernel_size=int(tr_cfg.get("kernel_size", 5)),
        temporal_refine_dropout=float(tr_cfg.get("dropout", 0.1)),
        # v8 (2026-05-05): structured head. Default off (legacy heads).
        structured_head=bool(sh_cfg.get("enabled", False)),
        structured_head_d_emb=int(sh_cfg.get("d_emb", 64)),
        structured_head_hidden=int(sh_cfg.get("hidden", 256)),
        structured_head_attn_heads=int(sh_cfg.get("attn_heads", 6)),
        # v8.1: downstream conditioning mode + target attention output
        # form. Defaults preserve v8 behaviour.
        structured_head_downstream_mode=str(sh_cfg.get("downstream_mode", "tf")),
        structured_head_target_attn_output=str(
            sh_cfg.get("target_attn_output", "softmax")
        ),
        # v9: target attention kind + decoder hyperparameters.
        structured_head_target_attn_kind=str(sh_cfg.get("target_attn_kind", "single_layer")),
        structured_head_target_decoder_layers=int(sh_cfg.get("target_decoder_layers", 4)),
        structured_head_target_decoder_ffn=int(sh_cfg.get("target_decoder_ffn", 1024)),
        # v9.4: positional encoding on object tokens for the mask decoder.
        structured_head_target_pos_enc=bool(sh_cfg.get("target_pos_enc", False)),
        structured_head_target_pos_enc_frequencies=int(
            sh_cfg.get("target_pos_enc_frequencies", 6)
        ),
        structured_head_target_pos_enc_coord_scale=float(
            sh_cfg.get("target_pos_enc_coord_scale", 1.0)
        ),
        # v9.5: hierarchical mask decoder patch count.
        structured_head_target_num_patches=int(
            sh_cfg.get("target_num_patches", 16)
        ),
        # v9.2: motion-aware trunk + random masking. Merge model defaults
        # with training-yaml overrides (same pattern as structured_head).
        motion_aware_trunk=bool(motion_cfg_merged.get("enabled", False)),
        motion_input_dim=int(motion_cfg_merged.get("joint_input_dim", 66)),
    )
    # v9.5: allow training-yaml to override object_encoder hyperparameters.
    # Same pattern as structured_head — model file holds defaults
    # (`obj_cfg.pointnet.*`), training yaml's `cfg.model.object_encoder`
    # block overrides per-key. Used by v9.5 to enable smaller SA2
    # radius + more output tokens without forking the encoder yaml.
    obj_overrides = cfg.model.get("object_encoder", {})
    object_encoder = ObjectEncoder(
        num_input_points=int(obj_overrides.get(
            "num_input_points", obj_cfg.pointnet.num_input_points,
        )),
        num_output_tokens=int(obj_overrides.get(
            "num_output_tokens", obj_cfg.pointnet.num_output_tokens,
        )),
        feature_dim=int(obj_overrides.get(
            "feature_dim", obj_cfg.pointnet.feature_dim,
        )),
        # v9.5: SA stage hyperparameters. Defaults preserve v9.1 behaviour.
        sa1_num_points=int(obj_overrides.get("sa1_num_points", 512)),
        sa1_radius=float(obj_overrides.get("sa1_radius", 0.15)),
        sa1_num_samples=int(obj_overrides.get("sa1_num_samples", 32)),
        sa2_radius=float(obj_overrides.get("sa2_radius", 0.30)),
        sa2_num_samples=int(obj_overrides.get("sa2_num_samples", 64)),
    )
    accelerator.print(
        f"[object encoder cfg] num_output_tokens={object_encoder.num_output_tokens} "
        f"sa2_radius={object_encoder.sa2_radius:.2f} "
        f"sa2_num_samples={object_encoder.sa2_num_samples} "
        f"(sa1: r={object_encoder.sa1_radius:.2f}, "
        f"num_samples={object_encoder.sa1_num_samples})"
    )

    # SyncBatchNorm on the object encoder under multi-GPU DDP. The
    # PointNet++ SA layers use BatchNorm1d; per-rank running stats
    # would otherwise diverge across A6000 cards. Only valid when
    # there is actually more than one process — the in-place conversion
    # inserts collective comms that fail on single-GPU runs.
    if accelerator.num_processes > 1:
        object_encoder = nn.SyncBatchNorm.convert_sync_batchnorm(object_encoder)

    # CLIP text encoder (frozen). Kept OUT of accelerator.prepare() so
    # it doesn't get wrapped by DDP — it has no trainable parameters.
    # HF Accelerate's recommended pattern for frozen sub-modules.
    clip_model = load_clip_text_encoder(
        device=device,
        model_name=cfg.model.get("text_encoder", "ViT-B/32"),
    )

    # Build the train dataset early — Logit Adjustment class priors
    # are computed from it before the criterion is built. (The
    # dataloader is constructed later, after the criterion + optimizer
    # + scheduler so all four can share one accelerator.prepare call.)
    dataset = _build_dataset(cfg)
    train_split_info = _resolve_split(cfg, split_override=None)
    accelerator.print(
        f"Train dataset: {len(dataset)} clips across {len(cfg.data.datasets)} roots "
        f"(split={train_split_info['label']})",
    )

    # Logit Adjustment class priors (Menon ICLR'21) — needed before
    # building the criterion. Scan the training-only dataset once to
    # tally per-class frequencies for phase + support, then pass
    # ``log π_y`` as a buffer into the loss. Cheap (~30 s for
    # 6400 clips × 196 frames). Only computed when enabled.
    logit_adjust_phase = None
    logit_adjust_support = None
    contact_pos_weight: torch.Tensor | None = None
    # v9: when either Logit Adjustment OR contact pos_weight is on,
    # we need to scan the training set once to compute per-class
    # frequencies. Do it once for both signals.
    needs_priors = (
        cfg.loss.get("use_logit_adjustment", False)
        or cfg.loss.get("use_contact_pos_weight", False)
    )
    if needs_priors:
        accelerator.print("Computing class priors (logit adjust + contact pos_weight)...")
        phase_freq, support_freq, contact_part_freq = compute_class_priors(
            dataset,
            num_phases=int(output_cfg.num_phases),
            num_support=int(output_cfg.num_support_states),
            num_body_parts=int(output_cfg.num_body_parts),
        )
        accelerator.print(f"  phase freq:        {phase_freq.round(4).tolist()}")
        accelerator.print(f"  support freq:      {support_freq.round(4).tolist()}")
        accelerator.print(f"  contact part rate: {contact_part_freq.round(4).tolist()}")
        # log(p + eps) so empty bins don't blow up to -inf
        eps = 1e-12
        if cfg.loss.get("use_logit_adjustment", False):
            logit_adjust_phase = torch.log(torch.from_numpy(phase_freq) + eps)
            logit_adjust_support = torch.log(torch.from_numpy(support_freq) + eps)
        if cfg.loss.get("use_contact_pos_weight", False):
            cap = float(cfg.loss.get("contact_pos_weight_cap", 15.0))
            # pos_weight = (1 - π) / π for each body part. Capped to
            # avoid blow-up on near-zero priors (e.g., foot if data
            # extraction failed).
            part_freq_t = torch.from_numpy(contact_part_freq).float().clamp(min=eps)
            pw = ((1.0 - part_freq_t) / part_freq_t).clamp(max=cap)
            contact_pos_weight = pw
            accelerator.print(f"  contact pos_weight: {pw.round(decimals=3).tolist()}  (cap={cap})")

    # Loss and priors. With Kendall multi-task weights enabled, the
    # static contact/target/phase/support weights are ignored and the
    # optimiser learns per-task log-variances (one extra scalar param
    # per task) that auto-balance loss-scale differences.
    criterion = PredictorLoss(
        contact_weight=cfg.loss.contact_weight,
        target_weight=cfg.loss.target_weight,
        phase_weight=cfg.loss.phase_weight,
        support_weight=cfg.loss.support_weight,
        label_smoothing=cfg.loss.get("label_smoothing", 0.0),
        focal_gamma=cfg.loss.get("focal_gamma", 0.0),
        use_kendall_weights=cfg.loss.get("use_kendall_weights", False),
        # v7-fix (2026-05-04): "all" instead of "contact" recovers target
        # supervision on every valid (frame, part) cell, since
        # closest-surface-point xyz is well-defined irrespective of
        # contact state. Default "contact" preserves v6 behaviour.
        target_gate_kind=cfg.loss.get("target_gate_kind", "contact"),
        logit_adjust_phase=logit_adjust_phase,
        logit_adjust_support=logit_adjust_support,
        logit_adjust_tau=float(cfg.loss.get("logit_adjust_tau", 1.0)),
        # v8 (2026-05-05) flags. Defaults preserve v7-fix behaviour.
        target_loss_kind=cfg.loss.get("target_loss_kind", "smooth_l1"),
        target_kernel_sigma=float(cfg.loss.get("target_kernel_sigma", 0.08)),
        consistency_weight=float(cfg.loss.get("consistency_weight", 0.0)),
        # v8.1 focal+dice hyperparameters (only used when
        # target_loss_kind="focal_dice"). target_tau_per_part order
        # matches PIANO body-part indexing: left/right hand, left/right
        # foot, pelvis. Default (5cm, 5cm, 3cm, 3cm, 12cm) matches
        # v12_strict tight contact thresholds.
        target_focal_alpha=float(cfg.loss.get("target_focal_alpha", 0.25)),
        target_focal_gamma=float(cfg.loss.get("target_focal_gamma", 2.0)),
        target_tau_per_part=tuple(cfg.loss.get(
            "target_tau_per_part", (0.05, 0.05, 0.03, 0.03, 0.12),
        )),
        target_topk_min_positives=int(cfg.loss.get(
            "target_topk_min_positives", 0,
        )),
        # v9 contact pos_weight (None when use_contact_pos_weight=false).
        contact_pos_weight=contact_pos_weight,
        # v9.2 ASL flags.
        contact_loss_kind=str(cfg.loss.get("contact_loss_kind", "bce")),
        contact_asl_gamma_pos=float(cfg.loss.get("contact_asl_gamma_pos", 0.0)),
        contact_asl_gamma_neg=float(cfg.loss.get("contact_asl_gamma_neg", 4.0)),
        contact_asl_prob_shift=float(cfg.loss.get("contact_asl_prob_shift", 0.05)),
        # v9.4: aux xyz L2 loss alongside focal+dice.
        target_aux_xyz_weight=float(cfg.loss.get("target_aux_xyz_weight", 0.0)),
        # v9.5: hierarchical patch CE loss (only fires when predictor
        # emits contact_target_patch_logits — see HierarchicalMaskDecoder).
        target_patch_weight=float(cfg.loss.get("target_patch_weight", 0.0)),
    )
    criterion = criterion.to(device)
    priors = PhysicalPriors(
        reachability_weight=cfg.priors.reachability_weight,
        contact_persistence_weight=cfg.priors.contact_persistence_weight,
        support_smoothness_weight=cfg.priors.support_smoothness_weight,
        phase_monotonicity_weight=cfg.priors.phase_monotonicity_weight,
    )

    # Train dataloader — dataset already built above (so logit-adjust
    # priors could be computed first).
    dataloader = DataLoader(
        dataset, batch_size=cfg.training.batch_size,
        shuffle=True, collate_fn=collate_hoi, num_workers=4,
        pin_memory=True, drop_last=True,
    )

    # Val dataloader — same builder with split=val, augmentation
    # disabled. Used by the in-training keep-best-val loop in
    # run_training_loop. Skipped when val_every_epochs <= 0.
    # (As of 2026-04-26 / v6+ we drop the test bucket from any
    # development-time signal — the predictor's only metric that
    # matters for the paper is the downstream generation quality
    # in Stage C, so a held-out test set on the predictor is unused.
    # The 85/15 train/val subject_split + this "val" bucket is the
    # entire eval surface during Stage A development.)
    val_dataloader = None
    val_every_epochs = int(cfg.training.get("val_every_epochs", 0))
    if val_every_epochs > 0:
        val_dataset = _build_dataset(
            cfg, split_override="val", enable_augment=False,
        )
        val_split_info = _resolve_split(cfg, split_override="val")
        accelerator.print(
            f"Val dataset:   {len(val_dataset)} clips "
            f"(split={val_split_info['label']}, augmentation disabled)",
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=cfg.training.batch_size,
            shuffle=False, collate_fn=collate_hoi, num_workers=4,
            pin_memory=True, drop_last=False,
        )

    # Optimizer: AdamW with ViT/T5-style weight-decay groups (no decay
    # on biases, LayerNorm / BatchNorm weights, positional embeddings).
    # When Kendall multi-task weights are active, criterion holds 4
    # learnable scalars — include it in the optimised modules so they
    # actually update (otherwise they sit at 0 forever, equivalent to
    # all-ones static weights). v4 fix: route the Kendall log-var
    # scalars to a separate param group at ~100× higher lr; v3 found
    # that with the main lr=1e-4 the log-vars only crawled 0→-0.25
    # over 6300 steps vs the predicted equilibrium of ~-3.3, so the
    # auto-balancing was functionally inactive.
    optim_modules = [predictor, object_encoder]
    kendall_lr = None
    if criterion.use_kendall_weights:
        optim_modules.append(criterion)
        kendall_lr = float(cfg.training.optimizer.get("kendall_lr", 1e-2))
    optimizer = build_optimizer_with_decay_groups(
        modules=optim_modules,
        lr=cfg.training.optimizer.lr,
        weight_decay=cfg.training.optimizer.weight_decay,
        betas=tuple(cfg.training.optimizer.betas),
        kendall_lr=kendall_lr,
    )
    # Scheduler total_steps measured in optimizer steps — Accelerate's
    # prepared scheduler handles the accumulation skip, so we feed it
    # (len(dataloader) / accum) × epochs.
    accum = cfg.training.gradient_accumulation_steps
    steps_per_epoch = max(1, len(dataloader) // accum)
    total_steps = steps_per_epoch * cfg.training.num_epochs
    scheduler = build_scheduler(
        optimizer, cfg.training.scheduler.warmup_steps, total_steps,
    )

    # Prepare trainables with accelerator. When Kendall is on, the
    # criterion holds 4 learnable scalars; under DDP these need grad
    # sync across ranks (otherwise each rank drifts to a different
    # value and the loss formula on rank 0 ≠ rank 1, breaking gradient
    # consistency on the predictor itself). prepare() wraps the
    # criterion in DDP so its parameter grads are all-reduced.
    if criterion.use_kendall_weights:
        predictor, object_encoder, criterion, optimizer, dataloader, scheduler = accelerator.prepare(
            predictor, object_encoder, criterion, optimizer, dataloader, scheduler,
        )
    else:
        predictor, object_encoder, optimizer, dataloader, scheduler = accelerator.prepare(
            predictor, object_encoder, optimizer, dataloader, scheduler,
        )
    if val_dataloader is not None:
        val_dataloader = accelerator.prepare(val_dataloader)

    # v8 teacher-forcing schedule for the StructuredHead. Pass through
    # only when structured_head is on AND a schedule block is provided.
    tf_schedule = None
    epochs_per_step = None
    if bool(sh_cfg.get("enabled", False)):
        tf_block = cfg.training.get("teacher_forcing", None)
        if tf_block is not None:
            tf_schedule = OmegaConf.to_container(tf_block, resolve=True) \
                if hasattr(tf_block, "_content") else dict(tf_block)
            steps_per_epoch = max(1, len(dataloader) //
                                  max(1, int(cfg.training.get("gradient_accumulation_steps", 1))))
            epochs_per_step = 1.0 / float(steps_per_epoch)

    step_fn = build_predictor_step_fn(
        predictor, object_encoder, clip_model, criterion, priors, device,
        prior_warmup_steps=cfg.priors.get("prior_warmup_steps", 0),
        teacher_forcing_schedule=tf_schedule,
        epochs_per_step=epochs_per_step,
    )

    # Wandb (optional)
    wandb_run = None
    if accelerator.is_main_process:
        try:
            import wandb
            wandb_run = wandb.init(project=cfg.logging.project, name=cfg.logging.run_name)
        except ImportError:
            pass

    run_training_loop(
        accelerator=accelerator,
        model=predictor,
        dataloader=dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        step_fn=step_fn,
        num_epochs=cfg.training.num_epochs,
        output_dir=cfg.output_dir,
        log_every=cfg.logging.log_every_n_steps,
        save_every_epochs=cfg.logging.save_every_n_epochs,
        max_grad_norm=cfg.training.max_grad_norm,
        wandb_run=wandb_run,
        # Persist the object encoder's weights into every checkpoint
        # — predictor alone can't run inference; the object cross-
        # attention KV tokens come from this encoder and its weights
        # are trained from scratch (no pretrained fallback exists).
        extra_modules={"object_encoder": object_encoder},
        # Keep-best-val: re-run the step_fn on val_dataloader every N
        # epochs, save a best_val.pt when val ``val_best_key`` improves.
        # Does not interrupt training — best checkpoint is kept in
        # parallel to the final one. Default ``val_best_key`` is
        # "loss" (the Kendall-combined total), but when Kendall is on
        # this is decoupled from supervision quality (see losses.py
        # comment) — set ``training.val_best_key: "loss_unweighted"``
        # in the config to track raw supervision instead.
        val_dataloader=val_dataloader,
        val_every_epochs=val_every_epochs,
        val_best_key=cfg.training.get("val_best_key", "loss"),
    )


def main() -> None:
    """CLI entry point for ``piano-train-predictor`` (Stage A)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/training/predictor.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
