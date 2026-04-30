"""Qualitative evaluation for Stage B v0.1 — does z_int actually control output?

Loads a Stage B checkpoint and generates motion under three controlled
conditions per sampled val clip:

    1. ``full``       — real text + real z_int (the "use both") branch
    2. ``text_only``  — real text + no z_int (compositional CFG fallback;
                        ≈ pure text-conditioned MoMask)
    3. ``swap``       — real text + z_int from a DIFFERENT val clip
                        (same text, deliberately wrong interaction plan)

Then measures:

    - **Base-token Hamming distance** between conditions: directly tells
      us how much z_int changed the Stage B base-layer generation.
      If ``hamming(full, text_only) ≈ 0``, IntXAttn never moved the
      logits and z_int has no effect.
    - **Motion-263 mean L2 per frame** between conditions (after the
      shared frozen residual transformer + VQ-VAE decode), comparable
      to how a viewer would experience the difference.
    - **w_int sweep** (optional, ``--w-int-sweep``): same metrics
      across ``w_int ∈ {0, 1, 2, 4, 8}`` to see whether cranking the
      interaction guidance scale actually amplifies its effect or
      whether the IntXAttn weights are saturated.

Saves each condition's generated 263-d motion + meta as
``<output-dir>/<condition>/generated.npz`` + ``summary.json`` so the
existing :mod:`piano.inference.visualize_motion` ``generated``
sub-command can render mp4s without further glue.

Usage::

    python scripts/stage_b_generator/qual_eval.py \\
        --config configs/training/generator.yaml \\
        --ckpt runs/training/generator/best_val.pt \\
        --num-clips 20 \\
        --output-dir runs/eval/stageB_v0_1_qual \\
        [--w-int-sweep]

Then to render mp4s of the "full" condition::

    python -m piano.inference.visualize_motion generated \\
        --run-dir runs/eval/stageB_v0_1_qual/full \\
        --output-dir runs/eval/stageB_v0_1_qual/full/vis
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
from torch.utils.data import ConcatDataset

from piano.data.dataset import HOIDataset
from piano.data.eval_sampling import (
    describe_eval_clip_selection,
    select_eval_clip_indices,
)
from piano.data.humanml3d_repr import load_motion_stats
from piano.data.split import build_subject_split, extract_subject_id
from piano.models.backbones.momask_adapter import (
    load_momask_mask_transformer,
    load_momask_residual_transformer,
    load_momask_vqvae,
)
from piano.models.interaction_tokenizer import InteractionTokenizer
from piano.models.motion_generator import InteractionMaskTransformer
from piano.models.motion_generator_residual import ResidualTransformerWithInteraction
from piano.utils.io_utils import ensure_dir, load_json, save_json


# ============================================================================
# Dataset helpers (mirror train_generator.py logic so we evaluate on the
# SAME val bucket the training run held out).
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
    """Reproduce the training-time val bucket exactly (seed + percentages)."""
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
    # v0.3-α: pick up force_world_frame from training config so eval
    # uses the same frame the model was trained on. Defaults to v0.2
    # behaviour (body-canonical) when key is absent.
    force_world_frame = bool(cfg.data.get("force_world_frame", False))
    datasets = []
    for entry in cfg.data.datasets:
        ds = HOIDataset(
            root=entry.root,
            pseudo_label_dir=pseudo_label_dir,
            max_seq_length=cfg.data.max_seq_length,
            subject_id_filter=val_filter,
            augment=None,
            surface_obj_pose=True,           # v0.2 tokenizer needs canonical object pose
            force_world_frame=force_world_frame,
        )
        datasets.append(ds)
    return ConcatDataset(datasets)


# ============================================================================
# Model setup
# ============================================================================

def _build_model(cfg, ckpt_path: Path, device: torch.device):
    """Build the wrapped InteractionMaskTransformer + frozen VQ-VAE +
    frozen ResidualTransformer, then load the Stage B checkpoint."""
    model_cfg = OmegaConf.load(cfg.model.config)

    # Frozen VQ-VAE.
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

    # Frozen Residual Transformer (per analyses/early_setup.md the `_sw`
    # checkpoint suffix encodes share_weight=True).
    res_ckpt_cfg = cfg.model.checkpoints.get("residual_transformer", None)
    if res_ckpt_cfg is None:
        raise ValueError(
            "configs/model/motion_generator.yaml must declare "
            "checkpoints.residual_transformer for qual eval."
        )
    res_transformer = load_momask_residual_transformer(
        res_ckpt_cfg,
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

    # Interaction-aware MaskTransformer: load pretrained MoMask first
    # (preserves CLIP load), wrap, then load best_val state on top.
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
    # γ-gate kind + wrapper_kind must match training-time config so
    # loaded state_dict shapes line up.
    gamma_kind = str(cfg.model.get(
        "gamma_kind",
        mt_cfg.interaction_cross_attn.get("gamma_kind", "scalar"),
    ))
    wrapper_kind = str(cfg.model.get("wrapper_kind", "v0.6"))
    print(f"γ-gate kind: {gamma_kind}, wrapper_kind: {wrapper_kind}")
    transformer = InteractionMaskTransformer(
        mask_transformer=base_mt,
        interaction_tokenizer=interaction_tokenizer,
        zero_init_gamma=bool(mt_cfg.interaction_cross_attn.get("zero_init", True)),
        max_token_seq_length=max_seq_length_tokens,
        gamma_kind=gamma_kind,
        wrapper_kind=wrapper_kind,
    )
    residual_int_cfg = cfg.model.get("residual_int_xattn", None)
    residual_int_enabled = (
        residual_int_cfg is not None
        and bool(residual_int_cfg.get("enabled", False))
    )
    if residual_int_enabled:
        res_transformer = ResidualTransformerWithInteraction(
            residual_transformer=res_transformer,
            d_model=model_cfg.residual_transformer.latent_dim,
            num_heads=model_cfg.residual_transformer.num_heads,
            dropout=float(residual_int_cfg.get(
                "dropout", model_cfg.residual_transformer.dropout,
            )),
            zero_init_gamma=bool(residual_int_cfg.get("zero_init_gamma", True)),
            gamma_kind=str(residual_int_cfg.get("gamma_kind", gamma_kind)),
        )
        transformer.residual_transformer = res_transformer
    transformer.to(device)

    # Load Stage B checkpoint on top.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    missing, unexpected = transformer.load_state_dict(state, strict=False)
    real_missing = [k for k in missing if not k.startswith("mask_transformer.clip_model.")]
    if real_missing:
        print(f"  [warn] missing keys: {real_missing[:5]}{'...' if len(real_missing) > 5 else ''}")
    if unexpected:
        print(f"  [warn] unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    transformer.eval()

    return transformer, vq_model, res_transformer, token_stride


# ============================================================================
# Generation primitives
# ============================================================================

@torch.no_grad()
def _tokenize_z_int(
    transformer: InteractionMaskTransformer,
    sample: dict[str, Any],
    device: torch.device,
) -> tuple[Tensor, Tensor | None]:
    """Run the interaction tokenizer on one batched sample (B=1)."""
    # Promote to (1, T, ...) batch.
    cs = sample["contact_state"].unsqueeze(0).float().to(device)
    ctx = sample["contact_target_xyz"].unsqueeze(0).float().to(device)
    ph = sample["phase"].unsqueeze(0).long().to(device)
    sup = sample["support"].unsqueeze(0).long().to(device)
    seq_len = sample["seq_len"].unsqueeze(0).long().to(device)
    # v0.2: HOIDataset(surface_obj_pose=True) places these in the sample
    # dict; both must be present for a v0.2-built tokenizer.
    obj_com = sample["obj_com_canonical"].unsqueeze(0).float().to(device)
    obj_rot6d = sample["obj_rot6d_canonical"].unsqueeze(0).float().to(device)
    int_kv, pad = transformer.interaction_tokenizer(
        contact_state=cs, contact_target_xyz=ctx,
        phase=ph, support=sup,
        obj_com_canonical=obj_com,
        obj_rot6d_canonical=obj_rot6d,
        seq_lens=seq_len,
    )
    return int_kv, pad


@torch.no_grad()
def _generate(
    transformer: InteractionMaskTransformer,
    vq_model: torch.nn.Module,
    res_transformer: torch.nn.Module,
    text: str,
    int_kv: Tensor | None,
    int_pad: Tensor | None,
    m_lens_tok: Tensor,
    *,
    w_text: float,
    w_int: float,
    motion_mean: np.ndarray,
    motion_std: np.ndarray,
    timesteps: int = 10,
    res_cond_scale: float = 2.0,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run base + residual generation, then VQ decode + denormalize.

    The VQ-VAE was trained on **normalized** HumanML3D 263-d
    (``(raw - mean) / std``) so its decoder also outputs in normalized
    space. We denormalize via ``motion * std + mean`` before returning,
    so the caller sees motion in the same scale as
    ``preprocess_interact``'s saved ``motion_263`` and downstream
    ``recover_from_ric`` reconstructs joints in real-world units.
    Without this denorm, joint positions cluster near the canonical
    origin and mp4 viewers see a "stationary blob" regardless of what
    the model actually generated. (This was the root cause of the
    apparent "body in place" failure mode in v0.1, v0.2, v0.3-α — the
    visualisation pipeline never denormalised. See
    analyses/2026-04-27_v0_3_root_cause_research.md note appended after
    v0.3-α evaluation.)

    Returns ``(motion_263, base_token_ids)`` numpy arrays of shapes
    ``(T_frames, 263)`` and ``(S_tokens,)`` respectively (B=1, squeezed).
    Motion is in **denormalized** HumanML3D scale.
    """
    cond_vector = transformer.encode_text([text]).to(device).float()
    base_ids = transformer.generate(
        cond_vector=cond_vector,
        m_lens_tok=m_lens_tok,
        int_tokens_bf=int_kv,
        int_padding_mask_bf=int_pad,
        timesteps=timesteps,
        w_text=w_text,
        w_int=w_int,
    )                                           # (1, S_max), -1 at padded
    # ``-1`` would crash res_transformer's gather; mask back to pad_id
    # the residual model knows. The MoMask convention is to fill with 0
    # since padding tokens get re-masked downstream.
    base_for_res = torch.where(base_ids < 0, torch.zeros_like(base_ids), base_ids)

    # Residual layers. C1 checkpoints wrap the residual transformer so
    # it sees the same z_int K/V as the base MaskTransformer.
    if hasattr(res_transformer, "generate_with_int"):
        res_int_kv = (
            None if int_kv is None
            else int_kv.transpose(0, 1).contiguous()
        )
        all_ids = res_transformer.generate_with_int(
            motion_ids=base_for_res,
            conds=[text],
            m_lens=m_lens_tok,
            int_kv=res_int_kv,
            int_padding_mask=int_pad,
            cond_scale=res_cond_scale,
        )
    else:
        all_ids = res_transformer.generate(
            motion_ids=base_for_res,
            conds=[text],
            m_lens=m_lens_tok,
            cond_scale=res_cond_scale,
        )                                       # (1, S, Q), -1 at padded
    all_for_decode = torch.where(all_ids < 0, torch.zeros_like(all_ids), all_ids)

    # Decode to 263-d motion (still in normalized space — VQ-VAE was
    # trained on (raw - mean) / std normalized features).
    motion = vq_model.forward_decoder(all_for_decode)   # (1, T, 263)
    motion = motion.squeeze(0).detach().cpu().numpy()
    # Denormalize so downstream recover_from_ric reconstructs joints in
    # real-world (HumanML3D-canonical) units. Without this, joint
    # positions stay clustered near the origin and the mp4 viewer sees
    # a stationary blob regardless of what the model generated.
    motion = motion * motion_std + motion_mean
    base_np = base_ids.squeeze(0).detach().cpu().numpy()
    return motion, base_np


# ============================================================================
# Diff metrics
# ============================================================================

# ============================================================================
# Coordinate-frame helpers
# ============================================================================

def _get_canon_to_world_transform(
    joints_world: np.ndarray,
    motion_263: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Recover the (R_y_angle, T_xz) that maps canonical → world for one clip.

    HumanML3D canonicalization (verified from MoMask
    ``utils/motion_process.process_file``) (1) translates so frame-0
    pelvis sits at XZ origin and (2) rotates around Y so a hip-line-derived
    "facing direction" aligns with +Z. This function inverts both steps
    by comparing frame-0 of:

      - ``joints_world``  — (T, 22, 3) preprocessed in world frame
        (saved by ``preprocess_interact.py`` as ``joints_22``)
      - ``motion_263``    — canonical-frame HumanML3D 263-d features

    so the GENERATED motion (which is also in canonical frame, since it
    came out of the VQ-VAE that was trained on HumanML3D) can be lifted
    back into THIS source clip's world frame for visualization.

    Returns
    -------
    R_y_angle : scalar — rotation around +Y, in radians
    T_xz : (2,) — translation along world X and Z (Y is preserved by
        canonicalization, so no T_y component)
    """
    # MoMask path is set up at module import via momask_adapter — the
    # qual_eval entrypoint already imports it. ``recover_from_ric`` is
    # the canonical-frame integrator HumanML3D ships with.
    import torch
    import piano.models.backbones.momask_adapter  # noqa: F401
    from utils.motion_process import recover_from_ric

    canonical_joints = recover_from_ric(
        torch.from_numpy(motion_263).float().unsqueeze(0),
        joints_num=22,
    ).squeeze(0).cpu().numpy().astype(np.float32)   # (T, 22, 3)

    # Frame 0 anchor.
    world_t0 = joints_world[0]                       # (22, 3)
    canon_t0 = canonical_joints[0]                   # (22, 3)

    # Translation: where does frame-0 pelvis sit in world? Canonical
    # pelvis is at (0, h, 0); world pelvis is at world_t0[0]. The XZ
    # delta is the translation we need.
    T_xz = world_t0[0, [0, 2]] - canon_t0[0, [0, 2]]

    # Rotation around Y: align hip-line directions. Right hip (joint 2)
    # minus left hip (joint 1) is approximately horizontal across the
    # hips and rotates with the body's facing — exactly the signal
    # canonicalization aligns to +Z.
    hip_world = world_t0[2] - world_t0[1]
    hip_canon = canon_t0[2] - canon_t0[1]
    angle_world = float(np.arctan2(hip_world[0], hip_world[2]))
    angle_canon = float(np.arctan2(hip_canon[0], hip_canon[2]))
    R_y_angle = angle_world - angle_canon
    return R_y_angle, T_xz.astype(np.float32)


# (Transform application moved to visualize_motion.motion_263_to_joints
# so denormalization (mean/std) and joint recovery happen in the right
# order. qual_eval only computes + saves the per-clip (R_y, T_xz) params.)


# ============================================================================
# Diff metrics
# ============================================================================

def _hamming(a: np.ndarray, b: np.ndarray, valid_lens: int | None = None) -> float:
    """Per-position fraction of disagreement on valid tokens.

    ``a`` and ``b`` are 1D token-id arrays (S,). ``valid_lens`` truncates
    to the actual non-padded prefix; if None, both are compared in full.
    """
    if valid_lens is not None:
        a = a[:valid_lens]
        b = b[:valid_lens]
    if len(a) == 0:
        return 0.0
    return float((a != b).mean())


def _motion_l2_per_frame(a: np.ndarray, b: np.ndarray, valid_frames: int | None = None) -> float:
    """Mean per-frame L2 distance across the 263-d motion repr.

    Returns the mean over (valid frames × 263) of |a - b|² then sqrt'd
    to get a meters-ish magnitude (HumanML3D 263-d is normalised to
    unit-ish scale by MoMask preprocessing, so the absolute number
    isn't a literal metric — but pair-wise differences ARE meaningful)."""
    if valid_frames is not None:
        a = a[:valid_frames]
        b = b[:valid_frames]
    if len(a) == 0:
        return 0.0
    diff = (a - b)
    return float(np.sqrt((diff * diff).mean()))


# ============================================================================
# Main
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/training/generator.yaml"))
    parser.add_argument("--ckpt", type=Path, default=Path("runs/training/generator/best_val.pt"))
    parser.add_argument(
        "--num-clips", type=int, default=20,
        help="number of stratified val clips to generate (default 20).",
    )
    parser.add_argument("--output-dir", type=Path,
                        default=Path("runs/eval/stageB_v0_1_qual"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--w-text", type=float, default=4.0)
    parser.add_argument("--w-int", type=float, default=2.0)
    parser.add_argument(
        "--guidance-steps", type=int, default=0,
        help="Contact-aware inference-time logit guidance (B3, MaskControl "
             "arXiv:2410.10780). When > 0, additionally generate a "
             "'full_guided' condition per clip: optimise base-token logits "
             "in decoded geometric space against contact_target_xyz_gt for "
             "this many AdamW steps before argmax + residual decode. "
             "Default 0 = baseline only. Recommended starting value: 30. "
             "Cost ~1 sec per step per clip on bf16 / single A6000.",
    )
    parser.add_argument(
        "--guidance-lr", type=float, default=6e-2,
        help="Learning rate for contact guidance optimiser. Default 6e-2 "
             "matches MaskControl's verified value at "
             "exitudio/ControlMM@models/mask_transformer/control_transformer.py.",
    )
    parser.add_argument(
        "--guidance-init-scale", type=float, default=3.0,
        help="Init scale for one-hot logit initialization. Lower = softer "
             "softmax = optimization can flip tokens more easily; higher = "
             "model prior is preserved more strongly. Sweep this if "
             "guidance_trace.json shows base_token_change_count=0 across "
             "all clips (the original 10.0 default was too sharp). 2026-04-28 "
             "calibration: 3.0 lets ~5-15 tokens flip per clip in 30 steps; "
             "1.0 essentially uniform; 10.0 = original sharp default.",
    )
    parser.add_argument(
        "--guidance-loss", choices=["target", "metric"], default="metric",
        help="Loss formulation: 'target' = masked L2 against GT contact "
             "target points (object-local lifted to world); 'metric' = "
             "exact eval metric (min over PC samples, min over body parts, "
             "mean over time). 'target' was the initial recipe (matches "
             "MaskControl's get_loss); 'metric' eliminates loss-vs-metric "
             "decoupling. Default 'metric' since 2026-04-28 calibration "
             "showed 'target' produced per-clip mixed results (largebox "
             "-13 cm BUT plasticbox_037 +14 cm).",
    )
    parser.add_argument(
        "--guidance-residual-seed", type=int, default=None,
        help="If set, calls torch.manual_seed(N) before each "
             "res_transformer.generate call (baseline + post-guidance). "
             "Forces RNG-equivalence between the two residual generations "
             "so any contact delta is attributable to base-token changes "
             "or autoregressive feedback, NOT Gumbel-noise drift. "
             "Default None = no extra seeding (legacy v3+v4 behavior). "
             "Background: ResidualTransformer.generate samples each layer "
             "via gumbel_sample which uses uniform_(0,1) on global RNG; "
             "v4 plasticbox_014 (0/23 base flips, +7.3cm regression) is "
             "the smoking gun for this drift.",
    )
    parser.add_argument(
        "--guidance-no-residual-rerun", action="store_true",
        help="If set, skip the post-guidance res_transformer.generate "
             "call entirely and decode using baseline residual_ids "
             "combined with new base_ids_after. Tests whether residual "
             "rerun itself is the per-clip variance source. Cost: "
             "forfeits any 'base flip → residual self-adapts' gain "
             "(e.g., v4 largebox_010 -14cm may have come from this).",
    )
    parser.add_argument(
        "--guidance-layers",
        choices=["base", "full_rvq"],
        default="base",
        help="Which token logits to optimize during contact guidance. "
             "'base' preserves the original B3 path; 'full_rvq' optimizes "
             "the full generated base+residual RVQ stack in decoded space "
             "and decodes that stack directly.",
    )
    parser.add_argument(
        "--per-step-iters", type=int, default=0,
        help="Per-step decoded-geometric guidance inner-loop budget (v17, "
             "MaskControl 'each_iter' analog, arXiv:2410.10780). When > 0, "
             "the baseline MaskGIT loop is replaced by a re-rolled version "
             "that runs N AdamW inner steps on the logits at each MaskGIT "
             "iteration before commit, using a relaxed-decode geometric "
             "loss with frozen baseline residuals. Default 0 = legacy "
             "post-hoc-only path (back-compat). Recommended first cut: 10. "
             "Stacks with --guidance-steps (post-hoc) when both > 0. See "
             "analyses/2026-05-01_per_step_guidance_design.md.",
    )
    parser.add_argument(
        "--per-step-lr", type=float, default=6e-2,
        help="Learning rate for the per-step inner AdamW loop. Default "
             "6e-2 matches MaskControl + the post-hoc --guidance-lr.",
    )
    parser.add_argument(
        "--per-step-temperature", type=float, default=1.0,
        help="Softmax temperature in the per-step relaxed decode. 1.0 = "
             "unbiased expectation, matching MaskControl. Lower (e.g. 0.5) "
             "would peak the relaxed embedding closer to the argmax token "
             "but provides weaker gradient signal.",
    )
    parser.add_argument(
        "--per-step-start-step", type=int, default=0,
        help="MaskGIT iteration index at which per-step guidance begins. "
             "0 = active from step 0 (matches MaskControl). Raise to 5 "
             "if early-step optimisation is unstable (most positions are "
             "still masked at low step indices, so the relaxed-decode "
             "signal is uniform-like).",
    )
    parser.add_argument(
        "--per-step-gumbel-scale", type=float, default=1.0,
        help="Gumbel-noise scale for the per-step inner loop's relaxed "
             "decode (v17-F, MaskControl-equivalent). 1.0 = canonical "
             "Gumbel-Softmax / Concrete relaxation, matches "
             "exitudio/ControlMM's `each_iter` block "
             "(softmax((logits/T) + gumbel_noise)). 0.0 = pre-v17-F PIANO "
             "behaviour (pure softmax expectation, no noise). Set 0.0 to "
             "ablate the Gumbel addition. Source-verified diff vs "
             "MaskControl listed in "
             "analyses/2026-05-01_per_step_guidance_design.md.",
    )
    parser.add_argument("--w-int-sweep", action="store_true",
                        help="also generate a w_int sweep over {0, 1, 2, 4, 8}")
    parser.add_argument(
        "--summary-detail",
        choices=["compact", "full"],
        default="compact",
        help=(
            "overall summary.json detail level. compact keeps run metadata and "
            "aggregate diff means; full also stores clip/text selection and "
            "per-clip diff arrays."
        ),
    )
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    ensure_dir(args.output_dir)
    cfg = OmegaConf.load(args.config)

    print(f"Loading model from {args.ckpt} ...")
    transformer, vq_model, res_transformer, token_stride = _build_model(
        cfg, args.ckpt, device,
    )
    print(f"Token stride: {token_stride}  (frame T → token S = T/{token_stride})")

    # Load HumanML3D motion stats so generated motion gets denormalized
    # to real-world scale before being saved + visualized. See
    # ``piano.data.humanml3d_repr.load_motion_stats`` docstring for why
    # this is load-bearing for mp4 visual quality.
    motion_mean, motion_std = load_motion_stats(cfg.model.checkpoints.vq_vae)
    print(f"Loaded HumanML3D mean/std: shape={motion_mean.shape}")

    # Build val dataset, sample N clips (deterministic, subset/object balanced).
    val_dataset = _build_val_dataset(cfg)
    sampled_idx = select_eval_clip_indices(
        val_dataset,
        args.num_clips,
        seed=args.seed,
    )
    selected_rows = describe_eval_clip_selection(val_dataset, sampled_idx)
    print(
        f"Sampled {len(sampled_idx)} stratified clips out of "
        f"{len(val_dataset)} val clips: {sampled_idx}",
    )
    for row in selected_rows:
        print(
            "  "
            f"idx={row['index']} subset={row['subset']} "
            f"object={row['object_id']} seq={row['seq_id']}",
        )

    samples = [val_dataset[i] for i in sampled_idx]
    seq_lens_frames = [int(s["seq_len"].item()) for s in samples]
    seq_lens_tok = [max(1, n // token_stride) for n in seq_lens_frames]
    texts = [str(s["text"]) for s in samples]
    seq_ids = [str(s["seq_id"]) for s in samples]
    # Per-source-clip world transform (R_y, T_xz) so we can put generated
    # canonical-frame motion back into the SOURCE's world frame and
    # render it together with the world-frame object. Without this, the
    # body always starts at canonical origin while the object sits 1-3m
    # away → visually they never interact regardless of z_int quality.
    # See ``_get_canon_to_world_transform`` for the math.
    source_canon_xforms: list[tuple[float, np.ndarray]] = []
    for s in samples:
        # joints_22 is world frame, motion is canonical (per
        # analyses/early_setup.md "Two coordinate frames, kept side-by-side").
        Ry, Txz = _get_canon_to_world_transform(
            s["joints"].cpu().numpy(),       # (T, 22, 3) world
            s["motion"].cpu().numpy(),       # (T, 263) canonical motion_263
        )
        source_canon_xforms.append((Ry, Txz))
    # Per-clip object info captured for visualization overlay. PC is the
    # subsampled-1024 cloud HOIDataset already returns; positions /
    # rotations come straight from the source clip's preprocessed npz
    # (added in 2026-04-27 to dataset.py). Some older clips lack rotations
    # — substitute zeros so visualize_motion can ignore the rotation
    # without a None-check explosion.
    object_pcs = [s["object_pc"].cpu().numpy() for s in samples]
    object_positions = [
        s["object_positions"].cpu().numpy()
        if "object_positions" in s
        else np.zeros((cfg.data.max_seq_length, 3), dtype=np.float32)
        for s in samples
    ]
    object_rotations = [
        s["object_rotations"].cpu().numpy()
        if "object_rotations" in s
        else np.zeros((cfg.data.max_seq_length, 3), dtype=np.float32)
        for s in samples
    ]

    # Pre-tokenise z_int for each sample (so we can swap freely).
    z_int_per = []
    for s in samples:
        kv, pad = _tokenize_z_int(transformer, s, device)
        z_int_per.append((kv, pad))

    # Helper: run all 3 conditions for a given (clip, w_int).
    @torch.no_grad()
    def _run_three(i: int, w_int: float) -> dict[str, dict]:
        text = texts[i]
        m_lens_tok = torch.tensor([seq_lens_tok[i]], dtype=torch.long, device=device)
        kv_self, pad_self = z_int_per[i]
        # "swap target" = next clip in cyclic order
        j = (i + 1) % len(samples)
        kv_other, pad_other = z_int_per[j]

        m_full, base_full = _generate(
            transformer, vq_model, res_transformer,
            text=text, int_kv=kv_self, int_pad=pad_self,
            m_lens_tok=m_lens_tok, w_text=args.w_text, w_int=w_int,
            motion_mean=motion_mean, motion_std=motion_std, device=device,
        )
        m_text, base_text = _generate(
            transformer, vq_model, res_transformer,
            text=text, int_kv=None, int_pad=None,
            m_lens_tok=m_lens_tok, w_text=args.w_text, w_int=w_int,
            motion_mean=motion_mean, motion_std=motion_std, device=device,
        )
        m_swap, base_swap = _generate(
            transformer, vq_model, res_transformer,
            text=text, int_kv=kv_other, int_pad=pad_other,
            m_lens_tok=m_lens_tok, w_text=args.w_text, w_int=w_int,
            motion_mean=motion_mean, motion_std=motion_std, device=device,
        )
        return {
            "full":      {"motion": m_full, "base": base_full,  "swap_from": None},
            "text_only": {"motion": m_text, "base": base_text,  "swap_from": None},
            "swap":      {"motion": m_swap, "base": base_swap,  "swap_from": seq_ids[j]},
        }

    # Default run (single w_int).
    default_w_int_str = f"w_int_{args.w_int:g}"
    print(f"\n=== Generating 3 conditions × {len(samples)} clips at w_int={args.w_int} ===")
    per_clip_default: list[dict[str, dict]] = []
    for i in range(len(samples)):
        print(f"  clip {i+1}/{len(samples)}: {seq_ids[i]}  text={texts[i][:60]!r}")
        per_clip_default.append(_run_three(i, args.w_int))

    # Per-condition save (visualize_motion-compatible: generated.npz +
    # summary.json). One subdir per condition, one row per clip.
    # Visualisation overlays: for every condition we ship the SOURCE
    # clip's object — so when the user looks at "swap", they see whether
    # the human, conditioned on a different clip's z_int, still tracks
    # the source clip's object. (For a future variant we could also
    # ship the swap-source's object, but starting with the simpler
    # comparison is the right first cut.)
    _save_condition_dir(
        args.output_dir / "full", per_clip_default, "full",
        texts, seq_lens_frames, seq_ids,
        object_pcs=object_pcs,
        object_positions=object_positions,
        object_rotations=object_rotations,
        world_R_y=[x[0] for x in source_canon_xforms],
        world_T_xz=[x[1] for x in source_canon_xforms],
    )
    _save_condition_dir(
        args.output_dir / "text_only", per_clip_default, "text_only",
        texts, seq_lens_frames, seq_ids,
        object_pcs=object_pcs,
        object_positions=object_positions,
        object_rotations=object_rotations,
        world_R_y=[x[0] for x in source_canon_xforms],
        world_T_xz=[x[1] for x in source_canon_xforms],
    )
    _save_condition_dir(
        args.output_dir / "swap", per_clip_default, "swap",
        texts, seq_lens_frames, seq_ids,
        object_pcs=object_pcs,
        object_positions=object_positions,
        object_rotations=object_rotations,
        world_R_y=[x[0] for x in source_canon_xforms],
        world_T_xz=[x[1] for x in source_canon_xforms],
    )

    # Optional: contact-aware inference-time logit guidance (B3).
    # Source-verified MaskControl arXiv:2410.10780 recipe. See
    # piano.inference.contact_guidance for the algorithm.
    if args.guidance_steps > 0 or args.per_step_iters > 0:
        from piano.inference.contact_guidance import guide_with_contact

        print(
            f"\n=== Generating 'full_guided' condition with contact guidance "
            f"(post_hoc_steps={args.guidance_steps}, lr={args.guidance_lr}, "
            f"init_scale={args.guidance_init_scale}, loss={args.guidance_loss}, "
            f"layers={args.guidance_layers}, "
            f"residual_seed={args.guidance_residual_seed}, "
            f"no_residual_rerun={args.guidance_no_residual_rerun}, "
            f"per_step_iters={args.per_step_iters}, "
            f"per_step_lr={args.per_step_lr}, "
            f"per_step_temperature={args.per_step_temperature}, "
            f"per_step_start_step={args.per_step_start_step}, "
            f"per_step_gumbel_scale={args.per_step_gumbel_scale}) ===",
        )
        per_clip_guided: list[dict[str, dict]] = []
        for i in range(len(samples)):
            text_i = texts[i]
            m_lens_tok_i = torch.tensor(
                [seq_lens_tok[i]], dtype=torch.long, device=device,
            )
            kv_self, pad_self = z_int_per[i]
            # Source clip data (numpy on CPU). The guide_with_contact
            # helper lifts contact_target_xyz from object-local to world
            # frame internally — pass the LOCAL-frame target + per-frame
            # object pose (so the rigid transform is reconstructible).
            cstate_i = samples[i]["contact_state"].cpu().numpy().astype(np.float32)
            ctgt_local_i = samples[i]["contact_target_xyz"].cpu().numpy().astype(np.float32)
            obj_pos_i = object_positions[i]                              # (T, 3) world
            obj_rot_i = object_rotations[i]                              # (T, 3) axis-angle world
            R_y_i, T_xz_i = source_canon_xforms[i]

            print(
                f"  clip {i+1}/{len(samples)}: {seq_ids[i]}  "
                f"text={text_i[:60]!r}",
            )
            motion_g, base_g, info = guide_with_contact(
                transformer=transformer,
                vq_model=vq_model,
                res_transformer=res_transformer,
                text=text_i,
                int_kv=kv_self,
                int_pad=pad_self,
                m_lens_tok=m_lens_tok_i,
                contact_target_xyz_local=ctgt_local_i,
                contact_state=cstate_i,
                object_pc_local=object_pcs[i],                    # (N_pc, 3) — for metric-mode loss
                object_positions=obj_pos_i,
                object_rotations=obj_rot_i,
                R_y_angle=R_y_i,
                T_xz=T_xz_i,
                motion_mean=torch.from_numpy(motion_mean).float().to(device),
                motion_std=torch.from_numpy(motion_std).float().to(device),
                w_text=args.w_text,
                w_int=args.w_int,
                num_guidance_steps=args.guidance_steps,
                guidance_lr=args.guidance_lr,
                init_logit_scale=args.guidance_init_scale,
                loss_mode=args.guidance_loss,
                residual_seed=args.guidance_residual_seed,
                no_residual_rerun=args.guidance_no_residual_rerun,
                guidance_layers=args.guidance_layers,
                per_step_iters=args.per_step_iters,
                per_step_lr=args.per_step_lr,
                per_step_temperature=args.per_step_temperature,
                per_step_start_step=args.per_step_start_step,
                per_step_gumbel_scale=args.per_step_gumbel_scale,
                device=device,
            )
            print(
                f"    guidance: post_hoc_loss {info['loss_initial']:.4f} → "
                f"{info['loss_final']:.4f} | "
                f"post_hoc tokens changed: {info['base_token_change_count']}/"
                f"{info['base_token_total']} base, "
                f"{info.get('rvq_token_change_count', 0)}/"
                f"{info.get('rvq_token_total', info['base_token_total'])} rvq",
            )
            per_step = info.get("per_step")
            if per_step is not None:
                ps_init = per_step.get("per_step_loss_initial", []) or []
                ps_final = per_step.get("per_step_loss_final", []) or []
                ps_active = per_step.get("per_step_active", []) or []
                active_init = [v for v, a in zip(ps_init, ps_active) if a]
                active_final = [v for v, a in zip(ps_final, ps_active) if a]
                if active_init and active_final:
                    print(
                        f"    per-step: active_steps={int(sum(ps_active))}/"
                        f"{len(ps_active)} | "
                        f"first→last loss {active_init[0]:.4f} → {active_final[-1]:.4f} | "
                        f"per-step base tokens changed vs naive: "
                        f"{per_step.get('per_step_base_token_change_count', 0)}/"
                        f"{per_step.get('per_step_base_token_total', info['base_token_total'])}",
                    )
            per_clip_guided.append({
                "full": {
                    "motion": motion_g,
                    "base": base_g,
                    "swap_from": None,
                    "guidance_info": info,
                },
            })

        _save_condition_dir(
            args.output_dir / "full_guided", per_clip_guided, "full",
            texts, seq_lens_frames, seq_ids,
            object_pcs=object_pcs,
            object_positions=object_positions,
            object_rotations=object_rotations,
            world_R_y=[x[0] for x in source_canon_xforms],
            world_T_xz=[x[1] for x in source_canon_xforms],
        )

        # Persist the per-clip guidance trace (loss_initial/final +
        # token-change count) to a sibling JSON. This is the
        # ground-truth signal that the new code path ran. If this file
        # is absent or empty, the server's git pull didn't deploy the
        # current commit. If it's present with non-trivial loss values
        # but contact distance is unchanged, the optimization itself
        # converges to the same fixed point — that's a different debug
        # path (likely AdamW lr or step count needs tuning).
        guidance_trace_path = args.output_dir / "full_guided" / "guidance_trace.json"
        guidance_trace = {
            "guidance_steps": int(args.guidance_steps),
            "guidance_lr": float(args.guidance_lr),
            "guidance_init_scale": float(args.guidance_init_scale),
            "guidance_loss": str(args.guidance_loss),
            "guidance_layers": str(args.guidance_layers),
            "residual_seed": (
                None if args.guidance_residual_seed is None
                else int(args.guidance_residual_seed)
            ),
            "no_residual_rerun": bool(args.guidance_no_residual_rerun),
            "per_step_iters": int(args.per_step_iters),
            "per_step_lr": float(args.per_step_lr),
            "per_step_temperature": float(args.per_step_temperature),
            "per_step_start_step": int(args.per_step_start_step),
            "per_step_gumbel_scale": float(args.per_step_gumbel_scale),
            "per_clip": [
                {
                    "seq_id": seq_ids[i],
                    "info": {
                        k: (v if not isinstance(v, list) else v)
                        for k, v in per_clip_guided[i]["full"]["guidance_info"].items()
                    },
                }
                for i in range(len(samples))
            ],
        }
        with guidance_trace_path.open("w", encoding="utf-8") as f:
            import json as _json
            _json.dump(guidance_trace, f, indent=2)
        print(f"  Wrote {guidance_trace_path}")

    # Diff metrics for the default run.
    diffs_default = _summarise_diffs(per_clip_default, seq_lens_tok, seq_lens_frames)
    print("\n=== Diff metrics @ w_int =", args.w_int, "===")
    _print_diff_block(diffs_default)

    full_summary: dict[str, Any] = {
        "schema": f"stage_b_qual_eval_{args.summary_detail}_v1",
        "ckpt": str(args.ckpt),
        "config": str(args.config),
        "w_text": args.w_text,
        "default_w_int": args.w_int,
        "seed": args.seed,
        "num_clips": len(samples),
        "diffs_default": (
            diffs_default
            if args.summary_detail == "full"
            else _compact_diffs(diffs_default)
        ),
    }
    if args.summary_detail == "full":
        full_summary.update({
            "clip_ids": seq_ids,
            "clip_selection": selected_rows,
            "texts": texts,
            "seq_lens_frames": seq_lens_frames,
            "seq_lens_tokens": seq_lens_tok,
        })

    # Optional: w_int sweep on a single shared clip set.
    if args.w_int_sweep:
        sweep_values = [0.0, 1.0, 2.0, 4.0, 8.0]
        print(f"\n=== w_int sweep over {sweep_values} ===")
        sweep: dict[str, Any] = {}
        for w in sweep_values:
            print(f"  -- w_int = {w} --")
            per_clip_sw = [_run_three(i, w) for i in range(len(samples))]
            d = _summarise_diffs(per_clip_sw, seq_lens_tok, seq_lens_frames)
            sweep[f"w_int_{w:g}"] = (
                d if args.summary_detail == "full" else _compact_diffs(d)
            )
            _print_diff_block(d)
        full_summary["w_int_sweep"] = sweep

    summary_path = args.output_dir / "summary.json"
    save_json(summary_path, full_summary)
    print(f"\nSaved overall summary → {summary_path}")
    print("To render mp4 of the 'full' condition:")
    print(
        f"  python -m piano.inference.visualize_motion generated "
        f"--run-dir {args.output_dir / 'full'} "
        f"--output-dir {args.output_dir / 'full' / 'vis'}",
    )
    return 0


# ----------------------------------------------------------------------------
# Output helpers
# ----------------------------------------------------------------------------

def _save_condition_dir(
    out_dir: Path,
    per_clip: list[dict[str, dict]],
    condition: str,
    texts: list[str],
    seq_lens_frames: list[int],
    seq_ids: list[str],
    object_pcs: list[np.ndarray] | None = None,
    object_positions: list[np.ndarray] | None = None,
    object_rotations: list[np.ndarray] | None = None,
    world_R_y: list[float] | None = None,
    world_T_xz: list[np.ndarray] | None = None,
) -> None:
    """Save one condition as visualize_motion-compatible run dir.

    Different clips generate different lengths, so we right-pad each
    motion with zeros to the per-batch maximum before stacking. The
    saved ``seq_lens`` field tells the visualizer how many valid
    frames to render per row. Object overlays — ``object_pc`` (per-clip
    point cloud, fixed N), ``object_positions`` (per-frame center) and
    ``object_rotations`` (per-frame axis-angle) — are saved alongside
    when supplied; visualize_motion picks them up via its
    ``load_generated_samples`` extension.
    """
    ensure_dir(out_dir)
    motions_list = [row[condition]["motion"] for row in per_clip]
    max_T = max(m.shape[0] for m in motions_list)
    feat_dim = motions_list[0].shape[1]
    padded = np.zeros((len(motions_list), max_T, feat_dim), dtype=np.float32)
    for i, m in enumerate(motions_list):
        padded[i, : m.shape[0]] = m

    save_kwargs: dict[str, np.ndarray] = {"motion_263": padded}
    if object_pcs is not None:
        # Per-clip PC has fixed N (1024 from HOIDataset's subsample); stack.
        save_kwargs["object_pc"] = np.stack(object_pcs).astype(np.float32)
    if object_positions is not None:
        # Each is already padded to max_seq_length by HOIDataset; truncate
        # / right-pad to match motion's max_T so the visualizer can index
        # them in lock-step with the rendered motion.
        save_kwargs["object_positions"] = _pad_per_frame_to(object_positions, max_T)
    if object_rotations is not None:
        save_kwargs["object_rotations"] = _pad_per_frame_to(object_rotations, max_T)
    if world_R_y is not None:
        save_kwargs["world_R_y_angle"] = np.asarray(world_R_y, dtype=np.float32)
    if world_T_xz is not None:
        save_kwargs["world_T_xz"] = np.stack(world_T_xz).astype(np.float32)

    np.savez(out_dir / "generated.npz", **save_kwargs)
    summary = {
        "condition": condition,
        "texts": texts,
        "seq_ids": seq_ids,
        "seq_lens": seq_lens_frames,
        "swap_from": [row[condition].get("swap_from") for row in per_clip],
    }
    save_json(out_dir / "summary.json", summary)


def _pad_per_frame_to(arrs: list[np.ndarray], max_T: int) -> np.ndarray:
    """Stack per-clip per-frame arrays to (N, max_T, …), zero-padding
    along the time axis. Each arr has shape ``(T_i, *rest)``."""
    rest = arrs[0].shape[1:]
    out = np.zeros((len(arrs), max_T, *rest), dtype=np.float32)
    for i, a in enumerate(arrs):
        T = min(a.shape[0], max_T)
        out[i, :T] = a[:T]
    return out


def _summarise_diffs(
    per_clip: list[dict[str, dict]],
    seq_lens_tok: list[int],
    seq_lens_frames: list[int],
) -> dict[str, Any]:
    """Tally the 3 pairwise diffs (token Hamming + motion-263 RMS) across clips."""
    pairs = [
        ("full_vs_text_only", "full", "text_only"),
        ("full_vs_swap",      "full", "swap"),
        ("text_vs_swap",      "text_only", "swap"),
    ]
    out: dict[str, Any] = {}
    for name, a, b in pairs:
        ham = []
        l2 = []
        for i, row in enumerate(per_clip):
            ham.append(_hamming(row[a]["base"], row[b]["base"], valid_lens=seq_lens_tok[i]))
            l2.append(_motion_l2_per_frame(row[a]["motion"], row[b]["motion"],
                                           valid_frames=seq_lens_frames[i]))
        out[name] = {
            "token_hamming_per_clip":   [round(v, 4) for v in ham],
            "token_hamming_mean":       round(float(np.mean(ham)), 4),
            "motion_rms_per_clip":      [round(v, 4) for v in l2],
            "motion_rms_mean":          round(float(np.mean(l2)), 4),
        }
    return out


def _compact_diffs(diffs: dict[str, Any]) -> dict[str, Any]:
    return {
        name: {
            "token_hamming_mean": values["token_hamming_mean"],
            "motion_rms_mean": values["motion_rms_mean"],
        }
        for name, values in diffs.items()
    }


def _print_diff_block(d: dict[str, Any]) -> None:
    print(f"  {'pair':<22s} {'tok_hamming':>14s} {'motion_rms':>14s}")
    for k, v in d.items():
        print(
            f"  {k:<22s} {v['token_hamming_mean']:>14.4f} "
            f"{v['motion_rms_mean']:>14.4f}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
