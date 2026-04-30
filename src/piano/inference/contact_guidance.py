"""Contact-aware inference-time logit guidance for Stage B (B3).

Adapts the MaskControl/ControlMM recipe (Pinyoanuntapong et al.,
**ICCV 2025**, arXiv:2410.10780, code ``exitudio/ControlMM``) to PIANO's
MoMask + RVQ pipeline. Source-verified against
``exitudio/ControlMM@models/mask_transformer/control_transformer.py::generate_with_control``
2026-04-28.

Why this exists
---------------

Per [analyses/2026-04-28_v0_3_delta_retrain_and_v0_5_contact.md] and
Codex SUGGESTION.md 2026-04-28: PIANO's training objective (masked-CE on
base RVQ tokens) is empirically decoupled from the ship metric
(geometric body-to-object contact distance). Two same-architecture
ablations (v0.4→v0.5 epochs, v0.5→v0.6 γ_kind) showed CE and contact
move in opposite directions on isolated knobs. So fixing the training
loss directly is a multi-day commitment with uncertain payoff.

This module closes the loop **at inference time**: given a trained
generator, optimize the base-token logits in *decoded geometric space*
against `contact_target_xyz_gt`, then argmax + standard residual decode.
No model weight changes; runs on `v0.6 best_val.pt` as-is.

v15 adds `guidance_layers="full_rvq"` for the current alignment failure mode:
the optimized variable can be the complete generated base+residual RVQ stack,
then the full stack is argmax-decoded directly. This is the final-stage
full-RVQ counterpart to MaskControl-style decoded-space guidance.

Recipe (verified from `exitudio/ControlMM` source)
--------------------------------------------------

Differentiable chain:

    relaxed_base[B, S, d] = softmax(logits / T) @ codebooks[0]    # base layer only
    all_emb[Q, B, S, d]   = stack(relaxed_base, residual_codes_detached[1:])
    x[B, d, S]            = all_emb.sum(dim=0).permute(0, 2, 1)
    motion_norm[B, T, 263]= vq.decoder(x)
    motion[B, T, 263]     = motion_norm * std + mean
    joints_canon[B, T, 22, 3] = recover_from_ric(motion, 22)
    joints_world          = R_y(angle) @ joints_canon + T_xz   # source-clip anchor
    body[B, T, 5, 3]      = joints_world[:, :, [20, 21, 10, 11, 0]]
    loss                  = masked_L2(body, contact_target_xyz_gt) under contact_state

Optimizer: AdamW(betas=(0.5, 0.9), weight_decay=1e-6, lr=6e-2) on
`logits` only. Residual transformer is detached (frozen, no gradient
flow); after optimization we re-run residual on argmax(logits) to
capture any base-token change downstream.

First-prototype scope reduction (vs MaskControl's full recipe)
--------------------------------------------------------------

MaskControl optimizes BOTH inside each MaskGIT iteration
(``each_iter=100`` per step) AND post-hoc (``iter_last=600``). The
canonical recipe is therefore ~1600 optimization steps per clip, ~5-10
min on bf16. We start with **post-hoc final-stage only**, ~30 steps,
because:

- It's a strict subset of the published recipe (faithful, just slower
  to converge).
- Lets us isolate "does logit guidance move decoded contact at all?"
  before paying the per-iter cost.
- If 30 steps move contact ≥ 3 cm vs baseline → expand to MaskControl's
  full schedule. If < 1 cm → guidance ceiling reached, escalate to C1
  (residual stage z_int adapter).

References
----------
- MaskControl/ControlMM: Pinyoanuntapong, E. et al. ICCV 2025.
  arXiv:2410.10780. Source path verified:
  ``exitudio/ControlMM@models/mask_transformer/control_transformer.py``
  lines ~750-820 (per-iter TTT) and ~890-920 (final-stage TTT).
- MotionLCM cross-reference: ``Dai-Wenxun/MotionLCM@mld/models/modeltype/mld.py``
  uses ``Adam([current_latents.requires_grad_(True)], lr=...)`` with
  identical decoder→joints→hint-loss path. Confirms the optimizer +
  gradient-path pattern is the canonical one in masked-token motion
  control.
- analyses/2026-04-28_v0_3_delta_retrain_and_v0_5_contact.md — empirical
  motivation (CE/contact decoupling).
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from piano.utils.canonical_frame import axis_angle_to_matrix_np, y_rotation_matrix
from piano.utils.smpl_utils import BODY_PART_INDICES


# ============================================================================
# Differentiable decode (relaxed base + detached residual)
# ============================================================================

def _decode_relaxed_base(
    base_logits: Tensor,             # (B, S, V) — the optimised tensor
    residual_ids: Tensor,            # (B, S, Q-1) — argmax residuals, detached
    vq_model: torch.nn.Module,       # frozen MoMask RVQ-VAE
    *,
    temperature: float = 1.0,
) -> Tensor:
    """Differentiably decode a relaxed-base + frozen-residual mixture.

    Returns motion_norm of shape ``(B, T, 263)`` in **normalized** space
    (callers must apply ``* std + mean`` to denormalize). Gradient flows
    only through ``base_logits``; residual code lookup is in a no-grad
    context so the residual transformer is treated as frozen feature
    storage.

    Mirrors MoMask's ``RVQVAE.forward_decoder`` but replaces the base
    layer's ``get_codes_from_indices`` lookup with the relaxed
    expectation ``softmax(logits/T) @ codebook[0]``. The sum-then-permute
    + decode steps are bytewise the same as the original.
    """
    quantizer = vq_model.quantizer

    # Codebook stack: (Q, V, d). Property recomputes every call; that's
    # cheap (small tensor) and keeps it autograd-clean.
    codebooks = quantizer.codebooks                                           # (Q, V, d)
    base_codebook = codebooks[0]                                              # (V, d)

    # Relaxed base embedding (the differentiable bit).
    probs = F.softmax(base_logits / max(temperature, 1e-6), dim=-1)           # (B, S, V)
    relaxed_base = probs @ base_codebook                                      # (B, S, d)

    # Residual code embeddings (detached). Pad with -1 sentinel for
    # clipped quantizer count, exactly like ``get_codes_from_indices``.
    Q = quantizer.num_quantizers
    if residual_ids.shape[-1] != Q - 1:
        raise ValueError(
            f"residual_ids should have {Q - 1} layers (got {residual_ids.shape[-1]})",
        )
    with torch.no_grad():
        # Build full (B, S, Q) indices tensor with base set to a sentinel
        # 0 (we don't actually use that slice; the relaxed embedding
        # replaces it below).
        B, S = base_logits.shape[0], base_logits.shape[1]
        full_ids = torch.zeros((B, S, Q), dtype=torch.long, device=base_logits.device)
        full_ids[..., 1:] = residual_ids
        all_codes = quantizer.get_codes_from_indices(full_ids)                # (Q, B, S, d)
        residual_emb_sum = all_codes[1:].sum(dim=0)                           # (B, S, d) — Q≥2

    # Combine: replace the base layer's contribution with the relaxed
    # embedding, sum with the (detached) residual contributions.
    x_emb = relaxed_base + residual_emb_sum                                   # (B, S, d)
    x = x_emb.permute(0, 2, 1)                                                # (B, d, S)

    return vq_model.decoder(x)                                                # (B, T, 263)


def _force_pad_logits_to_zero_token(logits: Tensor, token_mask: Tensor) -> Tensor:
    """Force padded token slots to code id 0 for every RVQ layer."""
    pad_logits = torch.full_like(logits, -30.0)
    pad_logits[..., 0] = 30.0
    return torch.where(token_mask[..., None], logits, pad_logits)


def _decode_relaxed_full_rvq(
    rvq_logits: Tensor,              # (B, S, Q, V)
    vq_model: torch.nn.Module,
    *,
    token_mask: Tensor,              # (B, S)
    temperature: float = 1.0,
) -> Tensor:
    """Differentiably decode relaxed logits for every RVQ layer.

    This is the final-stage full-RVQ analogue of MaskControl-style logit
    optimization: all quantizer layers can move in decoded geometric space,
    instead of freezing residual ids while only base ids are optimized.
    """
    quantizer = vq_model.quantizer
    codebooks = quantizer.codebooks                                           # (Q, V, d)
    Q, V = int(codebooks.shape[0]), int(codebooks.shape[1])
    if rvq_logits.ndim != 4:
        raise ValueError(f"rvq_logits must be (B, S, Q, V), got {tuple(rvq_logits.shape)}")
    if int(rvq_logits.shape[2]) != Q:
        raise ValueError(f"rvq_logits has Q={rvq_logits.shape[2]}, VQ model has Q={Q}")
    if int(rvq_logits.shape[3]) < V:
        raise ValueError(f"rvq_logits vocab {rvq_logits.shape[3]} is smaller than codebook {V}")

    logits = rvq_logits[..., :V]
    logits = _force_pad_logits_to_zero_token(logits, token_mask[:, :, None])
    probs = F.softmax(logits / max(float(temperature), 1e-6), dim=-1)
    layer_emb = torch.einsum("bsqv,qvd->bsqd", probs, codebooks)
    x_emb = layer_emb.sum(dim=2)
    return vq_model.decoder(x_emb.permute(0, 2, 1))


# ============================================================================
# Object PC → world frame lift (constant per-clip, pre-computed once)
# ============================================================================

def _lift_pc_to_world_np(
    object_pc_local: np.ndarray,        # (N_pc, 3)
    object_positions: np.ndarray,       # (T, 3)
    object_rotations: np.ndarray,       # (T, 3) axis-angle world
) -> np.ndarray:                         # (T, N_pc, 3)
    """Lift the object's sampled PC into per-frame world coordinates.

    Mirrors ``_world_object_pc_per_frame`` in ``measure_contact_distance.py``
    and ``contact_eval.py`` — same einsum pattern. This is the canonical
    eval-metric reference: the metric is min over these PC samples per
    body part per frame.
    """
    R_obj = axis_angle_to_matrix_np(object_rotations.astype(np.float32))   # (T, 3, 3)
    pc_world = np.einsum("tij,nj->tni", R_obj, object_pc_local.astype(np.float32))
    pc_world += object_positions[:, None, :].astype(np.float32)
    return pc_world


# ============================================================================
# Object-local → world target lift (constant per-clip, pre-computed once)
# ============================================================================

def _lift_target_to_world_np(
    target_local: np.ndarray,           # (T, n_parts, 3) object-local frame
    object_positions: np.ndarray,       # (T, 3) world frame
    object_rotations: np.ndarray,       # (T, 3) axis-angle world frame
) -> np.ndarray:
    """Lift per-body-part contact target from object-local to world frame.

    The pseudo-label ``contact_target_xyz_gt`` field is stored in
    object-local frame (per ``src/piano/data/pseudo_labels/extract_target.py``
    docstring: "exact closest-surface-point in object-local frame via
    trimesh.proximity.closest_point"). To use it as the L2 target in
    a world-frame contact loss, we rigidly transform via the source
    clip's per-frame object pose:

        target_world[t, p, :] = R_obj[t] @ target_local[t, p, :] + obj_pos[t]

    Same convention as ``_world_object_pc_per_frame`` (used by the
    eval metric). Pre-computed once per clip; constant across the
    optimization loop.
    """
    R_obj = axis_angle_to_matrix_np(object_rotations.astype(np.float32))   # (T, 3, 3)
    rotated = np.einsum("tij,tpj->tpi", R_obj, target_local.astype(np.float32))
    return rotated + object_positions[:, None, :].astype(np.float32)         # (T, n_parts, 3)


# ============================================================================
# Loss mode "metric": min-over-PC, min-over-body-parts, mean-over-time
# (mirrors measure_contact_distance.py's eval metric exactly — used as loss)
# ============================================================================

def _eval_metric_as_loss(
    body_world: Tensor,             # (B, T, n_parts, 3)
    pc_world: Tensor,               # (B, T, N_pc, 3)
) -> Tensor:
    """The exact eval metric, used as a differentiable loss.

    Computes ``mean_t min_p min_n ||body[t,p,:] - pc_world[t,n,:]||``.
    This matches ``measure_contact_distance.py``'s
    ``mean_min_dist_per_frame`` exactly.

    Differentiable via PyTorch's subgradient on min(). Each body part
    + frame backprops gradient only through the closest PC sample —
    similar pattern to max-margin / nearest-neighbor losses.

    NOTE: this loss intentionally does NOT mask by contact_state. The
    eval metric considers all frames; using contact_state-masked L2
    against the GT target was the bug behind the 2026-04-28 mixed
    per-clip results (see analyses/...). Matching the metric exactly
    eliminates the loss-vs-metric decoupling.

    Returns scalar tensor.
    """
    # Pairwise distances: (B, T, n_parts, N_pc)
    diff = body_world[:, :, :, None, :] - pc_world[:, :, None, :, :]      # (B, T, n_parts, N_pc, 3)
    d = torch.linalg.vector_norm(diff, dim=-1)                            # (B, T, n_parts, N_pc)
    # Min over PC samples per body part:
    d_min_pc, _ = d.min(dim=-1)                                            # (B, T, n_parts)
    # Min over body parts per frame:
    d_min_parts, _ = d_min_pc.min(dim=-1)                                  # (B, T)
    # Mean over time:
    return d_min_parts.mean()


# ============================================================================
# Loss mode "target": masked L2 against contact_target_xyz_gt (lifted to world) under contact_state
# ============================================================================

def _masked_contact_l2(
    body_world: Tensor,              # (B, T, n_parts, 3)
    target_world: Tensor,            # (B, T, n_parts, 3)
    contact_state: Tensor,           # (B, T, n_parts) — 1 = in contact
) -> Tensor:
    """MSE on (body, target) summed over xyz, masked by contact_state.

    Mirrors MaskControl's ``get_loss``: per-frame, per-part L2; reduce
    by mean over frames and parts where contact is asserted. Returns a
    scalar tensor.

    If no part is in contact across the whole batch (contact_state all
    zero), returns 0 — caller should detect this and skip the
    optimization loop.
    """
    diff_sq = ((body_world - target_world) ** 2).sum(dim=-1)                  # (B, T, n_parts)
    mask = contact_state.float()                                              # (B, T, n_parts)
    denom = mask.sum().clamp(min=1.0)
    return (diff_sq * mask).sum() / denom


# ============================================================================
# Canonical → world lift (differentiable; mirrors contact_eval helper)
# ============================================================================

def _lift_canonical_to_world_torch(
    joints_canon: Tensor,            # (B, T, 22, 3)
    R_y_angle: float,
    T_xz: np.ndarray,
) -> Tensor:
    """Differentiable canonical→world lift; same math as contact_eval._lift_canonical_to_world."""
    R = torch.from_numpy(y_rotation_matrix(float(R_y_angle))).to(
        joints_canon.device, dtype=joints_canon.dtype,
    )                                                                          # (3, 3)
    rotated = joints_canon @ R.T                                              # (B, T, 22, 3)
    out = rotated.clone()
    out[..., 0] = out[..., 0] + float(T_xz[0])
    out[..., 2] = out[..., 2] + float(T_xz[1])
    return out


# ============================================================================
# no-residual-rerun helper: combine post-guidance base with baseline residuals
# ============================================================================

def _build_decode_ids_with_baseline_residuals(
    base_ids_after: Tensor,           # (1, S) post-guidance argmax
    baseline_residual_ids: Tensor,    # (1, S, Q-1) from baseline residual call
    m_lens_tok: Tensor,               # (1,) actual token-space sequence length
) -> Tensor:                          # (1, S, Q)
    """Build decoder input from post-guidance base + frozen baseline residuals.

    Used when ``no_residual_rerun=True``: the post-guidance base may differ
    from the baseline base (token flips from optimization), but residuals
    are reused as-is from the baseline call. This isolates "what does base
    flip alone do?" by removing the residual rerun's contribution to
    contact deltas (RNG drift + autoregressive feedback both eliminated).

    Pad convention: the residual-rerun path zeros pad positions in the
    decoder input via ``where(all_ids_after < 0, 0, all_ids_after)``. To
    match that convention, this helper zeros pad positions in
    ``base_ids_after`` (residuals are already 0 at pad — inherited from
    baseline's where clause).
    """
    S = int(base_ids_after.shape[-1])
    pad_mask = (
        torch.arange(S, device=base_ids_after.device).unsqueeze(0)
        >= m_lens_tok.unsqueeze(-1)
    )                                                                          # (1, S)
    base_padded = torch.where(
        pad_mask,
        torch.zeros_like(base_ids_after),
        base_ids_after,
    )
    return torch.cat(
        [base_padded.unsqueeze(-1), baseline_residual_ids],
        dim=-1,
    )


def _precompute_residual_emb_sum(
    all_ids: Tensor,                    # (B, S, Q) — full RVQ ids from a baseline pass
    vq_model: torch.nn.Module,
) -> Tensor:                             # (B, S, d) — frozen residual context
    """Precompute the sum of residual codebook embeddings for layers 1..Q-1.

    Used by per-step guidance as the frozen residual context approximation
    while the MaskGIT loop is still committing base-layer tokens (the
    real residual tokens for the in-progress base distribution don't
    exist yet). Computed once from a baseline residual pass and reused
    across all MaskGIT iterations + inner guidance steps. Bias source:
    the baseline residuals were generated on a different base
    distribution. After per-step guidance finishes, we re-run the
    residual transformer on the post-guidance base to absorb the drift.
    """
    quantizer = vq_model.quantizer
    Q = int(quantizer.num_quantizers)
    if int(all_ids.shape[-1]) != Q:
        raise ValueError(
            f"all_ids should have Q={Q} layers (got {all_ids.shape[-1]})",
        )
    all_codes = quantizer.get_codes_from_indices(all_ids)                     # (Q, B, S, d)
    return all_codes[1:].sum(dim=0)                                            # (B, S, d)


def _decode_with_relaxed_masked_base(
    base_logits: Tensor,                 # (B, S, V) — optimised inside per-step inner loop
    committed_ids: Tensor,               # (B, S) — hard tokens at non-masked positions, anything at masked
    differentiable_mask: Tensor,         # (B, S) bool — True at positions where logits matter
    baseline_residual_emb_sum: Tensor,   # (B, S, d) frozen
    vq_model: torch.nn.Module,
    *,
    temperature: float = 1.0,
    gumbel_noise_scale: float = 0.0,
) -> Tensor:                              # (B, T, 263) normalised motion
    """Differentiable decode for per-step MaskGIT guidance.

    At ``differentiable_mask=True`` positions the base layer uses the
    relaxed expectation ``softmax(logits/T) @ codebook[0]`` (gradient
    flows here). At ``False`` positions it uses the hard codebook entry
    ``codebook[0][committed_ids]`` (gradient blocked because committed
    tokens won't change). Residual context is the frozen
    ``baseline_residual_emb_sum`` precomputed once from the baseline
    residual pass.

    ``gumbel_noise_scale`` (added 2026-05-01 to match MaskControl's
    canonical recipe; verified from
    ``exitudio/ControlMM@models/mask_transformer/control_transformer.py``
    `each_iter` block): when > 0, adds standard Gumbel(0, 1) noise to
    ``logits`` before softmax. This is the Gumbel-Softmax / Concrete
    relaxation: ``softmax((logits + scale*gumbel_noise) / T)`` is an
    expected-loss-under-sampled-tokens estimator instead of a pure
    softmax expectation. PIANO v17-C/D/E used 0.0 (pure softmax); v17-F
    onwards uses 1.0 (matching MaskControl).
    """
    quantizer = vq_model.quantizer
    base_codebook = quantizer.codebooks[0]                                    # (V, d)
    V = int(base_codebook.shape[0])

    if gumbel_noise_scale > 0.0:
        # Standard Gumbel(0, 1): -log(-log(U)). Inlined to keep this helper
        # CPU-importable (MoMask's tools.gumbel_noise lives behind the
        # backbone adapter which isn't on path during pure-CPU tests).
        # Numerically identical to ``models.mask_transformer.tools.gumbel_noise``.
        u = torch.rand_like(base_logits).clamp_(min=1e-20, max=1.0 - 1e-7)
        gn = -torch.log(-torch.log(u))
        noisy_logits = base_logits + float(gumbel_noise_scale) * gn
    else:
        noisy_logits = base_logits
    soft_emb = F.softmax(noisy_logits / max(temperature, 1e-6), dim=-1) @ base_codebook
    safe_ids = committed_ids.clamp(min=0, max=V - 1)
    hard_emb = base_codebook[safe_ids]                                        # (B, S, d)
    relaxed_base = torch.where(
        differentiable_mask.unsqueeze(-1), soft_emb, hard_emb,
    )                                                                          # (B, S, d)
    x_emb = relaxed_base + baseline_residual_emb_sum                          # (B, S, d)
    return vq_model.decoder(x_emb.permute(0, 2, 1))                           # (B, T, 263)


def _generate_residual_tokens(
    res_transformer: torch.nn.Module,
    *,
    motion_ids: Tensor,
    text: str,
    m_lens_tok: Tensor,
    int_kv: Tensor | None,
    int_pad: Tensor | None,
    res_cond_scale: float,
) -> Tensor:
    """Generate RVQ residual tokens, using C1 z_int path when available."""
    if hasattr(res_transformer, "generate_with_int"):
        res_int_kv = (
            None if int_kv is None
            else int_kv.transpose(0, 1).contiguous()
        )
        return res_transformer.generate_with_int(
            motion_ids=motion_ids,
            conds=[text],
            m_lens=m_lens_tok,
            int_kv=res_int_kv,
            int_padding_mask=int_pad,
            cond_scale=res_cond_scale,
        )
    return res_transformer.generate(
        motion_ids=motion_ids,
        conds=[text],
        m_lens=m_lens_tok,
        cond_scale=res_cond_scale,
    )


# ============================================================================
# Per-step decoded-geometric guidance (re-rolled MaskGIT loop)
# ============================================================================

def _generate_with_per_step_guidance(
    transformer: torch.nn.Module,             # InteractionMaskTransformer
    vq_model: torch.nn.Module,
    *,
    cond_vector: Tensor,                      # (1, clip_dim) — pre-encoded text
    m_lens_tok: Tensor,                       # (1,) token-space length
    int_tokens_bf: Tensor | None,             # (1, S_int, d) or None
    int_padding_mask_bf: Tensor | None,       # (1, S_int) or None
    baseline_residual_emb_sum: Tensor,        # (1, S, d) frozen residual context
    target_world_t: Tensor | None,            # (1, T, n_parts, 3) for loss_mode="target"
    contact_state_t: Tensor | None,           # (1, T, n_parts) for loss_mode="target"
    pc_world_t: Tensor | None,                # (1, T, N_pc, 3) for loss_mode="metric"
    R_y_angle: float,
    T_xz: np.ndarray,
    motion_mean: Tensor,
    motion_std: Tensor,
    body_part_indices: tuple[int, ...],
    loss_mode: str,                           # "target" or "metric"
    timesteps: int = 10,
    w_text: float = 4.0,
    w_int: float = 2.0,
    base_temperature: float = 1.0,
    topk_filter_thres: float = 0.9,
    per_step_iters: int = 10,
    per_step_lr: float = 6e-2,
    per_step_temperature: float = 1.0,
    per_step_start_step: int = 0,
    per_step_gumbel_scale: float = 1.0,  # 1.0 matches MaskControl; 0.0 = pre-v17-F PIANO behaviour
    device: torch.device,
    log_progress: bool = False,
) -> tuple[Tensor, dict[str, Any]]:
    """Re-rolled InteractionMaskTransformer.generate with per-step decoded-geometric guidance.

    Replaces ``transformer.generate(...)`` for the v17 ablation. At each
    MaskGIT iteration, after the model produces logits but before the
    commit decision, runs ``per_step_iters`` AdamW inner steps on the
    logits using a relaxed-decode geometric loss (MaskControl-style
    ``each_iter`` test-time training). The optimised logits then drive
    the standard MaskGIT top-k + gumbel commit. Padded positions stay
    pad_id; their logits are not optimised. Residual context for the
    relaxed decode comes from ``baseline_residual_emb_sum``, frozen.

    Mirrors the unrolled MaskGIT loop in
    ``InteractionMaskTransformer.generate`` (see
    ``src/piano/models/motion_generator.py:866-948``); the only change
    is the per-step inner optimisation between logit computation and
    sampling. Returns the same ids tensor convention as the wrapper:
    pad positions = -1, real positions = code id.

    See ``analyses/2026-05-01_per_step_guidance_design.md`` for design
    rationale, MaskControl source-code references, ablation plan, and
    failure modes.
    """
    # Lazy MoMask imports (consistent with the post-hoc path).
    from models.mask_transformer.tools import (
        cosine_schedule, gumbel_sample, lengths_to_mask, top_k,
    )
    from utils.motion_process import recover_from_ric

    if loss_mode not in {"target", "metric"}:
        raise ValueError(f"loss_mode must be 'target' or 'metric', got {loss_mode!r}")

    mt = transformer.mask_transformer
    B = int(cond_vector.shape[0])
    if B != 1:
        raise NotImplementedError(
            "per-step guidance currently supports single-clip B=1 only; "
            "batched per-clip eval pipeline already iterates one clip at a time."
        )
    S = int(m_lens_tok.max().item())

    non_pad_mask = lengths_to_mask(m_lens_tok, S)                              # (B, S)
    padding_mask = ~non_pad_mask
    ids = torch.where(padding_mask, mt.pad_id, mt.mask_id)
    scores = torch.where(
        padding_mask,
        torch.full_like(ids, 1e5, dtype=torch.float),
        torch.zeros_like(ids, dtype=torch.float),
    )

    body_idx_t = torch.tensor(list(body_part_indices), device=device, dtype=torch.long)

    per_step_loss_initial: list[float] = []
    per_step_loss_final: list[float] = []
    per_step_active: list[int] = []  # int (0/1) for JSON friendliness

    timestep_grid = torch.linspace(0, 1, timesteps, device=device)
    for step_idx, timestep in enumerate(timestep_grid):
        rand_mask_prob = cosine_schedule(timestep)
        num_token_masked = torch.round(
            rand_mask_prob * m_lens_tok.float(),
        ).clamp(min=1)
        sorted_indices = scores.argsort(dim=1)
        ranks = sorted_indices.argsort(dim=1)
        is_mask = ranks < num_token_masked.unsqueeze(-1)
        ids = torch.where(is_mask, mt.mask_id, ids)

        # Forward through 3-pass compositional CFG (no grad — we'll detach
        # and re-instantiate logits as the optimisation parameter).
        with torch.no_grad():
            logits_raw = transformer.forward_with_cond_scale(
                motion_ids=ids,
                cond_vector=cond_vector,
                token_padding_mask=padding_mask,
                int_tokens_bf=int_tokens_bf,
                int_padding_mask_bf=int_padding_mask_bf,
                w_text=w_text,
                w_int=w_int,
            ).permute(0, 2, 1)                                                # (B, S, V)

        active = (per_step_iters > 0) and (step_idx >= per_step_start_step)
        loss_init_step = float("nan")
        loss_final_step = float("nan")

        if active:
            # Differentiable mask: only the still-masked + non-pad positions
            # contribute gradient. Committed tokens (and pad) use their hard
            # codebook embedding.
            differentiable_mask = is_mask & non_pad_mask                       # (B, S)
            # Committed-id tensor: anything safe at masked positions (gets
            # overwritten by the soft branch in _decode_with_relaxed_masked_base
            # via the where-mask).
            committed_ids = torch.where(
                differentiable_mask,
                torch.zeros_like(ids),
                ids,
            )
            logits_param = logits_raw.detach().clone().requires_grad_(True)
            optimizer = torch.optim.AdamW(
                [logits_param], lr=per_step_lr, betas=(0.5, 0.9),
                weight_decay=1e-6,
            )
            for inner_idx in range(per_step_iters):
                motion_norm = _decode_with_relaxed_masked_base(
                    base_logits=logits_param,
                    committed_ids=committed_ids,
                    differentiable_mask=differentiable_mask,
                    baseline_residual_emb_sum=baseline_residual_emb_sum,
                    vq_model=vq_model,
                    temperature=per_step_temperature,
                    gumbel_noise_scale=per_step_gumbel_scale,
                )                                                              # (B, T, 263)
                motion = motion_norm * motion_std + motion_mean
                joints_canon = recover_from_ric(motion, 22)                    # (B, T, 22, 3)
                joints_world = _lift_canonical_to_world_torch(
                    joints_canon, R_y_angle, T_xz,
                )
                body_world = joints_world.index_select(2, body_idx_t)          # (B, T, 5, 3)
                T_dec = body_world.shape[1]
                if loss_mode == "target":
                    T_use = min(T_dec, target_world_t.shape[1])
                    loss = _masked_contact_l2(
                        body_world=body_world[:, :T_use],
                        target_world=target_world_t[:, :T_use],
                        contact_state=contact_state_t[:, :T_use],
                    )
                else:
                    T_use = min(T_dec, pc_world_t.shape[1])
                    loss = _eval_metric_as_loss(
                        body_world=body_world[:, :T_use],
                        pc_world=pc_world_t[:, :T_use],
                    )
                if inner_idx == 0:
                    loss_init_step = float(loss.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_final_step = float(loss.detach())
            logits_for_commit = logits_param.detach()
        else:
            logits_for_commit = logits_raw

        per_step_loss_initial.append(loss_init_step)
        per_step_loss_final.append(loss_final_step)
        per_step_active.append(int(active))

        # Standard MaskGIT commit on the (possibly optimised) logits.
        filtered_logits = top_k(logits_for_commit, topk_filter_thres, dim=-1)
        pred_ids = gumbel_sample(
            filtered_logits, temperature=base_temperature, dim=-1,
        )
        ids = torch.where(is_mask, pred_ids, ids)

        probs = logits_for_commit.softmax(dim=-1)
        tok_scores = probs.gather(2, pred_ids.unsqueeze(-1)).squeeze(-1)
        scores = torch.where(is_mask, tok_scores, scores)
        scores = scores.masked_fill(~is_mask, 1e5)

        if log_progress:
            tag = "ON " if active else "off"
            print(
                f"      [per-step {step_idx:2d}/{int(timesteps) - 1}] {tag} "
                f"loss {loss_init_step:.4f} -> {loss_final_step:.4f}",
            )

    ids = torch.where(padding_mask, torch.full_like(ids, -1), ids)
    info = {
        "per_step_iters": int(per_step_iters),
        "per_step_lr": float(per_step_lr),
        "per_step_temperature": float(per_step_temperature),
        "per_step_start_step": int(per_step_start_step),
        "per_step_gumbel_scale": float(per_step_gumbel_scale),
        "per_step_loss_initial": per_step_loss_initial,
        "per_step_loss_final": per_step_loss_final,
        "per_step_active": per_step_active,
    }
    return ids, info


# ============================================================================
# Public entry point
# ============================================================================

def guide_with_contact(
    transformer: torch.nn.Module,    # InteractionMaskTransformer (full wrapper)
    vq_model: torch.nn.Module,       # frozen MoMask RVQ-VAE
    res_transformer: torch.nn.Module,
    *,
    text: str,
    int_kv: Tensor,                  # (S_int, 1, d) — interaction K/V (B=1)
    int_pad: Tensor | None,          # (1, S_int) or None
    m_lens_tok: Tensor,              # (1,)
    contact_target_xyz_local: np.ndarray,   # (T, 5, 3) — closest-surface-point per body part, OBJECT-LOCAL frame
    contact_state: np.ndarray,              # (T, 5) — binary in-contact mask
    object_pc_local: np.ndarray,            # (N_pc, 3) — object PC in object-local frame
    object_positions: np.ndarray,           # (T, 3) — world-frame object COM per frame
    object_rotations: np.ndarray,           # (T, 3) — world-frame object axis-angle per frame
    R_y_angle: float,                # source clip's canonical→world rotation (for body lift)
    T_xz: np.ndarray,                # (2,) source clip's canonical→world translation (for body lift)
    motion_mean: Tensor,             # (263,) on device
    motion_std: Tensor,              # (263,) on device
    w_text: float = 4.0,
    w_int: float = 2.0,
    timesteps: int = 10,
    res_cond_scale: float = 2.0,
    num_guidance_steps: int = 30,
    guidance_lr: float = 6e-2,
    guidance_temperature: float = 1.0,
    init_logit_scale: float = 3.0,
    loss_mode: str = "target",       # "target" or "metric"
    residual_seed: int | None = None,
    no_residual_rerun: bool = False,
    guidance_layers: str = "base",   # "base" or "full_rvq"
    per_step_iters: int = 0,         # 0 = no per-step guidance (back-compat)
    per_step_lr: float = 6e-2,
    per_step_temperature: float = 1.0,
    per_step_start_step: int = 0,
    per_step_gumbel_scale: float = 1.0,  # 1.0 matches MaskControl; 0.0 = pre-v17-F PIANO
    body_part_indices: tuple[int, ...] = tuple(BODY_PART_INDICES),
    device: torch.device,
    log_progress: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Run baseline generate, then optimize base-token logits in decoded space.

    Returns
    -------
    motion_263_denorm : (T, 263) float32, denormalized HumanML3D scale
    base_ids_after_guidance : (S,) int64, the post-optimization argmax tokens
    info : dict with optimization trace (loss_initial, loss_final,
        steps_taken, base_token_change_count). Useful for logging / wandb.

    Hyperparameters defaults match MaskControl arXiv:2410.10780 verified
    source values where possible. ``num_guidance_steps=30`` is a
    deliberate scope reduction vs MaskControl's ``iter_last=600``;
    increase if 30 doesn't move contact distance vs baseline.

    ``residual_seed`` and ``no_residual_rerun`` (added 2026-04-28 v5):
    diagnostic toggles for isolating the residual-rerun variance
    surfaced by v4. Background: MoMask's ``ResidualTransformer.generate``
    samples each of the 5 residual layers via ``gumbel_sample``
    (transformer.py:949 + tools.py:90-95) using ``torch.uniform_(0, 1)``
    on the global RNG. So calling it twice with identical inputs
    produces different residual_ids — a "RNG drift" channel for
    contact deltas. v4 plasticbox_014 (0/23 base flips, +7.3 cm
    contact regression) is the smoking gun.

    - ``residual_seed=N`` (default ``None``): if set, calls
      ``torch.manual_seed(N)`` immediately before each
      ``res_transformer.generate`` call (baseline + post-guidance).
      Forces RNG-equivalence between the two calls so any contact
      delta is attributable to base-token changes or autoregressive
      feedback, NOT Gumbel-noise drift.
    - ``no_residual_rerun=True`` (default ``False``): skip the
      post-guidance ``res_transformer.generate`` call entirely;
      decode using baseline residual_ids combined with new
      ``base_ids_after``. Tests whether residual rerun itself is
      the per-clip variance source. Cost: forfeits any "base
      flip → residual self-adapts" gain (e.g., v4's
      largebox_010 −14 cm may have come from this self-adaptation).

    Side effect: ``residual_seed`` mutates the global ``torch`` RNG.
    Callers wanting strict reproducibility of downstream code should
    re-seed afterward.
    """
    # Lazy MoMask import (the inference module is also imported by
    # CPU-only test paths).
    import piano.models.backbones.momask_adapter  # noqa: F401
    from utils.motion_process import recover_from_ric

    transformer.eval()
    vq_model.eval()
    res_transformer.eval()
    if guidance_layers not in {"base", "full_rvq"}:
        raise ValueError(
            f"guidance_layers must be 'base' or 'full_rvq', got {guidance_layers!r}",
        )

    # --- Step 1: baseline generation (no grad) ---
    with torch.no_grad():
        cond_vector = transformer.encode_text([text]).to(device).float()
        base_ids_baseline = transformer.generate(
            cond_vector=cond_vector,
            m_lens_tok=m_lens_tok,
            int_tokens_bf=int_kv,
            int_padding_mask_bf=int_pad,
            timesteps=timesteps,
            w_text=w_text,
            w_int=w_int,
            temperature=1.0,
        )                                                                      # (1, S_max), -1 at pad
        base_for_res = torch.where(
            base_ids_baseline < 0,
            torch.zeros_like(base_ids_baseline),
            base_ids_baseline,
        )
        # Seed before residual generation so the post-guidance call (Step 4)
        # can use the same seed and get RNG-identical Gumbel noise. Without
        # this, gumbel_sample drift alone can change residual_ids even when
        # base_ids_after == base_for_res bit-identical.
        if residual_seed is not None:
            torch.manual_seed(int(residual_seed))
        all_ids = _generate_residual_tokens(
            res_transformer,
            motion_ids=base_for_res,
            text=text,
            m_lens_tok=m_lens_tok,
            int_kv=int_kv,
            int_pad=int_pad,
            res_cond_scale=res_cond_scale,
        )                                                                      # (1, S, Q)
        all_for_decode = torch.where(all_ids < 0, torch.zeros_like(all_ids), all_ids)

    # If there's nothing to guide against (no contact frames), short-circuit:
    # baseline's the answer.
    contact_state_np = np.asarray(contact_state, dtype=np.float32)
    if float(contact_state_np.sum()) < 0.5:
        with torch.no_grad():
            motion = vq_model.forward_decoder(all_for_decode)
            motion = motion.squeeze(0) * motion_std + motion_mean
        return (
            motion.detach().cpu().numpy().astype(np.float32),
            base_ids_baseline.squeeze(0).detach().cpu().numpy(),
            {"skipped": "no_contact_frames", "loss_initial": float("nan"),
             "loss_final": float("nan"), "steps_taken": 0,
             "base_token_change_count": 0,
             "rvq_token_change_count": 0,
             "residual_seed": residual_seed,
             "no_residual_rerun": bool(no_residual_rerun),
             "guidance_layers": guidance_layers,
             "per_step": None},
        )

    # Pre-compute the world-frame loss reference ONCE (constant across
    # the optimization loop). Per the data-loader docstring, the
    # ``contact_target_xyz_gt`` field is in OBJECT-LOCAL frame; the
    # body we compare against is in WORLD frame after the
    # canonical→world lift. Without this rigid transform the loss
    # compares two frames whose alignment depends on where the object
    # sits in the world — which produces wildly variable gradients
    # across clips and was the bug behind an earlier mixed-result run.
    if loss_mode == "target":
        # Lift target points from object-local to world. (T, 5, 3).
        target_world_np = _lift_target_to_world_np(
            target_local=np.asarray(contact_target_xyz_local, dtype=np.float32),
            object_positions=np.asarray(object_positions, dtype=np.float32),
            object_rotations=np.asarray(object_rotations, dtype=np.float32),
        )
        target_world_t = torch.from_numpy(target_world_np).to(device).float().unsqueeze(0)
        contact_state_t = torch.from_numpy(contact_state_np).to(device).float().unsqueeze(0)
        pc_world_t = None
    elif loss_mode == "metric":
        # Lift PC samples from object-local to world. (T, N_pc, 3).
        # This becomes the loss reference set: distances are min over
        # the lifted PC, matching measure_contact_distance.py's eval
        # metric exactly.
        pc_world_np = _lift_pc_to_world_np(
            object_pc_local=np.asarray(object_pc_local, dtype=np.float32),
            object_positions=np.asarray(object_positions, dtype=np.float32),
            object_rotations=np.asarray(object_rotations, dtype=np.float32),
        )
        pc_world_t = torch.from_numpy(pc_world_np).to(device).float().unsqueeze(0)
        target_world_t = None
        contact_state_t = None
    else:
        raise ValueError(f"loss_mode must be 'target' or 'metric', got {loss_mode!r}")

    # --- Step 1.5: optional per-step decoded-geometric guidance ---
    # When ``per_step_iters > 0`` we discard the naive baseline base_ids
    # and re-run MaskGIT with per-step inner optimisation. The naive
    # baseline residual pass is reused only as the frozen residual
    # context approximation during the inner loops; after per-step
    # generation completes we re-run the residual transformer on the
    # post-guidance base_ids so the saved motion uses fresh residuals.
    # See analyses/2026-05-01_per_step_guidance_design.md.
    per_step_info: dict[str, Any] | None = None
    if per_step_iters > 0:
        with torch.no_grad():
            baseline_residual_emb_sum = _precompute_residual_emb_sum(
                all_for_decode, vq_model,
            )                                                                  # (1, S, d), frozen
        naive_base_ids = base_ids_baseline.detach().clone()
        base_ids_baseline, per_step_info = _generate_with_per_step_guidance(
            transformer=transformer,
            vq_model=vq_model,
            cond_vector=cond_vector,
            m_lens_tok=m_lens_tok,
            int_tokens_bf=int_kv,
            int_padding_mask_bf=int_pad,
            baseline_residual_emb_sum=baseline_residual_emb_sum,
            target_world_t=target_world_t,
            contact_state_t=contact_state_t,
            pc_world_t=pc_world_t,
            R_y_angle=R_y_angle,
            T_xz=T_xz,
            motion_mean=motion_mean,
            motion_std=motion_std,
            body_part_indices=body_part_indices,
            loss_mode=loss_mode,
            timesteps=timesteps,
            w_text=w_text,
            w_int=w_int,
            base_temperature=1.0,
            per_step_iters=per_step_iters,
            per_step_lr=per_step_lr,
            per_step_temperature=per_step_temperature,
            per_step_start_step=per_step_start_step,
            per_step_gumbel_scale=per_step_gumbel_scale,
            device=device,
            log_progress=log_progress,
        )
        base_for_res = torch.where(
            base_ids_baseline < 0,
            torch.zeros_like(base_ids_baseline),
            base_ids_baseline,
        )
        with torch.no_grad():
            if residual_seed is not None:
                torch.manual_seed(int(residual_seed))
            all_ids = _generate_residual_tokens(
                res_transformer,
                motion_ids=base_for_res,
                text=text,
                m_lens_tok=m_lens_tok,
                int_kv=int_kv,
                int_pad=int_pad,
                res_cond_scale=res_cond_scale,
            )
            all_for_decode = torch.where(
                all_ids < 0, torch.zeros_like(all_ids), all_ids,
            )
        # Record how many base tokens the per-step guidance changed
        # vs the naive baseline (excludes pad). Post-hoc Step 3 then
        # operates on top of this and reports its own change count
        # in the standard info["base_token_change_count"] field.
        with torch.no_grad():
            naive_token_mask = (
                torch.arange(int(naive_base_ids.shape[-1]), device=device).unsqueeze(0)
                < m_lens_tok.to(device=device).long().clamp(min=1).unsqueeze(1)
            )
            per_step_info["per_step_base_token_change_count"] = int(
                ((base_ids_baseline != naive_base_ids) & naive_token_mask).sum().item(),
            )
            per_step_info["per_step_base_token_total"] = int(naive_token_mask.sum().item())

    # --- Step 2: initialize logits to optimize ---
    # One-hot(base_ids) * init_scale. Stronger init = optimization
    # respects the model's prior more, but softer init allows tokens
    # to flip within the budget.
    #
    # 2026-04-28 calibration: init_scale=10 made softmax effectively
    # one-hot (p(argmax) ≈ 0.99996), so 30 AdamW steps × lr=6e-2 (≈1.8
    # max logit change) couldn't overcome the gap → 0/N tokens flipped
    # across all 5 clips even though loss dropped 30-64%. Default
    # lowered to 3.0 (p(argmax) ≈ 0.038, model preference preserved
    # but flip-able).
    #
    # MaskControl's actual recipe inits from the model's last-iteration
    # logits (top-k filtered) which have similar magnitude but a
    # softer top-of-distribution because model uncertainty distributes
    # among multiple plausible tokens. Achieving that exactly requires
    # exposing logits from generate(); deferred (init_scale=3.0 is the
    # cheaper approximation).
    V = vq_model.quantizer.codebooks.shape[1]                                 # 512
    S = base_for_res.shape[-1]
    token_mask = (
        torch.arange(S, device=device).unsqueeze(0)
        < m_lens_tok.to(device=device).long().clamp(min=1).unsqueeze(1)
    )
    if guidance_layers == "base":
        logits = F.one_hot(base_for_res.long(), num_classes=V).float() * init_logit_scale
        logits = logits.detach().clone().requires_grad_(True)                 # (1, S, V)
        # Detached residual indices (Q-1 layers).
        residual_ids = all_for_decode[..., 1:].detach()                       # (1, S, Q-1)
    else:
        # Full-RVQ final-stage guidance. We initialize from the generated
        # base+residual stack and let every quantizer layer move through the
        # frozen VQ decoder. This is the path to use when residual layers are
        # part of the observed contact misalignment.
        logits = (
            F.one_hot(all_for_decode.long().clamp(min=0, max=int(V) - 1), num_classes=V)
            .float()
            * init_logit_scale
        )
        logits = logits.detach().clone().requires_grad_(True)                 # (1, S, Q, V)
        residual_ids = None

    # --- Step 3: optimization loop ---
    optimizer = torch.optim.AdamW(
        [logits], lr=guidance_lr, betas=(0.5, 0.9), weight_decay=1e-6,
    )
    body_idx_t = torch.tensor(list(body_part_indices), device=device, dtype=torch.long)

    loss_initial: float = float("nan")
    loss_trace: list[float] = []
    for step in range(num_guidance_steps):
        if guidance_layers == "base":
            motion_norm = _decode_relaxed_base(
                base_logits=logits,
                residual_ids=residual_ids,
                vq_model=vq_model,
                temperature=guidance_temperature,
            )                                                                  # (1, T, 263)
        else:
            motion_norm = _decode_relaxed_full_rvq(
                rvq_logits=logits,
                vq_model=vq_model,
                token_mask=token_mask,
                temperature=guidance_temperature,
            )                                                                  # (1, T, 263)
        motion = motion_norm * motion_std + motion_mean
        joints_canon = recover_from_ric(motion, 22)                            # (1, T, 22, 3)
        joints_world = _lift_canonical_to_world_torch(joints_canon, R_y_angle, T_xz)
        body_world = joints_world.index_select(2, body_idx_t)                  # (1, T, 5, 3)

        # Truncate to actual decoded T (residual decode may differ from
        # seq_lens-derived T by 1-3 frames due to VQ stride).
        T_dec = body_world.shape[1]
        if loss_mode == "target":
            T_tgt = target_world_t.shape[1]
            T_use = min(T_dec, T_tgt)
            loss = _masked_contact_l2(
                body_world=body_world[:, :T_use],
                target_world=target_world_t[:, :T_use],
                contact_state=contact_state_t[:, :T_use],
            )
        else:  # loss_mode == "metric"
            T_pc = pc_world_t.shape[1]
            T_use = min(T_dec, T_pc)
            loss = _eval_metric_as_loss(
                body_world=body_world[:, :T_use],
                pc_world=pc_world_t[:, :T_use],
            )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_trace.append(float(loss.detach()))
        if step == 0:
            loss_initial = float(loss.detach())
        if log_progress and step % max(num_guidance_steps // 5, 1) == 0:
            print(f"    [guidance step {step:3d}] loss={float(loss.detach()):.4f}")

    loss_final = loss_trace[-1] if loss_trace else float("nan")

    # --- Step 4: argmax + (optionally) re-run residual on possibly-changed base ---
    with torch.no_grad():
        if guidance_layers == "base":
            base_ids_after = logits.argmax(dim=-1)                             # (1, S)
            token_change_count = int(
                ((base_ids_after != base_for_res) & token_mask).sum().item(),
            )
            rvq_token_change_count = token_change_count

            if no_residual_rerun:
                # Skip res_transformer.generate; reuse baseline residual_ids
                # (already detached; pad already 0).
                all_for_decode_after = _build_decode_ids_with_baseline_residuals(
                    base_ids_after=base_ids_after,
                    baseline_residual_ids=residual_ids,
                    m_lens_tok=m_lens_tok,
                )
            else:
                # Existing path: re-run residual on the post-guidance base.
                # Re-seed (same value as Step 1) so any contact delta is NOT
                # attributable to Gumbel-noise drift between the two calls.
                if residual_seed is not None:
                    torch.manual_seed(int(residual_seed))
                all_ids_after = _generate_residual_tokens(
                    res_transformer,
                    motion_ids=base_ids_after,
                    text=text,
                    m_lens_tok=m_lens_tok,
                    int_kv=int_kv,
                    int_pad=int_pad,
                    res_cond_scale=res_cond_scale,
                )
                all_for_decode_after = torch.where(
                    all_ids_after < 0, torch.zeros_like(all_ids_after), all_ids_after,
                )
        else:
            all_ids_after = logits.argmax(dim=-1)                               # (1, S, Q)
            all_ids_after = torch.where(
                token_mask[:, :, None],
                all_ids_after,
                torch.zeros_like(all_ids_after),
            )
            base_ids_after = all_ids_after[..., 0]
            token_change_count = int(
                ((base_ids_after != base_for_res) & token_mask).sum().item(),
            )
            rvq_token_change_count = int(
                ((all_ids_after != all_for_decode) & token_mask[:, :, None]).sum().item(),
            )
            all_for_decode_after = all_ids_after

        motion = vq_model.forward_decoder(all_for_decode_after)
        motion = motion.squeeze(0) * motion_std + motion_mean

    return (
        motion.detach().cpu().numpy().astype(np.float32),
        base_ids_after.squeeze(0).detach().cpu().numpy(),
        {
            "loss_initial": loss_initial,
            "loss_final": loss_final,
            "loss_trace": loss_trace,
            "steps_taken": len(loss_trace),
            "base_token_change_count": token_change_count,
            "rvq_token_change_count": rvq_token_change_count,
            "base_token_total": int(S),
            "rvq_token_total": int(S * all_for_decode.shape[-1]),
            "guidance_layers": guidance_layers,
            "residual_seed": residual_seed,
            "no_residual_rerun": bool(no_residual_rerun),
            "per_step": per_step_info,
        },
    )
