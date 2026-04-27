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
# Loss: masked L2 against contact_target_xyz_gt (lifted to world) under contact_state
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
    """
    # Lazy MoMask import (the inference module is also imported by
    # CPU-only test paths).
    import piano.models.backbones.momask_adapter  # noqa: F401
    from utils.motion_process import recover_from_ric

    transformer.eval()
    vq_model.eval()
    res_transformer.eval()

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
        all_ids = res_transformer.generate(
            motion_ids=base_for_res,
            conds=[text],
            m_lens=m_lens_tok,
            cond_scale=res_cond_scale,
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
             "base_token_change_count": 0},
        )

    # Pre-compute the world-frame target ONCE (constant across the
    # optimization loop). Per the data-loader docstring, the
    # `contact_target_xyz_gt` field is in OBJECT-LOCAL frame; the body
    # we compare against is in WORLD frame after the canonical→world
    # lift. Without this rigid transform the loss compares two frames
    # whose alignment depends on where the object sits in the world
    # — which produces wildly variable gradients across clips and was
    # the bug behind the +8 cm regression on guidance30 (4/5 clips
    # got worse on canonical val set).
    target_world_np = _lift_target_to_world_np(
        target_local=np.asarray(contact_target_xyz_local, dtype=np.float32),
        object_positions=np.asarray(object_positions, dtype=np.float32),
        object_rotations=np.asarray(object_rotations, dtype=np.float32),
    )                                                                          # (T, 5, 3)
    target_world_t = torch.from_numpy(target_world_np).to(device).float().unsqueeze(0)   # (1, T, 5, 3)
    contact_state_t = torch.from_numpy(contact_state_np).to(device).float().unsqueeze(0) # (1, T, 5)

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
    logits = F.one_hot(base_for_res.long(), num_classes=V).float() * init_logit_scale
    logits = logits.detach().clone().requires_grad_(True)                     # (1, S, V)

    # Detached residual indices (Q-1 layers).
    residual_ids = all_for_decode[..., 1:].detach()                           # (1, S, Q-1)

    # --- Step 3: optimization loop ---
    optimizer = torch.optim.AdamW(
        [logits], lr=guidance_lr, betas=(0.5, 0.9), weight_decay=1e-6,
    )
    body_idx_t = torch.tensor(list(body_part_indices), device=device, dtype=torch.long)

    loss_initial: float = float("nan")
    loss_trace: list[float] = []
    for step in range(num_guidance_steps):
        motion_norm = _decode_relaxed_base(
            base_logits=logits,
            residual_ids=residual_ids,
            vq_model=vq_model,
            temperature=guidance_temperature,
        )                                                                      # (1, T, 263)
        motion = motion_norm * motion_std + motion_mean
        joints_canon = recover_from_ric(motion, 22)                            # (1, T, 22, 3)
        joints_world = _lift_canonical_to_world_torch(joints_canon, R_y_angle, T_xz)
        body_world = joints_world.index_select(2, body_idx_t)                  # (1, T, 5, 3)

        # Truncate target / mask to actual decoded T (residual decode
        # may be slightly different from the seq_lens-derived T; align
        # by taking the min).
        T_dec = body_world.shape[1]
        T_tgt = target_world_t.shape[1]
        T_use = min(T_dec, T_tgt)

        loss = _masked_contact_l2(
            body_world=body_world[:, :T_use],
            target_world=target_world_t[:, :T_use],
            contact_state=contact_state_t[:, :T_use],
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

    # --- Step 4: argmax + re-run residual on possibly-changed base ---
    with torch.no_grad():
        base_ids_after = logits.argmax(dim=-1)                                 # (1, S)
        token_change_count = int(
            (base_ids_after != base_for_res).sum().item(),
        )
        all_ids_after = res_transformer.generate(
            motion_ids=base_ids_after,
            conds=[text],
            m_lens=m_lens_tok,
            cond_scale=res_cond_scale,
        )
        all_for_decode_after = torch.where(
            all_ids_after < 0, torch.zeros_like(all_ids_after), all_ids_after,
        )
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
            "base_token_total": int(S),
        },
    )
