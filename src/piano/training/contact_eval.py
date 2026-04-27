"""Contact-distance eval for Stage B during training.

PIANO Stage B's training objective is masked-CE on base RVQ tokens; the
ship metric is geometric body-to-object distance (B0 metric in
``analyses/2026-04-27_v0_5_premise_review.md``). Per
``analyses/2026-04-28_v0_3_delta_retrain_and_v0_5_contact.md``, the
two metrics are empirically anti-correlated on isolated ablations
(v0.4 → v0.5 same-arch +epoch: CE↓ contact↑; v0.5 → v0.6 only γ_kind:
CE↑ contact↓), so ``best_val.pt`` selected by val_loss is structurally
the wrong checkpoint for the ship metric.

This module provides a callable that, given a fixed mini-batch of val
clips and the trained generator's components, returns a dict of
contact metrics (the headline being ``mean_min_dist_per_frame``,
averaged across the mini-batch). The trainer hooks this into
``_run_validation`` so that ``best_contact.pt`` is saved alongside the
existing ``best_val.pt``.

Cost: ~1 sec per clip on bf16 / 2× A6000 (10-step base + 10-step
residual MaskGIT decoding + VQ decode). For 5 clips × every val
interval (default 5 epochs) over an 80-epoch run = ~80 sec total
overhead, vs ~13 min per run. Negligible.

The compute follows the same logic as the offline diagnostic scripts:
- ``scripts/stage_b_generator/qual_eval.py::_generate`` for generation.
- ``scripts/stage_b_generator/measure_contact_distance.py`` for the
  per-frame body-to-object min distance over a sampled object PC.

The two scripts hold the load-bearing copies; this module duplicates a
small subset of helpers for in-process eval. They will be consolidated
once the contact-aware checkpointing has shipped.

References
----------
- ``analyses/2026-04-28_v0_3_delta_retrain_and_v0_5_contact.md`` — B1
  motivation + decision rule.
- ``analyses/2026-04-27_final_synthesis.md`` §"Phase B B1" — design.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor

from piano.utils.canonical_frame import (
    axis_angle_to_matrix_np,
    get_canonicalize_transform_from_clip,
    y_rotation_matrix,
)
from piano.utils.smpl_utils import BODY_PART_INDICES


# ============================================================================
# Geometric helpers (mirror measure_contact_distance.py — load-bearing copy)
# ============================================================================

def _lift_canonical_to_world(
    joints_canon: np.ndarray,   # (T, 22, 3)
    R_y_angle: float,
    T_xz: np.ndarray,           # (2,)
) -> np.ndarray:
    """Apply ``world = R_y(angle) @ canonical + [T_xz[0], 0, T_xz[1]]``."""
    R = y_rotation_matrix(float(R_y_angle))
    rotated = joints_canon @ R.T
    rotated[..., 0] += float(T_xz[0])
    rotated[..., 2] += float(T_xz[1])
    return rotated.astype(np.float32)


def _world_object_pc_per_frame(
    object_pc_local: np.ndarray,    # (N_pc, 3)
    object_positions: np.ndarray,   # (T, 3)
    object_rotations: np.ndarray,   # (T, 3) axis-angle
) -> np.ndarray:
    """Lift the object PC to per-frame world position + orientation."""
    R_obj = axis_angle_to_matrix_np(object_rotations.astype(np.float32))   # (T, 3, 3)
    pc_world = np.einsum(
        "tij,nj->tni",
        R_obj,
        object_pc_local.astype(np.float32),
    )                                                                       # (T, N_pc, 3)
    pc_world += object_positions[:, None, :].astype(np.float32)
    return pc_world


def _per_frame_body_to_object_distance(
    body_joints_world: np.ndarray,    # (T, n_parts, 3)
    object_pc_local: np.ndarray,      # (N_pc, 3)
    object_positions: np.ndarray,     # (T, 3)
    object_rotations: np.ndarray,     # (T, 3) axis-angle
) -> np.ndarray:
    """Return ``(T, n_parts)`` min distance per body part per frame."""
    pc_world = _world_object_pc_per_frame(
        object_pc_local, object_positions, object_rotations,
    )
    diff = body_joints_world[:, :, None, :] - pc_world[:, None, :, :]
    d = np.linalg.norm(diff, axis=-1)
    return d.min(axis=-1)


# ============================================================================
# Per-clip contact compute (numpy-side; takes already-generated motion)
# ============================================================================

def compute_clip_contact_distance(
    motion_263_generated: np.ndarray,   # (T, 263) generated, denormalized
    R_y_angle: float,                   # source-clip canonical→world rotation
    T_xz: np.ndarray,                   # (2,) source-clip canonical→world translation
    object_pc_local: np.ndarray,        # (N_pc, 3)
    object_positions: np.ndarray,       # (T, 3) world frame
    object_rotations: np.ndarray,       # (T, 3) axis-angle, world frame
    seq_len: int,
    *,
    recover_from_ric_fn: Callable,
) -> float:
    """Lift generated motion to source's world frame, return mean_min_dist_per_frame.

    Mirrors ``measure_contact_distance.py``'s per-clip logic but takes
    the generated motion array directly (no .npz round-trip). The
    ``(R_y_angle, T_xz)`` transform is the source clip's canonical→world
    anchor — caller pre-computes it once via
    :func:`piano.utils.canonical_frame.get_canonicalize_transform_from_clip`
    on the source ``joints_world`` + canonical-joints derived from the
    source ``motion_263``. We re-use that transform on the GENERATED
    motion's canonical joints so the generated body is anchored in the
    same world frame as the source's object trajectory.
    """
    T = int(seq_len)

    motion_t = torch.from_numpy(motion_263_generated[:T]).float().unsqueeze(0)
    canon_gen = recover_from_ric_fn(motion_t, 22).squeeze(0).cpu().numpy().astype(np.float32)

    world_joints_gen = _lift_canonical_to_world(canon_gen, R_y_angle, T_xz)
    body_joints = world_joints_gen[:, BODY_PART_INDICES, :]                  # (T, 5, 3)

    d = _per_frame_body_to_object_distance(
        body_joints,
        object_pc_local,
        object_positions[:T],
        object_rotations[:T],
    )                                                                        # (T, 5)
    min_per_frame = d.min(axis=1)                                            # (T,)
    return float(min_per_frame.mean())


# ============================================================================
# Generation (bf16-friendly, single-clip)
# ============================================================================

@torch.no_grad()
def _generate_full_condition(
    transformer: torch.nn.Module,
    vq_model: torch.nn.Module,
    res_transformer: torch.nn.Module,
    text: str,
    int_kv: Tensor,                # (S_int, 1, d)
    int_pad: Tensor | None,        # (1, S_int) or None
    m_lens_tok: Tensor,            # (1,)
    *,
    motion_mean: Tensor,           # (263,) on device
    motion_std: Tensor,
    w_text: float = 4.0,
    w_int: float = 2.0,
    timesteps: int = 10,
    res_cond_scale: float = 2.0,
    device: torch.device,
) -> np.ndarray:
    """Generate motion for one clip in the ``full`` (text + z_int) condition.

    Returns ``(T, 263)`` numpy array in **denormalized** HumanML3D scale
    (matching what ``preprocess_interact.py`` saves), so downstream
    ``recover_from_ric`` recovers joints in real-world units.
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
    )                                              # (1, S_max), -1 at padded
    base_for_res = torch.where(base_ids < 0, torch.zeros_like(base_ids), base_ids)

    all_ids = res_transformer.generate(
        motion_ids=base_for_res,
        conds=[text],
        m_lens=m_lens_tok,
        cond_scale=res_cond_scale,
    )
    all_for_decode = torch.where(all_ids < 0, torch.zeros_like(all_ids), all_ids)

    motion = vq_model.forward_decoder(all_for_decode)   # (1, T, 263)
    motion = motion.squeeze(0)                          # (T, 263), still on device
    motion = motion * motion_std + motion_mean          # denormalize
    return motion.detach().cpu().numpy().astype(np.float32)


# ============================================================================
# Public factory: build_contact_eval_fn
# ============================================================================

def build_contact_eval_fn(
    transformer: torch.nn.Module,
    vq_model: torch.nn.Module,
    res_transformer: torch.nn.Module,
    fixed_val_batch: dict[str, Any],
    *,
    motion_mean: Tensor,
    motion_std: Tensor,
    device: torch.device,
    token_stride: int = 4,
    w_text: float = 4.0,
    w_int: float = 2.0,
) -> Callable[[], dict[str, float]]:
    """Build a no-arg callable that evaluates contact distance on a fixed batch.

    The callable runs in eval mode + ``torch.no_grad`` and returns a dict
    with at least ``mean_min_dist`` (5-clip mean of
    ``mean_min_dist_per_frame``). The trainer uses
    ``mean_min_dist`` as the best-checkpoint key.

    Parameters
    ----------
    transformer
        The wrapped :class:`InteractionMaskTransformer` (already on
        ``device``). The caller is responsible for passing
        ``accelerator.unwrap_model(transformer)`` so we don't go through
        DDP wrappers when generating.
    vq_model, res_transformer
        Frozen MoMask components. Generated tokens go base + residual
        + decode + denormalize.
    fixed_val_batch
        Output of ``collate_hoi`` on a fixed list of N val clips. Must
        contain ``motion`` (T, 263), ``joints`` (T, 22, 3), ``seq_len``,
        ``text`` (list[str]), ``object_pc``, ``object_positions``,
        ``object_rotations``, ``contact_state``, ``contact_target_xyz``,
        ``phase``, ``support``, ``obj_com_canonical``,
        ``obj_rot6d_canonical``. (HOIDataset's standard output keys.)
    motion_mean, motion_std
        ``(263,)`` tensors on ``device`` — same ones used in the train
        step's VQ encode (for the denormalize after decode).
    token_stride
        VQ-VAE temporal downsample (4 for MoMask).
    w_text, w_int
        CFG strengths to evaluate at. Defaults match qual_eval.py
        defaults so during-training contact metric matches offline B0.

    Returns
    -------
    Callable[[], dict[str, float]]
        ``eval_fn() -> {"mean_min_dist": float, "n_clips": int}``.
    """
    # Lazy import to avoid forcing MoMask sys.path setup at module import
    # (the trainer module is loaded by Stage A predictor too, which
    # doesn't need MoMask).
    import piano.models.backbones.momask_adapter  # noqa: F401
    from utils.motion_process import recover_from_ric

    # Pre-extract per-clip CPU numpy arrays from the fixed batch. These
    # don't change across val intervals.
    motion_src_cpu = fixed_val_batch["motion"].cpu().numpy().astype(np.float32)        # (N, T, 263) source motion (raw HumanML3D scale)
    joints_src_cpu = fixed_val_batch["joints"].cpu().numpy().astype(np.float32)        # (N, T, 22, 3)
    seq_lens_cpu = fixed_val_batch["seq_len"].cpu().numpy().astype(np.int64)            # (N,)
    object_pc_cpu = fixed_val_batch["object_pc"].cpu().numpy().astype(np.float32)      # (N, N_pc, 3)
    object_positions_cpu = fixed_val_batch["object_positions"].cpu().numpy().astype(np.float32)
    object_rotations_cpu = fixed_val_batch["object_rotations"].cpu().numpy().astype(np.float32)
    texts: list[str] = list(fixed_val_batch["text"])
    n_clips = len(texts)

    # Pre-compute the source clip canonical→world transforms ONCE.
    # These depend only on the val clip's source motion + joints (both
    # fixed for this run), so re-computing them every val interval would
    # be wasted work. Each clip uses its OWN transform (anchors the
    # generated motion to the source's world frame for object alignment).
    src_R_y: list[float] = []
    src_T_xz: list[np.ndarray] = []
    for i in range(n_clips):
        T_i = int(seq_lens_cpu[i])
        if T_i < 1:
            src_R_y.append(0.0)
            src_T_xz.append(np.zeros(2, dtype=np.float32))
            continue
        motion_src_t = torch.from_numpy(motion_src_cpu[i, :T_i]).float().unsqueeze(0)
        canon_src = recover_from_ric(motion_src_t, 22).squeeze(0).cpu().numpy().astype(np.float32)
        R_y, T_xz = get_canonicalize_transform_from_clip(
            joints_src_cpu[i, :T_i], canon_src,
        )
        src_R_y.append(float(R_y))
        src_T_xz.append(np.asarray(T_xz, dtype=np.float32))

    def eval_fn() -> dict[str, float]:
        transformer.eval()
        # vq_model + res_transformer are pre-frozen; .eval() is idempotent.
        vq_model.eval()
        res_transformer.eval()

        per_clip_dists: list[float] = []
        for i in range(n_clips):
            T = int(seq_lens_cpu[i])
            if T < token_stride:    # need at least one VQ token
                continue

            # Build single-clip z_int K/V on device.
            with torch.no_grad():
                int_kv, int_pad = transformer.interaction_tokenizer(
                    contact_state=fixed_val_batch["contact_state"][i:i+1].to(device).float(),
                    contact_target_xyz=fixed_val_batch["contact_target_xyz"][i:i+1].to(device).float(),
                    phase=fixed_val_batch["phase"][i:i+1].to(device).long(),
                    support=fixed_val_batch["support"][i:i+1].to(device).long(),
                    obj_com_canonical=fixed_val_batch["obj_com_canonical"][i:i+1].to(device).float(),
                    obj_rot6d_canonical=fixed_val_batch["obj_rot6d_canonical"][i:i+1].to(device).float(),
                    seq_lens=fixed_val_batch["seq_len"][i:i+1].to(device).long(),
                )
            m_lens_tok = (fixed_val_batch["seq_len"][i:i+1].to(device).long() // token_stride).clamp(min=1)

            motion_gen = _generate_full_condition(
                transformer, vq_model, res_transformer,
                text=texts[i],
                int_kv=int_kv, int_pad=int_pad,
                m_lens_tok=m_lens_tok,
                motion_mean=motion_mean, motion_std=motion_std,
                w_text=w_text, w_int=w_int,
                device=device,
            )                                                              # (T_dec, 263)

            # Compute body-to-object distance against source clip's
            # world-frame anchor + object trajectory. Anchors are
            # pre-computed at factory time (fixed across val intervals).
            d = compute_clip_contact_distance(
                motion_263_generated=motion_gen,
                R_y_angle=src_R_y[i],
                T_xz=src_T_xz[i],
                object_pc_local=object_pc_cpu[i],
                object_positions=object_positions_cpu[i],
                object_rotations=object_rotations_cpu[i],
                seq_len=T,
                recover_from_ric_fn=recover_from_ric,
            )
            per_clip_dists.append(d)

        if not per_clip_dists:
            return {"mean_min_dist": float("inf"), "n_clips": 0}

        return {
            "mean_min_dist": float(np.mean(per_clip_dists)),
            "n_clips": float(len(per_clip_dists)),
        }

    return eval_fn
