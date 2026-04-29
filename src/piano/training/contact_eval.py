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
contact metrics. Early Stage B used ``mean_min_dist_per_frame`` as the
headline; v15 additionally reports strict GT-part/object-local alignment
metrics so ``best_contact.pt`` can be selected by
``alignment_contact_score`` when the visual failure is wrong part/patch.
The trainer hooks this into ``_run_validation`` so that
``best_contact.pt`` is saved alongside the existing ``best_val.pt``.

Cost: ~1 sec per clip on bf16 / 2× A6000 (10-step base + 10-step
residual MaskGIT decoding + VQ decode). For 20 clips × every val
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
from piano.data.pseudo_labels.extract_contact import (
    ContactConfig,
    _kinematic_contact_score,
)
from piano.utils.smpl_utils import BODY_PART_NAMES
from piano.training.decoded_contact_loss import (
    _object_motion_speed_from_canonical,
    body_canonical_to_object_local_torch,
)


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


def _object_motion_speed(
    object_positions: np.ndarray,
    object_rotations: np.ndarray | None,
    cfg: ContactConfig,
) -> np.ndarray:
    """Match the object-speed proxy used by `_kinematic_contact_score`."""
    T = len(object_positions)
    trans_vel = np.zeros(T, dtype=np.float32)
    if T > 1:
        trans_vel[1:] = (
            np.linalg.norm(np.diff(object_positions, axis=0), axis=-1) * cfg.fps
        )

    ang_vel = np.zeros(T, dtype=np.float32)
    if object_rotations is not None and T > 1:
        ang_vel[1:] = (
            np.linalg.norm(np.diff(object_rotations, axis=0), axis=-1) * cfg.fps
        )

    return trans_vel + float(cfg.kin_radius_proxy) * ang_vel


def _mean_finite(values: list[float | None]) -> float | None:
    xs = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not xs:
        return None
    return float(np.mean(xs))


# ============================================================================
# Per-clip contact compute (numpy-side; takes already-generated motion)
# ============================================================================

def compute_clip_contact_distance(
    motion_263_generated: np.ndarray,   # (T_gen, 263) generated, denormalized
    R_y_angle: float,                   # source-clip canonical→world rotation
    T_xz: np.ndarray,                   # (2,) source-clip canonical→world translation
    object_pc_local: np.ndarray,        # (N_pc, 3)
    object_positions: np.ndarray,       # (T_src, 3) world frame
    object_rotations: np.ndarray,       # (T_src, 3) axis-angle, world frame
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

    Frame-count alignment
    ---------------------

    MoMask VQ-VAE has total temporal stride 4 (down_t=2, stride_t=2).
    For source ``seq_len`` not divisible by 4, the generated motion has
    ``(seq_len // 4) * 4`` frames — up to 3 fewer than ``seq_len``. We
    truncate to ``T = min(seq_len, T_gen)`` so the body-trajectory and
    object-trajectory tensors broadcast cleanly. Any source frames
    beyond ``T_gen`` simply aren't evaluated (the model didn't generate
    them).
    """
    T = min(int(seq_len), int(motion_263_generated.shape[0]))

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


def compute_clip_contact_and_temporal_metrics(
    motion_263_generated: np.ndarray,
    R_y_angle: float,
    T_xz: np.ndarray,
    object_pc_local: np.ndarray,
    object_positions: np.ndarray,
    object_rotations: np.ndarray,
    seq_len: int,
    *,
    recover_from_ric_fn: Callable,
    fps: float = 20.0,
    coupling_threshold: float = 0.5,
    moving_speed_threshold: float | None = None,
) -> dict[str, float | None]:
    """Return distance plus moving-object temporal-coupling metrics."""
    T = min(int(seq_len), int(motion_263_generated.shape[0]))
    if T < 1:
        return {
            "mean_min_dist": float("inf"),
            "moving_frame_frac": None,
            "moving_close_frame_frac": None,
            "moving_coupled_frame_frac": None,
            "moving_close_but_uncoupled_frac": None,
            "moving_mean_best_kin_score": None,
        }

    cfg = ContactConfig(fps=float(fps))
    speed_threshold = (
        float(moving_speed_threshold)
        if moving_speed_threshold is not None
        else float(cfg.kin_world_eps)
    )

    motion_t = torch.from_numpy(motion_263_generated[:T]).float().unsqueeze(0)
    canon_gen = (
        recover_from_ric_fn(motion_t, 22)
        .squeeze(0)
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    world_joints_gen = _lift_canonical_to_world(canon_gen, R_y_angle, T_xz)
    body_joints = world_joints_gen[:, BODY_PART_INDICES, :]

    obj_pos = object_positions[:T]
    obj_rot = object_rotations[:T] if object_rotations is not None else None
    d = _per_frame_body_to_object_distance(
        body_joints,
        object_pc_local,
        obj_pos,
        obj_rot if obj_rot is not None else np.zeros((T, 3), dtype=np.float32),
    )
    min_per_frame = d.min(axis=1)
    close_thresholds = np.array(
        [cfg.distance_thresholds[name] for name in BODY_PART_NAMES],
        dtype=np.float32,
    )
    close_any = (d <= close_thresholds[None, :]).any(axis=1)

    kin_scores = np.stack([
        _kinematic_contact_score(body_joints[:, p, :], obj_pos, obj_rot, cfg)
        for p in range(len(BODY_PART_NAMES))
    ], axis=1)
    best_kin = kin_scores.max(axis=1)
    coupled_any = best_kin >= float(coupling_threshold)

    speed = _object_motion_speed(obj_pos, obj_rot, cfg)
    moving = speed >= speed_threshold
    n_moving = int(moving.sum())
    if n_moving > 0:
        moving_close = float(close_any[moving].mean())
        moving_coupled = float(coupled_any[moving].mean())
        moving_close_uncoupled = float((close_any[moving] & ~coupled_any[moving]).mean())
        moving_best_kin = float(best_kin[moving].mean())
    else:
        moving_close = None
        moving_coupled = None
        moving_close_uncoupled = None
        moving_best_kin = None

    return {
        "mean_min_dist": float(min_per_frame.mean()),
        "moving_frame_frac": float(n_moving / T),
        "close_frame_frac": float(close_any.mean()),
        "moving_close_frame_frac": moving_close,
        "moving_coupled_frame_frac": moving_coupled,
        "moving_close_but_uncoupled_frac": moving_close_uncoupled,
        "moving_mean_best_kin_score": moving_best_kin,
    }


def _weighted_mean_or_none_torch(values: Tensor, weights: Tensor) -> float | None:
    weights = weights.to(device=values.device, dtype=values.dtype)
    denom = weights.sum()
    if float(denom.detach().cpu()) <= 1e-6:
        return None
    return float(((values * weights).sum() / denom.clamp(min=1e-6)).detach().cpu())


def _fraction_or_none_torch(mask: Tensor, denom_mask: Tensor) -> float | None:
    denom = int(denom_mask.sum().detach().cpu())
    if denom <= 0:
        return None
    num = int((mask & denom_mask).sum().detach().cpu())
    return float(num) / float(denom)


def compute_clip_contact_alignment_metrics(
    motion_263_generated: np.ndarray,
    contact_state: np.ndarray,
    contact_target_xyz: np.ndarray,
    obj_com_canonical: np.ndarray,
    obj_rot6d_canonical: np.ndarray,
    seq_len: int,
    *,
    recover_from_ric_fn: Callable,
    fps: float = 20.0,
    moving_speed_threshold: float = 0.15,
    kin_radius_proxy: float = 0.3,
    contact_threshold: float = 0.5,
) -> dict[str, float | None]:
    """Strict part/patch alignment metric for in-training contact eval.

    This mirrors ``k_sample_oracle.py``'s alignment score: the generated body
    part named by GT ``contact_state`` must reach that same part's
    ``contact_target_xyz`` in object-local coordinates. It deliberately does
    not minimize over arbitrary body parts or arbitrary object points.
    """
    T = min(
        int(seq_len),
        int(motion_263_generated.shape[0]),
        int(contact_state.shape[0]),
        int(contact_target_xyz.shape[0]),
        int(obj_com_canonical.shape[0]),
        int(obj_rot6d_canonical.shape[0]),
    )
    if T < 1:
        return {
            "alignment_primary_error": None,
            "alignment_target_error": None,
            "alignment_moving_target_error": None,
            "alignment_same_part_recall": None,
            "alignment_moving_same_part_recall": None,
        }

    motion_t = torch.from_numpy(motion_263_generated[:T]).float().unsqueeze(0)
    joints = recover_from_ric_fn(motion_t, 22).float()
    body_idx = torch.as_tensor(BODY_PART_INDICES, dtype=torch.long)
    body = joints.index_select(dim=2, index=body_idx)

    obj_com = torch.from_numpy(obj_com_canonical[:T]).float().unsqueeze(0)
    obj_rot6d = torch.from_numpy(obj_rot6d_canonical[:T]).float().unsqueeze(0)
    body_local = body_canonical_to_object_local_torch(body, obj_com, obj_rot6d)

    target = torch.from_numpy(contact_target_xyz[:T]).float().unsqueeze(0)
    contact = torch.from_numpy(contact_state[:T]).float().unsqueeze(0).clamp(0.0, 1.0)
    contact_binary = contact >= float(contact_threshold)
    frame_mask = torch.arange(T).view(1, T, 1) < int(seq_len)
    valid_part = contact_binary & frame_mask
    weights = contact * valid_part.to(dtype=body_local.dtype)

    pos_dist = torch.linalg.vector_norm(body_local - target, dim=-1)
    thresholds = torch.tensor(
        [ContactConfig(fps=float(fps)).distance_thresholds[name] for name in BODY_PART_NAMES],
        dtype=body_local.dtype,
    ).view(1, 1, -1)
    same_part_hit = pos_dist <= thresholds

    obj_speed = _object_motion_speed_from_canonical(
        obj_com,
        obj_rot6d,
        fps=float(fps),
        radius_proxy=float(kin_radius_proxy),
    )
    moving = obj_speed >= float(moving_speed_threshold)
    moving_valid = valid_part & moving[:, :, None]
    moving_weights = weights * moving[:, :, None].to(dtype=body_local.dtype)

    target_error = _weighted_mean_or_none_torch(pos_dist, weights)
    moving_target_error = _weighted_mean_or_none_torch(pos_dist, moving_weights)
    same_part_recall = _fraction_or_none_torch(same_part_hit, valid_part)
    moving_same_part_recall = _fraction_or_none_torch(same_part_hit, moving_valid)

    valid_frames = torch.any(valid_part, dim=-1)
    moving_valid_frames = torch.any(moving_valid, dim=-1)
    moving_frames = moving & (torch.arange(T).view(1, T) < int(seq_len))

    return {
        "alignment_primary_error": (
            moving_target_error if moving_target_error is not None else target_error
        ),
        "alignment_target_error": target_error,
        "alignment_moving_target_error": moving_target_error,
        "alignment_same_part_recall": same_part_recall,
        "alignment_moving_same_part_recall": moving_same_part_recall,
        "alignment_contact_part_frame_frac": float(valid_frames.float().mean().item()),
        "alignment_moving_contact_part_frame_frac": (
            float(moving_valid_frames.sum().item()) / max(int(moving_frames.sum().item()), 1)
        ),
    }


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

    if hasattr(res_transformer, "generate_with_int"):
        all_ids = res_transformer.generate_with_int(
            motion_ids=base_for_res,
            conds=[text],
            m_lens=m_lens_tok,
            int_kv=int_kv.transpose(0, 1).contiguous(),
            int_padding_mask=int_pad,
            cond_scale=res_cond_scale,
        )
    else:
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
    fps: float = 20.0,
    coupling_threshold: float = 0.5,
    moving_speed_threshold: float | None = None,
    composite_coupling_weight: float = 0.12,
    composite_uncoupled_penalty: float = 0.05,
    composite_min_moving_frame_frac: float = 0.05,
    alignment_recall_penalty: float = 0.25,
    alignment_distance_weight: float = 0.05,
    alignment_coupling_weight: float = 0.0,
) -> Callable[[], dict[str, float]]:
    """Build a no-arg callable that evaluates contact distance on a fixed batch.

    The callable runs in eval mode + ``torch.no_grad`` and returns a dict
    with at least ``mean_min_dist`` (fixed-subset mean of
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
        ``eval_fn() -> {"mean_min_dist": float, "n_clips": int, ...}``.
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
    contact_state_cpu = fixed_val_batch["contact_state"].cpu().numpy().astype(np.float32)
    contact_target_cpu = fixed_val_batch["contact_target_xyz"].cpu().numpy().astype(np.float32)
    obj_com_canon_cpu = fixed_val_batch["obj_com_canonical"].cpu().numpy().astype(np.float32)
    obj_rot6d_canon_cpu = fixed_val_batch["obj_rot6d_canonical"].cpu().numpy().astype(np.float32)
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

        per_clip_metrics: list[dict[str, float | None]] = []
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

            # Compute distance and temporal binding against the source clip's
            # world-frame anchor + object trajectory. Anchors are pre-computed
            # at factory time (fixed across val intervals).
            metrics = compute_clip_contact_and_temporal_metrics(
                motion_263_generated=motion_gen,
                R_y_angle=src_R_y[i],
                T_xz=src_T_xz[i],
                object_pc_local=object_pc_cpu[i],
                object_positions=object_positions_cpu[i],
                object_rotations=object_rotations_cpu[i],
                seq_len=T,
                recover_from_ric_fn=recover_from_ric,
                fps=float(fps),
                coupling_threshold=float(coupling_threshold),
                moving_speed_threshold=moving_speed_threshold,
            )
            metrics.update(compute_clip_contact_alignment_metrics(
                motion_263_generated=motion_gen,
                contact_state=contact_state_cpu[i],
                contact_target_xyz=contact_target_cpu[i],
                obj_com_canonical=obj_com_canon_cpu[i],
                obj_rot6d_canonical=obj_rot6d_canon_cpu[i],
                seq_len=T,
                recover_from_ric_fn=recover_from_ric,
                fps=float(fps),
                moving_speed_threshold=(
                    float(moving_speed_threshold)
                    if moving_speed_threshold is not None
                    else 0.15
                ),
                kin_radius_proxy=float(ContactConfig(fps=float(fps)).kin_radius_proxy),
            ))
            per_clip_metrics.append(metrics)

        if not per_clip_metrics:
            return {"mean_min_dist": float("inf"), "n_clips": 0}

        out: dict[str, float] = {
            "mean_min_dist": float(np.mean([
                float(m["mean_min_dist"]) for m in per_clip_metrics
            ])),
            "n_clips": float(len(per_clip_metrics)),
        }
        for key in (
            "moving_frame_frac",
            "close_frame_frac",
            "moving_close_frame_frac",
            "moving_coupled_frame_frac",
            "moving_close_but_uncoupled_frac",
            "moving_mean_best_kin_score",
            "alignment_primary_error",
            "alignment_target_error",
            "alignment_moving_target_error",
            "alignment_same_part_recall",
            "alignment_moving_same_part_recall",
            "alignment_contact_part_frame_frac",
            "alignment_moving_contact_part_frame_frac",
        ):
            mean = _mean_finite([m.get(key) for m in per_clip_metrics])
            if mean is not None:
                out[key] = float(mean)

        moving_frac = out.get("moving_frame_frac", 0.0)
        coupled = out.get("moving_coupled_frame_frac", None)
        uncoupled = out.get("moving_close_but_uncoupled_frac", 0.0)
        if coupled is not None and moving_frac >= float(composite_min_moving_frame_frac):
            out["composite_contact_score"] = (
                out["mean_min_dist"]
                + float(composite_coupling_weight) * (1.0 - float(coupled))
                + float(composite_uncoupled_penalty) * float(uncoupled)
            )
        else:
            out["composite_contact_score"] = out["mean_min_dist"]
        primary = out.get("alignment_primary_error")
        if primary is not None and np.isfinite(float(primary)):
            recall = out.get("alignment_moving_same_part_recall")
            if recall is None or not np.isfinite(float(recall)):
                recall = out.get("alignment_same_part_recall", 0.0)
            coupling_term = (
                1.0 - float(coupled)
                if coupled is not None and np.isfinite(float(coupled))
                else 0.0
            )
            out["alignment_contact_score"] = (
                float(primary)
                + float(alignment_recall_penalty) * (1.0 - float(recall or 0.0))
                + float(alignment_distance_weight) * float(out["mean_min_dist"])
                + float(alignment_coupling_weight) * coupling_term
            )
        else:
            out["alignment_contact_score"] = out["composite_contact_score"]
        return out

    return eval_fn
