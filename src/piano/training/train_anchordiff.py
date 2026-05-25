"""Stage B (new): Train PIANO-AnchorDiff.

Anchor-conditioned continuous motion diffusion. Replaces the closed
MoMask Stage B track. Trained with GT ``z_int`` + classifier-free
guidance dropout; inference uses Stage A v10 predicted ``z_int``.

Design source of truth:
    analyses/2026-05-08_piano_anchordiff_design.md

Usage:
    accelerate launch --config_file configs/accelerate_config.yaml \\
        -m piano.training.train_anchordiff \\
        --config configs/training/anchordiff_v1.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from omegaconf import OmegaConf
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader, Subset

from piano.data.dataset import (
    AugmentConfig,
    HOIDataset,
    build_subject_split,
    collate_hoi,
    extract_subject_id,
)
from piano.data.stage1_coarse_oracle import (
    extract_coarse_v1_batched,
    load_stage1_coarse_norm,
)
from piano.models.motion_anchordiff import (
    AnchorDenoiserConfig,
    AnchorDiffConfig,
    DiffusionConfig,
    MotionAnchorDiff,
    ZIntDims,
    pack_z_int,
)
from piano.models.object_encoder import ObjectEncoder
from piano.training.anchor_consistency_loss import (
    AnchorConsistencyConfig,
    PART_TO_JOINT,
    anchor_consistency_loss_world_joints,
    lift_object_local_to_world,
)
from piano.training.feature_groups import FEATURE_GROUPS
from piano.training.feature_weight_state import FeatureWeightState
from piano.training.anchordiff_geometric_losses import (
    MotionGeometricLossConfig,
    compute_motion_geometric_losses,
    feature_velocity_loss,
)
from piano.training.trainer import (
    build_optimizer_with_decay_groups,
    build_scheduler,
    run_training_loop,
)
from piano.utils.clip_utils import encode_text_per_token, load_clip_text_encoder
from piano.utils.io_utils import load_json


# ---------------------------------------------------------------------------
# Dataset assembly (subject-split path, mirrors train_predictor.py)
# ---------------------------------------------------------------------------


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


def _resolve_subject_split(cfg, bucket: str) -> set | None:
    subj_cfg = cfg.data.get("subject_split")
    if subj_cfg is None or not subj_cfg.get("enabled", False):
        return None
    keys = sorted({
        (subset, extract_subject_id(subset, m.get("seq_id", "")))
        for subset, m in _read_metadata(cfg.data.datasets)
        if extract_subject_id(subset, m.get("seq_id", "")) is not None
    })
    splits = build_subject_split(
        keys,
        train_pct=subj_cfg.train_pct,
        val_pct=subj_cfg.val_pct,
        seed=subj_cfg.seed,
    )
    if bucket == "all":
        return None
    return splits[bucket]


def _build_dataset(cfg, bucket: str = "train", augment: bool = True) -> ConcatDataset:
    subj_filter = _resolve_subject_split(cfg, bucket)

    aug_cfg = cfg.data.get("augmentation", None)
    augment_obj = None
    if augment and aug_cfg is not None and aug_cfg.get("enabled", False):
        augment_obj = AugmentConfig(
            enabled=True,
            mirror_prob=float(aug_cfg.get("mirror_prob", 0.0)),
            mirror_duplicate=bool(aug_cfg.get("mirror_duplicate", False)),
            rotate_around_y_prob=float(aug_cfg.get("rotate_around_y_prob", 0.0)),
            pc_jitter_std=float(aug_cfg.get("pc_jitter_std", 0.0)),
            timewarp_scales=tuple(
                float(s) for s in aug_cfg.get("timewarp_scales", [])
            ),
            timewarp_mode=str(aug_cfg.get("timewarp_mode", "online")),
        )

    pseudo_label_subdir = cfg.data.get("pseudo_label_subdir", None)
    pseudo_label_dir = cfg.data.get("pseudo_label_dir", None)

    datasets = []
    for entry in cfg.data.datasets:
        if pseudo_label_dir is not None:
            sub_dir = pseudo_label_dir
        elif pseudo_label_subdir is not None:
            sub_dir = str(Path(entry.root) / pseudo_label_subdir)
        else:
            sub_dir = None
        ds = HOIDataset(
            root=entry.root,
            pseudo_label_dir=sub_dir,
            max_seq_length=cfg.data.max_seq_length,
            subject_id_filter=subj_filter,
            subsample_n_per_object=cfg.data.get("subsample_n_per_object", None),
            subsample_seed=int(cfg.data.get("subsample_seed", 42)),
            augment=augment_obj,
            support_collapse_hand_support=bool(
                cfg.data.get("support_collapse_hand_support", True)
            ),
            surface_obj_pose=True,
            force_world_frame=bool(cfg.data.get("force_world_frame", False)),
            motion_representation=str(
                cfg.data.get("motion_representation", "smpl_pose_135_plan")
            ),
            use_oracle_interaction_hint=bool(
                cfg.data.get("use_oracle_interaction_hint", False)
            ),
            oracle_hint_variant=str(
                cfg.data.get("oracle_hint_variant", "full")
            ),
            oracle_hint_fps=float(cfg.data.get("oracle_hint_fps", 20.0)),
            surface_temporal_aux_fields=bool(
                cfg.data.get("surface_temporal_aux_fields", False)
            ),
            use_body_action_hint=bool(
                cfg.data.get("use_body_action_hint", False)
            ),
            body_action_hint_mask_mode=str(
                cfg.data.get("body_action_hint_mask_mode", "all_on")
            ),
            body_action_energy_threshold=float(
                cfg.data.get("body_action_energy_threshold", 0.05)
            ),
        )
        datasets.append(ds)
    return ConcatDataset(datasets)


def _load_state_dict_compatible(
    module: torch.nn.Module,
    incoming: dict[str, Tensor],
) -> tuple[list[str], list[str]]:
    """Load matching checkpoint tensors and partially copy resized tensors.

    Used by v4b to warm-start from v4a while expanding the per-frame object
    trajectory channel from 9 dims to 24 dims. Exact-shape parameters load
    normally; same-rank shape mismatches copy the overlapping slice and keep
    the freshly initialized values for new rows/columns.
    """
    current = module.state_dict()
    loadable: dict[str, Tensor] = {}
    partial: list[str] = []
    skipped: list[str] = []

    for name, src in incoming.items():
        if name not in current:
            skipped.append(name)
            continue
        dst = current[name]
        if dst.shape == src.shape:
            loadable[name] = src
            continue
        if dst.ndim == src.ndim and dst.ndim > 0:
            merged = dst.clone()
            slices = tuple(slice(0, min(a, b)) for a, b in zip(dst.shape, src.shape))
            merged[slices] = src.to(dtype=dst.dtype)[slices]
            loadable[name] = merged
            partial.append(name)
            continue
        skipped.append(name)

    module.load_state_dict(loadable, strict=False)
    return partial, skipped


# ---------------------------------------------------------------------------
# Step function
# ---------------------------------------------------------------------------


def _phase_to_softmax(phase_idx: Tensor, num_classes: int) -> Tensor:
    """One-hot phase ids → soft 'softmax' representation we feed into z_int.

    Stage A produces softmax; GT during training is an integer index.
    We one-hot it so the input shape is consistent with predicted-z_int
    inference. (Identity at training time, soft at inference time.)
    """
    onehot = F.one_hot(phase_idx.clamp_min(0).long(), num_classes=num_classes)
    return onehot.float()


def _support_to_softmax(support_idx: Tensor, num_classes: int) -> Tensor:
    return _phase_to_softmax(support_idx, num_classes)


def build_anchordiff_step_fn(
    model: MotionAnchorDiff,
    object_encoder: ObjectEncoder,
    clip_model: torch.nn.Module,
    anchor_cfg: AnchorConsistencyConfig,
    z_dims: ZIntDims,
    device: torch.device,
    feature_weight_state: FeatureWeightState | None,
    geometric_cfg: MotionGeometricLossConfig,
    motion_representation: str = "smpl_pose_135_plan",
    world_joint_velocity_weight: float = 0.0,
    object_traj_dim: int = 9,
    fk_consistency_weight: float = 0.0,
    pos_loss_weight: float = 0.0,
    # Round-25 D5: per-joint weighting of the dense FK position loss.
    # Default 1.0 = back-compat (uniform 22-joint MSE). Set > 1 to
    # emphasize wrist (joints 20/21) and ankle (joints 10/11)
    # endpoints. Used to test H2 (loss imbalance) — see
    # analyses/2026-05-23_round25_diagnostic_bundle_design.md §D5.
    hand_endpoint_weight: float = 1.0,
    foot_endpoint_weight: float = 1.0,
    # Round-26: sparse fine-limb supervision at active contact/anchor parts.
    # Unlike dense FK L_pos, this is not averaged over all joints and all
    # frames; it directly matches the D2/D3 endpoint-error diagnostic.
    anchor_joint_pos_weight: float = 0.0,
    anchor_joint_vel_weight: float = 0.0,
    anchor_joint_part_weights: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0),
    use_interaction_plan: bool = False,
    plan_anchor_weight: float = 0.0,
    plan_segment_weight: float = 0.0,
    plan_transition_vel_weight: float = 0.0,
    plan_transition_acc_weight: float = 0.0,
    plan_transition_window: int = 3,
    stable_root_vel_weight: float = 0.0,
    stable_root_acc_weight: float = 0.0,
    stable_local_vel_weight: float = 0.0,
    stable_local_acc_weight: float = 0.0,
    stable_local_vel_cm_weight: float = 0.0,
    stable_local_acc_cm_weight: float = 0.0,
    stable_local_speed_moment_weight: float = 0.0,
    stable_support_erode: int = 4,
    # ── Min-SNR-γ weighting (Hang et al. arXiv:2303.09556, ICCV 2023) ──────
    # Reweights the diffusion mse_x0 loss by min(SNR(t), γ) per sample,
    # where SNR(t) = ᾱ_t / (1 - ᾱ_t). For x_0-prediction this is the
    # standard form (paper Table 1, official code). For ε-pred the form is
    # min(SNR, γ)/SNR; for v-pred it is min(SNR, γ)/(SNR+1).
    # The weight is normalized per-batch so mean(weight) = 1, preserving the
    # absolute scale of mse_x0 (so other loss weights don't need re-tuning).
    # Only the diffusion mse_x0 term is weighted; auxiliary losses (anchor,
    # FK pos, plan, stable, fix_v) are untouched per spec §4.3.
    use_min_snr_weighting: bool = False,
    min_snr_gamma: float = 5.0,
    # Round-22: Stage-1 Coarse-v1 (23-D route) oracle condition. When
    # ``stage1_coarse_dim > 0``: step_fn extracts Coarse-v1 from each batch's
    # ``motion`` + ``rest_offsets`` via ``extract_coarse_v1_batched``, z-scores
    # it with the Stage-1 train norm stats, and attaches it to
    # ``cond["stage1_coarse"]``. The Stage-2 denoiser's V12InputProjection
    # consumes it via a zero-init projection. See
    # ``analyses/2026-05-22_stage2_condition_reframe_and_next_plan.md`` §6.
    stage1_coarse_dim: int = 0,
    stage1_coarse_norm_mean: np.ndarray | None = None,
    stage1_coarse_norm_std: np.ndarray | None = None,
    stage1_coarse_noise_std: float = 0.0,
    # Round-27 Tier-0B: temporal interaction loss config + per-term weights
    # (per src/piano/training/temporal_interaction_losses.py + roadmap §7).
    # All zero by default → no behaviour change for v27 / earlier configs.
    # Requires data.surface_temporal_aux_fields=true so the trainer can
    # read walking_mask + foot_stance_gt from each batch.
    temporal_loss_cfg: object | None = None,
    fps: float = 20.0,
):
    """Build the AnchorDiff step_fn closure.

    The step_fn reads `feature_weight_state.current` every batch via
    `to_per_frame_tensor` (cheap; lazily refreshed when state has changed).
    Dynamic-update mode: an external epoch hook calls
    `feature_weight_state.update(...)` after every K epochs; the next
    batch's `cache.get()` automatically picks up the new weights.
    """

    # Round-22: pre-convert Stage-1 Coarse-v1 norm stats to device tensors
    # once at step_fn construction (avoids per-step host→device copies). When
    # the branch is disabled (stage1_coarse_dim == 0) these stay None.
    if stage1_coarse_dim > 0:
        if stage1_coarse_norm_mean is None or stage1_coarse_norm_std is None:
            raise ValueError(
                "stage1_coarse_dim > 0 requires stage1_coarse_norm_mean + std "
                "(load via piano.data.stage1_coarse_oracle.load_stage1_coarse_norm)."
            )
        if stage1_coarse_norm_mean.shape != (stage1_coarse_dim,):
            raise ValueError(
                f"stage1_coarse_norm_mean shape {stage1_coarse_norm_mean.shape} "
                f"!= ({stage1_coarse_dim},)"
            )
        stage1_coarse_mean_t = torch.from_numpy(stage1_coarse_norm_mean).to(device).float()
        stage1_coarse_std_t = torch.from_numpy(stage1_coarse_norm_std).to(device).float()
    else:
        stage1_coarse_mean_t = None
        stage1_coarse_std_t = None
    if len(anchor_joint_part_weights) != len(PART_TO_JOINT):
        raise ValueError(
            "anchor_joint_part_weights must have 5 entries "
            "(left_hand, right_hand, left_foot, right_foot, pelvis)"
        )
    anchor_joint_part_weights_t = torch.tensor(
        anchor_joint_part_weights, device=device, dtype=torch.float32,
    ).view(1, 1, -1)
    anchor_joint_part_to_joint_t = torch.tensor(
        PART_TO_JOINT, device=device, dtype=torch.long,
    )

    # DDP compatibility: when `model` has been wrapped by
    # `accelerator.prepare(model)` under multi-process training it becomes
    # a `DistributedDataParallel` whose `__getattr__` does NOT forward
    # arbitrary attribute lookups to the underlying module. Save a
    # reference to the underlying module for read-only access to
    # `model.diffusion.{alphas_cumprod, prediction_target}` below. Forward
    # training passes still go through the wrapped `model(...)` to keep
    # DDP gradient sync intact.
    _unwrapped_model_for_diff = model.module if hasattr(model, "module") else model

    class _WeightCache:
        def __init__(self) -> None:
            self._t: Tensor | None = None
            self._ver = -1
            if feature_weight_state is not None:
                self._t = feature_weight_state.to_per_frame_tensor(device)
                self._ver = feature_weight_state.last_update_epoch

        def get(self, motion_dim: int) -> Tensor:
            if feature_weight_state is None:
                if self._t is None or self._t.shape[-1] != motion_dim:
                    self._t = torch.ones(
                        1, 1, motion_dim, device=device, dtype=torch.float32,
                    )
                return self._t
            if feature_weight_state.last_update_epoch != self._ver:
                self._t = feature_weight_state.to_per_frame_tensor(device)
                self._ver = feature_weight_state.last_update_epoch
            if self._t is None or self._t.shape[-1] != motion_dim:
                raise ValueError(
                    "FeatureWeightState is only valid for motion_263; "
                    f"got weight_dim={None if self._t is None else self._t.shape[-1]} "
                    f"for motion_dim={motion_dim}"
                )
            return self._t

    cache = _WeightCache()

    def _build_object_traj(
        obj_com: Tensor,
        obj_rot6d: Tensor,
        contact_target_xyz: Tensor,
        obj_pos_world: Tensor,
        obj_rot_world: Tensor,
    ) -> Tensor:
        # COM (3) + rot6d (6) = 9 base dims. v4b+ models (object_traj_dim
        # >= 24) append 5 body-part anchor world targets (5x3=15).
        components = [obj_com, obj_rot6d]
        if object_traj_dim >= 24:
            target_world = lift_object_local_to_world(
                contact_target_xyz,
                obj_pos_world,
                obj_rot_world,
            ).reshape(obj_com.shape[0], obj_com.shape[1], -1)
            components.append(target_world)
        out = torch.cat(components, dim=-1)
        if out.shape[-1] != object_traj_dim:
            raise ValueError(
                f"object_traj_dim={object_traj_dim} but built {out.shape[-1]} dims"
            )
        return out

    def step_fn(_model, batch: dict, global_step: int = 0) -> dict[str, Tensor]:
        motion = batch["motion"].to(device)                       # (B, T, D_motion)
        joints = batch["joints"].to(device)                       # (B, T, 22, 3)
        object_pc = batch["object_pc"].to(device)
        contact_state = batch["contact_state"].to(device)         # (B, T, 5)
        contact_target_xyz = batch["contact_target_xyz"].to(device)  # (B, T, 5, 3)
        phase = batch["phase"].to(device)                         # (B, T)
        support = batch["support"].to(device)                     # (B, T)
        obj_com = batch["obj_com_canonical"].to(device)           # (B, T, 3)
        obj_rot6d = batch["obj_rot6d_canonical"].to(device)       # (B, T, 6)
        obj_pos_world = batch["object_positions"].to(device)      # (B, T, 3)
        obj_rot_world = batch["object_rotations"].to(device)      # (B, T, 3) axis-angle
        seq_len = batch["seq_len"].to(device)                     # (B,)

        B, T, motion_dim = motion.shape
        seq_idx = torch.arange(T, device=device).unsqueeze(0)
        seq_mask = (seq_idx < seq_len.unsqueeze(1)).float()        # (B, T)

        # Round-28 cleanup: only smpl_pose_135_plan is supported. The
        # legacy per-clip canonical->world transform (motion_263 path) and
        # the v8 keyframed contact-zeroing were removed; downstream code
        # treats R_y / T_xz_canon / T_y_canon as None.
        R_y = T_xz_canon = T_y_canon = None
        contact_target_xyz_for_z = contact_target_xyz

        # --- Pack z_int (training: GT contact + GT target + GT phase/support) ---
        phase_soft = _phase_to_softmax(phase, z_dims.phase_classes)
        support_soft = _support_to_softmax(support, z_dims.support_classes)

        # Round-28 cleanup: all active configs zero the z_int channels.
        # Hard-coded since the Stage-1.5 plan supersedes per-frame z_int.
        contact_state_for_z = torch.zeros_like(contact_state)
        contact_target_xyz_for_z = torch.zeros_like(contact_target_xyz_for_z)
        phase_soft_for_z = torch.zeros_like(phase_soft)
        support_soft_for_z = torch.zeros_like(support_soft)
        z_int = pack_z_int(
            contact_state_for_z, contact_target_xyz_for_z,
            phase_soft_for_z, support_soft_for_z, z_dims,
        )                                                          # (B, T, total)

        # --- Object trajectory channel. v1-v4a use object pose only
        # (3 COM + 6 rot6d). v4b appends the five body-part anchor targets
        # already transformed into world frame (5 * 3), so Stage A's
        # predictor signal reaches the denoiser in task-space coordinates.
        # v8 keyframed: appends 6-keyjoint positions only at keyframe
        # frames + 1-D keyframe indicator (zero elsewhere).
        object_traj = _build_object_traj(
            obj_com=obj_com,
            obj_rot6d=obj_rot6d,
            contact_target_xyz=contact_target_xyz_for_z,
            obj_pos_world=obj_pos_world,
            obj_rot_world=obj_rot_world,
        )

        # --- Init pose: SMPL-22 frame 0 ---
        init_pose = joints[:, 0, :, :].reshape(B, -1)              # (B, 66)

        # --- Text features via CLIP per-token ---
        text_features, _text_mask = encode_text_per_token(
            clip_model, batch["text"], device,
        )                                                          # (B, L, text_dim)

        # --- Object tokens via PointNet++ encoder ---
        obj_tokens = object_encoder(object_pc)                     # (B, N, obj_dim)

        # --- Stage B PLAN_ONLY condition mode (per
        # Round-28 cleanup: PLAN_ONLY condition mode is the only mode now.
        # Zero z_int and the contact-target portion of object_traj (first 9
        # dims are object pose; last 15 are 5 lifted contact targets).
        # Interaction plan tokens carry contact / part / phase / support info.
        z_int = torch.zeros_like(z_int)
        if object_traj.shape[-1] >= 24:
            object_traj = object_traj.clone()
            object_traj[..., 9:] = 0.0

        cond = {
            "z_int": z_int,
            "object_world_traj": object_traj,
            "init_pose": init_pose,
            "text": text_features.float(),
            "object_tokens": obj_tokens,
        }

        # ── Round-22: Stage-1 Coarse-v1 oracle condition ──
        # Extract 23-D Coarse-v1 from GT motion_135 + rest_offsets, z-score
        # with Stage-1 train stats, attach to cond. The denoiser's
        # V12InputProjection.stage1_coarse_proj (zero-init) consumes it.
        # See analyses/2026-05-22_stage2_condition_reframe_and_next_plan.md.
        if stage1_coarse_dim > 0:
            rest_offsets_for_coarse = batch["rest_offsets"].to(device).float()  # (B, 22, 3)
            coarse_v1_raw = extract_coarse_v1_batched(
                motion=motion, rest_offsets=rest_offsets_for_coarse,
            )                                                                    # (B, T, 23)
            if coarse_v1_raw.shape[-1] != stage1_coarse_dim:
                raise ValueError(
                    f"Oracle Coarse-v1 dim {coarse_v1_raw.shape[-1]} != "
                    f"stage1_coarse_dim={stage1_coarse_dim} from config."
                )
            coarse_v1_norm = (coarse_v1_raw - stage1_coarse_mean_t) / stage1_coarse_std_t
            if _model.training and stage1_coarse_noise_std > 0.0:
                coarse_v1_norm = coarse_v1_norm + (
                    torch.randn_like(coarse_v1_norm) * float(stage1_coarse_noise_std)
                )
            cond["stage1_coarse"] = coarse_v1_norm

        # ── Round-27 Tier-0A: oracle interaction hint condition ──
        # Per roadmap §6.11 / §6.12. The dataset builds the (T, D) hint
        # tensor in __getitem__ when data.use_oracle_interaction_hint=true;
        # the denoiser's oracle_hint_proj (zero-init) consumes it via a
        # direct addition into the per-frame motion-token embedding.
        if "oracle_interaction_hint" in batch:
            cond["oracle_interaction_hint"] = (
                batch["oracle_interaction_hint"].to(device).float()
            )
        # ── Round-28: body-action oracle hint condition (24D) ──
        # When data.use_body_action_hint=true, the dataset surfaces a
        # per-frame 24D channel; the denoiser's body_action_hint_proj
        # consumes it through a separate branch (gated / per-layer
        # adapter / input-add depending on injection_mode).
        if "body_action_hint" in batch:
            cond["body_action_hint"] = (
                batch["body_action_hint"].to(device).float()
            )

        # --- v10 InteractionPlan: thread the compiled plan through cond ---
        # The dataset compiles the plan in __getitem__ for the
        # smpl_pose_135_plan motion_representation; we just collect the
        # plan_* tensors back into a single dict the denoiser's
        # InteractionPlanEncoder expects.
        if use_interaction_plan:
            plan_keys = [
                "anchor_time", "anchor_part", "anchor_target_local",
                "anchor_target_world", "anchor_type", "anchor_phase",
                "anchor_support", "anchor_conf", "anchor_mask",
                "segment_start", "segment_end", "segment_part",
                "segment_target_summary_local", "segment_phase",
                "segment_support", "segment_conf", "segment_mask",
            ]
            cond["interaction_plan"] = {
                k: batch[f"plan_{k}"].to(device) for k in plan_keys
            }

        # --- Diffusion training step (x₀-prediction or v-prediction) ---
        # NOTE: call via __call__ (not .training_step directly) so DDP can
        # intercept and reset its per-iteration reducer. MotionAnchorDiff
        # .forward delegates to .training_step.
        out = model(motion, cond)
        x0_pred = out["x0_pred"]
        x0_target = out["x0_target"]
        diff_pred = out["diff_pred"]
        diff_target = out["diff_target"]

        # Diffusion MSE — masked to valid frames. FEATURE-WEIGHTED via the
        # FeatureWeightState (static or dynamic). See feature_groups.py.
        # Under x_0-pred: MSE(x0_pred, x0_target). Under v-pred:
        # MSE(v_pred, v_target). Both target the same parameterisation
        # the network natively predicts.
        mse_per_dim = (diff_pred - diff_target).pow(2)              # (B, T, D)
        weighted = (mse_per_dim * cache.get(motion_dim)).sum(-1)    # (B, T)

        # ── Min-SNR-γ per-sample weighting on diffusion mse ────────────
        # Hang et al. arXiv:2303.09556, x_0-pred form: w_b = min(SNR_{t_b}, γ).
        # Normalized per-batch so mean(w) = 1, preserving overall mse scale
        # → other-loss weights are unaffected, only the timestep balance is.
        min_snr_weight_mean = torch.zeros((), device=device, dtype=weighted.dtype)
        min_snr_weight_max = torch.zeros((), device=device, dtype=weighted.dtype)
        min_snr_weight_min = torch.zeros((), device=device, dtype=weighted.dtype)
        if use_min_snr_weighting and "t" in out:
            t_b = out["t"]                                                 # (B,) long
            alpha_bar = _unwrapped_model_for_diff.diffusion.alphas_cumprod.gather(0, t_b)  # (B,)
            snr = alpha_bar / (1.0 - alpha_bar + 1e-8)                     # (B,)
            snr_clamped = torch.clamp_max(snr, float(min_snr_gamma))       # (B,)
            pred_target = _unwrapped_model_for_diff.diffusion.prediction_target
            if pred_target == "x0":
                w_b = snr_clamped                                          # (B,)
            elif pred_target == "v":
                w_b = snr_clamped / (snr + 1.0)
            else:                                                          # "eps"
                w_b = snr_clamped / (snr + 1e-8)
            min_snr_weight_mean = w_b.mean().detach()
            min_snr_weight_max = w_b.max().detach()
            min_snr_weight_min = w_b.min().detach()
            # Normalize: mean across batch becomes 1.
            w_b_norm = w_b / w_b.mean().clamp_min(1e-8)                    # (B,)
            weighted = weighted * w_b_norm.view(-1, 1)                     # (B, T)

        mse = (weighted * seq_mask).sum() / seq_mask.sum().clamp_min(1.0)
        mse_unweighted = (
            mse_per_dim.sum(-1) * seq_mask
        ).sum() / seq_mask.sum().clamp_min(1.0)
        mse_main = mse
        mse_kf = torch.zeros((), device=device, dtype=mse.dtype)

        # Anchor consistency in WORLD frame (smpl_pose_135_plan path).
        # 135-D output: first 132 = global rot_6d, last 3 = root world xyz.
        # jpos derived by FK from rot_6d + per-clip rest_offsets + root_world.
        from piano.training.smpl_kinematics import (
            rotation_6d_to_matrix as _rot6d_to_mat,
            fk_from_global_rotations as _fk_from_global,
        )
        rot_6d = x0_pred[..., :132].view(B, T, 22, 6).float()
        root_world_pred = x0_pred[..., 132:135].float()                # (B, T, 3)
        rot_mat_global = _rot6d_to_mat(rot_6d)                         # (B, T, 22, 3, 3)
        rest_offsets = batch["rest_offsets"].to(device).float()        # (B, 22, 3)
        rest_per_frame = rest_offsets.unsqueeze(1).expand(B, T, 22, 3)
        jpos_pred = _fk_from_global(
            rot_mat_global, rest_per_frame, root_world_pred,
        )                                                              # (B, T, 22, 3)

        anchor = anchor_consistency_loss_world_joints(
            joints_world_pred=jpos_pred,
            contact_state_gt=contact_state,
            contact_target_xyz_local=contact_target_xyz,
            object_positions=obj_pos_world,
            object_rotations=obj_rot_world,
            cfg=anchor_cfg,
            seq_mask=seq_mask.bool(),
            stability_mask=None,
        )

        geom = compute_motion_geometric_losses(
            x0_pred=x0_pred,
            x0_target=x0_target,
            seq_mask=seq_mask,
            cfg=geometric_cfg,
        )
        # World-frame velocity loss on the full 135-D motion vector.
        if world_joint_velocity_weight > 0.0:
            wv_pred, wv_target = x0_pred.float(), x0_target.float()
            world_vel = feature_velocity_loss(wv_pred, wv_target, seq_mask.float())
            geom = {
                **geom,
                "loss_geometric": geom["loss_geometric"]
                + world_joint_velocity_weight * world_vel,
                "loss_vel": world_vel,
            }

        # FK consistency removed in Round-28 cleanup (was v5-only).
        loss_fk = torch.zeros((), device=device, dtype=x0_pred.dtype)

        # Full-body L_pos: MSE between FK-derived predicted joints and GT
        # joints_22, all 22 joints x all valid frames. Dense temporal
        # supervision (MDM Eq. 3, Tevet et al. ICLR 2023).
        loss_pos_full = torch.zeros((), device=device, dtype=x0_pred.dtype)
        loss_pos_full_obs_monitor = torch.zeros((), device=device, dtype=x0_pred.dtype)
        loss_pos_full_unobs_monitor = torch.zeros((), device=device, dtype=x0_pred.dtype)
        if pos_loss_weight > 0.0:
            joints_gt = joints.float()                                    # (B, T, 22, 3)
            err = (jpos_pred.float() - joints_gt).pow(2).sum(-1)          # (B, T, 22)
            # Round-25 D5: per-joint weighting for hand+foot endpoints.
            if hand_endpoint_weight != 1.0 or foot_endpoint_weight != 1.0:
                jw = torch.ones(22, device=err.device, dtype=err.dtype)
                jw[20] = hand_endpoint_weight
                jw[21] = hand_endpoint_weight
                jw[10] = foot_endpoint_weight
                jw[11] = foot_endpoint_weight
                err = err * jw                                            # (B, T, 22)
                weight_sum = jw.sum()
            else:
                weight_sum = err.new_tensor(22.0)
            denom = (seq_mask.sum() * weight_sum).clamp_min(1.0)
            loss_pos_full = (err * seq_mask.unsqueeze(-1).float()).sum() / denom

        # Sparse active-part endpoint supervision (Round-26 strategy fix).
        # Dense FK L_pos averages over all joints and frames, which lets
        # hands/feet stay poor while the global loss looks acceptable. This
        # term uses the same 5 body parts as the interaction/contact labels
        # and supervises only active contact/anchor slots against GT joints.
        loss_anchor_joint_pos = torch.zeros((), device=device, dtype=motion.dtype)
        loss_anchor_joint_vel = torch.zeros((), device=device, dtype=motion.dtype)
        anchor_joint_active_ratio = torch.zeros((), device=device)
        if anchor_joint_pos_weight > 0.0 or anchor_joint_vel_weight > 0.0:
            pred_part = jpos_pred.float().index_select(
                2, anchor_joint_part_to_joint_t,
            )                                                           # (B, T, P, 3)
            gt_part = joints.float().index_select(
                2, anchor_joint_part_to_joint_t,
            )
            active_part = (
                (contact_state >= anchor_cfg.contact_threshold)
                & seq_mask.bool().unsqueeze(-1)
            )                                                           # (B, T, P)
            active_f = active_part.float()
            part_w = anchor_joint_part_weights_t.to(
                device=device, dtype=pred_part.dtype,
            )
            weighted_active = active_f * part_w
            anchor_joint_active_ratio = (
                active_f.sum()
                / (seq_mask.float().sum() * len(PART_TO_JOINT)).clamp_min(1.0)
            )
            if anchor_joint_pos_weight > 0.0:
                err_p = (pred_part - gt_part).pow(2).sum(-1)             # (B, T, P)
                loss_anchor_joint_pos = (
                    (err_p * weighted_active).sum()
                    / weighted_active.sum().clamp_min(1.0)
                )
            if anchor_joint_vel_weight > 0.0 and T >= 2:
                vel_pred_part = pred_part[:, 1:] - pred_part[:, :-1]
                vel_gt_part = gt_part[:, 1:] - gt_part[:, :-1]
                active_pair = active_part[:, 1:] & active_part[:, :-1]
                weighted_pair = active_pair.float() * part_w
                err_vp = (vel_pred_part - vel_gt_part).pow(2).sum(-1)
                loss_anchor_joint_vel = (
                    (err_vp * weighted_pair).sum()
                    / weighted_pair.sum().clamp_min(1.0)
                )

        # --- Round-27 Tier-0B / v28 temporal interaction losses ---
        # Per src/piano/training/temporal_interaction_losses.py and
        # piano_stage2_full_architecture_roadmap.md §7. Five losses:
        #   - contact_rel_offset (object-local SmoothL1)
        #   - contact_drift      (segment-level drift in object-local)
        #   - contact_tracking   (asymmetric projection along obj disp)
        #   - gait_both_airborne (pred ankle height vs sample floor)
        #   - gait_stance_vel    (stance foot xz velocity penalty)
        # All zero by default → no behaviour change for v27 / earlier.
        loss_contact_rel = torch.zeros((), device=device, dtype=motion.dtype)
        loss_contact_drift = torch.zeros((), device=device, dtype=motion.dtype)
        loss_contact_track = torch.zeros((), device=device, dtype=motion.dtype)
        loss_gait_air = torch.zeros((), device=device, dtype=motion.dtype)
        loss_gait_stance_vel = torch.zeros((), device=device, dtype=motion.dtype)
        # Round-28 consistency losses (prompt §7.3 / §7.4).
        loss_hint_contact_cons = torch.zeros((), device=device, dtype=motion.dtype)
        loss_body_action_cons = torch.zeros((), device=device, dtype=motion.dtype)
        if temporal_loss_cfg is not None and (
            float(getattr(temporal_loss_cfg, "contact_rel_offset_weight", 0.0)) > 0.0
            or float(getattr(temporal_loss_cfg, "contact_drift_weight", 0.0)) > 0.0
            or float(getattr(temporal_loss_cfg, "contact_tracking_weight", 0.0)) > 0.0
            or float(getattr(temporal_loss_cfg, "gait_both_airborne_weight", 0.0)) > 0.0
            or float(getattr(temporal_loss_cfg, "gait_stance_velocity_weight", 0.0)) > 0.0
            or float(getattr(temporal_loss_cfg, "hint_contact_consistency_weight", 0.0)) > 0.0
            or float(getattr(temporal_loss_cfg, "body_action_consistency_weight", 0.0)) > 0.0
        ):
            from piano.training.temporal_interaction_losses import (
                loss_body_action_consistency,
                loss_contact_drift_smoothl1,
                loss_contact_rel_offset_smoothl1,
                loss_contact_tracking_projection,
                loss_gait_both_airborne,
                loss_gait_stance_velocity,
                loss_hint_contact_consistency,
            )

            # Aux walking/foot_stance fields are required only for the
            # T0-B gait losses; the consistency losses don't need them.
            gait_terms_active = (
                float(getattr(temporal_loss_cfg, "gait_both_airborne_weight", 0.0)) > 0.0
                or float(getattr(temporal_loss_cfg, "gait_stance_velocity_weight", 0.0)) > 0.0
            )
            if gait_terms_active and (
                "walking_mask" not in batch or "foot_stance_gt" not in batch
            ):
                raise KeyError(
                    "gait temporal losses are enabled but the batch is "
                    "missing walking_mask / foot_stance_gt — set "
                    "data.surface_temporal_aux_fields=true in the config."
                )
            walking_mask_b = (
                batch["walking_mask"].to(device).float()                       # (B, T, 1)
                if "walking_mask" in batch else None
            )
            foot_stance_b = (
                batch["foot_stance_gt"].to(device).float()                     # (B, T, 2)
                if "foot_stance_gt" in batch else None
            )

            jpf = jpos_pred.float()
            jgf = joints.float()
            cs_f = contact_state.float()
            op_f = obj_pos_world.float()
            or_f = obj_rot_world.float()
            sm_f = seq_mask.float()

            if float(temporal_loss_cfg.contact_rel_offset_weight) > 0.0:
                loss_contact_rel = loss_contact_rel_offset_smoothl1(
                    pred_joints=jpf, gt_joints=jgf,
                    object_positions=op_f, object_rotations=or_f,
                    contact_state=cs_f, cfg=temporal_loss_cfg, seq_mask=sm_f,
                )
            if float(temporal_loss_cfg.contact_drift_weight) > 0.0:
                loss_contact_drift = loss_contact_drift_smoothl1(
                    pred_joints=jpf, gt_joints=jgf,
                    object_positions=op_f, object_rotations=or_f,
                    contact_state=cs_f, cfg=temporal_loss_cfg, seq_mask=sm_f,
                )
            if float(temporal_loss_cfg.contact_tracking_weight) > 0.0:
                loss_contact_track = loss_contact_tracking_projection(
                    pred_joints=jpf, gt_joints=jgf,
                    object_positions=op_f, object_rotations=or_f,
                    contact_state=cs_f, cfg=temporal_loss_cfg, seq_mask=sm_f,
                )
            if float(temporal_loss_cfg.gait_both_airborne_weight) > 0.0:
                loss_gait_air = loss_gait_both_airborne(
                    pred_joints=jpf, gt_joints=jgf,
                    walking_mask=walking_mask_b, cfg=temporal_loss_cfg,
                    seq_mask=sm_f,
                )
            if float(temporal_loss_cfg.gait_stance_velocity_weight) > 0.0:
                loss_gait_stance_vel = loss_gait_stance_velocity(
                    pred_joints=jpf, foot_stance_gt=foot_stance_b,
                    walking_mask=walking_mask_b, fps=float(fps),
                    seq_mask=sm_f,
                )
            # Round-28 §7.3 — hint-contact consistency. Pull pred wrist
            # toward the oracle's hand_object_local_offset only on
            # contact frames. Small weight (start at 0.25-0.5).
            if (
                float(getattr(temporal_loss_cfg, "hint_contact_consistency_weight", 0.0)) > 0.0
                and "oracle_interaction_hint" in batch
            ):
                loss_hint_contact_cons = loss_hint_contact_consistency(
                    pred_joints=jpf,
                    oracle_interaction_hint=batch["oracle_interaction_hint"]
                    .to(device).float(),
                    object_positions=op_f, object_rotations=or_f,
                    contact_state=cs_f, cfg=temporal_loss_cfg,
                    seq_mask=sm_f,
                )
            elif float(getattr(temporal_loss_cfg, "hint_contact_consistency_weight", 0.0)) > 0.0:
                raise KeyError(
                    "hint_contact_consistency_weight > 0 but batch is missing "
                    "oracle_interaction_hint; set data.use_oracle_interaction_hint=true."
                )
            # Round-28 §7.4 — body-action consistency. Pull pred
            # six-joint deltas toward the oracle body_action_hint,
            # masked by the hint's joint mask.
            if (
                float(getattr(temporal_loss_cfg, "body_action_consistency_weight", 0.0)) > 0.0
                and "body_action_hint" in batch
            ):
                loss_body_action_cons = loss_body_action_consistency(
                    pred_joints=jpf,
                    body_action_hint=batch["body_action_hint"]
                    .to(device).float(),
                    seq_mask=sm_f,
                )
            elif float(getattr(temporal_loss_cfg, "body_action_consistency_weight", 0.0)) > 0.0:
                raise KeyError(
                    "body_action_consistency_weight > 0 but batch is missing "
                    "body_action_hint; set data.use_body_action_hint=true."
                )

        # --- v10 plan-aware losses ---
        # Active only when the trainer was built with use_interaction_plan
        # and the rep is smpl_pose_135_plan. Three sub-losses (per
        # piano_interaction_plan_pipeline_reframe_for_claude_code.md §7.2):
        # 1. plan_anchor: FK joint at anchor time × active part vs
        #    anchor_target_world. Sparse — only fires at anchor frames
        #    where parts are active.
        # 2. plan_transition_vel/acc: ±W-frame window around each anchor,
        #    MSE on world joint velocity / acceleration vs GT. Pulls the
        #    near-anchor motion to GT dynamics so the unobserved frames
        #    don't pop at boundaries.
        # 3. plan_segment: optional segment-realization loss; default 0
        #    weight to avoid recreating dense conditioning.
        loss_plan_anchor = torch.zeros((), device=device, dtype=motion.dtype)
        loss_plan_segment = torch.zeros((), device=device, dtype=motion.dtype)
        loss_plan_trans_vel = torch.zeros((), device=device, dtype=motion.dtype)
        loss_plan_trans_acc = torch.zeros((), device=device, dtype=motion.dtype)
        if use_interaction_plan and (
            plan_anchor_weight > 0.0
            or plan_segment_weight > 0.0
            or plan_transition_vel_weight > 0.0
            or plan_transition_acc_weight > 0.0
        ):
            # Body-part → SMPL-22 joint index map (matches
            # piano.utils.smpl_utils.INTERACTION_BODY_PARTS).
            part_to_joint = torch.tensor(
                [20, 21, 10, 11, 0], dtype=torch.long, device=device,
            )
            anchor_time = batch["plan_anchor_time"].to(device).long()       # (B, K)
            anchor_part = batch["plan_anchor_part"].to(device).float()      # (B, K, P)
            anchor_target_world = (
                batch["plan_anchor_target_world"].to(device).float()
            )                                                                 # (B, K, P, 3)
            anchor_mask = batch["plan_anchor_mask"].to(device).bool()        # (B, K)
            B_p, K_p, P_p = anchor_part.shape

            # Plan anchor realization loss --------------------------------
            if plan_anchor_weight > 0.0:
                # FK joints at active anchor positions per part. We gather
                # by anchor_time across the time dim, then by joint index
                # per part. Active mask = anchor_mask AND parts_active.
                # All ops are vectorised; works on B<=8 K=12 P=5.
                t_idx = anchor_time.clamp(0, T - 1).view(B_p, K_p, 1, 1).expand(B_p, K_p, 22, 3)
                fk_at_anchor = torch.gather(jpos_pred, 1, t_idx)             # (B, K, 22, 3)
                joint_at_part = fk_at_anchor[:, :, part_to_joint, :]         # (B, K, P, 3)
                err = (joint_at_part - anchor_target_world).pow(2).sum(-1)   # (B, K, P)
                # Active = anchor valid AND part active
                act = anchor_mask.unsqueeze(-1).float() * anchor_part        # (B, K, P)
                num_active = act.sum().clamp_min(1.0)
                loss_plan_anchor = (err * act).sum() / num_active

            # Transition window losses -----------------------------------
            if (
                plan_transition_vel_weight > 0.0
                or plan_transition_acc_weight > 0.0
            ):
                W = int(plan_transition_window)
                # window_mask[b, t] = True if t is within ±W of any valid
                # anchor frame in clip b.
                t_grid = torch.arange(T, device=device).view(1, 1, T)
                anchor_t_view = anchor_time.view(B_p, K_p, 1)                # (B, K, 1)
                amask = anchor_mask.view(B_p, K_p, 1)
                near = (
                    (anchor_t_view - W <= t_grid)
                    & (t_grid <= anchor_t_view + W)
                    & amask
                )                                                             # (B, K, T) bool
                window_mask = near.any(dim=1) & seq_mask.bool()              # (B, T)

                joints_gt_w = joints.float()
                if plan_transition_vel_weight > 0.0:
                    vel_pred = jpos_pred[:, 1:] - jpos_pred[:, :-1]            # (B, T-1, 22, 3)
                    vel_gt = joints_gt_w[:, 1:] - joints_gt_w[:, :-1]
                    vw = window_mask[:, 1:] & window_mask[:, :-1]              # (B, T-1)
                    err_v = (vel_pred - vel_gt).pow(2).sum(-1).mean(-1)        # (B, T-1)
                    denom_v = vw.float().sum().clamp_min(1.0)
                    loss_plan_trans_vel = (err_v * vw.float()).sum() / denom_v

                if plan_transition_acc_weight > 0.0:
                    if jpos_pred.shape[1] >= 3:
                        vel_pred = jpos_pred[:, 1:] - jpos_pred[:, :-1]
                        vel_gt = joints_gt_w[:, 1:] - joints_gt_w[:, :-1]
                        acc_pred = vel_pred[:, 1:] - vel_pred[:, :-1]
                        acc_gt = vel_gt[:, 1:] - vel_gt[:, :-1]
                        aw = (
                            window_mask[:, 2:]
                            & window_mask[:, 1:-1]
                            & window_mask[:, :-2]
                        )                                                       # (B, T-2)
                        err_a = (acc_pred - acc_gt).pow(2).sum(-1).mean(-1)     # (B, T-2)
                        denom_a = aw.float().sum().clamp_min(1.0)
                        loss_plan_trans_acc = (err_a * aw.float()).sum() / denom_a

            # Segment consistency loss (optional) ------------------------
            if plan_segment_weight > 0.0:
                seg_start = batch["plan_segment_start"].to(device).long()     # (B, S)
                seg_end = batch["plan_segment_end"].to(device).long()
                seg_mask = batch["plan_segment_mask"].to(device).bool()
                seg_part = batch["plan_segment_part"].to(device).float()      # (B, S, P)
                seg_target = (
                    batch["plan_segment_target_summary_local"].to(device).float()
                )                                                              # (B, S, P, 3)
                # Lift segment summary local→world via the per-frame object
                # pose at the segment midpoint. Simple, avoids re-deriving
                # full per-frame world targets here.
                S_p = seg_start.shape[1]
                mid = ((seg_start + seg_end) // 2).clamp(0, T - 1)              # (B, S)
                from piano.training.anchor_consistency_loss import (
                    lift_object_local_to_world as _lift,
                )
                # Build per-segment object pose at midpoint (B, S, 3)
                obj_pos_seg = torch.gather(
                    obj_pos_world, 1, mid.unsqueeze(-1).expand(-1, -1, 3),
                )
                obj_rot_seg = torch.gather(
                    obj_rot_world, 1, mid.unsqueeze(-1).expand(-1, -1, 3),
                )
                seg_target_world = _lift(
                    seg_target, obj_pos_seg, obj_rot_seg,
                )                                                              # (B, S, P, 3)
                # Mean FK joint position over the segment, per active part
                t_grid_s = torch.arange(T, device=device).view(1, 1, T)
                in_seg = (
                    (t_grid_s >= seg_start.unsqueeze(-1))
                    & (t_grid_s <= seg_end.unsqueeze(-1))
                    & seg_mask.unsqueeze(-1)
                )                                                              # (B, S, T)
                in_seg_f = in_seg.float()
                # jpos at part joints: (B, T, P, 3)
                jpos_at_parts = jpos_pred[:, :, part_to_joint, :]              # (B, T, P, 3)
                # mean over t-in-segment per (B, S, P, 3)
                num_t = in_seg_f.sum(dim=-1).clamp_min(1.0).unsqueeze(-1).unsqueeze(-1)
                jpos_mean = (
                    in_seg_f.unsqueeze(-1).unsqueeze(-1)
                    * jpos_at_parts.unsqueeze(1)
                ).sum(dim=2) / num_t                                            # (B, S, P, 3)
                err_s = (jpos_mean - seg_target_world).pow(2).sum(-1)           # (B, S, P)
                act_s = seg_mask.unsqueeze(-1).float() * seg_part               # (B, S, P)
                denom_s = act_s.sum().clamp_min(1.0)
                loss_plan_segment = (err_s * act_s).sum() / denom_s

        # --- Stable-support root stability loss (per
        # claude_code_v11_planonly_stability_next_steps.md §A).
        # Active on stable-support frames only (pseudo-label support != 0
        # OR pelvis contact > 0.5), eroded by ``stable_support_erode`` to
        # exclude approach/contact/release transitions. Penalises root
        # velocity / acceleration RMS toward GT — fixes the learned root
        # drift that v10/v11 jitter diagnostic identified (predicted root
        # vel rms ×1.4 GT on stable-support segments). Do NOT raise the
        # global velocity weight — that risks frozen-body collapse.
        loss_stable_root_vel = torch.zeros((), device=device, dtype=motion.dtype)
        loss_stable_root_acc = torch.zeros((), device=device, dtype=motion.dtype)
        loss_stable_local_vel = torch.zeros((), device=device, dtype=motion.dtype)
        loss_stable_local_acc = torch.zeros((), device=device, dtype=motion.dtype)
        # cm-scale local-dynamics losses (per
        # claude_code_v11_next_localdyn_target_routing.md §A.2-A.4):
        # the prior m²-space MSE contributed ~10⁻⁵ of total → ineffective.
        # cm-space SmoothL1 + cm-space speed-moment loss give gradient
        # signals at usable magnitudes for shipping weights ≤ 0.1.
        loss_stable_local_vel_cm = torch.zeros((), device=device, dtype=motion.dtype)
        loss_stable_local_acc_cm = torch.zeros((), device=device, dtype=motion.dtype)
        loss_stable_local_speed_moment = torch.zeros((), device=device, dtype=motion.dtype)
        stable_support_frame_ratio = torch.zeros((), device=device)
        stable_root_vel_rms_pred = torch.zeros((), device=device)
        stable_root_vel_rms_gt = torch.zeros((), device=device)
        stable_root_acc_rms_pred = torch.zeros((), device=device)
        stable_root_acc_rms_gt = torch.zeros((), device=device)
        stable_local_vel_rms_pred = torch.zeros((), device=device)
        stable_local_vel_rms_gt = torch.zeros((), device=device)
        stable_local_acc_rms_pred = torch.zeros((), device=device)
        stable_local_acc_rms_gt = torch.zeros((), device=device)
        stable_local_speed_mean_pred = torch.zeros((), device=device)
        stable_local_speed_mean_gt = torch.zeros((), device=device)
        stable_local_speed_std_pred = torch.zeros((), device=device)
        stable_local_speed_std_gt = torch.zeros((), device=device)
        if (
            stable_root_vel_weight > 0.0
            or stable_root_acc_weight > 0.0
            or stable_local_vel_weight > 0.0
            or stable_local_acc_weight > 0.0
            or stable_local_vel_cm_weight > 0.0
            or stable_local_acc_cm_weight > 0.0
            or stable_local_speed_moment_weight > 0.0
        ):
            joints_gt_s = joints.float()
            # stable_support[b, t] = (support != 0) | (pelvis_contact > 0.5)
            pelvis_contact = contact_state[..., 4] > 0.5  # (B, T)
            support_active = support != 0                  # (B, T)
            stable_raw = (support_active | pelvis_contact) & seq_mask.bool()
            # Erode by half-window each side so transitions are excluded.
            half = max(int(stable_support_erode) // 2, 0)
            stable_mask_t = stable_raw.clone()
            for shift in range(1, half + 1):
                left = torch.roll(stable_raw, shifts=-shift, dims=-1)
                right = torch.roll(stable_raw, shifts=shift, dims=-1)
                if shift > 0:
                    left[..., -shift:] = False
                    right[..., :shift] = False
                stable_mask_t = stable_mask_t & left & right

            stable_support_frame_ratio = (
                stable_mask_t.float().sum() / seq_mask.float().sum().clamp_min(1.0)
            )

            root_pred = jpos_pred[..., 0, :].float()    # (B, T, 3)
            root_gt = joints_gt_s[..., 0, :]            # (B, T, 3)
            vel_pred = root_pred[:, 1:] - root_pred[:, :-1]
            vel_gt = root_gt[:, 1:] - root_gt[:, :-1]
            vel_mask = stable_mask_t[:, 1:] & stable_mask_t[:, :-1]   # (B, T-1)

            # Root-aligned local positions (per
            # claude_code_v11_after_full_frozen_fix_handoff.md §A.2):
            # subtract root from every joint so root drift cancels and
            # the resulting velocity / acceleration loss supervises
            # body-relative dynamics — restoring this is the direct
            # frozen-body fix, separate from condition routing.
            jpos_pred_f = jpos_pred.float()                                # (B, T, 22, 3)
            local_pred = jpos_pred_f - root_pred.unsqueeze(-2)             # (B, T, 22, 3)
            local_gt = joints_gt_s - root_gt.unsqueeze(-2)                 # (B, T, 22, 3)
            vel_local_pred = local_pred[:, 1:] - local_pred[:, :-1]        # (B, T-1, 22, 3)
            vel_local_gt = local_gt[:, 1:] - local_gt[:, :-1]              # (B, T-1, 22, 3)
            if vel_mask.any():
                err_v = (vel_pred - vel_gt).pow(2).sum(-1)             # (B, T-1)
                loss_stable_root_vel = (err_v[vel_mask]).mean()
                stable_root_vel_rms_pred = (
                    vel_pred.pow(2).sum(-1)[vel_mask].mean().sqrt()
                )
                stable_root_vel_rms_gt = (
                    vel_gt.pow(2).sum(-1)[vel_mask].mean().sqrt()
                )

                # --- Local velocity matching (legacy m²-MSE form, per
                # claude_code_v11_after_full_frozen_fix_handoff.md §A.2).
                # Kept for back-compat; gradient contribution is ~10⁻⁵ at
                # weight 0.1 — superseded by the cm-scale block below.
                err_lv = (vel_local_pred - vel_local_gt).pow(2).sum(-1)    # (B, T-1, 22)
                m_lv = vel_mask.unsqueeze(-1).expand_as(err_lv)
                loss_stable_local_vel = err_lv[m_lv].mean()
                stable_local_vel_rms_pred = (
                    vel_local_pred.pow(2).sum(-1)[m_lv].mean().sqrt()
                )
                stable_local_vel_rms_gt = (
                    vel_local_gt.pow(2).sum(-1)[m_lv].mean().sqrt()
                )

                # --- cm-scale local-dynamics losses (per
                # claude_code_v11_next_localdyn_target_routing.md §A.2-A.3).
                # Scale velocity diffs to cm/frame then SmoothL1; scale
                # speed magnitudes to cm/frame then moment-match (mean+std).
                # m_lv selects (b, t-1, j) tuples; we apply the same mask
                # to the (B, T-1, 22, 3) tensors via broadcasting.
                vel_pred_cm = vel_local_pred * 100.0          # m → cm
                vel_gt_cm = vel_local_gt * 100.0
                vel_diff_cm = vel_pred_cm - vel_gt_cm          # (B, T-1, 22, 3)
                m_lv_xyz = m_lv.unsqueeze(-1).expand_as(vel_diff_cm)
                # SmoothL1 with default beta=1.0 (cm-scale): half-quadratic
                # below ±1 cm, linear above. Tolerant of large outliers,
                # provides usable gradient at sub-cm scale.
                vel_diff_cm_valid = vel_diff_cm[m_lv_xyz]
                if vel_diff_cm_valid.numel() > 0:
                    loss_stable_local_vel_cm = F.smooth_l1_loss(
                        vel_diff_cm_valid,
                        torch.zeros_like(vel_diff_cm_valid),
                        reduction="mean",
                        beta=1.0,
                    )

                # Speed-moment loss (§A.3). Per (b, t-1, j) joint-speed in
                # cm/frame; reduce to scalar mean and std over masked
                # (b, t-1, j) tuples; match pred to GT in both moments.
                speed_pred_cm = vel_pred_cm.pow(2).sum(-1).clamp_min(1e-12).sqrt()  # (B, T-1, 22)
                speed_gt_cm = vel_gt_cm.pow(2).sum(-1).clamp_min(1e-12).sqrt()
                speed_pred_valid = speed_pred_cm[m_lv]
                speed_gt_valid = speed_gt_cm[m_lv]
                if speed_pred_valid.numel() > 0:
                    stable_local_speed_mean_pred = speed_pred_valid.mean()
                    stable_local_speed_mean_gt = speed_gt_valid.mean()
                    stable_local_speed_std_pred = speed_pred_valid.std(unbiased=False)
                    stable_local_speed_std_gt = speed_gt_valid.std(unbiased=False)
                    loss_stable_local_speed_moment = (
                        (stable_local_speed_mean_pred - stable_local_speed_mean_gt).pow(2)
                        + (stable_local_speed_std_pred - stable_local_speed_std_gt).pow(2)
                    )

                if T >= 3:
                    acc_pred = vel_pred[:, 1:] - vel_pred[:, :-1]
                    acc_gt = vel_gt[:, 1:] - vel_gt[:, :-1]
                    acc_mask = vel_mask[:, 1:] & vel_mask[:, :-1]      # (B, T-2)
                    if acc_mask.any():
                        err_a = (acc_pred - acc_gt).pow(2).sum(-1)      # (B, T-2)
                        loss_stable_root_acc = (err_a[acc_mask]).mean()
                        stable_root_acc_rms_pred = (
                            acc_pred.pow(2).sum(-1)[acc_mask].mean().sqrt()
                        )
                        stable_root_acc_rms_gt = (
                            acc_gt.pow(2).sum(-1)[acc_mask].mean().sqrt()
                        )

                        # --- Local acceleration matching — legacy m²-MSE
                        # (kept for back-compat with prior FULL+localvel
                        # weight 0 runs) plus cm-scale SmoothL1 (§A.4).
                        acc_local_pred = vel_local_pred[:, 1:] - vel_local_pred[:, :-1]
                        acc_local_gt = vel_local_gt[:, 1:] - vel_local_gt[:, :-1]
                        err_la = (acc_local_pred - acc_local_gt).pow(2).sum(-1)  # (B, T-2, 22)
                        m_la = acc_mask.unsqueeze(-1).expand_as(err_la)
                        loss_stable_local_acc = err_la[m_la].mean()
                        stable_local_acc_rms_pred = (
                            acc_local_pred.pow(2).sum(-1)[m_la].mean().sqrt()
                        )
                        stable_local_acc_rms_gt = (
                            acc_local_gt.pow(2).sum(-1)[m_la].mean().sqrt()
                        )

                        acc_diff_cm = (acc_local_pred - acc_local_gt) * 100.0  # (B, T-2, 22, 3)
                        m_la_xyz = m_la.unsqueeze(-1).expand_as(acc_diff_cm)
                        acc_diff_cm_valid = acc_diff_cm[m_la_xyz]
                        if acc_diff_cm_valid.numel() > 0:
                            loss_stable_local_acc_cm = F.smooth_l1_loss(
                                acc_diff_cm_valid,
                                torch.zeros_like(acc_diff_cm_valid),
                                reduction="mean",
                                beta=1.0,
                            )

        total = (
            mse
            + anchor
            + geom["loss_geometric"]
            + fk_consistency_weight * loss_fk
            + pos_loss_weight * loss_pos_full
            + anchor_joint_pos_weight * loss_anchor_joint_pos
            + anchor_joint_vel_weight * loss_anchor_joint_vel
            + plan_anchor_weight * loss_plan_anchor
            + plan_segment_weight * loss_plan_segment
            + plan_transition_vel_weight * loss_plan_trans_vel
            + plan_transition_acc_weight * loss_plan_trans_acc
            + stable_root_vel_weight * loss_stable_root_vel
            + stable_root_acc_weight * loss_stable_root_acc
            + stable_local_vel_weight * loss_stable_local_vel
            + stable_local_acc_weight * loss_stable_local_acc
            + stable_local_vel_cm_weight * loss_stable_local_vel_cm
            + stable_local_acc_cm_weight * loss_stable_local_acc_cm
            + stable_local_speed_moment_weight * loss_stable_local_speed_moment
            # Round-27 Tier-0B temporal interaction losses (zero by default).
            # Round-28 adds two consistency terms (also zero by default).
            + (
                float(temporal_loss_cfg.contact_rel_offset_weight) * loss_contact_rel
                + float(temporal_loss_cfg.contact_drift_weight) * loss_contact_drift
                + float(temporal_loss_cfg.contact_tracking_weight) * loss_contact_track
                + float(temporal_loss_cfg.gait_both_airborne_weight) * loss_gait_air
                + float(temporal_loss_cfg.gait_stance_velocity_weight) * loss_gait_stance_vel
                + float(getattr(temporal_loss_cfg, "hint_contact_consistency_weight", 0.0)) * loss_hint_contact_cons
                + float(getattr(temporal_loss_cfg, "body_action_consistency_weight", 0.0)) * loss_body_action_cons
                if temporal_loss_cfg is not None else 0.0
            )
        )
        out = {
            "loss": total,
            "mse_x0": mse.detach(),
            "mse_x0_unweighted": mse_unweighted.detach(),
            "mse_main": mse_main.detach(),
            "mse_kf": mse_kf.detach(),
            "anchor_l2": anchor.detach(),
            "loss_geometric": geom["loss_geometric"].detach(),
            "loss_pos": geom["loss_pos"].detach(),
            "loss_vel": geom["loss_vel"].detach(),
            "loss_foot": geom["loss_foot"].detach(),
            "loss_fk": loss_fk.detach(),
            "loss_pos_full": loss_pos_full.detach(),
            "loss_pos_full_obs": loss_pos_full_obs_monitor.detach(),
            "loss_pos_full_unobs": loss_pos_full_unobs_monitor.detach(),
            "loss_anchor_joint_pos": loss_anchor_joint_pos.detach(),
            "loss_anchor_joint_vel": loss_anchor_joint_vel.detach(),
            "weighted_anchor_joint_pos": (
                anchor_joint_pos_weight * loss_anchor_joint_pos
            ).detach(),
            "weighted_anchor_joint_vel": (
                anchor_joint_vel_weight * loss_anchor_joint_vel
            ).detach(),
            "anchor_joint_active_ratio": anchor_joint_active_ratio.detach(),
            "loss_plan_anchor": loss_plan_anchor.detach(),
            "loss_plan_segment": loss_plan_segment.detach(),
            "loss_plan_trans_vel": loss_plan_trans_vel.detach(),
            "loss_plan_trans_acc": loss_plan_trans_acc.detach(),
            "loss_stable_root_vel": loss_stable_root_vel.detach(),
            "loss_stable_root_acc": loss_stable_root_acc.detach(),
            "loss_stable_local_vel": loss_stable_local_vel.detach(),
            "loss_stable_local_acc": loss_stable_local_acc.detach(),
            "loss_stable_local_vel_cm": loss_stable_local_vel_cm.detach(),
            "loss_stable_local_acc_cm": loss_stable_local_acc_cm.detach(),
            "loss_stable_local_speed_moment": loss_stable_local_speed_moment.detach(),
            # Round-27 Tier-0B temporal interaction loss components.
            "loss_contact_rel": loss_contact_rel.detach(),
            "loss_contact_drift_t": loss_contact_drift.detach(),
            "loss_contact_track": loss_contact_track.detach(),
            "loss_gait_air": loss_gait_air.detach(),
            "loss_gait_stance_vel": loss_gait_stance_vel.detach(),
            # Round-28 consistency loss components.
            "loss_hint_contact_cons": loss_hint_contact_cons.detach(),
            "loss_body_action_cons": loss_body_action_cons.detach(),
            "weighted_stable_local_vel_cm": (
                stable_local_vel_cm_weight * loss_stable_local_vel_cm
            ).detach(),
            "weighted_stable_local_acc_cm": (
                stable_local_acc_cm_weight * loss_stable_local_acc_cm
            ).detach(),
            "weighted_stable_local_speed_moment": (
                stable_local_speed_moment_weight * loss_stable_local_speed_moment
            ).detach(),
            "stable_support_frame_ratio": stable_support_frame_ratio.detach(),
            "stable_root_vel_rms_pred": stable_root_vel_rms_pred.detach(),
            "stable_root_vel_rms_gt": stable_root_vel_rms_gt.detach(),
            "stable_root_acc_rms_pred": stable_root_acc_rms_pred.detach(),
            "stable_root_acc_rms_gt": stable_root_acc_rms_gt.detach(),
            "stable_local_vel_rms_pred": stable_local_vel_rms_pred.detach(),
            "stable_local_vel_rms_gt": stable_local_vel_rms_gt.detach(),
            "stable_local_acc_rms_pred": stable_local_acc_rms_pred.detach(),
            "stable_local_acc_rms_gt": stable_local_acc_rms_gt.detach(),
            "stable_local_speed_mean_pred": stable_local_speed_mean_pred.detach(),
            "stable_local_speed_mean_gt": stable_local_speed_mean_gt.detach(),
            "stable_local_speed_std_pred": stable_local_speed_std_pred.detach(),
            "stable_local_speed_std_gt": stable_local_speed_std_gt.detach(),
            # Un-detached loss components for the gradient-audit smoke
            # path (per claude_code_v11_next_localdyn_target_routing.md §A.6).
            # Underscore prefix marks them as private (not logged to wandb).
            "_raw_mse_x0": mse,
            "_raw_anchor": anchor,
            "_raw_loss_pos_full": loss_pos_full,
            "_raw_loss_anchor_joint_pos": loss_anchor_joint_pos,
            "_raw_loss_anchor_joint_vel": loss_anchor_joint_vel,
            "_raw_loss_plan_anchor": loss_plan_anchor,
            "_raw_loss_stable_root_vel": loss_stable_root_vel,
            "_raw_loss_stable_local_vel_cm": loss_stable_local_vel_cm,
            "_raw_loss_stable_local_acc_cm": loss_stable_local_acc_cm,
            "_raw_loss_stable_local_speed_moment": loss_stable_local_speed_moment,
            # Min-SNR-γ diagnostics (active only when use_min_snr_weighting=True;
            # all-zero tensors otherwise).
            "min_snr_weight_mean": min_snr_weight_mean,
            "min_snr_weight_min": min_snr_weight_min,
            "min_snr_weight_max": min_snr_weight_max,
        }

        # Round-28 hint-interface diagnostics. The denoiser caches
        # scalar stats during forward (hint norms, gate stats, adapter
        # output norms). The generic trainer adds branch grad norms
        # after backward when these r28_* keys are present.
        _inner_model = _model.module if hasattr(_model, "module") else _model
        _denoiser_stats = getattr(
            getattr(_inner_model, "denoiser", None),
            "_last_oracle_hint_stats",
            {},
        )
        for _k, _v in _denoiser_stats.items():
            if isinstance(_v, torch.Tensor) and _v.numel() == 1:
                out[_k] = _v.detach()

        # v12 architecture-utilization metrics (per
        # analyses/2026-05-11_v12_implementation_doc.md §2.4 / §3.2).
        # Cheap to compute (norm of weight tensors), useful for diagnosing
        # whether the new conditioning pathways are actually engaging.
        denoiser = getattr(model, "denoiser", None)
        if denoiser is not None and getattr(denoiser.cfg, "use_dit_block", False):
            with torch.no_grad():
                out["v12_input_proj_norm_motion"] = (
                    denoiser.v12_input_proj.motion_proj.weight.norm().detach()
                )
                out["v12_input_proj_norm_zint"] = (
                    denoiser.v12_input_proj.zint_proj.weight.norm().detach()
                )
                out["v12_input_proj_norm_obj"] = (
                    denoiser.v12_input_proj.obj_proj.weight.norm().detach()
                )
                out["v12_input_proj_norm_hint"] = (
                    denoiser.v12_input_proj.hint_proj.weight.norm().detach()
                )
                for i, block in enumerate(denoiser.v12_blocks):
                    out[f"v12_adaLN_norm_layer{i}"] = (
                        block.adaLN_modulation[-1].weight.norm().detach()
                    )
                    out[f"v12_xattn_out_proj_norm_layer{i}"] = (
                        block.plan_xattn.out_proj.weight.norm().detach()
                    )
                fl = denoiser.v12_final_layer
                if hasattr(fl, "linear"):
                    # v12 FinalLayer: AdaLN + single linear
                    out["v12_final_adaLN_norm"] = (
                        fl.adaLN_modulation[-1].weight.norm().detach()
                    )
                    out["v12_final_linear_norm"] = (
                        fl.linear.weight.norm().detach()
                    )
                else:
                    # v13 DynamicsHead: base + delta branches + learnable γ
                    out["v13_final_adaLN_base_norm"] = (
                        fl.adaLN_base[-1].weight.norm().detach()
                    )
                    out["v13_final_base_linear_norm"] = (
                        fl.base_linear.weight.norm().detach()
                    )
                    out["v13_final_adaLN_delta_norm"] = (
                        fl.adaLN_delta[-1].weight.norm().detach()
                    )
                    out["v13_final_delta_linear_norm"] = (
                        fl.delta_linear.weight.norm().detach()
                    )
                    out["v13_gamma"] = fl.gamma.detach()
                    # Per-block temporal conv gate norm (if v13 P2 active)
                    if hasattr(denoiser.v12_blocks[0], "temporal_conv"):
                        for i, block in enumerate(denoiser.v12_blocks):
                            out[f"v13_temporal_conv_gate_layer{i}"] = (
                                block.temporal_conv.gate.detach().squeeze()
                            )
        return out

    return step_fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--grad-audit", action="store_true",
                        help="With --smoke-test: report per-loss-term L2 grad norm at "
                             "model.in_proj.weight (per spec §A.6).")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run a single batch + backward to verify wiring; do not save.")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    # find_unused_parameters=True is defensive for v27-style configs where
    # `plan_tokens_force_null: true` etc. zero out branches whose parameters
    # may never receive gradient — DDP would otherwise hang on the per-iter
    # all-reduce waiting for those params. Small overhead (extra bookkeeping
    # of the autograd graph), no correctness impact.
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.get("gradient_accumulation_steps", 1),
        mixed_precision=cfg.training.get("mixed_precision", "bf16"),
        kwargs_handlers=[ddp_kwargs],
    )
    set_seed(int(cfg.training.get("seed", 42)))
    device = accelerator.device

    accelerator.print("===== PIANO-AnchorDiff training =====")
    accelerator.print(f"output_dir = {cfg.output_dir}")
    accelerator.print(f"smoke_test = {args.smoke_test}")

    # --- Build dataset ---
    train_dataset = _build_dataset(cfg, bucket="train", augment=True)
    train_subset_indices: list[int] | None = None
    overfit_n = int(cfg.data.get("overfit_n_clips", 0))
    # Scale curve diagnostic (per analyses/stageB_root_cause_analysis_v2_and_next_strategy.md §5):
    # when scale_subset_seed is set, shuffle the train_dataset indices with that
    # seed BEFORE taking the first overfit_n. This produces representative random
    # mixed subsets across the 4 datasets. With the same seed across scales,
    # smaller subsets are STRICT SUBSETS of larger ones (nested).
    #
    # Controlled multimodality diagnostic (per same doc §6, P1): when
    # data.subset_indices_file is set, load explicit indices from a JSON file
    # (built by scripts/stage_b_generator/build_multimodality_subsets.py).
    # Takes precedence over overfit_n_clips / scale_subset_seed.
    scale_subset_seed = cfg.data.get("scale_subset_seed", None)
    subset_indices_file = cfg.data.get("subset_indices_file", None)
    if subset_indices_file is not None:
        with open(str(subset_indices_file), encoding="utf-8") as _f:
            _meta = json.load(_f)
        indices = list(_meta["indices"])
        n_avail = len(train_dataset)
        indices = [i for i in indices if 0 <= i < n_avail]
        train_subset_indices = indices
        train_dataset = Subset(train_dataset, indices)
        accelerator.print(
            f"Train dataset (SUBSET file {subset_indices_file}): "
            f"{len(train_dataset)} clips"
        )
    elif overfit_n > 0:
        n_avail = len(train_dataset)
        indices = list(range(n_avail))
        if scale_subset_seed is not None:
            import random as _random
            _rng = _random.Random(int(scale_subset_seed))
            _rng.shuffle(indices)
        indices = indices[:min(overfit_n, n_avail)]
        train_subset_indices = indices
        train_dataset = Subset(train_dataset, indices)
        accelerator.print(
            f"Train dataset (OVERFIT): {len(train_dataset)} clips "
            f"{'shuffled, seed=' + str(scale_subset_seed) if scale_subset_seed is not None else 'first-N'}"
        )
    else:
        accelerator.print(f"Train dataset: {len(train_dataset)} clips")

    val_dataset = None
    if int(cfg.training.get("val_every_epochs", 0)) > 0:
        if bool(cfg.training.get("val_on_train_subset", False)):
            val_base = _build_dataset(cfg, bucket="train", augment=False)
            if train_subset_indices is not None:
                val_dataset = Subset(val_base, train_subset_indices)
                accelerator.print(
                    "Val dataset (TRAIN SUBSET, no augment): "
                    f"{len(val_dataset)} clips"
                )
            else:
                val_dataset = val_base
                accelerator.print(
                    "Val dataset (FULL TRAIN, no augment): "
                    f"{len(val_dataset)} clips"
                )
        else:
            val_dataset = _build_dataset(cfg, bucket="val", augment=False)
            if overfit_n > 0:
                n_avail_val = len(val_dataset)
                v_indices = list(range(n_avail_val))
                if scale_subset_seed is not None:
                    import random as _random
                    _rng_v = _random.Random(int(scale_subset_seed))
                    _rng_v.shuffle(v_indices)
                v_indices = v_indices[:min(overfit_n, n_avail_val)]
                val_dataset = Subset(val_dataset, v_indices)
            accelerator.print(f"Val dataset:   {len(val_dataset)} clips")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=collate_hoi,
        num_workers=int(cfg.training.get("num_workers", 4)),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            collate_fn=collate_hoi,
            num_workers=int(cfg.training.get("num_workers", 4)),
            pin_memory=True,
            drop_last=False,
        )

    # --- Build model ---
    z_dims = ZIntDims(
        num_parts=int(cfg.model.z_int.num_parts),
        phase_classes=int(cfg.model.z_int.phase_classes),
        support_classes=int(cfg.model.z_int.support_classes),
    )
    denoiser_cfg = AnchorDenoiserConfig(
        motion_dim=int(cfg.model.denoiser.motion_dim),
        z_int=z_dims,
        object_traj_dim=int(cfg.model.denoiser.object_traj_dim),
        init_pose_dim=int(cfg.model.denoiser.init_pose_dim),
        text_dim=int(cfg.model.denoiser.text_dim),
        object_token_dim=int(cfg.model.denoiser.object_token_dim),
        object_num_tokens=int(cfg.model.denoiser.object_num_tokens),
        use_interaction_plan=bool(cfg.model.denoiser.get("use_interaction_plan", False)),
        plan_k_max=int(cfg.model.denoiser.get("plan_k_max", 12)),
        plan_s_max=int(cfg.model.denoiser.get("plan_s_max", 12)),
        plan_num_anchor_types=int(cfg.model.denoiser.get("plan_num_anchor_types", 5)),
        plan_num_parts=int(cfg.model.denoiser.get("plan_num_parts", 5)),
        plan_use_segment_tokens=bool(cfg.model.denoiser.get("plan_use_segment_tokens", False)),
        plan_use_context_hint=bool(cfg.model.denoiser.get("plan_use_context_hint", True)),
        plan_d_hint=int(cfg.model.denoiser.get("plan_d_hint", 32)),
        plan_d_time_embed=int(cfg.model.denoiser.get("plan_d_time_embed", 64)),
        cfg_drop_plan=bool(cfg.model.denoiser.get("cfg_drop_plan", False)),
        plan_per_part_tokens=bool(cfg.model.denoiser.get("plan_per_part_tokens", False)),
        plan_context_hint_mode=str(cfg.model.denoiser.get("plan_context_hint_mode", "time_only")),
        use_dit_block=bool(cfg.model.denoiser.get("use_dit_block", False)),
        dit_block_use_plan_pool_in_cond=bool(
            cfg.model.denoiser.get("dit_block_use_plan_pool_in_cond", True)
        ),
        stage1_coarse_dim=int(cfg.model.denoiser.get("stage1_coarse_dim", 0)),
        cfg_drop_stage1_coarse=bool(
            cfg.model.denoiser.get("cfg_drop_stage1_coarse", False)
        ),
        plan_xattn_relative_time_bias=bool(
            cfg.model.denoiser.get("plan_xattn_relative_time_bias", False)
        ),
        plan_xattn_time_bias_init=float(
            cfg.model.denoiser.get("plan_xattn_time_bias_init", 0.5)
        ),
        plan_tokens_force_null=bool(
            cfg.model.denoiser.get("plan_tokens_force_null", False)
        ),
        use_oracle_interaction_hint=bool(
            cfg.model.denoiser.get("use_oracle_interaction_hint", False)
        ),
        oracle_hint_dim=int(cfg.model.denoiser.get("oracle_hint_dim", 0)),
        use_body_action_hint=bool(
            cfg.model.denoiser.get("use_body_action_hint", False)
        ),
        body_action_hint_dim=int(
            cfg.model.denoiser.get("body_action_hint_dim", 0)
        ),
        oracle_hint_injection_mode=str(
            cfg.model.denoiser.get("oracle_hint_injection_mode", "input_add")
        ),
        oracle_hint_gate_bias_init=float(
            cfg.model.denoiser.get("oracle_hint_gate_bias_init", -3.0)
        ),
        separate_hint_branches=bool(
            cfg.model.denoiser.get("separate_hint_branches", True)
        ),
        zero_init_hint_adapters=bool(
            cfg.model.denoiser.get("zero_init_hint_adapters", True)
        ),
        d_model=int(cfg.model.denoiser.d_model),
        n_layers=int(cfg.model.denoiser.n_layers),
        n_heads=int(cfg.model.denoiser.n_heads),
        ff_mult=int(cfg.model.denoiser.ff_mult),
        dropout=float(cfg.model.denoiser.dropout),
        max_seq_length=int(cfg.data.max_seq_length),
    )
    diff_cfg = DiffusionConfig(
        num_steps=int(cfg.model.diffusion.num_steps),
        schedule=str(cfg.model.diffusion.schedule),
        objective=str(cfg.model.diffusion.get("objective", "ddpm")),
        prediction_target=str(cfg.model.diffusion.get("prediction_target", "x0")),
        rf_eps_time=float(cfg.model.diffusion.get("rf_eps_time", 0.05)),
        rf_time_schedule=str(cfg.model.diffusion.get("rf_time_schedule", "uniform")),
        rf_denoiser_p_mean=float(cfg.model.diffusion.get("rf_denoiser_p_mean", -1.5)),
        rf_denoiser_p_std=float(cfg.model.diffusion.get("rf_denoiser_p_std", 0.8)),
        rf_denoiser_noise_scale=float(cfg.model.diffusion.get("rf_denoiser_noise_scale", 1.0)),
        rf_num_sampling_steps=int(cfg.model.diffusion.get("rf_num_sampling_steps", 100)),
        rf_sampler_type=str(cfg.model.diffusion.get("rf_sampler_type", "rectified_flow_ode")),
        rf_sde_gamma=float(cfg.model.diffusion.get("rf_sde_gamma", 0.0)),
    )
    model = MotionAnchorDiff(
        AnchorDiffConfig(
            diffusion=diff_cfg,
            denoiser=denoiser_cfg,
            cfg_drop_prob=float(cfg.model.cfg_drop_prob),
        )
    )

    object_encoder = ObjectEncoder(
        num_input_points=int(cfg.model.object_encoder.num_input_points),
        num_output_tokens=int(cfg.model.object_encoder.num_output_tokens),
        feature_dim=int(cfg.model.object_encoder.feature_dim),
    )

    init_ckpt = cfg.training.get("init_checkpoint", None)
    if init_ckpt:
        init_path = Path(str(init_ckpt))
        accelerator.print(f"Loading model init checkpoint: {init_path}")
        state = torch.load(init_path, map_location="cpu", weights_only=False)
        model_state = state.get("model", state)
        partial_init = bool(
            cfg.training.get("partial_init_allow_shape_mismatch", False)
        )
        if partial_init:
            partial, skipped = _load_state_dict_compatible(model, model_state)
            accelerator.print(
                f"Partial model init: {len(partial)} resized tensors, "
                f"{len(skipped)} skipped tensors"
            )
            if partial:
                accelerator.print("  resized: " + ", ".join(partial[:8]))
        else:
            model.load_state_dict(model_state)
        if "object_encoder" in state:
            object_encoder.load_state_dict(state["object_encoder"])
        elif "extra_modules" in state and "object_encoder" in state["extra_modules"]:
            object_encoder.load_state_dict(state["extra_modules"]["object_encoder"])
        else:
            raise KeyError(
                f"init checkpoint {init_path} does not contain object_encoder weights"
            )
        accelerator.print(
            "Loaded init checkpoint weights only; optimizer/scheduler are reset.",
        )

    clip_model = load_clip_text_encoder(
        device=device,
        model_name=str(cfg.model.text_encoder.clip_version),
        download_root=str(cfg.model.text_encoder.get("download_root", "cache/clip")),
    )

    anchor_cfg = AnchorConsistencyConfig(
        weight=float(cfg.loss.anchor_weight),
        contact_threshold=float(cfg.loss.contact_threshold),
    )
    geom_cfg_raw = cfg.loss.get("motion_geometric", None)
    if geom_cfg_raw is None:
        geometric_cfg = MotionGeometricLossConfig()
    else:
        foot_indices = tuple(
            int(v)
            for v in geom_cfg_raw.get(
                "foot_joint_indices",
                MotionGeometricLossConfig().foot_joint_indices,
            )
        )
        if len(foot_indices) != 4:
            raise ValueError("loss.motion_geometric.foot_joint_indices must have 4 entries")
        geometric_cfg = MotionGeometricLossConfig(
            enabled=bool(geom_cfg_raw.get("enabled", False)),
            pos_weight=float(geom_cfg_raw.get("pos_weight", 0.0)),
            vel_weight=float(geom_cfg_raw.get("vel_weight", 0.0)),
            foot_weight=float(geom_cfg_raw.get("foot_weight", 0.0)),
            foot_contact_threshold=float(
                geom_cfg_raw.get("foot_contact_threshold", 0.5)
            ),
            foot_velocity_threshold=float(
                geom_cfg_raw.get("foot_velocity_threshold", 0.01)
            ),
            foot_joint_indices=foot_indices,
        )
    accelerator.print(
        "Motion geometric losses: "
        f"enabled={geometric_cfg.enabled} "
        f"pos={geometric_cfg.pos_weight} "
        f"vel={geometric_cfg.vel_weight} "
        f"foot={geometric_cfg.foot_weight}",
    )
    motion_representation = str(cfg.data.get("motion_representation", "smpl_pose_135_plan"))
    if motion_representation != "smpl_pose_135_plan":
        raise ValueError(
            "Round-28 trainer only supports motion_representation='smpl_pose_135_plan'; "
            f"got {motion_representation!r}"
        )
    world_joint_velocity_weight = float(
        cfg.loss.get("world_joint_velocity_weight", 0.0)
    )
    fk_consistency_weight = float(cfg.loss.get("fk_consistency_weight", 0.0))
    pos_loss_weight = float(cfg.loss.get("pos_loss_weight", 0.0))
    # Round-25 D5: per-joint endpoint weighting for the FK pos loss.
    hand_endpoint_weight = float(cfg.loss.get("hand_endpoint_weight", 1.0))
    foot_endpoint_weight = float(cfg.loss.get("foot_endpoint_weight", 1.0))
    anchor_joint_pos_weight = float(cfg.loss.get("anchor_joint_pos_weight", 0.0))
    anchor_joint_vel_weight = float(cfg.loss.get("anchor_joint_vel_weight", 0.0))
    anchor_joint_part_weights = tuple(
        float(v)
        for v in cfg.loss.get(
            "anchor_joint_part_weights",
            [1.0, 1.0, 1.0, 1.0, 1.0],
        )
    )
    if len(anchor_joint_part_weights) != 5:
        raise ValueError(
            "loss.anchor_joint_part_weights must have 5 entries "
            "(left_hand, right_hand, left_foot, right_foot, pelvis)"
        )
    accelerator.print(
        f"Motion representation: smpl_pose_135_plan "
        f"(world_joint_velocity_weight={world_joint_velocity_weight} "
        f"pos_loss_weight={pos_loss_weight} "
        f"hand_endpoint_weight={hand_endpoint_weight} "
        f"foot_endpoint_weight={foot_endpoint_weight} "
        f"anchor_joint_pos_weight={anchor_joint_pos_weight} "
        f"anchor_joint_vel_weight={anchor_joint_vel_weight})",
    )
    _fk_pos_enabled = pos_loss_weight > 0.0
    accelerator.print(
        f"[AnchorDiff] dense FK L_pos enabled: {_fk_pos_enabled} "
        f"(weight={pos_loss_weight})"
    )
    if int(cfg.model.denoiser.motion_dim) != 135:
        raise ValueError(
            "smpl_pose_135_plan requires model.denoiser.motion_dim=135 "
            f"(got {int(cfg.model.denoiser.motion_dim)})"
        )
    if not bool(cfg.model.denoiser.get("use_interaction_plan", False)):
        raise ValueError(
            "smpl_pose_135_plan requires model.denoiser.use_interaction_plan=true"
        )

    # --- Optimizer + scheduler ---
    accum = int(cfg.training.get("gradient_accumulation_steps", 1))
    steps_per_epoch = max(1, len(train_loader) // accum)
    total_steps = steps_per_epoch * int(cfg.training.num_epochs)
    optimizer = build_optimizer_with_decay_groups(
        modules=[model, object_encoder],
        lr=float(cfg.training.optimizer.lr),
        weight_decay=float(cfg.training.optimizer.weight_decay),
        betas=tuple(cfg.training.optimizer.betas),
    )
    scheduler = build_scheduler(
        optimizer,
        int(cfg.training.scheduler.warmup_steps),
        total_steps,
    )

    model, object_encoder, optimizer, train_loader, scheduler = accelerator.prepare(
        model, object_encoder, optimizer, train_loader, scheduler,
    )
    if val_loader is not None:
        val_loader = accelerator.prepare(val_loader)

    # --- FeatureWeightState (static v2 / dynamic v2.1) ---
    static_w_cfg = (
        OmegaConf.to_container(cfg.loss.motion_feature_weights, resolve=True)
        if "motion_feature_weights" in cfg.loss else None
    )
    dyn_cfg = cfg.loss.get("dynamic_metric", None)
    if dyn_cfg is not None and dyn_cfg.get("enabled", False):
        raise ValueError(
            "dynamic_metric is not supported in Round-28 (was motion_263-only)"
        )
    feature_weight_state = FeatureWeightState.static_from_config(static_w_cfg)
    accelerator.print("FeatureWeightState STATIC mode")

    use_interaction_plan = bool(
        cfg.model.denoiser.get("use_interaction_plan", False)
    )
    plan_anchor_weight = float(cfg.loss.get("plan_anchor_weight", 0.0))
    plan_segment_weight = float(cfg.loss.get("plan_segment_weight", 0.0))
    plan_transition_vel_weight = float(
        cfg.loss.get("plan_transition_vel_weight", 0.0)
    )
    plan_transition_acc_weight = float(
        cfg.loss.get("plan_transition_acc_weight", 0.0)
    )
    plan_transition_window = int(cfg.loss.get("plan_transition_window", 3))
    stable_root_vel_weight = float(cfg.loss.get("stable_root_vel_weight", 0.0))
    stable_root_acc_weight = float(cfg.loss.get("stable_root_acc_weight", 0.0))
    stable_local_vel_weight = float(cfg.loss.get("stable_local_vel_weight", 0.0))
    stable_local_acc_weight = float(cfg.loss.get("stable_local_acc_weight", 0.0))
    # cm-scale local-dynamics losses (per
    # claude_code_v11_next_localdyn_target_routing.md §A.2–A.4).
    stable_local_vel_cm_weight = float(cfg.loss.get("stable_local_vel_cm_weight", 0.0))
    stable_local_acc_cm_weight = float(cfg.loss.get("stable_local_acc_cm_weight", 0.0))
    stable_local_speed_moment_weight = float(
        cfg.loss.get("stable_local_speed_moment_weight", 0.0)
    )
    stable_support_erode = int(cfg.loss.get("stable_support_erode", 4))
    # Min-SNR-γ (Hang et al. arXiv:2303.09556) — per spec
    # analyses/stageB_updated_training_strategy_and_diagnostics_plan.md §4.
    use_min_snr_weighting = bool(cfg.loss.get("use_min_snr_weighting", False))
    min_snr_gamma = float(cfg.loss.get("min_snr_gamma", 5.0))

    # Round-27 Tier-0B: per-term weights for the 5 temporal interaction
    # losses (src/piano/training/temporal_interaction_losses.py). Built
    # whenever ANY weight is positive; defaults preserve back-compat for
    # v27 / earlier configs.
    temporal_loss_cfg = None
    _tloss = cfg.loss.get("temporal_interaction", None)
    if _tloss is not None:
        from piano.training.temporal_interaction_losses import (
            TemporalInteractionLossConfig,
        )
        temporal_loss_cfg = TemporalInteractionLossConfig(
            contact_rel_offset_weight=float(
                _tloss.get("contact_rel_offset_weight", 0.0)
            ),
            contact_drift_weight=float(_tloss.get("contact_drift_weight", 0.0)),
            contact_tracking_weight=float(_tloss.get("contact_tracking_weight", 0.0)),
            gait_both_airborne_weight=float(
                _tloss.get("gait_both_airborne_weight", 0.0)
            ),
            gait_stance_velocity_weight=float(
                _tloss.get("gait_stance_velocity_weight", 0.0)
            ),
            hint_contact_consistency_weight=float(
                _tloss.get("hint_contact_consistency_weight", 0.0)
            ),
            body_action_consistency_weight=float(
                _tloss.get("body_action_consistency_weight", 0.0)
            ),
            contact_threshold=float(_tloss.get("contact_threshold", 0.5)),
            contact_rel_clamp_m=float(_tloss.get("contact_rel_clamp_m", 2.0)),
            tracking_margin_m=float(_tloss.get("tracking_margin_m", 0.03)),
            tracking_min_obj_disp_m=float(_tloss.get("tracking_min_obj_disp_m", 0.05)),
            floor_quantile=float(_tloss.get("floor_quantile", 0.05)),
            grounded_threshold_above_floor_m=float(
                _tloss.get("grounded_threshold_above_floor_m", 0.10)
            ),
            grounded_softness_m=float(_tloss.get("grounded_softness_m", 0.03)),
        )
    # Round-28 cleanup: all zero_*_for_stageB flags hardcoded inside
    # step_fn (active configs all set them True; the Stage-1.5 plan
    # tokens carry contact / part / phase / support info now).

    # Round-22: Stage-1 Coarse-v1 oracle condition wiring.
    stage1_coarse_dim = int(cfg.model.denoiser.get("stage1_coarse_dim", 0))
    stage1_coarse_noise_std = float(
        cfg.training.get("stage1_coarse_noise_std", 0.0)
    )
    if stage1_coarse_dim > 0:
        s1_cache_root = cfg.data.get("stage1_coarse_cache_root", None)
        if s1_cache_root is None:
            raise ValueError(
                "stage1_coarse_dim > 0 requires data.stage1_coarse_cache_root in "
                "the YAML config (path to the Stage-1 cache directory containing "
                "normalization_train.json)."
            )
        stage1_coarse_norm_mean, stage1_coarse_norm_std = load_stage1_coarse_norm(
            str(s1_cache_root)
        )
        accelerator.print(
            f"[StageB Round-22] Stage-1 coarse oracle conditioning ENABLED - "
            f"cache_root={s1_cache_root}  dim={stage1_coarse_dim}  "
            f"train_noise_std={stage1_coarse_noise_std}  "
            f"mean[:3]={stage1_coarse_norm_mean[:3].tolist()}"
        )
    else:
        stage1_coarse_norm_mean = None
        stage1_coarse_norm_std = None
    accelerator.print(
        "[StageB condition mode]\n"
        f"  use_interaction_plan: {use_interaction_plan}\n"
        f"  plan_per_part_tokens: {bool(cfg.model.denoiser.get('plan_per_part_tokens', False))}\n"
        f"  plan_context_hint_mode: {cfg.model.denoiser.get('plan_context_hint_mode', 'time_only')!r}\n"
        f"  object_traj_dim: {int(cfg.model.denoiser.object_traj_dim)}\n"
        f"  z_int channels: forced to zero (Round-28 PLAN-only mode)\n"
        f"  dense contact-target portion of object_traj: forced to zero\n"
        f"  dense FK L_pos enabled: {_fk_pos_enabled}\n"
        f"  active-part endpoint losses: "
        f"{anchor_joint_pos_weight > 0 or anchor_joint_vel_weight > 0}"
        f"  (pos={anchor_joint_pos_weight}, vel={anchor_joint_vel_weight}, "
        f"part_weights={list(anchor_joint_part_weights)})\n"
        f"  stable-support root loss enabled: "
        f"{stable_root_vel_weight > 0 or stable_root_acc_weight > 0}"
        f"  (root_vel={stable_root_vel_weight}, root_acc={stable_root_acc_weight}, erode={stable_support_erode})\n"
        f"  stable-support local loss (m^2-MSE, legacy): "
        f"{stable_local_vel_weight > 0 or stable_local_acc_weight > 0}"
        f"  (local_vel={stable_local_vel_weight}, local_acc={stable_local_acc_weight})\n"
        f"  stable-support local loss (cm-scale): "
        f"{stable_local_vel_cm_weight > 0 or stable_local_acc_cm_weight > 0 or stable_local_speed_moment_weight > 0}"
        f"  (local_vel_cm={stable_local_vel_cm_weight}, local_acc_cm={stable_local_acc_cm_weight}, speed_moment={stable_local_speed_moment_weight})\n"
        f"  diffusion objective: {cfg.model.diffusion.get('objective', 'ddpm')!r}  "
        f"(prediction_target={getattr(accelerator.unwrap_model(model).diffusion, 'prediction_target', 'x0')}, "
        f"rf_steps={getattr(accelerator.unwrap_model(model).diffusion, 'rf_num_sampling_steps', 0)}, "
        f"rf_schedule={getattr(accelerator.unwrap_model(model).diffusion, 'rf_time_schedule', 'n/a')!r})\n"
        f"  min-SNR weighting: {use_min_snr_weighting}  (gamma={min_snr_gamma}, "
        f"pred_target={getattr(accelerator.unwrap_model(model).diffusion, 'prediction_target', 'x0')})\n"
    )
    step_fn = build_anchordiff_step_fn(
        model=model,
        object_encoder=object_encoder,
        clip_model=clip_model,
        anchor_cfg=anchor_cfg,
        z_dims=z_dims,
        device=device,
        feature_weight_state=feature_weight_state,
        geometric_cfg=geometric_cfg,
        motion_representation=motion_representation,
        world_joint_velocity_weight=world_joint_velocity_weight,
        object_traj_dim=int(cfg.model.denoiser.object_traj_dim),
        fk_consistency_weight=fk_consistency_weight,
        pos_loss_weight=pos_loss_weight,
        hand_endpoint_weight=hand_endpoint_weight,
        foot_endpoint_weight=foot_endpoint_weight,
        anchor_joint_pos_weight=anchor_joint_pos_weight,
        anchor_joint_vel_weight=anchor_joint_vel_weight,
        anchor_joint_part_weights=anchor_joint_part_weights,
        use_interaction_plan=use_interaction_plan,
        plan_anchor_weight=plan_anchor_weight,
        plan_segment_weight=plan_segment_weight,
        plan_transition_vel_weight=plan_transition_vel_weight,
        plan_transition_acc_weight=plan_transition_acc_weight,
        plan_transition_window=plan_transition_window,
        stable_root_vel_weight=stable_root_vel_weight,
        stable_root_acc_weight=stable_root_acc_weight,
        stable_local_vel_weight=stable_local_vel_weight,
        stable_local_acc_weight=stable_local_acc_weight,
        stable_local_vel_cm_weight=stable_local_vel_cm_weight,
        stable_local_acc_cm_weight=stable_local_acc_cm_weight,
        stable_local_speed_moment_weight=stable_local_speed_moment_weight,
        use_min_snr_weighting=use_min_snr_weighting,
        min_snr_gamma=min_snr_gamma,
        stable_support_erode=stable_support_erode,
        stage1_coarse_dim=stage1_coarse_dim,
        stage1_coarse_norm_mean=stage1_coarse_norm_mean,
        stage1_coarse_norm_std=stage1_coarse_norm_std,
        stage1_coarse_noise_std=stage1_coarse_noise_std,
        temporal_loss_cfg=temporal_loss_cfg,
        fps=float(cfg.data.get("oracle_hint_fps", 20.0)),
    )

    # --- Smoke test: one batch through forward + backward ---
    if args.smoke_test:
        accelerator.print("Running smoke test (1 batch)...")
        batch = next(iter(train_loader))
        out = step_fn(model, batch, global_step=0)
        accelerator.print(
            f"  loss={out['loss'].item():.4f}  "
            f"mse_x0={out['mse_x0'].item():.4f}  "
            f"anchor_l2={out['anchor_l2'].item():.4f}  "
            f"geom={out['loss_geometric'].item():.4f}  "
            f"pos={out['loss_pos'].item():.4f}  "
            f"vel={out['loss_vel'].item():.4f}  "
            f"foot={out['loss_foot'].item():.4f}  "
            f"fk={out.get('loss_fk', torch.zeros(())).item():.4f}  "
            f"pos_full={out.get('loss_pos_full', torch.zeros(())).item():.4f}  "
            f"anchor_joint_pos={out.get('loss_anchor_joint_pos', torch.zeros(())).item():.4f}  "
            f"anchor_joint_vel={out.get('loss_anchor_joint_vel', torch.zeros(())).item():.4f}"
        )
        if "loss_stable_local_vel" in out:
            accelerator.print(
                f"  stable_root_vel={out.get('loss_stable_root_vel', torch.zeros(())).item():.4f}  "
                f"stable_root_acc={out.get('loss_stable_root_acc', torch.zeros(())).item():.4f}  "
                f"stable_local_vel={out['loss_stable_local_vel'].item():.4f}  "
                f"stable_local_acc={out.get('loss_stable_local_acc', torch.zeros(())).item():.4f}\n"
                f"  stable_local_vel_cm={out.get('loss_stable_local_vel_cm', torch.zeros(())).item():.4f}  "
                f"stable_local_acc_cm={out.get('loss_stable_local_acc_cm', torch.zeros(())).item():.4f}  "
                f"stable_local_speed_moment={out.get('loss_stable_local_speed_moment', torch.zeros(())).item():.4f}\n"
                f"  RMS root_vel pred/gt = "
                f"{out.get('stable_root_vel_rms_pred', torch.zeros(())).item():.4f} / "
                f"{out.get('stable_root_vel_rms_gt', torch.zeros(())).item():.4f}  "
                f"local_vel pred/gt = "
                f"{out.get('stable_local_vel_rms_pred', torch.zeros(())).item():.4f} / "
                f"{out.get('stable_local_vel_rms_gt', torch.zeros(())).item():.4f}\n"
                f"  speed cm/fr pred mean/std = "
                f"{out.get('stable_local_speed_mean_pred', torch.zeros(())).item():.4f} / "
                f"{out.get('stable_local_speed_std_pred', torch.zeros(())).item():.4f}  "
                f"gt mean/std = "
                f"{out.get('stable_local_speed_mean_gt', torch.zeros(())).item():.4f} / "
                f"{out.get('stable_local_speed_std_gt', torch.zeros(())).item():.4f}"
            )

        # --- Gradient audit (per claude_code_v11_next_localdyn_target_routing.md §A.6).
        # For each named loss term, do a separate retain_graph backward
        # and read the total L2 grad-norm at the model's input
        # projection (`model.in_proj.weight`). This proves the new
        # cm-scale losses actually contribute non-trivial gradient
        # signal, avoiding the previous mistake where a config-enabled
        # loss had effectively zero gradient.
        if args.grad_audit:
            unwrapped = accelerator.unwrap_model(model)
            # Pick the appropriate input projection weight to measure grad
            # against. v11 has a single in_proj.weight; v12 has the
            # motion_proj.weight inside the v12_input_proj module.
            if getattr(unwrapped.denoiser.cfg, "use_dit_block", False):
                in_proj = unwrapped.denoiser.v12_input_proj.motion_proj.weight
                proj_name = "model.denoiser.v12_input_proj.motion_proj.weight"
            else:
                in_proj = unwrapped.denoiser.in_proj.weight
                proj_name = "model.denoiser.in_proj.weight"
            accelerator.print(f"[grad audit] per-loss-term L2 grad norm at {proj_name}:")
            audit_terms = [
                ("mse_x0",                       out.get("_raw_mse_x0")),
                ("loss_pos_full",                out.get("_raw_loss_pos_full")),
                ("loss_anchor_joint_pos",        out.get("_raw_loss_anchor_joint_pos")),
                ("loss_anchor_joint_vel",        out.get("_raw_loss_anchor_joint_vel")),
                ("loss_plan_anchor",             out.get("_raw_loss_plan_anchor")),
                ("loss_stable_root_vel",         out.get("_raw_loss_stable_root_vel")),
                ("loss_stable_local_vel_cm",     out.get("_raw_loss_stable_local_vel_cm")),
                ("loss_stable_local_speed_mom",  out.get("_raw_loss_stable_local_speed_moment")),
                ("loss_stable_local_acc_cm",     out.get("_raw_loss_stable_local_acc_cm")),
            ]
            for i, (name, term) in enumerate(audit_terms):
                if term is None or not term.requires_grad:
                    accelerator.print(f"  {name:30s}  (skipped - no grad)")
                    continue
                if in_proj.grad is not None:
                    in_proj.grad.zero_()
                retain = (i < len(audit_terms) - 1) or True  # keep graph for final backward below
                try:
                    term.backward(retain_graph=True)
                except RuntimeError as e:
                    accelerator.print(f"  {name:30s}  (backward failed: {e})")
                    continue
                gnorm_abs = in_proj.grad.detach().norm().item() if in_proj.grad is not None else 0.0
                tval = term.detach().item()
                accelerator.print(
                    f"  {name:30s}  raw={tval:.6f}  grad_norm={gnorm_abs:.6e}"
                )
            if in_proj.grad is not None:
                in_proj.grad.zero_()

        accelerator.backward(out["loss"])
        optimizer.step()
        accelerator.print("Smoke test PASSED.")
        return

    # --- Wandb (optional) ---
    wandb_run = None
    if accelerator.is_main_process:
        try:
            import wandb
            wandb_run = wandb.init(
                project=cfg.logging.project, name=cfg.logging.run_name,
            )
        except ImportError:
            pass

    # --- Calibration data + dynamic-weight update hook (v2.1 only) ---
    epoch_end_hook = None
    if feature_weight_state is not None and feature_weight_state.enable_dynamic:
        from piano.data.dataset import HOIDataset
        from piano.training.feature_groups import FEATURE_GROUPS as _FG
        from piano.utils.io_utils import load_json as _load_json

        cal_path = Path(dyn_cfg["calibration_clips"])
        cal_manifest = _load_json(cal_path)
        accelerator.print(
            f"Loaded calibration manifest: {cal_manifest['num_clips']} clips from {cal_path}"
        )

        # Pre-build calibration HOIDatasets per subset.
        cal_datasets: dict[str, HOIDataset] = {}
        cal_index: dict[str, dict[str, int]] = {}
        cal_subj_filter = _resolve_subject_split(cfg, "val")
        pseudo_label_subdir_local = cfg.data.get("pseudo_label_subdir", None)
        for entry in cfg.data.datasets:
            sub_dir = (str(Path(entry.root) / pseudo_label_subdir_local)
                       if pseudo_label_subdir_local is not None else None)
            ds = HOIDataset(
                root=entry.root,
                pseudo_label_dir=sub_dir,
                max_seq_length=cfg.data.max_seq_length,
                subject_id_filter=cal_subj_filter,
                augment=None,
                support_collapse_hand_support=True,
                surface_obj_pose=True,
            )
            cal_datasets[entry.name] = ds
            if hasattr(ds, "metadata"):
                cal_index[entry.name] = {
                    str(meta.get("seq_id", "")): i
                    for i, meta in enumerate(ds.metadata)
                }
            else:
                cal_index[entry.name] = {}

        # Use a local generator for the noise sampling at every update so
        # the residual is stable across epochs (only the model state
        # changes) without resetting the training RNG state.
        cal_seed = int(dyn_cfg.get("calibration_seed", 12345))
        cal_t = int(dyn_cfg.get("fixed_timestep", 200))     # mid-noise level

        def _calibration_eval(
            unwrapped_model: torch.nn.Module,
            unwrapped_obj_enc: torch.nn.Module,
        ) -> tuple[dict[str, float], dict[str, float]]:
            """Single-step denoising on calibration clips.

            Returns (per_group_RMSE, per_group_GT_std).
            """
            unwrapped_model.eval()
            unwrapped_obj_enc.eval()
            noise_gen = torch.Generator(device=device)
            noise_gen.manual_seed(cal_seed)

            n_groups = len(_FG)
            sum_se = {g.name: 0.0 for g in _FG}
            sum_gt = {g.name: 0.0 for g in _FG}
            sum_gt_sq = {g.name: 0.0 for g in _FG}
            count_dim = {g.name: 0 for g in _FG}

            for clip_meta in cal_manifest["clips"]:
                ds = cal_datasets[clip_meta["subset"]]
                seq_to_idx = cal_index.get(clip_meta["subset"], {})
                # Locate by seq_id. Prefer the manifest index when it still
                # matches the fixed val-filtered calibration dataset; fall
                # back to the precomputed seq_id index without scanning and
                # loading every sample.
                target_id = clip_meta["seq_id"]
                idx = clip_meta.get("clip_idx_in_filtered_dataset", -1)
                if not (0 <= idx < len(ds)):
                    idx = seq_to_idx.get(target_id, -1)
                elif seq_to_idx and seq_to_idx.get(target_id, idx) != idx:
                    idx = seq_to_idx.get(target_id, -1)
                if not (0 <= idx < len(ds)):
                    continue
                sample = ds[idx]
                if str(sample["seq_id"]) != target_id:
                    continue
                seq_len_s = int(sample["seq_len"].item())
                if seq_len_s < 5:
                    continue

                with torch.inference_mode():
                    batch = collate_hoi([sample])
                    motion_b = batch["motion"].to(device)
                    joints_b = batch["joints"].to(device)
                    cs_b = batch["contact_state"].to(device)
                    ctx_b = batch["contact_target_xyz"].to(device)
                    ph_b = batch["phase"].to(device)
                    sp_b = batch["support"].to(device)
                    obj_com_b = batch["obj_com_canonical"].to(device)
                    obj_rot6d_b = batch["obj_rot6d_canonical"].to(device)
                    object_pc_b = batch["object_pc"].to(device)

                    Tlen = motion_b.shape[1]
                    seq_idx_b = torch.arange(Tlen, device=device).unsqueeze(0)
                    seq_mask_b = (
                        seq_idx_b
                        < torch.tensor([seq_len_s], device=device).unsqueeze(1)
                    ).float()

                    phase_soft_b = F.one_hot(
                        ph_b.clamp_min(0).long(), z_dims.phase_classes,
                    ).float()
                    support_soft_b = F.one_hot(
                        sp_b.clamp_min(0).long(), z_dims.support_classes,
                    ).float()
                    z_int_b = pack_z_int(cs_b, ctx_b, phase_soft_b, support_soft_b, z_dims)
                    obj_traj_b = torch.cat([obj_com_b, obj_rot6d_b], dim=-1)
                    init_pose_b = joints_b[:, 0, :, :].reshape(1, -1)
                    text_features_b, _ = encode_text_per_token(clip_model, batch["text"], device)
                    obj_tokens_b = unwrapped_obj_enc(object_pc_b)

                    cond_b = {
                        "z_int": z_int_b,
                        "object_world_traj": obj_traj_b,
                        "init_pose": init_pose_b,
                        "text": text_features_b.float(),
                        "object_tokens": obj_tokens_b,
                    }
                    # Single-step denoising at fixed t (cheaper than full DDPM)
                    t_tensor = torch.full((1,), cal_t, device=device, dtype=torch.long)
                    noise = torch.randn(
                        motion_b.shape,
                        device=device,
                        dtype=motion_b.dtype,
                        generator=noise_gen,
                    )
                    x_t = unwrapped_model.diffusion.q_sample(motion_b, t_tensor, noise)
                    x0_pred = unwrapped_model.denoiser(x_t, t_tensor, cond_b, cond_drop_mask=None)

                    # Per-group RMSE numerator + denominator
                    err_sq = (x0_pred - motion_b).pow(2)            # (1, T, 263)
                    gt_sq = motion_b.pow(2)
                    gt_v = motion_b
                    mask3 = seq_mask_b.unsqueeze(-1)                # (1, T, 1)
                    for g in _FG:
                        e_grp = err_sq[..., g.lo:g.hi]
                        gt_grp = gt_v[..., g.lo:g.hi]
                        gt_sq_grp = gt_sq[..., g.lo:g.hi]
                        se = (e_grp * mask3).sum().item()
                        s = (gt_grp * mask3).sum().item()
                        s_sq = (gt_sq_grp * mask3).sum().item()
                        sum_se[g.name] += se
                        sum_gt[g.name] += s
                        sum_gt_sq[g.name] += s_sq
                        count_dim[g.name] += int(mask3.sum().item() * (g.hi - g.lo))

            unwrapped_model.train()
            unwrapped_obj_enc.train()

            rmse = {n: (sum_se[n] / max(count_dim[n], 1)) ** 0.5 for n in sum_se}
            mean_gt = {n: sum_gt[n] / max(count_dim[n], 1) for n in sum_gt}
            var_gt = {n: max(sum_gt_sq[n] / max(count_dim[n], 1) - mean_gt[n] ** 2, 0.0)
                      for n in sum_gt}
            std_gt = {n: var_gt[n] ** 0.5 for n in var_gt}
            return rmse, std_gt

        def epoch_end_hook(epoch, accelerator, model, global_step, output_dir,
                           metrics_appender, wandb_run):
            if not feature_weight_state.should_update(epoch):
                return
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_obj_enc = accelerator.unwrap_model(object_encoder)
            rmse, gt_std = _calibration_eval(unwrapped_model, unwrapped_obj_enc)
            log = feature_weight_state.update(
                epoch=epoch, group_rmse=rmse, group_gt_std=gt_std,
            )
            feature_weight_state.broadcast(accelerator)
            # Log to metrics + wandb + JSON snapshot
            if accelerator.is_main_process:
                metrics_appender("dynamic_weight_update", {
                    "epoch": epoch,
                    "global_step": global_step,
                    **{f"weight_{k}": v for k, v in log["new_w_g"].items()},
                    **{f"rmse_{k}": rmse[k] for k in rmse},
                    **{f"norm_err_{k}": log["norm_err_g"][k] for k in log["norm_err_g"]},
                })
                snap_path = Path(output_dir) / f"feature_weights_epoch_{epoch:04d}.json"
                snap_path.write_text(json.dumps(log, indent=2))
                if wandb_run is not None:
                    flat = {}
                    for k, v in log["new_w_g"].items():
                        flat[f"weight/{k}"] = v
                    for k, v in rmse.items():
                        flat[f"calib_rmse/{k}"] = v
                    wandb_run.log(flat, step=epoch)

    run_training_loop(
        accelerator=accelerator,
        model=model,
        dataloader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        step_fn=step_fn,
        num_epochs=int(cfg.training.num_epochs),
        output_dir=cfg.output_dir,
        log_every=int(cfg.logging.log_every_n_steps),
        save_every_epochs=int(cfg.logging.save_every_n_epochs),
        max_grad_norm=float(cfg.training.max_grad_norm),
        wandb_run=wandb_run,
        extra_modules={"object_encoder": object_encoder},
        epoch_end_hook=epoch_end_hook,
        extra_state_fn=(
            (lambda: {"feature_weight_state": feature_weight_state.state_dict()})
            if feature_weight_state is not None else None
        ),
        val_dataloader=val_loader,
        val_every_epochs=int(cfg.training.get("val_every_epochs", 0)),
        val_best_key=str(cfg.training.get("val_best_key", "loss")),
    )


if __name__ == "__main__":
    main()
