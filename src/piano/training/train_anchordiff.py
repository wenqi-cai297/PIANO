"""Stage B (new): Train PIANO-AnchorDiff.

Anchor-conditioned continuous motion diffusion. Replaces the closed
Stage B track trained with classifier-free guidance dropout. Operates on
HumanML3D motion_135 with object-trajectory + object-pc + text + init_pose
+ Stage-1 Coarse-v1 + Round-29 typed C/I/S/B conditioning.

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
import os
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
)
from piano.models.object_encoder import ObjectEncoder
from piano.training.anchor_consistency_loss import (
    AnchorConsistencyConfig,
    PART_TO_JOINT,
    anchor_consistency_loss_world_joints,
    lift_object_local_to_world,
)
from piano.training.anchordiff_geometric_losses import (
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
            oracle_hint_fps=float(cfg.data.get("oracle_hint_fps", 20.0)),
            surface_temporal_aux_fields=bool(
                cfg.data.get("surface_temporal_aux_fields", False)
            ),
            r29_coarse_variant=str(
                cfg.data.get("r29_coarse_variant", "C23")
            ),
            r29_interaction_variant=str(
                cfg.data.get("r29_interaction_variant", "I0")
            ),
            r29_support_variant=str(
                cfg.data.get("r29_support_variant", "S0")
            ),
            r29_body_variant=str(
                cfg.data.get("r29_body_variant", "B0")
            ),
            r29_body_coord_frame=(
                str(cfg.data.get("r29_body_coord_frame"))
                if cfg.data.get("r29_body_coord_frame") is not None
                else None
            ),
            r29_body_energy_threshold=float(
                cfg.data.get("r29_body_energy_threshold", 0.05)
            ),
            r29_body_lowpass_window=int(
                cfg.data.get("r29_body_lowpass_window", 9)
            ),
            r29_hand_offset_clamp_m=float(
                cfg.data.get("r29_hand_offset_clamp_m", 2.0)
            ),
        )
        datasets.append(ds)
    return ConcatDataset(datasets)


# ---------------------------------------------------------------------------
# Step function
# ---------------------------------------------------------------------------


def build_anchordiff_step_fn(
    model: MotionAnchorDiff,
    object_encoder: ObjectEncoder,
    clip_model: torch.nn.Module | None,
    anchor_cfg: AnchorConsistencyConfig,
    device: torch.device,
    motion_representation: str = "smpl_pose_135_plan",
    world_joint_velocity_weight: float = 0.0,
    object_traj_dim: int = 9,
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
    """Build the AnchorDiff step_fn closure."""

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

        # --- Object trajectory channel. v1-v4a use object pose only
        # (3 COM + 6 rot6d). v4b appends the five body-part anchor targets
        # already transformed into world frame (5 * 3), so Stage A's
        # predictor signal reaches the denoiser in task-space coordinates.
        # v8 keyframed: appends 6-keyjoint positions only at keyframe
        # frames + 1-D keyframe indicator (zero elsewhere).
        object_traj = _build_object_traj(
            obj_com=obj_com,
            obj_rot6d=obj_rot6d,
            contact_target_xyz=contact_target_xyz,
            obj_pos_world=obj_pos_world,
            obj_rot_world=obj_rot_world,
        )

        # --- Init pose: SMPL-22 frame 0 (optional — Tier-2 ablation) ---
        _denoiser = model.module.denoiser if hasattr(model, "module") else model.denoiser
        if _denoiser.use_init_pose:
            init_pose = joints[:, 0, :, :].reshape(B, -1)              # (B, 66)
        else:
            init_pose = None

        # --- Text features via CLIP per-token (optional — Tier-2 ablation) ---
        if _denoiser.use_text:
            text_features, _text_mask = encode_text_per_token(
                clip_model, batch["text"], device,
            )                                                          # (B, L, text_dim)
            text_features = text_features.float()
        else:
            text_features = None

        # --- Object tokens via PointNet++ encoder ---
        obj_tokens = object_encoder(object_pc)                     # (B, N, obj_dim)

        # R29 PLAN-only condition mode: zero the dense contact-target suffix
        # of object_traj if a wider variant is used (first 9 dims are object
        # pose, kept as the only object-pose channel).
        if object_traj.shape[-1] >= 24:
            object_traj = object_traj.clone()
            object_traj[..., 9:] = 0.0

        cond = {
            "object_world_traj": object_traj,
            "object_tokens": obj_tokens,
        }
        if init_pose is not None:
            cond["init_pose"] = init_pose
        if text_features is not None:
            cond["text"] = text_features

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

        # ── Round-29: typed Stage-2 condition bundle ──
        # The dataset's Stage2ConditionBundle is surfaced as four
        # optional keys; each is present iff the corresponding family
        # variant is enabled. The model's Round29CondInjectionModule
        # raises a KeyError if a required family key is missing, so
        # we just forward whatever the dataset produced.
        for _r29_key in (
            "stage2_coarse_extra",
            "stage2_interaction",
            "stage2_support",
            "stage2_body_refine",
        ):
            if _r29_key in batch:
                cond[_r29_key] = batch[_r29_key].to(device).float()

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
        weighted = mse_per_dim.sum(-1)                              # (B, T)

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

        # World-frame velocity loss on the full 135-D motion vector.
        if world_joint_velocity_weight > 0.0:
            wv_pred, wv_target = x0_pred.float(), x0_target.float()
            loss_world_vel = feature_velocity_loss(wv_pred, wv_target, seq_mask.float())
        else:
            loss_world_vel = torch.zeros((), device=device, dtype=x0_pred.dtype)

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
        # Round-29 failure-targeted ablation (R2/R3/R4/R5) — pre-allocated
        # zero scalars so the out dict is shape-consistent across variants.
        loss_r29_gait_one_foot_supp = torch.zeros((), device=device, dtype=motion.dtype)
        loss_r29_gait_pred_stance_v = torch.zeros((), device=device, dtype=motion.dtype)
        loss_r29_gait_ankle_smooth_v = torch.zeros((), device=device, dtype=motion.dtype)
        loss_r29_gait_antiphase = torch.zeros((), device=device, dtype=motion.dtype)
        loss_r29_s4_stance_bce_v = torch.zeros((), device=device, dtype=motion.dtype)
        loss_r29_s4_footstep_v = torch.zeros((), device=device, dtype=motion.dtype)
        loss_r29_contact_lock_off = torch.zeros((), device=device, dtype=motion.dtype)
        loss_r29_contact_lock_drift = torch.zeros((), device=device, dtype=motion.dtype)
        loss_r29_contact_lock_track = torch.zeros((), device=device, dtype=motion.dtype)
        # Round-29 next-baseline ablation (G1) — phase-free gait losses.
        loss_r29_gait_soft_stance_v = torch.zeros((), device=device, dtype=motion.dtype)
        loss_r29_gait_trans_rate = torch.zeros((), device=device, dtype=motion.dtype)
        loss_r29_gait_duty = torch.zeros((), device=device, dtype=motion.dtype)
        loss_r29_gait_both_state = torch.zeros((), device=device, dtype=motion.dtype)
        # Round-28 consistency losses (prompt §7.3 / §7.4).
        loss_hint_contact_cons = torch.zeros((), device=device, dtype=motion.dtype)
        loss_body_action_cons = torch.zeros((), device=device, dtype=motion.dtype)
        # Round-29 condition-consistency losses (analyses/2026-05-27_round29_loss_strategy_ablation_prompt_for_claude_code.md).
        loss_r29_interaction_cons = torch.zeros((), device=device, dtype=motion.dtype)
        loss_r29_support_air = torch.zeros((), device=device, dtype=motion.dtype)
        loss_r29_support_stance_vel = torch.zeros((), device=device, dtype=motion.dtype)
        # Round-29 swing clearance (post-Codex v1 review): forces swing
        # ankle off the floor during walking-non-stance frames.
        loss_r29_swing_clear = torch.zeros((), device=device, dtype=motion.dtype)
        if temporal_loss_cfg is not None and (
            float(getattr(temporal_loss_cfg, "contact_rel_offset_weight", 0.0)) > 0.0
            or float(getattr(temporal_loss_cfg, "contact_drift_weight", 0.0)) > 0.0
            or float(getattr(temporal_loss_cfg, "contact_tracking_weight", 0.0)) > 0.0
            or float(getattr(temporal_loss_cfg, "gait_both_airborne_weight", 0.0)) > 0.0
            or float(getattr(temporal_loss_cfg, "gait_stance_velocity_weight", 0.0)) > 0.0
            or float(getattr(temporal_loss_cfg, "hint_contact_consistency_weight", 0.0)) > 0.0
            or float(getattr(temporal_loss_cfg, "body_action_consistency_weight", 0.0)) > 0.0
            or float(getattr(temporal_loss_cfg, "r29_interaction_consistency_weight", 0.0)) > 0.0
            or float(getattr(temporal_loss_cfg, "r29_support_both_airborne_weight", 0.0)) > 0.0
            or float(getattr(temporal_loss_cfg, "r29_support_stance_velocity_weight", 0.0)) > 0.0
            or float(getattr(temporal_loss_cfg, "r29_swing_clearance_weight", 0.0)) > 0.0
            or float(getattr(temporal_loss_cfg, "r29_gait_one_foot_support_weight", 0.0)) > 0.0
            or float(getattr(temporal_loss_cfg, "r29_gait_pred_stance_velocity_weight", 0.0)) > 0.0
            or float(getattr(temporal_loss_cfg, "r29_gait_ankle_smooth_weight", 0.0)) > 0.0
            or float(getattr(temporal_loss_cfg, "r29_gait_antiphase_corr_weight", 0.0)) > 0.0
            or float(getattr(temporal_loss_cfg, "r29_s4_stance_bce_weight", 0.0)) > 0.0
            or float(getattr(temporal_loss_cfg, "r29_s4_footstep_target_weight", 0.0)) > 0.0
            or float(getattr(temporal_loss_cfg, "r29_contact_lock_offset_weight", 0.0)) > 0.0
            or float(getattr(temporal_loss_cfg, "r29_contact_lock_segment_drift_weight", 0.0)) > 0.0
            or float(getattr(temporal_loss_cfg, "r29_contact_lock_tracking_weight", 0.0)) > 0.0
            or float(getattr(temporal_loss_cfg, "r29_gait_soft_stance_velocity_weight", 0.0)) > 0.0
            or float(getattr(temporal_loss_cfg, "r29_gait_transition_rate_weight", 0.0)) > 0.0
            or float(getattr(temporal_loss_cfg, "r29_gait_duty_cycle_weight", 0.0)) > 0.0
            or float(getattr(temporal_loss_cfg, "r29_gait_both_state_match_weight", 0.0)) > 0.0
        ):
            from piano.training.temporal_interaction_losses import (
                loss_body_action_consistency,
                loss_contact_drift_smoothl1,
                loss_contact_rel_offset_smoothl1,
                loss_contact_tracking_projection,
                loss_gait_both_airborne,
                loss_gait_stance_velocity,
                loss_hint_contact_consistency,
                loss_r29_interaction_consistency,
                loss_r29_support_both_airborne,
                loss_r29_support_stance_velocity,
                loss_r29_swing_clearance,
                loss_r29_gait_one_foot_support,
                loss_r29_gait_pred_stance_velocity,
                loss_r29_gait_ankle_smooth,
                loss_r29_gait_antiphase_corr,
                loss_r29_s4_stance_bce,
                loss_r29_s4_footstep_target,
                loss_r29_contact_lock_offset,
                loss_r29_contact_lock_segment_drift,
                loss_r29_contact_lock_tracking,
                loss_r29_gait_soft_stance_velocity,
                loss_r29_gait_transition_rate,
                loss_r29_gait_duty_cycle,
                loss_r29_gait_both_state_match,
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

            # Round-29 P0 — interaction consistency. Pull pred wrist
            # toward the I3 target_offset channel (object-local), masked
            # by the I3 hand_contact channel. Self-contained on the R29
            # condition; does NOT need data.surface_temporal_aux_fields.
            if float(getattr(temporal_loss_cfg, "r29_interaction_consistency_weight", 0.0)) > 0.0:
                if "stage2_interaction" not in cond:
                    raise KeyError(
                        "r29_interaction_consistency_weight > 0 but cond is "
                        "missing stage2_interaction. Enable an I-family variant "
                        "via data.r29_interaction_variant (e.g. I3-contact-offset-masked)."
                    )
                loss_r29_interaction_cons = loss_r29_interaction_consistency(
                    pred_joints=jpf,
                    object_positions=op_f, object_rotations=or_f,
                    stage2_interaction=cond["stage2_interaction"].float(),
                    cfg=temporal_loss_cfg, seq_mask=sm_f,
                    hand_offset_clamp_m=float(
                        getattr(temporal_loss_cfg, "r29_hand_offset_clamp_m", 2.0)
                    ),
                )

            # Round-29 P0 — support both-airborne. Walking mask comes
            # from stage2_support[..., 4:5], not from a dataset aux field.
            if float(getattr(temporal_loss_cfg, "r29_support_both_airborne_weight", 0.0)) > 0.0:
                if "stage2_support" not in cond:
                    raise KeyError(
                        "r29_support_both_airborne_weight > 0 but cond is "
                        "missing stage2_support. Enable an S-family variant "
                        "with dim>=5 via data.r29_support_variant (S1/S2/S3/S4)."
                    )
                loss_r29_support_air = loss_r29_support_both_airborne(
                    pred_joints=jpf, gt_joints=jgf,
                    stage2_support=cond["stage2_support"].float(),
                    cfg=temporal_loss_cfg, seq_mask=sm_f,
                )

            # Round-29 P0 — support stance velocity. Foot stance + walking
            # mask both come from stage2_support.
            if float(getattr(temporal_loss_cfg, "r29_support_stance_velocity_weight", 0.0)) > 0.0:
                if "stage2_support" not in cond:
                    raise KeyError(
                        "r29_support_stance_velocity_weight > 0 but cond is "
                        "missing stage2_support. Enable an S-family variant "
                        "with dim>=5 via data.r29_support_variant (S1/S2/S3/S4)."
                    )
                loss_r29_support_stance_vel = loss_r29_support_stance_velocity(
                    pred_joints=jpf,
                    stage2_support=cond["stage2_support"].float(),
                    fps=float(fps), seq_mask=sm_f,
                )

            # Round-29 P0+ — swing clearance. Forces swing ankle above
            # the per-clip floor during walking-non-stance frames. Required
            # because both_airborne+stance_velocity alone do not prevent
            # the "both feet planted" trivial solution v1 produced.
            if float(getattr(temporal_loss_cfg, "r29_swing_clearance_weight", 0.0)) > 0.0:
                if "stage2_support" not in cond:
                    raise KeyError(
                        "r29_swing_clearance_weight > 0 but cond is missing "
                        "stage2_support. Enable an S-family variant with "
                        "dim>=5 via data.r29_support_variant (S1/S2/S3/S4)."
                    )
                loss_r29_swing_clear = loss_r29_swing_clearance(
                    pred_joints=jpf, gt_joints=jgf,
                    stage2_support=cond["stage2_support"].float(),
                    cfg=temporal_loss_cfg, seq_mask=sm_f,
                )

            # Round-29 failure-targeted ablation R2 — behavior-level gait
            # losses. Use S4 walking_mask only (not GT stance) so the model
            # can pick either left-first or right-first phase.
            r2_active = (
                float(getattr(temporal_loss_cfg, "r29_gait_one_foot_support_weight", 0.0)) > 0.0
                or float(getattr(temporal_loss_cfg, "r29_gait_pred_stance_velocity_weight", 0.0)) > 0.0
                or float(getattr(temporal_loss_cfg, "r29_gait_ankle_smooth_weight", 0.0)) > 0.0
                or float(getattr(temporal_loss_cfg, "r29_gait_antiphase_corr_weight", 0.0)) > 0.0
            )
            if r2_active and "stage2_support" not in cond:
                raise KeyError(
                    "R2 behavior-gait losses require cond['stage2_support'] "
                    "(walking_mask channel). Enable an S-family variant with "
                    "dim>=5 via data.r29_support_variant (S1/S2/S3/S4)."
                )
            if float(getattr(temporal_loss_cfg, "r29_gait_one_foot_support_weight", 0.0)) > 0.0:
                loss_r29_gait_one_foot_supp = loss_r29_gait_one_foot_support(
                    pred_joints=jpf, gt_joints=jgf,
                    stage2_support=cond["stage2_support"].float(),
                    cfg=temporal_loss_cfg, seq_mask=sm_f,
                )
            if float(getattr(temporal_loss_cfg, "r29_gait_pred_stance_velocity_weight", 0.0)) > 0.0:
                loss_r29_gait_pred_stance_v = loss_r29_gait_pred_stance_velocity(
                    pred_joints=jpf, gt_joints=jgf,
                    stage2_support=cond["stage2_support"].float(),
                    cfg=temporal_loss_cfg, fps=float(fps), seq_mask=sm_f,
                )
            if float(getattr(temporal_loss_cfg, "r29_gait_ankle_smooth_weight", 0.0)) > 0.0:
                loss_r29_gait_ankle_smooth_v = loss_r29_gait_ankle_smooth(
                    pred_joints=jpf,
                    stage2_support=cond["stage2_support"].float(),
                    seq_mask=sm_f,
                )
            if float(getattr(temporal_loss_cfg, "r29_gait_antiphase_corr_weight", 0.0)) > 0.0:
                loss_r29_gait_antiphase = loss_r29_gait_antiphase_corr(
                    pred_joints=jpf,
                    stage2_support=cond["stage2_support"].float(),
                    cfg=temporal_loss_cfg, seq_mask=sm_f,
                )

            # Round-29 failure-targeted ablation R3 — exact S4 execution.
            # Requires S4 layout (dim >= 13) since footstep target lives
            # at stage2_support[..., 9:13].
            r3_active = (
                float(getattr(temporal_loss_cfg, "r29_s4_stance_bce_weight", 0.0)) > 0.0
                or float(getattr(temporal_loss_cfg, "r29_s4_footstep_target_weight", 0.0)) > 0.0
            )
            if r3_active and "stage2_support" not in cond:
                raise KeyError(
                    "R3 exact-S4 losses require cond['stage2_support']. "
                    "Enable data.r29_support_variant=S4-S1-phase-footstep."
                )
            if float(getattr(temporal_loss_cfg, "r29_s4_stance_bce_weight", 0.0)) > 0.0:
                loss_r29_s4_stance_bce_v = loss_r29_s4_stance_bce(
                    pred_joints=jpf, gt_joints=jgf,
                    stage2_support=cond["stage2_support"].float(),
                    cfg=temporal_loss_cfg, seq_mask=sm_f,
                )
            if float(getattr(temporal_loss_cfg, "r29_s4_footstep_target_weight", 0.0)) > 0.0:
                loss_r29_s4_footstep_v = loss_r29_s4_footstep_target(
                    pred_joints=jpf, gt_joints=jgf,
                    stage2_support=cond["stage2_support"].float(),
                    seq_mask=sm_f,
                )

            # Round-29 next-baseline ablation G1 — phase-free gait losses.
            # Address R2's height-only loophole: foot is stance-like only when
            # low AND slow. Match aggregate gait stats (transition rate, sorted
            # duty cycle, both-state) without per-frame left/right alignment.
            g1_active = (
                float(getattr(temporal_loss_cfg, "r29_gait_soft_stance_velocity_weight", 0.0)) > 0.0
                or float(getattr(temporal_loss_cfg, "r29_gait_transition_rate_weight", 0.0)) > 0.0
                or float(getattr(temporal_loss_cfg, "r29_gait_duty_cycle_weight", 0.0)) > 0.0
                or float(getattr(temporal_loss_cfg, "r29_gait_both_state_match_weight", 0.0)) > 0.0
            )
            if g1_active and "stage2_support" not in cond:
                raise KeyError(
                    "G1 phase-free gait losses require cond['stage2_support'] "
                    "(walking_mask + L/R stance channels). Enable an S-family "
                    "variant with dim>=5 via data.r29_support_variant."
                )
            if float(getattr(temporal_loss_cfg, "r29_gait_soft_stance_velocity_weight", 0.0)) > 0.0:
                loss_r29_gait_soft_stance_v = loss_r29_gait_soft_stance_velocity(
                    pred_joints=jpf, gt_joints=jgf,
                    stage2_support=cond["stage2_support"].float(),
                    cfg=temporal_loss_cfg, fps=float(fps), seq_mask=sm_f,
                )
            if float(getattr(temporal_loss_cfg, "r29_gait_transition_rate_weight", 0.0)) > 0.0:
                loss_r29_gait_trans_rate = loss_r29_gait_transition_rate(
                    pred_joints=jpf, gt_joints=jgf,
                    stage2_support=cond["stage2_support"].float(),
                    cfg=temporal_loss_cfg, fps=float(fps), seq_mask=sm_f,
                )
            if float(getattr(temporal_loss_cfg, "r29_gait_duty_cycle_weight", 0.0)) > 0.0:
                loss_r29_gait_duty = loss_r29_gait_duty_cycle(
                    pred_joints=jpf, gt_joints=jgf,
                    stage2_support=cond["stage2_support"].float(),
                    cfg=temporal_loss_cfg, fps=float(fps), seq_mask=sm_f,
                )
            if float(getattr(temporal_loss_cfg, "r29_gait_both_state_match_weight", 0.0)) > 0.0:
                loss_r29_gait_both_state = loss_r29_gait_both_state_match(
                    pred_joints=jpf, gt_joints=jgf,
                    stage2_support=cond["stage2_support"].float(),
                    cfg=temporal_loss_cfg, fps=float(fps), seq_mask=sm_f,
                )

            # Round-29 failure-targeted ablation R4 / R5 — contact-lock
            # losses. Driven by I3 (dim=8) or I5 (dim=20).
            cl_active = (
                float(getattr(temporal_loss_cfg, "r29_contact_lock_offset_weight", 0.0)) > 0.0
                or float(getattr(temporal_loss_cfg, "r29_contact_lock_segment_drift_weight", 0.0)) > 0.0
                or float(getattr(temporal_loss_cfg, "r29_contact_lock_tracking_weight", 0.0)) > 0.0
            )
            if cl_active and "stage2_interaction" not in cond:
                raise KeyError(
                    "R4/R5 contact-lock losses require cond['stage2_interaction']. "
                    "Enable an I-family variant via data.r29_interaction_variant "
                    "(I3 dim=8 or I5 dim=20)."
                )
            if float(getattr(temporal_loss_cfg, "r29_contact_lock_offset_weight", 0.0)) > 0.0:
                loss_r29_contact_lock_off = loss_r29_contact_lock_offset(
                    pred_joints=jpf, object_positions=op_f, object_rotations=or_f,
                    stage2_interaction=cond["stage2_interaction"].float(),
                    cfg=temporal_loss_cfg, seq_mask=sm_f,
                    hand_offset_clamp_m=float(
                        getattr(temporal_loss_cfg, "r29_hand_offset_clamp_m", 2.0)
                    ),
                )
            if float(getattr(temporal_loss_cfg, "r29_contact_lock_segment_drift_weight", 0.0)) > 0.0:
                loss_r29_contact_lock_drift = loss_r29_contact_lock_segment_drift(
                    pred_joints=jpf, object_positions=op_f, object_rotations=or_f,
                    stage2_interaction=cond["stage2_interaction"].float(),
                    cfg=temporal_loss_cfg, seq_mask=sm_f,
                    hand_offset_clamp_m=float(
                        getattr(temporal_loss_cfg, "r29_hand_offset_clamp_m", 2.0)
                    ),
                )
            if float(getattr(temporal_loss_cfg, "r29_contact_lock_tracking_weight", 0.0)) > 0.0:
                loss_r29_contact_lock_track = loss_r29_contact_lock_tracking(
                    pred_joints=jpf, object_positions=op_f,
                    stage2_interaction=cond["stage2_interaction"].float(),
                    cfg=temporal_loss_cfg, seq_mask=sm_f,
                )

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
            + world_joint_velocity_weight * loss_world_vel
            + pos_loss_weight * loss_pos_full
            + anchor_joint_pos_weight * loss_anchor_joint_pos
            + anchor_joint_vel_weight * loss_anchor_joint_vel
            + stable_root_vel_weight * loss_stable_root_vel
            + stable_root_acc_weight * loss_stable_root_acc
            + stable_local_vel_weight * loss_stable_local_vel
            + stable_local_acc_weight * loss_stable_local_acc
            + stable_local_vel_cm_weight * loss_stable_local_vel_cm
            + stable_local_acc_cm_weight * loss_stable_local_acc_cm
            + stable_local_speed_moment_weight * loss_stable_local_speed_moment
            # Round-27 Tier-0B temporal interaction losses (zero by default).
            # Round-28 adds two consistency terms (also zero by default).
            # Round-29 adds three condition-consistency terms (also zero by default).
            + (
                float(temporal_loss_cfg.contact_rel_offset_weight) * loss_contact_rel
                + float(temporal_loss_cfg.contact_drift_weight) * loss_contact_drift
                + float(temporal_loss_cfg.contact_tracking_weight) * loss_contact_track
                + float(temporal_loss_cfg.gait_both_airborne_weight) * loss_gait_air
                + float(temporal_loss_cfg.gait_stance_velocity_weight) * loss_gait_stance_vel
                + float(getattr(temporal_loss_cfg, "hint_contact_consistency_weight", 0.0)) * loss_hint_contact_cons
                + float(getattr(temporal_loss_cfg, "body_action_consistency_weight", 0.0)) * loss_body_action_cons
                + float(getattr(temporal_loss_cfg, "r29_interaction_consistency_weight", 0.0)) * loss_r29_interaction_cons
                + float(getattr(temporal_loss_cfg, "r29_support_both_airborne_weight", 0.0)) * loss_r29_support_air
                + float(getattr(temporal_loss_cfg, "r29_support_stance_velocity_weight", 0.0)) * loss_r29_support_stance_vel
                + float(getattr(temporal_loss_cfg, "r29_swing_clearance_weight", 0.0)) * loss_r29_swing_clear
                # Round-29 failure-targeted ablation (R2/R3/R4/R5).
                + float(getattr(temporal_loss_cfg, "r29_gait_one_foot_support_weight", 0.0)) * loss_r29_gait_one_foot_supp
                + float(getattr(temporal_loss_cfg, "r29_gait_pred_stance_velocity_weight", 0.0)) * loss_r29_gait_pred_stance_v
                + float(getattr(temporal_loss_cfg, "r29_gait_ankle_smooth_weight", 0.0)) * loss_r29_gait_ankle_smooth_v
                + float(getattr(temporal_loss_cfg, "r29_gait_antiphase_corr_weight", 0.0)) * loss_r29_gait_antiphase
                + float(getattr(temporal_loss_cfg, "r29_s4_stance_bce_weight", 0.0)) * loss_r29_s4_stance_bce_v
                + float(getattr(temporal_loss_cfg, "r29_s4_footstep_target_weight", 0.0)) * loss_r29_s4_footstep_v
                + float(getattr(temporal_loss_cfg, "r29_contact_lock_offset_weight", 0.0)) * loss_r29_contact_lock_off
                + float(getattr(temporal_loss_cfg, "r29_contact_lock_segment_drift_weight", 0.0)) * loss_r29_contact_lock_drift
                + float(getattr(temporal_loss_cfg, "r29_contact_lock_tracking_weight", 0.0)) * loss_r29_contact_lock_track
                # Round-29 next-baseline ablation G1 — phase-free gait losses.
                + float(getattr(temporal_loss_cfg, "r29_gait_soft_stance_velocity_weight", 0.0)) * loss_r29_gait_soft_stance_v
                + float(getattr(temporal_loss_cfg, "r29_gait_transition_rate_weight", 0.0)) * loss_r29_gait_trans_rate
                + float(getattr(temporal_loss_cfg, "r29_gait_duty_cycle_weight", 0.0)) * loss_r29_gait_duty
                + float(getattr(temporal_loss_cfg, "r29_gait_both_state_match_weight", 0.0)) * loss_r29_gait_both_state
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
            "loss_vel": loss_world_vel.detach(),
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
            # Round-29 condition-consistency loss components.
            "loss_r29_interaction_cons": loss_r29_interaction_cons.detach(),
            "loss_r29_support_air": loss_r29_support_air.detach(),
            "loss_r29_support_stance_vel": loss_r29_support_stance_vel.detach(),
            "weighted_r29_interaction_cons": (
                float(getattr(temporal_loss_cfg, "r29_interaction_consistency_weight", 0.0))
                * loss_r29_interaction_cons
            ).detach() if temporal_loss_cfg is not None else loss_r29_interaction_cons.detach(),
            "weighted_r29_support_air": (
                float(getattr(temporal_loss_cfg, "r29_support_both_airborne_weight", 0.0))
                * loss_r29_support_air
            ).detach() if temporal_loss_cfg is not None else loss_r29_support_air.detach(),
            "weighted_r29_support_stance_vel": (
                float(getattr(temporal_loss_cfg, "r29_support_stance_velocity_weight", 0.0))
                * loss_r29_support_stance_vel
            ).detach() if temporal_loss_cfg is not None else loss_r29_support_stance_vel.detach(),
            "loss_r29_swing_clear": loss_r29_swing_clear.detach(),
            "weighted_r29_swing_clear": (
                float(getattr(temporal_loss_cfg, "r29_swing_clearance_weight", 0.0))
                * loss_r29_swing_clear
            ).detach() if temporal_loss_cfg is not None else loss_r29_swing_clear.detach(),
            # Round-29 failure-targeted ablation R2 — behavior-level gait.
            "loss_r29_gait_one_foot_support": loss_r29_gait_one_foot_supp.detach(),
            "loss_r29_gait_pred_stance_vel": loss_r29_gait_pred_stance_v.detach(),
            "loss_r29_gait_ankle_smooth": loss_r29_gait_ankle_smooth_v.detach(),
            "loss_r29_gait_antiphase_corr": loss_r29_gait_antiphase.detach(),
            "weighted_r29_gait_one_foot_support": (
                float(getattr(temporal_loss_cfg, "r29_gait_one_foot_support_weight", 0.0))
                * loss_r29_gait_one_foot_supp
            ).detach() if temporal_loss_cfg is not None else loss_r29_gait_one_foot_supp.detach(),
            "weighted_r29_gait_pred_stance_vel": (
                float(getattr(temporal_loss_cfg, "r29_gait_pred_stance_velocity_weight", 0.0))
                * loss_r29_gait_pred_stance_v
            ).detach() if temporal_loss_cfg is not None else loss_r29_gait_pred_stance_v.detach(),
            "weighted_r29_gait_ankle_smooth": (
                float(getattr(temporal_loss_cfg, "r29_gait_ankle_smooth_weight", 0.0))
                * loss_r29_gait_ankle_smooth_v
            ).detach() if temporal_loss_cfg is not None else loss_r29_gait_ankle_smooth_v.detach(),
            "weighted_r29_gait_antiphase_corr": (
                float(getattr(temporal_loss_cfg, "r29_gait_antiphase_corr_weight", 0.0))
                * loss_r29_gait_antiphase
            ).detach() if temporal_loss_cfg is not None else loss_r29_gait_antiphase.detach(),
            # Round-29 failure-targeted ablation R3 — exact S4 execution.
            "loss_r29_s4_stance_bce": loss_r29_s4_stance_bce_v.detach(),
            "loss_r29_s4_footstep_target": loss_r29_s4_footstep_v.detach(),
            "weighted_r29_s4_stance_bce": (
                float(getattr(temporal_loss_cfg, "r29_s4_stance_bce_weight", 0.0))
                * loss_r29_s4_stance_bce_v
            ).detach() if temporal_loss_cfg is not None else loss_r29_s4_stance_bce_v.detach(),
            "weighted_r29_s4_footstep_target": (
                float(getattr(temporal_loss_cfg, "r29_s4_footstep_target_weight", 0.0))
                * loss_r29_s4_footstep_v
            ).detach() if temporal_loss_cfg is not None else loss_r29_s4_footstep_v.detach(),
            # Round-29 failure-targeted ablation R4 / R5 — contact-lock.
            "loss_r29_contact_lock_offset": loss_r29_contact_lock_off.detach(),
            "loss_r29_contact_lock_segment_drift": loss_r29_contact_lock_drift.detach(),
            "loss_r29_contact_lock_tracking": loss_r29_contact_lock_track.detach(),
            "weighted_r29_contact_lock_offset": (
                float(getattr(temporal_loss_cfg, "r29_contact_lock_offset_weight", 0.0))
                * loss_r29_contact_lock_off
            ).detach() if temporal_loss_cfg is not None else loss_r29_contact_lock_off.detach(),
            "weighted_r29_contact_lock_segment_drift": (
                float(getattr(temporal_loss_cfg, "r29_contact_lock_segment_drift_weight", 0.0))
                * loss_r29_contact_lock_drift
            ).detach() if temporal_loss_cfg is not None else loss_r29_contact_lock_drift.detach(),
            "weighted_r29_contact_lock_tracking": (
                float(getattr(temporal_loss_cfg, "r29_contact_lock_tracking_weight", 0.0))
                * loss_r29_contact_lock_track
            ).detach() if temporal_loss_cfg is not None else loss_r29_contact_lock_track.detach(),
            # Round-29 next-baseline ablation G1 — phase-free gait.
            "loss_r29_gait_soft_stance_velocity": loss_r29_gait_soft_stance_v.detach(),
            "loss_r29_gait_transition_rate": loss_r29_gait_trans_rate.detach(),
            "loss_r29_gait_duty_cycle": loss_r29_gait_duty.detach(),
            "loss_r29_gait_both_state_match": loss_r29_gait_both_state.detach(),
            "weighted_r29_gait_soft_stance_velocity": (
                float(getattr(temporal_loss_cfg, "r29_gait_soft_stance_velocity_weight", 0.0))
                * loss_r29_gait_soft_stance_v
            ).detach() if temporal_loss_cfg is not None else loss_r29_gait_soft_stance_v.detach(),
            "weighted_r29_gait_transition_rate": (
                float(getattr(temporal_loss_cfg, "r29_gait_transition_rate_weight", 0.0))
                * loss_r29_gait_trans_rate
            ).detach() if temporal_loss_cfg is not None else loss_r29_gait_trans_rate.detach(),
            "weighted_r29_gait_duty_cycle": (
                float(getattr(temporal_loss_cfg, "r29_gait_duty_cycle_weight", 0.0))
                * loss_r29_gait_duty
            ).detach() if temporal_loss_cfg is not None else loss_r29_gait_duty.detach(),
            "weighted_r29_gait_both_state_match": (
                float(getattr(temporal_loss_cfg, "r29_gait_both_state_match_weight", 0.0))
                * loss_r29_gait_both_state
            ).detach() if temporal_loss_cfg is not None else loss_r29_gait_both_state.detach(),
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

        # Round-29 typed-condition diagnostics. The r29_inject module
        _inner_model = _model.module if hasattr(_model, "module") else _model
        _denoiser = getattr(_inner_model, "denoiser", None)
        # caches its own scalar stats during forward; we surface them
        # alongside the r28_* keys so the generic trainer's grad-norm
        # helper can fire on the r29_grad_norm_* groups.
        _r29 = getattr(_denoiser, "r29_inject", None) if _denoiser is not None else None
        if _r29 is not None:
            for _k, _v in _r29.last_stats().items():
                if isinstance(_v, torch.Tensor) and _v.numel() == 1:
                    out[_k] = _v.detach()

        # v12 architecture-utilization metrics (per
        # analyses/2026-05-11_v12_implementation_doc.md §2.4 / §3.2).
        # Cheap to compute (norm of weight tensors), useful for diagnosing
        # whether the new conditioning pathways are actually engaging.
        denoiser = getattr(model, "denoiser", None)
        if denoiser is not None:
            with torch.no_grad():
                out["v12_input_proj_norm_motion"] = (
                    denoiser.v12_input_proj.motion_proj.weight.norm().detach()
                )
                out["v12_input_proj_norm_obj"] = (
                    denoiser.v12_input_proj.obj_proj.weight.norm().detach()
                )
                for i, block in enumerate(denoiser.v12_blocks):
                    out[f"v12_adaLN_norm_layer{i}"] = (
                        block.adaLN_modulation[-1].weight.norm().detach()
                    )
                fl = denoiser.v12_final_layer
                out["v12_final_adaLN_norm"] = (
                    fl.adaLN_modulation[-1].weight.norm().detach()
                )
                out["v12_final_linear_norm"] = (
                    fl.linear.weight.norm().detach()
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
    denoiser_cfg = AnchorDenoiserConfig(
        motion_dim=int(cfg.model.denoiser.motion_dim),
        object_traj_dim=int(cfg.model.denoiser.object_traj_dim),
        init_pose_dim=int(cfg.model.denoiser.init_pose_dim),
        text_dim=int(cfg.model.denoiser.text_dim),
        object_token_dim=int(cfg.model.denoiser.object_token_dim),
        object_num_tokens=int(cfg.model.denoiser.object_num_tokens),
        stage1_coarse_dim=int(cfg.model.denoiser.get("stage1_coarse_dim", 0)),
        # Round-29 typed condition injection.
        use_round29_cond_injection=bool(
            cfg.model.denoiser.get("use_round29_cond_injection", False)
        ),
        r29_coarse_extra_dim=int(
            cfg.model.denoiser.get("r29_coarse_extra_dim", 0)
        ),
        r29_interaction_dim=int(
            cfg.model.denoiser.get("r29_interaction_dim", 0)
        ),
        r29_support_dim=int(cfg.model.denoiser.get("r29_support_dim", 0)),
        r29_body_refine_dim=int(
            cfg.model.denoiser.get("r29_body_refine_dim", 0)
        ),
        r29_injection_mode=str(
            cfg.model.denoiser.get("r29_injection_mode", "input_add")
        ),
        r29_gate_bias_init=float(
            cfg.model.denoiser.get("r29_gate_bias_init", -1.0)
        ),
        r29_per_family_modes=(
            dict(cfg.model.denoiser.get("r29_per_family_modes"))
            if cfg.model.denoiser.get("r29_per_family_modes") is not None
            else None
        ),
        r29_zero_init_adapters=bool(
            cfg.model.denoiser.get("r29_zero_init_adapters", True)
        ),
        # PB1 — AdaLN-cond branch (Codex §4.3 / §4.4).
        r29_use_cond_adaln=bool(
            cfg.model.denoiser.get("r29_use_cond_adaln", False)
        ),
        r29_adaln_families=(
            list(cfg.model.denoiser.get("r29_adaln_families"))
            if cfg.model.denoiser.get("r29_adaln_families") is not None
            else None
        ),
        r29_adaln_pool=str(
            cfg.model.denoiser.get("r29_adaln_pool", "mean")
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

    # Tier-2 ablation: skip loading CLIP entirely when text is disabled.
    # Saves ~1 GB GPU memory + per-step encode cost.
    if int(cfg.model.denoiser.text_dim) > 0:
        clip_model = load_clip_text_encoder(
            device=device,
            model_name=str(cfg.model.text_encoder.clip_version),
            download_root=str(cfg.model.text_encoder.get("download_root", "cache/clip")),
        )
    else:
        clip_model = None
        accelerator.print("text_dim=0: skipping CLIP text encoder load.")

    anchor_cfg = AnchorConsistencyConfig(
        weight=float(cfg.loss.anchor_weight),
        contact_threshold=float(cfg.loss.contact_threshold),
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
            # Round-29 loss-strategy ablation (analyses/2026-05-27_round29_loss_strategy_ablation_prompt_for_claude_code.md).
            r29_interaction_consistency_weight=float(
                _tloss.get("r29_interaction_consistency_weight", 0.0)
            ),
            r29_support_both_airborne_weight=float(
                _tloss.get("r29_support_both_airborne_weight", 0.0)
            ),
            r29_support_stance_velocity_weight=float(
                _tloss.get("r29_support_stance_velocity_weight", 0.0)
            ),
            # Round-29 swing clearance (post-Codex v1 review).
            r29_swing_clearance_weight=float(
                _tloss.get("r29_swing_clearance_weight", 0.0)
            ),
            r29_swing_clearance_m=float(
                _tloss.get("r29_swing_clearance_m", 0.05)
            ),
            # Round-29 failure-targeted ablation (R2 behavior gait,
            # R3 exact S4, R4/R5 contact-lock). All default to zero so
            # pre-R29-FT configs are behaviorally unchanged.
            r29_gait_one_foot_support_weight=float(
                _tloss.get("r29_gait_one_foot_support_weight", 0.0)
            ),
            r29_gait_pred_stance_velocity_weight=float(
                _tloss.get("r29_gait_pred_stance_velocity_weight", 0.0)
            ),
            r29_gait_ankle_smooth_weight=float(
                _tloss.get("r29_gait_ankle_smooth_weight", 0.0)
            ),
            r29_gait_antiphase_corr_weight=float(
                _tloss.get("r29_gait_antiphase_corr_weight", 0.0)
            ),
            r29_gait_antiphase_min_walking_frames=int(
                _tloss.get("r29_gait_antiphase_min_walking_frames", 10)
            ),
            r29_s4_stance_bce_weight=float(
                _tloss.get("r29_s4_stance_bce_weight", 0.0)
            ),
            r29_s4_footstep_target_weight=float(
                _tloss.get("r29_s4_footstep_target_weight", 0.0)
            ),
            r29_contact_lock_offset_weight=float(
                _tloss.get("r29_contact_lock_offset_weight", 0.0)
            ),
            r29_contact_lock_segment_drift_weight=float(
                _tloss.get("r29_contact_lock_segment_drift_weight", 0.0)
            ),
            r29_contact_lock_tracking_weight=float(
                _tloss.get("r29_contact_lock_tracking_weight", 0.0)
            ),
            # Round-29 next-baseline ablation (G1) — phase-free gait. All
            # default to zero so pre-G1 configs are behaviorally unchanged.
            r29_gait_soft_stance_velocity_weight=float(
                _tloss.get("r29_gait_soft_stance_velocity_weight", 0.0)
            ),
            r29_gait_transition_rate_weight=float(
                _tloss.get("r29_gait_transition_rate_weight", 0.0)
            ),
            r29_gait_duty_cycle_weight=float(
                _tloss.get("r29_gait_duty_cycle_weight", 0.0)
            ),
            r29_gait_both_state_match_weight=float(
                _tloss.get("r29_gait_both_state_match_weight", 0.0)
            ),
            r29_gait_soft_stance_speed_threshold_mps=float(
                _tloss.get("r29_gait_soft_stance_speed_threshold_mps", 0.30)
            ),
            r29_gait_soft_stance_speed_softness_mps=float(
                _tloss.get("r29_gait_soft_stance_speed_softness_mps", 0.10)
            ),
            # Pulled from cfg.data (used by both the dataset's condition
            # builder and the R29 interaction-consistency loss; must match).
            r29_hand_offset_clamp_m=float(
                cfg.data.get("r29_hand_offset_clamp_m", 2.0)
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
        f"  object_traj_dim: {int(cfg.model.denoiser.object_traj_dim)}\n"
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
        device=device,
        motion_representation=motion_representation,
        world_joint_velocity_weight=world_joint_velocity_weight,
        object_traj_dim=int(cfg.model.denoiser.object_traj_dim),
        pos_loss_weight=pos_loss_weight,
        hand_endpoint_weight=hand_endpoint_weight,
        foot_endpoint_weight=foot_endpoint_weight,
        anchor_joint_pos_weight=anchor_joint_pos_weight,
        anchor_joint_vel_weight=anchor_joint_vel_weight,
        anchor_joint_part_weights=anchor_joint_part_weights,
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
            f"vel={out['loss_vel'].item():.4f}  "
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
            in_proj = unwrapped.denoiser.v12_input_proj.motion_proj.weight
            proj_name = "model.denoiser.v12_input_proj.motion_proj.weight"
            accelerator.print(f"[grad audit] per-loss-term L2 grad norm at {proj_name}:")
            audit_terms = [
                ("mse_x0",                       out.get("_raw_mse_x0")),
                ("loss_pos_full",                out.get("_raw_loss_pos_full")),
                ("loss_anchor_joint_pos",        out.get("_raw_loss_anchor_joint_pos")),
                ("loss_anchor_joint_vel",        out.get("_raw_loss_anchor_joint_vel")),
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
    # Honour WANDB_MODE env (e.g. WANDB_MODE=offline / WANDB_DISABLED=true).
    # Any init failure (no API key, no network, no install) is non-fatal —
    # training continues with wandb_run=None and metrics still go to
    # metrics.jsonl + accelerator.print.
    wandb_run = None
    if accelerator.is_main_process and os.environ.get("WANDB_DISABLED", "").lower() not in ("1", "true", "yes"):
        try:
            import wandb
            wandb_run = wandb.init(
                project=cfg.logging.project, name=cfg.logging.run_name,
            )
        except Exception as exc:  # noqa: BLE001
            accelerator.print(
                f"[wandb] init failed ({type(exc).__name__}: {exc}); "
                "continuing without wandb. Set WANDB_DISABLED=1 to silence, "
                "or run `wandb login` to enable."
            )
            wandb_run = None

    # Calibration / dynamic feature-weighting was a motion_263-only Round-28
    # feature. Removed alongside FeatureWeightState in the Tier-1 cleanup.
    epoch_end_hook = None

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
        val_dataloader=val_loader,
        val_every_epochs=int(cfg.training.get("val_every_epochs", 0)),
        val_best_key=str(cfg.training.get("val_best_key", "loss")),
    )


if __name__ == "__main__":
    main()
