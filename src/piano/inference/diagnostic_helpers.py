"""Shared dataset / model / condition builders for diagnostic + inference scripts.

These helpers mirror the trainer's data loading + cond construction so
diagnostic / visual-review / sample-generation scripts can run a saved
checkpoint forward on the same inputs the trainer used at training time.

Extracted from the now-deleted ``scripts/stage_b_generator/plan_condition_diagnostics.py``
during the 2026-05-27 ``z_int`` + ``interaction_plan`` dead-path removal.
Lives in ``src/`` so dependent scripts can import via the package path
(``from piano.inference.diagnostic_helpers import ...``) instead of the
old ``sys.path``-hack pattern.

Consumers (as of 2026-05-27):
    scripts/stage_b_generator/render_round24_visual_review.py
    scripts/stage_b_generator/anchor_realization_diagnostic.py
    scripts/stage_b_generator/round26_sustained_contact_diag.py
    scripts/stage_b_generator/round26_gait_diag.py
    scripts/stage_b_generator/round27_build_tier0_train_indices.py
    scripts/stage_b_generator/round28_body_action_diag.py
    scripts/stage_b_generator/round28_build_body_action_subset.py
    scripts/stage_b_generator/round29_inspect_i3_hand_contact_distribution.py
    tests/test_stage2_stage1_coarse_condition.py
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, Subset

from piano.data.dataset import (
    HOIDataset, build_subject_split, extract_subject_id,
)
from piano.data.stage1_coarse_oracle import (
    extract_coarse_v1_batched,
    load_stage1_coarse_norm,
)
from piano.models.motion_anchordiff import (
    AnchorDenoiserConfig, AnchorDiffConfig, DiffusionConfig, MotionAnchorDiff,
)
from piano.models.object_encoder import ObjectEncoder
from piano.training.anchor_consistency_loss import lift_object_local_to_world
from piano.training.smpl_kinematics import (
    fk_from_global_rotations,
    rotation_6d_to_matrix,
)
from piano.utils.clip_utils import encode_text_per_token
from piano.utils.io_utils import load_json


# ---------------------------------------------------------------------------
# Checkpoint training-time metadata
# ---------------------------------------------------------------------------


def extract_train_time_meta(state: dict) -> dict[str, float | str | None]:
    """Pull training-wallclock metadata out of a checkpoint payload.

    Trainer (src/piano/training/trainer.py:_save_checkpoint) embeds
    ``train_started_at`` / ``train_saved_at`` / ``train_wallclock_seconds``
    in every checkpoint it writes. Older checkpoints (pre-2026-05-27)
    do not have these fields; we return None for those.

    Returns a flat dict suitable for splatting into a diag stats JSON:
    ``{"train_started_at": ..., "train_saved_at": ...,
       "train_wallclock_seconds": ..., "train_wallclock_hms": "1h23m45s"}``.
    """
    started_at = state.get("train_started_at")
    saved_at = state.get("train_saved_at")
    wall_sec = state.get("train_wallclock_seconds")
    hms: str | None = None
    if isinstance(wall_sec, (int, float)) and wall_sec >= 0:
        hh, rem = divmod(int(wall_sec), 3600)
        mm, ss = divmod(rem, 60)
        hms = f"{hh:d}h{mm:02d}m{ss:02d}s"
    return {
        "train_started_at": float(started_at) if isinstance(started_at, (int, float)) else None,
        "train_saved_at": float(saved_at) if isinstance(saved_at, (int, float)) else None,
        "train_wallclock_seconds": float(wall_sec) if isinstance(wall_sec, (int, float)) else None,
        "train_wallclock_hms": hms,
    }


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------


def _build_dataset(cfg, bucket: str, augment: bool) -> Subset | ConcatDataset:
    subj_filter: set | None = None
    subj_cfg = cfg.data.get("subject_split")
    if subj_cfg is not None and subj_cfg.get("enabled", False):
        keys: set[tuple[str, str]] = set()
        for entry in cfg.data.datasets:
            meta = load_json(Path(entry.root) / "metadata_clean.json")
            for m in meta:
                sid = extract_subject_id(Path(entry.root).name, m.get("seq_id", ""))
                if sid is not None:
                    keys.add((Path(entry.root).name, sid))
        splits = build_subject_split(
            sorted(keys),
            train_pct=subj_cfg.train_pct,
            val_pct=subj_cfg.val_pct,
            seed=subj_cfg.seed,
        )
        subj_filter = splits[bucket]
    pseudo_label_subdir = cfg.data.get("pseudo_label_subdir", None)
    pseudo_label_dir = cfg.data.get("pseudo_label_dir", None)
    datasets = []
    for entry in cfg.data.datasets:
        if pseudo_label_dir is not None:
            sub_dir = pseudo_label_dir
        elif pseudo_label_subdir:
            sub_dir = str(Path(entry.root) / pseudo_label_subdir)
        else:
            sub_dir = None
        ds = HOIDataset(
            root=entry.root,
            pseudo_label_dir=sub_dir,
            max_seq_length=int(cfg.data.max_seq_length),
            subject_id_filter=subj_filter,
            subsample_n_per_object=cfg.data.get("subsample_n_per_object", None),
            subsample_seed=int(cfg.data.get("subsample_seed", 42)),
            support_collapse_hand_support=bool(
                cfg.data.get("support_collapse_hand_support", True)
            ),
            surface_obj_pose=True,
            force_world_frame=bool(cfg.data.get("force_world_frame", False)),
            motion_representation=str(cfg.data.get("motion_representation", "motion_263")),
            oracle_hint_fps=float(cfg.data.get("oracle_hint_fps", 20.0)),
            surface_temporal_aux_fields=bool(
                cfg.data.get("surface_temporal_aux_fields", False)
            ),
            # Round-29 typed condition bundle (off by default; emits the
            # stage2_* keys when any family variant is non-zero).
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
# Model construction
# ---------------------------------------------------------------------------


def _build_model(cfg, device: torch.device) -> tuple[MotionAnchorDiff, ObjectEncoder]:
    denoiser_cfg = AnchorDenoiserConfig(
        motion_dim=int(cfg.model.denoiser.motion_dim),
        object_traj_dim=int(cfg.model.denoiser.object_traj_dim),
        init_pose_dim=int(cfg.model.denoiser.init_pose_dim),
        text_dim=int(cfg.model.denoiser.text_dim),
        object_token_dim=int(cfg.model.denoiser.object_token_dim),
        object_num_tokens=int(cfg.model.denoiser.object_num_tokens),
        stage1_coarse_dim=int(cfg.model.denoiser.get("stage1_coarse_dim", 0)),
        # Round-29 typed condition injection. Required so diagnostic
        # scripts can load R29 ckpts (which carry r29_inject.* keys)
        # without falling back to strict=False (which would silently
        # drop the conditioning the trainer learned).
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
            diffusion=diff_cfg, denoiser=denoiser_cfg,
            cfg_drop_prob=float(cfg.model.cfg_drop_prob),
        )
    ).to(device).eval()
    object_encoder = ObjectEncoder(
        num_input_points=int(cfg.model.object_encoder.num_input_points),
        num_output_tokens=int(cfg.model.object_encoder.num_output_tokens),
        feature_dim=int(cfg.model.object_encoder.feature_dim),
    ).to(device).eval()
    return model, object_encoder


# ---------------------------------------------------------------------------
# Stage-1 coarse norm stats
# ---------------------------------------------------------------------------


def _stage1_norm_for_cfg(
    cfg,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    stage1_coarse_dim = int(cfg.model.denoiser.get("stage1_coarse_dim", 0))
    if stage1_coarse_dim <= 0:
        return None
    cache_root = cfg.data.get("stage1_coarse_cache_root", None)
    if cache_root is None:
        raise ValueError(
            "stage1_coarse_dim > 0 requires data.stage1_coarse_cache_root "
            "in the YAML config (directory containing normalization_train.json)."
        )
    mean, std = load_stage1_coarse_norm(str(cache_root))
    if mean.shape != (stage1_coarse_dim,) or std.shape != (stage1_coarse_dim,):
        raise ValueError(
            f"Stage-1 norm stats shape mismatch: mean={mean.shape}, "
            f"std={std.shape}, stage1_coarse_dim={stage1_coarse_dim}"
        )
    mean_t = torch.from_numpy(mean).to(device=device, dtype=torch.float32).view(1, 1, -1)
    std_t = torch.from_numpy(std).to(device=device, dtype=torch.float32).view(1, 1, -1)
    return mean_t, std_t


# ---------------------------------------------------------------------------
# Object-trajectory channel
# ---------------------------------------------------------------------------


def _build_object_traj_for_cfg(
    *,
    cfg,
    obj_com: torch.Tensor,
    obj_rot6d: torch.Tensor,
    contact_target_xyz: torch.Tensor,
    obj_pos_world: torch.Tensor,
    obj_rot_world: torch.Tensor,
) -> torch.Tensor:
    """Mirror train_anchordiff._build_object_traj for diagnostic conds."""
    object_traj_dim = int(cfg.model.denoiser.object_traj_dim)
    components = [obj_com, obj_rot6d]
    if object_traj_dim >= 24:
        B, T = obj_com.shape[:2]
        target_world = lift_object_local_to_world(
            contact_target_xyz, obj_pos_world, obj_rot_world,
        ).reshape(B, T, -1)
        components.append(target_world)
    object_traj = torch.cat(components, dim=-1)
    if object_traj.shape[-1] != object_traj_dim:
        raise ValueError(
            f"object_traj_dim={object_traj_dim} but diagnostic built "
            f"{object_traj.shape[-1]} dims"
        )
    return object_traj


# ---------------------------------------------------------------------------
# cond dict construction
# ---------------------------------------------------------------------------


def _build_cond(
    batch: dict, model: MotionAnchorDiff, object_encoder: ObjectEncoder,
    clip_model, cfg, device: torch.device,
    stage1_norm: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[dict, int]:
    """Mirror the trainer's cond construction for diagnostic / inference."""
    motion = batch["motion"].to(device)
    joints = batch["joints"].to(device)
    object_pc = batch["object_pc"].to(device)
    contact_target_xyz = batch["contact_target_xyz"].to(device)
    obj_com = batch["obj_com_canonical"].to(device)
    obj_rot6d = batch["obj_rot6d_canonical"].to(device)
    obj_pos_world = batch["object_positions"].to(device)
    obj_rot_world = batch["object_rotations"].to(device)
    B, T, _ = motion.shape

    object_traj = _build_object_traj_for_cfg(
        cfg=cfg,
        obj_com=obj_com,
        obj_rot6d=obj_rot6d,
        contact_target_xyz=contact_target_xyz,
        obj_pos_world=obj_pos_world,
        obj_rot_world=obj_rot_world,
    )

    # init_pose and text are optional in R29 ablations.
    _denoiser = model.denoiser if model is not None else None
    use_init_pose = _denoiser is None or getattr(_denoiser, "use_init_pose", True)
    use_text = _denoiser is None or getattr(_denoiser, "use_text", True)
    obj_tokens = object_encoder(object_pc)
    cond = {
        "object_world_traj": object_traj,
        "object_tokens": obj_tokens,
    }
    if use_init_pose:
        cond["init_pose"] = joints[:, 0, :, :].reshape(B, -1)
    if use_text and clip_model is not None:
        text_features, _ = encode_text_per_token(clip_model, batch["text"], device)
        cond["text"] = text_features.float()
    # Round-29 typed condition bundle. The dataset surfaces each of
    # stage2_coarse_extra / stage2_interaction / stage2_support /
    # stage2_body_refine iff the corresponding family is active in
    # the diagnostic config (data.r29_<family>_variant != *0). The
    # model's Round29CondInjectionModule raises KeyError when an
    # active family's key is missing, so we just forward whatever
    # the dataset emitted.
    for _r29_key in (
        "stage2_coarse_extra",
        "stage2_interaction",
        "stage2_support",
        "stage2_body_refine",
    ):
        if _r29_key in batch:
            cond[_r29_key] = batch[_r29_key].to(device).float()
    stage1_coarse_dim = int(cfg.model.denoiser.get("stage1_coarse_dim", 0))
    if stage1_coarse_dim > 0:
        if str(cfg.data.get("motion_representation", "motion_263")) != "smpl_pose_135_plan":
            raise ValueError(
                "stage1_coarse_dim > 0 requires "
                "data.motion_representation='smpl_pose_135_plan'."
            )
        if "rest_offsets" not in batch:
            raise KeyError(
                "stage1_coarse_dim > 0 requires batch['rest_offsets'] for "
                "Coarse-v1 oracle extraction."
            )
        if stage1_norm is None:
            stage1_norm = _stage1_norm_for_cfg(cfg, device)
        assert stage1_norm is not None
        mean_t, std_t = stage1_norm
        coarse_raw = extract_coarse_v1_batched(
            motion=motion.float(),
            rest_offsets=batch["rest_offsets"].to(device).float(),
        )
        if coarse_raw.shape[-1] != stage1_coarse_dim:
            raise ValueError(
                f"Oracle Coarse-v1 dim {coarse_raw.shape[-1]} != "
                f"stage1_coarse_dim={stage1_coarse_dim}"
            )
        cond["stage1_coarse"] = (coarse_raw - mean_t) / std_t
    return cond, T


# ---------------------------------------------------------------------------
# FK / axis-angle helpers (formerly in anchor_realization_diagnostic.py)
# ---------------------------------------------------------------------------


def _fk_22joints(motion: torch.Tensor, rest_offsets: torch.Tensor) -> torch.Tensor:
    """motion: (B, T, 135) → joints: (B, T, 22, 3) world frame."""
    B, T, _ = motion.shape
    rot6d = motion[..., :132].reshape(B, T, 22, 6).float()
    root_world = motion[..., 132:135].float()
    rot_mat = rotation_6d_to_matrix(rot6d)
    rest_per_frame = rest_offsets.unsqueeze(1).expand(B, T, 22, 3).float()
    return fk_from_global_rotations(rot_mat, rest_per_frame, root_world)


def _aa_matrix(aa: torch.Tensor) -> torch.Tensor:
    """Axis-angle (3,) → rotation matrix (3, 3). Rodrigues, batched-able."""
    aa = aa.float()
    theta = torch.linalg.norm(aa).clamp_min(1e-9)
    k = aa / theta
    K = torch.zeros(3, 3, device=aa.device, dtype=aa.dtype)
    K[0, 1] = -k[2]; K[0, 2] = k[1]
    K[1, 0] = k[2];  K[1, 2] = -k[0]
    K[2, 0] = -k[1]; K[2, 1] = k[0]
    I3 = torch.eye(3, device=aa.device, dtype=aa.dtype)
    return I3 + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)
