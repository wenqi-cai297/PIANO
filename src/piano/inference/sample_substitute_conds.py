"""Sample Stage-1 / Stage-1.5 outputs on a selection subset, cache per-clip.

Used by the R31/R32/end-to-end downstream-coupling diag. The Stage-2
diag scripts (sustained_contact / gait / body_action / g1_soft_stance)
take a ``--substitute-conds-dir`` argument; this script populates that
directory.

Cache layout::

    <out_dir>/<subset>/<seq_id>.npz
        cond_keys: stage1_coarse (T, 23) z-scored   -- if Stage-1 sample
                   stage2_coarse_extra (T, 18) raw  -- if Stage-1.5 sample
                   stage2_support      (T, 13) raw  -- if Stage-1.5 sample
        meta: valid_T (int), seed (int)

The caller is responsible for matching the substitute key set to what
the Stage-2 diag's YAML config actually surfaces (we error early in
``_build_cond`` if not).

Sampling uses the same ``GaussianDiffusion.p_sample_loop`` machinery
Stage-2 uses, with the per-stage Denoiser passed in as the network. We
fix ``torch.manual_seed(seed)`` before every clip so the same input
batch always produces the same sample (matches Phase 0 probe's
contract).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from piano.data.dataset import collate_hoi
from piano.data.stage1_coarse_oracle import load_stage1_coarse_norm
from piano.models.motion_anchordiff import (
    DiffusionConfig,
    GaussianDiffusion,
)
from piano.models.object_encoder import ObjectEncoder
from piano.models.stage1_trajectory import (
    STAGE1_COARSE_DIM,
    Stage1Denoiser,
    Stage1DenoiserConfig,
)
from piano.models.stage1p5_interaction import (
    STAGE1P5_C41_DIM,
    STAGE1P5_S4_DIM,
    STAGE1P5_TOTAL_DIM,
    Stage1p5Denoiser,
    Stage1p5DenoiserConfig,
)
from piano.training.train_anchordiff import _build_dataset
from piano.utils.clip_utils import encode_text_per_token, load_clip_text_encoder


def _read_selection(path: Path) -> set[tuple[str, str]]:
    """Selection JSONs accept three legacy schemas (see Stage-2 diags)."""
    sel_obj = json.loads(path.read_text("utf-8"))
    selection = (
        sel_obj.get("selected")
        or sel_obj.get("candidates")
        or sel_obj.get("clips")
        or []
    )
    if not selection:
        raise SystemExit(f"empty selection: {path}")
    return {(e["subset"], e["seq_id"]) for e in selection}


def _build_stage1_model(cfg, device: torch.device) -> tuple[Stage1Denoiser, ObjectEncoder]:
    d = cfg.model.denoiser
    denoiser_cfg = Stage1DenoiserConfig(
        motion_dim=int(d.motion_dim),
        object_traj_dim=int(d.object_traj_dim),
        text_dim=int(d.text_dim),
        object_token_dim=int(d.object_token_dim),
        object_num_tokens=int(d.object_num_tokens),
        d_model=int(d.d_model),
        n_layers=int(d.n_layers),
        n_heads=int(d.n_heads),
        ff_mult=int(d.ff_mult),
        dropout=float(d.dropout),
        max_seq_length=int(cfg.data.max_seq_length),
        use_text=bool(d.get("use_text", True)),
    )
    model = Stage1Denoiser(denoiser_cfg).to(device)
    encoder = ObjectEncoder(
        num_input_points=int(cfg.model.object_encoder.num_input_points),
        num_output_tokens=int(cfg.model.object_encoder.num_output_tokens),
        feature_dim=int(cfg.model.object_encoder.feature_dim),
    ).to(device)
    return model, encoder


def _build_stage1p5_model(cfg, device: torch.device) -> tuple[Stage1p5Denoiser, ObjectEncoder]:
    d = cfg.model.denoiser
    denoiser_cfg = Stage1p5DenoiserConfig(
        motion_dim=int(d.motion_dim),
        stage1_coarse_dim=int(d.stage1_coarse_dim),
        object_traj_dim=int(d.object_traj_dim),
        text_dim=int(d.text_dim),
        object_token_dim=int(d.object_token_dim),
        object_num_tokens=int(d.object_num_tokens),
        d_model=int(d.d_model),
        n_layers=int(d.n_layers),
        n_heads=int(d.n_heads),
        ff_mult=int(d.ff_mult),
        dropout=float(d.dropout),
        max_seq_length=int(cfg.data.max_seq_length),
        use_text=bool(d.get("use_text", True)),
    )
    model = Stage1p5Denoiser(denoiser_cfg).to(device)
    encoder = ObjectEncoder(
        num_input_points=int(cfg.model.object_encoder.num_input_points),
        num_output_tokens=int(cfg.model.object_encoder.num_output_tokens),
        feature_dim=int(cfg.model.object_encoder.feature_dim),
    ).to(device)
    return model, encoder


def _build_diffusion(cfg) -> GaussianDiffusion:
    return GaussianDiffusion(
        DiffusionConfig(
            num_steps=int(cfg.model.diffusion.num_steps),
            schedule=str(cfg.model.diffusion.schedule),
            objective=str(cfg.model.diffusion.get("objective", "ddpm")),
            prediction_target=str(
                cfg.model.diffusion.get("prediction_target", "x0"),
            ),
        )
    )


def _load_ckpt(model: torch.nn.Module, encoder: ObjectEncoder, ckpt_path: Path):
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model_state = state.get("model", state)
    model.load_state_dict(model_state)
    if "object_encoder" in state:
        encoder.load_state_dict(state["object_encoder"])
    elif "extra_modules" in state and "object_encoder" in state["extra_modules"]:
        encoder.load_state_dict(state["extra_modules"]["object_encoder"])
    else:
        raise SystemExit(
            f"FATAL: ckpt {ckpt_path} has no object_encoder state. "
            "Aborting — without it the object features stay at random "
            "init and the substitute_conds cache would be meaningless."
        )


def _maybe_load_clip(cfg, device: torch.device):
    if int(cfg.model.denoiser.get("text_dim", 0)) > 0:
        return load_clip_text_encoder(
            device=device,
            model_name=str(cfg.model.text_encoder.clip_version),
            download_root=str(
                cfg.model.text_encoder.get("download_root", "cache/clip"),
            ),
        )
    return None


def _read_substitute_clip(path: Path) -> dict[str, np.ndarray]:
    """Read a previously cached substitute_conds .npz for one clip."""
    data = np.load(path)
    out: dict[str, np.ndarray] = {}
    for k in ("stage1_coarse", "stage2_coarse_extra", "stage2_support"):
        if k in data.files:
            out[k] = data[k]
    return out


def _load_substitute_clip_into_cond(
    upstream_dir: Path,
    subset: str, seq_id: str,
    cond: dict, device: torch.device, T: int,
):
    """Mutate ``cond`` in place by overwriting any keys present in the
    upstream cache for this clip. Same shape contract as the diag-side
    ``substitute_conds`` hook in ``diagnostic_helpers._build_cond``.
    """
    p = upstream_dir / subset / f"{seq_id}.npz"
    if not p.exists():
        raise FileNotFoundError(
            f"upstream substitute cache missing for ({subset!r}, {seq_id!r}): {p}"
        )
    payload = _read_substitute_clip(p)
    for k, v in payload.items():
        # cond[k] is (B=1, T_cond, D); the cache stores (T_cached, D).
        # Truncate / pad-trim to match cond's T.
        tv = torch.from_numpy(v).to(device).float().unsqueeze(0)  # (1, T_cached, D)
        if tv.shape[1] >= T:
            tv = tv[:, :T]
        else:
            raise ValueError(
                f"upstream cache for ({subset!r}, {seq_id!r}) key {k!r} has "
                f"T={tv.shape[1]} < diag T={T}; was the cache sampled with "
                "the same max_seq_length?"
            )
        cond[k] = tv


def _build_object_traj_canonical(batch: dict, device: torch.device) -> torch.Tensor:
    """9-D obj_traj matching what Stage-1 / Stage-1.5 trainers used.

    Both Stage-1 and Stage-1.5 take a canonical obj_traj = COM (3) +
    rot6d (6). Matches src/piano/training/train_stage1.py + train_stage1p5.py.
    """
    obj_com = batch["obj_com_canonical"].to(device)
    obj_rot6d = batch["obj_rot6d_canonical"].to(device)
    return torch.cat([obj_com, obj_rot6d], dim=-1)


def _build_cond_for_upstream(
    *, batch: dict, encoder: ObjectEncoder, clip_model,
    stage: Literal["stage1", "stage1p5"],
    device: torch.device,
    upstream_dir: Path | None = None,
    stage1_norm: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[dict, int]:
    """Build the cond dict the upstream model itself consumes at sample time.

    For Stage-1 this is just object_world_traj + object_tokens + optional
    text. For Stage-1.5 it additionally needs stage1_coarse — either the
    oracle z-scored version or the Stage-1 sample from upstream_dir.
    """
    motion = batch["motion"].to(device)
    object_pc = batch["object_pc"].to(device)
    B, T, _ = motion.shape

    object_traj = _build_object_traj_canonical(batch, device)
    obj_tokens = encoder(object_pc)
    cond: dict[str, torch.Tensor] = {
        "object_world_traj": object_traj,
        "object_tokens": obj_tokens,
    }
    if clip_model is not None:
        text_features, _ = encode_text_per_token(clip_model, batch["text"], device)
        cond["text"] = text_features.float()

    if stage == "stage1p5":
        # Need stage1_coarse cond.
        if upstream_dir is not None:
            # End-to-end (D): Stage-1.5 reads Stage-1's cached sample.
            subset = str(batch["subset"][0])
            seq_id = str(batch["seq_id"][0])
            _load_substitute_clip_into_cond(
                upstream_dir, subset, seq_id, cond, device, T,
            )
            if "stage1_coarse" not in cond:
                raise KeyError(
                    f"upstream cache at {upstream_dir} did not provide "
                    f"stage1_coarse for ({subset!r}, {seq_id!r})"
                )
        else:
            # C only (Stage-1.5 alone): use oracle z-scored stage1_coarse.
            if stage1_norm is None:
                raise ValueError(
                    "stage1p5 sampling without upstream_dir requires "
                    "stage1_norm to z-score the oracle stage1_coarse."
                )
            from piano.data.stage1_coarse_oracle import extract_coarse_v1_batched
            rest_offsets = batch["rest_offsets"].to(device).float()
            mean_t, std_t = stage1_norm
            coarse_raw = extract_coarse_v1_batched(
                motion=motion.float(), rest_offsets=rest_offsets,
            )
            cond["stage1_coarse"] = (coarse_raw - mean_t) / std_t

    return cond, T


def sample_substitute_conds(
    *, config_path: Path, ckpt_path: Path, selection_json: Path,
    out_dir: Path, bucket: str, stage: Literal["stage1", "stage1p5"],
    upstream_dir: Path | None = None,
    seed: int = 42, cfg_scale: float = 1.0,
    sampler: str = "ddim_eta0",
) -> int:
    """Sample upstream cond conds on the selected clips, cache per-clip.

    Returns number of clips written.
    """
    cfg = OmegaConf.load(str(config_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model + diffusion + load ckpt.
    if stage == "stage1":
        model, encoder = _build_stage1_model(cfg, device)
    elif stage == "stage1p5":
        model, encoder = _build_stage1p5_model(cfg, device)
    else:
        raise ValueError(f"unknown stage {stage!r}")
    _load_ckpt(model, encoder, ckpt_path)
    model.eval()
    encoder.eval()
    diffusion = _build_diffusion(cfg).to(device)
    clip_model = _maybe_load_clip(cfg, device)

    # Selection.
    sel_pairs = _read_selection(selection_json)
    print(
        f"[sample_substitute] stage={stage} selection={len(sel_pairs)} "
        f"out_dir={out_dir} ckpt={ckpt_path.name}"
    )

    # Need stage1 norm when stage1p5 has no upstream_dir (oracle path).
    stage1_norm: tuple[torch.Tensor, torch.Tensor] | None = None
    if stage == "stage1p5" and upstream_dir is None:
        mean_np, std_np = load_stage1_coarse_norm(
            str(cfg.data.stage1_coarse_cache_root),
        )
        stage1_norm = (
            torch.from_numpy(mean_np).to(device).float(),
            torch.from_numpy(std_np).to(device).float(),
        )

    # Dataset loader.
    dataset = _build_dataset(cfg, bucket, augment=False)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        collate_fn=collate_hoi, num_workers=0,
    )

    # Output dim for sanity.
    if stage == "stage1":
        sample_dim = STAGE1_COARSE_DIM
    else:
        sample_dim = STAGE1P5_TOTAL_DIM

    out_dir.mkdir(parents=True, exist_ok=True)
    n_written = 0
    for batch in loader:
        subset = str(batch["subset"][0])
        seq_id = str(batch["seq_id"][0])
        if (subset, seq_id) not in sel_pairs:
            continue

        cond, T = _build_cond_for_upstream(
            batch=batch, encoder=encoder, clip_model=clip_model,
            stage=stage, device=device,
            upstream_dir=upstream_dir, stage1_norm=stage1_norm,
        )

        seq_len = int(batch["seq_len"][0].item())
        valid_T = min(T, seq_len)

        torch.manual_seed(seed)
        with torch.no_grad():
            x0_pred = diffusion.p_sample_loop(
                denoiser=model,
                shape=(1, T, sample_dim),
                cond=cond,
                cfg_scale=cfg_scale,
                device=device,
                sampler=sampler,
            )                                                # (1, T, sample_dim)

        x0_np = x0_pred[0].cpu().numpy().astype(np.float32)  # (T, sample_dim)

        # Pack output.
        out_sub = out_dir / subset
        out_sub.mkdir(parents=True, exist_ok=True)
        save_path = out_sub / f"{seq_id}.npz"
        if stage == "stage1":
            np.savez(
                save_path,
                stage1_coarse=x0_np,                          # (T, 23) z-scored
                valid_T=np.int32(valid_T),
                seed=np.int32(seed),
            )
        else:
            c41 = x0_np[:, :STAGE1P5_C41_DIM]                # (T, 18) raw
            s4 = x0_np[:, STAGE1P5_C41_DIM:]                 # (T, 13) raw
            np.savez(
                save_path,
                stage2_coarse_extra=c41,
                stage2_support=s4,
                valid_T=np.int32(valid_T),
                seed=np.int32(seed),
            )
        n_written += 1
        if n_written % 4 == 0:
            print(f"  [sample_substitute] {n_written} clips written so far")

    print(f"[sample_substitute] wrote {n_written} clips to {out_dir}")
    return n_written
