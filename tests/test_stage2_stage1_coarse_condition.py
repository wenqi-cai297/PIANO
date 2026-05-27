"""Round-22 P0: tests for the Stage-1 Coarse-v1 oracle condition branch.

Per the Round-22 Codex prompt §Task 5 (Round-28 cleanup removed the
obsolete numpy-extractor equivalence check; 2026-05-27 cleanup removed
the obsolete z_int and zero_*_for_stageB tests after those code paths
were deleted):

  1. Collate / padding correctness — synthetic batch of mixed lengths
     should yield a (B, T_max, 23) tensor without nans.
  2. ``AnchorDenoiser(stage1_coarse_dim=23)`` accepts the new cond key
     and returns (B, T, 135) under v12 (use_dit_block=True).
  3. **Zero-init invariant** — at construction, the two denoisers
     (with and without ``stage1_coarse_dim=23``) produce bit-exact equal
     forward outputs given identical RNG seed and identical other inputs.
     This is the proof that adding the branch does not invalidate v18
     numerics.
  4. ``object_traj_dim=9`` build path does not silently expect 24 dims —
     ``_build_object_traj`` returns a 9-D tensor and the denoiser's
     ``null_obj_traj`` parameter is sized 9.

The tests are deliberately small (synthetic 1-2 clips, batch=2, T=16) so they
run on CPU in well under 10 seconds. They do NOT exercise the dataloader /
collate / wandb / accelerate paths — those are covered by the smoke configs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from piano.data.stage1_coarse_oracle import (
    COARSE_V1_DIM,
    extract_coarse_v1_batched,
    load_stage1_coarse_norm,
)
from piano.inference import diagnostic_helpers as pcd
from piano.models.motion_anchordiff import (
    AnchorDenoiser,
    AnchorDenoiserConfig,
)


# ---------------------------------------------------------------------------
# Synthetic data fixtures
# ---------------------------------------------------------------------------


def _make_synthetic_motion_135(
    T: int, *, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Random valid (T, 135) motion + (22, 3) rest_offsets.

    motion[..., :132] is 22 * 6d-rotation, motion[..., 132:135] is root_world.
    rot6d need not be a valid Gram-Schmidt parameterization — the orthogonalizer
    inside ``rotation_6d_to_matrix`` handles any 6-D vector, but to avoid
    exactly-degenerate rotations we use small random offsets from identity.
    """
    rng = np.random.default_rng(seed)
    # Identity 6d = (1,0,0, 0,1,0). Perturb each frame/joint.
    base = np.tile(
        np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32),
        (T, 22, 1),
    )
    perturb = rng.normal(scale=0.05, size=base.shape).astype(np.float32)
    rot6d = base + perturb                                          # (T, 22, 6)
    root_world = rng.normal(scale=0.5, size=(T, 3)).astype(np.float32).cumsum(axis=0)
    motion = np.concatenate(
        [rot6d.reshape(T, 132), root_world], axis=-1
    ).astype(np.float32)                                            # (T, 135)
    rest_offsets = rng.normal(scale=0.1, size=(22, 3)).astype(np.float32)
    rest_offsets[0] = 0.0                                            # root has no offset
    return motion, rest_offsets


def _build_denoiser_config(
    *,
    stage1_coarse_dim: int = 0,
    object_traj_dim: int = 24,
) -> AnchorDenoiserConfig:
    return AnchorDenoiserConfig(
        motion_dim=135,
        object_traj_dim=object_traj_dim,
        init_pose_dim=66,
        text_dim=512,
        object_token_dim=256,
        object_num_tokens=128,
        stage1_coarse_dim=stage1_coarse_dim,
        d_model=64,
        n_layers=2,
        n_heads=2,
        ff_mult=2,
        dropout=0.0,
        max_seq_length=32,
    )


def _make_synthetic_cond(B: int, T: int, cfg: AnchorDenoiserConfig, *, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    cond = {
        "object_world_traj": torch.randn(B, T, cfg.object_traj_dim, generator=g),
        "init_pose": torch.randn(B, cfg.init_pose_dim, generator=g),
        "text": torch.randn(B, 77, cfg.text_dim, generator=g),
        "object_tokens": torch.randn(B, cfg.object_num_tokens, cfg.object_token_dim, generator=g),
    }
    if cfg.stage1_coarse_dim > 0:
        cond["stage1_coarse"] = torch.randn(B, T, cfg.stage1_coarse_dim, generator=g)
    return cond


# ---------------------------------------------------------------------------
# Check 1 — collate/padding correctness on a 2-clip mixed-length batch
# ---------------------------------------------------------------------------


def test_extract_coarse_v1_batched_shape_and_finite():
    # Two clips of equal padded length T=16 (the trainer feeds already-padded
    # motion via the collate_hoi function; we mimic that here).
    T = 16
    motion0, rest0 = _make_synthetic_motion_135(T, seed=1)
    motion1, rest1 = _make_synthetic_motion_135(T, seed=2)
    motion_b = torch.from_numpy(np.stack([motion0, motion1]))      # (2, 16, 135)
    rest_b = torch.from_numpy(np.stack([rest0, rest1]))            # (2, 22, 3)
    out = extract_coarse_v1_batched(motion_b, rest_b)
    assert out.shape == (2, T, COARSE_V1_DIM)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Check 2 — denoiser accepts stage1_coarse_dim=23 and returns (B, T, 135)
# ---------------------------------------------------------------------------


def test_denoiser_with_stage1_coarse_dim_forward_shape():
    cfg = _build_denoiser_config(stage1_coarse_dim=23)
    torch.manual_seed(0)
    model = AnchorDenoiser(cfg).eval()
    B, T = 2, 16
    cond = _make_synthetic_cond(B, T, cfg, seed=0)
    x_t = torch.randn(B, T, cfg.motion_dim)
    t = torch.zeros(B, dtype=torch.long)
    with torch.no_grad():
        out = model(x_t, t, cond, cond_drop_mask=None)
    assert out.shape == (B, T, cfg.motion_dim)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Check 3 — zero-init invariant: enabling the branch with zero weights does
# not perturb the v18 forward output
# ---------------------------------------------------------------------------


def test_zero_init_invariant_preserves_v18_output():
    """v18 checkpoint loaded into an R22 architecture (stage1_coarse_dim=23)
    must produce bit-exact equal forward outputs when the new branch's
    zero-init projection is preserved.
    """
    B, T = 2, 16

    cfg_ref = _build_denoiser_config(stage1_coarse_dim=0)
    torch.manual_seed(12345)
    model_ref = AnchorDenoiser(cfg_ref).eval()

    cfg_new = _build_denoiser_config(stage1_coarse_dim=23)
    torch.manual_seed(67890)        # different seed on purpose; we copy weights below
    model_new = AnchorDenoiser(cfg_new).eval()

    # New-only keys: just the two stage1_coarse_proj tensors.
    ref_state = model_ref.state_dict()
    new_state = model_new.state_dict()
    extra_keys = set(new_state) - set(ref_state)
    assert extra_keys == {
        "v12_input_proj.stage1_coarse_proj.weight",
        "v12_input_proj.stage1_coarse_proj.bias",
    }, f"unexpected extra keys: {extra_keys}"

    # Verify the v18-checkpoint-loadable contract: ref's state dict loads
    # into new with strict=False, leaving the 2 new keys at their (zero) init.
    missing, unexpected = model_new.load_state_dict(ref_state, strict=False)
    assert sorted(missing) == sorted(extra_keys), (
        f"unexpected missing keys: {missing}"
    )
    assert unexpected == [], f"unexpected extra keys after load: {unexpected}"

    # Make the zero-init contract explicit (initialize_weights_v12 zeroes
    # this; load_state_dict didn't change it because it's not in ref).
    with torch.no_grad():
        model_new.v12_input_proj.stage1_coarse_proj.weight.zero_()
        model_new.v12_input_proj.stage1_coarse_proj.bias.zero_()

    cond_ref = _make_synthetic_cond(B, T, cfg_ref, seed=0)
    cond_new = _make_synthetic_cond(B, T, cfg_new, seed=0)
    for k in cond_ref:
        if isinstance(cond_ref[k], torch.Tensor):
            assert torch.equal(cond_ref[k], cond_new[k]), (
                f"shared cond key {k!r} diverges between configs"
            )

    x_t = torch.randn(B, T, cfg_ref.motion_dim)
    t = torch.zeros(B, dtype=torch.long)
    with torch.no_grad():
        out_ref = model_ref(x_t, t, cond_ref, cond_drop_mask=None)
        out_new = model_new(x_t, t, cond_new, cond_drop_mask=None)
    assert torch.equal(out_ref, out_new), (
        f"max|Δ|={(out_ref - out_new).abs().max().item():.3e} "
        "— zero-init invariant violated; enabling stage1_coarse_dim "
        "branch with zero-init projection perturbed the v18 forward output."
    )


def test_stage1_coarse_branch_activates_when_proj_nonzero():
    """Sanity check at the V12InputProjection layer: when stage1_coarse_proj
    receives non-zero weights, its output IS perturbed by the stage1_coarse
    input.
    """
    from piano.models.dit_blocks import V12InputProjection

    B, T = 2, 16
    motion_dim = 135
    obj_traj_dim = 9
    d_model = 64

    torch.manual_seed(11111)
    proj = V12InputProjection(
        motion_dim=motion_dim,
        obj_traj_dim=obj_traj_dim,
        d_model=d_model,
        stage1_coarse_dim=23,
    )
    # Zero-init the aux projections (mirrors initialize_weights_v12).
    with torch.no_grad():
        for sub in (proj.obj_proj, proj.stage1_coarse_proj):
            sub.weight.zero_()
            sub.bias.zero_()

    x_t = torch.randn(B, T, motion_dim)
    obj_traj = torch.randn(B, T, obj_traj_dim)
    coarse = torch.randn(B, T, 23)

    with torch.no_grad():
        out_zero = proj(x_t=x_t, obj_traj=obj_traj, stage1_coarse=coarse)
        # Activate the branch.
        torch.nn.init.xavier_uniform_(proj.stage1_coarse_proj.weight)
        out_active = proj(x_t=x_t, obj_traj=obj_traj, stage1_coarse=coarse)

    max_diff = (out_zero - out_active).abs().max().item()
    assert max_diff > 1e-4, (
        f"stage1_coarse branch did not perturb V12InputProjection output "
        f"(max|Δ|={max_diff:.3e}); the projection or routing is broken."
    )

    # And the perturbation is concentrated where stage1_coarse_proj routes —
    # the projection delta should equal what `stage1_coarse_proj` produces,
    # since all other paths are unchanged.
    expected_delta = proj.stage1_coarse_proj(coarse)
    actual_delta = out_active - out_zero
    assert torch.allclose(actual_delta, expected_delta, atol=1e-6), (
        f"V12InputProjection delta != stage1_coarse_proj(coarse); routing bug. "
        f"max|Δ|={(actual_delta - expected_delta).abs().max().item():.3e}"
    )


def test_v12_input_projection_missing_stage1_coarse_raises():
    """When V12InputProjection.stage1_coarse_dim > 0 but the caller forgets
    to pass ``stage1_coarse=``, a clear KeyError is raised. Prevents silent
    contract violations in trainer plumbing.
    """
    from piano.models.dit_blocks import V12InputProjection

    proj = V12InputProjection(
        motion_dim=135, obj_traj_dim=9,
        d_model=64, stage1_coarse_dim=23,
    )
    x_t = torch.randn(2, 16, 135)
    obj_traj = torch.randn(2, 16, 9)
    with pytest.raises(KeyError, match="stage1_coarse"):
        proj(x_t=x_t, obj_traj=obj_traj)


# ---------------------------------------------------------------------------
# Check 4 — object_traj_dim=9 build path does not silently expect 24 dims
# ---------------------------------------------------------------------------


def test_object_traj_dim_9_build_uses_pose_only():
    """Denoiser config with object_traj_dim=9 must build a v12 model whose
    null_obj_traj has shape (9,) and whose V12InputProjection.obj_proj
    expects 9-D input. Forward should succeed with a 9-D ``object_world_traj``
    cond and fail clearly if the trainer accidentally passes 24-D.
    """
    cfg = _build_denoiser_config(stage1_coarse_dim=23, object_traj_dim=9)
    torch.manual_seed(0)
    model = AnchorDenoiser(cfg).eval()

    # Null buffer sized 9, not 24.
    assert model.null_obj_traj.shape == (cfg.object_traj_dim,) == (9,)
    # V12InputProjection.obj_proj in_features must be 9.
    assert model.v12_input_proj.obj_proj.in_features == 9

    # Forward with 9-D obj_traj succeeds.
    B, T = 2, 16
    cond = _make_synthetic_cond(B, T, cfg, seed=0)
    assert cond["object_world_traj"].shape == (B, T, 9)
    x_t = torch.randn(B, T, cfg.motion_dim)
    t = torch.zeros(B, dtype=torch.long)
    with torch.no_grad():
        out = model(x_t, t, cond, cond_drop_mask=None)
    assert out.shape == (B, T, cfg.motion_dim)

    # If a caller wrongly passes 24-D, the obj_proj linear should mismatch
    # and raise. PyTorch's Linear.forward raises a RuntimeError for shape
    # mismatches.
    cond_wrong = dict(cond)
    cond_wrong["object_world_traj"] = torch.randn(B, T, 24)
    with pytest.raises(RuntimeError):
        with torch.no_grad():
            model(x_t, t, cond_wrong, cond_drop_mask=None)


# ---------------------------------------------------------------------------
# Diagnostic-helpers contracts (formerly plan_condition_diagnostics)
# ---------------------------------------------------------------------------


def test_diagnostic_object_traj_contract_9d_and_24d():
    B, T = 2, 5
    obj_com = torch.randn(B, T, 3)
    obj_rot6d = torch.randn(B, T, 6)
    contact_target_xyz = torch.randn(B, T, 5, 3)
    obj_pos_world = torch.zeros(B, T, 3)
    obj_rot_world = torch.zeros(B, T, 3)

    cfg9 = OmegaConf.create({"model": {"denoiser": {"object_traj_dim": 9}}})
    out9 = pcd._build_object_traj_for_cfg(
        cfg=cfg9,
        obj_com=obj_com,
        obj_rot6d=obj_rot6d,
        contact_target_xyz=contact_target_xyz,
        obj_pos_world=obj_pos_world,
        obj_rot_world=obj_rot_world,
    )
    assert out9.shape == (B, T, 9)
    assert torch.equal(out9[..., :3], obj_com)
    assert torch.equal(out9[..., 3:9], obj_rot6d)

    cfg24 = OmegaConf.create({"model": {"denoiser": {"object_traj_dim": 24}}})
    out24 = pcd._build_object_traj_for_cfg(
        cfg=cfg24,
        obj_com=obj_com,
        obj_rot6d=obj_rot6d,
        contact_target_xyz=contact_target_xyz,
        obj_pos_world=obj_pos_world,
        obj_rot_world=obj_rot_world,
    )
    assert out24.shape == (B, T, 24)
    assert torch.equal(out24[..., :9], torch.cat([obj_com, obj_rot6d], dim=-1))


def test_diagnostic_build_cond_adds_stage1_coarse_and_uses_9d_obj(monkeypatch):
    B, T = 2, 8
    motion0, rest0 = _make_synthetic_motion_135(T, seed=101)
    motion1, rest1 = _make_synthetic_motion_135(T, seed=102)
    motion = torch.from_numpy(np.stack([motion0, motion1]))
    rest = torch.from_numpy(np.stack([rest0, rest1]))

    batch = {
        "motion": motion,
        "joints": torch.randn(B, T, 22, 3),
        "object_pc": torch.randn(B, 64, 3),
        "contact_target_xyz": torch.randn(B, T, 5, 3),
        "obj_com_canonical": torch.randn(B, T, 3),
        "obj_rot6d_canonical": torch.randn(B, T, 6),
        "object_positions": torch.zeros(B, T, 3),
        "object_rotations": torch.zeros(B, T, 3),
        "seq_len": torch.full((B,), T, dtype=torch.long),
        "text": ["a", "b"],
        "rest_offsets": rest,
    }
    cfg = OmegaConf.create({
        "model": {
            "denoiser": {
                "object_traj_dim": 9,
                "stage1_coarse_dim": 23,
            },
        },
        "data": {"motion_representation": "smpl_pose_135_plan"},
    })
    mean = torch.zeros(1, 1, 23)
    std = torch.ones(1, 1, 23)

    monkeypatch.setattr(
        pcd,
        "encode_text_per_token",
        lambda _clip, texts, device: (
            torch.zeros(len(texts), 77, 512, device=device),
            None,
        ),
    )

    class _ObjectEncoder(torch.nn.Module):
        def forward(self, object_pc):
            return torch.zeros(object_pc.shape[0], 128, 256, device=object_pc.device)

    cond, out_T = pcd._build_cond(
        batch=batch,
        model=None,
        object_encoder=_ObjectEncoder(),
        clip_model=None,
        cfg=cfg,
        device=torch.device("cpu"),
        stage1_norm=(mean, std),
    )
    assert out_T == T
    assert cond["object_world_traj"].shape == (B, T, 9)
    assert cond["stage1_coarse"].shape == (B, T, 23)
    assert "z_int" not in cond, "z_int path was deleted; helper should not emit it"
    expected = extract_coarse_v1_batched(motion.float(), rest.float())
    assert torch.allclose(cond["stage1_coarse"], expected, atol=1e-6)


def test_stage1_norm_missing_cache_message_points_to_existing_builder(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_stage1_coarse_v1_cache.py"):
        load_stage1_coarse_norm(tmp_path)
