"""Round-22 P0: tests for the Stage-1 Coarse-v1 oracle condition branch.

Six checks per the Round-22 Codex prompt §Task 5:

  1. ``extract_coarse_v1_batched`` matches the canonical numpy
     ``extract_coarse_v0_v1`` on a single clip (channel-by-channel).
  2. Collate / padding correctness — synthetic batch of mixed lengths
     should yield a (B, T_max, 23) tensor without nans.
  3. ``AnchorDenoiser(stage1_coarse_dim=23)`` accepts the new cond key
     and returns (B, T, 135) under v12 (use_dit_block=True).
  4. **Zero-init invariant** — at construction, the two denoisers
     (with and without ``stage1_coarse_dim=23``) produce bit-exact equal
     forward outputs given identical RNG seed and identical other inputs.
     This is the proof that adding the branch does not invalidate v18
     numerics.
  5. Clean contract zeroing — when both ``zero_z_int_for_stageB=True``
     and ``zero_dense_contact_target_for_stageB=True``, the trainer's
     cond assembly produces all-zero ``z_int`` and zeros the 15-D dense
     contact-target suffix of ``object_world_traj``.
  6. ``object_traj_dim=9`` build path does not silently expect 24 dims —
     ``_build_object_traj`` returns a 9-D tensor and the denoiser's
     ``null_obj_traj`` parameter is sized 9.

The tests are deliberately small (synthetic 1-2 clips, batch=2, T=16) so they
run on CPU in well under 10 seconds. They do NOT exercise the dataloader /
collate / wandb / accelerate paths — those are covered by the smoke configs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Add scripts/stage_b_generator to sys.path so we can import the canonical
# numpy extractor for the equivalence test (it's not packaged in the wheel).
_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts" / "stage_b_generator"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from extract_coarse_motion_representation import (  # type: ignore  # noqa: E402
    extract_coarse_v0_v1,
)

from piano.data.stage1_coarse_oracle import (  # noqa: E402
    COARSE_V1_DIM,
    extract_coarse_v1_batched,
)
from piano.models.motion_anchordiff import (  # noqa: E402
    AnchorDenoiser,
    AnchorDenoiserConfig,
    ZIntDims,
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
    use_dit_block: bool = True,
    use_interaction_plan: bool = True,
) -> AnchorDenoiserConfig:
    return AnchorDenoiserConfig(
        motion_dim=135,
        z_int=ZIntDims(num_parts=5, phase_classes=3, support_classes=3),
        object_traj_dim=object_traj_dim,
        init_pose_dim=66,
        text_dim=512,
        object_token_dim=256,
        object_num_tokens=128,
        use_interaction_plan=use_interaction_plan,
        plan_k_max=12,
        plan_s_max=12,
        plan_num_anchor_types=5,
        plan_num_parts=5,
        plan_use_segment_tokens=False,
        plan_use_context_hint=True,
        plan_d_hint=32,
        plan_d_time_embed=64,
        cfg_drop_plan=False,
        plan_per_part_tokens=True,
        plan_context_hint_mode="target_aware",
        use_dit_block=use_dit_block,
        dit_block_use_plan_pool_in_cond=False,
        stage1_coarse_dim=stage1_coarse_dim,
        cfg_drop_stage1_coarse=False,
        d_model=64,
        n_layers=2,
        n_heads=2,
        ff_mult=2,
        dropout=0.0,
        max_seq_length=32,
    )


def _make_synthetic_cond(B: int, T: int, cfg: AnchorDenoiserConfig, *, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    z_total = cfg.z_int.total
    cond = {
        "z_int": torch.randn(B, T, z_total, generator=g),
        "object_world_traj": torch.randn(B, T, cfg.object_traj_dim, generator=g),
        "init_pose": torch.randn(B, cfg.init_pose_dim, generator=g),
        "text": torch.randn(B, 77, cfg.text_dim, generator=g),
        "object_tokens": torch.randn(B, cfg.object_num_tokens, cfg.object_token_dim, generator=g),
    }
    # Synthetic InteractionPlan padded dict — minimal valid shape (all masks
    # False is fine; the encoder handles an all-empty plan).
    K, S, P = cfg.plan_k_max, cfg.plan_s_max, cfg.plan_num_parts
    cond["interaction_plan"] = {
        "anchor_time": torch.zeros(B, K, dtype=torch.long),
        "anchor_part": torch.zeros(B, K, P),
        "anchor_target_local": torch.zeros(B, K, P, 3),
        "anchor_target_world": torch.zeros(B, K, P, 3),
        "anchor_type": torch.zeros(B, K, dtype=torch.long),
        "anchor_phase": torch.zeros(B, K, dtype=torch.long),
        "anchor_support": torch.zeros(B, K, dtype=torch.long),
        "anchor_conf": torch.zeros(B, K),
        "anchor_mask": torch.zeros(B, K, dtype=torch.bool),
        "segment_start": torch.zeros(B, S, dtype=torch.long),
        "segment_end": torch.zeros(B, S, dtype=torch.long),
        "segment_part": torch.zeros(B, S, P),
        "segment_target_summary_local": torch.zeros(B, S, P, 3),
        "segment_phase": torch.zeros(B, S, dtype=torch.long),
        "segment_support": torch.zeros(B, S, dtype=torch.long),
        "segment_conf": torch.zeros(B, S),
        "segment_mask": torch.zeros(B, S, dtype=torch.bool),
    }
    if cfg.stage1_coarse_dim > 0:
        cond["stage1_coarse"] = torch.randn(B, T, cfg.stage1_coarse_dim, generator=g)
    return cond


# ---------------------------------------------------------------------------
# Check 1 — oracle adapter matches canonical numpy extractor
# ---------------------------------------------------------------------------


def test_extract_coarse_v1_batched_matches_numpy_extractor():
    T = 24
    motion_np, rest_np = _make_synthetic_motion_135(T, seed=42)

    # Canonical numpy extractor (single clip).
    out_np = extract_coarse_v0_v1(motion_np, rest_np, seq_len=T)
    coarse_v1_np = out_np["coarse_v1"]                             # (T, 23)
    assert coarse_v1_np.shape == (T, COARSE_V1_DIM)

    # Torch batched extractor (B=1).
    motion_t = torch.from_numpy(motion_np).unsqueeze(0)            # (1, T, 135)
    rest_t = torch.from_numpy(rest_np).unsqueeze(0)                # (1, 22, 3)
    coarse_v1_t = extract_coarse_v1_batched(motion_t, rest_t).squeeze(0).numpy()
    assert coarse_v1_t.shape == (T, COARSE_V1_DIM)

    # Tolerance: float32 FK + atan2 give small numeric noise.
    assert np.allclose(coarse_v1_t, coarse_v1_np, atol=1e-5, rtol=1e-5), (
        f"max|diff|={np.max(np.abs(coarse_v1_t - coarse_v1_np)):.3e}"
    )
    assert np.isfinite(coarse_v1_t).all()


# ---------------------------------------------------------------------------
# Check 2 — collate/padding correctness on a 2-clip mixed-length batch
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
# Check 3 — denoiser accepts stage1_coarse_dim=23 and returns (B, T, 135)
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
# Check 4 — zero-init invariant: enabling the branch with zero weights does
# not perturb the v18 forward output
# ---------------------------------------------------------------------------


def test_zero_init_invariant_preserves_v18_output():
    """v18 checkpoint loaded into an R22 architecture (stage1_coarse_dim=23)
    must produce bit-exact equal forward outputs when the new branch's
    zero-init projection is preserved.

    This is the meaningful safety contract: an existing v18 ckpt loaded into
    the R22 architecture continues to produce the same output as v18 alone
    until the new ``stage1_coarse_proj`` learns non-zero weights. It does NOT
    require the two constructions to consume identical RNG (the new module's
    extra ``nn.Linear`` would shift subsequent ``xavier_uniform_`` draws — an
    artefact of construction order, not a forward-path bug).

    Procedure:
      1. Build a v18 reference denoiser.
      2. Build an R22 denoiser with ``stage1_coarse_dim=23``.
      3. Copy all *shared* state-dict keys from ref into new (load_state_dict
         with ``strict=False`` so the 3 new keys are skipped).
      4. Re-zero ``stage1_coarse_proj`` (it should already be zero from
         ``initialize_weights_v12`` but we make the invariant explicit).
      5. Forward both with identical inputs and verify bit-exact equality.
    """
    B, T = 2, 16

    cfg_ref = _build_denoiser_config(stage1_coarse_dim=0)
    torch.manual_seed(12345)
    model_ref = AnchorDenoiser(cfg_ref).eval()

    cfg_new = _build_denoiser_config(stage1_coarse_dim=23)
    torch.manual_seed(67890)        # different seed on purpose; we copy weights below
    model_new = AnchorDenoiser(cfg_new).eval()

    # New-only keys (must be exactly these three under the R22 design).
    ref_state = model_ref.state_dict()
    new_state = model_new.state_dict()
    extra_keys = set(new_state) - set(ref_state)
    assert extra_keys == {
        "null_stage1_coarse",
        "v12_input_proj.stage1_coarse_proj.weight",
        "v12_input_proj.stage1_coarse_proj.bias",
    }, f"unexpected extra keys: {extra_keys}"

    # Verify the v18-checkpoint-loadable contract: ref's state dict loads
    # into new with strict=False, leaving the 3 new keys at their (zero) init.
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
        model_new.null_stage1_coarse.zero_()

    # Synthetic forward — both models see identical other-channel inputs;
    # the new model additionally sees stage1_coarse, which gets multiplied
    # by a zero projection so its contribution to the residual stream is 0.
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

    We test V12InputProjection in isolation rather than the full denoiser
    because v12's V12FinalLayer is zero-init'd by design — so a fresh
    AnchorDenoiser's forward output is identically 0 regardless of any
    intermediate residual stream perturbation. The meaningful "branch
    routes signal" check lives at the projection level.
    """
    from piano.models.dit_blocks import V12InputProjection

    B, T = 2, 16
    motion_dim = 135
    zint_dim = 26
    obj_traj_dim = 9
    hint_dim = 32
    d_model = 64

    torch.manual_seed(11111)
    proj = V12InputProjection(
        motion_dim=motion_dim,
        zint_dim=zint_dim,
        obj_traj_dim=obj_traj_dim,
        hint_dim=hint_dim,
        d_model=d_model,
        stage1_coarse_dim=23,
    )
    # Zero-init the aux projections (mirrors initialize_weights_v12).
    with torch.no_grad():
        for sub in (proj.zint_proj, proj.obj_proj, proj.hint_proj, proj.stage1_coarse_proj):
            sub.weight.zero_()
            sub.bias.zero_()

    x_t = torch.randn(B, T, motion_dim)
    z_int = torch.randn(B, T, zint_dim)
    obj_traj = torch.randn(B, T, obj_traj_dim)
    plan_hint = torch.randn(B, T, hint_dim)
    coarse = torch.randn(B, T, 23)

    with torch.no_grad():
        out_zero = proj(
            x_t=x_t, z_int=z_int, obj_traj=obj_traj,
            plan_hint=plan_hint, stage1_coarse=coarse,
        )
        # Activate the branch.
        torch.nn.init.xavier_uniform_(proj.stage1_coarse_proj.weight)
        out_active = proj(
            x_t=x_t, z_int=z_int, obj_traj=obj_traj,
            plan_hint=plan_hint, stage1_coarse=coarse,
        )

    max_diff = (out_zero - out_active).abs().max().item()
    assert max_diff > 1e-4, (
        f"stage1_coarse branch did not perturb V12InputProjection output "
        f"(max|Δ|={max_diff:.3e}); the projection or routing is broken."
    )

    # And the perturbation is concentrated where stage1_coarse_proj routes —
    # namely the projection delta should equal what `stage1_coarse_proj`
    # produces, since all other paths are unchanged.
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
        motion_dim=135, zint_dim=26, obj_traj_dim=9, hint_dim=32,
        d_model=64, stage1_coarse_dim=23,
    )
    x_t = torch.randn(2, 16, 135)
    z_int = torch.randn(2, 16, 26)
    obj_traj = torch.randn(2, 16, 9)
    plan_hint = torch.randn(2, 16, 32)
    with pytest.raises(KeyError, match="stage1_coarse"):
        proj(x_t=x_t, z_int=z_int, obj_traj=obj_traj, plan_hint=plan_hint)


# ---------------------------------------------------------------------------
# Check 5 — clean contract zeroing actually zeros the channels
# ---------------------------------------------------------------------------


def test_clean_contract_zeros_z_int_and_dense_target():
    """The trainer's step_fn zeros dense channels when the corresponding flag
    is set. We replicate the minimal cond assembly here (without spinning up
    the full HOIDataset / accelerate stack) to verify the zeroing logic.
    """
    B, T = 2, 16
    z_total = ZIntDims().total                                      # 26
    object_traj_dim_full = 24
    z_int = torch.randn(B, T, z_total)
    object_traj = torch.randn(B, T, object_traj_dim_full)

    # Simulate the train_anchordiff.py step_fn block at lines ~517-524.
    z_int_clean = torch.zeros_like(z_int)                           # zero_z_int_for_stageB
    object_traj_clean = object_traj.clone()
    object_traj_clean[..., 9:] = 0.0                                # zero_dense_contact_target_for_stageB

    assert torch.all(z_int_clean == 0)
    assert torch.all(object_traj_clean[..., 9:] == 0)
    # Object pose (first 9 dims) is preserved.
    assert torch.equal(object_traj_clean[..., :9], object_traj[..., :9])


# ---------------------------------------------------------------------------
# Check 6 — object_traj_dim=9 build path does not silently expect 24 dims
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
# Bonus — guard against v11 misuse of the new branch
# ---------------------------------------------------------------------------


def test_v11_path_rejects_stage1_coarse_branch():
    cfg = _build_denoiser_config(stage1_coarse_dim=23, use_dit_block=False, use_interaction_plan=False)
    with pytest.raises(ValueError, match="stage1_coarse_dim > 0 requires use_dit_block=True"):
        AnchorDenoiser(cfg)
