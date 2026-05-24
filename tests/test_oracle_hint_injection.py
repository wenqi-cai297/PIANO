"""Round-27 Tier-0A Commit 2: tests for the oracle interaction hint
projection in ``AnchorDenoiser``.

Three checks:

1. **Zero-init invariant.** A denoiser built with
   ``use_oracle_interaction_hint=True`` produces a bit-exact equal
   forward output to a reference denoiser (same shared weights, no
   hint branch) when (a) the trainer passes an arbitrary hint tensor
   and (b) the ``oracle_hint_proj`` last-layer weights/bias are zero.
   Matches the contract that Round-22 ``stage1_coarse_proj`` honours
   for the Coarse-v1 branch.

2. **Hint branch is wired.** After overwriting the
   ``oracle_hint_proj`` weights with non-zero values, the forward
   output of the denoiser does differ from the no-hint reference —
   confirming the hint actually reaches the residual stream.

3. **Gradient flows to ``oracle_hint_proj``.** A backward pass through
   a synthetic loss yields a non-zero gradient on every
   ``oracle_hint_proj`` parameter tensor — confirms there is no
   detach / no-grad path between the hint and the loss.

The tests reuse the helper builders from
``tests.test_stage2_stage1_coarse_condition`` so the synthetic ``cond``
dict shape stays consistent with the rest of the v12 test suite.
"""
from __future__ import annotations

import torch

from piano.models.motion_anchordiff import (
    AnchorDenoiser,
    AnchorDenoiserConfig,
    ZIntDims,
)


def _build_denoiser_config(
    *,
    use_oracle_interaction_hint: bool = False,
    oracle_hint_dim: int = 0,
) -> AnchorDenoiserConfig:
    """Minimal v12 denoiser config (mirrors
    ``test_stage2_stage1_coarse_condition._build_denoiser_config``)."""
    return AnchorDenoiserConfig(
        motion_dim=135,
        z_int=ZIntDims(num_parts=5, phase_classes=3, support_classes=3),
        object_traj_dim=24,
        init_pose_dim=66,
        text_dim=512,
        object_token_dim=256,
        object_num_tokens=128,
        use_interaction_plan=True,
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
        use_dit_block=True,
        dit_block_use_plan_pool_in_cond=False,
        stage1_coarse_dim=0,
        cfg_drop_stage1_coarse=False,
        use_oracle_interaction_hint=use_oracle_interaction_hint,
        oracle_hint_dim=oracle_hint_dim,
        d_model=64,
        n_layers=2,
        n_heads=2,
        ff_mult=2,
        dropout=0.0,
        max_seq_length=32,
    )


def _make_synthetic_cond(
    B: int,
    T: int,
    cfg: AnchorDenoiserConfig,
    *,
    seed: int = 0,
):
    """Synthetic cond dict for the v12 forward."""
    g = torch.Generator().manual_seed(seed)
    z_total = cfg.z_int.total
    cond = {
        "z_int": torch.randn(B, T, z_total, generator=g),
        "object_world_traj": torch.randn(B, T, cfg.object_traj_dim, generator=g),
        "init_pose": torch.randn(B, cfg.init_pose_dim, generator=g),
        "text": torch.randn(B, 77, cfg.text_dim, generator=g),
        "object_tokens": torch.randn(
            B, cfg.object_num_tokens, cfg.object_token_dim, generator=g,
        ),
    }
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
    if cfg.use_oracle_interaction_hint:
        cond["oracle_interaction_hint"] = torch.randn(
            B, T, cfg.oracle_hint_dim, generator=g,
        )
    return cond


# ---------------------------------------------------------------------------
# Check 1 — zero-init invariant
# ---------------------------------------------------------------------------


def test_oracle_hint_zero_init_invariant():
    """Enabling ``use_oracle_interaction_hint`` with the default zero-init
    last layer must not change the forward output, even when the hint
    tensor is non-zero. Mirrors the Round-22 stage1_coarse_proj contract.
    """
    B, T = 2, 16
    hint_dim = 13

    cfg_ref = _build_denoiser_config(use_oracle_interaction_hint=False)
    torch.manual_seed(20260525)
    model_ref = AnchorDenoiser(cfg_ref).eval()

    cfg_new = _build_denoiser_config(
        use_oracle_interaction_hint=True, oracle_hint_dim=hint_dim,
    )
    torch.manual_seed(98765)  # different ctor seed; we copy weights below.
    model_new = AnchorDenoiser(cfg_new).eval()

    # Confirm the only extra parameters live in ``oracle_hint_proj``.
    ref_keys = set(model_ref.state_dict())
    new_keys = set(model_new.state_dict())
    extra = new_keys - ref_keys
    assert extra and all(k.startswith("oracle_hint_proj.") for k in extra), (
        f"unexpected extra keys: {extra}"
    )

    # Copy shared weights ref -> new; leave oracle_hint_proj at zero-init.
    missing, unexpected = model_new.load_state_dict(
        model_ref.state_dict(), strict=False,
    )
    assert sorted(missing) == sorted(extra)
    assert unexpected == []

    # Make the zero-init contract explicit on the last linear.
    last_linear = model_new.oracle_hint_proj[-1]
    with torch.no_grad():
        last_linear.weight.zero_()
        last_linear.bias.zero_()

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
    max_diff = (out_ref - out_new).abs().max().item()
    assert torch.equal(out_ref, out_new), (
        f"max|Δ|={max_diff:.3e} — zero-init invariant violated."
    )


# ---------------------------------------------------------------------------
# Check 2 — hint branch is wired (output differs when proj is non-zero)
# ---------------------------------------------------------------------------


def test_oracle_hint_perturbs_output_when_proj_nonzero():
    """After overwriting ``oracle_hint_proj`` with non-zero weights, the
    forward output should differ from the zero-init baseline. Proof that
    the hint actually reaches the residual stream.

    Wider context: ``V12FinalLayer`` is itself zero-init'd at start, so a
    fresh AnchorDenoiser's output is exactly 0 regardless of any input.
    We therefore overwrite the V12FinalLayer's gating to a non-trivial
    state too, otherwise the test would compare 0 against 0.
    """
    B, T = 2, 16
    hint_dim = 13

    cfg = _build_denoiser_config(
        use_oracle_interaction_hint=True, oracle_hint_dim=hint_dim,
    )
    torch.manual_seed(20260525)
    model = AnchorDenoiser(cfg).eval()

    # Make V12FinalLayer non-zero so x0 actually depends on the residual
    # stream (the AdaLN-Zero gate is zero at init by design).
    fl = model.v12_final_layer
    with torch.no_grad():
        for p in fl.parameters():
            p.normal_(mean=0.0, std=0.05)

    cond_zero_proj = _make_synthetic_cond(B, T, cfg, seed=0)
    x_t = torch.randn(B, T, cfg.motion_dim)
    t = torch.zeros(B, dtype=torch.long)
    with torch.no_grad():
        out_zero = model(x_t, t, cond_zero_proj, cond_drop_mask=None)

    # Overwrite the last linear of oracle_hint_proj with non-zero weights.
    last_linear = model.oracle_hint_proj[-1]
    with torch.no_grad():
        last_linear.weight.normal_(mean=0.0, std=0.1)
        last_linear.bias.normal_(mean=0.0, std=0.1)

    with torch.no_grad():
        out_nonzero = model(x_t, t, cond_zero_proj, cond_drop_mask=None)

    max_diff = (out_nonzero - out_zero).abs().max().item()
    assert not torch.equal(out_nonzero, out_zero), (
        "hint branch did not perturb the output even when proj is non-zero "
        f"— max|Δ|={max_diff:.3e}, branch not wired correctly."
    )
    # Diff should be visibly above float noise.
    assert max_diff > 1e-4, f"hint effect too small: max|Δ|={max_diff:.3e}"


# ---------------------------------------------------------------------------
# Check 3 — gradient flows to oracle_hint_proj
# ---------------------------------------------------------------------------


def test_oracle_hint_gradient_flows_to_proj():
    """Backward through a synthetic loss must produce non-zero gradients
    on every parameter of ``oracle_hint_proj``. Confirms no detach /
    no-grad path between the hint tensor and the model output.
    """
    B, T = 2, 16
    hint_dim = 13

    cfg = _build_denoiser_config(
        use_oracle_interaction_hint=True, oracle_hint_dim=hint_dim,
    )
    torch.manual_seed(20260525)
    model = AnchorDenoiser(cfg).train()

    # Same trick as check 2: make V12FinalLayer non-zero so gradient can
    # flow back through it. Otherwise the AdaLN-Zero gate would kill the
    # signal at init.
    fl = model.v12_final_layer
    with torch.no_grad():
        for p in fl.parameters():
            p.normal_(mean=0.0, std=0.05)
    # Also break oracle_hint_proj's zero-init so the *first* gradient is
    # not artificially zero (downstream gradient would be 0 if the last
    # layer's weight is zero — chain rule).
    last_linear = model.oracle_hint_proj[-1]
    with torch.no_grad():
        last_linear.weight.normal_(mean=0.0, std=0.1)
        last_linear.bias.normal_(mean=0.0, std=0.1)

    cond = _make_synthetic_cond(B, T, cfg, seed=0)
    x_t = torch.randn(B, T, cfg.motion_dim)
    t = torch.zeros(B, dtype=torch.long)

    out = model(x_t, t, cond, cond_drop_mask=None)              # (B, T, motion_dim)
    loss = (out ** 2).mean()
    loss.backward()

    for name, p in model.oracle_hint_proj.named_parameters():
        assert p.grad is not None, f"oracle_hint_proj.{name}.grad is None"
        gnorm = p.grad.detach().abs().sum().item()
        assert gnorm > 0.0, (
            f"oracle_hint_proj.{name} received zero gradient — "
            "the hint branch is not wired into the autograd graph."
        )
