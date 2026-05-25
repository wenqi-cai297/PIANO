"""Round-28 injection-mechanism smoke tests.

Verifies (prompt §6.3):

1. **Zero-init invariant — input_add + body-action branch.** Adding the
   body-action hint branch with zero-init last-layer produces output
   equal to the no-body-action baseline.
2. **Zero-init invariant — gated_input.** The gated_input injection
   keeps output unchanged at step 0 thanks to zero-init projection
   weights AND a strongly negative gate bias (initial gate ≈ 0).
3. **Zero-init invariant — per_layer_adapter.** The per_layer adapters
   are zero-init, so step-0 forward equals the no-adapter baseline.
4. **Gradient flow — gated_input.** Gradients reach interaction +
   body-action projection AND gate parameters.
5. **Gradient flow — per_layer_adapter.** Gradients reach every layer's
   adapter.
6. **Branch separation.** Interaction and body-action MLP last-layer
   weight shapes match (cfg.d_model, cfg.d_model) and are independent
   nn.Modules (separate parameter trees).

Reuses ``_build_denoiser_config`` / ``_make_synthetic_cond`` helpers
from ``test_oracle_hint_injection`` for consistency with the R27 tests.
"""
from __future__ import annotations

import torch

from piano.models.motion_anchordiff import (
    AnchorDenoiser,
    AnchorDenoiserConfig,
    ZIntDims,
)


def _build_cfg(
    *,
    use_interaction: bool = False,
    interaction_dim: int = 13,
    use_body_action: bool = False,
    body_action_dim: int = 24,
    injection_mode: str = "input_add",
    zero_init_adapters: bool = True,
) -> AnchorDenoiserConfig:
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
        use_oracle_interaction_hint=use_interaction,
        oracle_hint_dim=interaction_dim if use_interaction else 0,
        use_body_action_hint=use_body_action,
        body_action_hint_dim=body_action_dim if use_body_action else 0,
        oracle_hint_injection_mode=injection_mode,
        separate_hint_branches=True,
        zero_init_hint_adapters=zero_init_adapters,
        d_model=64,
        n_layers=2,
        n_heads=2,
        ff_mult=2,
        dropout=0.0,
        max_seq_length=32,
    )


def _make_cond(B, T, cfg, seed=0):
    g = torch.Generator().manual_seed(seed)
    cond = {
        "z_int": torch.randn(B, T, cfg.z_int.total, generator=g),
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
    if cfg.use_body_action_hint:
        cond["body_action_hint"] = torch.randn(
            B, T, cfg.body_action_hint_dim, generator=g,
        )
    return cond


def _share_weights(ref: torch.nn.Module, new: torch.nn.Module):
    """Copy shared weights ref -> new; leaves any new-only keys at init."""
    ref_keys = set(ref.state_dict())
    new_keys = set(new.state_dict())
    extra = new_keys - ref_keys
    missing, unexpected = new.load_state_dict(ref.state_dict(), strict=False)
    assert sorted(missing) == sorted(extra), (missing, extra)
    assert unexpected == [], unexpected
    return extra


# ---------------------------------------------------------------------------
# 1. zero-init invariant: body-action branch (input_add)
# ---------------------------------------------------------------------------


def test_zero_init_invariant_body_action_input_add():
    B, T = 2, 16
    cfg_ref = _build_cfg(use_interaction=False, use_body_action=False)
    cfg_new = _build_cfg(use_interaction=False, use_body_action=True)

    torch.manual_seed(20260525)
    m_ref = AnchorDenoiser(cfg_ref).eval()
    torch.manual_seed(99)
    m_new = AnchorDenoiser(cfg_new).eval()
    extra = _share_weights(m_ref, m_new)
    assert all(k.startswith("body_action_hint_proj.") for k in extra), extra

    cond_ref = _make_cond(B, T, cfg_ref, seed=0)
    cond_new = _make_cond(B, T, cfg_new, seed=0)
    x_t = torch.randn(B, T, cfg_ref.motion_dim)
    t = torch.zeros(B, dtype=torch.long)
    with torch.no_grad():
        out_ref = m_ref(x_t, t, cond_ref, cond_drop_mask=None)
        out_new = m_new(x_t, t, cond_new, cond_drop_mask=None)
    assert torch.equal(out_ref, out_new), (
        f"body-action zero-init invariant violated; "
        f"max|Δ|={(out_ref - out_new).abs().max().item():.3e}"
    )


# ---------------------------------------------------------------------------
# 2. zero-init invariant: gated_input
# ---------------------------------------------------------------------------


def test_zero_init_invariant_gated_input():
    """gated_input: the projection's last layer is zero (so emb=0) AND
    the gate weight is zero with bias -3 (so sigmoid(0+(-3))≈0.047, but
    multiplied by emb=0 → contribution=0). Output must equal baseline."""
    B, T = 2, 16
    cfg_ref = _build_cfg(use_interaction=False, use_body_action=False)
    cfg_new = _build_cfg(
        use_interaction=True, use_body_action=True,
        injection_mode="gated_input",
    )

    torch.manual_seed(20260525)
    m_ref = AnchorDenoiser(cfg_ref).eval()
    torch.manual_seed(77)
    m_new = AnchorDenoiser(cfg_new).eval()
    _share_weights(m_ref, m_new)

    cond_ref = _make_cond(B, T, cfg_ref, seed=1)
    cond_new = _make_cond(B, T, cfg_new, seed=1)
    x_t = torch.randn(B, T, cfg_ref.motion_dim)
    t = torch.zeros(B, dtype=torch.long)
    with torch.no_grad():
        out_ref = m_ref(x_t, t, cond_ref, cond_drop_mask=None)
        out_new = m_new(x_t, t, cond_new, cond_drop_mask=None)
    max_diff = (out_ref - out_new).abs().max().item()
    assert max_diff < 1e-6, (
        f"gated_input zero-init invariant violated; max|Δ|={max_diff:.3e}"
    )


# ---------------------------------------------------------------------------
# 3. zero-init invariant: per_layer_adapter
# ---------------------------------------------------------------------------


def test_zero_init_invariant_per_layer_adapter():
    B, T = 2, 16
    cfg_ref = _build_cfg(use_interaction=False, use_body_action=False)
    cfg_new = _build_cfg(
        use_interaction=True, use_body_action=True,
        injection_mode="per_layer_adapter",
    )

    torch.manual_seed(20260525)
    m_ref = AnchorDenoiser(cfg_ref).eval()
    torch.manual_seed(88)
    m_new = AnchorDenoiser(cfg_new).eval()
    _share_weights(m_ref, m_new)

    cond_ref = _make_cond(B, T, cfg_ref, seed=2)
    cond_new = _make_cond(B, T, cfg_new, seed=2)
    x_t = torch.randn(B, T, cfg_ref.motion_dim)
    t = torch.zeros(B, dtype=torch.long)
    with torch.no_grad():
        out_ref = m_ref(x_t, t, cond_ref, cond_drop_mask=None)
        out_new = m_new(x_t, t, cond_new, cond_drop_mask=None)
    max_diff = (out_ref - out_new).abs().max().item()
    assert max_diff < 1e-6, (
        f"per_layer_adapter zero-init invariant violated; "
        f"max|Δ|={max_diff:.3e}"
    )


# ---------------------------------------------------------------------------
# 4. gradient flow: gated_input
# ---------------------------------------------------------------------------


def test_gradients_reach_gated_input_branches():
    B, T = 2, 16
    cfg = _build_cfg(
        use_interaction=True, use_body_action=True,
        injection_mode="gated_input",
    )
    torch.manual_seed(20260525)
    model = AnchorDenoiser(cfg).train()

    # Wake up V12FinalLayer (AdaLN-Zero gate kills gradient at init).
    for p in model.v12_final_layer.parameters():
        with torch.no_grad():
            p.normal_(mean=0.0, std=0.05)
    # Wake up hint projections (zero-init last layer kills downstream grads).
    for proj in (model.oracle_hint_proj, model.body_action_hint_proj):
        with torch.no_grad():
            proj[-1].weight.normal_(mean=0.0, std=0.1)
            proj[-1].bias.normal_(mean=0.0, std=0.1)
    # Open the gates a bit (currently bias=-3 so sigmoid≈0.05; small but
    # non-zero enough for gradient).
    # Note: gate weight is zero-init so gradient still needs activation
    # via sigmoid(bias), which is non-zero — OK.

    cond = _make_cond(B, T, cfg, seed=3)
    x_t = torch.randn(B, T, cfg.motion_dim)
    t = torch.zeros(B, dtype=torch.long)
    out = model(x_t, t, cond, cond_drop_mask=None)
    loss = (out ** 2).mean()
    loss.backward()

    # interaction projection
    for name, p in model.oracle_hint_proj.named_parameters():
        assert p.grad is not None and p.grad.abs().sum().item() > 0.0, (
            f"oracle_hint_proj.{name} got no gradient"
        )
    # body-action projection
    for name, p in model.body_action_hint_proj.named_parameters():
        assert p.grad is not None and p.grad.abs().sum().item() > 0.0, (
            f"body_action_hint_proj.{name} got no gradient"
        )
    # interaction gate
    assert model.interaction_gate.bias.grad is not None
    assert model.interaction_gate.bias.grad.abs().sum().item() > 0.0
    # body-action gate
    assert model.body_action_gate.bias.grad is not None
    assert model.body_action_gate.bias.grad.abs().sum().item() > 0.0


# ---------------------------------------------------------------------------
# 5. gradient flow: per_layer_adapter
# ---------------------------------------------------------------------------


def test_gradients_reach_per_layer_adapters():
    B, T = 2, 16
    cfg = _build_cfg(
        use_interaction=True, use_body_action=True,
        injection_mode="per_layer_adapter",
    )
    torch.manual_seed(20260525)
    model = AnchorDenoiser(cfg).train()
    # Wake up V12FinalLayer + hint projections + every adapter's last
    # linear (otherwise the chain-rule downstream is zero).
    for p in model.v12_final_layer.parameters():
        with torch.no_grad():
            p.normal_(mean=0.0, std=0.05)
    for proj in (model.oracle_hint_proj, model.body_action_hint_proj):
        with torch.no_grad():
            proj[-1].weight.normal_(mean=0.0, std=0.1)
            proj[-1].bias.normal_(mean=0.0, std=0.1)
    for ad_list in (model.interaction_adapters, model.body_action_adapters):
        for ad in ad_list:
            with torch.no_grad():
                ad[-1].weight.normal_(mean=0.0, std=0.1)
                ad[-1].bias.normal_(mean=0.0, std=0.1)

    cond = _make_cond(B, T, cfg, seed=4)
    x_t = torch.randn(B, T, cfg.motion_dim)
    t = torch.zeros(B, dtype=torch.long)
    out = model(x_t, t, cond, cond_drop_mask=None)
    loss = (out ** 2).mean()
    loss.backward()

    for i, ad in enumerate(model.interaction_adapters):
        for name, p in ad.named_parameters():
            assert p.grad is not None and p.grad.abs().sum().item() > 0.0, (
                f"interaction_adapters[{i}].{name} got no gradient"
            )
    for i, ad in enumerate(model.body_action_adapters):
        for name, p in ad.named_parameters():
            assert p.grad is not None and p.grad.abs().sum().item() > 0.0, (
                f"body_action_adapters[{i}].{name} got no gradient"
            )


# ---------------------------------------------------------------------------
# 6. branch separation: independent parameter trees, expected shapes
# ---------------------------------------------------------------------------


def test_branch_independence_and_shapes():
    cfg = _build_cfg(
        use_interaction=True, use_body_action=True,
        injection_mode="input_add",
    )
    torch.manual_seed(0)
    m = AnchorDenoiser(cfg)
    assert m.oracle_hint_proj is not None
    assert m.body_action_hint_proj is not None
    # They are independent nn.Sequential modules.
    assert m.oracle_hint_proj is not m.body_action_hint_proj
    # Last-layer shape == (d_model, d_model).
    assert tuple(m.oracle_hint_proj[-1].weight.shape) == (cfg.d_model, cfg.d_model)
    assert tuple(m.body_action_hint_proj[-1].weight.shape) == (cfg.d_model, cfg.d_model)
    # First-layer in-features must match dims.
    assert m.oracle_hint_proj[0].in_features == cfg.oracle_hint_dim
    assert m.body_action_hint_proj[0].in_features == cfg.body_action_hint_dim
