"""Tests for R33 — per-block object cross-attention in ConditionedEncoderLayer.

Invariants:
  - enable_obj_xattn=False (default): block forward unchanged; state_dict
    has no new keys (V0/V7/V8 ckpt compatibility).
  - enable_obj_xattn=True: AdaLN-Zero on the cross-attn sub-block means
    its contribution is 0 at init, so the step-0 forward equals the
    enable_obj_xattn=False forward bit-identical (given equal weights
    on the shared modules).
  - enable_obj_xattn=True without obj_kv → ValueError.
  - With non-zero AdaLN_xattn weights, cross-attn contributes (changes
    output when obj_kv changes).
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from piano.models.dit_blocks import (
    ConditionedEncoderLayer,
    GlobalCondSummary,
    V12FinalLayer,
    V12InputProjection,
    initialize_weights_v12,
)


# ──────────────────────────────────────────────────────────────────────────
# Defaults: enable_obj_xattn=False keeps the layer as-is
# ──────────────────────────────────────────────────────────────────────────


def test_layer_default_has_no_obj_xattn_submodules():
    layer = ConditionedEncoderLayer(d_model=32, n_heads=4)
    assert layer.enable_obj_xattn is False
    assert layer.obj_xattn is None
    assert layer.norm_xattn is None
    assert layer.adaLN_modulation_xattn is None


def test_layer_default_state_dict_matches_pre_r33_layout():
    """V0/V7/V8 ckpts must load without unexpected keys when the layer
    is constructed with default args."""
    layer = ConditionedEncoderLayer(d_model=32, n_heads=4)
    sd = layer.state_dict()
    # No R33-specific keys.
    for k in sd:
        assert "xattn" not in k or k == "obj_xattn"  # no obj_xattn here
        assert "norm_xattn" not in k
        assert "adaLN_modulation_xattn" not in k


def test_layer_default_forward_doesnt_need_obj_kv():
    layer = ConditionedEncoderLayer(d_model=32, n_heads=4)
    x = torch.randn(2, 8, 32)
    c = torch.randn(2, 32)
    out = layer(x, c)
    assert out.shape == (2, 8, 32)
    assert torch.isfinite(out).all()


# ──────────────────────────────────────────────────────────────────────────
# enable_obj_xattn=True: AdaLN-Zero preserves step-0 identity
# ──────────────────────────────────────────────────────────────────────────


def test_r33_xattn_zero_init_yields_bit_identical_to_default():
    """When the cross-attn AdaLN is zero-init'd, the cross-attn sub-block
    contributes 0, so a R33 layer with enable_obj_xattn=True has the
    same forward as the same layer with it False (given equal weights
    on shared modules)."""
    d_model = 32
    n_heads = 4
    layer_v0 = ConditionedEncoderLayer(d_model, n_heads, enable_obj_xattn=False)
    layer_r33 = ConditionedEncoderLayer(d_model, n_heads, enable_obj_xattn=True)

    # Copy shared weights v0 -> r33.
    layer_r33.norm1 = layer_v0.norm1
    layer_r33.norm2 = layer_v0.norm2
    layer_r33.self_attn.load_state_dict(layer_v0.self_attn.state_dict())
    layer_r33.mlp.load_state_dict(layer_v0.mlp.state_dict())
    layer_r33.adaLN_modulation.load_state_dict(layer_v0.adaLN_modulation.state_dict())

    # Apply the v12 zero-init to a 1-block list so the cross-attn AdaLN
    # is zero'd (the rest of zero-init is idempotent on already-zero
    # main AdaLN weights, but we replicate it consistently).
    final = V12FinalLayer(d_model=d_model, motion_dim=23)
    proj = V12InputProjection(motion_dim=23, obj_traj_dim=9, d_model=d_model)
    cs = GlobalCondSummary(d_model=d_model)
    initialize_weights_v12(
        input_proj=proj,
        blocks=nn.ModuleList([layer_v0]),
        final_layer=final,
        cond_summary=cs,
    )
    initialize_weights_v12(
        input_proj=proj,
        blocks=nn.ModuleList([layer_r33]),
        final_layer=final,
        cond_summary=cs,
    )

    torch.manual_seed(7)
    x = torch.randn(2, 8, d_model)
    c = torch.randn(2, d_model)
    obj_kv = torch.randn(2, 16, d_model)

    out_v0 = layer_v0(x, c)
    out_r33 = layer_r33(x, c, obj_kv=obj_kv)
    assert torch.allclose(out_v0, out_r33, atol=1e-6)


def test_r33_xattn_missing_obj_kv_raises():
    layer = ConditionedEncoderLayer(d_model=32, n_heads=4, enable_obj_xattn=True)
    x = torch.randn(2, 8, 32)
    c = torch.randn(2, 32)
    with pytest.raises(ValueError, match="enable_obj_xattn=True"):
        layer(x, c, obj_kv=None)


def test_r33_xattn_perturbed_gate_changes_output_with_obj_kv():
    """After we manually set the AdaLN-xattn gate non-zero, swapping
    obj_kv should change the forward output. Confirms the cross-attn
    is wired into the gradient graph."""
    layer = ConditionedEncoderLayer(d_model=32, n_heads=4, enable_obj_xattn=True)

    # Make AdaLN_xattn non-zero by perturbing the final Linear.
    with torch.no_grad():
        layer.adaLN_modulation_xattn[-1].weight.fill_(0.1)
        layer.adaLN_modulation_xattn[-1].bias.fill_(0.1)

    x = torch.randn(2, 8, 32)
    c = torch.randn(2, 32)
    obj_kv_a = torch.randn(2, 16, 32)
    obj_kv_b = torch.randn(2, 16, 32) * 5.0   # different content

    out_a = layer(x, c, obj_kv=obj_kv_a)
    out_b = layer(x, c, obj_kv=obj_kv_b)
    assert not torch.allclose(out_a, out_b, atol=1e-4)


def test_r33_gradient_flows_through_obj_xattn():
    layer = ConditionedEncoderLayer(d_model=32, n_heads=4, enable_obj_xattn=True)
    # Non-zero AdaLN so cross-attn actually contributes.
    with torch.no_grad():
        layer.adaLN_modulation_xattn[-1].weight.fill_(0.1)
        layer.adaLN_modulation_xattn[-1].bias.fill_(0.1)

    x = torch.randn(2, 8, 32)
    c = torch.randn(2, 32)
    obj_kv = torch.randn(2, 16, 32, requires_grad=True)
    out = layer(x, c, obj_kv=obj_kv)
    out.sum().backward()
    assert obj_kv.grad is not None
    assert torch.isfinite(obj_kv.grad).all()
    assert obj_kv.grad.abs().sum().item() > 0.0


# ──────────────────────────────────────────────────────────────────────────
# initialize_weights_v12 — R33 zero-init invariant
# ──────────────────────────────────────────────────────────────────────────


def test_initialize_weights_v12_zeros_xattn_adaln():
    """The v12 init recipe must zero the R33 cross-attn AdaLN final Linear,
    or step-0 identity is violated."""
    d_model = 32
    block = ConditionedEncoderLayer(
        d_model, n_heads=4, enable_obj_xattn=True,
    )
    # Set non-zero to verify init zeros them.
    with torch.no_grad():
        block.adaLN_modulation_xattn[-1].weight.fill_(1.0)
        block.adaLN_modulation_xattn[-1].bias.fill_(1.0)

    final = V12FinalLayer(d_model=d_model, motion_dim=23)
    proj = V12InputProjection(motion_dim=23, obj_traj_dim=9, d_model=d_model)
    cs = GlobalCondSummary(d_model=d_model)
    initialize_weights_v12(
        input_proj=proj,
        blocks=nn.ModuleList([block]),
        final_layer=final,
        cond_summary=cs,
    )

    assert block.adaLN_modulation_xattn[-1].weight.abs().max().item() == 0.0
    assert block.adaLN_modulation_xattn[-1].bias.abs().max().item() == 0.0
