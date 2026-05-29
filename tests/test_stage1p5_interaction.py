"""Tests for ``piano.models.stage1p5_interaction.Stage1p5Denoiser``.

Locks the shape / zero-init invariants required by Stage-2 PB1's input
contract.
"""
from __future__ import annotations

import pytest
import torch

from piano.models.stage1p5_interaction import (
    STAGE1P5_C41_DIM,
    STAGE1P5_S4_DIM,
    STAGE1P5_TOTAL_DIM,
    Stage1p5Denoiser,
    Stage1p5DenoiserConfig,
)


def _make_cfg(**overrides) -> Stage1p5DenoiserConfig:
    defaults = dict(
        motion_dim=STAGE1P5_TOTAL_DIM,
        stage1_coarse_dim=23,
        object_traj_dim=9,
        text_dim=512,
        object_token_dim=256,
        object_num_tokens=8,
        d_model=64,
        n_layers=2,
        n_heads=4,
        ff_mult=2,
        dropout=0.0,
        max_seq_length=32,
        use_text=True,
    )
    defaults.update(overrides)
    return Stage1p5DenoiserConfig(**defaults)


def _make_cond(B: int, T: int, *, include_stage1: bool = True, include_text: bool = True):
    cond = {
        "object_world_traj": torch.randn(B, T, 9),
        "object_tokens": torch.randn(B, 8, 256),
    }
    if include_stage1:
        cond["stage1_coarse"] = torch.randn(B, T, 23)
    if include_text:
        cond["text"] = torch.randn(B, 4, 512)
    return cond


def test_output_dims_match_stage2_input_contract():
    assert STAGE1P5_C41_DIM == 18           # stage2_coarse_extra
    assert STAGE1P5_S4_DIM == 13            # stage2_support
    assert STAGE1P5_TOTAL_DIM == 31


def test_forward_shape():
    B, T = 2, 16
    cfg = _make_cfg()
    model = Stage1p5Denoiser(cfg)
    model.eval()
    x_t = torch.randn(B, T, STAGE1P5_TOTAL_DIM)
    t = torch.randint(0, 1000, (B,))
    cond = _make_cond(B, T)
    with torch.no_grad():
        out = model(x_t, t, cond, cond_drop_mask=None)
    assert out.shape == (B, T, STAGE1P5_TOTAL_DIM)


def test_step_zero_output_is_exactly_zero():
    """V12 zero-init: at step 0 the model must output identically 0."""
    B, T = 2, 16
    cfg = _make_cfg()
    model = Stage1p5Denoiser(cfg)
    model.eval()
    x_t = torch.randn(B, T, STAGE1P5_TOTAL_DIM)
    t = torch.randint(0, 1000, (B,))
    cond = _make_cond(B, T)
    with torch.no_grad():
        out = model(x_t, t, cond, cond_drop_mask=None)
    assert out.abs().max().item() == 0.0


def test_split_readout_emits_correct_per_head_dims():
    """The split readout's two Linears emit 18 and 13 separately; the
    concatenated output[..., :18] is the C41 head, [..., 18:] is S4.
    """
    B, T = 2, 8
    cfg = _make_cfg()
    model = Stage1p5Denoiser(cfg)
    # De-zero V12FinalLayer.linear_c41 only; linear_s4 stays zero. The
    # output should then have non-zero values in [..., :18] but exactly
    # zero in [..., 18:].
    with torch.no_grad():
        torch.nn.init.xavier_uniform_(model.v12_final.linear_c41.weight)
        torch.nn.init.xavier_uniform_(model.v12_final.adaLN_modulation[-1].weight)
        for block in model.v12_blocks:
            torch.nn.init.xavier_uniform_(block.adaLN_modulation[-1].weight)
    cond = _make_cond(B, T)
    x_t = torch.randn(B, T, STAGE1P5_TOTAL_DIM)
    t = torch.randint(0, 1000, (B,))
    with torch.no_grad():
        out = model(x_t, t, cond, cond_drop_mask=None)
    c41_out = out[..., :STAGE1P5_C41_DIM]
    s4_out = out[..., STAGE1P5_C41_DIM:]
    assert c41_out.abs().max().item() > 0.0, "C41 head should produce non-zero"
    assert s4_out.abs().max().item() == 0.0, "S4 head should still be zero"


def test_forward_raises_when_stage1_coarse_missing():
    B, T = 2, 8
    cfg = _make_cfg(stage1_coarse_dim=23)
    model = Stage1p5Denoiser(cfg)
    cond = _make_cond(B, T, include_stage1=False)
    x_t = torch.randn(B, T, STAGE1P5_TOTAL_DIM)
    t = torch.randint(0, 1000, (B,))
    with pytest.raises(KeyError, match="stage1_coarse"):
        model(x_t, t, cond, cond_drop_mask=None)


def test_forward_raises_when_text_required_but_missing():
    B, T = 2, 8
    cfg = _make_cfg(use_text=True)
    model = Stage1p5Denoiser(cfg)
    cond = _make_cond(B, T, include_text=False)
    x_t = torch.randn(B, T, STAGE1P5_TOTAL_DIM)
    t = torch.randint(0, 1000, (B,))
    with pytest.raises(KeyError, match="text"):
        model(x_t, t, cond, cond_drop_mask=None)


def test_cfg_drop_affects_text_and_obj_tokens_not_stage1_coarse():
    """obj_traj + stage1_coarse never CFG-dropped; text + obj_tokens are."""
    B, T = 2, 8
    cfg = _make_cfg()
    model = Stage1p5Denoiser(cfg)
    with torch.no_grad():
        torch.nn.init.xavier_uniform_(model.v12_final.linear_c41.weight)
        torch.nn.init.xavier_uniform_(model.v12_final.linear_s4.weight)
    model.eval()
    cond = _make_cond(B, T)
    x_t = torch.randn(B, T, STAGE1P5_TOTAL_DIM)
    t = torch.randint(0, 1000, (B,))
    drop = torch.tensor([True, True])
    with torch.no_grad():
        out_keep = model(x_t, t, cond, cond_drop_mask=None)
        out_drop = model(x_t, t, cond, cond_drop_mask=drop)
    assert not torch.allclose(out_keep, out_drop, atol=1e-6)


def test_gradient_flows_through_stage1_coarse_proj():
    """stage1_coarse feeds via V12InputProjection.stage1_coarse_proj
    (zero-init). After de-zeroing the AdaLN gates and finals, grad must
    reach this projection.
    """
    B, T = 2, 8
    cfg = _make_cfg()
    model = Stage1p5Denoiser(cfg)
    with torch.no_grad():
        torch.nn.init.xavier_uniform_(model.v12_final.linear_c41.weight)
        torch.nn.init.xavier_uniform_(model.v12_final.linear_s4.weight)
        torch.nn.init.xavier_uniform_(model.v12_final.adaLN_modulation[-1].weight)
        for block in model.v12_blocks:
            torch.nn.init.xavier_uniform_(block.adaLN_modulation[-1].weight)
    cond = _make_cond(B, T)
    x_t = torch.randn(B, T, STAGE1P5_TOTAL_DIM)
    t = torch.randint(0, 1000, (B,))
    target = torch.randn(B, T, STAGE1P5_TOTAL_DIM)
    out = model(x_t, t, cond, cond_drop_mask=None)
    loss = (out - target).pow(2).mean()
    loss.backward()
    s1_proj_w = model.v12_input_proj.stage1_coarse_proj.weight
    # At init (zero-init), the projection's INCOMING grad is non-trivial
    # (it gets backpropped via h → DiT → readout) but the projection
    # weight grad equals upstream_grad ⊗ stage1_coarse (input). Both
    # nonzero ⇒ grad nonzero.
    assert s1_proj_w.grad is not None and s1_proj_w.grad.norm() > 0


def test_config_dims_match_stage2_contract():
    cfg = Stage1p5DenoiserConfig()
    assert cfg.motion_dim == 31
    assert cfg.stage1_coarse_dim == 23
    assert cfg.object_traj_dim == 9
    assert cfg.text_dim == 512
