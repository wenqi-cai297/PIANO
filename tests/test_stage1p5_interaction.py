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


# ──────────────────────────────────────────────────────────────────────────
# R33 — per-block obj_xattn integration
# ──────────────────────────────────────────────────────────────────────────


def test_r33_default_flag_off():
    cfg = _make_cfg()
    assert cfg.enable_per_block_obj_xattn is False
    model = Stage1p5Denoiser(cfg)
    assert model.use_per_block_obj_xattn is False
    for block in model.v12_blocks:
        assert block.enable_obj_xattn is False


def test_r33_flag_on_creates_per_block_xattn_modules():
    cfg = _make_cfg(enable_per_block_obj_xattn=True)
    model = Stage1p5Denoiser(cfg)
    assert model.use_per_block_obj_xattn is True
    for block in model.v12_blocks:
        assert block.enable_obj_xattn is True
        assert block.obj_xattn is not None
        assert block.adaLN_modulation_xattn is not None


def test_r33_step_zero_output_still_exactly_zero():
    """Zero-init AdaLN of cross-attn must preserve the step-0 identity."""
    B, T = 2, 16
    cfg = _make_cfg(enable_per_block_obj_xattn=True)
    model = Stage1p5Denoiser(cfg)
    model.eval()
    x_t = torch.randn(B, T, STAGE1P5_TOTAL_DIM)
    t = torch.randint(0, 1000, (B,))
    cond = _make_cond(B, T)
    with torch.no_grad():
        out = model(x_t, t, cond, cond_drop_mask=None)
    assert out.abs().max().item() == 0.0, (
        "R33 cross-attn AdaLN zero-init failed; step-0 output should "
        "still be exactly 0."
    )


def test_r33_forward_shape():
    B, T = 2, 16
    cfg = _make_cfg(enable_per_block_obj_xattn=True)
    model = Stage1p5Denoiser(cfg)
    model.eval()
    x_t = torch.randn(B, T, STAGE1P5_TOTAL_DIM)
    t = torch.randint(0, 1000, (B,))
    cond = _make_cond(B, T)
    with torch.no_grad():
        out = model(x_t, t, cond, cond_drop_mask=None)
    assert out.shape == (B, T, STAGE1P5_TOTAL_DIM)


def test_r33_perturbed_gate_makes_obj_tokens_load_bearing():
    """With AdaLN-xattn gates perturbed non-zero, swapping object_tokens
    changes the prediction — confirming the per-block xattn pathway is
    actually used."""
    cfg = _make_cfg(enable_per_block_obj_xattn=True)
    model = Stage1p5Denoiser(cfg)

    # Make the V12 final layer non-zero so we can read out a non-zero
    # response, and perturb each block's xattn AdaLN gate.
    with torch.no_grad():
        torch.nn.init.xavier_uniform_(model.v12_final.linear_c41.weight)
        torch.nn.init.xavier_uniform_(model.v12_final.linear_s4.weight)
        torch.nn.init.xavier_uniform_(model.v12_final.adaLN_modulation[-1].weight)
        for block in model.v12_blocks:
            torch.nn.init.xavier_uniform_(
                block.adaLN_modulation_xattn[-1].weight
            )
            block.adaLN_modulation_xattn[-1].bias.fill_(0.1)

    B, T = 2, 8
    x_t = torch.randn(B, T, STAGE1P5_TOTAL_DIM)
    t = torch.randint(0, 1000, (B,))
    cond_a = _make_cond(B, T)
    cond_b = _make_cond(B, T)
    # Same x_t/t and same other cond fields, only object_tokens differ.
    cond_b["object_tokens"] = cond_a["object_tokens"] * 5.0 + 1.0
    cond_b["object_world_traj"] = cond_a["object_world_traj"]
    cond_b["stage1_coarse"] = cond_a["stage1_coarse"]
    cond_b["text"] = cond_a["text"]

    model.eval()
    with torch.no_grad():
        out_a = model(x_t, t, cond_a, cond_drop_mask=None)
        out_b = model(x_t, t, cond_b, cond_drop_mask=None)
    assert not torch.allclose(out_a, out_b, atol=1e-4)


def test_r33_no_state_dict_keys_when_flag_off():
    """A model trained with flag=False should serialize without R33 keys
    (V0/V7 backward compatibility)."""
    cfg = _make_cfg(enable_per_block_obj_xattn=False)
    model = Stage1p5Denoiser(cfg)
    sd = model.state_dict()
    for k in sd:
        assert "norm_xattn" not in k
        assert "adaLN_modulation_xattn" not in k
        assert "v12_blocks" not in k or "obj_xattn" not in k.split(".")[-1] or k.endswith("obj_xattn")
    # And in the symmetric "flag on" case, we DO see the keys.
    cfg_r33 = _make_cfg(enable_per_block_obj_xattn=True)
    model_r33 = Stage1p5Denoiser(cfg_r33)
    sd_r33 = model_r33.state_dict()
    has_per_block_xattn = any("v12_blocks.0.obj_xattn" in k for k in sd_r33)
    has_per_block_adaln = any(
        "v12_blocks.0.adaLN_modulation_xattn" in k for k in sd_r33
    )
    assert has_per_block_xattn
    assert has_per_block_adaln


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
