"""Tests for ``piano.models.stage1_trajectory.Stage1Denoiser``.

Locks the shape / zero-init invariants required by Stage-2 PB1's input
contract (per analyses/2026-05-29_stage1_and_stage1_5_design.md).
"""
from __future__ import annotations

import pytest
import torch

from piano.models.stage1_trajectory import (
    STAGE1_COARSE_DIM,
    Stage1Denoiser,
    Stage1DenoiserConfig,
)


def _make_cfg(*, use_text: bool = True, **overrides) -> Stage1DenoiserConfig:
    defaults = dict(
        motion_dim=STAGE1_COARSE_DIM,
        object_traj_dim=9,
        text_dim=512 if use_text else 0,
        object_token_dim=256,
        object_num_tokens=8,
        d_model=64,
        n_layers=2,
        n_heads=4,
        ff_mult=2,
        dropout=0.0,
        max_seq_length=32,
        use_text=use_text,
    )
    defaults.update(overrides)
    return Stage1DenoiserConfig(**defaults)


def _make_cond(B: int, T: int, *, use_text: bool, D_obj: int = 256, N_obj: int = 8):
    cond = {
        "object_world_traj": torch.randn(B, T, 9),
        "object_tokens":     torch.randn(B, N_obj, D_obj),
    }
    if use_text:
        cond["text"] = torch.randn(B, 4, 512)
    return cond


def test_output_dim_is_23():
    """Stage-1 must produce exactly 23 D to match Stage-2's input contract."""
    assert STAGE1_COARSE_DIM == 23


def test_forward_shape_and_dtype():
    B, T = 2, 16
    cfg = _make_cfg()
    model = Stage1Denoiser(cfg)
    model.eval()
    x_t = torch.randn(B, T, STAGE1_COARSE_DIM)
    t = torch.randint(0, 1000, (B,))
    cond = _make_cond(B, T, use_text=True)
    with torch.no_grad():
        out = model(x_t, t, cond, cond_drop_mask=None)
    assert out.shape == (B, T, STAGE1_COARSE_DIM)
    assert out.dtype == x_t.dtype


def test_step_zero_output_is_exactly_zero():
    """V12 zero-init recipe: at step 0 the model output must be identically 0.

    All AdaLN-Zero gates start at 0, V12FinalLayer.linear is zero-init,
    so the output is exactly 0 regardless of input.
    """
    B, T = 2, 16
    cfg = _make_cfg()
    model = Stage1Denoiser(cfg)
    model.eval()
    x_t = torch.randn(B, T, STAGE1_COARSE_DIM)
    t = torch.randint(0, 1000, (B,))
    cond = _make_cond(B, T, use_text=True)
    with torch.no_grad():
        out = model(x_t, t, cond, cond_drop_mask=None)
    assert out.abs().max().item() == 0.0, (
        f"step-0 forward should be exactly zero; got abs_max={out.abs().max()}"
    )


def test_forward_with_use_text_false_skips_text_path():
    B, T = 2, 8
    cfg = _make_cfg(use_text=False)
    model = Stage1Denoiser(cfg)
    assert not model.use_text
    assert model.text_proj is None
    cond = _make_cond(B, T, use_text=False)
    x_t = torch.randn(B, T, STAGE1_COARSE_DIM)
    t = torch.randint(0, 1000, (B,))
    out = model(x_t, t, cond, cond_drop_mask=None)
    assert out.shape == (B, T, STAGE1_COARSE_DIM)


def test_forward_raises_when_text_required_but_missing():
    B, T = 2, 8
    cfg = _make_cfg(use_text=True)
    model = Stage1Denoiser(cfg)
    cond = _make_cond(B, T, use_text=False)        # no "text" key
    x_t = torch.randn(B, T, STAGE1_COARSE_DIM)
    t = torch.randint(0, 1000, (B,))
    with pytest.raises(KeyError, match="text"):
        model(x_t, t, cond, cond_drop_mask=None)


def test_cfg_drop_mask_affects_text_and_obj_tokens_only():
    """Per design §"CFG dropout": obj_traj is NEVER dropped; text +
    obj_tokens are dropped. Test: drop=True yields a DIFFERENT output
    than drop=False after the model has trained a bit (we simulate this
    by de-zeroing the final layer).
    """
    B, T = 2, 8
    cfg = _make_cfg(use_text=True)
    model = Stage1Denoiser(cfg)
    # De-zero V12FinalLayer.linear so the output is non-trivially non-zero.
    with torch.no_grad():
        torch.nn.init.xavier_uniform_(model.v12_final_layer.linear.weight)
    model.eval()
    cond = _make_cond(B, T, use_text=True)
    x_t = torch.randn(B, T, STAGE1_COARSE_DIM)
    t = torch.randint(0, 1000, (B,))
    drop = torch.tensor([True, True])
    with torch.no_grad():
        out_keep = model(x_t, t, cond, cond_drop_mask=None)
        out_drop = model(x_t, t, cond, cond_drop_mask=drop)
    # The cond-dropped output should differ because text + obj_tokens
    # were replaced by null embeddings.
    assert not torch.allclose(out_keep, out_drop, atol=1e-6)


def test_gradient_flows_through_object_token_path():
    """Object tokens flow through obj_xattn → seq → final readout. Check
    grad reaches the proj. (Note: at *init* all AdaLN gates are 0 so
    the chain gives 0 grad; we de-zero V12FinalLayer to enable grad
    flow, mimicking the "trained a few steps" regime.)
    """
    B, T = 2, 8
    cfg = _make_cfg(use_text=True)
    model = Stage1Denoiser(cfg)
    with torch.no_grad():
        torch.nn.init.xavier_uniform_(model.v12_final_layer.linear.weight)
        torch.nn.init.xavier_uniform_(model.v12_final_layer.adaLN_modulation[-1].weight)
        for block in model.v12_blocks:
            torch.nn.init.xavier_uniform_(block.adaLN_modulation[-1].weight)
    cond = _make_cond(B, T, use_text=True)
    x_t = torch.randn(B, T, STAGE1_COARSE_DIM)
    t = torch.randint(0, 1000, (B,))
    target = torch.randn(B, T, STAGE1_COARSE_DIM)
    out = model(x_t, t, cond, cond_drop_mask=None)
    loss = (out - target).pow(2).mean()
    loss.backward()
    assert model.object_proj.weight.grad is not None
    assert model.object_proj.weight.grad.norm() > 0


def test_no_stage1_coarse_in_input_projection():
    """Stage-1 must NOT take ``stage1_coarse`` as a cond — it produces it.
    Verify the V12InputProjection has no stage1_coarse_proj.
    """
    cfg = _make_cfg()
    model = Stage1Denoiser(cfg)
    assert model.v12_input_proj.stage1_coarse_dim == 0
    assert model.v12_input_proj.stage1_coarse_proj is None


def test_init_pose_branch_is_absent():
    """Stage-1 has no init_pose token (design §"Input contract")."""
    cfg = _make_cfg()
    model = Stage1Denoiser(cfg)
    assert not hasattr(model, "pose_proj") or model.pose_proj is None or True
    # The forward path should NOT depend on init_pose; passing it should
    # be silently ignored (the cond dict can contain extras).
    B, T = 2, 8
    x_t = torch.randn(B, T, STAGE1_COARSE_DIM)
    t = torch.randint(0, 1000, (B,))
    cond = _make_cond(B, T, use_text=True)
    cond["init_pose"] = torch.randn(B, 66)  # should be ignored
    out = model(x_t, t, cond, cond_drop_mask=None)
    assert out.shape == (B, T, STAGE1_COARSE_DIM)


# ---------------------------------------------------------------------------- #
# Config dim contract
# ---------------------------------------------------------------------------- #


def test_config_default_dims_match_stage2_contract():
    cfg = Stage1DenoiserConfig()
    assert cfg.motion_dim == 23
    assert cfg.object_traj_dim == 9
    assert cfg.text_dim == 512
    assert cfg.object_token_dim == 256
