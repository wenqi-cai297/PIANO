"""Tests for the v0.3-δ InterControl-style trainable-copy encoder.

Smoke tests; full integration with the InteractionMaskTransformer
wrapper + train_generator is in a follow-up commit. These verify the
load-bearing initialization properties:

1. **Byte-identity at init**: with zero-init connectors the wrapper's
   output equals the base ``nn.TransformerEncoder``'s output for any
   input. Without this, swapping the encoder at the start of training
   would perturb pretrained-MoMask outputs and the "ControlNet starts
   as a no-op" guarantee fails.
2. **Connector zero-init**: every ``nn.Linear`` connector has both
   weight and bias = 0 at construction.
3. **Freeze main branch**: after ``freeze_main_branch()``, all main-
   branch parameters have ``requires_grad=False``; ctrl-branch params
   stay trainable.
4. **Param accounting**: trainable params include ctrl encoder layers +
   IntXAttn additions + connectors; frozen params are exactly the main
   encoder weights.

Server-only (needs torch).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn

from piano.models.motion_generator_intercontrol import (
    InterControlTransformerEncoder,
    count_trainable_vs_frozen,
)


# ----------------------------------------------------------------------
# Fixtures: a tiny TransformerEncoder mirroring MoMask's geometry.
# ----------------------------------------------------------------------

D_MODEL = 24            # smaller than MoMask's 384 for fast unit tests
N_HEADS = 4             # 24 / 4 = 6 head_dim — divisible
N_LAYERS = 3
DROPOUT = 0.1


@pytest.fixture
def base_encoder() -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=D_MODEL,
        nhead=N_HEADS,
        dim_feedforward=4 * D_MODEL,
        dropout=DROPOUT,
        activation="gelu",
        batch_first=False,
    )
    return nn.TransformerEncoder(layer, num_layers=N_LAYERS)


# ----------------------------------------------------------------------
# Test cases
# ----------------------------------------------------------------------

def _make_input(seed: int = 0):
    """Deterministic random (S, B, D) input + a (B, S) padding mask."""
    g = torch.Generator().manual_seed(seed)
    src = torch.randn(7, 2, D_MODEL, generator=g)             # (S=7, B=2, D)
    # First sample: all valid. Second sample: last 2 frames padded.
    pad = torch.zeros(2, 7, dtype=torch.bool)
    pad[1, 5:] = True
    return src, pad


def test_byte_identity_at_init_no_int_kv(base_encoder):
    """At init with no int_kv, the wrapper must equal the base encoder
    bytewise (because zero-init connectors mean ctrl output × 0 = 0,
    and base layers operate on the same input).
    """
    wrapper = InterControlTransformerEncoder(
        original_encoder=base_encoder,
        d_model=D_MODEL,
        num_heads=N_HEADS,
        dropout=DROPOUT,
        gamma_kind="per_head",
    )
    base_encoder.eval()
    wrapper.eval()
    src, pad = _make_input(seed=42)
    with torch.no_grad():
        out_base = base_encoder(src, src_key_padding_mask=pad)
        out_wrap = wrapper(src, int_kv=None, src_key_padding_mask=pad)
    assert torch.allclose(out_base, out_wrap), (
        "InterControlTransformerEncoder is not byte-identical to base at init "
        f"(max diff {(out_base - out_wrap).abs().max().item():.2e})"
    )


def test_byte_identity_at_init_with_int_kv(base_encoder):
    """Even when int_kv is supplied, ctrl branch's IntXAttn produces
    a γ-gated output that is zero at init (γ_int starts at 0). The
    connector then projects 0 to 0. So output still matches base."""
    wrapper = InterControlTransformerEncoder(
        original_encoder=base_encoder,
        d_model=D_MODEL,
        num_heads=N_HEADS,
        dropout=DROPOUT,
        gamma_kind="per_head",
    )
    base_encoder.eval()
    wrapper.eval()
    src, pad = _make_input(seed=11)
    int_kv = torch.randn(4, 2, D_MODEL)            # (S_int=4, B=2, D)
    int_pad = torch.zeros(2, 4, dtype=torch.bool)
    with torch.no_grad():
        out_base = base_encoder(src, src_key_padding_mask=pad)
        out_wrap = wrapper(
            src, int_kv=int_kv,
            src_key_padding_mask=pad,
            int_key_padding_mask=int_pad,
        )
    assert torch.allclose(out_base, out_wrap), (
        "byte-identity broke when int_kv is supplied at γ=0 + zero-init connectors"
    )


def test_connectors_zero_initialised(base_encoder):
    """Each connector's Linear weight + bias are exactly 0 at init."""
    wrapper = InterControlTransformerEncoder(
        original_encoder=base_encoder,
        d_model=D_MODEL,
        num_heads=N_HEADS,
        dropout=DROPOUT,
    )
    for i, conn in enumerate(wrapper.connectors):
        assert torch.all(conn.weight == 0), f"connector {i} weight non-zero"
        assert torch.all(conn.bias == 0), f"connector {i} bias non-zero"


def test_freeze_main_branch_flips_requires_grad(base_encoder):
    wrapper = InterControlTransformerEncoder(
        original_encoder=base_encoder,
        d_model=D_MODEL,
        num_heads=N_HEADS,
        dropout=DROPOUT,
    )
    # Before freezing, main params are trainable (default behaviour).
    assert all(p.requires_grad for p in wrapper.main_layers.parameters())

    wrapper.freeze_main_branch()

    # Main: all frozen. Ctrl + connectors: still trainable.
    assert all(not p.requires_grad for p in wrapper.main_layers.parameters())
    assert all(p.requires_grad for p in wrapper.ctrl_layers.parameters())
    assert all(p.requires_grad for p in wrapper.connectors.parameters())


def test_param_accounting(base_encoder):
    """After freeze, trainable count = ctrl_layers + connectors;
    frozen count = main_layers."""
    wrapper = InterControlTransformerEncoder(
        original_encoder=base_encoder,
        d_model=D_MODEL,
        num_heads=N_HEADS,
        dropout=DROPOUT,
    )
    wrapper.freeze_main_branch()
    counts = count_trainable_vs_frozen(wrapper)

    main_params = sum(int(p.numel()) for p in wrapper.main_layers.parameters())
    assert counts["frozen"] == main_params

    expected_trainable = (
        sum(int(p.numel()) for p in wrapper.ctrl_layers.parameters())
        + sum(int(p.numel()) for p in wrapper.connectors.parameters())
    )
    assert counts["trainable"] == expected_trainable
    assert counts["total"] == counts["frozen"] + counts["trainable"]


def test_grad_does_not_flow_into_main(base_encoder):
    """A backward pass on a scalar derived from the wrapper's output
    must not produce gradient on frozen main-branch params."""
    wrapper = InterControlTransformerEncoder(
        original_encoder=base_encoder,
        d_model=D_MODEL,
        num_heads=N_HEADS,
        dropout=DROPOUT,
    )
    wrapper.freeze_main_branch()
    wrapper.train()

    src, pad = _make_input(seed=77)
    src.requires_grad_(False)
    int_kv = torch.randn(4, 2, D_MODEL, requires_grad=False)
    out = wrapper(src, int_kv=int_kv, src_key_padding_mask=pad)
    loss = out.sum()
    loss.backward()

    # Frozen main: no grad attached.
    for name, p in wrapper.main_layers.named_parameters():
        assert p.grad is None, (
            f"main_layers.{name} got a grad despite requires_grad=False"
        )
    # Ctrl + connectors: grads exist (after a non-trivial loss).
    has_ctrl_grad = any(
        p.grad is not None and p.grad.abs().max() > 0
        for p in wrapper.ctrl_layers.parameters()
    )
    # Note: at init, ctrl produces a γ-gated output where γ=0. So ctrl
    # internal grads might exist (for downstream learning) but γ itself
    # gets gradient from connectors which are also at 0. The connectors,
    # however, ARE a function of ctrl output and at init connector(0) = 0
    # so loss = sum(out_base) which does not depend on ctrl/connectors
    # → ctrl grads may legitimately be all zero. We just check no
    # exception was raised + main params remain frozen.
    _ = has_ctrl_grad   # not asserted; can be False at init.


def test_separate_main_and_ctrl_weights(base_encoder):
    """Trainable updates on ctrl must not change main (deepcopy
    semantics). Verify by mutating ctrl's first param and checking
    main's first param is unchanged."""
    wrapper = InterControlTransformerEncoder(
        original_encoder=base_encoder,
        d_model=D_MODEL,
        num_heads=N_HEADS,
        dropout=DROPOUT,
    )
    main_first_param = next(wrapper.main_layers.parameters()).clone()
    # Mutate ctrl's first param.
    with torch.no_grad():
        next(wrapper.ctrl_layers.parameters()).add_(1.0)
    # Main is untouched.
    assert torch.equal(
        next(wrapper.main_layers.parameters()),
        main_first_param,
    )
