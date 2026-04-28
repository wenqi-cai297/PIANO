"""Tests for ResidualTransformerWithInteraction (Stage B C1).

The full ``forward`` / ``generate`` paths call into MoMask backbone
(needs CLIP, the vendored ``ResidualTransformer`` class which initializes
CLIP eagerly even in cond_mode='uncond' depending on path). For
CPU-only test runs we focus on:

- Constructor's in-place upgrade of seqTransEncoder.
- Byte-identity at γ_int_res=0 with int_kv=None (the un-wrapped
  forward path).
- Byte-identity at γ_int_res=0 with int_kv=tensor (γ-gate zeros the
  contribution regardless of int_kv content).
- Gradient flow into γ_int_res when γ != 0.
- Parameter accounting (no double-counting; new params added are
  norm_int + int_attn + gamma_int per block).

We use a minimal fake "ResidualTransformer-like" object — just enough
to satisfy the wrapper constructor. The encoder swap is the only
thing we need to test; full integration (with CLIP encoded text +
input_process + position_enc + output_process) is exercised on the
server via train_generator.py + qual_eval.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn


D_MODEL = 384
NUM_HEADS = 6
NUM_LAYERS = 8
DROPOUT = 0.1


def _make_fake_residual_transformer() -> nn.Module:
    """Construct a minimal stand-in for ``ResidualTransformer``.

    Holds just the attributes the wrapper constructor inspects:
    ``latent_dim`` and ``seqTransEncoder`` (an ``nn.TransformerEncoder``).
    """

    class FakeResidual(nn.Module):
        def __init__(self):
            super().__init__()
            self.latent_dim = D_MODEL
            layer = nn.TransformerEncoderLayer(
                d_model=D_MODEL,
                nhead=NUM_HEADS,
                dim_feedforward=4 * D_MODEL,
                dropout=DROPOUT,
                activation="gelu",
            )
            self.seqTransEncoder = nn.TransformerEncoder(layer, num_layers=NUM_LAYERS)
            self.forward_seen = None
            self.generate_seen = None

        def forward(self, *args, **kwargs):
            self.forward_seen = (args, kwargs)
            return "forward-result"

        def generate(self, *args, **kwargs):
            self.generate_seen = (args, kwargs)
            return "generate-result"

    return FakeResidual()


def test_constructor_swaps_seqTransEncoder_in_place():
    """After wrapping, residual.seqTransEncoder is the
    MaskTransformerEncoderWithInteraction, not the original.
    """
    from piano.models.motion_generator import MaskTransformerEncoderWithInteraction
    from piano.models.motion_generator_residual import (
        ResidualTransformerWithInteraction,
    )

    residual = _make_fake_residual_transformer()
    original_encoder_id = id(residual.seqTransEncoder)

    wrapper = ResidualTransformerWithInteraction(
        residual,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
    )

    # In-place rebind happened.
    assert isinstance(residual.seqTransEncoder, MaskTransformerEncoderWithInteraction)
    assert id(residual.seqTransEncoder) != original_encoder_id
    assert wrapper.encoder is residual.seqTransEncoder
    assert wrapper.residual is residual


def test_constructor_validates_d_model_mismatch():
    from piano.models.motion_generator_residual import (
        ResidualTransformerWithInteraction,
    )

    residual = _make_fake_residual_transformer()
    try:
        ResidualTransformerWithInteraction(
            residual, d_model=D_MODEL + 1, num_heads=NUM_HEADS,
        )
    except ValueError as e:
        assert "d_model" in str(e)
        return
    raise AssertionError("Expected ValueError for d_model mismatch")


def test_constructor_validates_num_heads_mismatch():
    from piano.models.motion_generator_residual import (
        ResidualTransformerWithInteraction,
    )

    residual = _make_fake_residual_transformer()
    try:
        ResidualTransformerWithInteraction(
            residual, d_model=D_MODEL, num_heads=NUM_HEADS + 1,
        )
    except ValueError as e:
        assert "num_heads" in str(e)
        return
    raise AssertionError("Expected ValueError for num_heads mismatch")


def test_byte_identity_at_gamma_zero_int_kv_none():
    """With γ_int_res=0 (default zero-init) and int_kv=None, the wrapped
    encoder produces byte-identical output to the original encoder.

    This is the load-bearing invariant: training step 0 of a wrapped
    residual must match the un-wrapped residual exactly. Otherwise we'd
    be starting from a different distribution than v0.6's pretrained
    residual, defeating the warm-start.
    """
    from piano.models.motion_generator_residual import (
        ResidualTransformerWithInteraction,
    )

    torch.manual_seed(0)
    residual = _make_fake_residual_transformer()
    residual.eval()

    # Save the ORIGINAL encoder's reference + a fixed input. We need a
    # snapshot of the un-wrapped output before the wrapper rebinds.
    original_encoder = residual.seqTransEncoder
    S, B = 12, 2
    src = torch.randn(S, B, D_MODEL)

    # Forward through original (un-wrapped).
    with torch.no_grad():
        out_orig = original_encoder(src)

    # Wrap in-place. γ_int_res defaults to zero-init.
    ResidualTransformerWithInteraction(
        residual,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        zero_init_gamma=True,
    )

    # Forward through wrapped, no int_kv (mimics the original code path
    # `r.seqTransEncoder(xseq, src_key_padding_mask=mask)`).
    with torch.no_grad():
        out_wrapped = residual.seqTransEncoder(src)

    # Byte-identity (the wrapped block's IntXAttn branch is skipped
    # entirely when int_kv=None).
    torch.testing.assert_close(out_wrapped, out_orig, atol=0, rtol=0)


def test_byte_identity_at_gamma_zero_int_kv_provided():
    """With γ_int_res=0 and int_kv=tensor, output is still byte-identical
    to the un-wrapped original.

    This is the more demanding invariant: the IntXAttn sublayer DOES run
    (int_kv is not None) but its contribution is multiplied by γ=0
    before adding to the residual stream → exact zero contribution →
    output bytes match original.
    """
    from piano.models.motion_generator_residual import (
        ResidualTransformerWithInteraction,
    )

    torch.manual_seed(0)
    residual = _make_fake_residual_transformer()
    residual.eval()

    original_encoder = residual.seqTransEncoder
    S, B = 12, 2
    S_int = 49
    src = torch.randn(S, B, D_MODEL)
    int_kv = torch.randn(S_int, B, D_MODEL)

    with torch.no_grad():
        out_orig = original_encoder(src)

    ResidualTransformerWithInteraction(
        residual, d_model=D_MODEL, num_heads=NUM_HEADS, dropout=DROPOUT,
        zero_init_gamma=True,
    )

    with torch.no_grad():
        out_wrapped = residual.seqTransEncoder(src, int_kv=int_kv)

    torch.testing.assert_close(out_wrapped, out_orig, atol=0, rtol=0)


def test_gamma_nonzero_init_breaks_byte_identity():
    """Sanity check the inverse: with zero_init_gamma=False (γ=1), the
    wrapped output differs from the original. Confirms the γ-gate is
    the operative knob for the byte-identity guarantee.
    """
    from piano.models.motion_generator_residual import (
        ResidualTransformerWithInteraction,
    )

    torch.manual_seed(0)
    residual = _make_fake_residual_transformer()
    residual.eval()
    original_encoder = residual.seqTransEncoder
    S, B = 12, 2
    S_int = 49
    src = torch.randn(S, B, D_MODEL)
    int_kv = torch.randn(S_int, B, D_MODEL)

    with torch.no_grad():
        out_orig = original_encoder(src)

    ResidualTransformerWithInteraction(
        residual, d_model=D_MODEL, num_heads=NUM_HEADS, dropout=DROPOUT,
        zero_init_gamma=False,                                # γ_init = 1
    )

    with torch.no_grad():
        out_wrapped = residual.seqTransEncoder(src, int_kv=int_kv)

    diff = (out_wrapped - out_orig).abs().max().item()
    assert diff > 1e-3, (
        f"With zero_init_gamma=False, expected byte-identity to BREAK; "
        f"got max abs diff {diff:.6e}"
    )


def test_gradient_flows_into_gamma_int_res():
    """When γ_int_res != 0 and int_kv is provided, optimization should
    push gradient into γ_int_res. Otherwise the new sublayer cannot
    learn.
    """
    from piano.models.motion_generator_residual import (
        ResidualTransformerWithInteraction,
    )

    torch.manual_seed(0)
    residual = _make_fake_residual_transformer()
    residual.train()  # gradient mode

    ResidualTransformerWithInteraction(
        residual, d_model=D_MODEL, num_heads=NUM_HEADS, dropout=DROPOUT,
        zero_init_gamma=False,  # γ=1 so the IntXAttn sublayer is active
    )

    S, B = 8, 2
    S_int = 12
    src = torch.randn(S, B, D_MODEL)
    int_kv = torch.randn(S_int, B, D_MODEL)

    out = residual.seqTransEncoder(src, int_kv=int_kv)
    loss = out.pow(2).mean()
    loss.backward()

    # Every block's gamma_int should have a gradient.
    grad_count = 0
    for blk in residual.seqTransEncoder.layers:
        assert blk.gamma_int.grad is not None, (
            f"gamma_int has no grad on block {grad_count}"
        )
        assert torch.isfinite(blk.gamma_int.grad).all()
        # Non-trivial gradient (we provided non-zero int_kv).
        assert blk.gamma_int.grad.abs().max() > 0
        grad_count += 1
    assert grad_count == NUM_LAYERS


def test_gamma_grad_zero_when_int_kv_none():
    """When int_kv=None, the IntXAttn sublayer is skipped → γ_int_res
    receives no gradient (it's not in the computational graph for that
    forward).
    """
    from piano.models.motion_generator_residual import (
        ResidualTransformerWithInteraction,
    )

    torch.manual_seed(0)
    residual = _make_fake_residual_transformer()
    residual.train()

    ResidualTransformerWithInteraction(
        residual, d_model=D_MODEL, num_heads=NUM_HEADS, dropout=DROPOUT,
        zero_init_gamma=False,
    )

    S, B = 8, 2
    src = torch.randn(S, B, D_MODEL)
    out = residual.seqTransEncoder(src, int_kv=None)
    loss = out.pow(2).mean()
    loss.backward()

    for blk in residual.seqTransEncoder.layers:
        # gamma_int wasn't part of the graph (skip branch on int_kv=None).
        assert blk.gamma_int.grad is None or blk.gamma_int.grad.abs().max() == 0


def test_param_accounting_no_double_count():
    """Wrapping should not double-count the original transformer's
    parameters. The wrapped encoder holds refs to original layers
    (no deepcopy), so iterating ``residual.parameters()`` should
    visit each weight exactly once.
    """
    from piano.models.motion_generator_residual import (
        ResidualTransformerWithInteraction,
    )

    residual = _make_fake_residual_transformer()
    original_param_ids = {id(p) for p in residual.parameters()}
    original_param_count = sum(p.numel() for p in residual.parameters())

    ResidualTransformerWithInteraction(
        residual, d_model=D_MODEL, num_heads=NUM_HEADS, dropout=DROPOUT,
    )

    # After wrapping, every original parameter ID should still appear
    # exactly once via residual.parameters() (which now traverses the
    # wrapped encoder).
    seen_ids = []
    for p in residual.parameters():
        seen_ids.append(id(p))
    seen_ids_set = set(seen_ids)
    assert len(seen_ids) == len(seen_ids_set), (
        "Some params appear multiple times in residual.parameters() — "
        "wrapper double-counts."
    )

    # Every original param is still reachable.
    assert original_param_ids.issubset(seen_ids_set)

    # New params (norm_int + int_attn + gamma_int per block) are added.
    new_param_count = sum(p.numel() for p in residual.parameters())
    assert new_param_count > original_param_count, (
        f"Wrapping should add new params; got original={original_param_count} "
        f"new={new_param_count}"
    )

    # Per block, expected new params:
    #   norm_int weight + bias = 2 * D_MODEL
    #   int_attn (in_proj_weight 3*D_MODEL*D_MODEL + in_proj_bias 3*D_MODEL +
    #             out_proj.weight D_MODEL*D_MODEL + out_proj.bias D_MODEL)
    #   gamma_int = NUM_HEADS (per-head default)
    new_per_block = 2 * D_MODEL + (3 * D_MODEL * D_MODEL + 3 * D_MODEL
                                   + D_MODEL * D_MODEL + D_MODEL) + NUM_HEADS
    expected_new = new_per_block * NUM_LAYERS
    assert new_param_count - original_param_count == expected_new, (
        f"Expected {expected_new} new params, got "
        f"{new_param_count - original_param_count}"
    )


def test_residual_layer_metrics_group_by_active_q_layer():
    from piano.models.motion_generator_residual import _residual_layer_metrics

    # logits shape follows MoMask CE convention: (B, vocab, S).
    logits = torch.tensor([
        [[5.0, 0.0], [0.0, 5.0], [0.0, 0.0], [0.0, 0.0]],
        [[0.0, 5.0], [5.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        [[0.0, 5.0], [5.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
    ])
    labels = torch.tensor([
        [0, 1],
        [0, 1],
        [1, -1],
    ])
    active_q_layers = torch.tensor([1, 2, 2])

    metrics = _residual_layer_metrics(
        logits,
        labels,
        active_q_layers,
        pad_id=-1,
        num_quant_layers=4,
    )

    assert set(metrics) == {
        "loss_residual_q1",
        "acc_residual_q1",
        "tokens_residual_q1",
        "loss_residual_q2",
        "acc_residual_q2",
        "tokens_residual_q2",
    }
    assert metrics["tokens_residual_q1"].item() == 2
    assert metrics["tokens_residual_q2"].item() == 3
    assert metrics["acc_residual_q1"].item() == 1.0
    torch.testing.assert_close(
        metrics["acc_residual_q2"],
        torch.tensor(1.0 / 3.0),
    )


def test_gamma_kind_scalar_vs_per_head_dofs():
    """gamma_kind controls the dof count: scalar=1 per layer, per_head=NUM_HEADS."""
    from piano.models.motion_generator_residual import (
        ResidualTransformerWithInteraction,
    )

    for gamma_kind, expected_dof in [("scalar", 1), ("per_head", NUM_HEADS)]:
        residual = _make_fake_residual_transformer()
        ResidualTransformerWithInteraction(
            residual, d_model=D_MODEL, num_heads=NUM_HEADS, dropout=DROPOUT,
            gamma_kind=gamma_kind,
        )
        for blk in residual.seqTransEncoder.layers:
            assert blk.gamma_int.numel() == expected_dof, (
                f"gamma_kind={gamma_kind}: expected {expected_dof} dof, "
                f"got {blk.gamma_int.numel()}"
            )


def test_new_and_backbone_parameters_partition_correctly():
    """``new_parameters()`` returns only the C1 IntXAttn additions
    (norm_int / int_attn / gamma_int per block); ``backbone_parameters()``
    returns the rest. The two sets must partition (no overlap, no gap)
    the wrapper's parameters_wo_clip set.
    """
    from piano.models.motion_generator_residual import (
        ResidualTransformerWithInteraction,
    )

    residual = _make_fake_residual_transformer()
    wrapper = ResidualTransformerWithInteraction(
        residual, d_model=D_MODEL, num_heads=NUM_HEADS, dropout=DROPOUT,
    )

    new_ids = {id(p) for p in wrapper.new_parameters()}
    bb_ids = {id(p) for p in wrapper.backbone_parameters()}
    all_ids = {id(p) for p in wrapper.parameters_wo_clip()}

    # Disjoint:
    assert new_ids.isdisjoint(bb_ids), (
        "new and backbone overlap — same param appears in both groups"
    )
    # Cover everything (parameters_wo_clip == new ∪ backbone):
    assert new_ids | bb_ids == all_ids, (
        f"partition gap: new+backbone={len(new_ids | bb_ids)}, "
        f"all_wo_clip={len(all_ids)}"
    )

    # Sanity: new should be exactly NUM_LAYERS × (norm_int weight +
    # norm_int bias + int_attn in_proj_weight + in_proj_bias +
    # out_proj.weight + out_proj.bias + gamma_int) = 7 params per block.
    assert len(new_ids) == NUM_LAYERS * 7, (
        f"expected {NUM_LAYERS * 7} new params, got {len(new_ids)}"
    )


def test_new_parameters_excludes_original_layer_weights():
    """Sanity check: original transformer layer's self_attn / FFN / norm1
    / norm2 weights must NOT appear in new_parameters() — they're
    backbone (held by reference inside MaskTransformerBlockWithInteraction.layer).
    """
    from piano.models.motion_generator_residual import (
        ResidualTransformerWithInteraction,
    )

    residual = _make_fake_residual_transformer()
    # Snapshot original layer param ids before wrapping.
    original_layer_param_ids = set()
    for blk_layer in residual.seqTransEncoder.layers:
        for p in blk_layer.parameters():
            original_layer_param_ids.add(id(p))

    wrapper = ResidualTransformerWithInteraction(
        residual, d_model=D_MODEL, num_heads=NUM_HEADS, dropout=DROPOUT,
    )

    new_ids = {id(p) for p in wrapper.new_parameters()}
    leak = original_layer_param_ids & new_ids
    assert not leak, (
        f"new_parameters() leaked {len(leak)} original layer param ids — "
        f"the .layer.* path should be classified as backbone"
    )


def test_parameters_wo_clip_excludes_clip_module():
    """``parameters_wo_clip`` should exclude ``clip_model.*`` params,
    matching the un-wrapped ResidualTransformer's helper.
    """
    from piano.models.motion_generator_residual import (
        ResidualTransformerWithInteraction,
    )

    residual = _make_fake_residual_transformer()
    # Add a fake clip_model so we can verify exclusion.
    residual.clip_model = nn.Linear(8, 8)

    wrapper = ResidualTransformerWithInteraction(
        residual, d_model=D_MODEL, num_heads=NUM_HEADS, dropout=DROPOUT,
    )
    wo_clip_ids = {id(p) for p in wrapper.parameters_wo_clip()}
    clip_param_ids = {id(p) for p in residual.clip_model.parameters()}
    assert wo_clip_ids.isdisjoint(clip_param_ids), (
        "parameters_wo_clip leaked clip_model.* params"
    )


def test_drop_in_forward_and_generate_passthroughs():
    """Wrapper remains usable where callers expect raw ResidualTransformer."""
    from piano.models.motion_generator_residual import (
        ResidualTransformerWithInteraction,
    )

    residual = _make_fake_residual_transformer()
    wrapper = ResidualTransformerWithInteraction(
        residual, d_model=D_MODEL, num_heads=NUM_HEADS, dropout=DROPOUT,
    )

    assert wrapper("ids", y="text", m_lens="lens") == "forward-result"
    assert residual.forward_seen == (("ids",), {"y": "text", "m_lens": "lens"})

    assert wrapper.generate("base", conds=["text"]) == "generate-result"
    assert residual.generate_seen == (("base",), {"conds": ["text"]})
