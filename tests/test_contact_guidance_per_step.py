"""CPU-friendly tests for the v17 per-step guidance building blocks
and the v17-G γ_int inference-time boost helper.

The full ``_generate_with_per_step_guidance`` integration test requires a
``InteractionMaskTransformer`` plus MoMask weights and is server-only.
Here we test the two new composable helpers added on 2026-05-01:

- ``_precompute_residual_emb_sum``: lookup of residual codebook
  embeddings for layers 1..Q-1 + sum.
- ``_decode_with_relaxed_masked_base``: differentiable decode with a
  mix of soft (masked) and hard (committed) base embeddings, plus
  frozen residual context.

See ``analyses/2026-05-01_per_step_guidance_design.md`` for design.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _make_fake_vq(B: int, S: int, Q: int, V: int, code_dim: int):
    """Build a minimal RVQ-VAE-like module that satisfies the helpers' contract.

    Mirrors the fake module pattern used in
    ``test_contact_guidance.py::test_decode_relaxed_base_shape_and_gradient``.
    """

    class FakeQuantizer(nn.Module):
        def __init__(self):
            super().__init__()
            self._codebooks = nn.Parameter(torch.randn(Q, V, code_dim) * 0.1)
            self.num_quantizers = Q

        @property
        def codebooks(self):
            return self._codebooks

        def get_codes_from_indices(self, indices):
            B_, S_, Q_ = indices.shape
            assert Q_ == Q
            out = torch.zeros(Q_, B_, S_, code_dim, device=indices.device)
            for q in range(Q_):
                out[q] = self._codebooks[q][indices[..., q]]
            return out

    class FakeDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.ConvTranspose1d(code_dim, 263, kernel_size=4, stride=4)

        def forward(self, x):
            return self.conv(x).permute(0, 2, 1)

    class FakeRVQVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.quantizer = FakeQuantizer()
            self.decoder = FakeDecoder()

    return FakeRVQVAE()


def test_precompute_residual_emb_sum_matches_manual():
    from piano.inference.contact_guidance import _precompute_residual_emb_sum

    B, S, Q, V, code_dim = 1, 5, 4, 7, 16
    vq = _make_fake_vq(B, S, Q, V, code_dim)
    all_ids = torch.randint(0, V, (B, S, Q))

    out = _precompute_residual_emb_sum(all_ids, vq)
    assert out.shape == (B, S, code_dim)

    # Manual: sum codebooks[q][all_ids[..., q]] for q=1..Q-1.
    manual = torch.zeros(B, S, code_dim)
    for q in range(1, Q):
        manual = manual + vq.quantizer.codebooks[q][all_ids[..., q]]
    torch.testing.assert_close(out, manual, atol=1e-6, rtol=1e-6)


def test_decode_with_relaxed_masked_base_shape_and_gradient_at_masked_only():
    """Gradient flows into base_logits only at differentiable_mask=True positions.

    At committed positions the relaxed-decode helper uses the hard
    codebook entry (no path back to logits), so logits.grad must be 0
    there. At masked positions softmax(logits/T) @ codebook[0] is on
    the autograd path and grad must be non-zero.
    """
    from piano.inference.contact_guidance import (
        _decode_with_relaxed_masked_base,
    )

    B, S, Q, V, code_dim = 1, 6, 4, 7, 16
    T_decoded = S * 4
    vq = _make_fake_vq(B, S, Q, V, code_dim)

    base_logits = torch.randn(B, S, V, requires_grad=True)
    committed_ids = torch.randint(0, V, (B, S))
    # Mask the first 3 positions, commit the last 3.
    differentiable_mask = torch.tensor(
        [[True, True, True, False, False, False]],
    )
    baseline_residual_emb_sum = torch.randn(B, S, code_dim)

    motion_norm = _decode_with_relaxed_masked_base(
        base_logits=base_logits,
        committed_ids=committed_ids,
        differentiable_mask=differentiable_mask,
        baseline_residual_emb_sum=baseline_residual_emb_sum,
        vq_model=vq,
        temperature=1.0,
    )

    assert motion_norm.shape == (B, T_decoded, 263)

    motion_norm.sum().backward()
    assert base_logits.grad is not None
    assert torch.isfinite(base_logits.grad).all()

    # Per the strict-MaskGIT semantics: gradient at masked positions is
    # finite and nonzero in expectation; at committed positions it must
    # be exactly zero (hard embedding doesn't depend on logits).
    grad = base_logits.grad
    assert grad[0, :3].abs().sum().item() > 0.0
    assert grad[0, 3:].abs().sum().item() == 0.0


def test_decode_with_relaxed_masked_base_all_masked_matches_pure_relaxed():
    """When every position is masked, the helper should agree with a pure
    relaxed-base decode of softmax(logits) @ codebook[0] + residual_emb_sum.
    """
    from piano.inference.contact_guidance import (
        _decode_with_relaxed_masked_base,
    )

    B, S, Q, V, code_dim = 1, 4, 3, 8, 16
    vq = _make_fake_vq(B, S, Q, V, code_dim)

    logits = torch.randn(B, S, V)
    committed_ids = torch.zeros(B, S, dtype=torch.long)
    differentiable_mask = torch.ones(B, S, dtype=torch.bool)
    res_sum = torch.randn(B, S, code_dim)

    out = _decode_with_relaxed_masked_base(
        base_logits=logits,
        committed_ids=committed_ids,
        differentiable_mask=differentiable_mask,
        baseline_residual_emb_sum=res_sum,
        vq_model=vq,
        temperature=1.0,
    )

    soft = torch.softmax(logits, dim=-1) @ vq.quantizer.codebooks[0]
    expected = vq.decoder((soft + res_sum).permute(0, 2, 1))
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)


def test_decode_with_relaxed_masked_base_gumbel_noise_zero_matches_no_noise():
    """gumbel_noise_scale=0 must be exactly equivalent to the
    pre-v17-F path (pure softmax expectation).
    """
    from piano.inference.contact_guidance import (
        _decode_with_relaxed_masked_base,
    )

    B, S, Q, V, code_dim = 1, 4, 3, 8, 16
    vq = _make_fake_vq(B, S, Q, V, code_dim)

    logits = torch.randn(B, S, V)
    committed_ids = torch.zeros(B, S, dtype=torch.long)
    differentiable_mask = torch.ones(B, S, dtype=torch.bool)
    res_sum = torch.randn(B, S, code_dim)

    no_noise = _decode_with_relaxed_masked_base(
        base_logits=logits,
        committed_ids=committed_ids,
        differentiable_mask=differentiable_mask,
        baseline_residual_emb_sum=res_sum,
        vq_model=vq,
        temperature=1.0,
        gumbel_noise_scale=0.0,
    )
    explicit_zero = _decode_with_relaxed_masked_base(
        base_logits=logits,
        committed_ids=committed_ids,
        differentiable_mask=differentiable_mask,
        baseline_residual_emb_sum=res_sum,
        vq_model=vq,
        temperature=1.0,
    )
    torch.testing.assert_close(no_noise, explicit_zero, atol=0.0, rtol=0.0)


def test_decode_with_relaxed_masked_base_gumbel_noise_changes_output():
    """gumbel_noise_scale=1 must produce a different (stochastic) output
    than gumbel_noise_scale=0 with the same logits, confirming the
    noise injection is wired up.
    """
    from piano.inference.contact_guidance import (
        _decode_with_relaxed_masked_base,
    )

    B, S, Q, V, code_dim = 1, 6, 3, 16, 16
    vq = _make_fake_vq(B, S, Q, V, code_dim)

    logits = torch.randn(B, S, V) * 2.0     # moderately peaked
    committed_ids = torch.zeros(B, S, dtype=torch.long)
    differentiable_mask = torch.ones(B, S, dtype=torch.bool)
    res_sum = torch.zeros(B, S, code_dim)

    torch.manual_seed(0)
    no_noise = _decode_with_relaxed_masked_base(
        base_logits=logits, committed_ids=committed_ids,
        differentiable_mask=differentiable_mask,
        baseline_residual_emb_sum=res_sum, vq_model=vq,
        temperature=1.0, gumbel_noise_scale=0.0,
    )
    torch.manual_seed(0)
    with_noise = _decode_with_relaxed_masked_base(
        base_logits=logits, committed_ids=committed_ids,
        differentiable_mask=differentiable_mask,
        baseline_residual_emb_sum=res_sum, vq_model=vq,
        temperature=1.0, gumbel_noise_scale=1.0,
    )
    diff = (with_noise - no_noise).abs().max()
    assert diff > 1e-3, f"Gumbel noise injection had no observable effect (max diff={float(diff)})"
    # Also: gradient still flows through logits when Gumbel noise is on.
    logits_param = logits.clone().requires_grad_(True)
    out = _decode_with_relaxed_masked_base(
        base_logits=logits_param, committed_ids=committed_ids,
        differentiable_mask=differentiable_mask,
        baseline_residual_emb_sum=res_sum, vq_model=vq,
        temperature=1.0, gumbel_noise_scale=1.0,
    )
    out.sum().backward()
    assert logits_param.grad is not None
    assert logits_param.grad.abs().sum().item() > 0.0


# ============================================================================
# v17-G γ_int boost context manager
# ============================================================================

def _make_fake_model_with_gamma_int():
    """Build a tiny module with `gamma_int` and `gamma_int_other` parameters
    at multiple nesting levels, mimicking the encoder-layers layout where
    PIANO's IntXAttn lives.
    """
    class Block(nn.Module):
        def __init__(self, init_value: float):
            super().__init__()
            self.gamma_int = nn.Parameter(torch.full((1,), init_value))
            self.other_param = nn.Parameter(torch.zeros(3))

    class Encoder(nn.Module):
        def __init__(self, n_layers: int, init_value: float):
            super().__init__()
            self.layers = nn.ModuleList([Block(init_value) for _ in range(n_layers)])

    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.seqTransEncoder = Encoder(n_layers=4, init_value=0.02)

    return Wrapper()


def test_scaled_gamma_int_multiplies_and_restores():
    from piano.inference.contact_guidance import _scaled_gamma_int

    model = _make_fake_model_with_gamma_int()
    originals = [
        blk.gamma_int.data.clone() for blk in model.seqTransEncoder.layers
    ]

    with _scaled_gamma_int([model], boost=10.0):
        for blk, original in zip(model.seqTransEncoder.layers, originals):
            assert torch.allclose(blk.gamma_int.data, original * 10.0)

    # After exit, restored exactly.
    for blk, original in zip(model.seqTransEncoder.layers, originals):
        assert torch.allclose(blk.gamma_int.data, original, atol=0.0, rtol=0.0)


def test_scaled_gamma_int_skips_non_gamma_params():
    """`other_param` (not ending in gamma_int) must NOT be scaled."""
    from piano.inference.contact_guidance import _scaled_gamma_int

    model = _make_fake_model_with_gamma_int()
    other_originals = [
        blk.other_param.data.clone() for blk in model.seqTransEncoder.layers
    ]

    with _scaled_gamma_int([model], boost=5.0):
        for blk, original in zip(model.seqTransEncoder.layers, other_originals):
            assert torch.allclose(blk.other_param.data, original, atol=0.0, rtol=0.0)


def test_scaled_gamma_int_handles_none_modules():
    """When residual_transformer is None, _scaled_gamma_int must skip it
    silently (matches the optional-residual code path in qual_eval)."""
    from piano.inference.contact_guidance import _scaled_gamma_int

    model = _make_fake_model_with_gamma_int()
    originals = [
        blk.gamma_int.data.clone() for blk in model.seqTransEncoder.layers
    ]

    with _scaled_gamma_int([model, None], boost=3.0):
        for blk, original in zip(model.seqTransEncoder.layers, originals):
            assert torch.allclose(blk.gamma_int.data, original * 3.0)

    for blk, original in zip(model.seqTransEncoder.layers, originals):
        assert torch.allclose(blk.gamma_int.data, original, atol=0.0, rtol=0.0)


def test_scaled_gamma_int_boost_one_is_no_op_fast_path():
    """boost == 1.0 must not touch any parameters (saves a clone per gamma)."""
    from piano.inference.contact_guidance import _scaled_gamma_int

    model = _make_fake_model_with_gamma_int()
    pre = [blk.gamma_int.data.clone() for blk in model.seqTransEncoder.layers]

    with _scaled_gamma_int([model], boost=1.0):
        for blk, original in zip(model.seqTransEncoder.layers, pre):
            assert torch.allclose(blk.gamma_int.data, original, atol=0.0, rtol=0.0)


def test_scaled_gamma_int_restores_after_exception():
    """If the body raises, the context manager must still restore γ_int."""
    from piano.inference.contact_guidance import _scaled_gamma_int

    model = _make_fake_model_with_gamma_int()
    originals = [blk.gamma_int.data.clone() for blk in model.seqTransEncoder.layers]

    class _SentinelError(Exception):
        pass

    try:
        with _scaled_gamma_int([model], boost=7.0):
            raise _SentinelError("body raised")
    except _SentinelError:
        pass

    for blk, original in zip(model.seqTransEncoder.layers, originals):
        assert torch.allclose(blk.gamma_int.data, original, atol=0.0, rtol=0.0)
