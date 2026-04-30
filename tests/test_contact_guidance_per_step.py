"""CPU-friendly tests for the v17 per-step guidance building blocks.

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
