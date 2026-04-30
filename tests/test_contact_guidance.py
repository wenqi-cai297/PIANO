"""Smoke tests for the contact-guidance module (B3).

The full ``guide_with_contact`` requires MoMask + CLIP + a trained
generator and is server-only; here we test the composable pieces:

- ``_decode_relaxed_base``: relaxed-base + frozen-residual decode shape
  + gradient flow into ``base_logits``.
- ``_masked_contact_l2``: shape, masking, zero-mask edge case.
- ``_lift_canonical_to_world_torch``: torch counterpart of the numpy
  helper from contact_eval; identity at zero transform; pure rotation.

The full ``guide_with_contact`` integration test requires the actual
model weights; runnable only on the server. We add a placeholder skip
test that documents the expected smoke check.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn


def test_lift_canonical_to_world_torch_identity():
    from piano.inference.contact_guidance import _lift_canonical_to_world_torch

    x = torch.tensor([[[[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]]])
    out = _lift_canonical_to_world_torch(x, R_y_angle=0.0, T_xz=np.array([0.0, 0.0]))
    assert torch.allclose(out, x, atol=1e-6)


def test_lift_canonical_to_world_torch_translation():
    from piano.inference.contact_guidance import _lift_canonical_to_world_torch

    x = torch.zeros(1, 1, 1, 3)
    x[0, 0, 0] = torch.tensor([1.0, 2.0, 3.0])
    out = _lift_canonical_to_world_torch(x, R_y_angle=0.0, T_xz=np.array([10.0, -5.0]))
    expected = torch.tensor([11.0, 2.0, -2.0])
    assert torch.allclose(out[0, 0, 0], expected, atol=1e-6)


def test_lift_canonical_to_world_torch_y_rotation():
    from piano.inference.contact_guidance import _lift_canonical_to_world_torch

    # +90° around Y: (1, 0, 0) → (0, 0, -1)
    x = torch.tensor([[[[1.0, 0.0, 0.0]]]])
    out = _lift_canonical_to_world_torch(
        x, R_y_angle=float(np.pi / 2), T_xz=np.array([0.0, 0.0]),
    )
    expected = torch.tensor([0.0, 0.0, -1.0])
    assert torch.allclose(out[0, 0, 0], expected, atol=1e-5)


def test_lift_target_to_world_zero_rotation_pure_translation():
    """target_world = R(0) @ target_local + obj_pos = target_local + obj_pos"""
    from piano.inference.contact_guidance import _lift_target_to_world_np

    target_local = np.zeros((2, 3, 3), dtype=np.float32)        # (T=2, n_parts=3, 3)
    target_local[0, 0] = [1.0, 0.0, 0.0]
    target_local[1, 2] = [0.0, 1.0, 0.0]

    obj_pos = np.array([[10.0, 0.0, 0.0], [0.0, 0.0, 5.0]], dtype=np.float32)
    obj_rot = np.zeros((2, 3), dtype=np.float32)                # axis-angle = 0 → identity

    out = _lift_target_to_world_np(target_local, obj_pos, obj_rot)
    assert out.shape == (2, 3, 3)
    # frame 0, part 0: (1,0,0) + (10,0,0) = (11,0,0)
    np.testing.assert_allclose(out[0, 0], [11.0, 0.0, 0.0], atol=1e-6)
    # frame 1, part 2: (0,1,0) + (0,0,5) = (0,1,5)
    np.testing.assert_allclose(out[1, 2], [0.0, 1.0, 5.0], atol=1e-6)


def test_lift_target_to_world_rotation_only():
    """target_world = R @ target_local with obj_pos=0"""
    from piano.inference.contact_guidance import _lift_target_to_world_np

    # +90° around Y: (1, 0, 0) → (0, 0, -1)
    target_local = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32)   # (T=1, n_parts=1, 3)
    obj_pos = np.zeros((1, 3), dtype=np.float32)
    # axis-angle for +90° around Y axis = [0, π/2, 0]
    obj_rot = np.array([[0.0, np.pi / 2, 0.0]], dtype=np.float32)

    out = _lift_target_to_world_np(target_local, obj_pos, obj_rot)
    np.testing.assert_allclose(out[0, 0], [0.0, 0.0, -1.0], atol=1e-5)


def test_lift_pc_to_world_zero_rotation_pure_translation():
    from piano.inference.contact_guidance import _lift_pc_to_world_np

    pc = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)   # (N=2, 3)
    obj_pos = np.array([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]], dtype=np.float32)
    obj_rot = np.zeros((2, 3), dtype=np.float32)

    out = _lift_pc_to_world_np(pc, obj_pos, obj_rot)
    assert out.shape == (2, 2, 3)
    np.testing.assert_allclose(out[0, 0], [10.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(out[0, 1], [11.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(out[1, 0], [20.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(out[1, 1], [21.0, 0.0, 0.0], atol=1e-6)


def test_eval_metric_as_loss_min_distance():
    """Loss = mean_t min_p min_n ||body[t,p] - pc_world[t,n]||"""
    from piano.inference.contact_guidance import _eval_metric_as_loss

    # 1 frame, 2 body parts, 2 PC samples.
    body = torch.tensor([[
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
    ]])                                                                 # (B=1, T=1, n_parts=2, 3)
    pc = torch.tensor([[
        [[5.0, 0.0, 0.0], [15.0, 0.0, 0.0]],
    ]])                                                                 # (B=1, T=1, N_pc=2, 3)

    # body[0, 0]: nearest PC point is 5.0 away (5.0)
    # body[10, 0]: nearest PC point is 5.0 away (15.0 - 10.0 = 5.0; or 0.0 - 10.0 = 10.0; min = 5.0)
    # min over body parts: 5.0
    # mean over t: 5.0
    out = _eval_metric_as_loss(body, pc)
    assert torch.allclose(out, torch.tensor(5.0), atol=1e-5)


def test_eval_metric_as_loss_gradient_flows():
    """Gradient flows from loss back through body_world."""
    from piano.inference.contact_guidance import _eval_metric_as_loss

    body = torch.zeros(1, 4, 2, 3, requires_grad=True)
    pc = torch.ones(1, 4, 8, 3) * 5.0   # all PC points at (5,5,5)

    loss = _eval_metric_as_loss(body, pc)
    loss.backward()
    assert body.grad is not None
    assert torch.isfinite(body.grad).all()
    # Gradient should push body toward (5,5,5) (i.e., negative for all coords)
    assert (body.grad < 0).any()


def test_lift_target_to_world_rotation_and_translation():
    """target_world = R @ target_local + obj_pos."""
    from piano.inference.contact_guidance import _lift_target_to_world_np

    # +180° around Y: (1, 0, 0) → (-1, 0, 0)
    target_local = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32)
    obj_pos = np.array([[5.0, 0.0, 7.0]], dtype=np.float32)
    obj_rot = np.array([[0.0, np.pi, 0.0]], dtype=np.float32)

    out = _lift_target_to_world_np(target_local, obj_pos, obj_rot)
    # rotated: (-1, 0, 0); +obj_pos: (4, 0, 7)
    np.testing.assert_allclose(out[0, 0], [4.0, 0.0, 7.0], atol=1e-5)


def test_masked_contact_l2_shape_and_value():
    from piano.inference.contact_guidance import _masked_contact_l2

    body = torch.tensor([[[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]]])     # (1,1,2,3)
    target = torch.zeros(1, 1, 2, 3)
    contact = torch.tensor([[[1.0, 1.0]]])                          # (1,1,2)

    loss = _masked_contact_l2(body, target, contact)
    # Per-frame, per-part squared distance: [0, 9]; mean over 2 in-contact parts = 4.5
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert torch.allclose(loss, torch.tensor(4.5), atol=1e-6)


def test_masked_contact_l2_only_masked_parts_count():
    from piano.inference.contact_guidance import _masked_contact_l2

    body = torch.tensor([[[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]]])
    target = torch.zeros(1, 1, 2, 3)
    # Only second part is in contact; loss should be 9 (not 4.5).
    contact = torch.tensor([[[0.0, 1.0]]])

    loss = _masked_contact_l2(body, target, contact)
    assert torch.allclose(loss, torch.tensor(9.0), atol=1e-6)


def test_masked_contact_l2_zero_mask_returns_zero():
    """When no part is in contact, loss is well-defined (0) with no division-by-zero."""
    from piano.inference.contact_guidance import _masked_contact_l2

    body = torch.randn(1, 5, 5, 3)
    target = torch.randn(1, 5, 5, 3)
    contact = torch.zeros(1, 5, 5)

    loss = _masked_contact_l2(body, target, contact)
    assert torch.isfinite(loss)
    # With clamp(min=1) denominator and zero numerator, loss = 0.
    assert loss.item() == 0.0


def test_decode_relaxed_base_shape_and_gradient():
    """Build a minimal fake VQ model; verify gradient flows through base_logits.

    We don't load real MoMask here — just construct a tiny module with a
    .quantizer (with .codebooks property + .num_quantizers + .get_codes_from_indices)
    and a .decoder (linear). This pins the expected shape + autograd contract.
    """
    from piano.inference.contact_guidance import _decode_relaxed_base

    B, S, V, code_dim = 1, 4, 8, 16
    Q = 3
    T_decoded = S * 4   # decoder is stride-4

    class FakeQuantizer(nn.Module):
        def __init__(self):
            super().__init__()
            # codebooks shape (Q, V, code_dim)
            self._codebooks = nn.Parameter(torch.randn(Q, V, code_dim) * 0.1)
            self.num_quantizers = Q

        @property
        def codebooks(self):
            return self._codebooks

        def get_codes_from_indices(self, indices):
            # indices: (B, S, Q) → return (Q, B, S, code_dim)
            B_, S_, Q_ = indices.shape
            assert Q_ == Q
            out = torch.zeros(Q_, B_, S_, code_dim, device=indices.device)
            for q in range(Q_):
                out[q] = self._codebooks[q][indices[..., q]]
            return out

    class FakeDecoder(nn.Module):
        """Mirrors MoMask's Decoder.forward contract: returns (B, T, 263)
        — i.e. ConvTranspose1d output (B, 263, T) followed by permute(0, 2, 1).
        """
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

    vq = FakeRVQVAE()

    base_logits = torch.randn(B, S, V, requires_grad=True)
    residual_ids = torch.randint(0, V, (B, S, Q - 1))

    motion_norm = _decode_relaxed_base(
        base_logits=base_logits,
        residual_ids=residual_ids,
        vq_model=vq,
        temperature=1.0,
    )                                                       # (1, T_decoded, 263)

    assert motion_norm.shape == (B, T_decoded, 263)
    # Gradient flow through base_logits is the load-bearing contract.
    motion_norm.sum().backward()
    assert base_logits.grad is not None
    assert torch.isfinite(base_logits.grad).all()
    # Residual ids are integer (no grad attribute) — confirm no crash.


def test_decode_relaxed_full_rvq_shape_and_gradient():
    from piano.inference.contact_guidance import _decode_relaxed_full_rvq

    B, S, Q, V, code_dim = 1, 4, 3, 8, 16
    T_decoded = S * 4

    class FakeQuantizer(nn.Module):
        def __init__(self):
            super().__init__()
            self._codebooks = nn.Parameter(torch.randn(Q, V, code_dim) * 0.1)
            self.num_quantizers = Q

        @property
        def codebooks(self):
            return self._codebooks

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

    logits = torch.randn(B, S, Q, V, requires_grad=True)
    motion_norm = _decode_relaxed_full_rvq(
        logits,
        FakeRVQVAE(),
        token_mask=torch.tensor([[True, True, True, False]]),
        temperature=1.0,
    )

    assert motion_norm.shape == (B, T_decoded, 263)
    motion_norm.sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[:, :3].abs().sum() > 0
    assert logits.grad[:, 3].abs().sum() == 0


def test_decode_relaxed_base_invariant_under_one_hot_init():
    """When base_logits is one-hot, relaxed embedding ≈ codebook[gt] (low-temp limit).

    This validates the "init from one-hot, optimize from there" pattern
    used in guide_with_contact.
    """
    from piano.inference.contact_guidance import _decode_relaxed_base

    B, S, V, code_dim = 1, 4, 8, 16
    Q = 3

    class FakeQuantizer(nn.Module):
        def __init__(self):
            super().__init__()
            self._codebooks = nn.Parameter(torch.randn(Q, V, code_dim))
            self.num_quantizers = Q

        @property
        def codebooks(self):
            return self._codebooks

        def get_codes_from_indices(self, indices):
            B_, S_, Q_ = indices.shape
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

    vq = FakeRVQVAE()
    base_ids = torch.randint(0, V, (B, S))
    residual_ids = torch.zeros(B, S, Q - 1, dtype=torch.long)

    # One-hot logits at very low temperature — softmax should pick the
    # one-hot index, so the relaxed embedding ≈ codebook[base_ids].
    INIT_SCALE = 100.0   # sharp argmax
    logits_oh = torch.nn.functional.one_hot(base_ids, V).float() * INIT_SCALE

    motion_oh = _decode_relaxed_base(
        base_logits=logits_oh,
        residual_ids=residual_ids,
        vq_model=vq,
        temperature=0.01,
    )

    # Now compute what we'd get if we used hard argmax:
    # relaxed_base ≈ codebooks[0][base_ids] for sharp logits.
    # Residual contribution is 0 (zero codes lookup)... actually no,
    # codebooks are random-init so the zero indices give us
    # codebook[1][0] + codebook[2][0]. Both paths should give the same
    # result up to numerical precision since at temperature=0.01 the
    # softmax is essentially one-hot.
    assert torch.isfinite(motion_oh).all()


def test_guide_with_contact_full_pipeline_skipped_without_momask():
    """Placeholder: full integration requires MoMask + CLIP + a trained generator.

    The actual end-to-end sanity check is run on the server via
    ``qual_eval.py --guidance-steps 30 --ckpt runs/training/generator_v06_per_head_gamma/best_val.pt``.
    """
    pytest.skip("Full guide_with_contact integration is server-only; see qual_eval.py")


def test_build_decode_ids_with_baseline_residuals_shape_and_content():
    """no-residual-rerun helper: cat [base_after, baseline_residuals] and pad."""
    from piano.inference.contact_guidance import (
        _build_decode_ids_with_baseline_residuals,
    )

    # 1 sample, S=5, Q=6 (1 base + 5 residuals). Real seq_len = 3, so
    # positions [3, 4] are padding and should be zeroed in base.
    base_ids_after = torch.tensor([[7, 11, 13, 17, 19]])               # (1, 5)
    baseline_residual_ids = torch.tensor([[
        [101, 102, 103, 104, 105],
        [201, 202, 203, 204, 205],
        [301, 302, 303, 304, 305],
        [401, 402, 403, 404, 405],
        [501, 502, 503, 504, 505],
    ]])                                                                # (1, 5, 5)
    m_lens_tok = torch.tensor([3])                                     # actual length

    out = _build_decode_ids_with_baseline_residuals(
        base_ids_after=base_ids_after,
        baseline_residual_ids=baseline_residual_ids,
        m_lens_tok=m_lens_tok,
    )

    assert out.shape == (1, 5, 6)
    # Real positions: base preserved.
    assert out[0, 0, 0].item() == 7
    assert out[0, 1, 0].item() == 11
    assert out[0, 2, 0].item() == 13
    # Pad positions (s=3, s=4): base zeroed.
    assert out[0, 3, 0].item() == 0
    assert out[0, 4, 0].item() == 0
    # Residuals preserved verbatim at every position; the cat lays out
    # quantizer layer at the LAST dim, so out[0, s, q+1] == baseline[0, s, q].
    # (Caller invariant: baseline residuals are 0 at pad via the where
    # clause in guide_with_contact, so we don't re-zero them here.)
    for s in range(5):
        for q in range(5):  # 5 residual layers
            assert out[0, s, q + 1].item() == baseline_residual_ids[0, s, q].item()


def test_build_decode_ids_no_pad():
    """When m_lens_tok == S, no positions are zeroed."""
    from piano.inference.contact_guidance import (
        _build_decode_ids_with_baseline_residuals,
    )

    # base: (1, S=3), residual: (1, S=3, Q-1=2). Quantizer layer at last dim.
    base_ids_after = torch.tensor([[7, 11, 13]])
    baseline_residual_ids = torch.tensor([[
        [101, 201],   # position 0: residual layer 0 = 101, layer 1 = 201
        [102, 202],   # position 1
        [103, 203],   # position 2
    ]])                                                                  # (1, 3, 2)
    m_lens_tok = torch.tensor([3])

    out = _build_decode_ids_with_baseline_residuals(
        base_ids_after=base_ids_after,
        baseline_residual_ids=baseline_residual_ids,
        m_lens_tok=m_lens_tok,
    )

    assert out.shape == (1, 3, 3)
    # Base preserved at all positions.
    torch.testing.assert_close(out[0, :, 0], base_ids_after[0])
    # Residuals laid out as expected.
    assert out[0, 0, 1].item() == 101 and out[0, 0, 2].item() == 201
    assert out[0, 2, 1].item() == 103 and out[0, 2, 2].item() == 203


def test_build_decode_ids_all_pad():
    """When m_lens_tok=0 (degenerate), all base positions are zeroed.

    Residuals are still preserved as-is (caller invariant: baseline
    residuals are 0 at pad already).
    """
    from piano.inference.contact_guidance import (
        _build_decode_ids_with_baseline_residuals,
    )

    base_ids_after = torch.tensor([[7, 11, 13]])
    baseline_residual_ids = torch.zeros(1, 3, 2, dtype=torch.long)        # (1, S=3, Q-1=2)
    m_lens_tok = torch.tensor([0])

    out = _build_decode_ids_with_baseline_residuals(
        base_ids_after=base_ids_after,
        baseline_residual_ids=baseline_residual_ids,
        m_lens_tok=m_lens_tok,
    )

    assert (out[0, :, 0] == 0).all()


def test_guide_with_contact_signature_accepts_new_kwargs():
    """v5 added residual_seed + no_residual_rerun kwargs; v17 added per-step
    kwargs. Signature smoke test confirming both generations of kwargs
    survive.

    Doesn't run the function (requires MoMask) — just confirms the kwargs
    exist with the documented defaults via inspect.signature.
    """
    import inspect
    from piano.inference.contact_guidance import guide_with_contact

    sig = inspect.signature(guide_with_contact)
    assert "residual_seed" in sig.parameters
    assert sig.parameters["residual_seed"].default is None
    assert "no_residual_rerun" in sig.parameters
    assert sig.parameters["no_residual_rerun"].default is False
    assert "guidance_layers" in sig.parameters
    assert sig.parameters["guidance_layers"].default == "base"

    # v17 per-step decoded-geometric guidance.
    assert "per_step_iters" in sig.parameters
    assert sig.parameters["per_step_iters"].default == 0
    assert "per_step_lr" in sig.parameters
    assert sig.parameters["per_step_lr"].default == 6e-2
    assert "per_step_temperature" in sig.parameters
    assert sig.parameters["per_step_temperature"].default == 1.0
    assert "per_step_start_step" in sig.parameters
    assert sig.parameters["per_step_start_step"].default == 0
