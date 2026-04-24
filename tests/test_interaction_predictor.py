"""Forward-shape + masking smoke tests for the Interaction Predictor stack.

These tests only check that the model wires up correctly — they do not
validate learning dynamics. Runs CPU-only with tiny batches so they stay
fast in the same pytest suite as the pseudo-label tests.
"""
from __future__ import annotations

import pytest
import torch

from piano.models.interaction_predictor import InteractionPredictor, PredictorBlock
from piano.models.object_encoder import ObjectEncoder
from piano.training.losses import PredictorLoss


TEXT_DIM = 512
D_MODEL = 64      # small for CPU speed
NUM_HEADS = 4
N_LAYERS = 2
MAX_SEQ = 32
N_BODY = 5
N_PATCH = 16
N_PHASE = 5
N_SUPPORT = 4


@pytest.fixture
def predictor() -> InteractionPredictor:
    return InteractionPredictor(
        d_model=D_MODEL,
        num_layers=N_LAYERS,
        num_heads=NUM_HEADS,
        dim_feedforward=128,
        dropout=0.0,
        text_dim=TEXT_DIM,
        pose_dim=66,
        max_seq_length=MAX_SEQ,
        num_body_parts=N_BODY,
        num_object_patches=N_PATCH,
        num_phases=N_PHASE,
        num_support_states=N_SUPPORT,
    )


def _make_inputs(B: int, T: int, M: int = 16):
    torch.manual_seed(0)
    text_tokens = torch.randn(B, 77, TEXT_DIM)
    object_tokens = torch.randn(B, M, D_MODEL)
    init_pose = torch.randn(B, 66)
    return text_tokens, object_tokens, init_pose


def test_predictor_forward_shapes(predictor: InteractionPredictor) -> None:
    """Output heads emit per-frame predictions of the right shape."""
    B, T = 2, MAX_SEQ
    text_tokens, object_tokens, init_pose = _make_inputs(B, T)

    out = predictor(text_tokens, object_tokens, init_pose, seq_length=T)

    assert out["contact_state"].shape == (B, T, N_BODY)
    assert out["contact_target"].shape == (B, T, N_BODY, N_PATCH)
    assert out["phase"].shape == (B, T, N_PHASE)
    assert out["support"].shape == (B, T, N_SUPPORT)
    # Softmax rows sum to 1
    assert torch.allclose(
        out["phase"].sum(-1), torch.ones(B, T), atol=1e-5,
    )
    assert torch.allclose(
        out["contact_target"].sum(-1), torch.ones(B, T, N_BODY), atol=1e-5,
    )


def test_predictor_variable_seq_length(predictor: InteractionPredictor) -> None:
    """Predictor handles seq_length shorter than max_seq_length."""
    B = 2
    T = 7
    text_tokens, object_tokens, init_pose = _make_inputs(B, T)
    out = predictor(text_tokens, object_tokens, init_pose, seq_length=T)
    assert out["phase"].shape == (B, T, N_PHASE)


def test_predictor_respects_text_padding_mask(predictor: InteractionPredictor) -> None:
    """Masking padded CLIP positions must not error, and must actually be
    respected (identical text features after EOT → identical outputs)."""
    B, T = 1, MAX_SEQ
    text_tokens, object_tokens, init_pose = _make_inputs(B, T)

    mask = torch.zeros(B, 77, dtype=torch.bool)
    mask[:, 10:] = True  # treat positions 10+ as padding

    out_a = predictor(
        text_tokens, object_tokens, init_pose, seq_length=T,
        text_key_padding_mask=mask,
    )

    # Replace padding with random garbage — output should be identical
    text_tokens_b = text_tokens.clone()
    text_tokens_b[:, 10:] = torch.randn_like(text_tokens_b[:, 10:]) * 100

    predictor.eval()
    with torch.no_grad():
        out_a = predictor(
            text_tokens, object_tokens, init_pose, seq_length=T,
            text_key_padding_mask=mask,
        )
        out_b = predictor(
            text_tokens_b, object_tokens, init_pose, seq_length=T,
            text_key_padding_mask=mask,
        )
    assert torch.allclose(out_a["phase"], out_b["phase"], atol=1e-5)


def test_predictor_no_block_attn_res_attrs(predictor: InteractionPredictor) -> None:
    """Block AttnRes was removed in the 2026-04-24 rewrite — sanity check."""
    block: PredictorBlock = predictor.layers[0]
    for attr in (
        "sa_attn_res_proj", "sa_attn_res_norm",
        "ca_attn_res_proj", "ca_attn_res_norm",
        "ff_attn_res_proj", "ff_attn_res_norm",
        "adaln_sa", "adaln_ca", "adaln_ff",
    ):
        assert not hasattr(block, attr), f"{attr} should have been removed"


def test_object_encoder_shapes() -> None:
    """Object encoder emits the configured token count."""
    B, N = 2, 256   # keep N small for test speed
    enc = ObjectEncoder(num_input_points=N, num_output_tokens=32, feature_dim=D_MODEL)
    pc = torch.randn(B, N, 3)
    out = enc(pc)
    assert out.shape == (B, 32, D_MODEL)


def test_predictor_loss_target_gated_by_contact() -> None:
    """When every GT contact is below threshold, the target KL loss
    contributes nothing — verifying the new gating."""
    B, T = 2, 8
    torch.manual_seed(0)

    pred = {
        "contact_logits": torch.randn(B, T, N_BODY, requires_grad=True),
        "target_logits": torch.randn(B, T, N_BODY, N_PATCH, requires_grad=True),
        "phase_logits":  torch.randn(B, T, N_PHASE, requires_grad=True),
        "support_logits": torch.randn(B, T, N_SUPPORT, requires_grad=True),
    }
    gt_contact_zero = torch.zeros(B, T, N_BODY)
    gt_target = torch.full((B, T, N_BODY, N_PATCH), 1.0 / N_PATCH)
    gt_phase = torch.randint(0, N_PHASE, (B, T))
    gt_support = torch.randint(0, N_SUPPORT, (B, T))

    loss = PredictorLoss(contact_weight=0.0, target_weight=1.0,
                         phase_weight=0.0, support_weight=0.0)
    out = loss(pred, gt_contact_zero, gt_target, gt_phase, gt_support, mask=None)

    # No active contact anywhere → target loss averaged over ~0 → ~0
    assert out["loss_target"].abs().item() < 1e-4
    # With active contact, the target loss is non-trivial
    gt_contact_on = torch.ones(B, T, N_BODY)
    out2 = loss(pred, gt_contact_on, gt_target, gt_phase, gt_support, mask=None)
    assert out2["loss_target"].item() > 0.0
