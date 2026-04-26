"""Stage B forward-shape + zero-init smoke tests.

These tests verify the new IntXAttn sublayer implementation without
loading the full MoMask checkpoint (CLIP would need network + 600 MB
weights). The key invariant — **the wrapped model is byte-identical
to the un-wrapped MaskTransformer at γ_int=0** — can be tested
directly on the encoder, since the wrapper preserves the original
``nn.TransformerEncoderLayer`` instances unchanged.

What they cover:

- InteractionTokenizer: channel concat math, T → S downsample, per-sample
  padding mask shape and content.
- MaskTransformerBlockWithInteraction: byte-identity to its original
  ``nn.TransformerEncoderLayer`` at γ_int=0 (both ``int_kv=None`` and
  ``int_kv=given``).
- MaskTransformerEncoderWithInteraction: identity to a stock
  ``nn.TransformerEncoder`` at γ_int=0.
- sample_cfg_buckets: bucket marginal probabilities (large-N law).
- Param-group sets ``new_parameters()`` / ``backbone_parameters()``:
  disjoint, complete, and the right things in each.

Server-only:
- End-to-end load + 100-step finetune (needs MoMask ckpt + CLIP).
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from piano.models.interaction_tokenizer import (
    DEFAULT_NUM_BODY_PARTS,
    DEFAULT_NUM_OBJ_POSE_CHANNELS,
    DEFAULT_NUM_PHASES,
    DEFAULT_NUM_SUPPORT,
    DEFAULT_TARGET_COORD_DIM,
    DEFAULT_TOKEN_STRIDE,
    InteractionTokenizer,
    z_int_input_dim,
)
from piano.models.motion_generator import (
    MaskTransformerBlockWithInteraction,
    MaskTransformerEncoderWithInteraction,
    sample_cfg_buckets,
)


D_MODEL = 64       # small for CPU speed
NUM_HEADS = 4
N_LAYERS = 3
T_FRAMES = 32
TOKEN_STRIDE = 4
S_TOKENS = T_FRAMES // TOKEN_STRIDE


# ============================================================================
# InteractionTokenizer
# ============================================================================

@pytest.fixture
def tokenizer() -> InteractionTokenizer:
    return InteractionTokenizer(
        d_model=D_MODEL,
        num_body_parts=DEFAULT_NUM_BODY_PARTS,
        target_coord_dim=DEFAULT_TARGET_COORD_DIM,
        num_phases=DEFAULT_NUM_PHASES,
        num_support=DEFAULT_NUM_SUPPORT,
        token_stride=TOKEN_STRIDE,
        max_seq_length=T_FRAMES,
    )


def _make_z_int(B: int, T: int, with_obj_pose: bool = True):
    torch.manual_seed(0)
    contact_state = torch.rand(B, T, DEFAULT_NUM_BODY_PARTS)
    contact_target_xyz = torch.randn(B, T, DEFAULT_NUM_BODY_PARTS, DEFAULT_TARGET_COORD_DIM)
    phase = torch.randint(0, DEFAULT_NUM_PHASES, (B, T))
    support = torch.randint(0, DEFAULT_NUM_SUPPORT, (B, T))
    out = (contact_state, contact_target_xyz, phase, support)
    if not with_obj_pose:
        return out
    # v0.2: add per-frame body-canonical object pose (3 + 6).
    obj_com = torch.randn(B, T, 3)
    obj_rot6d = torch.randn(B, T, 6)
    return out + (obj_com, obj_rot6d)


def test_z_int_input_dim_default() -> None:
    """5 + 5×3 + 3 + 4 + 9 = 36 (v0.2: includes obj pose channels per
    analyses/2026-04-27_object_conditioning_review.md §5.2)."""
    assert z_int_input_dim() == 36
    # And without obj pose: 27 (v0.1).
    assert z_int_input_dim(num_obj_pose_channels=0) == 27


def test_tokenizer_shape_and_downsample(tokenizer: InteractionTokenizer) -> None:
    B = 2
    cs, ctx, ph, sup, oc, or6d = _make_z_int(B, T_FRAMES)
    seq_lens = torch.tensor([T_FRAMES, T_FRAMES // 2])

    kv, pad_mask = tokenizer(
        cs, ctx, ph, sup,
        obj_com_canonical=oc, obj_rot6d_canonical=or6d,
        seq_lens=seq_lens,
    )

    # Output shape: (B, S, d_model) where S = T // stride.
    assert kv.shape == (B, S_TOKENS, D_MODEL)
    # Padding mask: (B, S), True at padded token positions.
    assert pad_mask is not None
    assert pad_mask.shape == (B, S_TOKENS)
    # Sample 0 has full length → no positions padded.
    assert pad_mask[0].sum() == 0
    # Sample 1 has T/2 frames → S/2 valid tokens, S/2 padded.
    assert pad_mask[1].sum() == S_TOKENS // 2


def test_tokenizer_accepts_one_hot_inputs(tokenizer: InteractionTokenizer) -> None:
    """phase / support can also come in as one-hot floats — useful when
    feeding predictor outputs (which are softmax probs) into Stage 4."""
    B = 2
    cs, ctx, ph_int, sup_int, oc, or6d = _make_z_int(B, T_FRAMES)
    ph_oh = torch.nn.functional.one_hot(ph_int, DEFAULT_NUM_PHASES).float()
    sup_oh = torch.nn.functional.one_hot(sup_int, DEFAULT_NUM_SUPPORT).float()
    kv_a, _ = tokenizer(
        cs, ctx, ph_int, sup_int,
        obj_com_canonical=oc, obj_rot6d_canonical=or6d,
    )
    kv_b, _ = tokenizer(
        cs, ctx, ph_oh, sup_oh,
        obj_com_canonical=oc, obj_rot6d_canonical=or6d,
    )
    assert torch.allclose(kv_a, kv_b, atol=1e-6)


def test_tokenizer_rejects_wrong_target_shape(tokenizer: InteractionTokenizer) -> None:
    B = 1
    cs = torch.zeros(B, T_FRAMES, DEFAULT_NUM_BODY_PARTS)
    ctx_wrong = torch.zeros(B, T_FRAMES, DEFAULT_NUM_BODY_PARTS * 3)   # flat instead of (B,T,5,3)
    ph = torch.zeros(B, T_FRAMES, dtype=torch.long)
    sup = torch.zeros(B, T_FRAMES, dtype=torch.long)
    oc = torch.zeros(B, T_FRAMES, 3)
    or6d = torch.zeros(B, T_FRAMES, 6)
    with pytest.raises(ValueError, match="contact_target_xyz"):
        tokenizer(cs, ctx_wrong, ph, sup,
                  obj_com_canonical=oc, obj_rot6d_canonical=or6d)


def test_tokenizer_rejects_missing_obj_pose() -> None:
    """v0.2: when num_obj_pose_channels > 0 but obj pose isn't passed,
    we should fail loudly rather than silently produce an undersized
    concat."""
    tok = InteractionTokenizer(
        d_model=D_MODEL, token_stride=TOKEN_STRIDE,
        max_seq_length=T_FRAMES, num_obj_pose_channels=9,
    )
    cs, ctx, ph, sup, _oc, _or6d = _make_z_int(1, T_FRAMES)
    with pytest.raises(ValueError, match="num_obj_pose_channels"):
        tok(cs, ctx, ph, sup)


def test_tokenizer_v0_1_compat() -> None:
    """Caller can opt out of v0.2 channels with num_obj_pose_channels=0."""
    tok = InteractionTokenizer(
        d_model=D_MODEL, token_stride=TOKEN_STRIDE,
        max_seq_length=T_FRAMES, num_obj_pose_channels=0,
    )
    cs, ctx, ph, sup = _make_z_int(2, T_FRAMES, with_obj_pose=False)
    kv, _ = tok(cs, ctx, ph, sup)
    assert kv.shape == (2, S_TOKENS, D_MODEL)


def test_canonical_frame_roundtrip() -> None:
    """world_to_canonical_object_pose followed by an inverse rotation +
    translation should recover the original world-frame object COM."""
    from piano.utils.canonical_frame import (
        world_to_canonical_object_pose, y_rotation_matrix,
    )
    rng = torch.manual_seed(0)
    T = 12
    obj_pos_world = torch.randn(T, 3).numpy()
    obj_rot_world_aa = torch.randn(T, 3).numpy() * 0.5
    R_y = 0.7
    T_xz = torch.tensor([1.5, -0.3]).numpy()

    obj_com_can, _obj_rot6d_can = world_to_canonical_object_pose(
        obj_pos_world, obj_rot_world_aa, R_y, T_xz,
    )
    # Re-apply the forward transform and check we get back the world pos.
    R = y_rotation_matrix(R_y)
    recovered = obj_com_can @ R.T
    recovered[..., 0] += T_xz[0]
    recovered[..., 2] += T_xz[1]
    import numpy as np
    assert np.allclose(recovered, obj_pos_world, atol=1e-5)


def test_compute_canonical_object_pose_force_world_frame() -> None:
    """v0.3-α: ``force_world_frame=True`` short-circuits canonicalization.

    The output ``obj_com`` should equal the input world ``object_positions``
    exactly (identity transform), and ``obj_rot6d`` should equal the 6D
    rep of the world axis-angle (no R_y composition). This branch DOES
    NOT trigger MoMask path setup or ``recover_from_ric``, so the test
    runs without a server.
    """
    import numpy as np
    from piano.data.dataset import HOIDataset
    from piano.utils.canonical_frame import (
        axis_angle_to_matrix_np, matrix_to_rotation_6d_np,
    )

    rng = np.random.default_rng(42)
    T = 12
    # motion_263 + joints are NOT touched in the force_world_frame=True
    # branch — pass small dummies so the call signature is satisfied.
    motion_dummy = rng.standard_normal((T, 263)).astype(np.float32)
    joints_dummy = rng.standard_normal((T, 22, 3)).astype(np.float32)
    obj_pos_world = rng.standard_normal((T, 3)).astype(np.float32)
    obj_rot_world_aa = (rng.standard_normal((T, 3)) * 0.5).astype(np.float32)

    obj_com, obj_rot6d = HOIDataset._compute_canonical_object_pose(
        motion_dummy, joints_dummy, obj_pos_world, obj_rot_world_aa,
        force_world_frame=True,
    )
    # Identity-transform expectation: COM equals the input world position.
    assert obj_com.shape == (T, 3)
    np.testing.assert_allclose(obj_com, obj_pos_world, atol=1e-6)
    # 6D rotation should match the axis-angle → matrix → first-2-cols
    # composition (no R_y rotation applied since R_y=0 → identity).
    expected_rot6d = matrix_to_rotation_6d_np(
        axis_angle_to_matrix_np(obj_rot_world_aa),
    )
    assert obj_rot6d.shape == (T, 6)
    np.testing.assert_allclose(obj_rot6d, expected_rot6d, atol=1e-6)


# ============================================================================
# Block byte-identity at γ_int = 0
# ============================================================================

def _stock_encoder_layer() -> nn.TransformerEncoderLayer:
    """Build a MoMask-style ``nn.TransformerEncoderLayer`` (post-norm,
    GELU activation, defaults match
    ``backbones/momask/models/mask_transformer/transformer.py:110-114``)."""
    return nn.TransformerEncoderLayer(
        d_model=D_MODEL, nhead=NUM_HEADS, dim_feedforward=128,
        dropout=0.0, activation="gelu",
    )


def test_block_byte_identical_at_gamma_zero_no_int_kv() -> None:
    """When ``int_kv=None`` the block must equal the original MoMask layer.

    This is the easy direction: the IntXAttn branch is skipped entirely,
    so the wrapper is a structural pass-through.
    """
    torch.manual_seed(0)
    layer = _stock_encoder_layer().eval()
    block = MaskTransformerBlockWithInteraction(
        original_layer=layer, d_model=D_MODEL, num_heads=NUM_HEADS, dropout=0.0,
    ).eval()

    src = torch.randn(S_TOKENS + 1, 2, D_MODEL)
    with torch.no_grad():
        ref = layer(src)
        out = block(src, int_kv=None)
    assert torch.allclose(out, ref, atol=1e-6)


def test_block_byte_identical_at_gamma_zero_with_int_kv() -> None:
    """When ``int_kv`` is supplied but γ_int=0, the new sublayer's
    contribution is exactly zero (γ-gated residual). The block must
    still equal the original layer.

    This is the load-bearing invariant from
    analyses/2026-04-26_stageB_design.md §1.3 / §6.5: zero-init makes
    the modified MoMask byte-identical to pretrained at step 0.
    """
    torch.manual_seed(0)
    layer = _stock_encoder_layer().eval()
    block = MaskTransformerBlockWithInteraction(
        original_layer=layer, d_model=D_MODEL, num_heads=NUM_HEADS,
        dropout=0.0, zero_init_gamma=True,
    ).eval()

    src = torch.randn(S_TOKENS + 1, 2, D_MODEL)
    int_kv = torch.randn(S_TOKENS, 2, D_MODEL)
    with torch.no_grad():
        ref = layer(src)
        out = block(src, int_kv=int_kv)

    assert torch.allclose(out, ref, atol=1e-6), (
        "γ_int=0 must make the IntXAttn contribution exactly zero so "
        "the wrapped model is byte-identical to pretrained MoMask."
    )


def test_block_diverges_when_gamma_grows() -> None:
    """Sanity check: setting γ_int to non-zero should change the output."""
    torch.manual_seed(0)
    layer = _stock_encoder_layer().eval()
    block = MaskTransformerBlockWithInteraction(
        original_layer=layer, d_model=D_MODEL, num_heads=NUM_HEADS,
        dropout=0.0, zero_init_gamma=True,
    ).eval()

    src = torch.randn(S_TOKENS + 1, 2, D_MODEL)
    int_kv = torch.randn(S_TOKENS, 2, D_MODEL)
    with torch.no_grad():
        out_a = block(src, int_kv=int_kv)
        block.gamma_int.data.fill_(1.0)
        out_b = block(src, int_kv=int_kv)
    assert not torch.allclose(out_a, out_b)


# ============================================================================
# Encoder wrapper byte-identity
# ============================================================================

def test_encoder_byte_identical_at_gamma_zero() -> None:
    """The full :class:`MaskTransformerEncoderWithInteraction` must
    equal a stock ``nn.TransformerEncoder`` at γ_int=0, independent of
    whether ``int_kv`` is provided. This is the key property the
    Stage B finetune relies on: at step 0, the wrapped model produces
    the exact same logits as pretrained MoMask, so finetuning starts
    from the published FID=0.045 baseline rather than a random
    initialisation."""
    torch.manual_seed(0)
    layer = _stock_encoder_layer()
    encoder = nn.TransformerEncoder(layer, num_layers=N_LAYERS).eval()
    wrapped = MaskTransformerEncoderWithInteraction(
        original_encoder=encoder, d_model=D_MODEL,
        num_heads=NUM_HEADS, dropout=0.0, zero_init_gamma=True,
    ).eval()

    src = torch.randn(S_TOKENS + 1, 2, D_MODEL)
    int_kv = torch.randn(S_TOKENS, 2, D_MODEL)

    with torch.no_grad():
        ref = encoder(src)
        out_no_int = wrapped(src, int_kv=None)
        out_with_int = wrapped(src, int_kv=int_kv)

    assert torch.allclose(out_no_int, ref, atol=1e-6)
    assert torch.allclose(out_with_int, ref, atol=1e-6)


def test_encoder_preserves_original_layer_weights() -> None:
    """``MaskTransformerEncoderWithInteraction`` must not clone or
    re-initialise the original layers — it holds them by reference.
    Verifying via param identity (``id(p)``) so an accidental
    ``deepcopy`` on the layer would be caught."""
    layer = _stock_encoder_layer()
    encoder = nn.TransformerEncoder(layer, num_layers=N_LAYERS)
    wrapped = MaskTransformerEncoderWithInteraction(
        original_encoder=encoder, d_model=D_MODEL,
        num_heads=NUM_HEADS, dropout=0.0,
    )
    for orig_layer, wrapped_blk in zip(encoder.layers, wrapped.layers):
        assert wrapped_blk.layer is orig_layer
        for orig_p, wrap_p in zip(orig_layer.parameters(), wrapped_blk.layer.parameters()):
            assert id(orig_p) == id(wrap_p)


# ============================================================================
# CFG bucket sampling
# ============================================================================

def test_cfg_buckets_marginals_match_design() -> None:
    """Large-N marginal-probability check.

    Design §2.2 dictates p(drop_text) = 10% + 5% = 15% and
    p(drop_int) = 10% + 10% = 20%, with the joint event
    p(drop_both) = 10%. Sampling 50 000 buckets must hit each within
    ~1 percentage point.
    """
    torch.manual_seed(0)
    N = 50_000
    drop_text, drop_int = sample_cfg_buckets(
        N, p_drop_both=0.10, p_drop_int_only=0.10, p_drop_text_only=0.05,
    )
    p_drop_text = drop_text.float().mean().item()
    p_drop_int = drop_int.float().mean().item()
    p_drop_both = (drop_text & drop_int).float().mean().item()

    assert abs(p_drop_text - 0.15) < 0.01
    assert abs(p_drop_int - 0.20) < 0.01
    assert abs(p_drop_both - 0.10) < 0.01


def test_cfg_buckets_rejects_oversum() -> None:
    with pytest.raises(ValueError, match="bucket probabilities"):
        sample_cfg_buckets(8, p_drop_both=0.5, p_drop_int_only=0.5, p_drop_text_only=0.5)


# ============================================================================
# Param-group separation (uses a fake MaskTransformer to avoid loading CLIP)
# ============================================================================

class _FakeMaskTransformer(nn.Module):
    """Minimal stand-in for MoMask's MaskTransformer that
    :class:`InteractionMaskTransformer.__init__` can patch.

    Has the attributes the wrapper inspects:
        - ``latent_dim`` (int)
        - ``dropout`` (float)
        - ``cond_drop_prob`` (mutable float — wrapper sets it to 0)
        - ``seqTransEncoder`` (nn.TransformerEncoder)
    Anything else (token_emb, output_process, mask_id, opt, ...) is
    not exercised by these tests.
    """

    def __init__(self) -> None:
        super().__init__()
        self.latent_dim = D_MODEL
        self.dropout = 0.0
        self.cond_drop_prob = 0.1
        layer = _stock_encoder_layer()
        self.seqTransEncoder = nn.TransformerEncoder(layer, num_layers=N_LAYERS)
        # The wrapper also reads token_emb / cond_emb / output_process
        # only from inside trans_forward; we don't call trans_forward
        # here, so empty stubs suffice.
        self.token_emb = nn.Embedding(8, D_MODEL)
        self.cond_emb = nn.Linear(D_MODEL, D_MODEL)


def test_param_group_separation() -> None:
    """``new_parameters`` and ``backbone_parameters`` must be disjoint
    AND together cover all trainable params (excluding CLIP, but our
    fake has no CLIP). This is what
    :func:`build_two_group_optimizer` relies on to assign LRs without
    double-counting or missing weights."""
    from piano.models.motion_generator import InteractionMaskTransformer

    fake = _FakeMaskTransformer()
    tok = InteractionTokenizer(
        d_model=D_MODEL, token_stride=TOKEN_STRIDE, max_seq_length=T_FRAMES,
    )
    wrapper = InteractionMaskTransformer(
        mask_transformer=fake, interaction_tokenizer=tok,
        max_token_seq_length=S_TOKENS,
    )

    new_ids = {id(p) for p in wrapper.new_parameters()}
    bb_ids = {id(p) for p in wrapper.backbone_parameters()}
    all_trainable = {
        id(p) for p in wrapper.parameters() if p.requires_grad
    }

    # Disjoint
    assert new_ids.isdisjoint(bb_ids)
    # Complete (every trainable param falls into exactly one group).
    # The wrapper has no CLIP under the fake, so all trainable params
    # should be covered.
    assert new_ids | bb_ids == all_trainable

    # The new group must contain γ_int (one per encoder layer) and
    # null_int_kv. Sanity-check counts.
    gamma_count = sum(
        1 for blk in wrapper.mask_transformer.seqTransEncoder.layers
        if id(blk.gamma_int) in new_ids
    )
    assert gamma_count == N_LAYERS
    assert id(wrapper.null_int_kv) in new_ids


def test_wrapper_disables_momask_internal_text_drop() -> None:
    """At construction, the wrapper sets ``mask_transformer.cond_drop_prob = 0``
    so MoMask's own per-batch Bernoulli text drop doesn't fire on top
    of our explicit per-sample CFG bucket drops. Without this, training
    would see an additional uncoordinated 10% text-drop and the CFG
    arithmetic would no longer match the design."""
    from piano.models.motion_generator import InteractionMaskTransformer

    fake = _FakeMaskTransformer()
    fake.cond_drop_prob = 0.1
    tok = InteractionTokenizer(
        d_model=D_MODEL, token_stride=TOKEN_STRIDE, max_seq_length=T_FRAMES,
    )
    wrapper = InteractionMaskTransformer(
        mask_transformer=fake, interaction_tokenizer=tok,
        max_token_seq_length=S_TOKENS,
    )
    assert wrapper.mask_transformer.cond_drop_prob == 0.0
