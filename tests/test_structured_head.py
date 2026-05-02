"""Local sanity tests for the v8 StructuredHead + KL target loss.

Run via:
    python -m pytest tests/test_structured_head.py -xvs
or directly:
    python tests/test_structured_head.py
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from piano.models.interaction_predictor import (
    InteractionPredictor,
    StructuredHead,
)
from piano.models.object_encoder import ObjectEncoder
from piano.training.losses import PredictorLoss


# --------------------------------------------------------------------------
# Test fixtures
# --------------------------------------------------------------------------

def _make_inputs(B=2, T=8, M=128, d_model=384, num_parts=5):
    text_tokens = torch.randn(B, 77, 512)
    object_tokens = torch.randn(B, M, d_model)
    object_xyz = torch.randn(B, M, 3) * 0.5
    init_pose = torch.randn(B, 66) * 0.5
    gt_contact = (torch.rand(B, T, num_parts) > 0.5).float()
    gt_target = torch.randn(B, T, num_parts, 3) * 0.3
    gt_phase = torch.randint(0, 3, (B, T))
    gt_support = torch.randint(0, 4, (B, T))
    return dict(
        text_tokens=text_tokens, object_tokens=object_tokens,
        object_xyz=object_xyz, init_pose=init_pose,
        gt_contact=gt_contact, gt_target=gt_target,
        gt_phase=gt_phase, gt_support=gt_support,
    )


# --------------------------------------------------------------------------
# Test 1: Forward shape — StructuredHead alone
# --------------------------------------------------------------------------

def test_structured_head_forward_shape():
    B, T, M, d = 2, 8, 128, 384
    head = StructuredHead(d_model=d, num_body_parts=5,
                          num_phases=3, num_support_states=4)
    x = torch.randn(B, T, d)
    obj_tok = torch.randn(B, M, d)
    obj_xyz = torch.randn(B, M, 3)
    out = head(x, obj_tok, obj_xyz)
    assert out["contact_logits"].shape == (B, T, 5)
    assert out["contact_state"].shape == (B, T, 5)
    assert out["contact_target_attn"].shape == (B, T, 5, M)
    assert out["contact_target_xyz"].shape == (B, T, 5, 3)
    assert out["phase_logits"].shape == (B, T, 3)
    assert out["support_logits"].shape == (B, T, 4)
    # Attention sums to 1 over object-token dim
    attn_sum = out["contact_target_attn"].sum(dim=-1)
    assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-4)
    print("[PASS] test_structured_head_forward_shape")


# --------------------------------------------------------------------------
# Test 2: Full predictor forward — both head modes
# --------------------------------------------------------------------------

def test_predictor_legacy_forward_unchanged():
    """structured_head=False produces v7-fix-shape output."""
    inp = _make_inputs(T=8)
    pred_model = InteractionPredictor(
        d_model=384, num_layers=2, num_heads=6, dim_feedforward=512,
        max_seq_length=196, structured_head=False,
    )
    pred_model.eval()
    with torch.no_grad():
        out = pred_model(
            inp["text_tokens"], inp["object_tokens"], inp["init_pose"],
            seq_length=8,
        )
    assert "contact_target_xyz" in out and out["contact_target_xyz"].shape == (2, 8, 5, 3)
    assert "contact_target_attn" not in out  # legacy path doesn't emit attn
    print("[PASS] test_predictor_legacy_forward_unchanged")


def test_predictor_structured_forward():
    """structured_head=True returns affordance attn + back-compat xyz."""
    inp = _make_inputs(T=8)
    pred_model = InteractionPredictor(
        d_model=384, num_layers=2, num_heads=6, dim_feedforward=512,
        max_seq_length=196, structured_head=True,
    )
    pred_model.eval()
    with torch.no_grad():
        out = pred_model(
            inp["text_tokens"], inp["object_tokens"], inp["init_pose"],
            seq_length=8,
            object_xyz=inp["object_xyz"],
        )
    assert out["contact_target_attn"].shape == (2, 8, 5, 128)
    assert out["contact_target_xyz"].shape == (2, 8, 5, 3)
    # Back-compat xyz must be inside the convex hull of object_xyz.
    obj_min = inp["object_xyz"].amin(dim=1, keepdim=True).unsqueeze(2)  # (B,1,1,3)
    obj_max = inp["object_xyz"].amax(dim=1, keepdim=True).unsqueeze(2)
    pred_xyz = out["contact_target_xyz"]
    assert torch.all(pred_xyz >= obj_min - 1e-5), "xyz below obj min"
    assert torch.all(pred_xyz <= obj_max + 1e-5), "xyz above obj max"
    print("[PASS] test_predictor_structured_forward")


# --------------------------------------------------------------------------
# Test 3: Teacher forcing toggle
# --------------------------------------------------------------------------

def test_teacher_forcing_toggle():
    """With TF=True, downstream heads receive GT contact (not pred)."""
    torch.manual_seed(0)
    inp = _make_inputs(T=8)
    pred_model = InteractionPredictor(
        d_model=384, num_layers=2, num_heads=6, dim_feedforward=512,
        max_seq_length=196, structured_head=True,
    )
    pred_model.train()

    # Two forwards: first WITHOUT TF (use prediction), second WITH TF
    # (use GT). With deterministic non-zero gt_contact, these should
    # give different downstream outputs.
    out_no_tf = pred_model(
        inp["text_tokens"], inp["object_tokens"], inp["init_pose"],
        seq_length=8, object_xyz=inp["object_xyz"],
        gt_contact=inp["gt_contact"], gt_phase=inp["gt_phase"],
        teacher_forcing=False,
    )
    out_tf = pred_model(
        inp["text_tokens"], inp["object_tokens"], inp["init_pose"],
        seq_length=8, object_xyz=inp["object_xyz"],
        gt_contact=inp["gt_contact"], gt_phase=inp["gt_phase"],
        teacher_forcing=True,
    )
    # Phase output (Level 1b) should differ between the two modes
    # because phase head consumes contact_emb derived from GT vs pred.
    diff = (out_no_tf["phase_logits"] - out_tf["phase_logits"]).abs().max()
    assert diff > 1e-4, f"TF on/off should change phase output, got diff={diff}"
    print(f"[PASS] test_teacher_forcing_toggle (max phase_logits diff = {diff:.4f})")


# --------------------------------------------------------------------------
# Test 4: KL target loss positivity + zero at perfect prediction
# --------------------------------------------------------------------------

def test_kl_target_loss():
    B, T, P, M = 2, 4, 5, 128
    gt_xyz = torch.randn(B, T, P, 3) * 0.3
    object_xyz = torch.randn(B, M, 3) * 0.5
    sigma = 0.08

    # Compute the GT distribution exactly as the loss does, then feed
    # it back as the prediction → KL should be ~0.
    diff = gt_xyz.unsqueeze(-2) - object_xyz.view(B, 1, 1, M, 3)
    d_sq = diff.pow(2).sum(dim=-1)
    gt_attn = F.softmax(-d_sq / (2 * sigma * sigma), dim=-1)
    perfect_pred = gt_attn.clone()
    loss_perfect = PredictorLoss._kl_div_target_loss(
        perfect_pred, gt_xyz, object_xyz, sigma=sigma,
    )
    assert loss_perfect.shape == (B, T, P)
    assert loss_perfect.abs().max() < 1e-5, \
        f"KL of pred=GT should be ~0, got max {loss_perfect.abs().max()}"

    # Random prediction should give strictly positive loss
    random_pred = F.softmax(torch.randn(B, T, P, M), dim=-1)
    loss_random = PredictorLoss._kl_div_target_loss(
        random_pred, gt_xyz, object_xyz, sigma=sigma,
    )
    assert loss_random.mean() > 0
    print(f"[PASS] test_kl_target_loss "
          f"(perfect={loss_perfect.abs().max():.2e}, random_mean={loss_random.mean():.4f})")


# --------------------------------------------------------------------------
# Test 5: Consistency loss = 0 when constraints satisfied
# --------------------------------------------------------------------------

def test_consistency_loss_zero_when_satisfied():
    B, T, P, M = 2, 6, 5, 128
    # contact_logits all positive (sigmoid → ~1 — strong contact),
    # support all 0 logits (uniform softmax) → support[hand_supp]=0.25
    # vs hand_contact=1 → constraint satisfied.
    pred = {
        "contact_logits": torch.full((B, T, P), 5.0),  # sigmoid ≈ 1
        "support_logits": torch.zeros(B, T, 4),         # uniform
        "phase_logits":   torch.zeros(B, T, 3),         # uniform
        "contact_target_attn": F.softmax(torch.randn(B, T, P, M), dim=-1),
    }
    mask = torch.ones(B, T)
    loss_cons = PredictorLoss._consistency_loss(pred, mask)
    # All 4 hinges should be 0 (each p_dependent ≤ p_prerequisite).
    # Plus a small target-attention-entropy term ≈ 0 because contact ≈ 1
    # everywhere → no_contact factor is ~0.
    assert loss_cons.item() < 0.01, f"loss_cons should be ~0, got {loss_cons.item()}"
    print(f"[PASS] test_consistency_loss_zero_when_satisfied (loss={loss_cons.item():.6f})")


# --------------------------------------------------------------------------
# Test 6: Full PredictorLoss with v8 config — backward pass works
# --------------------------------------------------------------------------

def test_full_v8_loss_backward():
    inp = _make_inputs(T=8)
    pred_model = InteractionPredictor(
        d_model=384, num_layers=2, num_heads=6, dim_feedforward=512,
        max_seq_length=196, structured_head=True,
    )
    pred_model.train()
    out = pred_model(
        inp["text_tokens"], inp["object_tokens"], inp["init_pose"],
        seq_length=8, object_xyz=inp["object_xyz"],
        gt_contact=inp["gt_contact"], gt_phase=inp["gt_phase"],
        teacher_forcing=True,
    )
    loss_fn = PredictorLoss(
        contact_weight=2.0, target_weight=5.0,
        phase_weight=0.3, support_weight=0.1,
        target_loss_kind="kl_div", target_kernel_sigma=0.08,
        consistency_weight=0.1,
    )
    loss_dict = loss_fn(
        out,
        gt_contact=inp["gt_contact"],
        gt_target=inp["gt_target"],
        gt_phase=inp["gt_phase"].long(),
        gt_support=inp["gt_support"].long(),
        mask=torch.ones(2, 8, dtype=torch.bool),
        object_xyz=inp["object_xyz"],
    )
    total = loss_dict["loss"]
    assert total.requires_grad
    total.backward()
    # Spot-check: at least the structured head's parameters got grads
    assert pred_model.head.contact_head[0].weight.grad is not None
    assert pred_model.head.target_attn.in_proj_weight.grad is not None
    assert pred_model.head.support_head[0].weight.grad is not None
    print(f"[PASS] test_full_v8_loss_backward "
          f"(loss={total.item():.4f}, "
          f"loss_target={loss_dict['loss_target'].item():.4f}, "
          f"loss_consistency={loss_dict['loss_consistency'].item():.4f})")


# --------------------------------------------------------------------------
# Test 7: Backward compatibility regression — legacy v7-fix loss path
# --------------------------------------------------------------------------

def test_v7fix_legacy_loss_unchanged():
    """structured_head=False + smooth_l1 + consistency=0 reproduces v7-fix."""
    inp = _make_inputs(T=8)
    pred_model = InteractionPredictor(
        d_model=384, num_layers=2, num_heads=6, dim_feedforward=512,
        max_seq_length=196, structured_head=False,
    )
    pred_model.train()
    out = pred_model(
        inp["text_tokens"], inp["object_tokens"], inp["init_pose"],
        seq_length=8,
    )
    loss_fn = PredictorLoss(
        contact_weight=2.0, target_weight=5.0,
        phase_weight=0.3, support_weight=0.1,
        target_loss_kind="smooth_l1",
        target_gate_kind="all",
        consistency_weight=0.0,
    )
    loss_dict = loss_fn(
        out,
        gt_contact=inp["gt_contact"],
        gt_target=inp["gt_target"],
        gt_phase=inp["gt_phase"].long(),
        gt_support=inp["gt_support"].long(),
        mask=torch.ones(2, 8, dtype=torch.bool),
    )
    total = loss_dict["loss"]
    assert total.requires_grad
    total.backward()
    print(f"[PASS] test_v7fix_legacy_loss_unchanged (loss={total.item():.4f})")


# --------------------------------------------------------------------------
# Test 8: ObjectEncoder return_xyz toggle
# --------------------------------------------------------------------------

def test_object_encoder_return_xyz():
    enc = ObjectEncoder(num_input_points=256, num_output_tokens=64, feature_dim=128)
    enc.eval()
    pc = torch.randn(2, 256, 3)
    with torch.no_grad():
        feat = enc(pc, return_xyz=False)
        assert feat.shape == (2, 64, 128)
        xyz, feat2 = enc(pc, return_xyz=True)
        assert xyz.shape == (2, 64, 3)
        assert feat2.shape == (2, 64, 128)
    print("[PASS] test_object_encoder_return_xyz")


# --------------------------------------------------------------------------
# Test 9: End-to-end yaml load → predictor → loss compatibility
# --------------------------------------------------------------------------
#
# Regression test for the bug where ``structured_head`` was only read
# from the model yaml file (default off), ignoring the training yaml's
# override. Caught at server training start with ValueError on first
# batch: predictor produced no ``contact_target_attn`` because it was
# built with structured_head=False, but loss expected one because
# target_loss_kind=kl_div was correctly read from the training yaml.

def test_v8_config_yaml_end_to_end_compat():
    """Loading predictor_v8_structured.yaml must produce a predictor
    that emits contact_target_attn AND a loss configured for kl_div."""
    from pathlib import Path
    from omegaconf import OmegaConf

    repo_root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.load(
        repo_root / "configs/training/predictor_v8_structured.yaml"
    )
    model_cfg = OmegaConf.load(repo_root / cfg.model.config)
    sh_cfg = OmegaConf.merge(
        model_cfg.get("structured_head", {}),
        cfg.model.get("structured_head", {}),
    )
    sh_enabled = bool(sh_cfg.get("enabled", False))
    assert sh_enabled, \
        "predictor_v8_structured.yaml must enable structured_head; got False"

    # Build the loss with the same args train_predictor.py uses.
    crit = PredictorLoss(
        contact_weight=cfg.loss.contact_weight,
        target_weight=cfg.loss.target_weight,
        phase_weight=cfg.loss.phase_weight,
        support_weight=cfg.loss.support_weight,
        target_loss_kind=cfg.loss.get("target_loss_kind", "smooth_l1"),
        target_kernel_sigma=float(cfg.loss.get("target_kernel_sigma", 0.08)),
        consistency_weight=float(cfg.loss.get("consistency_weight", 0.0)),
    )
    assert crit.target_loss_kind == "kl_div", \
        f"v8 yaml must request kl_div loss; got {crit.target_loss_kind!r}"

    # Build the predictor with the merged config (matches train_predictor.py).
    pred_model = InteractionPredictor(
        d_model=int(model_cfg.encoder.d_model),
        num_layers=2,  # smaller for unit-test speed
        num_heads=int(model_cfg.encoder.num_heads),
        dim_feedforward=int(model_cfg.encoder.dim_feedforward),
        max_seq_length=8,
        num_body_parts=int(model_cfg.output.num_body_parts),
        num_phases=int(model_cfg.output.num_phases),
        num_support_states=int(model_cfg.output.num_support_states),
        structured_head=sh_enabled,
        structured_head_d_emb=int(sh_cfg.get("d_emb", 64)),
        structured_head_hidden=int(sh_cfg.get("hidden", 256)),
        structured_head_attn_heads=int(sh_cfg.get("attn_heads", 6)),
    )
    pred_model.train()
    inp = _make_inputs(T=8, d_model=int(model_cfg.encoder.d_model))
    out = pred_model(
        inp["text_tokens"], inp["object_tokens"], inp["init_pose"],
        seq_length=8, object_xyz=inp["object_xyz"],
        gt_contact=inp["gt_contact"], gt_phase=inp["gt_phase"],
        teacher_forcing=False,
    )
    # Predictor must emit attn so the loss can compute KL.
    assert "contact_target_attn" in out, \
        "v8 yaml's structured_head=True must produce contact_target_attn"

    # Full loss compute — this is the path that crashed at server startup.
    loss_dict = crit(
        out,
        gt_contact=inp["gt_contact"],
        gt_target=inp["gt_target"],
        gt_phase=inp["gt_phase"].long(),
        gt_support=inp["gt_support"].long(),
        mask=torch.ones(2, 8, dtype=torch.bool),
        object_xyz=inp["object_xyz"],
    )
    assert torch.isfinite(loss_dict["loss"])
    print("[PASS] test_v8_config_yaml_end_to_end_compat")


if __name__ == "__main__":
    test_structured_head_forward_shape()
    test_predictor_legacy_forward_unchanged()
    test_predictor_structured_forward()
    test_teacher_forcing_toggle()
    test_kl_target_loss()
    test_consistency_loss_zero_when_satisfied()
    test_full_v8_loss_backward()
    test_v7fix_legacy_loss_unchanged()
    test_object_encoder_return_xyz()
    test_v8_config_yaml_end_to_end_compat()
    print("\nAll v8 sanity tests passed.")
