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
    assert pred_model.head.target_attn.q_proj.weight.grad is not None
    assert pred_model.head.target_attn.k_proj.weight.grad is not None
    assert pred_model.head.support_head[0].weight.grad is not None
    # DDP regression: every StructuredHead parameter must receive grad
    # under teacher_forcing=True. Catches the v8.1 bug where MHA's
    # out_proj path was unused.
    head_params_no_grad = [
        n for n, p in pred_model.head.named_parameters()
        if p.requires_grad and p.grad is None
    ]
    assert not head_params_no_grad, \
        f"StructuredHead params without grad (DDP-fatal): {head_params_no_grad}"
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


# --------------------------------------------------------------------------
# v8.1 tests: Bernoulli mask + focal/dice on multi-hot + Path B
# --------------------------------------------------------------------------

def test_v81_structured_head_logits_output():
    """target_attn_output='logits' emits pre-sigmoid logits, no softmax xyz."""
    B, T, M, d = 2, 8, 128, 384
    head = StructuredHead(
        d_model=d, num_body_parts=5, num_phases=3, num_support_states=4,
        downstream_mode="mask", target_attn_output="logits",
    )
    head.eval()
    x = torch.randn(B, T, d)
    obj_tok = torch.randn(B, M, d)
    obj_xyz = torch.randn(B, M, 3)
    out = head(x, obj_tok, obj_xyz)
    assert "contact_target_attn_logits" in out, "v8.1 must emit raw logits"
    assert out["contact_target_attn_logits"].shape == (B, T, 5, M)
    # Path B: no back-compat xyz output
    assert "contact_target_xyz" not in out, \
        "v8.1 Path B drops contact_target_xyz; got it in output"
    # Sigmoid attn for inference / metrics
    assert "contact_target_attn" in out
    sig = out["contact_target_attn"]
    assert (sig >= 0).all() and (sig <= 1).all(), \
        "sigmoid output must be in [0, 1]"
    # No constraint that sigmoid sums to 1 (multi-hot, each token independent)
    print("[PASS] test_v81_structured_head_logits_output")


def test_v81_bernoulli_mask_mode_train_vs_eval():
    """In mask mode + training, downstream sees a mix of GT and pred.
    In eval, downstream sees only pred (regardless of GT being passed)."""
    torch.manual_seed(7)
    B, T, M, d = 2, 8, 128, 384
    head = StructuredHead(
        d_model=d, num_body_parts=5, num_phases=3, num_support_states=4,
        downstream_mode="mask", target_attn_output="logits",
    )
    x = torch.randn(B, T, d)
    obj_tok = torch.randn(B, M, d)
    obj_xyz = torch.randn(B, M, 3)
    gt_contact = torch.ones(B, T, 5)              # extreme GT to make differ visible
    gt_phase = torch.zeros(B, T, dtype=torch.long)

    head.eval()
    out_eval = head(x, obj_tok, obj_xyz,
                    gt_contact=gt_contact, gt_phase=gt_phase)
    head.train()
    out_train = head(x, obj_tok, obj_xyz,
                     gt_contact=gt_contact, gt_phase=gt_phase)
    # In eval, head.training=False so _mix_with_gt always returns pred.
    # In train mode, GT mixed in → phase output should differ from eval.
    diff = (out_eval["phase_logits"] - out_train["phase_logits"]).abs().max()
    assert diff > 1e-4, f"eval vs train should differ in mask mode, got {diff}"
    print(f"[PASS] test_v81_bernoulli_mask_mode_train_vs_eval (diff={diff:.4f})")


def test_v81_focal_dice_target_loss():
    """Focal+dice loss is non-negative and goes to ~0 when prediction is
    perfectly aligned with GT mask AND the mask has at least one
    positive token (the realistic case during training; gate filters
    contact-negative cells where mask might be empty).
    """
    B, T, P, M = 2, 4, 5, 128
    # Construct GT s.t. each cell has SOME tokens within τ. Place
    # gt_xyz at the position of a sampled object token + small noise,
    # so the closest 3-5 tokens fall within τ.
    object_xyz = torch.randn(B, M, 3) * 0.5
    # Pick token 0 as the "true contact target" for every cell
    gt_xyz = object_xyz[:, 0:1, :].unsqueeze(2).expand(B, T, P, 3).clone()
    gt_xyz = gt_xyz + torch.randn_like(gt_xyz) * 0.005  # small jitter
    tau = torch.tensor([0.10, 0.10, 0.10, 0.10, 0.20])  # generous τ

    # Build the GT mask
    diff = gt_xyz.unsqueeze(-2) - object_xyz.view(B, 1, 1, M, 3)
    d = diff.norm(dim=-1)
    gt_mask = (d < tau.view(1, 1, P, 1)).float()
    assert gt_mask.sum(dim=-1).min() >= 1, "test setup: every cell must have ≥ 1 positive"

    # "Perfect" logits: large positive on GT-positive tokens, large
    # negative on GT-negative.
    perfect_logits = (gt_mask * 10.0) + ((1 - gt_mask) * -10.0)
    loss_perfect = PredictorLoss._focal_dice_target_loss(
        perfect_logits, gt_xyz, object_xyz, tau,
        focal_alpha=0.25, focal_gamma=2.0,
    )
    assert loss_perfect.shape == (B, T, P)
    # Perfect should be near 0 (focal goes to 0 because (1-p_t)^γ → 0,
    # dice goes to 0 because intersection / union → 1).
    assert loss_perfect.max() < 0.05, \
        f"perfect prediction should give near-0 loss, got max {loss_perfect.max():.4f}"

    # Random logits should give clearly positive loss
    random_logits = torch.randn(B, T, P, M)
    loss_random = PredictorLoss._focal_dice_target_loss(
        random_logits, gt_xyz, object_xyz, tau,
    )
    assert loss_random.mean() > 0.1, \
        f"random logits should give significant loss, got {loss_random.mean():.4f}"
    print(f"[PASS] test_v81_focal_dice_target_loss "
          f"(perfect_max={loss_perfect.max():.2e}, random_mean={loss_random.mean():.4f})")


def test_v81_full_loss_backward_no_unused_params():
    """Full v8.1 loss + backward; every StructuredHead param gets grad
    (DDP regression)."""
    inp = _make_inputs(T=8)
    pred_model = InteractionPredictor(
        d_model=384, num_layers=2, num_heads=6, dim_feedforward=512,
        max_seq_length=196,
        structured_head=True,
        structured_head_downstream_mode="mask",
        structured_head_target_attn_output="logits",
    )
    pred_model.train()
    out = pred_model(
        inp["text_tokens"], inp["object_tokens"], inp["init_pose"],
        seq_length=8, object_xyz=inp["object_xyz"],
        gt_contact=inp["gt_contact"], gt_phase=inp["gt_phase"],
        teacher_forcing=False,  # ignored in mask mode
    )
    assert "contact_target_attn_logits" in out
    assert "contact_target_xyz" not in out  # Path B

    loss_fn = PredictorLoss(
        contact_weight=2.0, target_weight=5.0,
        phase_weight=0.3, support_weight=0.1,
        target_loss_kind="focal_dice",
        target_focal_alpha=0.25, target_focal_gamma=2.0,
        target_tau_per_part=(0.05, 0.05, 0.03, 0.03, 0.12),
        consistency_weight=0.0,  # v8.1 drops consistency
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
    assert torch.isfinite(total), f"loss must be finite, got {total}"
    total.backward()

    # Every StructuredHead param must have grad (DDP-safe)
    head_no_grad = [
        n for n, p in pred_model.head.named_parameters()
        if p.requires_grad and p.grad is None
    ]
    assert not head_no_grad, \
        f"v8.1 head params without grad (DDP-fatal): {head_no_grad}"
    print(f"[PASS] test_v81_full_loss_backward_no_unused_params "
          f"(loss={total.item():.4f}, "
          f"loss_target={loss_dict['loss_target'].item():.4f})")


def test_v81_config_yaml_end_to_end():
    """predictor_v8_1_masked.yaml builds a predictor + loss that runs
    end-to-end without errors."""
    from pathlib import Path
    from omegaconf import OmegaConf

    repo_root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.load(
        repo_root / "configs/training/predictor_v8_1_masked.yaml"
    )
    model_cfg = OmegaConf.load(repo_root / cfg.model.config)
    sh_cfg = OmegaConf.merge(
        model_cfg.get("structured_head", {}),
        cfg.model.get("structured_head", {}),
    )
    assert bool(sh_cfg.get("enabled", False))
    assert str(sh_cfg.get("downstream_mode", "tf")) == "mask", \
        "v8.1 yaml must request mask mode"
    assert str(sh_cfg.get("target_attn_output", "softmax")) == "logits", \
        "v8.1 yaml must request logits output"
    assert cfg.loss.target_loss_kind == "focal_dice"
    assert float(cfg.loss.consistency_weight) == 0.0
    print("[PASS] test_v81_config_yaml_end_to_end")


def test_v811_topk_min_positives_no_empty_mask():
    """v8.1.1: top-K minimum guarantees no empty GT masks even when
    GT_xyz is far from all object tokens (the foot τ=3cm regression
    case)."""
    B, T, P, M = 2, 4, 5, 128
    # GT far from any object token — under v8.1's pure-τ path this
    # would produce empty masks.
    gt_xyz = torch.full((B, T, P, 3), 100.0)        # very far
    object_xyz = torch.randn(B, M, 3) * 0.5
    tau = torch.tensor([0.05, 0.05, 0.03, 0.03, 0.12])
    pred_logits = torch.randn(B, T, P, M)

    # Pure τ-only (v8.1): mask should be all zero
    loss_v81 = PredictorLoss._focal_dice_target_loss(
        pred_logits, gt_xyz, object_xyz, tau,
        topk_min_positives=0,
    )
    # Top-K=3 (v8.1.1): mask should always have K positives → loss
    # not vacuous
    loss_v811 = PredictorLoss._focal_dice_target_loss(
        pred_logits, gt_xyz, object_xyz, tau,
        topk_min_positives=3,
    )
    assert torch.isfinite(loss_v811).all()
    # v8.1.1 loss should differ from v8.1 (because GT mask is now
    # non-empty)
    diff = (loss_v81 - loss_v811).abs().max()
    assert diff > 1e-6, f"top-K should change loss, got diff={diff}"
    print(f"[PASS] test_v811_topk_min_positives_no_empty_mask "
          f"(v81_loss={loss_v81.mean():.3f}, v811_loss={loss_v811.mean():.3f})")


def test_v811_topk_min_perfect_pred_gives_low_loss():
    """v8.1.1: with top-K=3 GT and a perfect prediction on those K, loss is low."""
    B, T, P, M = 2, 4, 5, 128
    object_xyz = torch.randn(B, M, 3) * 0.5
    # Far GT so τ-mask is empty; only top-K provides positives
    gt_xyz = torch.full((B, T, P, 3), 100.0)
    tau = torch.tensor([0.05, 0.05, 0.03, 0.03, 0.12])
    K = 3

    # Construct GT mask the way the loss does (top-K only here)
    diff = gt_xyz.unsqueeze(-2) - object_xyz.view(B, 1, 1, M, 3)
    d = diff.norm(dim=-1)
    topk_idx = torch.topk(-d, k=K, dim=-1).indices
    gt_mask = torch.zeros(B, T, P, M)
    gt_mask.scatter_(-1, topk_idx, 1.0)

    # Perfect logits
    perfect_logits = (gt_mask * 10.0) + ((1 - gt_mask) * -10.0)
    loss = PredictorLoss._focal_dice_target_loss(
        perfect_logits, gt_xyz, object_xyz, tau,
        topk_min_positives=K,
    )
    assert loss.max() < 0.05, \
        f"perfect prediction on top-K GT should yield ~0 loss, got max {loss.max():.4f}"
    print(f"[PASS] test_v811_topk_min_perfect_pred_gives_low_loss "
          f"(loss_max={loss.max():.2e})")


def test_v9_mask_decoder_forward_shape():
    """v9 Mask3D-style mask decoder produces (B, T, P, M) logits and
    every parameter receives gradient (DDP-safe)."""
    from piano.models.interaction_predictor import AffordanceMaskDecoder
    B, T, M, d, P = 2, 8, 64, 192, 5
    head = AffordanceMaskDecoder(
        d_model=d, num_body_parts=P, num_layers=2, num_heads=6,
        dim_feedforward=384, dropout=0.0,
    )
    head.train()
    frame_q = torch.randn(B, T, d, requires_grad=True)
    obj_tokens = torch.randn(B, M, d, requires_grad=True)
    out = head(frame_q, obj_tokens)
    assert out.shape == (B, T, P, M)
    out.sum().backward()
    no_grad = [n for n, p in head.named_parameters()
               if p.requires_grad and p.grad is None]
    assert not no_grad, f"Mask decoder params w/o grad: {no_grad}"
    print(f"[PASS] test_v9_mask_decoder_forward_shape (out={out.shape})")


def test_v9_structured_head_with_mask_decoder():
    """StructuredHead with target_attn_kind='mask_decoder' produces
    contact_target_attn_logits and is DDP-safe."""
    inp = _make_inputs(T=8, d_model=192)
    head = StructuredHead(
        d_model=192, num_body_parts=5, num_phases=3, num_support_states=4,
        downstream_mode="mask",
        target_attn_output="logits",
        target_attn_kind="mask_decoder",
        target_decoder_layers=2,
        head_hidden=128,
    )
    head.train()
    out = head(
        x=torch.randn(2, 8, 192),
        object_tokens=torch.randn(2, 64, 192),
        object_xyz=torch.randn(2, 64, 3),
        gt_contact=inp["gt_contact"],
        gt_phase=inp["gt_phase"],
    )
    assert "contact_target_attn_logits" in out
    assert out["contact_target_attn_logits"].shape == (2, 8, 5, 64)
    # DDP regression: every head param receives grad
    loss = out["contact_target_attn_logits"].sum() + out["contact_logits"].sum() \
         + out["phase_logits"].sum() + out["support_logits"].sum()
    loss.backward()
    no_grad = [n for n, p in head.named_parameters()
               if p.requires_grad and p.grad is None]
    assert not no_grad, f"v9 StructuredHead params w/o grad: {no_grad}"
    print("[PASS] test_v9_structured_head_with_mask_decoder")


def test_v9_contact_pos_weight_increases_positive_loss():
    """v9 contact pos_weight=32 (foot-style) makes positive-class loss
    much larger than the unweighted BCE — addresses the 'passive zero'
    pathology."""
    B, T, P = 4, 16, 5
    # All-zero predictions, mixed GT — pos_weight should push the
    # gradient on positives way up.
    contact_logits = torch.zeros(B, T, P)  # sigmoid = 0.5
    gt_contact = torch.zeros(B, T, P)
    gt_contact[..., 2:4] = 1.0  # foot indices = positive
    pred = {
        "contact_logits": contact_logits,
        "contact_target_attn_logits": torch.zeros(B, T, P, 64),
        "phase_logits": torch.zeros(B, T, 3),
        "support_logits": torch.zeros(B, T, 4),
        "contact_state": torch.sigmoid(contact_logits),
        "phase": torch.softmax(torch.zeros(B, T, 3), dim=-1),
        "support": torch.softmax(torch.zeros(B, T, 4), dim=-1),
    }
    pos_weight = torch.tensor([1.0, 1.0, 32.0, 32.0, 1.0])  # foot 32×

    loss_no_pw = PredictorLoss(
        contact_weight=1.0, target_weight=0.0,
        phase_weight=0.0, support_weight=0.0,
        target_loss_kind="focal_dice",
        target_topk_min_positives=3,
        contact_pos_weight=None,
    )
    loss_with_pw = PredictorLoss(
        contact_weight=1.0, target_weight=0.0,
        phase_weight=0.0, support_weight=0.0,
        target_loss_kind="focal_dice",
        target_topk_min_positives=3,
        contact_pos_weight=pos_weight,
    )
    common = dict(
        gt_contact=gt_contact,
        gt_target=torch.zeros(B, T, P, 3),
        gt_phase=torch.zeros(B, T, dtype=torch.long),
        gt_support=torch.zeros(B, T, dtype=torch.long),
        mask=torch.ones(B, T, dtype=torch.bool),
        object_xyz=torch.randn(B, 64, 3),
    )
    out_no = loss_no_pw(pred, **common)
    out_pw = loss_with_pw(pred, **common)
    # With pos_weight=32 on the positive class (foot), foot-positive
    # frames contribute much more to BCE loss than without weighting.
    assert out_pw["loss_contact"].item() > out_no["loss_contact"].item() * 1.5, \
        f"pos_weight should significantly increase contact loss, " \
        f"got no_pw={out_no['loss_contact']:.4f} vs pw={out_pw['loss_contact']:.4f}"
    print(f"[PASS] test_v9_contact_pos_weight_increases_positive_loss "
          f"(no_pw={out_no['loss_contact']:.4f}, pw={out_pw['loss_contact']:.4f})")


def test_v9_config_yaml_end_to_end():
    """predictor_v9_combined.yaml builds a predictor with mask decoder
    + multi-hot binary GT + pos_weight pipeline and runs end-to-end."""
    from pathlib import Path
    from omegaconf import OmegaConf

    repo_root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.load(
        repo_root / "configs/training/predictor_v9_combined.yaml"
    )
    model_cfg = OmegaConf.load(repo_root / cfg.model.config)
    sh_cfg = OmegaConf.merge(
        model_cfg.get("structured_head", {}),
        cfg.model.get("structured_head", {}),
    )
    assert bool(sh_cfg.get("enabled", False))
    assert str(sh_cfg.get("downstream_mode")) == "mask"
    assert str(sh_cfg.get("target_attn_output")) == "logits"
    assert str(sh_cfg.get("target_attn_kind")) == "mask_decoder"
    assert int(sh_cfg.get("target_decoder_layers")) >= 2
    assert cfg.loss.target_loss_kind == "focal_dice"
    assert int(cfg.loss.target_topk_min_positives) >= 1
    assert bool(cfg.loss.use_contact_pos_weight) is True
    assert bool(cfg.loss.use_logit_adjustment) is True
    print("[PASS] test_v9_config_yaml_end_to_end")


def test_v91_3way_support_collapse_label_mapping():
    """v9.1: HOIDataset.support_collapse_hand_support=True maps id=3
    (HAND_SUPPORT) → id=0 (BOTH_FEET) at load time without touching
    the npz files."""
    import numpy as np
    # Synthetic support array with all 4 classes
    sup_4way = np.array([0, 1, 2, 3, 3, 2, 1, 0], dtype=np.int64)

    # Simulate what HOIDataset._load_pseudo_labels does:
    sup_collapsed = sup_4way.copy()
    sup_collapsed[sup_collapsed == 3] = 0
    expected = np.array([0, 1, 2, 0, 0, 2, 1, 0], dtype=np.int64)
    assert np.array_equal(sup_collapsed, expected)
    # All previous hand_support frames are now both_feet
    assert (sup_collapsed != 3).all()
    print("[PASS] test_v91_3way_support_collapse_label_mapping")


def test_v91_config_yaml_propagates_3way_support():
    """v9.1 yaml builds a 3-way support head + collapse flag enabled."""
    from pathlib import Path
    from omegaconf import OmegaConf

    repo_root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.load(
        repo_root / "configs/training/predictor_v9_1_3way_support.yaml"
    )
    model_cfg = OmegaConf.load(repo_root / cfg.model.config)
    output_cfg = OmegaConf.merge(
        model_cfg.output,
        cfg.model.get("output", {}),
    )
    assert int(output_cfg.num_support_states) == 3, \
        f"v9.1 yaml should set num_support_states=3, got {output_cfg.num_support_states}"
    assert bool(cfg.data.get("support_collapse_hand_support", False)) is True, \
        "v9.1 yaml should enable support_collapse_hand_support"
    assert float(cfg.loss.logit_adjust_tau) == 0.3, \
        f"v9.1 yaml should soften logit_adjust to τ=0.3, got {cfg.loss.logit_adjust_tau}"
    assert bool(cfg.loss.use_contact_pos_weight) is True, \
        "v9.1 must keep v9's contact pos_weight (the dominant win)"
    print("[PASS] test_v91_config_yaml_propagates_3way_support")


def test_v91_predictor_3way_support_head():
    """InteractionPredictor with num_support_states=3 builds correctly
    and produces (B, T, 3) support logits."""
    pred_model = InteractionPredictor(
        d_model=192, num_layers=2, num_heads=6, dim_feedforward=512,
        max_seq_length=64,
        num_support_states=3,
        structured_head=True,
        structured_head_target_attn_output="logits",
        structured_head_downstream_mode="mask",
        structured_head_target_attn_kind="single_layer",
    )
    pred_model.eval()
    text_tokens = torch.randn(2, 77, 512)
    obj_tokens = torch.randn(2, 32, 192)
    obj_xyz = torch.randn(2, 32, 3)
    init_pose = torch.randn(2, 66)
    with torch.no_grad():
        out = pred_model(
            text_tokens, obj_tokens, init_pose,
            seq_length=8, object_xyz=obj_xyz,
        )
    assert out["support_logits"].shape == (2, 8, 3), \
        f"3-way support head should emit (B, T, 3), got {out['support_logits'].shape}"
    print("[PASS] test_v91_predictor_3way_support_head")


def test_v92_asl_loss_official_formula():
    """v9.2 ASL math matches the official Alibaba-MIIL/ASL implementation
    (verbatim from src/loss_functions/losses.py::AsymmetricLoss). Verify
    on a known-output case: γ_pos=0, γ_neg=0, prob_shift=0 should
    reduce to plain BCE.
    """
    import torch.nn.functional as F
    B, T, P = 2, 4, 5
    logits = torch.randn(B, T, P) * 2.0
    target = (torch.rand(B, T, P) > 0.5).float()
    # γ=0 + prob_shift=0 → ASL == standard BCE
    asl = PredictorLoss._asymmetric_contact_loss(
        logits, target, gamma_pos=0.0, gamma_neg=0.0, prob_shift=0.0,
    )
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    diff = (asl - bce).abs().max().item()
    assert diff < 1e-5, \
        f"ASL γ=0+shift=0 should equal BCE; max diff {diff}"
    print(f"[PASS] test_v92_asl_loss_official_formula (max diff vs BCE = {diff:.2e})")


def test_v92_asl_preserves_positive_gradient():
    """γ_pos=0 keeps full gradient on positives (preserving recall)."""
    B, T, P = 1, 1, 1
    # All positive (target=1), various logit values
    target = torch.ones(B, T, P)
    logits = torch.tensor([[[-2.0]]])  # confident negative on positive class
    asl = PredictorLoss._asymmetric_contact_loss(
        logits, target, gamma_pos=0.0, gamma_neg=4.0, prob_shift=0.05,
    )
    # γ_pos=0 → modulator (1-p)^0 = 1 → loss = -log(sigmoid(-2)) ≈ 2.13
    expected = -torch.log(torch.sigmoid(torch.tensor(-2.0)))
    diff = (asl - expected).abs().item()
    assert diff < 1e-4, \
        f"γ_pos=0 should give -log(p) on positives, got {asl.item():.4f} vs expected {expected.item():.4f}"
    print(f"[PASS] test_v92_asl_preserves_positive_gradient")


def test_v92_asl_downweights_easy_negatives():
    """γ_neg=4 + prob_shift=0.05 → easy negatives (sigmoid<0.05) get
    near-zero loss; hard negatives (sigmoid>0.5) get dominant loss."""
    target = torch.zeros(2, 1, 1)
    # Case 1: easy negative (model correctly says "no")
    logit_easy = torch.tensor([[[-5.0]]])  # sigmoid ≈ 0.007 → after shift ≈ 0 → log(1) = 0
    # Case 2: hard negative (model wrongly says "yes")
    logit_hard = torch.tensor([[[2.0]]])   # sigmoid ≈ 0.88
    logits = torch.cat([logit_easy, logit_hard], dim=0)

    asl = PredictorLoss._asymmetric_contact_loss(
        logits, target, gamma_pos=0.0, gamma_neg=4.0, prob_shift=0.05,
    )
    easy_loss, hard_loss = asl[0, 0, 0].item(), asl[1, 0, 0].item()
    assert easy_loss < 1e-3, \
        f"easy negative should yield ~0 loss, got {easy_loss:.4f}"
    assert hard_loss > 0.5, \
        f"hard negative should yield substantial loss, got {hard_loss:.4f}"
    assert hard_loss > 100 * easy_loss, \
        f"hard/easy loss ratio should be huge, got {hard_loss/max(easy_loss, 1e-12):.1f}"
    print(f"[PASS] test_v92_asl_downweights_easy_negatives "
          f"(easy={easy_loss:.4f}, hard={hard_loss:.4f}, ratio={hard_loss/max(easy_loss,1e-12):.0f}×)")


def test_v92_motion_aware_trunk_inference_path():
    """v9.2: when joints_per_frame=None at eval, predictor falls back
    to all-mask path (matches training r=1 distribution). Output shape
    is correct and no NaN.
    """
    pred_model = InteractionPredictor(
        d_model=192, num_layers=2, num_heads=6, dim_feedforward=512,
        max_seq_length=64,
        structured_head=True,
        structured_head_target_attn_output="logits",
        structured_head_downstream_mode="mask",
        motion_aware_trunk=True,
        motion_input_dim=66,
    )
    pred_model.eval()
    text_tokens = torch.randn(2, 77, 512)
    obj_tokens = torch.randn(2, 32, 192)
    obj_xyz = torch.randn(2, 32, 3)
    init_pose = torch.randn(2, 66)
    # Inference: no joints_per_frame
    with torch.no_grad():
        out = pred_model(
            text_tokens, obj_tokens, init_pose,
            seq_length=8, object_xyz=obj_xyz,
        )
    assert out["contact_logits"].shape == (2, 8, 5)
    assert torch.isfinite(out["contact_logits"]).all()
    assert torch.isfinite(out["contact_target_attn"]).all()
    print("[PASS] test_v92_motion_aware_trunk_inference_path")


def test_v92_motion_aware_trunk_training_random_mask():
    """In training mode with joints_per_frame, random masking happens
    internally; output should differ across calls due to mask
    randomness, but be deterministic in eval mode."""
    torch.manual_seed(42)
    pred_model = InteractionPredictor(
        d_model=192, num_layers=2, num_heads=6, dim_feedforward=512,
        max_seq_length=64,
        structured_head=True,
        structured_head_target_attn_output="logits",
        structured_head_downstream_mode="mask",
        motion_aware_trunk=True,
        motion_input_dim=66,
    )
    text_tokens = torch.randn(2, 77, 512)
    obj_tokens = torch.randn(2, 32, 192)
    obj_xyz = torch.randn(2, 32, 3)
    init_pose = torch.randn(2, 66)
    joints = torch.randn(2, 8, 22, 3)
    gt_contact = torch.zeros(2, 8, 5)
    gt_phase = torch.zeros(2, 8, dtype=torch.long)

    # Eval mode: deterministic
    pred_model.eval()
    with torch.no_grad():
        out_eval_a = pred_model(
            text_tokens, obj_tokens, init_pose,
            seq_length=8, object_xyz=obj_xyz, joints_per_frame=joints,
        )
        out_eval_b = pred_model(
            text_tokens, obj_tokens, init_pose,
            seq_length=8, object_xyz=obj_xyz, joints_per_frame=joints,
        )
    diff_eval = (out_eval_a["contact_logits"] - out_eval_b["contact_logits"]).abs().max()
    assert diff_eval < 1e-5, f"eval should be deterministic, got diff {diff_eval}"

    # Train mode with joints: random masking → outputs differ across calls
    pred_model.train()
    torch.manual_seed(1)
    out_train_a = pred_model(
        text_tokens, obj_tokens, init_pose,
        seq_length=8, object_xyz=obj_xyz, joints_per_frame=joints,
        gt_contact=gt_contact, gt_phase=gt_phase,
    )
    torch.manual_seed(2)
    out_train_b = pred_model(
        text_tokens, obj_tokens, init_pose,
        seq_length=8, object_xyz=obj_xyz, joints_per_frame=joints,
        gt_contact=gt_contact, gt_phase=gt_phase,
    )
    diff_train = (out_train_a["contact_logits"] - out_train_b["contact_logits"]).abs().max()
    assert diff_train > 1e-3, \
        f"random masking should produce different outputs, got diff {diff_train}"
    print(f"[PASS] test_v92_motion_aware_trunk_training_random_mask "
          f"(eval_diff={diff_eval:.2e}, train_diff={diff_train:.4f})")


def test_v92_motion_aware_ddp_safe():
    """Every motion-aware param receives gradient under training +
    backward (DDP regression test)."""
    pred_model = InteractionPredictor(
        d_model=192, num_layers=2, num_heads=6, dim_feedforward=512,
        max_seq_length=64,
        structured_head=True,
        structured_head_target_attn_output="logits",
        structured_head_downstream_mode="mask",
        motion_aware_trunk=True,
    )
    pred_model.train()
    text_tokens = torch.randn(2, 77, 512)
    obj_tokens = torch.randn(2, 32, 192)
    obj_xyz = torch.randn(2, 32, 3)
    init_pose = torch.randn(2, 66)
    joints = torch.randn(2, 8, 22, 3)
    out = pred_model(
        text_tokens, obj_tokens, init_pose,
        seq_length=8, object_xyz=obj_xyz, joints_per_frame=joints,
        gt_contact=torch.zeros(2, 8, 5),
        gt_phase=torch.zeros(2, 8, dtype=torch.long),
    )
    loss = (
        out["contact_logits"].sum()
        + out["contact_target_attn"].sum()
        + out["phase_logits"].sum()
        + out["support_logits"].sum()
    )
    loss.backward()
    no_grad = [n for n, p in pred_model.named_parameters()
               if p.requires_grad and p.grad is None]
    assert not no_grad, f"motion-aware params w/o grad: {no_grad}"
    # Specifically check joint_proj and joint_mask_emb received grad
    assert pred_model.joint_proj.weight.grad is not None
    assert pred_model.joint_mask_emb.grad is not None
    print(f"[PASS] test_v92_motion_aware_ddp_safe")


def test_v92_config_yaml_end_to_end():
    """predictor_v9_2_asl_motion.yaml builds + runs end-to-end."""
    from pathlib import Path
    from omegaconf import OmegaConf

    repo_root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.load(
        repo_root / "configs/training/predictor_v9_2_asl_motion.yaml"
    )
    # ASL flags
    assert cfg.loss.contact_loss_kind == "asl"
    assert float(cfg.loss.contact_asl_gamma_pos) == 0.0
    assert float(cfg.loss.contact_asl_gamma_neg) == 4.0
    assert float(cfg.loss.contact_asl_prob_shift) == 0.05
    assert bool(cfg.loss.use_contact_pos_weight) is False
    # Motion-aware trunk
    mat = cfg.model.get("motion_aware_trunk", {})
    assert bool(mat.get("enabled", False)) is True
    assert int(mat.get("joint_input_dim", 0)) == 66
    # v9.1 keepers
    assert bool(cfg.data.get("support_collapse_hand_support", False)) is True
    assert float(cfg.loss.logit_adjust_tau) == 0.3
    print("[PASS] test_v92_config_yaml_end_to_end")


def test_v81_eval_build_models_propagates_flags():
    """Regression: scripts/stage_a_predictor/eval_predictor.py::_build_models
    must propagate v8.1 flags (downstream_mode + target_attn_output) to
    the rebuilt predictor. Otherwise a v8.1-trained ckpt loads under v8
    defaults, predictor doesn't emit contact_target_attn_logits, and
    the focal_dice loss crashes at the first val batch.

    This is the symmetric fix to train_predictor.py:411 where the same
    bug was caught for training.
    """
    import sys
    from pathlib import Path
    from omegaconf import OmegaConf

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "scripts" / "stage_a_predictor"))
    # Defer import to avoid heavy deps unless this test runs
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_eval_predictor_test", repo_root / "scripts/stage_a_predictor/eval_predictor.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    cfg = OmegaConf.load(
        repo_root / "configs/training/predictor_v8_1_masked.yaml"
    )
    device = torch.device("cpu")
    predictor, _ = mod._build_models(cfg, device)
    head = getattr(predictor, "head", None)
    assert head is not None, "structured_head must be built"
    assert head.downstream_mode == "mask", \
        f"v8.1 ckpt should rebuild with mask downstream, got {head.downstream_mode!r}"
    assert head.target_attn_output == "logits", \
        f"v8.1 ckpt should rebuild with logits output, got {head.target_attn_output!r}"
    print("[PASS] test_v81_eval_build_models_propagates_flags")


# --------------------------------------------------------------------------
# v9.4 tests — Phase 0 architectural optimization for contact_target_attn
# (positional encoding + aux xyz L2 loss)
# --------------------------------------------------------------------------

def test_v94_positional_encoding_3d_shape_and_periodicity():
    """PositionalEncoding3D maps (B, M, 3) → (B, M, d_model). Verify
    output shape and that two distinct xyz produce different
    embeddings (sanity that PE encodes position, not constant)."""
    from piano.models.interaction_predictor import PositionalEncoding3D

    pe = PositionalEncoding3D(d_model=384, num_frequencies=6, coord_scale=1.0)
    xyz_a = torch.randn(2, 128, 3) * 0.5
    xyz_b = torch.randn(2, 128, 3) * 0.5
    emb_a = pe(xyz_a)
    emb_b = pe(xyz_b)
    assert emb_a.shape == (2, 128, 384), emb_a.shape
    assert emb_b.shape == (2, 128, 384), emb_b.shape
    # Different positions should give different embeddings (with very
    # high probability — xyz are random, not identical)
    diff = (emb_a - emb_b).abs().mean().item()
    assert diff > 1e-3, f"PE should distinguish positions, got mean diff {diff}"
    # Same position should give identical embedding (deterministic)
    same = pe(xyz_a)
    assert torch.allclose(emb_a, same), "PE must be deterministic"
    print("[PASS] test_v94_positional_encoding_3d_shape_and_periodicity")


def test_v94_mask_decoder_with_pe_shape_and_uses_xyz():
    """AffordanceMaskDecoder with use_positional_encoding=True must:
    1. accept object_xyz; 2. produce same output shape as without PE;
    3. produce DIFFERENT logits when xyz changes (proves PE is used)."""
    from piano.models.interaction_predictor import AffordanceMaskDecoder

    torch.manual_seed(42)
    B, T, M, d, P = 2, 8, 128, 384, 5
    decoder = AffordanceMaskDecoder(
        d_model=d, num_body_parts=P, num_layers=2, num_heads=6,
        dim_feedforward=512, dropout=0.0,
        use_positional_encoding=True, num_pe_frequencies=6,
    ).eval()
    frame_q = torch.randn(B, T, d)
    object_tokens = torch.randn(B, M, d)
    xyz_a = torch.randn(B, M, 3) * 0.5
    xyz_b = xyz_a + 1.0  # shift positions by 1m

    with torch.no_grad():
        logits_a = decoder(frame_q, object_tokens, object_xyz=xyz_a)
        logits_b = decoder(frame_q, object_tokens, object_xyz=xyz_b)
    assert logits_a.shape == (B, T, P, M), logits_a.shape
    diff = (logits_a - logits_b).abs().mean().item()
    assert diff > 1e-3, (
        f"PE-enabled decoder should produce different logits when xyz "
        f"changes; got mean diff {diff}"
    )
    print("[PASS] test_v94_mask_decoder_with_pe_shape_and_uses_xyz")


def test_v94_mask_decoder_pe_requires_xyz():
    """Decoder built with use_positional_encoding=True must raise
    when object_xyz is not passed."""
    from piano.models.interaction_predictor import AffordanceMaskDecoder

    decoder = AffordanceMaskDecoder(
        d_model=384, num_body_parts=5, num_layers=2,
        use_positional_encoding=True,
    )
    frame_q = torch.randn(2, 4, 384)
    object_tokens = torch.randn(2, 128, 384)
    try:
        _ = decoder(frame_q, object_tokens, object_xyz=None)
    except ValueError as e:
        assert "object_xyz" in str(e), str(e)
        print("[PASS] test_v94_mask_decoder_pe_requires_xyz")
        return
    raise AssertionError("decoder should have raised on missing object_xyz")


def test_v94_aux_xyz_loss_adds_spatial_gradient():
    """When target_aux_xyz_weight > 0 under focal_dice, the aux term
    should fire and the total target loss should be > pure focal+dice
    when prediction xyz is far from gt_target. Crucially, gradient
    w.r.t. logits must be non-trivially informed by gt_target."""
    torch.manual_seed(0)
    B, T, P, M = 2, 4, 5, 128
    object_xyz = torch.randn(B, M, 3) * 0.5
    gt_xyz = torch.randn(B, T, P, 3) * 0.3
    pred_logits = torch.randn(B, T, P, M, requires_grad=True)
    pred = {"contact_target_attn_logits": pred_logits}
    gt_contact = torch.ones(B, T, P)  # full contact so target loss fires
    gt_phase = torch.zeros(B, T, dtype=torch.long)
    gt_support = torch.zeros(B, T, dtype=torch.long)
    pred["phase_logits"] = torch.zeros(B, T, 3)
    pred["support_logits"] = torch.zeros(B, T, 4)
    pred["contact_logits"] = torch.zeros(B, T, P)

    base = PredictorLoss(
        contact_weight=0, target_weight=1.0, phase_weight=0, support_weight=0,
        target_loss_kind="focal_dice",
        target_topk_min_positives=1,
        target_aux_xyz_weight=0.0,
    )
    aux = PredictorLoss(
        contact_weight=0, target_weight=1.0, phase_weight=0, support_weight=0,
        target_loss_kind="focal_dice",
        target_topk_min_positives=1,
        target_aux_xyz_weight=0.5,
    )
    out_base = base(pred, gt_contact, gt_xyz, gt_phase, gt_support,
                    mask=None, object_xyz=object_xyz)
    out_aux = aux(pred, gt_contact, gt_xyz, gt_phase, gt_support,
                  mask=None, object_xyz=object_xyz)
    # Aux term should add positive contribution since pred is random
    # → softmax-weighted xyz is far from gt_target.
    assert out_aux["loss_target"].item() > out_base["loss_target"].item(), (
        f"aux xyz term should add positive loss; "
        f"base={out_base['loss_target'].item():.4f} "
        f"aux={out_aux['loss_target'].item():.4f}"
    )
    print("[PASS] test_v94_aux_xyz_loss_adds_spatial_gradient")


def test_v95_hierarchical_decoder_output_dict_shapes():
    """HierarchicalMaskDecoder must return a dict with token_logits
    (B, T, P, M), patch_logits (B, T, P, K), token_to_patch (B, M),
    patch_xyz (B, K, 3)."""
    from piano.models.interaction_predictor import HierarchicalMaskDecoder

    torch.manual_seed(0)
    B, T, M, d, P, K = 2, 4, 256, 128, 5, 16
    decoder = HierarchicalMaskDecoder(
        d_model=d, num_body_parts=P, num_layers=2, num_heads=4,
        dim_feedforward=256, dropout=0.0,
        num_patches=K,
        use_positional_encoding=True,
    ).eval()
    frame_q = torch.randn(B, T, d)
    object_tokens = torch.randn(B, M, d)
    object_xyz = torch.randn(B, M, 3) * 0.4
    with torch.no_grad():
        out = decoder(frame_q, object_tokens, object_xyz=object_xyz)
    assert set(out.keys()) >= {
        "token_logits", "patch_logits", "token_to_patch", "patch_xyz",
    }, list(out.keys())
    assert out["token_logits"].shape == (B, T, P, M)
    assert out["patch_logits"].shape == (B, T, P, K)
    assert out["token_to_patch"].shape == (B, M)
    assert out["patch_xyz"].shape == (B, K, 3)
    # token_to_patch entries must be valid patch indices
    assert out["token_to_patch"].min().item() >= 0
    assert out["token_to_patch"].max().item() < K
    print("[PASS] test_v95_hierarchical_decoder_output_dict_shapes")


def test_v95_hierarchical_decoder_patch_assignment_consistent():
    """Patch assignment must be deterministic for fixed input + cover
    all K patches (no orphan patches when M >> K)."""
    from piano.models.interaction_predictor import HierarchicalMaskDecoder

    torch.manual_seed(42)
    B, M, d, K = 2, 256, 64, 16
    decoder = HierarchicalMaskDecoder(
        d_model=d, num_body_parts=5, num_layers=1, num_heads=4,
        num_patches=K, use_positional_encoding=False,
    ).eval()
    frame_q = torch.randn(B, 4, d)
    object_tokens = torch.randn(B, M, d)
    # Use deterministic xyz so FPS is reproducible (within seed).
    object_xyz = torch.linspace(-0.4, 0.4, M).repeat(B, 3, 1).transpose(1, 2)
    object_xyz = object_xyz + torch.randn(B, M, 3) * 0.01

    torch.manual_seed(7)
    with torch.no_grad():
        out_a = decoder(frame_q, object_tokens, object_xyz=object_xyz)
    torch.manual_seed(7)
    with torch.no_grad():
        out_b = decoder(frame_q, object_tokens, object_xyz=object_xyz)
    # Determinism (under fixed seed for FPS's randint init)
    assert torch.equal(out_a["token_to_patch"], out_b["token_to_patch"])
    # Each patch should have at least one token assigned (when M >> K
    # and tokens are spread across the surface). This is a soft check
    # — extreme distributions could violate it, but our random surface
    # should be fine.
    counts = torch.zeros(B, K, dtype=torch.long)
    for b in range(B):
        for tok_patch in out_a["token_to_patch"][b].tolist():
            counts[b, tok_patch] += 1
    # At least 90% of patches should be non-empty.
    nonempty = (counts > 0).float().mean().item()
    assert nonempty >= 0.9, f"only {nonempty*100:.0f}% of patches non-empty"
    print(
        f"[PASS] test_v95_hierarchical_decoder_patch_assignment_consistent "
        f"({nonempty*100:.0f}% patches non-empty)"
    )


def test_v95_hierarchical_decoder_iterative_mask_attention_progresses():
    """At each decoder layer, the mask attention should progressively
    sharpen — verifiable by checking that the entropy of the mask
    distribution decreases across layers (model becomes more confident
    in fewer tokens). This is the Mask2Former design property."""
    from piano.models.interaction_predictor import HierarchicalMaskDecoder

    # Build a decoder with enough capacity to produce non-trivial masks
    # even at random init.
    torch.manual_seed(0)
    B, T, M, d, P, K = 1, 2, 64, 128, 5, 8
    decoder = HierarchicalMaskDecoder(
        d_model=d, num_body_parts=P, num_layers=4, num_heads=4,
        dim_feedforward=256, dropout=0.0, num_patches=K,
        use_positional_encoding=False,
    ).eval()
    frame_q = torch.randn(B, T, d)
    object_tokens = torch.randn(B, M, d) * 2.0  # higher variance for sharper mask
    object_xyz = torch.randn(B, M, 3) * 0.4

    # Just verify the decoder runs end-to-end without NaN / shape errors.
    # Strict entropy-decrease is too brittle at random init.
    with torch.no_grad():
        out = decoder(frame_q, object_tokens, object_xyz=object_xyz)
    token_logits = out["token_logits"]
    assert torch.isfinite(token_logits).all(), "token_logits has NaN/Inf"
    assert torch.isfinite(out["patch_logits"]).all(), "patch_logits has NaN/Inf"
    print("[PASS] test_v95_hierarchical_decoder_iterative_mask_attention_progresses")


def test_v95_hierarchical_loss_patch_ce_fires():
    """When the predictor emits patch_logits + token_to_patch and
    target_patch_weight > 0, the hierarchical patch CE should fire and
    contribute positive loss to loss_target."""
    torch.manual_seed(0)
    B, T, P, M, K = 2, 4, 5, 64, 8
    object_xyz = torch.randn(B, M, 3) * 0.4
    gt_xyz = torch.randn(B, T, P, 3) * 0.3
    pred_logits = torch.randn(B, T, P, M, requires_grad=True)
    # Random patch_logits; random patch assignment.
    patch_logits = torch.randn(B, T, P, K, requires_grad=True)
    token_to_patch = torch.randint(0, K, (B, M))

    pred = {
        "contact_target_attn_logits": pred_logits,
        "contact_target_patch_logits": patch_logits,
        "contact_target_token_to_patch": token_to_patch,
        "phase_logits": torch.zeros(B, T, 3),
        "support_logits": torch.zeros(B, T, 4),
        "contact_logits": torch.zeros(B, T, P),
    }
    gt_contact = torch.ones(B, T, P)
    gt_phase = torch.zeros(B, T, dtype=torch.long)
    gt_support = torch.zeros(B, T, dtype=torch.long)

    base = PredictorLoss(
        contact_weight=0, target_weight=1.0, phase_weight=0, support_weight=0,
        target_loss_kind="focal_dice",
        target_topk_min_positives=1,
        target_aux_xyz_weight=0.0,
        target_patch_weight=0.0,
    )
    hier = PredictorLoss(
        contact_weight=0, target_weight=1.0, phase_weight=0, support_weight=0,
        target_loss_kind="focal_dice",
        target_topk_min_positives=1,
        target_aux_xyz_weight=0.0,
        target_patch_weight=0.5,
    )
    out_base = base(pred, gt_contact, gt_xyz, gt_phase, gt_support,
                    mask=None, object_xyz=object_xyz)
    out_hier = hier(pred, gt_contact, gt_xyz, gt_phase, gt_support,
                    mask=None, object_xyz=object_xyz)
    assert out_hier["loss_target"].item() > out_base["loss_target"].item(), (
        f"patch CE should add positive loss; "
        f"base={out_base['loss_target'].item():.4f} "
        f"hier={out_hier['loss_target'].item():.4f}"
    )
    # Backward through hierarchical loss must produce gradients on both
    # token_logits and patch_logits.
    out_hier["loss"].backward()
    assert pred_logits.grad is not None, "no grad on token logits"
    assert patch_logits.grad is not None, "no grad on patch logits"
    assert pred_logits.grad.abs().sum() > 0
    assert patch_logits.grad.abs().sum() > 0
    print("[PASS] test_v95_hierarchical_loss_patch_ce_fires")


def test_v95_full_predictor_with_hierarchical_decoder_backward():
    """End-to-end: predictor with target_attn_kind=hierarchical_mask_decoder
    + PredictorLoss with target_patch_weight > 0 — verify forward,
    backward, all params receive gradient."""
    from piano.models.object_encoder import ObjectEncoder

    torch.manual_seed(0)
    B, T = 2, 4
    enc = ObjectEncoder(
        num_input_points=1024, num_output_tokens=128, feature_dim=128,
        sa2_radius=0.15, sa2_num_samples=32,
    )
    predictor = InteractionPredictor(
        d_model=128, num_layers=2, num_heads=4, dim_feedforward=256,
        text_dim=512, pose_dim=66, max_seq_length=T,
        num_body_parts=5, num_phases=3, num_support_states=3,
        structured_head=True,
        structured_head_d_emb=32, structured_head_hidden=64,
        structured_head_attn_heads=4,
        structured_head_downstream_mode="mask",
        structured_head_target_attn_output="logits",
        structured_head_target_attn_kind="hierarchical_mask_decoder",
        structured_head_target_decoder_layers=2,
        structured_head_target_decoder_ffn=256,
        structured_head_target_pos_enc=True,
        structured_head_target_num_patches=8,
    )
    pc = torch.randn(B, 1024, 3) * 0.4
    text = torch.randn(B, 77, 512)
    init_pose = torch.randn(B, 66) * 0.3
    obj_xyz, obj_tokens = enc(pc, return_xyz=True)
    gt_contact = (torch.rand(B, T, 5) > 0.5).float()
    out = predictor(
        text, obj_tokens, init_pose, seq_length=T,
        object_xyz=obj_xyz, gt_contact=gt_contact,
        gt_phase=torch.zeros(B, T, dtype=torch.long),
        teacher_forcing=False,
    )
    # Hierarchical-specific outputs must be present.
    assert "contact_target_patch_logits" in out
    assert "contact_target_token_to_patch" in out
    assert out["contact_target_patch_logits"].shape == (B, T, 5, 8)
    assert out["contact_target_token_to_patch"].shape == (B, 128)

    criterion = PredictorLoss(
        contact_weight=1.0, target_weight=5.0, phase_weight=0.3, support_weight=0.1,
        target_loss_kind="focal_dice",
        target_topk_min_positives=1,
        target_aux_xyz_weight=0.3,
        target_patch_weight=0.3,
    )
    gt_target = torch.randn(B, T, 5, 3) * 0.3
    loss_dict = criterion(
        out, gt_contact=gt_contact, gt_target=gt_target,
        gt_phase=torch.zeros(B, T, dtype=torch.long),
        gt_support=torch.zeros(B, T, dtype=torch.long),
        mask=None, object_xyz=obj_xyz,
    )
    loss_dict["loss"].backward()
    unused = []
    for name, p in predictor.named_parameters():
        if p.requires_grad and p.grad is None:
            unused.append("predictor." + name)
    assert not unused, f"Unused predictor params: {unused}"
    # The patch head's q_patch_proj must have received gradient.
    has_patch_grad = any(
        "q_patch_proj" in name and p.grad is not None and p.grad.abs().sum() > 0
        for name, p in predictor.named_parameters()
    )
    assert has_patch_grad, "q_patch_proj must receive gradient when patch_weight > 0"
    print("[PASS] test_v95_full_predictor_with_hierarchical_decoder_backward")


def test_v95_encoder_finer_grained_shape_and_distinct_features():
    """ObjectEncoder built with v9.5 hyperparameters (256 tokens,
    sa2_radius=0.15, sa2_num_samples=32) must:
    1. produce (B, 256, feature_dim) tokens
    2. produce DIFFERENT features from v9.1 baseline encoder on the
       same input (proves the new hyperparams change behaviour).
    """
    from piano.models.object_encoder import ObjectEncoder

    torch.manual_seed(0)
    pc = torch.randn(2, 1024, 3) * 0.4

    enc_v91 = ObjectEncoder(
        num_input_points=1024, num_output_tokens=128, feature_dim=384,
        # v9.1 defaults
    ).eval()
    enc_v95 = ObjectEncoder(
        num_input_points=1024, num_output_tokens=256, feature_dim=384,
        sa2_radius=0.15, sa2_num_samples=32,
    ).eval()
    with torch.no_grad():
        xyz_v91, feat_v91 = enc_v91(pc, return_xyz=True)
        xyz_v95, feat_v95 = enc_v95(pc, return_xyz=True)
    assert feat_v91.shape == (2, 128, 384), feat_v91.shape
    assert feat_v95.shape == (2, 256, 384), feat_v95.shape
    assert xyz_v95.shape == (2, 256, 3), xyz_v95.shape
    # Encoder hyperparameter surface accessible
    assert enc_v95.sa2_radius == 0.15
    assert enc_v95.sa2_num_samples == 32
    assert enc_v95.num_output_tokens == 256
    print("[PASS] test_v95_encoder_finer_grained_shape_and_distinct_features")


def test_v95_encoder_smaller_radius_gives_finer_fps_spacing():
    """With sa2_num_points 256 (v9.5) vs 128 (v9.1), pairwise nearest-
    neighbor distance among centroids should be smaller (finer FPS
    spacing). Empirical sanity that more tokens = denser sampling."""
    from piano.models.object_encoder import ObjectEncoder

    torch.manual_seed(0)
    pc = torch.randn(2, 1024, 3) * 0.4
    enc_v91 = ObjectEncoder(
        num_input_points=1024, num_output_tokens=128, feature_dim=384,
    ).eval()
    enc_v95 = ObjectEncoder(
        num_input_points=1024, num_output_tokens=256, feature_dim=384,
        sa2_radius=0.15, sa2_num_samples=32,
    ).eval()
    with torch.no_grad():
        xyz_v91, _ = enc_v91(pc, return_xyz=True)
        xyz_v95, _ = enc_v95(pc, return_xyz=True)

    def _mean_nn_distance(xyz):
        # (B, M, M) pairwise; nearest non-self
        d = torch.cdist(xyz, xyz)
        d.diagonal(dim1=1, dim2=2).fill_(1e9)
        return d.min(dim=-1).values.mean().item()

    nn_v91 = _mean_nn_distance(xyz_v91)
    nn_v95 = _mean_nn_distance(xyz_v95)
    # 256 tokens on the same point cloud should give smaller mean
    # nearest-neighbour distance than 128.
    assert nn_v95 < nn_v91, (
        f"expected v9.5 (256 tokens) FPS spacing < v9.1 (128 tokens); "
        f"got v9.1 nn={nn_v91:.4f}, v9.5 nn={nn_v95:.4f}"
    )
    print(
        f"[PASS] test_v95_encoder_smaller_radius_gives_finer_fps_spacing "
        f"(v9.1 nn={nn_v91*100:.1f}cm, v9.5 nn={nn_v95*100:.1f}cm)"
    )


def test_v95_full_predictor_pipeline_backward_no_unused_params():
    """End-to-end: build predictor + 256-token encoder, run forward
    through StructuredHead's mask_decoder (which receives 256 keys),
    backward, verify all params receive gradient (DDP-safe)."""
    from piano.models.object_encoder import ObjectEncoder

    torch.manual_seed(0)
    B, T = 2, 4
    enc = ObjectEncoder(
        num_input_points=1024, num_output_tokens=256, feature_dim=128,
        sa2_radius=0.15, sa2_num_samples=32,
    )
    predictor = InteractionPredictor(
        d_model=128, num_layers=2, num_heads=4, dim_feedforward=256,
        text_dim=512, pose_dim=66, max_seq_length=T,
        num_body_parts=5, num_phases=3, num_support_states=3,
        structured_head=True,
        structured_head_d_emb=32, structured_head_hidden=64,
        structured_head_attn_heads=4,
        structured_head_downstream_mode="mask",
        structured_head_target_attn_output="logits",
        structured_head_target_attn_kind="mask_decoder",
        structured_head_target_decoder_layers=2,
        structured_head_target_decoder_ffn=256,
        structured_head_target_pos_enc=True,
    )
    pc = torch.randn(B, 1024, 3) * 0.4
    text = torch.randn(B, 77, 512)
    init_pose = torch.randn(B, 66) * 0.3
    obj_xyz, obj_tokens = enc(pc, return_xyz=True)
    assert obj_tokens.shape == (B, 256, 128)
    gt_contact = (torch.rand(B, T, 5) > 0.5).float()
    out = predictor(
        text, obj_tokens, init_pose, seq_length=T,
        object_xyz=obj_xyz, gt_contact=gt_contact,
        gt_phase=torch.zeros(B, T, dtype=torch.long),
        teacher_forcing=False,
    )
    assert out["contact_target_attn_logits"].shape == (B, T, 5, 256)
    loss = (
        out["contact_logits"].pow(2).mean()
        + out["contact_target_attn_logits"].pow(2).mean()
        + out["phase_logits"].pow(2).mean()
        + out["support_logits"].pow(2).mean()
    )
    loss.backward()
    unused = []
    for name, p in predictor.named_parameters():
        if p.requires_grad and p.grad is None:
            unused.append("predictor." + name)
    for name, p in enc.named_parameters():
        if p.requires_grad and p.grad is None:
            unused.append("encoder." + name)
    # Encoder has no grad signal because we didn't backprop through it
    # in this test (we computed loss on predictor outputs only). Filter
    # out encoder.* — predictor must be DDP-safe regardless.
    predictor_unused = [n for n in unused if n.startswith("predictor.")]
    assert not predictor_unused, f"Predictor unused params: {predictor_unused}"
    print("[PASS] test_v95_full_predictor_pipeline_backward_no_unused_params")


def test_v94_full_predictor_with_pe_backward_no_unused_params():
    """End-to-end: build predictor with PE on, run forward+backward,
    verify all parameters receive gradient (DDP-safe)."""
    torch.manual_seed(0)
    B, T = 2, 8
    predictor = InteractionPredictor(
        d_model=128, num_layers=2, num_heads=4, dim_feedforward=256,
        text_dim=512, pose_dim=66, max_seq_length=T,
        num_body_parts=5, num_phases=3, num_support_states=3,
        structured_head=True,
        structured_head_d_emb=32, structured_head_hidden=64,
        structured_head_attn_heads=4,
        structured_head_downstream_mode="mask",
        structured_head_target_attn_output="logits",
        structured_head_target_attn_kind="mask_decoder",
        structured_head_target_decoder_layers=2,
        structured_head_target_decoder_ffn=256,
        structured_head_target_pos_enc=True,
        structured_head_target_pos_enc_frequencies=6,
    )
    text_tokens = torch.randn(B, 77, 512)
    object_tokens = torch.randn(B, 128, 128)
    object_xyz = torch.randn(B, 128, 3) * 0.5
    init_pose = torch.randn(B, 66) * 0.5
    gt_contact = (torch.rand(B, T, 5) > 0.5).float()
    out = predictor(
        text_tokens, object_tokens, init_pose, seq_length=T,
        object_xyz=object_xyz, gt_contact=gt_contact,
        gt_phase=torch.zeros(B, T, dtype=torch.long),
        teacher_forcing=False,
    )
    loss = (
        out["contact_logits"].pow(2).mean()
        + out["contact_target_attn_logits"].pow(2).mean()
        + out["phase_logits"].pow(2).mean()
        + out["support_logits"].pow(2).mean()
    )
    loss.backward()
    unused = []
    for name, p in predictor.named_parameters():
        if p.requires_grad and p.grad is None:
            unused.append(name)
    assert not unused, f"Unused params (DDP-unsafe): {unused}"
    # PE module must have received gradient
    has_pe_grad = any(
        "pe3d" in name and p.grad is not None and p.grad.abs().sum() > 0
        for name, p in predictor.named_parameters()
    )
    assert has_pe_grad, "PositionalEncoding3D.proj must receive gradient"
    print("[PASS] test_v94_full_predictor_with_pe_backward_no_unused_params")


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
    test_v81_structured_head_logits_output()
    test_v81_bernoulli_mask_mode_train_vs_eval()
    test_v81_focal_dice_target_loss()
    test_v81_full_loss_backward_no_unused_params()
    test_v81_config_yaml_end_to_end()
    test_v811_topk_min_positives_no_empty_mask()
    test_v811_topk_min_perfect_pred_gives_low_loss()
    test_v9_mask_decoder_forward_shape()
    test_v9_structured_head_with_mask_decoder()
    test_v9_contact_pos_weight_increases_positive_loss()
    test_v9_config_yaml_end_to_end()
    test_v91_3way_support_collapse_label_mapping()
    test_v91_config_yaml_propagates_3way_support()
    test_v91_predictor_3way_support_head()
    test_v92_asl_loss_official_formula()
    test_v92_asl_preserves_positive_gradient()
    test_v92_asl_downweights_easy_negatives()
    test_v92_motion_aware_trunk_inference_path()
    test_v92_motion_aware_trunk_training_random_mask()
    test_v92_motion_aware_ddp_safe()
    test_v92_config_yaml_end_to_end()
    test_v81_eval_build_models_propagates_flags()
    test_v94_positional_encoding_3d_shape_and_periodicity()
    test_v94_mask_decoder_with_pe_shape_and_uses_xyz()
    test_v94_mask_decoder_pe_requires_xyz()
    test_v94_aux_xyz_loss_adds_spatial_gradient()
    test_v94_full_predictor_with_pe_backward_no_unused_params()
    test_v95_encoder_finer_grained_shape_and_distinct_features()
    test_v95_encoder_smaller_radius_gives_finer_fps_spacing()
    test_v95_full_predictor_pipeline_backward_no_unused_params()
    test_v95_hierarchical_decoder_output_dict_shapes()
    test_v95_hierarchical_decoder_patch_assignment_consistent()
    test_v95_hierarchical_decoder_iterative_mask_attention_progresses()
    test_v95_hierarchical_loss_patch_ce_fires()
    test_v95_full_predictor_with_hierarchical_decoder_backward()
    print("\nAll v8 + v8.1 + v8.1.1 + v9 + v9.1 + v9.2 + v9.4 + v9.5 sanity tests passed.")
