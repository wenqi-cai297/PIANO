"""Tests for ``piano.models.round29_cond_injection`` (Round-29 prompt §9.9)."""
from __future__ import annotations

import pytest
import torch

from piano.models.round29_cond_injection import (
    Round29CondInjectionConfig,
    Round29CondInjectionModule,
)


def _make_cond(B: int, T: int) -> dict[str, torch.Tensor]:
    return {
        "stage2_coarse_extra": torch.randn(B, T, 18),
        "stage2_interaction":  torch.randn(B, T, 8),
        "stage2_support":      torch.randn(B, T, 13),
        "stage2_body_refine":  torch.randn(B, T, 20),
    }


def _module(mode: str, *, D: int = 32, per_family_modes=None, n_layers: int = 4):
    cfg = Round29CondInjectionConfig(
        coarse_extra_dim=18,
        interaction_dim=8,
        support_dim=13,
        body_refine_dim=20,
        injection_mode=mode,
        gate_bias_init=-1.0,
        per_family_modes=per_family_modes,
    )
    mod = Round29CondInjectionModule(cfg, d_model=D)
    mod.configure_adapter_layers(n_layers=n_layers)
    return mod


@pytest.mark.parametrize("mode", ["input_add", "gated_input", "adapter_only", "input_add_adapter"])
def test_synthetic_forward_each_mode(mode: str) -> None:
    B, T, D = 2, 16, 32
    mod = _module(mode, D=D)
    cond = _make_cond(B, T)
    c_summary = torch.randn(B, D)
    h_in = torch.zeros(B, T, D)
    h = mod.apply_input_injection(h_in, cond, c_summary=c_summary)
    # All zero-init last Linear / no input add (adapter_only) → h equals h_in.
    assert torch.allclose(h, h_in, atol=0.0)
    seq = torch.cat([torch.zeros(B, 1, D), h], dim=1)
    for layer_idx in range(4):
        seq = mod.apply_per_layer_adapter(seq, layer_idx=layer_idx, motion_token_start=1)
    assert seq.shape == (B, T + 1, D)


def test_typed_per_family_modes() -> None:
    B, T, D = 2, 8, 32
    mod = _module(
        "typed", D=D,
        per_family_modes={
            "coarse_extra": "input_add",
            "interaction":  "gated_input",
            "support":      "adapter_only",
            "body_refine":  "input_add_adapter",
        },
    )
    cond = _make_cond(B, T)
    h_in = torch.zeros(B, T, D)
    c = torch.randn(B, D)
    h = mod.apply_input_injection(h_in, cond, c_summary=c)
    assert torch.allclose(h, h_in, atol=0.0)


def test_missing_key_raises_keyerror() -> None:
    mod = _module("input_add")
    cond = _make_cond(2, 4)
    cond.pop("stage2_interaction")
    with pytest.raises(KeyError, match="stage2_interaction"):
        mod.apply_input_injection(
            torch.zeros(2, 4, 32), cond, c_summary=None,
        )


def test_wrong_last_dim_raises_valueerror() -> None:
    mod = _module("input_add")
    cond = _make_cond(2, 4)
    cond["stage2_support"] = torch.randn(2, 4, 7)  # wrong width
    with pytest.raises(ValueError, match="last-dim"):
        mod.apply_input_injection(
            torch.zeros(2, 4, 32), cond, c_summary=None,
        )


def test_disabled_module_when_all_dims_zero() -> None:
    cfg = Round29CondInjectionConfig()  # all dims = 0
    mod = Round29CondInjectionModule(cfg, d_model=16)
    mod.configure_adapter_layers(n_layers=2)
    h = mod.apply_input_injection(
        torch.zeros(1, 4, 16), cond={}, c_summary=None,
    )
    assert torch.equal(h, torch.zeros(1, 4, 16))


def test_invalid_mode_raises_valueerror() -> None:
    with pytest.raises(ValueError):
        Round29CondInjectionConfig(injection_mode="bogus")
        Round29CondInjectionModule(
            Round29CondInjectionConfig(injection_mode="bogus"),
            d_model=8,
        )


def test_per_family_modes_typed_unknown_family_raises() -> None:
    with pytest.raises(ValueError, match="unknown family"):
        Round29CondInjectionModule(
            Round29CondInjectionConfig(
                coarse_extra_dim=18,
                injection_mode="typed",
                per_family_modes={"bogus_family": "input_add"},
            ),
            d_model=8,
        )


# ---------------------------------------------------------------------------
# Codex post-review §P1/§P2: R29 stats must be visible after forward.
# Grad-norm helper must group r29_inject parameters correctly.
# ---------------------------------------------------------------------------

def test_last_stats_populated_after_forward_gated_input() -> None:
    """gated_input mode produces gate_mean/gate_std stats per family."""
    mod = _module("gated_input", D=32)
    cond = _make_cond(2, 8)
    c = torch.randn(2, 32)
    mod.apply_input_injection(torch.zeros(2, 8, 32), cond, c_summary=c)
    stats = mod.last_stats()
    # Hint + emb norms per active family.
    for fam in ("coarse_extra", "interaction", "support", "body_refine"):
        assert f"r29_{fam}_hint_norm" in stats
        assert f"r29_{fam}_emb_norm" in stats
        assert f"r29_{fam}_gate_mean" in stats
        assert f"r29_{fam}_gate_std" in stats


def test_last_stats_populated_after_forward_input_add_adapter() -> None:
    """input_add_adapter mode: hint/emb norms + per-layer adapter norms."""
    mod = _module("input_add_adapter", D=32, n_layers=4)
    cond = _make_cond(2, 8)
    h = mod.apply_input_injection(torch.zeros(2, 8, 32), cond, c_summary=None)
    seq = torch.cat([torch.zeros(2, 1, 32), h], dim=1)
    for layer_idx in range(4):
        seq = mod.apply_per_layer_adapter(
            seq, layer_idx=layer_idx, motion_token_start=1,
        )
    stats = mod.last_stats()
    # Per-layer adapter norms must appear.
    for fam in ("coarse_extra", "interaction", "support", "body_refine"):
        for li in range(4):
            assert f"r29_{fam}_adapter_norm_layer{li}" in stats


def test_trainer_r29_grad_norm_helper_groups_params() -> None:
    """``_maybe_add_r29_cond_grad_stats`` must attach r29_grad_norm_*
    keys when an r29_inject submodule is present and r29_* stats exist."""
    import torch.nn as nn
    from piano.training.trainer import _maybe_add_r29_cond_grad_stats

    class StubInject(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.ModuleDict(
                {"coarse_extra": nn.Linear(8, 8)}
            )
            self.gate = nn.ModuleDict({"interaction": nn.Linear(16, 1)})
            self.adapters = nn.ModuleDict(
                {"body_refine": nn.ModuleList([nn.Linear(8, 8) for _ in range(2)])}
            )

    class StubModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.r29_inject = StubInject()

    model = StubModel()
    # Backward through a synthetic loss so .grad is populated.
    x = torch.randn(2, 8, requires_grad=True)
    y = model.r29_inject.proj["coarse_extra"](x).sum()
    y = y + model.r29_inject.gate["interaction"](
        torch.cat([x, x], dim=-1)
    ).sum()
    for adapter in model.r29_inject.adapters["body_refine"]:
        y = y + adapter(x).sum()
    y.backward()

    loss_dict: dict[str, torch.Tensor] = {"r29_coarse_extra_hint_norm": torch.tensor(0.1)}
    _maybe_add_r29_cond_grad_stats(model, loss_dict)
    assert "r29_grad_norm_proj" in loss_dict
    assert "r29_grad_norm_gate" in loss_dict
    assert "r29_grad_norm_adapters" in loss_dict
    # Real grads -> non-zero norms.
    assert float(loss_dict["r29_grad_norm_proj"]) > 0
    assert float(loss_dict["r29_grad_norm_gate"]) > 0
    assert float(loss_dict["r29_grad_norm_adapters"]) > 0


def test_trainer_r29_grad_norm_helper_no_op_without_r29_stats() -> None:
    """When no r29_* keys are present in loss_dict, the helper does nothing."""
    import torch.nn as nn
    from piano.training.trainer import _maybe_add_r29_cond_grad_stats

    class StubModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.r29_inject = nn.Linear(4, 4)

    loss_dict: dict[str, torch.Tensor] = {"loss": torch.tensor(0.0)}
    _maybe_add_r29_cond_grad_stats(StubModel(), loss_dict)
    assert "r29_grad_norm_proj" not in loss_dict
    assert "r29_grad_norm_gate" not in loss_dict
    assert "r29_grad_norm_adapters" not in loss_dict


# ============================================================================ #
# PB1 — AdaLN-cond branch (compute_family_embeddings + pool_cond_summary)
# ============================================================================ #


def _module_a1_like(D: int = 32, n_layers: int = 4):
    """A1's active families: coarse_extra (18) + support (13). PB1 base config."""
    cfg = Round29CondInjectionConfig(
        coarse_extra_dim=18,
        interaction_dim=0,
        support_dim=13,
        body_refine_dim=0,
        injection_mode="input_add_adapter",
        gate_bias_init=-1.0,
        per_family_modes=None,
    )
    mod = Round29CondInjectionModule(cfg, d_model=D)
    mod.configure_adapter_layers(n_layers=n_layers)
    return mod


def _cond_a1(B: int, T: int) -> dict[str, torch.Tensor]:
    return {
        "stage2_coarse_extra": torch.randn(B, T, 18),
        "stage2_support":      torch.randn(B, T, 13),
    }


def test_compute_family_embeddings_populates_cache_for_active_families():
    B, T, D = 2, 16, 32
    mod = _module_a1_like(D=D)
    cond = _cond_a1(B, T)
    cache = mod.compute_family_embeddings(cond)
    # Only active families (coarse_extra + support) are cached.
    assert set(cache.keys()) == {"coarse_extra", "support"}
    for f, emb in cache.items():
        assert emb.shape == (B, T, D)
    # The cache is also the module's internal cache (alias, not copy).
    assert cache is mod._cond_emb_cache


def test_compute_family_embeddings_zero_init_means_zero_emb_at_init():
    """A1's projection has a zero-init final Linear under input_add_adapter
    (see Round29CondInjectionModule.__init__). Right after init the
    family embedding must therefore be all-zero.
    """
    B, T, D = 2, 16, 32
    mod = _module_a1_like(D=D)
    cond = _cond_a1(B, T)
    cache = mod.compute_family_embeddings(cond)
    for f, emb in cache.items():
        assert torch.allclose(emb, torch.zeros_like(emb), atol=0.0)


def test_pool_cond_summary_mean_returns_mean_over_T_and_families():
    B, T, D = 2, 16, 32
    mod = _module_a1_like(D=D)
    cond = _cond_a1(B, T)
    # Bypass the zero-init by writing the cache directly with non-zero embs
    # so we can verify the pooling math, not the init.
    e_coarse = torch.randn(B, T, D)
    e_support = torch.randn(B, T, D)
    mod._cond_emb_cache = {"coarse_extra": e_coarse, "support": e_support}
    pooled = mod.pool_cond_summary(["coarse_extra", "support"], cond, pool="mean")
    expected = (e_coarse.mean(dim=1) + e_support.mean(dim=1)) / 2.0
    assert pooled.shape == (B, D)
    assert torch.allclose(pooled, expected, atol=1e-6)


def test_pool_cond_summary_support_walking_mean_uses_S4_dim4_as_weight():
    """walking_mask = S4 dim 4 (per the dataset; see r29_support_variant
    S4-S1-phase-footstep layout). The pool must compute a walking-mask-
    weighted mean of the support embedding with denominator clamped ≥ 1.0.
    """
    B, T, D = 2, 8, 32
    mod = _module_a1_like(D=D)
    cond = _cond_a1(B, T)
    # Force a known walking_mask: sample b=0 walks frames [0,1,2,3] only;
    # sample b=1 has no walking frames (all zero -> denom should clamp ≥ 1).
    cond["stage2_support"] = torch.zeros(B, T, 13)
    cond["stage2_support"][0, :4, 4] = 1.0
    e_support = torch.randn(B, T, D)
    mod._cond_emb_cache = {"support": e_support}
    pooled = mod.pool_cond_summary(["support"], cond, pool="support_walking_mean")
    assert pooled.shape == (B, D)
    # b=0: weighted = sum(emb[:4]); denom = 4; pooled = sum(emb[:4]) / 4 = mean(emb[:4]).
    expected_b0 = e_support[0, :4].mean(dim=0)
    assert torch.allclose(pooled[0], expected_b0, atol=1e-6)
    # b=1: no walking frames → denom = 1 (clamped), weighted = 0, pooled = 0.
    assert torch.allclose(pooled[1], torch.zeros(D), atol=1e-6)


def test_pool_cond_summary_falls_back_to_mean_when_support_not_active():
    """Per design: if the requested pool is support_walking_mean but
    support isn't an active family, fall back to mean over the requested
    families and record a fallback stat.
    """
    B, T, D = 2, 8, 32
    # Module with ONLY coarse_extra active (no support).
    cfg = Round29CondInjectionConfig(
        coarse_extra_dim=18,
        interaction_dim=0,
        support_dim=0,
        body_refine_dim=0,
        injection_mode="input_add",
    )
    mod = Round29CondInjectionModule(cfg, d_model=D)
    e_coarse = torch.randn(B, T, D)
    mod._cond_emb_cache = {"coarse_extra": e_coarse}
    pooled = mod.pool_cond_summary(
        ["coarse_extra"], {"stage2_coarse_extra": torch.randn(B, T, 18)},
        pool="support_walking_mean",
    )
    # Fallback path: mean over T then mean over families (only coarse here).
    expected = e_coarse.mean(dim=1)
    assert torch.allclose(pooled, expected, atol=1e-6)
    # Fallback was recorded as a warning stat.
    stats = mod.last_stats()
    assert "r29_adaln_support_walking_mean_fallback" in stats


def test_pool_cond_summary_raises_without_compute_first():
    B, T, D = 2, 8, 32
    mod = _module_a1_like(D=D)
    with pytest.raises(RuntimeError, match="compute_family_embeddings"):
        mod.pool_cond_summary(["support"], _cond_a1(B, T), pool="mean")


def test_apply_input_injection_reuses_precomputed_cache():
    """When the parent has already called compute_family_embeddings (PB1
    path), apply_input_injection MUST reuse the cache and not re-project.
    Test: pre-populate the cache with a sentinel tensor; call
    apply_input_injection; the residual must reflect the sentinel, not a
    fresh re-projection.
    """
    B, T, D = 2, 16, 32
    mod = _module_a1_like(D=D)
    cond = _cond_a1(B, T)
    # Sentinel: force coarse_extra emb to all-ones, support emb to all-twos.
    mod._cond_emb_cache = {
        "coarse_extra": torch.ones(B, T, D),
        "support":      torch.full((B, T, D), 2.0),
    }
    h = mod.apply_input_injection(torch.zeros(B, T, D), cond, c_summary=None)
    # input_add_adapter: h += proj(cond) for each family.
    # If the cache was reused, h = 0 + ones + twos = threes everywhere.
    # If the cache was clobbered (bug), h would equal 0 (zero-init proj).
    assert torch.allclose(h, torch.full((B, T, D), 3.0), atol=1e-6)


# ============================================================================ #
# PB1 — GlobalCondSummary + AnchorDenoiser end-to-end invariants
# ============================================================================ #


def _build_a1_denoiser(*, r29_use_cond_adaln: bool, seed: int = 42):
    """Build an AnchorDenoiser matching the A1 family-dim/injection config.
    Tiny capacity so the test runs fast.
    """
    from piano.models.motion_anchordiff import (
        AnchorDenoiser, AnchorDenoiserConfig,
    )
    torch.manual_seed(seed)
    cfg = AnchorDenoiserConfig(
        motion_dim=135,
        object_traj_dim=9,
        init_pose_dim=66,
        text_dim=512,
        object_token_dim=256,
        object_num_tokens=8,
        stage1_coarse_dim=23,
        use_round29_cond_injection=True,
        r29_coarse_extra_dim=18,
        r29_interaction_dim=0,
        r29_support_dim=13,
        r29_body_refine_dim=0,
        r29_injection_mode="input_add_adapter",
        r29_per_family_modes=None,
        r29_zero_init_adapters=True,
        r29_use_cond_adaln=r29_use_cond_adaln,
        r29_adaln_families=("support",) if r29_use_cond_adaln else None,
        r29_adaln_pool="support_walking_mean" if r29_use_cond_adaln else "mean",
        d_model=32,
        n_layers=2,
        n_heads=2,
        ff_mult=2,
        dropout=0.0,
        max_seq_length=32,
    )
    return AnchorDenoiser(cfg)


def _build_dummy_cond(B: int, T: int, D_obj: int = 256, N_obj: int = 8):
    return {
        "object_world_traj": torch.randn(B, T, 9),
        "object_tokens":     torch.randn(B, N_obj, D_obj),
        "text":              torch.randn(B, 4, 512),
        "init_pose":         torch.randn(B, 66),
        "stage1_coarse":     torch.randn(B, T, 23),
        "stage2_coarse_extra": torch.randn(B, T, 18),
        "stage2_support":      torch.randn(B, T, 13),
    }


def test_pb1_step0_forward_bit_identical_to_a1_when_seeded():
    """Codex §10 invariant: PB1 at init must produce the same step-0 output
    as A1 at init when their SHARED parameters match. PB1 has one extra
    submodule (cond_summary_mlp); its zero-init contributes 0 to ``c``,
    so the forward must match A1's bit-identically.

    Two tricks needed:

    1. ``nn.Linear.__init__`` consumes the global RNG to init weights.
       PB1 creates one extra Linear before being zeroed, so even with the
       same construction seed the two models have DIFFERENT shared
       weights. We work around this by loading PB1's shared keys from
       A1's state_dict (load_state_dict with strict=False, then sanity-
       check the missing keys are only the cond_summary_mlp ones).

    2. V12FinalLayer is zero-init, so the full step-0 output is
       identically 0 — that would pass the assertion trivially even if
       PB1 was misconfigured. We de-zero V12FinalLayer with identical
       weights across both models so the comparison is non-trivial.
    """
    B, T = 2, 16
    a1 = _build_a1_denoiser(r29_use_cond_adaln=False, seed=42)
    pb1 = _build_a1_denoiser(r29_use_cond_adaln=True, seed=42)
    a1.eval(); pb1.eval()

    # 1. Align all shared parameters: copy A1's state_dict into PB1
    # (PB1's cond_summary_mlp keys are missing from A1's sd; that's OK).
    a1_sd = a1.state_dict()
    missing, unexpected = pb1.load_state_dict(a1_sd, strict=False)
    # The only "missing" (i.e. unaligned) keys should be cond_summary_mlp.*
    # because A1 doesn't have them.
    assert all("cond_summary_mlp" in k for k in missing), (
        f"unexpected missing keys after PB1 ⟵ A1 state-dict load: {missing}"
    )
    assert unexpected == []
    # Confirm PB1's cond_summary_mlp is still zero-init after the load.
    cs_w = pb1.v12_cond_summary.cond_summary_mlp[-1].weight
    cs_b = pb1.v12_cond_summary.cond_summary_mlp[-1].bias
    assert torch.allclose(cs_w, torch.zeros_like(cs_w), atol=0.0)
    assert torch.allclose(cs_b, torch.zeros_like(cs_b), atol=0.0)

    # 2. De-zero the final layer in BOTH models with identical weights so
    # the step-0 output isn't trivially 0.
    torch.manual_seed(7)
    weight = torch.empty_like(a1.v12_final_layer.linear.weight)
    torch.nn.init.xavier_uniform_(weight)
    with torch.no_grad():
        for m in (a1, pb1):
            m.v12_final_layer.linear.weight.copy_(weight)
            m.v12_final_layer.linear.bias.zero_()

    torch.manual_seed(0)
    x_t = torch.randn(B, T, 135)
    t = torch.randint(0, 1000, (B,))
    cond = _build_dummy_cond(B, T)

    with torch.no_grad():
        out_a1 = a1(x_t, t, cond, cond_drop_mask=None)
        out_pb1 = pb1(x_t, t, cond, cond_drop_mask=None)

    assert out_a1.shape == out_pb1.shape == (B, T, 135)
    assert out_a1.abs().max() > 0.0, "test setup failed: outputs are still zero"
    # Bit-identical: the PB1 cond_summary branch contributes 0.
    assert torch.allclose(out_a1, out_pb1, atol=0.0, rtol=0.0)


def test_pb1_breaks_invariant_at_GlobalCondSummary_when_perturbed():
    """Negative control for the zero-init invariant: confirm the test
    above is actually sensitive to the cond_summary_mlp parameters.

    We can't check this end-to-end because V12FinalLayer is also zero-
    init — the whole forward is exactly 0 at step 0, regardless of any
    intermediate perturbation. So we test at the right layer: the
    GlobalCondSummary itself. If its Linear is zero, output equals
    t_emb; if we perturb it, output differs.
    """
    from piano.models.dit_blocks import GlobalCondSummary
    torch.manual_seed(0)
    D = 32
    mod_pb1 = GlobalCondSummary(d_model=D, use_cond_summary_mlp=True)
    t_emb = torch.randn(2, D)
    cs = torch.randn(2, D)
    # Zero-init: cond_summary contribution is 0 → output equals t_emb.
    out_zero = mod_pb1(t_emb, cs)
    assert torch.allclose(out_zero, t_emb, atol=0.0)
    # Perturb: cond_summary now contributes non-zero → output differs.
    with torch.no_grad():
        mod_pb1.cond_summary_mlp[-1].weight.fill_(0.1)
        mod_pb1.cond_summary_mlp[-1].bias.fill_(0.05)
    out_perturbed = mod_pb1(t_emb, cs)
    assert not torch.allclose(out_perturbed, t_emb, atol=1e-6)


def test_pb1_a1_state_dict_loads_into_a1_baseline_cleanly():
    """A pre-PB1 A1 ckpt must load cleanly into an A1-config denoiser
    (no PB1 fields). Backward-compat check: nothing we did to PB1
    broke state_dict shape for the existing A1 path.
    """
    a1_a = _build_a1_denoiser(r29_use_cond_adaln=False, seed=42)
    a1_b = _build_a1_denoiser(r29_use_cond_adaln=False, seed=123)
    sd = a1_a.state_dict()
    missing, unexpected = a1_b.load_state_dict(sd, strict=True)
    assert missing == []
    assert unexpected == []


def test_pb1_has_extra_cond_summary_mlp_in_state_dict():
    """PB1's state_dict has cond_summary_mlp.* keys; A1's does not.
    Verifies we don't accidentally instantiate the MLP under
    r29_use_cond_adaln=False (would cost params + break A1 ckpt load).
    """
    a1 = _build_a1_denoiser(r29_use_cond_adaln=False, seed=42)
    pb1 = _build_a1_denoiser(r29_use_cond_adaln=True, seed=42)
    a1_keys = set(a1.state_dict().keys())
    pb1_keys = set(pb1.state_dict().keys())
    pb1_extra = pb1_keys - a1_keys
    # The only difference must be the cond_summary_mlp's two parameters.
    assert any("cond_summary_mlp" in k for k in pb1_extra), (
        f"PB1 should expose cond_summary_mlp.* keys; got extras {pb1_extra}"
    )
    a1_extra = a1_keys - pb1_keys
    assert a1_extra == set(), (
        f"A1 should not have keys PB1 lacks; got extras {a1_extra}"
    )
