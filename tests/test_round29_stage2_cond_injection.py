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
