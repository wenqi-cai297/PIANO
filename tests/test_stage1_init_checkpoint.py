"""Tests for the R41 warm-start loader.

`_maybe_load_stage1_init_checkpoint` is the load-bearing prereq for the
R41 cascade experiment: without it, ``training.init_checkpoint`` in the
cfg is silently ignored and "fine-tune V8 V6" runs from scratch instead.

The tests below cover the three failure paths plus the no-op default.
They DO NOT round-trip a real Stage1Denoiser ckpt because that needs
the full repo dependency stack (omegaconf etc.) which is not always
available on the dev laptop. Forward round-trip is verified on the
server during the R41 smoke test (`--smoke-test` flag in the trainer).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

# The loader transitively imports omegaconf (via the trainer module). Skip
# the whole file when the dependency is missing on the dev laptop; the
# tests run on the server where the env is complete.
pytest.importorskip("omegaconf")


from piano.training.train_stage1 import (  # noqa: E402
    _maybe_load_stage1_init_checkpoint,
)


# ──────────────────────────────────────────────────────────────────────────
# Fixtures — minimal stand-in modules with the right state_dict shape.
# We don't need a real Stage1Denoiser for the negative-path tests; any
# nn.Module with a state_dict will exercise the load contract.
# ──────────────────────────────────────────────────────────────────────────


class _TinyModel(torch.nn.Module):
    """Stand-in for Stage1Denoiser. Same state_dict contract is all we
    need to verify the loader's error handling."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(4, 4)


class _TinyEncoder(torch.nn.Module):
    """Stand-in for ObjectEncoder."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)


# ──────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────


def test_warm_start_noop_when_path_is_none():
    """No-op when ckpt_path is None — preserves from-scratch behavior."""
    model = _TinyModel()
    encoder = _TinyEncoder()
    before_model = {k: v.clone() for k, v in model.state_dict().items()}
    before_encoder = {k: v.clone() for k, v in encoder.state_dict().items()}

    _maybe_load_stage1_init_checkpoint(
        model=model, object_encoder=encoder, ckpt_path=None,
    )

    # State unchanged.
    for k, v in model.state_dict().items():
        assert torch.equal(v, before_model[k])
    for k, v in encoder.state_dict().items():
        assert torch.equal(v, before_encoder[k])


def test_warm_start_noop_when_path_is_empty_string():
    """Empty string also no-op — defensive against cfg writers passing ''."""
    model = _TinyModel()
    encoder = _TinyEncoder()
    before = {k: v.clone() for k, v in model.state_dict().items()}

    _maybe_load_stage1_init_checkpoint(
        model=model, object_encoder=encoder, ckpt_path="",
    )

    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k])


def test_warm_start_raises_when_path_missing(tmp_path: Path):
    """Non-empty path that doesn't exist → FileNotFoundError.

    Catches the common typo / stale-config case where someone points to
    a ckpt that no longer exists. Silently no-op'ing here would be the
    worst outcome: the run starts from scratch but the user thinks it's
    fine-tuning.
    """
    bogus = tmp_path / "does_not_exist.pt"
    model = _TinyModel()
    encoder = _TinyEncoder()

    with pytest.raises(FileNotFoundError, match="init_checkpoint"):
        _maybe_load_stage1_init_checkpoint(
            model=model, object_encoder=encoder,
            ckpt_path=str(bogus),
        )


def test_warm_start_raises_when_object_encoder_state_missing(tmp_path: Path):
    """Ckpt has denoiser but no object_encoder state → KeyError.

    Without the object encoder, Stage-1 sees random-init object features
    while the denoiser is loaded — a silent failure mode that we
    explicitly refuse.
    """
    model = _TinyModel()
    encoder = _TinyEncoder()

    # Save a ckpt that only contains the denoiser state (no
    # object_encoder, no extra_modules dict).
    ckpt = tmp_path / "no_encoder.pt"
    torch.save({"model": model.state_dict()}, str(ckpt))

    # Fresh modules so loading should mutate them.
    fresh_model = _TinyModel()
    fresh_encoder = _TinyEncoder()
    with pytest.raises(KeyError, match="object_encoder"):
        _maybe_load_stage1_init_checkpoint(
            model=fresh_model, object_encoder=fresh_encoder,
            ckpt_path=str(ckpt),
        )


def test_warm_start_loads_when_object_encoder_under_extra_modules(
    tmp_path: Path,
):
    """Legacy save format: state["extra_modules"]["object_encoder"]
    should also load cleanly (mirrors sample_substitute_conds.py:149-150).
    """
    src_model = _TinyModel()
    src_encoder = _TinyEncoder()
    # Mutate so we can verify the load actually moved weights, not just
    # silently no-op'd on matching random init.
    with torch.no_grad():
        src_model.fc.weight.fill_(0.123)
        src_encoder.proj.weight.fill_(0.456)
    ckpt = tmp_path / "legacy_format.pt"
    torch.save(
        {
            "model": src_model.state_dict(),
            "extra_modules": {"object_encoder": src_encoder.state_dict()},
        },
        str(ckpt),
    )

    dst_model = _TinyModel()
    dst_encoder = _TinyEncoder()
    _maybe_load_stage1_init_checkpoint(
        model=dst_model, object_encoder=dst_encoder,
        ckpt_path=str(ckpt),
    )

    assert torch.allclose(dst_model.fc.weight, src_model.fc.weight)
    assert torch.allclose(dst_encoder.proj.weight, src_encoder.proj.weight)


def test_warm_start_loads_when_object_encoder_at_top_level(
    tmp_path: Path,
):
    """Current save format: state["object_encoder"] at top level
    (mirrors sample_substitute_conds.py:147-148). This is the format
    that ``run_training_loop`` writes."""
    src_model = _TinyModel()
    src_encoder = _TinyEncoder()
    with torch.no_grad():
        src_model.fc.weight.fill_(0.789)
        src_encoder.proj.weight.fill_(0.321)
    ckpt = tmp_path / "current_format.pt"
    torch.save(
        {
            "model": src_model.state_dict(),
            "object_encoder": src_encoder.state_dict(),
        },
        str(ckpt),
    )

    dst_model = _TinyModel()
    dst_encoder = _TinyEncoder()
    _maybe_load_stage1_init_checkpoint(
        model=dst_model, object_encoder=dst_encoder,
        ckpt_path=str(ckpt),
    )

    assert torch.allclose(dst_model.fc.weight, src_model.fc.weight)
    assert torch.allclose(dst_encoder.proj.weight, src_encoder.proj.weight)


def test_warm_start_strict_mismatch_raises(tmp_path: Path):
    """strict=True (default) should raise on state_dict key mismatch,
    so we never silently partial-load and ship a broken warm-start."""

    class _Mismatch(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.different_name = torch.nn.Linear(4, 4)

    src_model = _Mismatch()
    src_encoder = _TinyEncoder()
    ckpt = tmp_path / "mismatch.pt"
    torch.save(
        {
            "model": src_model.state_dict(),
            "object_encoder": src_encoder.state_dict(),
        },
        str(ckpt),
    )

    dst_model = _TinyModel()       # has "fc.*", ckpt has "different_name.*"
    dst_encoder = _TinyEncoder()
    with pytest.raises(RuntimeError):  # PyTorch raises RuntimeError on strict mismatch
        _maybe_load_stage1_init_checkpoint(
            model=dst_model, object_encoder=dst_encoder,
            ckpt_path=str(ckpt),
            strict=True,
        )


def test_warm_start_strict_false_tolerates_mismatch(tmp_path: Path):
    """strict=False allows partial load — escape hatch for cfg evolution."""

    class _ExtraKey(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = torch.nn.Linear(4, 4)
            self.extra = torch.nn.Linear(4, 4)  # not in target

    src_model = _ExtraKey()
    src_encoder = _TinyEncoder()
    ckpt = tmp_path / "partial.pt"
    torch.save(
        {
            "model": src_model.state_dict(),
            "object_encoder": src_encoder.state_dict(),
        },
        str(ckpt),
    )

    dst_model = _TinyModel()
    dst_encoder = _TinyEncoder()
    # strict=False — should not raise even though ckpt has an extra key.
    _maybe_load_stage1_init_checkpoint(
        model=dst_model, object_encoder=dst_encoder,
        ckpt_path=str(ckpt),
        strict=False,
    )
    # fc.* loaded.
    assert torch.allclose(dst_model.fc.weight, src_model.fc.weight)
