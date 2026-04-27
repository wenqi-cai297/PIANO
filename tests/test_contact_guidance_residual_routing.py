from __future__ import annotations

import torch

from piano.inference.contact_guidance import _generate_residual_tokens


class _FakeResidualWithInt(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = []

    def generate_with_int(self, **kwargs):
        self.calls.append(kwargs)
        return torch.full((1, 3, 6), 7, dtype=torch.long)


class _FakeRawResidual(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return torch.full((1, 3, 6), 5, dtype=torch.long)


def test_generate_residual_tokens_uses_c1_int_path_and_seq_first_kv():
    model = _FakeResidualWithInt()
    motion_ids = torch.zeros((1, 3), dtype=torch.long)
    int_kv_bf = torch.randn(1, 4, 8)
    int_pad = torch.zeros((1, 4), dtype=torch.bool)

    out = _generate_residual_tokens(
        model,
        motion_ids=motion_ids,
        text="pick up the box",
        m_lens_tok=torch.tensor([3]),
        int_kv=int_kv_bf,
        int_pad=int_pad,
        res_cond_scale=2.0,
    )

    assert out.unique().item() == 7
    call = model.calls[0]
    assert call["motion_ids"] is motion_ids
    assert call["conds"] == ["pick up the box"]
    assert call["int_padding_mask"] is int_pad
    assert call["int_kv"].shape == (4, 1, 8)
    torch.testing.assert_close(call["int_kv"], int_kv_bf.transpose(0, 1))


def test_generate_residual_tokens_falls_back_to_raw_generate():
    model = _FakeRawResidual()
    motion_ids = torch.zeros((1, 3), dtype=torch.long)

    out = _generate_residual_tokens(
        model,
        motion_ids=motion_ids,
        text="walk around",
        m_lens_tok=torch.tensor([3]),
        int_kv=None,
        int_pad=None,
        res_cond_scale=2.0,
    )

    assert out.unique().item() == 5
    call = model.calls[0]
    assert call["motion_ids"] is motion_ids
    assert call["conds"] == ["walk around"]
    torch.testing.assert_close(call["m_lens"], torch.tensor([3]))
    assert call["cond_scale"] == 2.0
