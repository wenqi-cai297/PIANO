from __future__ import annotations

import torch
import torch.nn as nn


def test_rotation_6d_to_matrix_torch_inverts_repo_layout_identity():
    from piano.training.decoded_contact_loss import rotation_6d_to_matrix_torch

    d6 = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    R = rotation_6d_to_matrix_torch(d6)
    assert torch.allclose(R, torch.eye(3), atol=1e-6)


def test_object_pc_to_canonical_torch_identity_pose():
    from piano.training.decoded_contact_loss import object_pc_to_canonical_torch

    pc = torch.tensor([[[1.0, 2.0, 3.0], [0.0, 0.0, 1.0]]])
    com = torch.tensor([[[10.0, 0.0, -5.0], [20.0, 1.0, -6.0]]])
    rot6d = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]).view(1, 1, 6).repeat(1, 2, 1)

    out = object_pc_to_canonical_torch(pc, com, rot6d)
    assert out.shape == (1, 2, 2, 3)
    assert torch.allclose(out[0, 0, 0], torch.tensor([11.0, 2.0, -2.0]), atol=1e-6)
    assert torch.allclose(out[0, 1, 1], torch.tensor([20.0, 1.0, -5.0]), atol=1e-6)


def test_decoded_contact_aux_loss_gradient_flows_to_base_logits():
    from piano.training.decoded_contact_loss import decoded_contact_aux_loss

    torch.manual_seed(7)
    B, S, V, code_dim = 1, 3, 8, 16
    Q = 3
    T = S * 4

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
    base_logits = torch.randn(B, V, S, requires_grad=True)
    all_indices = torch.zeros(B, S, Q, dtype=torch.long)
    motion_mean = torch.zeros(263)
    motion_std = torch.ones(263)

    rot6d_identity = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]).view(1, 1, 6)
    batch = {
        "object_pc": torch.ones(B, 12, 3),
        "obj_com_canonical": torch.zeros(B, T, 3),
        "obj_rot6d_canonical": rot6d_identity.repeat(B, T, 1),
        "seq_len": torch.tensor([T - 1]),
    }

    loss, metrics = decoded_contact_aux_loss(
        base_logits=base_logits,
        all_indices=all_indices,
        vq_model=vq,
        motion_mean=motion_mean,
        motion_std=motion_std,
        batch=batch,
        m_lens_tok=torch.tensor([S - 1]),
        num_object_points=8,
        temperature=1.0,
    )

    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert metrics["decoded_contact_aux_mean_min_dist"].dim() == 0
    assert metrics["decoded_contact_aux_valid_frames"].item() == float((S - 1) * 4)

    loss.backward()
    assert base_logits.grad is not None
    assert torch.isfinite(base_logits.grad).all()
    assert base_logits.grad.abs().sum() > 0
    assert base_logits.grad[:, :, -1].abs().sum() == 0
