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


def test_body_canonical_to_object_local_torch_identity_pose():
    from piano.training.decoded_contact_loss import body_canonical_to_object_local_torch

    body = torch.tensor([[[[11.0, 2.0, -2.0], [20.0, 1.0, -5.0]]]])
    com = torch.tensor([[[10.0, 0.0, -5.0]]])
    rot6d = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]).view(1, 1, 6)

    out = body_canonical_to_object_local_torch(body, com, rot6d)
    expected = torch.tensor([[[[1.0, 2.0, 3.0], [10.0, 1.0, 0.0]]]])
    assert torch.allclose(out, expected, atol=1e-6)


def test_target_trajectory_loss_uses_part_specific_contact_targets():
    from piano.training.decoded_contact_loss import _target_trajectory_loss_canonical

    rot6d_identity = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]).view(1, 1, 6)
    body = torch.tensor(
        [[[
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]]],
        requires_grad=True,
    )
    contact = torch.tensor([[[1.0, 0.0]]])
    target = torch.tensor([[[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]])

    loss, metrics = _target_trajectory_loss_canonical(
        body_canonical=body,
        obj_com_canonical=torch.zeros(1, 1, 3),
        obj_rot6d_canonical=rot6d_identity,
        contact_state=contact,
        contact_target_xyz=target,
        frame_mask=torch.tensor([[True]]),
        position_weight=1.0,
        velocity_weight=0.0,
        metric_loss=torch.zeros(()),
        metric_weight=0.0,
        moving_frame_extra_weight=0.0,
        contact_threshold=0.5,
        use_soft_contact_weights=True,
        velocity_moving_only=True,
        fps=20.0,
        moving_speed_threshold=0.15,
        kin_radius_proxy=0.3,
    )

    assert torch.allclose(metrics["decoded_contact_aux_target_position"], torch.tensor(1.0))
    loss.backward()
    assert body.grad is not None
    assert body.grad[0, 0, 0].abs().sum() > 0
    assert body.grad[0, 0, 1].abs().sum() == 0


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
        "contact_state": torch.ones(B, T, 5),
        "contact_target_xyz": torch.zeros(B, T, 5, 3),
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

    base_logits_target = torch.randn(B, V, S, requires_grad=True)
    target_loss, target_metrics = decoded_contact_aux_loss(
        base_logits=base_logits_target,
        all_indices=all_indices,
        vq_model=vq,
        motion_mean=motion_mean,
        motion_std=motion_std,
        batch=batch,
        m_lens_tok=torch.tensor([S - 1]),
        num_object_points=8,
        temperature=1.0,
        mode="target_trajectory",
        target_position_weight=1.0,
        target_velocity_weight=0.5,
    )
    assert torch.isfinite(target_loss)
    assert target_metrics["decoded_contact_aux_target_position"].dim() == 0
    assert target_metrics["decoded_contact_aux_target_velocity"].dim() == 0
    target_loss.backward()
    assert base_logits_target.grad is not None
    assert base_logits_target.grad.abs().sum() > 0


def test_decoded_contact_aux_loss_full_prediction_flows_to_residual_path():
    from piano.training.decoded_contact_loss import decoded_contact_aux_loss

    torch.manual_seed(11)
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

    class FakeResidual(nn.Module):
        def __init__(self):
            super().__init__()
            self.cond_mode = "uncond"
            self.latent_dim = code_dim
            self.token_embed_weight = nn.Parameter(torch.randn(Q - 1, V + 1, code_dim) * 0.1)
            self.output_proj_weight = nn.Parameter(torch.randn(Q - 1, V + 1, code_dim) * 0.1)
            self.output_proj_bias = None

        def process_embed_proj_weight(self):
            return None

        def trans_forward_with_int(
            self,
            motion_codes,
            qids,
            cond,
            padding_mask,
            *,
            int_kv=None,
            int_padding_mask=None,
        ):
            return motion_codes.transpose(1, 2).contiguous()

        def output_project(self, logits, qids):
            weight = self.output_proj_weight[qids]
            return torch.einsum("bnc,bcs->bns", weight, logits)

    vq = FakeRVQVAE()
    residual = FakeResidual()
    base_logits = torch.randn(B, S, V, requires_grad=True)
    all_indices = torch.zeros(B, S, Q, dtype=torch.long)
    motion_mean = torch.zeros(263)
    motion_std = torch.ones(263)

    rot6d_identity = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]).view(1, 1, 6)
    batch = {
        "object_pc": torch.ones(B, 12, 3),
        "obj_com_canonical": torch.zeros(B, T, 3),
        "obj_rot6d_canonical": rot6d_identity.repeat(B, T, 1),
        "seq_len": torch.tensor([T]),
    }

    loss, metrics = decoded_contact_aux_loss(
        base_logits=base_logits,
        all_indices=all_indices,
        vq_model=vq,
        motion_mean=motion_mean,
        motion_std=motion_std,
        batch=batch,
        m_lens_tok=torch.tensor([S]),
        num_object_points=8,
        temperature=1.0,
        rvq_path="full_prediction",
        residual_transformer=residual,
        text=None,
    )

    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert metrics["decoded_contact_aux_residual_layers"].item() == Q - 1

    loss.backward()
    assert base_logits.grad is not None
    assert base_logits.grad.abs().sum() > 0
    assert residual.token_embed_weight.grad is not None
    assert residual.token_embed_weight.grad.abs().sum() > 0
    assert residual.output_proj_weight.grad is not None
    assert residual.output_proj_weight.grad.abs().sum() > 0
