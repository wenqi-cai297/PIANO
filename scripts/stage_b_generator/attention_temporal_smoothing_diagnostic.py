"""Temporal smoothing diagnostic for the full-data v18 Stage B checkpoint.

This diagnostic asks a narrow question:

    does the trained DiT self-attention stack progressively flatten temporal
    structure inside the denoiser, or is the remaining frozen-body gap better
    explained by the already-documented multimodality / sampling effects?

The script runs two linked measurements on a small clip subset:

1. One-step reconstruction at a fixed diffusion timestep, with lightweight
   diagnostic monkey patches on the v12 DiT blocks. These patches expose:
   - self-attention maps for motion tokens only;
   - hidden states after input projection, each sub-block, final-head input,
     and reconstructed ``x0_pred``;
   - the actual residual contributions added by self-attn, plan-xattn, and
     MLP branches.
2. A normal full DDPM sample on the same clips, without the diagnostic block
   patch enabled, so the external frozen-body metrics stay comparable to the
   existing ``dynamics_diagnostic.py`` reports.

Normal training/inference code is not modified. The attention-weight path is
only activated inside this standalone script by temporarily wrapping the
already-instantiated checkpoint model.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from dynamics_diagnostic import (
    _build_cond,
    _build_dataset,
    _build_model,
    _fft_band_energy,
    _fk_from_motion_135,
    _per_joint_vel_stats,
)
from piano.data.dataset import collate_hoi
from piano.models.dit_blocks import modulate
from piano.utils.clip_utils import load_clip_text_encoder


PLAN_KEYS = [
    "anchor_time",
    "anchor_part",
    "anchor_target_local",
    "anchor_target_world",
    "anchor_type",
    "anchor_phase",
    "anchor_support",
    "anchor_conf",
    "anchor_mask",
    "segment_start",
    "segment_end",
    "segment_part",
    "segment_target_summary_local",
    "segment_phase",
    "segment_support",
    "segment_conf",
    "segment_mask",
]

LOCAL_WINDOWS = (1, 3, 5, 10)


def _extract_plan(batch: dict[str, Any], device: torch.device) -> dict[str, Tensor]:
    return {k: batch[f"plan_{k}"].to(device) for k in PLAN_KEYS}


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if abs(den) > 1e-12 else 0.0


class TemporalAccumulator:
    """Pooled temporal metrics for tensors shaped ``(B, T, D)``."""

    def __init__(self, fps: float, track_norm: bool = False) -> None:
        self.fps = float(fps)
        self.track_norm = bool(track_norm)
        self.velocity_sum = 0.0
        self.velocity_count = 0
        self.acceleration_sum = 0.0
        self.acceleration_count = 0
        self.norm_sum = 0.0
        self.norm_count = 0
        self.fft_low = 0.0
        self.fft_mid = 0.0
        self.fft_high = 0.0
        self.n_clips = 0

    @torch.no_grad()
    def update(self, tensor: Tensor, seq_mask: Tensor) -> None:
        x = tensor.detach().float()
        mask = seq_mask.detach().bool()
        batch_size, total_t, _ = x.shape
        if mask.shape[1] != total_t:
            raise ValueError(
                f"Temporal metric mask mismatch: tensor T={total_t}, mask T={mask.shape[1]}"
            )

        for b in range(batch_size):
            valid = int(mask[b].sum().item())
            if valid <= 0:
                continue
            xb = x[b, :valid]
            if self.track_norm:
                norms = torch.linalg.vector_norm(xb, dim=-1)
                self.norm_sum += float(norms.sum().item())
                self.norm_count += int(norms.numel())

            if valid >= 2:
                vel = torch.linalg.vector_norm(xb[1:] - xb[:-1], dim=-1)
                self.velocity_sum += float(vel.sum().item())
                self.velocity_count += int(vel.numel())

            if valid >= 3:
                acc = torch.linalg.vector_norm(
                    xb[2:] - 2.0 * xb[1:-1] + xb[:-2],
                    dim=-1,
                )
                self.acceleration_sum += float(acc.sum().item())
                self.acceleration_count += int(acc.numel())

            if valid >= 32:
                xc = xb - xb.mean(dim=0, keepdim=True)
                fft = torch.fft.rfft(xc, dim=0)
                power = (fft.real.square() + fft.imag.square()).sum(dim=-1)
                freqs = torch.fft.rfftfreq(valid, d=1.0 / self.fps).to(power.device)
                self.fft_low += float(power[(freqs >= 0.0) & (freqs < 1.0)].sum().item())
                self.fft_mid += float(power[(freqs >= 1.0) & (freqs < 4.0)].sum().item())
                self.fft_high += float(power[freqs >= 4.0].sum().item())
            self.n_clips += 1

    def summary(self) -> dict[str, float | int]:
        total_fft = self.fft_low + self.fft_mid + self.fft_high
        out: dict[str, float | int] = {
            "temporal_velocity_mean": _safe_div(self.velocity_sum, float(self.velocity_count)),
            "temporal_acceleration_mean": _safe_div(
                self.acceleration_sum,
                float(self.acceleration_count),
            ),
            "fft_energy_total": float(total_fft),
            "fft_fraction_low": _safe_div(self.fft_low, total_fft),
            "fft_fraction_mid": _safe_div(self.fft_mid, total_fft),
            "fft_fraction_high": _safe_div(self.fft_high, total_fft),
            "n_clips": int(self.n_clips),
            "velocity_count": int(self.velocity_count),
            "acceleration_count": int(self.acceleration_count),
        }
        if self.track_norm:
            out["norm_mean"] = _safe_div(self.norm_sum, float(self.norm_count))
            out["norm_count"] = int(self.norm_count)
        return out


class AttentionAccumulator:
    """Pooled motion-token self-attention statistics for one DiT block."""

    def __init__(self) -> None:
        self.weight = 0
        self.entropy_sum = 0.0
        self.entropy_norm_sum = 0.0
        self.motion_key_mass_sum = 0.0
        self.init_pose_mass_sum = 0.0
        self.mean_distance_sum = 0.0
        self.diagonal_mass_sum = 0.0
        self.diagonal_uniform_ratio_sum = 0.0
        self.query_similarity_sum = 0.0
        self.local_mass_sum = {w: 0.0 for w in LOCAL_WINDOWS}

    @torch.no_grad()
    def update(self, attn_weights: Tensor, seq_mask: Tensor) -> None:
        # Expected shape from MultiheadAttention with average_attn_weights=False:
        # (B, H, L_query, L_key), where L = T_motion + 1 init-pose token.
        weights = attn_weights.detach().float()
        mask = seq_mask.detach().bool()
        batch_size, n_heads, seq_q, seq_k = weights.shape
        if seq_q != seq_k:
            raise ValueError(f"Self-attn expected square map, got {weights.shape}")
        if seq_q != mask.shape[1] + 1:
            raise ValueError(
                f"Self-attn token mismatch: attn L={seq_q}, motion T={mask.shape[1]}"
            )

        for b in range(batch_size):
            valid = int(mask[b].sum().item())
            if valid <= 0:
                continue
            # Query tokens 1..valid, key tokens 1..valid. Token 0 is init_pose.
            sub_raw = weights[b, :, 1 : valid + 1, 1 : valid + 1]  # (H, L, L)
            motion_mass = sub_raw.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            sub = sub_raw / motion_mass
            init_pose_mass = weights[b, :, 1 : valid + 1, 0]

            entropy = -(sub.clamp_min(1e-12) * sub.clamp_min(1e-12).log()).sum(dim=-1)
            entropy_norm = entropy / max(math.log(max(valid, 2)), 1e-12)

            q = torch.arange(valid, device=sub.device).view(1, valid, 1)
            k = torch.arange(valid, device=sub.device).view(1, 1, valid)
            dist = (q - k).abs().float()
            mean_distance = (sub * dist).sum(dim=-1)

            diag = torch.diagonal(sub, dim1=-2, dim2=-1)
            diag_mass = diag
            diag_uniform_ratio = diag * float(valid)

            query_mean = sub.mean(dim=1, keepdim=True)
            query_similarity = torch.nn.functional.cosine_similarity(
                sub,
                query_mean.expand_as(sub),
                dim=-1,
                eps=1e-12,
            )

            n = int(n_heads * valid)
            self.weight += n
            self.entropy_sum += float(entropy.sum().item())
            self.entropy_norm_sum += float(entropy_norm.sum().item())
            self.motion_key_mass_sum += float(motion_mass.squeeze(-1).sum().item())
            self.init_pose_mass_sum += float(init_pose_mass.sum().item())
            self.mean_distance_sum += float(mean_distance.sum().item())
            self.diagonal_mass_sum += float(diag_mass.sum().item())
            self.diagonal_uniform_ratio_sum += float(diag_uniform_ratio.sum().item())
            self.query_similarity_sum += float(query_similarity.sum().item())

            for window in LOCAL_WINDOWS:
                local = (dist <= float(window)).float()
                mass = (sub * local).sum(dim=-1)
                self.local_mass_sum[window] += float(mass.sum().item())

    def summary(self) -> dict[str, float | int]:
        denom = float(max(self.weight, 1))
        out: dict[str, float | int] = {
            "attention_entropy": self.entropy_sum / denom,
            "attention_entropy_normalized": self.entropy_norm_sum / denom,
            "motion_key_mass_raw": self.motion_key_mass_sum / denom,
            "init_pose_key_mass_raw": self.init_pose_mass_sum / denom,
            "mean_attended_temporal_distance": self.mean_distance_sum / denom,
            "diagonal_mass": self.diagonal_mass_sum / denom,
            "diagonal_uniform_ratio": self.diagonal_uniform_ratio_sum / denom,
            "query_distribution_cosine_to_mean": self.query_similarity_sum / denom,
            "count_query_head_pairs": int(self.weight),
        }
        for window in LOCAL_WINDOWS:
            out[f"local_attention_mass_pm{window}"] = self.local_mass_sum[window] / denom
        out["global_attention_mass_outside_pm10"] = (
            1.0 - float(out["local_attention_mass_pm10"])
        )
        return out


class DiagnosticRecorder:
    """Runtime sink used by temporary diagnostic block wrappers."""

    def __init__(self, fps: float) -> None:
        self.fps = float(fps)
        self.enabled = False
        self.seq_mask: Tensor | None = None
        self.hidden: dict[str, TemporalAccumulator] = defaultdict(
            lambda: TemporalAccumulator(self.fps, track_norm=False)
        )
        self.branches: dict[str, TemporalAccumulator] = defaultdict(
            lambda: TemporalAccumulator(self.fps, track_norm=True)
        )
        self.attention: dict[str, AttentionAccumulator] = defaultdict(AttentionAccumulator)

    def begin(self, seq_mask: Tensor) -> None:
        self.seq_mask = seq_mask.detach()
        self.enabled = True

    def end(self) -> None:
        self.enabled = False
        self.seq_mask = None

    def _mask(self) -> Tensor:
        if self.seq_mask is None:
            raise RuntimeError("DiagnosticRecorder used without seq_mask context")
        return self.seq_mask

    def record_hidden(self, name: str, tensor: Tensor) -> None:
        if self.enabled:
            self.hidden[name].update(tensor, self._mask())

    def record_branch(self, name: str, tensor: Tensor) -> None:
        if self.enabled:
            self.branches[name].update(tensor, self._mask())

    def record_attention(self, name: str, weights: Tensor) -> None:
        if self.enabled:
            self.attention[name].update(weights, self._mask())

    def summary(self) -> dict[str, dict[str, float | int]]:
        return {
            "hidden_states": {k: v.summary() for k, v in self.hidden.items()},
            "residual_branches": {k: v.summary() for k, v in self.branches.items()},
            "attention_maps": {k: v.summary() for k, v in self.attention.items()},
        }


def _install_diagnostic_patches(model, recorder: DiagnosticRecorder) -> list[Any]:
    """Temporarily patch v18's DiT blocks so recon forwards expose internals."""

    denoiser = model.denoiser
    if not getattr(denoiser.cfg, "use_dit_block", False):
        raise ValueError("This diagnostic expects use_dit_block=True")

    restore_actions: list[Any] = []

    def _input_hook(_module, _inputs, output):
        recorder.record_hidden("input_projection", output)

    input_handle = denoiser.v12_input_proj.register_forward_hook(_input_hook)
    restore_actions.append(input_handle.remove)

    def _final_pre_hook(_module, inputs):
        recorder.record_hidden("final_head_input", inputs[0])

    final_handle = denoiser.v12_final_layer.register_forward_pre_hook(_final_pre_hook)
    restore_actions.append(final_handle.remove)

    for idx, block in enumerate(denoiser.v12_blocks):
        original_forward = block.forward

        def _diag_forward(
            self,
            x: Tensor,
            c: Tensor,
            plan_kv: Tensor,
            plan_key_padding_mask: Tensor,
            motion_token_start: int = 1,
            *,
            _idx: int = idx,
            _original=original_forward,
        ) -> Tensor:
            if not recorder.enabled:
                return _original(
                    x,
                    c,
                    plan_kv,
                    plan_key_padding_mask,
                    motion_token_start,
                )

            prefix = f"block_{_idx:02d}"
            recorder.record_hidden(f"{prefix}_input", x[:, motion_token_start:])

            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(c).chunk(6, dim=-1)
            )

            h_self = modulate(self.norm1(x), shift_msa, scale_msa)
            attn_out, attn_weights = self.self_attn(
                h_self,
                h_self,
                h_self,
                need_weights=True,
                average_attn_weights=False,
            )
            self_residual = gate_msa.unsqueeze(1) * attn_out
            x_after_self = x + self_residual
            recorder.record_attention(prefix, attn_weights)
            recorder.record_branch(
                f"{prefix}_self_attn_out",
                self_residual[:, motion_token_start:],
            )
            recorder.record_hidden(
                f"{prefix}_after_self_attn",
                x_after_self[:, motion_token_start:],
            )

            x_curr = x_after_self
            if self.use_temporal_conv:
                prefix_tokens = x_curr[:, :motion_token_start]
                motion_tokens = x_curr[:, motion_token_start:]
                motion_tokens = self.temporal_conv(motion_tokens)
                x_curr = torch.cat([prefix_tokens, motion_tokens], dim=1)
                recorder.record_hidden(
                    f"{prefix}_after_temporal_conv",
                    x_curr[:, motion_token_start:],
                )

            xattn_out, _ = self.plan_xattn(
                x_curr,
                plan_kv,
                plan_kv,
                key_padding_mask=plan_key_padding_mask,
                need_weights=False,
            )
            x_after_plan = x_curr + xattn_out
            recorder.record_branch(
                f"{prefix}_plan_xattn_out",
                xattn_out[:, motion_token_start:],
            )
            recorder.record_hidden(
                f"{prefix}_after_plan_xattn",
                x_after_plan[:, motion_token_start:],
            )

            h_mlp = modulate(self.norm2(x_after_plan), shift_mlp, scale_mlp)
            mlp_out = self.mlp(h_mlp)
            mlp_residual = gate_mlp.unsqueeze(1) * mlp_out
            x_out = x_after_plan + mlp_residual
            recorder.record_branch(
                f"{prefix}_mlp_out",
                mlp_residual[:, motion_token_start:],
            )
            recorder.record_hidden(
                f"{prefix}_after_mlp",
                x_out[:, motion_token_start:],
            )
            recorder.record_hidden(
                f"{prefix}_output",
                x_out[:, motion_token_start:],
            )
            return x_out

        block.forward = MethodType(_diag_forward, block)
        restore_actions.append(
            lambda _block=block, _original=original_forward: setattr(
                _block,
                "forward",
                _original,
            )
        )

    return restore_actions


def _restore_patches(actions: list[Any]) -> None:
    for action in reversed(actions):
        action()


def _mean_hand_velocity_ratio(
    numerator: dict[str, Any], denominator: dict[str, Any],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for hand in ("L_hand", "R_hand"):
        num = numerator["per_joint_vel_cm_per_frame"][hand]["mean"]
        den = denominator["per_joint_vel_cm_per_frame"][hand]["mean"]
        out[hand] = _safe_div(float(num), float(den))
    out["mean"] = float(np.mean([out["L_hand"], out["R_hand"]]))
    return out


def _external_motion_metrics(
    gt_joints: Tensor,
    sample_joints: Tensor,
    recon_joints: Tensor,
    seq_masks: Tensor,
    fps: float,
) -> dict[str, Any]:
    gt_joint = _per_joint_vel_stats(gt_joints, seq_masks)
    sample_joint = _per_joint_vel_stats(sample_joints, seq_masks)
    recon_joint = _per_joint_vel_stats(recon_joints, seq_masks)
    gt_fft = _fft_band_energy(gt_joints, seq_masks, fps=fps)
    sample_fft = _fft_band_energy(sample_joints, seq_masks, fps=fps)
    recon_fft = _fft_band_energy(recon_joints, seq_masks, fps=fps)

    gt_vel = gt_joint["body_local_vel_cm_per_frame"]["mean"]
    sample_vel = sample_joint["body_local_vel_cm_per_frame"]["mean"]
    recon_vel = recon_joint["body_local_vel_cm_per_frame"]["mean"]
    gt_acc_p95 = gt_joint["body_local_acc_cm_per_frame"]["p95"]
    sample_acc_p95 = sample_joint["body_local_acc_cm_per_frame"]["p95"]
    recon_acc_p95 = recon_joint["body_local_acc_cm_per_frame"]["p95"]

    return {
        "gt": {
            "joint_vel_stats": gt_joint,
            "fft_spectrum": gt_fft,
        },
        "sampled": {
            "joint_vel_stats": sample_joint,
            "fft_spectrum": sample_fft,
        },
        "recon_one_step": {
            "joint_vel_stats": recon_joint,
            "fft_spectrum": recon_fft,
        },
        "ratios": {
            "sample_body_local_velocity_xgt": _safe_div(sample_vel, gt_vel),
            "recon_body_local_velocity_xgt": _safe_div(recon_vel, gt_vel),
            "recon_minus_sample_velocity_gap_xgt": _safe_div(recon_vel, gt_vel)
            - _safe_div(sample_vel, gt_vel),
            "sample_body_local_acceleration_p95_xgt": _safe_div(
                sample_acc_p95,
                gt_acc_p95,
            ),
            "recon_body_local_acceleration_p95_xgt": _safe_div(
                recon_acc_p95,
                gt_acc_p95,
            ),
            "sample_hand_velocity_xgt": _mean_hand_velocity_ratio(sample_joint, gt_joint),
            "recon_hand_velocity_xgt": _mean_hand_velocity_ratio(recon_joint, gt_joint),
            "sample_fft_mid_fraction": float(sample_fft["fraction_mid"]),
            "recon_fft_mid_fraction": float(recon_fft["fraction_mid"]),
            "gt_fft_mid_fraction": float(gt_fft["fraction_mid"]),
        },
    }


def _sort_block_rows(rows: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return sorted(rows.items(), key=lambda item: item[0])


def _interpret_case(summary: dict[str, Any]) -> dict[str, Any]:
    hidden = summary["internal_diagnostic"]["hidden_states"]
    attention = summary["internal_diagnostic"]["attention_maps"]

    entry = hidden.get("input_projection", {})
    block0 = hidden.get("block_00_after_mlp", entry)
    deep = hidden.get("block_07_after_mlp", block0)
    final = hidden.get("final_head_input", deep)

    velocity_ratio_deep_vs_block0 = _safe_div(
        float(deep.get("temporal_velocity_mean", 0.0)),
        float(block0.get("temporal_velocity_mean", 0.0)),
    )
    mid_ratio_deep_vs_block0 = _safe_div(
        float(deep.get("fft_fraction_mid", 0.0)),
        float(block0.get("fft_fraction_mid", 0.0)),
    )
    velocity_ratio_final_vs_entry = _safe_div(
        float(final.get("temporal_velocity_mean", 0.0)),
        float(entry.get("temporal_velocity_mean", 0.0)),
    )
    mid_ratio_final_vs_entry = _safe_div(
        float(final.get("fft_fraction_mid", 0.0)),
        float(entry.get("fft_fraction_mid", 0.0)),
    )

    block_attention = list(attention.values())
    avg_entropy_norm = float(
        np.mean([float(x["attention_entropy_normalized"]) for x in block_attention])
    )
    avg_local_pm10 = float(
        np.mean([float(x["local_attention_mass_pm10"]) for x in block_attention])
    )
    avg_query_similarity = float(
        np.mean([float(x["query_distribution_cosine_to_mean"]) for x in block_attention])
    )
    attention_global_like = (
        avg_entropy_norm >= 0.80
        and avg_local_pm10 <= 0.55
        and avg_query_similarity >= 0.90
    )

    # The user explicitly asked us not to conflate "global-looking attention"
    # with "self-attn smooths the hidden state". In v18 the deepest blocks can
    # become diffuse even when the hidden-state dynamics have already settled.
    # Track that late-stack condition separately so Case C can fire when it
    # should.
    late_attention_rows = [
        attention[name]
        for name in ("block_04", "block_05", "block_06", "block_07")
        if name in attention
    ]
    late_entropy_norm = float(
        np.mean([float(x["attention_entropy_normalized"]) for x in late_attention_rows])
    ) if late_attention_rows else 0.0
    late_local_pm10 = float(
        np.mean([float(x["local_attention_mass_pm10"]) for x in late_attention_rows])
    ) if late_attention_rows else 0.0
    late_query_similarity = float(
        np.mean([float(x["query_distribution_cosine_to_mean"]) for x in late_attention_rows])
    ) if late_attention_rows else 0.0
    late_attention_global_like = (
        late_entropy_norm >= 0.85
        and late_local_pm10 <= 0.30
        and late_query_similarity >= 0.80
    )

    late_block4 = hidden.get("block_04_after_mlp", deep)
    late_block7 = hidden.get("block_07_after_mlp", deep)
    late_velocity_ratio = _safe_div(
        float(late_block7.get("temporal_velocity_mean", 0.0)),
        float(late_block4.get("temporal_velocity_mean", 0.0)),
    )
    late_mid_ratio = _safe_div(
        float(late_block7.get("fft_fraction_mid", 0.0)),
        float(late_block4.get("fft_fraction_mid", 0.0)),
    )
    late_hidden_drop = late_velocity_ratio <= 0.90 and late_mid_ratio <= 0.90

    hidden_drop = (
        velocity_ratio_deep_vs_block0 <= 0.85
        and mid_ratio_deep_vs_block0 <= 0.85
    )
    if hidden_drop:
        case = "Case A"
        headline = (
            "DiT block outputs lose temporal velocity and mid-band energy across depth; "
            "a trained-model temporal smoothing bias remains plausible."
        )
    elif late_attention_global_like and not late_hidden_drop:
        case = "Case C"
        headline = (
            "Late self-attention maps look global/diffuse, but late-stack hidden temporal "
            "dynamics do not continue collapsing; attention maps alone do not explain "
            "frozen motion."
        )
    else:
        case = "Case B"
        headline = (
            "Hidden temporal dynamics do not show a strong depth-wise smoothing collapse; "
            "self-attention is not the leading explanation for the remaining frozen-body gap."
        )

    return {
        "case": case,
        "headline": headline,
        "hidden_drop": bool(hidden_drop),
        "attention_global_like": bool(attention_global_like),
        "velocity_ratio_deep_vs_block0": velocity_ratio_deep_vs_block0,
        "mid_fft_ratio_deep_vs_block0": mid_ratio_deep_vs_block0,
        "velocity_ratio_final_vs_entry": velocity_ratio_final_vs_entry,
        "mid_fft_ratio_final_vs_entry": mid_ratio_final_vs_entry,
        "attention_entropy_norm_mean": avg_entropy_norm,
        "attention_local_mass_pm10_mean": avg_local_pm10,
        "attention_query_similarity_mean": avg_query_similarity,
        "late_attention_entropy_norm_mean": late_entropy_norm,
        "late_attention_local_mass_pm10_mean": late_local_pm10,
        "late_attention_query_similarity_mean": late_query_similarity,
        "late_attention_global_like": bool(late_attention_global_like),
        "late_hidden_velocity_ratio_block7_vs_block4": late_velocity_ratio,
        "late_hidden_mid_fft_ratio_block7_vs_block4": late_mid_ratio,
        "late_hidden_drop": bool(late_hidden_drop),
    }


def _write_markdown(path: Path, results: dict[str, Any]) -> None:
    attention = results["internal_diagnostic"]["attention_maps"]
    hidden = results["internal_diagnostic"]["hidden_states"]
    branches = results["internal_diagnostic"]["residual_branches"]
    external = results["external_motion_metrics"]
    ratios = external["ratios"]
    interpretation = results["interpretation"]

    lines: list[str] = []
    lines.append("# v18 Attention Temporal Smoothing Diagnostic")
    lines.append("")
    lines.append(f"**Config:** `{results['config']}`")
    lines.append(f"**Checkpoint:** `{results['ckpt']}`")
    lines.append(
        f"**Clips:** {results['num_clips']}    **bucket:** {results['bucket']}    "
        f"**recon_t:** {results['recon_t']}    **cfg_scale:** {results['cfg_scale']}"
    )
    lines.append("")
    lines.append(
        "Internal hook statistics are measured on one-step reconstruction forwards at "
        f"`t={results['recon_t']}`. Motion-token temporal statistics exclude the prepended "
        "`init_pose` token. Full DDPM sampling is run separately, with the diagnostic "
        "patch disabled, for external frozen-body readouts."
    )

    lines.append("")
    lines.append("## 1. Self-attention temporal map statistics")
    lines.append("")
    lines.append(
        "| block | entropy | entropy norm | local ±1 | local ±3 | local ±5 | local ±10 | "
        "global >10 | mean |Δt| | diag mass | diag × uniform | query dist cosine |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, row in _sort_block_rows(attention):
        lines.append(
            f"| {name} | {row['attention_entropy']:.3f} | "
            f"{row['attention_entropy_normalized']:.3f} | "
            f"{row['local_attention_mass_pm1']:.3f} | "
            f"{row['local_attention_mass_pm3']:.3f} | "
            f"{row['local_attention_mass_pm5']:.3f} | "
            f"{row['local_attention_mass_pm10']:.3f} | "
            f"{row['global_attention_mass_outside_pm10']:.3f} | "
            f"{row['mean_attended_temporal_distance']:.2f} | "
            f"{row['diagonal_mass']:.4f} | "
            f"{row['diagonal_uniform_ratio']:.2f} | "
            f"{row['query_distribution_cosine_to_mean']:.3f} |"
        )

    lines.append("")
    lines.append("## 2. Hidden-state temporal smoothing statistics")
    lines.append("")
    lines.append("| stage | temporal vel | temporal acc | FFT low | FFT mid | FFT high |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    ordered_hidden = ["input_projection"]
    for idx in range(8):
        prefix = f"block_{idx:02d}"
        ordered_hidden.extend(
            [
                f"{prefix}_input",
                f"{prefix}_after_self_attn",
                f"{prefix}_after_plan_xattn",
                f"{prefix}_after_mlp",
            ]
        )
    ordered_hidden.extend(["final_head_input", "x0_pred_recon"])
    for name in ordered_hidden:
        row = hidden.get(name)
        if row is None:
            continue
        lines.append(
            f"| {name} | {row['temporal_velocity_mean']:.4f} | "
            f"{row['temporal_acceleration_mean']:.4f} | "
            f"{row['fft_fraction_low']:.3f} | "
            f"{row['fft_fraction_mid']:.3f} | "
            f"{row['fft_fraction_high']:.3f} |"
        )

    lines.append("")
    lines.append("## 3. Residual-branch temporal contribution")
    lines.append("")
    lines.append("| branch | norm | temporal vel | temporal acc | FFT mid |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, row in _sort_block_rows(branches):
        lines.append(
            f"| {name} | {row.get('norm_mean', 0.0):.4f} | "
            f"{row['temporal_velocity_mean']:.4f} | "
            f"{row['temporal_acceleration_mean']:.4f} | "
            f"{row['fft_fraction_mid']:.3f} |"
        )

    lines.append("")
    lines.append("## 4. Recon vs sample comparison")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    lines.append(
        f"| sampled body local velocity ×GT | "
        f"{ratios['sample_body_local_velocity_xgt']:.3f} |"
    )
    lines.append(
        f"| recon body local velocity ×GT | "
        f"{ratios['recon_body_local_velocity_xgt']:.3f} |"
    )
    lines.append(
        f"| recon - sample velocity gap ×GT | "
        f"{ratios['recon_minus_sample_velocity_gap_xgt']:.3f} |"
    )
    lines.append(
        f"| sampled hand velocity ×GT (mean L/R) | "
        f"{ratios['sample_hand_velocity_xgt']['mean']:.3f} |"
    )
    lines.append(
        f"| recon hand velocity ×GT (mean L/R) | "
        f"{ratios['recon_hand_velocity_xgt']['mean']:.3f} |"
    )
    lines.append(
        f"| sampled acceleration p95 ×GT | "
        f"{ratios['sample_body_local_acceleration_p95_xgt']:.3f} |"
    )
    lines.append(
        f"| recon acceleration p95 ×GT | "
        f"{ratios['recon_body_local_acceleration_p95_xgt']:.3f} |"
    )
    lines.append(f"| GT FFT mid-band fraction | {ratios['gt_fft_mid_fraction']:.3f} |")
    lines.append(f"| sampled FFT mid-band fraction | {ratios['sample_fft_mid_fraction']:.3f} |")
    lines.append(f"| recon FFT mid-band fraction | {ratios['recon_fft_mid_fraction']:.3f} |")

    lines.append("")
    lines.append("## 5. Interpretation")
    lines.append("")
    lines.append(f"**Verdict:** **{interpretation['case']}**. {interpretation['headline']}")
    lines.append("")
    lines.append("| diagnostic summary | value |")
    lines.append("|---|---:|")
    lines.append(
        f"| deep/block0 hidden temporal velocity ratio | "
        f"{interpretation['velocity_ratio_deep_vs_block0']:.3f} |"
    )
    lines.append(
        f"| deep/block0 hidden FFT mid-band ratio | "
        f"{interpretation['mid_fft_ratio_deep_vs_block0']:.3f} |"
    )
    lines.append(
        f"| final-head/input-proj hidden temporal velocity ratio | "
        f"{interpretation['velocity_ratio_final_vs_entry']:.3f} |"
    )
    lines.append(
        f"| final-head/input-proj hidden FFT mid-band ratio | "
        f"{interpretation['mid_fft_ratio_final_vs_entry']:.3f} |"
    )
    lines.append(
        f"| mean normalized self-attn entropy | "
        f"{interpretation['attention_entropy_norm_mean']:.3f} |"
    )
    lines.append(
        f"| mean local self-attn mass within ±10 | "
        f"{interpretation['attention_local_mass_pm10_mean']:.3f} |"
    )
    lines.append(
        f"| mean query-distribution cosine similarity | "
        f"{interpretation['attention_query_similarity_mean']:.3f} |"
    )
    lines.append(
        f"| late-stack normalized self-attn entropy (blocks 4-7) | "
        f"{interpretation['late_attention_entropy_norm_mean']:.3f} |"
    )
    lines.append(
        f"| late-stack local self-attn mass within ±10 (blocks 4-7) | "
        f"{interpretation['late_attention_local_mass_pm10_mean']:.3f} |"
    )
    lines.append(
        f"| late-stack query cosine similarity (blocks 4-7) | "
        f"{interpretation['late_attention_query_similarity_mean']:.3f} |"
    )
    lines.append(
        f"| late hidden velocity ratio block7 / block4 | "
        f"{interpretation['late_hidden_velocity_ratio_block7_vs_block4']:.3f} |"
    )
    lines.append(
        f"| late hidden FFT mid-band ratio block7 / block4 | "
        f"{interpretation['late_hidden_mid_fft_ratio_block7_vs_block4']:.3f} |"
    )

    if interpretation["case"] == "Case A":
        lines.append("")
        lines.append(
            "The evidence supports a remaining architectural temporal-smoothing bias. "
            "The smallest worthwhile full-data ablation is a **post-plan-xattn local "
            "temporal module**: keep the existing A1/v18 path unchanged through "
            "`plan_xattn`, then add a tiny zero-gated local Conv1D or local-attention "
            "residual after plan routing. This avoids the v13 failure mode where the "
            "temporal module sat before plan cross-attention and damaged routing."
        )
    elif interpretation["case"] == "Case C":
        lines.append("")
        lines.append(
            "The self-attention maps are diffuse enough that they look averaging-like, "
            "but hidden-state temporal velocity / mid-band energy do **not** collapse "
            "through the stack. This is exactly the case where attention maps by "
            "themselves are not causal evidence. The next path should stay focused on "
            "full-data scaling / augmentation / denoising-objective work rather than "
            "restarting temporal Conv1D solely from attention heatmaps."
        )
    else:
        lines.append("")
        lines.append(
            "The hidden-state stack does not show the strong monotonic temporal flattening "
            "needed to blame self-attention as the main frozen-body source. The remaining "
            "gap is better aligned with the existing multimodality and iterative-denoising "
            "diagnosis. Prioritize the data/objective path over a full-data temporal-conv "
            "revival."
        )

    lines.append("")
    lines.append("## 6. Clear decision")
    lines.append("")
    if interpretation["case"] == "Case A":
        lines.append("1. **Self-attn temporal smoothing:** supported on v18.")
        lines.append("2. **Restart temporal conv / local attention on full data:** yes, but only as a minimal post-plan-routing ablation.")
        lines.append("3. **Smallest ablation:** zero-gated depthwise temporal Conv1D after `plan_xattn` inside each DiT block, no dynamics head, no change to plan pool.")
        lines.append("4. **Next path after that:** compare against v18 on the same dynamics + plan diagnostics before touching broader mode-collapse losses.")
    else:
        lines.append("1. **Self-attn temporal smoothing:** not established as the main v18 bottleneck.")
        lines.append("2. **Restart temporal conv / local attention on full data:** not yet justified.")
        lines.append("3. **Smallest ablation if revisited later:** one post-plan-xattn local temporal residual on v18 only, not the old pre-plan v13 placement.")
        lines.append("4. **Preferred next path now:** continue with the data/objective axis: all-7 full data or augmentation first; keep `v-pred + min-SNR` as a bounded parameterization follow-up, not as the lead hypothesis.")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--md", type=Path, required=True)
    parser.add_argument("--num-clips", type=int, default=8)
    parser.add_argument("--bucket", default="train", choices=["train", "val"])
    parser.add_argument("--recon-t", type=int, default=100)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=20.0)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    dataset = _build_dataset(cfg, args.bucket)
    dataset = Subset(dataset, list(range(min(args.num_clips, len(dataset)))))
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_hoi,
        num_workers=0,
    )

    model, object_encoder, z_dims = _build_model(cfg, device)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("model", state))
    if "object_encoder" in state:
        object_encoder.load_state_dict(state["object_encoder"])
    elif "extra_modules" in state and "object_encoder" in state["extra_modules"]:
        object_encoder.load_state_dict(state["extra_modules"]["object_encoder"])

    clip_model = load_clip_text_encoder(
        device=device,
        model_name=str(cfg.model.text_encoder.clip_version),
        download_root=str(cfg.model.text_encoder.get("download_root", "cache/clip")),
    )

    recorder = DiagnosticRecorder(fps=args.fps)
    restore_actions = _install_diagnostic_patches(model, recorder)

    all_gt_joints: list[Tensor] = []
    all_sample_joints: list[Tensor] = []
    all_recon_joints: list[Tensor] = []
    all_seq_masks: list[Tensor] = []
    per_clip: list[dict[str, Any]] = []

    try:
        for i, batch in enumerate(loader):
            cond, total_t = _build_cond(
                batch,
                model,
                object_encoder,
                clip_model,
                z_dims,
                cfg,
                device,
            )
            plan = _extract_plan(batch, device)
            cond_full = {**cond, "interaction_plan": plan}

            motion_gt = batch["motion"].to(device).float()
            rest_offsets = batch["rest_offsets"].to(device).float()
            seq_len = batch["seq_len"].to(device)
            seq_idx = torch.arange(total_t, device=device).unsqueeze(0)
            seq_mask = seq_idx < seq_len.unsqueeze(1)
            joints_gt = batch["joints"].to(device).float()

            # Hooked one-step reconstruction at a fixed diffusion step.
            torch.manual_seed(args.seed + 1000 + i)
            t_recon = torch.full(
                (1,),
                int(args.recon_t),
                dtype=torch.long,
                device=device,
            )
            noise = torch.randn_like(motion_gt)
            x_t = model.diffusion.q_sample(motion_gt, t_recon, noise)
            recorder.begin(seq_mask)
            with torch.no_grad():
                x0_recon = model.denoiser(
                    x_t,
                    t_recon,
                    cond_full,
                    cond_drop_mask=None,
                )
            recorder.record_hidden("x0_pred_recon", x0_recon)
            recorder.end()
            joints_recon = _fk_from_motion_135(x0_recon, rest_offsets)

            # Normal DDPM sample. Diagnostic wrappers stay installed but recorder
            # is disabled, so each block routes to its original forward method.
            torch.manual_seed(args.seed + i)
            with torch.no_grad():
                x0_sample = model.sample(
                    cond=cond_full,
                    seq_length=total_t,
                    cfg_scale=args.cfg_scale,
                    replacement="none",
                    output_skip=False,
                )
            joints_sample = _fk_from_motion_135(x0_sample, rest_offsets)

            all_gt_joints.append(joints_gt)
            all_sample_joints.append(joints_sample)
            all_recon_joints.append(joints_recon)
            all_seq_masks.append(seq_mask)
            per_clip.append(
                {
                    "subset": batch["subset"][0],
                    "seq_id": batch["seq_id"][0],
                    "seq_len": int(seq_len.item()),
                    "text": batch["text"][0][:120],
                }
            )
            print(
                f"  [{i + 1}/{len(loader)}] "
                f"{batch['subset'][0]}/{batch['seq_id'][0]} "
                f"T={int(seq_len.item())}"
            )
    finally:
        recorder.end()
        _restore_patches(restore_actions)

    gt_joints = torch.cat(all_gt_joints, dim=0)
    sample_joints = torch.cat(all_sample_joints, dim=0)
    recon_joints = torch.cat(all_recon_joints, dim=0)
    seq_masks = torch.cat(all_seq_masks, dim=0)

    results: dict[str, Any] = {
        "config": str(args.config),
        "ckpt": str(args.ckpt),
        "bucket": str(args.bucket),
        "num_clips": int(gt_joints.shape[0]),
        "recon_t": int(args.recon_t),
        "cfg_scale": float(args.cfg_scale),
        "seed": int(args.seed),
        "fps": float(args.fps),
        "per_clip": per_clip,
        "internal_diagnostic": recorder.summary(),
        "external_motion_metrics": _external_motion_metrics(
            gt_joints,
            sample_joints,
            recon_joints,
            seq_masks,
            fps=args.fps,
        ),
    }
    results["interpretation"] = _interpret_case(results)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    _write_markdown(args.md, results)

    print(f"\nWrote JSON to {args.output}")
    print(f"Wrote Markdown to {args.md}")
    print("\n=== Diagnostic verdict ===")
    print(f"  {results['interpretation']['case']}: {results['interpretation']['headline']}")
    print(
        "  sample vel ×GT="
        f"{results['external_motion_metrics']['ratios']['sample_body_local_velocity_xgt']:.3f}, "
        "recon vel ×GT="
        f"{results['external_motion_metrics']['ratios']['recon_body_local_velocity_xgt']:.3f}, "
        "gap="
        f"{results['external_motion_metrics']['ratios']['recon_minus_sample_velocity_gap_xgt']:.3f}"
    )


if __name__ == "__main__":
    main()
