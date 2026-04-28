"""Residual transformer wrapper with z_int K/V conditioning (Stage B C1).

Why this exists
---------------

B3 v5 (2026-04-28) sweep on v0.6 b1_bestval found that base-token logit
guidance is **directionally uncontrolled**: the residual transformer
(5/6 of decoded capacity) is text-only at inference and propagates base
flips through 5 layers of autoregressive sampling, sometimes aligning
with the contact target (largebox −14 cm) and sometimes catastrophically
diverging (plasticbox_037 +16 cm). Same mechanism produces both biggest
win and biggest loss → no inference-time fix can ship.

C1 makes the residual transformer's 5 layers see ``z_int`` (per-frame
contact / phase / support / object pose channels), turning text-only
autoregressive cascades into z_int-conditioned cascades whose direction
is aligned with contact targets. This is the InterControl/MaskControl
recipe applied to the residual stage:

- One IntXAttn(z_int K/V) sublayer per residual transformer block.
- ``γ_int_res`` per-block (or per-head) gate, **zero-init** so the
  wrapped residual is byte-identical to the original at training step 0.
- During training, ``γ_int_res`` grows as residual learns to use z_int.

Architectural choice (mirror v0.6 SOTA, not v0.3-δ)
---------------------------------------------------

We adopt v0.6's per-block IntXAttn + γ-gate pattern (the ckpt of record,
mean_min_dist 16.0 cm pre-B1) rather than v0.3-δ's trainable-copy +
zero-init connector (regressed +14.92 cm vs v0.6, see
analyses/2026-04-28_v0_3_delta_retrain_and_v0_5_contact.md).

Reuse strategy
--------------

ResidualTransformer's inner ``self.seqTransEncoder`` is structurally
identical to MaskTransformer's encoder (an ``nn.TransformerEncoder``
with ``nn.TransformerEncoderLayer`` blocks at the same d_model / num_heads).
So we can directly reuse :class:`MaskTransformerEncoderWithInteraction`
from ``motion_generator.py`` — the wrapper holds references to original
layers without deepcopy, preserving pretrained weights byte-exactly.

The only thing this module adds is method-level threading of ``int_kv``
through ResidualTransformer's forward/generate path, since the original
methods don't accept ``int_kv``.

References
----------

- v0.6 per-block IntXAttn + γ-gate: ``src/piano/models/motion_generator.py``
  ``MaskTransformerBlockWithInteraction``.
- MaskControl/ControlMM: Pinyoanuntapong et al. ICCV 2025. arXiv:2410.10780.
- InterControl: Wang et al. NeurIPS 2024. arXiv:2311.15864.
- analyses/2026-04-28_b1_b3_iteration_log.md §"v5 — Sweep results" for
  empirical motivation.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from piano.models.motion_generator import MaskTransformerEncoderWithInteraction


def _residual_layer_metrics(
    logits: Tensor,
    labels: Tensor,
    active_q_layers: Tensor,
    *,
    pad_id: int,
    num_quant_layers: int,
) -> dict[str, Tensor]:
    """Compute per-residual-layer CE/accuracy for sampled active layers."""
    with torch.no_grad():
        logits_f = logits.detach().float()
        labels_d = labels.detach()
        active_q = active_q_layers.detach()
        valid = labels_d.ne(pad_id)
        per_token_loss = F.cross_entropy(
            logits_f, labels_d, ignore_index=pad_id, reduction="none",
        )
        pred = logits_f.argmax(dim=1)

        metrics: dict[str, Tensor] = {}
        for q in range(1, int(num_quant_layers)):
            q_mask = active_q.eq(q).view(-1, 1)
            mask = valid & q_mask
            count = mask.sum()
            if int(count.item()) == 0:
                continue
            count_f = count.to(dtype=logits_f.dtype)
            suffix = f"q{q}"
            metrics[f"loss_residual_{suffix}"] = (
                per_token_loss.masked_select(mask).sum() / count_f
            )
            metrics[f"acc_residual_{suffix}"] = (
                pred.eq(labels_d).masked_select(mask).float().mean()
            )
            metrics[f"tokens_residual_{suffix}"] = count_f
        return metrics


class ResidualTransformerWithInteraction(nn.Module):
    """ResidualTransformer wrapper that adds z_int K/V conditioning.

    Wraps a vendored MoMask ``ResidualTransformer`` instance in-place by
    replacing its inner ``seqTransEncoder`` with a
    :class:`MaskTransformerEncoderWithInteraction`. The wrapped encoder
    holds references to the original transformer layers (no deepcopy),
    preserving pretrained weights. New parameters added: per-block
    ``norm_int`` + ``int_attn`` + ``gamma_int`` (zero-init).

    Backward-compat: the original's ``forward(all_indices, y, m_lens)``
    and ``generate(motion_ids, conds, m_lens, ...)`` methods continue
    to work unchanged — they call ``self.seqTransEncoder(xseq,
    src_key_padding_mask=...)`` which the wrapped encoder accepts via
    ``int_kv=None`` default. With ``γ_int_res = 0``, those calls are
    byte-identical to the un-wrapped original.

    z_int-aware methods (``forward_with_int``, ``generate_with_int``)
    take additional ``int_kv`` + ``int_padding_mask`` arguments and
    thread them through ``trans_forward`` → wrapped ``seqTransEncoder``
    so every residual layer's IntXAttn sublayer sees the interaction
    K/V.

    Parameters
    ----------
    residual_transformer : MoMask ``ResidualTransformer`` instance.
    d_model : encoder latent dim. Must equal ``residual_transformer.latent_dim``.
    num_heads : self-attn head count. Must equal the heads in
        ``residual_transformer.seqTransEncoder.layers[0].self_attn``.
    dropout : dropout for the new IntXAttn + ``norm_int``.
    zero_init_gamma : if True (default), ``γ_int_res = 0`` at init —
        the wrapped residual is byte-identical to the original. If
        False, ``γ_int_res = 1`` (matches v0.3-δ trainable-copy default
        we landed in commit ``6deea63``; useful for ablation).
    gamma_kind : ``"scalar"`` or ``"per_head"``. v0.6 SOTA used
        per-head; default per-head here for parity.
    """

    def __init__(
        self,
        residual_transformer: nn.Module,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        zero_init_gamma: bool = True,
        gamma_kind: str = "per_head",
    ) -> None:
        super().__init__()

        # Sanity-check d_model + num_heads match the original.
        actual_d = int(residual_transformer.latent_dim)
        if actual_d != int(d_model):
            raise ValueError(
                f"d_model={d_model} doesn't match "
                f"residual_transformer.latent_dim={actual_d}",
            )
        first_layer = residual_transformer.seqTransEncoder.layers[0]
        actual_h = int(first_layer.self_attn.num_heads)
        if actual_h != int(num_heads):
            raise ValueError(
                f"num_heads={num_heads} doesn't match self_attn.num_heads={actual_h}",
            )

        # Wrap the inner encoder. Holds refs to original layers (no copy).
        wrapped_encoder = MaskTransformerEncoderWithInteraction(
            residual_transformer.seqTransEncoder,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            zero_init_gamma=zero_init_gamma,
            gamma_kind=gamma_kind,
        )
        # In-place rebind so the original's methods (forward, generate,
        # trans_forward) call our wrapped encoder. The wrapped encoder's
        # default int_kv=None makes those calls byte-identical to the
        # un-wrapped original at γ_int_res=0.
        residual_transformer.seqTransEncoder = wrapped_encoder

        self.residual = residual_transformer
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.gamma_kind = str(gamma_kind)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def encoder(self) -> MaskTransformerEncoderWithInteraction:
        """Convenience accessor to the wrapped encoder (for diagnostics).

        Returns the wrapped ``MaskTransformerEncoderWithInteraction`` so
        callers can inspect ``.layers[i].gamma_int`` directly without
        digging through ``self.residual.seqTransEncoder``.
        """
        return self.residual.seqTransEncoder       # type: ignore[return-value]

    def parameters_wo_clip(self) -> list[nn.Parameter]:
        """Mirror :meth:`ResidualTransformer.parameters_wo_clip`."""
        return [
            p for name, p in self.named_parameters()
            if "clip_model." not in name
        ]

    # ------------------------------------------------------------------
    # Param group splits for the two-LR optimizer (mirrors the base
    # InteractionMaskTransformer.new_parameters / .backbone_parameters
    # convention so the same `build_two_group_optimizer` machinery can
    # consume them).
    # ------------------------------------------------------------------

    def _is_new_residual_param(self, name: str) -> bool:
        """Identify "new" params added by C1 wrapping.

        New params live inside the wrapped seqTransEncoder's per-block
        IntXAttn additions:
        - ``residual.seqTransEncoder.layers.<i>.norm_int.*``
        - ``residual.seqTransEncoder.layers.<i>.int_attn.*``
        - ``residual.seqTransEncoder.layers.<i>.gamma_int``

        Everything else under ``residual.*`` (input_process, position_enc,
        output_process, embeddings, cond_emb, the wrapped layer's original
        self_attn/FFN/norms, CLIP, etc.) is treated as "backbone".
        """
        # We're called with names from self.named_parameters(), so they're
        # rooted at "residual." (the only submodule of self).
        # Markers: the IntXAttn additions are inside .seqTransEncoder.layers.
        if "seqTransEncoder.layers." not in name:
            return False
        # Within a layer's submodules:
        #   .layer.*           = original self_attn/FFN/norms (backbone)
        #   .norm_int.*        = new
        #   .int_attn.*        = new
        #   .gamma_int         = new
        if ".layer." in name:
            return False
        return any(
            marker in name
            for marker in (".norm_int.", ".int_attn.", ".gamma_int")
        )

    def new_parameters(self) -> list[nn.Parameter]:
        """Per-block norm_int + int_attn + gamma_int (the C1 additions)."""
        return [
            p for name, p in self.named_parameters()
            if self._is_new_residual_param(name)
            and "clip_model." not in name
        ]

    def backbone_parameters(self) -> list[nn.Parameter]:
        """Original residual transformer weights (excluding CLIP)."""
        return [
            p for name, p in self.named_parameters()
            if not self._is_new_residual_param(name)
            and "clip_model." not in name
        ]

    # ------------------------------------------------------------------
    # Drop-in passthroughs for code that still expects a raw
    # ResidualTransformer instance. C1-specific paths should call
    # forward_with_int / generate_with_int below.
    # ------------------------------------------------------------------

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate to the wrapped residual transformer's original forward."""
        return self.residual(*args, **kwargs)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate to the wrapped residual transformer's original generate."""
        return self.residual.generate(*args, **kwargs)

    # ------------------------------------------------------------------
    # z_int-aware trans_forward / forward / generate
    # ------------------------------------------------------------------

    def trans_forward_with_int(
        self,
        motion_codes: Tensor,                      # (B, S, d) — RVQ token embeddings cumsum
        qids: Tensor,                              # (B,) — quantizer layer ids
        cond: Tensor,                              # (B, clip_dim) text or (B, num_actions)
        padding_mask: Tensor,                      # (B, S) — TRUE = pad
        *,
        int_kv: Tensor | None = None,              # (S_int, B, d) — z_int K/V
        int_padding_mask: Tensor | None = None,    # (B, S_int)
        force_mask: bool = False,                  # text-drop branch for CFG
    ) -> Tensor:
        """Z_int-aware mirror of ``ResidualTransformer.trans_forward``.

        Only adds ``int_kv`` / ``int_padding_mask`` keyword pass-through;
        otherwise byte-identical to the original method (
        ``transformer.py:786-812`` in MoMask vendor).
        """
        r = self.residual
        cond = r.mask_cond(cond, force_mask=force_mask)
        x = r.input_process(motion_codes)                      # (S, B, d)
        q_onehot = r.encode_quant(qids).float().to(x.device)
        q_emb = r.quant_emb(q_onehot).unsqueeze(0)             # (1, B, d)
        cond_emb = r.cond_emb(cond).unsqueeze(0)               # (1, B, d)
        x = r.position_enc(x)
        xseq = torch.cat([cond_emb, q_emb, x], dim=0)          # (S+2, B, d)
        # Pad the front 2 positions (cond + q_emb) as non-pad.
        prefix_mask = padding_mask.new_zeros((padding_mask.shape[0], 2))
        padding_mask_aug = torch.cat([prefix_mask, padding_mask], dim=1)
        # If int_kv provided, also augment its key_padding_mask layout.
        # IntXAttn's K/V are along dim 0 of int_kv (S_int) so its mask
        # is independent of the source-side cond/q_emb prepend.
        output = r.seqTransEncoder(
            xseq,
            int_kv=int_kv,
            src_key_padding_mask=padding_mask_aug,
            int_key_padding_mask=int_padding_mask,
        )[2:]                                                  # (S, B, d) — drop cond + q_emb
        logits = r.output_process(output)                      # (B, code_dim, S)
        return logits

    def forward_with_cond_scale_with_int(
        self,
        motion_codes: Tensor,
        q_id: int,
        cond_vector: Tensor,
        padding_mask: Tensor,
        *,
        int_kv: Tensor | None = None,
        int_padding_mask: Tensor | None = None,
        cond_scale: float = 3.0,
        force_mask: bool = False,
    ) -> Tensor:
        """Z_int-aware mirror of ``ResidualTransformer.forward_with_cond_scale``.

        Single-condition CFG: ``logits = aux_logits + (logits - aux_logits) * cond_scale``
        where ``aux_logits`` is the text-dropped branch. ``int_kv`` is
        kept across both branches (we don't compositionally CFG over int
        in residual stage — that'd require a 3-pass forward and the
        residual stage doesn't get separate int-vs-text scales in
        existing MoMask CFG).
        """
        r = self.residual
        bs = motion_codes.shape[0]
        qids = torch.full((bs,), q_id, dtype=torch.long, device=motion_codes.device)

        if force_mask:
            logits = self.trans_forward_with_int(
                motion_codes, qids, cond_vector, padding_mask,
                int_kv=int_kv, int_padding_mask=int_padding_mask,
                force_mask=True,
            )
            return r.output_project(logits, qids - 1)

        logits = self.trans_forward_with_int(
            motion_codes, qids, cond_vector, padding_mask,
            int_kv=int_kv, int_padding_mask=int_padding_mask,
        )
        logits = r.output_project(logits, qids - 1)
        if cond_scale == 1:
            return logits

        aux_logits = self.trans_forward_with_int(
            motion_codes, qids, cond_vector, padding_mask,
            int_kv=int_kv, int_padding_mask=int_padding_mask,
            force_mask=True,
        )
        aux_logits = r.output_project(aux_logits, qids - 1)
        return aux_logits + (logits - aux_logits) * cond_scale

    def forward_with_int(
        self,
        all_indices: Tensor,                       # (B, S, Q)
        y: Any,                                    # raw text or action labels
        m_lens: Tensor,                            # (B,)
        *,
        int_kv: Tensor | None = None,
        int_padding_mask: Tensor | None = None,
        return_layer_metrics: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        """Z_int-aware mirror of ``ResidualTransformer.forward``.

        Returns ``(ce_loss, pred_id, acc)`` for the active-quantizer-layer
        masked-CE training objective. Identical to the un-wrapped
        ``forward`` except ``int_kv`` is threaded into ``trans_forward``.
        """
        # Lazy imports to avoid einops + MoMask backbone import on
        # CPU-only test paths.
        from einops import repeat
        from piano.models.backbones.momask.models.mask_transformer.tools import (
            cal_performance, q_schedule,
        )
        from piano.models.backbones.momask.models.mask_transformer.transformer import (
            lengths_to_mask,
        )

        r = self.residual
        r.process_embed_proj_weight()
        bs, ntokens, num_quant_layers = all_indices.shape
        device = all_indices.device

        non_pad_mask = lengths_to_mask(m_lens, ntokens)                         # (B, n)
        q_non_pad_mask = repeat(non_pad_mask, "b n -> b n q", q=num_quant_layers)
        all_indices = torch.where(q_non_pad_mask, all_indices, r.pad_id)

        active_q_layers = q_schedule(bs, low=1, high=num_quant_layers, device=device)

        token_embed = repeat(r.token_embed_weight, "q c d -> b c d q", b=bs)
        gather_indices = repeat(
            all_indices[..., :-1], "b n q -> b n d q", d=token_embed.shape[2],
        )
        all_codes = token_embed.gather(1, gather_indices)                       # (B, n, d, q-1)
        cumsum_codes = torch.cumsum(all_codes, dim=-1)
        active_indices = all_indices[torch.arange(bs), :, active_q_layers]
        history_sum = cumsum_codes[torch.arange(bs), :, :, active_q_layers - 1]

        force_mask = False
        if r.cond_mode == "text":
            with torch.no_grad():
                cond_vector = r.encode_text(y)
        elif r.cond_mode == "action":
            cond_vector = r.enc_action(y).to(device).float()
        elif r.cond_mode == "uncond":
            cond_vector = torch.zeros(bs, r.latent_dim).float().to(device)
            force_mask = True
        else:
            raise NotImplementedError(f"Unsupported cond_mode {r.cond_mode!r}")

        logits = self.trans_forward_with_int(
            history_sum, active_q_layers, cond_vector, ~non_pad_mask,
            int_kv=int_kv, int_padding_mask=int_padding_mask,
            force_mask=force_mask,
        )
        logits = r.output_project(logits, active_q_layers - 1)
        ce_loss, pred_id, acc = cal_performance(
            logits, active_indices, ignore_index=r.pad_id,
        )
        if return_layer_metrics:
            layer_metrics = _residual_layer_metrics(
                logits,
                active_indices,
                active_q_layers,
                pad_id=int(r.pad_id),
                num_quant_layers=int(num_quant_layers),
            )
            return ce_loss, pred_id, acc, layer_metrics
        return ce_loss, pred_id, acc

    @torch.no_grad()
    def generate_with_int(
        self,
        motion_ids: Tensor,                        # (B, S) base-layer ids
        conds: Any,                                # raw text or action labels
        m_lens: Tensor,                            # (B,)
        *,
        int_kv: Tensor | None = None,
        int_padding_mask: Tensor | None = None,
        temperature: float = 1.0,
        topk_filter_thres: float = 0.9,
        cond_scale: float = 2.0,
        num_res_layers: int = -1,
    ) -> Tensor:
        """Z_int-aware mirror of ``ResidualTransformer.generate``.

        Iteratively samples the 5 residual quantizer layers; at each
        layer's ``forward_with_cond_scale`` call we pass ``int_kv`` so
        every residual layer's IntXAttn sublayer sees the interaction
        K/V. Output shape ``(B, S, Q)`` with ``-1`` at pad positions.
        """
        from einops import repeat
        from piano.models.backbones.momask.models.mask_transformer.tools import (
            gumbel_sample, top_k,
        )
        from piano.models.backbones.momask.models.mask_transformer.transformer import (
            lengths_to_mask,
        )

        r = self.residual
        r.process_embed_proj_weight()
        device = next(r.parameters()).device
        seq_len = motion_ids.shape[1]
        batch_size = len(conds) if not isinstance(conds, Tensor) else conds.shape[0]

        if r.cond_mode == "text":
            with torch.no_grad():
                cond_vector = r.encode_text(conds)
        elif r.cond_mode == "action":
            cond_vector = r.enc_action(conds).to(device)
        elif r.cond_mode == "uncond":
            cond_vector = torch.zeros(batch_size, r.latent_dim).float().to(device)
        else:
            raise NotImplementedError(f"Unsupported cond_mode {r.cond_mode!r}")

        padding_mask = ~lengths_to_mask(m_lens, seq_len)
        motion_ids = torch.where(padding_mask, r.pad_id, motion_ids)
        all_indices = [motion_ids]
        history_sum = 0
        num_quant_layers = (
            r.opt.num_quantizers if num_res_layers == -1 else num_res_layers + 1
        )

        for i in range(1, num_quant_layers):
            token_embed = r.token_embed_weight[i - 1]
            token_embed = repeat(token_embed, "c d -> b c d", b=batch_size)
            gathered_ids = repeat(
                motion_ids, "b n -> b n d", d=token_embed.shape[-1],
            )
            history_sum = history_sum + token_embed.gather(1, gathered_ids)

            logits = self.forward_with_cond_scale_with_int(
                history_sum, i, cond_vector, padding_mask,
                int_kv=int_kv, int_padding_mask=int_padding_mask,
                cond_scale=cond_scale,
            )
            logits = logits.permute(0, 2, 1)                          # (B, S, V)
            filtered_logits = top_k(logits, topk_filter_thres, dim=-1)
            pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)
            ids = torch.where(padding_mask, r.pad_id, pred_ids)
            motion_ids = ids
            all_indices.append(ids)

        all_indices_t = torch.stack(all_indices, dim=-1)
        all_indices_t = torch.where(all_indices_t == r.pad_id, -1, all_indices_t)
        return all_indices_t

    # ------------------------------------------------------------------
    # Original methods still callable via self.residual.forward / .generate
    # When called with int_kv=None semantics (i.e. the original methods
    # don't pass int_kv to seqTransEncoder), the wrapped encoder defaults
    # to int_kv=None → byte-identical to original. Test coverage in
    # tests/test_motion_generator_residual.py.
    # ------------------------------------------------------------------
