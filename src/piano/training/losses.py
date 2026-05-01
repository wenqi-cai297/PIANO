"""Loss functions for all PIANO training stages.

Stage A (Predictor): pseudo-label supervision losses
Stage B (Generator): masked token prediction loss
Stage C (Joint):     predictor + generator + consistency loss
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ============================================================================
# Kendall et al. CVPR 2018 — homoscedastic uncertainty multi-task weighting
# ============================================================================

class KendallTaskWeights(nn.Module):
    """Learnable per-task log-variance weights (Kendall et al. CVPR 2018).

    Total loss is::

        L_total = Σ_i  exp(-s_i) · L_i  +  0.5 · s_i

    where ``s_i = log σ_i^2`` is a learnable scalar per task. At
    optimum ``∂L/∂s_i = 0`` gives ``exp(-s_i) · L_i = 0.5`` so each
    task contributes 0.5 to the total — the optimiser automatically
    rebalances tasks of very different scales (our problem: train
    target loss ≈ 0.018 vs train contact loss ≈ 0.43, a 24× scale gap
    that no fixed weight could compensate cleanly).

    At init ``s_i = 0`` → all tasks are weighted ×1 (which is equal
    to the manual unit-weight starting point). The optimiser learns
    s_i over the first few epochs.

    Reference: Kendall, Gal, Cipolla. "Multi-Task Learning Using
    Uncertainty to Weigh Losses for Scene Geometry and Semantics."
    CVPR 2018.
    """

    def __init__(self, task_names: tuple[str, ...]) -> None:
        super().__init__()
        self.task_names = tuple(task_names)
        # One nn.Parameter per task. Stored as a flat ParameterDict so
        # they're picked up by .parameters() and DDP synchronises them.
        self.log_vars = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(())) for name in task_names}
        )

    def forward(self, raw_losses: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        """Combine per-task losses with learned uncertainty weights.

        Returns ``(total, log_dict)`` where log_dict carries the
        learned weights and log-variances for wandb.
        """
        device = next(self.log_vars.values()).device
        total = torch.zeros((), device=device)
        log: dict[str, Tensor] = {}
        for name in self.task_names:
            if name not in raw_losses:
                continue
            s = self.log_vars[name]
            li = raw_losses[name]
            # Numerically more stable than literal exp(-s) for very
            # negative s: clamp the exponent.
            inv_var = torch.exp(-s.clamp(min=-10.0, max=10.0))
            total = total + inv_var * li + 0.5 * s
            log[f"weight_{name}"] = inv_var.detach()
            log[f"log_var_{name}"] = s.detach()
        return total, log


# ============================================================================
# Stage A: Interaction Predictor losses
# ============================================================================

class PredictorLoss(nn.Module):
    """Combined loss for the Interaction Predictor.

    Supervises contact state, contact target (xyz), interaction phase,
    and support state against pseudo-labels.

    **Target head is xyz regression, not patch classification.** The
    contact-gate is still applied: for (t, body_part) cells where GT
    contact < threshold, the body part isn't touching anything and the
    regression target is undefined — we zero out the per-cell loss and
    normalise by the count of gated cells.

    **Focal loss** on the phase + support CE heads (Lin et al.,
    RetinaNet / ICCV 2017). Our pseudo-label class frequencies are
    severely skewed — phase `pre_contact` is 0.4% of frames, support
    `hand_support` 3%. With naive CE the model ignored rare classes
    entirely (F1 = 0 even on the training set). The focal factor
    ``(1 - p_t)^γ`` down-weights confident / easy frames so gradient
    mass stays on the hard / rare-class ones.
    """

    def __init__(
        self,
        contact_weight: float = 1.0,
        target_weight: float = 0.5,
        phase_weight: float = 0.5,
        support_weight: float = 0.5,
        contact_threshold: float = 0.5,
        label_smoothing: float = 0.0,
        focal_gamma: float = 0.0,
        use_kendall_weights: bool = False,
        target_gate_kind: str = "contact",
        logit_adjust_phase: Tensor | None = None,
        logit_adjust_support: Tensor | None = None,
        logit_adjust_tau: float = 1.0,
        # v8 (2026-05-05): affordance-style target loss + DAG consistency.
        # See analyses/2026-05-05_predictor_v8_design.md Section 3.3.
        target_loss_kind: str = "smooth_l1",
        target_kernel_sigma: float = 0.08,
        consistency_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.contact_weight = contact_weight
        self.target_weight = target_weight
        self.phase_weight = phase_weight
        self.support_weight = support_weight
        self.contact_threshold = contact_threshold
        # Label smoothing on phase / support CE (ViT / T5 / PointNeXt /
        # MoMask convention). Stops the model from over-confidently
        # fitting ~10%-noisy pseudo-labels.
        self.label_smoothing = label_smoothing
        # Focal loss gamma. 0 disables. RetinaNet default is 2.0.
        self.focal_gamma = focal_gamma
        # Kendall et al. CVPR'18 multi-task uncertainty weighting.
        # When True, the static {contact,target,phase,support}_weight
        # values above are IGNORED and the optimiser learns per-task
        # log-variances that auto-balance scale differences. v2 found
        # that fixed weights left the target term contributing only
        # ~2% of total gradient because target loss ≈ 0.018 while
        # contact loss ≈ 0.43; Kendall weights remove that hand-tuning
        # by construction.
        self.use_kendall_weights = use_kendall_weights
        # target_gate_kind selects WHICH frame/part cells contribute to
        # target xyz regression loss:
        #   "contact" — only cells where gt_contact > contact_threshold
        #               (legacy v6 behaviour; appropriate when contact is
        #               dense, e.g. v11 labels with ~70% contact frac)
        #   "all"     — every (frame, part) cell with valid frame mask
        #               regardless of contact state. Justified because
        #               extract_target.py emits closest-surface-point xyz
        #               for every cell (100% non-zero), and that quantity
        #               is well-defined whether or not the body actually
        #               makes contact at this frame. v12 strict labels
        #               drop contact frame frac to ~50%, leaving 50% of
        #               target supervision unused under "contact" gating
        #               — switch to "all" to recover supervision.
        if target_gate_kind not in ("contact", "all"):
            raise ValueError(
                f"target_gate_kind must be 'contact' or 'all', got {target_gate_kind!r}"
            )
        self.target_gate_kind = target_gate_kind
        # v8 target loss kind:
        #   "smooth_l1" — legacy Huber on xyz regression (v6 / v7 / v7-fix)
        #   "kl_div"    — KL(GT_attn || pred_attn) where GT_attn is a
        #                 Gaussian-kernelled distribution over object tokens
        #                 derived from contact_target_xyz_gt + object_xyz.
        #                 Requires pred to contain ``contact_target_attn``
        #                 and forward() to receive ``object_xyz``. See
        #                 Move-as-You-Say CVPR'24 (arXiv 2403.18036) for the
        #                 affordance-heatmap precedent.
        if target_loss_kind not in ("smooth_l1", "kl_div"):
            raise ValueError(
                f"target_loss_kind must be 'smooth_l1' or 'kl_div', "
                f"got {target_loss_kind!r}"
            )
        self.target_loss_kind = target_loss_kind
        self.target_kernel_sigma = target_kernel_sigma
        # Consistency loss is the auxiliary term that mirrors the
        # extract_*.py DAG: hand_support ⊂ hand_contact, sitting ⊂
        # pelvis_contact, phase != non_contact ⊂ any_contact, target
        # attention is high-entropy on no-contact frames. Weight 0
        # disables. Default 0 so legacy training reproduces v7-fix.
        self.consistency_weight = consistency_weight
        if use_kendall_weights:
            self.kendall = KendallTaskWeights(
                task_names=("contact", "target", "phase", "support"),
            )
        else:
            self.kendall = None
        # Logit Adjustment (Menon et al. ICLR'21) for long-tailed
        # categorical heads — Bayes-optimal under a known class prior
        # at extreme imbalance. v3 with focal γ=2 alone left
        # `pre_contact` (0.4%) F1 = 0 and `single_foot` (5%) F1 = 0
        # even on train; LDAM (Cao NeurIPS'19) and Menon ICLR'21
        # both show focal underperforms margin/logit-based methods
        # at ratios >100:1. Add ``τ × log π_y`` to the logits at
        # train time only — at inference the raw logits are used.
        # ``logit_adjust_*`` is a 1D tensor of class log-priors
        # (== log(N_y / Σ N_y)). Pass None to disable per-head.
        self.logit_adjust_tau = float(logit_adjust_tau)
        if logit_adjust_phase is not None:
            self.register_buffer("logit_adjust_phase", logit_adjust_phase.float())
        else:
            self.logit_adjust_phase = None  # type: ignore[assignment]
        if logit_adjust_support is not None:
            self.register_buffer("logit_adjust_support", logit_adjust_support.float())
        else:
            self.logit_adjust_support = None  # type: ignore[assignment]

    @staticmethod
    def _focal_weight(logits: Tensor, gt: Tensor, gamma: float) -> Tensor:
        """(1 - p_t)^gamma per-element, where p_t is the predicted
        probability of the true class. logits shape (N, C), gt shape (N,)."""
        with torch.no_grad():
            p = F.softmax(logits, dim=-1)                            # (N, C)
            p_t = p.gather(-1, gt.unsqueeze(-1)).squeeze(-1)         # (N,)
            return (1.0 - p_t).clamp(min=0.0).pow(gamma)

    def forward(
        self,
        pred: dict[str, Tensor],
        gt_contact: Tensor,
        gt_target: Tensor,
        gt_phase: Tensor,
        gt_support: Tensor,
        mask: Tensor | None = None,
        object_xyz: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute all predictor losses.

        Parameters
        ----------
        pred : output dict from InteractionPredictor. Must contain
            ``contact_logits``, ``contact_target_xyz``, ``phase_logits``,
            ``support_logits``. v8 ``target_loss_kind="kl_div"`` also
            requires ``contact_target_attn`` ∈ (B, T, 5, M).
        gt_contact : (B, T, 5) — soft contact pseudo-labels in [0, 1]
        gt_target : (B, T, 5, 3) — xyz target in object-local frame
            (zero / arbitrary where gt_contact < threshold — gated out)
        gt_phase : (B, T) — integer phase labels
        gt_support : (B, T) — integer support labels
        mask : (B, T) — True for valid (non-padded) frames
        object_xyz : (B, M, 3) — object encoder centroid positions in
            object-local frame. Required when ``target_loss_kind="kl_div"``
            (v8 affordance-heatmap target).

        Returns
        -------
        Dictionary with individual losses and total.
        """
        # Contact: BCE on soft labels
        loss_contact = F.binary_cross_entropy_with_logits(
            pred["contact_logits"], gt_contact, reduction="none",
        )  # (B, T, 5)

        # Target loss: dispatch on target_loss_kind.
        # smooth_l1 path is the legacy v6/v7/v7-fix Huber regression.
        # kl_div path is v8: KL(GT_attn || pred_attn) where GT_attn is
        # a Gaussian kernel over the distance from gt_target_xyz to each
        # of the M object-token centroid positions.
        if self.target_loss_kind == "smooth_l1":
            loss_target = F.smooth_l1_loss(
                pred["contact_target_xyz"], gt_target, reduction="none",
            ).sum(dim=-1)  # (B, T, 5)
        elif self.target_loss_kind == "kl_div":
            if "contact_target_attn" not in pred:
                raise ValueError(
                    "target_loss_kind='kl_div' requires the predictor to "
                    "emit 'contact_target_attn' (use structured_head=True)."
                )
            if object_xyz is None:
                raise ValueError(
                    "target_loss_kind='kl_div' requires object_xyz to be "
                    "passed into PredictorLoss.forward()."
                )
            loss_target = self._kl_div_target_loss(
                pred_attn=pred["contact_target_attn"],   # (B, T, P, M)
                gt_xyz=gt_target,                        # (B, T, P, 3)
                object_xyz=object_xyz,                   # (B, M, 3)
                sigma=self.target_kernel_sigma,
            )  # (B, T, P)
        else:
            raise ValueError(self.target_loss_kind)  # unreachable

        # Phase: CE on integer labels + label smoothing + optional
        # logit-adjustment (Menon ICLR'21) + optional focal.
        B, T, P = pred["phase_logits"].shape
        phase_logits_flat = pred["phase_logits"].reshape(-1, P)
        phase_gt_flat = gt_phase.reshape(-1)
        if self.logit_adjust_phase is not None:
            phase_logits_flat = phase_logits_flat + (
                self.logit_adjust_tau * self.logit_adjust_phase
            )
        loss_phase_flat = F.cross_entropy(
            phase_logits_flat, phase_gt_flat,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        if self.focal_gamma > 0:
            loss_phase_flat = loss_phase_flat * self._focal_weight(
                phase_logits_flat, phase_gt_flat, self.focal_gamma,
            )
        loss_phase = loss_phase_flat.reshape(B, T)  # (B, T)

        # Support: CE on integer labels + label smoothing + optional
        # logit-adjustment + optional focal.
        S = pred["support_logits"].shape[-1]
        support_logits_flat = pred["support_logits"].reshape(-1, S)
        support_gt_flat = gt_support.reshape(-1)
        if self.logit_adjust_support is not None:
            support_logits_flat = support_logits_flat + (
                self.logit_adjust_tau * self.logit_adjust_support
            )
        loss_support_flat = F.cross_entropy(
            support_logits_flat, support_gt_flat,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        if self.focal_gamma > 0:
            loss_support_flat = loss_support_flat * self._focal_weight(
                support_logits_flat, support_gt_flat, self.focal_gamma,
            )
        loss_support = loss_support_flat.reshape(B, T)  # (B, T)

        # Build frame mask and contact gate
        if mask is not None:
            frame_mask = mask.float()                             # (B, T)
        else:
            frame_mask = torch.ones(
                gt_contact.shape[:2],
                device=gt_contact.device, dtype=torch.float32,
            )
        # Contact gate: only supervise contact_target where the GT
        # contact label crosses threshold — outside contact the pseudo-
        # target is uninformative.
        contact_gate = (
            (gt_contact > self.contact_threshold).float()
            * frame_mask.unsqueeze(-1)
        )                                                         # (B, T, 5)

        n_frames = frame_mask.sum() + 1e-8
        n_contact = contact_gate.sum() + 1e-8
        num_parts = gt_contact.shape[-1]

        loss_contact = (
            (loss_contact * frame_mask.unsqueeze(-1)).sum()
            / (n_frames * num_parts)
        )
        # KL-div target loss is only meaningful where the body part is
        # actually contacting (closest-mesh-point is the contact site).
        # Force "contact" gate regardless of target_gate_kind for kl_div.
        # smooth_l1 retains the v7-fix "all" path for backward compat.
        if self.target_loss_kind == "kl_div":
            loss_target = (loss_target * contact_gate).sum() / n_contact
        elif self.target_gate_kind == "all":
            # Supervise every valid (frame, part) cell — closest-surface-
            # point is well-defined regardless of contact (extract_target
            # emits 100% non-zero xyz). Recovers supervision when contact
            # frac drops, e.g. v12 strict labels.
            target_full_gate = frame_mask.unsqueeze(-1).expand_as(loss_target)  # (B, T, 5)
            n_target = target_full_gate.sum() + 1e-8
            loss_target = (loss_target * target_full_gate).sum() / n_target
        else:
            loss_target = (loss_target * contact_gate).sum() / n_contact
        loss_phase = (loss_phase * frame_mask).sum() / n_frames
        loss_support = (loss_support * frame_mask).sum() / n_frames

        # v8 consistency loss — auxiliary term enforcing the extraction
        # DAG's physical priors. Disabled when consistency_weight == 0,
        # which preserves the v6 / v7 / v7-fix legacy training contract.
        if self.consistency_weight > 0:
            loss_consistency = self._consistency_loss(pred, frame_mask)
        else:
            loss_consistency = torch.zeros((), device=loss_contact.device)

        if self.kendall is not None:
            # Kendall et al. CVPR'18 multi-task uncertainty weighting.
            # Static *_weight values are ignored. The optimiser learns
            # per-task log-variances; combined loss equalises gradient
            # contribution by construction.
            total, kendall_log = self.kendall({
                "contact": loss_contact,
                "target": loss_target,
                "phase": loss_phase,
                "support": loss_support,
            })
        else:
            total = (
                self.contact_weight * loss_contact
                + self.target_weight * loss_target
                + self.phase_weight * loss_phase
                + self.support_weight * loss_support
            )
            kendall_log = {}

        if self.consistency_weight > 0:
            total = total + self.consistency_weight * loss_consistency

        # Unweighted sum of raw per-task losses — the supervision
        # signal that actually tracks model quality, with no Kendall
        # combinator and no `+0.5·s_i` term. v4 found that selecting
        # best_val on the Kendall-combined ``loss`` was broken: as
        # log_var_target descended (good), the `0.5·s` contribution
        # made total ``loss`` decrease monotonically regardless of
        # supervision quality, so best_val.pt got saved at epoch 4
        # (before Kendall took off) and final.pt was 3× better on
        # phase macro-F1. Use this key for ``val_best_key`` instead.
        loss_unweighted = loss_contact + loss_target + loss_phase + loss_support

        return {
            "loss": total,
            "loss_unweighted": loss_unweighted,
            "loss_contact": loss_contact,
            "loss_target": loss_target,
            "loss_phase": loss_phase,
            "loss_support": loss_support,
            "loss_consistency": loss_consistency,
            "n_contact_frames": n_contact.detach(),
            **kendall_log,
        }

    @staticmethod
    def _kl_div_target_loss(
        pred_attn: Tensor,
        gt_xyz: Tensor,
        object_xyz: Tensor,
        sigma: float,
    ) -> Tensor:
        """KL-divergence target loss for v8 affordance-style heads.

        Constructs a Gaussian-kernelled GT distribution over the M object
        tokens for each (frame, body_part) cell, then computes
        KL(GT || pred). Returned shape (B, T, P) — caller applies the
        contact gate.

        Parameters
        ----------
        pred_attn : (B, T, P, M) — softmax-normalised predicted attention
            over object tokens. Must already be a valid distribution
            (StructuredHead's MultiheadAttention emits softmax weights).
        gt_xyz : (B, T, P, 3) — closest-mesh-point xyz per body part,
            in object-local frame.
        object_xyz : (B, M, 3) — object encoder centroid positions in
            object-local frame.
        sigma : Gaussian kernel width (m). Move-as-You-Say uses 0.8 m at
            scene scale; 0.08 m at object scale is the v8 default
            (~ 1/10 of their σ since our objects are 10-100× smaller).

        Returns
        -------
        loss_target : (B, T, P) — per-cell KL divergence (≥ 0).
        """
        B, T, P, _ = gt_xyz.shape
        M = object_xyz.shape[1]
        # Pairwise squared distance between gt_xyz and each object token.
        # gt_xyz: (B, T, P, 3) → (B, T, P, 1, 3); object_xyz: (B, 1, 1, M, 3)
        diff = gt_xyz.unsqueeze(-2) - object_xyz.view(B, 1, 1, M, 3)
        d_sq = diff.pow(2).sum(dim=-1)                              # (B, T, P, M)
        # Gaussian-kernel softmax → GT distribution
        gt_attn = F.softmax(-d_sq / (2.0 * sigma * sigma), dim=-1)  # (B, T, P, M)
        # KL(gt || pred) = sum gt * (log gt - log pred)
        # F.kl_div expects log-prob input + prob target.
        log_pred = (pred_attn.clamp_min(1e-12)).log()
        # reduction='none' → (B, T, P, M); sum over M for per-cell loss
        loss = F.kl_div(log_pred, gt_attn, reduction="none").sum(dim=-1)
        return loss                                                 # (B, T, P)

    @staticmethod
    def _consistency_loss(
        pred: dict[str, Tensor],
        frame_mask: Tensor,
    ) -> Tensor:
        """Auxiliary consistency loss enforcing the extraction-DAG priors.

        Four relu-hinge constraints, all = 0 when satisfied:
        1. P(target attention spread > 0) when contact = 0
           (target attention should be near-uniform on no-contact frames)
        2. P(hand_support) ≤ max(P(left_hand), P(right_hand))
           ([extract_support.py:184-202](src/piano/data/pseudo_labels/extract_support.py#L184))
        3. P(sitting) ≤ P(pelvis_contact)
        4. P(phase != non_contact) ≤ P(any_part_contact)
           ([extract_phase.py:122](src/piano/data/pseudo_labels/extract_phase.py#L122))

        The hinge form `relu(p_dependent - p_prerequisite)` lets the
        model leverage cases where it is more confident than the
        extractor without forcing exact label match.
        """
        contact_prob = torch.sigmoid(pred["contact_logits"])           # (B, T, P)

        # 1. Target attention entropy on no-contact frames
        if "contact_target_attn" in pred:
            attn = pred["contact_target_attn"].clamp_min(1e-12)        # (B, T, P, M)
            entropy = -(attn * attn.log()).sum(dim=-1)                 # (B, T, P)
            max_entropy = math.log(attn.shape[-1])
            no_contact = 1.0 - contact_prob                            # (B, T, P)
            l_attn = (no_contact * (max_entropy - entropy)).mean()
        else:
            l_attn = torch.zeros((), device=contact_prob.device)

        # 2. hand_support ⊂ hand contact
        # support classes: {0=both_feet, 1=single_foot, 2=sitting, 3=hand_support}
        support_prob = F.softmax(pred["support_logits"], dim=-1)       # (B, T, 4)
        hand_contact = torch.maximum(contact_prob[..., 0], contact_prob[..., 1])
        l_hand = F.relu(support_prob[..., 3] - hand_contact)
        l_hand = (l_hand * frame_mask).sum() / (frame_mask.sum() + 1e-8)

        # 3. sitting ⊂ pelvis contact (pelvis is body-part index 4)
        pelvis_contact = contact_prob[..., 4]
        l_sit = F.relu(support_prob[..., 2] - pelvis_contact)
        l_sit = (l_sit * frame_mask).sum() / (frame_mask.sum() + 1e-8)

        # 4. phase != non_contact ⊂ any part contact
        phase_prob = F.softmax(pred["phase_logits"], dim=-1)
        any_contact = contact_prob.max(dim=-1).values
        p_in_contact = 1.0 - phase_prob[..., 0]
        l_phase = F.relu(p_in_contact - any_contact)
        l_phase = (l_phase * frame_mask).sum() / (frame_mask.sum() + 1e-8)

        return l_attn + l_hand + l_sit + l_phase


# ============================================================================
# Stage B: Generator losses
# ============================================================================

class GeneratorLoss(nn.Module):
    """Loss for the Motion Generator (MoMask masked transformer).

    The primary loss is masked token prediction CE (computed inside the
    MaskedTransformer forward). This module adds optional velocity
    smoothness regularization on the decoded motion.
    """

    def __init__(self, velocity_smoothness_weight: float = 0.01) -> None:
        super().__init__()
        self.velocity_smoothness_weight = velocity_smoothness_weight

    def forward(
        self,
        mask_pred_loss: Tensor,
        decoded_motion: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute generator loss.

        Parameters
        ----------
        mask_pred_loss : scalar — CE loss on masked tokens (from MaskedTransformer.forward)
        decoded_motion : (B, T, 263) — optionally decoded motion for smoothness
        """
        total = mask_pred_loss

        losses = {
            "loss_mask_pred": mask_pred_loss,
        }

        if decoded_motion is not None and self.velocity_smoothness_weight > 0:
            loss_smooth = velocity_smoothness_loss(decoded_motion)
            total = total + self.velocity_smoothness_weight * loss_smooth
            losses["loss_smoothness"] = loss_smooth

        losses["loss"] = total
        return losses


# ============================================================================
# Stage C: Consistency loss
# ============================================================================

class ConsistencyLoss(nn.Module):
    """Consistency loss between input interaction latent and extracted latent.

    Ensures the generator doesn't ignore the interaction conditioning by
    requiring that the generated motion, when passed through the extractor,
    recovers the original interaction labels.
    """

    def __init__(
        self,
        contact_weight: float = 1.0,
        phase_weight: float = 0.5,
        support_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.contact_weight = contact_weight
        self.phase_weight = phase_weight
        self.support_weight = support_weight

    def forward(
        self,
        extracted: dict[str, Tensor],
        original: dict[str, Tensor],
        mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute consistency between extracted and original interaction labels.

        Parameters
        ----------
        extracted : output from InteractionExtractor on generated motion
        original : output from InteractionPredictor (the conditioning labels)
        mask : (B, T) — True for valid frames
        """
        # Contact consistency (BCE between two soft predictions)
        loss_contact = F.mse_loss(
            extracted["contact_state"], original["contact_state"].detach(),
            reduction="none",
        )

        # Phase consistency (KL between two distributions)
        loss_phase = F.kl_div(
            F.log_softmax(extracted["phase_logits"], dim=-1),
            original["phase"].detach(),
            reduction="none",
        ).sum(dim=-1)

        # Support consistency
        loss_support = F.kl_div(
            F.log_softmax(extracted["support_logits"], dim=-1),
            original["support"].detach(),
            reduction="none",
        ).sum(dim=-1)

        if mask is not None:
            frame_mask = mask.float()
            loss_contact = (loss_contact * frame_mask.unsqueeze(-1)).sum() / (frame_mask.sum() * 5 + 1e-8)
            loss_phase = (loss_phase * frame_mask).sum() / (frame_mask.sum() + 1e-8)
            loss_support = (loss_support * frame_mask).sum() / (frame_mask.sum() + 1e-8)
        else:
            loss_contact = loss_contact.mean()
            loss_phase = loss_phase.mean()
            loss_support = loss_support.mean()

        total = (
            self.contact_weight * loss_contact
            + self.phase_weight * loss_phase
            + self.support_weight * loss_support
        )

        return {
            "loss": total,
            "loss_consistency_contact": loss_contact,
            "loss_consistency_phase": loss_phase,
            "loss_consistency_support": loss_support,
        }


# ============================================================================
# Helpers
# ============================================================================

def velocity_smoothness_loss(motion: Tensor) -> Tensor:
    """Penalize high acceleration (second-order finite difference).

    Parameters
    ----------
    motion : (B, T, D) — motion feature sequence

    Returns
    -------
    Scalar loss — mean squared acceleration.
    """
    # velocity: (B, T-1, D)
    vel = motion[:, 1:, :] - motion[:, :-1, :]
    # acceleration: (B, T-2, D)
    acc = vel[:, 1:, :] - vel[:, :-1, :]
    return acc.pow(2).mean()
