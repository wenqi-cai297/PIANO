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
        # v8.1 (2026-05-05): multi-hot binary GT under target_loss_kind="focal_dice".
        # Per body part, tokens within radius τ_part of GT closest_xyz are
        # positive; others negative. EgoChoir / Text2HOI HOI affordance
        # convention. τ matches v12_strict pseudo-label tight thresholds.
        target_focal_alpha: float = 0.25,
        target_focal_gamma: float = 2.0,
        target_dice_eps: float = 1e-6,
        # τ per part: hand=5cm, foot=3cm, pelvis=12cm. Order matches
        # PIANO body-part indexing (left/right hand, left/right foot,
        # pelvis). Override per dataset via cfg.loss.target_tau_per_part.
        target_tau_per_part: tuple[float, ...] | None = None,
        # v8.1.1 (2026-05-05): top-K minimum to avoid empty GT masks
        # in low-density regions (e.g. foot τ=3cm has 0 tokens in
        # neighborhood when 128 FPS spacing ~ 8.8cm). GT mask =
        # (top-K nearest) ∪ (within-τ). K=0 disables (pure τ-only,
        # v8.1 behaviour). K=3 is the v8.1.1 default — matches typical
        # palm-contact region size.
        target_topk_min_positives: int = 0,
        # v9 (2026-05-03): per-part pos_weight for contact BCE. Fixes
        # the "passive zero" pathology where foot contact (~3% positive
        # rate) BCE is dominated 32:1 by negatives → model trivially
        # predicts negative everywhere → recall = 0. DECO ICCV 2023 +
        # HACO NeurIPS 2025 confirm class-balanced BCE beats focal for
        # per-element binary contact. Pass shape (num_body_parts,);
        # None disables. Computed at train start as
        # ((1 - π_part) / π_part).clamp(max=cap) where π_part is the
        # training-set positive rate for that body part.
        contact_pos_weight: Tensor | None = None,
        # v9.2 (2026-05-03): Asymmetric Loss for contact head. ASL
        # decouples positive vs negative gradient handling — γ_pos=0
        # keeps positives' full gradient (preserves recall) while
        # γ_neg=4 down-weights easy negatives via (p_shifted)^γ_neg
        # modulation, focusing gradient on hard negatives (FPs).
        # Reference: Ben-Baruch et al. ICCV 2021, arXiv:2009.14119,
        # github.com/Alibaba-MIIL/ASL (797★). Replaces pos_weight
        # when contact_loss_kind="asl"; "bce" preserves v9 behaviour.
        contact_loss_kind: str = "bce",
        contact_asl_gamma_pos: float = 0.0,
        contact_asl_gamma_neg: float = 4.0,
        contact_asl_prob_shift: float = 0.05,
        # v9.4 (2026-05-04): auxiliary xyz L2 loss alongside focal+dice
        # for the target_attn head. focal+dice operates on per-token
        # binary mask but provides no gradient that distinguishes
        # "closer to true xyz" from "farther but still in mask" within
        # the GT positive set. The aux term computes a single xyz
        # prediction via softmax(logits) @ object_xyz and Huber-
        # regresses against gt_target. This injects the missing
        # spatial-distance gradient.
        # Only active when target_loss_kind="focal_dice" and weight > 0.
        target_aux_xyz_weight: float = 0.0,
        # v9.5 (2026-05-04): hierarchical patch-level CE loss for the
        # HierarchicalMaskDecoder. Only fires when the predictor emits
        # ``contact_target_patch_logits`` and ``contact_target_token_to_patch``.
        # Provides explicit "right region" supervision (1-of-K patches,
        # K=16) — much cleaner training signal than the 1-of-256 token
        # task which is encoder-smoothness-bound. Per-cell loss is
        # F.cross_entropy(patch_logits, gt_patch_id) gated by contact.
        # Weight 0.3 default — same magnitude as aux_xyz_weight.
        target_patch_weight: float = 0.0,
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
        if target_loss_kind not in ("smooth_l1", "kl_div", "focal_dice"):
            raise ValueError(
                f"target_loss_kind must be 'smooth_l1', 'kl_div', or "
                f"'focal_dice', got {target_loss_kind!r}"
            )
        self.target_loss_kind = target_loss_kind
        self.target_kernel_sigma = target_kernel_sigma
        # Consistency loss is the auxiliary term that mirrors the
        # extract_*.py DAG: hand_support ⊂ hand_contact, sitting ⊂
        # pelvis_contact, phase != non_contact ⊂ any_contact, target
        # attention is high-entropy on no-contact frames. Weight 0
        # disables. Default 0 so legacy training reproduces v7-fix.
        self.consistency_weight = consistency_weight
        # v8.1 focal+dice hyperparameters
        self.target_focal_alpha = target_focal_alpha
        self.target_focal_gamma = target_focal_gamma
        self.target_dice_eps = target_dice_eps
        if target_tau_per_part is None:
            # Default matches v12_strict tight thresholds:
            # hand 5 cm, foot 3 cm, pelvis 12 cm.
            target_tau_per_part = (0.05, 0.05, 0.03, 0.03, 0.12)
        self.register_buffer(
            "target_tau_per_part",
            torch.tensor(target_tau_per_part, dtype=torch.float32),
        )
        if int(target_topk_min_positives) < 0:
            raise ValueError(
                f"target_topk_min_positives must be >= 0, "
                f"got {target_topk_min_positives}"
            )
        self.target_topk_min_positives = int(target_topk_min_positives)
        # v9: contact pos_weight for class-balanced BCE.
        if contact_pos_weight is not None:
            if contact_pos_weight.ndim != 1:
                raise ValueError(
                    f"contact_pos_weight must be 1-D (num_body_parts,), "
                    f"got shape {tuple(contact_pos_weight.shape)}"
                )
            self.register_buffer("contact_pos_weight", contact_pos_weight.float())
        else:
            self.contact_pos_weight = None  # type: ignore[assignment]
        # v9.2: ASL contact loss flags
        if contact_loss_kind not in ("bce", "asl"):
            raise ValueError(
                f"contact_loss_kind must be 'bce' or 'asl', "
                f"got {contact_loss_kind!r}"
            )
        self.contact_loss_kind = contact_loss_kind
        self.contact_asl_gamma_pos = float(contact_asl_gamma_pos)
        self.contact_asl_gamma_neg = float(contact_asl_gamma_neg)
        self.contact_asl_prob_shift = float(contact_asl_prob_shift)
        # v9.4: aux xyz L2 weight on top of focal+dice. 0.3 default in
        # the v9.4 config; 0 disables (legacy behaviour).
        if float(target_aux_xyz_weight) < 0.0:
            raise ValueError(
                f"target_aux_xyz_weight must be >= 0, "
                f"got {target_aux_xyz_weight}"
            )
        self.target_aux_xyz_weight = float(target_aux_xyz_weight)
        # v9.5: hierarchical patch CE weight. 0 disables.
        if float(target_patch_weight) < 0.0:
            raise ValueError(
                f"target_patch_weight must be >= 0, "
                f"got {target_patch_weight}"
            )
        self.target_patch_weight = float(target_patch_weight)
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
        # Contact loss. Two modes:
        # - "bce" (v9 default): BCE with optional per-part pos_weight.
        # - "asl" (v9.2): Asymmetric Loss (Ben-Baruch et al. ICCV'21).
        #   Decouples positive vs negative gradient handling. Fixes
        #   the precision regression introduced by aggressive pos_weight
        #   (foot precision 0.06 with pos_weight cap=15) without losing
        #   recall.
        if self.contact_loss_kind == "asl":
            loss_contact = self._asymmetric_contact_loss(
                logits=pred["contact_logits"],
                target=gt_contact,
                gamma_pos=self.contact_asl_gamma_pos,
                gamma_neg=self.contact_asl_gamma_neg,
                prob_shift=self.contact_asl_prob_shift,
            )  # (B, T, 5)
        else:
            # v9 BCE path (with optional pos_weight).
            contact_pw = getattr(self, "contact_pos_weight", None)
            if contact_pw is not None:
                pw = contact_pw.view(1, 1, -1).to(
                    device=pred["contact_logits"].device,
                    dtype=pred["contact_logits"].dtype,
                )
                loss_contact = F.binary_cross_entropy_with_logits(
                    pred["contact_logits"], gt_contact,
                    reduction="none", pos_weight=pw,
                )
            else:
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
        elif self.target_loss_kind == "focal_dice":
            if "contact_target_attn_logits" not in pred:
                raise ValueError(
                    "target_loss_kind='focal_dice' requires the predictor "
                    "to emit 'contact_target_attn_logits' (use "
                    "structured_head_target_attn_output='logits')."
                )
            if object_xyz is None:
                raise ValueError(
                    "target_loss_kind='focal_dice' requires object_xyz "
                    "to be passed into PredictorLoss.forward()."
                )
            loss_target = self._focal_dice_target_loss(
                pred_logits=pred["contact_target_attn_logits"],  # (B, T, P, M)
                gt_xyz=gt_target,                                # (B, T, P, 3)
                object_xyz=object_xyz,                           # (B, M, 3)
                tau_per_part=self.target_tau_per_part,           # (P,)
                focal_alpha=self.target_focal_alpha,
                focal_gamma=self.target_focal_gamma,
                dice_eps=self.target_dice_eps,
                topk_min_positives=self.target_topk_min_positives,
            )  # (B, T, P)
            # v9.4: auxiliary xyz L2 — softmax-weighted token xyz vs
            # gt_target. Only adds spatial-distance gradient; per-cell
            # additive, gated identically to focal+dice below.
            if self.target_aux_xyz_weight > 0.0:
                attn_softmax = F.softmax(
                    pred["contact_target_attn_logits"], dim=-1,
                )                                                # (B, T, P, M)
                pred_xyz = torch.einsum(
                    "btpm,bmc->btpc", attn_softmax, object_xyz,
                )                                                # (B, T, P, 3)
                aux_l2 = F.smooth_l1_loss(
                    pred_xyz, gt_target, reduction="none",
                ).sum(dim=-1)                                    # (B, T, P)
                loss_target = loss_target + self.target_aux_xyz_weight * aux_l2
            # v9.5: hierarchical patch CE — gt_patch_id derived from
            # gt_target via "find nearest token, look up its patch_id".
            # Only fires when the predictor emits patch_logits +
            # token_to_patch (HierarchicalMaskDecoder path).
            has_patch = (
                self.target_patch_weight > 0.0
                and "contact_target_patch_logits" in pred
                and "contact_target_token_to_patch" in pred
            )
            if has_patch:
                patch_logits = pred["contact_target_patch_logits"]   # (B, T, P, K)
                token_to_patch = pred["contact_target_token_to_patch"]  # (B, M)
                B_, T_, P_, K_ = patch_logits.shape
                M_ = token_to_patch.shape[1]
                # Find nearest token to gt_target per (frame, part) cell:
                # gt_target (B, T, P, 3) vs object_xyz (B, M, 3).
                diff = gt_target.unsqueeze(-2) - object_xyz.view(
                    B_, 1, 1, M_, 3,
                )                                                # (B, T, P, M, 3)
                d2 = diff.pow(2).sum(dim=-1)                     # (B, T, P, M)
                nearest_token = d2.argmin(dim=-1)                # (B, T, P) ∈ [0, M)
                # Gather token_to_patch[b, nearest_token[b, t, p]]
                # → (B, T, P) ∈ [0, K)
                gt_patch_id = torch.gather(
                    token_to_patch.unsqueeze(1).unsqueeze(1).expand(
                        -1, T_, P_, -1,
                    ),
                    -1, nearest_token.unsqueeze(-1),
                ).squeeze(-1)                                    # (B, T, P)
                # Per-cell CE — flatten B*T*P, run F.cross_entropy.
                patch_logits_flat = patch_logits.reshape(-1, K_)
                gt_patch_id_flat = gt_patch_id.reshape(-1).long()
                loss_patch_flat = F.cross_entropy(
                    patch_logits_flat, gt_patch_id_flat,
                    reduction="none",
                )                                                # (B*T*P,)
                loss_patch = loss_patch_flat.reshape(B_, T_, P_)
                loss_target = loss_target + self.target_patch_weight * loss_patch
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
        # KL-div / focal-dice target losses are only meaningful where
        # the body part is actually contacting (closest-mesh-point is
        # the contact site; outside contact frames the multi-hot mask
        # would be all zero or noise). Force "contact" gate regardless
        # of target_gate_kind. smooth_l1 retains the v7-fix "all" path
        # for backward compat.
        if self.target_loss_kind in ("kl_div", "focal_dice"):
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
    def _asymmetric_contact_loss(
        logits: Tensor,
        target: Tensor,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        prob_shift: float = 0.05,
        eps: float = 1e-8,
    ) -> Tensor:
        """Asymmetric Loss for multi-label binary classification.

        Verbatim implementation of Ben-Baruch et al. ICCV 2021,
        arXiv:2009.14119 — see github.com/Alibaba-MIIL/ASL (797★)
        ``src/loss_functions/losses.py::AsymmetricLoss``. Adaptations:
        - Returns per-element loss (B, T, P) for masked-mean aggregation
          by the caller, instead of the upstream ``-loss.sum()``.
        - Default ``gamma_pos=0`` (paper's "passive zero protection"
          recipe — keeps positives' full gradient to preserve recall;
          upstream's default of 1 down-weights confident positives).
        - Focal modulator gradient flow is enabled (we don't replicate
          the upstream's ``disable_torch_grad_focal_loss`` because the
          gradient through ``one_sided_w`` is harmless at our scale).

        Mechanism (why it fixes our foot precision = 0.06):
        - γ_pos = 0 → positives' (1 - p_t)^0 = 1 → no down-weighting.
          Recall is preserved (same gradient as plain BCE on positives).
        - γ_neg = 4 → easy negatives (p ≈ 0) get weight ≈ 0; hard
          negatives (model wrongly says positive, p ≈ 0.7) get weight
          ≈ (0.7)^4 ≈ 0.24 — dominant gradient signal. Optimizer
          focuses on the False Positives we're trying to reduce.
        - prob_shift (clip = 0.05) → for negatives, treat any p < 0.05
          as fully correct. Protects against ~10 % pseudo-label noise
          from making the model overcompensate on near-zero negatives.

        Parameters
        ----------
        logits : (B, T, P) — pre-sigmoid scores from contact head
        target : (B, T, P) — soft labels in [0, 1] (we use binarised
            > 0.5 thresholding on the soft pseudo-labels in practice)
        gamma_pos, gamma_neg : focusing parameters per class polarity
        prob_shift : negative-class probability shift (asymmetric clip)
        eps : log clamp

        Returns
        -------
        loss : (B, T, P) — per-element non-negative loss
        """
        # Probabilities
        x_sigmoid = torch.sigmoid(logits)
        xs_pos = x_sigmoid
        xs_neg = 1.0 - x_sigmoid

        # Asymmetric clipping: shift negative-class probability up by
        # prob_shift, clamp to 1. Negatives with sigmoid < prob_shift
        # become "1 - 0" → log(1) = 0 → zero loss.
        if prob_shift > 0:
            xs_neg = (xs_neg + prob_shift).clamp(max=1.0)

        # Cross-entropy per polarity (verbatim from upstream)
        los_pos = target * torch.log(xs_pos.clamp(min=eps))
        los_neg = (1.0 - target) * torch.log(xs_neg.clamp(min=eps))
        loss = los_pos + los_neg

        # Asymmetric focal modulator
        if gamma_neg > 0 or gamma_pos > 0:
            pt0 = xs_pos * target
            pt1 = xs_neg * (1.0 - target)
            pt = pt0 + pt1
            one_sided_gamma = gamma_pos * target + gamma_neg * (1.0 - target)
            one_sided_w = torch.pow(1.0 - pt, one_sided_gamma)
            loss = loss * one_sided_w

        # Per-element non-negative loss (caller does masked mean).
        return -loss

    @staticmethod
    def _focal_dice_target_loss(
        pred_logits: Tensor,
        gt_xyz: Tensor,
        object_xyz: Tensor,
        tau_per_part: Tensor,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        dice_eps: float = 1e-6,
        topk_min_positives: int = 0,
    ) -> Tensor:
        """v8.1 affordance loss: focal BCE + dice on multi-hot binary GT.

        For each (frame, body_part) cell, GT is the binary mask of
        object tokens within ``τ_part`` of the GT closest-mesh-point
        xyz. Multiple adjacent tokens can be positive — palm contact
        covers a few cm region, multiple FPS-sampled tokens fall in it.
        This is the EgoChoir (NeurIPS 2024) / Text2HOI (CVPR 2024)
        convention for HOI affordance supervision.

        v8.1.1 addition (2026-05-05): when ``topk_min_positives > 0``,
        the GT mask is the union of (top-K nearest tokens) ∪
        (within-τ tokens). This guarantees every cell has at least K
        positives regardless of FPS density, fixing the v8.1 foot
        regression where τ_foot=3cm produced empty masks at FPS
        spacing ~8.8cm.

        Predicted output is per-token sigmoid (each token is an
        independent binary classifier). Loss is focal BCE + dice on
        the multi-hot mask.

        Parameters
        ----------
        pred_logits : (B, T, P, M) — pre-sigmoid attention scores from
            ``CrossAttentionWeightsOnly`` with ``output="logits"``.
        gt_xyz : (B, T, P, 3) — closest-mesh-point xyz per body part,
            object-local frame.
        object_xyz : (B, M, 3) — object encoder centroid positions.
        tau_per_part : (P,) — per-body-part contact radius (m).
            Default (0.05, 0.05, 0.03, 0.03, 0.12) matches v12_strict
            tight thresholds.
        focal_alpha : positive class weight (Lin et al. ICCV 2017
            RetinaNet default 0.25).
        focal_gamma : focusing parameter (default 2.0).
        dice_eps : numerical stability for dice denominator.
        topk_min_positives : minimum positive tokens per cell. 0
            disables (pure τ-only, v8.1 behaviour). 3 is the v8.1.1
            recommended default.

        Returns
        -------
        loss_target : (B, T, P) — per-cell loss; caller applies the
            contact gate before reduction.
        """
        B, T, P, _ = gt_xyz.shape
        M = object_xyz.shape[1]
        # (a) Build multi-hot GT.
        # gt_xyz: (B, T, P, 3) -> (B, T, P, 1, 3); object_xyz: (B, 1, 1, M, 3)
        diff = gt_xyz.unsqueeze(-2) - object_xyz.view(B, 1, 1, M, 3)
        d = diff.norm(dim=-1)                                      # (B, T, P, M)
        tau = tau_per_part.to(d.device, dtype=d.dtype).view(1, 1, P, 1)
        gt_mask_tau = (d < tau)                                    # (B, T, P, M) bool
        # v8.1.1 top-K minimum: union with the K nearest tokens to
        # guarantee at least K positives per cell. Avoids empty masks
        # for parts whose τ is below FPS-token spacing (foot τ=3cm).
        if topk_min_positives > 0 and topk_min_positives < M:
            # topk on negated distance → nearest tokens
            topk_idx = torch.topk(-d, k=topk_min_positives, dim=-1).indices  # (B, T, P, K)
            gt_mask_topk = torch.zeros_like(gt_mask_tau)
            gt_mask_topk.scatter_(-1, topk_idx, True)
            gt_mask_bool = gt_mask_tau | gt_mask_topk
        else:
            gt_mask_bool = gt_mask_tau
        gt_mask = gt_mask_bool.to(pred_logits.dtype)               # (B, T, P, M)

        # (b) Focal BCE per token
        # F.binary_cross_entropy_with_logits is numerically stable.
        bce = F.binary_cross_entropy_with_logits(
            pred_logits, gt_mask, reduction="none",
        )                                                           # (B, T, P, M)
        pred = torch.sigmoid(pred_logits)
        p_t = pred * gt_mask + (1.0 - pred) * (1.0 - gt_mask)
        focal_mod = (1.0 - p_t).clamp(min=0.0).pow(focal_gamma)
        alpha_t = focal_alpha * gt_mask + (1.0 - focal_alpha) * (1.0 - gt_mask)
        loss_focal = (alpha_t * focal_mod * bce).mean(dim=-1)      # (B, T, P)

        # (c) Soft Dice on the multi-hot mask. With pred ∈ [0, 1] and
        # gt ∈ {0, 1}, dice = 2·|pred ∩ gt| / (|pred| + |gt|), and
        # loss = 1 - dice. Per-cell: aggregate across the M-dim.
        intersection = (pred * gt_mask).sum(dim=-1)                 # (B, T, P)
        pred_sum = pred.sum(dim=-1)                                 # (B, T, P)
        gt_sum = gt_mask.sum(dim=-1)                                # (B, T, P)
        loss_dice = 1.0 - (
            (2.0 * intersection + dice_eps) / (pred_sum + gt_sum + dice_eps)
        )                                                           # (B, T, P)

        return 0.5 * loss_focal + 0.5 * loss_dice                   # (B, T, P)

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
